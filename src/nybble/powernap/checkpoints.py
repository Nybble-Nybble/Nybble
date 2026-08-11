"""Durable local metadata for Tinker and retriever checkpoints."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class CheckpointRecord:
    step: int
    state_path: str
    sampler_path: str
    retriever_path: str
    model: str
    renderer: str
    last_sample_id: str
    created_at: float
    config: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""
    sample_ids: tuple[str, ...] = ()
    sample_digests: tuple[str, ...] = ()
    reward_contract: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step cannot be negative")
        for name in ("state_path", "sampler_path", "retriever_path", "model", "renderer"):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(self.config, dict):
            raise TypeError("config must be a dictionary")
        json.dumps(self.config, allow_nan=False)
        if not isinstance(self.data_fingerprint, str):
            raise TypeError("data_fingerprint must be a string")
        if not isinstance(self.sample_ids, tuple) or not all(
            isinstance(sample_id, str) and sample_id for sample_id in self.sample_ids
        ):
            raise ValueError("sample_ids must be a tuple of nonempty strings")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("sample_ids must be unique")
        if not isinstance(self.sample_digests, tuple) or not all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in self.sample_digests
        ):
            raise ValueError("sample_digests must be a tuple of SHA-256 hex digests")
        if self.sample_digests and len(self.sample_digests) != len(self.sample_ids):
            raise ValueError("sample_digests must correspond one-to-one with sample_ids")
        if not isinstance(self.reward_contract, dict):
            raise TypeError("reward_contract must be a dictionary")
        json.dumps(self.reward_contract, allow_nan=False, sort_keys=True)
        if self.schema_version not in (1, 2):
            raise ValueError(f"unsupported checkpoint schema_version {self.schema_version}")

    @classmethod
    def create(
        cls,
        *,
        step: int,
        state_path: str,
        sampler_path: str,
        retriever_path: str,
        model: str,
        renderer: str,
        last_sample_id: str,
        config: dict[str, Any] | None = None,
        data_fingerprint: str = "",
        sample_ids: tuple[str, ...] = (),
        sample_digests: tuple[str, ...] = (),
        reward_contract: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        return cls(
            step=step,
            state_path=state_path,
            sampler_path=sampler_path,
            retriever_path=retriever_path,
            model=model,
            renderer=renderer,
            last_sample_id=last_sample_id,
            created_at=time.time(),
            config=dict(config or {}),
            data_fingerprint=data_fingerprint,
            sample_ids=tuple(sample_ids),
            sample_digests=tuple(sample_digests),
            reward_contract=dict(reward_contract or {}),
        )

    @classmethod
    def from_dict(cls, value: dict) -> CheckpointRecord:
        for name in ("sample_ids", "sample_digests"):
            if name in value:
                value = {**value, name: tuple(value[name])}
        return cls(**value)


class CheckpointStore:
    """Append-only checkpoint manifest that never deletes user checkpoints."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CheckpointRecord) -> None:
        if not isinstance(record, CheckpointRecord):
            raise TypeError("record must be a CheckpointRecord")
        line = json.dumps(
            asdict(record),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._read_locked(handle, repair=True)
                handle.seek(0, os.SEEK_END)
                handle.write(line + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def all(self) -> list[CheckpointRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records, _ = self._read_locked(handle, repair=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return records

    def latest(self) -> CheckpointRecord | None:
        records = self.all()
        return records[-1] if records else None

    def recover(self) -> int:
        """Repair an interrupted final append and return bytes removed."""

        if not self.path.exists():
            return 0
        with self.path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                _, removed = self._read_locked(handle, repair=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return removed

    @staticmethod
    def _read_locked(handle: BinaryIO, *, repair: bool) -> tuple[list[CheckpointRecord], int]:
        handle.seek(0)
        content = handle.read()
        if not content:
            return [], 0

        lines = content.splitlines(keepends=True)
        records: list[CheckpointRecord] = []
        offset = 0
        truncate_at: int | None = None

        for index, raw_line in enumerate(lines):
            line_start = offset
            offset += len(raw_line)
            is_last = index == len(lines) - 1
            terminated = raw_line.endswith((b"\n", b"\r"))
            stripped = raw_line.strip()

            if not stripped:
                if is_last and not terminated:
                    truncate_at = line_start
                    break
                raise ValueError(f"invalid checkpoint manifest line {index + 1}: blank record")

            try:
                payload = json.loads(stripped.decode("utf-8"))
                record = CheckpointRecord.from_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                if is_last and not terminated:
                    truncate_at = line_start
                    break
                raise ValueError(f"invalid checkpoint manifest line {index + 1}: {exc}") from exc
            records.append(record)

        removed = 0
        if repair and truncate_at is not None:
            removed = len(content) - truncate_at
            handle.seek(truncate_at)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        elif repair and lines and not lines[-1].endswith((b"\n", b"\r")):
            # A valid complete JSON object is durable data. Normalize its ending
            # before a later append so two records can never be concatenated.
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        return records, removed
