"""Output: sharded JSONL plus a run manifest.

One file per source keeps a failed collector from corrupting everything else and
makes it trivial to drop a source after the fact ("actually, not my texts").
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .redact import Redactor
from .schema import Record, validate


@dataclass
class SourceResult:
    source_id: str
    records: int = 0
    chars: int = 0
    skipped: int = 0
    error: Optional[str] = None
    seconds: float = 0.0
    paths_seen: int = 0


@dataclass
class RunManifest:
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    nybble_version: str = "0.1.0"
    platform: str = ""
    redacted: bool = True
    results: List[SourceResult] = field(default_factory=list)
    redaction_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_records(self) -> int:
        return sum(r.records for r in self.results)

    @property
    def total_chars(self) -> int:
        return sum(r.chars for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_records"] = self.total_records
        d["total_chars"] = self.total_chars
        return d


class Writer:
    """Append-only JSONL writer with per-source files and content dedupe."""

    def __init__(self, root: Path, redact: bool = True, dedupe: bool = True):
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.redactor = Redactor(enabled=redact)
        self.dedupe = dedupe
        self._seen: set = set()
        self._handles: Dict[str, Any] = {}
        self.manifest = RunManifest(redacted=redact)

    # --- lifecycle ---

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._handles.clear()

    def _handle(self, source_id: str):
        if source_id not in self._handles:
            path = self.records_dir / f"{source_id}.jsonl"
            self._handles[source_id] = path.open("a", encoding="utf-8")
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return self._handles[source_id]

    # --- writing ---

    def write(self, rec: Record) -> bool:
        """Write one record. Returns False if it was dropped (empty or dupe)."""
        rec.text = self.redactor.scrub(rec.text or "")
        if not rec.text.strip():
            return False
        # Recompute after redaction so identical scrubbed rows collapse.
        rec.id = rec.compute_id()
        if self.dedupe:
            if rec.id in self._seen:
                return False
            self._seen.add(rec.id)
        validate(rec)
        self._handle(rec.source).write(rec.to_json() + "\n")
        return True

    def write_all(self, source_id: str, records: Iterable[Record]) -> SourceResult:
        """Drain a collector, recording counts and containing its failures."""
        result = SourceResult(source_id=source_id)
        started = time.time()
        try:
            for rec in records:
                if self.write(rec):
                    result.records += 1
                    result.chars += len(rec.text)
                else:
                    result.skipped += 1
        except Exception as exc:  # a broken collector must not kill the run
            result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = round(time.time() - started, 2)
        self.manifest.results.append(result)
        return result

    # --- finish ---

    def finalize(self, platform_name: str = "") -> Path:
        self.manifest.finished_at = time.time()
        self.manifest.platform = platform_name
        self.manifest.redaction_counts = self.redactor.report()
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, default=str), "utf-8")
        return path


def iter_records(root: Path, source_ids: Optional[Iterable[str]] = None):
    """Read collected records back. Used by the taste stage and by `bundle`."""
    records_dir = Path(root) / "records"
    if not records_dir.is_dir():
        return
    wanted = set(source_ids) if source_ids else None
    for path in sorted(records_dir.glob("*.jsonl")):
        if wanted and path.stem not in wanted:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except OSError:
            continue
