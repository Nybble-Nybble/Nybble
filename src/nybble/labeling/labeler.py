"""Deterministic frame discovery and Gemini-first semantic action labeling."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
)

from PIL import ExifTags, Image

from nybble.labeling.schema import (
    FrameLabel,
    GroupingResult,
    LabeledChunk,
    LabelingResult,
    SourceFrame,
    grouping_from_payload,
)
from nybble.vlm.gemini import GeminiActionGrouper, GeminiCaptioner

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_NATURAL_PART = re.compile(r"(\d+)")
_UNIX_TOKEN = re.compile(r"(?<!\d)(\d{10}|\d{13}|\d{16}|\d{19})(?!\d)")
_MIN_UNIX_SECONDS = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
_MAX_UNIX_SECONDS = datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp()


class TimestampResolutionError(ValueError):
    """Raised when capture timestamps cannot be resolved without guessing order."""


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    """Sort filenames so frame2 precedes frame10, case-insensitively."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_PART.split(Path(path).name)
    )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, bool):
        raise TimestampResolutionError("boolean values are not timestamps")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) >= 1e18:
            numeric /= 1e9
        elif abs(numeric) >= 1e15:
            numeric /= 1e6
        elif abs(numeric) >= 1e12:
            numeric /= 1e3
        try:
            parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise TimestampResolutionError(f"invalid Unix timestamp {value!r}") from exc
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise TimestampResolutionError("timestamp must not be empty")
        try:
            numeric = float(stripped)
        except ValueError:
            normalized = stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise TimestampResolutionError(
                    f"unsupported timestamp {value!r}; use ISO 8601 or Unix seconds"
                ) from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
        else:
            return _parse_timestamp(numeric)
    else:
        raise TimestampResolutionError(f"unsupported timestamp value {value!r}")
    return parsed.astimezone(timezone.utc)


def _load_manifest(path: Path) -> dict[str, datetime]:
    """Load a JSON, JSONL, or CSV path-to-timestamp manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    records: list[tuple[str, Any]] = []
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise TimestampResolutionError("CSV manifest requires a path column")
            timestamp_field = next(
                (
                    field
                    for field in ("captured_at", "timestamp", "time")
                    if field in reader.fieldnames
                ),
                None,
            )
            if timestamp_field is None:
                raise TimestampResolutionError(
                    "CSV manifest requires captured_at, timestamp, or time"
                )
            records.extend((row["path"], row[timestamp_field]) for row in reader)
    elif suffix == ".jsonl":
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TimestampResolutionError(
                    f"invalid JSONL manifest line {line_number}"
                ) from exc
            records.append(_manifest_record(record, line_number))
    else:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TimestampResolutionError("invalid JSON timestamp manifest") from exc
        if isinstance(payload, dict):
            records.extend((str(key), value) for key, value in payload.items())
        elif isinstance(payload, list):
            records.extend(_manifest_record(record, index) for index, record in enumerate(payload))
        else:
            raise TimestampResolutionError(
                "JSON manifest must be an object mapping paths to times or a record list"
            )

    result: dict[str, datetime] = {}
    for raw_path, raw_timestamp in records:
        key = str(raw_path).strip()
        if not key:
            raise TimestampResolutionError("manifest path must not be empty")
        if key in result:
            raise TimestampResolutionError(f"duplicate manifest entry for {key!r}")
        result[key] = _parse_timestamp(raw_timestamp)
    return result


def _manifest_record(record: Any, index: int) -> tuple[str, Any]:
    if not isinstance(record, dict) or "path" not in record:
        raise TimestampResolutionError(f"manifest record {index} requires path")
    timestamp_field = next(
        (field for field in ("captured_at", "timestamp", "time") if field in record),
        None,
    )
    if timestamp_field is None:
        raise TimestampResolutionError(f"manifest record {index} requires a timestamp")
    return str(record["path"]), record[timestamp_field]


def _timestamp_from_manifest(
    frame: Path, root: Path, manifest: Mapping[str, datetime]
) -> datetime | None:
    candidates = (str(frame), str(frame.relative_to(root)), frame.name)
    found = [manifest[candidate] for candidate in candidates if candidate in manifest]
    if not found:
        return None
    if any(value != found[0] for value in found[1:]):
        raise TimestampResolutionError(f"conflicting manifest entries for {frame}")
    return found[0]


def _timestamp_from_exif(frame: Path) -> datetime | None:
    try:
        with Image.open(frame) as image:
            exif = image.getexif()
    except (OSError, ValueError):
        return None
    if not exif:
        return None

    values = {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in exif.items()}
    if hasattr(ExifTags, "IFD"):
        try:
            nested = exif.get_ifd(ExifTags.IFD.Exif)
        except (KeyError, TypeError, ValueError):
            nested = {}
        values.update({ExifTags.TAGS.get(tag, str(tag)): value for tag, value in nested.items()})
    raw = next(
        (
            values.get(field)
            for field in ("DateTimeOriginal", "DateTimeDigitized", "DateTime")
            if values.get(field)
        ),
        None,
    )
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    raw_offset = values.get("OffsetTimeOriginal") or values.get("OffsetTime")
    if isinstance(raw_offset, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", raw_offset):
        sign = 1 if raw_offset[0] == "+" else -1
        hours, minutes = (int(part) for part in raw_offset[1:].split(":"))
        parsed = parsed.replace(tzinfo=timezone(sign * timedelta(hours=hours, minutes=minutes)))
    else:
        # EXIF timestamps commonly omit an offset. UTC is an explicit, stable
        # default; a timestamp manifest should be used when local timezone matters.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_from_unix_filename(frame: Path) -> datetime | None:
    for match in _UNIX_TOKEN.finditer(frame.stem):
        digits = match.group(1)
        seconds = int(digits) / (10 ** (len(digits) - 10))
        if _MIN_UNIX_SECONDS <= seconds < _MAX_UNIX_SECONDS:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
    return None


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{prefix}_{digest.hexdigest()[:20]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_bound_frame_id(path: Path, captured_at: datetime, content_sha256: str) -> str:
    return _stable_id(
        "frame",
        (
            str(Path(path).expanduser().resolve()),
            captured_at.isoformat(),
            content_sha256,
        ),
    )


def _bind_current_image_content(frame: SourceFrame) -> SourceFrame:
    """Return a frame whose identity matches the image bytes read this run."""

    if not frame.path.is_file():
        # Synthetic SourceFrame objects are useful for offline unit tests. Real
        # labeling still validates paths in the Gemini adapter.
        return frame
    content_sha256 = _file_sha256(frame.path)
    frame_id = _content_bound_frame_id(frame.path, frame.captured_at, content_sha256)
    if frame.content_sha256 == content_sha256 and frame.frame_id == frame_id:
        return frame
    return SourceFrame(
        index=frame.index,
        path=frame.path,
        captured_at=frame.captured_at,
        frame_id=frame_id,
        content_sha256=content_sha256,
    )


def discover_frames(
    folder: Path,
    manifest_path: Path | None = None,
    fps: float | None = None,
    start_time: Any | None = None,
    recursive: bool = False,
) -> tuple[SourceFrame, ...]:
    """Discover naturally ordered frames and resolve an aware UTC timestamp.

    Resolution order is manifest, EXIF, then a plausible Unix timestamp embedded in
    the filename. Missing timestamps can be filled from ``fps`` using the first
    resolved timestamp, an explicit ``start_time``, or the first file's mtime as the
    final anchor. A supplied manifest is authoritative and must cover every frame.
    """

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (path.resolve() for path in iterator if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in _NATURAL_PART.split(str(path.relative_to(root)))
        ),
    )
    if not paths:
        raise FileNotFoundError(f"no supported image frames in {root}")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be greater than zero")

    manifest = _load_manifest(Path(manifest_path)) if manifest_path else None
    timestamps: list[datetime | None] = []
    for path in paths:
        if manifest is not None:
            timestamp = _timestamp_from_manifest(path, root, manifest)
            if timestamp is None:
                raise TimestampResolutionError(
                    f"timestamp manifest does not cover {path.relative_to(root)}"
                )
        else:
            timestamp = _timestamp_from_exif(path) or _timestamp_from_unix_filename(path)
        timestamps.append(timestamp)

    if any(timestamp is None for timestamp in timestamps):
        if fps is None:
            missing = [
                str(path.relative_to(root))
                for path, timestamp in zip(paths, timestamps, strict=True)
                if timestamp is None
            ]
            raise TimestampResolutionError(
                "could not resolve timestamps for "
                + ", ".join(missing)
                + "; supply a manifest or fps"
            )
        known_index = next(
            (index for index, value in enumerate(timestamps) if value is not None),
            None,
        )
        if start_time is not None:
            anchor_index = 0
            anchor_time = _parse_timestamp(start_time)
        elif known_index is not None:
            anchor_index = known_index
            anchor_time = timestamps[known_index]
            assert anchor_time is not None
        else:
            anchor_index = 0
            anchor_time = datetime.fromtimestamp(paths[0].stat().st_mtime, timezone.utc)
        for index, timestamp in enumerate(timestamps):
            if timestamp is None:
                timestamps[index] = anchor_time + timedelta(seconds=(index - anchor_index) / fps)

    resolved = [timestamp for timestamp in timestamps if timestamp is not None]
    for index in range(1, len(resolved)):
        if resolved[index] < resolved[index - 1]:
            raise TimestampResolutionError(
                f"timestamps go backward between {paths[index - 1].name} and "
                f"{paths[index].name}; fix the manifest or filename order"
            )

    frames = []
    for index, (path, captured_at) in enumerate(zip(paths, resolved, strict=True)):
        content_sha256 = _file_sha256(path)
        frame_id = _content_bound_frame_id(path, captured_at, content_sha256)
        frames.append(
            SourceFrame(
                index=index,
                path=path,
                captured_at=captured_at,
                frame_id=frame_id,
                content_sha256=content_sha256,
            )
        )
    return tuple(frames)


def split_sessions(
    frames: Sequence[SourceFrame], max_gap_seconds: float
) -> tuple[tuple[SourceFrame, ...], ...]:
    """Split whenever the gap between adjacent frames exceeds ``max_gap_seconds``."""

    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be nonnegative")
    if not frames:
        return ()
    sessions: list[list[SourceFrame]] = [[frames[0]]]
    for previous, current in pairwise(frames):
        gap = (current.captured_at - previous.captured_at).total_seconds()
        if gap < 0:
            raise TimestampResolutionError("frames must be ordered by captured_at")
        if gap > max_gap_seconds:
            sessions.append([])
        sessions[-1].append(current)
    return tuple(tuple(session) for session in sessions)


def iter_frame_chunks(
    frames: Sequence[SourceFrame], chunk_size: int
) -> Iterator[tuple[SourceFrame, ...]]:
    """Yield fixed-size chunks and always flush the final nonempty tail."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    for start in range(0, len(frames), chunk_size):
        yield tuple(frames[start : start + chunk_size])


class JsonlLabelingCache:
    """Crash-tolerant, process-safe cache for resumable Gemini labeling."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(str(self.path))
        self._lock = threading.Lock()
        self._captions: dict[str, str] = {}
        self._groupings: dict[str, GroupingResult] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                captions, groupings, _ = self._read_locked(handle, repair=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self._captions = captions
        self._groupings = groupings

    @staticmethod
    def _store_loaded(mapping: dict[str, Any], key: str, value: Any, line: int) -> None:
        existing = mapping.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting labeling cache key {key!r} on line {line}")
        mapping[key] = value

    @classmethod
    def _decode_record(
        cls,
        record: Any,
        *,
        line_number: int,
        captions: dict[str, str],
        groupings: dict[str, GroupingResult],
    ) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"labeling cache line {line_number} must be an object")
        record_type = record.get("record_type")
        cache_key = record.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError(f"labeling cache line {line_number} has no cache_key")
        if record_type == "frame_caption":
            caption = record.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"labeling cache line {line_number} has an invalid caption")
            cls._store_loaded(captions, cache_key, caption.strip(), line_number)
        elif record_type == "action_grouping":
            grouping = grouping_from_payload(
                record.get("grouping"),
                frame_count=record.get("frame_count"),
            )
            cls._store_loaded(groupings, cache_key, grouping, line_number)
        else:
            raise ValueError(f"unknown labeling cache record type on line {line_number}")

    @classmethod
    def _read_locked(
        cls, handle: BinaryIO, *, repair: bool
    ) -> tuple[dict[str, str], dict[str, GroupingResult], int]:
        handle.seek(0)
        content = handle.read()
        if not content:
            return {}, {}, 0

        lines = content.splitlines(keepends=True)
        captions: dict[str, str] = {}
        groupings: dict[str, GroupingResult] = {}
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
                raise ValueError(f"blank labeling cache record on line {index + 1}")
            try:
                record = json.loads(stripped.decode("utf-8"))
                cls._decode_record(
                    record,
                    line_number=index + 1,
                    captions=captions,
                    groupings=groupings,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                if is_last and not terminated:
                    truncate_at = line_start
                    break
                raise ValueError(f"invalid labeling cache record on line {index + 1}") from exc

        removed = 0
        if repair and truncate_at is not None:
            removed = len(content) - truncate_at
            handle.seek(truncate_at)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        elif repair and lines and not lines[-1].endswith((b"\n", b"\r")):
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return captions, groupings, removed

    def _append(
        self,
        record: dict[str, Any],
        *,
        key: str,
        value: str | GroupingResult,
        record_type: str,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                captions, groupings, _ = self._read_locked(handle, repair=True)
                mapping: dict[str, Any]
                mapping = captions if record_type == "frame_caption" else groupings
                existing = mapping.get(key)
                if existing is not None:
                    if existing != value:
                        raise ValueError(f"conflicting labeling cache entry for {key}")
                else:
                    handle.seek(0, os.SEEK_END)
                    handle.write(serialized + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    mapping[key] = value
                self._captions = captions
                self._groupings = groupings
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def recover(self) -> int:
        """Repair an interrupted final append and return bytes removed."""

        with self._lock:
            if not self.path.exists():
                return 0
            with self.path.open("r+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    captions, groupings, removed = self._read_locked(handle, repair=True)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self._captions = captions
            self._groupings = groupings
            return removed

    def get_caption(self, key: str) -> str | None:
        with self._lock:
            return self._captions.get(key)

    def put_caption(self, key: str, caption: str) -> None:
        normalized = caption.strip()
        if not normalized:
            raise ValueError("cached caption must not be empty")
        with self._lock:
            self._append(
                {
                    "record_type": "frame_caption",
                    "cache_key": key,
                    "caption": normalized,
                },
                key=key,
                value=normalized,
                record_type="frame_caption",
            )

    def get_grouping(self, key: str) -> GroupingResult | None:
        with self._lock:
            return self._groupings.get(key)

    def put_grouping(self, key: str, grouping: GroupingResult, frame_count: int) -> None:
        with self._lock:
            self._append(
                {
                    "record_type": "action_grouping",
                    "cache_key": key,
                    "frame_count": frame_count,
                    "grouping": {
                        "actions": [
                            {
                                "start_index": action.start_index,
                                "end_index": action.end_index,
                                "caption": action.caption,
                            }
                            for action in grouping.actions
                        ],
                        "dense_context": grouping.dense_context,
                    },
                },
                key=key,
                value=grouping,
                record_type="action_grouping",
            )


class FrameLabeler:
    """Caption frames independently, then group each temporal chunk with Gemini."""

    def __init__(
        self,
        captioner: GeminiCaptioner | None = None,
        grouper: GeminiActionGrouper | None = None,
        chunk_size: int = 10,
        max_gap_seconds: float = 2.0,
        caption_workers: int = 4,
        cache_path: Path | None = None,
        cache: JsonlLabelingCache | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be nonnegative")
        if caption_workers < 1:
            raise ValueError("caption_workers must be at least 1")
        if cache_path is not None and cache is not None:
            raise ValueError("pass either cache_path or cache, not both")
        self.captioner = captioner or GeminiCaptioner()
        self.grouper = grouper or GeminiActionGrouper()
        self.chunk_size = chunk_size
        self.max_gap_seconds = max_gap_seconds
        self.caption_workers = caption_workers
        self.cache = cache or (JsonlLabelingCache(cache_path) if cache_path else None)

    def _caption_cache_key(self, frame: SourceFrame) -> str:
        return _stable_id(
            "caption",
            (
                frame.frame_id,
                self.captioner.model,
                self.captioner.prompt_version,
            ),
        )

    def _caption_frame(self, frame: SourceFrame) -> str:
        cache_key = self._caption_cache_key(frame)
        if self.cache is not None:
            cached = self.cache.get_caption(cache_key)
            if cached is not None:
                return cached
        caption = self.captioner.caption_frame(frame.path).strip()
        if not caption:
            raise ValueError(f"captioner returned an empty caption for {frame.path}")
        if self.cache is not None:
            self.cache.put_caption(cache_key, caption)
        return caption

    def _caption_chunk(self, chunk: Sequence[SourceFrame]) -> tuple[str, ...]:
        if self.caption_workers == 1 or len(chunk) == 1:
            return tuple(self._caption_frame(frame) for frame in chunk)
        with ThreadPoolExecutor(
            max_workers=min(self.caption_workers, len(chunk)),
            thread_name_prefix="nybble-caption",
        ) as executor:
            return tuple(executor.map(self._caption_frame, chunk))

    def label_directory(
        self,
        folder: Path,
        manifest_path: Path | None = None,
        fps: float | None = None,
        start_time: Any | None = None,
        recursive: bool = False,
        finalize_tail: bool = False,
    ) -> LabelingResult:
        if type(finalize_tail) is not bool:
            raise TypeError("finalize_tail must be a boolean")
        frames = discover_frames(
            folder=folder,
            manifest_path=manifest_path,
            fps=fps,
            start_time=start_time,
            recursive=recursive,
        )
        return self.label_frames(frames, finalize_tail=finalize_tail)

    def label_frames(
        self, frames: Sequence[SourceFrame], *, finalize_tail: bool = False
    ) -> LabelingResult:
        if type(finalize_tail) is not bool:
            raise TypeError("finalize_tail must be a boolean")
        frames = tuple(_bind_current_image_content(frame) for frame in frames)
        if tuple(frame.index for frame in frames) != tuple(range(len(frames))):
            raise ValueError("source frames must have contiguous zero-based indices")

        labeled_frames: list[FrameLabel] = []
        labeled_chunks: list[LabeledChunk] = []
        sessions = split_sessions(frames, self.max_gap_seconds)
        for session_index, session in enumerate(sessions):
            # Session ownership must survive corrected pixels at the same capture
            # path and time so a relabel can replace stale journal actions. Frame
            # and chunk identities separately bind the actual image bytes.
            session_id = _stable_id(
                "session",
                (
                    str(session[0].path.expanduser().resolve()),
                    session[0].captured_at.isoformat(),
                ),
            )
            for chunk in iter_frame_chunks(session, self.chunk_size):
                chunk_id = _stable_id(
                    "chunk",
                    (
                        session_id,
                        self.captioner.model,
                        self.captioner.prompt_version,
                        self.grouper.model,
                        self.grouper.prompt_version,
                        *(frame.frame_id for frame in chunk),
                    ),
                )
                available_at = chunk[-1].captured_at
                captions = self._caption_chunk(chunk)
                chunk_labels = tuple(
                    FrameLabel(
                        index=frame.index,
                        path=frame.path,
                        captured_at=frame.captured_at,
                        caption=caption,
                        frame_id=frame.frame_id,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        available_at=available_at,
                        model=self.captioner.model,
                        prompt_version=self.captioner.prompt_version,
                    )
                    for frame, caption in zip(chunk, captions, strict=True)
                )
                local_grouping = (
                    self.cache.get_grouping(chunk_id) if self.cache is not None else None
                )
                if local_grouping is None:
                    local_grouping = self.grouper.group(chunk_labels)
                    if self.cache is not None:
                        self.cache.put_grouping(
                            chunk_id, local_grouping, frame_count=len(chunk_labels)
                        )
                global_grouping = local_grouping.shifted(chunk[0].index)
                labeled_frames.extend(chunk_labels)
                labeled_chunks.append(
                    LabeledChunk(
                        chunk_id=chunk_id,
                        session_id=session_id,
                        start_index=chunk[0].index,
                        end_index=chunk[-1].index,
                        available_at=available_at,
                        closed=(
                            len(chunk) == self.chunk_size
                            or session_index < len(sessions) - 1
                            or finalize_tail
                        ),
                        frames=chunk_labels,
                        grouping=global_grouping,
                    )
                )
        return LabelingResult(
            frames=tuple(labeled_frames),
            chunks=tuple(labeled_chunks),
        )


__all__ = [
    "FrameLabeler",
    "JsonlLabelingCache",
    "TimestampResolutionError",
    "discover_frames",
    "iter_frame_chunks",
    "natural_sort_key",
    "split_sessions",
]
