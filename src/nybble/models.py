"""Versioned, immutable data models shared across the Nybble pipeline."""

# Python 3.9 compatibility requires Optional instead of PEP 604 union syntax.
# ruff: noqa: UP045

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Optional

SCHEMA_VERSION = 1


def _require_exact_type(value: Any, expected: type, name: str) -> Any:
    if type(value) is not expected:
        raise TypeError(f"{name} must be {expected.__name__}, got {type(value).__name__}")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    _require_exact_type(value, str, name)
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_string(value: Any, name: str) -> str:
    return _require_exact_type(value, str, name)


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_string(value, name)


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite, non-negative timestamp")
    return result


def _schema_version(value: Any) -> int:
    _require_exact_type(value, int, "schema_version")
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {value}; expected {SCHEMA_VERSION}")
    return value


def _freeze_json(value: Any, path: str = "metadata") -> Any:
    """Validate JSON data and recursively make it immutable."""

    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze_json(value, name)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _strict_payload(
    payload: Any,
    *,
    record_type: str,
    required: Iterable[str],
    optional: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    data = dict(payload)
    allowed = set(required) | set(optional) | {"record_type"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {record_type} fields: {sorted(unknown)}")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"missing {record_type} fields: {sorted(missing)}")
    actual_type = data.pop("record_type", record_type)
    if actual_type != record_type:
        raise ValueError(f"record_type must be {record_type!r}, got {actual_type!r}")
    return data


def event_order_key(event: ActionEvent) -> tuple[float, str]:
    """Canonical ordering for events, including ties at the same timestamp."""

    if not isinstance(event, ActionEvent):
        raise TypeError("event must be an ActionEvent")
    return event.start_ts, event.id


@dataclass(frozen=True)
class FrameObservation:
    """A captured frame and its optional Gemini-derived description."""

    RECORD_TYPE: ClassVar[str] = "frame_observation"

    id: str
    timestamp: float
    source: str
    image_path: str
    text: str = ""
    dense_context: str = ""
    chunk_id: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_nonempty_string(self.id, "id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp, "timestamp"))
        object.__setattr__(self, "source", _require_nonempty_string(self.source, "source"))
        object.__setattr__(
            self, "image_path", _require_nonempty_string(self.image_path, "image_path")
        )
        object.__setattr__(self, "text", _require_string(self.text, "text"))
        object.__setattr__(
            self,
            "dense_context",
            _require_string(self.dense_context, "dense_context"),
        )
        object.__setattr__(self, "chunk_id", _optional_string(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "model", _optional_string(self.model, "model"))
        object.__setattr__(
            self,
            "prompt_version",
            _optional_string(self.prompt_version, "prompt_version"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.schema_version,
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "image_path": self.image_path,
            "text": self.text,
            "dense_context": self.dense_context,
            "chunk_id": self.chunk_id,
            "provenance": _thaw_json(self.provenance),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrameObservation:
        data = _strict_payload(
            payload,
            record_type=cls.RECORD_TYPE,
            required=("schema_version", "id", "timestamp", "source", "image_path"),
            optional=(
                "text",
                "dense_context",
                "chunk_id",
                "provenance",
                "model",
                "prompt_version",
                "metadata",
            ),
        )
        return cls(**data)


@dataclass(frozen=True)
class ActionEvent:
    """A completed user action derived from one or more frame observations."""

    RECORD_TYPE: ClassVar[str] = "action_event"

    id: str
    start_ts: float
    end_ts: float
    available_at: float
    source: str
    text: str
    dense_context: str = ""
    start_frame: Optional[str] = None
    end_frame: Optional[str] = None
    image_path: Optional[str] = None
    chunk_id: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_nonempty_string(self.id, "id"))
        object.__setattr__(self, "start_ts", _timestamp(self.start_ts, "start_ts"))
        object.__setattr__(self, "end_ts", _timestamp(self.end_ts, "end_ts"))
        object.__setattr__(self, "available_at", _timestamp(self.available_at, "available_at"))
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        if self.available_at < self.end_ts:
            raise ValueError("available_at must be greater than or equal to end_ts")
        object.__setattr__(self, "source", _require_nonempty_string(self.source, "source"))
        object.__setattr__(self, "text", _require_nonempty_string(self.text, "text"))
        object.__setattr__(
            self,
            "dense_context",
            _require_string(self.dense_context, "dense_context"),
        )
        for name in ("start_frame", "end_frame", "image_path", "chunk_id"):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "model", _optional_string(self.model, "model"))
        object.__setattr__(
            self,
            "prompt_version",
            _optional_string(self.prompt_version, "prompt_version"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.schema_version,
            "id": self.id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "available_at": self.available_at,
            "source": self.source,
            "text": self.text,
            "dense_context": self.dense_context,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "image_path": self.image_path,
            "chunk_id": self.chunk_id,
            "provenance": _thaw_json(self.provenance),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionEvent:
        data = _strict_payload(
            payload,
            record_type=cls.RECORD_TYPE,
            required=(
                "schema_version",
                "id",
                "start_ts",
                "end_ts",
                "available_at",
                "source",
                "text",
            ),
            optional=(
                "dense_context",
                "start_frame",
                "end_frame",
                "image_path",
                "chunk_id",
                "provenance",
                "model",
                "prompt_version",
                "metadata",
            ),
        )
        return cls(**data)


@dataclass(frozen=True)
class TrainingSample:
    """An exact, bounded past/future window used by the trainer."""

    RECORD_TYPE: ClassVar[str] = "training_sample"

    id: str
    past_events: tuple[ActionEvent, ...]
    future_events: tuple[ActionEvent, ...]
    cutoff_ts: float
    start_ts: float
    end_ts: float
    context_text: str
    solution_text: str
    image_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_nonempty_string(self.id, "id"))
        for name in ("past_events", "future_events"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{name} must be a sequence of ActionEvent objects")
            events = tuple(value)
            if not events:
                raise ValueError(f"{name} must not be empty")
            if not all(isinstance(event, ActionEvent) for event in events):
                raise TypeError(f"{name} must contain only ActionEvent objects")
            if events != tuple(sorted(events, key=event_order_key)):
                raise ValueError(f"{name} must use canonical timestamp and ID order")
            object.__setattr__(self, name, events)

        selected_ids = self.past_event_ids + self.future_event_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("past_events and future_events must have unique, disjoint IDs")
        selected_events = self.past_events + self.future_events
        stream_keys = {
            (
                event.source,
                event.provenance.get("session_id").strip()
                if isinstance(event.provenance.get("session_id"), str)
                else "",
            )
            for event in selected_events
        }
        if len(stream_keys) != 1:
            raise ValueError("past_events and future_events must share one source session")
        if any(event.metadata.get("provisional") is True for event in selected_events):
            raise ValueError("training samples cannot contain provisional tail actions")

        object.__setattr__(self, "cutoff_ts", _timestamp(self.cutoff_ts, "cutoff_ts"))
        object.__setattr__(self, "start_ts", _timestamp(self.start_ts, "start_ts"))
        object.__setattr__(self, "end_ts", _timestamp(self.end_ts, "end_ts"))
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be greater than or equal to start_ts")
        if self.start_ts != self.past_events[0].start_ts:
            raise ValueError("start_ts must match the first selected past event")
        if self.end_ts != self.future_events[-1].end_ts:
            raise ValueError("end_ts must match the last selected future event")
        if self.cutoff_ts < max(event.available_at for event in self.past_events):
            raise ValueError("cutoff_ts cannot precede a selected past event's availability")
        if any(event.start_ts < self.cutoff_ts for event in self.future_events):
            raise ValueError("future event start_ts cannot precede cutoff_ts")
        past_chunks = {event.chunk_id for event in self.past_events if event.chunk_id is not None}
        future_chunks = {
            event.chunk_id for event in self.future_events if event.chunk_id is not None
        }
        shared_chunks = past_chunks & future_chunks
        if shared_chunks:
            raise ValueError("past_events and future_events cannot cross a shared labeling chunk")

        object.__setattr__(
            self, "context_text", _require_nonempty_string(self.context_text, "context_text")
        )
        object.__setattr__(
            self,
            "solution_text",
            _require_nonempty_string(self.solution_text, "solution_text"),
        )
        if not isinstance(self.image_paths, (list, tuple)):
            raise TypeError("image_paths must be a sequence")
        image_paths = tuple(
            _require_nonempty_string(path, "image_paths[]") for path in self.image_paths
        )
        object.__setattr__(self, "image_paths", image_paths)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    @property
    def past_event_ids(self) -> tuple[str, ...]:
        return tuple(event.id for event in self.past_events)

    @property
    def future_event_ids(self) -> tuple[str, ...]:
        return tuple(event.id for event in self.future_events)

    @property
    def future_len(self) -> int:
        return len(self.future_events)

    @property
    def past_len(self) -> int:
        return len(self.past_events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.schema_version,
            "id": self.id,
            "past_events": [event.to_dict() for event in self.past_events],
            "future_events": [event.to_dict() for event in self.future_events],
            "cutoff_ts": self.cutoff_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "context_text": self.context_text,
            "solution_text": self.solution_text,
            "image_paths": list(self.image_paths),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingSample:
        data = _strict_payload(
            payload,
            record_type=cls.RECORD_TYPE,
            required=(
                "schema_version",
                "id",
                "past_events",
                "future_events",
                "cutoff_ts",
                "start_ts",
                "end_ts",
                "context_text",
                "solution_text",
            ),
            optional=("image_paths", "metadata"),
        )
        data["past_events"] = tuple(ActionEvent.from_dict(item) for item in data["past_events"])
        data["future_events"] = tuple(ActionEvent.from_dict(item) for item in data["future_events"])
        return cls(**data)


@dataclass(frozen=True)
class Prediction:
    """A persisted next-action prediction and its generation trace."""

    RECORD_TYPE: ClassVar[str] = "prediction"

    id: str
    created_at: float
    cutoff_ts: float
    actions: tuple[str, ...]
    raw_text: str
    rationale: str = ""
    revision: str = ""
    retrieved: tuple[str, ...] = ()
    context_event_ids: tuple[str, ...] = ()
    model: Optional[str] = None
    checkpoint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_nonempty_string(self.id, "id"))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "cutoff_ts", _timestamp(self.cutoff_ts, "cutoff_ts"))
        if self.created_at < self.cutoff_ts:
            raise ValueError("created_at must be greater than or equal to cutoff_ts")

        for name, allow_empty in (
            ("actions", False),
            ("retrieved", True),
            ("context_event_ids", True),
        ):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{name} must be a sequence")
            values = tuple(_require_nonempty_string(item, f"{name}[]") for item in value)
            if not allow_empty and not values:
                raise ValueError(f"{name} must not be empty")
            if name == "context_event_ids" and len(values) != len(set(values)):
                raise ValueError("context_event_ids must be unique")
            object.__setattr__(self, name, values)

        object.__setattr__(self, "raw_text", _require_string(self.raw_text, "raw_text"))
        object.__setattr__(self, "rationale", _require_string(self.rationale, "rationale"))
        object.__setattr__(self, "revision", _require_string(self.revision, "revision"))
        object.__setattr__(self, "model", _optional_string(self.model, "model"))
        object.__setattr__(self, "checkpoint", _optional_string(self.checkpoint, "checkpoint"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.schema_version,
            "id": self.id,
            "created_at": self.created_at,
            "cutoff_ts": self.cutoff_ts,
            "actions": list(self.actions),
            "raw_text": self.raw_text,
            "rationale": self.rationale,
            "revision": self.revision,
            "retrieved": list(self.retrieved),
            "context_event_ids": list(self.context_event_ids),
            "model": self.model,
            "checkpoint": self.checkpoint,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Prediction:
        data = _strict_payload(
            payload,
            record_type=cls.RECORD_TYPE,
            required=(
                "schema_version",
                "id",
                "created_at",
                "cutoff_ts",
                "actions",
                "raw_text",
            ),
            optional=(
                "rationale",
                "revision",
                "retrieved",
                "context_event_ids",
                "model",
                "checkpoint",
                "metadata",
            ),
        )
        return cls(**data)


__all__ = [
    "SCHEMA_VERSION",
    "ActionEvent",
    "FrameObservation",
    "Prediction",
    "TrainingSample",
    "event_order_key",
]
