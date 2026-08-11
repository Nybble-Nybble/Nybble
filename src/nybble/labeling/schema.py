"""Strict, serializable schemas for timestamped frame labels and action spans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class GroupingValidationError(ValueError):
    """Raised when Gemini returns an unsafe or ambiguous span partition."""


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class SourceFrame:
    """A naturally ordered input frame with a resolved capture timestamp."""

    index: int
    path: Path
    captured_at: datetime
    frame_id: str
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be nonnegative")
        _require_aware(self.captured_at, "captured_at")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.content_sha256 is not None and (
            len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": str(self.path),
            "captured_at": self.captured_at.isoformat(),
            "frame_id": self.frame_id,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class FrameLabel:
    """An observed-only single-frame caption and its availability boundary.

    ``available_at`` is the end of the containing grouping chunk. This prevents a
    trainer from treating a label produced with chunk-level processing as available
    before all inputs to that processing existed. Chunk-wide ``dense_context`` is
    intentionally not stored here.
    """

    index: int
    path: Path
    captured_at: datetime
    caption: str
    frame_id: str
    session_id: str
    chunk_id: str
    available_at: datetime
    model: str
    prompt_version: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be nonnegative")
        _require_aware(self.captured_at, "captured_at")
        _require_aware(self.available_at, "available_at")
        if self.available_at < self.captured_at:
            raise ValueError("available_at cannot precede captured_at")
        for field_name in (
            "caption",
            "frame_id",
            "session_id",
            "chunk_id",
            "model",
            "prompt_version",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": str(self.path),
            "captured_at": self.captured_at.isoformat(),
            "caption": self.caption,
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "available_at": self.available_at.isoformat(),
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class ActionSpan:
    """A semantic action over zero-based indices with an inclusive end index."""

    start_index: int
    end_index: int
    caption: str

    def __post_init__(self) -> None:
        if isinstance(self.start_index, bool) or not isinstance(self.start_index, int):
            raise GroupingValidationError("start_index must be an integer")
        if isinstance(self.end_index, bool) or not isinstance(self.end_index, int):
            raise GroupingValidationError("end_index must be an integer")
        if self.start_index < 0:
            raise GroupingValidationError("start_index must be nonnegative")
        if self.end_index < self.start_index:
            raise GroupingValidationError("end_index must be at least start_index")
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise GroupingValidationError("action caption must not be empty")

    def shifted(self, offset: int) -> ActionSpan:
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        return ActionSpan(
            start_index=self.start_index + offset,
            end_index=self.end_index + offset,
            caption=self.caption.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "caption": self.caption,
            "index_basis": "zero_based",
            "end_inclusive": True,
        }


@dataclass(frozen=True)
class GroupingResult:
    """A validated exact partition of a sequence of frame observations."""

    actions: tuple[ActionSpan, ...]
    dense_context: str = ""
    index_basis: str = "zero_based"
    end_inclusive: bool = True

    def __post_init__(self) -> None:
        if self.index_basis != "zero_based":
            raise GroupingValidationError("only zero_based indices are supported")
        if self.end_inclusive is not True:
            raise GroupingValidationError("end_index must be inclusive")
        if not isinstance(self.dense_context, str):
            raise GroupingValidationError("dense_context must be a string")

    def shifted(self, offset: int) -> GroupingResult:
        return GroupingResult(
            actions=tuple(action.shifted(offset) for action in self.actions),
            dense_context=self.dense_context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "dense_context": self.dense_context,
            "index_basis": self.index_basis,
            "end_inclusive": self.end_inclusive,
        }


@dataclass(frozen=True)
class LabeledChunk:
    """One idempotent labeling unit inside a temporal session."""

    chunk_id: str
    session_id: str
    start_index: int
    end_index: int
    available_at: datetime
    closed: bool
    frames: tuple[FrameLabel, ...]
    grouping: GroupingResult

    def __post_init__(self) -> None:
        _require_aware(self.available_at, "available_at")
        if not self.chunk_id or not self.session_id:
            raise ValueError("chunk_id and session_id must not be empty")
        if not self.frames:
            raise ValueError("a labeled chunk must contain at least one frame")
        if type(self.closed) is not bool:
            raise TypeError("closed must be a boolean")
        if self.start_index != self.frames[0].index:
            raise ValueError("chunk start_index does not match its first frame")
        if self.end_index != self.frames[-1].index:
            raise ValueError("chunk end_index does not match its last frame")
        validate_action_partition(
            self.grouping.actions,
            self.start_index,
            self.end_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "index_basis": "zero_based",
            "end_inclusive": True,
            "available_at": self.available_at.isoformat(),
            "closed": self.closed,
            "frames": [frame.to_dict() for frame in self.frames],
            "grouping": self.grouping.to_dict(),
        }


@dataclass(frozen=True)
class LabelingResult:
    """A complete set of frame labels and strict semantic action partitions."""

    frames: tuple[FrameLabel, ...]
    chunks: tuple[LabeledChunk, ...]

    def __post_init__(self) -> None:
        expected_indices = tuple(range(len(self.frames)))
        actual_indices = tuple(frame.index for frame in self.frames)
        if actual_indices != expected_indices:
            raise ValueError("frames must have contiguous zero-based indices")
        if self.frames:
            validate_action_partition(self.actions, 0, len(self.frames) - 1)
        elif self.chunks:
            raise ValueError("chunks cannot exist without frames")

    @property
    def actions(self) -> tuple[ActionSpan, ...]:
        return tuple(action for chunk in self.chunks for action in chunk.grouping.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "actions": [action.to_dict() for action in self.actions],
            "index_basis": "zero_based",
            "end_inclusive": True,
        }


def validate_action_partition(
    actions: Sequence[ActionSpan], start_index: int, end_index: int
) -> None:
    """Reject gaps, overlaps, reordering, and out-of-range action spans."""

    if end_index < start_index:
        if actions:
            raise GroupingValidationError("empty ranges cannot contain actions")
        return
    if not actions:
        raise GroupingValidationError("actions must cover the input range")

    expected_start = start_index
    for action in actions:
        if action.start_index != expected_start:
            if action.start_index < expected_start:
                raise GroupingValidationError("action ranges overlap or are out of order")
            raise GroupingValidationError("action ranges leave an uncovered gap")
        if action.end_index > end_index:
            raise GroupingValidationError("action range exceeds the input range")
        expected_start = action.end_index + 1

    if expected_start != end_index + 1:
        raise GroupingValidationError("action ranges leave an uncovered tail")


def grouping_from_payload(payload: Any, frame_count: int) -> GroupingResult:
    """Parse and strictly validate a Gemini structured-grouping payload."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if not isinstance(payload, dict):
        raise GroupingValidationError("grouping response must be a JSON object")

    allowed_top_level = {"actions", "dense_context"}
    unknown = set(payload) - allowed_top_level
    if unknown:
        raise GroupingValidationError(
            "grouping response contains unknown fields: " + ", ".join(sorted(unknown))
        )
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise GroupingValidationError("actions must be a JSON array")
    dense_context = payload.get("dense_context", "")
    if not isinstance(dense_context, str):
        raise GroupingValidationError("dense_context must be a string")

    actions = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            raise GroupingValidationError(f"action {index} must be a JSON object")
        if set(raw_action) != {"start_index", "end_index", "caption"}:
            raise GroupingValidationError(
                f"action {index} must contain only start_index, end_index, and caption"
            )
        actions.append(
            ActionSpan(
                start_index=raw_action["start_index"],
                end_index=raw_action["end_index"],
                caption=raw_action["caption"].strip()
                if isinstance(raw_action["caption"], str)
                else raw_action["caption"],
            )
        )

    if frame_count == 0:
        if actions:
            raise GroupingValidationError("no actions are allowed for zero frames")
    else:
        validate_action_partition(actions, 0, frame_count - 1)
    return GroupingResult(
        actions=tuple(actions),
        dense_context=dense_context.strip(),
    )


__all__ = [
    "ActionSpan",
    "FrameLabel",
    "GroupingResult",
    "GroupingValidationError",
    "LabeledChunk",
    "LabelingResult",
    "SourceFrame",
    "grouping_from_payload",
    "validate_action_partition",
]
