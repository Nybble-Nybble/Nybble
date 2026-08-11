"""Adapters that turn Gemini labeling output into durable Nybble records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nybble.labeling.schema import LabelingResult
from nybble.models import ActionEvent, FrameObservation, TrainingSample
from nybble.powernap.types import RolloutSample


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        value = part.encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"{prefix}_{digest.hexdigest()[:24]}"


def labeling_result_to_records(
    result: LabelingResult,
    *,
    source: str,
    grouping_model: str,
    grouping_prompt_version: str,
) -> tuple[tuple[FrameObservation, ...], tuple[ActionEvent, ...]]:
    """Convert strict frame/chunk labels into versioned causal records.

    The grouping request sees every caption in its chunk, so each action label is
    considered available only at that chunk's final capture time. Chunk-wide dense
    context is attached only to the final action to avoid repeating it in prompts.
    """

    if not source.strip():
        raise ValueError("source must not be empty")
    if not grouping_model.strip() or not grouping_prompt_version.strip():
        raise ValueError("grouping model and prompt version must not be empty")

    observations = tuple(
        FrameObservation(
            id=frame.frame_id,
            timestamp=frame.captured_at.timestamp(),
            source=source,
            image_path=str(frame.path),
            text=frame.caption,
            chunk_id=frame.chunk_id,
            provenance={
                "session_id": frame.session_id,
                "frame_index": frame.index,
                "available_at": frame.available_at.timestamp(),
            },
            model=frame.model,
            prompt_version=frame.prompt_version,
        )
        for frame in result.frames
    )

    frames_by_index = {frame.index: frame for frame in result.frames}
    actions = []
    for chunk in result.chunks:
        for span in chunk.grouping.actions:
            first = frames_by_index[span.start_index]
            last = frames_by_index[span.end_index]
            frame_ids = tuple(
                frames_by_index[index].frame_id
                for index in range(span.start_index, span.end_index + 1)
            )
            action_id = _stable_id(
                "action",
                (
                    chunk.chunk_id,
                    str(span.start_index),
                    str(span.end_index),
                    span.caption.strip(),
                    grouping_model,
                    grouping_prompt_version,
                ),
            )
            actions.append(
                ActionEvent(
                    id=action_id,
                    start_ts=first.captured_at.timestamp(),
                    end_ts=last.captured_at.timestamp(),
                    available_at=chunk.available_at.timestamp(),
                    source=source,
                    text=span.caption.strip(),
                    dense_context=(
                        chunk.grouping.dense_context.strip()
                        if span.end_index == chunk.end_index
                        else ""
                    ),
                    start_frame=first.frame_id,
                    end_frame=last.frame_id,
                    image_path=str(last.path),
                    chunk_id=chunk.chunk_id,
                    provenance={
                        "session_id": chunk.session_id,
                        "frame_indices": [span.start_index, span.end_index],
                        "frame_ids": list(frame_ids),
                        "caption_model": first.model,
                        "caption_prompt_version": first.prompt_version,
                    },
                    model=grouping_model,
                    prompt_version=grouping_prompt_version,
                    metadata={
                        "index_basis": "zero_based",
                        "end_inclusive": True,
                        "provisional": not chunk.closed,
                    },
                )
            )
    return observations, tuple(actions)


def to_rollout_sample(sample: TrainingSample) -> RolloutSample:
    """Project a durable training sample into the network runtime record."""

    return RolloutSample(
        id=sample.id,
        context_text=sample.context_text,
        solution_text=sample.solution_text,
        cutoff_ts=sample.cutoff_ts,
        target_end_ts=sample.end_ts,
        future_len=sample.future_len,
        past_event_ids=sample.past_event_ids,
        future_event_ids=sample.future_event_ids,
        image_paths=sample.image_paths,
        target_visible_after_ts=max(event.available_at for event in sample.future_events),
    )


def write_json_atomic(path: Path, value: object) -> Path:
    """Write one JSON artifact durably without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Replace a JSONL artifact durably after every record has serialized."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


__all__ = [
    "labeling_result_to_records",
    "to_rollout_sample",
    "write_json_atomic",
    "write_jsonl_atomic",
]
