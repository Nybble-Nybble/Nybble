"""Frame discovery and Gemini-backed semantic action labeling.

Exports are loaded lazily because the labeling pipeline and Gemini adapter refer to
each other's narrow schema and protocol surfaces.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ActionSpan",
    "FrameLabel",
    "FrameLabeler",
    "GroupingResult",
    "JsonlLabelingCache",
    "LabeledChunk",
    "LabelingResult",
    "SourceFrame",
    "TimestampResolutionError",
    "discover_frames",
    "iter_frame_chunks",
    "natural_sort_key",
    "split_sessions",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActionSpan": ("nybble.labeling.schema", "ActionSpan"),
    "FrameLabel": ("nybble.labeling.schema", "FrameLabel"),
    "GroupingResult": ("nybble.labeling.schema", "GroupingResult"),
    "LabeledChunk": ("nybble.labeling.schema", "LabeledChunk"),
    "LabelingResult": ("nybble.labeling.schema", "LabelingResult"),
    "SourceFrame": ("nybble.labeling.schema", "SourceFrame"),
    "FrameLabeler": ("nybble.labeling.labeler", "FrameLabeler"),
    "JsonlLabelingCache": ("nybble.labeling.labeler", "JsonlLabelingCache"),
    "TimestampResolutionError": (
        "nybble.labeling.labeler",
        "TimestampResolutionError",
    ),
    "discover_frames": ("nybble.labeling.labeler", "discover_frames"),
    "iter_frame_chunks": ("nybble.labeling.labeler", "iter_frame_chunks"),
    "natural_sort_key": ("nybble.labeling.labeler", "natural_sort_key"),
    "split_sessions": ("nybble.labeling.labeler", "split_sessions"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
