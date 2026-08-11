"""Network-independent validation for rendered policy turns."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodedTurn:
    text: str
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class TaggedBlock:
    raw: str
    content: str
    valid: bool
    error: str | None = None


def parse_tagged_block(text: str, expected_tag: str) -> TaggedBlock:
    """Validate one exact, plain-text XML phase block."""

    if not isinstance(text, str) or not text.strip():
        return TaggedBlock("", "", False, "empty tagged turn")
    raw = text.strip()
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return TaggedBlock(raw, "", False, f"invalid XML: {exc}")
    if root.tag != expected_tag:
        return TaggedBlock(raw, "", False, f"root must be <{expected_tag}>")
    if root.attrib or list(root):
        return TaggedBlock(raw, "", False, "phase block cannot have attributes or children")
    content = " ".join((root.text or "").split())
    if not content:
        return TaggedBlock(raw, "", False, "phase block cannot be empty")
    return TaggedBlock(raw, content, True)


def decode_model_turn(
    renderer: Any,
    action: Any,
    extra: Mapping[str, Any] | None,
    get_text_content: Callable[[Any], str],
) -> DecodedTurn:
    """Decode one turn and reject length stops or malformed renderer output."""

    if extra and extra.get("stop_reason") == "length":
        return DecodedTurn("", "max_tokens")
    message, termination = renderer.parse_response(action)
    if not getattr(termination, "is_clean", False):
        return DecodedTurn("", "parse_error")
    text = get_text_content(message).strip()
    if not text:
        return DecodedTurn("", "empty_turn")
    return DecodedTurn(text)


__all__ = ["DecodedTurn", "TaggedBlock", "decode_model_turn", "parse_tagged_block"]
