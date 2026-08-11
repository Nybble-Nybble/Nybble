"""Deterministic token budgeting for retrieved PowerNap memories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _token_count(tokenizer: Any, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    try:
        return len(tokens)
    except TypeError as exc:
        raise TypeError("tokenizer.encode must return a sized token sequence") from exc


def _number(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def pack_retrieved_memories(
    hits: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Fit whole memories under a token cap, dropping weakest and oldest first.

    The caller adds the returned data to an already-complete phase prompt. This
    function never truncates the observed context, images, or phase instruction.
    Whole documents are retained so JSON-backed memories are never cut into a
    misleading partial example. Kept documents preserve their incoming MMR order.
    """

    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens must be a nonnegative integer")
    if max_tokens == 0 or not hits:
        return ""
    if not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer must provide encode")

    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            raise TypeError("retrieval hits must be mappings")
        text = hit.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("retrieval hit text must be a nonempty string")
        if _token_count(tokenizer, text.strip()) <= max_tokens:
            candidates.append((index, hit))

    def render() -> str:
        return "\n\n".join(str(hit["text"]).strip() for _index, hit in candidates)

    while candidates:
        packed = render()
        if _token_count(tokenizer, packed) <= max_tokens:
            return packed

        # Lower semantic score is less useful. For equal scores, an older event
        # is less useful. A later MMR rank loses the final deterministic tie.
        drop_position = min(
            range(len(candidates)),
            key=lambda position: (
                _number(candidates[position][1].get("score"), default=float("-inf")),
                _number(candidates[position][1].get("event_ts"), default=float("-inf")),
                -candidates[position][0],
            ),
        )
        candidates.pop(drop_position)
    return ""


__all__ = ["pack_retrieved_memories"]
