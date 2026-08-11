"""Safe construction and admission rules for PowerNap reflection memory."""

from __future__ import annotations

import json

from nybble.powernap.rewards import RewardResult


def build_memory_text(
    *,
    context: str,
    revision: str,
    prediction: str,
    outcome: str,
    accuracy: float,
) -> str:
    """Serialize one completed historical case without duplicating XML tags."""

    return json.dumps(
        {
            "observed_context": context,
            "revision": revision,
            "predicted_actions": prediction,
            "observed_outcome": outcome,
            "accuracy": accuracy,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def should_index_memory(result: RewardResult, minimum_accuracy: float) -> bool:
    if not 0 <= minimum_accuracy <= 1:
        raise ValueError("minimum_accuracy must be between 0 and 1")
    return result.valid and result.accuracy > 0 and result.accuracy >= minimum_accuracy


__all__ = ["build_memory_text", "should_index_memory"]
