"""Network-independent runtime records for PowerNap rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutSample:
    id: str
    context_text: str
    solution_text: str
    cutoff_ts: float
    target_end_ts: float
    future_len: int
    past_event_ids: tuple[str, ...]
    future_event_ids: tuple[str, ...]
    image_paths: tuple[str, ...] = ()
    target_visible_after_ts: float | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.context_text or not self.solution_text:
            raise ValueError("rollout sample id, context, and solution are required")
        if self.future_len < 1 or self.future_len != len(self.future_event_ids):
            raise ValueError("future_len must match future_event_ids")
        if set(self.past_event_ids) & set(self.future_event_ids):
            raise ValueError("past and future event IDs cannot overlap")
        if self.target_end_ts < self.cutoff_ts:
            raise ValueError("target_end_ts cannot precede cutoff_ts")
        visible_after = (
            self.target_end_ts
            if self.target_visible_after_ts is None
            else self.target_visible_after_ts
        )
        if visible_after < self.target_end_ts:
            raise ValueError("target_visible_after_ts cannot precede target_end_ts")
        object.__setattr__(self, "target_visible_after_ts", visible_after)
        if not isinstance(self.image_paths, tuple) or not all(
            isinstance(path, str) and path.strip() for path in self.image_paths
        ):
            raise ValueError("image_paths must be a tuple of nonempty strings")


@dataclass(frozen=True)
class PredictionTrace:
    rationale: str
    retrieved: str
    revision: str
    raw_actions: str
    actions: tuple[str, ...]
