"""Current tinker-cookbook environment for PowerNap's three-turn rollout."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    StepResult,
    Trajectory,
)

from nybble.powernap.memory import build_memory_text, should_index_memory
from nybble.powernap.prompts import (
    append_actions,
    append_revise,
    build_think_messages,
    retrieval_query,
)
from nybble.powernap.rendering import decode_model_turn, parse_tagged_block
from nybble.powernap.retrieval_budget import pack_retrieved_memories
from nybble.powernap.rewards import GeminiRewardJudge, RewardResult
from nybble.powernap.types import RolloutSample


def _mmr_select(hits: Sequence[dict], top_m: int, alpha: float) -> list[dict]:
    """Small adapter that keeps the optional environment independent of internals."""
    if not hits:
        return []
    from nybble.retrieval import mmr_select

    items = [(hit["text"], float(hit["score"]), hit) for hit in hits]
    return [item[2] for item in mmr_select(items, top_m=top_m, alpha=alpha)]


class LongNAPEnv(Env):
    THINK = 0
    REVISE = 1
    ACTIONS = 2

    def __init__(
        self,
        sample: RolloutSample,
        renderer: Renderer,
        tokenizer: Any,
        retriever: Any | None,
        *,
        retrieval_top_k: int,
        retrieval_mmr_k: int,
        retrieval_max_tokens: int,
        retrieval_mmr_alpha: float,
        retrieval_time_decay: float,
        effort: float,
    ) -> None:
        self.sample = sample
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_mmr_k = retrieval_mmr_k
        self.retrieval_max_tokens = retrieval_max_tokens
        self.retrieval_mmr_alpha = retrieval_mmr_alpha
        self.retrieval_time_decay = retrieval_time_decay
        self.effort = effort
        self.phase = self.THINK
        self.messages: list[dict] = []
        self.rationale = ""
        self.retrieved = ""
        self.revision = ""
        self.actions_text = ""

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _render(self, messages: list[dict]) -> tinker.ModelInput:
        """Inkling's renderer accepts effort; retain a safe fallback for custom renderers."""
        try:
            return self.renderer.build_generation_prompt(messages, effort=self.effort)
        except TypeError:
            return self.renderer.build_generation_prompt(messages)

    def _retrieve(self, rationale: str) -> str:
        if self.retriever is None or self.retrieval_max_tokens == 0:
            return ""
        hits = self.retriever.query(
            retrieval_query(rationale, self.sample.context_text),
            k=self.retrieval_top_k,
            cutoff_ts=self.sample.cutoff_ts,
            namespaces=("train",),
            time_decay_lambda=self.retrieval_time_decay,
        )
        selected = _mmr_select(hits, self.retrieval_mmr_k, self.retrieval_mmr_alpha)
        return pack_retrieved_memories(selected, self.tokenizer, self.retrieval_max_tokens)

    def _invalid_step(self, error: str, detail: str = "") -> StepResult:
        self.actions_text = ""
        stop_metric = "stop/max_tokens" if error == "max_tokens" else "stop/parse_error"
        metrics = {stop_metric: 1.0, "valid_turn": 0.0}
        if error == "empty_turn":
            metrics["empty_turn"] = 1.0
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={"decode_error": detail or error},
        )

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        self.messages = build_think_messages(self.sample.context_text, self.sample.image_paths)
        return self._render(self.messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        decoded = decode_model_turn(
            self.renderer,
            action,
            extra,
            get_text_content,
        )
        if not decoded.valid:
            return self._invalid_step(decoded.error or "parse_error")
        text = decoded.text
        if self.phase == self.THINK:
            rationale = parse_tagged_block(text, "rationale")
            if not rationale.valid:
                return self._invalid_step("parse_error", rationale.error or "invalid rationale")
            self.rationale = text
            self.retrieved = self._retrieve(rationale.content)
            self.messages = append_revise(copy.deepcopy(self.messages), text, self.retrieved)
            self.phase = self.REVISE
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self._render(self.messages),
                next_stop_condition=self.stop_condition,
                metrics={"phase/think": 1.0, "tokens": float(len(action))},
                logs={},
            )

        if self.phase == self.REVISE:
            revision = parse_tagged_block(text, "revise")
            if not revision.valid:
                return self._invalid_step("parse_error", revision.error or "invalid revision")
            self.revision = text
            self.messages = append_actions(
                copy.deepcopy(self.messages), text, self.sample.future_len
            )
            self.phase = self.ACTIONS
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self._render(self.messages),
                next_stop_condition=self.stop_condition,
                metrics={"phase/revise": 1.0, "tokens": float(len(action))},
                logs={},
            )

        self.actions_text = text
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"phase/actions": 1.0, "tokens": float(len(action))},
            logs={},
        )


@dataclass
class LongNAPGroupBuilder(EnvGroupBuilder):
    sample: RolloutSample
    renderer: Renderer
    tokenizer: Any
    retriever: Any | None
    reward_judge: GeminiRewardJudge
    num_envs: int
    retrieval_top_k: int = 10
    retrieval_mmr_k: int = 5
    retrieval_max_tokens: int = 4096
    retrieval_mmr_alpha: float = 0.5
    retrieval_time_decay: float = 0.5
    retrieval_min_accuracy: float = 0.5
    effort: float = 0.0
    score_results: list[RewardResult] = field(default_factory=list, init=False)
    retriever_candidates: list[dict | None] = field(default_factory=list, init=False)

    async def make_envs(self) -> Sequence[Env]:
        return [
            LongNAPEnv(
                self.sample,
                self.renderer,
                self.tokenizer,
                self.retriever,
                retrieval_top_k=self.retrieval_top_k,
                retrieval_mmr_k=self.retrieval_mmr_k,
                retrieval_max_tokens=self.retrieval_max_tokens,
                retrieval_mmr_alpha=self.retrieval_mmr_alpha,
                retrieval_time_decay=self.retrieval_time_decay,
                effort=self.effort,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        del trajectory_group
        longnap_envs = [env for env in env_group if isinstance(env, LongNAPEnv)]
        if len(longnap_envs) != len(env_group):
            raise TypeError("LongNAPGroupBuilder received an unexpected environment type")

        self.score_results = await self.reward_judge.score_many(
            [env.actions_text for env in longnap_envs],
            self.sample.solution_text,
            self.sample.future_len,
        )
        self.retriever_candidates = [
            {
                "text": build_memory_text(
                    context=self.sample.context_text,
                    revision=env.revision,
                    prediction=env.actions_text,
                    outcome=self.sample.solution_text,
                    accuracy=result.accuracy,
                ),
                "event_ts": self.sample.cutoff_ts,
                "visible_after_ts": self.sample.target_visible_after_ts,
                "metadata": {
                    "sample_id": self.sample.id,
                    "target_end_ts": self.sample.target_end_ts,
                    "target_visible_after_ts": self.sample.target_visible_after_ts,
                },
            }
            if env.revision.strip() and env.actions_text.strip()
            else None
            for env, result in zip(longnap_envs, self.score_results, strict=True)
        ]
        return [
            (
                result.reward,
                {
                    "accuracy": result.accuracy,
                    "formatting": result.formatting,
                    "valid": float(result.valid),
                },
            )
            for result in self.score_results
        ]

    def add_winner_to_retriever(self, rewards: Sequence[float]) -> None:
        if self.retriever is None or not rewards or not self.retriever_candidates:
            return
        winner = max(range(len(rewards)), key=lambda index: rewards[index])
        if rewards[winner] <= 0 or not should_index_memory(
            self.score_results[winner], self.retrieval_min_accuracy
        ):
            return
        candidate = self.retriever_candidates[winner]
        if not candidate:
            return
        self.retriever.add(
            candidate["text"],
            event_ts=candidate["event_ts"],
            visible_after_ts=candidate["visible_after_ts"],
            namespace="train",
            metadata={**candidate["metadata"], "utility": float(rewards[winner])},
        )

    def logging_tags(self) -> list[str]:
        return ["nybble", "powernap"]
