"""Checkpoint inference with the same prompts, context, and retrieval as training."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nybble.config import PowerNapConfig
from nybble.powernap.checkpoints import CheckpointRecord
from nybble.powernap.prompts import (
    append_actions,
    append_revise,
    build_think_messages,
    retrieval_query,
    validate_image_paths,
)
from nybble.powernap.rendering import decode_model_turn, parse_tagged_block
from nybble.powernap.retrieval_budget import pack_retrieved_memories
from nybble.powernap.rewards import parse_actions
from nybble.powernap.types import PredictionTrace


def _mmr(hits: list[dict], top_m: int, alpha: float) -> list[dict]:
    if not hits:
        return []
    from nybble.retrieval import mmr_select

    items = [(hit["text"], float(hit["score"]), hit) for hit in hits]
    return [value[2] for value in mmr_select(items, top_m=top_m, alpha=alpha)]


class TinkerPowerNapPredictor:
    def __init__(
        self,
        *,
        config: PowerNapConfig,
        checkpoint: CheckpointRecord,
        retriever: Any,
        tinker_api_key: str,
        predictions_path: Path | None = None,
    ) -> None:
        if not tinker_api_key:
            raise ValueError("tinker_api_key cannot be empty")
        self.config = config
        self.checkpoint = checkpoint
        self.retriever = retriever
        self.tinker_api_key = tinker_api_key
        self.predictions_path = Path(predictions_path) if predictions_path else None
        self._sampling_client: Any | None = None
        self._renderer: Any | None = None
        self._tokenizer: Any | None = None

    async def initialize(self) -> None:
        try:
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PowerNap inference requires Python 3.11+ and `pip install -e '.[tinker]'`"
            ) from exc

        service = tinker.ServiceClient(
            api_key=self.tinker_api_key,
            user_metadata={"app": "nybble", "mode": "powernap-inference"},
        )
        self._sampling_client = await service.create_sampling_client_async(
            model_path=self.checkpoint.sampler_path
        )
        tokenizer = get_tokenizer(self.checkpoint.model)
        self._tokenizer = tokenizer
        self._renderer = renderers.get_renderer(
            self.checkpoint.renderer, tokenizer, model_name=self.checkpoint.model
        )
        if self.checkpoint.retriever_path:
            self.retriever.load_checkpoint(self.checkpoint.retriever_path)

    async def predict(
        self,
        context_text: str,
        cutoff_ts: float,
        *,
        image_paths: Sequence[str] = (),
        future_len: int | None = None,
        action_attempts: int = 3,
    ) -> PredictionTrace:
        count = self.config.future_len if future_len is None else future_len
        if type(count) is not int or count < 1:
            raise ValueError("future_len must be a positive integer")
        if type(action_attempts) is not int or action_attempts < 1:
            raise ValueError("action_attempts must be a positive integer")
        if not isinstance(context_text, str) or not context_text.strip():
            raise ValueError("context_text must not be empty")
        if not isinstance(cutoff_ts, (int, float)) or cutoff_ts < 0:
            raise ValueError("cutoff_ts must be nonnegative")
        validated_images = validate_image_paths(image_paths)
        if self._sampling_client is None or self._renderer is None:
            await self.initialize()
        messages = build_think_messages(context_text, validated_images)
        rationale, rationale_content = await self._sample_tagged(
            messages, "rationale", action_attempts
        )

        hits = (
            self.retriever.query(
                retrieval_query(rationale_content, context_text),
                k=self.config.retrieval_top_k,
                cutoff_ts=cutoff_ts,
                namespaces=("train",),
                time_decay_lambda=self.config.retrieval_time_decay,
            )
            if self.config.retrieval_max_tokens
            else []
        )
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("predictor tokenizer is not initialized")
        retrieved = pack_retrieved_memories(
            _mmr(hits, self.config.retrieval_mmr_k, self.config.retrieval_mmr_alpha),
            tokenizer,
            self.config.retrieval_max_tokens,
        )
        messages = append_revise(messages, rationale, retrieved)
        revision, _revision_content = await self._sample_tagged(messages, "revise", action_attempts)
        messages = append_actions(messages, revision, count)
        errors: tuple[str, ...] = ()
        for _attempt in range(action_attempts):
            try:
                raw_actions = await self._sample(messages)
            except RuntimeError as exc:
                errors = (str(exc),)
                continue
            parsed = parse_actions(raw_actions, count)
            if parsed.valid:
                break
            errors = parsed.errors
        else:
            raise ValueError(
                "invalid checkpoint prediction after "
                f"{action_attempts} attempts: " + "; ".join(errors)
            )

        trace = PredictionTrace(
            rationale=rationale,
            retrieved=retrieved,
            revision=revision,
            raw_actions=raw_actions,
            actions=parsed.actions,
        )
        if self.predictions_path:
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
            value = {
                "cutoff_ts": cutoff_ts,
                "future_len": count,
                "checkpoint": self.checkpoint.sampler_path,
                "model": self.checkpoint.model,
                "rationale": trace.rationale,
                "retrieved": trace.retrieved,
                "revision": trace.revision,
                "raw_actions": trace.raw_actions,
                "actions": list(trace.actions),
            }
            with self.predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
        return trace

    async def _sample_tagged(
        self, messages: list[dict], expected_tag: str, attempts: int
    ) -> tuple[str, str]:
        last_error = "invalid phase output"
        for _attempt in range(attempts):
            try:
                raw = await self._sample(messages)
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            parsed = parse_tagged_block(raw, expected_tag)
            if parsed.valid:
                return parsed.raw, parsed.content
            last_error = parsed.error or last_error
        raise ValueError(
            f"invalid <{expected_tag}> response after {attempts} attempts: {last_error}"
        )

    async def _sample(self, messages: list[dict]) -> str:
        import tinker
        from tinker_cookbook.renderers import get_text_content

        renderer = self._renderer
        client = self._sampling_client
        if renderer is None or client is None:
            raise RuntimeError("predictor is not initialized")
        try:
            prompt = renderer.build_generation_prompt(messages, effort=self.config.inkling_effort)
        except TypeError:
            prompt = renderer.build_generation_prompt(messages)
        response = await client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=renderer.get_stop_sequences(),
            ),
        )
        if not response.sequences:
            raise RuntimeError("Tinker returned no prediction sequence")
        sequence = response.sequences[0]
        decoded = decode_model_turn(
            renderer,
            sequence.tokens,
            {"stop_reason": sequence.stop_reason},
            get_text_content,
        )
        if not decoded.valid:
            raise RuntimeError(f"prediction turn failed: {decoded.error}")
        return decoded.text
