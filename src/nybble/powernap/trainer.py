"""Bounded, resumable GRPO/LoRA training against the current Tinker SDK."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from nybble.config import PowerNapConfig
from nybble.powernap.checkpoints import CheckpointRecord, CheckpointStore
from nybble.powernap.prompts import validate_image_paths
from nybble.powernap.rewards import GeminiRewardJudge
from nybble.powernap.types import RolloutSample

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class TrainingRunResult:
    completed_steps: int
    checkpoint: CheckpointRecord
    metrics_path: str


async def _retry(
    operation: Callable[[], Awaitable[T]],
    *,
    name: str,
    attempts: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 15.0,
) -> T:
    """Retry by recreating the operation, never by polling a failed future."""
    delay = initial_delay
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            logger.warning("%s failed (%s/%s): %s", name, attempt, attempts, exc)
            await asyncio.sleep(delay)
            delay = min(max_delay, max(delay * 2, 0.1))
    raise RuntimeError(f"{name} failed after {attempts} attempts") from last_error


def _remove_mask(tinker: Any, datum: Any) -> Any:
    """The importance-sampling loss does not accept cookbook's diagnostic mask."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
    )


def distinct_batch(
    samples: Sequence[RolloutSample], step: int, batch_size: int, seed: int = 0
) -> list[RolloutSample]:
    """Choose unique windows deterministically, without cloning the latest sample."""
    if not samples:
        return []
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    count = min(batch_size, len(order))
    start = (step * count) % len(order)
    indices = [order[(start + offset) % len(order)] for offset in range(count)]
    return [samples[index] for index in indices]


def _file_manifest(
    path: str,
    cache: dict[str, tuple[int, str]] | None = None,
) -> dict[str, str | int]:
    cached = cache.get(path) if cache is not None else None
    if cached is not None:
        size, sha256 = cached
        return {"path": path, "size": size, "sha256": sha256}
    resolved = Path(path)
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    sha256 = digest.hexdigest()
    if cache is not None:
        cache[path] = (size, sha256)
    return {"path": path, "size": size, "sha256": sha256}


def rollout_sample_digest(
    sample: RolloutSample,
    *,
    image_digest_cache: dict[str, tuple[int, str]] | None = None,
) -> str:
    """Bind a resume checkpoint to every model-visible sample input and target."""

    if not isinstance(sample, RolloutSample):
        raise TypeError("sample must be a RolloutSample")
    validated_images = validate_image_paths(sample.image_paths)
    image_manifest = [_file_manifest(path, image_digest_cache) for path in validated_images]
    payload = {
        "id": sample.id,
        "context_text": sample.context_text,
        "solution_text": sample.solution_text,
        "cutoff_ts": sample.cutoff_ts,
        "target_end_ts": sample.target_end_ts,
        "target_visible_after_ts": sample.target_visible_after_ts,
        "future_len": sample.future_len,
        "past_event_ids": sample.past_event_ids,
        "future_event_ids": sample.future_event_ids,
        "images": image_manifest,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resume_samples(
    resume: CheckpointRecord,
    *,
    sample_ids: tuple[str, ...],
    sample_digests: tuple[str, ...],
    data_fingerprint: str,
) -> None:
    """Accept appended data, but never a changed checkpoint-era sample prefix."""

    if resume.sample_digests:
        prefix = sample_digests[: len(resume.sample_digests)]
        if prefix != resume.sample_digests:
            raise ValueError("resume checkpoint sample content is not a prefix of current samples")
    elif resume.sample_ids:
        # Schema v1 checkpoints did not persist content digests. Preserve their
        # historical ID-prefix behavior so existing runs remain loadable.
        prefix = sample_ids[: len(resume.sample_ids)]
        if prefix != resume.sample_ids:
            raise ValueError("resume checkpoint samples are not a prefix of current samples")
    elif resume.data_fingerprint and resume.data_fingerprint != data_fingerprint:
        raise ValueError("resume checkpoint training data does not match current samples")


async def _forward_backward_once(training_client: Any, datums: Any) -> Any:
    """Enqueue and await one non-idempotent gradient mutation exactly once."""

    future = await training_client.forward_backward_async(datums, loss_fn="importance_sampling")
    return await future.result_async()


async def _optimizer_step_once(training_client: Any, adam_params: Any) -> Any:
    """Enqueue and await one non-idempotent optimizer mutation exactly once."""

    future = await training_client.optim_step_async(adam_params)
    return await future.result_async()


class TinkerPowerNapTrainer:
    def __init__(
        self,
        *,
        config: PowerNapConfig,
        reward_judge: GeminiRewardJudge,
        retriever: Any,
        run_dir: Path,
        tinker_api_key: str,
        resume: CheckpointRecord | None = None,
    ) -> None:
        if not tinker_api_key:
            raise ValueError("tinker_api_key cannot be empty")
        self.config = config
        self.reward_judge = reward_judge
        self.retriever = retriever
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tinker_api_key = tinker_api_key
        self.resume = resume
        reward_contract = getattr(reward_judge, "contract", {})
        if not isinstance(reward_contract, dict):
            raise TypeError("reward judge contract must be a dictionary")
        json.dumps(reward_contract, allow_nan=False, sort_keys=True)
        self._reward_contract = dict(reward_contract)
        self.checkpoints = CheckpointStore(self.run_dir / "checkpoints.jsonl")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._data_fingerprint = ""
        self._sample_ids: tuple[str, ...] = ()
        self._sample_digests: tuple[str, ...] = ()
        existing = self.checkpoints.all()
        if resume is None and (existing or self.metrics_path.exists()):
            raise RuntimeError(
                f"run directory already contains training state: {self.run_dir}; "
                "resume it or choose a new run directory"
            )
        if resume is not None:
            if resume.model != config.model or resume.renderer != config.renderer:
                raise ValueError("resume checkpoint model/renderer does not match config")
            if resume.config and resume.config != asdict(config):
                raise ValueError("resume checkpoint PowerNap config does not match config")
            if resume.reward_contract and resume.reward_contract != self._reward_contract:
                raise ValueError("resume checkpoint Gemini reward contract does not match")

    async def train(
        self,
        samples: Sequence[RolloutSample],
        *,
        steps: int,
        max_step_attempts: int | None = None,
    ) -> TrainingRunResult:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if not samples:
            raise ValueError("no eligible training windows")
        if max_step_attempts is not None and max_step_attempts < steps:
            raise ValueError("max_step_attempts must be at least steps")
        if self.config.temperature <= 0:
            raise ValueError("PowerNap GRPO training requires temperature greater than zero")
        image_digest_cache: dict[str, tuple[int, str]] = {}
        self._sample_digests = tuple(
            rollout_sample_digest(sample, image_digest_cache=image_digest_cache)
            for sample in samples
        )
        fingerprint_payload = json.dumps(
            self._sample_digests,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._data_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        self._sample_ids = tuple(sample.id for sample in samples)
        if self.resume is not None:
            _validate_resume_samples(
                self.resume,
                sample_ids=self._sample_ids,
                sample_digests=self._sample_digests,
                data_fingerprint=self._data_fingerprint,
            )

        try:
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.rl.data_processing import (
                assemble_training_data,
                compute_advantages,
            )
            from tinker_cookbook.rl.rollouts import (
                do_group_rollout_and_filter_constant_reward,
            )
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PowerNap training requires Python 3.11+ and `pip install -e '.[tinker]'`"
            ) from exc

        from nybble.powernap.environment import LongNAPGroupBuilder

        service = tinker.ServiceClient(
            api_key=self.tinker_api_key,
            user_metadata={"app": "nybble", "algorithm": "powernap"},
        )
        if self.resume:
            training_client = await _retry(
                lambda: service.create_training_client_from_state_with_optimizer_async(
                    self.resume.state_path, base_model=self.resume.model
                ),
                name="resume Tinker training state",
            )
            if self.resume.retriever_path:
                self.retriever.load_checkpoint(self.resume.retriever_path)
            completed = self.resume.step
        else:
            training_client = await _retry(
                lambda: service.create_lora_training_client_async(
                    base_model=self.config.model,
                    rank=self.config.lora_rank,
                    user_metadata={"app": "nybble", "algorithm": "powernap"},
                ),
                name="create Tinker LoRA client",
            )
            completed = 0

        tokenizer = get_tokenizer(self.config.model)
        renderer = renderers.get_renderer(
            self.config.renderer, tokenizer, model_name=self.config.model
        )
        requested_final_step = completed + steps
        attempts = 0
        batch_cursor = completed
        attempt_limit = (
            max_step_attempts if max_step_attempts is not None else max(steps * 5, steps + 4)
        )
        latest_sample_id = ""

        while completed < requested_final_step and attempts < attempt_limit:
            batch = distinct_batch(samples, batch_cursor, self.config.batch_size)
            batch_cursor += 1
            sampling_client = await _retry(
                training_client.save_weights_and_get_sampling_client_async,
                name="refresh on-policy sampler",
            )
            builders = [
                LongNAPGroupBuilder(
                    sample=sample,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    retriever=self.retriever,
                    reward_judge=self.reward_judge,
                    num_envs=self.config.group_size,
                    retrieval_top_k=self.config.retrieval_top_k,
                    retrieval_mmr_k=self.config.retrieval_mmr_k,
                    retrieval_max_tokens=self.config.retrieval_max_tokens,
                    retrieval_mmr_alpha=self.config.retrieval_mmr_alpha,
                    retrieval_time_decay=self.config.retrieval_time_decay,
                    retrieval_min_accuracy=self.config.retrieval_min_accuracy,
                    effort=self.config.inkling_effort,
                )
                for sample in batch
            ]

            async def rollout(
                builder: LongNAPGroupBuilder,
                client: Any = sampling_client,
            ) -> tuple[LongNAPGroupBuilder, Any]:
                trajectory = await _retry(
                    lambda: do_group_rollout_and_filter_constant_reward(
                        sampling_client=client,
                        env_group_builder=builder,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        do_remove_constant_reward_groups=True,
                        enable_logging=False,
                    ),
                    name=f"rollout {builder.sample.id}",
                )
                return builder, trajectory

            raw_results = await asyncio.gather(
                *(rollout(builder) for builder in builders), return_exceptions=True
            )
            usable = []
            failures = []
            for result in raw_results:
                if isinstance(result, BaseException):
                    failures.append(result)
                elif result[1] is not None:
                    usable.append(result)
            if failures:
                logger.warning("%s rollout group(s) failed", len(failures))
            attempts += 1
            if not usable:
                self._append_metrics(
                    {"attempt": attempts, "step": completed, "skipped": True, "reason": "no_signal"}
                )
                continue

            used_builders = [result[0] for result in usable]
            trajectory_groups = [result[1] for result in usable]
            advantages = compute_advantages(trajectory_groups)
            datums, _ = assemble_training_data(trajectory_groups, advantages)
            datums = [_remove_mask(tinker, datum) for datum in datums]
            if not datums:
                continue

            # These calls mutate remote training state. Reissuing one after a lost
            # response could apply gradients or an optimizer step twice, so any
            # error aborts the run and recovery starts from the last checkpoint.
            fwd_result = await _forward_backward_once(training_client, datums)
            await _optimizer_step_once(
                training_client,
                tinker.AdamParams(
                    learning_rate=self.config.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                ),
            )
            rewards = []
            for builder, group in zip(used_builders, trajectory_groups, strict=True):
                group_rewards = group.get_total_rewards()
                rewards.extend(group_rewards)
                builder.add_winner_to_retriever(group_rewards)

            completed += 1
            latest_sample_id = used_builders[-1].sample.id
            loss = float(fwd_result.metrics.get("loss:sum", 0.0))
            metrics = {
                "attempt": attempts,
                "step": completed,
                "loss_sum": loss,
                "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
                "groups": len(trajectory_groups),
                "datums": len(datums),
                "sample_ids": [builder.sample.id for builder in used_builders],
                "timestamp": time.time(),
            }
            self._append_metrics(metrics)

            if self.config.checkpoint_every and completed % self.config.checkpoint_every == 0:
                await self._save_checkpoint(
                    training_client, completed, latest_sample_id, durable=False
                )

        if completed < requested_final_step:
            raise RuntimeError(
                f"only completed {completed} of {requested_final_step} steps after {attempts} attempts"
            )
        checkpoint = await self._save_checkpoint(
            training_client, completed, latest_sample_id, durable=True
        )
        return TrainingRunResult(completed, checkpoint, str(self.metrics_path))

    async def _save_checkpoint(
        self, training_client: Any, step: int, sample_id: str, *, durable: bool
    ) -> CheckpointRecord:
        ttl = None if durable else 7 * 24 * 60 * 60
        kind = "final" if durable else "step"
        suffix = f"{kind}-{step:06d}-{uuid.uuid4().hex[:10]}"
        name = f"nybble-powernap-{suffix}"

        async def save_state() -> str:
            future = await training_client.save_state_async(name + "-state", ttl_seconds=ttl)
            return (await future.result_async()).path

        async def save_sampler() -> str:
            future = await training_client.save_weights_for_sampler_async(
                name + "-sampler", ttl_seconds=ttl
            )
            return (await future.result_async()).path

        state_path, sampler_path = await asyncio.gather(
            _retry(save_state, name="save Tinker state"),
            _retry(save_sampler, name="save Tinker sampler"),
        )
        retriever_path = (self.run_dir / f"retriever-{suffix}.json.gz").resolve()
        self.retriever.save_checkpoint(retriever_path)
        record = CheckpointRecord.create(
            step=step,
            state_path=state_path,
            sampler_path=sampler_path,
            retriever_path=str(retriever_path),
            model=self.config.model,
            renderer=self.config.renderer,
            last_sample_id=sample_id,
            config=asdict(self.config),
            data_fingerprint=self._data_fingerprint,
            sample_ids=self._sample_ids,
            sample_digests=self._sample_digests,
            reward_contract=self._reward_contract,
        )
        self.checkpoints.append(record)
        return record

    def _append_metrics(self, value: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
