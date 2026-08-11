"""Continuous, journal-driven PowerNap training and prediction.

The runner deliberately owns no capture UI or process lifecycle. It polls an
``EventJournal``, waits for a causal sample set to settle, resumes the latest
optimizer checkpoint for a bounded update, and can emit one prediction from the
new sampler. All network-bearing components are injected behind small protocols
so the orchestration can be tested without Gemini or Tinker credentials.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import json
import logging
import math
import os
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from nybble.config import PowerNapConfig
from nybble.data import (
    ContextBuilder,
    EventJournal,
    SampleBuilder,
    select_latest_visible_context,
    select_recent_image_paths,
)
from nybble.models import ActionEvent, Prediction, TrainingSample
from nybble.pipeline import to_rollout_sample
from nybble.powernap.checkpoints import CheckpointRecord, CheckpointStore
from nybble.powernap.trainer import TrainingRunResult, rollout_sample_digest
from nybble.powernap.types import PredictionTrace, RolloutSample

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ContinuousRunnerError(RuntimeError):
    """Base class for continuous-runner failures."""


class RunnerAlreadyActiveError(ContinuousRunnerError):
    """Raised when another runner owns the same state lease."""


class RunnerStateError(ContinuousRunnerError):
    """Raised for invalid or inconsistent durable runner state."""


class SampleHistoryChangedError(ContinuousRunnerError):
    """Raised when checkpoint-era samples are removed, reordered, or edited."""


class JournalReader(Protocol):
    def read(self) -> tuple[ActionEvent, ...]: ...


class SampleBuilderLike(Protocol):
    def build(self, events: Sequence[ActionEvent]) -> tuple[TrainingSample, ...]: ...


class CheckpointSource(Protocol):
    def latest(self) -> CheckpointRecord | None: ...


class IncrementalTrainer(Protocol):
    async def train(
        self,
        samples: Sequence[RolloutSample],
        *,
        steps: int,
        max_step_attempts: int | None = None,
    ) -> TrainingRunResult: ...


class PowerNapPredictor(Protocol):
    async def predict(
        self,
        context_text: str,
        cutoff_ts: float,
        *,
        image_paths: Sequence[str] = (),
        future_len: int | None = None,
        action_attempts: int = 3,
    ) -> PredictionTrace: ...


class PredictionSink(Protocol):
    def get(self, prediction_id: str) -> Prediction | Awaitable[Prediction | None] | None: ...

    def append(self, prediction: Prediction) -> bool | Awaitable[bool]: ...


TrainerFactory = Callable[[CheckpointRecord | None], IncrementalTrainer]
PredictorFactory = Callable[[CheckpointRecord], PowerNapPredictor]
ResultCallback = Callable[["PollResult"], object | Awaitable[object]]
ErrorCallback = Callable[[BaseException], object | Awaitable[object]]


@dataclass(frozen=True)
class ContinuousRunnerConfig:
    """Operational bounds for one poll-based training service."""

    poll_interval_seconds: float = 5.0
    stable_for_seconds: float = 15.0
    steps_per_update: int = 1
    max_step_attempts: int | None = None
    predict_after_update: bool = False
    action_attempts: int = 3
    error_backoff_seconds: float = 15.0
    max_consecutive_errors: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "poll_interval_seconds",
            "stable_for_seconds",
            "error_backoff_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        for name in ("steps_per_update", "action_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_step_attempts", "max_consecutive_errors"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_step_attempts is not None and self.max_step_attempts < self.steps_per_update:
            raise ValueError("max_step_attempts cannot be less than steps_per_update")
        if type(self.predict_after_update) is not bool:
            raise TypeError("predict_after_update must be a boolean")


@dataclass(frozen=True)
class RunnerState:
    """Small durable cursor written only after a checkpoint is locally visible."""

    sample_fingerprint: str
    sample_count: int
    last_sample_id: str
    checkpoint_step: int
    checkpoint_state_path: str
    checkpoint_sampler_path: str
    updated_at: float
    updates: int = 0
    pending_prediction_id: str = ""
    pending_prediction_cutoff_ts: float | None = None
    pending_context_event_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.sample_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.sample_fingerprint
        ):
            raise ValueError("sample_fingerprint must be a SHA-256 hex digest")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        if not self.last_sample_id:
            raise ValueError("last_sample_id cannot be empty")
        if type(self.checkpoint_step) is not int or self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be a nonnegative integer")
        if not self.checkpoint_state_path or not self.checkpoint_sampler_path:
            raise ValueError("checkpoint paths cannot be empty")
        if (
            isinstance(self.updated_at, bool)
            or not isinstance(self.updated_at, (int, float))
            or not math.isfinite(float(self.updated_at))
            or self.updated_at < 0
        ):
            raise ValueError("updated_at must be a finite nonnegative timestamp")
        if type(self.updates) is not int or self.updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported runner state schema_version {self.schema_version}")
        if not isinstance(self.pending_context_event_ids, tuple) or not all(
            isinstance(event_id, str) and event_id for event_id in self.pending_context_event_ids
        ):
            raise ValueError("pending_context_event_ids must contain nonempty strings")
        if len(self.pending_context_event_ids) != len(set(self.pending_context_event_ids)):
            raise ValueError("pending_context_event_ids must be unique")
        pending_values = (
            bool(self.pending_prediction_id),
            self.pending_prediction_cutoff_ts is not None,
            bool(self.pending_context_event_ids),
        )
        if any(pending_values) and not all(pending_values):
            raise ValueError("pending prediction fields must be set or cleared together")
        if self.pending_prediction_cutoff_ts is not None:
            cutoff = self.pending_prediction_cutoff_ts
            if (
                isinstance(cutoff, bool)
                or not isinstance(cutoff, (int, float))
                or not math.isfinite(float(cutoff))
                or cutoff < 0
            ):
                raise ValueError(
                    "pending_prediction_cutoff_ts must be a finite nonnegative timestamp"
                )

    @property
    def has_pending_prediction(self) -> bool:
        return bool(self.pending_prediction_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> RunnerState:
        if not isinstance(value, dict):
            raise TypeError("runner state must be a JSON object")
        allowed = {
            "sample_fingerprint",
            "sample_count",
            "last_sample_id",
            "checkpoint_step",
            "checkpoint_state_path",
            "checkpoint_sampler_path",
            "updated_at",
            "updates",
            "pending_prediction_id",
            "pending_prediction_cutoff_ts",
            "pending_context_event_ids",
            "schema_version",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown runner state fields: {sorted(unknown)}")
        required = allowed - {
            "updates",
            "pending_prediction_id",
            "pending_prediction_cutoff_ts",
            "pending_context_event_ids",
            "schema_version",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing runner state fields: {sorted(missing)}")
        normalized = dict(value)
        if "pending_context_event_ids" in normalized:
            normalized["pending_context_event_ids"] = tuple(normalized["pending_context_event_ids"])
        return cls(**normalized)


class RunnerStateStore:
    """Atomic, fsynced persistence for the continuous runner cursor."""

    MAX_STATE_BYTES = 1024 * 1024

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(str(self.path))

    def load(self) -> RunnerState | None:
        if not self.path.exists():
            return None
        if self.path.stat().st_size > self.MAX_STATE_BYTES:
            raise RunnerStateError(f"runner state is unexpectedly large: {self.path}")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return RunnerState.from_dict(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RunnerStateError(f"invalid runner state {self.path}: {exc}") from exc

    def save(self, state: RunnerState) -> None:
        if not isinstance(state, RunnerState):
            raise TypeError("state must be a RunnerState")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    state.to_dict(),
                    handle,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


class JsonlPredictionSink:
    """Idempotent, crash-tolerant JSONL prediction persistence."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(str(self.path))
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def append(self, prediction: Prediction) -> bool:
        if not isinstance(prediction, Prediction):
            raise TypeError("prediction must be a Prediction")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("a+b") as handle:
                    existing = self._read_locked(handle)
                    previous = next(
                        (record for record in existing if record.id == prediction.id), None
                    )
                    if previous is not None:
                        if previous != prediction:
                            raise RunnerStateError(
                                f"prediction ID {prediction.id!r} already has different content"
                            )
                        return False
                    encoded = json.dumps(
                        prediction.to_dict(),
                        allow_nan=False,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    handle.seek(0, os.SEEK_END)
                    handle.write(encoded + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    return True
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def get(self, prediction_id: str) -> Prediction | None:
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError("prediction_id must be a nonempty string")
        if not self.path.exists():
            return None
        with self.lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    return None
                with self.path.open("r+b") as handle:
                    return next(
                        (
                            record
                            for record in self._read_locked(handle)
                            if record.id == prediction_id
                        ),
                        None,
                    )
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_locked(handle: Any) -> tuple[Prediction, ...]:
        handle.seek(0)
        content = handle.read()
        if not content:
            return ()
        lines = content.splitlines(keepends=True)
        records: list[Prediction] = []
        offset = 0
        truncate_at: int | None = None
        for index, raw_line in enumerate(lines):
            line_start = offset
            offset += len(raw_line)
            is_last = index == len(lines) - 1
            terminated = raw_line.endswith((b"\n", b"\r"))
            try:
                payload = json.loads(raw_line.strip().decode("utf-8"))
                records.append(Prediction.from_dict(payload))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                if is_last and not terminated:
                    truncate_at = line_start
                    break
                raise RunnerStateError(
                    f"invalid prediction journal line {index + 1}: {exc}"
                ) from exc
        if truncate_at is not None:
            handle.seek(truncate_at)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        elif lines and not lines[-1].endswith((b"\n", b"\r")):
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return tuple(records)


class PollStatus(str, Enum):
    EMPTY = "empty"
    STABILIZING = "stabilizing"
    UNCHANGED = "unchanged"
    UPDATED = "updated"
    PREDICTION_RECOVERED = "prediction_recovered"


@dataclass(frozen=True)
class PollResult:
    status: PollStatus
    sample_count: int
    new_sample_count: int = 0
    checkpoint: CheckpointRecord | None = None
    prediction: Prediction | None = None
    detail: str = ""


@dataclass(frozen=True)
class RunSummary:
    polls: int
    updates: int
    predictions: int
    consecutive_errors: int


@dataclass(frozen=True)
class _SampleSnapshot:
    events: tuple[ActionEvent, ...]
    samples: tuple[TrainingSample, ...]
    rollout_samples: tuple[RolloutSample, ...]
    sample_ids: tuple[str, ...]
    sample_digests: tuple[str, ...]
    fingerprint: str


def rollout_sample_set_fingerprint(
    samples: Sequence[RolloutSample],
) -> tuple[tuple[str, ...], str]:
    """Return the exact digest tuple and fingerprint used by Tinker checkpoints."""

    values = tuple(samples)
    if not values:
        raise ValueError("samples cannot be empty")
    image_digest_cache: dict[str, tuple[int, str]] = {}
    digests = tuple(
        rollout_sample_digest(sample, image_digest_cache=image_digest_cache) for sample in values
    )
    encoded = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digests, hashlib.sha256(encoded).hexdigest()


def is_stable_action_event(event: ActionEvent) -> bool:
    """Exclude mutable labeling-tail records from durable training windows."""

    if not isinstance(event, ActionEvent):
        raise TypeError("event must be an ActionEvent")
    return event.metadata.get("provisional") is not True


class _RunnerLease:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> _RunnerLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RunnerAlreadyActiveError(f"another continuous runner owns {self.path}") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


class ContinuousPowerNapService:
    """Poll an action journal and perform bounded, exactly-cursored updates."""

    def __init__(
        self,
        *,
        powernap_config: PowerNapConfig,
        runner_config: ContinuousRunnerConfig,
        journal: JournalReader,
        sample_builder: SampleBuilderLike,
        checkpoint_store: CheckpointSource,
        state_store: RunnerStateStore,
        trainer_factory: TrainerFactory,
        predictor_factory: PredictorFactory | None = None,
        prediction_sink: PredictionSink | None = None,
        context_builder: ContextBuilder | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        on_result: ResultCallback | None = None,
        on_error: ErrorCallback | None = None,
        stable_event_predicate: Callable[[ActionEvent], bool] = is_stable_action_event,
    ) -> None:
        if not isinstance(powernap_config, PowerNapConfig):
            raise TypeError("powernap_config must be a PowerNapConfig")
        if not isinstance(runner_config, ContinuousRunnerConfig):
            raise TypeError("runner_config must be a ContinuousRunnerConfig")
        if runner_config.predict_after_update and (
            predictor_factory is None or prediction_sink is None
        ):
            raise ValueError("predict_after_update requires predictor_factory and prediction_sink")
        for name in ("past_len", "future_len", "max_images"):
            actual = getattr(sample_builder, name, getattr(powernap_config, name))
            expected = getattr(powernap_config, name)
            if actual != expected:
                raise ValueError(
                    f"sample_builder {name}={actual!r} does not match PowerNap config "
                    f"{name}={expected!r}"
                )
        if not callable(stable_event_predicate):
            raise TypeError("stable_event_predicate must be callable")
        self.powernap_config = powernap_config
        self.runner_config = runner_config
        self.journal = journal
        self.sample_builder = sample_builder
        self.checkpoint_store = checkpoint_store
        self.state_store = state_store
        self.trainer_factory = trainer_factory
        self.predictor_factory = predictor_factory
        self.prediction_sink = prediction_sink
        self.context_builder = context_builder or ContextBuilder(include_dense_context=True)
        self.monotonic_clock = monotonic_clock
        self.wall_clock = wall_clock
        self.on_result = on_result
        self.on_error = on_error
        self.stable_event_predicate = stable_event_predicate
        self._pending_fingerprint = ""
        self._pending_since = 0.0
        self._lease_path = state_store.path.with_name(f".{state_store.path.name}.runner.lock")

    @classmethod
    def for_tinker(
        cls,
        *,
        journal_path: Path,
        run_dir: Path,
        powernap_config: PowerNapConfig,
        runner_config: ContinuousRunnerConfig,
        reward_judge: Any,
        tinker_api_key: str,
        predictions_path: Path | None = None,
    ) -> ContinuousPowerNapService:
        """Build the production Tinker implementation while keeping APIs injectable."""

        from nybble.powernap.predictor import TinkerPowerNapPredictor
        from nybble.powernap.trainer import TinkerPowerNapTrainer
        from nybble.retrieval import BM25TemporalRetriever

        directory = Path(run_dir)
        checkpoint_store = CheckpointStore(directory / "checkpoints.jsonl")

        def trainer_factory(resume: CheckpointRecord | None) -> IncrementalTrainer:
            return TinkerPowerNapTrainer(
                config=powernap_config,
                reward_judge=reward_judge,
                retriever=BM25TemporalRetriever(),
                run_dir=directory,
                tinker_api_key=tinker_api_key,
                resume=resume,
            )

        def predictor_factory(checkpoint: CheckpointRecord) -> PowerNapPredictor:
            return TinkerPowerNapPredictor(
                config=powernap_config,
                checkpoint=checkpoint,
                retriever=BM25TemporalRetriever(),
                tinker_api_key=tinker_api_key,
            )

        sink = (
            JsonlPredictionSink(predictions_path or directory / "predictions.jsonl")
            if runner_config.predict_after_update
            else None
        )
        return cls(
            powernap_config=powernap_config,
            runner_config=runner_config,
            journal=EventJournal(journal_path),
            sample_builder=SampleBuilder(
                past_len=powernap_config.past_len,
                future_len=powernap_config.future_len,
                max_images=powernap_config.max_images,
                context_builder=ContextBuilder(include_dense_context=True),
            ),
            checkpoint_store=checkpoint_store,
            state_store=RunnerStateStore(directory / "continuous-state.json"),
            trainer_factory=trainer_factory,
            predictor_factory=(predictor_factory if runner_config.predict_after_update else None),
            prediction_sink=sink,
        )

    async def poll_once(self) -> PollResult:
        """Run one non-overlapping poll and at most one bounded training update."""

        snapshot = self._snapshot()
        if snapshot is None:
            self._clear_pending_stability()
            return PollResult(PollStatus.EMPTY, 0, detail="no eligible causal windows")

        checkpoint = self.checkpoint_store.latest()
        self._validate_checkpoint_config(checkpoint)
        state = self.state_store.load()

        if state is not None and checkpoint is None:
            raise RunnerStateError("runner state exists but its checkpoint manifest is missing")

        if state is not None and state.has_pending_prediction:
            prediction = await self._complete_pending_prediction(snapshot.events, state, checkpoint)
            cleared = replace(
                state,
                pending_prediction_id="",
                pending_prediction_cutoff_ts=None,
                pending_context_event_ids=(),
            )
            self.state_store.save(cleared)
            self._clear_pending_stability()
            return PollResult(
                PollStatus.PREDICTION_RECOVERED,
                len(snapshot.samples),
                checkpoint=checkpoint,
                prediction=prediction,
                detail="completed a prediction left pending by an earlier update",
            )

        if self._checkpoint_covers_snapshot(checkpoint, snapshot):
            assert checkpoint is not None
            if state is None or not self._state_matches_checkpoint(state, checkpoint, snapshot):
                recovered = self._state_for_checkpoint(
                    snapshot,
                    checkpoint,
                    previous=state,
                    include_pending_prediction=self.runner_config.predict_after_update,
                )
                self.state_store.save(recovered)
                if recovered.has_pending_prediction:
                    prediction = await self._complete_pending_prediction(
                        snapshot.events, recovered, checkpoint
                    )
                    self.state_store.save(
                        replace(
                            recovered,
                            pending_prediction_id="",
                            pending_prediction_cutoff_ts=None,
                            pending_context_event_ids=(),
                        )
                    )
                    return PollResult(
                        PollStatus.PREDICTION_RECOVERED,
                        len(snapshot.samples),
                        checkpoint=checkpoint,
                        prediction=prediction,
                        detail="recovered cursor and prediction from the latest checkpoint",
                    )
            self._clear_pending_stability()
            return PollResult(
                PollStatus.UNCHANGED,
                len(snapshot.samples),
                checkpoint=checkpoint,
                detail="sample set already belongs to the latest checkpoint",
            )

        if state is not None and state.sample_fingerprint == snapshot.fingerprint:
            raise RunnerStateError(
                "runner cursor says the sample set was trained, but the latest checkpoint "
                "does not contain that fingerprint"
            )

        previous_count = self._validate_append_only_history(snapshot, checkpoint)
        now = self.monotonic_clock()
        if self._pending_fingerprint != snapshot.fingerprint:
            self._pending_fingerprint = snapshot.fingerprint
            self._pending_since = now
            if self.runner_config.stable_for_seconds > 0:
                return PollResult(
                    PollStatus.STABILIZING,
                    len(snapshot.samples),
                    new_sample_count=len(snapshot.samples) - previous_count,
                    checkpoint=checkpoint,
                    detail="observed a new sample set and started its stability timer",
                )
        elapsed = now - self._pending_since
        if elapsed < self.runner_config.stable_for_seconds:
            return PollResult(
                PollStatus.STABILIZING,
                len(snapshot.samples),
                new_sample_count=len(snapshot.samples) - previous_count,
                checkpoint=checkpoint,
                detail=f"sample set has been stable for {max(elapsed, 0.0):.3f} seconds",
            )

        trainer = self.trainer_factory(checkpoint)
        result = await trainer.train(
            snapshot.rollout_samples,
            steps=self.runner_config.steps_per_update,
            max_step_attempts=self.runner_config.max_step_attempts,
        )
        trained_checkpoint = self._validate_training_result(snapshot, result)
        persisted_checkpoint = self.checkpoint_store.latest()
        if persisted_checkpoint != trained_checkpoint:
            raise RunnerStateError(
                "trainer returned before its checkpoint was visible in the checkpoint store"
            )

        next_state = self._state_for_checkpoint(
            snapshot,
            trained_checkpoint,
            previous=state,
            include_pending_prediction=self.runner_config.predict_after_update,
        )
        self.state_store.save(next_state)
        self._clear_pending_stability()

        prediction = None
        detail = "completed a bounded incremental training update"
        if next_state.has_pending_prediction:
            prediction = await self._complete_pending_prediction(
                snapshot.events, next_state, trained_checkpoint
            )
            self.state_store.save(
                replace(
                    next_state,
                    pending_prediction_id="",
                    pending_prediction_cutoff_ts=None,
                    pending_context_event_ids=(),
                )
            )
        elif self.runner_config.predict_after_update:
            detail += "; no complete latest context was available for prediction"

        return PollResult(
            PollStatus.UPDATED,
            len(snapshot.samples),
            new_sample_count=len(snapshot.samples) - previous_count,
            checkpoint=trained_checkpoint,
            prediction=prediction,
            detail=detail,
        )

    async def run(
        self,
        *,
        stop_event: asyncio.Event | None = None,
        max_updates: int | None = None,
    ) -> RunSummary:
        """Poll until stopped, releasing the single-runner lease on every exit path."""

        if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
            raise ValueError("max_updates must be a positive integer or None")
        stop = stop_event or asyncio.Event()
        polls = updates = predictions = consecutive_errors = 0
        with _RunnerLease(self._lease_path):
            while not stop.is_set():
                delay = self.runner_config.poll_interval_seconds
                try:
                    result = await self.poll_once()
                    polls += 1
                    consecutive_errors = 0
                    if result.status is PollStatus.UPDATED:
                        updates += 1
                    if result.prediction is not None:
                        predictions += 1
                    if self.on_result is not None:
                        await _maybe_await(self.on_result(result))
                    if max_updates is not None and updates >= max_updates:
                        break
                except asyncio.CancelledError:
                    logger.info("continuous PowerNap runner cancelled")
                    raise
                except Exception as exc:
                    consecutive_errors += 1
                    logger.exception("continuous PowerNap poll failed")
                    if self.on_error is not None:
                        await _maybe_await(self.on_error(exc))
                    if (
                        self.runner_config.max_consecutive_errors is not None
                        and consecutive_errors >= self.runner_config.max_consecutive_errors
                    ):
                        raise
                    delay = self.runner_config.error_backoff_seconds
                await _wait_or_stop(stop, delay)
        return RunSummary(polls, updates, predictions, consecutive_errors)

    async def run_until_signalled(self, *, max_updates: int | None = None) -> RunSummary:
        """Run until SIGINT/SIGTERM requests a graceful boundary stop."""

        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        installed: list[signal.Signals] = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop.set)
            except (NotImplementedError, RuntimeError):
                continue
            installed.append(signum)
        try:
            return await self.run(stop_event=stop, max_updates=max_updates)
        finally:
            for signum in installed:
                loop.remove_signal_handler(signum)

    def _snapshot(self) -> _SampleSnapshot | None:
        journal_events = tuple(self.journal.read())
        if not all(isinstance(event, ActionEvent) for event in journal_events):
            raise TypeError("journal.read() must return only ActionEvent records")
        events = tuple(event for event in journal_events if self.stable_event_predicate(event))
        samples = tuple(self.sample_builder.build(events))
        if not samples:
            return None
        if not all(isinstance(sample, TrainingSample) for sample in samples):
            raise TypeError("sample_builder.build() must return TrainingSample records")
        rollout_samples = tuple(to_rollout_sample(sample) for sample in samples)
        sample_digests, fingerprint = rollout_sample_set_fingerprint(rollout_samples)
        return _SampleSnapshot(
            events=events,
            samples=samples,
            rollout_samples=rollout_samples,
            sample_ids=tuple(sample.id for sample in rollout_samples),
            sample_digests=sample_digests,
            fingerprint=fingerprint,
        )

    def _validate_checkpoint_config(self, checkpoint: CheckpointRecord | None) -> None:
        if checkpoint is None:
            return
        if (
            checkpoint.model != self.powernap_config.model
            or checkpoint.renderer != self.powernap_config.renderer
        ):
            raise RunnerStateError("latest checkpoint model/renderer does not match runner config")
        if checkpoint.config and checkpoint.config != asdict(self.powernap_config):
            raise RunnerStateError("latest checkpoint PowerNap config does not match runner config")

    @staticmethod
    def _checkpoint_covers_snapshot(
        checkpoint: CheckpointRecord | None, snapshot: _SampleSnapshot
    ) -> bool:
        if checkpoint is None:
            return False
        if checkpoint.data_fingerprint:
            return checkpoint.data_fingerprint == snapshot.fingerprint
        if checkpoint.sample_digests:
            return checkpoint.sample_digests == snapshot.sample_digests
        if checkpoint.sample_ids:
            return checkpoint.sample_ids == snapshot.sample_ids
        return False

    @staticmethod
    def _validate_append_only_history(
        snapshot: _SampleSnapshot, checkpoint: CheckpointRecord | None
    ) -> int:
        if checkpoint is None:
            return 0
        if checkpoint.sample_digests:
            previous_count = len(checkpoint.sample_digests)
            if snapshot.sample_digests[:previous_count] != checkpoint.sample_digests:
                raise SampleHistoryChangedError(
                    "checkpoint-era sample content is no longer an exact prefix"
                )
            return previous_count
        if checkpoint.sample_ids:
            previous_count = len(checkpoint.sample_ids)
            if snapshot.sample_ids[:previous_count] != checkpoint.sample_ids:
                raise SampleHistoryChangedError(
                    "checkpoint-era sample IDs are no longer an exact prefix"
                )
            return previous_count
        raise SampleHistoryChangedError(
            "latest checkpoint has no sample cursor and cannot be incrementally extended safely"
        )

    def _validate_training_result(
        self, snapshot: _SampleSnapshot, result: TrainingRunResult
    ) -> CheckpointRecord:
        if not isinstance(result, TrainingRunResult):
            raise TypeError("trainer.train() must return TrainingRunResult")
        checkpoint = result.checkpoint
        self._validate_checkpoint_config(checkpoint)
        if checkpoint.data_fingerprint != snapshot.fingerprint:
            raise RunnerStateError("new checkpoint fingerprint does not match the trained samples")
        if checkpoint.sample_ids and checkpoint.sample_ids != snapshot.sample_ids:
            raise RunnerStateError("new checkpoint sample IDs do not match the trained samples")
        if checkpoint.sample_digests and checkpoint.sample_digests != snapshot.sample_digests:
            raise RunnerStateError("new checkpoint sample digests do not match the trained samples")
        if checkpoint.step != result.completed_steps:
            raise RunnerStateError("training result step does not match its checkpoint")
        return checkpoint

    @staticmethod
    def _state_matches_checkpoint(
        state: RunnerState, checkpoint: CheckpointRecord, snapshot: _SampleSnapshot
    ) -> bool:
        return (
            state.sample_fingerprint == snapshot.fingerprint
            and state.sample_count == len(snapshot.samples)
            and state.last_sample_id == snapshot.sample_ids[-1]
            and state.checkpoint_step == checkpoint.step
            and state.checkpoint_state_path == checkpoint.state_path
            and state.checkpoint_sampler_path == checkpoint.sampler_path
        )

    def _state_for_checkpoint(
        self,
        snapshot: _SampleSnapshot,
        checkpoint: CheckpointRecord,
        *,
        previous: RunnerState | None,
        include_pending_prediction: bool,
    ) -> RunnerState:
        pending_id = ""
        pending_cutoff: float | None = None
        pending_context_ids: tuple[str, ...] = ()
        if include_pending_prediction:
            cutoff = max(event.available_at for event in snapshot.events)
            context = select_latest_visible_context(
                snapshot.events,
                cutoff_ts=cutoff,
                past_len=self.powernap_config.past_len,
            )
            if len(context) == self.powernap_config.past_len:
                identity = json.dumps(
                    {
                        "checkpoint": checkpoint.sampler_path,
                        "sample_fingerprint": snapshot.fingerprint,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                pending_id = f"prediction-{hashlib.sha256(identity).hexdigest()[:24]}"
                pending_cutoff = cutoff
                pending_context_ids = tuple(event.id for event in context)
        updates = (previous.updates if previous is not None else 0) + 1
        updated_at = max(float(self.wall_clock()), pending_cutoff or 0.0)
        return RunnerState(
            sample_fingerprint=snapshot.fingerprint,
            sample_count=len(snapshot.samples),
            last_sample_id=snapshot.sample_ids[-1],
            checkpoint_step=checkpoint.step,
            checkpoint_state_path=checkpoint.state_path,
            checkpoint_sampler_path=checkpoint.sampler_path,
            updated_at=updated_at,
            updates=updates,
            pending_prediction_id=pending_id,
            pending_prediction_cutoff_ts=pending_cutoff,
            pending_context_event_ids=pending_context_ids,
        )

    async def _complete_pending_prediction(
        self,
        events: tuple[ActionEvent, ...],
        state: RunnerState,
        checkpoint: CheckpointRecord | None,
    ) -> Prediction:
        if checkpoint is None:
            raise RunnerStateError("pending prediction has no checkpoint")
        if not state.has_pending_prediction:
            raise RunnerStateError("runner state has no pending prediction")
        if (
            checkpoint.step != state.checkpoint_step
            or checkpoint.state_path != state.checkpoint_state_path
            or checkpoint.sampler_path != state.checkpoint_sampler_path
        ):
            raise RunnerStateError("pending prediction checkpoint is no longer latest")
        predictor_factory = self.predictor_factory
        prediction_sink = self.prediction_sink
        if predictor_factory is None or prediction_sink is None:
            raise RunnerStateError("pending prediction dependencies are not configured")
        cutoff = state.pending_prediction_cutoff_ts
        if cutoff is None:
            raise RunnerStateError("pending prediction cutoff is missing")
        context = self.context_builder.select(events, state.pending_context_event_ids)
        canonical_context = select_latest_visible_context(
            events,
            cutoff_ts=cutoff,
            past_len=self.powernap_config.past_len,
        )
        if tuple(event.id for event in canonical_context) != state.pending_context_event_ids:
            raise SampleHistoryChangedError(
                "events for the pending prediction no longer form the same visible context"
            )
        context_text = self.context_builder.render_context(context)
        image_paths = select_recent_image_paths(context, self.powernap_config.max_images)
        trace = await predictor_factory(checkpoint).predict(
            context_text,
            cutoff,
            image_paths=image_paths,
            future_len=self.powernap_config.future_len,
            action_attempts=self.runner_config.action_attempts,
        )
        prediction = Prediction(
            id=state.pending_prediction_id,
            created_at=max(state.updated_at, cutoff),
            cutoff_ts=cutoff,
            actions=trace.actions,
            raw_text=trace.raw_actions,
            rationale=trace.rationale,
            revision=trace.revision,
            retrieved=(trace.retrieved,) if trace.retrieved else (),
            context_event_ids=state.pending_context_event_ids,
            model=checkpoint.model,
            checkpoint=checkpoint.sampler_path,
            metadata={
                "future_len": self.powernap_config.future_len,
                "image_paths": list(image_paths),
                "sample_fingerprint": state.sample_fingerprint,
                "continuous_runner": True,
            },
        )
        await _maybe_await(prediction_sink.append(prediction))
        return prediction

    def _clear_pending_stability(self) -> None:
        self._pending_fingerprint = ""
        self._pending_since = 0.0


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


async def _wait_or_stop(stop: asyncio.Event, delay: float) -> None:
    if stop.is_set() or delay <= 0:
        await asyncio.sleep(0)
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        return


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ContinuousPowerNapService",
    "ContinuousRunnerConfig",
    "ContinuousRunnerError",
    "JsonlPredictionSink",
    "PollResult",
    "PollStatus",
    "RunSummary",
    "RunnerAlreadyActiveError",
    "RunnerState",
    "RunnerStateError",
    "RunnerStateStore",
    "SampleHistoryChangedError",
    "rollout_sample_set_fingerprint",
]
