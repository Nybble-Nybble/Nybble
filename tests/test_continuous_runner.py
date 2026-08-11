import asyncio
from dataclasses import asdict, replace

import pytest

from nybble.config import PowerNapConfig
from nybble.data import ContextBuilder, SampleBuilder
from nybble.models import ActionEvent, Prediction
from nybble.powernap.checkpoints import CheckpointRecord, CheckpointStore
from nybble.powernap.continuous import (
    ContinuousPowerNapService,
    ContinuousRunnerConfig,
    JsonlPredictionSink,
    PollStatus,
    RunnerStateError,
    RunnerStateStore,
    SampleHistoryChangedError,
    rollout_sample_set_fingerprint,
)
from nybble.powernap.trainer import TrainingRunResult
from nybble.powernap.types import PredictionTrace


def event(index: int, *, text: str | None = None) -> ActionEvent:
    timestamp = float(index)
    return ActionEvent(
        id=f"event-{index}",
        start_ts=timestamp,
        end_ts=timestamp,
        available_at=timestamp,
        source="meta-glasses",
        text=text or f"action {index}",
        chunk_id=f"chunk-{index}",
        provenance={"session_id": "session-1"},
    )


class MutableJournal:
    def __init__(self, events=()):
        self.events = tuple(events)

    def read(self):
        return self.events


class FakeTrainerFactory:
    def __init__(self, config, store):
        self.config = config
        self.store = store
        self.resumes = []
        self.sample_counts = []

    def __call__(self, resume):
        self.resumes.append(resume)
        owner = self

        class Trainer:
            async def train(self, samples, *, steps, max_step_attempts=None):
                assert max_step_attempts is None or max_step_attempts >= steps
                sample_values = tuple(samples)
                owner.sample_counts.append(len(sample_values))
                digests, fingerprint = rollout_sample_set_fingerprint(sample_values)
                completed = (resume.step if resume else 0) + steps
                checkpoint = CheckpointRecord.create(
                    step=completed,
                    state_path=f"tinker://state/{completed}",
                    sampler_path=f"tinker://sampler/{completed}",
                    retriever_path=f"/tmp/retriever-{completed}.json.gz",
                    model=owner.config.model,
                    renderer=owner.config.renderer,
                    last_sample_id=sample_values[-1].id,
                    config=asdict(owner.config),
                    data_fingerprint=fingerprint,
                    sample_ids=tuple(sample.id for sample in sample_values),
                    sample_digests=digests,
                )
                owner.store.append(checkpoint)
                return TrainingRunResult(completed, checkpoint, "/tmp/metrics.jsonl")

        return Trainer()


class FakePredictor:
    def __init__(self, *, failures=0):
        self.failures = failures
        self.calls = []

    async def predict(
        self,
        context_text,
        cutoff_ts,
        *,
        image_paths=(),
        future_len=None,
        action_attempts=3,
    ):
        self.calls.append(
            (context_text, cutoff_ts, tuple(image_paths), future_len, action_attempts)
        )
        if self.failures:
            self.failures -= 1
            raise ConnectionError("temporary prediction failure")
        return PredictionTrace(
            rationale="<rationale>reason</rationale>",
            retrieved="memory",
            revision="<revise>revision</revise>",
            raw_actions="<actions><action>next action</action></actions>",
            actions=("next action",),
        )


class MemoryPredictionSink:
    def __init__(self):
        self.records = {}

    def append(self, prediction):
        previous = self.records.get(prediction.id)
        if previous is not None and previous != prediction:
            raise AssertionError("prediction ID conflict")
        self.records[prediction.id] = prediction
        return previous is None


def service(
    tmp_path,
    journal,
    *,
    stable_for=0,
    predict=False,
    predictor=None,
    sink=None,
    monotonic_clock=None,
    on_result=None,
):
    config = PowerNapConfig(
        past_len=2,
        future_len=1,
        max_images=0,
        batch_size=2,
        checkpoint_every=0,
    )
    store = CheckpointStore(tmp_path / "run" / "checkpoints.jsonl")
    factory = FakeTrainerFactory(config, store)
    runner = ContinuousPowerNapService(
        powernap_config=config,
        runner_config=ContinuousRunnerConfig(
            poll_interval_seconds=60,
            stable_for_seconds=stable_for,
            steps_per_update=1,
            predict_after_update=predict,
            error_backoff_seconds=0,
        ),
        journal=journal,
        sample_builder=SampleBuilder(
            past_len=2,
            future_len=1,
            max_images=0,
            context_builder=ContextBuilder(include_dense_context=True),
        ),
        checkpoint_store=store,
        state_store=RunnerStateStore(tmp_path / "run" / "continuous-state.json"),
        trainer_factory=factory,
        predictor_factory=(lambda _checkpoint: predictor) if predictor else None,
        prediction_sink=sink,
        monotonic_clock=monotonic_clock or (lambda: 0.0),
        wall_clock=lambda: 100.0,
        on_result=on_result,
    )
    return runner, factory, store


def test_incremental_runner_skips_unchanged_samples_and_resumes_latest_checkpoint(tmp_path):
    journal = MutableJournal((event(1), event(2), event(3)))
    runner, factory, _store = service(tmp_path, journal)

    async def scenario():
        first = await runner.poll_once()
        unchanged = await runner.poll_once()
        journal.events = (*journal.events, event(4))
        second = await runner.poll_once()
        return first, unchanged, second

    first, unchanged, second = asyncio.run(scenario())

    assert first.status is PollStatus.UPDATED
    assert first.new_sample_count == 1
    assert unchanged.status is PollStatus.UNCHANGED
    assert second.status is PollStatus.UPDATED
    assert second.new_sample_count == 1
    assert factory.sample_counts == [1, 2]
    assert factory.resumes[0] is None
    assert factory.resumes[1] == first.checkpoint
    state = runner.state_store.load()
    assert state is not None
    assert state.sample_count == 2
    assert state.checkpoint_step == 2
    assert state.updates == 2


def test_stability_timer_resets_when_the_eligible_sample_set_changes(tmp_path):
    clock = [0.0]
    journal = MutableJournal((event(1), event(2), event(3)))
    runner, factory, _store = service(
        tmp_path,
        journal,
        stable_for=10,
        monotonic_clock=lambda: clock[0],
    )

    async def scenario():
        first = await runner.poll_once()
        clock[0] = 4.0
        second = await runner.poll_once()
        journal.events = (*journal.events, event(4))
        clock[0] = 5.0
        changed = await runner.poll_once()
        clock[0] = 14.0
        waiting = await runner.poll_once()
        clock[0] = 15.0
        trained = await runner.poll_once()
        return first, second, changed, waiting, trained

    results = asyncio.run(scenario())

    assert [result.status for result in results] == [
        PollStatus.STABILIZING,
        PollStatus.STABILIZING,
        PollStatus.STABILIZING,
        PollStatus.STABILIZING,
        PollStatus.UPDATED,
    ]
    assert factory.sample_counts == [2]


def test_changed_checkpoint_history_is_rejected_before_remote_training(tmp_path):
    journal = MutableJournal((event(1), event(2), event(3)))
    runner, factory, _store = service(tmp_path, journal)

    async def scenario():
        await runner.poll_once()
        journal.events = (event(1, text="rewritten action"), event(2), event(3))
        await runner.poll_once()

    with pytest.raises(SampleHistoryChangedError, match="exact prefix"):
        asyncio.run(scenario())
    assert factory.sample_counts == [1]


def test_failed_prediction_is_retried_without_retraining_the_same_sample_set(tmp_path):
    journal = MutableJournal((event(1), event(2), event(3)))
    predictor = FakePredictor(failures=1)
    sink = MemoryPredictionSink()
    runner, factory, _store = service(
        tmp_path,
        journal,
        predict=True,
        predictor=predictor,
        sink=sink,
    )

    async def scenario():
        with pytest.raises(ConnectionError, match="temporary"):
            await runner.poll_once()
        pending = runner.state_store.load()
        assert pending is not None and pending.has_pending_prediction
        recovered = await runner.poll_once()
        unchanged = await runner.poll_once()
        return recovered, unchanged

    recovered, unchanged = asyncio.run(scenario())

    assert recovered.status is PollStatus.PREDICTION_RECOVERED
    assert recovered.prediction is not None
    assert recovered.prediction.context_event_ids == ("event-2", "event-3")
    assert unchanged.status is PollStatus.UNCHANGED
    assert factory.sample_counts == [1]
    assert len(predictor.calls) == 2
    assert 'id="event-2"' in predictor.calls[-1][0]
    assert 'id="event-3"' in predictor.calls[-1][0]
    assert len(sink.records) == 1
    state = runner.state_store.load()
    assert state is not None and not state.has_pending_prediction


def test_run_releases_its_lease_when_cancelled(tmp_path):
    observed = asyncio.Event()

    async def on_result(_result):
        observed.set()

    runner, _factory, _store = service(
        tmp_path,
        MutableJournal(),
        on_result=on_result,
    )

    async def scenario():
        task = asyncio.create_task(runner.run())
        await asyncio.wait_for(observed.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        stopped = asyncio.Event()
        stopped.set()
        summary = await runner.run(stop_event=stopped)
        assert summary.polls == 0

    asyncio.run(scenario())


def test_prediction_sink_is_idempotent_and_repairs_an_interrupted_tail(tmp_path):
    path = tmp_path / "predictions.jsonl"
    sink = JsonlPredictionSink(path)
    prediction = Prediction(
        id="prediction-1",
        created_at=10,
        cutoff_ts=9,
        actions=("next",),
        raw_text="<actions><action>next</action></actions>",
    )

    assert sink.append(prediction)
    assert not sink.append(prediction)
    with pytest.raises(RunnerStateError, match="different content"):
        sink.append(replace(prediction, actions=("different",)))

    with path.open("ab") as handle:
        handle.write(b'{"record_type":"prediction"')
    second = replace(prediction, id="prediction-2")
    assert sink.append(second)
    assert path.read_bytes().endswith(b"\n")
    assert path.read_text(encoding="utf-8").count("\n") == 2
