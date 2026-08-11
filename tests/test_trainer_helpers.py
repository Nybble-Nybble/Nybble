import asyncio
from dataclasses import replace

import pytest
from PIL import Image

from nybble.config import PowerNapConfig
from nybble.powernap.checkpoints import CheckpointRecord
from nybble.powernap.trainer import (
    _forward_backward_once,
    _optimizer_step_once,
    _validate_resume_samples,
    distinct_batch,
    rollout_sample_digest,
)
from nybble.powernap.types import RolloutSample


def sample(number):
    return RolloutSample(
        id=f"sample-{number}",
        context_text=f"context {number}",
        solution_text="<actions><action>next</action></actions>",
        cutoff_ts=float(number),
        target_end_ts=float(number + 1),
        future_len=1,
        past_event_ids=(f"past-{number}",),
        future_event_ids=(f"future-{number}",),
    )


def test_distinct_batch_never_clones_latest_window():
    samples = [sample(index) for index in range(5)]
    batch = distinct_batch(samples, step=0, batch_size=8)
    assert len(batch) == 5
    assert len({item.id for item in batch}) == 5


def test_distinct_batch_is_deterministic_and_rotates():
    samples = [sample(index) for index in range(8)]
    first = distinct_batch(samples, step=0, batch_size=3, seed=9)
    assert first == distinct_batch(samples, step=0, batch_size=3, seed=9)
    second = distinct_batch(samples, step=1, batch_size=3, seed=9)
    assert {item.id for item in first} != {item.id for item in second}


def test_rollout_sample_digest_binds_all_semantic_fields_and_image_content(tmp_path):
    original = sample(1)
    original_digest = rollout_sample_digest(original)
    changed_samples = (
        replace(original, id="different-id"),
        replace(original, context_text="different context"),
        replace(original, solution_text="<actions><action>different</action></actions>"),
        replace(original, cutoff_ts=1.5),
        replace(original, target_end_ts=2.5, target_visible_after_ts=2.5),
        replace(original, target_visible_after_ts=3.0),
        replace(original, past_event_ids=("different-past",)),
        replace(original, future_event_ids=("different-future",)),
    )
    assert all(rollout_sample_digest(changed) != original_digest for changed in changed_samples)

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    with_image = replace(original, image_paths=(str(image_path),))
    first_image_digest = rollout_sample_digest(with_image)
    Image.new("RGB", (4, 4), "blue").save(image_path)

    assert rollout_sample_digest(with_image) != first_image_digest


def test_resume_content_digests_allow_appends_but_reject_a_changed_prefix():
    originals = (sample(1), sample(2))
    sample_ids = tuple(item.id for item in originals)
    sample_digests = tuple(rollout_sample_digest(item) for item in originals)
    record = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path="/tmp/retriever.json.gz",
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
        sample_ids=sample_ids,
        sample_digests=sample_digests,
    )
    appended = (*originals, sample(3))
    _validate_resume_samples(
        record,
        sample_ids=tuple(item.id for item in appended),
        sample_digests=tuple(rollout_sample_digest(item) for item in appended),
        data_fingerprint="new-full-dataset-fingerprint",
    )

    changed = (originals[0], replace(originals[1], context_text="changed"), sample(3))
    with pytest.raises(ValueError, match="sample content"):
        _validate_resume_samples(
            record,
            sample_ids=tuple(item.id for item in changed),
            sample_digests=tuple(rollout_sample_digest(item) for item in changed),
            data_fingerprint="new-full-dataset-fingerprint",
        )


class FailedFuture:
    async def result_async(self):
        raise ConnectionError("response was lost")


class MutationClient:
    def __init__(self):
        self.forward_calls = 0
        self.optimizer_calls = 0

    async def forward_backward_async(self, datums, *, loss_fn):
        assert datums == ["datum"]
        assert loss_fn == "importance_sampling"
        self.forward_calls += 1
        return FailedFuture()

    async def optim_step_async(self, params):
        assert params == "adam"
        self.optimizer_calls += 1
        return FailedFuture()


def test_remote_training_mutations_are_not_reissued_after_lost_response():
    client = MutationClient()

    with pytest.raises(ConnectionError, match="lost"):
        asyncio.run(_forward_backward_once(client, ["datum"]))
    with pytest.raises(ConnectionError, match="lost"):
        asyncio.run(_optimizer_step_once(client, "adam"))

    assert client.forward_calls == 1
    assert client.optimizer_calls == 1


def test_max_step_attempts_cannot_be_less_than_required_steps(tmp_path):
    from nybble.powernap.trainer import TinkerPowerNapTrainer

    trainer = TinkerPowerNapTrainer(
        config=PowerNapConfig(past_len=1, future_len=1),
        reward_judge=object(),
        retriever=object(),
        run_dir=tmp_path,
        tinker_api_key="fake",
    )

    with pytest.raises(ValueError, match="at least steps"):
        asyncio.run(trainer.train([sample(0)], steps=2, max_step_attempts=1))
