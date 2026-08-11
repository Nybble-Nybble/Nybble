import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from nybble.cli import _reward_model_for_run
from nybble.config import PowerNapConfig
from nybble.powernap.checkpoints import CheckpointRecord, CheckpointStore
from nybble.powernap.rewards import GeminiRewardJudge
from nybble.powernap.trainer import TinkerPowerNapTrainer


class RewardGenerator:
    def __init__(self, model="gemini-3.6-flash"):
        self.model = model


def test_checkpoint_manifest_round_trip(tmp_path):
    store = CheckpointStore(tmp_path / "run" / "checkpoints.jsonl")
    first = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever-2.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
        sample_ids=("sample-2",),
        sample_digests=("a" * 64,),
        reward_contract=GeminiRewardJudge(RewardGenerator()).contract,
    )
    second = CheckpointRecord.create(
        step=4,
        state_path="tinker://state/4",
        sampler_path="tinker://sampler/4",
        retriever_path=str(tmp_path / "retriever-4.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-4",
    )
    store.append(first)
    store.append(second)
    assert store.all() == [first, second]
    assert store.latest() == second


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("model", "gemini-different"),
        ("prompt_version", "different-prompt-v2"),
        ("schema_version", 999),
    ),
)
def test_resume_rejects_a_changed_gemini_reward_contract(tmp_path, field, changed):
    config = PowerNapConfig()
    judge = GeminiRewardJudge(RewardGenerator())
    checkpoint_contract = {**judge.contract, field: changed}
    record = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever.json.gz"),
        model=config.model,
        renderer=config.renderer,
        last_sample_id="sample-2",
        config=asdict(config),
        reward_contract=checkpoint_contract,
    )

    with pytest.raises(ValueError, match="reward contract"):
        TinkerPowerNapTrainer(
            config=config,
            reward_judge=judge,
            retriever=object(),
            run_dir=tmp_path / f"resume-{field}",
            tinker_api_key="fake",
            resume=record,
        )


def test_cli_resume_uses_the_checkpoint_reward_model(tmp_path):
    contract = GeminiRewardJudge(RewardGenerator("gemini-checkpoint")).contract
    record = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
        reward_contract=contract,
    )

    assert _reward_model_for_run("gemini-new-default", record) == "gemini-checkpoint"


def test_checkpoint_manifest_repairs_an_interrupted_final_append(tmp_path):
    path = tmp_path / "run" / "checkpoints.jsonl"
    store = CheckpointStore(path)
    first = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever-2.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
    )
    second = CheckpointRecord.create(
        step=4,
        state_path="tinker://state/4",
        sampler_path="tinker://sampler/4",
        retriever_path=str(tmp_path / "retriever-4.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-4",
    )
    store.append(first)
    partial = b'{"created_at":123,"step":'
    with path.open("ab") as handle:
        handle.write(partial)

    assert store.recover() == len(partial)
    assert store.all() == [first]
    store.append(second)
    assert store.all() == [first, second]
    assert path.read_bytes().endswith(b"\n")


def test_checkpoint_manifest_rejects_malformed_complete_records(tmp_path):
    path = tmp_path / "run" / "checkpoints.jsonl"
    store = CheckpointStore(path)
    path.write_bytes(b'{"step":}\n')

    with pytest.raises(ValueError, match="line 1"):
        store.all()


def test_checkpoint_manifest_normalizes_a_valid_unterminated_record(tmp_path):
    path = tmp_path / "run" / "checkpoints.jsonl"
    store = CheckpointStore(path)
    record = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever-2.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record)), encoding="utf-8")

    assert store.all() == [record]
    assert path.read_bytes().endswith(b"\n")


def test_checkpoint_manifest_loads_schema_one_records_without_content_digests(tmp_path):
    path = tmp_path / "run" / "checkpoints.jsonl"
    store = CheckpointStore(path)
    current = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever-2.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
        sample_ids=("sample-1", "sample-2"),
    )
    legacy = asdict(current)
    legacy.pop("sample_digests")
    legacy["schema_version"] = 1
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    loaded = store.latest()
    assert loaded is not None
    assert loaded.schema_version == 1
    assert loaded.sample_ids == ("sample-1", "sample-2")
    assert loaded.sample_digests == ()


class SaveResult:
    def __init__(self, path):
        self.path = path


class SaveFuture:
    def __init__(self, path):
        self.path = path

    async def result_async(self):
        return SaveResult(self.path)


class TrainingClient:
    def __init__(self):
        self.calls = 0

    async def save_state_async(self, name, *, ttl_seconds):
        self.calls += 1
        return SaveFuture(f"tinker://state/{name}/{self.calls}")

    async def save_weights_for_sampler_async(self, name, *, ttl_seconds):
        self.calls += 1
        return SaveFuture(f"tinker://sampler/{name}/{self.calls}")


class Retriever:
    def __init__(self):
        self.calls = 0

    def save_checkpoint(self, path):
        self.calls += 1
        Path(path).write_text(str(self.calls), encoding="utf-8")


def test_final_retriever_snapshots_are_immutable_and_unique(tmp_path):
    trainer = TinkerPowerNapTrainer(
        config=PowerNapConfig(),
        reward_judge=object(),
        retriever=Retriever(),
        run_dir=tmp_path / "run",
        tinker_api_key="fake",
    )
    client = TrainingClient()

    first = asyncio.run(trainer._save_checkpoint(client, 2, "sample-2", durable=True))
    second = asyncio.run(trainer._save_checkpoint(client, 4, "sample-4", durable=True))

    assert first.retriever_path != second.retriever_path
    assert Path(first.retriever_path).read_text(encoding="utf-8") == "1"
    assert Path(second.retriever_path).read_text(encoding="utf-8") == "2"
    assert Path(first.retriever_path).is_absolute()


def test_fresh_training_refuses_a_nonempty_run_and_resume_validates_config(tmp_path):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir / "checkpoints.jsonl")
    record = CheckpointRecord.create(
        step=2,
        state_path="tinker://state/2",
        sampler_path="tinker://sampler/2",
        retriever_path=str(tmp_path / "retriever.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample-2",
        config={"different": True},
    )
    store.append(record)

    with pytest.raises(RuntimeError, match="already contains"):
        TinkerPowerNapTrainer(
            config=PowerNapConfig(),
            reward_judge=object(),
            retriever=object(),
            run_dir=run_dir,
            tinker_api_key="fake",
        )
    with pytest.raises(ValueError, match="config"):
        TinkerPowerNapTrainer(
            config=PowerNapConfig(),
            reward_judge=object(),
            retriever=object(),
            run_dir=tmp_path / "resume",
            tinker_api_key="fake",
            resume=record,
        )
