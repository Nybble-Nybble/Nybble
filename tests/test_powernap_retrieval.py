import asyncio

import pytest

from nybble.config import PowerNapConfig
from nybble.powernap.checkpoints import CheckpointRecord
from nybble.powernap.predictor import TinkerPowerNapPredictor
from nybble.powernap.retrieval_budget import pack_retrieved_memories


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)


def hit(text, score, event_ts, doc_id):
    return {"text": text, "score": score, "event_ts": event_ts, "id": doc_id}


def test_budget_drops_lowest_scoring_oldest_memory_and_preserves_mmr_order():
    hits = [
        hit("high value has four", 3, 10, "high"),
        hit("older low memory", 1, 1, "old-low"),
        hit("newer low memory", 1, 2, "new-low"),
    ]

    packed = pack_retrieved_memories(hits, WhitespaceTokenizer(), max_tokens=7)

    assert packed == "high value has four\n\nnewer low memory"


def test_individually_oversized_memory_does_not_crowd_out_one_that_fits():
    hits = [
        hit("one two three four five six", 10, 10, "oversized"),
        hit("small memory", 1, 1, "small"),
    ]

    assert pack_retrieved_memories(hits, WhitespaceTokenizer(), max_tokens=4) == "small memory"


def test_budget_counts_the_exact_joined_payload_including_separators():
    hits = [hit("aa", 2, 1, "first"), hit("bb", 1, 2, "second")]

    assert pack_retrieved_memories(hits, CharacterTokenizer(), max_tokens=5) == "aa"
    assert pack_retrieved_memories(hits, CharacterTokenizer(), max_tokens=6) == "aa\n\nbb"


def test_zero_budget_disables_retrieval_and_invalid_budget_is_rejected():
    hits = [hit("memory", 1, 1, "one")]

    assert pack_retrieved_memories(hits, WhitespaceTokenizer(), max_tokens=0) == ""
    with pytest.raises(ValueError, match="nonnegative integer"):
        pack_retrieved_memories(hits, WhitespaceTokenizer(), max_tokens=-1)


def test_powernap_config_allows_disabling_retrieval_memory_budget():
    assert PowerNapConfig(retrieval_max_tokens=0).retrieval_max_tokens == 0
    for value in (-1, True):
        with pytest.raises(ValueError, match="retrieval_max_tokens"):
            PowerNapConfig(retrieval_max_tokens=value)


def test_predictor_applies_the_same_budget_to_oversized_memories(tmp_path):
    class Retriever:
        def query(self, *_args, **_kwargs):
            return [
                hit("one two three four five", 2, 2, "oversized"),
                hit("small memory", 1, 1, "fits"),
            ]

    checkpoint = CheckpointRecord.create(
        step=1,
        state_path="tinker://state/1",
        sampler_path="tinker://sampler/1",
        retriever_path=str(tmp_path / "retriever.json.gz"),
        model="thinkingmachines/Inkling-Small",
        renderer="tml_v0",
        last_sample_id="sample",
    )
    predictor = TinkerPowerNapPredictor(
        config=PowerNapConfig(
            future_len=1,
            retrieval_top_k=2,
            retrieval_mmr_k=2,
            retrieval_max_tokens=2,
        ),
        checkpoint=checkpoint,
        retriever=Retriever(),
        tinker_api_key="fake",
    )
    predictor._sampling_client = object()
    predictor._renderer = object()
    predictor._tokenizer = WhitespaceTokenizer()
    tagged = iter(
        (
            ("<rationale>reason</rationale>", "reason"),
            ("<revise>revision</revise>", "revision"),
        )
    )

    async def sample_tagged(_messages, _tag, _attempts):
        return next(tagged)

    async def sample_actions(_messages):
        return "<actions><action>next</action></actions>"

    predictor._sample_tagged = sample_tagged
    predictor._sample = sample_actions

    trace = asyncio.run(predictor.predict("context", 10, future_len=1, action_attempts=1))

    assert trace.retrieved == "small memory"
