import asyncio

import pytest

pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

import nybble.powernap.environment as environment_module
from nybble.powernap.environment import LongNAPEnv
from nybble.powernap.types import RolloutSample


class Termination:
    is_clean = True


class Renderer:
    def __init__(self, text):
        self.text = text
        self.parse_calls = 0

    def parse_response(self, action):
        self.parse_calls += 1
        return self.text, Termination()

    def get_stop_sequences(self):
        return [200006]


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()


class Retriever:
    def query(self, *_args, **_kwargs):
        return [
            {
                "id": "oversized",
                "text": "one two three four five",
                "score": 2.0,
                "event_ts": 2,
            },
            {"id": "fits", "text": "small memory", "score": 1.0, "event_ts": 1},
        ]


def sample():
    return RolloutSample(
        id="sample",
        context_text="context",
        solution_text="<actions><action>next</action></actions>",
        cutoff_ts=1,
        target_end_ts=2,
        future_len=1,
        past_event_ids=("past",),
        future_event_ids=("future",),
    )


def make_env(renderer):
    return LongNAPEnv(
        sample(),
        renderer,
        tokenizer=object(),
        retriever=None,
        retrieval_top_k=1,
        retrieval_mmr_k=1,
        retrieval_max_tokens=32,
        retrieval_mmr_alpha=0.5,
        retrieval_time_decay=0,
        effort=0,
    )


def test_length_stop_terminates_with_training_metric_without_parsing():
    renderer = Renderer("<rationale>unused</rationale>")
    result = asyncio.run(make_env(renderer).step([1], extra={"stop_reason": "length"}))

    assert result.episode_done
    assert result.metrics["stop/max_tokens"] == 1
    assert renderer.parse_calls == 0


def test_wrong_phase_tag_terminates_with_parse_metric(monkeypatch):
    monkeypatch.setattr(environment_module, "get_text_content", lambda message: message)
    env = make_env(Renderer("<revise>wrong phase</revise>"))
    result = asyncio.run(env.step([1], extra={"stop_reason": "stop"}))

    assert result.episode_done
    assert result.metrics["stop/parse_error"] == 1
    assert env.actions_text == ""


def test_environment_drops_oversized_retrieval_memory_before_prompting():
    env = LongNAPEnv(
        sample(),
        Renderer("<rationale>unused</rationale>"),
        tokenizer=Tokenizer(),
        retriever=Retriever(),
        retrieval_top_k=2,
        retrieval_mmr_k=2,
        retrieval_max_tokens=2,
        retrieval_mmr_alpha=0.5,
        retrieval_time_decay=0,
        effort=0,
    )

    assert env._retrieve("rationale") == "small memory"
