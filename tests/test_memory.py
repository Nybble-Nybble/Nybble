import json

from nybble.powernap.memory import build_memory_text, should_index_memory
from nybble.powernap.rewards import RewardResult


def test_memory_contains_completed_case_without_nested_revision_tags():
    text = build_memory_text(
        context="observed",
        revision="<revise>one tag</revise>",
        prediction="<actions><action>next</action></actions>",
        outcome="<actions><action>actual</action></actions>",
        accuracy=0.75,
    )
    value = json.loads(text)

    assert value["revision"] == "<revise>one tag</revise>"
    assert text.count("<revise>") == 1
    assert value["predicted_actions"]
    assert value["observed_outcome"]


def test_memory_rejects_format_only_or_invalid_winners():
    format_only = RewardResult(0.5, 0.0, 1.0, True)
    invalid = RewardResult(0.0, 0.9, 0.0, False)
    useful = RewardResult(0.8, 0.6, 1.0, True)

    assert not should_index_memory(format_only, 0.5)
    assert not should_index_memory(invalid, 0.5)
    assert should_index_memory(useful, 0.5)
