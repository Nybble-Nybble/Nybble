import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from nybble.labeling.schema import FrameLabel, GroupingValidationError
from nybble.vlm.gemini import (
    GeminiActionGrouper,
    GeminiCaptioner,
    GeminiGenerator,
    RetryPolicy,
)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def make_frame_label(index=0, caption="The wearer looks at a mug."):
    now = datetime(2026, 8, 10, 12, 0, index, tzinfo=timezone.utc)
    return FrameLabel(
        index=index,
        path=Path(f"frame{index}.jpg"),
        captured_at=now,
        caption=caption,
        frame_id=f"frame-{index}",
        session_id="session-1",
        chunk_id="chunk-1",
        available_at=datetime(2026, 8, 10, 12, 0, 2, tzinfo=timezone.utc),
        model="gemini-3.6-flash",
        prompt_version="test-prompt",
    )


def test_captioner_sends_exactly_one_frame_and_retries_once(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (1200, 600), "red").save(frame)
    client = FakeClient([RuntimeError("transient"), "The wearer sees a red field."])
    delays = []
    captioner = GeminiCaptioner(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.1, max_delay_seconds=1),
        sleep=delays.append,
    )

    assert captioner.caption_frame(frame) == "The wearer sees a red field."
    assert len(client.models.calls) == 2
    assert delays == [0.1]
    call = client.models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert len(call["contents"]) == 1
    parts = call["contents"][0]["parts"]
    assert sum("inline_data" in part for part in parts) == 1
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"
    assert isinstance(parts[0]["inline_data"]["data"], bytes)
    assert "Do not infer" in parts[1]["text"]


def test_generator_has_a_hard_request_attempt_bound():
    client = FakeClient([RuntimeError("one"), RuntimeError("two"), "unused"])
    generator = GeminiGenerator(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
    )

    with pytest.raises(RuntimeError, match="two"):
        generator.generate_content("hello", {})
    assert len(client.models.calls) == 2


def test_generator_structured_retries_parse_failures_and_returns_an_object():
    client = FakeClient(["not json", '{"score":0.75}'])
    generator = GeminiGenerator(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
    )

    result = generator.generate_structured(
        "Score this prediction.",
        {"type": "object", "properties": {"score": {"type": "number"}}},
    )

    assert result == {"score": 0.75}
    assert len(client.models.calls) == 2
    config = client.models.calls[-1]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["type"] == "object"


def test_generator_structured_rejects_non_object_json():
    client = FakeClient(["[1,2,3]"])
    generator = GeminiGenerator(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        generator.generate_structured("Return an object.", {"type": "object"})


def test_grouper_uses_structured_json_and_returns_inclusive_zero_based_spans():
    payload = {
        "actions": [
            {"start_index": 0, "end_index": 1, "caption": "The wearer lifts a mug."},
            {"start_index": 2, "end_index": 2, "caption": "The wearer drinks."},
        ],
        "dense_context": "The wearer handles and drinks from a mug.",
    }
    client = FakeClient([json.dumps(payload)])
    grouper = GeminiActionGrouper(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    result = grouper.group([make_frame_label(i) for i in range(3)])

    assert [(span.start_index, span.end_index) for span in result.actions] == [
        (0, 1),
        (2, 2),
    ]
    assert result.index_basis == "zero_based"
    assert result.end_inclusive is True
    call = client.models.calls[0]
    assert call["config"]["response_mime_type"] == "application/json"
    assert "response_json_schema" in call["config"]
    assert '"index":0' in call["contents"]


@pytest.mark.parametrize(
    "actions, error",
    [
        (
            [
                {"start_index": 0, "end_index": 1, "caption": "one"},
                {"start_index": 1, "end_index": 2, "caption": "two"},
            ],
            "overlap",
        ),
        (
            [
                {"start_index": 1, "end_index": 1, "caption": "one"},
                {"start_index": 0, "end_index": 2, "caption": "two"},
            ],
            "gap",
        ),
        (
            [
                {"start_index": 0, "end_index": 0, "caption": "one"},
                {"start_index": 2, "end_index": 2, "caption": "two"},
            ],
            "gap",
        ),
        (
            [{"start_index": 0, "end_index": 1, "caption": "one"}],
            "tail",
        ),
        (
            [{"start_index": 0, "end_index": 3, "caption": "one"}],
            "exceeds",
        ),
    ],
)
def test_grouper_rejects_invalid_ranges_instead_of_clamping(actions, error):
    payload = {"actions": actions, "dense_context": "context"}
    client = FakeClient([json.dumps(payload)])
    grouper = GeminiActionGrouper(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(GroupingValidationError, match=error):
        grouper.group([make_frame_label(i) for i in range(3)])


def test_grouper_retries_malformed_json_but_never_accepts_it():
    valid = {
        "actions": [{"start_index": 0, "end_index": 0, "caption": "looks"}],
        "dense_context": "",
    }
    client = FakeClient(["not json", json.dumps(valid)])
    grouper = GeminiActionGrouper(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
    )

    assert grouper.group([make_frame_label()]).actions[0].caption == "looks"
    assert len(client.models.calls) == 2
