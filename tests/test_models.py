import json
from dataclasses import FrozenInstanceError

import pytest

from nybble.models import ActionEvent, FrameObservation, Prediction, TrainingSample


def event(event_id="event-1", timestamp=10.0, **overrides):
    values = {
        "id": event_id,
        "start_ts": timestamp,
        "end_ts": timestamp + 1,
        "available_at": timestamp + 2,
        "source": "glasses",
        "text": f"action {event_id}",
        "dense_context": "At a desk",
        "start_frame": "frame-a",
        "end_frame": "frame-b",
        "image_path": f"/frames/{event_id}.jpg",
        "chunk_id": "chunk-1",
        "provenance": {"frames": ["frame-a", "frame-b"]},
        "model": "gemini-test",
        "prompt_version": "v1",
        "metadata": {"confidence": 0.9},
    }
    values.update(overrides)
    return ActionEvent(**values)


def test_frame_observation_is_strict_immutable_and_roundtrips():
    observation = FrameObservation(
        id="frame-1",
        timestamp=10,
        source="glasses",
        image_path="/frames/1.jpg",
        text="Looking at a laptop",
        provenance={"capture": {"fps": 5}, "indices": [1, 2]},
        model="gemini-test",
    )

    with pytest.raises(FrozenInstanceError):
        observation.text = "changed"
    with pytest.raises(TypeError):
        observation.provenance["capture"] = {}
    with pytest.raises(TypeError):
        observation.provenance["capture"]["fps"] = 10

    payload = observation.to_dict()
    assert payload["schema_version"] == 1
    assert FrameObservation.from_dict(payload) == observation
    json.dumps(payload)


def test_action_event_validates_temporal_and_schema_invariants():
    with pytest.raises(ValueError, match="end_ts"):
        event(end_ts=9)
    with pytest.raises(ValueError, match="available_at"):
        event(available_at=10.5)
    with pytest.raises(TypeError, match="start_ts"):
        event(start_ts=True)
    with pytest.raises(ValueError, match="schema_version"):
        event(schema_version=2)
    with pytest.raises(ValueError, match="text"):
        event(text=" ")


def test_action_event_roundtrip_rejects_unknown_fields():
    original = event()
    assert ActionEvent.from_dict(original.to_dict()) == original

    payload = original.to_dict()
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        ActionEvent.from_dict(payload)


def test_training_sample_embeds_exact_events_and_roundtrips():
    past = (event("past", 10, chunk_id=None),)
    future = (event("future", 20, chunk_id=None),)
    sample = TrainingSample(
        id="sample-1",
        past_events=past,
        future_events=future,
        cutoff_ts=12,
        start_ts=10,
        end_ts=21,
        context_text="<observed_actions />",
        solution_text="<actions><action>next</action></actions>",
        image_paths=("/frames/past.jpg",),
    )

    assert sample.past_event_ids == ("past",)
    assert sample.future_event_ids == ("future",)
    assert sample.past_len == 1
    assert sample.future_len == 1
    assert TrainingSample.from_dict(sample.to_dict()) == sample
    json.dumps(sample.to_dict())


def test_training_sample_rejects_overlap_and_inexact_boundaries():
    past = (event("past", 10),)
    with pytest.raises(ValueError, match="disjoint"):
        TrainingSample(
            id="sample-1",
            past_events=past,
            future_events=past,
            cutoff_ts=12,
            start_ts=10,
            end_ts=11,
            context_text="context",
            solution_text="solution",
        )

    with pytest.raises(ValueError, match="first selected"):
        TrainingSample(
            id="sample-2",
            past_events=past,
            future_events=(event("future", 20),),
            cutoff_ts=12,
            start_ts=9,
            end_ts=21,
            context_text="context",
            solution_text="solution",
        )


def test_training_sample_rejects_cross_session_and_provisional_events():
    past = (
        event(
            "past",
            10,
            chunk_id=None,
            provenance={"session_id": "first"},
        ),
    )
    future = (
        event(
            "future",
            20,
            chunk_id=None,
            provenance={"session_id": "second"},
        ),
    )
    values = {
        "id": "sample-session",
        "past_events": past,
        "future_events": future,
        "cutoff_ts": 12,
        "start_ts": 10,
        "end_ts": 21,
        "context_text": "context",
        "solution_text": "solution",
    }

    with pytest.raises(ValueError, match="one source session"):
        TrainingSample(**values)

    provisional_future = (
        event(
            "future",
            20,
            chunk_id=None,
            provenance={"session_id": "first"},
            metadata={"provisional": True},
        ),
    )
    with pytest.raises(ValueError, match="provisional"):
        TrainingSample(**{**values, "future_events": provisional_future})


def test_prediction_trace_roundtrip_and_validation():
    prediction = Prediction(
        id="prediction-1",
        created_at=30,
        cutoff_ts=20,
        actions=("Open the editor", "Run the tests"),
        raw_text="<actions>...</actions>",
        rationale="The user is coding",
        revision="Tests likely follow",
        retrieved=("A previous coding session",),
        context_event_ids=("event-a", "event-b"),
        model="thinkingmachines/Inkling-Small",
        checkpoint="tinker://checkpoint",
        metadata={"latency_ms": 12},
    )

    assert Prediction.from_dict(prediction.to_dict()) == prediction
    json.dumps(prediction.to_dict())

    with pytest.raises(ValueError, match="actions"):
        Prediction(
            id="prediction-2",
            created_at=30,
            cutoff_ts=20,
            actions=(),
            raw_text="",
        )
    with pytest.raises(ValueError, match="created_at"):
        Prediction(
            id="prediction-3",
            created_at=19,
            cutoff_ts=20,
            actions=("next",),
            raw_text="next",
        )
