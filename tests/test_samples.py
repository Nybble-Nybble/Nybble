import pytest

from nybble.data.samples import (
    ContextBuilder,
    SampleBuilder,
    event_stream_key,
    select_latest_visible_context,
    select_recent_image_paths,
)
from nybble.models import ActionEvent, TrainingSample
from nybble.pipeline import to_rollout_sample


def event(event_id, timestamp, image=True, **overrides):
    values = {
        "id": event_id,
        "start_ts": timestamp,
        "end_ts": timestamp + 0.5,
        "available_at": timestamp + 1,
        "source": "glasses",
        "text": f"text-{event_id}",
        "dense_context": f"dense-{event_id}",
        "image_path": f"/frames/{event_id}.jpg" if image else None,
    }
    values.update(overrides)
    return ActionEvent(
        **values,
    )


def test_sliding_windows_are_distinct_bounded_and_complete():
    events = [event(f"e{index}", index) for index in range(7)]
    samples = SampleBuilder(past_len=3, future_len=2).build(reversed(events))

    assert len(samples) == 3
    assert len({sample.id for sample in samples}) == 3
    assert [sample.past_event_ids for sample in samples] == [
        ("e0", "e1", "e2"),
        ("e1", "e2", "e3"),
        ("e2", "e3", "e4"),
    ]
    assert [sample.future_event_ids for sample in samples] == [
        ("e3", "e4"),
        ("e4", "e5"),
        ("e5", "e6"),
    ]
    assert all(sample.past_len == 3 and sample.future_len == 2 for sample in samples)


def test_equal_timestamps_do_not_expand_context_across_boundary():
    events = [event(event_id, 10, end_ts=10, available_at=10) for event_id in ("a", "b", "c", "d")]
    samples = SampleBuilder(past_len=2, future_len=1).build(events)

    first = samples[0]
    assert first.past_event_ids == ("a", "b")
    assert first.future_event_ids == ("c",)
    assert "text-a" in first.context_text
    assert "text-b" in first.context_text
    assert "text-c" not in first.context_text
    assert "text-c" in first.solution_text
    assert "text-d" not in first.solution_text


def test_context_builder_selects_ids_not_timestamp_ranges():
    events = [event(event_id, 10) for event_id in ("a", "b", "c")]
    builder = ContextBuilder()

    rendered = builder.render_selected(events, ("b",))
    assert "text-b" in rendered
    assert "text-a" not in rendered
    assert "text-c" not in rendered
    with pytest.raises(KeyError):
        builder.render_selected(events, ("missing",))


def test_window_is_skipped_when_future_started_before_past_labels_were_available():
    events = [
        event("past-a", 1, end_ts=2, available_at=10),
        event("past-b", 3, end_ts=4, available_at=10),
        event("overlapping-future", 9, end_ts=10, available_at=10),
    ]

    builder = SampleBuilder(past_len=2, future_len=1)
    assert builder.build(events) == ()
    with pytest.raises(ValueError, match="availability cutoff"):
        builder.build_selected(events[:2], events[2:3])


def test_window_is_skipped_when_labeling_chunk_crosses_boundary():
    events = [
        event("past-a", 1, end_ts=1, available_at=1, chunk_id="chunk-a"),
        event("past-b", 2, end_ts=2, available_at=2, chunk_id="shared"),
        event("future", 3, end_ts=3, available_at=3, chunk_id="shared"),
    ]

    builder = SampleBuilder(past_len=2, future_len=1)
    assert builder.build(events) == ()
    with pytest.raises(ValueError, match="share a labeling chunk"):
        builder.build_selected(events[:2], events[2:])


def test_context_and_solution_escape_model_data():
    unsafe = ActionEvent(
        id='id-"',
        start_ts=1,
        end_ts=2,
        available_at=2,
        source="a&b",
        text="open <settings>",
    )
    builder = ContextBuilder()

    assert "open &lt;settings&gt;" in builder.render_context((unsafe,))
    assert "a&amp;b" in builder.render_context((unsafe,))
    assert "open &lt;settings&gt;" in builder.render_solution((unsafe,))


def test_image_selection_is_past_only_and_capped():
    events = [event(f"e{index}", index) for index in range(5)]
    sample = SampleBuilder(past_len=3, future_len=2, max_images=2).latest(events)

    assert sample is not None
    assert sample.image_paths == ("/frames/e1.jpg", "/frames/e2.jpg")
    assert "/frames/e3.jpg" not in sample.image_paths


def test_image_selection_filters_missing_before_cap_and_can_be_disabled():
    events = [
        event("e0", 0),
        event("e1", 1, image=False),
        event("e2", 2),
        event("e3", 3),
    ]

    assert select_recent_image_paths(events, 2) == (
        "/frames/e2.jpg",
        "/frames/e3.jpg",
    )
    assert select_recent_image_paths(events, 0) == ()


def test_sample_roundtrip_preserves_shared_rendering():
    events = [event(f"e{index}", index) for index in range(3)]
    sample = SampleBuilder(past_len=2, future_len=1).latest(events)

    assert sample is not None
    restored = TrainingSample.from_dict(sample.to_dict())
    assert restored == sample
    assert restored.context_text == sample.context_text
    assert restored.solution_text == sample.solution_text


def test_insufficient_events_and_stride():
    events = [event(f"e{index}", index) for index in range(8)]
    assert SampleBuilder(past_len=3, future_len=2).build(events[:4]) == ()
    samples = SampleBuilder(past_len=2, future_len=2, stride=2).build(events)
    assert [sample.past_event_ids for sample in samples] == [
        ("e0", "e1"),
        ("e2", "e3"),
        ("e4", "e5"),
    ]


def test_windows_never_bridge_capture_sessions_or_sources():
    events = [
        event(
            f"a-{index}",
            index,
            provenance={"session_id": "session-a"},
        )
        for index in range(3)
    ] + [
        event(
            f"b-{index}",
            index + 10,
            source="other-glasses",
            provenance={"session_id": "session-b"},
        )
        for index in range(3)
    ]

    samples = SampleBuilder(past_len=2, future_len=1).build(reversed(events))

    assert [sample.past_event_ids for sample in samples] == [
        ("a-0", "a-1"),
        ("b-0", "b-1"),
    ]
    assert [sample.future_event_ids for sample in samples] == [("a-2",), ("b-2",)]
    assert all(
        len({event_stream_key(event) for event in sample.past_events + sample.future_events}) == 1
        for sample in samples
    )


def test_cross_session_explicit_window_is_rejected():
    past = (
        event("a", 1, provenance={"session_id": "first"}),
        event("b", 2, provenance={"session_id": "first"}),
    )
    future = (event("c", 3, provenance={"session_id": "second"}),)

    with pytest.raises(ValueError, match="one source session"):
        SampleBuilder(past_len=2, future_len=1).build_selected(past, future)


def test_latest_context_uses_only_the_latest_active_session():
    events = [
        event(
            f"old-{index}",
            index,
            provenance={"session_id": "old"},
        )
        for index in range(4)
    ] + [
        event(
            f"new-{index}",
            index + 10,
            provenance={"session_id": "new"},
        )
        for index in range(2)
    ]

    selected = select_latest_visible_context(events, cutoff_ts=20, past_len=4)

    assert tuple(item.id for item in selected) == ("new-0", "new-1")


def test_provisional_tail_actions_are_visible_for_prediction_but_never_training():
    stable = [event(f"stable-{index}", index) for index in range(3)]
    provisional = event("tail", 3, metadata={"provisional": True})
    events = [*stable, provisional]

    samples = SampleBuilder(past_len=2, future_len=1).build(events)
    visible = select_latest_visible_context(events, cutoff_ts=10, past_len=4)

    assert len(samples) == 1
    assert samples[0].future_event_ids == ("stable-2",)
    assert tuple(item.id for item in visible) == (
        "stable-0",
        "stable-1",
        "stable-2",
        "tail",
    )
    with pytest.raises(ValueError, match="provisional"):
        SampleBuilder(past_len=2, future_len=1).build_selected(
            tuple(stable[1:]),
            (provisional,),
        )


def test_rollout_memory_waits_for_future_label_availability():
    past = event("past", 1, end_ts=1, available_at=2, image=False)
    future = event("future", 3, end_ts=3, available_at=5, image=False)
    sample = SampleBuilder(past_len=1, future_len=1).build_selected((past,), (future,))

    rollout = to_rollout_sample(sample)
    assert rollout.target_end_ts == 3
    assert rollout.target_visible_after_ts == 5
