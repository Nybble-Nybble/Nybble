from pathlib import Path

from PIL import Image

from nybble.data import EventJournal, SampleBuilder
from nybble.labeling import ActionSpan, FrameLabeler, GroupingResult
from nybble.pipeline import labeling_result_to_records, to_rollout_sample


class FakeCaptioner:
    model = "gemini-fake"
    prompt_version = "caption-v1"

    def __init__(self):
        self.calls = 0

    def caption_frame(self, frame: Path) -> str:
        self.calls += 1
        return f"The wearer observes {frame.stem}."


class FakeGrouper:
    model = "gemini-fake"
    prompt_version = "group-v1"

    def __init__(self):
        self.calls = 0

    def group(self, frames):
        self.calls += 1
        return GroupingResult(
            actions=(ActionSpan(0, len(frames) - 1, f"Reviews {frames[-1].path.stem}"),),
            dense_context="The wearer is reviewing a short sequence.",
        )


def _write_frames(folder: Path, count: int) -> None:
    folder.mkdir(exist_ok=True)
    for index in range(count):
        Image.new("RGB", (12, 8), (index * 20, 10, 30)).save(folder / f"frame-{index}.jpg")


def test_frames_to_resumable_labels_to_causal_powernap_sample(tmp_path):
    frame_dir = tmp_path / "frames"
    _write_frames(frame_dir, 6)
    cache = tmp_path / "label-cache.jsonl"
    captioner = FakeCaptioner()
    grouper = FakeGrouper()
    labeler = FrameLabeler(
        captioner=captioner,
        grouper=grouper,
        chunk_size=2,
        max_gap_seconds=30,
        caption_workers=3,
        cache_path=cache,
    )

    labels = labeler.label_directory(frame_dir, fps=1, start_time="2026-01-01T00:00:00Z")
    assert captioner.calls == 6
    assert grouper.calls == 3

    resumed_captioner = FakeCaptioner()
    resumed_grouper = FakeGrouper()
    resumed = FrameLabeler(
        captioner=resumed_captioner,
        grouper=resumed_grouper,
        chunk_size=2,
        max_gap_seconds=30,
        caption_workers=2,
        cache_path=cache,
    ).label_directory(frame_dir, fps=1, start_time="2026-01-01T00:00:00Z")
    assert resumed == labels
    assert resumed_captioner.calls == 0
    assert resumed_grouper.calls == 0

    frames, actions = labeling_result_to_records(
        labels,
        source="meta-glasses",
        grouping_model=grouper.model,
        grouping_prompt_version=grouper.prompt_version,
    )
    assert len(frames) == 6
    assert len(actions) == 3
    assert all(action.available_at >= action.end_ts for action in actions)
    assert all(action.dense_context for action in actions)

    journal = EventJournal(tmp_path / "actions.jsonl")
    assert journal.append_many(actions) == 3
    assert journal.append_many(actions) == 0
    samples = SampleBuilder(past_len=2, future_len=1, max_images=None).build(journal.read())
    assert len(samples) == 1
    rollout = to_rollout_sample(samples[0])
    assert rollout.future_len == 1
    assert rollout.image_paths == samples[0].image_paths
    assert set(rollout.past_event_ids).isdisjoint(rollout.future_event_ids)
    assert "Reviews frame-5" in rollout.solution_text


def test_growing_tail_transactionally_replaces_stale_session_actions(tmp_path):
    frame_dir = tmp_path / "frames"
    _write_frames(frame_dir, 3)
    labeler = FrameLabeler(
        captioner=FakeCaptioner(),
        grouper=FakeGrouper(),
        chunk_size=2,
        max_gap_seconds=30,
        caption_workers=1,
        cache_path=tmp_path / "label-cache.jsonl",
    )
    first = labeler.label_directory(
        frame_dir,
        fps=1,
        start_time="2026-01-01T00:00:00Z",
    )
    _frames, old_actions = labeling_result_to_records(
        first,
        source="meta-glasses",
        grouping_model="gemini-fake",
        grouping_prompt_version="group-v1",
    )
    assert [action.metadata["provisional"] for action in old_actions] == [False, True]
    journal = EventJournal(tmp_path / "actions.jsonl")
    session_ids = {chunk.session_id for chunk in first.chunks}
    assert journal.replace_source_sessions(
        old_actions,
        source="meta-glasses",
        session_ids=session_ids,
    ) == (0, 2)

    _write_frames(frame_dir, 5)
    grown = labeler.label_directory(
        frame_dir,
        fps=1,
        start_time="2026-01-01T00:00:00Z",
    )
    _frames, current_actions = labeling_result_to_records(
        grown,
        source="meta-glasses",
        grouping_model="gemini-fake",
        grouping_prompt_version="group-v1",
    )
    assert [action.metadata["provisional"] for action in current_actions] == [
        False,
        False,
        True,
    ]
    current_sessions = {chunk.session_id for chunk in grown.chunks}

    assert current_sessions == session_ids
    assert journal.replace_source_sessions(
        current_actions,
        source="meta-glasses",
        session_ids=current_sessions,
    ) == (2, 3)
    assert journal.read() == tuple(current_actions)
    assert set(event.id for event in old_actions) - set(event.id for event in current_actions)
    assert not (
        set(event.id for event in old_actions) - set(event.id for event in current_actions)
    ) & {event.id for event in journal.read()}
