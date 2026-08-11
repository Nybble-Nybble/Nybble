import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from nybble.labeling.labeler import (
    FrameLabeler,
    JsonlLabelingCache,
    TimestampResolutionError,
    discover_frames,
    iter_frame_chunks,
    split_sessions,
)
from nybble.labeling.schema import ActionSpan, GroupingResult, SourceFrame
from nybble.pipeline import labeling_result_to_records


class FakeCaptioner:
    model = "gemini-3.6-flash"
    prompt_version = "test-frame-v1"

    def __init__(self):
        self.calls = []

    def caption_frame(self, path):
        self.calls.append(Path(path))
        return f"The wearer observes {Path(path).stem}."


class FakeGrouper:
    model = "gemini-3.6-flash"
    prompt_version = "test-group-v1"

    def __init__(self):
        self.calls = []

    def group(self, frames):
        self.calls.append(tuple(frames))
        return GroupingResult(
            actions=(
                ActionSpan(
                    start_index=0,
                    end_index=len(frames) - 1,
                    caption="The wearer continues one observed action.",
                ),
            ),
            dense_context="Chunk-wide context available only at chunk end.",
        )


def make_image(path):
    Image.new("RGB", (8, 8), "blue").save(path)


def test_natural_filename_order_and_manifest_timestamps(tmp_path):
    names = ["frame10.jpg", "frame2.jpg", "frame1.jpg"]
    for name in names:
        make_image(tmp_path / name)
    manifest = tmp_path / "timestamps.json"
    manifest.write_text(
        json.dumps(
            {
                "frame1.jpg": "2026-08-10T12:00:00Z",
                "frame2.jpg": "2026-08-10T12:00:01Z",
                "frame10.jpg": "2026-08-10T12:00:02Z",
            }
        ),
        encoding="utf-8",
    )

    frames = discover_frames(tmp_path, manifest_path=manifest)

    assert [frame.path.name for frame in frames] == [
        "frame1.jpg",
        "frame2.jpg",
        "frame10.jpg",
    ]
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert all(frame.captured_at.tzinfo is not None for frame in frames)


def test_frame_and_downstream_ids_bind_pixels_but_session_identity_does_not(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    manifest = tmp_path / "timestamps.json"
    manifest.write_text(json.dumps({"frame.jpg": "2026-08-10T12:00:00Z"}), encoding="utf-8")
    cache_path = tmp_path / "labels.jsonl"

    Image.new("RGB", (8, 8), "red").save(frame_path)
    first_frames = discover_frames(tmp_path, manifest_path=manifest)
    first_captioner = FakeCaptioner()
    first_grouper = FakeGrouper()
    first = FrameLabeler(
        captioner=first_captioner,
        grouper=first_grouper,
        chunk_size=1,
        cache_path=cache_path,
    ).label_frames(first_frames)

    Image.new("RGB", (8, 8), "blue").save(frame_path)
    second_frames = discover_frames(tmp_path, manifest_path=manifest)
    second_captioner = FakeCaptioner()
    second_grouper = FakeGrouper()
    second = FrameLabeler(
        captioner=second_captioner,
        grouper=second_grouper,
        chunk_size=1,
        cache_path=cache_path,
    ).label_frames(second_frames)

    assert first_frames[0].frame_id != second_frames[0].frame_id
    assert first.frames[0].frame_id != second.frames[0].frame_id
    assert first.frames[0].session_id == second.frames[0].session_id
    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id
    assert len(first_captioner.calls) == len(second_captioner.calls) == 1
    assert len(first_grouper.calls) == len(second_grouper.calls) == 1

    _, first_actions = labeling_result_to_records(
        first,
        source="test",
        grouping_model=first_grouper.model,
        grouping_prompt_version=first_grouper.prompt_version,
    )
    _, second_actions = labeling_result_to_records(
        second,
        source="test",
        grouping_model=second_grouper.model,
        grouping_prompt_version=second_grouper.prompt_version,
    )
    assert first_actions[0].id != second_actions[0].id


def test_manifest_is_authoritative_and_rejects_missing_or_backward_times(tmp_path):
    make_image(tmp_path / "frame1.jpg")
    make_image(tmp_path / "frame2.jpg")
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"frame1.jpg": "2026-08-10T12:00:00Z"}), encoding="utf-8")
    with pytest.raises(TimestampResolutionError, match="does not cover"):
        discover_frames(tmp_path, manifest_path=missing)

    backward = tmp_path / "backward.json"
    backward.write_text(
        json.dumps(
            {
                "frame1.jpg": "2026-08-10T12:00:01Z",
                "frame2.jpg": "2026-08-10T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TimestampResolutionError, match="go backward"):
        discover_frames(tmp_path, manifest_path=backward)


def test_unix_filename_then_fps_fallback(tmp_path):
    for name in ("capture_1786388400000.jpg", "capture_unknown.jpg"):
        make_image(tmp_path / name)

    frames = discover_frames(tmp_path, fps=2)

    assert frames[1].captured_at - frames[0].captured_at == timedelta(seconds=0.5)


def test_fps_resolves_a_plain_numbered_sequence(tmp_path):
    for name in ("frame1.jpg", "frame2.jpg", "frame3.jpg"):
        make_image(tmp_path / name)

    frames = discover_frames(
        tmp_path,
        fps=5,
        start_time="2026-08-10T12:00:00Z",
    )

    assert frames[2].captured_at - frames[0].captured_at == timedelta(seconds=0.4)


def test_exif_original_timestamp_resolution(tmp_path):
    frame = tmp_path / "frame.jpg"
    exif = Image.Exif()
    exif[36867] = "2026:08:10 12:34:56"
    Image.new("RGB", (8, 8), "blue").save(frame, exif=exif)

    frames = discover_frames(tmp_path)

    assert frames[0].captured_at == datetime(2026, 8, 10, 12, 34, 56, tzinfo=timezone.utc)


def source_frames(count=5, gap_after=2):
    start = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    frames = []
    for index in range(count):
        extra = 10 if index > gap_after else 0
        frames.append(
            SourceFrame(
                index=index,
                path=Path(f"frame{index}.jpg"),
                captured_at=start + timedelta(seconds=index + extra),
                frame_id=f"frame-{index}",
            )
        )
    return tuple(frames)


def test_max_gap_session_split_and_final_tail_flush():
    frames = source_frames()
    sessions = split_sessions(frames, max_gap_seconds=2)

    assert [len(session) for session in sessions] == [3, 2]
    assert [len(chunk) for chunk in iter_frame_chunks(sessions[0], 2)] == [2, 1]


def test_labeler_is_idempotent_and_keeps_dense_context_off_frames():
    frames = source_frames()
    first_captioner = FakeCaptioner()
    first_grouper = FakeGrouper()
    first = FrameLabeler(
        captioner=first_captioner,
        grouper=first_grouper,
        chunk_size=2,
        max_gap_seconds=2,
        caption_workers=1,
    ).label_frames(frames)
    second = FrameLabeler(
        captioner=FakeCaptioner(),
        grouper=FakeGrouper(),
        chunk_size=2,
        max_gap_seconds=2,
        caption_workers=1,
    ).label_frames(frames)

    assert [len(chunk.frames) for chunk in first.chunks] == [2, 1, 2]
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert [(action.start_index, action.end_index) for action in first.actions] == [
        (0, 1),
        (2, 2),
        (3, 4),
    ]
    assert first.frames[0].available_at == frames[1].captured_at
    assert first.frames[2].available_at == frames[2].captured_at
    assert first.frames[3].available_at == frames[4].captured_at
    assert "dense_context" not in first.frames[0].to_dict()
    assert first.chunks[0].grouping.dense_context
    assert [path.name for path in first_captioner.calls] == [
        f"frame{index}.jpg" for index in range(5)
    ]


def test_ids_change_when_prompt_version_changes():
    frames = source_frames(count=1)
    first = FrameLabeler(
        captioner=FakeCaptioner(), grouper=FakeGrouper(), chunk_size=1
    ).label_frames(frames)

    changed = FakeCaptioner()
    changed.prompt_version = "test-frame-v2"
    second = FrameLabeler(captioner=changed, grouper=FakeGrouper(), chunk_size=1).label_frames(
        frames
    )

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


def test_caption_workers_preserve_frame_order():
    frames = source_frames(count=3, gap_after=10)
    result = FrameLabeler(
        captioner=FakeCaptioner(),
        grouper=FakeGrouper(),
        chunk_size=3,
        caption_workers=3,
    ).label_frames(frames)

    assert [frame.caption for frame in result.frames] == [
        f"The wearer observes frame{index}." for index in range(3)
    ]


def test_jsonl_cache_resumes_without_repeating_model_calls(tmp_path):
    frames = source_frames(count=3, gap_after=10)
    cache_path = tmp_path / "labels.jsonl"
    first_captioner = FakeCaptioner()
    first_grouper = FakeGrouper()
    first = FrameLabeler(
        captioner=first_captioner,
        grouper=first_grouper,
        chunk_size=2,
        caption_workers=2,
        cache_path=cache_path,
    ).label_frames(frames)

    second_captioner = FakeCaptioner()
    second_grouper = FakeGrouper()
    second = FrameLabeler(
        captioner=second_captioner,
        grouper=second_grouper,
        chunk_size=2,
        caption_workers=2,
        cache_path=cache_path,
    ).label_frames(frames)

    assert first.to_dict() == second.to_dict()
    assert len(first_captioner.calls) == 3
    assert len(first_grouper.calls) == 2
    assert second_captioner.calls == []
    assert second_grouper.calls == []
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 5


def test_jsonl_cache_repairs_partial_tail_and_normalizes_valid_unterminated_line(tmp_path):
    cache_path = tmp_path / "labels.jsonl"
    cache = JsonlLabelingCache(cache_path)
    cache.put_caption("one", "caption one")
    partial = b'{"record_type":"frame_caption"'
    with cache_path.open("ab") as handle:
        handle.write(partial)

    assert cache.recover() == len(partial)
    assert cache.get_caption("one") == "caption one"
    cache.put_caption("two", "caption two")
    assert cache_path.read_bytes().endswith(b"\n")

    cache_path.write_bytes(cache_path.read_bytes().removesuffix(b"\n"))
    reloaded = JsonlLabelingCache(cache_path)
    assert reloaded.get_caption("two") == "caption two"
    assert cache_path.read_bytes().endswith(b"\n")


def test_jsonl_cache_refreshes_under_file_lock_before_append(tmp_path):
    cache_path = tmp_path / "labels.jsonl"
    first = JsonlLabelingCache(cache_path)
    second = JsonlLabelingCache(cache_path)
    first.put_caption("shared", "same caption")
    second.put_caption("shared", "same caption")

    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ValueError, match="conflicting"):
        second.put_caption("shared", "different caption")


def test_final_partial_chunk_is_provisional_until_explicitly_finalized():
    frames = source_frames(count=3, gap_after=10)
    labeler = FrameLabeler(
        captioner=FakeCaptioner(),
        grouper=FakeGrouper(),
        chunk_size=2,
        caption_workers=1,
    )

    growing = labeler.label_frames(frames)
    finalized = labeler.label_frames(frames, finalize_tail=True)

    assert [chunk.closed for chunk in growing.chunks] == [True, False]
    assert [chunk.closed for chunk in finalized.chunks] == [True, True]
    assert [chunk.chunk_id for chunk in growing.chunks] == [
        chunk.chunk_id for chunk in finalized.chunks
    ]
