import json

import pytest

from nybble.data.journal import (
    EventJournal,
    JournalConflictError,
    JournalCorruptionError,
)
from nybble.models import ActionEvent


def event(event_id, timestamp, text=None, *, source="test", session_id=None):
    return ActionEvent(
        id=event_id,
        start_ts=timestamp,
        end_ts=timestamp + 1,
        available_at=timestamp + 1,
        source=source,
        text=text or f"action {event_id}",
        provenance={"session_id": session_id} if session_id else {},
    )


def test_append_is_idempotent_and_read_order_is_deterministic(tmp_path):
    journal = EventJournal(tmp_path / "events.jsonl")
    same_time_b = event("b", 10)
    later = event("z", 20)
    same_time_a = event("a", 10)

    assert journal.append_many((later, same_time_b, same_time_a)) == 3
    assert journal.append(same_time_a) is False
    assert [item.id for item in journal.read()] == ["a", "b", "z"]
    assert len(journal) == 3
    assert journal.get("b") == same_time_b

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["record_type"] == "action_event" for line in lines)


def test_conflicting_event_id_is_rejected(tmp_path):
    journal = EventJournal(tmp_path / "events.jsonl")
    journal.append(event("same", 1, "first"))

    with pytest.raises(JournalConflictError, match="different content"):
        journal.append(event("same", 1, "second"))
    assert journal.read()[0].text == "first"


def test_partial_final_line_is_removed_and_following_append_is_safe(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    first = event("first", 1)
    journal.append(first)
    complete_size = path.stat().st_size

    with path.open("ab") as handle:
        handle.write(b'{"record_type":"action_event","schema_version":1,"id":"bro')

    assert journal.read() == (first,)
    assert path.stat().st_size == complete_size
    assert journal.append(event("second", 2)) is True
    assert [item.id for item in journal.read()] == ["first", "second"]


def test_recover_reports_removed_byte_count(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    journal.append(event("first", 1))
    suffix = b'{"broken":'
    with path.open("ab") as handle:
        handle.write(suffix)

    assert journal.recover() == len(suffix)
    assert journal.recover() == 0


def test_valid_unterminated_record_is_preserved_and_normalized(tmp_path):
    path = tmp_path / "events.jsonl"
    original = event("first", 1)
    path.write_text(json.dumps(original.to_dict()))

    journal = EventJournal(path)
    assert journal.read() == (original,)
    assert path.read_bytes().endswith(b"\n")
    journal.append(event("second", 2))
    assert len(path.read_text().splitlines()) == 2


def test_malformed_complete_line_is_hard_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("{not json}\n")

    with pytest.raises(JournalCorruptionError, match="line 1"):
        EventJournal(path).read()


def test_missing_journal_reads_as_empty(tmp_path):
    journal = EventJournal(tmp_path / "missing" / "events.jsonl")
    assert journal.read() == ()
    assert journal.recover() == 0


def test_source_session_replacement_is_atomic_scoped_and_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    stable = event("stable", 1, session_id="session-a")
    old_tail = event("old-tail", 2, session_id="session-a")
    unrelated = event(
        "unrelated",
        20,
        source="other-source",
        session_id="session-b",
    )
    journal.append_many((stable, old_tail, unrelated))
    replacement = event("new-tail", 3, session_id="session-a")

    assert journal.replace_source_sessions(
        (stable, replacement),
        source="test",
        session_ids={"session-a"},
    ) == (2, 2)
    assert [item.id for item in journal.read()] == ["stable", "new-tail", "unrelated"]
    assert journal.replace_source_sessions(
        (stable, replacement),
        source="test",
        session_ids={"session-a"},
    ) == (0, 0)
    assert all(
        json.loads(line)["record_type"] == "action_event" for line in path.read_text().splitlines()
    )


def test_source_session_replacement_rejects_foreign_events(tmp_path):
    journal = EventJournal(tmp_path / "events.jsonl")
    foreign = event("foreign", 1, source="other", session_id="session-a")

    with pytest.raises(ValueError, match="requested source sessions"):
        journal.replace_source_sessions(
            (foreign,),
            source="test",
            session_ids={"session-a"},
        )
