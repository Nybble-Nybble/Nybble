"""Each collector is exercised against real files in the formats it claims."""

from __future__ import annotations

from pathlib import Path

import pytest

from nybble import platform_paths as pp
from nybble.schema import validate
from nybble.sources import Context
from nybble.sources import archives, calendars, filesystem, shell_git, sqlite_apps


@pytest.fixture
def ctx() -> Context:
    return Context(min_chars=20)


def _collect(fn, paths, ctx):
    records = list(fn(paths, ctx))
    for rec in records:
        validate(rec)
    return records


# --- documents ---------------------------------------------------------------

def test_documents_extracts_prose_and_skips_data(documents, ctx):
    records = _collect(filesystem.collect_documents, [documents], ctx)
    titles = {r.title for r in records}

    assert "essay-on-attention" in titles
    assert "notes" in titles
    # A CSV-shaped .txt is not prose and must not become a training record.
    assert "data" not in titles


def test_documents_skip_files_another_collector_owns(documents, kindle_clippings,
                                                     ctx):
    """Book passages must not be attributed to the user as their own writing."""
    records = _collect(filesystem.collect_documents, [documents], ctx)
    assert "My Clippings" not in {r.title for r in records}

    essay = next(r for r in records if r.title == "essay-on-attention")
    assert essay.authorship == "self"
    assert essay.doc_category == "written_by_user"
    # Frontmatter and heading markup are stripped; the prose survives.
    assert "---" not in essay.text
    assert "path dependent" in essay.text


def test_obsidian_captures_tags_and_links(obsidian_vault, ctx):
    records = _collect(filesystem.collect_obsidian, [obsidian_vault], ctx)
    assert len(records) == 1

    note = records[0]
    assert note.app == "Obsidian"
    assert set(note.meta["tags"]) == {"philosophy", "attention"}
    assert set(note.meta["links"]) == {"On Attention", "Consensus"}
    # Wikilink syntax is resolved to its label, not left as brackets.
    assert "[[" not in note.text
    assert "On Attention" in note.text


# --- kindle ------------------------------------------------------------------

def test_kindle_separates_highlights_from_notes(kindle_clippings, ctx):
    records = _collect(filesystem.collect_kindle, [kindle_clippings], ctx)
    assert len(records) == 3

    highlights = [r for r in records if r.kind == "highlight"]
    notes = [r for r in records if r.kind == "comment"]
    assert len(highlights) == 2
    assert len(notes) == 1

    scott = next(r for r in highlights if "legibility" in r.text)
    assert scott.title == "Seeing Like a State"
    assert scott.author == "James C. Scott"
    assert scott.authorship == "other"          # the book's words
    assert scott.signals["highlighted"] is True
    assert scott.created_at.startswith("2024-06-03")

    # The user's own margin note is their writing, not the author's.
    assert notes[0].authorship == "self"


# --- browsers ----------------------------------------------------------------

def test_history_records_visit_counts(chrome_profile, ctx):
    records = _collect(sqlite_apps.collect_history, [chrome_profile], ctx)
    by_title = {r.title: r for r in records}

    assert "Files are hard" in by_title
    assert by_title["Files are hard"].signals["visit_count"] == 14
    assert by_title["Files are hard"].signals["revisited"] is True
    # A single visit is not evidence of affinity.
    assert by_title["Shoe sale"].signals["revisited"] is False
    assert by_title["Files are hard"].meta["domain"] == "danluu.com"


def test_bookmarks_walk_nested_folders(chrome_profile, ctx):
    records = _collect(sqlite_apps.collect_bookmarks, [chrome_profile], ctx)
    titles = {r.title for r in records}

    assert "Notes on distributed consensus" in titles
    # The nested one proves the folder walk recurses.
    assert "The Grug Brained Developer" in titles

    nested = next(r for r in records if r.title == "The Grug Brained Developer")
    assert nested.meta["folder"] == "Reading"
    assert nested.signals["bookmarked"] is True


# --- twitter archive ---------------------------------------------------------

def test_twitter_splits_authorship_and_reads_interests(twitter_archive, ctx):
    records = _collect(archives.collect_twitter, [twitter_archive], ctx)

    posts = [r for r in records if r.kind == "post"]
    likes = [r for r in records if r.kind == "reaction"]
    prefs = [r for r in records if r.kind == "preference"]

    assert len(posts) == 1
    assert posts[0].authorship == "self"
    assert posts[0].signals["favorite_count"] == 12

    # A liked tweet is someone else's writing the user endorsed.
    assert len(likes) == 1
    assert likes[0].authorship == "other"
    assert likes[0].doc_category == "liked_writing"
    assert likes[0].signals["liked"] is True

    # Plan doc §6: the platform's own inferred interest list.
    assert len(prefs) == 1
    assert "Distributed systems" in prefs[0].meta["interests"]
    assert prefs[0].meta["platform_declared"] is True


# --- whatsapp ----------------------------------------------------------------

def test_whatsapp_parses_turns_and_drops_media(whatsapp_export, ctx):
    records = _collect(archives.collect_whatsapp, [whatsapp_export], ctx)

    assert len(records) == 3                      # <Media omitted> is dropped
    assert all(r.thread_id == "Alex" for r in records)
    assert records[0].author == "Alex"
    assert records[0].created_at.startswith("2024-06-03")
    assert "clock skew" in records[1].text


# --- calendar ----------------------------------------------------------------

def test_calendar_unfolds_lines_and_strips_dial_in(calendar_file, ctx):
    records = _collect(calendars.collect_calendars, [calendar_file], ctx)
    assert len(records) == 1

    event = records[0]
    assert event.title == "Design review: storage layer"
    assert event.created_at.startswith("2024-06-03")
    assert set(event.meta["attendees"]) == {"Jane Doe", "Sam Lee"}
    # RFC 5545 line folding must be undone before the text is usable.
    assert "compaction strategy" in event.text
    # Conference boilerplate carries no personal signal.
    assert "zoom.us" not in event.text
    assert "Meeting ID" not in event.text


# --- shell history -----------------------------------------------------------

def test_shell_history_parses_timestamps_and_drops_credentials(shell_history, ctx):
    records = _collect(shell_git.collect_shell_history, [shell_history], ctx)
    commands = [r.text for r in records]

    assert "git status" in commands
    assert "pytest tests/ -k consensus" in commands
    assert records[0].created_at.startswith("2024-06-03")
    # A curl carrying a bearer token is dropped wholesale, not just redacted.
    assert not any("api.example.com" in c for c in commands)


# --- discovery ---------------------------------------------------------------

def test_archive_discovery_finds_downloaded_bundle(twitter_archive, fake_home):
    found = archives.find_archives(["twitter"])
    assert twitter_archive in found


def test_probes_locate_fixtures(chrome_profile, obsidian_vault, kindle_clippings,
                                fake_home):
    assert chrome_profile in pp.chromium_profiles()
    assert kindle_clippings in pp.kindle_clippings()

    from nybble.registry import BY_ID
    assert obsidian_vault in BY_ID["obsidian"].candidates()
