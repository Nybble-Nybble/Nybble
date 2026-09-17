"""Builds a synthetic home directory that exercises the real collectors.

Nothing here is mocked: the fixtures write actual SQLite databases, actual zip
archives, and actual export-format files, so the tests fail if a parser drifts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    (home / "Downloads").mkdir()
    (home / "Desktop").mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("NYBBLE_HOME", str(tmp_path / "nybble"))
    return home


@pytest.fixture
def documents(fake_home) -> Path:
    docs = fake_home / "Documents"
    (docs / "essay-on-attention.md").write_text(
        "---\ntitle: On Attention\n---\n\n"
        "# On Attention\n\n"
        "The thing nobody tells you about attention is that it is not a "
        "resource you spend, it is a shape your day takes. I have been "
        "keeping notes on this for two years and the pattern is consistent: "
        "the days I remember are the days I held one question long enough "
        "for it to change.\n\n"
        "Every productivity system I have tried optimizes the wrong variable. "
        "They all assume attention is fungible across tasks when in practice "
        "it is deeply path dependent.\n",
        encoding="utf-8")

    (docs / "notes.txt").write_text(
        "A short note about distributed systems and why consensus protocols "
        "are harder to reason about than their pseudocode suggests. The "
        "failure modes that matter are the ones the paper does not model, "
        "particularly around clock skew and partial network partitions that "
        "resolve asymmetrically.",
        encoding="utf-8")

    # Should be filtered out by looks_like_prose.
    (docs / "data.txt").write_text("1,2,3\n4,5,6\n7,8,9\n" * 20, encoding="utf-8")
    return docs


@pytest.fixture
def obsidian_vault(fake_home) -> Path:
    vault = fake_home / "Documents" / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "reading-notes.md").write_text(
        "# Reading notes\n\n"
        "Tags: #philosophy #attention\n\n"
        "Linked to [[On Attention]] and [[Consensus]].\n\n"
        "The argument in chapter four is that institutions decay not through "
        "corruption but through the slow substitution of proxy metrics for the "
        "thing the metric was meant to track. This is the most useful frame I "
        "have picked up this year.\n",
        encoding="utf-8")
    return vault


@pytest.fixture
def kindle_clippings(fake_home) -> Path:
    path = fake_home / "Documents" / "My Clippings.txt"
    path.write_text(
        "Seeing Like a State (James C. Scott)\n"
        "- Your Highlight on page 88 | Location 1341-1343 | "
        "Added on Monday, 3 June 2024 09:12:33\n"
        "\n"
        "The legibility of a society provides the capacity for large-scale "
        "social engineering, high modernist ideology provides the desire.\n"
        "==========\n"
        "The Design of Everyday Things (Don Norman)\n"
        "- Your Highlight on page 12 | Location 201-203 | "
        "Added on Tuesday, 4 June 2024 11:02:00\n"
        "\n"
        "When a device as simple as a door has to come with an instruction "
        "manual, even if it is only a one word manual, then it is a failure, "
        "poorly designed.\n"
        "==========\n"
        "Seeing Like a State (James C. Scott)\n"
        "- Your Note on page 88 | Location 1341 | "
        "Added on Monday, 3 June 2024 09:14:00\n"
        "\n"
        "This is the same argument as the metrics point in my reading notes.\n"
        "==========\n",
        encoding="utf-8")
    return path


@pytest.fixture
def chrome_profile(fake_home) -> Path:
    profile = fake_home / ".config" / "google-chrome" / "Default"
    profile.mkdir(parents=True)

    db = sqlite3.connect(profile / "History")
    db.execute("""CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT,
                  title TEXT, visit_count INTEGER, typed_count INTEGER,
                  last_visit_time INTEGER, hidden INTEGER)""")
    rows = [
        (1, "https://danluu.com/deconstruct-files/",
         "Files are hard", 14, 2, 13350000000000000, 0),
        (2, "https://www.example.com/sale", "Shoe sale", 1, 0,
         13350000000000000, 0),
        (3, "https://arxiv.org/abs/1706.03762",
         "Attention Is All You Need", 9, 1, 13350000000000000, 0),
    ]
    db.executemany("INSERT INTO urls VALUES (?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()

    bookmarks = {
        "roots": {
            "bookmark_bar": {
                "type": "folder", "name": "Bookmarks bar",
                "children": [
                    {"type": "url", "name": "Notes on distributed consensus",
                     "url": "https://aphyr.com/posts/288-jepsen",
                     "date_added": "13350000000000000"},
                    {"type": "folder", "name": "Reading",
                     "children": [
                         {"type": "url", "name": "The Grug Brained Developer",
                          "url": "https://grugbrain.dev/",
                          "date_added": "13350000000000000"},
                     ]},
                ],
            }
        }
    }
    (profile / "Bookmarks").write_text(json.dumps(bookmarks), encoding="utf-8")
    return profile


@pytest.fixture
def twitter_archive(fake_home) -> Path:
    path = fake_home / "Downloads" / "twitter-2024-06-01-archive.zip"
    tweets = [{"tweet": {
        "id_str": "1",
        "full_text": "The best writing about systems is writing that admits "
                     "where the model breaks down.",
        "created_at": "Wed Jun 05 12:00:00 +0000 2024",
        "favorite_count": "12", "retweet_count": "3",
    }}]
    likes = [{"like": {
        "tweetId": "999",
        "fullText": "A long thread on why most benchmarks measure the wrong "
                    "thing, with a careful worked example.",
        "expandedUrl": "https://x.com/i/status/999",
    }}]
    personalization = {"p13nData": {"interests": [
        {"name": "Distributed systems"}, {"name": "Cognitive science"},
    ]}}

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/tweets.js",
                   "window.YTD.tweets.part0 = " + json.dumps(tweets))
        z.writestr("data/like.js",
                   "window.YTD.like.part0 = " + json.dumps(likes))
        z.writestr("data/personalization.js",
                   "window.YTD.personalization.part0 = " +
                   json.dumps([personalization]))
    return path


@pytest.fixture
def calendar_file(fake_home) -> Path:
    path = fake_home / "Documents" / "work.ics"
    path.write_text(
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:abc-123\r\n"
        "SUMMARY:Design review: storage layer\r\n"
        "DTSTART:20240603T160000Z\r\n"
        "DTEND:20240603T170000Z\r\n"
        "DESCRIPTION:Walk through the write path and the compaction\r\n"
        "  strategy. Zoom: https://zoom.us/j/123456\\nMeeting ID: 123 456\r\n"
        "ATTENDEE;CN=Jane Doe:mailto:jane@example.com\r\n"
        "ATTENDEE;CN=Sam Lee:mailto:sam@example.com\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n",
        encoding="utf-8")
    return path


@pytest.fixture
def whatsapp_export(fake_home) -> Path:
    path = fake_home / "Downloads" / "WhatsApp Chat with Alex.txt"
    path.write_text(
        "[2024-06-03, 09:12:33] Alex: did you read the jepsen post\n"
        "[2024-06-03, 09:13:02] Me: yeah, the part about clock skew is the "
        "only honest treatment of it I have seen\n"
        "[2024-06-03, 09:13:40] Me: everyone else hand waves it\n"
        "[2024-06-03, 09:14:10] Alex: <Media omitted>\n",
        encoding="utf-8")
    return path


@pytest.fixture
def shell_history(fake_home) -> Path:
    path = fake_home / ".zsh_history"
    path.write_text(
        ": 1717401153:0;git status\n"
        ": 1717401160:0;pytest tests/ -k consensus\n"
        ": 1717401200:0;curl -H 'Authorization: Bearer sk-abc123456789012345678901' https://api.example.com\n"
        ": 1717401260:0;rg --type py 'def collect'\n",
        encoding="utf-8")
    return path
