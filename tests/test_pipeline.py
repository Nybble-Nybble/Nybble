"""End-to-end: consent → collect → taste, plus the guarantees around them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nybble import consent
from nybble.cli import main
from nybble.consent import ConsentError, Ledger, discover
from nybble.redact import Redactor
from nybble.schema import Record
from nybble.taste.pool import build_pool, score
from nybble.writer import Writer, iter_records


@pytest.fixture
def populated(fake_home, documents, obsidian_vault, kindle_clippings,
              chrome_profile, twitter_archive, calendar_file, whatsapp_export,
              shell_history):
    """A machine with data from every group the plan doc names."""
    return fake_home


@pytest.fixture
def nybble_home(fake_home) -> Path:
    return Path(consent.default_root())


# ===========================================================================
# Consent gates collection
# ===========================================================================

def test_collect_refuses_without_consent(populated, nybble_home):
    with pytest.raises(ConsentError, match="No consent on record"):
        consent.require(nybble_home)


def test_scan_reads_no_file_contents(populated, nybble_home):
    """Discovery must be safe to run before any approval exists."""
    found = discover()
    assert found
    # Probing writes nothing and creates no ledger.
    assert not (nybble_home / "consent.json").exists()


def test_yes_all_approves_everything_found(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    found = discover()
    ledger = consent.run_interactive(found, root=nybble_home, yes_all=True)

    assert len(ledger.sources) == len([a for a in found if a.present])
    assert ledger.blanket is True
    # A blanket yes to local reading is NOT a yes to uploading.
    assert ledger.off_device_llm is False
    assert ledger.redact is True
    assert (nybble_home / "consent.json").exists()


def test_stale_consent_is_refused(populated, nybble_home):
    ledger = Ledger(sources=["local_documents"])
    ledger.granted_at -= (consent.LEDGER_MAX_AGE_DAYS + 1) * 86400
    ledger.save(nybble_home)

    with pytest.raises(ConsentError, match="Re-run"):
        consent.require(nybble_home)


def test_unapproved_source_is_never_collected(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    Ledger(sources=["local_documents"]).save(nybble_home)

    assert main(["--home", str(nybble_home), "collect"]) == 0

    collected = {r["source"] for r in iter_records(nybble_home)}
    assert collected == {"local_documents"}
    # The Kindle and browser data is present on disk but was not approved.
    assert not (nybble_home / "records" / "kindle_clippings.jsonl").exists()


# ===========================================================================
# Full run
# ===========================================================================

def test_full_pipeline(populated, nybble_home, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert main(["--home", str(nybble_home), "consent", "--yes-all"]) == 0
    assert main(["--home", str(nybble_home), "collect"]) == 0

    records = list(iter_records(nybble_home))
    assert len(records) > 10

    sources = {r["source"] for r in records}
    # At least one source from each past-behavior category the fixtures cover.
    assert "local_documents" in sources          # §1 things you wrote
    assert "whatsapp_export" in sources          # §2 messages
    assert "kindle_clippings" in sources         # §4 writing you like
    assert "browser_bookmarks" in sources        # §5 content you like
    assert "twitter_archive" in sources          # §6 platform preferences

    categories = {r["doc_category"] for r in records}
    assert {"written_by_user", "messages", "liked_writing",
            "liked_content", "social_preferences"} <= categories

    manifest = json.loads((nybble_home / "manifest.json").read_text())
    assert manifest["total_records"] == len(records)
    assert manifest["redacted"] is True

    # Taste inference runs on the local backend without a key or network.
    assert main(["--home", str(nybble_home), "taste"]) == 0
    profile = json.loads((nybble_home / "taste_profile.json").read_text())
    assert profile["backend"] == "local"
    assert profile["topics"]


def test_collect_is_idempotent(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(["--home", str(nybble_home), "consent", "--yes-all"])

    main(["--home", str(nybble_home), "collect"])
    first = len(list(iter_records(nybble_home)))

    main(["--home", str(nybble_home), "collect", "--fresh"])
    second = len(list(iter_records(nybble_home)))

    assert first == second > 0


def test_revoke_deletes_everything(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(["--home", str(nybble_home), "consent", "--yes-all"])
    main(["--home", str(nybble_home), "collect"])
    assert (nybble_home / "records").exists()

    assert main(["--home", str(nybble_home), "revoke", "--yes"]) == 0

    assert not (nybble_home / "records").exists()
    assert not (nybble_home / "consent.json").exists()
    assert not (nybble_home / "manifest.json").exists()


# ===========================================================================
# Redaction
# ===========================================================================

def test_redactor_removes_identifiers_and_secrets():
    r = Redactor(enabled=True)
    out = r.scrub(
        "mail me at jane.doe@example.com or call +1 415 555 0134. "
        "key=sk-ant-abcdefghijklmnopqrstuvwx card 4111 1111 1111 1111")

    assert "jane.doe@example.com" not in out
    assert "sk-ant-abcdefghijklmnopqrstuvwx" not in out
    assert "4111 1111 1111 1111" not in out
    assert "[REDACTED_EMAIL]" in out
    assert r.report()["EMAIL"] == 1


def test_secrets_are_scrubbed_even_with_redaction_off():
    r = Redactor(enabled=False)
    out = r.scrub("write to jane@example.com using AWS_SECRET_KEY=hunter2abcdef")

    # Opting out of PII redaction is allowed; leaking credentials is not.
    assert "jane@example.com" in out
    assert "hunter2abcdef" not in out


def test_card_redaction_ignores_non_card_digit_runs():
    r = Redactor(enabled=True)
    out = r.scrub("order 1234567890123456789 shipped")
    # Fails Luhn, so it is an order number, not a card.
    assert "1234567890123456789" in out


def test_writer_applies_redaction_before_writing(tmp_path):
    with Writer(tmp_path, redact=True) as w:
        w.write(Record(source="local_documents", kind="document",
                       authorship="self", doc_category="written_by_user",
                       text="reach me at a@b.com"))
    raw = (tmp_path / "records" / "local_documents.jsonl").read_text()
    assert "a@b.com" not in raw
    assert "[REDACTED_EMAIL]" in raw


def test_writer_dedupes_identical_records(tmp_path):
    def make():
        return Record(source="obsidian", kind="document", authorship="self",
                      doc_category="written_by_user", text="the same note body")

    with Writer(tmp_path, redact=False) as w:
        assert w.write(make()) is True
        assert w.write(make()) is False


# ===========================================================================
# Affinity scoring
# ===========================================================================

def test_explicit_keeps_outrank_passive_exposure():
    highlight = {"kind": "highlight", "authorship": "other",
                 "signals": {"highlighted": True}}
    one_visit = {"kind": "consumption", "authorship": "other",
                 "signals": {"visit_count": 1}}
    many_visits = {"kind": "consumption", "authorship": "other",
                   "signals": {"visit_count": 40}}
    own_essay = {"kind": "document", "authorship": "self", "signals": {}}

    assert score(highlight) > score(many_visits) > score(one_visit)
    # The user's own writing is voice signal, never taste evidence.
    assert score(own_essay) == 0.0


def test_ratings_scale_with_their_value():
    low = {"kind": "reaction", "authorship": "other", "signals": {"rating": 1}}
    high = {"kind": "reaction", "authorship": "other", "signals": {"rating": 5}}
    assert score(high) > score(low)


def test_pool_excludes_self_authored_writing(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(["--home", str(nybble_home), "consent", "--yes-all"])
    main(["--home", str(nybble_home), "collect"])

    pool = build_pool(nybble_home, limit=500)
    assert pool

    sources = {c.source for c in pool}
    assert "kindle_clippings" in sources
    assert "browser_bookmarks" in sources
    # The user's own essays and notes are not evidence of what they like.
    assert "local_documents" not in sources
    assert "obsidian" not in sources

    # The book passage the user highlighted made it into the pool.
    assert any("legibility" in c.text for c in pool)


def test_pool_is_ranked_by_weight(populated, nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(["--home", str(nybble_home), "consent", "--yes-all"])
    main(["--home", str(nybble_home), "collect"])

    pool = build_pool(nybble_home, limit=500)
    weights = [c.weight for c in pool]
    assert weights == sorted(weights, reverse=True)
