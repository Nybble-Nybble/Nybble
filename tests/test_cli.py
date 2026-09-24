"""The CLI and the consent gate, with no sources registered."""

from __future__ import annotations

import json

import pytest

from nybble import consent
from nybble.cli import main
from nybble.consent import ConsentError, Ledger
from nybble.redact import Redactor
from nybble.schema import Record
from nybble.writer import Writer


def test_no_command_prints_help_and_fails():
    assert main([]) == 1


def test_scan_finds_nothing_with_empty_registry(nybble_home, capsys):
    assert main(["scan"]) == 0
    assert "No supported sources found" in capsys.readouterr().out
    # Discovery creates no ledger.
    assert not (nybble_home / "consent.json").exists()


def test_scan_json_is_valid(nybble_home, capsys):
    assert main(["scan", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_sources_lists_catalog(capsys):
    assert main(["sources", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_consent_with_nothing_found_saves_empty_ledger(nybble_home, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["consent", "--yes-all"]) == 0
    ledger = Ledger.load(nybble_home)
    assert ledger is not None
    assert ledger.sources == []
    assert ledger.redact is True


def test_collect_refuses_without_consent(nybble_home):
    with pytest.raises(ConsentError, match="No consent on record"):
        consent.require(nybble_home)
    assert main(["collect"]) == 2


def test_collect_refuses_empty_ledger(nybble_home):
    Ledger().save(nybble_home)
    with pytest.raises(ConsentError, match="approves no sources"):
        consent.require(nybble_home)


def test_stale_consent_is_refused(nybble_home):
    ledger = Ledger(sources=["imessage"])
    ledger.granted_at -= (consent.LEDGER_MAX_AGE_DAYS + 1) * 86400
    ledger.save(nybble_home)
    with pytest.raises(ConsentError, match="Re-run"):
        consent.require(nybble_home)


def test_status_before_consent(nybble_home, capsys):
    assert main(["status"]) == 0
    assert "No consent on record" in capsys.readouterr().out


def test_status_and_revoke_round_trip(nybble_home, capsys):
    Ledger(sources=["imessage"]).save(nybble_home)
    with Writer(nybble_home, redact=True) as w:
        w.write_all("imessage", iter([
            Record(source="imessage", kind="message", authorship="self",
                   doc_category="messages", text="x" * 300)]))
        w.finalize("darwin")

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Approved sources: 1" in out
    assert "imessage" in out

    assert main(["revoke", "--yes"]) == 0
    assert not (nybble_home / "records").exists()
    assert not (nybble_home / "consent.json").exists()
    assert not (nybble_home / "manifest.json").exists()


def test_redactor_removes_identifiers_and_secrets():
    r = Redactor(enabled=True)
    out = r.scrub(
        "mail me at jane.doe@example.com or call +1 415 555 0134. "
        "key=sk-ant-abcdefghijklmnopqrstuvwx card 4111 1111 1111 1111")
    assert "jane.doe@example.com" not in out
    assert "sk-ant-abcdefghijklmnopqrstuvwx" not in out
    assert "4111 1111 1111 1111" not in out
    assert "[REDACTED_EMAIL]" in out


def test_secrets_are_scrubbed_even_with_redaction_off():
    r = Redactor(enabled=False)
    out = r.scrub("write to jane@example.com using AWS_SECRET_KEY=hunter2abcdef")
    assert "jane@example.com" in out
    assert "hunter2abcdef" not in out


def test_writer_applies_redaction_and_dedupes(tmp_path):
    def make():
        return Record(source="imessage", kind="message", authorship="self",
                      doc_category="messages", text="reach me at a@b.com")

    with Writer(tmp_path, redact=True) as w:
        assert w.write(make()) is True
        assert w.write(make()) is False
    raw = (tmp_path / "records" / "imessage.jsonl").read_text()
    assert "a@b.com" not in raw
    assert "[REDACTED_EMAIL]" in raw
