"""The taste stage: pooling, the local profiler, and the hosted contract."""

from __future__ import annotations

import json

import pytest

from nybble.taste.classify import (TasteProfile, _dedupe_topics, _parse_json,
                                   infer)
from nybble.taste.pool import Candidate


def _cand(text: str, weight: float = 0.9, **kw) -> Candidate:
    kw.setdefault("id", text[:12])
    kw.setdefault("source", "kindle_clippings")
    kw.setdefault("kind", "highlight")
    return Candidate(text=text, weight=weight, **kw)


# --- topic hygiene ------------------------------------------------------------

def test_dedupe_collapses_case_and_subsumed_terms():
    topics = _dedupe_topics([
        {"topic": "Distributed systems", "basis": "platform-declared"},
        {"topic": "distributed systems", "basis": "frequent phrase"},
        {"topic": "distributed", "basis": "frequent term"},
        {"topic": "systems", "basis": "frequent term"},
        {"topic": "consensus", "basis": "frequent term"},
    ])
    labels = [t["topic"] for t in topics]

    assert labels == ["Distributed systems", "consensus"]


def test_dedupe_keeps_distinct_multiword_topics():
    topics = _dedupe_topics([
        {"topic": "distributed systems"},
        {"topic": "systems programming"},
    ])
    # Overlapping words are fine; these are genuinely different topics.
    assert len(topics) == 2


# --- local backend ------------------------------------------------------------

def test_local_profiler_runs_without_network_or_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    candidates = [
        _cand("Consensus protocols are harder to reason about than their "
              "pseudocode suggests, especially under clock skew."),
        _cand("The failure modes that matter in distributed systems are the "
              "ones the paper does not model."),
        _cand("Clock skew and partial partitions resolve asymmetrically, "
              "which breaks most consensus implementations."),
    ]
    profile = infer(candidates, allow_off_device=False)

    assert profile.backend == "local"
    assert profile.topics
    assert profile.evidence["candidates"] == 3
    # The local backend must say plainly that it is not a real judgement.
    assert any("Local profiler" in w for w in profile.warnings)


def test_empty_pool_returns_actionable_warning():
    profile = infer([], allow_off_device=True)
    assert profile.topics == []
    assert profile.warnings
    assert "No affinity evidence" in profile.warnings[0]


def test_approved_but_keyless_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    profile = infer([_cand("a passage about typography and legibility")],
                    allow_off_device=True)

    assert profile.backend == "local"
    assert any("ANTHROPIC_API_KEY" in w for w in profile.warnings)


def test_local_profiler_weights_platform_declared_topics_highest():
    candidates = [
        Candidate(id="p", source="twitter_archive", kind="preference",
                  weight=0.9,
                  text="Interests X inferred: Distributed systems, Typography"),
        _cand("some passage mentioning kubernetes kubernetes kubernetes"),
    ]
    profile = infer(candidates, allow_off_device=False)
    declared = [t for t in profile.topics if t.get("basis") == "platform-declared"]

    assert {t["topic"] for t in declared} >= {"Distributed systems", "Typography"}
    assert profile.topics[0]["basis"] == "platform-declared"


# --- hosted backend contract ---------------------------------------------------

def test_model_response_parsing_tolerates_markdown_fences():
    payload = _parse_json(
        '```json\n{"items": [{"n": 1, "liked": true}], "summary": "ok"}\n```')
    assert payload["items"][0]["liked"] is True
    assert payload["summary"] == "ok"


def test_model_response_parsing_tolerates_surrounding_prose():
    payload = _parse_json(
        'Here is the analysis:\n{"topics": [{"topic": "x"}]}\nHope that helps.')
    assert payload["topics"][0]["topic"] == "x"


def test_model_response_parsing_returns_none_on_garbage():
    assert _parse_json("not json at all") is None
    assert _parse_json("") is None


def test_hosted_backend_is_skipped_without_consent(monkeypatch):
    """A key in the environment must not override a declined grant."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-used")

    def explode(*args, **kwargs):
        raise AssertionError("hosted inference ran without consent")

    monkeypatch.setattr("nybble.taste.classify._infer_hosted", explode)

    profile = infer([_cand("a highlighted passage")], allow_off_device=False)
    assert profile.backend == "local"


def test_profile_round_trips_to_disk(tmp_path):
    profile = TasteProfile(backend="local", topics=[{"topic": "x"}],
                           summary="a summary")
    path = profile.save(tmp_path / "taste_profile.json")
    loaded = json.loads(path.read_text())

    assert loaded["backend"] == "local"
    assert loaded["summary"] == "a summary"
