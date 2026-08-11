import math

import pytest

from nybble.models import ActionEvent
from nybble.retrieval import BM25TemporalRetriever, jaccard_ngrams, mmr_select


def test_event_is_invisible_until_its_label_is_available():
    event = ActionEvent(
        id="event-1",
        start_ts=10,
        end_ts=20,
        available_at=30,
        source="test",
        text="open project settings",
    )
    retriever = BM25TemporalRetriever()
    retriever.add_event(event)

    assert retriever.query("settings", k=5, cutoff_ts=20) == []
    assert retriever.query("settings", k=5, cutoff_ts=29.999) == []
    result = retriever.query("settings", k=5, cutoff_ts=30)
    assert [item["id"] for item in result] == ["event-1"]
    assert result[0]["visible_after_ts"] == 30


def test_invisible_future_docs_do_not_change_query_statistics():
    retriever = BM25TemporalRetriever(dedup_threshold=1.0)
    retriever.add("open project settings", event_ts=1, end_ts=2, doc_id="past")
    before = retriever.query("open settings", k=5, cutoff_ts=5)

    retriever.add(
        "open another settings panel",
        event_ts=100,
        end_ts=101,
        doc_id="future",
    )
    after = retriever.query("open settings", k=5, cutoff_ts=5)

    assert after == before


def test_global_idf_and_all_norms_are_rebuilt_after_each_add():
    retriever = BM25TemporalRetriever(dedup_threshold=1.0)
    retriever.add("alpha", event_ts=1, doc_id="short")
    old_norm = retriever.doc_norm[0]
    retriever.add("alpha beta gamma delta", event_ts=2, doc_id="long")

    assert retriever.N == 2
    assert retriever.avgdl == 2.5
    assert retriever.doc_norm[0] != old_norm
    expected_idf = math.log(1.0 + (2 - 2 + 0.5) / (2 + 0.5))
    assert retriever.idf["alpha"] == pytest.approx(expected_idf)


def test_dedup_replaces_with_newer_document_but_short_texts_stay_distinct():
    retriever = BM25TemporalRetriever(dedup_threshold=0.8)
    assert retriever.add("open project settings now", event_ts=1, doc_id="old") == "old"
    assert retriever.add("open project settings now", event_ts=2, doc_id="new") == "new"
    assert retriever.N == 1
    assert retriever.docs[0]["id"] == "new"

    retriever.add("open terminal", event_ts=3, doc_id="terminal")
    retriever.add("close browser", event_ts=4, doc_id="browser")
    assert retriever.N == 3
    assert jaccard_ngrams("open terminal", "close browser") == 0.0


def test_namespace_filter_and_time_decay():
    retriever = BM25TemporalRetriever(dedup_threshold=1.0)
    retriever.add("run project tests", event_ts=1, namespace="train", doc_id="old-train")
    retriever.add("run project tests", event_ts=86_401, namespace="eval", doc_id="new-eval")

    train = retriever.query(
        "project tests",
        k=5,
        cutoff_ts=86_402,
        namespaces=("train",),
        time_decay_lambda=1.0,
    )
    assert [item["id"] for item in train] == ["old-train"]
    all_results = retriever.query("project tests", k=5, cutoff_ts=86_402, time_decay_lambda=1.0)
    assert all_results[0]["id"] == "new-eval"


def test_mmr_prefers_a_diverse_second_result():
    items = [
        ("open project settings panel", 1.0, "first"),
        ("open project settings panel", 0.99, "duplicate"),
        ("run terminal test suite", 0.9, "diverse"),
    ]
    selected = mmr_select(items, top_m=2, alpha=0.5)
    assert [item[2] for item in selected] == ["first", "diverse"]


def test_checkpoint_roundtrip_rebuilds_index_and_results(tmp_path):
    retriever = BM25TemporalRetriever(k1=1.2, b=0.6, dedup_threshold=1.0)
    retriever.add(
        "open project editor",
        event_ts=1,
        end_ts=2,
        namespace="train",
        metadata={"reward": 0.8},
        doc_id="one",
    )
    retriever.add(
        "run project tests",
        event_ts=3,
        end_ts=4,
        namespace="train",
        metadata={"reward": 1.0},
        doc_id="two",
    )
    expected = retriever.query("project", k=5, cutoff_ts=10)

    checkpoint = retriever.save_checkpoint(tmp_path / "checkpoint")
    assert checkpoint.name == "retriever.json.gz"
    restored = BM25TemporalRetriever.from_checkpoint(tmp_path / "checkpoint")

    assert restored.query("project", k=5, cutoff_ts=10) == expected
    assert restored.idf == retriever.idf
    assert restored.doc_norm == retriever.doc_norm


def test_query_integrates_mmr_and_reset():
    retriever = BM25TemporalRetriever(dedup_threshold=1.0)
    retriever.add("open project settings alpha", event_ts=1, doc_id="one")
    retriever.add("open project settings beta", event_ts=2, doc_id="two")
    retriever.add("open project terminal tests", event_ts=3, doc_id="three")

    results = retriever.query("open project", k=3, cutoff_ts=10, mmr_k=2, mmr_alpha=0.3)
    assert len(results) == 2
    assert len({item["id"] for item in results}) == 2

    retriever.reset()
    assert retriever.N == 0
    assert retriever.idf == {}
    assert retriever.doc_norm == []
