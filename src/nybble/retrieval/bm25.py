"""In-memory BM25 retrieval with temporal visibility, deduplication, and MMR."""

# Python 3.9 compatibility requires Union and Optional instead of PEP 604 syntax.
# ruff: noqa: UP007, UP045

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from nybble.models import ActionEvent

PathLike = Union[str, os.PathLike]
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
CHECKPOINT_SCHEMA_VERSION = 1


@lru_cache(maxsize=100_000)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


@lru_cache(maxsize=100_000)
def _ngrams(text: str, n: int = 3) -> frozenset:
    tokens = _tokens(text)
    if not tokens:
        return frozenset()
    width = min(n, len(tokens))
    return frozenset(
        tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)
    )


def jaccard_ngrams(left: str, right: str, n: int = 3) -> float:
    """Return token n-gram Jaccard similarity with a short-text fallback."""

    if type(left) is not str or type(right) is not str:
        raise TypeError("left and right must be strings")
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    left_grams = _ngrams(left, n)
    right_grams = _ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    intersection = len(left_grams & right_grams)
    return intersection / max(1, len(left_grams | right_grams))


def mmr_select(
    items: Sequence[tuple[str, float, Any]],
    top_m: int,
    alpha: float = 0.7,
) -> list[tuple[str, float, Any]]:
    """Select relevant but non-redundant items using maximal marginal relevance."""

    if type(top_m) is not int or top_m <= 0:
        raise ValueError("top_m must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if not items:
        return []

    values = list(items)
    for item in values:
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("items must be (text, utility, payload) tuples")
        if type(item[0]) is not str:
            raise TypeError("item text must be a string")
        if isinstance(item[1], bool) or not isinstance(item[1], (int, float)):
            raise TypeError("item utility must be numeric")

    grams = [_ngrams(item[0], 3) for item in values]

    def similarity(left_index: int, right_index: int) -> float:
        left = grams[left_index]
        right = grams[right_index]
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    remaining = sorted(
        range(len(values)),
        key=lambda index: (-float(values[index][1]), index),
    )
    selected = [remaining.pop(0)]

    while remaining and len(selected) < top_m:
        best_index = remaining[0]
        best_value = float("-inf")
        for candidate in remaining:
            redundancy = max(similarity(candidate, chosen) for chosen in selected)
            score = alpha * float(values[candidate][1]) - (1.0 - alpha) * redundancy
            if score > best_value:
                best_index = candidate
                best_value = score
        remaining.remove(best_index)
        selected.append(best_index)

    return [values[index] for index in selected]


class BM25TemporalRetriever:
    """A causal BM25 index whose documents become visible only when complete.

    Global index statistics are rebuilt after every mutation. Queries compute
    statistics again over only the documents visible at ``cutoff_ts`` so future
    documents cannot affect even IDF or length normalization.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        dedup_threshold: float = 0.8,
        replace_on_dedup: bool = True,
    ) -> None:
        self.k1 = self._positive_number(k1, "k1")
        self.b = self._unit_interval(b, "b")
        self.dedup_threshold = self._unit_interval(dedup_threshold, "dedup_threshold")
        if type(replace_on_dedup) is not bool:
            raise TypeError("replace_on_dedup must be a boolean")
        self.replace_on_dedup = replace_on_dedup

        self.docs: list[dict[str, Any]] = []
        self.N = 0
        self.total_len = 0
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_norm: list[float] = []

    @staticmethod
    def _positive_number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return result

    @staticmethod
    def _unit_interval(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
        return result

    @staticmethod
    def _timestamp(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be a finite, non-negative timestamp")
        return result

    @staticmethod
    def _metadata(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("metadata must be a mapping")
        try:
            encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata must contain only JSON-compatible values") from exc
        return json.loads(encoded)

    @staticmethod
    def _doc_identity(text: str, event_ts: float, visible_after_ts: float, namespace: str) -> str:
        payload = json.dumps(
            [text, event_ts, visible_after_ts, namespace],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"doc-{hashlib.sha256(payload).hexdigest()[:24]}"

    def reset(self) -> None:
        self.docs = []
        self._rebuild_index()

    def add(
        self,
        text: str,
        *,
        event_ts: float,
        visible_after_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        namespace: str = "train",
        metadata: Optional[Mapping[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a document and return the ID that remains after deduplication."""

        if type(text) is not str or not text.strip():
            raise ValueError("text must be a non-empty string")
        event_ts = self._timestamp(event_ts, "event_ts")
        if end_ts is not None:
            end_ts = self._timestamp(end_ts, "end_ts")
            if end_ts < event_ts:
                raise ValueError("end_ts must be greater than or equal to event_ts")
        if visible_after_ts is None:
            visible_after_ts = end_ts if end_ts is not None else event_ts
        visible_after_ts = self._timestamp(visible_after_ts, "visible_after_ts")
        if visible_after_ts < event_ts:
            raise ValueError("visible_after_ts cannot precede event_ts")
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        metadata_dict = self._metadata(metadata)
        if doc_id is None:
            doc_id = self._doc_identity(text, event_ts, visible_after_ts, namespace)
        elif type(doc_id) is not str or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")

        tokens = _tokens(text)
        if not tokens:
            raise ValueError("text must contain at least one word token")
        new_doc = {
            "id": doc_id,
            "text": text,
            "event_ts": event_ts,
            "visible_after_ts": visible_after_ts,
            "namespace": namespace,
            "meta": metadata_dict,
            "toks": tokens,
            "tf": Counter(tokens),
            "uniq_toks": frozenset(tokens),
            "tri": _ngrams(text, 3),
            "len": len(tokens),
        }

        for existing in self.docs:
            if existing["id"] == doc_id:
                if self._persistent_doc(existing) == self._persistent_doc(new_doc):
                    return doc_id
                raise ValueError(f"doc_id {doc_id!r} already has different content")

        duplicate_index: Optional[int] = None
        duplicate_similarity = float("-inf")
        for index, existing in enumerate(self.docs):
            if existing["namespace"] != namespace:
                continue
            similarity = self._jaccard_sets(new_doc["tri"], existing["tri"])
            if similarity >= self.dedup_threshold and similarity > duplicate_similarity:
                duplicate_index = index
                duplicate_similarity = similarity

        if duplicate_index is not None:
            existing = self.docs[duplicate_index]
            new_order = (event_ts, visible_after_ts, doc_id)
            existing_order = (
                existing["event_ts"],
                existing["visible_after_ts"],
                existing["id"],
            )
            if self.replace_on_dedup and new_order >= existing_order:
                self.docs[duplicate_index] = new_doc
                self._rebuild_index()
                return doc_id
            return existing["id"]

        self.docs.append(new_doc)
        self._rebuild_index()
        return doc_id

    def add_event(
        self,
        event: ActionEvent,
        *,
        text: Optional[str] = None,
        namespace: str = "train",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Index an action event once its derived label is actually available."""

        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent")
        merged_metadata = event.to_dict()["metadata"]
        merged_metadata.update(
            {
                "event_id": event.id,
                "source": event.source,
                "available_at": event.available_at,
            }
        )
        if metadata is not None:
            merged_metadata.update(self._metadata(metadata))
        return self.add(
            text if text is not None else event.text,
            event_ts=event.start_ts,
            end_ts=event.end_ts,
            visible_after_ts=event.available_at,
            namespace=namespace,
            metadata=merged_metadata,
            doc_id=event.id,
        )

    def query(
        self,
        text: str,
        *,
        k: int,
        cutoff_ts: float,
        namespaces: Optional[Iterable[str]] = None,
        time_decay_lambda: Optional[float] = None,
        mmr_k: Optional[int] = None,
        mmr_alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Return documents visible by the cutoff, optionally reranked by MMR."""

        if type(text) is not str:
            raise TypeError("text must be a string")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        cutoff_ts = self._timestamp(cutoff_ts, "cutoff_ts")
        if mmr_k is not None and (type(mmr_k) is not int or mmr_k <= 0):
            raise ValueError("mmr_k must be a positive integer")
        mmr_alpha = self._unit_interval(mmr_alpha, "mmr_alpha")
        if time_decay_lambda is not None:
            if isinstance(time_decay_lambda, bool) or not isinstance(
                time_decay_lambda, (int, float)
            ):
                raise TypeError("time_decay_lambda must be a number")
            time_decay_lambda = float(time_decay_lambda)
            if not math.isfinite(time_decay_lambda) or time_decay_lambda < 0:
                raise ValueError("time_decay_lambda must be finite and non-negative")

        namespace_set: Optional[set[str]] = None
        if namespaces is not None:
            namespace_set = set(namespaces)
            if not all(type(item) is str and item.strip() for item in namespace_set):
                raise ValueError("namespaces must contain non-empty strings")

        query_tokens = tuple(dict.fromkeys(_tokens(text)))
        if not query_tokens or not self.docs:
            return []

        eligible = [
            index
            for index, doc in enumerate(self.docs)
            if doc["event_ts"] <= cutoff_ts
            and doc["visible_after_ts"] <= cutoff_ts
            and (namespace_set is None or doc["namespace"] in namespace_set)
        ]
        if not eligible:
            return []

        # Query-time statistics prevent not-yet-visible documents from affecting
        # IDF or document-length normalization.
        eligible_df: Counter = Counter()
        total_len = 0
        for index in eligible:
            doc = self.docs[index]
            eligible_df.update(doc["uniq_toks"])
            total_len += doc["len"]
        eligible_n = len(eligible)
        avgdl = total_len / eligible_n

        scores: dict[int, float] = {}
        for index in eligible:
            doc = self.docs[index]
            norm = self.k1 * (1.0 - self.b + self.b * (doc["len"] / avgdl if avgdl else 0.0))
            score = 0.0
            for token in query_tokens:
                term_frequency = doc["tf"].get(token, 0)
                if not term_frequency:
                    continue
                document_frequency = eligible_df[token]
                idf = math.log(
                    1.0 + (eligible_n - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + norm
                score += idf * (term_frequency * (self.k1 + 1.0)) / denominator
            if score <= 0:
                continue
            if time_decay_lambda:
                age_days = max(0.0, cutoff_ts - doc["event_ts"]) / 86_400.0
                score *= math.exp(-time_decay_lambda * age_days)
            scores[index] = score

        ranked = sorted(
            scores,
            key=lambda index: (
                -scores[index],
                -self.docs[index]["event_ts"],
                self.docs[index]["id"],
            ),
        )
        pool = ranked[:k]
        if mmr_k is not None and pool:
            items = [(self.docs[index]["text"], scores[index], index) for index in pool]
            pool = [
                item[2] for item in mmr_select(items, top_m=min(mmr_k, len(items)), alpha=mmr_alpha)
            ]

        return [self._result(index, scores[index]) for index in pool]

    def save_checkpoint(self, checkpoint_path: PathLike) -> Path:
        """Write a compact gzip checkpoint and return its resolved path."""

        path = self._checkpoint_path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config": {
                "k1": self.k1,
                "b": self.b,
                "dedup_threshold": self.dedup_threshold,
                "replace_on_dedup": self.replace_on_dedup,
            },
            "docs": [self._persistent_doc(doc) for doc in self.docs],
        }

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(file_descriptor)
        try:
            with gzip.open(temporary_name, "wt", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path

    def load_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Replace this index with checkpoint contents and rebuild all statistics."""

        path = self._checkpoint_path(checkpoint_path, for_read=True)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported or invalid retriever checkpoint")
        if set(payload) != {"schema_version", "config", "docs"}:
            raise ValueError("retriever checkpoint has unknown or missing fields")
        config = payload["config"]
        if not isinstance(config, dict) or set(config) != {
            "k1",
            "b",
            "dedup_threshold",
            "replace_on_dedup",
        }:
            raise ValueError("retriever checkpoint has invalid config")

        replacement = BM25TemporalRetriever(**config)
        if not isinstance(payload["docs"], list):
            raise ValueError("retriever checkpoint docs must be a list")
        seen_ids = set()
        for raw_doc in payload["docs"]:
            doc = replacement._restore_doc(raw_doc)
            if doc["id"] in seen_ids:
                raise ValueError(f"duplicate checkpoint doc ID {doc['id']!r}")
            seen_ids.add(doc["id"])
            replacement.docs.append(doc)
        replacement._rebuild_index()

        self.k1 = replacement.k1
        self.b = replacement.b
        self.dedup_threshold = replacement.dedup_threshold
        self.replace_on_dedup = replacement.replace_on_dedup
        self.docs = replacement.docs
        self.N = replacement.N
        self.total_len = replacement.total_len
        self.avgdl = replacement.avgdl
        self.df = replacement.df
        self.idf = replacement.idf
        self.postings = replacement.postings
        self.doc_norm = replacement.doc_norm

    @classmethod
    def from_checkpoint(cls, checkpoint_path: PathLike) -> BM25TemporalRetriever:
        retriever = cls()
        retriever.load_checkpoint(checkpoint_path)
        return retriever

    def _rebuild_index(self) -> None:
        self.N = len(self.docs)
        self.total_len = sum(doc["len"] for doc in self.docs)
        self.avgdl = self.total_len / self.N if self.N else 0.0
        self.df = Counter()
        self.postings = defaultdict(list)
        for index, doc in enumerate(self.docs):
            self.df.update(doc["uniq_toks"])
            for token, term_frequency in doc["tf"].items():
                self.postings[token].append((index, term_frequency))
        self.idf = {
            token: math.log(1.0 + (self.N - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in self.df.items()
        }
        self.doc_norm = [
            self.k1 * (1.0 - self.b + self.b * (doc["len"] / self.avgdl if self.avgdl else 0.0))
            for doc in self.docs
        ]

    @staticmethod
    def _jaccard_sets(left: frozenset, right: frozenset) -> float:
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def _persistent_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": doc["id"],
            "text": doc["text"],
            "event_ts": doc["event_ts"],
            "visible_after_ts": doc["visible_after_ts"],
            "namespace": doc["namespace"],
            "meta": doc["meta"],
        }

    def _restore_doc(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != {
            "id",
            "text",
            "event_ts",
            "visible_after_ts",
            "namespace",
            "meta",
        }:
            raise ValueError("invalid retriever checkpoint document")
        text = payload["text"]
        if type(text) is not str or not text.strip() or not _tokens(text):
            raise ValueError("checkpoint document text is invalid")
        doc_id = payload["id"]
        namespace = payload["namespace"]
        if type(doc_id) is not str or not doc_id.strip():
            raise ValueError("checkpoint document ID is invalid")
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("checkpoint document namespace is invalid")
        event_ts = self._timestamp(payload["event_ts"], "event_ts")
        visible_after_ts = self._timestamp(payload["visible_after_ts"], "visible_after_ts")
        if visible_after_ts < event_ts:
            raise ValueError("checkpoint visible_after_ts cannot precede event_ts")
        tokens = _tokens(text)
        return {
            "id": doc_id,
            "text": text,
            "event_ts": event_ts,
            "visible_after_ts": visible_after_ts,
            "namespace": namespace,
            "meta": self._metadata(payload["meta"]),
            "toks": tokens,
            "tf": Counter(tokens),
            "uniq_toks": frozenset(tokens),
            "tri": _ngrams(text, 3),
            "len": len(tokens),
        }

    @staticmethod
    def _checkpoint_path(checkpoint_path: PathLike, for_read: bool = False) -> Path:
        path = Path(checkpoint_path)
        if path.is_dir() or (not for_read and not path.suffix):
            return path / "retriever.json.gz"
        if for_read and not path.exists() and not path.suffix:
            return path / "retriever.json.gz"
        return path

    def _result(self, index: int, score: float) -> dict[str, Any]:
        doc = self.docs[index]
        return {
            "id": doc["id"],
            "text": doc["text"],
            "meta": json.loads(json.dumps(doc["meta"])),
            "score": float(score),
            "event_ts": doc["event_ts"],
            "visible_after_ts": doc["visible_after_ts"],
            "namespace": doc["namespace"],
        }


InMemoryBM25Temporal = BM25TemporalRetriever


__all__ = [
    "BM25TemporalRetriever",
    "InMemoryBM25Temporal",
    "jaccard_ngrams",
    "mmr_select",
]
