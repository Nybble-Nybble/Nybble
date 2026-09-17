"""Pool and rank candidate affinity evidence.

Every other-authored record is a candidate. What separates them is how
deliberate the act of keeping was: highlighting a passage is a stronger
statement than a browser recording that you loaded a page once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..writer import iter_records

# How much each kind of evidence counts. These are priors handed to the model as
# context, not a verdict — the model still decides what the user actually likes.
SIGNAL_WEIGHTS: Dict[str, float] = {
    # Deliberate, effortful keeps.
    "highlighted": 1.00,
    "bookmarked": 0.95,
    "saved": 0.90,
    "favorited": 0.90,
    "starred": 0.85,
    "liked": 0.85,
    "upvoted": 0.80,
    "subscribed": 0.75,
    "wishlisted": 0.70,
    # Weaker, but still a choice.
    "revisited": 0.55,
    "purchased": 0.50,
    "watched": 0.35,
    "play_counts": 0.45,
}

# Record kinds that are affinity evidence on their own.
KIND_WEIGHTS: Dict[str, float] = {
    "highlight": 0.95,
    "bookmark": 0.85,
    "reaction": 0.80,
    "preference": 0.90,   # the platform stating an inferred interest
    "consumption": 0.30,
    "comment": 0.40,      # the user responding to someone else's thing
    "document": 0.35,     # a saved PDF/paper
    "post": 0.25,
}

MIN_WEIGHT = 0.15


@dataclass
class Candidate:
    """One piece of evidence about what the user likes."""

    id: str
    source: str
    text: str
    weight: float
    kind: str
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    created_at: Optional[str] = None
    signals: Dict[str, Any] = field(default_factory=dict)

    def excerpt(self, limit: int = 600) -> str:
        text = self.text.strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "…"

    def label(self) -> str:
        """Compact human-readable header used in the model prompt."""
        bits = []
        if self.title and self.title.strip() != self.text.strip()[:len(self.title)]:
            bits.append(self.title.strip())
        if self.author:
            bits.append(f"by {self.author}")
        if self.domain:
            bits.append(f"({self.domain})")
        return " ".join(bits)[:200]


def score(record: Dict[str, Any]) -> float:
    """How strongly this record says 'the user likes this'."""
    if record.get("authorship") == "self":
        # A comment the user wrote on someone else's work still points at
        # something they engaged with, but it is weak evidence.
        if record.get("kind") not in ("comment",):
            return 0.0

    weight = KIND_WEIGHTS.get(record.get("kind", ""), 0.0)

    signals = record.get("signals") or {}
    for name, value in signals.items():
        if value in (None, False, 0, ""):
            continue
        bump = SIGNAL_WEIGHTS.get(name)
        if bump:
            weight = max(weight, bump)

    # Ratings are explicit and ordinal: use the value, not just its presence.
    rating = signals.get("rating")
    if isinstance(rating, (int, float)) and rating > 0:
        normalized = rating / 10.0 if rating > 5 else rating / 5.0
        weight = max(weight, 0.4 + 0.6 * min(normalized, 1.0))

    # Revisiting a page many times is a real signal; visiting once is not.
    visits = signals.get("visit_count")
    if isinstance(visits, (int, float)) and visits > 1:
        weight = max(weight, min(0.30 + 0.12 * math.log2(visits), 0.75))

    return round(min(weight, 1.0), 3)


_DOMAIN = re.compile(r"https?://([^/:]+)")


def _domain_of(record: Dict[str, Any]) -> Optional[str]:
    meta = record.get("meta") or {}
    if meta.get("domain"):
        return str(meta["domain"])
    m = _DOMAIN.match(record.get("url") or "")
    return m.group(1).lower().lstrip("www.") if m else None


def build_pool(
    root: Path,
    limit: int = 1200,
    min_weight: float = MIN_WEIGHT,
    per_source_cap: Optional[int] = None,
) -> List[Candidate]:
    """Collect and rank affinity candidates from collected records.

    Stratifies by source so a 40,000-row browser history cannot crowd out 200
    Kindle highlights, which are far better evidence.
    """
    buckets: Dict[str, List[Candidate]] = {}

    for record in iter_records(root):
        weight = score(record)
        if weight < min_weight:
            continue
        text = (record.get("text") or "").strip()
        if len(text) < 8:
            continue
        cand = Candidate(
            id=record.get("id") or "",
            source=record.get("source") or "unknown",
            text=text,
            weight=weight,
            kind=record.get("kind") or "",
            title=record.get("title"),
            author=record.get("author"),
            url=record.get("url"),
            domain=_domain_of(record),
            created_at=record.get("created_at"),
            signals=record.get("signals") or {},
        )
        buckets.setdefault(cand.source, []).append(cand)

    if not buckets:
        return []

    for items in buckets.values():
        items.sort(key=lambda c: (-c.weight, -len(c.text)))

    cap = per_source_cap or max(20, limit // max(len(buckets), 1))
    pooled: List[Candidate] = []
    for source, items in buckets.items():
        pooled.extend(_diversify(items, cap))

    pooled.sort(key=lambda c: -c.weight)
    return pooled[:limit]


def _diversify(items: List[Candidate], cap: int) -> List[Candidate]:
    """Take the strongest items, but avoid ten highlights from one book."""
    if len(items) <= cap:
        return items
    per_group = max(2, cap // 12)
    counts: Dict[str, int] = {}
    out: List[Candidate] = []
    overflow: List[Candidate] = []

    for cand in items:
        key = (cand.title or cand.domain or cand.author or cand.source).lower()
        if counts.get(key, 0) < per_group:
            counts[key] = counts.get(key, 0) + 1
            out.append(cand)
        else:
            overflow.append(cand)
        if len(out) >= cap:
            return out
    # Backfill from the overflow if diversification left us short.
    out.extend(overflow[: cap - len(out)])
    return out


def pool_stats(candidates: Iterable[Candidate]) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    total_weight = 0.0
    count = 0
    for cand in candidates:
        by_source[cand.source] = by_source.get(cand.source, 0) + 1
        by_kind[cand.kind] = by_kind.get(cand.kind, 0) + 1
        total_weight += cand.weight
        count += 1
    return {
        "candidates": count,
        "mean_weight": round(total_weight / count, 3) if count else 0.0,
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
    }


def voice_sample(root: Path, limit: int = 400,
                 min_chars: int = 200) -> List[Dict[str, Any]]:
    """Self-authored writing, for profiling the user's own voice.

    Separate from the taste pool: this is doc §1-3 (how the user writes), not
    §4-6 (what the user likes). It never leaves the machine.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in iter_records(root):
        if record.get("authorship") != "self":
            continue
        text = (record.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        buckets.setdefault(record.get("source", "unknown"), []).append(record)

    if not buckets:
        return []
    cap = max(5, limit // len(buckets))
    out: List[Dict[str, Any]] = []
    for items in buckets.values():
        items.sort(key=lambda r: -len(r.get("text") or ""))
        out.extend(items[:cap])
    return out[:limit]
