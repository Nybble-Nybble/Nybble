"""Infer a taste profile from the pooled evidence.

Two backends:

  hosted — batches excerpts to a Claude model, which labels each item and then
           writes the profile. Requires the separate off-device consent grant
           and an API key.
  local  — a statistical profiler over the same pool. No network, cruder, but
           it always runs and it makes the hosted path optional rather than
           required.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .pool import Candidate, pool_stats

DEFAULT_MODEL = "claude-sonnet-5"
BATCH_SIZE = 40
MAX_BATCHES = 40


@dataclass
class TasteProfile:
    backend: str = "local"
    model: Optional[str] = None
    generated_at: float = field(default_factory=time.time)

    # What the user likes, in decreasing confidence.
    topics: List[Dict[str, Any]] = field(default_factory=list)
    authors: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    styles: List[str] = field(default_factory=list)
    summary: str = ""

    # Things the evidence says the user avoids or bounces off.
    dislikes: List[str] = field(default_factory=list)

    # Per-item verdicts, when the hosted backend ran.
    labeled: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        "utf-8")
        return path


def infer(
    candidates: List[Candidate],
    allow_off_device: bool = False,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_batches: int = MAX_BATCHES,
    progress=None,
) -> TasteProfile:
    """Build a taste profile, preferring the hosted backend when permitted."""
    if not candidates:
        return TasteProfile(warnings=["No affinity evidence found. Collect more "
                                      "sources, or check that reading/social "
                                      "sources were approved."])

    if allow_off_device:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            profile = _infer_hosted(candidates, model, key, max_batches, progress)
            if profile is not None:
                return profile
        else:
            pass  # fall through to local, with a warning attached below

    profile = _infer_local(candidates)
    if allow_off_device and not (api_key or os.environ.get("ANTHROPIC_API_KEY")):
        profile.warnings.append(
            "Off-device inference was approved but ANTHROPIC_API_KEY is not set; "
            "used the local profiler instead.")
    return profile


# ===========================================================================
# Hosted backend
# ===========================================================================

_SYSTEM = """\
You are analyzing one person's saved material to work out what they actually \
like, so a model can be personalized to them.

You will receive numbered items. Each is something this person kept, \
highlighted, bookmarked, liked, followed, rated, or returned to. Each carries a \
prior weight (0-1) reflecting how deliberate that act was: a highlighted passage \
is a stronger signal than a page they happened to load.

The prior is evidence, not an answer. Judge the content itself. People save \
things for many reasons — obligation, work, curiosity about something they \
dislike, a link someone sent them. Your job is to separate genuine affinity \
from incidental retention.

Return ONLY a JSON object, no prose, in this shape:

{
  "items": [
    {"n": 1, "liked": true, "confidence": 0.8, "topics": ["distributed systems"],
     "why": "highlighted a dense technical passage, not a headline"}
  ],
  "topics": [{"topic": "...", "confidence": 0.9, "evidence_count": 12}],
  "authors": [{"name": "...", "confidence": 0.7}],
  "styles": ["dense technical prose with worked examples"],
  "dislikes": ["..."],
  "summary": "2-4 sentences on what this person's taste actually is."
}

Rules:
- "liked" false is a real and useful answer. Do not label everything true.
- Topics should be specific ("post-training RL for code models"), not generic \
("technology").
- "styles" describes the writing and content they gravitate toward, not the \
subject matter.
- Base everything on the items. Do not speculate about the person beyond what \
the material supports.
"""


def _infer_hosted(candidates: List[Candidate], model: str, api_key: str,
                  max_batches: int, progress=None) -> Optional[TasteProfile]:
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None

    profile = TasteProfile(backend="hosted", model=model)
    batches = list(_batched(candidates, BATCH_SIZE))[:max_batches]
    if len(candidates) > max_batches * BATCH_SIZE:
        profile.warnings.append(
            f"Pool of {len(candidates)} truncated to {max_batches * BATCH_SIZE} "
            f"items for inference.")

    topic_votes: Counter = Counter()
    topic_conf: Dict[str, List[float]] = {}
    author_votes: Counter = Counter()
    styles: List[str] = []
    dislikes: List[str] = []
    summaries: List[str] = []
    failures = 0

    for index, batch in enumerate(batches):
        if progress:
            progress(index + 1, len(batches))
        payload = _render_batch(batch)
        result = _call(client, model, payload)
        if result is None:
            failures += 1
            if failures >= 3:
                profile.warnings.append(
                    "Hosted inference failed repeatedly; results are partial.")
                break
            continue

        for item in result.get("items") or []:
            n = item.get("n")
            if not isinstance(n, int) or not (1 <= n <= len(batch)):
                continue
            cand = batch[n - 1]
            profile.labeled.append({
                "id": cand.id,
                "source": cand.source,
                "title": cand.title,
                "url": cand.url,
                "liked": bool(item.get("liked")),
                "confidence": _clamp(item.get("confidence")),
                "topics": [str(t) for t in (item.get("topics") or [])][:6],
                "why": str(item.get("why") or "")[:300],
                "prior_weight": cand.weight,
            })

        for entry in result.get("topics") or []:
            name = str(entry.get("topic") or "").strip()
            if not name:
                continue
            count = int(entry.get("evidence_count") or 1)
            topic_votes[name.lower()] += max(count, 1)
            topic_conf.setdefault(name.lower(), []).append(
                _clamp(entry.get("confidence")))

        for entry in result.get("authors") or []:
            name = str(entry.get("name") or "").strip()
            if name:
                author_votes[name] += 1

        styles.extend(str(s) for s in (result.get("styles") or []))
        dislikes.extend(str(s) for s in (result.get("dislikes") or []))
        if result.get("summary"):
            summaries.append(str(result["summary"]))

    if not profile.labeled and not topic_votes:
        return None  # nothing usable came back; caller falls back to local

    profile.topics = [
        {
            "topic": topic,
            "evidence_count": count,
            "confidence": round(sum(topic_conf.get(topic, [0.5])) /
                                len(topic_conf.get(topic, [0.5])), 2),
        }
        for topic, count in topic_votes.most_common(40)
    ]
    profile.authors = [{"name": name, "evidence_count": count}
                       for name, count in author_votes.most_common(30)]
    profile.styles = _dedupe(styles)[:15]
    profile.dislikes = _dedupe(dislikes)[:15]
    profile.sources = _source_affinity(profile.labeled, candidates)
    profile.summary = _merge_summaries(summaries)
    profile.evidence = pool_stats(candidates)
    profile.evidence["batches_run"] = len(batches) - failures
    profile.evidence["liked_rate"] = _liked_rate(profile.labeled)
    return profile


def _render_batch(batch: List[Candidate]) -> str:
    lines = []
    for i, cand in enumerate(batch, 1):
        header = cand.label()
        signals = ", ".join(
            f"{k}={v}" for k, v in cand.signals.items()
            if v not in (None, False, "", 0))[:160]
        lines.append(
            f"[{i}] source={cand.source} kind={cand.kind} "
            f"prior={cand.weight:.2f}"
            + (f" signals=({signals})" if signals else "")
            + (f"\n    {header}" if header else "")
            + f"\n    {cand.excerpt()}"
        )
    return "\n\n".join(lines)


def _call(client, model: str, payload: str, retries: int = 2) -> Optional[Dict]:
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM,
                messages=[{"role": "user", "content": payload}],
            )
            text = "".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text")
            return _parse_json(text)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
    return None


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def _batched(items: List[Candidate], size: int) -> Iterable[List[Candidate]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 2)
    except (TypeError, ValueError):
        return default


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _liked_rate(labeled: List[Dict[str, Any]]) -> float:
    if not labeled:
        return 0.0
    return round(sum(1 for item in labeled if item.get("liked")) / len(labeled), 3)


def _merge_summaries(summaries: List[str]) -> str:
    """Batch summaries repeat; keep the longest few as the profile summary."""
    if not summaries:
        return ""
    unique = _dedupe(summaries)
    unique.sort(key=len, reverse=True)
    return " ".join(unique[:2])[:1200]


def _source_affinity(labeled: List[Dict[str, Any]],
                     candidates: List[Candidate]) -> List[Dict[str, Any]]:
    """Which domains and publications the liked items came from."""
    by_id = {c.id: c for c in candidates}
    counts: Counter = Counter()
    for item in labeled:
        if not item.get("liked"):
            continue
        cand = by_id.get(item.get("id", ""))
        key = (cand.domain or cand.author or cand.title) if cand else None
        if key:
            counts[str(key)] += 1
    return [{"name": name, "liked_items": count}
            for name, count in counts.most_common(30)]


# ===========================================================================
# Local backend
# ===========================================================================

# Words that dominate any English corpus and say nothing about taste.
_STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn do
does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just ll me more most mustn my myself no nor not now of off on
once only or other otherwise ought our ours ourselves out over own re s same
shan she should shouldn so some such t than that the their theirs them
themselves then there these they this those through to too under until up ve
very was wasn we were weren what when where which while who whom why will with
won would wouldn you your yours yourself yourselves via get got one two also
new like make made way much many well back even still time year years day days
thing things people
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _infer_local(candidates: List[Candidate]) -> TasteProfile:
    """Weighted keyword and entity statistics over the pool.

    Crude next to a model, but honest: it reports what is actually frequent in
    the material the user kept, weighted by how deliberately they kept it.
    """
    profile = TasteProfile(backend="local")

    term_weight: Dict[str, float] = {}
    term_docs: Counter = Counter()
    bigram_weight: Dict[str, float] = {}
    domains: Dict[str, float] = {}
    authors: Dict[str, float] = {}
    declared_topics: Counter = Counter()

    for cand in candidates:
        words = [w.lower() for w in _WORD.findall(cand.text)]
        content = [w for w in words if w not in _STOPWORDS and len(w) > 3]

        for term in set(content):
            term_weight[term] = term_weight.get(term, 0.0) + cand.weight
            term_docs[term] += 1

        for a, b in zip(content, content[1:]):
            phrase = f"{a} {b}"
            bigram_weight[phrase] = bigram_weight.get(phrase, 0.0) + cand.weight

        if cand.domain:
            domains[cand.domain] = domains.get(cand.domain, 0.0) + cand.weight
        if cand.author:
            authors[cand.author] = authors.get(cand.author, 0.0) + cand.weight

        # Platform-declared interest lists are already topic labels.
        if cand.kind == "preference":
            for part in re.split(r"[,;]", cand.text.split(":", 1)[-1]):
                label = part.strip()
                if 2 < len(label) < 60:
                    declared_topics[label] += 1

    # A term appearing in one item is noise; in everything, it is background.
    total = max(len(candidates), 1)
    scored: List[Tuple[str, float]] = []
    for term, weight in term_weight.items():
        docs = term_docs[term]
        if docs < 2:
            continue
        if docs / total > 0.4:
            continue
        scored.append((term, weight * (1.0 + 1.0 / docs ** 0.5)))
    scored.sort(key=lambda kv: -kv[1])

    phrases = sorted(
        ((p, w) for p, w in bigram_weight.items() if w >= 1.5),
        key=lambda kv: -kv[1])[:25]

    # Ordered best-basis-first so that when a term is subsumed by a phrase
    # already accepted ("systems" under "distributed systems"), the specific
    # one is what survives.
    topics: List[Dict[str, Any]] = []
    for label, count in declared_topics.most_common(20):
        topics.append({"topic": label, "evidence_count": count,
                       "confidence": 0.75, "basis": "platform-declared"})
    for phrase, weight in phrases[:20]:
        topics.append({"topic": phrase, "evidence_count": term_docs.get(phrase, 1),
                       "confidence": round(min(weight / 10.0, 0.7), 2),
                       "basis": "frequent phrase"})
    for term, weight in scored[:25]:
        topics.append({"topic": term, "evidence_count": term_docs[term],
                       "confidence": round(min(weight / 12.0, 0.6), 2),
                       "basis": "frequent term"})

    profile.topics = _dedupe_topics(topics)[:50]
    profile.authors = [{"name": name, "weight": round(w, 2)}
                       for name, w in sorted(authors.items(),
                                             key=lambda kv: -kv[1])[:30]]
    profile.sources = [{"name": domain, "weight": round(w, 2)}
                       for domain, w in sorted(domains.items(),
                                               key=lambda kv: -kv[1])[:30]]
    profile.evidence = pool_stats(candidates)
    profile.summary = _local_summary(profile, candidates)
    profile.warnings.append(
        "Local profiler: frequency statistics over kept material, not a "
        "judgement about what you like. Approve off-device inference and set "
        "ANTHROPIC_API_KEY for a real reading of the evidence.")
    return profile


def _dedupe_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse case variants and terms already covered by an accepted phrase.

    "Distributed systems", "distributed systems", "distributed" and "systems"
    are one topic, and the specific phrasing is the useful one.
    """
    kept: List[Dict[str, Any]] = []
    seen_exact: set = set()
    accepted_words: set = set()

    for topic in topics:
        label = str(topic["topic"]).strip()
        key = label.lower()
        if not key or key in seen_exact:
            continue

        words = set(key.split())
        # A single word already contained in an accepted multi-word topic adds
        # nothing; a multi-word topic is specific enough to keep either way.
        if len(words) == 1 and words <= accepted_words:
            continue

        seen_exact.add(key)
        accepted_words |= words
        kept.append(topic)

    return kept


def _local_summary(profile: TasteProfile, candidates: List[Candidate]) -> str:
    top = [t["topic"] for t in profile.topics[:8]]
    domains = [s["name"] for s in profile.sources[:5]]
    parts = [
        f"Pooled {len(candidates)} pieces of kept material across "
        f"{len(profile.evidence.get('by_source', {}))} sources."
    ]
    if top:
        parts.append("Most frequent themes: " + ", ".join(top) + ".")
    if domains:
        parts.append("Most-kept sources: " + ", ".join(domains) + ".")
    return " ".join(parts)
