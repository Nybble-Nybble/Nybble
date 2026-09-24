"""Normalized record schema.

Everything the importer collects — a text message, a Kindle highlight, an
inferred ad-interest tag, a meeting transcript — lands in one flat record type.
Downstream tuning code should never need to know which collector produced a row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# --- controlled vocabularies -------------------------------------------------

# Who produced the text. Downstream tuning keys off this: "self" is style/voice
# training signal, "other" is what the user chose to keep around, which is
# preference signal. "mixed" is a two-sided artifact (a chat thread, a meeting
# transcript) where authorship varies row to row or cannot be resolved.
AUTHORSHIP = ("self", "other", "mixed", "system")

# Shape of the record, independent of source app.
KINDS = (
    "document",     # essay, note, doc, draft
    "message",      # chat/DM/SMS/email turn
    "transcript",   # speech converted to text
    "post",         # public/social authored post
    "comment",      # reply on someone else's thing
    "highlight",    # excerpt the user marked in someone else's work
    "bookmark",     # saved link/item
    "reaction",     # like/upvote/star/favorite
    "consumption",  # watched/read/listened event
    "preference",   # platform-declared interest or setting
    "event",        # calendar entry
    "contact",      # person record
    "code",         # commit message, code comment
    "activity",     # shell command, app usage
)

# How sensitive the row is. Drives redaction defaults and whether the row is
# allowed to leave the device during taste inference.
SENSITIVITY = ("low", "medium", "high")

# The six past-behavior categories from the Nybble plan doc, plus "context" for
# sources the doc did not list but that carry real signal.
DOC_CATEGORIES = (
    "written_by_user",      # doc §1  Essays / content written by user
    "messages",             # doc §2  Text messages
    "transcripts",          # doc §3  Transcriptions of audio interactions
    "liked_writing",        # doc §4  Writing of others that you like
    "liked_content",        # doc §5  Content in general of others that you like
    "social_preferences",   # doc §6  Preferences saved by social media
    "context",              # not in doc: calendar, contacts, shell, git, ...
)


@dataclass
class Record:
    """One unit of extracted behavior."""

    source: str                              # registry source id, e.g. "imessage"
    kind: str                                # one of KINDS
    authorship: str                          # one of AUTHORSHIP
    text: str                                # the payload
    doc_category: str = "context"            # one of DOC_CATEGORIES
    sensitivity: str = "medium"              # one of SENSITIVITY

    created_at: Optional[str] = None         # ISO-8601, UTC where known
    modified_at: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None             # display name or handle
    recipient: Optional[str] = None
    thread_id: Optional[str] = None          # groups a conversation
    url: Optional[str] = None
    path: Optional[str] = None               # local provenance
    app: Optional[str] = None                # human-facing app name
    lang: Optional[str] = None

    # Affinity evidence: liked=True, upvoted=True, rating=4.5, dwell_ms=91000,
    # play_count=12, highlighted=True, saved=True ...
    signals: Dict[str, Any] = field(default_factory=dict)
    # Anything source-specific worth keeping but not worth a column.
    meta: Dict[str, Any] = field(default_factory=dict)

    id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = self.compute_id()

    def compute_id(self) -> str:
        """Stable content hash so re-running the importer does not duplicate rows."""
        basis = "\x1f".join(
            [
                self.source,
                self.kind,
                self.created_at or "",
                self.thread_id or "",
                self.author or "",
                self.path or "",
                self.text[:2048],
            ]
        )
        return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:32]

    def to_json(self) -> str:
        d = asdict(self)
        # Drop empties to keep JSONL small over millions of rows.
        d = {k: v for k, v in d.items() if v not in (None, "", {}, [])}
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    @property
    def is_affinity_evidence(self) -> bool:
        """True when this row is evidence about what the user likes.

        The plan doc asks for "writing of others that you like" and "content of
        others that you like". No platform reports that directly, so we treat
        any other-authored item the user kept, marked, or replayed as a
        candidate for downstream ranking.
        """
        if self.authorship == "self":
            return False
        if self.kind in ("highlight", "bookmark", "reaction", "preference", "consumption"):
            return True
        return bool(self.signals)


def validate(rec: Record) -> None:
    """Raise ValueError on a malformed record. Used in tests and by the writer."""
    if rec.kind not in KINDS:
        raise ValueError(f"unknown kind: {rec.kind!r}")
    if rec.authorship not in AUTHORSHIP:
        raise ValueError(f"unknown authorship: {rec.authorship!r}")
    if rec.sensitivity not in SENSITIVITY:
        raise ValueError(f"unknown sensitivity: {rec.sensitivity!r}")
    if rec.doc_category not in DOC_CATEGORIES:
        raise ValueError(f"unknown doc_category: {rec.doc_category!r}")
    if not rec.source:
        raise ValueError("record has no source")
