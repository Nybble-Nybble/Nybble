"""The catalog of every data source Nybble knows how to import.

This is the authoritative list. `nybble scan` probes it against the current
machine; `nybble consent` renders it for approval; `nybble collect` runs the
approved entries.

Each source maps to one of the six "past behavior" categories from the plan doc,
or to `context` for signal the doc did not enumerate but that is worth having.

Acquisition modes
-----------------
local    Data already sits on this machine. Nothing for the user to do.
export   The user must request a data export from the platform first (GDPR /
         "download your data"). We parse the resulting archive. `export_hint`
         says exactly where to click.
mount    Data lives on an attached device (a plugged-in Kindle, a phone backup).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import platform_paths as pp

ALL_PLATFORMS = ("darwin", "linux", "win32")


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    group: str                       # consent group shown to the user
    doc_category: str                # schema.DOC_CATEGORIES
    authorship: str                  # dominant authorship of produced records
    sensitivity: str                 # low | medium | high
    acquisition: str                 # local | export | mount
    collector: str                   # "module:function" inside nybble.sources
    what: str                        # plain-language: what gets read
    platforms: Sequence[str] = ALL_PLATFORMS
    probe: Optional[Callable[[], List]] = None   # returns candidate paths
    export_hint: str = ""
    needs: Sequence[str] = ()        # optional pip extras this source wants
    # macOS Full Disk Access, or similar OS-level grant, required.
    needs_os_permission: bool = False

    def available_on_platform(self) -> bool:
        return pp.PLATFORM in self.platforms

    def candidates(self) -> List:
        if self.probe is None:
            return []
        try:
            return self.probe() or []
        except Exception:
            return []

    def resolve_collector(self) -> Callable:
        mod_name, _, fn_name = self.collector.partition(":")
        mod = importlib.import_module(f"nybble.sources.{mod_name}")
        return getattr(mod, fn_name)


# Consent groups: how the approval prompt is chunked. Order matters — it is the
# order the user sees.
GROUPS: Dict[str, str] = {
    "writing": "Things you wrote (notes, docs, essays, drafts, code)",
    "messages": "Your conversations (SMS, chat apps, email, DMs)",
    "speech": "Recordings and transcripts of you talking",
    "reading": "What you read, saved, and highlighted",
    "social": "Your social accounts: posts, likes, and declared interests",
    "context": "Surrounding context (calendar, contacts, shell, repos)",
}

GROUP_NOTES: Dict[str, str] = {
    "messages": "Highest-sensitivity group. Includes other people's words, not "
                "just yours. Redaction is on by default here.",
    "speech": "Audio is transcribed locally; the audio itself is never uploaded.",
    "social": "Mostly requires you to download an export from each platform first.",
}


# Sources are added by their own branches. See the iMessage collector.
SOURCES: List[Source] = []


# --- lookups ------------------------------------------------------------------

BY_ID: Dict[str, Source] = {s.id: s for s in SOURCES}


def get(source_id: str) -> Source:
    return BY_ID[source_id]


def for_platform() -> List[Source]:
    return [s for s in SOURCES if s.available_on_platform()]


def by_group() -> Dict[str, List[Source]]:
    out: Dict[str, List[Source]] = {g: [] for g in GROUPS}
    for s in for_platform():
        out.setdefault(s.group, []).append(s)
    return out


def by_doc_category() -> Dict[str, List[Source]]:
    out: Dict[str, List[Source]] = {}
    for s in SOURCES:
        out.setdefault(s.doc_category, []).append(s)
    return out
