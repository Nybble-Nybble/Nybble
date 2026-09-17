"""Calendar (.ics) and contacts (.vcf) collectors.

Calendar establishes rhythm and recurring collaborators; contacts exist only to
turn phone numbers in message archives into names, so the model learns that the
user writes differently to their manager than to their sibling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import parse_date, relpath, walk_files

# RFC 5545 folds long lines by starting the continuation with a space or tab.
_UNFOLD = re.compile(r"\r?\n[ \t]")
_PROP = re.compile(r"^(?P<name>[A-Za-z-]+)(?P<params>;[^:]*)?:(?P<value>.*)$")


def _unescape(value: str) -> str:
    return (value.replace("\\n", "\n").replace("\\N", "\n")
                 .replace("\\,", ",").replace("\\;", ";")
                 .replace("\\\\", "\\"))


# Properties whose parameters carry the information we actually want: the
# display name in ATTENDEE;CN=Jane Doe:mailto:jane@example.com lives in the
# parameter, not the value.
_PARAMS_MATTER = {"ATTENDEE", "ORGANIZER"}


def _parse_components(raw: str, component: str) -> Iterator[Dict[str, List[str]]]:
    """Yield property dicts for each BEGIN:<component> block."""
    text = _UNFOLD.sub("", raw)
    current: Optional[Dict[str, List[str]]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == f"BEGIN:{component}":
            current = {}
            continue
        if line.upper() == f"END:{component}":
            if current:
                yield current
            current = None
            continue
        if current is None:
            continue
        m = _PROP.match(line)
        if not m:
            continue
        name = m.group("name").upper()
        value = _unescape(m.group("value"))
        if name in _PARAMS_MATTER and m.group("params"):
            value = m.group("params") + ":" + value
        current.setdefault(name, []).append(value)


def _first(props: Dict[str, List[str]], key: str) -> Optional[str]:
    values = props.get(key)
    return values[0].strip() if values and values[0].strip() else None


def collect_calendars(paths: List[Path], ctx: Context) -> Iterator[Record]:
    seen = 0
    for path in walk_files(paths, suffixes={".ics", ".ical", ".icbu"},
                           max_files=20_000, max_depth=12):
        if seen >= ctx.max_records_per_source:
            return
        raw = tx.read_text(path)
        if not raw or "BEGIN:VEVENT" not in raw:
            continue
        for props in _parse_components(raw, "VEVENT"):
            if seen >= ctx.max_records_per_source:
                return
            summary = _first(props, "SUMMARY")
            if not summary:
                continue
            start = parse_date(_first(props, "DTSTART"))
            if ctx.since and (start or "") < ctx.since:
                continue

            description = _first(props, "DESCRIPTION") or ""
            location = _first(props, "LOCATION")
            attendees = [_clean_attendee(a) for a in props.get("ATTENDEE", [])]
            attendees = [a for a in attendees if a][:40]
            organizer = _clean_attendee(_first(props, "ORGANIZER") or "")

            body = summary
            if description:
                # Meeting descriptions are mostly dial-in boilerplate.
                cleaned = _strip_conference_boilerplate(description)
                if cleaned:
                    body = f"{summary}\n\n{cleaned}"

            seen += 1
            yield Record(
                source="calendar",
                kind="event",
                authorship="mixed",
                doc_category="context",
                sensitivity="high",
                text=tx.clean(body),
                title=summary,
                author=organizer or None,
                created_at=start,
                modified_at=parse_date(_first(props, "LAST-MODIFIED")),
                app="Calendar",
                path=relpath(path),
                thread_id=_first(props, "UID"),
                meta={
                    "end": parse_date(_first(props, "DTEND")),
                    "location": location,
                    "attendees": attendees,
                    "attendee_count": len(attendees),
                    "recurring": bool(props.get("RRULE")),
                    "status": _first(props, "STATUS"),
                },
            )


# A conferencing link is noise wherever it sits, including mid-sentence after
# real meeting notes, so it is removed inline rather than by the line.
_CONF_URL = re.compile(
    r"\s*(?:\b\w+:\s*)?https?://\S*(?:zoom\.us|meet\.google|teams\.microsoft"
    r"|webex\.com|whereby\.com|meet\.jit\.si|bluejeans)\S*", re.I)

# Lines that are nothing but dial-in details. Anchored whole-line so a line
# that also carries agenda text survives.
_CONF_LINE = re.compile(
    r"(?im)^\s*(?:Meeting ID|Passcode|Password|One tap mobile|Dial[- ]?in|"
    r"Find your local number|Join (?:Zoom|Microsoft Teams|the) meeting|"
    r"\+?\d[\d\s().-]{8,})\s*:?.*$")


def _strip_conference_boilerplate(text: str) -> str:
    out = _CONF_URL.sub("", text)
    out = _CONF_LINE.sub("", out)
    out = re.sub(r"[-_]{5,}", "", out)
    return tx.clean(out)


def _clean_attendee(value: str) -> Optional[str]:
    """'mailto:a@b.com' or 'CN=Jane Doe:mailto:...' → a readable identifier."""
    if not value:
        return None
    m = re.search(r"CN=([^;:]+)", value)
    if m:
        return m.group(1).strip().strip('"')
    return value.replace("mailto:", "").strip() or None


# ===========================================================================
# Contacts
# ===========================================================================

def collect_contacts(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Names, emails, and phone numbers — identity mapping only, no notes."""
    seen = 0
    for path in walk_files(paths, suffixes={".vcf", ".vcard", ".abbu"},
                           max_files=10_000, max_depth=12):
        if seen >= ctx.max_records_per_source:
            return
        raw = tx.read_text(path)
        if not raw or "BEGIN:VCARD" not in raw:
            continue
        for props in _parse_components(raw, "VCARD"):
            if seen >= ctx.max_records_per_source:
                return
            name = _first(props, "FN") or _structured_name(props)
            if not name:
                continue
            emails = [e.strip() for e in props.get("EMAIL", []) if e.strip()]
            phones = [_normalize_phone(p) for p in props.get("TEL", [])]
            phones = [p for p in phones if p]
            org = _first(props, "ORG")

            seen += 1
            yield Record(
                source="contacts",
                kind="contact",
                authorship="system",
                doc_category="context",
                sensitivity="high",
                # The text is the name alone; identifiers stay in meta so the
                # redactor does not blank out the record entirely.
                text=name,
                title=name,
                app="Contacts",
                path=relpath(path),
                meta={
                    "emails": emails[:10],
                    "phones": phones[:10],
                    "organization": org.replace(";", " ").strip() if org else None,
                },
            )


def _structured_name(props: Dict[str, List[str]]) -> Optional[str]:
    n = _first(props, "N")
    if not n:
        return None
    parts = [p.strip() for p in n.split(";")]
    family = parts[0] if parts else ""
    given = parts[1] if len(parts) > 1 else ""
    full = " ".join(x for x in (given, family) if x)
    return full or None


def _normalize_phone(value: str) -> Optional[str]:
    digits = re.sub(r"[^\d+]", "", value or "")
    return digits if len(re.sub(r"\D", "", digits)) >= 7 else None
