"""Email collectors.

Sent mail is usually the largest and highest-quality sample of how someone
writes at length to a real audience, so it is worth the parsing effort.
"""

from __future__ import annotations

import email
import email.policy
import mailbox
import re
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import parse_date, relpath, walk_files

MAIL_SUFFIXES = {".mbox", ".emlx", ".eml", ".partial.emlx"}

# Quoted replies and signatures add no signal and triple the corpus size.
_QUOTE_LINE = re.compile(r"^\s*(>|\|)")
_REPLY_HEADER = re.compile(
    r"^\s*(On .{5,120}\bwrote:|-{2,}\s*Original Message\s*-{2,}|"
    r"From:\s.*|_{10,}|Sent from my \w+)", re.I | re.M)
_SIGNATURE = re.compile(r"^--\s*$", re.M)


def collect_local_mail(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Apple Mail (.emlx), Thunderbird (mbox), Maildir, and loose .eml files."""
    seen = 0
    for path in walk_files(paths, max_files=200_000, max_depth=12):
        if seen >= ctx.max_records_per_source:
            return
        name = path.name.lower()
        try:
            if name.endswith(".emlx"):
                for rec in _from_emlx(path, "email_local", ctx):
                    yield rec
                    seen += 1
            elif name.endswith(".eml"):
                raw = path.read_bytes()
                rec = _record_from_message(_parse(raw), path, "email_local", ctx)
                if rec:
                    yield rec
                    seen += 1
            elif name.endswith(".mbox") or _is_mbox(path):
                for rec in _from_mbox(path, "email_local", ctx):
                    yield rec
                    seen += 1
                    if seen >= ctx.max_records_per_source:
                        return
        except (OSError, ValueError, AttributeError):
            continue


def collect_mbox_export(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Gmail Takeout .mbox files, which are often a single multi-GB file."""
    seen = 0
    for base in paths:
        base = Path(base)
        candidates = [base] if base.is_file() else list(base.rglob("*.mbox"))
        for mbox_path in candidates:
            for rec in _from_mbox(mbox_path, "email_export", ctx):
                yield rec
                seen += 1
                if seen >= ctx.max_records_per_source:
                    return


def _is_mbox(path: Path) -> bool:
    if path.suffix.lower() in (".mbox",):
        return True
    try:
        if path.stat().st_size < 64:
            return False
        with path.open("rb") as fh:
            return fh.read(5) == b"From "
    except OSError:
        return False


def _parse(raw: bytes) -> Message:
    return email.message_from_bytes(raw, policy=email.policy.default)


def _from_emlx(path: Path, source_id: str, ctx: Context) -> Iterator[Record]:
    """Apple Mail's .emlx: a byte-count line, the RFC822 message, then a plist."""
    try:
        raw = path.read_bytes()
    except OSError:
        return
    first_newline = raw.find(b"\n")
    if first_newline == -1:
        return
    header = raw[:first_newline].strip()
    body = raw[first_newline + 1:]
    if header.isdigit():
        body = body[: int(header)]
    rec = _record_from_message(_parse(body), path, source_id, ctx)
    if rec:
        yield rec


def _from_mbox(path: Path, source_id: str, ctx: Context) -> Iterator[Record]:
    try:
        box = mailbox.mbox(str(path), factory=None, create=False)
    except (OSError, ValueError):
        return
    try:
        for key in box.keys():
            try:
                msg = box[key]
            except Exception:
                continue
            rec = _record_from_message(msg, path, source_id, ctx)
            if rec:
                yield rec
    finally:
        try:
            box.close()
        except Exception:
            pass


def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        return str(value)


def _addresses(value: Optional[str]) -> str:
    return _decode(value).strip()


def _body_text(msg: Message) -> Optional[str]:
    """Plain-text body, falling back to stripped HTML."""
    plain, html = None, None
    try:
        parts = msg.walk() if msg.is_multipart() else [msg]
    except Exception:
        return None
    for part in parts:
        try:
            ctype = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            if ctype not in ("text/plain", "text/html"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            text = tx.decode(payload)
            if not text:
                continue
            if ctype == "text/plain" and plain is None:
                plain = text
            elif ctype == "text/html" and html is None:
                html = text
        except Exception:
            continue
    if plain:
        return plain
    return tx.from_html(html) if html else None


def _strip_quoted(text: str) -> str:
    """Drop quoted replies and signature blocks; keep what this person wrote."""
    cut = _REPLY_HEADER.search(text)
    if cut and cut.start() > 40:
        text = text[: cut.start()]
    sig = _SIGNATURE.search(text)
    if sig and sig.start() > 40:
        text = text[: sig.start()]
    lines = [ln for ln in text.splitlines() if not _QUOTE_LINE.match(ln)]
    return tx.clean("\n".join(lines))


def _record_from_message(msg: Message, path: Path, source_id: str,
                         ctx: Context) -> Optional[Record]:
    if msg is None:
        return None
    body = _body_text(msg)
    if not body:
        return None
    body = _strip_quoted(body)
    if len(body) < ctx.min_chars:
        return None

    when = parse_date(msg.get("Date"))
    if ctx.since and (when or "") < ctx.since:
        return None

    sender = _addresses(msg.get("From"))
    recipient = _addresses(msg.get("To"))
    subject = _decode(msg.get("Subject"))

    # Folder name is the only reliable local hint about direction.
    lowered = str(path).lower()
    from_me = any(k in lowered for k in ("sent", "outbox", "drafts"))

    return Record(
        source=source_id,
        kind="message",
        authorship="self" if from_me else "other",
        doc_category="messages",
        sensitivity="high",
        text=body,
        title=subject or None,
        author="me" if from_me else (sender or None),
        recipient=recipient or None,
        thread_id=(msg.get("In-Reply-To") or msg.get("Message-ID") or "").strip() or None,
        created_at=when,
        app="Mail",
        path=relpath(path),
        meta={"from": sender or None, "list": msg.get("List-Id")},
    )
