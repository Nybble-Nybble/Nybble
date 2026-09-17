"""Collectors for data that is just files on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Optional

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import file_times, relpath, walk_files

# ---------------------------------------------------------------------------
# Documents, essays, drafts
# ---------------------------------------------------------------------------

# Files that live among the user's documents but are not the user's writing.
# Each is claimed by a dedicated collector that knows its real authorship, so
# picking them up here would attribute other people's words to the user.
_CLAIMED_ELSEWHERE = {
    "my clippings.txt",          # kindle_clippings
    "readme.md", "license.md", "changelog.md", "contributing.md",
    "code_of_conduct.md", "requirements.txt", "package.json",
}
_CLAIMED_PREFIXES = ("whatsapp chat ",)  # whatsapp_export


def _claimed_by_another_collector(path: Path) -> bool:
    name = path.name.lower()
    return name in _CLAIMED_ELSEWHERE or name.startswith(_CLAIMED_PREFIXES)


def collect_documents(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Prose from the document formats people write in."""
    suffixes = tx.ALL_SUFFIXES - {".epub"}  # ebooks are other-authored; see below
    seen = 0
    for path in walk_files(paths, suffixes=suffixes, max_files=60_000):
        if seen >= ctx.max_records_per_source:
            return
        if _claimed_by_another_collector(path):
            continue
        text = tx.extract(path)
        if not text:
            continue
        if path.suffix.lower() in (".md", ".markdown"):
            text = tx.strip_markdown(text)
        if len(text) < ctx.min_chars or not tx.looks_like_prose(text):
            continue
        created, modified = file_times(path)
        if ctx.since and (modified or "") < ctx.since:
            continue
        seen += 1
        yield Record(
            source="local_documents",
            kind="document",
            authorship="self",
            doc_category="written_by_user",
            sensitivity="medium",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app="filesystem",
            meta={"format": path.suffix.lstrip(".")},
        )


def collect_saved_pdfs(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """PDFs the user downloaded and kept — papers, essays, reports.

    Treated as other-authored affinity evidence: choosing to keep a paper is a
    taste signal even when the user never annotated it.
    """
    seen = 0
    for path in walk_files(paths, suffixes={".pdf", ".epub"}, max_files=20_000):
        if seen >= ctx.max_records_per_source:
            return
        text = tx.extract(path)
        if not text or len(text) < 400:
            continue
        created, modified = file_times(path)
        seen += 1
        # Cap the body: we want enough to characterize the piece, not the book.
        yield Record(
            source="saved_pdfs",
            kind="document",
            authorship="other",
            doc_category="liked_writing",
            sensitivity="low",
            text=text[:20_000],
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app="filesystem",
            signals={"saved": True},
            meta={"format": path.suffix.lstrip("."), "truncated": len(text) > 20_000},
        )


# ---------------------------------------------------------------------------
# Note vaults
# ---------------------------------------------------------------------------

_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_TAG = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]{1,40})")


def _markdown_vault(vaults: List[Path], source_id: str, app: str,
                    ctx: Context) -> Iterator[Record]:
    seen = 0
    for vault in vaults:
        for path in walk_files([vault], suffixes={".md", ".markdown"}, max_files=30_000):
            if seen >= ctx.max_records_per_source:
                return
            raw = tx.read_text(path)
            if not raw:
                continue
            title = tx.frontmatter_title(raw) or path.stem
            tags = sorted(set(_TAG.findall(raw)))
            links = sorted(set(_WIKILINK.findall(raw)))
            text = tx.strip_markdown(_WIKILINK.sub(r"\1", raw))
            if len(text) < ctx.min_chars:
                continue
            created, modified = file_times(path)
            if ctx.since and (modified or "") < ctx.since:
                continue
            seen += 1
            yield Record(
                source=source_id,
                kind="document",
                authorship="self",
                doc_category="written_by_user",
                sensitivity="medium",
                text=text,
                title=title,
                created_at=created,
                modified_at=modified,
                path=relpath(path),
                app=app,
                meta={
                    "vault": vault.name,
                    "tags": tags[:40],
                    "links": links[:40],
                },
            )


def collect_obsidian(paths: List[Path], ctx: Context) -> Iterator[Record]:
    yield from _markdown_vault(paths, "obsidian", "Obsidian", ctx)


def collect_logseq(paths: List[Path], ctx: Context) -> Iterator[Record]:
    yield from _markdown_vault(paths, "logseq", "Logseq", ctx)


# ---------------------------------------------------------------------------
# Notebooks and LaTeX
# ---------------------------------------------------------------------------

def collect_notebooks(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Markdown cells from Jupyter notebooks — the prose around the code."""
    seen = 0
    for path in walk_files(paths, suffixes={".ipynb"}, max_files=10_000):
        if seen >= ctx.max_records_per_source:
            return
        raw = tx.read_text(path)
        if not raw:
            continue
        try:
            nb = json.loads(raw)
        except ValueError:
            continue
        cells = nb.get("cells") or []
        chunks = []
        for cell in cells:
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source")
            body = "".join(src) if isinstance(src, list) else str(src or "")
            body = tx.strip_markdown(body)
            if body:
                chunks.append(body)
        text = "\n\n".join(chunks)
        if len(text) < ctx.min_chars:
            continue
        created, modified = file_times(path)
        seen += 1
        yield Record(
            source="jupyter",
            kind="document",
            authorship="self",
            doc_category="written_by_user",
            sensitivity="low",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app="Jupyter",
            meta={"markdown_cells": len(chunks), "total_cells": len(cells)},
        )


def collect_latex(paths: List[Path], ctx: Context) -> Iterator[Record]:
    seen = 0
    for path in walk_files(paths, suffixes={".tex"}, max_files=5_000):
        if seen >= ctx.max_records_per_source:
            return
        raw = tx.read_text(path)
        if not raw:
            continue
        text = tx.from_latex(raw)
        if not text or len(text) < ctx.min_chars:
            continue
        created, modified = file_times(path)
        seen += 1
        yield Record(
            source="latex",
            kind="document",
            authorship="self",
            doc_category="written_by_user",
            sensitivity="low",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app="LaTeX",
        )


# ---------------------------------------------------------------------------
# Kindle clippings
# ---------------------------------------------------------------------------

_CLIPPING_SEP = "=========="
# "- Your Highlight on page 42 | Location 601-602 | Added on Monday, 3 June 2024 09:12:33"
_CLIPPING_META = re.compile(
    r"-\s*Your\s+(?P<type>Highlight|Note|Bookmark)[^|]*"
    r"(?:\|\s*(?P<loc>[^|]*))?"
    r"\|\s*Added on\s+(?P<when>.+)", re.I)
_TITLE_AUTHOR = re.compile(r"^(?P<title>.+?)\s*\((?P<author>[^()]+)\)\s*$")


def collect_kindle(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """`My Clippings.txt` — the cleanest 'writing of others that you like' source."""
    for path in paths:
        raw = tx.read_text(Path(path))
        if not raw:
            continue
        for block in raw.split(_CLIPPING_SEP):
            block = block.strip()
            if not block:
                continue
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue
            header, meta_line = lines[0], lines[1]
            body = "\n".join(lines[2:]).strip()
            if not body or len(body) < 10:
                continue

            title, author = header, None
            m = _TITLE_AUTHOR.match(header)
            if m:
                title, author = m.group("title").strip(), m.group("author").strip()

            clip_type, when, loc = "Highlight", None, None
            mm = _CLIPPING_META.search(meta_line)
            if mm:
                clip_type = (mm.group("type") or "Highlight").title()
                loc = (mm.group("loc") or "").strip() or None
                from .base import parse_date
                when = parse_date(_strip_weekday(mm.group("when")))

            yield Record(
                source="kindle_clippings",
                kind="highlight" if clip_type != "Note" else "comment",
                authorship="other" if clip_type == "Highlight" else "self",
                doc_category="liked_writing",
                sensitivity="low",
                text=body,
                title=title,
                author=author,
                created_at=when,
                path=relpath(Path(path)),
                app="Kindle",
                signals={"highlighted": True} if clip_type == "Highlight" else {},
                meta={"clipping_type": clip_type, "location": loc},
            )


def _strip_weekday(when: str) -> str:
    """'Monday, 3 June 2024 09:12:33' → '3 June 2024 09:12:33'."""
    return re.sub(r"^\s*\w+day,\s*", "", when or "").strip()


# ---------------------------------------------------------------------------
# Screenshots (OCR)
# ---------------------------------------------------------------------------

_SCREENSHOT_NAME = re.compile(
    r"(screen\s?shot|screenshot|capture|cleanshot|Bildschirmfoto)", re.I)


def collect_screenshots(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Text inside screenshots the user kept.

    People screenshot what they want to keep, so the OCR'd text is affinity
    evidence. Runs entirely locally via tesseract; if tesseract is missing the
    collector yields nothing rather than failing the run.
    """
    if not ctx.ocr:
        return
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return

    seen = 0
    for path in walk_files(paths, suffixes={".png", ".jpg", ".jpeg"}, max_files=5_000):
        if seen >= min(ctx.max_records_per_source, 3_000):
            return
        if not _SCREENSHOT_NAME.search(path.name):
            continue
        try:
            with Image.open(path) as img:
                text = pytesseract.image_to_string(img)
        except Exception:
            continue
        text = tx.clean(text or "")
        if len(text) < 80:
            continue
        created, modified = file_times(path)
        seen += 1
        yield Record(
            source="screenshots",
            kind="bookmark",
            authorship="other",
            doc_category="liked_content",
            sensitivity="high",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app="screenshot",
            signals={"saved": True},
            meta={"ocr": True},
        )
