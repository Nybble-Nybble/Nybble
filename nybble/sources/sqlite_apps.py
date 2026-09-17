"""Collectors for apps that keep their data in a local SQLite database."""

from __future__ import annotations

import gzip
import json
import plistlib
import re
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import (columns, from_apple, from_unix, from_webkit, open_sqlite,
                   parse_date, query, relpath, table_exists, walk_files)

# ===========================================================================
# iMessage / SMS
# ===========================================================================

def collect_imessage(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Messages from chat.db, both directions, grouped into threads.

    Needs Full Disk Access on macOS; without it the copy raises EPERM and this
    yields nothing rather than failing the run.
    """
    for db_path in paths:
        with open_sqlite(Path(db_path)) as conn:
            if conn is None or not table_exists(conn, "message"):
                continue

            cols = columns(conn, "message")
            has_attr = "attributedBody" in cols
            date_expr = "m.date"

            sql = f"""
                SELECT m.ROWID          AS rowid,
                       m.text           AS text,
                       {"m.attributedBody AS attributed," if has_attr else ""}
                       {date_expr}      AS date,
                       m.is_from_me     AS is_from_me,
                       m.service        AS service,
                       h.id             AS handle,
                       c.chat_identifier AS chat_id,
                       c.display_name   AS chat_name,
                       c.style          AS chat_style
                FROM message m
                LEFT JOIN handle h            ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                LEFT JOIN chat c              ON c.ROWID = cmj.chat_id
                WHERE (m.text IS NOT NULL AND m.text != '')
                   {"OR m.attributedBody IS NOT NULL" if has_attr else ""}
                ORDER BY m.date
            """
            seen = 0
            for row in query(conn, sql):
                if seen >= ctx.max_records_per_source:
                    break
                text = row["text"]
                if not text and has_attr:
                    text = _decode_attributed_body(row["attributed"])
                if not text:
                    continue
                text = tx.clean(text)
                if not text:
                    continue

                when = from_apple(row["date"])
                if ctx.since and (when or "") < ctx.since:
                    continue

                from_me = bool(row["is_from_me"])
                handle = row["handle"]
                is_group = (row["chat_style"] == 43)

                seen += 1
                yield Record(
                    source="imessage",
                    kind="message",
                    authorship="self" if from_me else "other",
                    doc_category="messages",
                    sensitivity="high",
                    text=text,
                    created_at=when,
                    author="me" if from_me else (handle or "unknown"),
                    recipient=(handle or row["chat_id"]) if from_me else "me",
                    thread_id=row["chat_id"] or handle,
                    app="Messages",
                    path=relpath(Path(db_path)),
                    meta={
                        "service": row["service"],
                        "group_chat": is_group,
                        "chat_name": row["chat_name"],
                    },
                )


# Modern macOS stores message text inside an NSAttributedString archive rather
# than the `text` column. The body is wrapped in a typedstream; the plain text
# follows the NSString marker. This pulls it out without an NSKeyedUnarchiver.
_ATTR_MARKERS = (b"NSString\x01\x94\x84\x01+", b"NSString\x01\x95\x84\x01+")


def _decode_attributed_body(blob) -> Optional[str]:
    if not blob:
        return None
    if isinstance(blob, str):
        blob = blob.encode("utf-8", "replace")
    if not isinstance(blob, (bytes, bytearray)):
        return None
    data = bytes(blob)

    start = -1
    for marker in _ATTR_MARKERS:
        idx = data.find(marker)
        if idx != -1:
            start = idx + len(marker)
            break
    if start == -1:
        idx = data.find(b"NSString")
        if idx == -1:
            return None
        # Skip the class descriptor and the length prefix that follows it.
        start = idx + 8
        while start < len(data) and data[start] in (0x01, 0x94, 0x95, 0x84, 0x2B):
            start += 1

    if start >= len(data):
        return None

    # A 1-byte length, or 0x81 followed by a little-endian uint16.
    length = data[start]
    start += 1
    if length == 0x81 and start + 2 <= len(data):
        length = int.from_bytes(data[start:start + 2], "little")
        start += 2
    elif length == 0x82 and start + 4 <= len(data):
        length = int.from_bytes(data[start:start + 4], "little")
        start += 4

    chunk = data[start:start + length] if 0 < length < len(data) else data[start:]
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        text = chunk.decode("utf-8", "ignore")
    text = text.split("\x00")[0].strip()
    # Trailing archive class names leak in when the length prefix was wrong.
    text = re.split(r"(?:NSDictionary|NSAttributedString|__kIM|NSNumber)", text)[0]
    return text.strip() or None


# ===========================================================================
# Apple Notes
# ===========================================================================

def collect_apple_notes(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Apple Notes bodies.

    Note text is gzipped protobuf in ZICNOTEDATA.ZDATA. Rather than depend on a
    protobuf schema that Apple changes between releases, decompress and pull the
    printable runs out of the wire format.
    """
    for db_path in paths:
        with open_sqlite(Path(db_path)) as conn:
            if conn is None or not table_exists(conn, "ZICCLOUDSYNCINGOBJECT"):
                continue
            if not table_exists(conn, "ZICNOTEDATA"):
                continue

            note_cols = columns(conn, "ZICCLOUDSYNCINGOBJECT")
            title_col = "ZTITLE1" if "ZTITLE1" in note_cols else "ZTITLE"
            folder_join = "ZFOLDER" in note_cols

            sql = f"""
                SELECT o.Z_PK             AS pk,
                       o.{title_col}      AS title,
                       o.ZCREATIONDATE1   AS created,
                       o.ZMODIFICATIONDATE1 AS modified,
                       {"f.ZTITLE2 AS folder," if folder_join else "NULL AS folder,"}
                       d.ZDATA            AS data
                FROM ZICCLOUDSYNCINGOBJECT o
                JOIN ZICNOTEDATA d ON d.ZNOTE = o.Z_PK
                {"LEFT JOIN ZICCLOUDSYNCINGOBJECT f ON f.Z_PK = o.ZFOLDER" if folder_join else ""}
                WHERE d.ZDATA IS NOT NULL
            """
            rows = query(conn, sql)
            if not rows:  # older schema without the ZTITLE1 suffix
                rows = query(conn, """
                    SELECT o.Z_PK AS pk, o.ZTITLE AS title,
                           o.ZCREATIONDATE AS created, o.ZMODIFICATIONDATE AS modified,
                           NULL AS folder, d.ZDATA AS data
                    FROM ZICCLOUDSYNCINGOBJECT o
                    JOIN ZICNOTEDATA d ON d.ZNOTE = o.Z_PK
                    WHERE d.ZDATA IS NOT NULL
                """)

            seen = 0
            for row in rows:
                if seen >= ctx.max_records_per_source:
                    break
                text = _decode_note_blob(row["data"])
                if not text or len(text) < ctx.min_chars:
                    continue
                modified = from_apple(row["modified"])
                if ctx.since and (modified or "") < ctx.since:
                    continue
                seen += 1
                yield Record(
                    source="apple_notes",
                    kind="document",
                    authorship="self",
                    doc_category="written_by_user",
                    sensitivity="medium",
                    text=text,
                    title=row["title"],
                    created_at=from_apple(row["created"]),
                    modified_at=modified,
                    app="Apple Notes",
                    path=relpath(Path(db_path)),
                    meta={"folder": row["folder"]},
                )


_PRINTABLE_RUN = re.compile(rb"[\x20-\x7E\n\t\xc2-\xf4][\x20-\x7E\n\t\x80-\xbf]{6,}")


def _decode_note_blob(blob) -> Optional[str]:
    if not blob:
        return None
    if isinstance(blob, str):
        blob = blob.encode("utf-8", "replace")
    raw = bytes(blob)

    try:
        raw = gzip.decompress(raw)
    except (OSError, EOFError, zlib.error):
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass  # some notes are stored uncompressed

    # The note body is the first long UTF-8 run in the protobuf payload; the
    # runs after it are style/attachment metadata.
    runs = []
    for match in _PRINTABLE_RUN.finditer(raw):
        try:
            piece = match.group(0).decode("utf-8")
        except UnicodeDecodeError:
            continue
        piece = piece.strip()
        if len(piece) < 4:
            continue
        # Skip protobuf-ish identifiers and UUIDs.
        if re.fullmatch(r"[A-Fa-f0-9-]{16,}", piece):
            continue
        if piece.startswith(("com.apple.", "public.", "Helvetica", "SFUI")):
            continue
        runs.append(piece)
        if len(runs) > 200:
            break
    if not runs:
        return None
    return tx.clean("\n".join(runs))


# ===========================================================================
# Bear, Day One, Joplin
# ===========================================================================

def collect_bear(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for db_path in paths:
        with open_sqlite(Path(db_path)) as conn:
            if conn is None or not table_exists(conn, "ZSFNOTE"):
                continue
            rows = query(conn, """
                SELECT ZTITLE AS title, ZTEXT AS text,
                       ZCREATIONDATE AS created, ZMODIFICATIONDATE AS modified,
                       ZTRASHED AS trashed
                FROM ZSFNOTE
                WHERE ZTEXT IS NOT NULL
            """)
            for row in rows:
                if row["trashed"]:
                    continue
                text = tx.strip_markdown(row["text"] or "")
                if len(text) < ctx.min_chars:
                    continue
                yield Record(
                    source="bear", kind="document", authorship="self",
                    doc_category="written_by_user", sensitivity="medium",
                    text=text, title=row["title"],
                    created_at=from_apple(row["created"]),
                    modified_at=from_apple(row["modified"]),
                    app="Bear", path=relpath(Path(db_path)),
                )


def collect_dayone(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for db_path in paths:
        with open_sqlite(Path(db_path)) as conn:
            if conn is None or not table_exists(conn, "ZENTRY"):
                continue
            cols = columns(conn, "ZENTRY")
            text_col = "ZMARKDOWNTEXT" if "ZMARKDOWNTEXT" in cols else "ZTEXT"
            rows = query(conn, f"""
                SELECT {text_col} AS text, ZCREATIONDATE AS created,
                       ZMODIFIEDDATE AS modified
                FROM ZENTRY WHERE {text_col} IS NOT NULL
            """)
            for row in rows:
                text = tx.strip_markdown(row["text"] or "")
                if len(text) < ctx.min_chars:
                    continue
                yield Record(
                    source="dayone", kind="document", authorship="self",
                    doc_category="written_by_user", sensitivity="high",
                    text=text,
                    created_at=from_apple(row["created"]),
                    modified_at=from_apple(row["modified"]),
                    app="Day One", path=relpath(Path(db_path)),
                )


def collect_joplin(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for base in paths:
        db_path = Path(base) / "database.sqlite"
        if not db_path.exists():
            continue
        with open_sqlite(db_path) as conn:
            if conn is None or not table_exists(conn, "notes"):
                continue
            for row in query(conn, """
                SELECT title, body, created_time, updated_time
                FROM notes WHERE body IS NOT NULL
            """):
                text = tx.strip_markdown(row["body"] or "")
                if len(text) < ctx.min_chars:
                    continue
                yield Record(
                    source="joplin", kind="document", authorship="self",
                    doc_category="written_by_user", sensitivity="medium",
                    text=text, title=row["title"],
                    created_at=from_unix(row["created_time"], "ms"),
                    modified_at=from_unix(row["updated_time"], "ms"),
                    app="Joplin", path=relpath(db_path),
                )


# ===========================================================================
# Browsers
# ===========================================================================

def _browser_name(profile: Path) -> str:
    parts = [p.lower() for p in profile.parts]
    for needle, label in (
        ("brave", "Brave"), ("edge", "Edge"), ("arc", "Arc"),
        ("vivaldi", "Vivaldi"), ("opera", "Opera"), ("chromium", "Chromium"),
        ("chrome", "Chrome"), ("firefox", "Firefox"), ("safari", "Safari"),
    ):
        if any(needle in part for part in parts):
            return label
    return "Browser"


def collect_history(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Browsing history with visit counts.

    Visit count is the local stand-in for the attention signal the plan doc
    wants: a page you returned to eleven times is one you liked.
    """
    for profile in paths:
        profile = Path(profile)
        app = _browser_name(profile)

        chromium_db = profile / "History"
        if chromium_db.exists():
            yield from _chromium_history(chromium_db, app, ctx)
            continue

        firefox_db = profile / "places.sqlite"
        if firefox_db.exists():
            yield from _firefox_history(firefox_db, app, ctx)
            continue

        safari_db = profile / "History.db"
        if safari_db.exists():
            yield from _safari_history(safari_db, ctx)


def _emit_visit(source_url, title, visits, when, app, path, ctx) -> Optional[Record]:
    if not source_url or source_url.startswith(("chrome://", "about:", "file://",
                                                "edge://", "brave://")):
        return None
    title = tx.clean(title or "")
    if not title:
        return None
    if ctx.since and (when or "") < ctx.since:
        return None
    return Record(
        source="browser_history",
        kind="consumption",
        authorship="other",
        doc_category="liked_content",
        sensitivity="high",
        text=title,
        title=title,
        url=source_url,
        created_at=when,
        app=app,
        path=relpath(Path(path)),
        signals={"visit_count": int(visits or 0),
                 "revisited": bool((visits or 0) > 2)},
        meta={"domain": _domain(source_url)},
    )


def _domain(url: str) -> Optional[str]:
    m = re.match(r"https?://([^/:]+)", url or "")
    return m.group(1).lower() if m else None


def _chromium_history(db: Path, app: str, ctx: Context) -> Iterator[Record]:
    with open_sqlite(db) as conn:
        if conn is None or not table_exists(conn, "urls"):
            return
        rows = query(conn, """
            SELECT url, title, visit_count, last_visit_time
            FROM urls
            WHERE title IS NOT NULL AND title != ''
            ORDER BY visit_count DESC
            LIMIT 60000
        """)
        for row in rows:
            rec = _emit_visit(row["url"], row["title"], row["visit_count"],
                              from_webkit(row["last_visit_time"]), app, db, ctx)
            if rec:
                yield rec


def _firefox_history(db: Path, app: str, ctx: Context) -> Iterator[Record]:
    with open_sqlite(db) as conn:
        if conn is None or not table_exists(conn, "moz_places"):
            return
        rows = query(conn, """
            SELECT url, title, visit_count, last_visit_date
            FROM moz_places
            WHERE title IS NOT NULL AND title != ''
            ORDER BY visit_count DESC
            LIMIT 60000
        """)
        for row in rows:
            rec = _emit_visit(row["url"], row["title"], row["visit_count"],
                              from_unix(row["last_visit_date"], "us"), app, db, ctx)
            if rec:
                yield rec


def _safari_history(db: Path, ctx: Context) -> Iterator[Record]:
    with open_sqlite(db) as conn:
        if conn is None or not table_exists(conn, "history_items"):
            return
        rows = query(conn, """
            SELECT i.url AS url, v.title AS title, i.visit_count AS visit_count,
                   MAX(v.visit_time) AS visit_time
            FROM history_items i
            JOIN history_visits v ON v.history_item = i.id
            WHERE v.title IS NOT NULL AND v.title != ''
            GROUP BY i.id
            ORDER BY i.visit_count DESC
            LIMIT 60000
        """)
        for row in rows:
            rec = _emit_visit(row["url"], row["title"], row["visit_count"],
                              from_apple(row["visit_time"]), "Safari", db, ctx)
            if rec:
                yield rec


def collect_bookmarks(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Bookmarks across all browser families. A bookmark is a deliberate keep."""
    for profile in paths:
        profile = Path(profile)
        app = _browser_name(profile)

        chromium_json = profile / "Bookmarks"
        if chromium_json.exists():
            yield from _chromium_bookmarks(chromium_json, app, ctx)
            continue

        places = profile / "places.sqlite"
        if places.exists():
            yield from _firefox_bookmarks(places, app, ctx)
            continue

        safari_plist = profile / "Bookmarks.plist"
        if safari_plist.exists():
            yield from _safari_bookmarks(safari_plist, ctx)


def _bookmark_record(url, title, when, app, path, folder=None) -> Optional[Record]:
    if not url or not url.startswith("http"):
        return None
    title = tx.clean(title or "") or url
    return Record(
        source="browser_bookmarks",
        kind="bookmark",
        authorship="other",
        doc_category="liked_content",
        sensitivity="medium",
        text=title,
        title=title,
        url=url,
        created_at=when,
        app=app,
        path=relpath(Path(path)),
        signals={"saved": True, "bookmarked": True},
        meta={"folder": folder, "domain": _domain(url)},
    )


def _chromium_bookmarks(path: Path, app: str, ctx: Context) -> Iterator[Record]:
    raw = tx.read_text(path)
    if not raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return

    def walk(node, folder=None):
        if not isinstance(node, dict):
            return
        if node.get("type") == "url":
            rec = _bookmark_record(node.get("url"), node.get("name"),
                                   from_webkit(node.get("date_added")),
                                   app, path, folder)
            if rec:
                yield rec
        for child in node.get("children") or []:
            yield from walk(child, node.get("name") or folder)

    for root in (data.get("roots") or {}).values():
        yield from walk(root)


def _firefox_bookmarks(db: Path, app: str, ctx: Context) -> Iterator[Record]:
    with open_sqlite(db) as conn:
        if conn is None or not table_exists(conn, "moz_bookmarks"):
            return
        rows = query(conn, """
            SELECT b.title AS title, p.url AS url, b.dateAdded AS added,
                   parent.title AS folder
            FROM moz_bookmarks b
            JOIN moz_places p ON p.id = b.fk
            LEFT JOIN moz_bookmarks parent ON parent.id = b.parent
            WHERE b.type = 1
            LIMIT 40000
        """)
        for row in rows:
            rec = _bookmark_record(row["url"], row["title"],
                                   from_unix(row["added"], "us"), app, db,
                                   row["folder"])
            if rec:
                yield rec


def _safari_bookmarks(path: Path, ctx: Context) -> Iterator[Record]:
    try:
        with path.open("rb") as fh:
            data = plistlib.load(fh)
    except Exception:
        return

    def walk(node, folder=None):
        if isinstance(node, dict):
            if node.get("WebBookmarkType") == "WebBookmarkTypeLeaf":
                uri = node.get("URLString")
                title = (node.get("URIDictionary") or {}).get("title")
                rec = _bookmark_record(uri, title, None, "Safari", path, folder)
                if rec:
                    yield rec
            for child in node.get("Children") or []:
                yield from walk(child, node.get("Title") or folder)
        elif isinstance(node, list):
            for child in node:
                yield from walk(child, folder)

    yield from walk(data)


# ===========================================================================
# Reading: Zotero, Apple Books
# ===========================================================================

def collect_zotero(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Saved papers, plus the user's own notes and PDF annotations on them."""
    for db_path in paths:
        db_path = Path(db_path)
        with open_sqlite(db_path) as conn:
            if conn is None or not table_exists(conn, "items"):
                continue

            titles: Dict[int, str] = {}
            for row in query(conn, """
                SELECT idv.itemID AS item_id, v.value AS value
                FROM itemData idv
                JOIN itemDataValues v ON v.valueID = idv.valueID
                JOIN fields f ON f.fieldID = idv.fieldID
                WHERE f.fieldName = 'title'
            """):
                titles[row["item_id"]] = row["value"]

            # Library items: the fact that a paper was saved is the signal.
            for row in query(conn, """
                SELECT i.itemID AS item_id, i.dateAdded AS added,
                       v.value AS abstract
                FROM items i
                LEFT JOIN itemData idv ON idv.itemID = i.itemID
                LEFT JOIN itemDataValues v ON v.valueID = idv.valueID
                LEFT JOIN fields f ON f.fieldID = idv.fieldID
                    AND f.fieldName = 'abstractNote'
                WHERE f.fieldName = 'abstractNote'
                LIMIT 20000
            """):
                title = titles.get(row["item_id"])
                body = tx.clean(row["abstract"] or "")
                if not body and not title:
                    continue
                yield Record(
                    source="zotero", kind="bookmark", authorship="other",
                    doc_category="liked_writing", sensitivity="low",
                    text=body or title, title=title,
                    created_at=parse_date(row["added"]),
                    app="Zotero", path=relpath(db_path),
                    signals={"saved": True},
                )

            # The user's own notes on those items.
            if table_exists(conn, "itemNotes"):
                for row in query(conn, """
                    SELECT n.note AS note, n.parentItemID AS parent
                    FROM itemNotes n WHERE n.note IS NOT NULL LIMIT 20000
                """):
                    body = tx.from_html(row["note"] or "")
                    if not body or len(body) < 20:
                        continue
                    yield Record(
                        source="zotero", kind="comment", authorship="self",
                        doc_category="liked_writing", sensitivity="low",
                        text=body, title=titles.get(row["parent"]),
                        app="Zotero", path=relpath(db_path),
                        meta={"note_on": titles.get(row["parent"])},
                    )

            # Highlighted passages in PDFs.
            if table_exists(conn, "itemAnnotations"):
                for row in query(conn, """
                    SELECT text, comment, parentItemID AS parent
                    FROM itemAnnotations
                    WHERE text IS NOT NULL AND text != '' LIMIT 40000
                """):
                    body = tx.clean(row["text"] or "")
                    if len(body) < 15:
                        continue
                    yield Record(
                        source="zotero", kind="highlight", authorship="other",
                        doc_category="liked_writing", sensitivity="low",
                        text=body, title=titles.get(row["parent"]),
                        app="Zotero", path=relpath(db_path),
                        signals={"highlighted": True},
                        meta={"comment": tx.clean(row["comment"] or "") or None},
                    )


def collect_apple_books(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Highlights and margin notes from Apple Books."""
    annotation_dbs: List[Path] = []
    library_dbs: List[Path] = []
    for base in paths:
        base = Path(base)
        if "AEAnnotation" in base.name:
            annotation_dbs += list(base.glob("*.sqlite"))
        elif "BKLibrary" in base.name:
            library_dbs += list(base.glob("*.sqlite"))

    # asset id -> book title
    books: Dict[str, str] = {}
    for db in library_dbs:
        with open_sqlite(db) as conn:
            if conn is None or not table_exists(conn, "ZBKLIBRARYASSET"):
                continue
            for row in query(conn, """
                SELECT ZASSETID AS asset, ZTITLE AS title, ZAUTHOR AS author
                FROM ZBKLIBRARYASSET
            """):
                if row["asset"]:
                    books[row["asset"]] = f"{row['title']}||{row['author'] or ''}"

    for db in annotation_dbs:
        with open_sqlite(db) as conn:
            if conn is None or not table_exists(conn, "ZAEANNOTATION"):
                continue
            for row in query(conn, """
                SELECT ZANNOTATIONSELECTEDTEXT AS selected,
                       ZANNOTATIONNOTE AS note,
                       ZANNOTATIONASSETID AS asset,
                       ZANNOTATIONCREATIONDATE AS created
                FROM ZAEANNOTATION
                WHERE ZANNOTATIONDELETED = 0
                  AND (ZANNOTATIONSELECTEDTEXT IS NOT NULL
                       OR ZANNOTATIONNOTE IS NOT NULL)
                LIMIT 40000
            """):
                title, author = (books.get(row["asset"], "||").split("||") + [""])[:2]
                when = from_apple(row["created"])
                selected = tx.clean(row["selected"] or "")
                note = tx.clean(row["note"] or "")

                if selected and len(selected) > 10:
                    yield Record(
                        source="apple_books", kind="highlight", authorship="other",
                        doc_category="liked_writing", sensitivity="low",
                        text=selected, title=title or None, author=author or None,
                        created_at=when, app="Apple Books", path=relpath(db),
                        signals={"highlighted": True},
                    )
                if note and len(note) > 5:
                    yield Record(
                        source="apple_books", kind="comment", authorship="self",
                        doc_category="liked_writing", sensitivity="low",
                        text=note, title=title or None, author=author or None,
                        created_at=when, app="Apple Books", path=relpath(db),
                        meta={"note_on_highlight": selected[:200] or None},
                    )
