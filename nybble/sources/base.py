"""Helpers shared by collectors."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

# Directories that never contain personal writing but do contain millions of
# files. Walking into them is the difference between a 2-minute and a 2-hour run.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "site-packages", "dist-packages", ".tox", ".mypy_cache",
    ".pytest_cache", ".gradle", ".m2", "target", "build", "dist", ".next",
    ".nuxt", ".cache", "Cache", "Caches", "CachedData", "vendor", "Pods",
    ".terraform", "bower_components", ".idea", ".vscode", "Library",
    "AppData", "System", "Windows", "Program Files", "Applications",
    ".Trash", "$RECYCLE.BIN", ".DS_Store", ".nybble",
}

MAX_DEPTH = 8


def walk_files(
    roots: Sequence[Path],
    suffixes: Optional[set] = None,
    max_files: int = 50_000,
    max_depth: int = MAX_DEPTH,
    skip_dirs: Optional[set] = None,
) -> Iterator[Path]:
    """Depth-limited walk that skips build/cache trees and tolerates EPERM."""
    skip = SKIP_DIRS | (skip_dirs or set())
    count = 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            if root.is_file():
                yield root
                count += 1
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            here = Path(dirpath)
            if len(here.parts) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            # Prune in place so os.walk never descends.
            dirnames[:] = [d for d in dirnames
                           if d not in skip and not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                p = here / name
                if suffixes and p.suffix.lower() not in suffixes:
                    continue
                yield p
                count += 1
                if count >= max_files:
                    return


@contextmanager
def open_sqlite(path: Path, copy: bool = True) -> Iterator[Optional[sqlite3.Connection]]:
    """Open a SQLite DB read-only.

    Live app databases (Chrome, Messages) are usually locked by the running app,
    and a WAL sidecar means a bare copy of the .db can be stale. Copy the db plus
    its -wal/-shm companions to a temp dir and read that.
    """
    tmpdir = None
    conn = None
    try:
        target = path
        if copy:
            tmpdir = tempfile.mkdtemp(prefix="nybble-db-")
            target = Path(tmpdir) / path.name
            shutil.copy2(path, target)
            for suffix in ("-wal", "-shm"):
                side = Path(str(path) + suffix)
                if side.exists():
                    try:
                        shutil.copy2(side, Path(str(target) + suffix))
                    except OSError:
                        pass
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        yield conn
    except (sqlite3.Error, OSError, shutil.Error):
        yield None
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def query(conn: sqlite3.Connection, sql: str, params: Sequence = ()) -> List[sqlite3.Row]:
    """Run a query, returning [] when the schema does not match this app version."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    rows = query(
        conn, "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,))
    return bool(rows)


def columns(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in query(conn, f"PRAGMA table_info({table})")}


# --- timestamps ---------------------------------------------------------------

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
WEBKIT_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def from_unix(value: Optional[float], unit: str = "s") -> Optional[str]:
    """Unix timestamp to ISO. `unit` is s, ms, or us."""
    if value in (None, 0, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    divisor = {"s": 1.0, "ms": 1e3, "us": 1e6}.get(unit, 1.0)
    try:
        return iso(datetime.fromtimestamp(v / divisor, tz=timezone.utc))
    except (OverflowError, OSError, ValueError):
        return None


def from_apple(value: Optional[float]) -> Optional[str]:
    """Apple Core Data timestamps: seconds (or nanoseconds) since 2001-01-01."""
    if value in (None, 0, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if abs(v) > 1e12:  # nanosecond variant used by newer chat.db rows
        v /= 1e9
    try:
        return iso(APPLE_EPOCH + timedelta(seconds=v))
    except (OverflowError, ValueError):
        return None


def from_webkit(value: Optional[float]) -> Optional[str]:
    """Chromium timestamps: microseconds since 1601-01-01."""
    if value in (None, 0, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    try:
        return iso(WEBKIT_EPOCH + timedelta(microseconds=v))
    except (OverflowError, ValueError):
        return None


def parse_date(text: Optional[str]) -> Optional[str]:
    """Best-effort parse of the date formats found in exports."""
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None
    # Numeric epoch hiding in a string field.
    if text.isdigit():
        n = int(text)
        if n > 1e17:
            return from_unix(n, "us")
        if n > 1e11:
            return from_unix(n, "ms")
        if n > 1e8:
            return from_unix(n, "s")
    candidates = (
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%d/%m/%Y %H:%M",
        "%d/%m/%Y", "%b %d, %Y %I:%M:%S %p", "%b %d, %Y", "%B %d, %Y",
        # Kindle and other non-US locales write the day first, month spelled out.
        "%d %B %Y %H:%M:%S", "%d %b %Y %H:%M:%S", "%d %B %Y", "%d %b %Y",
        "%B %d, %Y %H:%M:%S", "%b %d, %Y %H:%M:%S",
        # iCalendar basic format, with and without the UTC designator.
        "%a %b %d %H:%M:%S %Y", "%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d",
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 (email)
    )
    cleaned = text.replace("Z", "+0000")
    for fmt in candidates:
        try:
            return iso(datetime.strptime(cleaned, fmt))
        except ValueError:
            continue
    try:  # 3.11+ handles most ISO-8601 variants directly
        return iso(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def file_times(path: Path) -> tuple:
    """(created_iso, modified_iso) for a file, where the OS records creation."""
    try:
        st = path.stat()
    except OSError:
        return None, None
    created = getattr(st, "st_birthtime", None) or st.st_ctime
    return from_unix(created), from_unix(st.st_mtime)


def relpath(path: Path) -> str:
    """Path with $HOME collapsed, so manifests are not full of usernames."""
    s = str(path)
    h = str(Path.home())
    return "~" + s[len(h):] if s.startswith(h) else s
