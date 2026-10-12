"""Local message sources (Messages, WhatsApp): check a database is readable and run its extractor."""

import logging
import sqlite3
import stat
from pathlib import Path

import imessage_extract
import store
import whatsapp_extract

# Connector id (see app/mcps.py) -> extractor module and a cheap query that proves the right database is open.
SOURCES = {
    "messages": {"module": imessage_extract, "count_sql": "SELECT COUNT(*) FROM message"},
    "whatsapp": {"module": whatsapp_extract, "count_sql": "SELECT COUNT(*) FROM ZWAMESSAGE"},
}

ACCESS_HELP = (
    "macOS blocked access to {path}. Give the app running the Nybble server (e.g. Terminal) "
    "Full Disk Access in System Settings > Privacy & Security, then restart the server."
)


log = logging.getLogger("nybble")


class SourceError(Exception):
    """A problem the user can fix, with a message safe to show in the UI."""


def resolve(source_id: str, db_path: str = "") -> Path:
    """The user's path with ~ expanded, or the extractor's default location when left blank."""
    raw = (db_path or "").strip()
    return Path(raw).expanduser() if raw else SOURCES[source_id]["module"].DB_PATH


def check(source_id: str, db_path: str = "") -> dict:
    """Open the database read-only and count its messages. Raises SourceError if it can't be read."""
    path = resolve(source_id, db_path)
    wrong_db = f"{path} doesn't look like the right database. Check the path, or leave it blank to use the default."
    try:
        st = path.stat()  # Path.exists() hides permission errors, which we need to tell apart from a missing file
    except (FileNotFoundError, NotADirectoryError):  # NotADirectory: part of the path is a file, e.g. notes.txt/chat.db
        raise SourceError(f"No database found at {path}. Is the app installed and signed in on this Mac?")
    except PermissionError:
        raise SourceError(ACCESS_HELP.format(path=path))
    except (OSError, ValueError):  # name too long, a stray null byte, a symlink loop...
        raise SourceError(f"{path} isn't a usable file path.")
    if stat.S_ISDIR(st.st_mode):
        raise SourceError(f"{path} is a folder. Give the path to the database file inside it.")
    if not stat.S_ISREG(st.st_mode):
        raise SourceError(wrong_db)

    try:
        conn = store.open_readonly(path)
        try:
            count = conn.execute(SOURCES[source_id]["count_sql"]).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.OperationalError as err:
        msg = str(err)
        if "unable to open" in msg or "authorization denied" in msg:
            raise SourceError(ACCESS_HELP.format(path=path))
        if "locked" in msg or "busy" in msg:
            raise SourceError(f"{path} is busy. Try again in a moment.")
        if "no such table" not in msg and "not a database" not in msg:
            log.warning("Unexpected error reading %s: %s", path, msg)
        raise SourceError(wrong_db)
    except sqlite3.DatabaseError:  # corrupt, encrypted, or not SQLite at all
        raise SourceError(wrong_db)

    return {"db_path": str(path), "messages": count}


def has_data(source_id: str) -> bool:
    return store.has(source_id)


def origin(source_id: str):
    """The app database the source's data was extracted from, if recorded."""
    return store.origin(source_id)


def delete(source_id: str) -> None:
    """Remove everything extracted for the source from the local store. The app's own database is never touched."""
    store.clear(source_id)


def extract(source_id: str, db_path: str = "") -> dict:
    """Run the source's extractor, replacing what the local store holds for it."""
    module = SOURCES[source_id]["module"]
    n_chats, n_sessions, n_msgs = module.write_sessions(db_path=resolve(source_id, db_path))
    return {"chats": n_chats, "sessions": n_sessions, "messages": n_msgs, "file": str(store.path(source_id))}
