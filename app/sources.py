"""Local message sources (iMessage, WhatsApp): check a database is readable and run its extractor."""

import sqlite3
from pathlib import Path

import imessage_extract
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


class SourceError(Exception):
    """A problem the user can fix, with a message safe to show in the UI."""


def resolve(source_id: str, db_path: str = "") -> Path:
    """The user's path with ~ expanded, or the extractor's default location when left blank."""
    raw = (db_path or "").strip()
    return Path(raw).expanduser() if raw else SOURCES[source_id]["module"].DB_PATH


def check(source_id: str, db_path: str = "") -> dict:
    """Open the database read-only and count its messages. Raises SourceError if it can't be read."""
    path = resolve(source_id, db_path)
    try:
        path.stat()  # Path.exists() hides permission errors, which we need to tell apart from a missing file
    except FileNotFoundError:
        raise SourceError(f"No database found at {path}. Is the app installed and signed in on this Mac?")
    except PermissionError:
        raise SourceError(ACCESS_HELP.format(path=path))

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            count = conn.execute(SOURCES[source_id]["count_sql"]).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.OperationalError as err:
        msg = str(err)
        if "no such table" in msg or "not a database" in msg:
            raise SourceError(f"{path} doesn't look like the right database.")
        if "unable to open" in msg or "authorization denied" in msg:
            raise SourceError(ACCESS_HELP.format(path=path))
        raise SourceError(f"Couldn't read {path}: {msg}")
    except sqlite3.DatabaseError:
        raise SourceError(f"{path} doesn't look like the right database.")

    return {"db_path": str(path), "messages": count}


def extract(source_id: str, db_path: str = "") -> dict:
    """Run the source's extractor and write its sessions file under data/."""
    module = SOURCES[source_id]["module"]
    n_chats, n_sessions, n_msgs = module.write_sessions(out_path=module.OUT_PATH, db_path=resolve(source_id, db_path))
    return {"chats": n_chats, "sessions": n_sessions, "messages": n_msgs, "file": str(module.OUT_PATH)}
