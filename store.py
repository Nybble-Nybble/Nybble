"""Local SQLite store for extracted messages: one database per source, data/<source>.db.

The extractors write into it and the Your data page reads from it, so nothing has to be held in memory.
A source is always written to a new file that replaces the old one only once it is complete, so a failed
extraction leaves the previous copy untouched. Disconnecting deletes the file outright, which leaves no
deleted rows lying around inside a database.
"""

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

DATA_DIR = Path(__file__).parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id  INTEGER PRIMARY KEY,    -- the chat's id in the app's own database
    chat     TEXT    NOT NULL,       -- phone number, email or group id
    is_group INTEGER NOT NULL,
    -- Totals worked out at write time so listing chats never scans messages.
    sessions INTEGER NOT NULL,
    messages INTEGER NOT NULL,
    mine     INTEGER NOT NULL,       -- messages you sent
    people   INTEGER NOT NULL,       -- everyone who wrote, plus you
    first    INTEGER NOT NULL,       -- unix seconds
    last     INTEGER NOT NULL,
    preview  TEXT    NOT NULL        -- start of the newest message
);
CREATE TABLE IF NOT EXISTS sessions (
    id      INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    seq     INTEGER NOT NULL,        -- 0 = the chat's oldest session
    start   INTEGER NOT NULL,
    end     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_chat ON sessions (chat_id, seq);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,  -- insertion order = time order within a session
    session_id INTEGER NOT NULL,
    chat_id    INTEGER NOT NULL,
    t          INTEGER NOT NULL,
    me         INTEGER NOT NULL,
    sender     TEXT,                 -- NULL for your own messages
    text       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages (session_id);
CREATE INDEX IF NOT EXISTS messages_by_time ON messages (t);

-- Substring search index over message text. Built in one pass after a full write (fast);
-- the triggers keep it in sync for the occasional single-chat update.
-- Where the data came from, so Update data re-reads the same database after a server restart.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,          -- 'db_path': the app database this was extracted from
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text, content='messages', content_rowid='id', tokenize='trigram'
);
"""

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def path(source: str) -> Path:
    return DATA_DIR / f"{source}.db"


def has(source: str) -> bool:
    return path(source).exists()


def open_readonly(db_path) -> sqlite3.Connection:
    """Open any SQLite file strictly read-only.

    The path goes into a file: URI, so it is percent-encoded first. Unencoded, a ? or # in a folder or
    file name would end the path early and could replace mode=ro with options of its own.
    """
    return sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)


def connect(source: str) -> sqlite3.Connection:
    """Open a source's database for reading."""
    conn = open_readonly(path(source))
    conn.row_factory = sqlite3.Row
    return conn


def origin(source: str) -> Optional[str]:
    """The app database a source was last extracted from, or None if that isn't recorded."""
    if not has(source):
        return None
    try:
        conn = connect(source)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'db_path'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:  # deleted in the meantime, or written before the meta table existed
        return None
    return row[0] if row else None


def clear(source: str) -> None:
    """Delete everything stored for a source."""
    path(source).unlink(missing_ok=True)


def save(source: str, chats, whole_source: bool = True, db_path=None) -> tuple:
    """Write (header, sessions) pairs for one source and return (chats, sessions, messages) written.

    header is {"chat_id", "chat", "group"}; sessions is a list of message lists, each message
    {"t", "me", "from", "text"}. With whole_source the source's data is replaced; otherwise only the
    chats passed in are, and the rest is kept. Chats with no sessions are skipped. db_path, the app
    database being read, is recorded so later updates read the same one (see origin()).
    """
    final = path(source)
    DATA_DIR.mkdir(exist_ok=True)
    # A temp file unique to this run, so two extractions of the same source at once (Connect all during
    # Update data, two tabs) never write into each other's file; whichever finishes last wins, complete.
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, prefix=f".{source}-", suffix=".db.tmp")
    os.close(fd)
    tmp = Path(tmp)
    if not whole_source and final.exists():
        shutil.copyfile(final, tmp)

    n_chats = n_sessions = n_msgs = 0
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        if not whole_source:
            conn.executescript(TRIGGERS)
        with conn:
            for header, sessions in chats:
                chat_id = header["chat_id"]
                if not whole_source:
                    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                    conn.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
                    conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
                if not sessions:
                    continue
                msgs = [m for s in sessions for m in s]
                conn.execute(
                    "INSERT INTO chats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (chat_id, header["chat"], header["group"], len(sessions), len(msgs),
                     sum(1 for m in msgs if m["me"]), len({m["from"] for m in msgs if m["from"]}) + 1,
                     sessions[0][0]["t"], sessions[-1][-1]["t"], msgs[-1]["text"][:90]),
                )
                for seq, s in enumerate(sessions):
                    sid = conn.execute(
                        "INSERT INTO sessions (chat_id, seq, start, end) VALUES (?, ?, ?, ?)",
                        (chat_id, seq, s[0]["t"], s[-1]["t"]),
                    ).lastrowid
                    conn.executemany(
                        "INSERT INTO messages (session_id, chat_id, t, me, sender, text) VALUES (?, ?, ?, ?, ?, ?)",
                        [(sid, chat_id, m["t"], m["me"], m["from"], m["text"]) for m in s],
                    )
                n_chats += 1
                n_sessions += len(sessions)
                n_msgs += len(msgs)
            if db_path is not None:
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('db_path', ?)", (str(db_path),))
            if whole_source:
                conn.execute("INSERT INTO messages_fts (messages_fts) VALUES ('rebuild')")
                conn.executescript(TRIGGERS)
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp, final)  # atomic: readers see the old file or the new one, never a half-written one
    return n_chats, n_sessions, n_msgs
