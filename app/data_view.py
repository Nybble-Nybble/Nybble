"""Read the extracted messages back for the "Your data" page. Everything stays on this machine.

Queries go straight to each source's SQLite database (store.py), so nothing is cached in memory and
each request only reads the rows it returns: a chat is read a window of sessions at a time, never whole.
"""

import sqlite3
from typing import Optional

import store
from app.sources import SOURCES

PAGE = 60          # sessions sent when a chat is opened, and per "Show earlier/later"
MAX_WINDOW = 200   # most sessions one request can ask for


def _query(source_id: str, sql: str, args=()) -> list:
    if not store.has(source_id):
        return []
    try:
        conn = store.connect(source_id)
    except sqlite3.OperationalError:
        if not store.has(source_id):  # a Disconnect deleted it between the two calls: same as no data
            return []
        raise
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def summary() -> list[dict]:
    """One row per chat across all sources, most recently active first."""
    rows = [
        {"source": source_id, "chat_id": r["chat_id"], "chat": r["chat"], "group": bool(r["is_group"]),
         "sessions": r["sessions"], "messages": r["messages"], "mine": r["mine"], "people": r["people"],
         "first": r["first"], "last": r["last"], "preview": r["preview"]}
        for source_id in SOURCES
        for r in _query(source_id, "SELECT * FROM chats")
    ]
    rows.sort(key=lambda r: r["last"], reverse=True)
    return rows


def _message(r) -> dict:
    return {"t": r["t"], "me": bool(r["me"]), "from": r["sender"], "text": r["text"]}


def chat(source_id: str, chat_id: int, start: Optional[int] = None, end: Optional[int] = None) -> Optional[dict]:
    """Sessions [start, end) of one chat (0 = oldest). By default the newest PAGE.

    The window is clamped to the chat and to MAX_WINDOW sessions; the reply says which one was sent,
    and how many sessions the chat has in all.
    """
    if source_id not in SOURCES:
        return None
    head = _query(source_id, "SELECT chat, is_group, sessions FROM chats WHERE chat_id = ?", (chat_id,))
    if not head:
        return None
    total = head[0]["sessions"]
    end = total if end is None else min(end, total)
    start = max(0, end - PAGE) if start is None else min(start, end)
    end = min(end, start + MAX_WINDOW)
    rows = _query(
        source_id,
        """SELECT s.seq, s.start, s.end, m.t, m.me, m.sender, m.text
           FROM sessions s JOIN messages m ON m.session_id = s.id
           WHERE s.chat_id = ? AND s.seq >= ? AND s.seq < ?
           ORDER BY s.seq, m.id""",
        (chat_id, start, end),
    )
    sessions = []
    for r in rows:
        if not sessions or sessions[-1]["seq"] != r["seq"]:
            sessions.append({"seq": r["seq"], "start": r["start"], "end": r["end"], "messages": []})
        sessions[-1]["messages"].append(_message(r))
    return {"chat": head[0]["chat"], "group": bool(head[0]["is_group"]),
            "total": total, "start": start, "end": end, "sessions": sessions}


SEARCH = """SELECT m.chat_id, c.chat, s.seq, m.t, m.me, m.sender, m.text
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            JOIN chats c ON c.chat_id = m.chat_id"""


def search(q: str, limit: int = 200) -> list[dict]:
    """Case-insensitive substring search over every message, newest first."""
    q = q.strip()
    if not q:
        return []
    if len(q) >= 3:
        # The trigram index matches any substring of 3+ characters; quote it so it's taken literally.
        sql = SEARCH + " WHERE m.id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?) ORDER BY m.t DESC LIMIT ?"
        args = ('"' + q.replace('"', '""') + '"', limit)
    else:
        # Too short for the index: scan. Cheap enough, since 1-2 characters is rarely a real search.
        sql = SEARCH + " WHERE instr(lower(m.text), lower(?)) ORDER BY m.t DESC LIMIT ?"
        args = (q, limit)
    hits = [
        {"source": source_id, "chat_id": r["chat_id"], "chat": r["chat"], "session": r["seq"], **_message(r)}
        for source_id in SOURCES
        for r in _query(source_id, sql, args)
    ]
    hits.sort(key=lambda h: h["t"], reverse=True)
    return hits[:limit]
