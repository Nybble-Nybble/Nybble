import sqlite3
import sys
from itertools import groupby
from pathlib import Path
from typing import Optional

import store

DB_PATH = Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"

# Core Data epoch (2001-01-01) to Unix epoch offset, in seconds.
APPLE_EPOCH_OFFSET = 978307200

# A gap longer than this between consecutive messages starts a new session.
SESSION_GAP_SECONDS = 12 * 60 * 60

# ZMESSAGETYPE values that carry a text body: 0 = plain text, 7 = text with a link preview.
TEXT_MESSAGE_TYPES = (0, 7)

QUERY = """
SELECT
    s.Z_PK                                    AS chat_id,
    s.ZCONTACTJID                             AS chat_identifier,   -- 1425...@s.whatsapp.net or ...@g.us
    s.ZSESSIONTYPE                            AS session_type,      -- 0 = 1:1, 1 = group
    m.Z_PK                                    AS message_id,
    CAST(m.ZMESSAGEDATE AS INTEGER) + :epoch  AS sent_at,           -- unix seconds
    m.ZISFROMME                               AS is_from_me,
    COALESCE(gm.ZMEMBERJID, m.ZFROMJID)       AS sender,            -- group member JID, else contact JID; NULL when sent by you
    m.ZTEXT                                   AS text,
    m.ZMESSAGETYPE                            AS message_type
FROM ZWAMESSAGE m
JOIN ZWACHATSESSION s      ON s.Z_PK = m.ZCHATSESSION
LEFT JOIN ZWAGROUPMEMBER gm ON gm.Z_PK = m.ZGROUPMEMBER   -- in groups ZFROMJID is the group itself; the member row has the real sender
WHERE m.ZMESSAGETYPE IN (0, 7)                -- text only: drops media, calls, stickers, system events
  AND (:chat IS NULL OR s.ZCONTACTJID = :chat)
ORDER BY s.Z_PK, m.ZMESSAGEDATE
"""


def to_jid(chat: str) -> str:
    """Accept a phone number like +14255550123 or a full JID; return the JID WhatsApp stores."""
    chat = chat.strip()
    if "@" in chat:
        return chat
    return "".join(ch for ch in chat if ch.isdigit()) + "@s.whatsapp.net"


def fetch_messages(db_path: Path = DB_PATH, chat: Optional[str] = None):
    """Yield every text message, grouped by chat and ordered by time. `chat` limits to one JID."""
    conn = store.open_readonly(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield from conn.execute(QUERY, {"epoch": APPLE_EPOCH_OFFSET, "chat": chat})
    finally:
        conn.close()


def chats(rows):
    """Group the ordered row stream into (chat_id, [rows]) one chat at a time."""
    for chat_id, group in groupby(rows, key=lambda r: r["chat_id"]):
        yield chat_id, list(group)


def clean(rows):
    """Drop empty bodies and shape each row for output. Media, calls and system events were already dropped in SQL."""
    for row in rows:
        text = (row["text"] or "").strip()
        if not text:
            continue
        me = bool(row["is_from_me"])
        yield {"t": row["sent_at"], "me": me, "from": None if me else row["sender"], "text": text}


def sessions(rows, gap=SESSION_GAP_SECONDS):
    """Split one chat's time-ordered rows wherever the gap between messages exceeds `gap`."""
    current = []
    for row in rows:
        if current and row["t"] - current[-1]["t"] > gap:
            yield current
            current = []
        current.append(row)
    if current:
        yield current


def write_sessions(db_path: Path = DB_PATH, chat: Optional[str] = None):
    """Extract every chat (or just `chat`) into the local store (store.py). Returns (chats, sessions, messages)."""
    def records():
        for chat_id, rows in chats(fetch_messages(db_path, chat)):
            header = {
                "chat_id": chat_id,
                "chat": rows[0]["chat_identifier"],
                "group": rows[0]["session_type"] != 0,   # 0 = 1:1; groups, broadcasts and communities are all multi-party
            }
            yield header, list(sessions(clean(rows)))

    return store.save("whatsapp", records(), whole_source=chat is None, db_path=db_path)


if __name__ == "__main__":
    chat = to_jid(sys.argv[1]) if len(sys.argv) > 1 else None   # e.g. +14255550123 or a full JID
    n_chats, n_sessions, n_msgs = write_sessions(chat=chat)
    print(f"{n_msgs} messages in {n_sessions} sessions across {n_chats} chats -> {store.path('whatsapp')}")
