import json
import sqlite3
import sys
from itertools import groupby
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"
OUT_PATH = Path(__file__).parent / "data/whatsapp_sessions.jsonl"

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
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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


def write_sessions(out_path: Path = OUT_PATH, db_path: Path = DB_PATH, chat: Optional[str] = None):
    """Stream every chat (or just `chat`) through clean + sessions and write one JSON object per session."""
    n_chats = n_sessions = n_msgs = 0
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chat_id, rows in chats(fetch_messages(db_path, chat)):
            n_chats += 1
            header = {
                "chat_id": chat_id,
                "chat": rows[0]["chat_identifier"],
                "group": rows[0]["session_type"] != 0,   # 0 = 1:1; groups, broadcasts and communities are all multi-party
            }
            for msgs in sessions(clean(rows)):
                record = {**header, "start": msgs[0]["t"], "end": msgs[-1]["t"], "messages": msgs}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_sessions += 1
                n_msgs += len(msgs)
    return n_chats, n_sessions, n_msgs


if __name__ == "__main__":
    chat = to_jid(sys.argv[1]) if len(sys.argv) > 1 else None   # e.g. +14255550123 or a full JID
    n_chats, n_sessions, n_msgs = write_sessions(chat=chat)
    print(f"{n_msgs} messages in {n_sessions} sessions across {n_chats} chats -> {OUT_PATH}")
