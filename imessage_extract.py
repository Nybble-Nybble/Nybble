import json
import sqlite3
import sys
from itertools import groupby
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / "Library/Messages/chat.db"
OUT_PATH = Path(__file__).parent / "data/sessions.jsonl"

# Apple epoch (2001-01-01) to Unix epoch offset, in seconds.
APPLE_EPOCH_OFFSET = 978307200

# A gap longer than this between consecutive messages starts a new session.
SESSION_GAP_SECONDS = 12 * 60 * 60

QUERY = """
SELECT
    c.ROWID                                   AS chat_id,
    c.chat_identifier                         AS chat_identifier,
    c.style                                   AS chat_style,        -- 45 = 1:1, 43 = group
    m.ROWID                                   AS message_id,
    m.date / 1000000000 + :epoch              AS sent_at,           -- unix seconds
    m.is_from_me                              AS is_from_me,
    h.id                                      AS sender,            -- NULL for messages you sent
    m.text                                    AS text,
    m.attributedBody                          AS attributed_body,
    m.cache_has_attachments                   AS has_attachments
FROM message m
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c                ON c.ROWID = cmj.chat_id
LEFT JOIN handle h         ON h.ROWID = m.handle_id
WHERE m.associated_message_type = 0           -- drop tapbacks / reactions
  AND (:chat IS NULL OR c.chat_identifier = :chat)
ORDER BY c.ROWID, m.date
"""


def fetch_messages(db_path: Path = DB_PATH, chat: Optional[str] = None):
    """Yield every real message, grouped by chat and ordered by time. `chat` limits to one chat_identifier."""
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


ATTACHMENT_PLACEHOLDER = "\ufffc"


def body(row) -> str:
    """Plain message text: the `text` column, or the NSString pulled out of the typedstream blob."""
    if row["text"]:
        return row["text"]
    blob = row["attributed_body"]
    if not blob:
        return ""
    i = blob.find(b"NSString")
    if i < 0:
        return ""
    i += len(b"NSString") + 5          # skip the 0x01 0x94 0x84 0x01 0x2b type header
    length = blob[i]
    i += 1
    if length == 0x81:                  # two-byte little-endian length follows
        length = int.from_bytes(blob[i:i + 2], "little")
        i += 2
    elif length == 0x82:                # four-byte
        length = int.from_bytes(blob[i:i + 4], "little")
        i += 4
    return blob[i:i + length].decode("utf-8", errors="replace")


def clean(rows):
    """Parse each row's body and drop system events, empty bodies and attachment-only messages."""
    for row in rows:
        if not row["is_from_me"] and not row["sender"]:
            continue                    # system event: renamed group, left chat, etc.
        text = body(row).replace(ATTACHMENT_PLACEHOLDER, "").strip()
        if not text:
            continue
        me = bool(row["is_from_me"])
        # handle_id points at the other party even on sent messages, so never report a sender for your own turns
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
                "group": rows[0]["chat_style"] == 43,
            }
            for msgs in sessions(clean(rows)):
                record = {**header, "start": msgs[0]["t"], "end": msgs[-1]["t"], "messages": msgs}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_sessions += 1
                n_msgs += len(msgs)
    return n_chats, n_sessions, n_msgs


if __name__ == "__main__":
    chat = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. +14155550123 or an iMessage email
    n_chats, n_sessions, n_msgs = write_sessions(chat=chat)
    print(f"{n_msgs} messages in {n_sessions} sessions across {n_chats} chats -> {OUT_PATH}")
