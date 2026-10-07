"""Read the extracted sessions files back for the "Your data" page. Everything stays on this machine."""

import json

from app.sources import SOURCES

# source id -> (file mtime, {chat_id: {"chat", "group", "sessions"}}); reloaded whenever the file changes.
_cache: dict[str, tuple[float, dict]] = {}


def _load(source_id: str) -> dict:
    path = SOURCES[source_id]["module"].OUT_PATH
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    hit = _cache.get(source_id)
    if hit and hit[0] == mtime:
        return hit[1]
    chats: dict = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            c = chats.setdefault(s["chat_id"], {"chat": s["chat"], "group": s["group"], "sessions": []})
            c["sessions"].append(s)
    _cache[source_id] = (mtime, chats)
    return chats


def summary() -> list[dict]:
    """One row per chat across all sources, most recently active first."""
    rows = []
    for source_id in SOURCES:
        for chat_id, c in _load(source_id).items():
            msgs = [m for s in c["sessions"] for m in s["messages"]]
            rows.append({
                "source": source_id,
                "chat_id": chat_id,
                "chat": c["chat"],
                "group": c["group"],
                "sessions": len(c["sessions"]),
                "messages": len(msgs),
                "mine": sum(1 for m in msgs if m["me"]),
                "people": len({m["from"] for m in msgs if m["from"]}) + 1,  # everyone who wrote, plus you
                "first": c["sessions"][0]["start"],
                "last": c["sessions"][-1]["end"],
                "preview": msgs[-1]["text"][:90],
            })
    rows.sort(key=lambda r: r["last"], reverse=True)
    return rows


def chat(source_id: str, chat_id: int) -> dict | None:
    if source_id not in SOURCES:
        return None
    return _load(source_id).get(chat_id)


def search(q: str, limit: int = 200) -> list[dict]:
    """Case-insensitive substring search over every message, newest first."""
    q = q.strip().lower()
    if not q:
        return []
    hits = []
    for source_id in SOURCES:
        for chat_id, c in _load(source_id).items():
            for i, s in enumerate(c["sessions"]):
                for m in s["messages"]:
                    if q in m["text"].lower():
                        hits.append({"source": source_id, "chat_id": chat_id, "chat": c["chat"],
                                     "session": i, **m})
    hits.sort(key=lambda h: h["t"], reverse=True)
    return hits[:limit]
