"""Parsers for platform data exports (GDPR / "download your data" bundles).

Most social and messaging data is not readable on disk — it has to be requested
from the platform. This module discovers those downloaded bundles and parses
them, working identically whether the user extracted the .zip or not.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .. import platform_paths as pp
from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import from_unix, parse_date, relpath

# ===========================================================================
# Archive discovery
# ===========================================================================

# Where people leave exports. ~/.nybble/archives is the canonical drop folder.
def _search_roots() -> List[Path]:
    roots = pp.downloads_dir() + pp.documents_dirs()
    roots += pp.existing(Path.home() / ".nybble" / "archives",
                         Path(os.environ.get("NYBBLE_ARCHIVES", "")) or None)
    seen, out = set(), []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out


# Filename fragments that identify each service's export bundle.
ARCHIVE_PATTERNS: Dict[str, Sequence[str]] = {
    "twitter": ("twitter-", "twitter_", "x-archive", "tweets.js", "twitter archive"),
    "reddit": ("reddit", "export_reddit"),
    "facebook": ("facebook-", "facebook_", "your_facebook_activity"),
    "meta": ("meta-", "meta_"),
    "instagram": ("instagram-", "instagram_"),
    "tiktok": ("tiktok", "user_data_tiktok", "user_data.json"),
    "linkedin": ("linkedin", "basic_linkedin"),
    "discord": ("discord", "package.zip"),
    "slack": ("slack", "-slack-export"),
    "telegram": ("telegram", "chat_export", "result.json"),
    "signal": ("signal-",),
    "whatsapp": ("whatsapp", "_chat.txt"),
    "sms": ("sms-", "sms_backup", "calls-"),
    "smsbackup": ("sms-",),
    "takeout": ("takeout", "google-"),
    "google": ("takeout", "google-data"),
    "youtube": ("youtube", "takeout"),
    "spotify": ("spotify", "my_spotify_data", "streaminghistory"),
    "netflix": ("netflix",),
    "amazon": ("amazon", "your orders"),
    "goodreads": ("goodreads",),
    "letterboxd": ("letterboxd",),
    "imdb": ("imdb", "ratings.csv"),
    "readwise": ("readwise",),
    "pocket": ("pocket", "ril_export", "getpocket"),
    "instapaper": ("instapaper",),
    "matter": ("matter",),
    "omnivore": ("omnivore",),
    "feedly": ("feedly",),
    "opml": (".opml",),
    "inoreader": ("inoreader",),
    "newsblur": ("newsblur",),
    "overcast": ("overcast",),
    "pocketcasts": ("pocketcasts", "pocket_casts"),
    "podcast": ("podcast",),
    "github": ("github", "_export"),
    "notion": ("notion", "export-"),
    "evernote": ("evernote",),
    "enex": (".enex",),
    "roam": ("roam", "roam-export"),
    "substack": ("substack",),
    "medium": ("medium-export", "medium_"),
    "wordpress": ("wordpress", "wp-export"),
    "ghost": ("ghost-",),
    "mastodon": ("mastodon", "archive-"),
    "bluesky": ("bluesky", ".car"),
    "outbox": ("outbox.json",),
    "hackernews": ("hackernews", "hn-"),
    "hn": ("hn-",),
    "stackexchange": ("stackexchange", "stack-exchange"),
    "stackoverflow": ("stackoverflow",),
    "mail": ("mail", ".mbox"),
    "mbox": (".mbox",),
}


def find_archives(names: Sequence[str], max_depth: int = 2) -> List[Path]:
    """Locate downloaded export bundles matching any of `names`.

    Matches zip files and extracted directories, shallowly — exports live in
    Downloads, not fifteen levels into a project tree.
    """
    patterns: List[str] = []
    for name in names:
        patterns.extend(ARCHIVE_PATTERNS.get(name, (name,)))
    patterns = [p.lower() for p in patterns]

    found: List[Path] = []
    for root in _search_roots():
        for depth in range(max_depth + 1):
            glob = "/".join(["*"] * depth) + ("/*" if depth else "*")
            try:
                entries = list(root.glob(glob.lstrip("/") or "*"))
            except (OSError, ValueError):
                continue
            for entry in entries:
                lowered = entry.name.lower()
                if not any(p in lowered for p in patterns):
                    continue
                if entry.is_dir() or entry.suffix.lower() in (
                        ".zip", ".json", ".js", ".csv", ".xml", ".enex", ".opml",
                        ".txt", ".mbox"):
                    found.append(entry)
    # Dedupe, preferring shorter paths (the archive root over a file inside it).
    seen, out = set(), []
    for p in sorted(found, key=lambda x: len(str(x))):
        key = str(p)
        if not any(key.startswith(s + os.sep) for s in seen):
            seen.add(key)
            out.append(p)
    return out[:50]


# ===========================================================================
# Uniform access to a zip or a directory
# ===========================================================================

class ArchiveFS:
    """Read files out of an export bundle, zipped or extracted."""

    MAX_MEMBER_BYTES = 256 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = Path(root)
        self._zip: Optional[zipfile.ZipFile] = None
        if self.root.is_file() and self.root.suffix.lower() == ".zip":
            try:
                self._zip = zipfile.ZipFile(self.root)
            except (zipfile.BadZipFile, OSError):
                self._zip = None

    def __enter__(self) -> "ArchiveFS":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            try:
                self._zip.close()
            except OSError:
                pass
            self._zip = None

    @property
    def is_single_file(self) -> bool:
        return self._zip is None and self.root.is_file()

    def names(self) -> List[str]:
        if self._zip is not None:
            return [n for n in self._zip.namelist() if not n.endswith("/")]
        if self.root.is_file():
            return [self.root.name]
        out = []
        for p in self.root.rglob("*"):
            if p.is_file():
                try:
                    out.append(str(p.relative_to(self.root)))
                except ValueError:
                    continue
                if len(out) > 200_000:
                    break
        return out

    def find(self, *fragments: str, suffix: Optional[str] = None) -> List[str]:
        """Member names containing any fragment (case-insensitive)."""
        frags = [f.lower() for f in fragments]
        out = []
        for name in self.names():
            lowered = name.lower()
            if suffix and not lowered.endswith(suffix):
                continue
            if not frags or any(f in lowered for f in frags):
                out.append(name)
        return out

    def read(self, name: str) -> Optional[bytes]:
        try:
            if self._zip is not None:
                info = self._zip.getinfo(name)
                if info.file_size > self.MAX_MEMBER_BYTES:
                    return None
                return self._zip.read(name)
            path = self.root if self.is_single_file else self.root / name
            if path.stat().st_size > self.MAX_MEMBER_BYTES:
                return None
            return path.read_bytes()
        except (KeyError, OSError, zipfile.BadZipFile, ValueError):
            return None

    def read_text(self, name: str) -> Optional[str]:
        raw = self.read(name)
        return tx.decode(raw) if raw else None

    def read_json(self, name: str) -> Optional[Any]:
        text = self.read_text(name)
        if not text:
            return None
        text = text.strip()
        # Twitter and some others wrap JSON in a JS assignment.
        if text.startswith(("window.", "var ")) or re.match(r"^\w[\w.]*\s*=", text):
            brace = min((i for i in (text.find("["), text.find("{")) if i != -1),
                        default=-1)
            if brace == -1:
                return None
            text = text[brace:].rstrip().rstrip(";")
        try:
            return json.loads(text)
        except ValueError:
            return None

    def read_csv(self, name: str) -> List[Dict[str, str]]:
        text = self.read_text(name)
        if not text:
            return []
        try:
            return list(csv.DictReader(io.StringIO(text)))
        except (csv.Error, ValueError):
            return []


def _each(paths: Sequence[Path]) -> Iterator[ArchiveFS]:
    for p in paths:
        fs = ArchiveFS(Path(p))
        try:
            yield fs
        finally:
            fs.close()


def _walk_values(node: Any, key: str) -> Iterator[Any]:
    """Yield every value stored under `key` anywhere in a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            else:
                yield from _walk_values(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_values(item, key)


def _fix_meta_mojibake(text: str) -> str:
    """Meta exports double-encode UTF-8 as latin-1; undo it."""
    if not text or not any(ch in text for ch in ("Ã", "â", "ð")):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# ===========================================================================
# X / Twitter
# ===========================================================================

def collect_twitter(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Tweets (self), likes and bookmarks (taste), and inferred interests."""
    for fs in _each(paths):
        for name in fs.find("tweets.js", "tweet.js", "tweets.json"):
            data = fs.read_json(name)
            for item in _as_list(data):
                tweet = item.get("tweet", item) if isinstance(item, dict) else None
                if not isinstance(tweet, dict):
                    continue
                text = tx.clean(tweet.get("full_text") or tweet.get("text") or "")
                if not text:
                    continue
                created = parse_date(tweet.get("created_at"))
                if ctx.since and (created or "") < ctx.since:
                    continue
                is_reply = bool(tweet.get("in_reply_to_status_id_str"))
                yield Record(
                    source="twitter_archive",
                    kind="comment" if is_reply else "post",
                    authorship="self",
                    doc_category="social_preferences",
                    sensitivity="medium",
                    text=text,
                    created_at=created,
                    author="me",
                    url=f"https://x.com/i/status/{tweet.get('id_str')}"
                        if tweet.get("id_str") else None,
                    app="X",
                    path=relpath(fs.root),
                    thread_id=tweet.get("in_reply_to_status_id_str"),
                    signals={
                        "favorite_count": _int(tweet.get("favorite_count")),
                        "retweet_count": _int(tweet.get("retweet_count")),
                    },
                    meta={"reply": is_reply},
                )

        # Likes: other people's writing the user endorsed.
        for name in fs.find("like.js", "likes.js", "like.json"):
            for item in _as_list(fs.read_json(name)):
                like = item.get("like", item) if isinstance(item, dict) else None
                if not isinstance(like, dict):
                    continue
                text = tx.clean(like.get("fullText") or like.get("full_text") or "")
                if not text:
                    continue
                yield Record(
                    source="twitter_archive", kind="reaction", authorship="other",
                    doc_category="liked_writing", sensitivity="medium",
                    text=text, app="X", path=relpath(fs.root),
                    url=like.get("expandedUrl"),
                    signals={"liked": True},
                )

        for name in fs.find("bookmark"):
            for item in _as_list(fs.read_json(name)):
                bm = item.get("bookmark", item) if isinstance(item, dict) else None
                if not isinstance(bm, dict):
                    continue
                text = tx.clean(bm.get("fullText") or bm.get("full_text") or "")
                if not text:
                    continue
                yield Record(
                    source="twitter_archive", kind="bookmark", authorship="other",
                    doc_category="liked_writing", sensitivity="medium",
                    text=text, app="X", path=relpath(fs.root),
                    signals={"bookmarked": True, "saved": True},
                )

        # The interest tags X assigned — doc §6, stated literally by the platform.
        for name in fs.find("personalization"):
            data = fs.read_json(name)
            interests = []
            for block in _walk_values(data, "interests"):
                for entry in _as_list(block):
                    label = entry.get("name") if isinstance(entry, dict) else entry
                    if isinstance(label, str) and label.strip():
                        interests.append(label.strip())
            if interests:
                yield Record(
                    source="twitter_archive", kind="preference", authorship="system",
                    doc_category="social_preferences", sensitivity="medium",
                    text="Interests X inferred: " + ", ".join(sorted(set(interests))[:300]),
                    app="X", path=relpath(fs.root),
                    meta={"interests": sorted(set(interests))[:300],
                          "platform_declared": True},
                )


def _as_list(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "results", "entries"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ===========================================================================
# Reddit
# ===========================================================================

def collect_reddit(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        # Comments and posts the user wrote.
        for name in fs.find("comments", "posts", suffix=".csv"):
            lowered = name.lower()
            if "saved" in lowered or "upvot" in lowered or "hidden" in lowered:
                continue
            is_comment = "comment" in lowered
            for row in fs.read_csv(name):
                text = tx.clean(row.get("body") or row.get("text") or
                                row.get("title") or "")
                if len(text) < 15:
                    continue
                created = parse_date(row.get("date") or row.get("created"))
                if ctx.since and (created or "") < ctx.since:
                    continue
                yield Record(
                    source="reddit_export",
                    kind="comment" if is_comment else "post",
                    authorship="self",
                    doc_category="social_preferences", sensitivity="medium",
                    text=text, created_at=created, author="me",
                    url=row.get("permalink") or row.get("url"),
                    app="Reddit", path=relpath(fs.root),
                    thread_id=row.get("parent") or row.get("subreddit"),
                    meta={"subreddit": row.get("subreddit")},
                )

        # Saved and upvoted: other people's content the user endorsed.
        for name in fs.find("saved", "upvot", suffix=".csv"):
            lowered = name.lower()
            signal = "saved" if "saved" in lowered else "upvoted"
            for row in fs.read_csv(name):
                text = tx.clean(row.get("title") or row.get("body") or "")
                permalink = row.get("permalink") or row.get("url")
                if not text and not permalink:
                    continue
                yield Record(
                    source="reddit_export",
                    kind="bookmark" if signal == "saved" else "reaction",
                    authorship="other",
                    doc_category="liked_content", sensitivity="medium",
                    text=text or permalink,
                    created_at=parse_date(row.get("date")),
                    url=permalink, app="Reddit", path=relpath(fs.root),
                    signals={signal: True},
                    meta={"subreddit": row.get("subreddit")},
                )

        # Subscriptions are a declared-preference list.
        for name in fs.find("subscribed_subreddits", "subscriptions", suffix=".csv"):
            subs = [r.get("subreddit") for r in fs.read_csv(name) if r.get("subreddit")]
            if subs:
                yield Record(
                    source="reddit_export", kind="preference", authorship="system",
                    doc_category="social_preferences", sensitivity="low",
                    text="Subreddits followed: " + ", ".join(subs[:500]),
                    app="Reddit", path=relpath(fs.root),
                    meta={"subreddits": subs[:500], "platform_declared": True},
                )


# ===========================================================================
# Meta (Facebook / Instagram): messages and preferences
# ===========================================================================

def collect_meta_messages(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Messenger and Instagram DM threads."""
    for fs in _each(paths):
        source_id = ("instagram_dms" if "instagram" in str(fs.root).lower()
                     else "messenger_export")
        app = "Instagram" if source_id == "instagram_dms" else "Messenger"

        for name in fs.find("message_", "message.json", suffix=".json"):
            if "/inbox/" not in name.lower() and "messages" not in name.lower():
                continue
            data = fs.read_json(name)
            if not isinstance(data, dict):
                continue
            thread = data.get("title") or Path(name).parent.name
            participants = [p.get("name") for p in data.get("participants") or []
                            if isinstance(p, dict) and p.get("name")]
            # The archive owner is conventionally the last participant listed.
            me = participants[-1] if participants else None

            for msg in data.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                text = tx.clean(_fix_meta_mojibake(msg.get("content") or ""))
                if len(text) < 2:
                    continue
                sender = _fix_meta_mojibake(msg.get("sender_name") or "")
                when = from_unix(msg.get("timestamp_ms"), "ms")
                if ctx.since and (when or "") < ctx.since:
                    continue
                from_me = bool(me and sender == me)
                yield Record(
                    source=source_id, kind="message",
                    authorship="self" if from_me else "other",
                    doc_category="messages", sensitivity="high",
                    text=text, created_at=when,
                    author="me" if from_me else (sender or None),
                    thread_id=_fix_meta_mojibake(str(thread)),
                    app=app, path=relpath(fs.root),
                    meta={"participants": participants[:30],
                          "group": len(participants) > 2},
                )


def collect_meta_prefs(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Posts, comments, saved items, follows, and Meta's inferred ad interests."""
    for fs in _each(paths):
        instagram = "instagram" in str(fs.root).lower()
        source_id = "instagram_prefs" if instagram else "facebook_prefs"
        app = "Instagram" if instagram else "Facebook"

        # Authored posts and comments.
        for name in fs.find("your_posts", "posts_1", "comments", "post_comments",
                            suffix=".json"):
            data = fs.read_json(name)
            for item in _flatten_meta(data):
                text = tx.clean(_fix_meta_mojibake(item.get("text", "")))
                if len(text) < 10:
                    continue
                when = from_unix(item.get("timestamp"), "s")
                if ctx.since and (when or "") < ctx.since:
                    continue
                yield Record(
                    source=source_id,
                    kind="comment" if "comment" in name.lower() else "post",
                    authorship="self",
                    doc_category="social_preferences", sensitivity="medium",
                    text=text, created_at=when, author="me",
                    app=app, path=relpath(fs.root),
                )

        # Saved posts: other people's content kept deliberately.
        for name in fs.find("saved_posts", "saved_collections", suffix=".json"):
            data = fs.read_json(name)
            for item in _flatten_meta(data):
                label = item.get("text") or item.get("title") or ""
                if not label:
                    continue
                yield Record(
                    source=source_id, kind="bookmark", authorship="other",
                    doc_category="liked_content", sensitivity="medium",
                    text=tx.clean(_fix_meta_mojibake(label)),
                    created_at=from_unix(item.get("timestamp"), "s"),
                    url=item.get("href"), app=app, path=relpath(fs.root),
                    signals={"saved": True},
                )

        # Platform-declared interests — the literal doc §6 artifact.
        for name in fs.find("ads_interests", "your_topics", "ad_preferences",
                            "advertisers", "topics", suffix=".json"):
            data = fs.read_json(name)
            labels = set()
            for key in ("name", "topic", "value", "string_map_data", "title"):
                for val in _walk_values(data, key):
                    if isinstance(val, str) and 2 < len(val) < 80:
                        labels.add(val.strip())
            labels = sorted(labels)[:400]
            if labels:
                yield Record(
                    source=source_id, kind="preference", authorship="system",
                    doc_category="social_preferences", sensitivity="medium",
                    text=f"Topics {app} inferred: " + ", ".join(labels),
                    app=app, path=relpath(fs.root),
                    meta={"interests": labels, "platform_declared": True,
                          "file": Path(name).name},
                )

        # Accounts followed: a curated taste list.
        for name in fs.find("following", "followed_", suffix=".json"):
            data = fs.read_json(name)
            names = sorted({v.strip() for v in _walk_values(data, "value")
                            if isinstance(v, str) and 1 < len(v) < 80})[:500]
            if names:
                yield Record(
                    source=source_id, kind="preference", authorship="system",
                    doc_category="liked_content", sensitivity="low",
                    text=f"Accounts followed on {app}: " + ", ".join(names),
                    app=app, path=relpath(fs.root),
                    meta={"following": names},
                )


def _flatten_meta(data: Any) -> Iterator[Dict[str, Any]]:
    """Meta nests post text under data[].post or attachments; normalize it."""
    for item in _as_list(data):
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp") or item.get("timestamp_ms")
        if isinstance(timestamp, (int, float)) and timestamp > 1e11:
            timestamp = timestamp / 1000.0
        text = item.get("title") or ""
        for block in item.get("data") or []:
            if isinstance(block, dict) and block.get("post"):
                text = block["post"]
                break
        if not text:
            for val in _walk_values(item, "comment"):
                if isinstance(val, dict) and val.get("comment"):
                    text = val["comment"]
                    break
        if text:
            yield {"text": str(text), "timestamp": timestamp,
                   "href": item.get("uri") or item.get("href")}


# ===========================================================================
# TikTok
# ===========================================================================

def collect_tiktok(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Watch history, likes, favorites, and searches.

    Watch timestamps are the closest export-available analogue to the
    attention/scroll signal the plan doc wants from present-behavior capture.
    """
    for fs in _each(paths):
        for name in fs.find("user_data", ".json", suffix=".json"):
            data = fs.read_json(name)
            if not isinstance(data, dict):
                continue

            for list_key, kind, signal, category in (
                ("VideoList", "consumption", "watched", "liked_content"),
                ("ItemFavoriteList", "bookmark", "favorited", "liked_content"),
                ("Likes", "reaction", "liked", "liked_content"),
                ("FavoriteVideoList", "bookmark", "favorited", "liked_content"),
            ):
                for block in _walk_values(data, list_key):
                    for item in _as_list(block):
                        if not isinstance(item, dict):
                            continue
                        link = item.get("Link") or item.get("VideoLink") or ""
                        when = parse_date(item.get("Date") or item.get("date"))
                        if not link:
                            continue
                        if ctx.since and (when or "") < ctx.since:
                            continue
                        yield Record(
                            source="tiktok_export", kind=kind, authorship="other",
                            doc_category=category, sensitivity="medium",
                            text=link, url=link.strip(), created_at=when,
                            app="TikTok", path=relpath(fs.root),
                            signals={signal: True},
                        )

            # Searches are a direct statement of interest.
            terms = []
            for block in _walk_values(data, "SearchList"):
                for item in _as_list(block):
                    if isinstance(item, dict) and item.get("SearchTerm"):
                        terms.append(item["SearchTerm"])
            if terms:
                yield Record(
                    source="tiktok_export", kind="preference", authorship="self",
                    doc_category="social_preferences", sensitivity="medium",
                    text="TikTok searches: " + ", ".join(terms[:400]),
                    app="TikTok", path=relpath(fs.root),
                    meta={"searches": terms[:400]},
                )


# ===========================================================================
# Discord / Slack / Telegram / WhatsApp / SMS
# ===========================================================================

def collect_discord(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("messages.csv", suffix=".csv"):
            channel = Path(name).parent.name
            for row in fs.read_csv(name):
                text = tx.clean(row.get("Contents") or row.get("contents") or "")
                if len(text) < 2:
                    continue
                when = parse_date(row.get("Timestamp") or row.get("timestamp"))
                if ctx.since and (when or "") < ctx.since:
                    continue
                yield Record(
                    source="discord_export", kind="message", authorship="self",
                    doc_category="messages", sensitivity="high",
                    text=text, created_at=when, author="me",
                    thread_id=channel, app="Discord", path=relpath(fs.root),
                    meta={"channel_id": channel},
                )
        # Newer packages ship JSON instead of CSV.
        for name in fs.find("messages.json", suffix=".json"):
            channel = Path(name).parent.name
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                text = tx.clean(item.get("Contents") or item.get("content") or "")
                if len(text) < 2:
                    continue
                yield Record(
                    source="discord_export", kind="message", authorship="self",
                    doc_category="messages", sensitivity="high",
                    text=text,
                    created_at=parse_date(item.get("Timestamp") or item.get("timestamp")),
                    author="me", thread_id=channel, app="Discord",
                    path=relpath(fs.root),
                )


def collect_slack(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        users: Dict[str, str] = {}
        for name in fs.find("users.json", suffix=".json"):
            for u in _as_list(fs.read_json(name)):
                if isinstance(u, dict) and u.get("id"):
                    users[u["id"]] = (u.get("real_name") or u.get("name") or u["id"])

        for name in fs.names():
            if not re.search(r"/\d{4}-\d{2}-\d{2}\.json$", name):
                continue
            channel = Path(name).parent.name
            for msg in _as_list(fs.read_json(name)):
                if not isinstance(msg, dict) or msg.get("subtype"):
                    continue
                text = tx.clean(_expand_slack_mentions(msg.get("text") or "", users))
                if len(text) < 5:
                    continue
                when = from_unix(msg.get("ts"), "s")
                if ctx.since and (when or "") < ctx.since:
                    continue
                author = users.get(msg.get("user", ""), msg.get("user"))
                yield Record(
                    source="slack_export", kind="message", authorship="mixed",
                    doc_category="messages", sensitivity="high",
                    text=text, created_at=when, author=author,
                    thread_id=msg.get("thread_ts") or channel,
                    app="Slack", path=relpath(fs.root),
                    meta={"channel": channel},
                )


_SLACK_MENTION = re.compile(r"<@([A-Z0-9]+)>")
_SLACK_LINK = re.compile(r"<(https?://[^|>]+)(?:\|([^>]*))?>")


def _expand_slack_mentions(text: str, users: Dict[str, str]) -> str:
    text = _SLACK_MENTION.sub(lambda m: "@" + users.get(m.group(1), m.group(1)), text)
    return _SLACK_LINK.sub(lambda m: m.group(2) or m.group(1), text)


def collect_telegram(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("result.json", suffix=".json"):
            data = fs.read_json(name)
            if not isinstance(data, dict):
                continue
            me = str((data.get("personal_information") or {}).get("user_id", ""))
            chats = data.get("chats", {})
            chat_list = chats.get("list", []) if isinstance(chats, dict) else chats
            for chat in _as_list(chat_list):
                if not isinstance(chat, dict):
                    continue
                thread = chat.get("name") or str(chat.get("id"))
                for msg in chat.get("messages") or []:
                    if not isinstance(msg, dict) or msg.get("type") != "message":
                        continue
                    text = _telegram_text(msg.get("text"))
                    if len(text) < 2:
                        continue
                    when = parse_date(msg.get("date"))
                    if ctx.since and (when or "") < ctx.since:
                        continue
                    sender = msg.get("from") or ""
                    from_me = bool(me and str(msg.get("from_id", "")).endswith(me))
                    yield Record(
                        source="telegram_export", kind="message",
                        authorship="self" if from_me else "other",
                        doc_category="messages", sensitivity="high",
                        text=tx.clean(text), created_at=when,
                        author="me" if from_me else (sender or None),
                        thread_id=str(thread), app="Telegram",
                        path=relpath(fs.root),
                    )


def _telegram_text(value: Any) -> str:
    """Telegram stores text as a string, or a list mixing strings and entities."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for piece in value:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict):
                parts.append(str(piece.get("text", "")))
        return "".join(parts)
    return ""


def collect_signal(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Signal plaintext exports only.

    Signal Desktop encrypts its database on purpose. Nybble does not attempt to
    extract or crack that key; if the user wants Signal included they export a
    plaintext backup themselves and drop it here.
    """
    for fs in _each(paths):
        for name in fs.find("signal", suffix=".json"):
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                text = tx.clean(item.get("body") or item.get("text") or "")
                if len(text) < 2:
                    continue
                from_me = bool(item.get("sent_at") or item.get("outgoing"))
                yield Record(
                    source="signal_export", kind="message",
                    authorship="self" if from_me else "other",
                    doc_category="messages", sensitivity="high",
                    text=text,
                    created_at=from_unix(item.get("timestamp") or item.get("sent_at"), "ms"),
                    thread_id=str(item.get("conversationId") or ""),
                    app="Signal", path=relpath(fs.root),
                )


# "[2024-06-03, 09:12:33] Jane Doe: hello" and the bracket-free US variant.
_WA_LINE = re.compile(
    r"^\[?(?P<date>\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4})[,]?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?)\]?\s*[-–]?\s*"
    r"(?P<sender>[^:]{1,60}):\s(?P<text>.*)$")


def collect_whatsapp(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("chat", "whatsapp", suffix=".txt"):
            raw = fs.read_text(name)
            if not raw:
                continue
            thread = re.sub(r"(?i)whatsapp chat (?:with )?", "",
                            Path(name).stem).strip() or Path(name).stem
            current = None
            for line in raw.splitlines():
                m = _WA_LINE.match(line.strip())
                if m:
                    if current:
                        yield current
                    text = m.group("text").strip()
                    if text in ("<Media omitted>", "This message was deleted", ""):
                        current = None
                        continue
                    sender = m.group("sender").strip()
                    current = Record(
                        source="whatsapp_export", kind="message",
                        authorship="other",  # refined below once we know the owner
                        doc_category="messages", sensitivity="high",
                        text=text,
                        created_at=parse_date(f"{m.group('date')} {m.group('time')}"),
                        author=sender, thread_id=thread,
                        app="WhatsApp", path=relpath(fs.root),
                    )
                elif current is not None and line.strip():
                    current.text += "\n" + line.strip()
            if current:
                yield current


def collect_sms_backup(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """'SMS Backup & Restore' XML from Android."""
    for fs in _each(paths):
        for name in fs.find("sms", "calls", suffix=".xml"):
            raw = fs.read(name)
            if not raw:
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            for node in root.iter():
                if node.tag not in ("sms", "mms"):
                    continue
                body = tx.clean(node.get("body") or "")
                if len(body) < 2:
                    continue
                when = from_unix(node.get("date"), "ms")
                if ctx.since and (when or "") < ctx.since:
                    continue
                # type 2 = sent, 1 = received
                from_me = node.get("type") == "2"
                address = node.get("address")
                yield Record(
                    source="android_sms_backup", kind="message",
                    authorship="self" if from_me else "other",
                    doc_category="messages", sensitivity="high",
                    text=body, created_at=when,
                    author="me" if from_me else (node.get("contact_name") or address),
                    recipient=address if from_me else "me",
                    thread_id=address, app="SMS", path=relpath(fs.root),
                )


# ===========================================================================
# Google Takeout
# ===========================================================================

def collect_takeout_chat(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("messages.json", "hangouts.json", suffix=".json"):
            data = fs.read_json(name)
            thread = Path(name).parent.name
            for msg in _as_list((data or {}).get("messages") if isinstance(data, dict) else data):
                if not isinstance(msg, dict):
                    continue
                text = tx.clean(msg.get("text") or "")
                if len(text) < 2:
                    continue
                creator = msg.get("creator") or {}
                author = creator.get("name") if isinstance(creator, dict) else None
                yield Record(
                    source="google_chat", kind="message", authorship="mixed",
                    doc_category="messages", sensitivity="high",
                    text=text,
                    created_at=parse_date(msg.get("created_date")),
                    author=author, thread_id=thread,
                    app="Google Chat", path=relpath(fs.root),
                )


def collect_takeout_activity(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Search queries and the ad-interest topics Google assigned."""
    for fs in _each(paths):
        for name in fs.find("myactivity", "my activity", suffix=".json"):
            queries = []
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or ""
                if title.startswith("Searched for "):
                    queries.append((title[len("Searched for "):],
                                    parse_date(item.get("time"))))
            for chunk_start in range(0, min(len(queries), 5000), 500):
                chunk = queries[chunk_start:chunk_start + 500]
                yield Record(
                    source="google_activity", kind="preference", authorship="self",
                    doc_category="social_preferences", sensitivity="high",
                    text="Google searches: " + ", ".join(q for q, _ in chunk),
                    created_at=chunk[0][1] if chunk else None,
                    app="Google", path=relpath(fs.root),
                    meta={"query_count": len(chunk)},
                )

        for name in fs.find("ads", "interest", suffix=".json"):
            data = fs.read_json(name)
            labels = sorted({v.strip() for v in _walk_values(data, "name")
                             if isinstance(v, str) and 2 < len(v) < 80})[:400]
            if labels:
                yield Record(
                    source="google_activity", kind="preference", authorship="system",
                    doc_category="social_preferences", sensitivity="medium",
                    text="Ad topics Google inferred: " + ", ".join(labels),
                    app="Google", path=relpath(fs.root),
                    meta={"interests": labels, "platform_declared": True},
                )


def collect_youtube(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("watch-history", "watch history", suffix=".json"):
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").replace("Watched ", "").strip()
                if not title or title == "a video that has been removed":
                    continue
                when = parse_date(item.get("time"))
                if ctx.since and (when or "") < ctx.since:
                    continue
                channel = None
                subtitles = item.get("subtitles") or []
                if subtitles and isinstance(subtitles[0], dict):
                    channel = subtitles[0].get("name")
                yield Record(
                    source="youtube_takeout", kind="consumption", authorship="other",
                    doc_category="liked_content", sensitivity="medium",
                    text=title, title=title, author=channel,
                    created_at=when, url=item.get("titleUrl"),
                    app="YouTube", path=relpath(fs.root),
                    signals={"watched": True},
                )

        for name in fs.find("subscriptions", suffix=".csv"):
            channels = [r.get("Channel Title") or r.get("Channel title")
                        for r in fs.read_csv(name)]
            channels = [c for c in channels if c]
            if channels:
                yield Record(
                    source="youtube_takeout", kind="preference", authorship="system",
                    doc_category="liked_content", sensitivity="low",
                    text="YouTube channels subscribed: " + ", ".join(channels[:400]),
                    app="YouTube", path=relpath(fs.root),
                    meta={"channels": channels[:400]},
                )

        for name in fs.find("playlist", suffix=".csv"):
            if "liked" not in name.lower() and "favorite" not in name.lower():
                continue
            rows = fs.read_csv(name)
            ids = [r.get("Video ID") for r in rows if r.get("Video ID")]
            if ids:
                yield Record(
                    source="youtube_takeout", kind="reaction", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text=f"Liked {len(ids)} YouTube videos",
                    app="YouTube", path=relpath(fs.root),
                    signals={"liked": True},
                    meta={"video_ids": ids[:500]},
                )


# ===========================================================================
# Media & commerce
# ===========================================================================

def collect_spotify(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("streaminghistory", "endsong", suffix=".json"):
            plays: Dict[str, int] = {}
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                artist = (item.get("artistName") or
                          item.get("master_metadata_album_artist_name"))
                track = (item.get("trackName") or
                         item.get("master_metadata_track_name"))
                if not artist or not track:
                    continue
                key = f"{artist} — {track}"
                plays[key] = plays.get(key, 0) + 1
            top = sorted(plays.items(), key=lambda kv: -kv[1])[:500]
            if top:
                yield Record(
                    source="spotify_export", kind="consumption", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text="Most-played tracks: " + "; ".join(
                        f"{k} ({v}x)" for k, v in top[:200]),
                    app="Spotify", path=relpath(fs.root),
                    signals={"play_counts": True},
                    meta={"top_tracks": [{"track": k, "plays": v} for k, v in top]},
                )

        for name in fs.find("yourlibrary", suffix=".json"):
            data = fs.read_json(name)
            tracks = []
            for block in _walk_values(data, "tracks"):
                for item in _as_list(block):
                    if isinstance(item, dict) and item.get("track"):
                        tracks.append(f"{item.get('artist', '')} — {item['track']}")
            if tracks:
                yield Record(
                    source="spotify_export", kind="reaction", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text="Saved tracks: " + "; ".join(tracks[:300]),
                    app="Spotify", path=relpath(fs.root),
                    signals={"saved": True},
                )


def collect_podcasts(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find(".opml", suffix=".opml"):
            shows = _parse_opml(fs.read(name))
            if shows:
                yield Record(
                    source="podcasts", kind="preference", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text="Podcasts subscribed: " + ", ".join(
                        s["title"] for s in shows[:300]),
                    app="Podcasts", path=relpath(fs.root),
                    meta={"shows": shows[:300]},
                )


def collect_opml(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """RSS subscriptions: a hand-curated list of sources the user trusts."""
    for fs in _each(paths):
        for name in fs.find(".opml", "feed", suffix=".opml"):
            feeds = _parse_opml(fs.read(name))
            if feeds:
                yield Record(
                    source="rss_subscriptions", kind="preference", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text="Feeds followed: " + ", ".join(
                        f["title"] for f in feeds[:400]),
                    app="RSS", path=relpath(fs.root),
                    signals={"subscribed": True},
                    meta={"feeds": feeds[:400]},
                )


def _parse_opml(raw: Optional[bytes]) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for node in root.iter("outline"):
        title = node.get("title") or node.get("text")
        url = node.get("xmlUrl") or node.get("htmlUrl")
        if title and url:
            out.append({"title": title, "url": url})
    return out


def collect_goodreads(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("goodreads", "library", suffix=".csv"):
            for row in fs.read_csv(name):
                title = row.get("Title")
                if not title:
                    continue
                rating = _float(row.get("My Rating"))
                review = tx.clean(row.get("My Review") or "")
                shelf = row.get("Exclusive Shelf") or ""
                body = f"{title} by {row.get('Author', '')}".strip()
                if review:
                    body += f"\n\n{review}"
                yield Record(
                    source="goodreads",
                    kind="comment" if review else "reaction",
                    authorship="self" if review else "other",
                    doc_category="liked_content", sensitivity="low",
                    text=body, title=title, author=row.get("Author"),
                    created_at=parse_date(row.get("Date Read") or row.get("Date Added")),
                    app="Goodreads", path=relpath(fs.root),
                    signals={"rating": rating, "liked": bool(rating and rating >= 4),
                             "shelf": shelf},
                    meta={"shelf": shelf, "has_review": bool(review)},
                )


def collect_ratings_csv(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Letterboxd and IMDb ratings/reviews."""
    for fs in _each(paths):
        for name in fs.find("ratings", "reviews", "watched", suffix=".csv"):
            app = "Letterboxd" if "letterboxd" in str(fs.root).lower() else "IMDb"
            for row in fs.read_csv(name):
                title = row.get("Name") or row.get("Title")
                if not title:
                    continue
                rating = _float(row.get("Rating") or row.get("Your Rating"))
                review = tx.clean(row.get("Review") or "")
                yield Record(
                    source="letterboxd",
                    kind="comment" if review else "reaction",
                    authorship="self" if review else "other",
                    doc_category="liked_content", sensitivity="low",
                    text=f"{title}\n\n{review}".strip(), title=title,
                    created_at=parse_date(row.get("Date") or row.get("Date Rated")),
                    url=row.get("Letterboxd URI") or row.get("URL"),
                    app=app, path=relpath(fs.root),
                    signals={"rating": rating,
                             "liked": bool(rating and rating >= (4 if app == "Letterboxd" else 8))},
                )


def collect_commerce(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Netflix viewing and Amazon purchases/wishlists."""
    for fs in _each(paths):
        netflix = "netflix" in str(fs.root).lower()
        for name in fs.find("viewingactivity", "viewing", suffix=".csv"):
            titles = [r.get("Title") for r in fs.read_csv(name) if r.get("Title")]
            if titles:
                yield Record(
                    source="netflix_amazon", kind="consumption", authorship="other",
                    doc_category="social_preferences", sensitivity="medium",
                    text="Watched on Netflix: " + "; ".join(titles[:400]),
                    app="Netflix", path=relpath(fs.root),
                    signals={"watched": True},
                )
        if netflix:
            continue
        for name in fs.find("order", "wishlist", suffix=".csv"):
            items = []
            for row in fs.read_csv(name):
                label = (row.get("Title") or row.get("Product Name") or
                         row.get("item-name"))
                if label:
                    items.append(label)
            if items:
                wish = "wishlist" in name.lower()
                yield Record(
                    source="netflix_amazon",
                    kind="bookmark" if wish else "consumption",
                    authorship="other",
                    doc_category="social_preferences", sensitivity="medium",
                    text=("Wishlisted: " if wish else "Purchased: ") +
                         "; ".join(items[:400]),
                    app="Amazon", path=relpath(fs.root),
                    signals={"wishlisted": True} if wish else {"purchased": True},
                )


def _float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ===========================================================================
# Read-later services
# ===========================================================================

def collect_readlater(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Readwise, Pocket, Instapaper, Matter, Omnivore.

    Highlights here are the purest form of doc §4: passages from other people's
    writing that the user chose to mark.
    """
    for fs in _each(paths):
        app = _readlater_app(fs.root)

        for name in fs.find("highlight", "readwise", suffix=".csv"):
            for row in fs.read_csv(name):
                text = tx.clean(row.get("Highlight") or row.get("text") or "")
                if len(text) < 15:
                    continue
                yield Record(
                    source="readwise_export", kind="highlight", authorship="other",
                    doc_category="liked_writing", sensitivity="low",
                    text=text,
                    title=row.get("Book Title") or row.get("title"),
                    author=row.get("Book Author") or row.get("author"),
                    created_at=parse_date(row.get("Highlighted at") or row.get("date")),
                    url=row.get("Amazon Book ID") or row.get("url"),
                    app=app, path=relpath(fs.root),
                    signals={"highlighted": True},
                    meta={"note": tx.clean(row.get("Note") or "") or None},
                )

        # Pocket ships HTML; Instapaper ships CSV.
        for name in fs.find("ril_export", "pocket", suffix=".html"):
            html = fs.read_text(name)
            for url, title in _pocket_links(html or ""):
                yield Record(
                    source="readwise_export", kind="bookmark", authorship="other",
                    doc_category="liked_writing", sensitivity="low",
                    text=title or url, title=title, url=url,
                    app="Pocket", path=relpath(fs.root),
                    signals={"saved": True},
                )

        for name in fs.find("instapaper", suffix=".csv"):
            for row in fs.read_csv(name):
                url = row.get("URL")
                if not url:
                    continue
                yield Record(
                    source="readwise_export", kind="bookmark", authorship="other",
                    doc_category="liked_writing", sensitivity="low",
                    text=row.get("Title") or url, title=row.get("Title"),
                    url=url, created_at=parse_date(row.get("Timestamp")),
                    app="Instapaper", path=relpath(fs.root),
                    signals={"saved": True, "folder": row.get("Folder")},
                )


def _readlater_app(root: Path) -> str:
    lowered = str(root).lower()
    for needle, label in (("readwise", "Readwise"), ("pocket", "Pocket"),
                          ("instapaper", "Instapaper"), ("matter", "Matter"),
                          ("omnivore", "Omnivore")):
        if needle in lowered:
            return label
    return "read-later"


_POCKET_LINK = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>([^<]*)</a>', re.I)


def _pocket_links(html: str) -> Iterator[tuple]:
    for m in _POCKET_LINK.finditer(html):
        url, title = m.group(1), tx.clean(m.group(2))
        if url.startswith("http"):
            yield url, title


# ===========================================================================
# Developer and forum platforms
# ===========================================================================

def collect_github_export(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("issue_comments", "issues_", "pull_requests",
                            "commit_comments", suffix=".json"):
            for item in _as_list(fs.read_json(name)):
                if not isinstance(item, dict):
                    continue
                text = tx.strip_markdown(item.get("body") or "")
                if len(text) < 20:
                    continue
                yield Record(
                    source="github_comments",
                    kind="comment" if "comment" in name.lower() else "post",
                    authorship="self",
                    doc_category="messages", sensitivity="low",
                    text=text, title=item.get("title"),
                    created_at=parse_date(item.get("created_at")),
                    url=item.get("url") or item.get("html_url"),
                    author="me", app="GitHub", path=relpath(fs.root),
                    meta={"repository": item.get("repository")},
                )


def collect_github_stars(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("star", suffix=".json"):
            repos = []
            for item in _as_list(fs.read_json(name)):
                if isinstance(item, dict):
                    repo = item.get("repository") or item.get("repo") or item.get("name")
                    if isinstance(repo, dict):
                        repo = repo.get("full_name") or repo.get("name")
                    if repo:
                        repos.append(str(repo))
                elif isinstance(item, str):
                    repos.append(item)
            if repos:
                yield Record(
                    source="github_stars", kind="reaction", authorship="other",
                    doc_category="liked_content", sensitivity="low",
                    text="Starred repositories: " + ", ".join(repos[:500]),
                    app="GitHub", path=relpath(fs.root),
                    signals={"starred": True},
                    meta={"repositories": repos[:500]},
                )


def collect_hackernews(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("hn", "hackernews", suffix=".json"):
            data = fs.read_json(name)
            hits = data.get("hits") if isinstance(data, dict) else data
            for item in _as_list(hits):
                if not isinstance(item, dict):
                    continue
                text = tx.from_html(item.get("comment_text") or "") or \
                       tx.clean(item.get("story_title") or item.get("title") or "")
                if len(text) < 20:
                    continue
                is_comment = bool(item.get("comment_text"))
                yield Record(
                    source="hackernews",
                    kind="comment" if is_comment else "post",
                    authorship="self",
                    doc_category="social_preferences", sensitivity="low",
                    text=text, title=item.get("story_title"),
                    created_at=parse_date(item.get("created_at")),
                    url=item.get("story_url"), author=item.get("author"),
                    app="Hacker News", path=relpath(fs.root),
                    signals={"points": _int(item.get("points"))},
                )


def collect_stackexchange(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("posts", "answers", "questions", suffix=".csv"):
            for row in fs.read_csv(name):
                text = tx.strip_markdown(row.get("Body") or row.get("body") or "")
                if len(text) < 30:
                    continue
                yield Record(
                    source="stackexchange", kind="post", authorship="self",
                    doc_category="social_preferences", sensitivity="low",
                    text=text, title=row.get("Title"),
                    created_at=parse_date(row.get("CreationDate") or row.get("Date")),
                    url=row.get("Url") or row.get("Link"),
                    author="me", app="Stack Exchange", path=relpath(fs.root),
                    signals={"score": _int(row.get("Score"))},
                )


# ===========================================================================
# Notes and blogging platforms
# ===========================================================================

def collect_notion(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("", suffix=".md"):
            raw = fs.read_text(name)
            if not raw:
                continue
            text = tx.strip_markdown(raw)
            if len(text) < ctx.min_chars:
                continue
            # Notion appends a 32-char hash to every exported filename.
            title = re.sub(r"\s+[0-9a-f]{32}$", "", Path(name).stem)
            yield Record(
                source="notion_export", kind="document", authorship="self",
                doc_category="written_by_user", sensitivity="medium",
                text=text, title=title or None,
                app="Notion", path=relpath(fs.root),
                meta={"export_path": name},
            )


def collect_enex(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("", suffix=".enex"):
            raw = fs.read(name)
            if not raw:
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            for note in root.iter("note"):
                title = (note.findtext("title") or "").strip()
                content = note.findtext("content") or ""
                text = tx.from_html(content) or ""
                if len(text) < ctx.min_chars:
                    continue
                yield Record(
                    source="evernote_export", kind="document", authorship="self",
                    doc_category="written_by_user", sensitivity="medium",
                    text=text, title=title or None,
                    created_at=parse_date(note.findtext("created")),
                    modified_at=parse_date(note.findtext("updated")),
                    app="Evernote", path=relpath(fs.root),
                )


def collect_roam(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("roam", suffix=".json"):
            for page in _as_list(fs.read_json(name)):
                if not isinstance(page, dict):
                    continue
                blocks: List[str] = []
                _roam_blocks(page.get("children") or [], blocks)
                text = tx.strip_markdown("\n".join(blocks))
                if len(text) < ctx.min_chars:
                    continue
                yield Record(
                    source="roam_export", kind="document", authorship="self",
                    doc_category="written_by_user", sensitivity="medium",
                    text=text, title=page.get("title"),
                    created_at=from_unix(page.get("create-time"), "ms"),
                    modified_at=from_unix(page.get("edit-time"), "ms"),
                    app="Roam", path=relpath(fs.root),
                )


def _roam_blocks(children: List[Any], out: List[str], depth: int = 0) -> None:
    if depth > 30:
        return
    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("string"):
            out.append(str(child["string"]))
        if child.get("children"):
            _roam_blocks(child["children"], out, depth + 1)


def collect_blog_export(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Substack, Medium, WordPress, Ghost."""
    for fs in _each(paths):
        app = _blog_app(fs.root)

        # Substack: posts.csv plus per-post HTML files.
        for name in fs.find("posts.csv", suffix=".csv"):
            for row in fs.read_csv(name):
                body = row.get("body") or ""
                text = tx.from_html(body) if "<" in body else tx.clean(body)
                title = row.get("title") or row.get("post_title")
                if not text and title:
                    text = title
                if not text or len(text) < 50:
                    continue
                yield Record(
                    source="blog_exports", kind="post", authorship="self",
                    doc_category="written_by_user", sensitivity="low",
                    text=text, title=title,
                    created_at=parse_date(row.get("post_date") or row.get("date")),
                    author="me", app=app, path=relpath(fs.root),
                )

        for name in fs.find("posts/", suffix=".html"):
            text = tx.from_html(fs.read_text(name) or "")
            if not text or len(text) < 200:
                continue
            yield Record(
                source="blog_exports", kind="post", authorship="self",
                doc_category="written_by_user", sensitivity="low",
                text=text, title=Path(name).stem, author="me",
                app=app, path=relpath(fs.root),
            )

        # WordPress/Ghost ship XML or JSON.
        for name in fs.find("wordpress", "wp-export", suffix=".xml"):
            raw = fs.read(name)
            if not raw:
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
            for item in root.iter("item"):
                encoded = item.find("content:encoded", ns)
                body = encoded.text if encoded is not None else None
                text = tx.from_html(body or "")
                if not text or len(text) < 100:
                    continue
                yield Record(
                    source="blog_exports", kind="post", authorship="self",
                    doc_category="written_by_user", sensitivity="low",
                    text=text, title=(item.findtext("title") or "").strip() or None,
                    created_at=parse_date(item.findtext("pubDate")),
                    url=item.findtext("link"), author="me",
                    app="WordPress", path=relpath(fs.root),
                )


def _blog_app(root: Path) -> str:
    lowered = str(root).lower()
    for needle, label in (("substack", "Substack"), ("medium", "Medium"),
                          ("wordpress", "WordPress"), ("ghost", "Ghost")):
        if needle in lowered:
            return label
    return "blog"


def collect_fediverse(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Mastodon ActivityPub outbox, and Bluesky exports."""
    for fs in _each(paths):
        app = "Bluesky" if "bluesky" in str(fs.root).lower() else "Mastodon"
        for name in fs.find("outbox", "posts", suffix=".json"):
            data = fs.read_json(name)
            items = data.get("orderedItems") if isinstance(data, dict) else data
            for item in _as_list(items):
                if not isinstance(item, dict):
                    continue
                obj = item.get("object")
                if isinstance(obj, dict):
                    content = obj.get("content") or ""
                    when = obj.get("published")
                    url = obj.get("id")
                elif isinstance(item.get("text"), str):
                    content, when, url = item["text"], item.get("createdAt"), None
                else:
                    continue
                text = tx.from_html(content) if "<" in content else tx.clean(content)
                if not text or len(text) < 10:
                    continue
                yield Record(
                    source="mastodon_bluesky", kind="post", authorship="self",
                    doc_category="social_preferences", sensitivity="low",
                    text=text, created_at=parse_date(when), url=url,
                    author="me", app=app, path=relpath(fs.root),
                )


def collect_linkedin(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for fs in _each(paths):
        for name in fs.find("shares", "comments", suffix=".csv"):
            is_comment = "comment" in name.lower()
            for row in fs.read_csv(name):
                text = tx.clean(row.get("ShareCommentary") or row.get("Message") or
                                row.get("Comment") or "")
                if len(text) < 20:
                    continue
                yield Record(
                    source="linkedin_export",
                    kind="comment" if is_comment else "post",
                    authorship="self",
                    doc_category="social_preferences", sensitivity="low",
                    text=text, created_at=parse_date(row.get("Date")),
                    url=row.get("ShareLink") or row.get("Link"),
                    author="me", app="LinkedIn", path=relpath(fs.root),
                )

        for name in fs.find("company follows", "follows", suffix=".csv"):
            orgs = [r.get("Organization") or r.get("FullName")
                    for r in fs.read_csv(name)]
            orgs = [o for o in orgs if o]
            if orgs:
                yield Record(
                    source="linkedin_export", kind="preference", authorship="system",
                    doc_category="social_preferences", sensitivity="low",
                    text="LinkedIn follows: " + ", ".join(orgs[:300]),
                    app="LinkedIn", path=relpath(fs.root),
                    meta={"follows": orgs[:300]},
                )
