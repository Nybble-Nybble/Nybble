"""The catalog of every data source Nybble knows how to import.

This is the authoritative list. `nybble scan` probes it against the current
machine; `nybble consent` renders it for approval; `nybble collect` runs the
approved entries.

Each source maps to one of the six "past behavior" categories from the plan doc,
or to `context` for signal the doc did not enumerate but that is worth having.

Acquisition modes
-----------------
local    Data already sits on this machine. Nothing for the user to do.
export   The user must request a data export from the platform first (GDPR /
         "download your data"). We parse the resulting archive. `export_hint`
         says exactly where to click.
mount    Data lives on an attached device (a plugged-in Kindle, a phone backup).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import platform_paths as pp

ALL_PLATFORMS = ("darwin", "linux", "win32")


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    group: str                       # consent group shown to the user
    doc_category: str                # schema.DOC_CATEGORIES
    authorship: str                  # dominant authorship of produced records
    sensitivity: str                 # low | medium | high
    acquisition: str                 # local | export | mount
    collector: str                   # "module:function" inside nybble.sources
    what: str                        # plain-language: what gets read
    platforms: Sequence[str] = ALL_PLATFORMS
    probe: Optional[Callable[[], List]] = None   # returns candidate paths
    export_hint: str = ""
    needs: Sequence[str] = ()        # optional pip extras this source wants
    # macOS Full Disk Access, or similar OS-level grant, required.
    needs_os_permission: bool = False

    def available_on_platform(self) -> bool:
        return pp.PLATFORM in self.platforms

    def candidates(self) -> List:
        if self.probe is None:
            return []
        try:
            return self.probe() or []
        except Exception:
            return []

    def resolve_collector(self) -> Callable:
        mod_name, _, fn_name = self.collector.partition(":")
        mod = importlib.import_module(f"nybble.sources.{mod_name}")
        return getattr(mod, fn_name)


# Consent groups: how the approval prompt is chunked. Order matters — it is the
# order the user sees.
GROUPS: Dict[str, str] = {
    "writing": "Things you wrote (notes, docs, essays, drafts, code)",
    "messages": "Your conversations (SMS, chat apps, email, DMs)",
    "speech": "Recordings and transcripts of you talking",
    "reading": "What you read, saved, and highlighted",
    "social": "Your social accounts: posts, likes, and declared interests",
    "context": "Surrounding context (calendar, contacts, shell, repos)",
}

GROUP_NOTES: Dict[str, str] = {
    "messages": "Highest-sensitivity group. Includes other people's words, not "
                "just yours. Redaction is on by default here.",
    "speech": "Audio is transcribed locally; the audio itself is never uploaded.",
    "social": "Mostly requires you to download an export from each platform first.",
}


def _archives(*names: str):
    """Probe helper for export bundles the user has downloaded."""
    def _probe():
        from .sources.archives import find_archives
        return find_archives(names)
    return _probe


SOURCES: List[Source] = [

    # ======================================================================
    # doc §1 — Essays / content written by the user
    # ======================================================================
    Source(
        id="local_documents",
        name="Documents, essays, and drafts on disk",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local",
        collector="filesystem:collect_documents",
        what="Text in .md .txt .docx .odt .rtf .pdf .tex .org files under your "
             "Documents, Desktop, Downloads, iCloud, Dropbox, and Drive folders.",
        probe=pp.documents_dirs,
        needs=("formats",),
    ),
    Source(
        id="obsidian",
        name="Obsidian vaults",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local",
        collector="filesystem:collect_obsidian",
        what="Every markdown note in folders containing an .obsidian directory.",
        probe=lambda: _find_vaults(".obsidian"),
    ),
    Source(
        id="logseq",
        name="Logseq graphs",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local",
        collector="filesystem:collect_logseq",
        what="Journals and pages from folders containing a logseq directory.",
        probe=lambda: _find_vaults("logseq"),
    ),
    Source(
        id="apple_notes",
        name="Apple Notes",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local", platforms=("darwin",),
        collector="sqlite_apps:collect_apple_notes",
        what="Note bodies and folders from the local NoteStore database.",
        probe=pp.apple_notes_db,
        needs_os_permission=True,
    ),
    Source(
        id="bear",
        name="Bear notes",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local", platforms=("darwin",),
        collector="sqlite_apps:collect_bear",
        what="Note text and tags from Bear's local database.",
        probe=pp.bear_db,
    ),
    Source(
        id="dayone",
        name="Day One journal",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="high", acquisition="local", platforms=("darwin",),
        collector="sqlite_apps:collect_dayone",
        what="Journal entries. Personal diary content — review before approving.",
        probe=pp.dayone_db,
    ),
    Source(
        id="joplin",
        name="Joplin notes",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="local",
        collector="sqlite_apps:collect_joplin",
        what="Notes and notebooks from Joplin's local database.",
        probe=pp.joplin_dir,
    ),
    Source(
        id="notion_export",
        name="Notion workspace export",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_notion",
        what="Pages and databases from a Notion markdown/CSV export.",
        probe=_archives("notion"),
        export_hint="Notion → Settings → Workspace → Export all workspace content "
                    "→ Markdown & CSV.",
    ),
    Source(
        id="evernote_export",
        name="Evernote export (.enex)",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_enex",
        what="Notes from one or more .enex export files.",
        probe=_archives("evernote", "enex"),
        export_hint="Evernote → select notebook → File → Export Notes as .enex.",
    ),
    Source(
        id="roam_export",
        name="Roam Research export",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_roam",
        what="Blocks and pages from a Roam JSON export.",
        probe=_archives("roam"),
        export_hint="Roam → ... menu → Export All → JSON.",
    ),
    Source(
        id="jupyter",
        name="Jupyter notebooks",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="low", acquisition="local",
        collector="filesystem:collect_notebooks",
        what="Markdown cells (your prose, not the code output) from .ipynb files.",
        probe=lambda: pp.documents_dirs() + pp.code_roots(),
    ),
    Source(
        id="latex",
        name="LaTeX / academic writing",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="low", acquisition="local",
        collector="filesystem:collect_latex",
        what="Prose extracted from .tex sources, with markup stripped.",
        probe=pp.documents_dirs,
    ),
    Source(
        id="blog_exports",
        name="Blog & newsletter exports (Substack, Medium, WordPress, Ghost)",
        group="writing", doc_category="written_by_user", authorship="self",
        sensitivity="low", acquisition="export",
        collector="archives:collect_blog_export",
        what="Posts you published, from a platform export archive.",
        probe=_archives("substack", "medium", "wordpress", "ghost"),
        export_hint="Substack → Settings → Export. Medium → Settings → Download "
                    "your information. WordPress → Tools → Export.",
    ),
    Source(
        id="git_authored",
        name="Your git commits and code comments",
        group="context", doc_category="written_by_user", authorship="self",
        sensitivity="low", acquisition="local",
        collector="shell_git:collect_git",
        what="Commit messages you authored in local repositories. Strong signal "
             "for how you explain technical decisions.",
        probe=lambda: pp.code_roots() + pp.documents_dirs(),
    ),

    # ======================================================================
    # doc §2 — Text messages
    # ======================================================================
    Source(
        id="imessage",
        name="iMessage / SMS (macOS)",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="local", platforms=("darwin",),
        collector="sqlite_apps:collect_imessage",
        what="Message text, timestamps, and who sent what, from chat.db. "
             "Includes messages other people sent you.",
        probe=pp.imessage_db,
        needs_os_permission=True,
    ),
    Source(
        id="android_sms_backup",
        name="Android SMS backup XML",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_sms_backup",
        what="SMS/MMS from an 'SMS Backup & Restore' XML file.",
        probe=_archives("sms", "smsbackup"),
        export_hint="Install 'SMS Backup & Restore' on Android → Back up → "
                    "transfer the .xml to this computer.",
    ),
    Source(
        id="whatsapp_export",
        name="WhatsApp chat exports",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_whatsapp",
        what="Per-chat .txt exports ('WhatsApp Chat with ...').",
        probe=_archives("whatsapp"),
        export_hint="WhatsApp → open a chat → ⋮ → More → Export chat → Without media.",
    ),
    Source(
        id="telegram_export",
        name="Telegram Desktop export",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_telegram",
        what="Messages from a Telegram JSON export.",
        probe=_archives("telegram"),
        export_hint="Telegram Desktop → Settings → Advanced → Export Telegram data "
                    "→ format JSON.",
    ),
    Source(
        id="signal_export",
        name="Signal export",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_signal",
        what="Messages from a Signal backup you have already decrypted. "
             "Nybble never attempts to break Signal's encryption.",
        probe=_archives("signal"),
        export_hint="Signal Desktop's database is encrypted by design. Export a "
                    "plaintext backup yourself if you want it included.",
    ),
    Source(
        id="discord_export",
        name="Discord data package",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_discord",
        what="Your messages across servers and DMs, from the official package.",
        probe=_archives("discord"),
        export_hint="Discord → Settings → Privacy & Safety → Request all of my data.",
    ),
    Source(
        id="slack_export",
        name="Slack export",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_slack",
        what="Channel and DM messages from a Slack workspace/user export.",
        probe=_archives("slack"),
        export_hint="Slack → workspace settings → Import/Export Data → Export.",
    ),
    Source(
        id="messenger_export",
        name="Facebook Messenger",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_meta_messages",
        what="Messenger threads inside a Facebook 'Download Your Information' archive.",
        probe=_archives("facebook", "meta"),
        export_hint="Facebook → Settings → Your information → Download your "
                    "information → JSON.",
    ),
    Source(
        id="instagram_dms",
        name="Instagram direct messages",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_meta_messages",
        what="DM threads inside an Instagram data download.",
        probe=_archives("instagram"),
        export_hint="Instagram → Settings → Accounts Center → Your information and "
                    "permissions → Download your information → JSON.",
    ),
    Source(
        id="google_chat",
        name="Google Chat / Hangouts",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_takeout_chat",
        what="Chat messages inside a Google Takeout archive.",
        probe=_archives("takeout", "google"),
        export_hint="takeout.google.com → select Google Chat / Hangouts → Export.",
    ),
    Source(
        id="email_local",
        name="Email (Apple Mail, Thunderbird, mbox, Maildir)",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="local",
        collector="mail:collect_local_mail",
        what="Message bodies from local mail stores. Mail you sent is the "
             "strongest writing sample most people have.",
        probe=pp.mail_dirs,
        needs_os_permission=True,
    ),
    Source(
        id="email_export",
        name="Email export (Gmail Takeout .mbox)",
        group="messages", doc_category="messages", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="mail:collect_mbox_export",
        what="Messages from a Gmail Takeout .mbox file.",
        probe=_archives("takeout", "mail", "mbox"),
        export_hint="takeout.google.com → select Mail → Export.",
    ),
    Source(
        id="github_comments",
        name="GitHub issue & PR comments",
        group="messages", doc_category="messages", authorship="self",
        sensitivity="low", acquisition="export",
        collector="archives:collect_github_export",
        what="Comments and issue bodies you wrote, from a GitHub data export.",
        probe=_archives("github"),
        export_hint="GitHub → Settings → Account → Export account data.",
    ),

    # ======================================================================
    # doc §3 — Transcriptions of audio interactions
    # ======================================================================
    Source(
        id="existing_transcripts",
        name="Transcripts you already have",
        group="speech", doc_category="transcripts", authorship="mixed",
        sensitivity="high", acquisition="local",
        collector="audio:collect_existing_transcripts",
        what=".vtt .srt .txt transcripts from Zoom, Teams, Granola, Otter, "
             "Fathom, Fireflies, and Rev folders.",
        probe=lambda: pp.zoom_dirs() + pp.meeting_note_dirs(),
    ),
    Source(
        id="voice_memos",
        name="Voice Memos",
        group="speech", doc_category="transcripts", authorship="self",
        sensitivity="high", acquisition="local", platforms=("darwin",),
        collector="audio:collect_voice_memos",
        what="Your voice recordings, transcribed locally with Whisper.",
        probe=pp.voice_memos_dirs,
        needs=("audio",),
        needs_os_permission=True,
    ),
    Source(
        id="local_audio",
        name="Other local audio & video recordings",
        group="speech", doc_category="transcripts", authorship="mixed",
        sensitivity="high", acquisition="local",
        collector="audio:collect_local_media",
        what="Meeting and call recordings (.m4a .mp3 .wav .mp4 .mov) found in "
             "your folders, transcribed locally. Audio never leaves the machine.",
        probe=lambda: pp.zoom_dirs() + pp.documents_dirs(),
        needs=("audio",),
    ),

    # ======================================================================
    # doc §4 and §5 — Writing and content of others that you like
    #
    # No platform exposes "things you like" directly. These sources pool
    # everything the user kept, marked, replayed, or lingered on; the taste
    # stage then infers affinity from the pool.
    # ======================================================================
    Source(
        id="browser_bookmarks",
        name="Browser bookmarks",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="medium", acquisition="local",
        collector="sqlite_apps:collect_bookmarks",
        what="Bookmarks from Chrome, Edge, Brave, Arc, Vivaldi, Firefox, Safari. "
             "A bookmark is a deliberate keep — strong affinity evidence.",
        probe=lambda: pp.chromium_profiles() + pp.firefox_profiles() + pp.safari_dir(),
    ),
    Source(
        id="browser_history",
        name="Browser history (with visit counts)",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="high", acquisition="local",
        collector="sqlite_apps:collect_history",
        what="URLs, titles, visit counts, and dwell where recorded. Revisits are "
             "the closest local proxy for 'content you actually liked'.",
        probe=lambda: pp.chromium_profiles() + pp.firefox_profiles() + pp.safari_dir(),
    ),
    Source(
        id="kindle_clippings",
        name="Kindle highlights (My Clippings.txt)",
        group="reading", doc_category="liked_writing", authorship="other",
        sensitivity="low", acquisition="mount",
        collector="filesystem:collect_kindle",
        what="Every passage you highlighted, with book and timestamp. The single "
             "cleanest signal for 'writing of others that you like'.",
        probe=pp.kindle_clippings,
    ),
    Source(
        id="apple_books",
        name="Apple Books highlights & notes",
        group="reading", doc_category="liked_writing", authorship="other",
        sensitivity="low", acquisition="local", platforms=("darwin",),
        collector="sqlite_apps:collect_apple_books",
        what="Highlighted passages and your margin notes from Apple Books.",
        probe=pp.apple_books_dirs,
    ),
    Source(
        id="zotero",
        name="Zotero library, notes, and annotations",
        group="reading", doc_category="liked_writing", authorship="other",
        sensitivity="low", acquisition="local",
        collector="sqlite_apps:collect_zotero",
        what="Papers you saved, plus your notes and PDF annotations on them.",
        probe=pp.zotero_db,
    ),
    Source(
        id="readwise_export",
        name="Readwise / Instapaper / Pocket / Matter / Omnivore",
        group="reading", doc_category="liked_writing", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_readlater",
        what="Saved articles and highlights from read-later services.",
        probe=_archives("readwise", "pocket", "instapaper", "matter", "omnivore"),
        export_hint="Readwise → Export. Pocket → getpocket.com/export. "
                    "Instapaper → Settings → Download .csv.",
    ),
    Source(
        id="rss_subscriptions",
        name="RSS subscriptions & starred items",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_opml",
        what="Feeds you chose to follow, from an OPML file or reader export.",
        probe=_archives("feedly", "opml", "inoreader", "newsblur"),
        export_hint="Most readers export OPML under Settings → Import/Export.",
    ),
    Source(
        id="goodreads",
        name="Goodreads library & ratings",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_goodreads",
        what="Books, shelves, your star ratings, and your reviews.",
        probe=_archives("goodreads"),
        export_hint="Goodreads → My Books → Import and export → Export Library.",
    ),
    Source(
        id="youtube_takeout",
        name="YouTube watch history, likes, and subscriptions",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_youtube",
        what="What you watched, liked, and subscribed to. Titles and channels; "
             "transcripts are not fetched.",
        probe=_archives("takeout", "youtube"),
        export_hint="takeout.google.com → select YouTube and YouTube Music → Export.",
    ),
    Source(
        id="spotify_export",
        name="Spotify listening history & saved tracks",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_spotify",
        what="Streaming history, saved tracks, and playlists you built.",
        probe=_archives("spotify"),
        export_hint="Spotify → Account → Privacy settings → Download your data "
                    "(request the extended history).",
    ),
    Source(
        id="podcasts",
        name="Podcast subscriptions & plays",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_podcasts",
        what="Shows you subscribe to and episodes you finished (Overcast, "
             "Pocket Casts, Apple Podcasts OPML).",
        probe=_archives("overcast", "pocketcasts", "podcast"),
        export_hint="Overcast → Settings → Export OPML. Pocket Casts → Settings "
                    "→ Export.",
    ),
    Source(
        id="github_stars",
        name="GitHub stars",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_github_stars",
        what="Repositories you starred, with their descriptions and topics.",
        probe=_archives("github"),
        export_hint="Included in the GitHub account data export.",
    ),
    Source(
        id="letterboxd",
        name="Letterboxd / IMDb ratings",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="low", acquisition="export",
        collector="archives:collect_ratings_csv",
        what="Films you rated and reviewed.",
        probe=_archives("letterboxd", "imdb"),
        export_hint="Letterboxd → Settings → Data → Export your data.",
    ),
    Source(
        id="saved_pdfs",
        name="Papers and PDFs you downloaded",
        group="reading", doc_category="liked_writing", authorship="other",
        sensitivity="low", acquisition="local",
        collector="filesystem:collect_saved_pdfs",
        what="Text from PDFs in your Downloads and Documents. Downloading and "
             "keeping a paper is a deliberate choice worth reading as taste.",
        probe=pp.documents_dirs,
        needs=("formats",),
    ),
    Source(
        id="screenshots",
        name="Screenshots (OCR)",
        group="reading", doc_category="liked_content", authorship="other",
        sensitivity="high", acquisition="local",
        collector="filesystem:collect_screenshots",
        what="Text read out of screenshots you kept. People screenshot what they "
             "want to remember; OCR runs locally.",
        probe=pp.screenshots_dirs,
        needs=("ocr",),
    ),

    # ======================================================================
    # doc §6 — Preferences saved by social media
    # ======================================================================
    Source(
        id="twitter_archive",
        name="X / Twitter archive (tweets, likes, bookmarks)",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_twitter",
        what="Your tweets and replies (your voice), plus likes and bookmarks "
             "(other people's writing you endorsed), plus the interest tags X "
             "inferred about you.",
        probe=_archives("twitter", "x-archive"),
        export_hint="X → Settings → Your account → Download an archive of your data.",
    ),
    Source(
        id="reddit_export",
        name="Reddit posts, comments, saved, and upvoted",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_reddit",
        what="Your comments and posts, plus saved/upvoted items and subreddit "
             "subscriptions.",
        probe=_archives("reddit"),
        export_hint="reddit.com/settings/data-request → request your data.",
    ),
    Source(
        id="facebook_prefs",
        name="Facebook profile, posts, and ad interests",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_meta_prefs",
        what="Posts, comments, likes, page follows, and the 'ads interests' "
             "Meta inferred — a literal list of topics a platform thinks you like.",
        probe=_archives("facebook", "meta"),
        export_hint="Facebook → Download your information → JSON.",
    ),
    Source(
        id="instagram_prefs",
        name="Instagram posts, saved, and topic interests",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_meta_prefs",
        what="Captions you wrote, posts you saved, accounts you follow, and "
             "Instagram's inferred topic list.",
        probe=_archives("instagram"),
        export_hint="Instagram → Download your information → JSON.",
    ),
    Source(
        id="tiktok_export",
        name="TikTok watch history, likes, and favorites",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_tiktok",
        what="Videos watched (with timestamps), liked, favorited, and searched. "
             "Watch timestamps approximate the attention signal the plan doc "
             "wants from scroll behavior.",
        probe=_archives("tiktok"),
        export_hint="TikTok → Settings → Account → Download your data → JSON.",
    ),
    Source(
        id="linkedin_export",
        name="LinkedIn posts, follows, and interests",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="low", acquisition="export",
        collector="archives:collect_linkedin",
        what="Your posts and comments, companies and people you follow.",
        probe=_archives("linkedin"),
        export_hint="LinkedIn → Settings → Data privacy → Get a copy of your data.",
    ),
    Source(
        id="mastodon_bluesky",
        name="Mastodon / Bluesky archive",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="low", acquisition="export",
        collector="archives:collect_fediverse",
        what="Your posts and favorites from a Mastodon or Bluesky export.",
        probe=_archives("mastodon", "bluesky", "outbox"),
        export_hint="Mastodon → Preferences → Import and export → Request archive. "
                    "Bluesky → Settings → Export my data.",
    ),
    Source(
        id="google_activity",
        name="Google search history, activity, and ad topics",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="high", acquisition="export",
        collector="archives:collect_takeout_activity",
        what="Search queries, My Activity, and the ad-interest topics Google "
             "assigned you.",
        probe=_archives("takeout", "google"),
        export_hint="takeout.google.com → select My Activity → Export.",
    ),
    Source(
        id="netflix_amazon",
        name="Netflix viewing / Amazon orders & wishlist",
        group="social", doc_category="social_preferences", authorship="other",
        sensitivity="medium", acquisition="export",
        collector="archives:collect_commerce",
        what="What you watched and what you bought or wishlisted.",
        probe=_archives("netflix", "amazon"),
        export_hint="Netflix → Account → Download your personal information. "
                    "Amazon → Request My Data.",
    ),
    Source(
        id="hackernews",
        name="Hacker News comments and favorites",
        group="social", doc_category="social_preferences", authorship="mixed",
        sensitivity="low", acquisition="export",
        collector="archives:collect_hackernews",
        what="Your comments and favorited submissions, from a saved JSON dump.",
        probe=_archives("hackernews", "hn"),
        export_hint="Fetch from hn.algolia.com/api/v1/search?tags=comment,"
                    "author_YOURNAME and save the JSON here.",
    ),
    Source(
        id="stackexchange",
        name="Stack Exchange answers",
        group="social", doc_category="social_preferences", authorship="self",
        sensitivity="low", acquisition="export",
        collector="archives:collect_stackexchange",
        what="Questions and answers you wrote, with their vote scores.",
        probe=_archives("stackexchange", "stackoverflow"),
        export_hint="Stack Exchange → Profile → Settings → Export your data.",
    ),

    # ======================================================================
    # context — not in the plan doc, but load-bearing for personalization
    # ======================================================================
    Source(
        id="calendar",
        name="Calendar events",
        group="context", doc_category="context", authorship="mixed",
        sensitivity="high", acquisition="local",
        collector="calendars:collect_calendars",
        what="Event titles, attendees, and notes. Establishes who you work with "
             "and what your weeks actually look like.",
        probe=lambda: pp.calendar_dirs() + pp.documents_dirs(),
        needs_os_permission=True,
    ),
    Source(
        id="contacts",
        name="Contacts",
        group="context", doc_category="context", authorship="system",
        sensitivity="high", acquisition="local",
        collector="calendars:collect_contacts",
        what="Names and handles only. Used to attribute messages to real people "
             "so the model learns how you talk to each of them.",
        probe=lambda: pp.contacts_dirs() + pp.documents_dirs(),
        needs_os_permission=True,
    ),
    Source(
        id="shell_history",
        name="Shell history",
        group="context", doc_category="context", authorship="self",
        sensitivity="high", acquisition="local",
        collector="shell_git:collect_shell_history",
        what="Commands you ran. Reveals tooling and working style. Secrets are "
             "scrubbed before anything is written.",
        probe=pp.shell_histories,
    ),
]


def _find_vaults(marker: str) -> List:
    """Locate note vaults by their marker directory (.obsidian, logseq)."""
    out = []
    for root in pp.notes_vault_roots():
        for p in pp.glob_existing(root, f"**/{marker}", limit=50):
            if p.is_dir():
                out.append(p.parent)
    return out


# --- lookups ------------------------------------------------------------------

BY_ID: Dict[str, Source] = {s.id: s for s in SOURCES}


def get(source_id: str) -> Source:
    return BY_ID[source_id]


def for_platform() -> List[Source]:
    return [s for s in SOURCES if s.available_on_platform()]


def by_group() -> Dict[str, List[Source]]:
    out: Dict[str, List[Source]] = {g: [] for g in GROUPS}
    for s in for_platform():
        out.setdefault(s.group, []).append(s)
    return out


def by_doc_category() -> Dict[str, List[Source]]:
    out: Dict[str, List[Source]] = {}
    for s in SOURCES:
        out.setdefault(s.doc_category, []).append(s)
    return out
