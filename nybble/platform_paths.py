"""Cross-platform location of the places personal data actually lives.

Every helper returns a list of existing paths, so callers never branch on
sys.platform themselves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

MACOS = sys.platform == "darwin"
WINDOWS = sys.platform in ("win32", "cygwin")
LINUX = sys.platform.startswith("linux")

PLATFORM = "darwin" if MACOS else "win32" if WINDOWS else "linux"


def home() -> Path:
    return Path.home()


def existing(*candidates: object) -> List[Path]:
    """Filter candidate paths down to the ones present on this machine."""
    out: List[Path] = []
    seen = set()
    for c in candidates:
        if c is None:
            continue
        items: Iterable[object] = c if isinstance(c, (list, tuple, set)) else [c]
        for item in items:
            try:
                p = Path(os.path.expandvars(str(item))).expanduser()
            except (OSError, ValueError):
                continue
            try:
                if p.exists() and str(p) not in seen:
                    seen.add(str(p))
                    out.append(p)
            except OSError:
                continue
    return out


def glob_existing(root: object, pattern: str, limit: int = 2000) -> List[Path]:
    """Glob under `root`, tolerating permission errors and missing roots."""
    try:
        base = Path(os.path.expandvars(str(root))).expanduser()
    except (OSError, ValueError):
        return []
    if not base.exists():
        return []
    out: List[Path] = []
    try:
        for p in base.glob(pattern):
            out.append(p)
            if len(out) >= limit:
                break
    except (OSError, PermissionError, ValueError):
        pass
    return out


# --- generic user directories ------------------------------------------------

def app_support() -> Path:
    if MACOS:
        return home() / "Library" / "Application Support"
    if WINDOWS:
        return Path(os.environ.get("APPDATA", home() / "AppData" / "Roaming"))
    return Path(os.environ.get("XDG_DATA_HOME", home() / ".local" / "share"))


def local_app_data() -> Path:
    if WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA", home() / "AppData" / "Local"))
    return app_support()


def documents_dirs() -> List[Path]:
    return existing(
        home() / "Documents",
        home() / "Desktop",
        home() / "Downloads",
        home() / "Dropbox",
        home() / "Google Drive",
        home() / "OneDrive",
        home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs",  # iCloud Drive
    )


def downloads_dir() -> List[Path]:
    return existing(home() / "Downloads")


# --- browsers ----------------------------------------------------------------

def chromium_profiles() -> List[Path]:
    """Profile directories for every Chromium-family browser installed."""
    roots: List[Path] = []
    if MACOS:
        base = home() / "Library" / "Application Support"
        roots = [
            base / "Google/Chrome",
            base / "Google/Chrome Beta",
            base / "Chromium",
            base / "Microsoft Edge",
            base / "BraveSoftware/Brave-Browser",
            base / "Vivaldi",
            base / "Arc/User Data",
            base / "Opera Software/Opera Stable",
        ]
    elif WINDOWS:
        lad = local_app_data()
        roots = [
            lad / "Google/Chrome/User Data",
            lad / "Chromium/User Data",
            lad / "Microsoft/Edge/User Data",
            lad / "BraveSoftware/Brave-Browser/User Data",
            lad / "Vivaldi/User Data",
            Path(os.environ.get("APPDATA", "")) / "Opera Software/Opera Stable",
        ]
    else:
        cfg = Path(os.environ.get("XDG_CONFIG_HOME", home() / ".config"))
        roots = [
            cfg / "google-chrome",
            cfg / "chromium",
            cfg / "microsoft-edge",
            cfg / "BraveSoftware/Brave-Browser",
            cfg / "vivaldi",
            cfg / "opera",
            home() / "snap/chromium/common/chromium",
            home() / ".var/app/com.google.Chrome/config/google-chrome",
        ]

    profiles: List[Path] = []
    for root in existing(*roots):
        for name in ("Default", "Profile 1", "Profile 2", "Profile 3", "Guest Profile"):
            p = root / name
            if p.is_dir():
                profiles.append(p)
        # Some installs use only the root as the profile.
        if (root / "History").exists():
            profiles.append(root)
    return profiles


def firefox_profiles() -> List[Path]:
    if MACOS:
        root = home() / "Library/Application Support/Firefox/Profiles"
    elif WINDOWS:
        root = Path(os.environ.get("APPDATA", "")) / "Mozilla/Firefox/Profiles"
    else:
        root = home() / ".mozilla/firefox"
    out = []
    for p in glob_existing(root, "*"):
        if p.is_dir() and (p / "places.sqlite").exists():
            out.append(p)
    # Flatpak / snap variants
    for alt in existing(
        home() / ".var/app/org.mozilla.firefox/.mozilla/firefox",
        home() / "snap/firefox/common/.mozilla/firefox",
    ):
        for p in glob_existing(alt, "*"):
            if (p / "places.sqlite").exists():
                out.append(p)
    return out


def safari_dir() -> List[Path]:
    return existing(home() / "Library/Safari") if MACOS else []


# --- messaging ---------------------------------------------------------------

def imessage_db() -> List[Path]:
    return existing(home() / "Library/Messages/chat.db") if MACOS else []


def signal_dir() -> List[Path]:
    if MACOS:
        return existing(home() / "Library/Application Support/Signal")
    if WINDOWS:
        return existing(Path(os.environ.get("APPDATA", "")) / "Signal")
    return existing(home() / ".config/Signal")


def telegram_dir() -> List[Path]:
    return existing(
        home() / "Library/Application Support/Telegram Desktop",
        Path(os.environ.get("APPDATA", "")) / "Telegram Desktop" if WINDOWS else None,
        home() / ".local/share/TelegramDesktop",
    )


def whatsapp_dirs() -> List[Path]:
    return existing(
        home() / "Library/Application Support/WhatsApp",
        home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared",
        Path(os.environ.get("APPDATA", "")) / "WhatsApp" if WINDOWS else None,
    )


# --- notes / knowledge apps ---------------------------------------------------

def apple_notes_db() -> List[Path]:
    if not MACOS:
        return []
    return existing(
        home() / "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"
    )


def bear_db() -> List[Path]:
    if not MACOS:
        return []
    return existing(
        home() / "Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/"
        "Application Data/database.sqlite"
    )


def dayone_db() -> List[Path]:
    if not MACOS:
        return []
    return existing(
        home() / "Library/Group Containers/5U8NS4GX82.dayoneapp2/Data/Documents/DayOne.sqlite"
    )


def notes_vault_roots() -> List[Path]:
    """Search roots likely to contain Obsidian/Logseq/plain-markdown vaults."""
    return documents_dirs() + existing(home() / "vaults", home() / "notes", home() / "wiki")


def joplin_dir() -> List[Path]:
    return existing(
        home() / ".config/joplin-desktop",
        home() / "Library/Application Support/joplin-desktop",
        Path(os.environ.get("APPDATA", "")) / "joplin-desktop" if WINDOWS else None,
    )


# --- reading / reference ------------------------------------------------------

def zotero_db() -> List[Path]:
    return existing(
        home() / "Zotero/zotero.sqlite",
        home() / "Library/Application Support/Zotero/zotero.sqlite",
    )


def apple_books_dirs() -> List[Path]:
    if not MACOS:
        return []
    return existing(
        home() / "Library/Containers/com.apple.iBooksX/Data/Documents/AEAnnotation",
        home() / "Library/Containers/com.apple.iBooksX/Data/Documents/BKLibrary",
    )


def kindle_clippings() -> List[Path]:
    """`My Clippings.txt` from a mounted Kindle, or a copy the user kept."""
    found = existing(
        Path("/Volumes/Kindle/documents/My Clippings.txt"),
        Path("D:/documents/My Clippings.txt") if WINDOWS else None,
    )
    for d in documents_dirs():
        found += glob_existing(d, "**/My Clippings.txt", limit=5)
    return found


# --- mail ---------------------------------------------------------------------

def mail_dirs() -> List[Path]:
    out = existing(
        home() / "Library/Mail",                       # Apple Mail
        home() / "Library/Thunderbird",
        home() / ".thunderbird",
        home() / "Mail",
        home() / "Maildir",
        Path(os.environ.get("APPDATA", "")) / "Thunderbird" if WINDOWS else None,
    )
    return out


# --- audio / meetings ---------------------------------------------------------

def voice_memos_dirs() -> List[Path]:
    if not MACOS:
        return []
    return existing(
        home() / "Library/Application Support/com.apple.voicememos/Recordings",
        home() / "Library/Group Containers/group.com.apple.VoiceMemos/Recordings",
    )


def zoom_dirs() -> List[Path]:
    return existing(
        home() / "Documents/Zoom",
        home() / "Zoom",
    )


def meeting_note_dirs() -> List[Path]:
    """Granola / Otter / Fathom / Fireflies exports people drop into Documents."""
    out: List[Path] = []
    for d in documents_dirs():
        for name in ("Granola", "Otter", "Otter.ai", "Fathom", "Fireflies", "Transcripts",
                     "Meetings", "Rev", "Descript"):
            p = d / name
            if p.is_dir():
                out.append(p)
    out += existing(home() / "Library/Application Support/Granola")
    return out


# --- shell / dev ---------------------------------------------------------------

def shell_histories() -> List[Path]:
    return existing(
        home() / ".zsh_history",
        home() / ".bash_history",
        home() / ".local/share/fish/fish_history",
        home() / ".config/fish/fish_history",
        home() / ".history",
        home() / ".python_history",
        home() / ".psql_history",
    )


def code_roots() -> List[Path]:
    return existing(
        home() / "code", home() / "src", home() / "dev", home() / "projects",
        home() / "repos", home() / "workspace", home() / "git",
        home() / "Documents/GitHub",
    )


# --- calendars / contacts -------------------------------------------------------

def calendar_dirs() -> List[Path]:
    return existing(
        home() / "Library/Calendars",
        home() / ".local/share/evolution/calendar",
    )


def contacts_dirs() -> List[Path]:
    return existing(
        home() / "Library/Application Support/AddressBook",
        home() / ".local/share/evolution/addressbook",
    )


def screenshots_dirs() -> List[Path]:
    return existing(
        home() / "Desktop",
        home() / "Pictures/Screenshots",
        home() / "Pictures/Screenshot",
        home() / "OneDrive/Pictures/Screenshots",
    )
