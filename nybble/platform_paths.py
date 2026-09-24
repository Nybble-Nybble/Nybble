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
