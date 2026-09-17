"""Shell history and git authorship.

Neither is in the plan doc's list, but both are unusually dense signal: commit
messages are the user explaining decisions to an audience, and shell history is
a record of how they actually work.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Set

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import from_unix, parse_date, relpath

# ===========================================================================
# Shell history
# ===========================================================================

# zsh extended history: ": 1699999999:0;git status"
_ZSH_EXT = re.compile(r"^:\s*(\d+):\d*;(.*)$")
# fish: "- cmd: git status" / "  when: 1699999999"
_FISH_CMD = re.compile(r"^- cmd:\s*(.*)$")
_FISH_WHEN = re.compile(r"^\s+when:\s*(\d+)$")

# Commands whose arguments are often secrets.
_SECRET_COMMAND = re.compile(
    r"\b(curl|wget|export|set|env|ssh|scp|mysql|psql|openssl|aws|gcloud|az|"
    r"docker\s+login|npm\s+(?:login|publish)|gh\s+auth|kubectl)\b", re.I)

# A credential riding along in the command line. The redactor would mask the
# token itself, but a command built to pass one is better dropped whole: what
# survives redaction still leaks the endpoint and the auth scheme.
_CARRIES_CREDENTIAL = re.compile(
    r"(--?(?:pass|password|passwd|token|key|secret|auth|header|H)\b"
    r"|Authorization\s*:"
    r"|\bBearer\s+\S+"
    r"|://[^\s/]+:[^\s/@]+@"          # credentials embedded in a URL
    r"|\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[=:]"
    r"|\bsk-[A-Za-z0-9_-]{16,})", re.I)


def collect_shell_history(paths: List[Path], ctx: Context) -> Iterator[Record]:
    for path in paths:
        path = Path(path)
        raw = tx.read_text(path)
        if not raw:
            continue
        shell = _shell_name(path)
        entries = (_parse_fish(raw) if shell == "fish" else _parse_shell(raw))

        for command, when in entries:
            command = command.strip()
            if not command or len(command) < 3:
                continue
            if ctx.since and (when or "") < ctx.since:
                continue
            # The redactor catches inline keys, but whole commands built to pass
            # credentials are better dropped than scrubbed.
            if _SECRET_COMMAND.search(command) and _CARRIES_CREDENTIAL.search(command):
                continue
            yield Record(
                source="shell_history",
                kind="activity",
                authorship="self",
                doc_category="context",
                sensitivity="high",
                text=command,
                created_at=when,
                app=shell,
                path=relpath(path),
                meta={"program": command.split()[0] if command.split() else None},
            )


def _shell_name(path: Path) -> str:
    name = path.name.lower()
    if "fish" in name or "fish" in str(path).lower():
        return "fish"
    if "zsh" in name:
        return "zsh"
    if "bash" in name:
        return "bash"
    if "python" in name:
        return "python"
    if "psql" in name:
        return "psql"
    return "shell"


def _parse_shell(raw: str) -> Iterator[tuple]:
    pending: List[str] = []
    for line in raw.splitlines():
        m = _ZSH_EXT.match(line)
        if m:
            if pending:
                yield "\n".join(pending), None
                pending = []
            yield m.group(2), from_unix(int(m.group(1)))
            continue
        # Multi-line commands continue with a trailing backslash.
        if line.endswith("\\"):
            pending.append(line[:-1])
            continue
        if pending:
            pending.append(line)
            yield "\n".join(pending), None
            pending = []
        else:
            yield line, None


def _parse_fish(raw: str) -> Iterator[tuple]:
    command: Optional[str] = None
    for line in raw.splitlines():
        m = _FISH_CMD.match(line)
        if m:
            if command is not None:
                yield command, None
            command = m.group(1)
            continue
        w = _FISH_WHEN.match(line)
        if w and command is not None:
            yield command, from_unix(int(w.group(1)))
            command = None
    if command is not None:
        yield command, None


# ===========================================================================
# Git
# ===========================================================================

_GIT_SEP = "\x1e"
_GIT_FIELD = "\x1f"


def collect_git(paths: List[Path], ctx: Context) -> Iterator[Record]:
    """Commit messages the user authored, across local repositories."""
    if not _has_git():
        return

    identities = _git_identities()
    repos = _find_repos(paths)
    seen = 0

    for repo in repos:
        if seen >= ctx.max_records_per_source:
            return
        for commit in _log(repo, identities, ctx):
            subject, body, when, author, sha = commit
            message = subject if not body else f"{subject}\n\n{body}"
            message = tx.clean(message)
            if len(message) < 12:
                continue
            # Merge commits and bot noise carry no authorial signal.
            if subject.startswith(("Merge branch", "Merge pull request",
                                   "Merge remote-tracking")):
                continue
            seen += 1
            if seen >= ctx.max_records_per_source:
                return
            yield Record(
                source="git_authored",
                kind="code",
                authorship="self",
                doc_category="written_by_user",
                sensitivity="low",
                text=message,
                title=subject[:200] or None,
                author=author,
                created_at=when,
                app="git",
                path=relpath(repo),
                meta={"repo": repo.name, "sha": sha[:12], "has_body": bool(body)},
            )


def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10,
                       check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _git_identities() -> Set[str]:
    """The emails this user commits under, so we only take their own commits."""
    out: Set[str] = set()
    for scope in ("--global", "--system"):
        try:
            r = subprocess.run(["git", "config", scope, "user.email"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                out.add(r.stdout.strip().lower())
        except (OSError, subprocess.SubprocessError):
            continue
    env = os.environ.get("GIT_AUTHOR_EMAIL") or os.environ.get("EMAIL")
    if env:
        out.add(env.strip().lower())
    return out


def _find_repos(roots: List[Path], limit: int = 300) -> List[Path]:
    repos: List[Path] = []
    for root in roots:
        root = Path(root)
        if (root / ".git").exists():
            repos.append(root)
            continue
        if not root.is_dir():
            continue
        # Repos are usually 1-3 levels below a code root; do not walk deeper.
        for depth in ("*/.git", "*/*/.git", "*/*/*/.git"):
            for git_dir in root.glob(depth):
                repos.append(git_dir.parent)
                if len(repos) >= limit:
                    return repos
    # Dedupe while preserving order.
    seen, out = set(), []
    for r in repos:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _log(repo: Path, identities: Set[str], ctx: Context) -> Iterator[tuple]:
    fmt = _GIT_FIELD.join(["%H", "%aI", "%aE", "%aN", "%s", "%b"]) + _GIT_SEP
    cmd = ["git", "-C", str(repo), "log", f"--pretty=format:{fmt}",
           "--no-merges", "-n", "4000"]
    # Restrict to this user's commits when we know who they are; otherwise take
    # everything and let the author field record it.
    for ident in identities:
        cmd += [f"--author={ident}"]
    if ctx.since:
        cmd += [f"--since={ctx.since[:10]}"]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           errors="replace")
    except (OSError, subprocess.SubprocessError):
        return
    if r.returncode != 0:
        return

    for chunk in r.stdout.split(_GIT_SEP):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        fields = chunk.split(_GIT_FIELD)
        if len(fields) < 6:
            continue
        sha, when, email, name, subject, body = fields[:6]
        yield subject.strip(), body.strip(), parse_date(when), name or email, sha
