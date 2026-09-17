"""Consent ledger.

Nothing is read from disk until it appears in the ledger. The ledger is a plain
JSON file the user can open, audit, and edit by hand; `nybble revoke` deletes it
along with the collected data.

Two separate grants exist, and one never implies the other:

  collection  — read these sources from this machine
  off_device  — send excerpts to a hosted LLM for taste inference

A user who says "yes to all" for collection is still asked about off_device,
because that one leaves the machine.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import registry
from .registry import GROUPS, GROUP_NOTES, Source

LEDGER_VERSION = 1
# Re-ask after this long so a one-time approval does not become permanent.
LEDGER_MAX_AGE_DAYS = 90


def default_root() -> Path:
    env = os.environ.get("NYBBLE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".nybble"


@dataclass
class Ledger:
    version: int = LEDGER_VERSION
    granted_at: float = field(default_factory=time.time)
    machine: str = ""
    user: str = ""
    # source ids approved for collection
    sources: List[str] = field(default_factory=list)
    # separate, explicit grant for sending excerpts off-device
    off_device_llm: bool = False
    # user asked to blanket-approve; recorded so `scan` can auto-include
    # newly-discovered sources on later runs without re-prompting.
    blanket: bool = False
    redact: bool = True
    notes: str = ""

    @property
    def approved(self) -> Set[str]:
        return set(self.sources)

    def age_days(self) -> float:
        return (time.time() - self.granted_at) / 86400.0

    def is_stale(self) -> bool:
        return self.age_days() > LEDGER_MAX_AGE_DAYS

    def allows(self, source_id: str) -> bool:
        return source_id in self.approved

    # --- persistence ---

    @classmethod
    def path(cls, root: Optional[Path] = None) -> Path:
        return (root or default_root()) / "consent.json"

    @classmethod
    def load(cls, root: Optional[Path] = None) -> Optional["Ledger"]:
        p = cls.path(root)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        known = {f for f in cls.__dataclass_fields__}  # tolerate older/newer files
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, root: Optional[Path] = None) -> Path:
        p = self.path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.machine = self.machine or f"{platform.node() or socket.gethostname()}"
        self.user = self.user or _current_user()
        p.write_text(json.dumps(asdict(self), indent=2), "utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
        return p


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


# --- discovery ----------------------------------------------------------------

@dataclass
class Availability:
    source: Source
    found: List[Path] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return bool(self.found)

    def summary(self) -> str:
        if not self.found:
            return "not found"
        head = str(self.found[0])
        h = str(Path.home())
        if head.startswith(h):
            head = "~" + head[len(h):]
        extra = f" (+{len(self.found) - 1} more)" if len(self.found) > 1 else ""
        return head + extra


def discover(include_missing: bool = False) -> List[Availability]:
    """Probe every registry source against this machine.

    Only checks for existence — no file contents are read during discovery, so
    a scan is safe to run before any consent is given.
    """
    out: List[Availability] = []
    for src in registry.for_platform():
        avail = Availability(source=src, found=src.candidates())
        if avail.present or include_missing:
            out.append(avail)
    return out


# --- interactive approval ------------------------------------------------------

BANNER = """\
Nybble reads personal data from this computer to personalize a model for you.

  • Everything runs locally. Nothing is uploaded unless you separately approve
    off-device inference at the end.
  • You choose what is read. Nothing is touched without approval.
  • Your approvals are written to {ledger}, in plain text you can edit.
  • `nybble revoke` deletes the collected data and the approvals.
"""


def _prompt(msg: str, valid: Set[str], default: str) -> str:
    """Read one keystroke-ish answer, defaulting on EOF (non-tty / piped input)."""
    try:
        raw = input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw if raw in valid else default


def run_interactive(
    availabilities: List[Availability],
    root: Optional[Path] = None,
    yes_all: bool = False,
    assume_no_tty_default: str = "n",
) -> Ledger:
    """Walk the user through approval and return a saved ledger.

    With `yes_all`, every discovered source is approved in one confirmation
    instead of one prompt per source. The off-device LLM grant is still asked
    separately — a blanket yes to reading local files is not a yes to uploading.
    """
    ledger = Ledger()
    print(BANNER.format(ledger=Ledger.path(root)))

    present = [a for a in availabilities if a.present]
    if not present:
        print("No supported data sources found on this machine.")
        print("Run `nybble sources` to see what Nybble looks for, and how to")
        print("request exports from platforms that require them.")
        ledger.save(root)
        return ledger

    # Show the full picture before asking for anything.
    _print_catalog(present)

    interactive = sys.stdin.isatty()

    if yes_all:
        print(f"\n--yes-all: approving all {len(present)} sources listed above.")
        if interactive:
            ans = _prompt("Type 'yes' to confirm, anything else to cancel: ",
                          {"yes", "y"}, "n")
            if ans not in ("yes", "y"):
                print("Cancelled. Nothing approved.")
                ledger.save(root)
                return ledger
        ledger.sources = [a.source.id for a in present]
        ledger.blanket = True
    elif not interactive:
        print("\nNo terminal attached and --yes-all not passed; approving nothing.")
        print("Re-run with --yes-all to approve everything non-interactively.")
        ledger.save(root)
        return ledger
    else:
        ledger.sources = _walk_groups(present)

    # Redaction: on by default, opt out explicitly.
    if interactive and not yes_all:
        ans = _prompt(
            "\nRedact emails, phone numbers, card numbers, and API keys from "
            "collected text? [Y/n]: ", {"y", "n", ""}, "y")
        ledger.redact = ans != "n"

    # The separate off-device grant.
    ledger.off_device_llm = _ask_off_device(interactive)

    path = ledger.save(root)
    print(f"\nApproved {len(ledger.sources)} source(s). Ledger: {path}")
    if not ledger.off_device_llm:
        print("Off-device inference declined — `nybble taste` will use the local "
              "heuristic profiler.")
    return ledger


def _print_catalog(present: List[Availability]) -> None:
    groups: Dict[str, List[Availability]] = {}
    for a in present:
        groups.setdefault(a.source.group, []).append(a)

    print(f"Found {len(present)} data source(s) on this machine:\n")
    for gid, gname in GROUPS.items():
        items = groups.get(gid)
        if not items:
            continue
        print(f"  {gname}")
        note = GROUP_NOTES.get(gid)
        if note:
            print(f"    note: {note}")
        for a in items:
            flag = "!" if a.source.sensitivity == "high" else " "
            print(f"    {flag} {a.source.name}")
            print(f"        {a.source.what}")
            print(f"        location: {a.summary()}")
        print()
    print("  ! = high sensitivity (contains other people's data, or yours at its "
          "most personal)")


def _walk_groups(present: List[Availability]) -> List[str]:
    """Per-group prompt with per-source drill-down."""
    groups: Dict[str, List[Availability]] = {}
    for a in present:
        groups.setdefault(a.source.group, []).append(a)

    approved: List[str] = []
    print("\nFor each group: [y] all of it  [n] none  [s] choose one by one  "
          "[A] yes to everything, stop asking")

    for gid, gname in GROUPS.items():
        items = groups.get(gid)
        if not items:
            continue
        ans = _prompt(f"\n{gname} — {len(items)} source(s). [y/n/s/A]: ",
                      {"y", "n", "s", "a", ""}, "n")
        if ans == "a":
            print("  → approving every remaining source.")
            return [a.source.id for a in present]
        if ans == "y":
            approved += [a.source.id for a in items]
        elif ans == "s":
            for a in items:
                sub = _prompt(f"    {a.source.name} ({a.summary()}) [y/N]: ",
                              {"y", "n", ""}, "n")
                if sub == "y":
                    approved.append(a.source.id)
    return approved


def _ask_off_device(interactive: bool) -> bool:
    print("\n" + "-" * 68)
    print("Optional: off-device taste inference.")
    print()
    print("Figuring out which writing and content you actually like needs a model")
    print("to read excerpts of what you saved. Nybble can do this two ways:")
    print()
    print("  local  — heuristic profiler, runs here, never uploads. Cruder.")
    print("  hosted — sends excerpts of saved/highlighted content to an LLM API.")
    print("           Your own messages and documents are never sent; only")
    print("           other-authored material you kept, capped and redacted.")
    if not interactive:
        print("\nNo terminal attached — defaulting to local-only.")
        return False
    ans = _prompt("\nAllow sending excerpts to a hosted LLM? [y/N]: ",
                  {"y", "n", ""}, "n")
    return ans == "y"


def require(root: Optional[Path] = None) -> Ledger:
    """Load the ledger for a collect/taste run, or explain how to create one."""
    ledger = Ledger.load(root)
    if ledger is None:
        raise ConsentError(
            "No consent on record. Run `nybble consent` (add --yes-all to "
            "approve everything found)."
        )
    if ledger.is_stale():
        raise ConsentError(
            f"Consent is {ledger.age_days():.0f} days old (limit "
            f"{LEDGER_MAX_AGE_DAYS}). Re-run `nybble consent` to confirm it "
            f"still reflects what you want."
        )
    if not ledger.sources:
        raise ConsentError(
            "Consent ledger exists but approves no sources. Run `nybble consent` "
            "again."
        )
    return ledger


class ConsentError(RuntimeError):
    pass
