"""Command-line interface.

    nybble scan                what personal data exists on this machine
    nybble sources            the full catalog, and how to get each one
    nybble consent            approve sources (--yes-all approves everything)
    nybble collect            read the approved sources into JSONL
    nybble taste              infer what you like from what you kept
    nybble status             what has been collected so far
    nybble bundle             package the corpus for fine-tuning
    nybble revoke             delete collected data and approvals
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__, consent, platform_paths as pp, registry
from .consent import ConsentError, Ledger, discover
from .registry import GROUPS
from .sources import Context
from .writer import Writer, iter_records


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except ConsentError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nybble",
        description="Import your own behavioral data, with your permission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"nybble {__version__}")
    parser.add_argument("--home", type=Path, default=None,
                        help="Data directory (default: ~/.nybble)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("scan", help="Show what data exists on this machine")
    p.add_argument("--all", action="store_true",
                   help="Include sources that were not found")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sources", help="List the full source catalog")
    p.add_argument("--category", help="Filter by plan-doc category")
    p.add_argument("--exports", action="store_true",
                   help="Only sources that need a platform export, with steps")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("consent", help="Approve sources for collection")
    p.add_argument("--yes-all", action="store_true",
                   help="Approve every source found, in one confirmation")
    p.add_argument("--allow-llm", action="store_true",
                   help="Also grant off-device taste inference")
    p.add_argument("--no-redact", action="store_true",
                   help="Do not scrub emails/phones/cards (secrets are always scrubbed)")
    p.set_defaults(func=cmd_consent)

    p = sub.add_parser("collect", help="Read approved sources into JSONL")
    p.add_argument("--since", help="Only records after this date (YYYY-MM-DD)")
    p.add_argument("--only", nargs="+", metavar="SOURCE",
                   help="Restrict to specific approved source ids")
    p.add_argument("--max-per-source", type=int, default=200_000)
    p.add_argument("--transcribe", action="store_true",
                   help="Transcribe local audio with Whisper (slow, local)")
    p.add_argument("--whisper-model", default="base")
    p.add_argument("--ocr", action="store_true",
                   help="OCR screenshots (needs tesseract)")
    p.add_argument("--archives", type=Path,
                   help="Directory holding platform export bundles")
    p.add_argument("--fresh", action="store_true",
                   help="Discard previously collected records first")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("taste", help="Infer what you like from what you kept")
    p.add_argument("--limit", type=int, default=1200,
                   help="How many pieces of evidence to reason over")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--local", action="store_true",
                   help="Force the local profiler even if off-device is approved")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_taste)

    p = sub.add_parser("status", help="Summarize what has been collected")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("bundle", help="Package the corpus for fine-tuning")
    p.add_argument("--out", type=Path, help="Output file (default: <home>/bundle.jsonl)")
    p.add_argument("--min-chars", type=int, default=200)
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("revoke", help="Delete collected data and approvals")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation")
    p.set_defaults(func=cmd_revoke)

    return parser


def _root(args) -> Path:
    return args.home or consent.default_root()


# ===========================================================================
# scan
# ===========================================================================

def cmd_scan(args) -> int:
    found = discover(include_missing=args.all)

    if args.json:
        print(json.dumps([
            {
                "id": a.source.id,
                "name": a.source.name,
                "group": a.source.group,
                "doc_category": a.source.doc_category,
                "sensitivity": a.source.sensitivity,
                "acquisition": a.source.acquisition,
                "present": a.present,
                "locations": [str(p) for p in a.found[:10]],
            }
            for a in found
        ], indent=2))
        return 0

    present = [a for a in found if a.present]
    print(f"\nNybble scan — {pp.PLATFORM}\n")
    if not present:
        print("No supported sources found on this machine.")
        print("Most social and messaging data needs a platform export first:")
        print("  nybble sources --exports\n")
        return 0

    by_group = {}
    for a in present:
        by_group.setdefault(a.source.group, []).append(a)

    for gid, gname in GROUPS.items():
        items = by_group.get(gid)
        if not items:
            continue
        print(f"{gname}")
        for a in items:
            flag = "!" if a.source.sensitivity == "high" else " "
            extra = ""
            if a.source.needs_os_permission:
                extra = "  [needs OS permission]"
            elif a.source.needs:
                extra = f"  [pip install 'nybble[{','.join(a.source.needs)}]']"
            print(f"  {flag} {a.source.id:<24} {a.summary()}{extra}")
        print()

    missing = [a for a in found if not a.present] if args.all else []
    if missing:
        print("Not found:")
        for a in missing:
            print(f"    {a.source.id:<24} {a.source.acquisition}")
        print()

    print(f"{len(present)} source(s) available. Next: nybble consent")
    return 0


# ===========================================================================
# sources
# ===========================================================================

def cmd_sources(args) -> int:
    sources = registry.SOURCES
    if args.category:
        sources = [s for s in sources if s.doc_category == args.category]
    if args.exports:
        sources = [s for s in sources if s.acquisition == "export"]

    if args.json:
        print(json.dumps([
            {
                "id": s.id, "name": s.name, "group": s.group,
                "doc_category": s.doc_category, "authorship": s.authorship,
                "sensitivity": s.sensitivity, "acquisition": s.acquisition,
                "platforms": list(s.platforms), "what": s.what,
                "export_hint": s.export_hint, "needs": list(s.needs),
            } for s in sources
        ], indent=2))
        return 0

    labels = {
        "written_by_user": "1. Essays / content written by you",
        "messages": "2. Text messages",
        "transcripts": "3. Transcriptions of audio interactions",
        "liked_writing": "4. Writing of others that you like",
        "liked_content": "5. Content of others that you like",
        "social_preferences": "6. Preferences saved by social media",
        "context": "+  Context the plan doc did not list",
    }

    grouped = {}
    for s in sources:
        grouped.setdefault(s.doc_category, []).append(s)

    print(f"\nNybble source catalog — {len(sources)} sources\n")
    for category, label in labels.items():
        items = grouped.get(category)
        if not items:
            continue
        print(f"{label}")
        print("-" * len(label))
        for s in items:
            plat = "" if len(s.platforms) == 3 else f" [{'/'.join(s.platforms)}]"
            print(f"  {s.id:<24} {s.name}{plat}")
            print(f"  {'':<24} {s.what}")
            if args.exports and s.export_hint:
                print(f"  {'':<24} → {s.export_hint}")
            print()
    return 0


# ===========================================================================
# consent
# ===========================================================================

def cmd_consent(args) -> int:
    root = _root(args)
    ledger = consent.run_interactive(
        discover(), root=root, yes_all=args.yes_all)

    changed = False
    if args.allow_llm and not ledger.off_device_llm:
        ledger.off_device_llm = True
        changed = True
    if args.no_redact and ledger.redact:
        ledger.redact = False
        changed = True
    if changed:
        ledger.save(root)

    if ledger.sources:
        print("\nNext: nybble collect")
    return 0


# ===========================================================================
# collect
# ===========================================================================

def cmd_collect(args) -> int:
    root = _root(args)
    ledger = consent.require(root)

    approved = list(ledger.approved)
    if args.only:
        unknown = [s for s in args.only if s not in ledger.approved]
        if unknown:
            print(f"Not approved (run `nybble consent` first): {', '.join(unknown)}",
                  file=sys.stderr)
            return 2
        approved = args.only

    if args.fresh:
        records_dir = root / "records"
        if records_dir.exists():
            shutil.rmtree(records_dir)

    ctx = Context(
        since=_normalize_since(args.since),
        max_records_per_source=args.max_per_source,
        archives_dir=args.archives,
        transcribe_audio=args.transcribe,
        whisper_model=args.whisper_model,
        ocr=args.ocr,
    )

    print(f"\nCollecting {len(approved)} approved source(s) → {root}")
    if ledger.redact:
        print("Redaction on: emails, phones, cards, and secrets are scrubbed.")
    print()

    started = time.time()
    with Writer(root, redact=ledger.redact) as writer:
        for source_id in sorted(approved):
            source = registry.BY_ID.get(source_id)
            if source is None:
                continue
            if not source.available_on_platform():
                continue

            paths = source.candidates()
            if not paths:
                print(f"  {source_id:<24} nothing found")
                continue

            print(f"  {source_id:<24} ", end="", flush=True)
            try:
                collector = source.resolve_collector()
            except (ImportError, AttributeError) as exc:
                print(f"collector unavailable ({exc})")
                continue

            result = writer.write_all(source_id, collector(paths, ctx))
            result.paths_seen = len(paths)
            if result.error:
                print(f"error: {result.error}")
            else:
                print(f"{result.records:>7,} records  "
                      f"{result.chars / 1_000_000:>6.1f}M chars  "
                      f"{result.seconds:>5.1f}s")

        manifest_path = writer.finalize(pp.PLATFORM)
        manifest = writer.manifest
        redactions = writer.redactor.report()

    elapsed = time.time() - started
    print(f"\n{manifest.total_records:,} records, "
          f"{manifest.total_chars / 1_000_000:.1f}M characters in {elapsed:.0f}s")
    if redactions:
        total = sum(redactions.values())
        top = ", ".join(f"{k.lower()} {v}" for k, v in list(redactions.items())[:5])
        print(f"Redacted {total:,} items ({top})")
    print(f"Manifest: {manifest_path}")
    print("\nNext: nybble taste")
    return 0


def _normalize_since(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value if "T" in value else f"{value}T00:00:00+00:00"


# ===========================================================================
# taste
# ===========================================================================

def cmd_taste(args) -> int:
    from .taste import build_pool, infer
    from .taste.pool import pool_stats

    root = _root(args)
    ledger = consent.require(root)

    if not (root / "records").exists():
        print("Nothing collected yet. Run `nybble collect` first.", file=sys.stderr)
        return 2

    print("\nPooling affinity evidence…")
    candidates = build_pool(root, limit=args.limit)
    if not candidates:
        print("No affinity evidence found.")
        print("The sources that carry it are bookmarks, highlights, saved "
              "articles, likes, and platform interest exports — approve those "
              "in `nybble consent`, or add exports and re-collect.")
        return 1

    stats = pool_stats(candidates)
    print(f"  {stats['candidates']:,} candidates from "
          f"{len(stats['by_source'])} source(s), "
          f"mean weight {stats['mean_weight']}")
    for source, count in list(stats["by_source"].items())[:8]:
        print(f"    {source:<24} {count:>6,}")

    allow = ledger.off_device_llm and not args.local
    print(f"\nInferring taste ({'hosted model' if allow else 'local profiler'})…")

    def progress(done: int, total: int) -> None:
        print(f"\r  batch {done}/{total}", end="", flush=True)

    profile = infer(candidates, allow_off_device=allow, model=args.model,
                    progress=progress if allow else None)
    if allow:
        print()

    path = profile.save(root / "taste_profile.json")

    if args.json:
        print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"\nBackend: {profile.backend}"
          + (f" ({profile.model})" if profile.model else ""))
    if profile.summary:
        print(f"\n{profile.summary}\n")

    if profile.topics:
        print("Topics:")
        for topic in profile.topics[:15]:
            conf = topic.get("confidence", 0)
            print(f"  {conf:>4.2f}  {topic['topic']}"
                  f"  ({topic.get('evidence_count', 0)} items)")
        print()

    if profile.styles:
        print("Style affinities:")
        for style in profile.styles[:8]:
            print(f"  - {style}")
        print()

    if profile.sources:
        print("Most-kept sources:")
        for source in profile.sources[:10]:
            count = source.get("liked_items") or source.get("weight")
            print(f"  {source['name']}  ({count})")
        print()

    if profile.labeled:
        liked = sum(1 for item in profile.labeled if item["liked"])
        print(f"Labeled {len(profile.labeled):,} items; "
              f"{liked:,} judged genuine affinity "
              f"({liked / len(profile.labeled):.0%})")

    for warning in profile.warnings:
        print(f"\nnote: {warning}")

    print(f"\nProfile: {path}")
    return 0


# ===========================================================================
# status
# ===========================================================================

def cmd_status(args) -> int:
    root = _root(args)
    ledger = Ledger.load(root)
    manifest_path = root / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except ValueError:
            manifest = None

    if args.json:
        print(json.dumps({
            "root": str(root),
            "consent": ledger.__dict__ if ledger else None,
            "manifest": manifest,
        }, indent=2, default=str))
        return 0

    print(f"\nNybble home: {root}")
    if ledger is None:
        print("No consent on record. Run `nybble consent`.")
        return 0

    print(f"Consent granted {ledger.age_days():.0f} days ago"
          + (" (STALE — re-run `nybble consent`)" if ledger.is_stale() else ""))
    print(f"Approved sources: {len(ledger.sources)}"
          + ("  [blanket approval]" if ledger.blanket else ""))
    print(f"Redaction: {'on' if ledger.redact else 'off'}")
    print(f"Off-device inference: {'allowed' if ledger.off_device_llm else 'declined'}")

    if manifest:
        print(f"\nLast collection: {manifest.get('total_records', 0):,} records, "
              f"{manifest.get('total_chars', 0) / 1_000_000:.1f}M characters")
        results = sorted(manifest.get("results", []),
                         key=lambda r: -(r.get("records") or 0))
        for result in results[:20]:
            if not result.get("records") and not result.get("error"):
                continue
            note = f"  error: {result['error']}" if result.get("error") else ""
            print(f"  {result['source_id']:<24} {result.get('records', 0):>8,}{note}")
    else:
        print("\nNothing collected yet. Run `nybble collect`.")

    profile_path = root / "taste_profile.json"
    if profile_path.exists():
        print(f"\nTaste profile: {profile_path}")
    return 0


# ===========================================================================
# bundle
# ===========================================================================

def cmd_bundle(args) -> int:
    root = _root(args)
    out = args.out or (root / "bundle.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    by_category: dict = {}
    with out.open("w", encoding="utf-8") as fh:
        for record in iter_records(root):
            text = (record.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            category = record.get("doc_category", "context")
            by_category[category] = by_category.get(category, 0) + 1
            fh.write(json.dumps({
                "text": text,
                "source": record.get("source"),
                "authorship": record.get("authorship"),
                "category": category,
                "created_at": record.get("created_at"),
                "title": record.get("title"),
                "author": record.get("author"),
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written:,} records to {out}")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {category:<22} {count:>8,}")
    return 0


# ===========================================================================
# revoke
# ===========================================================================

def cmd_revoke(args) -> int:
    root = _root(args)
    if not root.exists():
        print(f"Nothing to delete at {root}")
        return 0

    targets = [p for p in (root / "records", root / "consent.json",
                           root / "manifest.json", root / "taste_profile.json",
                           root / "bundle.jsonl") if p.exists()]
    if not targets:
        print(f"Nothing to delete at {root}")
        return 0

    print(f"\nThis permanently deletes, from {root}:")
    for path in targets:
        print(f"  {path.name}")

    if not args.yes:
        try:
            if input("\nType 'delete' to confirm: ").strip().lower() != "delete":
                print("Cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1

    for path in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass
    print("Deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
