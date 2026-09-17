"""Collectors.

Every collector is a generator with the signature:

    def collect_x(paths: list[Path], ctx: Context) -> Iterator[Record]

It must never raise for a single bad file — skip and continue. The writer
catches anything that escapes and records it in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Context:
    """Run-wide knobs handed to every collector."""

    # Skip records older than this ISO date, when set.
    since: Optional[str] = None
    # Per-source cap so one giant mail spool cannot dominate the corpus.
    max_records_per_source: int = 200_000
    # Ignore text shorter than this (in characters) for prose-like sources.
    min_chars: int = 40
    # Where the user's export archives live, if not auto-discovered.
    archives_dir: Optional[Path] = None
    # Transcribe audio found on disk (slow, needs faster-whisper).
    transcribe_audio: bool = False
    whisper_model: str = "base"
    # OCR screenshots (needs pytesseract).
    ocr: bool = False
    verbose: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)

    def note(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n
