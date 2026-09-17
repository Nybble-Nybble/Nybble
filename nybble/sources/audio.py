"""Speech → text.

Two paths. Transcripts that already exist (Zoom, Teams, Granola, Otter) are just
parsed. Raw audio is transcribed locally with faster-whisper; the audio itself
never leaves the machine and is never uploaded, including when off-device taste
inference is enabled.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .. import textextract as tx
from ..schema import Record
from . import Context
from .base import file_times, relpath, walk_files

TRANSCRIPT_SUFFIXES = {".vtt", ".srt", ".txt", ".json"}
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus",
                  ".mp4", ".mov", ".m4v", ".webm"}

# Files whose names say "this is a transcript" — used to avoid pulling every
# stray .txt in a Zoom folder.
_TRANSCRIPT_NAME = re.compile(
    r"(transcript|captions?|subtitle|\.vtt$|\.srt$|meeting|recording|notes)", re.I)


# ===========================================================================
# Existing transcripts
# ===========================================================================

def collect_existing_transcripts(paths: List[Path], ctx: Context) -> Iterator[Record]:
    seen = 0
    for path in walk_files(paths, suffixes=TRANSCRIPT_SUFFIXES, max_files=20_000):
        if seen >= ctx.max_records_per_source:
            return
        suffix = path.suffix.lower()
        if suffix in (".txt", ".json") and not _TRANSCRIPT_NAME.search(str(path)):
            continue

        parsed = _parse_transcript(path)
        if not parsed:
            continue
        text, speakers = parsed
        if len(text) < 200:
            continue

        created, modified = file_times(path)
        if ctx.since and (modified or "") < ctx.since:
            continue
        seen += 1
        yield Record(
            source="existing_transcripts",
            kind="transcript",
            authorship="mixed",
            doc_category="transcripts",
            sensitivity="high",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app=_guess_app(path),
            meta={"speakers": speakers[:30], "format": suffix.lstrip("."),
                  "pre_existing": True},
        )


def _guess_app(path: Path) -> str:
    lowered = str(path).lower()
    for needle, label in (("zoom", "Zoom"), ("granola", "Granola"),
                          ("otter", "Otter"), ("fathom", "Fathom"),
                          ("fireflies", "Fireflies"), ("teams", "Teams"),
                          ("meet", "Google Meet"), ("rev", "Rev"),
                          ("descript", "Descript")):
        if needle in lowered:
            return label
    return "transcript"


def _parse_transcript(path: Path) -> Optional[Tuple[str, List[str]]]:
    raw = tx.read_text(path)
    if not raw:
        return None
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return _parse_vtt(raw)
    if suffix == ".srt":
        return _parse_srt(raw)
    if suffix == ".json":
        return _parse_json_transcript(raw)
    return _parse_plain(raw)


_TIMESTAMP = re.compile(
    r"^\s*(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*"
    r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}")
_SPEAKER = re.compile(r"^\s*(?:<v\s+([^>]+)>|([A-Z][\w .'-]{1,40}?)\s*:)\s*(.*)$")
_CUE_INDEX = re.compile(r"^\s*\d+\s*$")


def _collapse_cues(lines: List[str]) -> Tuple[str, List[str]]:
    """Merge caption cues into continuous speaker turns."""
    turns: List[Tuple[Optional[str], List[str]]] = []
    speakers: List[str] = []
    last_line = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        speaker, body = None, line
        m = _SPEAKER.match(line)
        if m:
            speaker = (m.group(1) or m.group(2) or "").strip() or None
            body = (m.group(3) or "").strip()
        body = re.sub(r"</?v[^>]*>", "", body).strip()
        if not body or body == last_line:  # VTT often repeats a rolling caption
            continue
        last_line = body
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if turns and turns[-1][0] == speaker:
            turns[-1][1].append(body)
        else:
            turns.append((speaker, [body]))

    out = []
    for speaker, parts in turns:
        joined = " ".join(parts)
        out.append(f"{speaker}: {joined}" if speaker else joined)
    return tx.clean("\n".join(out)), speakers


def _parse_vtt(raw: str) -> Tuple[str, List[str]]:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.startswith(("WEBVTT", "NOTE", "STYLE", "REGION"))
                or _TIMESTAMP.match(line) or _CUE_INDEX.match(line)):
            continue
        lines.append(line)
    return _collapse_cues(lines)


def _parse_srt(raw: str) -> Tuple[str, List[str]]:
    lines = [ln for ln in raw.splitlines()
             if not _TIMESTAMP.match(ln) and not _CUE_INDEX.match(ln)]
    return _collapse_cues(lines)


def _parse_plain(raw: str) -> Tuple[str, List[str]]:
    # Strip leading "00:12:33" timestamps that plain-text exports prefix.
    lines = [re.sub(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*", "", ln)
             for ln in raw.splitlines()]
    return _collapse_cues(lines)


def _parse_json_transcript(raw: str) -> Optional[Tuple[str, List[str]]]:
    """Handle the common shapes: Whisper output, and generic segment lists."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None

    segments = None
    if isinstance(data, dict):
        if isinstance(data.get("text"), str) and len(data["text"]) > 200:
            return tx.clean(data["text"]), []
        for key in ("segments", "transcript", "results", "utterances", "monologues"):
            if isinstance(data.get(key), list):
                segments = data[key]
                break
    elif isinstance(data, list):
        segments = data
    if not segments:
        return None

    lines, speakers = [], []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text") or seg.get("transcript") or seg.get("content")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if not text:
            continue
        speaker = seg.get("speaker") or seg.get("speaker_label") or seg.get("name")
        if speaker and speaker not in speakers:
            speakers.append(str(speaker))
        lines.append(f"{speaker}: {text}" if speaker else str(text))
    if not lines:
        return None
    return tx.clean("\n".join(lines)), speakers


# ===========================================================================
# Local transcription
# ===========================================================================

def collect_voice_memos(paths: List[Path], ctx: Context) -> Iterator[Record]:
    yield from _transcribe_dirs(paths, "voice_memos", "Voice Memos", "self", ctx)


def collect_local_media(paths: List[Path], ctx: Context) -> Iterator[Record]:
    yield from _transcribe_dirs(paths, "local_audio", "recording", "mixed", ctx)


def _transcribe_dirs(paths: List[Path], source_id: str, app: str,
                     authorship: str, ctx: Context) -> Iterator[Record]:
    if not ctx.transcribe_audio:
        return
    model = _load_whisper(ctx.whisper_model)
    if model is None:
        return

    seen = 0
    for path in walk_files(paths, suffixes=AUDIO_SUFFIXES, max_files=2_000):
        if seen >= min(ctx.max_records_per_source, 500):
            return
        try:
            if path.stat().st_size < 20_000:  # under ~1s of audio
                continue
        except OSError:
            continue

        # Never re-transcribe something that already has a transcript beside it.
        if any((path.with_suffix(s)).exists() for s in (".vtt", ".srt", ".txt")):
            continue

        text = _transcribe(model, path)
        if not text or len(text) < 100:
            continue
        created, modified = file_times(path)
        seen += 1
        yield Record(
            source=source_id,
            kind="transcript",
            authorship=authorship,
            doc_category="transcripts",
            sensitivity="high",
            text=text,
            title=path.stem,
            created_at=created,
            modified_at=modified,
            path=relpath(path),
            app=app,
            meta={"transcribed_locally": True, "model": ctx.whisper_model,
                  "media_format": path.suffix.lstrip(".")},
        )


_MODEL_CACHE = {}


def _load_whisper(size: str):
    if size in _MODEL_CACHE:
        return _MODEL_CACHE[size]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        _MODEL_CACHE[size] = None
        return None
    try:
        model = WhisperModel(size, device="auto", compute_type="int8")
    except Exception:
        try:
            model = WhisperModel(size, device="cpu", compute_type="int8")
        except Exception:
            model = None
    _MODEL_CACHE[size] = model
    return model


def _transcribe(model, path: Path) -> Optional[str]:
    try:
        segments, _info = model.transcribe(str(path), vad_filter=True,
                                           beam_size=1)
        return tx.clean(" ".join(seg.text for seg in segments))
    except Exception:
        return None
