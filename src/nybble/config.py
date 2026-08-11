"""Typed configuration and secret loading for Nybble."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_TINKER_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_RENDERER = "tml_v0"


def load_secret(name: str, env_file: Path | None = None) -> str | None:
    """Read a secret from the environment, then a simple NAME=value file."""
    value = os.environ.get(name)
    if value:
        return value

    path = env_file or Path.cwd() / ".env"
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == name:
            return raw_value.strip().strip("\"'") or None
    return None


def require_secret(name: str, env_file: Path | None = None) -> str:
    value = load_secret(name, env_file)
    if not value:
        location = env_file or Path.cwd() / ".env"
        raise RuntimeError(f"{name} is missing; export it or add it to {location}")
    return value


@dataclass(frozen=True)
class GeminiConfig:
    model: str = DEFAULT_GEMINI_MODEL
    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("retry delays cannot be negative")


@dataclass(frozen=True)
class LabelingConfig:
    chunk_size: int = 10
    max_gap_seconds: float = 30.0
    fps: float | None = None
    caption_workers: int = 4

    def __post_init__(self) -> None:
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.caption_workers < 1:
            raise ValueError("caption_workers must be at least 1")


@dataclass(frozen=True)
class PowerNapConfig:
    model: str = DEFAULT_TINKER_MODEL
    renderer: str = DEFAULT_RENDERER
    past_len: int = 50
    future_len: int = 8
    max_images: int = 5
    batch_size: int = 8
    group_size: int = 4
    lora_rank: int = 32
    learning_rate: float = 5e-5
    max_tokens: int = 512
    temperature: float = 1.0
    inkling_effort: float = 0.0
    checkpoint_every: int = 2
    retrieval_top_k: int = 10
    retrieval_mmr_k: int = 5
    retrieval_max_tokens: int = 4096
    retrieval_mmr_alpha: float = 0.5
    retrieval_time_decay: float = 0.5
    retrieval_min_accuracy: float = 0.5
    accuracy_weight: float = 0.5
    formatting_weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.renderer.strip():
            raise ValueError("model and renderer must not be empty")
        positive_ints = {
            "past_len": self.past_len,
            "future_len": self.future_len,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "lora_rank": self.lora_rank,
            "max_tokens": self.max_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_mmr_k": self.retrieval_mmr_k,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if self.max_images < 0:
            raise ValueError("max_images must be nonnegative")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be nonnegative")
        if type(self.retrieval_max_tokens) is not int or self.retrieval_max_tokens < 0:
            raise ValueError("retrieval_max_tokens must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 <= self.inkling_effort < 1:
            raise ValueError("inkling_effort must be in [0, 1)")
        if not 0 <= self.retrieval_mmr_alpha <= 1:
            raise ValueError("retrieval_mmr_alpha must be between 0 and 1")
        if not math.isfinite(self.retrieval_time_decay) or self.retrieval_time_decay < 0:
            raise ValueError("retrieval_time_decay must be finite and nonnegative")
        if not 0 <= self.retrieval_min_accuracy <= 1:
            raise ValueError("retrieval_min_accuracy must be between 0 and 1")
        if not math.isfinite(self.accuracy_weight) or not math.isfinite(self.formatting_weight):
            raise ValueError("reward weights must be finite")
        if self.accuracy_weight < 0 or self.formatting_weight < 0:
            raise ValueError("reward weights cannot be negative")
        if self.accuracy_weight + self.formatting_weight <= 0:
            raise ValueError("at least one reward weight must be positive")
