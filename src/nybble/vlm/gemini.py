"""Gemini 3.6 Flash adapter for independent captions and structured grouping.

The Google Gen AI SDK is imported only when no client is injected. Tests and
offline pipelines can therefore use a small fake without installing the SDK or
creating a network client.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, ImageOps

from nybble.labeling.prompt import (
    FRAME_CAPTION_PROMPT,
    FRAME_CAPTION_PROMPT_VERSION,
    GROUPING_PROMPT_VERSION,
    build_grouping_prompt,
)
from nybble.labeling.schema import FrameLabel, GroupingResult, grouping_from_payload

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
MAX_IMAGE_SIDE = 1024
_T = TypeVar("_T")


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns an empty or non-JSON response."""


@dataclass(frozen=True)
class RetryPolicy:
    """A bounded exponential retry policy.

    ``max_attempts`` includes the initial request. Setting it to one disables
    retries. ``max_delay_seconds`` prevents a large retry count from producing an
    unbounded wait.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be nonnegative")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be nonnegative")


def _with_retries(
    operation: Callable[[], _T],
    policy: RetryPolicy,
    sleep: Callable[[float], None],
) -> _T:
    for attempt in range(policy.max_attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 >= policy.max_attempts:
                raise
            delay = min(
                policy.base_delay_seconds * (2**attempt),
                policy.max_delay_seconds,
            )
            if delay:
                sleep(delay)
    raise AssertionError("retry loop terminated without returning or raising")


def _read_env_key(name: str = "GEMINI_API_KEY") -> str | None:
    key = os.environ.get(name)
    if key:
        return key

    # Match the existing repository convention without importing api_vlm.py and
    # therefore without importing its OpenAI client at package import time.
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            candidate, separator, value = line.partition("=")
            if separator and candidate.strip() == name:
                return value.strip()
    return None


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Some fakes and future SDK versions may return a mapping-shaped response.
    if isinstance(response, dict):
        mapped = response.get("text")
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
    raise GeminiResponseError("Gemini returned no response text")


def _jpeg_inline_part(frame: Path) -> dict[str, Any]:
    with Image.open(frame) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": buffer.getvalue(),
        }
    }


class GeminiGenerator:
    """Small adapter over ``google.genai.Client.models.generate_content``."""

    def __init__(
        self,
        client: Any | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self._client = client
        self._client_lock = threading.Lock()
        self._api_key = api_key
        self.model = model
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            key = self._api_key or _read_env_key()
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY not found; export it or put GEMINI_API_KEY=... in .env"
                )
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is required for Gemini labeling; install project dependencies"
                ) from exc
            self._client = genai.Client(api_key=key)
            return self._client

    def generate_content(self, contents: Any, config: dict[str, Any]) -> Any:
        """Generate one response, retrying request failures with a hard bound."""

        return _with_retries(
            lambda: self.generate_content_once(contents, config),
            self.retry_policy,
            self._sleep,
        )

    def generate_content_once(self, contents: Any, config: dict[str, Any]) -> Any:
        """Make exactly one SDK request so parsing can share one retry boundary."""

        client = self._get_client()
        models = getattr(client, "models", client)
        return models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        """Return one schema-constrained JSON object with bounded parse retries."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary")
        config = {
            "temperature": 0,
            "max_output_tokens": 4000,
            "thinking_config": {"thinking_level": "LOW"},
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }

        def operation() -> dict:
            text = _response_text(self.generate_content_once(prompt, config))
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise GeminiResponseError("Gemini structured response was not JSON") from exc
            if not isinstance(payload, dict):
                raise GeminiResponseError("Gemini structured response must be a JSON object")
            return payload

        return _with_retries(operation, self.retry_policy, self._sleep)


class GeminiCaptioner:
    """Caption exactly one frame per Gemini request."""

    prompt_version = FRAME_CAPTION_PROMPT_VERSION

    def __init__(
        self,
        generator: GeminiGenerator | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        prompt: str = FRAME_CAPTION_PROMPT,
    ) -> None:
        if generator is not None and (client is not None or api_key is not None):
            raise ValueError("pass either generator or client/api_key, not both")
        self.generator = generator or GeminiGenerator(
            client=client,
            api_key=api_key,
            model=model,
            retry_policy=retry_policy,
            sleep=sleep,
        )
        self.prompt = prompt
        self.model = self.generator.model

    def caption_frame(self, frame: Path) -> str:
        frame_path = Path(frame)
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        contents = [
            {
                "role": "user",
                "parts": [
                    _jpeg_inline_part(frame_path),
                    {"text": self.prompt},
                ],
            }
        ]
        config = {
            "temperature": 0,
            "max_output_tokens": 2000,
            "thinking_config": {"thinking_level": "LOW"},
        }

        # Empty responses are response failures and receive the same bounded retry
        # treatment as transport failures.
        def operation() -> str:
            response = self.generator.generate_content_once(contents=contents, config=config)
            return _response_text(response)

        return _with_retries(operation, self.generator.retry_policy, self.generator._sleep)


_GROUPING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "caption": {"type": "string"},
                },
                "required": ["start_index", "end_index", "caption"],
            },
        },
        "dense_context": {"type": "string"},
    },
    "required": ["actions", "dense_context"],
}


class GeminiActionGrouper:
    """Group single-frame captions into a strict semantic action partition."""

    prompt_version = GROUPING_PROMPT_VERSION

    def __init__(
        self,
        generator: GeminiGenerator | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if generator is not None and (client is not None or api_key is not None):
            raise ValueError("pass either generator or client/api_key, not both")
        self.generator = generator or GeminiGenerator(
            client=client,
            api_key=api_key,
            model=model,
            retry_policy=retry_policy,
            sleep=sleep,
        )
        self.model = self.generator.model

    def group(self, frames: Sequence[FrameLabel]) -> GroupingResult:
        if not frames:
            return GroupingResult(actions=())
        prompt = build_grouping_prompt(frames)
        config = {
            "temperature": 0,
            "max_output_tokens": 4000,
            "thinking_config": {"thinking_level": "LOW"},
            "response_mime_type": "application/json",
            "response_json_schema": _GROUPING_RESPONSE_SCHEMA,
        }

        # Parsing and exact-partition validation happen inside this retry boundary.
        # A malformed structured response is never clamped or partially accepted.
        def operation() -> GroupingResult:
            response = self.generator.generate_content_once(contents=prompt, config=config)
            text = _response_text(response)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise GeminiResponseError("Gemini grouping response was not JSON") from exc
            return grouping_from_payload(payload, frame_count=len(frames))

        return _with_retries(operation, self.generator.retry_policy, self.generator._sleep)


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "GeminiActionGrouper",
    "GeminiCaptioner",
    "GeminiGenerator",
    "GeminiResponseError",
    "RetryPolicy",
]
