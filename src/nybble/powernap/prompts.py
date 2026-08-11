"""Prompts and deterministic formatting shared by training and inference."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = (
    "You predict the next actions in a first-person activity stream. "
    "Use only the observed action journal and retrieved historical patterns. "
    "Do not invent observations that are not present."
)

THINK_INSTRUCTION = """\
Analyze the observed sequence and identify the strongest short-horizon patterns.
Output only a <rationale>...</rationale> block. Do not output actions yet."""

REVISE_INSTRUCTION = """\
Re-evaluate the rationale using the historical context below. Historical examples may be
irrelevant, so keep only evidence that fits the current sequence.

{retrieved}

Output only a <revise>...</revise> block. Do not output actions yet."""

ACTIONS_INSTRUCTION = """\
Predict exactly {future_len} next observable wearer actions in chronological order.
Output only one <actions> block containing exactly {future_len} nonempty <action> tags.
Keep each action concrete, concise, and independently observable."""


def actions_xml(actions: Iterable[str]) -> str:
    cleaned = [" ".join(action.split()) for action in actions]
    return (
        "<actions>\n"
        + "\n".join(f"  <action>{html.escape(a)}</action>" for a in cleaned)
        + "\n</actions>"
    )


def validate_image_paths(image_paths: Sequence[str]) -> tuple[str, ...]:
    """Validate local Inkling inputs before any remote training request."""

    validated = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PowerNap context image not found: {path}")
        try:
            with Image.open(path) as image:
                image_format = image.format
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid PowerNap context image: {path}") from exc
        if image_format not in {"JPEG", "PNG"}:
            raise ValueError(
                f"Inkling accepts PNG or JPEG context images, got {image_format}: {path}"
            )
        validated.append(str(path))
    return tuple(validated)


def build_think_messages(context: str, image_paths: Sequence[str] = ()) -> list[dict]:
    user_text = "Observed action journal:\n\n" + context + "\n\n" + THINK_INSTRUCTION
    content: str | list[dict]
    if image_paths:
        content = [
            {"type": "text", "text": user_text},
            *({"type": "image", "image": path} for path in image_paths),
        ]
    else:
        content = user_text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def append_revise(messages: Sequence[dict], rationale: str, retrieved: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": rationale},
        {
            "role": "user",
            "content": REVISE_INSTRUCTION.format(retrieved=retrieved or "(none)"),
        },
    ]


def append_actions(messages: Sequence[dict], revision: str, future_len: int) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": revision},
        {"role": "user", "content": ACTIONS_INSTRUCTION.format(future_len=future_len)},
    ]


def retrieval_query(rationale: str, context: str) -> str:
    return rationale.strip() + "\n\n" + context.strip()
