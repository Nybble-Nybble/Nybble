"""Hosted-API captioning, same job as ondevice_vlm.py but over the network.

caption_stream(frames) matches ondevice_vlm's signature so run_captions.py can
drive either, and both share prompt.py — so a local-vs-hosted comparison is a
genuine A/B on the model, not on the prompt.

Every provider here speaks the OpenAI chat-completions dialect, which is why
switching is a base_url + model swap rather than a new client. Adding one is a
single PROVIDERS entry; nothing else in this file changes.
"""

import base64
import io
import os

from openai import OpenAI
from PIL import Image, ImageOps

from prompt import CAPTION_PROMPT

# ponytail: same cap as ondevice_vlm so local and hosted captions are comparable.
# Cost note for Gemini specifically: it bills 258 tokens for images with BOTH
# sides <=384px, and tiles anything larger into 768px tiles at 258 tokens each —
# so dropping this to 384 cuts input cost ~4x. It also cuts the fine detail that
# reads labels off packaging. Measure before trading it away.
MAX_SIDE = 1024

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.6-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
    },
}


def _data_url(frame):
    """Any frame -> a base64 JPEG data URL, downscaled the same way as on-device."""
    img = frame if isinstance(frame, Image.Image) else Image.open(frame)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # aspect-preserving, downscale-only
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def caption_stream(frames, provider="gemini", model=None, max_tokens=600):
    """Yield one caption per frame. One image per call, matching ondevice_vlm.

    ponytail: no retry/concurrency layer. The SDK already retries 429s and 5xxs,
    and serial calls are fine for iterating on a few hundred frames. When you
    batch a real corpus, wrap this in a thread pool — the network is idle
    latency, not CPU, so concurrency is where the wall-clock win is.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; have {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise RuntimeError(f"{cfg['key_env']} is not set (needed for {provider})")
    client = OpenAI(base_url=cfg["base_url"], api_key=key)

    # Validation above runs eagerly; only the request loop is deferred, so a bad
    # provider or missing key fails at the call rather than on the first frame.
    def stream():
        for frame in frames:
            resp = client.chat.completions.create(
                model=model or cfg["model"],
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(frame)}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ]}],
            )
            yield (resp.choices[0].message.content or "").strip()

    return stream()
