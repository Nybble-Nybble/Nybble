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
from pathlib import Path

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
        # ponytail: 3.6-flash always thinks, and the compat layer bills those thinking
        # tokens against max_tokens while leaving them OUT of usage.completion_tokens.
        # So a budget sized for the caption alone starves it: at max_tokens=600 we got
        # finish_reason="length" after 15 visible words. "none" is rejected outright —
        # "low" is the floor for this model. Anything under ~1500 will truncate.
        "extra": {"reasoning_effort": "low"},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "extra": {},
    },
}


def _key(name):
    """Read a key from the environment, falling back to a gitignored .env file.

    ponytail: eight lines instead of python-dotenv. No export/quote/interpolation
    syntax — one NAME=value per line. Swap in the real thing if you ever need more.
    """
    if os.environ.get(name):
        return os.environ[name]
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == name:
                return v.strip()
    return None


def _data_url(frame):
    """Any frame -> a base64 JPEG data URL, downscaled the same way as on-device."""
    img = frame if isinstance(frame, Image.Image) else Image.open(frame)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # aspect-preserving, downscale-only
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def caption_stream(frames, provider="gemini", model=None, max_tokens=2000):
    """Yield one caption per frame. One image per call, matching ondevice_vlm.

    max_tokens is far above ondevice_vlm's because on thinking models it has to
    cover the reasoning pass too — see the gemini entry in PROVIDERS. It stays a
    backstop, not a length control; the prompt's word cap does that.

    ponytail: no retry/concurrency layer. The SDK already retries 429s and 5xxs,
    and serial calls are fine for iterating on a few hundred frames. When you
    batch a real corpus, wrap this in a thread pool — the network is idle
    latency, not CPU, so concurrency is where the wall-clock win is.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; have {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = _key(cfg["key_env"])
    if not key:
        raise RuntimeError(
            f"{cfg['key_env']} not found — export it, or put "
            f"{cfg['key_env']}=... in .env (gitignored)"
        )
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
                **cfg.get("extra", {}),
            )
            if resp.choices[0].finish_reason == "length":
                raise RuntimeError(
                    f"caption truncated at max_tokens={max_tokens} — on a thinking "
                    f"model the reasoning pass shares this budget; raise it"
                )
            yield (resp.choices[0].message.content or "").strip()

    return stream()
