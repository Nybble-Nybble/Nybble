"""On-device captioning: Qwen3-VL on MLX, running locally on Apple Silicon.

Frames may be PIL Images, file paths, URLs, or data URIs — mlx_vlm.load_image
takes all of them, so the (not-yet-written) glasses stream can hand us whatever
it produces.

api_vlm.py is the hosted-API counterpart with the same caption_stream(frames)
signature, so run_captions.py can drive either. Both share prompt.py.
"""

from mlx_vlm import generate, load
from mlx_vlm.utils import load_image

from prompt import CAPTION_PROMPT

# ponytail: 4-bit, not 3-bit. At 3-bit this model failed at spatial grounding — it
# read a top-down frame of the wearer's own sneakers as "lying on their back on the
# floor". 4-bit fixed that outright and is *faster* (better stopping), for 4GB more
# RAM. Do not drop to 3-bit to save memory; the captions degrade silently.
MODEL = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"

# ponytail: longest side we feed the vision encoder. Cost is quadratic in this, and
# Ray-Ban stills are ~12MP, so uncapped frames are what would blow the time budget.
# Raise if the model misses small objects (a phone screen, a label); lower for headroom.
MAX_SIDE = 1024

_loaded = None


def load_model(model_path=MODEL):
    """Load once, first call wins. Call at startup so frame 1 doesn't pay the ~14s load."""
    global _loaded
    if _loaded is None:
        _loaded = load(model_path)
    return _loaded


def _prep(frame):
    """Normalize whatever the stream yields to a downscaled RGB image."""
    img = load_image(frame)  # accepts PIL, path, URL, data URI; always returns a copy
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # aspect-preserving, downscale-only
    return img


def caption_stream(frames, max_tokens=600, repetition_penalty=1.15):
    """Yield one caption per frame. One image per call, never a pair.

    ponytail: sending the previous frame alongside the current one *looks* like free
    motion context, but this 3-bit quant cannot tell which image is "now" — it fused
    two frames into one scene, and captioned the wrong frame in both orderings of a
    test pair, preferring whichever image was larger. Labelling the images inline did
    not help; the chat template was already correct, the model just ignored it. The
    temporal signal is recoverable anyway: the predictor sees the *sequence* of
    captions, so trajectory lives across captions rather than inside one.

    ponytail: max_tokens is a safety backstop, not the length control — set it well
    above the prompt's word cap, which this quant overshoots by ~45%. At 200 it was
    truncating captions mid-sentence, which is worse for training data than a long
    one. repetition_penalty is not optional either. On low-detail frames (dim, blurred,
    a blank wall) greedy decoding either loops a clause or finishes the caption and
    starts a second one. 1.1 did not stop the restart; 1.15 does. Watch for it on dark
    real-world frames.
    """
    model, processor = load_model()
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": CAPTION_PROMPT},
        ]}],
        add_generation_prompt=True,
    )

    for frame in frames:
        yield generate(
            model,
            processor,
            prompt,
            image=[_prep(frame)],
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
        ).text.strip()
