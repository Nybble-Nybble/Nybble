# Nybble

Next-action prediction on life, from Meta glasses footage. University of Washington.

This branch covers the first stage: turning egocentric frames into captions that
describe **what the wearer is doing**. Those captions become the input tokens for a
downstream text-only predictor, so they are training data — a wrong caption is a
wrong label, and a truncated one is worse than a long one.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

For the hosted backends, put your key in `.env` (gitignored):

```
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

An exported environment variable wins over `.env` if both are set.

## Usage

```bash
python run_captions.py                      # on-device, reads test_images/
python run_captions.py --backend gemini     # hosted
python run_captions.py path/to/frames       # any folder
```

Prints each caption with its wall-clock time and word count.

`test_images/` is **gitignored** — the local frames contain identifiable people, and
this repo is public. Clone it and you get no images; point the script at your own.

## The two backends

Both expose the same `caption_stream(frames)` and share `prompt.py`, so swapping one
for the other is a genuine A/B on the model rather than on the prompt.

| | `ondevice_vlm.py` | `api_vlm.py` |
|---|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct, 4-bit MLX | `gemini-3.6-flash` (or OpenRouter) |
| Speed | ~4 s/frame | ~20 s/frame (10.8–33.0 s observed) |
| Cost | free | ~$3.12 / 1k frames, ~$1.56 batch |
| Data leaves the Mac | no | yes |

Measured on an M3 Max, 36 GB. The API is a corpus-generation tool, not a realtime
one — its latency is thinking time plus a round trip and doesn't tune away, though
a thread pool makes it irrelevant for batch work.

Quality is close. On a test frame Gemini caught a pen on the floor that the local
model missed; the local model caught the wearer's own sneakers and correctly inferred
they were standing, which Gemini omitted. For next-action prediction the posture cue
is probably worth more than the pen.

**On the free tier, Google's terms say your content is used to improve their
products.** The paid tier says it is not. Egocentric footage of identifiable people
belongs on a paid key, or on-device.

## Things learned the hard way

Each of these is a comment in the file it applies to; they are collected here because
every one of them cost a debugging session.

- **Do not drop the local model to 3-bit.** It fails at spatial grounding — it read a
  top-down frame of the wearer's own sneakers as "lying on their back on the floor".
  4-bit fixes it outright and is *faster*. 6-bit exceeds the wired-memory limit and
  thrashes for no measured quality gain.
- **One frame per caption, never a pair.** Sending the previous frame as motion
  context made the model fuse two scenes and caption the wrong one, preferring
  whichever image was larger. The temporal signal lives across captions instead.
- **`max_tokens` is a backstop, not a length control.** The prompt's word cap does
  the real work. On Gemini the reasoning pass shares that budget while staying out of
  `usage.completion_tokens`, so a caption-sized budget starves the caption — at 600 it
  returned `finish_reason="length"` after 15 visible words with no obvious cause.
- **Greedy decoding on the quantized local model loops or restarts.** On low-detail
  frames it repeats a clause or finishes and begins a second caption.
  `repetition_penalty=1.15` stops it; 1.1 did not.
- **The prompt has to ask for people explicitly.** An earlier version listed only
  objects and surfaces, and the local model obeyed literally — walking straight past a
  person standing in frame.

## Files

| | |
|---|---|
| `prompt.py` | The captioning prompt. Shared by both backends. |
| `ondevice_vlm.py` | Local MLX captioner. |
| `api_vlm.py` | Hosted captioner. New provider = one `PROVIDERS` entry. |
| `run_captions.py` | Captions a folder, reports per-frame timing. |
