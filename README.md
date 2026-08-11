# Nybble

Nybble turns timestamped first-person frames into a causal action journal, then
fine-tunes an action predictor with the PowerNap training loop.

The default pipeline uses:

- Gemini 3.6 Flash for independent frame captions, semantic action grouping, and
  batched reward judging
- `thinkingmachines/Inkling-Small` as the trainable next-action policy
- Tinker's LoRA training service with group-relative policy optimization
- a time-aware BM25 and MMR memory for the Think, Retrieve, Revise, Actions loop

The original local MLX captioner remains available as an isolated experiment. It is
not imported by the PowerNap path.

## What the pipeline actually does

```text
timestamped frames
  -> one-frame Gemini captions
  -> exact semantic action spans per temporal chunk
  -> append-only action journal
  -> bounded past K / future N training windows plus the last 5 past frames
  -> Inkling Small: Think -> Retrieve -> Revise -> Actions
  -> one batched Gemini reward judgment per rollout group
  -> GRPO advantages -> Tinker LoRA update
  -> durable state, sampler, retriever, and metric checkpoints
```

This differs from a common shorthand description of PowerNap in two important ways.
Tada's runtime capture does not use ffmpeg as its primary recorder. Its Napsack path
uses screen and input hooks, groups input bursts, and gives the labeler every ordered
frame in a chunk. It does not caption only the first and final frame. Nybble keeps the
same causal grouping and training ideas, but treats Meta glasses frames plus their
timestamps as the source of truth. See [POWERNAP.md](POWERNAP.md) for the code-level
mapping and design choices.

## Install

Python 3.11 is required for the Tinker training extra.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[tinker,dev]'
```

For Gemini-only labeling, the base package is enough:

```bash
pip install -e .
```

Set secrets in the environment or in a local `.env` file:

```text
GEMINI_API_KEY=...
TINKER_API_KEY=...
```

Use a data policy appropriate for identifiable egocentric footage. Raw frames are
sent to Gemini for hosted labeling, and the selected past context frames are sent to
Tinker when Inkling Small training or inference renders a multimodal prompt.

## 1. Label timestamped frames

If filenames or EXIF contain timestamps:

```bash
nybble label path/to/frames \
  --journal data/actions.jsonl \
  --labels-output data/labels.json \
  --finalize
```

For an extracted video sequence, provide its frame rate and start time:

```bash
nybble label path/to/frames \
  --fps 2 \
  --start-time 2026-08-10T12:00:00Z \
  --chunk-size 10 \
  --max-gap-seconds 30 \
  --finalize
```

A JSON, JSONL, or CSV manifest can be used instead:

```bash
nybble label path/to/frames --manifest captures.csv
```

CSV manifests require `path` and one of `captured_at`, `timestamp`, or `time`.
Timestamps are never replaced with processing time. Natural filename order is used,
and a gap larger than `--max-gap-seconds` starts a new session.

`data/label-cache.jsonl` caches both frame captions and grouping responses. A rerun
therefore resumes without paying for completed Gemini work. The action journal
transactionally synchronizes each labeled source session, so a growing partial chunk
replaces its earlier interpretation instead of leaving overlapping stale actions.
The final nonempty chunk is always labeled, but it remains provisional and is
excluded from training until it fills, a later session closes it, or `--finalize`
declares that capture complete. Provisional actions remain available to prediction.

## 2. Inspect and materialize training windows

```bash
nybble inspect data/actions.jsonl --past-len 50 --future-len 8

nybble dataset data/actions.jsonl \
  --past-len 50 \
  --future-len 8 \
  --output data/training-samples.jsonl
```

Each sample contains exact event IDs, a bounded observed context, and a distinct
future target. Windows that cross a labeling-availability boundary or split one
Gemini grouping chunk between past and future are excluded.

## 3. Train Inkling Small

```bash
nybble train data/actions.jsonl \
  --steps 20 \
  --run-dir runs/powernap \
  --past-len 50 \
  --future-len 8 \
  --max-images 5
```

The defaults are:

```text
model       thinkingmachines/Inkling-Small
renderer    tml_v0
LoRA rank   32
batch size  8 distinct windows
group size  4 rollouts per window
past images 5 most recent image-bearing actions
retrieval memory 4096 tokenizer tokens maximum
```

Resume both the model and optimizer state, plus the temporal retriever:

```bash
nybble train data/actions.jsonl \
  --steps 20 \
  --run-dir runs/powernap \
  --resume-latest
```

Intermediate remote checkpoints use a seven-day TTL. The final state and sampler
are saved without an explicit TTL. Local checkpoint manifests, metrics, and
retriever snapshots are append-only and are never deleted by the trainer.

## 4. Predict

```bash
nybble predict data/actions.jsonl \
  --run-dir runs/powernap \
  --past-len 50 \
  --future-len 8
```

Inference uses the same bounded context renderer, three-turn prompt sequence,
ordered past-frame selection, Inkling renderer, and temporal retrieval rules as
training. Predictions are appended to `runs/powernap/predictions.jsonl`. Pass
`--max-images 0` to run an explicit text-only experiment. Both commands apply the
same `--retrieval-max-tokens` cap (4096 by default) only to retrieved reflections;
the observed action journal, screenshots, and phase instruction are never truncated.

## Legacy caption experiments

The original scripts still support direct caption comparisons:

```bash
python run_captions.py path/to/frames
python run_captions.py path/to/frames --backend gemini
```

The most important finding from those experiments remains enforced here: caption
one frame per request. Supplying a previous frame made the visual model fuse scenes.
Temporal evidence belongs in the grouping step and the Inkling context. The local
MLX path also retains its measured 4-bit and repetition-penalty settings in
`ondevice_vlm.py`, but it is not a PowerNap dependency.

## Repository layout

| Path | Role |
| --- | --- |
| `src/nybble/vlm/gemini.py` | Native Gemini image, grouping, and structured-output adapter |
| `src/nybble/labeling/` | Timestamp discovery, strict spans, concurrency, and resumable caches |
| `src/nybble/models.py` | Versioned immutable records |
| `src/nybble/data/` | Crash-tolerant journal and causal sample construction |
| `src/nybble/retrieval/` | Time-filtered BM25, deduplication, MMR, and checkpoints |
| `src/nybble/powernap/` | Prompts, rollout environment, Gemini rewards, GRPO trainer, and inference |
| `src/nybble/pipeline.py` | Label-to-event and sample-to-rollout adapters |
| `src/nybble/cli.py` | End-to-end command-line interface |
| `api_vlm.py` | Legacy OpenAI-compatible hosted caption experiment |
| `ondevice_vlm.py` | Optional local MLX caption experiment |

## Test

```bash
pytest
ruff check src tests
```

The default tests use fake Gemini and Tinker boundaries and make no paid network
requests. Tests marked `live_gemini` or `live_tinker` require explicit credentials.
