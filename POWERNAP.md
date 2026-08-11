# PowerNap in Nybble

This document separates the PowerNap algorithm from Tada-specific capture and UI
code, then maps the pieces that Nybble needs for real use.

## Correct mental model

PowerNap is not one model and it is not one source file. It is a loop spanning data
collection, semantic labeling, causal sample construction, policy rollout, reward,
retrieval, optimization, checkpointing, and inference.

In Tada, the collection path records screen frames and input events through Napsack.
Input inactivity helps create interaction bursts. The Gemini labeler receives the
ordered frames and interaction information for a chunk and emits semantic action
ranges. The stored final frame is an action artifact, not evidence that the model
saw only the first and final frame.

After labeling, a multimodal policy observes the most recent K semantic actions plus
up to five recent screenshots by default, and must predict the next N. During a
rollout it:

1. writes a rationale about the observed sequence;
2. retrieves similar earlier, already-visible reflections;
3. revises the rationale with that memory;
4. emits exactly N action tags.

Several candidates are sampled for the same target. Gemini judges their semantic
and chronological agreement in one structured request. A deterministic validator
separately scores the exact output format. Group-relative advantages turn those
rewards into a GRPO update, and Tinker applies that update to LoRA parameters.

Nybble uses Inkling Small for this trainable policy. Gemini is not fine-tuned here.
It remains the hosted visual labeler and reward judge.

## Tada to Nybble mapping

| PowerNap responsibility | Nybble implementation | Why it is central |
| --- | --- | --- |
| Capture identity and time | `labeling/labeler.py` | Original timestamps and stable frame IDs define causality |
| Independent visual labels | `vlm/gemini.py` | Converts private visual state into policy-ready observations |
| Semantic grouping | `labeling/schema.py`, `vlm/gemini.py` | Creates the action vocabulary and exact span boundaries |
| Resume and idempotency | `JsonlLabelingCache`, `EventJournal` | Prevents duplicate labels, API cost, and partial-run corruption |
| Dataset construction | `data/samples.py` | Enforces bounded K/N windows and blocks target leakage |
| Shared prompts | `powernap/prompts.py` | Keeps training and inference behavior identical |
| Reflection memory | `retrieval/bm25.py` | Supplies causal Retrieve context without future visibility |
| Rollout environment | `powernap/environment.py` | Implements Think, Retrieve, Revise, Actions |
| Reward | `powernap/rewards.py` | Combines strict formatting with batched Gemini accuracy |
| Optimization | `powernap/trainer.py` | Computes group advantages and performs Tinker LoRA updates |
| Recovery | `powernap/checkpoints.py` | Persists optimizer, sampler, retriever, and progress metadata |
| Inference | `powernap/predictor.py` | Replays the same renderer, prompts, and retrieval policy |

## Nybble-specific choices

### Frame source

The primary source is a directory of Meta glasses frames. Timestamps come from a
manifest, EXIF, or a filename Unix timestamp. When those are unavailable, `--fps`
fills missing timestamps from an explicit start time or a stable anchor. Processing
time is never substituted for capture time.

The optional `screen` dependency exists for future desktop adapters, but desktop
screen and keyboard capture is not silently mixed into the glasses journal.

### Gemini labeling contract

Nybble keeps the repository's proven one-image-per-caption contract. Gemini captions
frames independently in parallel. A second, text-only structured request groups the
ordered captions into a contiguous, nonoverlapping, zero-based inclusive partition.
Malformed, clamped, overlapping, or incomplete ranges are rejected.

This avoids fusing two visual scenes while retaining temporal grouping. Every action
created from a chunk has `available_at` equal to the chunk's final timestamp because
the grouping could not have existed earlier.

### Causal windows

The journal is ordered by `(start_ts, id)`, not by a broad `timestamp <= cutoff`
query. Past and future are selected by exact IDs and remain inside one source and
capture session. A window is excluded when:

- the first future action began before the selected past labels were available;
- the past and future share a nonempty grouping chunk ID;
- any action belongs to the still-mutable final partial chunk;
- it lacks exactly K past or N future events.

The journal transactionally replaces one source session when a growing tail is
relabeled. Full chunks and tails closed by a later session are stable. `--finalize`
explicitly closes the last partial chunk when a capture is complete. This prevents
stale overlaps and also handles equal timestamps without expanding the context.

### Inkling Small rendering

`thinkingmachines/Inkling-Small` uses the Cookbook tokenizer and `tml_v0` renderer.
The renderer owns assistant-message parsing and stop tokens. Nybble does not decode
raw tokens with a custom XML stop string. `inkling_effort` defaults to zero because
PowerNap already has explicit rationale and revision turns, but it can be set below
one for experiments. Only the exposed XML response is carried into the next phase;
renderer-internal hidden thinking is not replayed as conversation history.

Like Tada, Nybble also supplies the latest five image-bearing past actions by
default. It filters the entire selected past window first, then applies the image
cap, and uses that same helper in training and inference. Future images are never
included. The renderer receives Cookbook image parts with local PNG or JPEG paths,
so no Qwen image processor is involved. Set `max_images` to zero only for an
intentional text-only ablation.

### Retrieval visibility

Retriever documents have both an event time and a visibility time. Query-time BM25
statistics use only documents visible by the sample cutoff, so future documents do
not alter inverse-document-frequency or normalization. A winning revision becomes
visible only after every target action label is available, which can be later than
the target action's semantic end time. MMR removes redundant memories after
retrieval. The same Cookbook tokenizer then caps the retrieved-memory payload at
`retrieval_max_tokens` (4096 by default) in training and inference. Whole memories
are dropped by lowest score, then oldest event time, so retrieval can never truncate
the observed journal, screenshots, or phase instruction.

### Failure and recovery boundaries

- Gemini requests and response parsing use bounded exponential retries.
- Retryable setup, sampling, and checkpoint operations use bounded retries.
- Non-idempotent forward/backward and optimizer mutations are issued exactly once;
  an uncertain response aborts so recovery cannot apply an update twice.
- A training step can skip a constant-reward or failed rollout group without
  updating on empty data.
- Resume restores optimizer state, not only model weights.
- The final sampler and state are durable; local artifacts are fsynced or atomically
  replaced where they define recovery state.

## What is intentionally not copied

Tada's application lifecycle, desktop process manager, screen connector, input-event
schema, and UI are product-specific. They are useful sources for a future desktop
adapter, but they are not prerequisites for glasses-based PowerNap. Copying them into
the main path would couple the training algorithm to an unrelated capture surface.

The local Qwen caption path is also not part of this implementation. It remains a
standalone Nybble experiment while the requested production path uses Gemini for
captioning and Inkling Small for training and inference.
