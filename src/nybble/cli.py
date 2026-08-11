"""Command-line entry points for labeling, dataset construction, and PowerNap."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from nybble.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_RENDERER,
    DEFAULT_TINKER_MODEL,
    PowerNapConfig,
    require_secret,
)
from nybble.data import (
    ContextBuilder,
    EventJournal,
    SampleBuilder,
    select_latest_visible_context,
)
from nybble.models import Prediction
from nybble.pipeline import (
    labeling_result_to_records,
    to_rollout_sample,
    write_json_atomic,
    write_jsonl_atomic,
)
from nybble.powernap.checkpoints import CheckpointRecord


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _effort_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def _add_window_arguments(parser: argparse.ArgumentParser, *, use_defaults: bool = True) -> None:
    parser.add_argument("--past-len", type=_positive_int, default=50 if use_defaults else None)
    parser.add_argument("--future-len", type=_positive_int, default=8 if use_defaults else None)
    parser.add_argument("--max-images", type=_nonnegative_int, default=5 if use_defaults else None)


def _add_prediction_arguments(
    parser: argparse.ArgumentParser, *, use_defaults: bool = True
) -> None:
    _add_window_arguments(parser, use_defaults=use_defaults)
    parser.add_argument("--max-tokens", type=_positive_int, default=512 if use_defaults else None)
    parser.add_argument("--temperature", type=float, default=1.0 if use_defaults else None)
    parser.add_argument(
        "--inkling-effort", type=_effort_float, default=0.0 if use_defaults else None
    )
    parser.add_argument(
        "--retrieval-top-k", type=_positive_int, default=10 if use_defaults else None
    )
    parser.add_argument(
        "--retrieval-mmr-k", type=_positive_int, default=5 if use_defaults else None
    )
    parser.add_argument(
        "--retrieval-max-tokens",
        type=_nonnegative_int,
        default=4096 if use_defaults else None,
    )
    parser.add_argument(
        "--retrieval-mmr-alpha", type=_unit_float, default=0.5 if use_defaults else None
    )
    parser.add_argument("--retrieval-time-decay", type=float, default=0.5 if use_defaults else None)
    parser.add_argument(
        "--retrieval-min-accuracy",
        type=_unit_float,
        default=0.5 if use_defaults else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nybble",
        description=("Gemini-first egocentric action labeling and Inkling Small PowerNap training"),
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser(
        "label", help="caption timestamped frames and append semantic actions"
    )
    label.add_argument("frames", type=Path)
    label.add_argument("--journal", type=Path, default=Path("data/actions.jsonl"))
    label.add_argument("--labels-output", type=Path, default=Path("data/labels.json"))
    label.add_argument("--cache", type=Path, default=Path("data/label-cache.jsonl"))
    label.add_argument("--manifest", type=Path)
    label.add_argument("--fps", type=_positive_float)
    label.add_argument("--start-time")
    label.add_argument("--recursive", action="store_true")
    label.add_argument(
        "--finalize",
        action="store_true",
        help="declare the current final partial chunk closed for training",
    )
    label.add_argument("--chunk-size", type=_positive_int, default=10)
    label.add_argument("--max-gap-seconds", type=_positive_float, default=30.0)
    label.add_argument("--caption-workers", type=_positive_int, default=4)
    label.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    label.add_argument("--source", default="meta-glasses")
    label.add_argument("--max-attempts", type=_positive_int, default=4)
    label.add_argument("--retry-base-seconds", type=float, default=1.0)
    label.add_argument("--retry-max-seconds", type=float, default=20.0)
    label.set_defaults(handler=_label)

    dataset = subparsers.add_parser(
        "dataset", help="materialize causal past/future windows from an action journal"
    )
    dataset.add_argument("journal", type=Path)
    dataset.add_argument("--output", type=Path, default=Path("data/training-samples.jsonl"))
    _add_window_arguments(dataset)
    dataset.add_argument("--stride", type=_positive_int, default=1)
    dataset.set_defaults(handler=_dataset)

    inspect = subparsers.add_parser("inspect", help="summarize a labeled action journal")
    inspect.add_argument("journal", type=Path)
    _add_window_arguments(inspect)
    inspect.set_defaults(handler=_inspect)

    train = subparsers.add_parser(
        "train", help="fine-tune Inkling Small with PowerNap GRPO and LoRA"
    )
    train.add_argument("journal", type=Path)
    train.add_argument("--run-dir", type=Path, default=Path("runs/powernap"))
    train.add_argument("--steps", type=_positive_int, required=True)
    train.add_argument("--resume-latest", action="store_true")
    train.add_argument("--model", default=DEFAULT_TINKER_MODEL)
    train.add_argument("--renderer", default=DEFAULT_RENDERER)
    _add_prediction_arguments(train)
    train.add_argument("--batch-size", type=_positive_int, default=8)
    train.add_argument("--group-size", type=_positive_int, default=4)
    train.add_argument("--lora-rank", type=_positive_int, default=32)
    train.add_argument("--learning-rate", type=_positive_float, default=5e-5)
    train.add_argument("--checkpoint-every", type=_nonnegative_int, default=2)
    train.add_argument("--accuracy-weight", type=float, default=0.5)
    train.add_argument("--formatting-weight", type=float, default=0.5)
    train.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    train.add_argument("--max-step-attempts", type=_positive_int)
    train.set_defaults(handler=_train)

    predict = subparsers.add_parser(
        "predict", help="predict the next actions with a saved Inkling Small sampler"
    )
    predict.add_argument("journal", type=Path)
    predict.add_argument("--run-dir", type=Path, default=Path("runs/powernap"))
    predict.add_argument("--predictions", type=Path)
    predict.add_argument("--cutoff-ts", type=float)
    predict.add_argument("--action-attempts", type=_positive_int, default=3)
    _add_prediction_arguments(predict, use_defaults=False)
    predict.set_defaults(handler=_predict)
    return parser


def _label(args: argparse.Namespace) -> int:
    from nybble.labeling import FrameLabeler
    from nybble.labeling.prompt import GROUPING_PROMPT_VERSION
    from nybble.vlm import GeminiActionGrouper, GeminiCaptioner, GeminiGenerator, RetryPolicy

    generator = GeminiGenerator(
        api_key=require_secret("GEMINI_API_KEY"),
        model=args.gemini_model,
        retry_policy=RetryPolicy(
            max_attempts=args.max_attempts,
            base_delay_seconds=args.retry_base_seconds,
            max_delay_seconds=args.retry_max_seconds,
        ),
    )
    captioner = GeminiCaptioner(generator=generator)
    grouper = GeminiActionGrouper(generator=generator)
    labeler = FrameLabeler(
        captioner=captioner,
        grouper=grouper,
        chunk_size=args.chunk_size,
        max_gap_seconds=args.max_gap_seconds,
        caption_workers=args.caption_workers,
        cache_path=args.cache,
    )
    result = labeler.label_directory(
        args.frames,
        manifest_path=args.manifest,
        fps=args.fps,
        start_time=args.start_time,
        recursive=args.recursive,
        finalize_tail=args.finalize,
    )
    observations, actions = labeling_result_to_records(
        result,
        source=args.source,
        grouping_model=grouper.model,
        grouping_prompt_version=GROUPING_PROMPT_VERSION,
    )
    session_ids = {chunk.session_id for chunk in result.chunks}
    removed, installed = (0, 0)
    if session_ids:
        removed, installed = EventJournal(args.journal).replace_source_sessions(
            actions,
            source=args.source,
            session_ids=session_ids,
        )
    write_json_atomic(
        args.labels_output,
        {
            "schema_version": 1,
            "source": args.source,
            "caption_model": captioner.model,
            "grouping_model": grouper.model,
            "labeling": result.to_dict(),
            "observations": [record.to_dict() for record in observations],
            "events": [record.to_dict() for record in actions],
        },
    )
    print(
        json.dumps(
            {
                "frames": len(observations),
                "chunks": len(result.chunks),
                "actions": len(actions),
                "removed_journal_actions": removed,
                "installed_journal_actions": installed,
                "journal": str(args.journal.resolve()),
                "labels": str(args.labels_output.resolve()),
                "cache": str(args.cache.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_samples(args: argparse.Namespace):
    events = EventJournal(args.journal).read()
    builder = SampleBuilder(
        past_len=args.past_len,
        future_len=args.future_len,
        stride=getattr(args, "stride", 1),
        max_images=args.max_images,
        context_builder=ContextBuilder(include_dense_context=True),
    )
    return events, builder.build(events)


def _dataset(args: argparse.Namespace) -> int:
    events, samples = _build_samples(args)
    write_jsonl_atomic(args.output, (sample.to_dict() for sample in samples))
    print(
        json.dumps(
            {
                "events": len(events),
                "samples": len(samples),
                "past_len": args.past_len,
                "future_len": args.future_len,
                "max_images": args.max_images,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    events, samples = _build_samples(args)
    chunks = {event.chunk_id for event in events if event.chunk_id}
    sessions = {
        str(event.provenance.get("session_id"))
        for event in events
        if event.provenance.get("session_id")
    }
    payload = {
        "journal": str(args.journal.resolve()),
        "events": len(events),
        "chunks": len(chunks),
        "sessions": len(sessions),
        "eligible_training_windows": len(samples),
        "past_len": args.past_len,
        "future_len": args.future_len,
        "max_images": args.max_images,
        "start_ts": events[0].start_ts if events else None,
        "end_ts": events[-1].end_ts if events else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _powernap_config(
    args: argparse.Namespace, *, model: str | None = None, renderer: str | None = None
) -> PowerNapConfig:
    return PowerNapConfig(
        model=model or args.model,
        renderer=renderer or args.renderer,
        past_len=args.past_len,
        future_len=args.future_len,
        max_images=args.max_images,
        batch_size=getattr(args, "batch_size", 8),
        group_size=getattr(args, "group_size", 4),
        lora_rank=getattr(args, "lora_rank", 32),
        learning_rate=getattr(args, "learning_rate", 5e-5),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        inkling_effort=args.inkling_effort,
        checkpoint_every=getattr(args, "checkpoint_every", 2),
        retrieval_top_k=args.retrieval_top_k,
        retrieval_mmr_k=args.retrieval_mmr_k,
        retrieval_max_tokens=args.retrieval_max_tokens,
        retrieval_mmr_alpha=args.retrieval_mmr_alpha,
        retrieval_time_decay=args.retrieval_time_decay,
        retrieval_min_accuracy=args.retrieval_min_accuracy,
        accuracy_weight=getattr(args, "accuracy_weight", 0.5),
        formatting_weight=getattr(args, "formatting_weight", 0.5),
    )


def _reward_model_for_run(requested_model: str, resume: CheckpointRecord | None) -> str:
    if resume is None or not resume.reward_contract:
        return requested_model
    checkpoint_model = resume.reward_contract.get("model")
    if not isinstance(checkpoint_model, str) or not checkpoint_model.strip():
        raise ValueError("resume checkpoint has an invalid Gemini reward model")
    return checkpoint_model


async def _train_async(args: argparse.Namespace) -> dict[str, Any]:
    from nybble.powernap.checkpoints import CheckpointStore
    from nybble.powernap.rewards import GeminiRewardJudge
    from nybble.powernap.trainer import TinkerPowerNapTrainer
    from nybble.retrieval import BM25TemporalRetriever
    from nybble.vlm import GeminiGenerator

    resume = None
    if args.resume_latest:
        resume = CheckpointStore(args.run_dir / "checkpoints.jsonl").latest()
        if resume is None:
            raise RuntimeError(f"no checkpoint found in {args.run_dir}")
    config = (
        PowerNapConfig(**resume.config)
        if resume is not None and resume.config
        else _powernap_config(args)
    )
    events = EventJournal(args.journal).read()
    samples = SampleBuilder(
        past_len=config.past_len,
        future_len=config.future_len,
        max_images=config.max_images,
        context_builder=ContextBuilder(include_dense_context=True),
    ).build(events)
    rollout_samples = tuple(to_rollout_sample(sample) for sample in samples)
    if not rollout_samples:
        raise RuntimeError(
            "no causal training windows; label more data or lower past-len/future-len"
        )

    reward_model = _reward_model_for_run(args.gemini_model, resume)
    gemini = GeminiGenerator(api_key=require_secret("GEMINI_API_KEY"), model=reward_model)
    reward_judge = GeminiRewardJudge(
        gemini,
        accuracy_weight=config.accuracy_weight,
        formatting_weight=config.formatting_weight,
    )
    trainer = TinkerPowerNapTrainer(
        config=config,
        reward_judge=reward_judge,
        retriever=BM25TemporalRetriever(),
        run_dir=args.run_dir,
        tinker_api_key=require_secret("TINKER_API_KEY"),
        resume=resume,
    )
    result = await trainer.train(
        rollout_samples,
        steps=args.steps,
        max_step_attempts=args.max_step_attempts,
    )
    return {
        "completed_steps": result.completed_steps,
        "checkpoint": asdict(result.checkpoint),
        "metrics": result.metrics_path,
        "samples": len(rollout_samples),
        "model": config.model,
        "renderer": config.renderer,
    }


def _train(args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(_train_async(args)), indent=2, sort_keys=True))
    return 0


def _append_prediction(path: Path, prediction: Prediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            prediction.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


async def _predict_async(args: argparse.Namespace) -> Prediction:
    from nybble.data import select_recent_image_paths
    from nybble.powernap.checkpoints import CheckpointStore
    from nybble.powernap.predictor import TinkerPowerNapPredictor
    from nybble.retrieval import BM25TemporalRetriever

    checkpoint = CheckpointStore(args.run_dir / "checkpoints.jsonl").latest()
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found in {args.run_dir}")
    config = (
        PowerNapConfig(**checkpoint.config)
        if checkpoint.config
        else PowerNapConfig(model=checkpoint.model, renderer=checkpoint.renderer)
    )
    overrides = {
        name: getattr(args, name)
        for name in (
            "past_len",
            "future_len",
            "max_images",
            "max_tokens",
            "temperature",
            "inkling_effort",
            "retrieval_top_k",
            "retrieval_mmr_k",
            "retrieval_max_tokens",
            "retrieval_mmr_alpha",
            "retrieval_time_decay",
            "retrieval_min_accuracy",
        )
        if getattr(args, name) is not None
    }
    if overrides:
        config = replace(config, **overrides)
    events = EventJournal(args.journal).read()
    if not events:
        raise RuntimeError("the action journal is empty")
    cutoff = args.cutoff_ts
    if cutoff is None:
        cutoff = max(event.available_at for event in events)
    if cutoff < 0:
        raise ValueError("cutoff-ts must be nonnegative")
    context_events = select_latest_visible_context(
        events,
        cutoff_ts=cutoff,
        past_len=config.past_len,
    )
    if len(context_events) < config.past_len:
        raise RuntimeError(
            f"need {config.past_len} visible past events in the latest source session, "
            f"found {len(context_events)}"
        )
    context_text = ContextBuilder(include_dense_context=True).render_context(context_events)
    image_paths = select_recent_image_paths(context_events, config.max_images)
    predictor = TinkerPowerNapPredictor(
        config=config,
        checkpoint=checkpoint,
        retriever=BM25TemporalRetriever(),
        tinker_api_key=require_secret("TINKER_API_KEY"),
    )
    trace = await predictor.predict(
        context_text,
        cutoff,
        image_paths=image_paths,
        future_len=config.future_len,
        action_attempts=args.action_attempts,
    )
    created_at = max(time.time(), cutoff)
    identity = json.dumps(
        {
            "created_at": created_at,
            "cutoff": cutoff,
            "checkpoint": checkpoint.sampler_path,
            "context": [event.id for event in context_events],
            "actions": list(trace.actions),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prediction = Prediction(
        id=f"prediction-{hashlib.sha256(identity).hexdigest()[:24]}",
        created_at=created_at,
        cutoff_ts=cutoff,
        actions=trace.actions,
        raw_text=trace.raw_actions,
        rationale=trace.rationale,
        revision=trace.revision,
        retrieved=(trace.retrieved,) if trace.retrieved else (),
        context_event_ids=tuple(event.id for event in context_events),
        model=checkpoint.model,
        checkpoint=checkpoint.sampler_path,
        metadata={"future_len": config.future_len, "image_paths": list(image_paths)},
    )
    prediction_path = args.predictions or args.run_dir / "predictions.jsonl"
    _append_prediction(prediction_path, prediction)
    return prediction


def _predict(args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(_predict_async(args)).to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
