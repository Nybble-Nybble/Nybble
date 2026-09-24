# Nybble

A pipeline for tuning a language model to fit one person.

Every LLM talks the same way, and nobody is the average user. Nybble collects a
person's own writing, and optionally writing they like, and packages it for
fine-tuning so the model moves toward how that person actually communicates.

There is no benchmark for this. Each person's preferences are different, so for
now evaluation is the person using the model and judging whether it feels like
theirs.

## CLI

Everything runs on your machine. Nothing is read until you approve it, and
nothing is uploaded.

```
pip install -e .

nybble scan        # what personal data exists on this machine
nybble sources     # the full source catalog
nybble consent     # approve sources  (--yes-all approves everything found)
nybble collect     # read approved sources into ~/.nybble/records/*.jsonl
nybble status      # what has been collected so far
nybble revoke      # delete the data and the approvals
```

Approvals live in `~/.nybble/consent.json`, plain JSON you can read and edit.
They expire after 90 days. Redaction of emails, phones, card numbers, and
secrets is on by default; credentials are stripped even if you turn it off.

## Status

This branch establishes the CLI and the consent, record, and output plumbing.
No data sources are registered yet. iMessage is the first one coming.

## Tests

```
pip install -e '.[dev]' && pytest
```
