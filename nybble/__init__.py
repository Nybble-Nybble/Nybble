"""Nybble — consent-gated local importer for personal behavioral data.

Reads the past-behavior sources enumerated in the Nybble plan (things you wrote,
your messages, transcripts of you talking, writing and content of others you
like, and the preferences platforms saved about you), normalizes them into one
record schema, and infers a taste profile from the material you kept.

Everything runs on the user's machine. Nothing is uploaded without a separate,
explicit grant.
"""

__version__ = "0.1.0"

from .schema import Record  # noqa: F401
