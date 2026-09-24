"""Nybble — consent-gated local importer for personal behavioral data.

Reads the personal data sources the user approves, normalizes them into one
record schema, and packages the result for fine-tuning. Everything runs on the
user's machine.
"""

__version__ = "0.1.0"

from .schema import Record  # noqa: F401
