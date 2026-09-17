"""Redaction of secrets and direct identifiers.

On by default. The goal is not anonymization — message archives are inherently
re-identifiable — but keeping credentials and payment data out of a corpus that
will be fed to a model.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Order matters: longer/more specific patterns first so a card number is not
# partially eaten by the generic long-digit rule.
PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # Credentials and keys — always removed, even with redaction "off".
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("STRIPE_KEY", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL)),
    ("ENV_ASSIGNMENT", re.compile(
        r"\b([A-Z][A-Z0-9_]{2,}(?:_(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|"
        r"CREDENTIAL|API|AUTH))\s*=\s*)\S+")),
    ("PASSWORD_PHRASE", re.compile(
        r"\b(password|passwd|pwd|passphrase)\s*(?:is|:|=)\s*\S+", re.I)),

    # Direct identifiers — removed when redaction is enabled.
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(
        r"(?<![\w.])\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w.])")),
]

# These run regardless of the user's redaction preference.
ALWAYS = {
    "AWS_KEY", "GITHUB_TOKEN", "SLACK_TOKEN", "OPENAI_KEY", "ANTHROPIC_KEY",
    "GOOGLE_KEY", "STRIPE_KEY", "JWT", "PRIVATE_KEY", "ENV_ASSIGNMENT",
    "PASSWORD_PHRASE",
}


def _luhn_ok(digits: str) -> bool:
    """Real card numbers pass Luhn; long IDs and phone strings usually do not."""
    total, alt = 0, False
    for ch in reversed(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


class Redactor:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.counts: Dict[str, int] = {}

    def __call__(self, text: str) -> str:
        return self.scrub(text)

    def scrub(self, text: str) -> str:
        if not text:
            return text
        for label, pattern in PATTERNS:
            if not self.enabled and label not in ALWAYS:
                continue

            if label == "CARD":
                text = self._scrub_card(text)
                continue
            if label == "ENV_ASSIGNMENT":
                text, n = pattern.subn(r"\1[REDACTED_SECRET]", text)
            elif label == "PASSWORD_PHRASE":
                text, n = pattern.subn(r"\1: [REDACTED_SECRET]", text)
            else:
                text, n = pattern.subn(f"[REDACTED_{label}]", text)
            if n:
                self.counts[label] = self.counts.get(label, 0) + n
        return text

    def _scrub_card(self, text: str) -> str:
        pattern = dict(PATTERNS)["CARD"]

        def repl(m: "re.Match[str]") -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                self.counts["CARD"] = self.counts.get("CARD", 0) + 1
                return "[REDACTED_CARD]"
            return m.group(0)

        return pattern.sub(repl, text)

    def report(self) -> Dict[str, int]:
        return dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))
