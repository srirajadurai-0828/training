"""
pii/redactor.py
---------------
Regex-based PII redactor.
Runs AFTER is_pii() detection so the guardrail badge still fires on the
raw query. The redacted query is what reaches the agent and logs.
"""

import re

# ── Replacement tokens ────────────────────────────────────────────────────────
_RULES: list[tuple[str, re.Pattern, str]] = [

    # Card numbers  (13-19 digits, optional spaces/dashes)
    ("card",        re.compile(r"\b(?:\d[ -]?){13,19}\b"),              "[CARD_NUMBER]"),

    # Aadhaar  (12 digits, optional spaces)
    ("aadhaar",     re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),    "[AADHAAR]"),

    # PAN  (AAAAA9999A format)
    ("pan",         re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),          "[PAN]"),

    # Indian mobile  (10 digits, optional +91 / 0 prefix)
    ("phone",       re.compile(r"(?:\+91|0)?[6-9]\d{9}\b"),            "[PHONE]"),

    # Email
    ("email",       re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),  "[EMAIL]"),

    # PIN / OTP  (4-8 digit standalone numbers that aren't transaction IDs)
    # We skip 9+ digit standalone numbers — those are likely transaction IDs
    ("pin_otp",     re.compile(r"\b\d{4,8}\b"),                        "[PIN_OTP]"),

    # IFSC code  (4 letters + 0 + 6 alphanumeric)
    ("ifsc",        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),           "[IFSC]"),

    # CVV  (3-4 digits near keyword cvv/cvc)
    ("cvv",         re.compile(r"(?i)\b(?:cvv|cvc)\b[\s:]*\d{3,4}"),   "[CVV]"),

    # Password-like patterns  (keyword followed by value)
    ("password",    re.compile(r"(?i)\b(?:password|passwd|pwd)\b[\s:]*\S+"), "[PASSWORD]"),
]


def redact_pii(query: str) -> str:
    """
    Apply all redaction rules in order and return the cleaned query.
    Rules are ordered so more specific patterns (card, aadhaar) run before
    the generic pin_otp rule, preventing partial double-replacement.
    """
    redacted = query
    for _name, pattern, token in _RULES:
        redacted = pattern.sub(token, redacted)
    return redacted


def redaction_summary(original: str, redacted: str) -> list[str]:
    """
    Return a list of which PII types were redacted.
    Useful for logging without exposing the actual values.
    """
    found = []
    for name, pattern, token in _RULES:
        if pattern.search(original) and token in redacted:
            found.append(name)
    return found