"""
pii/redactor.py
---------------
Presidio-based PII redactor with explicit Indian recogniser registration
and fallback regex for Aadhaar/PAN formats Presidio misses by default.
"""
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ── NLP engine ────────────────────────────────────────────────────────────────
_nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
).create_engine()

# ── Registry: load defaults + explicitly add Indian recognisers ───────────────
_registry = RecognizerRegistry()
_registry.load_predefined_recognizers(languages=["en"], nlp_engine=_nlp_engine)

# Aadhaar: 12 digits — solid, spaced (1234 5678 9012), or hyphenated
_registry.add_recognizer(PatternRecognizer(
    supported_entity="IN_AADHAAR",
    patterns=[
        Pattern("AADHAAR_SOLID",    r"\b[2-9]\d{11}\b",                      0.85),
        Pattern("AADHAAR_SPACED",   r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b",        0.85),
        Pattern("AADHAAR_HYPHEN",   r"\b[2-9]\d{3}-\d{4}-\d{4}\b",          0.85),
    ],
    context=["aadhaar", "aadhar", "uid", "uidai"],
))

# PAN: AAAAA9999A (5 letters, 4 digits, 1 letter)
_registry.add_recognizer(PatternRecognizer(
    supported_entity="IN_PAN",
    patterns=[
        Pattern("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.85),
    ],
    context=["pan", "permanent account"],
))

# Indian mobile: 10 digits starting with 6–9, optional +91 prefix
_registry.add_recognizer(PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[
        Pattern("IN_MOBILE",         r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", 0.75),
        Pattern("IN_MOBILE_SPACED",  r"(?<!\d)[6-9]\d{4}\s\d{5}(?!\d)",         0.75),
    ],
    context=["phone", "mobile", "number", "contact", "call"],
))

# Card numbers: 13–19 digits with optional spaces/dashes
_registry.add_recognizer(PatternRecognizer(
    supported_entity="CREDIT_CARD",
    patterns=[
        Pattern("CARD_SPACED", r"\b(?:\d[ -]?){13,18}\d\b", 0.85),
    ],
    context=["card", "credit", "debit", "visa", "mastercard", "rupay"],
))

# ── Analyzer ──────────────────────────────────────────────────────────────────
_ENTITIES = [
    "CREDIT_CARD",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "PERSON",
    "LOCATION",
    "IN_AADHAAR",
    "IN_PAN",
    "US_SSN",
    "IP_ADDRESS",
    "URL",
    "DATE_TIME",
    "NRP",
]

_analyzer = AnalyzerEngine(
    registry=_registry,
    nlp_engine=_nlp_engine,
    supported_languages=["en"],
)

_anonymizer = AnonymizerEngine()

# ── Replacement tokens ────────────────────────────────────────────────────────
_OPERATORS: dict[str, OperatorConfig] = {
    "CREDIT_CARD":  OperatorConfig("replace", {"new_value": "[CARD_NUMBER]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS":OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "IBAN_CODE":    OperatorConfig("replace", {"new_value": "[IBAN]"}),
    "PERSON":       OperatorConfig("replace", {"new_value": "[NAME]"}),
    "LOCATION":     OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "IN_AADHAAR":   OperatorConfig("replace", {"new_value": "[AADHAAR]"}),
    "IN_PAN":       OperatorConfig("replace", {"new_value": "[PAN]"}),
    "US_SSN":       OperatorConfig("replace", {"new_value": "[SSN]"}),
    "IP_ADDRESS":   OperatorConfig("replace", {"new_value": "[IP_ADDRESS]"}),
    "URL":          OperatorConfig("replace", {"new_value": "[URL]"}),
    "DATE_TIME":    OperatorConfig("replace", {"new_value": "[DATE]"}),
    "NRP":          OperatorConfig("replace", {"new_value": "[NRP]"}),
    "DEFAULT":      OperatorConfig("replace", {"new_value": "[REDACTED]"}),
}

# Human-readable labels for the frontend
TOKEN_LABELS: dict[str, str] = {
    "[CARD_NUMBER]": "card number",
    "[PHONE]":       "phone number",
    "[EMAIL]":       "email address",
    "[IBAN]":        "IBAN",
    "[NAME]":        "name",
    "[LOCATION]":    "location",
    "[AADHAAR]":     "Aadhaar number",
    "[PAN]":         "PAN number",
    "[SSN]":         "SSN",
    "[IP_ADDRESS]":  "IP address",
    "[URL]":         "URL",
    "[DATE]":        "date",
    "[NRP]":         "personal identifier",
    "[REDACTED]":    "sensitive information",
}

# Tokens that trigger a hard block — these should never reach the agent
HARD_BLOCK_TOKENS = {"[CARD_NUMBER]", "[AADHAAR]", "[PAN]", "[SSN]", "[IBAN]"}


def redact_pii(query: str, score_threshold: float = 0.4) -> tuple[str, list[str]]:
    """
    Mask PII in `query` using Presidio.

    Returns:
        masked_query  — query with PII replaced by tokens e.g. [PHONE]
        found_tokens  — list of tokens inserted e.g. ["[PHONE]", "[EMAIL]"]
    """
    results = _analyzer.analyze(
        text=query,
        entities=_ENTITIES,
        language="en",
        score_threshold=score_threshold,
    )

    if not results:
        return query, []

    anonymised = _anonymizer.anonymize(
        text=query,
        analyzer_results=results,
        operators=_OPERATORS,
    )

    masked = anonymised.text

    found_tokens = [
        token for token in TOKEN_LABELS
        if token in masked and token not in query
    ]

    return masked, found_tokens