
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


_nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
).create_engine()


_registry = RecognizerRegistry()
_registry.load_predefined_recognizers(languages=["en"], nlp_engine=_nlp_engine)


_registry.add_recognizer(PatternRecognizer(
    supported_entity="IN_AADHAAR",
    patterns=[
        Pattern("AADHAAR_SOLID",    r"\b[2-9]\d{11}\b",                      0.85),
        Pattern("AADHAAR_SPACED",   r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b",        0.85),
        Pattern("AADHAAR_HYPHEN",   r"\b[2-9]\d{3}-\d{4}-\d{4}\b",          0.85),
    ],
    context=["aadhaar", "aadhar", "uid", "uidai"],
))


_registry.add_recognizer(PatternRecognizer(
    supported_entity="IN_PAN",
    patterns=[
        Pattern("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.85),
    ],
    context=["pan", "permanent account"],
))


_registry.add_recognizer(PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[
        Pattern("IN_MOBILE",         r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", 0.75),
        Pattern("IN_MOBILE_SPACED",  r"(?<!\d)[6-9]\d{4}\s\d{5}(?!\d)",         0.75),
    ],
    context=["phone", "mobile", "contact", "call", "tel", "whatsapp"],

))


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
    "IN_AADHAAR",
    "IN_PAN",
    "US_SSN",
    "IP_ADDRESS",
    "NRP",
]

_analyzer = AnalyzerEngine(
    registry=_registry,
    nlp_engine=_nlp_engine,
    supported_languages=["en"],
)

_anonymizer = AnonymizerEngine()


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

TOKEN_LABELS: dict[str, str] = {
    "[CARD_NUMBER]": "card number",
    "[PHONE]":       "phone number",
    "[EMAIL]":       "email address",
    "[IBAN]":        "IBAN",
    "[NAME]":        "name",
    "[AADHAAR]":     "Aadhaar number",
    "[PAN]":         "PAN number",
    "[SSN]":         "SSN",
    "[IP_ADDRESS]":  "IP address",
    "[URL]":         "URL",
    "[DATE]":        "date",
    "[NRP]":         "personal identifier",
    "[REDACTED]":    "sensitive information",
}


HARD_BLOCK_TOKENS = {"[CARD_NUMBER]", "[AADHAAR]", "[PAN]", "[SSN]", "[IBAN]"}


import re

def redact_pii(query: str, score_threshold: float = 0.4) -> tuple[str, list[str]]:
    
    # Step 1: temporarily replace transaction IDs with placeholders
    txn_pattern = re.compile(
        r'(?i)(transaction\s*id|txn\s*id|ref(?:erence)?\s*(?:id|no|number)?)\s*[:\-]?\s*([A-Z0-9\-]{6,20})',
    )
    placeholders = {}
    counter = [0]

    def protect_txn(match):
        key = f"__TXN_{counter[0]}__"
        placeholders[key] = match.group(0)
        counter[0] += 1
        return key

    protected_query = txn_pattern.sub(protect_txn, query)

    # Step 2: run Presidio on protected query
    results = _analyzer.analyze(
        text=protected_query,
        entities=_ENTITIES,
        language="en",
        score_threshold=score_threshold,
    )

    if not results:
        # Restore placeholders and return
        restored = protected_query
        for key, original in placeholders.items():
            restored = restored.replace(key, original)
        return restored, []

    anonymised = _anonymizer.anonymize(
        text=protected_query,
        analyzer_results=results,
        operators=_OPERATORS,
    )

    masked = anonymised.text

    # Step 3: restore protected transaction IDs
    for key, original in placeholders.items():
        masked = masked.replace(key, original)

    found_tokens = [
        token for token in TOKEN_LABELS
        if token in masked and token not in query
    ]

    return masked, found_tokens