from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Sample banking text with PII
text = """
Customer John Smith called from (555) 123-4567.
His email is johnsmith@gmail.com.
Account number is 4532-8821-9933-1147.
SSN is 543-22-1234.
"""

# Detect PII
results = analyzer.analyze(
    text=text,
    entities=[
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",    
        "ACCOUNT_NUMBER",
        "US_SSN"
    ],
    language="en"
)

# Redact PII
redacted = anonymizer.anonymize(
    text=text,
    analyzer_results=results
)

# Output
print("ORIGINAL TEXT:\n")
print(text)

print("\nREDACTED TEXT:\n")
print(redacted.text)