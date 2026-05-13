from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

pii_examples = [
    {
        "query": "My account number is 1234567890, can you check my balance?",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "Here is my phone number 9876543210, update my account",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My email is raja@gmail.com, please link it",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My Aadhaar number is 1234 5678 9012",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My PAN is ABCDE1234F",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My PIN is 4321, reset it",
        "label": "Contains PII",
        "confidence": "High"
    },
    # ── Transaction ID examples — should NOT be flagged ──
    {
        "query": "Transaction ID 114302729857 failed, raise a complaint",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "My payment with reference number 9876543210 is stuck",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "Transaction 4028312 to xxx977@paytm failed on 23-05-2021",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "A payment of ₹2.0 to Raunak on 23-05-2021 failed. Transaction ID: 114302729857.",
        "label": "No PII",
        "confidence": "High"
    },
    # ── General queries ──
    {
        "query": "How to open a savings account?",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "What is IFSC code?",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "Explain interest rates",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "I think my number is something like 12345, not sure",
        "label": "Contains PII",
        "confidence": "Low"
    }
]

pii_example_prompt = PromptTemplate.from_template(
    "Query: {query}\nLabel: {label}\nConfidence: {confidence}"
)

pii_few_shot_prompt = FewShotPromptTemplate(
    examples=pii_examples,
    example_prompt=pii_example_prompt,
    prefix=(
        "You are a PII (Personally Identifiable Information) detection classifier "
        "for a banking assistant.\n"
        "Your task is to identify whether the user query contains any sensitive personal data "
        "that should not be processed directly.\n\n"
        "Label as 'Contains PII' if the query includes:\n"
        "- Bank account numbers or card numbers\n"
        "- Phone numbers or email addresses\n"
        "- Aadhaar numbers, PAN numbers\n"
        "- Passwords, PINs, OTPs, or CVV codes\n\n"
        "Label as 'No PII' if the query contains:\n"
        "- General banking questions with no personal data\n"
        "- Transaction IDs or payment reference numbers (these are NOT PII)\n"
        "- UPI IDs used in a transaction context (e.g. xxx@paytm)\n"
        "- Amounts, dates, or merchant names related to a transaction\n\n"
        "Examples:"
    ),
    suffix="\nNow classify this:\nQuery: {input}\nLabel:",
    input_variables=["input"]
)