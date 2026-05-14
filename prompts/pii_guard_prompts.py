from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


pii_examples = [

    {
        "query": "My account number is [CARD_NUMBER], can you check my balance?",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "Here is my phone number [PHONE], update my account",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My email is [EMAIL], please link it",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My Aadhaar number is [AADHAAR]",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My PAN is [PAN]",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My PIN is [PIN_OTP], reset it",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "Please contact [NAME] at [EMAIL] regarding my account",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "I live at [LOCATION], please update my address",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "My IBAN is [IBAN] for the transfer",
        "label": "Contains PII",
        "confidence": "High"
    },
    {
        "query": "Call me at [PHONE] after resolving this",
        "label": "Contains PII",
        "confidence": "High"
    },

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
        "query": "Transaction 4028312 to xxx977@paytm failed on [DATE]",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "A payment of ₹2.0 to Raunak on [DATE] failed. Transaction ID: 114302729857.",
        "label": "No PII",
        "confidence": "High"
    },

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
        "query": "I want to dispute a transaction",
        "label": "No PII",
        "confidence": "High"
    },
    {
        "query": "What documents do I need for a home loan?",
        "label": "No PII",
        "confidence": "High"
    },
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
        "Queries have already been pre-processed by a PII redactor. Sensitive values "
        "have been replaced with tokens like [PHONE], [EMAIL], [CARD_NUMBER], [AADHAAR], "
        "[PAN], [NAME], [LOCATION], [IBAN], [DATE], [REDACTED].\n\n"
        "Your task: detect whether the query contains any of these masked PII tokens.\n\n"
        "Label as 'Contains PII' if the query includes any masked token such as:\n"
        "- [CARD_NUMBER] or [IBAN] — bank or card identifiers\n"
        "- [PHONE] or [EMAIL] — contact details\n"
        "- [AADHAAR], [PAN], [SSN] — government IDs\n"
        "- [NAME] or [LOCATION] — personal identifiers\n"
        "- [REDACTED] — any other masked sensitive value\n\n"
        "Label as 'No PII' if the query contains:\n"
        "- General banking questions with no masked tokens\n"
        "- Transaction IDs or payment reference numbers (raw numbers are NOT PII)\n"
        "- UPI IDs used in a transaction context (e.g. xxx@paytm)\n"
        "- Amounts, merchant names, or [DATE] tokens alone (dates are not PII in transaction context)\n\n"
        "Examples:"
    ),
    suffix="\nNow classify this:\nQuery: {input}\nLabel:",
    input_variables=["input"]
)