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
        "query": "Account number 56781234 transfer money",
        "label": "Contains PII",
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
        "confidence": "Medium"
    },
    {
        "query": "I think my number is something like 12345, not sure",
        "label": "Contains PII",
        "confidence": "Low"
    }
]

pii_example_prompt = PromptTemplate.from_template(
    "Query: {query}\n Label: {label}\n Confidence: {confidence}"
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
        "- Bank account numbers, card numbers, or IFSCs\n"
        "- Phone numbers, email addresses, or Aadhaar/PAN numbers\n"
        "- Passwords, PINs, OTPs, or CVV codes\n"
        "- Any partial or full personal identifiers\n\n"
        "Label as 'No PII' if the query contains only general banking questions with no personal data.\n\n"
        "Examples:"
    ),
    suffix="\nNow classify this:\nQuery: {input}\nLabel:",
    input_variables=["input"]
)
