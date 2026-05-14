from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


off_topic_examples = [
    {
        "query": "How to open a savings account?",
        "label": "Banking",
        "confidence": "High"
    },
    {
        "query": "What is interest rate in bank?",
        "label": "Banking",
        "confidence": "High"
    },
    {
        "query": "How to check my account balance?",
        "label": "Banking",
        "confidence": "High"
    },
    {
        "query": "Who won the cricket match yesterday?",
        "label": "Off-Topic",
        "confidence": "High"
    },
    {
        "query": "Tell me a joke",
        "label": "Off-Topic",
        "confidence": "High"
    },
    {
        "query": "Tell me a movie story",
        "label": "Off-Topic",
        "confidence": "High"
    },
    {
        "query": "What is the weather today?",
        "label": "Off-Topic",
        "confidence": "High"
    },
    {
        "query": "I am bored, suggest a movie",
        "label": "Off-Topic",
        "confidence": "Medium"
    },
    {
        "query": "Bank app is slow, what to do?",
        "label": "Banking",
        "confidence": "Medium"
    },
    {
        "query": "Summarize the chat",
        "label": "Banking",
        "confidence": "High"
    },
    {
        "query": "Explain how UPI works",
        "label": "Banking",
        "confidence": "High"
    }
]

off_topic_example_prompt = PromptTemplate.from_template(
    "Query: {query}\n Label: {label}\n Confidence: {confidence}"
)

off_topic_few_shot_prompt = FewShotPromptTemplate(
    examples=off_topic_examples,
    example_prompt=off_topic_example_prompt,
    prefix=(
        "You are a relevance classifier for a banking assistant.\n"
        "Your task is to determine whether the user query is related to banking "
        "and financial services, or is completely unrelated.\n\n"
        "Label as 'Banking' if the query is about:\n"
        "- Bank accounts, transactions, loans, cards, or financial products\n"
        "- Banking app issues or customer support\n"
        "- Financial planning, savings, or charges\n\n"
        "Label as 'Off-Topic' if the query has no connection to banking or finance.\n\n"
        "Examples:"
    ),
    suffix="\nNow classify this:\nQuery: {input}\nLabel:",
    input_variables=["input"]
)
