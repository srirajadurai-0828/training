from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

greeting_examples = [
    {
        "query": "Hi",
        "label": "Greeting",
        "confidence": "High"
    },
    {
        "query": "Hello, how are you?",
        "label": "Greeting",
        "confidence": "High"
    },
    {
        "query": "Good morning",
        "label": "Greeting",
        "confidence": "High"
    },
    {
        "query": "Hey there!",
        "label": "Greeting",
        "confidence": "High"
    },
    {
        "query": "How to open a bank account?",
        "label": "Not Greeting",
        "confidence": "High"
    },
    {
        "query": "Check my balance",
        "label": "Not Greeting",
        "confidence": "High"
    },
    {
        "query": "Hi, I want to know about loans",
        "label": "Greeting",
        "confidence": "Medium"
    },
    {
        "query": "Thanks for your help",
        "label": "Greeting",
        "confidence": "Medium"
    }
]

greeting_example_prompt = PromptTemplate.from_template(
    "Query: {query}\n Label: {label}\n Confidence: {confidence}"
)

greeting_few_shot_prompt = FewShotPromptTemplate(
    examples=greeting_examples,
    example_prompt=greeting_example_prompt,
    prefix=(
        "You are a greeting classifier for a banking assistant.\n"
        "Determine whether the user's message is a conversational greeting, "
        "farewell, or social pleasantry — as opposed to a direct banking request.\n\n"
        "Label as 'Greeting' if the message is:\n"
        "- A salutation or farewell (hi, hello, bye, thanks)\n"
        "- A social check-in with no specific banking intent\n"
        "- A greeting combined with a vague or general request\n\n"
        "Label as 'Not Greeting' if the message contains a clear banking query or task.\n\n"
        "Examples:"
    ),
    suffix="\nNow classify this:\nQuery: {input}\nLabel:",
    input_variables=["input"]
)
