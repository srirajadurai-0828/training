from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate
)

from llm.llm import agent_llm


class SentimentOutput(BaseModel):

    query: str

    sentiment: Literal[
        "calm",
        "neutral",
        "frustrated",
        "angry",
        "churn-risk"
    ]

    score: Literal[
        1,
        2,
        3,
        4,
        5
    ]

    escalation_required: bool


examples = [

    {
        "query": "Thank you for helping me",
        "sentiment": "calm",
        "score": 1,
        "escalation_required": False
    },

    {
        "query": "Can you check my balance?",
        "sentiment": "neutral",
        "score": 2,
        "escalation_required": False
    },

    {
        "query": "This issue is becoming annoying",
        "sentiment": "frustrated",
        "score": 3,
        "escalation_required": False
    },

    {
        "query": "Your bank is terrible",
        "sentiment": "angry",
        "score": 4,
        "escalation_required": True
    },

    {
        "query": "I will close my account",
        "sentiment": "churn-risk",
        "score": 5,
        "escalation_required": True
    }
]


example_prompt = PromptTemplate.from_template(
    """
    Query: {query}

    Sentiment: {sentiment}

    Score: {score}

    Escalation Required: {escalation_required}
    """
)


sentiment_prompt = FewShotPromptTemplate(

    examples=examples,

    example_prompt=example_prompt,

    prefix="""
    You are a Banking Sentiment Analyzer.

    Detect:
    - customer frustration
    - anger
    - churn risk
    - escalation need
    """,

    suffix="""
    Query: {input}
    """,

    input_variables=["input"]
)


def analyze_sentiment(query: str):

    final_prompt = sentiment_prompt.format(
        input=query
    )

    chain = agent_llm.with_structured_output(
        SentimentOutput
    )

    return chain.invoke(final_prompt)