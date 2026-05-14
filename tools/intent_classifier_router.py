from llm.llm import safe_llm_invoke
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
)

from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
)

from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings

from llm.llm import agent_llm

class IntentOutput(BaseModel):

    query: str = Field(
        description="The customer's original query"
    )

    intent: Literal[
        "account_inquiry",
        "card_dispute",
        "loan_query",
        "complaint",
        "general_faq",
    ]

    confidence: Literal[
        "Low",
        "Medium",
        "High"
    ]


examples = [

    
    {
        "query": "Can you check my account balance?",
        "intent": "account_inquiry",
        "confidence": "High",
    },
    {
        "query": "I want to know my available balance",
        "intent": "account_inquiry",
        "confidence": "High",
    },
    {
        "query": "How do I open a savings account?",
        "intent": "account_inquiry",
        "confidence": "High",
    },
    {
        "query": "Show me my last 5 transactions",
        "intent": "account_inquiry",
        "confidence": "High",
    },
    {
        "query": "How to download my bank statement?",
        "intent": "account_inquiry",
        "confidence": "High",
    },

    
    {
        "query": "My debit card transaction failed but money was deducted",
        "intent": "card_dispute",
        "confidence": "High",
    },
    {
        "query": "How do I block my lost debit card?",
        "intent": "card_dispute",
        "confidence": "High",
    },
    {
        "query": "My card was declined at the ATM",
        "intent": "card_dispute",
        "confidence": "High",
    },
    {
        "query": "Someone made a transaction using my card without my permission",
        "intent": "card_dispute",
        "confidence": "High",
    },

    
    {
        "query": "I need home loan details",
        "intent": "loan_query",
        "confidence": "High",
    },
    {
        "query": "What is the interest rate for a personal loan?",
        "intent": "loan_query",
        "confidence": "High",
    },
    {
        "query": "Am I eligible for an education loan?",
        "intent": "loan_query",
        "confidence": "High",
    },
    {
        "query": "What will be my EMI for a ₹5 lakh personal loan?",
        "intent": "loan_query",
        "confidence": "High",
    },

    
    {
        "query": "I was charged twice for the same transaction",
        "intent": "complaint",
        "confidence": "High",
    },
    {
        "query": "My money was debited but the recipient did not receive it",
        "intent": "complaint",
        "confidence": "High",
    },
    {
        "query": "I want to raise a formal complaint",
        "intent": "complaint",
        "confidence": "High",
    },
    {
        "query": "I am very frustrated, nobody is helping me",
        "intent": "complaint",
        "confidence": "High",
    },

    
    {
        "query": "How to reset internet banking password?",
        "intent": "general_faq",
        "confidence": "High",
    },
    {
        "query": "What are your branch working hours?",
        "intent": "general_faq",
        "confidence": "High",
    },
    {
        "query": "What is the bank's customer care number?",
        "intent": "general_faq",
        "confidence": "High",
    },
    {
        "query": "What is the difference between NEFT, RTGS and IMPS?",
        "intent": "general_faq",
        "confidence": "High",
    },
]


embeddings = OpenAIEmbeddings()



example_prompt = PromptTemplate.from_template(
    "Query: {query}\n"
    "Intent: {intent}\n"
    "Confidence: {confidence}"
)


example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=FAISS,
    k=4,
)



intent_prompt = FewShotPromptTemplate(

    example_selector=example_selector,

    example_prompt=example_prompt,

    prefix="""
You are a Banking Intent Classification Engine.

Classify the customer query into exactly one of:

- account_inquiry
- card_dispute
- loan_query
- complaint
- general_faq

Definitions:

account_inquiry:
balance, statements, KYC, account opening/closing,
profile updates, account services

card_dispute:
card disputes, failed transactions,
PIN issues, blocked cards, fraud,
activate/deactivate card

loan_query:
loan eligibility, EMI, rates,
loan applications, pre-closure,
loan documents

complaint:
customer dissatisfaction,
refund delays, unresolved issues,
wrong deductions, service complaints

general_faq:
banking education, charges,
policies, branch info,
bank details, contact info,
bank products, NEFT/RTGS/IMPS,
internet banking

Choose ONLY one intent.
Return confidence based on certainty.
""",

    suffix="""
Query: {input}

Intent:
""",

    input_variables=["input"],
)


def detect_intent(query: str) -> IntentOutput:

    final_prompt = intent_prompt.format(
        input=query
    )

    return agent_llm.with_structured_output(
        IntentOutput
    ).invoke(final_prompt)