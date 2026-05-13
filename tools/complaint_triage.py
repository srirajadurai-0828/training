from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate
)

from llm import llm

class ComplaintOutput(BaseModel):

    complaint: str

    priority: Literal[
        "P1",
        "P2",
        "P3",
        "P4"
    ]

    monetary_impact: str

    escalation_required: bool

    sla_hours: int

    reason: str

examples = [

    {
        "complaint": "I lost ₹2 lakh due to fraud",
        "priority": "P1",
        "monetary_impact": "Very High",
        "escalation_required": True,
        "sla_hours": 1,
        "reason": "Critical fraud issue"
    },

    {
        "complaint": "I was charged twice",
        "priority": "P2",
        "monetary_impact": "Medium",
        "escalation_required": True,
        "sla_hours": 4,
        "reason": "Duplicate debit issue"
    },

    {
        "complaint": "My card delivery is delayed",
        "priority": "P3",
        "monetary_impact": "Low",
        "escalation_required": False,
        "sla_hours": 24,
        "reason": "Service delay"
    },

    {
        "complaint": "The app UI is confusing",
        "priority": "P4",
        "monetary_impact": "None",
        "escalation_required": False,
        "sla_hours": 48,
        "reason": "General feedback"
    }
]

example_prompt = PromptTemplate.from_template(
    """
    Complaint: {complaint}

    Priority: {priority}

    Monetary Impact: {monetary_impact}

    Escalation Required: {escalation_required}

    SLA Hours: {sla_hours}

    Reason: {reason}
    """
)

triage_prompt = FewShotPromptTemplate(

    examples=examples,

    example_prompt=example_prompt,

    prefix="""
    You are a Banking Complaint Triage Engine.

    Assign:
    - complaint severity
    - priority
    - escalation need
    - SLA hours
    """,

    suffix="""
    Complaint: {input}
    """,

    input_variables=["input"]
)


def triage_complaint(query: str):

    final_prompt = triage_prompt.format(
        input=query
    )

    chain = llm.with_structured_output(
        ComplaintOutput
    )

    return chain.invoke(final_prompt)