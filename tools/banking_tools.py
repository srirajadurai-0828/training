from langchain_core.tools import tool

from tools.intent_classifier_router import detect_intent
from tools.sentiment_analyzer import analyze_sentiment
from tools.complaint_triage import triage_complaint


@tool
def intent_classifier_tool(query: str) -> str:
    """
    Classifies the customer's banking intent from their message.
    Returns the detected intent category and confidence level.
    Use this to understand what the customer is trying to do.
    """
    result = detect_intent(query)
    d = result.model_dump()
    return (
        f"Intent: {d['intent']} | "
        f"Confidence: {d['confidence']}"
    )

@tool
def sentiment_analysis_tool(query: str) -> str:
    """
    Detects the customer's emotional tone, frustration level, and whether
    the issue needs escalation to a human agent.
    Use this for complaints or when the customer seems upset.
    """
    result = analyze_sentiment(query)
    d = result.model_dump()
    return (
        f"Sentiment: {d['sentiment']} | "
        f"Score: {d['score']}/5 | "
        f"Escalation Required: {d['escalation_required']}"
    )

@tool
def complaint_triage_tool(query: str) -> str:
    """
    Triages a banking complaint: assigns severity priority (P1-P4),
    monetary impact, escalation requirement, and SLA hours.
    Use this whenever the customer reports a problem or complaint.
    """
    result = triage_complaint(query)
    d = result.model_dump()
    return (
        f"Priority: {d['priority']} | "
        f"Monetary Impact: {d['monetary_impact']} | "
        f"Escalation Required: {d['escalation_required']} | "
        f"SLA: {d['sla_hours']}h | "
        f"Reason: {d['reason']}"
    )
