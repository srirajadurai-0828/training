import prompts.topic_system_prompts as prompts
from llm import conversation_llm_call

# =========================================================
# TOPIC DISPATCHER
# =========================================================

def routing_topics(data: dict, session_id: str = "default") -> str:
    """
    Route an agent-analysed query to the correct topic-specific system prompt
    and generate a contextual response.

    Args:
        data:       Dict with keys: query, intent, sentiment, complaint.
                    All keys are optional — missing ones are handled gracefully.
        session_id: Passed through to conversation_llm_call for session context.

    Returns:
        The LLM's response string.
    """

    query     = data.get("query", "")
    intent    = data.get("intent", "general_faq")
    sentiment = data.get("sentiment", "neutral")
    complaint = data.get("complaint")      # may be None or a dict

    # ------------------------------------------------------------------
    # SELECT SYSTEM PROMPT BASED ON INTENT
    # ------------------------------------------------------------------

    if intent == "account_inquiry":
        system_prompt = prompts.ACCOUNT_PROMPT

    elif intent == "card_dispute":
        system_prompt = prompts.CARD_PROMPT

    elif intent == "loan_query":
        system_prompt = prompts.LOAN_PROMPT

    elif intent == "complaint":

        escalation_text = ""

        if complaint and isinstance(complaint, dict):
            escalation_text = (
                f"\n\nComplaint Priority: {complaint.get('priority', 'Unknown')}"
                f"\nEscalation Required: {complaint.get('escalation_required', False)}"
                f"\nSLA Hours: {complaint.get('sla_hours', 'N/A')}"
                f"\nReason: {complaint.get('reason', '')}"
            )

        sentiment_note = (
            f"\n\nCustomer Sentiment: {sentiment}"
            "\nIf frustration is high — apologise professionally, "
            "reassure the customer, and mention escalation if needed."
        )

        system_prompt = prompts.SUPPORT_PROMPT + escalation_text + sentiment_note

    else:
        system_prompt = prompts.SUPPORT_PROMPT

    # ------------------------------------------------------------------
    # CALL LLM WITH SESSION CONTEXT
    # ------------------------------------------------------------------

    return conversation_llm_call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        session_id=session_id,
    )
