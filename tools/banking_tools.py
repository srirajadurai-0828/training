# ─── banking_tools.py ───────────────────────────────────────────────────────

from langchain_core.tools import tool

from tools.intent_classifier_router import detect_intent
from tools.sentiment_analyzer import analyze_sentiment
from tools.complaint_triage import triage_complaint
from tools.rag.hm25_rag import hm25_retriever_tool

from llm.llm import agent_llm 



def _llm_judge_rag(query: str, retrieved_context: str) -> dict:
    """
    Runs an LLM-as-Judge pass over the retrieved RAG context.

    Returns a dict:
        {
            "verdict":     "RELEVANT" | "PARTIAL" | "IRRELEVANT",
            "score":       int (1-5),
            "reason":      str,
            "final_answer": str          # grounded answer or fallback
        }
    """
    judge_prompt = f"""You are a strict RAG quality judge for a banking assistant.

You will be given:
1. A customer QUERY
2. Retrieved CONTEXT from the knowledge base

Your job is two-fold:
A) Judge whether the context is sufficient to answer the query.
B) If sufficient, produce a concise, grounded answer using ONLY the context.

--- QUERY ---
{query}

--- RETRIEVED CONTEXT ---
{retrieved_context}

Respond ONLY in this exact format (no extra text):
VERDICT: <RELEVANT | PARTIAL | IRRELEVANT>
SCORE: <1-5>
REASON: <one sentence>
ANSWER: <grounded answer if RELEVANT/PARTIAL, otherwise "I could not find reliable information on this in my knowledge base.">
"""

    response = agent_llm.invoke(judge_prompt)

    # agent_llm returns an AIMessage; extract text content
    raw = response.content if hasattr(response, "content") else str(response)

    # Parse the structured output
    result = {
        "verdict": "IRRELEVANT",
        "score": 1,
        "reason": "Could not parse judge response.",
        "final_answer": (
            "I could not find reliable information on this "
            "in my knowledge base."
        ),
    }

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            result["verdict"] = line.split(":", 1)[1].strip()
        elif line.startswith("SCORE:"):
            try:
                result["score"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("ANSWER:"):
            result["final_answer"] = line.split(":", 1)[1].strip()

    return result


# ── Tools ────────────────────────────────────────────────────────────────────

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


@tool
def hm25_rag_tool(query: str) -> str:
    """
    Retrieves information from the HM25 knowledge base using the given query.
    Use this whenever the customer asks about HM25 or related topics,
    or whenever you need to look up any policy, product, or factual
    information to answer a customer question.

    Internally applies an LLM-as-Judge step to verify retrieval quality
    before returning the answer, so the result is always grounded and
    reliability-rated.
    """

    retrieved_context: str = hm25_retriever_tool(query)

    judgment = _llm_judge_rag(query, retrieved_context)


    return (
        f"[RAG Judge] Verdict: {judgment['verdict']} | "
        f"Score: {judgment['score']}/5 | "
        f"Reason: {judgment['reason']}\n\n"
        f"Answer: {judgment['final_answer']}"
    )