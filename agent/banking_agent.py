# ─── agent.py ────────────────────────────────────────────────────────────────

from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm import agent_llm

from agent.bank_config import (
    get_bank_context,
    BANK_NAME,
    BANK_HELPLINE,
)

from tools.banking_tools import (
    intent_classifier_tool,
    sentiment_analysis_tool,
    complaint_triage_tool,
    hm25_rag_tool,
)

from tools.record_tools import (
    raise_complaint_ticket,
    check_ticket_status,
    list_my_tickets,
    check_account_status,
)

tools = [
    intent_classifier_tool,
    sentiment_analysis_tool,
    complaint_triage_tool,
    hm25_rag_tool,

    raise_complaint_ticket,
    check_ticket_status,
    list_my_tickets,
    check_account_status,
]

# ─── agent.py  (_SYSTEM_TEMPLATE only) ──────────────────────────────────────

_SYSTEM_TEMPLATE = """You are a professional banking assistant for {bank_name}.

Your session ID for this conversation is: {{SESSION_ID}}

Always pass this session ID when calling any tool that requires it.

TOOL USAGE RULES:

1. For ANY question that requires factual knowledge — including product details,
   policies, procedures, fees, eligibility, terms, frozen accounts, blocked cards,
   KYC requirements, or any "what should I do" / "how do I" question:
- Call hm25_rag_tool(query="<the customer's question>") FIRST.
- If the verdict is RELEVANT or PARTIAL, use the returned Answer as the basis
  for your response and stay grounded in it.
- If the verdict is IRRELEVANT (score ≤ 2), tell the user you could not find
  reliable information and direct them to {helpline}.
- Do NOT answer knowledge questions from memory alone; always go through
  hm25_rag_tool first.

2. ONLY when a user explicitly asks to VIEW their account details, balance,
   or transaction history (e.g. "show my balance", "what is my account number"):
- Call check_account_status(session_id="{{SESSION_ID}}")
- If the result starts with "NO_ACCOUNT":
  Inform the user that no account is linked to this session.
  Direct them to visit the nearest branch or call {helpline}.
  Do NOT attempt to collect details or register an account.
- If account exists:
  Continue using the account details returned.

3. When a user reports:
- complaints
- duplicate charges
- failed transactions
- fraud
- unresolved banking issues

First call complaint_triage_tool to determine:
- priority
- escalation
- SLA

Then call raise_complaint_ticket.

Always provide the ticket ID to the user.

4. When a user asks for complaint status:
- Call check_ticket_status

5. When a user asks:
- "show my tickets"
- "list my complaints"

Call:
list_my_tickets(session_id="{{SESSION_ID}}")

6. Use:
- sentiment_analysis_tool for complaints or when the customer seems upset
- intent_classifier_tool when intent is unclear

ROUTING PRIORITY (follow in order):
- "what should I do", "how do I", "why is my", "what is the process" → Rule 1 (hm25_rag_tool)
- "show my balance", "what is my account number"                     → Rule 2 (check_account_status)
- complaint / fraud / failed transaction                             → Rule 3 (complaint_triage_tool)

{bank_context}

Rules:
- Never ask for OTP, PIN, passwords, or full card numbers
- Do not perform real transactions
- Do not collect personal details for account opening — direct users to a branch or {helpline}
- Guide users to official banking channels when needed
- Stay grounded in the bank data above
- If unsure, direct users to {helpline}
"""

_agent_store: dict = {}


def get_banking_agent(
    session_id: str = "default"
) -> AgentExecutor:

    if session_id not in _agent_store:

        system_prompt = (
            _SYSTEM_TEMPLATE
            .replace("{bank_name}", BANK_NAME)
            .replace("{bank_context}", get_bank_context())
            .replace("{helpline}", BANK_HELPLINE)
            .replace("{{SESSION_ID}}", session_id)
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),

            MessagesPlaceholder(
                "chat_history",
                optional=True
            ),

            ("human", "{input}"),

            MessagesPlaceholder(
                "agent_scratchpad"
            ),
        ])

        memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
        )

        agent = create_openai_tools_agent(
            llm=agent_llm,
            tools=tools,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            return_intermediate_steps=True,
        )

        _agent_store[session_id] = executor

    return _agent_store[session_id]