# Banking AI Chatbot

A production-ready, multi-layer conversational banking assistant built with **FastAPI**, **Streamlit**, and **LangChain**. The system uses LLM-powered guardrails, a tool-calling agent, and a full complaint management pipeline to simulate a real banking support experience.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Request Pipeline](#request-pipeline)
- [Project Structure](#project-structure)
- [File Reference](#file-reference)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
  - [Ubuntu / Linux](#ubuntu--linux)
  - [Windows](#windows)
  - [macOS](#macos)
- [API Endpoints](#api-endpoints)
- [Running Tests](#running-tests)

---

## Architecture Overview

```
User (Browser)
     │
     ▼
┌─────────────┐       HTTP       ┌──────────────────────────────────────────┐
│  app.py     │◄────────────────►│  main.py  (FastAPI Backend)              │
│ (Streamlit  │                  │                                          │
│  Frontend)  │                  │  POST /chat  →  routing/query_router.py  │
└─────────────┘                  └──────────────────────────────────────────┘
                                                  │
                              ┌───────────────────┼────────────────────────┐
                              │                   │                        │
                    ┌─────────▼──────┐  ┌────────▼──────────┐  ┌─────────▼──────┐
                    │  Guardrails    │  │  Banking Agent     │  │  Storage Layer │
                    │  (4 LLM checks)│  │  (LangChain +      │  │  (JSON files)  │
                    │                │  │   8 tools)         │  │                │
                    └────────────────┘  └───────────────────┘  └────────────────┘
                              │
                    ┌─────────▼──────┐
                    │  llm.py        │
                    │  OpenAI (main) │
                    │  Anthropic     │
                    │  (fallback)    │
                    └────────────────┘
```

---

## Request Pipeline

Every user message passes through the following sequential stages:

```
User Query
    │
    ▼
[1] Greeting Check  ──── Is greeting? ──► Return warm greeting response
    │ (Not greeting)
    ▼
[2] Attack Check    ──── Is malicious/jailbreak?
[3] Off-Topic Check ──── Is unrelated to banking?         ──► Return polite block response
[4] PII Check       ──── Contains sensitive data (card#, SSN)?
    │ (All clear)
    ▼
[5] Banking Agent (LangChain AgentExecutor)
    │   Tools available:
    │   ├── intent_classifier_tool      (detect what user wants)
    │   ├── sentiment_analysis_tool     (detect frustration/anger)
    │   ├── complaint_triage_tool       (assign P1–P4 priority + SLA)
    │   ├── raise_complaint_ticket      (create & persist ticket)
    │   ├── check_ticket_status         (look up ticket by ID)
    │   ├── list_my_tickets             (all tickets for session)
    │   ├── check_account_status        (find or prompt registration)
    │   └── register_new_account        (create account record)
    │
    ▼
[6] Response returned to FastAPI → logged to JSON → sent to Streamlit UI
```

---

## Project Structure

```
chatbot/
│
├── app.py                              # Streamlit frontend UI
├── main.py                             # FastAPI backend server
├── llm.py                              # LLM abstraction (OpenAI + Anthropic fallback)
├── requirements.txt                    # Python dependencies
│
├── agent/
│   ├── banking_agent.py                # LangChain AgentExecutor factory
│   └── bank_config.py                  # Bank product data & configuration
│
├── routing/
│   └── query_router.py                 # Multi-layer guardrail pipeline + routing logic
│
├── prompts/
│   ├── greeting_classifier_prompts.py  # Few-shot prompt: greeting detection
│   ├── pii_guard_prompts.py            # Few-shot prompt: PII detection
│   ├── relevance_guard_prompts.py      # Few-shot prompt: off-topic detection
│   ├── security_guard_prompts.py       # Few-shot prompt: attack/jailbreak detection
│   ├── topic_classifier_prompts.py     # Few-shot prompt: topic classification
│   └── topic_system_prompts.py         # System prompts per banking topic
│
├── tools/
│   ├── banking_tools.py                # LangChain tools: intent, sentiment, triage
│   ├── complaint_triage.py             # Complaint priority engine (P1–P4)
│   ├── intent_classifier_router.py     # FAISS-backed semantic intent classifier
│   ├── record_tools.py                 # LangChain tools: tickets & accounts
│   └── sentiment_analyzer.py           # Customer sentiment analyzer
│
├── topic_route/
│   └── topic_dispatcher.py             # Routes to topic-specific system prompts
│
├── storage/
│   ├── data_store.py                   # Thread-safe JSON persistence layer
│   └── data/                           # Auto-created at runtime
│       ├── tickets.json                # Complaint tickets store
│       ├── accounts.json               # User accounts store
│       └── query_log.json              # Audit log of all queries
│
├── tests/
│   └── test_chatbot_functionality.py   # Pytest unit & integration tests
│
└── chatbot_test/
    ├── automated_chatbot_test.py       # End-to-end automated test runner
    └── chatbot_test_results.json       # Saved test results
```

---

## File Reference

### Root Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI. Manages session state, renders chat history, displays guardrail check metadata and tool call outputs, communicates with the FastAPI backend via HTTP. |
| `main.py` | FastAPI application server. Defines `/chat`, `/reset`, `/health`, `/tickets`, and `/account` endpoints. Orchestrates the routing pipeline, logs every conversation turn to a per-session JSON file, and tracks token usage. |
| `llm.py` | LLM abstraction layer. Initialises both OpenAI and Anthropic LangChain clients with a shared `TokenUsageHandler`. Provides `safe_llm_invoke` (tries OpenAI first, falls back to Anthropic) and `SafeLLM` — a wrapper that also supports structured output with automatic provider fallback. |
| `requirements.txt` | All Python dependencies including FastAPI, Uvicorn, Streamlit, LangChain, OpenAI SDK, Anthropic SDK, FAISS, Pydantic, and testing libraries. |

---

### `agent/`

| File | Purpose |
|------|---------|
| `banking_agent.py` | Factory function `get_banking_agent(session_id)` that creates and caches a per-session `AgentExecutor`. Each agent has a `ConversationBufferWindowMemory` (last 10 turns), a custom system prompt injected with bank data, and access to all 8 banking tools. Agents are stored in the `_agent_store` dict keyed by session ID. |
| `bank_config.py` | Static configuration for the fictional "Horizon Bank". Defines all product details (savings, loans, credit cards, FD/RD), fee schedules, escalation contacts, and a `get_bank_context()` function that formats this data as a text block injected into the agent's system prompt. |

---

### `routing/`

| File | Purpose |
|------|---------|
| `query_router.py` | The central routing pipeline. Runs four sequential LLM-powered classification checks (greeting → attack → off-topic → PII) before deciding whether to: (a) return a warm greeting, (b) return a guardrail block response, or (c) pass the query to the banking agent. Returns a structured dict with type, guardrail flag, and check metadata. |

---

### `prompts/`

| File | Purpose |
|------|---------|
| `greeting_classifier_prompts.py` | Few-shot `PromptTemplate` for binary classification: `Greeting` vs `Not Greeting`. |
| `pii_guard_prompts.py` | Few-shot `PromptTemplate` that detects if a message contains personally identifiable information (card numbers, account numbers, passwords, etc.). |
| `relevance_guard_prompts.py` | Few-shot `PromptTemplate` that classifies queries as `Banking` or `Off-Topic` with confidence level. |
| `security_guard_prompts.py` | Few-shot `PromptTemplate` that detects adversarial inputs, jailbreak attempts, and prompt injection attacks (`Safe` vs `Attack`). |
| `topic_classifier_prompts.py` | Few-shot prompts for classifying queries by banking topic (account, card, loan, complaint, FAQ). |
| `topic_system_prompts.py` | Per-topic system prompt strings (`ACCOUNT_PROMPT`, `CARD_PROMPT`, `LOAN_PROMPT`, `SUPPORT_PROMPT`) used by the topic dispatcher. |

---

### `tools/`

| File | Purpose |
|------|---------|
| `banking_tools.py` | Wraps three analysis functions as `@tool`-decorated LangChain tools: `intent_classifier_tool`, `sentiment_analysis_tool`, and `complaint_triage_tool`. These are the "thinking" tools the agent calls to understand the customer's intent and emotional state before acting. |
| `intent_classifier_router.py` | Semantic intent classifier. Uses `OpenAIEmbeddings` + FAISS vector store to find the 4 most similar examples for a query, then calls the LLM with a `FewShotPromptTemplate` to classify into: `account_inquiry`, `card_dispute`, `loan_query`, `complaint`, or `general_faq`. |
| `sentiment_analyzer.py` | Detects the customer's emotional tone on a 5-point scale (`calm` → `churn-risk`) and determines if human escalation is required. Uses a few-shot prompt with the structured output schema `SentimentOutput`. |
| `complaint_triage.py` | Assigns severity to complaints using few-shot examples. Produces `ComplaintOutput` with priority (P1–P4), monetary impact level, escalation flag, SLA hours, and a one-line reason. |
| `record_tools.py` | Five `@tool`-decorated LangChain tools that interact with the storage layer: `raise_complaint_ticket`, `check_ticket_status`, `list_my_tickets`, `check_account_status`, and `register_new_account`. These are the "action" tools that create and read persistent records. |

---

### `topic_route/`

| File | Purpose |
|------|---------|
| `topic_dispatcher.py` | Routes an already-analysed query (with intent, sentiment, and complaint triage data) to the appropriate topic-specific system prompt from `topic_system_prompts.py`, then calls the LLM to generate the final response. Used as an alternative response path for topic-specific handling. |

---

### `storage/`

| File | Purpose |
|------|---------|
| `data_store.py` | Thread-safe JSON persistence layer backed by three files: `tickets.json`, `accounts.json`, and `query_log.json`. Provides `create_ticket`, `get_ticket`, `update_ticket_status`, `list_tickets_by_session`, `register_account`, `get_account_by_session`, and `log_query`. All writes use a `threading.Lock` for concurrency safety. |

---

### `tests/`

| File | Purpose |
|------|---------|
| `tests/test_chatbot_functionality.py` | Pytest test suite covering: guardrail checks (attack, PII, off-topic), greeting detection, agent responses for account/loan/complaint queries, ticket creation flow, and sentiment analysis. |
| `chatbot_test/automated_chatbot_test.py` | End-to-end test runner that fires a batch of predefined queries against the live FastAPI server and records results to `chatbot_test_results.json`. |

---

## Prerequisites

- Python **3.10+**
- An **OpenAI API key** (primary LLM)
- An **Anthropic API key** (fallback LLM — optional but recommended)

---

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key
ANTHROPIC_API_KEY=sk-your-api-key

MODEL_NAME=openai-model-name

LOG_LEVEL=INFO

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=ls-your-api-key
LANGSMITH_PROJECT="project-name"

LANGSMITH_TRACING_V2=true
```

> `MODEL_NAME` is used for the primary OpenAI model. `gpt-4o` or `gpt-4-turbo` recommended. The Anthropic fallback always uses `claude-opus-4-7`.

---

## How to Run

The project requires **two terminal windows** — one for the FastAPI backend and one for the Streamlit frontend.

---

### Ubuntu / Linux

**1. Download and enter the project**

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create the `.env` file**
```bash
cp .env.example .env   # or create manually
nano .env              # add your API keys
```

**5. Start the FastAPI backend** (Terminal 1)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
or
fastapi dev main.py
```

**6. Start the Streamlit frontend** (Terminal 2)
```bash
streamlit run app.py
```

**7. Open in browser**
```
http://localhost:8501
```

---

### Windows

**1. Download and enter the project**

**2. Create and activate a virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```cmd
pip install -r requirements.txt
```

**4. Create the `.env` file**

Create a file named `.env` in the project root using Notepad or any editor:
```
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o
ANTHROPIC_API_KEY=sk-ant-...
```

**5. Start the FastAPI backend** (Command Prompt 1)
```cmd
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
or
fastapi dev main.py
```

**6. Start the Streamlit frontend** (Command Prompt 2)
```cmd
streamlit run app.py
```

**7. Open in browser**
```
http://localhost:8501
```

> **Note:** If `uvicorn` or `streamlit` are not found, ensure your virtual environment is activated and the install step completed successfully.

---

### macOS

**1. Download and enter the project**

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create the `.env` file**
```bash
touch .env
open -e .env   # opens in TextEdit — add your API keys and save
```

Or via terminal:
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
echo "MODEL_NAME=gpt-4o" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

**5. Start the FastAPI backend** (Terminal 1)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**6. Start the Streamlit frontend** (Terminal 2)
```bash
streamlit run app.py
```

**7. Open in browser**
```
http://localhost:8501
```

---

## API Endpoints

Once the backend is running, interactive API docs are available at:

```
http://localhost:8000/docs       (Swagger UI)
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a user message. Body: `{ "query": "...", "session_id": "..." }` |
| `POST` | `/reset` | Clear session memory. Body: `{ "session_id": "..." }` |
| `GET`  | `/health` | Service status, uptime, token usage, active sessions |
| `GET`  | `/tickets/{ticket_id}` | Fetch a complaint ticket by ID |
| `GET`  | `/tickets?session_id=...` | List all tickets for a session |
| `GET`  | `/account?session_id=...` | Get registered account for a session |

---

## Running Tests

**Unit & integration tests (pytest):**
```bash
pytest tests/test_chatbot_functionality.py -v
```

**End-to-end automated tests** (requires the FastAPI server to be running):
```bash
python chatbot_test/automated_chatbot_test.py
```

Results are saved to `chatbot_test/chatbot_test_results.json`.

---

## Data Storage

All data is persisted as JSON files in `storage/data/` (auto-created on first run):

| File | Contents |
|------|----------|
| `tickets.json` | All complaint tickets across all sessions |
| `accounts.json` | Registered user accounts |
| `query_log.json` | Audit log of all queries and responses |

Per-session conversation logs are saved in `storage/<session_id>.json` by `main.py`.

---

## LLM Fallback Strategy

The system uses a two-provider fallback chain:

1. **Primary:** OpenAI (model set via `MODEL_NAME` in `.env`)
2. **Fallback:** Anthropic Claude (`claude-opus-4-7`)

If OpenAI fails for any reason (rate limit, timeout, auth error), the system automatically retries with Anthropic. If both fail, an error is raised with details from both providers. This logic lives in `llm.py`.
