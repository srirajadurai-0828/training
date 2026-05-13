import sys
import os
import logging
import time
import datetime
import json

from pathlib import Path
from typing import Optional, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request as FastAPIRequest
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from routing.query_router import routing
from agent.banking_agent import (
    _agent_store,
    get_banking_agent
)

from llm import (
    token_handler,
    MODEL_NAME
)

from storage.data_store import (
    get_ticket,
    list_tickets_by_session,
    get_account_by_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("banking_ai")

_start_time = time.time()

STORAGE_DIR = Path("storage")

STORAGE_DIR.mkdir(exist_ok=True)


def save_chat_log(

    session_id: str,

    query: str,

    result: dict,

    latency_ms: float,

    tool_outputs: list
):

    timestamp = datetime.datetime.now().isoformat()

    log_data = {

        "timestamp": timestamp,

        "session_id": session_id,

        "query": query,

        "type": result.get("type"),

        "guardrail": result.get("guardrail"),

        "latency_ms": latency_ms,

        "chain_metadata": {

            "attack_check":
                result.get("attack_check"),

            "off_topic_check":
                result.get("off_topic_check"),

            "pii_check":
                result.get("pii_check"),

            "greeting_check":
                result.get("greeting_check")
        },

        "tool_outputs": [

            {
                "tool": t.tool,
                "output": t.output
            }

            for t in tool_outputs
        ],

        "response":
            _extract_response(
                result.get("data")
            )
    }

    file_path = STORAGE_DIR / f"{session_id}.json"

    if file_path.exists():

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            try:

                existing_logs = json.load(f)

            except:

                existing_logs = []

    else:

        existing_logs = []

    existing_logs.append(log_data)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            existing_logs,
            f,
            indent=4,
            ensure_ascii=False
        )


app = FastAPI(
    title="Banking AI",

    description=(
        "Conversational banking assistant with "
        "per-session memory, multi-layer guardrails, "
        "and LangChain tool-calling agents."
    ),

    version="1.0.0",

    docs_url="/docs",

    redoc_url="/redoc",
)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(

    request: FastAPIRequest,

    exc: Exception
):

    logger.error(
        f"Unhandled exception on {request.url}: {exc}",
        exc_info=True
    )

    return JSONResponse(

        status_code=500,

        content={
            "error": "Internal server error.",
            "detail": str(exc)
        },
    )


class ChatRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )


class GuardrailCheck(BaseModel):

    label: str

    confidence: str


class ChainMetadata(BaseModel):

    attack_check:Optional[GuardrailCheck] = None

    off_topic_check:Optional[GuardrailCheck] = None

    pii_check:Optional[GuardrailCheck] = None

    greeting_check:Optional[GuardrailCheck] = None


class ToolOutput(BaseModel):

    tool: str

    output: str


class ChatResponse(BaseModel):

    session_id: str

    query: str

    response: str

    type: str

    guardrail: bool

    chain_metadata: ChainMetadata

    tool_outputs: list[ToolOutput] = Field(
        default_factory=list
    )

    latency_ms: float


class ResetRequest(BaseModel):

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class ResetResponse(BaseModel):

    session_id: str

    cleared: bool

    message: str


class HealthResponse(BaseModel):

    status: str

    uptime_seconds: float

    model: str

    active_sessions: int

    token_usage: dict


def _extract_response(data: Any) -> str:

    if isinstance(data, dict):

        return data.get(
            "output",
            str(data)
        )

    return str(data or "")


def _extract_tool_outputs(data: Any) -> list[ToolOutput]:

    if not isinstance(data, dict):

        return []

    steps = data.get(
        "intermediate_steps",
        []
    )

    outputs = []

    for action, observation in steps:

        tool_name = getattr(
            action,
            "tool",
            "unknown_tool"
        )

        outputs.append(

            ToolOutput(
                tool=tool_name,
                output=str(observation)
            )
        )

    return outputs


def _build_chain_metadata(
    result: dict
) -> ChainMetadata:

    def _guard(
        key: str
    ) -> Optional[GuardrailCheck]:

        raw = result.get(key)

        if raw and isinstance(raw, dict):

            return GuardrailCheck(

                label=raw.get(
                    "label",
                    ""
                ),

                confidence=raw.get(
                    "confidence",
                    ""
                ),
            )

        return None

    return ChainMetadata(

        attack_check=
            _guard("attack_check"),

        off_topic_check=
            _guard("off_topic_check"),

        pii_check=
            _guard("pii_check"),

        greeting_check=
            _guard("greeting_check"),
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
async def chat(
    request: ChatRequest
):

    start = time.time()

    logger.info(
        f"[{request.session_id}] "
        f"CHAT query={request.query[:80]!r}"
    )

    try:

        result = routing(
            query=request.query,
            session_id=request.session_id
        )

    except Exception as exc:

        logger.error(
            f"[{request.session_id}] "
            f"routing failed: {exc}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    latency_ms = round(
        (time.time() - start) * 1000,
        2
    )

    data = result.get("data")

    response_text = _extract_response(data)

    tool_outputs = _extract_tool_outputs(data)

    save_chat_log(

        session_id=request.session_id,

        query=request.query,

        result=result,

        latency_ms=latency_ms,

        tool_outputs=tool_outputs
    )

    chain_metadata = _build_chain_metadata(
        result
    )

    logger.info(

        f"[{request.session_id}] DONE "

        f"type={result.get('type')} "

        f"guardrail={result.get('guardrail')} "

        f"latency={latency_ms}ms"
    )

    return ChatResponse(

        session_id=request.session_id,

        query=request.query,

        response=response_text,

        type=result.get(
            "type",
            "unknown"
        ),

        guardrail=result.get(
            "guardrail",
            False
        ),

        chain_metadata=chain_metadata,

        tool_outputs=tool_outputs,

        latency_ms=latency_ms,
    )


@app.post(
    "/reset",
    response_model=ResetResponse,
    tags=["Session"],
)
async def reset(
    request: ResetRequest
):

    sid = request.session_id

    if sid in _agent_store:

        del _agent_store[sid]

        logger.info(
            f"[{sid}] session cleared"
        )

        return ResetResponse(

            session_id=sid,

            cleared=True,

            message=(
                f"Session '{sid}' "
                f"memory cleared."
            ),
        )

    logger.info(
        f"[{sid}] reset requested "
        f"but no active session found"
    )

    return ResetResponse(

        session_id=sid,

        cleared=False,

        message=(
            f"No active session found "
            f"for '{sid}'."
        ),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Ops"],
)
async def health():

    uptime = round(
        time.time() - _start_time,
        2
    )

    return HealthResponse(

        status="ok",

        uptime_seconds=uptime,

        model=MODEL_NAME or "unknown",

        active_sessions=len(_agent_store),

        token_usage={

            "total_tokens":
                token_handler.total_tokens,

            "input_tokens":
                token_handler.total_input_tokens,

            "output_tokens":
                token_handler.total_output_tokens,

            "total_requests":
                len(token_handler.requests),
        },
    )


@app.get(
    "/tickets/{ticket_id}",
    tags=["Data"],
)
async def get_ticket_endpoint(
    ticket_id: str
):

    ticket = get_ticket(
        ticket_id.upper()
    )

    if not ticket:

        raise HTTPException(

            status_code=404,

            detail=(
                f"Ticket '{ticket_id}' "
                f"not found."
            )
        )

    return ticket


@app.get(
    "/tickets",
    tags=["Data"],
)
async def list_tickets_endpoint(
    session_id: str
):

    return list_tickets_by_session(
        session_id
    )


@app.get(
    "/account",
    tags=["Data"],
)
async def get_account_endpoint(
    session_id: str
):

    account = get_account_by_session(
        session_id
    )

    if not account:

        raise HTTPException(

            status_code=404,

            detail=(
                f"No account found for "
                f"session '{session_id}'."
            )
        )

    return account


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )