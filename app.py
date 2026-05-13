import streamlit as st
import requests
import uuid
import time
from PIL import Image
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path="chatbot_db.db"))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Horizon Assist",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "health_cache" not in st.session_state:
    st.session_state.health_cache = None

if "health_ts" not in st.session_state:
    st.session_state.health_ts = 0

if "total_turns" not in st.session_state:
    st.session_state.total_turns = 0

if "processing" not in st.session_state:
    st.session_state.processing = False

if "ocr_image_key" not in st.session_state:       # ← fix 1: initialize missing key
    st.session_state.ocr_image_key = 0

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None


def api_chat(query: str, session_id: str):
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"query": query, "session_id": session_id},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_reset(session_id: str):
    try:
        response = requests.post(
            f"{API_BASE}/reset",
            json={"session_id": session_id},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def api_health():
    now = time.time()
    if st.session_state.health_cache and (now - st.session_state.health_ts) < 10:
        return st.session_state.health_cache
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        st.session_state.health_cache = data
        st.session_state.health_ts = now
        return data
    except Exception:
        return None                                # ← always returns None on failure


with st.sidebar:

    st.info("Horizon Bank AI Assistant")
    st.caption("AI-powered Banking Support")
    st.divider()

    st.subheader("Backend Status")
    health = api_health()

    if health and health.get("status") == "ok":   # ← fix 3: null-check before .get()
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

    st.subheader("Session")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    custom_sid = st.text_input(
        "Custom session ID",
        value=st.session_state.session_id,
    )

    if custom_sid != st.session_state.session_id:
        st.session_state.session_id = custom_sid
        st.session_state.messages = []
        st.session_state.total_turns = 0
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Chat", use_container_width=True):
            api_reset(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.total_turns = 0
            st.rerun()

    with col2:
        if st.button("Clear", use_container_width=True):
            result = api_reset(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.total_turns = 0
            if result:
                if result.get("cleared"):
                    st.toast("Memory cleared.")
                else:
                    st.toast("Nothing to clear.")
            st.rerun()

    st.divider()

    st.subheader("This Session")
    c1, c2 = st.columns(2)
    c1.metric("Turns", st.session_state.total_turns)
    c2.metric("Messages", len(st.session_state.messages))

    st.divider()

    st.subheader("Service Health")

    if st.button("Refresh", use_container_width=True):
        st.session_state.health_cache = None
        st.session_state.health_ts = 0

    if st.session_state.health_cache is None:
        with st.spinner("Checking service..."):
            health = api_health()
    else:
        health = api_health()

    if health:                                     # ← fix 3: all health blocks guarded
        status = health.get("status", "unknown")
        if status == "ok":
            st.success(f"{status.upper()}")
        else:
            st.error(f"{status.upper()}")

        token_usage = health.get("token_usage", {})
        uptime_s = health.get("uptime_seconds", 0)

        if uptime_s > 60:
            uptime_str = (
                f"{int(uptime_s // 3600)}h "
                f"{int((uptime_s % 3600) // 60)}m"
            )
        else:
            uptime_str = f"{int(uptime_s)}s"

        c1, c2 = st.columns(2)
        c1.metric("Uptime", uptime_str)
        c2.metric("Sessions", health.get("active_sessions", 0))
        st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")

        c3, c4 = st.columns(2)
        c3.metric("Input", f"{token_usage.get('input_tokens', 0):,}")
        c4.metric("Output", f"{token_usage.get('output_tokens', 0):,}")
        st.caption(f"Model: `{health.get('model', '—')}`")

    else:
        st.error("Backend unreachable")
        st.caption("Make sure FastAPI is running on port 8000.")

    st.divider()

    st.subheader("Quick Prompts")

    quick_prompts = [
        "How do I open a savings account?",
        "What documents do I need for a loan?",
        "I want to dispute a transaction",
        "Difference between FD and RD?",
        "My card was blocked, what do I do?",
    ]

    for q in quick_prompts:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state["_inject_query"] = q
            st.rerun()


st.title("Horizon Bank AI")
st.caption("Ask me anything about accounts, loans, cards, or complaints.")
st.divider()

with st.expander("Upload Transaction Failure Screenshot", expanded=False):
    st.caption(
        "Upload a banking transaction failure screenshot. "
        "The system uses OCR to validate and extract complaint details."
    )
    uploaded = st.file_uploader(
        "Choose a PNG, JPG, or JPEG file",
        type=["png", "jpg", "jpeg"],
        key=f"ocr_uploader_{st.session_state.ocr_image_key}",  
    )
    if uploaded:
        image = Image.open(uploaded)               
        st.image(image, caption="Uploaded Screenshot", use_container_width=True)

        if st.button("Validate and Extract Details", type="primary"):
            with st.spinner("Running OCR analysis... (this may take 10-20 seconds)"):
                try:
                    from ocr.transaction_validation import transaction_validator
                    ocr_result = transaction_validator(image)
                    st.session_state.ocr_result = ocr_result
                except Exception as e:
                    st.error(f"OCR pipeline error: {e}")
                    ocr_result = None

            if ocr_result:
                if ocr_result["valid"]:
                    st.success("Transaction proof validated successfully.")
                    st.info(
                        "The extracted summary will be appended to your next message "
                        "for automatic complaint registration."
                    )
                    st.text_area("Extracted Summary", ocr_result["summary"])
                else:
                    st.error("Image validation failed - cannot process as transaction proof.")
                    st.warning(ocr_result["summary"])
                    st.session_state.ocr_result = None

if not st.session_state.messages:
    st.info("Start a conversation by typing a message below or choosing a quick prompt.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            data = msg.get("data", {})
            tools = data.get("tool_outputs", [])

            if tools:
                with st.expander(f"Tool calls ({len(tools)})", expanded=False):
                    for t in tools:
                        tool_name = (
                            t.get("tool", "")
                            .replace("_tool", "")
                            .replace("_", " ")
                            .title()
                        )
                        st.caption(f"**{tool_name}**")
                        st.code(t.get("output", ""), language=None)

            meta = data.get("chain_metadata", {})
            checks = {
                "Attack":   meta.get("attack_check"),
                "Topic":    meta.get("off_topic_check"),
                "PII":      meta.get("pii_check"),
                "Greeting": meta.get("greeting_check"),
            }
            active_checks = {k: v for k, v in checks.items() if v}

            if active_checks:
                with st.expander("Guardrail checks", expanded=False):
                    cols = st.columns(len(active_checks))
                    for col, (name, check) in zip(cols, active_checks.items()):
                        col.metric(
                            label=name,
                            value=check.get("label", "—"),
                            delta=check.get("confidence", ""),
                            delta_color="off",
                        )

            if data.get("guardrail"):
                st.warning("This request was blocked by a security guardrail.")

            response_type = data.get("type", "")
            latency = data.get("latency_ms", "")
            type_map = {
                "greeting":       "Greeting",
                "agent_response": "Agent",
                "secure_response": "Blocked",
            }
            parts = []
            if response_type in type_map:
                parts.append(type_map[response_type])
            if latency:
                parts.append(f"{latency}ms")
            if parts:
                st.caption("  ·  ".join(parts))


injected = st.session_state.pop("_inject_query", None)

ocr_injected = ""

if st.session_state.get("ocr_result") and st.session_state.ocr_result.get("valid"):
    ocr_injected = (
        "The Image validator has already validated this transaction as valid. "
        "Use the summary below to register complaint:\n\n"
        + st.session_state.ocr_result["summary"]
    )

user_input = st.chat_input("Type your banking query...")
prompt = (user_input + "\n" + ocr_injected) if user_input else None

if injected:
    prompt = injected

if prompt and not st.session_state.processing:

    st.session_state.processing = True

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = api_chat(prompt, st.session_state.session_id)

        if data:
            response_text = data.get("response", "Sorry, I could not process that.")
            st.write(response_text)

            tools = data.get("tool_outputs", [])
            if tools:
                with st.expander(f"Tool calls ({len(tools)})", expanded=False):
                    for t in tools:
                        tool_name = (
                            t.get("tool", "")
                            .replace("_tool", "")
                            .replace("_", " ")
                            .title()
                        )
                        st.caption(f"**{tool_name}**")
                        st.code(t.get("output", ""), language=None)

            meta = data.get("chain_metadata", {})
            checks = {
                "Attack":   meta.get("attack_check"),
                "Topic":    meta.get("off_topic_check"),
                "PII":      meta.get("pii_check"),
                "Greeting": meta.get("greeting_check"),
            }
            active_checks = {k: v for k, v in checks.items() if v}

            if active_checks:
                with st.expander("Guardrail checks", expanded=False):
                    cols = st.columns(len(active_checks))
                    for col, (name, check) in zip(cols, active_checks.items()):
                        col.metric(
                            label=name,
                            value=check.get("label", "—"),
                            delta=check.get("confidence", ""),
                            delta_color="off",
                        )

            if data.get("guardrail"):
                st.warning("This request was blocked by a security guardrail.")

            response_type = data.get("type", "")
            latency = data.get("latency_ms", "")
            type_map = {
                "greeting":        "Greeting",
                "agent_response":  "Agent",
                "secure_response": "Blocked",
            }
            parts = []
            if response_type in type_map:
                parts.append(type_map[response_type])
            if latency:
                parts.append(f"{latency}ms")
            if parts:
                st.caption("  ·  ".join(parts))

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "data": data,
            })

            st.session_state.total_turns += 1

    st.session_state.processing = False
    st.rerun()