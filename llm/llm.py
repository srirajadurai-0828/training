from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from typing import Any, List, Optional
import traceback
import logging
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("llm")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL_NAME", "claude-opus-4-7")
CONVERSATION_WINDOW = 20

MODEL_NAME = ANTHROPIC_MODEL  # default to primary


class TokenUsageHandler(BaseCallbackHandler):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.requests = []

    def on_llm_end(self, response, **kwargs):
        try:
            msg = response.generations[0][0].message
            meta = getattr(msg, "usage_metadata", {}) or {}
            input_tokens = meta.get("input_tokens", 0)
            output_tokens = meta.get("output_tokens", 0)
            total = input_tokens + output_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total
            self.requests.append({"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total})
            if self.verbose:
                logger.info(f"Tokens | Input={input_tokens} Output={output_tokens} Total={total}")
        except Exception as e:
            logger.warning(f"Token tracking failed: {str(e)}")


token_handler = TokenUsageHandler()

anthropic_llm = ChatAnthropic(
    model=ANTHROPIC_MODEL, temperature=0.3, api_key=ANTHROPIC_API_KEY,
    streaming=False, callbacks=[token_handler],
)
openai_llm = ChatOpenAI(
    model=OPENAI_MODEL, temperature=0.3, api_key=OPENAI_API_KEY,
    streaming=False, callbacks=[token_handler],
)


def invoke_anthropic(messages):
    try:
        logger.info(f"Calling Anthropic model: {ANTHROPIC_MODEL}")
        response = anthropic_llm.invoke(messages)
        logger.info("Anthropic success")
        return response
    except Exception as e:
        logger.error(f"Anthropic failed [{type(e).__name__}]: {str(e)}")
        raise e


def invoke_openai(messages):
    try:
        logger.info(f"Calling OpenAI model: {OPENAI_MODEL}")
        response = openai_llm.invoke(messages)
        logger.info("OpenAI success")
        return response
    except Exception as e:
        logger.error(f"OpenAI failed [{type(e).__name__}]: {str(e)}")
        raise e


def safe_llm_invoke(messages):
    global MODEL_NAME
    try:
        response = invoke_anthropic(messages)
        MODEL_NAME = ANTHROPIC_MODEL
        return response
    except Exception as anthropic_error:
        logger.warning(f"Anthropic failed [{type(anthropic_error).__name__}]. Switching to OpenAI fallback...")
        try:
            response = invoke_openai(messages)
            MODEL_NAME = OPENAI_MODEL
            return response
        except Exception as openai_error:
            logger.critical("Both Anthropic and OpenAI failed")
            logger.critical(f"Anthropic Error: {str(anthropic_error)}")
            logger.critical(f"OpenAI Error: {str(openai_error)}")
            raise Exception(
                "Both LLM providers failed.\n\n"
                f"Anthropic Error: {str(anthropic_error)}\n\n"
                f"OpenAI Error: {str(openai_error)}"
            )


class SafeLLM:
    def invoke(self, messages):
        return safe_llm_invoke(messages)

    def with_structured_output(self, schema):
        anthropic_structured = anthropic_llm.with_structured_output(schema)
        openai_structured = openai_llm.with_structured_output(schema)

        class StructuredWrapper:
            def invoke(self, messages):
                global MODEL_NAME
                try:
                    logger.info("Structured output via Anthropic")
                    result = anthropic_structured.invoke(messages)
                    MODEL_NAME = ANTHROPIC_MODEL
                    return result
                except Exception as anthropic_error:
                    logger.warning(f"Structured output failed on Anthropic [{type(anthropic_error).__name__}]. Switching to OpenAI.")
                    try:
                        logger.info("Structured output via OpenAI fallback")
                        result = openai_structured.invoke(messages)
                        MODEL_NAME = OPENAI_MODEL
                        return result
                    except Exception as openai_error:
                        logger.critical("Structured output failed on both providers.")
                        logger.critical(f"Anthropic Error: {str(anthropic_error)}")
                        logger.critical(f"OpenAI Error: {str(openai_error)}")
                        raise Exception("Structured output failed on both providers.")

        return StructuredWrapper()


class SafeAgentLLM(BaseChatModel):
    """
    A proper LangChain BaseChatModel that wraps Anthropic (primary) and
    OpenAI (fallback) for agentic/tool-calling usage.
    Extends BaseChatModel so it is a Runnable and works with
    create_tool_calling_agent's LCEL pipeline (prompt | llm | parser).
    """

    # Private attributes (not Pydantic fields) — use object.__setattr__
    _anthropic_with_tools: Any = None
    _openai_with_tools: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_anthropic_with_tools", anthropic_llm)
        object.__setattr__(self, "_openai_with_tools", openai_llm)

    @property
    def _llm_type(self) -> str:
        return "safe-agent-llm"

    def bind_tools(self, tools, **kwargs):
        bound = SafeAgentLLM()
        object.__setattr__(bound, "_anthropic_with_tools", anthropic_llm.bind_tools(tools, **kwargs))
        object.__setattr__(bound, "_openai_with_tools", openai_llm.bind_tools(tools, **kwargs))
        return bound

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        global MODEL_NAME
        try:
            logger.info(f"Agent calling Anthropic model: {ANTHROPIC_MODEL}")
            response = self._anthropic_with_tools.invoke(messages)
            MODEL_NAME = ANTHROPIC_MODEL
            logger.info("Agent Anthropic success")
        except Exception as anthropic_error:
            logger.warning(f"Agent Anthropic failed [{type(anthropic_error).__name__}]. Switching to OpenAI fallback...")
            try:
                response = self._openai_with_tools.invoke(messages)
                MODEL_NAME = OPENAI_MODEL
                logger.info("Agent OpenAI fallback success")
            except Exception as openai_error:
                logger.critical("Agent: Both Anthropic and OpenAI failed")
                logger.critical(f"Anthropic Error: {str(anthropic_error)}")
                logger.critical(f"OpenAI Error: {str(openai_error)}")
                raise Exception(
                    "Agent: Both LLM providers failed.\n\n"
                    f"Anthropic Error: {str(anthropic_error)}\n\n"
                    f"OpenAI Error: {str(openai_error)}"
                )
        # BaseChatModel._generate must return a ChatResult
        return ChatResult(generations=[ChatGeneration(message=response)])


llm = SafeLLM()
agent_llm = SafeAgentLLM()


def conversation_llm_call(messages: list, session_id: str = "default") -> str:
    try:
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        windowed = non_system_messages[-CONVERSATION_WINDOW:]
        lc_messages = []
        for msg in system_messages + windowed:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
        return llm.invoke(lc_messages).content
    except Exception as e:
        logger.error(f"Conversation call failed: {str(e)}")
        logger.error(traceback.format_exc())
        return "I'm currently experiencing technical difficulties. Please try again later."