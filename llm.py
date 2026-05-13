from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import traceback
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("llm")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-opus-4-7"
CONVERSATION_WINDOW = 20


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

openai_llm = ChatOpenAI(
    model=MODEL_NAME, temperature=0.3, api_key=OPENAI_API_KEY,
    streaming=False, callbacks=[token_handler],
)
anthropic_llm = ChatAnthropic(
    model=ANTHROPIC_MODEL, temperature=0.3, api_key=ANTHROPIC_API_KEY,
    streaming=False, callbacks=[token_handler],
)


def invoke_openai(messages):
    try:
        logger.info(f"Calling OpenAI model: {MODEL_NAME}")
        response = openai_llm.invoke(messages)
        logger.info("OpenAI success")
        return response
    except Exception as e:
        logger.error(f"OpenAI failed [{type(e).__name__}]: {str(e)}")
        raise e


def invoke_anthropic(messages):
    try:
        logger.info(f"Calling Anthropic model: {ANTHROPIC_MODEL}")
        response = anthropic_llm.invoke(messages)
        logger.info("Anthropic success")
        return response
    except Exception as e:
        logger.error(f"Anthropic failed [{type(e).__name__}]: {str(e)}")
        raise e


def safe_llm_invoke(messages):
    try:
        return invoke_openai(messages)
    except Exception as openai_error:
        logger.warning(f"OpenAI failed [{type(openai_error).__name__}]. Switching to Anthropic fallback...")
        try:
            return invoke_anthropic(messages)
        except Exception as anthropic_error:
            logger.critical("Both OpenAI and Anthropic failed")
            logger.critical(f"OpenAI Error: {str(openai_error)}")
            logger.critical(f"Anthropic Error: {str(anthropic_error)}")
            raise Exception(
                "Both LLM providers failed.\n\n"
                f"OpenAI Error: {str(openai_error)}\n\n"
                f"Anthropic Error: {str(anthropic_error)}"
            )


class SafeLLM:
    def invoke(self, messages):
        return safe_llm_invoke(messages)

    def with_structured_output(self, schema):
        openai_structured = openai_llm.with_structured_output(schema)

        class StructuredWrapper:
            def invoke(self, messages):
                try:
                    logger.info("Structured output via OpenAI")
                    return openai_structured.invoke(messages)
                except Exception as openai_error:
                    logger.warning(f"Structured output failed on OpenAI [{type(openai_error).__name__}]. Switching to Anthropic.")
                    try:
                        raw = anthropic_llm.invoke(messages)
                        parser_prompt = f"Convert the following response into the required structured schema.\n\nResponse:\n{raw.content}"
                        logger.info("Repairing Anthropic output using OpenAI parser.")
                        return openai_structured.invoke(parser_prompt)
                    except Exception as anthropic_error:
                        logger.critical("Structured output failed on both providers.")
                        logger.critical(f"OpenAI Error: {str(openai_error)}")
                        logger.critical(f"Anthropic Error: {str(anthropic_error)}")
                        raise Exception("Structured output failed on both providers.")

        return StructuredWrapper()


llm = SafeLLM()
agent_llm = openai_llm


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