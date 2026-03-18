"""LLM factory for the Multi-Agent Job Application Assistant.

Provides a single entry point for creating Groq-backed LLM instances
with model selection logic based on task type.
"""

# LangSmith Tracing Setup:
# 1. Sign up free at https://smith.langchain.com
# 2. Create a project called "job-assistant"
# 3. Go to Settings → API Keys → Create API Key
# 4. Add the keys to your .env file (see .env.example)
# Tracing is automatically disabled if LANGCHAIN_API_KEY is not set

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Model constants — all configurable via environment variables
PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL: str = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
LONG_CONTEXT_MODEL: str = os.getenv("LONG_CONTEXT_MODEL", "mixtral-8x7b-32768")


def get_llm(model: str = PRIMARY_MODEL, temperature: float = 0.1) -> ChatGroq:
    """Create a Groq-backed LLM instance.

    Args:
        model: The Groq model name to use. Defaults to the primary model.
        temperature: Sampling temperature for generation. Lower values
            produce more deterministic output. Defaults to 0.1.

    Returns:
        A configured ChatGroq instance ready for use.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
        )

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def get_primary_llm(temperature: float = 0.1) -> ChatGroq:
    """Get the primary LLM for quality tasks (resume analysis, cover letters).

    Uses llama-3.3-70b-versatile by default.

    Args:
        temperature: Sampling temperature. Defaults to 0.1.

    Returns:
        A configured ChatGroq instance with the primary model.
    """
    return get_llm(model=PRIMARY_MODEL, temperature=temperature)


def get_fast_llm(temperature: float = 0.1) -> ChatGroq:
    """Get the fast LLM for high-volume tasks (fit scoring).

    Uses llama-3.1-8b-instant by default.

    Args:
        temperature: Sampling temperature. Defaults to 0.1.

    Returns:
        A configured ChatGroq instance with the fast model.
    """
    return get_llm(model=FAST_MODEL, temperature=temperature)


def get_long_context_llm(temperature: float = 0.1) -> ChatGroq:
    """Get the long-context LLM for processing large documents.

    Uses mixtral-8x7b-32768 by default. Suitable for resumes or JDs
    over 8,000 tokens.

    Args:
        temperature: Sampling temperature. Defaults to 0.1.

    Returns:
        A configured ChatGroq instance with the long-context model.
    """
    return get_llm(model=LONG_CONTEXT_MODEL, temperature=temperature)


def setup_tracing() -> bool:
    """
    Enables LangSmith tracing if LANGCHAIN_API_KEY is present in env.
    Returns True if tracing is enabled, False if skipped.
    LangChain reads these env vars automatically — no code changes
    needed in agents.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key or api_key == "your_langsmith_api_key_here":
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "job-assistant")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )
    return True

setup_tracing()
