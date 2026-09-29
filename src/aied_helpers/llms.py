"""Factories for LangChain chat models used throughout the course repositories."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b-20260604:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set before creating this model client.")
    return value


def get_openai_llm(model: str = "gpt-5-mini", temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """Return an OpenAI chat model using ``OPENAI_API_KEY``."""
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def get_openrouter_llm(
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return an OpenRouter chat model using its OpenAI-compatible API."""
    return ChatOpenAI(
        model=model,
        api_key=_required_environment("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        **kwargs,
    )


def get_databricks_llm(model: str, temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """Return a Databricks AI Gateway chat model for an explicit model-service name."""
    host = _required_environment("DATABRICKS_HOST").rstrip("/")
    return ChatOpenAI(
        model=model,
        api_key=_required_environment("DATABRICKS_TOKEN"),
        base_url=f"{host}/ai-gateway/mlflow/v1",
        temperature=temperature,
        **kwargs,
    )


def get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0, **kwargs: Any):
    """Return a Groq chat model using the optional ``langchain-groq`` extra."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as error:
        raise ImportError("Install aied-helpers[groq] to use the Groq provider.") from error
    return ChatGroq(model=model, temperature=temperature, **kwargs)


def get_experientiallabs_llm(
    model: str = "gpt-5.6-luna", temperature: float = 0, **kwargs: Any
) -> ChatOpenAI:
    """Return an Experiential Labs OpenAI-compatible chat model."""
    return ChatOpenAI(
        model=model,
        api_key=_required_environment("EXPERIENTIALLABS_API_KEY"),
        base_url="https://api.experientiallabs.ai/v1",
        temperature=temperature,
        **kwargs,
    )


_FACTORIES: dict[str, Callable[..., Any]] = {
    "openrouter": get_openrouter_llm,
    "openai": get_openai_llm,
    "databricks": get_databricks_llm,
    "groq": get_groq_llm,
    "experientiallabs": get_experientiallabs_llm,
}


def get_llm(
    *,
    provider: str = "openrouter",
    model: str | None = None,
    temperature: float = 0,
    **kwargs: Any,
):
    """Return a chat model; OpenRouter's free Nemotron 3 Ultra is the default."""
    factory = _FACTORIES.get(provider)
    if factory is None:
        choices = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"Unknown provider {provider!r}. Choose from: {choices}.")
    if provider == "databricks" and model is None:
        raise ValueError(
            "model must be an explicit Databricks model-service name, such as a system.ai.* service."
        )

    arguments = {"temperature": temperature, **kwargs}
    if model is not None:
        arguments["model"] = model
    return factory(**arguments)
