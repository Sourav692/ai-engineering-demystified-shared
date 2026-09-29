"""Factories for embedding models."""

from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings


def get_embeddings(*, provider: str = "openai", model: str | None = None, **kwargs: Any):
    """Return an embedding model for an explicit supported provider."""
    if provider == "openai":
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", **kwargs)
    if provider == "databricks":
        try:
            from databricks_langchain import DatabricksEmbeddings
        except ImportError as error:
            raise ImportError("Install aied-helpers[databricks] to use Databricks embeddings.") from error
        return DatabricksEmbeddings(endpoint=model or "databricks-gte-large-en", **kwargs)
    raise ValueError("Unknown embedding provider. Choose from: databricks, openai.")
