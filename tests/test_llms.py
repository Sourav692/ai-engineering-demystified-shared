import pytest

from aied_helpers.llms import _FACTORIES, DEFAULT_OPENROUTER_MODEL, get_llm, get_openrouter_llm


def test_openrouter_is_the_default_provider_and_nemotron_is_free_default():
    assert DEFAULT_OPENROUTER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b-20260604:free"
    assert "openrouter" in _FACTORIES
    assert "databricks_gateway" not in _FACTORIES


def test_openrouter_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_openrouter_llm()


def test_databricks_requires_an_explicit_model_service():
    with pytest.raises(ValueError, match="explicit Databricks model-service"):
        get_llm(provider="databricks")
