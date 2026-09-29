# AI Engineering Demystified Helpers

Shared, versioned helpers for the AI Engineering Demystified course repositories.

```python
from aied_helpers import get_llm

llm = get_llm()  # OpenRouter + NVIDIA Nemotron 3 Ultra (free)
```

Set `OPENROUTER_API_KEY` before invoking the default model. Provider and model can always be chosen explicitly:

```python
llm = get_llm(provider="openai", model="gpt-5-mini")
```

This package deliberately does not load `.env` files during import and never selects a Databricks profile automatically.

## Verify the default provider

Run [`examples/01_openrouter_smoke.ipynb`](examples/01_openrouter_smoke.ipynb) after installing the package and setting `OPENROUTER_API_KEY`. Its first code cell confirms that `get_llm()` creates the OpenRouter client for NVIDIA Nemotron 3 Ultra (free); its second code cell makes a minimal live request.
