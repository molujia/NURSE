import os

import pytest

from nurse_agent.llm_client import OpenAICompatibleClient


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") or os.environ.get("NURSE_RUN_LIVE_OPENAI_TESTS") != "1", reason="Live OpenAI smoke tests require OPENAI_API_KEY and NURSE_RUN_LIVE_OPENAI_TESTS=1")
def test_openai_smoke_returns_text():
    client = OpenAICompatibleClient()

    text = client.complete_text(
        "Return only this exact JSON object: {\"ok\": true}",
        max_output_tokens=512,
    )

    assert "ok" in text.lower()

