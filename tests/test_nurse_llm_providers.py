from __future__ import annotations

import pytest

from nurse.llm_providers import provider_for_model


@pytest.mark.parametrize(
    "model",
    [
        "ep-20260623160517-ckhj8",
        "ep-20260623103508-6dkw6",
        "ep-20260628204018-8mnz4",
    ],
)
def test_volcano_model_ids_route_to_volcano_provider(model):
    provider = provider_for_model(model)

    assert provider.name == "volcano"
    assert provider.env_key == "VOLCANO_API_KEY"
    assert provider.uses_openai_client is True
