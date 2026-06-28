from nurse.llm_providers import provider_for_model


def test_provider_config_routes_known_models():
    assert provider_for_model("gpt-5-nano").name == "openai"
    assert provider_for_model("ep-20260623160517-ckhj8").name == "volcano"
    assert provider_for_model("glm-4.5-air").name == "zhipu"
