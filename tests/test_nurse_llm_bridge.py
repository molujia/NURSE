from nurse import llm_bridge
from nurse.llm_bridge import NurseLLMBridge, llm_candidates_to_evidence


def test_llm_bridge_skips_cleanly_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "")

    result = NurseLLMBridge().propose_evidence(
        task="nurse_evidence_proposal",
        payload={"case_id": "case-1"},
        model="gpt-5-nano",
    )

    assert result["status"] == "skipped"
    assert "OPENAI_API_KEY" in result["reason"]


def test_llm_json_candidates_are_converted_to_unverified_evidence():
    candidates = llm_candidates_to_evidence(
        case_id="case-1",
        raw_candidates=[
            {
                "entity": "paymentservice",
                "metric": "error_rate",
                "kind": "metric",
                "confidence": "0.7",
                "reason": "spike after alarm",
            }
        ],
        time_range={"start": 1, "end": 2},
        model="gpt-5-nano",
    )

    assert candidates[0].entity == "payment"
    assert candidates[0].provenance["source"] == "llm_proposal"
    assert candidates[0].claim["llm_reason"] == "spike after alarm"
    assert candidates[0].claim["metric_role"] == "llm_verified_check"
    assert 0.0 <= candidates[0].claim["evidence_program_weight"] <= 0.05


def test_llm_candidate_confidence_is_bounded_before_verification():
    candidates = llm_candidates_to_evidence(
        case_id="case-1",
        raw_candidates=[
            {
                "entity": "paymentservice",
                "metric": "cpu",
                "kind": "metric",
                "confidence": 80.0,
            }
        ],
        time_range={"start": 1, "end": 2},
        model="gpt-5-nano",
    )

    assert candidates[0].confidence == 1.0
    assert candidates[0].claim["evidence_program_weight"] == 0.05


def test_llm_bridge_retries_once_when_json_response_is_malformed(monkeypatch):
    calls = []

    def fake_complete(self, *, config, system_prompt, user_payload, max_output_tokens):
        calls.append((config, system_prompt, user_payload, max_output_tokens))
        if len(calls) == 1:
            return '{"candidates": [{"entity": "payment"', {"total_tokens": 3}
        return (
            '{"candidates": [{"entity": "payment", "metric": "cpu"}], '
            '"hypotheses": [], "candidate_rules": []}',
            {"total_tokens": 12},
        )

    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "test-key")
    monkeypatch.setattr(NurseLLMBridge, "_complete_json_text", fake_complete)

    result = NurseLLMBridge().propose_evidence(
        task="nurse_evidence_proposal",
        payload={"case_id": "case-1"},
        model="gpt-5-nano",
    )

    assert result["status"] == "ok"
    assert result["retry_count"] == 1
    assert len(calls) == 2
    assert result["candidates"][0]["entity"] == "payment"


def test_llm_bridge_retains_rank_repair_proposals(monkeypatch):
    def fake_complete(self, *, config, system_prompt, user_payload, max_output_tokens):
        return (
            '{"rank_repair": ['
            '{"promote": "checkout", "to_rank": 1, '
            '"supporting_evidence_ids": ["ev-checkout"], "confidence": 0.9}'
            '], "candidates": [], "hypotheses": []}',
            {"total_tokens": 20},
        )

    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "test-key")
    monkeypatch.setattr(NurseLLMBridge, "_complete_json_text", fake_complete)

    result = NurseLLMBridge().propose_evidence(
        task="nurse_rank_repair",
        payload={"output_contract": "rank_repair_only", "candidate_cards": []},
        model="gpt-5-nano",
    )

    assert result["status"] == "ok"
    assert result["rank_repair"][0]["promote"] == "checkout"
    assert result["proposal"]["rank_repair"][0]["supporting_evidence_ids"] == ["ev-checkout"]


def test_llm_candidates_are_filtered_by_service_candidates():
    candidates = llm_candidates_to_evidence(
        case_id="case-1",
        raw_candidates=[
            {"entity": "code", "metric": "error", "kind": "metric"},
            {"entity": "paymentservice", "metric": "error_rate", "kind": "metric"},
        ],
        time_range={"start": 1, "end": 2},
        model="gpt-5-nano",
        service_candidates=["paymentservice", "checkoutservice"],
    )

    assert [candidate.entity for candidate in candidates] == ["payment"]


def test_openai_config_uses_openai_key_and_optional_base_url(monkeypatch):
    def fake_env(name):
        return {
            "NURSE_OPENAI_MODEL": "gpt-5-nano",
            "NURSE_OPENAI_BASE_URL": "https://example.test/v1",
        }.get(name, "")

    monkeypatch.setattr(llm_bridge, "_get_env_var", fake_env)
    config = NurseLLMBridge()._model_config("gpt-5-nano")

    assert config["env_key"] == "OPENAI_API_KEY"
    assert config["model_name"] == "gpt-5-nano"
    assert config["base_url"] == "https://example.test/v1"


def test_custom_model_still_uses_openai_compatible_config(monkeypatch):
    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "")

    config = NurseLLMBridge()._model_config("provider-model-id")

    assert config["env_key"] == "OPENAI_API_KEY"
    assert config["model_name"] == "provider-model-id"
    assert config["base_url"] == "https://api.openai.com/v1"


def test_llm_bridge_uses_rank_repair_only_contract(monkeypatch):
    captured = {"system_prompt": None, "max_output_tokens": None}

    def fake_complete(self, *, config, system_prompt, user_payload, max_output_tokens):
        captured["system_prompt"] = system_prompt
        captured["max_output_tokens"] = max_output_tokens
        return '{"rank_repair": []}', {"total_tokens": 7}

    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "test-key")
    monkeypatch.setattr(NurseLLMBridge, "_complete_json_text", fake_complete)

    result = NurseLLMBridge().propose_evidence(
        task="nurse_rank_repair",
        payload={"output_contract": "rank_repair_only", "candidate_cards": []},
        model="gpt-5-nano",
    )

    assert result["status"] == "ok"
    assert captured["max_output_tokens"] >= 3600
    assert "Return only valid JSON" in captured["system_prompt"]
    assert "rank_repair" in captured["system_prompt"]
    assert "candidate_rules" not in captured["system_prompt"]


def test_llm_bridge_uses_grounded_user_manager_contract(monkeypatch):
    captured = {"system_prompt": None, "max_output_tokens": None}

    def fake_complete(self, *, config, system_prompt, user_payload, max_output_tokens):
        captured["system_prompt"] = system_prompt
        captured["max_output_tokens"] = max_output_tokens
        return (
            '{"summary":"checkout is supported by ev-checkout",'
            '"top_candidate_explanation":"ev-checkout supports checkout",'
            '"evidence_bound_next_steps":["inspect checkout cpu"],'
            '"advisory_notes":[],"unsupported_claims":[]}',
            {"total_tokens": 31},
        )

    monkeypatch.setattr(llm_bridge, "_get_env_var", lambda name: "test-key")
    monkeypatch.setattr(NurseLLMBridge, "_complete_json_text", fake_complete)

    result = NurseLLMBridge().propose_evidence(
        task="nurse_user_manager",
        payload={
            "output_contract": "grounded_rcl_user_proposal",
            "final_ranking": ["checkout", "frontend"],
            "evidence_ids": {"checkout": ["ev-checkout"]},
        },
        model="gpt-5-nano",
    )

    assert result["status"] == "ok"
    assert result["proposal"]["summary"].startswith("checkout")
    assert result["rank_repair"] == []
    assert "must not change" in captured["system_prompt"].lower()
    assert "unsupported_claims" in captured["system_prompt"]
    assert captured["max_output_tokens"] == 1200