import pytest

from nurse_agent.tools.incident_prior_tool import run


def test_incident_prior_inactive_without_history():
    result = run(
        {
            "case_id": "c1",
            "dataset": "rcaeval-tt",
            "service_candidates": ["adservice", "checkoutservice"],
        }
    )

    assert result["incident_prior"]["active"] is False
    assert result["incident_prior"]["method"] == "none"
    assert result["incident_prior"]["score_contribution"] == {}


def test_incident_prior_with_approved_history_is_bounded_and_cited():
    result = run(
        {
            "case_id": "c1",
            "dataset": "rcaeval-tt",
            "service_candidates": ["adservice", "checkoutservice"],
            "history_index": [
                {
                    "incident_id": "H1",
                    "service": "adservice",
                    "similarity": 0.5,
                    "historical_score": 0.4,
                }
            ],
        }
    )

    prior = result["incident_prior"]
    assert prior["active"] is True
    assert prior["matched_history"][0]["incident_id"] == "H1"
    assert prior["score_contribution"]["adservice"] == pytest.approx(0.2)


def test_incident_prior_rejects_forbidden_history_fields():
    with pytest.raises(ValueError, match="forbidden inference field"):
        run(
            {
                "case_id": "c1",
                "dataset": "rcaeval-tt",
                "service_candidates": ["adservice"],
                "history_index": [{"incident_id": "H1", "root_cause": "adservice"}],
            }
        )


def test_incident_prior_can_use_bounded_llm_similarity(monkeypatch):
    from nurse_agent.tools import incident_prior_tool

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete_text(self, prompt, *, max_output_tokens=512):
            assert "H1" in prompt
            assert "answer_key" not in prompt
            return "H1 is the closest approved historical incident."

    monkeypatch.setattr(incident_prior_tool, "OpenAICompatibleClient", FakeClient)

    result = run(
        {
            "case_id": "c1",
            "dataset": "rcaeval-tt",
            "service_candidates": ["adservice"],
            "use_llm": True,
            "history_index": [
                {
                    "incident_id": "H1",
                    "service": "adservice",
                    "similarity": 0.25,
                    "historical_score": 0.8,
                    "summary": "memory pressure near auth path",
                }
            ],
        }
    )

    prior = result["incident_prior"]
    assert prior["active"] is True
    assert prior["method"] == "retrieval_then_llm_similarity"
    assert "H1" in prior["rationale"]
    assert prior["score_contribution"]["adservice"] == pytest.approx(0.2)
