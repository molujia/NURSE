from nurse.experiment_modes import load_experiment_mode


def test_without_symbolic_score_disables_llm_rank_facts():
    mode = load_experiment_mode("wo_L")

    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is False
    assert mode.use_llm_rank_repair is False
    assert mode.ranker_kwargs()["symbolic_score_enabled"] is False
