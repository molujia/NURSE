from nurse.experiment_modes import experiment_mode_choices, load_experiment_mode


def test_experiment_mode_choices_include_new_ephr_ablations():
    choices = experiment_mode_choices()

    assert choices == sorted(choices)
    assert choices == ["explanation_only", "full", "wo_E", "wo_H", "wo_L", "wo_P", "wo_R"]


def test_full_mode_only_uses_semantic_fact_agent_for_rank_updates():
    mode = load_experiment_mode("full")

    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is True
    assert mode.use_llm_rank_repair is False
    assert mode.symbolic_score_enabled is True
    kwargs = mode.ranker_kwargs()
    assert kwargs["allow_llm_scoring"] is True
    assert kwargs["symbolic_score_enabled"] is True


def test_wo_e_disables_direct_evidence_term():
    mode = load_experiment_mode("wo_E")

    assert mode.call_llm is False
    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is False
    assert mode.use_llm_rank_repair is False
    kwargs = mode.ranker_kwargs()
    assert kwargs["lambda_e"] == 0.0
    assert kwargs["allow_llm_scoring"] is False


def test_wo_h_disables_history_prior_only():
    mode = load_experiment_mode("wo_H")

    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is True
    kwargs = mode.ranker_kwargs()
    assert kwargs["lambda_h"] == 0.0
    assert kwargs["symbolic_score_enabled"] is True


def test_wo_p_disables_propagation_only():
    mode = load_experiment_mode("wo_P")

    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is True
    kwargs = mode.ranker_kwargs()
    assert kwargs["lambda_p"] == 0.0
    assert kwargs["symbolic_score_enabled"] is True


def test_wo_l_disables_semantic_fact_agent():
    mode = load_experiment_mode("wo_L")

    assert mode.call_llm is False
    assert mode.use_llm_candidates is False
    assert mode.use_llm_hypotheses is False
    assert mode.use_llm_semantic_prior is False
    assert mode.use_llm_rank_repair is False
    assert mode.symbolic_score_enabled is False
    kwargs = mode.ranker_kwargs()
    assert kwargs["symbolic_score_enabled"] is False


def test_wo_r_disables_sre_rule_term_only():
    mode = load_experiment_mode("wo_R")

    assert mode.use_llm_semantic_prior is True
    kwargs = mode.ranker_kwargs()
    assert kwargs["lambda_r"] == 0.0
