from nurse.llm_payload import build_candidate_cards, build_grounded_user_manager_payload, build_rank_repair_payload
from nurse.schemas import AlarmArtifact, NursePrediction, VerifiedEvidence, VerifiedRCLContext


def _evidence(evidence_id, entity, signal, metric_role):
    return VerifiedEvidence(
        evidence_id=evidence_id,
        case_id="case-hidden",
        entity=entity,
        signal=signal,
        kind="metric",
        time_range={"start": 1, "end": 2},
        claim={"metric_role": metric_role, "evidence_program_weight": 2.5},
        provenance={"source": "raw_metric"},
        observed={"observed": 7.0, "baseline": 1.0},
        verification_rule="unit-test",
        verified=True,
    )


def _prediction():
    return NursePrediction(
        case_id="case-hidden",
        ranked=["frontend", "checkout", "ad"],
        scores={"frontend": 9.0, "checkout": 7.0, "ad": 6.0},
        score_breakdown={
            "frontend": {"symptom_penalty": -0.55, "metric_deviation_magnitude": 2.0},
            "checkout": {"counterfactual_drop": 1.2, "incident_shape_topology_locality": 0.5},
            "ad": {"source_evidence": 0.7, "evidence_program": 0.9},
        },
        evidence_ids={
            "frontend": ["ev-front"],
            "checkout": ["ev-checkout"],
            "ad": ["ev-ad"],
        },
        closure_paths=[{"type": "propagates_to", "src": "ad", "dst": "frontend"}],
    )


def test_candidate_cards_expose_root_vs_symptom_signals_without_labels():
    cards = build_candidate_cards(
        prediction=_prediction(),
        verified_evidence=[
            _evidence("ev-front", "frontend", "rrt", "symptom_metric"),
            _evidence("ev-checkout", "checkout", "cpu", "generic_metric"),
            _evidence("ev-ad", "ad", "cpu", "source_metric"),
        ],
        topology_edges=[("ad", "frontend"), ("checkout", "frontend")],
        max_cards=3,
    )

    assert [card["service"] for card in cards] == ["frontend", "checkout", "ad"]
    assert cards[0]["symptom_like_evidence"] == 1
    assert cards[2]["source_like_evidence"] == 1
    assert cards[2]["out_degree"] == 1
    assert cards[0]["in_degree"] == 2
    assert cards[2]["supporting_evidence_ids"] == ["ev-ad"]
    assert "case_id" not in cards[0]
    assert "root_cause" not in str(cards).lower()
    assert "answer_key" not in str(cards).lower()


def test_rank_repair_payload_is_compact_and_rank_repair_only():
    alarm = AlarmArtifact(
        case_id="case-hidden",
        dataset="rcaeval-tt",
        trigger_time=123,
        window={"normal_start": 0, "normal_end": 100, "fault_start": 101, "fault_end": 200},
        alert_entity="frontend",
        signals=[],
        provenance={"source": "benchmark_trigger_fixed_window_v1"},
    )

    payload = build_rank_repair_payload(
        alarm=alarm,
        service_candidates=["frontend", "checkout", "ad"],
        prediction=_prediction(),
        verified_evidence=[_evidence("ev-ad", "ad", "cpu", "source_metric")],
        topology_edges=[("ad", "frontend")],
        prompt_profile="recall-heavy",
    )

    assert payload["output_contract"] == "rank_repair_only"
    assert payload["incident_ref"] == "current-incident"
    assert payload["current_ranking"] == ["frontend", "checkout", "ad"]
    assert payload["candidate_cards"][2]["service"] == "ad"
    assert "case-hidden" not in str(payload)
    assert len(str(payload)) < 12000

def test_rank_repair_payload_limits_cards_and_mentions_symptom_heavy_entrypoint():
    ranked = [f"svc-{index}" for index in range(15)]
    prediction = NursePrediction(
        case_id="case-hidden",
        ranked=ranked,
        scores={service: 100.0 - index for index, service in enumerate(ranked)},
        score_breakdown={service: {"generic_evidence": 1.0} for service in ranked},
        evidence_ids={service: [f"ev-{service}"] for service in ranked},
        closure_paths=[],
    )
    alarm = AlarmArtifact(
        case_id="case-hidden",
        dataset="rcaeval-tt",
        trigger_time=123,
        window={"normal_start": 0, "normal_end": 100, "fault_start": 101, "fault_end": 200},
        alert_entity="frontend",
        signals=[],
        provenance={"source": "benchmark_trigger_fixed_window_v1"},
    )

    payload = build_rank_repair_payload(
        alarm=alarm,
        service_candidates=ranked,
        prediction=prediction,
        verified_evidence=[_evidence(f"ev-{service}", service, "cpu", "generic_metric") for service in ranked],
        topology_edges=[],
        prompt_profile="recall-heavy",
    )

    assert len(payload["candidate_cards"]) == 12
    assert "symptom-heavy" in payload["repair_policy"]["entrypoint_repair"]
    assert "Top-2..Top-5" in payload["repair_policy"]["entrypoint_repair"]

def test_aiops_payload_marks_semantic_roles_without_labels():
    prediction = NursePrediction(
        case_id="case-hidden",
        ranked=["frontend", "ad", "k8s-master2"],
        scores={"frontend": 3.0, "ad": 2.0, "k8s-master2": 1.0},
        score_breakdown={"frontend": {}, "ad": {}, "k8s-master2": {}},
        evidence_ids={"frontend": ["ev-frontend"], "ad": ["ev-ad"], "k8s-master2": ["ev-k8s"]},
        closure_paths=[],
    )

    cards = build_candidate_cards(
        prediction=prediction,
        verified_evidence=[
            _evidence("ev-frontend", "frontend", "rrt", "symptom_metric"),
            _evidence("ev-ad", "ad", "cpu", "source_metric"),
            _evidence("ev-k8s", "k8s-master2", "node_cpu_usage_rate", "source_metric"),
        ],
        topology_edges=[],
        dataset="aiops25-retained-nonnode",
        max_cards=3,
    )

    by_service = {card["service"]: card for card in cards}
    assert by_service["frontend"]["semantic_role"] == "symptom_prone_entrypoint"
    assert by_service["ad"]["semantic_role"] == "application_candidate"
    assert by_service["k8s-master2"]["semantic_role"] == "blocked_infra_for_retained_nonnode"
    assert "root_cause" not in str(cards).lower()

def test_aiops_rank_repair_payload_matches_top_two_guardrail():
    ranked = ["payment", "ad", "pd", "frontend"]
    prediction = NursePrediction(
        case_id="case-hidden",
        ranked=ranked,
        scores={"payment": 13.0, "ad": 9.0, "pd": 7.0, "frontend": 6.0},
        score_breakdown={service: {"source_evidence": 1.0} for service in ranked},
        evidence_ids={service: [f"ev-{service}"] for service in ranked},
        closure_paths=[],
    )
    alarm = AlarmArtifact(
        case_id="case-hidden",
        dataset="aiops25-retained-nonnode",
        trigger_time=123,
        window={"normal_start": 0, "normal_end": 100, "fault_start": 101, "fault_end": 200},
        alert_entity="payment",
        signals=[],
        provenance={"source": "benchmark_trigger_fixed_window_v1"},
    )

    payload = build_rank_repair_payload(
        alarm=alarm,
        service_candidates=ranked,
        prediction=prediction,
        verified_evidence=[_evidence(f"ev-{service}", service, "cpu", "source_metric") for service in ranked],
        topology_edges=[],
        prompt_profile="balanced",
    )

    assert payload["current_ranking"] == ["payment", "ad"]
    assert [card["service"] for card in payload["candidate_cards"]] == ["payment", "ad"]
    assert "Top-2 application_candidate" in payload["repair_policy"]["entrypoint_repair"]
    assert "Top-2..Top-5" not in payload["repair_policy"]["entrypoint_repair"]


def test_aiops_recall_heavy_rank_repair_payload_exposes_top_five():
    ranked = ["payment", "ad", "product-catalog", "checkout", "cart", "frontend"]
    prediction = NursePrediction(
        case_id="case-hidden",
        ranked=ranked,
        scores={service: 20.0 - index for index, service in enumerate(ranked)},
        score_breakdown={service: {"source_evidence": 1.0} for service in ranked},
        evidence_ids={service: [f"ev-{service}"] for service in ranked},
        closure_paths=[],
    )
    alarm = AlarmArtifact(
        case_id="case-hidden",
        dataset="aiops25-retained-nonnode",
        trigger_time=123,
        window={"normal_start": 0, "normal_end": 100, "fault_start": 101, "fault_end": 200},
        alert_entity="payment",
        signals=[],
        provenance={"source": "benchmark_trigger_fixed_window_v1"},
    )

    payload = build_rank_repair_payload(
        alarm=alarm,
        service_candidates=ranked,
        prediction=prediction,
        verified_evidence=[_evidence(f"ev-{service}", service, "cpu", "source_metric") for service in ranked],
        topology_edges=[],
        prompt_profile="recall-heavy",
    )

    assert payload["current_ranking"] == ranked[:5]
    assert [card["service"] for card in payload["candidate_cards"]] == ranked[:5]
    assert "Top-2..Top-5 application_candidate" in payload["repair_policy"]["entrypoint_repair"]

def test_rcaeval_tt_payload_marks_primary_and_auxiliary_service_roles():
    prediction = NursePrediction(
        case_id="case-hidden",
        ranked=["payment-service", "auth-service", "admin-travel-service"],
        scores={"payment-service": 6.9, "auth-service": 3.0, "admin-travel-service": 2.0},
        score_breakdown={
            "payment-service": {"source_evidence": 1.0},
            "auth-service": {"source_evidence": 1.0},
            "admin-travel-service": {"source_evidence": 1.0},
        },
        evidence_ids={
            "payment-service": ["ev-payment"],
            "auth-service": ["ev-auth"],
            "admin-travel-service": ["ev-admin"],
        },
        closure_paths=[],
    )

    cards = build_candidate_cards(
        prediction=prediction,
        verified_evidence=[
            _evidence("ev-payment", "payment-service", "cpu", "source_metric"),
            _evidence("ev-auth", "auth-service", "cpu", "source_metric"),
            _evidence("ev-admin", "admin-travel-service", "cpu", "source_metric"),
        ],
        topology_edges=[],
        dataset="rcaeval-tt",
        max_cards=3,
    )

    by_service = {card["service"]: card for card in cards}
    assert by_service["auth-service"]["semantic_role"] == "primary_train_ticket_service"
    assert by_service["payment-service"]["semantic_role"] == "support_or_storage_candidate"
    assert by_service["admin-travel-service"]["semantic_role"] == "support_or_storage_candidate"
    assert "case-hidden" not in str(cards)

def test_grounded_user_manager_payload_preserves_rank_and_evidence_boundary():
    context = VerifiedRCLContext(
        case_id="case-hidden",
        dataset="synthetic",
        ranked_candidates=["checkout", "frontend"],
        score_summary={"checkout": 3.0, "frontend": 1.0},
        score_breakdown={"checkout": {"T": 1.0, "C": 2.0}},
        evidence_ids={"checkout": ["ev-checkout"]},
        evidence_summaries=[{"evidence_id": "ev-checkout", "entity": "checkout", "signal": "cpu"}],
        closure_paths=[{"type": "propagates_to", "src": "checkout", "dst": "frontend"}],
        rejected_items=[{"evidence_id": "bad-1", "reason": "missing provenance"}],
        advisory={"rank_repair": [{"promote": "frontend"}]},
        provenance={"rank_path": "evidence_core_v2", "llm_rank_mutation": False},
    )

    payload = build_grounded_user_manager_payload(context=context)

    assert payload["output_contract"] == "grounded_rcl_user_proposal"
    assert payload["final_ranking"] == ["checkout", "frontend"]
    assert payload["evidence_ids"]["checkout"] == ["ev-checkout"]
    assert payload["rank_policy"]["llm_can_change_final_rank"] is False
    assert "answer_key" not in str(payload).lower()
    assert "root_cause" not in str(payload).lower()
