from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def schemas() -> ModuleType:
    return importlib.import_module("nurse_rcl.v2.ranking_schemas")


def interval(start: int = 1_000, end: int = 1_200) -> Any:
    observation = importlib.import_module("nurse_rcl.v2.observation_schemas")
    return observation.TimeInterval(start_unix_ns=start, end_unix_ns=end)


def edge(
    module: ModuleType,
    edge_id: str,
    caller: str,
    callee: str,
) -> Any:
    return module.GraphEdge(
        edge_id=edge_id,
        caller_service_key=caller,
        callee_service_key=callee,
        relation_type="service_call",
        direction="caller_to_callee",
        interval=interval(),
        coverage_quality=1.0,
        request_count=10,
        error_count=2,
        latency_p95=0.2,
        change_state="increased_error",
        provenance_ids=("trace:source-a",),
    )


def score(
    module: ModuleType,
    service: str,
    value: float,
    *,
    total_score: float | None = None,
) -> Any:
    expected_total = 0.75 * value
    return module.build_score_decomposition(
        candidate_service_key=service,
        event_evidence_score=value,
        propagation_score=0.0,
        rule_score=0.0,
        history_score=0.0,
        lambda_p=0.25,
        eta_r=0.0,
        eta_h=0.0,
        total_score=expected_total if total_score is None else total_score,
        event_group_ids=("evt_" + "1" * 24,),
        propagation_path_ids=(),
        rule_ids=(),
        history_neighbor_ids=(),
        scoring_configuration_sha256="a" * 64,
    )


def test_graph_snapshot_preserves_caller_to_callee_and_shared_association() -> None:
    module = schemas()
    graph = module.build_graph_snapshot(
        opaque_case_id="opq_" + "2" * 32,
        window_kind="fault",
        observation_contract_sha256="b" * 64,
        nodes=("checkout", "payment"),
        edges=(edge(module, "edge-a", "checkout", "payment"),),
    )

    assert graph.verify()
    assert graph.edges[0].direction == "caller_to_callee"
    invalid = graph.edges[0].model_dump(mode="python")
    invalid["relation_type"] = "shared_resource_association"
    with pytest.raises(ValidationError, match="shared resource association is undirected"):
        module.GraphEdge.model_validate(invalid)


def test_scc_provenance_retains_original_cycle_without_internal_source() -> None:
    module = schemas()
    graph = module.build_graph_snapshot(
        opaque_case_id="opq_" + "2" * 32,
        window_kind="fault",
        observation_contract_sha256="b" * 64,
        nodes=("service-a", "service-b"),
        edges=(
            edge(module, "edge-a", "service-a", "service-b"),
            edge(module, "edge-b", "service-b", "service-a"),
        ),
    )
    component = module.SCCComponent(
        component_id="scc_" + "3" * 24,
        member_service_keys=("service-a", "service-b"),
        internal_edge_ids=("edge-a", "edge-b"),
        internal_source_service_key=None,
    )
    provenance = module.build_scc_provenance(
        graph_snapshot=graph,
        relation_type="service_call",
        components=(component,),
        condensed_edges=(),
    )

    assert provenance.verify()
    assert provenance.retained_original_edge_ids == ("edge-a", "edge-b")
    with pytest.raises(ValidationError):
        module.SCCComponent.model_validate(
            {
                **component.model_dump(mode="python"),
                "internal_source_service_key": "service-a",
            }
        )


def test_score_decomposition_is_recomputable_and_not_a_probability() -> None:
    module = schemas()
    decomposition = score(module, "checkout", 0.4)

    assert decomposition.verify()
    assert decomposition.total_score == pytest.approx(0.3)
    assert decomposition.quantized_total_score == "0.300000000000"
    assert decomposition.is_root_cause_probability is False
    with pytest.raises(ValueError, match="total score does not match decomposition"):
        module.build_score_decomposition(
            candidate_service_key="checkout",
            event_evidence_score=0.4,
            propagation_score=0.0,
            rule_score=0.0,
            history_score=0.0,
            lambda_p=0.25,
            eta_r=0.0,
            eta_h=0.0,
            total_score=0.9,
            event_group_ids=("evt_" + "1" * 24,),
            propagation_path_ids=(),
            rule_ids=(),
            history_neighbor_ids=(),
            scoring_configuration_sha256="a" * 64,
        )


def test_score_replay_accepts_bounded_error_in_the_same_quantization_bucket() -> None:
    module = schemas()
    decomposition = score(
        module,
        "bounded-replay",
        0.4,
        total_score=0.3 + 4e-13,
    )

    assert decomposition.verify()
    assert decomposition.quantized_total_score == "0.300000000000"


def test_score_replay_rejects_error_above_absolute_tolerance() -> None:
    module = schemas()

    with pytest.raises(ValueError, match="total score does not match decomposition"):
        score(
            module,
            "outside-absolute-tolerance",
            0.4,
            total_score=0.3 + 2e-10,
        )


def test_score_replay_rejects_in_tolerance_total_in_another_quantization_bucket() -> None:
    module = schemas()

    with pytest.raises(ValueError, match="quantized total differs"):
        score(
            module,
            "different-quantization-bucket",
            0.4,
            total_score=0.3 + 6e-13,
        )


def test_tie_group_and_top_k_keep_hash_tiebreak_separate_from_evidence() -> None:
    module = schemas()
    alpha = score(module, "alpha", 0.4)
    beta = score(module, "beta", 0.4)
    tie = module.build_tie_group(scores=(alpha, beta))
    seed_sha256 = "c" * 64
    expected_order = tuple(
        sorted(
            ("alpha", "beta"),
            key=lambda service: sha256_hex(
                canonical_json_bytes(
                    {
                        "seed_sha256": seed_sha256,
                        "service_entity_key": service,
                    }
                )
            ),
        )
    )
    ranking = module.build_top_k(
        opaque_case_id="opq_" + "2" * 32,
        candidate_universe_sha256="1" * 64,
        scores=(alpha, beta),
        tie_groups=(tie,),
        ordered_service_keys=expected_order,
        k=2,
        tie_break_seed_sha256=seed_sha256,
    )

    assert tie.verify() and ranking.verify()
    assert ranking.opaque_case_id == "opq_" + "2" * 32
    assert ranking.candidate_universe_sha256 == "1" * 64
    assert tie.member_service_keys == ("alpha", "beta")
    assert ranking.entries[0].service_entity_key == expected_order[0]
    assert ranking.entries[0].evidence_tie_group_id == tie.tie_group_id
    assert ranking.entries[0].tie_break_is_evidence is False
    with pytest.raises(ValueError, match="declared deterministic ordering"):
        module.build_top_k(
            opaque_case_id="opq_" + "2" * 32,
            candidate_universe_sha256="1" * 64,
            scores=(alpha, beta),
            tie_groups=(tie,),
            ordered_service_keys=tuple(reversed(expected_order)),
            k=2,
            tie_break_seed_sha256=seed_sha256,
        )


def test_rules_are_explicit_and_no_directional_default_is_possible() -> None:
    module = schemas()
    rule = module.build_scoring_rule(
        rule_key="promote_verified_local_pressure",
        rule_version="1",
        direction="promote",
        strength=0.2,
        predicate_result_ids=("prd_" + "4" * 24,),
        enabled=True,
        provenance_ids=("operator:declared-rule",),
    )
    rules = module.build_rule_set(rules=(rule,))

    assert rule.verify() and rules.verify()
    assert rules.default_directional_rules_enabled is False
    with pytest.raises(ValidationError):
        module.RuleSet.model_validate(
            {
                **rules.model_dump(mode="python"),
                "default_directional_rules_enabled": True,
            }
        )


def test_history_modes_are_isolated_nonrecursive_and_group_safe() -> None:
    module = schemas()
    off = module.build_history_context(
        dataset="rcabench",
        current_opaque_case_id="opq_" + "5" * 32,
        current_group_fingerprint="d" * 64,
        mode="off",
        library_sha256=None,
        neighbors=(),
    )
    assert off.verify() and off.neighbors == ()

    neighbor = module.HistoryNeighbor(
        neighbor_id="nbr_" + "6" * 24,
        source_opaque_case_id="opq_" + "7" * 32,
        source_group_fingerprint="e" * 64,
        source_history_mode="off",
        similarity=0.8,
        base_score_artifact_sha256="f" * 64,
        hint_reference="pattern:pressure",
        score_contribution=0.0,
    )
    hint = module.build_history_context(
        dataset="rcabench",
        current_opaque_case_id="opq_" + "5" * 32,
        current_group_fingerprint="d" * 64,
        mode="hint",
        library_sha256="1" * 64,
        neighbors=(neighbor,),
    )
    assert hint.verify()
    with pytest.raises(ValueError, match="current case or incident group"):
        module.build_history_context(
            dataset="rcabench",
            current_opaque_case_id=neighbor.source_opaque_case_id,
            current_group_fingerprint="d" * 64,
            mode="hint",
            library_sha256="1" * 64,
            neighbors=(neighbor,),
        )
    with pytest.raises(ValidationError, match="hint mode cannot inject score"):
        module.HistoryContext.model_validate(
            {
                **hint.model_dump(mode="python"),
                "neighbors": (neighbor.model_copy(update={"score_contribution": 0.1}),),
            }
        )


def test_frozen_report_context_binds_ranking_and_cannot_be_mutated() -> None:
    module = schemas()
    alpha = score(module, "alpha", 0.4)
    tie = module.build_tie_group(scores=(alpha,))
    ranking = module.build_top_k(
        opaque_case_id="opq_" + "2" * 32,
        candidate_universe_sha256="1" * 64,
        scores=(alpha,),
        tie_groups=(tie,),
        ordered_service_keys=("alpha",),
        k=1,
        tie_break_seed_sha256="c" * 64,
    )
    context = module.build_frozen_report_context(
        opaque_case_id="opq_" + "2" * 32,
        candidate_universe_sha256="1" * 64,
        focus_set_sha256="2" * 64,
        observation_set_sha256="3" * 64,
        entity_binding_set_sha256="4" * 64,
        event_group_set_sha256="5" * 64,
        hypothesis_set_sha256="6" * 64,
        graph_snapshot_sha256_by_window={"normal": "7" * 64, "fault": "8" * 64},
        scc_provenance_sha256="9" * 64,
        scores=(alpha,),
        tie_groups=(tie,),
        top_k=ranking,
        rule_set_sha256="a" * 64,
        history_context_sha256="b" * 64,
        frozen_at_utc="2026-09-05T09:00:00Z",
    )

    assert context.verify()
    assert context.schema_version == "nurse.v2.frozen-report-context/2"
    assert context.ranking_mutable is False
    changed = context.model_copy(update={"candidate_universe_sha256": "f" * 64})
    assert not changed.verify()

    legacy_top_document = ranking.model_dump(mode="json")
    legacy_top_document.pop("opaque_case_id")
    legacy_top_document.pop("candidate_universe_sha256")
    legacy_top_document["schema_version"] = "nurse.v2.top-k/1"
    legacy_top_unsigned = {
        key: value for key, value in legacy_top_document.items() if key != "ranking_sha256"
    }
    legacy_top_document["ranking_sha256"] = sha256_hex(canonical_json_bytes(legacy_top_unsigned))
    legacy_top = module.TopKArtifactV1.model_validate(legacy_top_document)
    assert legacy_top.verify()

    legacy_context_document = context.model_dump(mode="json")
    legacy_context_document["schema_version"] = "nurse.v2.frozen-report-context/1"
    legacy_context_document["top_k"] = legacy_top.model_dump(mode="json")
    legacy_context_unsigned = {
        key: value for key, value in legacy_context_document.items() if key != "context_sha256"
    }
    legacy_context_document["context_sha256"] = sha256_hex(
        canonical_json_bytes(legacy_context_unsigned)
    )
    legacy_context = module.FrozenReportContextV1.model_validate(legacy_context_document)
    assert legacy_context.verify()
    with pytest.raises(ValidationError):
        module.FrozenReportContext.model_validate(legacy_context.model_dump(mode="json"))
