"""Immutable case audit packages over exact NURSE V2 domain models."""

from __future__ import annotations

import importlib
import json
from collections import Counter

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.event_strength import EventStrengthResult, compile_event_strengths
from nurse_rcl.v2.family_aggregation import (
    REFERENCE_KEY,
    FamilyAggregationResult,
    compile_family_aggregation,
)
from nurse_rcl.v2.graph_snapshots import (
    GRAPH_SNAPSHOT_SCHEMA_VERSION,
    GraphSnapshot,
)
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult, evaluate_history
from nurse_rcl.v2.observation_schemas import FocusDecision, SemanticFocusSet, build_focus_set
from nurse_rcl.v2.propagation_scoring import (
    PropagationScoringResult,
    score_propagation_paths,
)
from nurse_rcl.v2.ranking_schemas import TopKArtifact, build_rule_set
from nurse_rcl.v2.reference_ranking import (
    RankingInputAuthority,
    ReferenceRankingResult,
    ReferenceScoringConfig,
    build_reference_scoring_config,
    rank_reference_candidates,
)
from nurse_rcl.v2.rule_scoring import RuleScoringResult, score_explicit_rules
from nurse_rcl.v2.temporal_condensation import build_relation_scc_condensation
from tests.test_v2_propagation_scoring import _authoritative_scenario, _time_contract


def _consistent_fault_graph(scenario: dict[str, object]) -> GraphSnapshot:
    frozen = scenario["frozen_candidates"]
    candidate_universe = scenario["candidate_universe"]
    entity_registry = scenario["entity_registry"]
    prior = scenario["condensations"][0]  # type: ignore[index]
    edge = prior.selected_original_edges[0]
    unsigned: dict[str, object] = {
        "schema_version": GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "opaque_case_id": frozen.opaque_case_id,
        "window_kind": "fault",
        "interval": {"start_unix_ns": 10, "end_unix_ns": 100},
        "decision_time_unix_ns": 100,
        "observation_contract_sha256": candidate_universe.observation_contract_sha256,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "entity_registry_sha256": entity_registry.registry_sha256,
        "node_service_keys": frozen.candidate_service_keys,
        "coverage": edge.coverage.model_dump(mode="json"),
        "edges": tuple(item.model_dump(mode="json") for item in prior.selected_original_edges),
        "provenance_ids": tuple(
            sorted(
                {
                    provenance_id
                    for item in prior.selected_original_edges
                    for provenance_id in (*item.provenance_ids, *item.coverage.provenance_ids)
                }
            )
        ),
    }
    return GraphSnapshot.model_validate(
        {**unsigned, "snapshot_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def _focus_set(scenario: dict[str, object]) -> SemanticFocusSet:
    candidate_universe = scenario["candidate_universe"]
    decisions = tuple(
        FocusDecision(
            service_entity_key=item.service_entity_key,
            selected=True,
            selection_phase="base",
            selection_reasons=("neutral_evidence",),
            exclusion_reason=None,
        )
        for item in candidate_universe.candidates
    )
    return build_focus_set(
        candidate_universe=candidate_universe,
        q_base=len(decisions),
        q_max=len(decisions),
        decisions=decisions,
    )


def build_real_replay_bundle(*, source_strength: float = 0.8) -> object:
    from nurse_rcl.v2.case_audit import (
        build_case_evidence_replay_bundle,
        build_case_ranking_replay_bundle,
    )

    scenario = _authoritative_scenario(
        additional_service_keys=("service-c",),
        source_strength=source_strength,
    )
    graph = _consistent_fault_graph(scenario)
    scenario["condensations"] = (
        build_relation_scc_condensation(
            graph_snapshot=graph,
            relation_type=scenario["condensations"][0].relation_type,  # type: ignore[index]
        ),
    )
    role_inputs = {
        "proposals": scenario["role_proposals"],
        "guard_results": scenario["role_guard_results"],
        "predicates": scenario["role_predicates"],
        "candidate_universe": scenario["candidate_universe"],
        "frozen_candidates": scenario["frozen_candidates"],
        "entity_registry": scenario["entity_registry"],
        "event_grouping": scenario["event_grouping"],
        "mechanism_library": scenario["mechanism_library"],
        "scope_interval": scenario["scope_interval"],
    }
    event_strength = compile_event_strengths(
        role_compilation=scenario["role_compilation"],
        **role_inputs,
    )
    family = compile_family_aggregation(
        event_strength=event_strength,
        aggregator_version=REFERENCE_KEY,
        role_compilation=scenario["role_compilation"],
        **role_inputs,
    )
    propagation = score_propagation_paths(**scenario)
    frozen = scenario["frozen_candidates"]
    rule_set = build_rule_set(rules=())
    rules = score_explicit_rules(
        frozen_candidates=frozen,
        rule_set=rule_set,
        assignments=(),
        predicates=(),
    )
    history = evaluate_history(frozen_candidates=frozen, mode="off")
    authority = RankingInputAuthority(
        candidate_freeze_sha256=frozen.freeze_sha256,
        family_aggregation_result_sha256=family.result_sha256,
        propagation_scoring_result_sha256=propagation.result_sha256,
        propagation_authoritative_input_sha256=propagation.authoritative_input_sha256,
        rule_scoring_result_sha256=rules.result_sha256,
        rule_authoritative_input_sha256=rules.authoritative_input_sha256,
        history_evaluation_result_sha256=history.result_sha256,
        history_authoritative_input_sha256=history.authoritative_input_sha256,
    )
    config = build_reference_scoring_config(history_mode="off")
    ranking = rank_reference_candidates(
        frozen_candidates=frozen,
        family_aggregation=family,
        propagation_scoring=propagation,
        rule_scoring=rules,
        history_evaluation=history,
        authority=authority,
        config=config,
    )
    evidence = build_case_evidence_replay_bundle(
        time_contract=_time_contract(),
        entity_registry=scenario["entity_registry"],
        event_grouping=scenario["event_grouping"],
        candidate_universe=scenario["candidate_universe"],
        frozen_candidates=frozen,
        focus_set=_focus_set(scenario),
        semantic_context=None,
        card_expansion_policy=None,
        query_sessions=(),
        mechanism_library=scenario["mechanism_library"],
        scope_interval=scenario["scope_interval"],
        role_predicates=scenario["role_predicates"],
        role_guard_results=scenario["role_guard_results"],
        role_proposals=scenario["role_proposals"],
        role_compilation=scenario["role_compilation"],
        event_strength=event_strength,
        graph_snapshots=(graph,),
        condensations=scenario["condensations"],
        onsets=scenario["onsets"],
        precedences=scenario["precedences"],
    )
    return build_case_ranking_replay_bundle(
        evidence=evidence,
        family_aggregation=family,
        propagation_scoring=propagation,
        rule_set=rule_set,
        rule_assignments=(),
        rule_predicates=(),
        rule_scoring=rules,
        history_mode="off",
        history_evaluation=history,
        ranking_authority=authority,
        scoring_config=config,
        ranking_result=ranking,
        top_k=ranking.top_k,
        semantic_calls=(),
        attempt_ledgers=(),
        costs=(),
        checkpoints=(),
        degradation_reason_codes=(),
    )


@pytest.fixture(scope="module")
def replay_bundle() -> object:
    return build_real_replay_bundle()


@pytest.fixture(scope="module")
def alternate_replay_bundle() -> object:
    return build_real_replay_bundle(source_strength=0.4)


def _resign_replay_bundle(bundle: object, **updates: object) -> object:
    from nurse_rcl.v2.case_audit import CaseRankingReplayBundle

    unsigned = bundle.model_dump(mode="json", exclude={"bundle_sha256"})
    for field_name, value in updates.items():
        unsigned[field_name] = (
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        )
    return CaseRankingReplayBundle.model_validate(
        {**unsigned, "bundle_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def test_replay_bundle_uses_exact_existing_v2_models(replay_bundle: object) -> None:
    bundle = replay_bundle

    assert type(bundle.evidence.event_strength) is EventStrengthResult
    assert type(bundle.evidence.frozen_candidates) is FrozenCandidateUniverse
    assert type(bundle.family_aggregation) is FamilyAggregationResult
    assert type(bundle.propagation_scoring) is PropagationScoringResult
    assert type(bundle.rule_scoring) is RuleScoringResult
    assert type(bundle.history_evaluation) is HistoryEvaluationResult
    assert type(bundle.ranking_authority) is RankingInputAuthority
    assert type(bundle.scoring_config) is ReferenceScoringConfig
    assert type(bundle.ranking_result) is ReferenceRankingResult
    assert type(bundle.top_k) is TopKArtifact
    assert bundle.verify_inputs()


def test_replay_bundle_closes_nonempty_runtime_records_to_the_same_case(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import build_case_ranking_replay_bundle, seal_case_audit
    from nurse_rcl.v2.runtime_schemas import (
        AttemptRecord,
        build_attempt_ledger,
        build_case_checkpoint,
        build_cost_record,
        build_semantic_call,
    )

    opaque_case_id = replay_bundle.evidence.time_contract.opaque_case_id
    attempt = AttemptRecord(
        attempt_number=1,
        started_at_utc="2026-09-10T09:59:58Z",
        completed_at_utc="2026-09-10T09:59:59Z",
        outcome="structurally_valid",
        retryable=False,
        response_sha256="a" * 64,
        error_code=None,
        input_tokens=100,
        output_tokens=20,
        latency_ms=1000,
    )
    ledger = build_attempt_ledger(
        logical_request_id="lrq_" + "b" * 24,
        attempts=(attempt,),
    )
    cost = build_cost_record(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=0,
        actual_cost="0.001234",
        currency="USD",
        source="provider_receipt",
    )
    call = build_semantic_call(
        opaque_case_id=opaque_case_id,
        stage="hypothesis",
        endpoint="https://ark.cn-beijing.volces.com/api/v3",
        model="deepseek-v4-flash-ga-260731",
        input_token_budget=12_000,
        output_token_budget=3_000,
        payload_sha256="2" * 64,
        prompt_sha256="3" * 64,
        output_schema_sha256="4" * 64,
        attempt_ledger=ledger,
        cost=cost,
        status="success",
    )
    checkpoint = build_case_checkpoint(
        opaque_case_id=opaque_case_id,
        experiment_version="B2/v2-hybrid",
        state="SEMANTIC_PENDING",
        sequence=6,
        upstream_hashes={"candidate_universe": "5" * 64},
        produced_artifact_hashes={"semantic_call": call.call_sha256},
        logical_request_ids=(ledger.logical_request_id,),
        created_at_utc="2026-09-10T10:00:00Z",
    )
    values = {
        field_name: getattr(replay_bundle, field_name)
        for field_name in type(replay_bundle).model_fields
        if field_name not in {"schema_version", "bundle_sha256"}
    }
    values.update(
        semantic_calls=(call,),
        attempt_ledgers=(ledger,),
        costs=(cost,),
        checkpoints=(checkpoint,),
    )
    rebuilt = build_case_ranking_replay_bundle(**values)

    assert rebuilt.verify_inputs()
    assert rebuilt.semantic_calls == (call,)
    from nurse_rcl.v2.case_audit import build_case_audit_lineage_index

    lineage = build_case_audit_lineage_index(rebuilt)
    node_by_id = {item.record_id: item for item in lineage.nodes}
    assert ledger.logical_request_id in node_by_id
    assert f"cost:{cost.cost_sha256}" in node_by_id
    assert call.semantic_call_id in node_by_id
    assert f"checkpoint:{checkpoint.checkpoint_sha256}" in node_by_id
    assert ledger.logical_request_id in node_by_id[call.semantic_call_id].source_record_ids
    assert f"cost:{cost.cost_sha256}" in (node_by_id[call.semantic_call_id].source_record_ids)
    with pytest.raises(ValueError, match="checkpoint experiment version"):
        seal_case_audit(
            replay_bundle=rebuilt,
            experiment_version="B0/v2-small",
            run_mode="full",
            sealed_at_utc="2026-09-10T10:00:00Z",
        )
    package = seal_case_audit(
        replay_bundle=rebuilt,
        experiment_version="B2/v2-hybrid",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    sealed_document = package.sealed_audit_package.model_dump(mode="json")
    sealed_document["experiment_version"] = "B0/v2-small"
    sealed_unsigned = {
        key: value for key, value in sealed_document.items() if key != "package_sha256"
    }
    sealed_document["package_sha256"] = sha256_hex(canonical_json_bytes(sealed_unsigned))
    tampered = _resign_audit_package(
        package,
        sealed_audit_package_json=canonical_json_bytes(sealed_document).decode("utf-8"),
    )
    assert not tampered.verify()
    wrong_case_call = build_semantic_call(
        opaque_case_id="opq_" + "d" * 32,
        stage="hypothesis",
        endpoint=call.endpoint,
        model=call.model,
        input_token_budget=call.input_token_budget,
        output_token_budget=call.output_token_budget,
        payload_sha256=call.payload_sha256,
        prompt_sha256=call.prompt_sha256,
        output_schema_sha256=call.output_schema_sha256,
        attempt_ledger=ledger,
        cost=cost,
        status="success",
    )
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(
            **{
                **values,
                "semantic_calls": (wrong_case_call,),
            }
        )
    wrong_case_checkpoint = build_case_checkpoint(
        opaque_case_id="opq_" + "d" * 32,
        experiment_version="B2/v2-hybrid",
        state="SEMANTIC_PENDING",
        sequence=6,
        upstream_hashes={"candidate_universe": "5" * 64},
        produced_artifact_hashes={"semantic_call": call.call_sha256},
        logical_request_ids=(ledger.logical_request_id,),
        created_at_utc="2026-09-10T10:00:00Z",
    )
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(**{**values, "checkpoints": (wrong_case_checkpoint,)})
    unknown_request_checkpoint = build_case_checkpoint(
        opaque_case_id=opaque_case_id,
        experiment_version="B2/v2-hybrid",
        state="SEMANTIC_PENDING",
        sequence=6,
        upstream_hashes={"candidate_universe": "5" * 64},
        produced_artifact_hashes={"semantic_call": call.call_sha256},
        logical_request_ids=("lrq_" + "e" * 24,),
        created_at_utc="2026-09-10T10:00:00Z",
    )
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(**{**values, "checkpoints": (unknown_request_checkpoint,)})
    unattached_ledger = build_attempt_ledger(
        logical_request_id="lrq_" + "c" * 24,
        attempts=(),
    )
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(
            **{**values, "attempt_ledgers": (ledger, unattached_ledger)}
        )
    unattached_cost = build_cost_record(
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        actual_cost="0.000000",
        currency="USD",
        source="estimated",
    )
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(**{**values, "costs": (cost, unattached_cost)})
    with pytest.raises(ValueError):
        build_case_ranking_replay_bundle(**{**values, "semantic_calls": (call, call)})


def test_replay_bundle_seals_the_authoritative_verification_query_session(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        build_case_evidence_replay_bundle,
        build_case_ranking_replay_bundle,
    )
    from nurse_rcl.v2.diagnostic_schemas import build_verification_request
    from nurse_rcl.v2.query_session import (
        CaseQuerySessionCoordinator,
        InMemoryCaseQuerySessionHeadStore,
    )
    from nurse_rcl.v2.verification_executor import (
        build_verification_execution_policy,
        build_verification_execution_scope,
    )

    old_evidence = replay_bundle.evidence
    candidate = next(
        item for item in old_evidence.candidate_universe.candidates if item.observation_ids
    )
    observation_by_id = {
        item.observation_id: item for item in old_evidence.event_grouping.observations
    }
    observation = observation_by_id[candidate.observation_ids[0]]
    policy = build_verification_execution_policy()
    scope = build_verification_execution_scope(
        policy=policy,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        operator_permissions=policy.operator_permissions,
    )
    request = build_verification_request(
        opaque_case_id=old_evidence.time_contract.opaque_case_id,
        candidate_freeze_sha256=old_evidence.frozen_candidates.freeze_sha256,
        verification_scope_sha256=scope.scope_sha256,
        candidate_service_key=candidate.service_entity_key,
        operator="compare_window_summary",
        signal_family=observation.signal_family,
        query_interval=observation.interval,
        decision_time_unix_ns=old_evidence.entity_registry.decision_time_unix_ns,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        requested_fields=("anomaly_strength",),
    )
    coordinator = CaseQuerySessionCoordinator(head_store=InMemoryCaseQuerySessionHeadStore())
    session = coordinator.execute_verification(
        requests=(request,),
        policy=policy,
        scope=scope,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
    )
    evidence_values = {
        field_name: getattr(old_evidence, field_name)
        for field_name in type(old_evidence).model_fields
        if field_name not in {"schema_version", "evidence_sha256"}
    }
    evidence_values.update(
        semantic_context=None,
        card_expansion_policy=None,
        query_sessions=(session,),
    )
    evidence = build_case_evidence_replay_bundle(**evidence_values)
    ranking_values = {
        field_name: getattr(replay_bundle, field_name)
        for field_name in type(replay_bundle).model_fields
        if field_name not in {"schema_version", "bundle_sha256", "evidence"}
    }
    rebuilt = build_case_ranking_replay_bundle(
        evidence=evidence,
        **ranking_values,
    )

    assert rebuilt.verify_inputs()
    assert rebuilt.evidence.query_sessions == (session,)
    assert session.verification_result is not None
    assert session.verification_result.requests == (request,)
    from nurse_rcl.v2.case_audit import build_case_audit_lineage_index, seal_case_audit

    lineage = build_case_audit_lineage_index(rebuilt)
    node_by_id = {item.record_id: item for item in lineage.nodes}
    result = session.verification_result.results[0]
    assert request.verification_request_id in node_by_id
    assert result.verification_result_id in node_by_id
    assert session.query_session_id in node_by_id
    assert request.verification_request_id in (
        node_by_id[result.verification_result_id].source_record_ids
    )
    package = seal_case_audit(
        replay_bundle=rebuilt,
        experiment_version="B2/v2-hybrid",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    assert package.sealed_audit_package.artifact_hashes["queries"] == sha256_hex(
        canonical_json_bytes((session.model_dump(mode="json"),))
    )
    assert result.verification_result_id in (package.sealed_audit_package.accepted_provenance_ids)
    assert session.verification_result.decisions[0].execution_decision_id in (
        package.sealed_audit_package.accepted_provenance_ids
    )


def test_evidence_bundle_replays_card_expansion_from_frozen_context(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        build_case_audit_lineage_index,
        build_case_evidence_replay_bundle,
        build_case_ranking_replay_bundle,
    )
    from nurse_rcl.v2.context_expansion import (
        CARD_EXPANSION_QUERY_KIND,
        build_card_expansion_policy,
        build_card_expansion_request,
    )
    from nurse_rcl.v2.diagnostic_schemas import build_verification_request
    from nurse_rcl.v2.focus_selection import build_neutral_focus_selection
    from nurse_rcl.v2.query_session import (
        CaseQuerySessionCoordinator,
        InMemoryCaseQuerySessionHeadStore,
    )
    from nurse_rcl.v2.semantic_context import build_semantic_context_bundle
    from nurse_rcl.v2.verification_executor import (
        build_verification_execution_policy,
        build_verification_execution_scope,
    )

    old_evidence = replay_bundle.evidence
    candidate_count = len(old_evidence.frozen_candidates.candidate_service_keys)
    selection = build_neutral_focus_selection(
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        event_grouping=old_evidence.event_grouping,
        q_base=1,
        q_max=candidate_count,
    )
    context = build_semantic_context_bundle(
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        focus_selection=selection,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
    )
    selected = set(selection.focus_set.selected_service_keys)
    requested_service = next(
        item
        for item in old_evidence.frozen_candidates.candidate_service_keys
        if item not in selected
    )
    policy = build_card_expansion_policy()
    request = build_card_expansion_request(
        opaque_case_id=old_evidence.time_contract.opaque_case_id,
        candidate_freeze_sha256=old_evidence.frozen_candidates.freeze_sha256,
        semantic_context_bundle_sha256=context.bundle_sha256,
        service_entity_key=requested_service,
        query_kind=CARD_EXPANSION_QUERY_KIND,
        decision_time_unix_ns=old_evidence.entity_registry.decision_time_unix_ns,
    )
    coordinator = CaseQuerySessionCoordinator(head_store=InMemoryCaseQuerySessionHeadStore())
    session = coordinator.execute_card_expansion(
        requests=(request,),
        policy=policy,
        semantic_context=context,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
    )
    candidate = next(
        item for item in old_evidence.candidate_universe.candidates if item.observation_ids
    )
    observation_by_id = {
        item.observation_id: item for item in old_evidence.event_grouping.observations
    }
    observation = observation_by_id[candidate.observation_ids[0]]
    verification_policy = build_verification_execution_policy()
    verification_scope = build_verification_execution_scope(
        policy=verification_policy,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        operator_permissions=verification_policy.operator_permissions,
    )
    verification_request = build_verification_request(
        opaque_case_id=old_evidence.time_contract.opaque_case_id,
        candidate_freeze_sha256=old_evidence.frozen_candidates.freeze_sha256,
        verification_scope_sha256=verification_scope.scope_sha256,
        candidate_service_key=candidate.service_entity_key,
        operator="compare_window_summary",
        signal_family=observation.signal_family,
        query_interval=observation.interval,
        decision_time_unix_ns=old_evidence.entity_registry.decision_time_unix_ns,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        requested_fields=("anomaly_strength",),
    )
    verification_session = coordinator.execute_verification(
        requests=(verification_request,),
        policy=verification_policy,
        scope=verification_scope,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        semantic_context=context,
        card_expansion_policy=policy,
    )
    values = {
        field_name: getattr(old_evidence, field_name)
        for field_name in type(old_evidence).model_fields
        if field_name not in {"schema_version", "evidence_sha256"}
    }
    values.update(
        focus_set=selection.focus_set,
        semantic_context=context,
        card_expansion_policy=policy,
        query_sessions=(session, verification_session),
    )
    rebuilt = build_case_evidence_replay_bundle(**values)

    assert rebuilt.verify_inputs()
    assert rebuilt.query_sessions == (session, verification_session)
    assert session.card_expansion_result is not None
    assert session.card_expansion_result.decisions[0].request == request
    assert verification_session.verification_result is not None
    assert verification_session.verification_result.requests == (verification_request,)
    ranking_values = {
        field_name: getattr(replay_bundle, field_name)
        for field_name in type(replay_bundle).model_fields
        if field_name not in {"schema_version", "bundle_sha256", "evidence"}
    }
    ranking_bundle = build_case_ranking_replay_bundle(
        evidence=rebuilt,
        **ranking_values,
    )
    lineage = build_case_audit_lineage_index(ranking_bundle)
    node_ids = {item.record_id for item in lineage.nodes}
    assert request.expansion_request_id in node_ids
    assert session.card_expansion_result.decisions[0].expansion_decision_id in node_ids
    assert verification_request.verification_request_id in node_ids
    assert verification_session.query_session_id in node_ids


def test_query_session_bound_guards_replay_through_ranking(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        build_case_evidence_replay_bundle,
        build_case_ranking_replay_bundle,
    )
    from nurse_rcl.v2.diagnostic_schemas import build_verification_request
    from nurse_rcl.v2.hypothesis_guard import execute_hypothesis_guard
    from nurse_rcl.v2.query_session import (
        CaseQuerySessionCoordinator,
        InMemoryCaseQuerySessionHeadStore,
    )
    from nurse_rcl.v2.role_compiler import (
        build_default_role_binding_proposals,
        compile_event_roles,
    )
    from nurse_rcl.v2.verification_executor import (
        build_verification_execution_policy,
        build_verification_execution_scope,
    )

    old_evidence = replay_bundle.evidence
    candidate = next(
        item for item in old_evidence.candidate_universe.candidates if item.observation_ids
    )
    observation_by_id = {
        item.observation_id: item for item in old_evidence.event_grouping.observations
    }
    observation = observation_by_id[candidate.observation_ids[0]]
    policy = build_verification_execution_policy()
    scope = build_verification_execution_scope(
        policy=policy,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        operator_permissions=policy.operator_permissions,
    )
    request = build_verification_request(
        opaque_case_id=old_evidence.time_contract.opaque_case_id,
        candidate_freeze_sha256=old_evidence.frozen_candidates.freeze_sha256,
        verification_scope_sha256=scope.scope_sha256,
        candidate_service_key=candidate.service_entity_key,
        operator="compare_window_summary",
        signal_family=observation.signal_family,
        query_interval=observation.interval,
        decision_time_unix_ns=old_evidence.entity_registry.decision_time_unix_ns,
        observation_ids=(observation.observation_id,),
        relation_ids=(),
        requested_fields=("anomaly_strength",),
    )
    coordinator = CaseQuerySessionCoordinator(head_store=InMemoryCaseQuerySessionHeadStore())
    session = coordinator.execute_verification(
        requests=(request,),
        policy=policy,
        scope=scope,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
    )
    predicate_by_id = {item.predicate_result_id: item for item in old_evidence.role_predicates}
    guards = []
    for old_guard in old_evidence.role_guard_results:
        hypothesis = old_guard.parsed_hypothesis
        assert hypothesis is not None
        predicate_ids = tuple(
            sorted(
                {
                    *hypothesis.necessary_predicate_result_ids,
                    *hypothesis.optional_predicate_result_ids,
                    *hypothesis.refuting_predicate_result_ids,
                }
            )
        )
        guards.append(
            execute_hypothesis_guard(
                submitted_record=json.loads(old_guard.submitted_record_json),
                predicates=tuple(predicate_by_id[item] for item in predicate_ids),
                candidate_universe=old_evidence.candidate_universe,
                frozen_candidates=old_evidence.frozen_candidates,
                entity_registry=old_evidence.entity_registry,
                event_grouping=old_evidence.event_grouping,
                mechanism_library=old_evidence.mechanism_library,
                scope_interval=old_evidence.scope_interval,
                query_session=session,
                query_session_coordinator=coordinator,
            )
        )
    guard_results = tuple(sorted(guards, key=lambda item: item.guard_result_id))
    query_mapping = {item.guard_result_id: session for item in guard_results}
    proposals = build_default_role_binding_proposals(
        guard_results=guard_results,
        event_grouping=old_evidence.event_grouping,
    )
    role_compilation = compile_event_roles(
        proposals=proposals,
        guard_results=guard_results,
        predicates=old_evidence.role_predicates,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        mechanism_library=old_evidence.mechanism_library,
        scope_interval=old_evidence.scope_interval,
        query_sessions_by_guard_result_id=query_mapping,
        query_session_coordinator=coordinator,
    )
    role_inputs = {
        "proposals": proposals,
        "guard_results": guard_results,
        "predicates": old_evidence.role_predicates,
        "candidate_universe": old_evidence.candidate_universe,
        "frozen_candidates": old_evidence.frozen_candidates,
        "entity_registry": old_evidence.entity_registry,
        "event_grouping": old_evidence.event_grouping,
        "mechanism_library": old_evidence.mechanism_library,
        "scope_interval": old_evidence.scope_interval,
        "query_sessions_by_guard_result_id": query_mapping,
        "query_session_coordinator": coordinator,
    }
    event_strength = compile_event_strengths(
        role_compilation=role_compilation,
        **role_inputs,
    )
    family = compile_family_aggregation(
        event_strength=event_strength,
        aggregator_version=REFERENCE_KEY,
        role_compilation=role_compilation,
        **role_inputs,
    )
    propagation = score_propagation_paths(
        frozen_candidates=old_evidence.frozen_candidates,
        condensations=old_evidence.condensations,
        event_grouping=old_evidence.event_grouping,
        onsets=old_evidence.onsets,
        precedences=old_evidence.precedences,
        role_compilation=role_compilation,
        role_proposals=proposals,
        role_guard_results=guard_results,
        role_predicates=old_evidence.role_predicates,
        candidate_universe=old_evidence.candidate_universe,
        entity_registry=old_evidence.entity_registry,
        mechanism_library=old_evidence.mechanism_library,
        scope_interval=old_evidence.scope_interval,
        query_sessions_by_guard_result_id=query_mapping,
        query_session_coordinator=coordinator,
        config=replay_bundle.propagation_scoring.config,
    )
    evidence = build_case_evidence_replay_bundle(
        time_contract=old_evidence.time_contract,
        entity_registry=old_evidence.entity_registry,
        event_grouping=old_evidence.event_grouping,
        candidate_universe=old_evidence.candidate_universe,
        frozen_candidates=old_evidence.frozen_candidates,
        focus_set=old_evidence.focus_set,
        semantic_context=None,
        card_expansion_policy=None,
        query_sessions=(session,),
        mechanism_library=old_evidence.mechanism_library,
        scope_interval=old_evidence.scope_interval,
        role_predicates=old_evidence.role_predicates,
        role_guard_results=guard_results,
        role_proposals=proposals,
        role_compilation=role_compilation,
        event_strength=event_strength,
        graph_snapshots=old_evidence.graph_snapshots,
        condensations=old_evidence.condensations,
        onsets=old_evidence.onsets,
        precedences=old_evidence.precedences,
    )
    authority = RankingInputAuthority(
        candidate_freeze_sha256=old_evidence.frozen_candidates.freeze_sha256,
        family_aggregation_result_sha256=family.result_sha256,
        propagation_scoring_result_sha256=propagation.result_sha256,
        propagation_authoritative_input_sha256=propagation.authoritative_input_sha256,
        rule_scoring_result_sha256=replay_bundle.rule_scoring.result_sha256,
        rule_authoritative_input_sha256=(replay_bundle.rule_scoring.authoritative_input_sha256),
        history_evaluation_result_sha256=(replay_bundle.history_evaluation.result_sha256),
        history_authoritative_input_sha256=(
            replay_bundle.history_evaluation.authoritative_input_sha256
        ),
    )
    ranking = rank_reference_candidates(
        frozen_candidates=old_evidence.frozen_candidates,
        family_aggregation=family,
        propagation_scoring=propagation,
        rule_scoring=replay_bundle.rule_scoring,
        history_evaluation=replay_bundle.history_evaluation,
        authority=authority,
        config=replay_bundle.scoring_config,
    )
    bundle = build_case_ranking_replay_bundle(
        evidence=evidence,
        family_aggregation=family,
        propagation_scoring=propagation,
        rule_set=replay_bundle.rule_set,
        rule_assignments=replay_bundle.rule_assignments,
        rule_predicates=replay_bundle.rule_predicates,
        rule_scoring=replay_bundle.rule_scoring,
        history_mode=replay_bundle.history_mode,
        history_evaluation=replay_bundle.history_evaluation,
        ranking_authority=authority,
        scoring_config=replay_bundle.scoring_config,
        ranking_result=ranking,
        top_k=ranking.top_k,
        semantic_calls=(),
        attempt_ledgers=(),
        costs=(),
        checkpoints=(),
        degradation_reason_codes=(),
    )

    assert bundle.verify_inputs()
    assert all(
        session.session_sha256 in guard.decisions[0].provenance_ids
        for guard in bundle.evidence.role_guard_results
    )


@pytest.mark.parametrize(
    "channel_name",
    ("family_aggregation", "propagation_scoring", "rule_scoring", "history_evaluation"),
)
def test_replay_rejects_individually_valid_cross_channel_contradiction(
    replay_bundle: object,
    alternate_replay_bundle: object,
    channel_name: str,
) -> None:
    alternate_channel = getattr(alternate_replay_bundle, channel_name)
    assert (
        alternate_channel.verify()
        if hasattr(alternate_channel, "verify")
        else alternate_channel.verify_structure()
    )
    tampered = _resign_replay_bundle(
        replay_bundle,
        **{channel_name: alternate_channel},
    )

    assert not tampered.verify_inputs()


def test_replay_rejects_individually_valid_config_contradiction(
    replay_bundle: object,
) -> None:
    alternate_config = build_reference_scoring_config(history_mode="off", top_k=3)
    tampered = _resign_replay_bundle(
        replay_bundle,
        scoring_config=alternate_config,
    )

    assert not tampered.verify_inputs()


def test_verify_inputs_rebuilds_ranking_from_channels(
    replay_bundle: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("nurse_rcl.v2.case_audit")
    original = module.rank_reference_candidates
    calls: list[dict[str, object]] = []

    def recording_rank(**kwargs: object) -> ReferenceRankingResult:
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(module, "rank_reference_candidates", recording_rank)

    assert replay_bundle.verify_inputs()
    assert len(calls) == 1
    assert calls[0]["family_aggregation"] == replay_bundle.family_aggregation
    assert calls[0]["propagation_scoring"] == replay_bundle.propagation_scoring
    assert calls[0]["rule_scoring"] == replay_bundle.rule_scoring
    assert calls[0]["history_evaluation"] == replay_bundle.history_evaluation


def _resign_lineage_index(index: object, *, nodes: tuple[object, ...]) -> object:
    from nurse_rcl.v2.case_audit import CaseAuditLineageIndex

    unsigned = index.model_dump(mode="json", exclude={"lineage_sha256"})
    unsigned["nodes"] = tuple(item.model_dump(mode="json") for item in nodes)
    return CaseAuditLineageIndex.model_validate(
        {**unsigned, "lineage_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def test_lineage_index_covers_score_records_with_bidirectional_edges(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import build_case_audit_lineage_index

    index = build_case_audit_lineage_index(replay_bundle)
    counts = Counter(item.record_kind for item in index.nodes)
    expected_counts = {
        "candidate_scope": len(replay_bundle.evidence.candidate_universe.candidates),
        "observation": len(replay_bundle.evidence.event_grouping.observations),
        "event_group": len(replay_bundle.evidence.event_grouping.event_groups),
        "predicate_result": len(replay_bundle.evidence.role_predicates),
        "diagnostic_hypothesis": len(
            {
                item.parsed_hypothesis.hypothesis_id
                for item in replay_bundle.evidence.role_guard_results
                if item.parsed_hypothesis is not None
            }
        ),
        "guard_decision": sum(
            len(item.decisions) for item in replay_bundle.evidence.role_guard_results
        ),
        "hypothesis_guard_result": len(replay_bundle.evidence.role_guard_results),
        "score_impact_compilation": sum(
            item.score_impact is not None for item in replay_bundle.evidence.role_guard_results
        ),
        "role_binding_proposal": len(replay_bundle.evidence.role_proposals),
        "role_assertion": len(replay_bundle.evidence.role_compilation.assertions),
        "graph_snapshot": len(replay_bundle.evidence.graph_snapshots),
        "scc_condensation": len(replay_bundle.evidence.condensations),
        "onset_interval": len(replay_bundle.evidence.onsets),
        "temporal_precedence": len(replay_bundle.evidence.precedences),
        "event_group_base_strength": len(replay_bundle.evidence.event_strength.group_strengths),
        "candidate_event_role_adjustment": len(
            replay_bundle.evidence.event_strength.candidate_event_adjustments
        ),
        "signal_family_maximum": sum(
            len(item.family_maxima) for item in replay_bundle.family_aggregation.candidates
        ),
        "candidate_family_aggregation": len(replay_bundle.family_aggregation.candidates),
        "propagation_relation_binding": len(replay_bundle.propagation_scoring.relation_bindings),
        "propagation_hop_audit": sum(
            len(item.hops) for item in replay_bundle.propagation_scoring.path_audits
        ),
        "propagation_path_audit": len(replay_bundle.propagation_scoring.path_audits),
        "propagation_target_score": len(replay_bundle.propagation_scoring.target_scores),
        "candidate_propagation_score": len(replay_bundle.propagation_scoring.candidate_scores),
        "scoring_rule": len(replay_bundle.rule_set.rules),
        "rule_candidate_assignment": len(replay_bundle.rule_assignments),
        "rule_contribution": sum(
            len(item.contributions) for item in replay_bundle.rule_scoring.candidates
        ),
        "candidate_rule_score": len(replay_bundle.rule_scoring.candidates),
        "history_neighbor_audit": len(replay_bundle.history_evaluation.neighbor_audits),
        "history_score_contribution": sum(
            len(item.contributions) for item in replay_bundle.history_evaluation.candidate_scores
        ),
        "candidate_history_score": len(replay_bundle.history_evaluation.candidate_scores),
        "score_decomposition": len(replay_bundle.ranking_result.scores),
        "evidence_tie_group": len(replay_bundle.ranking_result.evidence_tie_groups),
        "top_k_entry": len(replay_bundle.top_k.entries),
    }
    assert counts == Counter(expected_counts)
    assert index.verify()
    assert index.verify_against(replay_bundle)

    by_id = {item.record_id: item for item in index.nodes}
    assert len(by_id) == len(index.nodes)
    for node in index.nodes:
        assert node.score_affecting
        assert len(node.record_sha256) == 64
        assert node.source_record_ids == tuple(sorted(set(node.source_record_ids)))
        assert node.consumer_record_ids == tuple(sorted(set(node.consumer_record_ids)))
        for source_id in node.source_record_ids:
            assert source_id in by_id
            assert node.record_id in by_id[source_id].consumer_record_ids
        for consumer_id in node.consumer_record_ids:
            assert consumer_id in by_id
            assert node.record_id in by_id[consumer_id].source_record_ids

    observation = replay_bundle.evidence.event_grouping.observations[0]
    observation_node = by_id[observation.observation_id]
    assert observation_node.record_sha256 == sha256_hex(
        canonical_json_bytes(observation.model_dump(mode="json"))
    )
    for score in replay_bundle.ranking_result.scores:
        score_node = by_id[score.score_sha256]
        assert len(score_node.source_record_ids) == 4
        assert score_node.consumer_record_ids


def test_lineage_rejects_resigned_missing_reverse_edge(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        CaseAuditLineageNode,
        build_case_audit_lineage_index,
    )

    index = build_case_audit_lineage_index(replay_bundle)
    target = next(item for item in index.nodes if item.consumer_record_ids)
    replacement = CaseAuditLineageNode.model_validate(
        {
            **target.model_dump(mode="json"),
            "consumer_record_ids": target.consumer_record_ids[1:],
        }
    )
    nodes = tuple(
        replacement if item.record_id == target.record_id else item for item in index.nodes
    )
    tampered = _resign_lineage_index(index, nodes=nodes)

    assert not tampered.verify()
    assert not tampered.verify_against(replay_bundle)


def test_lineage_rejects_resigned_unknown_source(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        CaseAuditLineageNode,
        build_case_audit_lineage_index,
    )

    index = build_case_audit_lineage_index(replay_bundle)
    target = next(item for item in index.nodes if item.source_record_ids)
    replacement = CaseAuditLineageNode.model_validate(
        {
            **target.model_dump(mode="json"),
            "source_record_ids": tuple(sorted(("missing_record", *target.source_record_ids[1:]))),
        }
    )
    nodes = tuple(
        replacement if item.record_id == target.record_id else item for item in index.nodes
    )
    tampered = _resign_lineage_index(index, nodes=nodes)

    assert not tampered.verify()
    assert not tampered.verify_against(replay_bundle)


def test_lineage_rejects_resigned_disposition_forgery(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import (
        CaseAuditLineageNode,
        build_case_audit_lineage_index,
    )

    index = build_case_audit_lineage_index(replay_bundle)
    target = next(item for item in index.nodes if item.record_kind == "guard_decision")
    replacement = CaseAuditLineageNode.model_validate(
        {**target.model_dump(mode="json"), "disposition": "forged"}
    )
    nodes = tuple(
        replacement if item.record_id == target.record_id else item for item in index.nodes
    )
    tampered = _resign_lineage_index(index, nodes=nodes)

    assert tampered.verify()
    assert not tampered.verify_against(replay_bundle)


def _resign_audit_package(package: object, **updates: object) -> object:
    from nurse_rcl.v2.case_audit import ImmutableCaseAuditPackage

    unsigned = package.model_dump(mode="json", exclude={"audit_sha256"})
    unsigned.update(updates)
    return ImmutableCaseAuditPackage.model_validate(
        {**unsigned, "audit_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def test_success_audit_is_canonical_at_rest_and_replays_real_channels() -> None:
    from nurse_rcl.v2.case_audit import replay_case_audit, seal_case_audit

    source = build_real_replay_bundle()
    package = seal_case_audit(
        replay_bundle=source,
        experiment_version="B2/v2-hybrid",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    before = package.replay_bundle_json
    object.__setattr__(source, "degradation_reason_codes", ("caller_tampered",))

    assert package.replay_bundle_json == before
    assert package.replay_bundle.verify_inputs()
    assert package.lineage_index.verify_against(package.replay_bundle)
    assert package.sealed_audit_package.verify()
    receipt = replay_case_audit(
        package,
        expected_audit_sha256=package.audit_sha256,
    )
    assert receipt.terminal_state == "SEALED"
    assert receipt.ranking_result == package.replay_bundle.ranking_result
    assert receipt.top_k == package.replay_bundle.top_k
    assert receipt.failure_diagnostic is None

    mutable_view = package.sealed_audit_package
    mutable_view.artifact_hashes["scores"] = "0" * 64
    assert package.sealed_audit_package.artifact_hashes["scores"] != "0" * 64


def test_success_audit_partitions_all_status_bearing_provenance(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import seal_case_audit

    package = seal_case_audit(
        replay_bundle=replay_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    sealed = package.sealed_audit_package
    partitions = {
        "accepted": set(sealed.accepted_provenance_ids),
        "rejected": set(sealed.rejected_provenance_ids),
        "conflicted": set(sealed.conflicted_provenance_ids),
        "unknown": set(sealed.unknown_provenance_ids),
    }
    status_records = {
        hypothesis.hypothesis_id: (
            "unknown" if hypothesis.lifecycle_status == "proposed" else hypothesis.lifecycle_status
        )
        for guard in replay_bundle.evidence.role_guard_results
        if (hypothesis := guard.parsed_hypothesis) is not None
    }
    status_records.update(
        (proposal.proposal_id, "unknown") for proposal in replay_bundle.evidence.role_proposals
    )
    status_records.update(
        (
            assertion.assertion_id,
            "unknown" if assertion.lifecycle_status == "proposed" else assertion.lifecycle_status,
        )
        for assertion in replay_bundle.evidence.role_compilation.assertions
    )
    status_records.update(
        (guard.guard_result_id, guard.final_disposition)
        for guard in replay_bundle.evidence.role_guard_results
    )
    status_records.update(
        (decision.guard_decision_id, decision.disposition)
        for guard in replay_bundle.evidence.role_guard_results
        for decision in guard.decisions
    )

    assert status_records
    for record_id, expected_partition in status_records.items():
        memberships = {name for name, record_ids in partitions.items() if record_id in record_ids}
        assert memberships == {expected_partition}


def test_replay_requires_the_external_audit_anchor(replay_bundle: object) -> None:
    from nurse_rcl.v2.case_audit import replay_case_audit, seal_case_audit

    package = seal_case_audit(
        replay_bundle=replay_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )

    with pytest.raises(ValueError, match="external audit digest"):
        replay_case_audit(package, expected_audit_sha256="0" * 64)


def test_resigned_package_rejects_cross_artifact_contradictions(
    replay_bundle: object,
    alternate_replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import CaseAuditLineageNode, seal_case_audit

    package = seal_case_audit(
        replay_bundle=replay_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )

    alternate_json = canonical_json_bytes(alternate_replay_bundle.model_dump(mode="json")).decode(
        "utf-8"
    )
    bundle_tampered = _resign_audit_package(
        package,
        replay_bundle_json=alternate_json,
    )
    assert not bundle_tampered.verify()

    lineage = package.lineage_index
    target = next(item for item in lineage.nodes if item.record_kind == "guard_decision")
    replacement = CaseAuditLineageNode.model_validate(
        {**target.model_dump(mode="json"), "disposition": "forged"}
    )
    lineage_nodes = tuple(
        replacement if item.record_id == target.record_id else item for item in lineage.nodes
    )
    forged_lineage = _resign_lineage_index(lineage, nodes=lineage_nodes)
    lineage_tampered = _resign_audit_package(
        package,
        lineage_index_json=canonical_json_bytes(forged_lineage.model_dump(mode="json")).decode(
            "utf-8"
        ),
    )
    assert not lineage_tampered.verify()

    sealed_document = package.sealed_audit_package.model_dump(mode="json")
    sealed_document["artifact_hashes"]["scores"] = "0" * 64
    sealed_unsigned = {
        key: value for key, value in sealed_document.items() if key != "package_sha256"
    }
    sealed_document["package_sha256"] = sha256_hex(canonical_json_bytes(sealed_unsigned))
    sealed_tampered = _resign_audit_package(
        package,
        sealed_audit_package_json=canonical_json_bytes(sealed_document).decode("utf-8"),
    )
    assert not sealed_tampered.verify()


@pytest.mark.parametrize(
    "forbidden_field",
    ("answer", "label", "injection", "original_case_id", "source_path"),
)
def test_audit_rejects_resigned_forbidden_inference_fields(
    replay_bundle: object,
    forbidden_field: str,
) -> None:
    from nurse_rcl.v2.case_audit import seal_case_audit

    package = seal_case_audit(
        replay_bundle=replay_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    bundle_document = json.loads(package.replay_bundle_json)
    bundle_document[forbidden_field] = "must-not-enter-inference"
    tampered = _resign_audit_package(
        package,
        replay_bundle_json=canonical_json_bytes(bundle_document).decode("utf-8"),
    )

    assert not tampered.verify()


def test_audit_rejects_noncanonical_nan_overdeep_and_cyclic_inputs(
    replay_bundle: object,
) -> None:
    from nurse_rcl.v2.case_audit import ImmutableCaseAuditPackage, seal_case_audit

    package = seal_case_audit(
        replay_bundle=replay_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T10:00:00Z",
    )
    with pytest.raises(ValueError):
        ImmutableCaseAuditPackage.model_validate(
            {
                **package.model_dump(mode="json"),
                "replay_bundle_json": '{"value":NaN}',
            }
        )
    overdeep = "[" * 80 + "0" + "]" * 80
    with pytest.raises(ValueError):
        ImmutableCaseAuditPackage.model_validate(
            {
                **package.model_dump(mode="json"),
                "replay_bundle_json": overdeep,
            }
        )
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises((TypeError, ValueError)):
        seal_case_audit(
            replay_bundle=cyclic,  # type: ignore[arg-type]
            experiment_version="B0/v2-small",
            run_mode="full",
            sealed_at_utc="2026-09-10T10:00:00Z",
        )


def test_failure_audit_requires_and_preserves_real_empty_candidate_diagnostic() -> None:
    from nurse_rcl.v2.case_audit import replay_case_audit, seal_case_audit
    from tests.test_v2_candidate_invariance import _build_empty_failure

    failure, _, _ = _build_empty_failure()
    package = seal_case_audit(failure_diagnostic=failure)
    receipt = replay_case_audit(
        package,
        expected_audit_sha256=package.audit_sha256,
    )

    assert package.terminal_state == "FAILURE_SEALED"
    assert package.replay_bundle_or_none is None
    assert package.lineage_index_or_none is None
    assert receipt.terminal_state == "FAILURE_SEALED"
    assert receipt.ranking_result is None
    assert receipt.top_k is None
    assert receipt.prediction_service_keys == ()
    assert receipt.failure_diagnostic == failure
    with pytest.raises((TypeError, ValueError)):
        seal_case_audit(failure_diagnostic="empty_candidate_universe")  # type: ignore[arg-type]


def test_case_audit_exact_schemas_are_registered() -> None:
    from nurse_rcl.v2.case_audit import (
        CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
        CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION,
        CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION,
        IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
    )

    registry = importlib.import_module("nurse_rcl.v2.schema_compatibility")
    assert {
        "case_evidence_replay_bundle": CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION,
        "case_ranking_replay_bundle": CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION,
        "case_audit_lineage_index": CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
        "immutable_case_audit_package": IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
    }.items() <= registry.CURRENT_SCHEMA_VERSIONS.items()
