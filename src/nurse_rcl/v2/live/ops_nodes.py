"""Expose actual normal-B0 computations as bounded, transparent diagnostic cards."""

from __future__ import annotations

from nurse_rcl.v2.dynamic_verification.contracts import TruthValue
from nurse_rcl.v2.live.normal_artifacts import NormalEvidence
from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence, snapshot_id
from nurse_rcl.v2.live.ops_node_common import node_card, prepared_snapshot_id
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData


def initial_nodes(data: RepairPreparationData) -> tuple[OpsNodeEvidence, ...]:
    from nurse_rcl.v2.live.ops_relations import dependency_nodes

    contract = data.diagnostics.legacy.sources.adapter.time_contract
    snapshot = prepared_snapshot_id(data)
    modalities = sorted({s.observation.modality for s in data.sources})
    boundary = node_card(
        snapshot=snapshot, node="boundary", parameters={},
        payload=dict(
            candidate_service_keys=data.catalog.candidate_service_keys,
            baseline_window=contract.baseline_window.model_dump(mode="json"),
            fault_window=contract.fault_window.model_dump(mode="json"),
            scalar_modalities=modalities,
            missing_modalities=contract.missing_modalities,
            scalar_observation_count=len(data.sources),
            error_observation_count=len(data.diagnostics.errors),
            query_request_keys=tuple(r.request_key for r in data.catalog.requests),
            excluded_or_missing_modalities_are_not_normal=True,
        ),
        evidence_ids=(data.catalog.catalog_sha256,),
        source_refs=(data.source_descriptor_sha256, data.source_files_sha256),
        interval=contract.fault_window,
        limitations=("capability_inventory_not_health_or_causal_confirmation",),
    )
    return (boundary, *dependency_nodes(data))


def _symptoms(data: RepairPreparationData, evidence: NormalEvidence) -> tuple[OpsNodeEvidence, ...]:
    snapshot = prepared_snapshot_id(data)
    mechanisms = evidence.ranking.propagation.local_scoring.strengths.mechanisms
    sources = {s.source_sha256: s for s in data.sources}
    cards = []
    for fact in mechanisms.signal_facts:
        source = sources[fact.source_sha256]
        quality, obs = source.quality, source.observation
        general: TruthValue = next((p.truth_value for p in fact.facts
                        if p.predicate_kind == "local_anomaly_observed"), "unknown")
        present = quality.source_summary.fault_summary.valid_point_count > 0
        cards.append(node_card(
            snapshot=snapshot, node="symptoms",
            parameters=dict(observation_id=obs.observation_id,
                            candidate_service_key=source.candidate_service_key),
            payload=dict(
                kind="fixed_summary", observation_id=obs.observation_id,
                candidate_service_key=source.candidate_service_key,
                signal_family=obs.signal_family, metric_name=source.metric_name,
                baseline=quality.source_summary.baseline_summary.model_dump(mode="json"),
                fault=quality.source_summary.fault_summary.model_dump(mode="json"),
                baseline_window=source.time_contract.baseline_window.model_dump(mode="json"),
                fault_window=source.time_contract.fault_window.model_dump(mode="json"),
                signed_standardized_change=quality.signed_standardized_change,
                eligible=quality.formal_scoring_eligible,
                facts=fact.model_dump(mode="json"),
                episode_status="not_measured_by_summary",
            ),
            evidence_ids=(fact.result_sha256, obs.observation_id),
            source_refs=(source.source_sha256, *source.source_file_sha256s),
            interval=source.query_interval, unit=obs.unit, coverage=quality.coverage_quality,
            truth=general if present else "unknown",
            availability="available" if quality.formal_scoring_eligible else (
                "partial" if present else "unavailable"
            ),
            limitations=(*quality.quality_flags, "summary_cannot_prove_episode_or_normality"),
        ))
    for error_fact in mechanisms.error_facts.facts:
        error_truth: TruthValue = next((p.truth_value for p in error_fact.facts
                                       if p.predicate_kind == "local_anomaly_observed"), "unknown")
        cards.append(node_card(
            snapshot=snapshot, node="symptoms",
            parameters=dict(candidate_service_key=error_fact.candidate_service_key,
                            evidence_id=error_fact.result_sha256),
            payload=dict(kind="error_summary", facts=error_fact.model_dump(mode="json")),
            evidence_ids=(error_fact.result_sha256,), source_refs=error_fact.observation_sha256s,
            interval=error_fact.interval, unit="observed_record_fraction", coverage=None,
            truth=error_truth,
            limitations=(*error_fact.limitations, "record_fraction_not_request_population"),
        ))
    for step in evidence.session.steps:
        measured = step.measurement
        source = sources[measured.source_sha256]
        cards.append(node_card(
            snapshot=snapshot, node="symptoms",
            parameters=dict(request_key=step.feedback.request_key),
            payload=dict(
                kind="paid_episode", measurement_sha256=measured.measurement_sha256,
                candidate_service_key=source.candidate_service_key,
                observation_id=source.observation.observation_id,
                episodes=[e.model_dump(mode="json") for e in measured.episodes],
                persistent_up=measured.persistent_up, persistent_down=measured.persistent_down,
                general_deviation_present=measured.general_deviation_present,
                verified_normal=measured.verified_normal,
                predicates=[p.model_dump(mode="json") for p in step.feedback.predicates],
                reason_codes=measured.reason_codes,
            ),
            evidence_ids=(measured.measurement_sha256, step.feedback.result_sha256,
                          *(p.result_sha256 for p in step.feedback.predicates)),
            source_refs=(source.source_sha256,), interval=measured.interval,
            unit=source.observation.unit, coverage=measured.quality.coverage_quality,
            truth=measured.general_deviation_present,
            limitations=("episode_support_not_causal_confirmation", *measured.reason_codes),
        ))
    return tuple(cards)


def _mechanisms(
    data: RepairPreparationData, evidence: NormalEvidence,
) -> tuple[OpsNodeEvidence, ...]:
    snapshot = prepared_snapshot_id(data)
    compiled = evidence.ranking.propagation.local_scoring.strengths.mechanisms
    by_condition = {c.condition_id: c for c in compiled.conditions}
    interval = data.diagnostics.legacy.sources.adapter.time_contract.fault_window
    cards = []
    for mechanism in compiled.mechanisms:
        conditions = tuple(by_condition[key] for key in mechanism.condition_ids)
        distinguishing = [dict(
            condition_id=condition.condition_id, predicate_kind=condition.predicate_kind,
            requirement=condition.requirement,
            request_keys=tuple(r.request_key for r in data.catalog.requests if (
                r.candidate_service_key == mechanism.candidate_service_key
                and r.request_key not in evidence.session.submitted_request_keys
                and any(n.queryable and n.mechanism_key == mechanism.mechanism_key
                        and n.predicate_kind == condition.predicate_kind for n in r.needs)
            )),
        ) for condition in conditions if condition.truth_value == "unknown"]
        cards.append(node_card(
            snapshot=snapshot, node="mechanisms",
            parameters=dict(candidate_service_key=mechanism.candidate_service_key,
                            mechanism_key=mechanism.mechanism_key),
            payload=dict(
                decision=mechanism.model_dump(mode="json"),
                conditions=[c.model_dump(mode="json") for c in conditions],
                distinguishing_checks=distinguishing, score_authority=False,
            ),
            evidence_ids=(mechanism.mechanism_id, *(c.condition_id for c in conditions)),
            source_refs=(compiled.result_sha256,), interval=interval,
            truth="true" if mechanism.disposition == "accepted" else (
                "false" if mechanism.disposition == "rejected" else "unknown"
            ),
            limitations=("observable_contract_not_causal_confirmation",
                         "unmeasured_conditions_remain_unknown"),
        ))
    return tuple(cards)


def terminal_nodes(
    data: RepairPreparationData, evidence: NormalEvidence,
) -> tuple[OpsNodeEvidence, ...]:
    from nurse_rcl.v2.live.ops_relations import timing_nodes

    snapshot = prepared_snapshot_id(data)
    if snapshot != snapshot_id(evidence):
        raise ValueError("node/evidence snapshot differs")
    interval = data.diagnostics.legacy.sources.adapter.time_contract.fault_window
    ranking = evidence.ranking
    ranks = {key: i + 1 for i, key in enumerate(ranking.ordered_service_keys)}
    scores = tuple(node_card(
        snapshot=snapshot, node="scoring",
        parameters=dict(candidate_service_key=score.candidate_service_key),
        payload=dict(score=score.model_dump(mode="json"),
                     rank=ranks[score.candidate_service_key],
                     tie_break_is_evidence=False, is_root_cause_probability=False),
        evidence_ids=(score.score_sha256,), source_refs=(ranking.result_sha256,),
        interval=interval, unit="dimensionless_support_score",
        limitations=("score_not_probability", "hash_tie_break_not_evidence"),
    ) for score in ranking.scores)
    return (*_symptoms(data, evidence), *timing_nodes(data, evidence),
            *_mechanisms(data, evidence), *scores)
