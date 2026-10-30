"""Supplementary projections over admitted values; never a scoring session."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import NoReturn

from nurse_rcl.v2.live.ops_contracts import OpsNodeEvidence
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.repair_fixed_facts import _signal_values
from nurse_rcl.v2.live.repair_measurements import RepairSignalMeasurement
from nurse_rcl.v2.live.repair_mechanisms import _compose
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_queries import RepairQueryFeedback
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource


def measurement_card(
    snapshot: str,
    source: RepairMeasurementSource,
    measurement: RepairSignalMeasurement,
    feedback: RepairQueryFeedback,
) -> OpsNodeEvidence:
    return node_card(
        snapshot=snapshot,
        node="symptoms",
        parameters=dict(request_key=feedback.request_key),
        payload=dict(
            kind="supplementary_episode",
            measurement_sha256=measurement.measurement_sha256,
            candidate_service_key=source.candidate_service_key,
            observation_id=source.observation.observation_id,
            baseline_window=source.time_contract.baseline_window.model_dump(mode="json"),
            fault_window=source.time_contract.fault_window.model_dump(mode="json"),
            episodes=[e.model_dump(mode="json") for e in measurement.episodes],
            persistent_up=measurement.persistent_up,
            persistent_down=measurement.persistent_down,
            verified_normal=measurement.verified_normal,
            feedback=feedback.model_dump(mode="json"),
            score_authority=False,
        ),
        evidence_ids=(
            measurement.measurement_sha256,
            feedback.result_sha256,
            *(p.result_sha256 for p in feedback.predicates),
        ),
        source_refs=(source.source_sha256, *source.source_file_sha256s),
        interval=measurement.interval,
        unit=source.observation.unit,
        coverage=measurement.quality.coverage_quality,
        truth=measurement.general_deviation_present,
        availability="available" if measurement.quality.formal_scoring_eligible else "partial",
        limitations=(
            *measurement.reason_codes,
            "supplementary_not_scoring_input",
            "episode_support_not_causal_confirmation",
        ),
    )


def recheck_mechanism(
    card: OpsNodeEvidence,
    data: RepairPreparationData,
    feedback: tuple[RepairQueryFeedback, ...],
    *,
    reject: Callable[[str, str], NoReturn] | None = None,
) -> OpsNodeEvidence:
    """Recheck one published condition table with already obtained predicates.

    No implicit measurements, permissions, scores or synthetic formal session.
    Fixed leaves are matched by source fact ID, not by a model's chosen value.
    """
    body = json.loads(card.payload_json)
    if card.node != "mechanisms" or "decision" not in body:
        if reject is not None:
            reject("card_kind_mismatch", "registered formal mechanism card required")
        raise ValueError("registered formal mechanism card required")
    formal = body["decision"]
    candidate, mechanism = formal["candidate_service_key"], formal["mechanism_key"]
    facts = {
        _signal_values(s).result_sha256: s.observation.observation_id
        for s in data.sources
        if s.candidate_service_key == candidate
    }
    dynamic = {
        (p.observation_id, p.predicate_kind): p
        for f in feedback
        for p in f.predicates
        if p.candidate_service_key == candidate and p.mechanism_key == mechanism
    }
    used = []
    conditions = body["conditions"]
    for condition in conditions:
        for leaf in condition["leaves"]:
            observation = facts.get(leaf["evidence_id"])
            measured = (
                dynamic.get((observation, condition["predicate_kind"]))
                if observation is not None
                else None
            )
            if measured is not None:
                leaf.update(
                    dynamic_predicate_sha256=measured.result_sha256,
                    measurement_field=measured.measurement_field,
                    truth_value=measured.truth_value,
                    reason_codes=list(measured.reason_codes),
                )
                used.append(measured.result_sha256)
        condition["truth_value"] = _compose(
            tuple(leaf["truth_value"] for leaf in condition["leaves"]),
            condition["operator"],
        )
        # Original formal ID is provenance, not a digest of the updated content.
        condition["formal_condition_id"] = condition.pop("condition_id")
    support = _compose(
        tuple(c["truth_value"] for c in conditions if c["requirement"] == "necessary"), "and"
    )
    unknown = {c["formal_condition_id"] for c in conditions if c["truth_value"] == "unknown"}
    completed = {f.request_key for f in feedback}
    remaining = [
        dict(check, request_keys=[key for key in check["request_keys"] if key not in completed])
        for check in body["distinguishing_checks"]
        if check["condition_id"] in unknown
    ]
    return node_card(
        snapshot=card.case_snapshot_id,
        node="mechanisms",
        parameters=dict(candidate_service_key=candidate, mechanism_key=mechanism),
        payload=dict(
            formal_decision=formal,
            conditions=conditions,
            supplementary_necessary_support=support,
            score_authority=False,
            formal_distinguishing_checks=body["distinguishing_checks"],
            distinguishing_checks=remaining,
        ),
        evidence_ids=tuple(dict.fromkeys((*card.evidence_ids, *used))),
        source_refs=card.source_refs,
        interval=data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
        truth=support,
        limitations=(
            "supplementary_recheck_not_formal_replacement",
            "necessary_support_not_causal_confirmation",
            "refuting_and_unknown_conditions_must_be_retained",
        ),
    )
