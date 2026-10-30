"""Pure B1 expression values, never a measurement or scoring authority.

The B1a kernel verifies DAG structure and replays AND/OR operations. Leaf truth
and field references must be built by the condition consumers and independently
source-replayed by session/guard. Even a well-formed, correctly hashed leaf does
not prove its measurements. No source store, query, transport or ranker is used.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Literal, cast

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.b0_rule_binding import B0RuleBinding, require_b0_rule_binding
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.observation_schemas import CaseTimeContract, Observation, TimeInterval
from nurse_rcl.v2.predicate_plan_evaluator import _unavailable_reasons
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    require_frozen_clock_resolution,
)

from .contracts import (
    DynamicConditionEvaluation,
    DynamicExpressionNode,
    DynamicFieldReference,
    DynamicGraphTimePair,
    DynamicLocalTimePair,
    DynamicPredicateEvaluation,
    DynamicRequestCatalog,
    DynamicRequestConsumption,
    DynamicSeedEvaluation,
    DynamicSupportSource,
    DynamicTimeEvaluation,
    DynamicTimeSide,
    DynamicTimeSideEvaluation,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    ExpressionLeafKind,
    SignalEpisodeMeasurement,
    TruthValue,
    VerificationSeries,
    VerificationSourceCatalog,
    field_reference_key,
)
from .requests import (
    build_dynamic_request_catalog,
    build_dynamic_verification_policy,
    build_dynamic_verification_requests,
)
from .safety import canonical, plain, require_pin, require_unchanged, snapshot

# Seven predicate kinds in the six approved rows. Normal/refuting, lifecycle,
# dedicated storage/network and temporal predicates deliberately remain outside.
_PERSISTENCE_FIELDS = {
    "incoming_request_rate_increase": "persistent_up",
    "local_pressure_observed": "persistent_up",
    "local_error_increase": "persistent_up",
    "request_failure_increase": "persistent_up",
    "request_latency_increase": "persistent_up",
    "outgoing_dependency_failure_observed": "persistent_up",
    "local_anomaly_observed": "persistent_deviation_present",
}

_LOCAL_TIME_RECIPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "incoming_load_precedes_local_pressure": (
        ("incoming_request_rate_increase",),
        ("local_pressure_observed",),
    ),
    "dependency_failure_precedes_local_symptom": (
        ("outgoing_dependency_failure_observed",),
        ("local_error_increase", "request_latency_increase"),
    ),
}


def _references(
    values: tuple[DynamicFieldReference, ...],
) -> tuple[DynamicFieldReference, ...]:
    by_key = {field_reference_key(item): item for item in values}
    return tuple(by_key[key] for key in sorted(by_key))


def _merge_support_sources(
    values: tuple[DynamicSupportSource, ...],
) -> tuple[DynamicSupportSource, ...]:
    by_key: dict[str, DynamicSupportSource] = {}
    for item in values:
        key = item.source_key
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = item
            continue
        if (
            previous.candidate_service_key != item.candidate_service_key
            or previous.source_fingerprint != item.source_fingerprint
            or previous.adapter_semantics_sha256 != item.adapter_semantics_sha256
        ):
            raise ValueError("dynamic support source identity collision")
        quality = (
            None
            if previous.coverage_quality is None or item.coverage_quality is None
            else min(previous.coverage_quality, item.coverage_quality)
        )
        by_key[key] = DynamicSupportSource(
            candidate_service_key=item.candidate_service_key,
            source_fingerprint=item.source_fingerprint,
            adapter_semantics_sha256=item.adapter_semantics_sha256,
            observation_ids=tuple(sorted(set(previous.observation_ids + item.observation_ids))),
            field_references=_references(previous.field_references + item.field_references),
            coverage_quality=quality,
        )
    return tuple(by_key[key] for key in sorted(by_key))


def _seal(node: DynamicExpressionNode) -> DynamicExpressionNode:
    document = cast(dict[str, object], plain(node))
    document.pop("expression_id")
    return node.model_copy(
        update={"expression_id": "expr_" + sha256_hex(canonical_json_bytes(document))}
    )


def _fresh_node(value: DynamicExpressionNode) -> DynamicExpressionNode:
    raw = canonical(value, (DynamicExpressionNode,))
    fresh = DynamicExpressionNode.model_validate_json(raw)
    if canonical(fresh, (DynamicExpressionNode,)) != raw or _seal(fresh) != fresh:
        raise ValueError("dynamic expression node content integrity mismatch")
    return fresh


def _build_expression_leaf(
    *,
    kind: ExpressionLeafKind,
    truth_value: TruthValue,
    field_references: tuple[DynamicFieldReference, ...],
    support_sources: tuple[DynamicSupportSource, ...],
    pending_demand_keys: tuple[str, ...],
    reason_codes: tuple[str, ...],
) -> DynamicExpressionNode:
    """Internal value constructor, not an alternative to source-bound evaluation."""
    inputs = (
        kind,
        truth_value,
        field_references,
        support_sources,
        pending_demand_keys,
        reason_codes,
    )
    roots = (DynamicFieldReference, DynamicSupportSource)
    before = snapshot(inputs, roots)
    try:
        if type(kind) is not str or kind not in {
            "original_window",
            "directional_persistence",
            "directional_time",
            "summary_not_persistent",
            "pending",
        }:
            raise ValueError("dynamic expression leaf kind is invalid")
        if any(type(value) is not tuple for value in inputs[2:]):
            raise ValueError("dynamic expression leaf requires exact tuple collections")
        if any(type(item) is not str for item in pending_demand_keys + reason_codes):
            raise ValueError("dynamic expression leaf metadata requires exact strings")
        if any(type(value) is not DynamicFieldReference for value in field_references) or any(
            type(value) is not DynamicSupportSource for value in support_sources
        ):
            raise ValueError("dynamic expression leaf requires exact source rows")
        sources = _merge_support_sources(support_sources)
        node = DynamicExpressionNode(
            expression_id="expr_" + "0" * 64,
            kind=kind,
            truth_value=truth_value,
            child_ids=(),
            field_references=_references(
                field_references
                + tuple(ref for source in sources for ref in source.field_references)
            ),
            support_sources=sources,
            pending_demand_keys=tuple(sorted(set(pending_demand_keys))),
            reason_codes=tuple(sorted(set(reason_codes))),
        )
        return _fresh_node(_seal(node))
    finally:
        require_unchanged(inputs, roots, before)


def _compose(
    operator: Literal["and", "or"], operands: tuple[DynamicExpressionNode, ...]
) -> DynamicExpressionNode:
    by_id = {item.expression_id: item for item in operands}
    truths = {item.truth_value for item in by_id.values()}
    truth: TruthValue
    if operator == "and":
        truth = "false" if "false" in truths else "unknown" if "unknown" in truths else "true"
    else:
        truth = "true" if "true" in truths else "unknown" if "unknown" in truths else "false"
    return _seal(
        DynamicExpressionNode(
            expression_id="expr_" + "0" * 64,
            kind=operator,
            truth_value=truth,
            child_ids=tuple(sorted(by_id)),
            field_references=_references(
                tuple(ref for child in by_id.values() for ref in child.field_references)
            ),
            support_sources=_merge_support_sources(
                tuple(
                    source
                    for child in by_id.values()
                    if child.truth_value == "true"
                    for source in child.support_sources
                )
            )
            if truth == "true"
            else (),
            pending_demand_keys=tuple(
                sorted({key for child in by_id.values() for key in child.pending_demand_keys})
            ),
            reason_codes=(f"strong_kleene_{operator}:{truth}",),
        )
    )


def compose_dynamic_expressions(
    *, operator: Literal["and", "or"], operands: tuple[DynamicExpressionNode, ...]
) -> DynamicExpressionNode:
    """Compose nonempty exact values, retaining every true branch's source quality."""
    if type(operator) is not str or operator not in {"and", "or"}:
        raise ValueError("dynamic expression composition requires and/or")
    if type(operands) is not tuple or not operands:
        raise ValueError("dynamic expression composition requires a nonempty exact tuple")
    if any(type(item) is not DynamicExpressionNode for item in operands):
        raise ValueError("dynamic expression composition requires exact nodes")
    roots = (DynamicExpressionNode,)
    before = snapshot(operands, roots)
    try:
        fresh = tuple(_fresh_node(item) for item in operands)
        return _compose(operator, fresh)
    finally:
        require_unchanged(operands, roots, before)


def replay_dynamic_expression_dag(
    *, nodes: tuple[DynamicExpressionNode, ...], root_ids: tuple[str, ...]
) -> tuple[DynamicExpressionNode, ...]:
    """Replay closed, child-before-parent operations; leaf provenance is NOT verified.

    A session/guard must additionally regenerate every leaf from pinned sources.
    Internal replay alone intentionally grants no query, role or score authority.
    """
    if (
        type(nodes) is not tuple
        or not nodes
        or any(type(item) is not DynamicExpressionNode for item in nodes)
    ):
        raise ValueError("dynamic expression DAG requires a nonempty exact node tuple")
    if type(root_ids) is not tuple or not root_ids or any(type(key) is not str for key in root_ids):
        raise ValueError("dynamic expression DAG requires nonempty exact root IDs")
    roots = (DynamicExpressionNode,)
    inputs = (nodes, root_ids)
    before = snapshot(inputs, roots)
    try:
        if root_ids != tuple(sorted(set(root_ids))):
            raise ValueError("dynamic expression DAG roots must be unique and ordered")
        rebuilt: dict[str, DynamicExpressionNode] = {}
        for original in nodes:
            node = _fresh_node(original)
            if node.expression_id in rebuilt:
                raise ValueError("dynamic expression DAG contains duplicate nodes")
            if any(key not in rebuilt for key in node.child_ids):
                raise ValueError("dynamic expression DAG must be closed and strictly forward")
            if node.kind == "and" or node.kind == "or":
                expected = _compose(node.kind, tuple(rebuilt[key] for key in node.child_ids))
                if expected != node:
                    raise ValueError("dynamic expression operation differs from replay")
            rebuilt[node.expression_id] = node
        if any(key not in rebuilt for key in root_ids):
            raise ValueError("dynamic expression DAG has unresolved roots")
        pending = list(root_ids)
        reachable: set[str] = set()
        while pending:
            key = pending.pop()
            if key in reachable:
                continue
            reachable.add(key)
            pending.extend(rebuilt[key].child_ids)
        if reachable != set(rebuilt):
            raise ValueError("dynamic expression DAG contains unreferenced nodes")
        return tuple(rebuilt.values())
    finally:
        require_unchanged(inputs, roots, before)


def _observation_reference(observation: Observation, path: str) -> DynamicFieldReference:
    return DynamicFieldReference(
        source_kind="observation",
        source_sha256=sha256_hex(canonical(observation, (Observation,))),
        field_path=path,
    )


def _source_support(
    observation: Observation,
    candidate: str,
    references: tuple[DynamicFieldReference, ...],
    quality: float,
) -> DynamicSupportSource:
    if observation.adapter_semantics_sha256 is None:
        raise ValueError("dynamic condition support lacks frozen adapter semantics")
    return DynamicSupportSource(
        candidate_service_key=candidate,
        source_fingerprint=observation.source_fingerprint,
        adapter_semantics_sha256=observation.adapter_semantics_sha256,
        observation_ids=(observation.observation_id,),
        field_references=_references(references),
        coverage_quality=quality,
    )


def _original_condition_node(
    predicate: PredicateResult, candidate: str, observations: dict[str, Observation]
) -> DynamicExpressionNode:
    reference = DynamicFieldReference(
        source_kind="original_predicate",
        source_sha256=predicate.predicate_sha256,
        field_path="truth_value",
    )
    references = [reference]
    supports = []
    pending = [predicate]
    while pending:
        item = pending.pop()
        references.append(
            DynamicFieldReference(
                source_kind="original_predicate",
                source_sha256=item.predicate_sha256,
                field_path="truth_value",
            )
        )
        if item.operand_states:
            pending.extend(item.operand_states)
            continue
        for key in item.observation_ids:
            observation = observations[key]
            refs = (
                _observation_reference(observation, "anomaly_strength"),
                _observation_reference(observation, "coverage_quality"),
                DynamicFieldReference(
                    source_kind="original_predicate",
                    source_sha256=item.predicate_sha256,
                    field_path="truth_value",
                ),
            )
            references.extend(refs)
            if predicate.truth_value == "true" and item.truth_value == "true":
                supports.append(
                    _source_support(
                        observation,
                        candidate,
                        refs,
                        observation.coverage_quality,
                    )
                )
    return _build_expression_leaf(
        kind="original_window",
        truth_value=predicate.truth_value,
        field_references=tuple(references),
        support_sources=tuple(supports),
        pending_demand_keys=(),
        reason_codes=(predicate.reason or "original_predicate",),
    )


def _check_condition_sources(
    envelope: RuleHypothesisEnvelope,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    measurements: tuple[SignalEpisodeMeasurement, ...],
) -> None:
    """Check joins, not raw-block authority (which only A3/A4 can establish)."""
    if not catalog.verify() or (
        catalog.opaque_case_id != envelope.candidate_universe.opaque_case_id
        or catalog.candidate_freeze_sha256 != envelope.frozen_candidates.freeze_sha256
    ):
        raise ValueError("dynamic condition catalog case/freeze mismatch")
    observations = {row.observation_id: row for row in envelope.event_grouping.observations}
    resolutions = {
        row.observation.observation_id: row for row in envelope.entity_registry.resolutions
    }
    if {row.observation_id for row in catalog.rows} != set(observations):
        raise ValueError("dynamic condition catalog must cover every original observation")
    by_series = {row.canonical_series_key: row for row in catalog.series}
    for row in catalog.rows:
        observation = observations[row.observation_id]
        binding = resolutions[row.observation_id].binding
        event_ids = tuple(
            sorted(
                group.event_group_id
                for group in envelope.event_grouping.event_groups
                if row.observation_id in group.observation_ids
            )
        )
        if (
            row.candidate_service_key != binding.service_entity_key
            or row.event_group_ids != event_ids
        ):
            raise ValueError("dynamic condition observation candidate/event binding mismatch")
        if row.canonical_series_key is None:
            continue
        series = by_series[row.canonical_series_key]
        if (
            row.source_identity is None
            or row.source_identity.observed_entity_key
            != resolutions[row.observation_id].observed_entity_key
            or row.source_identity.entity_kind != resolutions[row.observation_id].entity_kind
            or series.source_fingerprint != observation.source_fingerprint
            or series.adapter_semantics_sha256 != observation.adapter_semantics_sha256
            or series.transform_result.transform != observation.transform
        ):
            raise ValueError("dynamic condition observation source identity mismatch")
        for field in (
            "signal_type",
            "unit",
            "aggregation",
            "baseline_summary",
            "fault_summary",
            "evidence_state",
            "anomaly_strength",
            "coverage_quality",
            "minimum_valid_points",
            "formal_scoring_eligible",
            "quality_flags",
        ):
            if getattr(series.quality_result, field) != getattr(observation, field):
                raise ValueError(
                    "dynamic condition observation source semantics mismatch: " + field
                )
    by_request = {row.request_sha256: row for row in requests}
    if {row.canonical_series_key for row in requests} != set(by_series):
        raise ValueError("dynamic condition request directory must cover all frozen series")
    if len(by_request) != len(requests) or len({r.canonical_series_key for r in requests}) != len(
        requests
    ):
        raise ValueError("dynamic condition duplicate request")
    for request in requests:
        request_series = by_series.get(request.canonical_series_key)
        if (
            not request.verify()
            or request_series is None
            or (
                request.catalog_sha256 != catalog.catalog_sha256
                or request.time_contract_sha256 != catalog.time_contract_sha256
                or request.candidate_freeze_sha256 != catalog.candidate_freeze_sha256
                or request.opaque_case_id != catalog.opaque_case_id
                or request.candidate_service_key != request_series.candidate_service_key
                or request.permission_scope != envelope.scope_interval
                or request.allowed_candidate_service_keys
                != envelope.frozen_candidates.candidate_service_keys
                or request.policy_sha256 != build_dynamic_verification_policy().policy_sha256
                or request.quality_configuration_sha256
                != sha256_hex(
                    canonical_json_bytes(plain(request_series.quality_result.configuration))
                )
                or request.transform_component != request_series.transform_component
                or request.baseline_window != request_series.transform_result.windows[0].interval
                or request.fault_window != request_series.transform_result.windows[1].interval
            )
        ):
            raise ValueError("dynamic condition request binding mismatch")
    if len({row.request_sha256 for row in measurements}) != len(measurements):
        raise ValueError("dynamic condition duplicate measurement request")
    for measured in measurements:
        result_request = by_request.get(measured.request_sha256)
        if not measured.verify() or result_request is None:
            raise ValueError("dynamic condition measurement request/integrity mismatch")
        series = by_series[result_request.canonical_series_key]
        if (
            measured.catalog_sha256 != catalog.catalog_sha256
            or measured.policy_sha256 != result_request.policy_sha256
            or measured.canonical_series_key != series.canonical_series_key
            or measured.source_input_sha256 != series.source_input_sha256
            or measured.quality != series.quality_result
            or measured.baseline_median != series.quality_result.baseline_summary.median
            or measured.baseline_scale != series.quality_result.baseline_scale
            or measured.abnormal_z_threshold
            != series.quality_result.configuration.abnormal_z_threshold
            or result_request.availability != "available"
        ):
            raise ValueError("dynamic condition measurement source/quality binding mismatch")


def _persistence_node(
    *,
    predicate_kind: str,
    candidate: str,
    observation: Observation,
    primary_family: str,
    series: VerificationSeries | None,
    request: DynamicVerificationRequest | None,
    measurement: SignalEpisodeMeasurement | None,
    pending_demand_keys: tuple[str, ...],
) -> DynamicExpressionNode:
    refs = [
        _observation_reference(observation, name)
        for name in (
            "signal_family",
            "unit",
            "aggregation",
            "coverage_quality",
            "quality_flags",
        )
    ]
    reasons = list(_unavailable_reasons(observation))
    contract = _PREDICATE_EVIDENCE_CONTRACTS[predicate_kind]
    if (
        primary_family not in contract.primary_signal_families
        and observation.signal_family not in contract.source_signal_families
    ):
        reasons.append("signal_family_denied")
    if series is None:
        reasons.append("source_unavailable")
    elif predicate_kind == "incoming_request_rate_increase" and (
        series.aggregation not in {"per_second_rate", "exposure_rate"}
        or not series.unit.endswith("/second")
    ):
        reasons.append("rate_semantics_unavailable")
    if request is not None:
        refs.append(
            DynamicFieldReference(
                source_kind="request",
                source_sha256=request.request_sha256,
                field_path="canonical_series_key",
            )
        )
    if measurement is not None:
        refs.extend(
            (
                DynamicFieldReference(
                    source_kind="measurement",
                    source_sha256=measurement.measurement_sha256,
                    field_path=_PERSISTENCE_FIELDS[predicate_kind],
                ),
                DynamicFieldReference(
                    source_kind="quality",
                    source_sha256=measurement.quality.result_sha256,
                    field_path="coverage_quality",
                ),
            )
        )
    elif not reasons and series is not None:
        proof = _derive_summary_nonpersistence(
            series=series, policy=build_dynamic_verification_policy()
        )
        return _build_expression_leaf(
            kind="summary_not_persistent",
            truth_value=proof.truth_value,
            field_references=(*refs, *proof.field_references),
            support_sources=(),
            pending_demand_keys=pending_demand_keys if proof.truth_value == "unknown" else (),
            reason_codes=proof.reason_codes,
        )
    truth: TruthValue = "unknown"
    if not reasons and measurement is not None:
        truth = cast(TruthValue, getattr(measurement, _PERSISTENCE_FIELDS[predicate_kind]))
        reasons.extend(measurement.reason_codes)
        reasons.append("directional_persistence:" + truth)
    return _build_expression_leaf(
        kind="pending" if measurement is None else "directional_persistence",
        truth_value=truth,
        field_references=tuple(refs),
        support_sources=(
            _source_support(
                observation, candidate, tuple(refs), measurement.quality.coverage_quality
            ),
        )
        if truth == "true" and measurement is not None
        else (),
        pending_demand_keys=pending_demand_keys if measurement is None else (),
        reason_codes=tuple(reasons),
    )


def _evaluate_bound_condition(
    *,
    envelope: RuleHypothesisEnvelope,
    original_binding: B0RuleBinding,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    measurements: tuple[SignalEpisodeMeasurement, ...],
    seed_hypothesis_id: str,
    seed_predicate_id: str,
    pending_demand_keys: tuple[str, ...],
) -> tuple[DynamicExpressionNode, ...]:
    """Internal condition compiler; not the independent-pin public B1 entry.

    Derives values from complete original seeds and already obtained measurements.
    Verifies joins and original replay, but never claims raw measurement authority.
    The enclosing session must A4-replay blocks before invoking B1. Demand keys
    must come from the frozen request catalog, not a caller-assigned priority.
    """
    roots: tuple[type[BaseModel], ...] = (
        RuleHypothesisEnvelope,
        B0RuleBinding,
        VerificationSourceCatalog,
        DynamicVerificationRequest,
        SignalEpisodeMeasurement,
    )
    inputs = (
        envelope,
        original_binding,
        catalog,
        requests,
        measurements,
        seed_hypothesis_id,
        seed_predicate_id,
        pending_demand_keys,
    )
    before = snapshot(inputs, roots)
    try:
        if (
            type(envelope) is not RuleHypothesisEnvelope
            or type(original_binding) is not B0RuleBinding
            or type(catalog) is not VerificationSourceCatalog
            or type(requests) is not tuple
            or type(measurements) is not tuple
            or type(pending_demand_keys) is not tuple
            or any(type(item) is not DynamicVerificationRequest for item in requests)
            or any(type(item) is not SignalEpisodeMeasurement for item in measurements)
            or any(
                type(item) is not str
                for item in (seed_hypothesis_id, seed_predicate_id, *pending_demand_keys)
            )
        ):
            raise ValueError("dynamic condition requires exact frozen inputs")
        source = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, roots))
        binding = require_b0_rule_binding(
            envelope=source,
            expected_rule_set_sha256=source.rule_hypothesis_set.result_sha256,
            binding=original_binding,
            expected_binding_sha256=original_binding.binding_sha256,
        )
        sources = VerificationSourceCatalog.model_validate_json(canonical(catalog, roots))
        frozen_requests = tuple(
            DynamicVerificationRequest.model_validate_json(canonical(row, roots))
            for row in requests
        )
        results = tuple(
            SignalEpisodeMeasurement.model_validate_json(canonical(row, roots))
            for row in measurements
        )
        _check_condition_sources(source, sources, frozen_requests, results)
        return _compile_bound_condition_values(
            source=source,
            binding=binding,
            sources=sources,
            frozen_requests=frozen_requests,
            results=results,
            seed_hypothesis_id=seed_hypothesis_id,
            seed_predicate_id=seed_predicate_id,
            pending_demand_keys=pending_demand_keys,
        )
    finally:
        require_unchanged(inputs, roots, before)


def _compile_bound_condition_values(
    *,
    source: RuleHypothesisEnvelope,
    binding: B0RuleBinding,
    sources: VerificationSourceCatalog,
    frozen_requests: tuple[DynamicVerificationRequest, ...],
    results: tuple[SignalEpisodeMeasurement, ...],
    seed_hypothesis_id: str,
    seed_predicate_id: str,
    pending_demand_keys: tuple[str, ...],
) -> tuple[DynamicExpressionNode, ...]:
    """Pure inner compilation after the caller has replayed complete frozen inputs.

    Not an alternative source verifier. Both entry wrappers retain their full
    replay/joins and finally guards; a complete evaluation validates once rather
    than rebuilding every original predicate for each individual condition.
    """
    hypothesis = next(
        (
            row
            for row in source.rule_hypothesis_set.hypotheses
            if row.hypothesis_id == seed_hypothesis_id
        ),
        None,
    )
    seed = next(
        (
            row
            for row in source.rule_hypothesis_set.predicates
            if row.predicate_result_id == seed_predicate_id
        ),
        None,
    )
    if (
        hypothesis is None
        or seed is None
        or seed_predicate_id
        not in (
            hypothesis.necessary_predicate_result_ids
            + hypothesis.optional_predicate_result_ids
            + hypothesis.refuting_predicate_result_ids
        )
    ):
        raise ValueError("dynamic condition requires a predicate of the original hypothesis")
    candidate = hypothesis.candidate_service_key
    observations = {row.observation_id: row for row in source.event_grouping.observations}
    rows = {row.observation_id: row for row in sources.rows}
    for key in seed.observation_ids:
        if (
            key not in hypothesis.observation_ids
            or rows[key].candidate_service_key != candidate
            or not set(rows[key].event_group_ids).issubset(hypothesis.event_group_ids)
        ):
            raise ValueError("dynamic condition source exceeds original candidate/event permission")
    original_id = dict(binding.predicate_id_mapping)[seed_predicate_id]
    predicate = next(row for row in binding.predicates if row.predicate_result_id == original_id)
    old = _original_condition_node(predicate, candidate, observations)
    if seed.predicate_kind not in _PERSISTENCE_FIELDS or seed.requirement == "refuting":
        return (old,)
    series = {row.canonical_series_key: row for row in sources.series}
    request_by_series = {row.canonical_series_key: row for row in frozen_requests}
    result_by_series = {row.canonical_series_key: row for row in results}
    families = {
        key: group.primary_signal_family
        for group in source.event_grouping.event_groups
        for key in group.observation_ids
    }
    dynamic = tuple(
        _persistence_node(
            predicate_kind=seed.predicate_kind,
            candidate=candidate,
            observation=observations[key],
            primary_family=families[key],
            series=series.get(rows[key].canonical_series_key or ""),
            request=request_by_series.get(rows[key].canonical_series_key or ""),
            measurement=result_by_series.get(rows[key].canonical_series_key or ""),
            pending_demand_keys=pending_demand_keys,
        )
        for key in seed.observation_ids
    )
    if not dynamic:
        dynamic = (
            _build_expression_leaf(
                kind="pending",
                truth_value="unknown",
                field_references=(),
                support_sources=(),
                pending_demand_keys=pending_demand_keys,
                reason_codes=("no_bound_observations",),
            ),
        )
    combined = compose_dynamic_expressions(operator="or", operands=(old, *dynamic))
    unique = {row.expression_id: row for row in (old, *dynamic, combined)}
    return tuple(unique.values())


def _derive_summary_nonpersistence(
    *,
    series: VerificationSeries,
    policy: DynamicVerificationPolicy,
) -> DynamicExpressionNode:
    """Prove no qualifying direction using existing extrema and timing metadata.

    This does not classify point values, build episodes or read a source store.
    Only the frozen transform's timestamps/source intervals establish no internal
    gaps; no unqueried segment/onset is derived. Integrity validation still reads
    the complete frozen object. A3/A4 source authority is required by the caller.
    The negative conclusion is not verified-normal and carries no score support.
    """
    roots = (VerificationSeries, DynamicVerificationPolicy)
    inputs = (series, policy)
    before = snapshot(inputs, roots)
    try:
        if type(series) is not VerificationSeries or type(policy) is not DynamicVerificationPolicy:
            raise ValueError("summary proof requires exact series and policy")
        source = VerificationSeries.model_validate_json(canonical(series, roots))
        strategy = DynamicVerificationPolicy.model_validate_json(canonical(policy, roots))
        if strategy != build_dynamic_verification_policy() or not strategy.verify():
            raise ValueError("summary proof policy differs from the fixed strategy")
        if canonical((source, strategy), roots) != before[0]:
            raise ValueError("summary proof input differs from canonical replay")
        quality = source.quality_result
        baseline, fault = quality.baseline_summary, quality.fault_summary
        refs = [
            DynamicFieldReference(
                source_kind="quality",
                source_sha256=quality.result_sha256,
                field_path=field,
            )
            for field in (
                "baseline_summary",
                "baseline_scale",
                "fault_summary.minimum",
                "fault_summary.maximum",
                "fault_summary.valid_point_count",
                "fault_summary.coverage_fraction",
                "baseline_summary_sufficient",
                "fault_summary_sufficient",
                "formal_scoring_eligible",
                "quality_flags",
                "configuration",
                "context.expected_cadence_ns",
            )
        ]
        refs.append(
            DynamicFieldReference(
                source_kind="policy",
                source_sha256=strategy.policy_sha256,
                field_path="maximum_gap_numerator",
            )
        )
        refs.append(
            DynamicFieldReference(
                source_kind="policy",
                source_sha256=strategy.policy_sha256,
                field_path="maximum_gap_denominator",
            )
        )
        reasons = []
        if (
            not quality.formal_scoring_eligible
            or quality.evidence_state in {"unknown", "degraded", "no_record"}
            or not quality.baseline_summary_sufficient
            or not quality.fault_summary_sufficient
            or baseline.coverage_fraction < quality.configuration.minimum_coverage_fraction
            or fault.coverage_fraction < quality.configuration.minimum_coverage_fraction
        ):
            reasons.append("summary_quality_unavailable")
        median, scale = baseline.median, quality.baseline_scale
        if (
            median is None
            or scale is None
            or scale <= 0.0
            or fault.minimum is None
            or fault.maximum is None
        ):
            reasons.append("summary_deviation_band_unavailable")
        else:
            lower_z, upper_z = (fault.minimum - median) / scale, (fault.maximum - median) / scale
            threshold = quality.configuration.abnormal_z_threshold
            if not (
                math.isfinite(lower_z)
                and math.isfinite(upper_z)
                and -threshold < lower_z <= upper_z < threshold
            ):
                reasons.append("summary_not_strictly_inside_band")
        component_index, component = next(
            (index, row)
            for index, row in enumerate(source.transform_result.windows[1].components)
            if row.name == source.transform_component
        )
        path = f"windows[1].components[{component_index}]"
        for field in ("name", "aggregation", "status", "formal_scoring_eligible", "quality_flags"):
            refs.append(
                DynamicFieldReference(
                    source_kind="transform",
                    source_sha256=source.transform_result.result_sha256,
                    field_path=f"{path}.{field}",
                )
            )
        if component.aggregation in {"window_count", "exposure_rate"}:
            reasons.append("temporal_resolution_unavailable")
        if (
            not component.formal_scoring_eligible
            or len(component.samples) != fault.valid_point_count
            or not component.samples
        ):
            reasons.append("summary_all_points_not_proven")
        for index, sample in enumerate(component.samples):
            for field in ("timestamp_unix_ns", "source_interval"):
                refs.append(
                    DynamicFieldReference(
                        source_kind="transform",
                        source_sha256=source.transform_result.result_sha256,
                        field_path=f"{path}.samples[{index}].{field}",
                    )
                )
            if index == 0:
                continue
            previous = component.samples[index - 1]
            if (
                strategy.maximum_gap_denominator
                * (sample.timestamp_unix_ns - previous.timestamp_unix_ns)
                > strategy.maximum_gap_numerator * source.expected_cadence_ns
            ):
                reasons.append("sampling_gap")
            if (
                previous.source_interval is not None
                and sample.source_interval is not None
                and sample.source_interval.start_unix_ns > previous.source_interval.end_unix_ns
            ):
                reasons.append("source_interval_gap")
        return _build_expression_leaf(
            kind="summary_not_persistent",
            truth_value="unknown" if reasons else "false",
            field_references=tuple(refs),
            support_sources=(),
            pending_demand_keys=(),
            reason_codes=tuple(reasons)
            if reasons
            else ("strict_in_band_summary_no_internal_gaps",),
        )
    finally:
        require_unchanged(inputs, roots, before)


def _freeze_time_side(
    candidate: str,
    kinds: tuple[str, ...],
    envelope: RuleHypothesisEnvelope,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
) -> DynamicTimeSide:
    families = {
        key: group.primary_signal_family
        for group in envelope.event_grouping.event_groups
        for key in group.observation_ids
    }
    observations = {row.observation_id: row for row in envelope.event_grouping.observations}
    rows = tuple(
        row
        for row in catalog.rows
        if row.candidate_service_key == candidate
        and any(
            families[row.observation_id]
            in _PREDICATE_EVIDENCE_CONTRACTS[kind].primary_signal_families
            or observations[row.observation_id].signal_family
            in _PREDICATE_EVIDENCE_CONTRACTS[kind].source_signal_families
            for kind in kinds
        )
    )
    keys = tuple(
        sorted({row.canonical_series_key for row in rows if row.canonical_series_key is not None})
    )
    return DynamicTimeSide(
        candidate_service_key=candidate,
        predicate_kinds=tuple(sorted(kinds)),
        observation_ids=tuple(sorted(row.observation_id for row in rows)),
        event_group_ids=tuple(sorted({key for row in rows for key in row.event_group_ids})),
        canonical_series_keys=keys,
        request_keys=tuple(
            sorted(
                row.canonical_request_key for row in requests if row.canonical_series_key in keys
            )
        ),
        direction="up",
    )


def _build_local_time_pairs(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
) -> tuple[DynamicLocalTimePair, ...]:
    """Freeze the two code-owned optional recipes before observing any query result.

    Private builder shared by the forthcoming complete request catalog. Graph
    edges are a separate source domain, never inferred from these local recipes.
    """
    roots = (
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceCatalog,
        DynamicVerificationRequest,
        DynamicVerificationPolicy,
    )
    inputs = (envelope, time_contract, catalog, requests)
    before = snapshot(inputs, roots)
    try:
        with ExitStack() as guards:
            if (
                type(envelope) is not RuleHypothesisEnvelope
                or type(time_contract) is not CaseTimeContract
                or type(catalog) is not VerificationSourceCatalog
                or type(requests) is not tuple
                or any(type(row) is not DynamicVerificationRequest for row in requests)
            ):
                raise ValueError("local time pairs require exact frozen inputs")
            source = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, roots))
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, roots))
            sources = VerificationSourceCatalog.model_validate_json(canonical(catalog, roots))
            strategy = build_dynamic_verification_policy()
            working = (source, contract, sources, strategy)
            guards.callback(require_unchanged, working, roots, snapshot(working, roots))
            expected = build_dynamic_verification_requests(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                expected_catalog_sha256=sources.catalog_sha256,
                policy=strategy,
                expected_policy_sha256=strategy.policy_sha256,
            )
            guards.callback(require_unchanged, expected, roots, snapshot(expected, roots))
            if canonical(requests, roots) != canonical(expected, roots):
                raise ValueError("local time request directory differs from frozen time/sources")
            _check_condition_sources(source, sources, expected, ())
            return _local_time_pair_values(
                envelope=source, time_contract=contract, catalog=sources, requests=expected
            )
    finally:
        require_unchanged(inputs, roots, before)


def _local_time_pair_values(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
) -> tuple[DynamicLocalTimePair, ...]:
    """Build only values; the enclosing operation owns authentication and snapshots."""
    source, contract, sources, expected = envelope, time_contract, catalog, requests
    seeds = {row.predicate_result_id: row for row in source.rule_hypothesis_set.predicates}
    pairs = []
    for hypothesis in source.rule_hypothesis_set.hypotheses:
        for key in hypothesis.optional_predicate_result_ids:
            seed = seeds[key]
            recipe = _LOCAL_TIME_RECIPES.get(seed.predicate_kind)
            if recipe is None:
                continue
            condition = "condition_" + sha256_hex(
                canonical_json_bytes(
                    [
                        hypothesis.candidate_service_key,
                        seed.predicate_kind,
                        "optional",
                    ]
                )
            )
            document = {
                "condition_key": condition,
                "opaque_case_id": sources.opaque_case_id,
                "candidate_freeze_sha256": sources.candidate_freeze_sha256,
                "time_contract_sha256": sources.time_contract_sha256,
                "catalog_sha256": sources.catalog_sha256,
                "source_rule_set_sha256": source.rule_hypothesis_set.result_sha256,
                "source_kind": "mechanism_seed",
                "seed_hypothesis_id": hypothesis.hypothesis_id,
                "seed_predicate_id": key,
                "predicate_kind": seed.predicate_kind,
                "requirement": "optional",
                "source": plain(
                    _freeze_time_side(
                        hypothesis.candidate_service_key, recipe[0], source, sources, expected
                    )
                ),
                "target": plain(
                    _freeze_time_side(
                        hypothesis.candidate_service_key, recipe[1], source, sources, expected
                    )
                ),
                "relation_window": plain(contract.fault_window),
            }
            document["time_pair_key"] = "timepair_" + sha256_hex(canonical_json_bytes(document))
            pairs.append(DynamicLocalTimePair.model_validate_json(canonical_json_bytes(document)))
    return tuple(sorted(pairs, key=lambda row: row.time_pair_key))


_TimeSideBounds = tuple[
    tuple[int, int] | None,
    tuple[DynamicFieldReference, ...],
    tuple[DynamicSupportSource, ...],
    tuple[str, ...],
    bool,
]
_BOUND_ROOTS = (DynamicFieldReference, DynamicSupportSource)


def _time_side_bounds(
    *,
    side: DynamicTimeSide,
    window: TimeInterval,
    envelope: RuleHypothesisEnvelope,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    measurements: tuple[SignalEpisodeMeasurement, ...],
) -> _TimeSideBounds:
    reasons = []
    refs: list[DynamicFieldReference] = []
    supports: list[DynamicSupportSource] = []
    rows = {row.observation_id: row for row in catalog.rows}
    by_series = {row.canonical_series_key: row for row in measurements}
    source_rows = {row.canonical_series_key: row for row in catalog.series}
    request_rows = {row.canonical_series_key: row for row in requests}
    observations = {row.observation_id: row for row in envelope.event_grouping.observations}
    families = {
        key: group.primary_signal_family
        for group in envelope.event_grouping.event_groups
        for key in group.observation_ids
    }
    if not side.observation_ids or any(
        rows[key].canonical_series_key is None for key in side.observation_ids
    ):
        reasons.append("time_source_unavailable")
    if len(side.canonical_series_keys) > 1:
        reasons.append("ambiguous_episode_binding")
    missing = any(key not in by_series for key in side.canonical_series_keys)
    if missing:
        reasons.append("time_measurement_pending")
    eligible: list[tuple[int, int]] = []
    qualifying_count = 0
    for key in side.canonical_series_keys:
        measured = by_series.get(key)
        if measured is None:
            continue
        refs.extend(
            (
                DynamicFieldReference(
                    source_kind="measurement",
                    source_sha256=measured.measurement_sha256,
                    field_path="episodes",
                ),
                DynamicFieldReference(
                    source_kind="measurement",
                    source_sha256=measured.measurement_sha256,
                    field_path="points",
                ),
            )
        )
        leaves = tuple(
            _persistence_node(
                predicate_kind=kind,
                candidate=side.candidate_service_key,
                observation=observations[obs_id],
                primary_family=families[obs_id],
                series=source_rows[key],
                request=request_rows[key],
                measurement=measured,
                pending_demand_keys=(),
            )
            for obs_id in side.observation_ids
            if rows[obs_id].canonical_series_key == key
            for kind in side.predicate_kinds
        )
        refs.extend(ref for leaf in leaves for ref in leaf.field_references)
        if not any(leaf.truth_value == "true" for leaf in leaves):
            reasons.append("time_direction_or_source_quality_unavailable")
        supports.extend(
            item for leaf in leaves if leaf.truth_value == "true" for item in leaf.support_sources
        )
        for index, episode in enumerate(measured.episodes):
            if not episode.qualifying or episode.direction != side.direction:
                continue
            if any(point_index >= len(measured.points) for point_index in episode.sample_indices):
                raise ValueError("dynamic time measurement episode has an invalid sample reference")
            points = tuple(measured.points[i] for i in episode.sample_indices)
            overlaps = any(
                window.start_unix_ns <= point.timestamp_unix_ns < window.end_unix_ns
                if point.source_interval is None
                else point.source_interval.start_unix_ns < window.end_unix_ns
                and point.source_interval.end_unix_ns > window.start_unix_ns
                for point in points
            )
            if not overlaps:
                continue
            qualifying_count += 1
            for field in (
                "onset_lower_unix_ns",
                "onset_upper_unix_ns",
                "left_censored",
                "right_censored",
                "direction",
                "qualifying",
                "sample_indices",
            ):
                refs.append(
                    DynamicFieldReference(
                        source_kind="measurement",
                        source_sha256=measured.measurement_sha256,
                        field_path=f"episodes[{index}].{field}",
                    )
                )
            lower, upper = episode.onset_lower_unix_ns, episode.onset_upper_unix_ns
            if lower is None or episode.left_censored:
                reasons.append("left_censored_onset")
            elif lower > upper:
                raise ValueError("dynamic time measurement onset bounds are invalid")
            elif lower < window.start_unix_ns or upper >= window.end_unix_ns:
                reasons.append("onset_outside_frozen_relation_window")
            else:
                eligible.append((lower, upper))
    if qualifying_count > 1:
        reasons.append("ambiguous_episode_binding")
    elif qualifying_count == 0 and not missing and "time_source_unavailable" not in reasons:
        reasons.append("no_qualifying_episode")
    bounds = eligible[0] if not reasons and len(eligible) == 1 else None
    return (
        bounds,
        _references(tuple(refs)),
        _merge_support_sources(tuple(supports)),
        tuple(sorted(set(reasons))),
        missing,
    )


def _compare_time_bounds(
    source: tuple[int, int] | None,
    target: tuple[int, int] | None,
    resolution: FrozenClockResolution,
    reasons: tuple[str, ...],
) -> tuple[TruthValue, tuple[str, ...]]:
    """Shared pure arithmetic after each source-specific frozen-domain gate."""
    collected = list(reasons)
    truth: TruthValue = "unknown"
    if resolution.resolution_status != "resolved":
        collected.append("clock_uncertainty_unresolved")
    if source is not None and target is not None and not collected:
        source_low, source_high = source
        target_low, target_high = target
        uncertainty = resolution.clock_uncertainty_ns
        source_low, source_high = max(0, source_low - uncertainty), source_high + uncertainty
        target_low, target_high = max(0, target_low - uncertainty), target_high + uncertainty
        if source_high < target_low:
            truth = "true"
            collected.append("strict_source_before_target")
        elif target_high < source_low:
            truth = "false"
            collected.append("strict_target_before_source")
        else:
            collected.append("onset_intervals_touch_or_overlap")
    return truth, tuple(collected)


def _evaluate_frozen_time_pair(
    *,
    pair: DynamicLocalTimePair,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    measurements: tuple[SignalEpisodeMeasurement, ...],
    clock_resolution: FrozenClockResolution,
    expected_clock_resolution_sha256: str,
    pending_demand_keys: tuple[str, ...],
) -> DynamicExpressionNode:
    """Private local time consumer; optional evidence only, never causation/demotion.

    Rebuilds the pre-query source domain and consumes only submitted A4 results.
    Cross-candidate graph-derived pairs are a separate forthcoming source variant.
    """
    roots = (
        DynamicLocalTimePair,
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceCatalog,
        DynamicVerificationRequest,
        SignalEpisodeMeasurement,
        FrozenClockResolution,
    )
    inputs = (
        pair,
        envelope,
        time_contract,
        catalog,
        requests,
        measurements,
        clock_resolution,
        expected_clock_resolution_sha256,
        pending_demand_keys,
    )
    before = snapshot(inputs, roots)
    try:
        with ExitStack() as guards:
            require_pin(expected_clock_resolution_sha256)
            if (
                type(pair) is not DynamicLocalTimePair
                or type(envelope) is not RuleHypothesisEnvelope
                or type(time_contract) is not CaseTimeContract
                or type(catalog) is not VerificationSourceCatalog
                or type(requests) is not tuple
                or any(type(row) is not DynamicVerificationRequest for row in requests)
                or type(clock_resolution) is not FrozenClockResolution
                or type(measurements) is not tuple
                or any(type(row) is not SignalEpisodeMeasurement for row in measurements)
                or type(pending_demand_keys) is not tuple
                or any(type(key) is not str for key in pending_demand_keys)
            ):
                raise ValueError("dynamic time evaluation requires exact frozen inputs")
            fresh_pair = DynamicLocalTimePair.model_validate_json(canonical(pair, roots))
            frozen_envelope = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, roots))
            frozen_time = CaseTimeContract.model_validate_json(canonical(time_contract, roots))
            frozen_catalog = VerificationSourceCatalog.model_validate_json(
                canonical(catalog, roots)
            )
            frozen_requests = tuple(
                DynamicVerificationRequest.model_validate_json(canonical(row, roots))
                for row in requests
            )
            working = (fresh_pair, frozen_envelope, frozen_time, frozen_catalog, frozen_requests)
            guards.callback(require_unchanged, working, roots, snapshot(working, roots))
            expected_pairs = _build_local_time_pairs(
                envelope=frozen_envelope,
                time_contract=frozen_time,
                catalog=frozen_catalog,
                requests=frozen_requests,
            )
            guards.callback(
                require_unchanged, expected_pairs, roots, snapshot(expected_pairs, roots)
            )
            if not any(
                canonical(row, roots) == canonical(fresh_pair, roots) for row in expected_pairs
            ):
                raise ValueError("dynamic time pair differs from its frozen mechanism source")
            resolution = require_frozen_clock_resolution(
                clock_resolution=clock_resolution,
                time_contract=frozen_time,
                expected_clock_resolution_sha256=expected_clock_resolution_sha256,
            )
            guards.callback(require_unchanged, resolution, roots, snapshot(resolution, roots))
            results = tuple(
                SignalEpisodeMeasurement.model_validate_json(canonical(row, roots))
                for row in measurements
            )
            guards.callback(require_unchanged, results, roots, snapshot(results, roots))
            _check_condition_sources(frozen_envelope, frozen_catalog, frozen_requests, results)
            source = _time_side_bounds(
                side=fresh_pair.source,
                window=fresh_pair.relation_window,
                envelope=frozen_envelope,
                catalog=frozen_catalog,
                requests=frozen_requests,
                measurements=results,
            )
            guards.callback(require_unchanged, source, _BOUND_ROOTS, snapshot(source, _BOUND_ROOTS))
            target = _time_side_bounds(
                side=fresh_pair.target,
                window=fresh_pair.relation_window,
                envelope=frozen_envelope,
                catalog=frozen_catalog,
                requests=frozen_requests,
                measurements=results,
            )
            guards.callback(require_unchanged, target, _BOUND_ROOTS, snapshot(target, _BOUND_ROOTS))
            return _local_time_comparison_values(
                fresh_pair=fresh_pair,
                resolution=resolution,
                source=source,
                target=target,
                pending_demand_keys=pending_demand_keys,
            )
    finally:
        require_unchanged(inputs, roots, before)


def _local_time_comparison_values(
    *,
    fresh_pair: DynamicLocalTimePair,
    resolution: FrozenClockResolution,
    source: _TimeSideBounds,
    target: _TimeSideBounds,
    pending_demand_keys: tuple[str, ...],
) -> DynamicExpressionNode:
    """Pure comparison over this call's source-certified pair and bounds."""
    truth, reasons = _compare_time_bounds(source[0], target[0], resolution, source[3] + target[3])
    refs = (
        *source[1],
        *target[1],
        DynamicFieldReference(
            source_kind="time_pair",
            source_sha256=fresh_pair.time_pair_sha256,
            field_path="source",
        ),
        DynamicFieldReference(
            source_kind="time_pair",
            source_sha256=fresh_pair.time_pair_sha256,
            field_path="target",
        ),
        DynamicFieldReference(
            source_kind="time_pair",
            source_sha256=fresh_pair.time_pair_sha256,
            field_path="relation_window",
        ),
        DynamicFieldReference(
            source_kind="clock_resolution",
            source_sha256=resolution.resolution_sha256,
            field_path="clock_uncertainty_ns",
        ),
        DynamicFieldReference(
            source_kind="clock_resolution",
            source_sha256=resolution.resolution_sha256,
            field_path="resolution_status",
        ),
    )
    return _build_expression_leaf(
        kind="directional_time",
        truth_value=truth,
        field_references=refs,
        support_sources=(*source[2], *target[2]) if truth == "true" else (),
        pending_demand_keys=pending_demand_keys if source[4] or target[4] else (),
        reason_codes=tuple(reasons),
    )


def _prepare_dynamic_measurements(
    *,
    source: RuleHypothesisEnvelope,
    sources: VerificationSourceCatalog,
    directory: DynamicRequestCatalog,
    measurements: tuple[SignalEpisodeMeasurement, ...],
    expected_measurement_sha256s: tuple[str, ...],
) -> tuple[SignalEpisodeMeasurement, ...]:
    """Fresh prefix shape, pins and joins; never raw-block authority."""
    roots = (
        RuleHypothesisEnvelope,
        VerificationSourceCatalog,
        DynamicRequestCatalog,
        SignalEpisodeMeasurement,
    )
    originals = (source, sources, directory, measurements, expected_measurement_sha256s)
    before = snapshot(originals, roots)
    try:
        with ExitStack() as guards:
            if (
                type(measurements) is not tuple
                or type(expected_measurement_sha256s) is not tuple
                or len(measurements) != len(expected_measurement_sha256s)
                or any(type(row) is not SignalEpisodeMeasurement for row in measurements)
            ):
                raise ValueError(
                    "dynamic evaluation requires the exact complete measurement prefix"
                )
            for measured, pin in zip(measurements, expected_measurement_sha256s, strict=True):
                if measured.measurement_sha256 != require_pin(pin):
                    raise ValueError("dynamic evaluation independent measurement pin mismatch")
            results = tuple(
                sorted(
                    (
                        SignalEpisodeMeasurement.model_validate_json(canonical(row, roots))
                        for row in measurements
                    ),
                    key=lambda row: row.request_sha256,
                )
            )
            guards.callback(require_unchanged, results, roots, snapshot(results, roots))
            _check_condition_sources(source, sources, directory.requests, results)
            return results
    finally:
        require_unchanged(originals, roots, before)


def _compile_dynamic_evaluation_values(
    *,
    source: RuleHypothesisEnvelope,
    binding: B0RuleBinding,
    sources: VerificationSourceCatalog,
    directory: DynamicRequestCatalog,
    resolution: FrozenClockResolution,
    results: tuple[SignalEpisodeMeasurement, ...],
    expected_rule_set_sha256: str,
    expected_original_binding_sha256: str,
    expected_catalog_sha256: str,
    expected_policy_sha256: str,
    expected_request_catalog_sha256: str,
) -> DynamicPredicateEvaluation:
    """Compile this operation's prepared values without source or head authority."""
    from .graph_time import _graph_time_comparison_values

    with ExitStack() as guards:
        nodes: dict[str, DynamicExpressionNode] = {}
        times = []
        local_time_nodes = {}
        time_demands = {
            (row.time_pair_key, row.time_side): row
            for row in directory.demands
            if row.time_pair_key is not None
        }
        for pair in directory.time_pairs:
            sides = []
            side_bounds: list[_TimeSideBounds] = []
            for side_name in ("source", "target"):
                side: DynamicTimeSide | None
                unavailable_reasons: tuple[str, ...] = ()
                if type(pair) is DynamicLocalTimePair:
                    side = pair.source if side_name == "source" else pair.target
                elif type(pair) is DynamicGraphTimePair:
                    graph_binding = pair.source if side_name == "source" else pair.target
                    side = graph_binding.domain
                    unavailable_reasons = graph_binding.reason_codes
                else:
                    raise ValueError("dynamic evaluation received an undeclared time pair")
                bounds = (
                    _time_side_bounds(
                        side=side,
                        window=pair.relation_window,
                        envelope=source,
                        catalog=sources,
                        requests=directory.requests,
                        measurements=results,
                    )
                    if side is not None
                    else (None, (), (), unavailable_reasons, False)
                )
                # Keep the first side guarded even if computing the second raises.
                guards.callback(
                    require_unchanged, bounds, _BOUND_ROOTS, snapshot(bounds, _BOUND_ROOTS)
                )
                side_bounds.append(bounds)
                sides.append(
                    DynamicTimeSideEvaluation(
                        demand_key=time_demands[(pair.time_pair_key, side_name)].demand_key,
                        onset_bounds=bounds[0],
                        field_references=bounds[1],
                        reason_codes=bounds[3],
                        measurement_pending=bounds[4],
                    )
                )
            pending = tuple(sorted(row.demand_key for row in sides if row.measurement_pending))
            if type(pair) is DynamicLocalTimePair:
                time_node = _local_time_comparison_values(
                    fresh_pair=pair,
                    resolution=resolution,
                    source=side_bounds[0],
                    target=side_bounds[1],
                    pending_demand_keys=pending,
                )
                local_time_nodes[(pair.seed_hypothesis_id, pair.seed_predicate_id)] = time_node
            elif type(pair) is DynamicGraphTimePair:
                time_node = _graph_time_comparison_values(
                    fresh=pair,
                    resolution=resolution,
                    left=side_bounds[0],
                    right=side_bounds[1],
                    pending_demand_keys=pending,
                )
            else:
                raise ValueError("dynamic evaluation received an undeclared time pair")
            nodes[time_node.expression_id] = time_node
            times.append(
                DynamicTimeEvaluation(
                    time_pair_key=pair.time_pair_key,
                    expression_id=time_node.expression_id,
                    source=sides[0],
                    target=sides[1],
                )
            )

        conditions = []
        seed_rows = []
        by_predicate = {
            row.predicate_result_id: row for row in source.rule_hypothesis_set.predicates
        }
        measured_keys = {"request_" + row.request_sha256 for row in results}
        for seed in sorted(
            source.rule_hypothesis_set.hypotheses, key=lambda row: row.hypothesis_id
        ):
            ids: dict[str, list[str]] = {"necessary": [], "optional": [], "refuting": []}
            for predicate_id in sorted(
                seed.necessary_predicate_result_ids
                + seed.optional_predicate_result_ids
                + seed.refuting_predicate_result_ids
            ):
                predicate = by_predicate[predicate_id]
                demand = next(
                    row
                    for row in directory.demands
                    if row.time_pair_key is None
                    and seed.hypothesis_id in row.seed_ids
                    and predicate_id in row.source_predicate_ids
                )
                condition_nodes = _compile_bound_condition_values(
                    source=source,
                    binding=binding,
                    sources=sources,
                    frozen_requests=directory.requests,
                    results=results,
                    seed_hypothesis_id=seed.hypothesis_id,
                    seed_predicate_id=predicate_id,
                    pending_demand_keys=(demand.demand_key,)
                    if not demand.request_keys or set(demand.request_keys) - measured_keys
                    else (),
                )
                for node in condition_nodes:
                    nodes[node.expression_id] = node
                active = local_time_nodes.get(
                    (seed.hypothesis_id, predicate_id), condition_nodes[-1]
                )
                conditions.append(
                    DynamicConditionEvaluation(
                        seed_hypothesis_id=seed.hypothesis_id,
                        seed_predicate_id=predicate_id,
                        requirement=predicate.requirement,
                        demand_key=demand.demand_key,
                        original_expression_id=condition_nodes[0].expression_id,
                        expression_id=active.expression_id,
                    )
                )
                ids[predicate.requirement].append(active.expression_id)
            necessary_ids = tuple(sorted(set(ids["necessary"])))
            necessary = compose_dynamic_expressions(
                operator="and", operands=tuple(nodes[key] for key in necessary_ids)
            )
            nodes[necessary.expression_id] = necessary
            seed_rows.append(
                DynamicSeedEvaluation(
                    seed_hypothesis_id=seed.hypothesis_id,
                    candidate_service_key=seed.candidate_service_key,
                    mechanism_key=seed.mechanism_key,
                    necessary_expression_ids=necessary_ids,
                    optional_expression_ids=tuple(sorted(set(ids["optional"]))),
                    refuting_expression_ids=tuple(sorted(set(ids["refuting"]))),
                    necessary_expression_id=necessary.expression_id,
                )
            )
        result_by_request = {"request_" + row.request_sha256: row for row in results}
        consumption = []
        for request in directory.requests:
            measurement = result_by_request.get(request.canonical_request_key)
            digest = measurement.measurement_sha256 if measurement is not None else None
            consumers = tuple(
                sorted(
                    node.expression_id
                    for node in nodes.values()
                    if any(
                        ref.source_kind == "measurement" and ref.source_sha256 == digest
                        for ref in node.field_references
                    )
                )
            )
            consumption.append(
                DynamicRequestConsumption(
                    request_key=request.canonical_request_key,
                    measurement_sha256=digest,
                    expression_ids=consumers,
                    reason_codes=(
                        "measurement_referenced"
                        if consumers
                        else "measurement_not_consumed"
                        if measurement is not None
                        else "not_submitted"
                        if request.availability == "available"
                        else "request_unavailable",
                    ),
                )
            )
        draft = DynamicPredicateEvaluation(
            opaque_case_id=directory.opaque_case_id,
            candidate_freeze_sha256=directory.candidate_freeze_sha256,
            source_rule_set_sha256=expected_rule_set_sha256,
            original_binding_sha256=expected_original_binding_sha256,
            time_contract_sha256=directory.time_contract_sha256,
            catalog_sha256=expected_catalog_sha256,
            policy_sha256=expected_policy_sha256,
            request_catalog_sha256=expected_request_catalog_sha256,
            clock_resolution=resolution,
            original_predicates=binding.predicates,
            expression_nodes=tuple(nodes.values()),
            conditions=tuple(conditions),
            seed_evaluations=tuple(seed_rows),
            time_evaluations=tuple(times),
            request_consumption=tuple(consumption),
            pending_demand_keys=tuple(
                sorted({key for node in nodes.values() for key in node.pending_demand_keys})
            ),
            evaluation_sha256="0" * 64,
        )
        document = cast(dict[str, object], plain(draft))
        document.pop("evaluation_sha256")
        result = draft.model_copy(
            update={"evaluation_sha256": sha256_hex(canonical_json_bytes(document))}
        )
        if not result.verify():
            raise ValueError("dynamic evaluation construction integrity failed")
        return result


def evaluate_dynamic_predicates(
    *,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    original_binding: B0RuleBinding,
    expected_original_binding_sha256: str,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    policy: DynamicVerificationPolicy,
    expected_policy_sha256: str,
    request_catalog: DynamicRequestCatalog,
    expected_request_catalog_sha256: str,
    measurements: tuple[SignalEpisodeMeasurement, ...],
    expected_measurement_sha256s: tuple[str, ...],
    clock_resolution: FrozenClockResolution,
    expected_clock_resolution_sha256: str,
) -> DynamicPredicateEvaluation:
    """Evaluate all frozen seeds from only the submitted measurement prefix.

    Pins are supplied by the enclosing trusted session, never inferred from an
    output being verified. Graph sources are inside the independently pinned
    pre-query directory; the session must additionally rebuild that directory
    from B0DynamicInputs. This pure layer has no store or execution authority.
    Measurement shape/hash and joins do not replace A4 raw-block replay.
    """
    roots = (
        RuleHypothesisEnvelope,
        B0RuleBinding,
        CaseTimeContract,
        VerificationSourceCatalog,
        DynamicVerificationPolicy,
        DynamicRequestCatalog,
        SignalEpisodeMeasurement,
        FrozenClockResolution,
    )
    inputs = (
        envelope,
        expected_rule_set_sha256,
        original_binding,
        expected_original_binding_sha256,
        time_contract,
        catalog,
        expected_catalog_sha256,
        policy,
        expected_policy_sha256,
        request_catalog,
        expected_request_catalog_sha256,
        measurements,
        expected_measurement_sha256s,
        clock_resolution,
        expected_clock_resolution_sha256,
    )
    before = snapshot(inputs, roots)
    try:
        with ExitStack() as guards:
            for value, kind in zip(
                (
                    envelope,
                    original_binding,
                    time_contract,
                    catalog,
                    policy,
                    request_catalog,
                    clock_resolution,
                ),
                (
                    RuleHypothesisEnvelope,
                    B0RuleBinding,
                    CaseTimeContract,
                    VerificationSourceCatalog,
                    DynamicVerificationPolicy,
                    DynamicRequestCatalog,
                    FrozenClockResolution,
                ),
                strict=True,
            ):
                if type(value) is not kind:
                    raise ValueError("dynamic evaluation requires exact complete frozen roots")
            for pin in (
                expected_rule_set_sha256,
                expected_original_binding_sha256,
                expected_catalog_sha256,
                expected_policy_sha256,
                expected_request_catalog_sha256,
                expected_clock_resolution_sha256,
            ):
                require_pin(pin)
            if (
                type(measurements) is not tuple
                or type(expected_measurement_sha256s) is not tuple
                or len(measurements) != len(expected_measurement_sha256s)
                or any(type(row) is not SignalEpisodeMeasurement for row in measurements)
            ):
                raise ValueError(
                    "dynamic evaluation requires the exact complete measurement prefix"
                )
            for measured, pin in zip(measurements, expected_measurement_sha256s, strict=True):
                if measured.measurement_sha256 != require_pin(pin):
                    raise ValueError("dynamic evaluation independent measurement pin mismatch")
            source = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, roots))
            guards.callback(require_unchanged, source, roots, snapshot(source, roots))
            binding = require_b0_rule_binding(
                envelope=source,
                expected_rule_set_sha256=expected_rule_set_sha256,
                binding=original_binding,
                expected_binding_sha256=expected_original_binding_sha256,
            )
            guards.callback(require_unchanged, binding, roots, snapshot(binding, roots))
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, roots))
            sources = VerificationSourceCatalog.model_validate_json(canonical(catalog, roots))
            strategy = DynamicVerificationPolicy.model_validate_json(canonical(policy, roots))
            submitted = DynamicRequestCatalog.model_validate_json(canonical(request_catalog, roots))
            working = (contract, sources, strategy, submitted)
            guards.callback(require_unchanged, working, roots, snapshot(working, roots))
            if (
                not submitted.verify()
                or submitted.request_catalog_sha256 != expected_request_catalog_sha256
            ):
                raise ValueError("dynamic evaluation independent request directory pin mismatch")
            directory = build_dynamic_request_catalog(
                envelope=source,
                expected_rule_set_sha256=expected_rule_set_sha256,
                time_contract=contract,
                catalog=sources,
                expected_catalog_sha256=expected_catalog_sha256,
                policy=strategy,
                expected_policy_sha256=expected_policy_sha256,
                graph_snapshots=submitted.graph_snapshots,
                expected_graph_snapshot_sha256s=tuple(
                    row.snapshot_sha256 for row in submitted.graph_snapshots
                ),
            )
            guards.callback(require_unchanged, directory, roots, snapshot(directory, roots))
            if canonical(directory, roots) != canonical(submitted, roots):
                raise ValueError(
                    "dynamic evaluation directory differs from complete frozen reconstruction"
                )
            resolution = require_frozen_clock_resolution(
                clock_resolution=clock_resolution,
                time_contract=contract,
                expected_clock_resolution_sha256=expected_clock_resolution_sha256,
            )
            guards.callback(require_unchanged, resolution, roots, snapshot(resolution, roots))
            results = _prepare_dynamic_measurements(
                source=source,
                sources=sources,
                directory=directory,
                measurements=measurements,
                expected_measurement_sha256s=expected_measurement_sha256s,
            )
            guards.callback(require_unchanged, results, roots, snapshot(results, roots))
            return _compile_dynamic_evaluation_values(
                source=source,
                binding=binding,
                sources=sources,
                directory=directory,
                resolution=resolution,
                results=results,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_original_binding_sha256=expected_original_binding_sha256,
                expected_catalog_sha256=expected_catalog_sha256,
                expected_policy_sha256=expected_policy_sha256,
                expected_request_catalog_sha256=expected_request_catalog_sha256,
            )
    finally:
        require_unchanged(inputs, roots, before)
