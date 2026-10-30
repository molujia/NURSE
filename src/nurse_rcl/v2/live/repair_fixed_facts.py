"""Source-replayed fixed B/F facts, separate from paid episode verification.

These leaves do not authorize a mechanism, role or score. The enclosing repair
scorer must retain the complete case, replay raw files and combine every
applicable condition. Error-record pins must come from that independent file
projection, not from a submitted result. No population or dependency is inferred
from observed status fractions. No episode or onset is computed in this module.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Final, Literal, TypeVar, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse, require_frozen_candidate_universe
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, TruthValue, _SourceModel
from nurse_rcl.v2.dynamic_verification.safety import (
    canonical,
    plain,
    require_pin,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.dynamic_verification.source_store import SourceBlockStore
from nurse_rcl.v2.hypothesis_guard import (
    _PREDICATE_EVIDENCE_CONTRACT_SHA256,
    _PREDICATE_EVIDENCE_CONTRACTS,
)
from nurse_rcl.v2.live.repair_errors import (
    ERROR_SEMANTICS_SHA256,
    ErrorObservation,
    Kind,
    LogErrorRecord,
    Record,
    TraceErrorRecord,
    _summarize,
)
from nurse_rcl.v2.live.repair_lifecycle import _AVAILABILITY_MEANINGS, LIFECYCLE_SEMANTICS_SHA256
from nurse_rcl.v2.live.repair_measurements import _NORMAL_PREDICATES, _UP_PREDICATES, _source_replay
from nurse_rcl.v2.live.repair_observations import REPAIR_QUALITY_SEMANTICS_SHA256
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.observation_schemas import CandidateUniverse, CaseTimeContract, TimeInterval

REPAIR_FIXED_SIGNAL_SCHEMA_VERSION: Final = "nurse.v2.live-repair-fixed-signal-facts/1"
REPAIR_FIXED_ERROR_SCHEMA_VERSION: Final = "nurse.v2.live-repair-fixed-error-facts/1"
REPAIR_FIXED_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-fixed-facts/1",
            quality=REPAIR_QUALITY_SEMANTICS_SHA256,
            lifecycle=LIFECYCLE_SEMANTICS_SHA256,
            errors=ERROR_SEMANTICS_SHA256,
            applicability=_PREDICATE_EVIDENCE_CONTRACT_SHA256,
            summary="general_deviation_only_never_pressure_persistence_or_normality",
            event="registered_observed_zero_existence_not_duration",
            error_increase="strict_fraction_increase_only_two_nonempty_fully_known_record_groups",
            request_failure="server_spans_only_not_logs_unknown_kind_or_request_population",
            dependency="no_remote_identity_in_this_projection_no_dependency_authority",
            normal="fixed_summary_or_absent_errors_never_prove_health",
            numeric_weight="none_full_mechanism_and_score_compiler_required",
            query_cost=0,
        )
    )
)


class RepairFixedFact(_SourceModel):
    predicate_kind: str
    measurement_field: str
    truth_value: TruthValue
    reason_codes: tuple[str, ...]


class RepairFixedSignalFacts(_SourceModel):
    schema_version: Literal["nurse.v2.live-repair-fixed-signal-facts/1"] = (
        REPAIR_FIXED_SIGNAL_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    source_sha256: SHA256
    source_summary_sha256: SHA256
    candidate_service_key: str | None
    observation_id: str
    support_source_key: str
    interval: TimeInterval
    event_timestamps_ns: tuple[int, ...]
    facts: tuple[RepairFixedFact, ...]
    dynamic_queries_used: Literal[0] = 0
    score_authority: Literal[False] = False
    result_sha256: SHA256


class RepairFixedErrorFacts(_SourceModel):
    schema_version: Literal["nurse.v2.live-repair-fixed-error-facts/1"] = (
        REPAIR_FIXED_ERROR_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    dataset: Literal["rcabench", "aiops25"]
    modality: Literal["trace", "log"]
    span_kind: Kind
    observed_entity_key: str | None
    candidate_service_key: str | None
    observation_sha256s: tuple[str, str]
    source_group_ids: tuple[str, ...]
    support_source_key: str
    interval: TimeInterval
    statistic: Literal["observed_span_status_fraction", "observed_log_severity_fraction"]
    severity_basis: Literal["not_applicable", "exported_severity", "preprocessor_message_heuristic"]
    baseline_known_denominator: int = Field(ge=0)
    fault_known_denominator: int = Field(ge=0)
    baseline_fraction: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fault_fraction: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    observed_error_count: int = Field(ge=0)
    sampling_coverage: Literal["unknown"] = "unknown"
    facts: tuple[RepairFixedFact, ...]
    limitations: tuple[str, ...]
    dynamic_queries_used: Literal[0] = 0
    score_authority: Literal[False] = False
    result_sha256: SHA256


class RepairErrorFactGroup(_SourceModel):
    """Transient numeric view, never a substitute for complete source replay."""

    entity_key: str | None
    span_kind: Kind
    evidence_group_id: str
    observed_count: int = Field(ge=0)
    known_denominator: int = Field(ge=0)
    error_count: int = Field(ge=0)
    error_fraction: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class RepairErrorFactSummary(_SourceModel):
    dataset: Literal["rcabench", "aiops25"]
    modality: Literal["trace", "log"]
    interval: TimeInterval
    result_sha256: SHA256
    groups: tuple[RepairErrorFactGroup, ...]
    severity_basis: Literal["not_applicable", "exported_severity", "preprocessor_message_heuristic"]
    limitations: tuple[str, ...]


_ROOTS = (
    RepairFixedSignalFacts,
    RepairFixedErrorFacts,
    RepairFixedFact,
    RepairMeasurementSource,
    CaseTimeContract,
    CandidateUniverse,
    FrozenCandidateUniverse,
    ErrorObservation,
    TraceErrorRecord,
    LogErrorRecord,
)
_T = TypeVar("_T", RepairFixedSignalFacts, RepairFixedErrorFacts)


def _seal(draft: _T) -> _T:
    values = cast(dict[str, object], plain(draft))
    values.pop("result_sha256")
    return draft.model_copy(update={"result_sha256": sha256_hex(canonical_json_bytes(values))})


def _fact(kind: str, field: str, truth: TruthValue, *reasons: str) -> RepairFixedFact:
    return RepairFixedFact(
        predicate_kind=kind,
        measurement_field=field,
        truth_value=truth,
        reason_codes=tuple(sorted(set((*reasons, "fixed_fact:" + truth)))),
    )


def _signal_values(source: RepairMeasurementSource) -> RepairFixedSignalFacts:
    """Pure projection inside the independently pinned scalar-source replay."""
    quality, event, observation = source.quality, source.lifecycle, source.observation
    abnormal = (
        quality.formal_scoring_eligible
        and quality.robust_z is not None
        and quality.robust_z >= quality.source_summary.configuration.abnormal_z_threshold
    )
    facts = []
    for kind, contract in sorted(_PREDICATE_EVIDENCE_CONTRACTS.items()):
        if observation.signal_family not in (
            *contract.primary_signal_families,
            *contract.source_signal_families,
        ):
            continue
        truth: TruthValue = "unknown"
        field = "unavailable"
        reasons = ["fixed_fact_semantics_unavailable"]
        if kind == "local_anomaly_observed":
            field = "quality.robust_z_or_lifecycle.formal_event_eligible"
            if abnormal or (event is not None and event.formal_event_eligible):
                truth, reasons = "true", ["observed_general_deviation_not_causality"]
            else:
                reasons = ["whole_window_summary_does_not_exclude_transient_deviation"]
        elif kind in _UP_PREDICATES:
            reasons = ["reserved_dynamic_measurement_required"]
        elif kind in _NORMAL_PREDICATES:
            if abnormal or (event is not None and event.formal_event_eligible):
                truth, field, reasons = "false", "observed_deviation", ["deviation_is_not_health"]
            else:
                reasons = ["normality_requires_authorized_point_verification"]
        elif kind in {"workload_unavailability_observed", "workload_availability_verified_normal"}:
            field = "lifecycle.availability_support"
            if event is None or event.state_meaning not in _AVAILABILITY_MEANINGS:
                reasons = ["availability_semantics_unavailable"]
            elif event.availability_support:
                truth = "true" if kind == "workload_unavailability_observed" else "false"
                reasons = ["observed_entity_zero_not_continuous_service_outage"]
            else:
                reasons = ["absence_of_zero_is_not_verified_health"]
        elif kind == "lifecycle_transition_observed":
            field = "lifecycle.events"
            if (
                event
                and event.formal_event_eligible
                and any(
                    e.onset_bounds_ns is not None or e.recovery_bounds_ns is not None
                    for e in event.events
                )
            ):
                truth, reasons = "true", ["source_bracketed_state_transition"]
            else:
                reasons = ["observed_transition_bracket_unavailable"]
        if source.candidate_service_key is None:
            truth, reasons = "unknown", ["candidate_binding_unavailable"]
        facts.append(_fact(kind, field, truth, *reasons))
    support = "support_" + sha256_hex(
        canonical_json_bytes(
            dict(
                candidate_service_key=source.candidate_service_key,
                source_fingerprint=observation.source_fingerprint,
                adapter_semantics_sha256=observation.adapter_semantics_sha256,
            )
        )
    )
    return _seal(
        RepairFixedSignalFacts(
            semantics_sha256=REPAIR_FIXED_SEMANTICS_SHA256,
            source_sha256=source.source_sha256,
            source_summary_sha256=quality.result_sha256,
            candidate_service_key=source.candidate_service_key,
            observation_id=observation.observation_id,
            support_source_key=support,
            interval=source.query_interval,
            event_timestamps_ns=tuple(
                sorted({t for e in event.events for t in e.zero_timestamps_ns})
            )
            if event
            else (),
            facts=tuple(facts),
            result_sha256="0" * 64,
        )
    )


def build_repair_fixed_signal_facts(
    *,
    source: RepairMeasurementSource,
    expected_source_sha256: str,
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    store: SourceBlockStore,
) -> RepairFixedSignalFacts:
    """Fixed facts remain available even with zero remaining dynamic queries."""
    with _source_replay(
        source=source,
        expected_source_sha256=expected_source_sha256,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        store=store,
    ) as fresh:
        return _signal_values(fresh)


def _error_values(
    baseline: ErrorObservation | RepairErrorFactSummary,
    fault: ErrorObservation | RepairErrorFactSummary,
    candidates: tuple[str, ...],
) -> tuple[RepairFixedErrorFacts, ...]:
    """Compare the same entity/kind, without substituting a missing denominator."""
    before = {(g.entity_key, g.span_kind): g for g in baseline.groups}
    after = {(g.entity_key, g.span_kind): g for g in fault.groups}
    rows = []
    for entity, kind in sorted(set(before) | set(after), key=lambda v: (v[0] or "", v[1])):
        b, f = before.get((entity, kind)), after.get((entity, kind))
        bound = entity if entity in candidates else None
        increase: TruthValue = "unknown"
        reason = "complete_known_baseline_and_fault_denominators_required"
        # Use integer cross products, avoiding fraction rounding as a truth rule.
        if all(
            g is not None and g.known_denominator > 0 and g.known_denominator == g.observed_count
            for g in (b, f)
        ):
            assert b is not None and f is not None
            increase = (
                "true"
                if (f.error_count * b.known_denominator > b.error_count * f.known_denominator)
                else "false"
            )
            reason = "comparison_of_observed_status_fractions_not_population_rate"
        observed: TruthValue = "true" if f is not None and f.error_count > 0 else "unknown"
        facts = [
            _fact(
                "local_anomaly_observed",
                "fault.error_count",
                observed,
                "observed_error_status_not_causality"
                if observed == "true"
                else "absent_errors_do_not_prove_health",
            ),
            _fact("local_error_increase", "fault_vs_baseline.error_fraction", increase, reason),
            _fact(
                "local_signals_verified_normal",
                "fault.error_count",
                "false" if observed == "true" else "unknown",
                "no_population_health_claim",
            ),
        ]
        request_truth = increase
        request_reason = reason
        if fault.modality == "log":
            request_truth, request_reason = "unknown", "log_severity_is_not_request_outcome"
        elif kind != "server":
            request_truth, request_reason = "unknown", "server_request_kind_required"
        facts.append(
            _fact(
                "request_failure_increase",
                "observed_server_status_fraction",
                request_truth,
                request_reason,
            )
        )
        if bound is None:
            facts = [
                _fact(
                    v.predicate_kind,
                    v.measurement_field,
                    "unknown",
                    "candidate_binding_unavailable",
                )
                for v in facts
            ]
        groups = tuple(sorted(g.evidence_group_id for g in (b, f) if g is not None))
        rows.append(
            _seal(
                RepairFixedErrorFacts(
                    semantics_sha256=REPAIR_FIXED_SEMANTICS_SHA256,
                    dataset=fault.dataset,
                    modality=fault.modality,
                    span_kind=kind,
                    observed_entity_key=entity,
                    candidate_service_key=bound,
                    observation_sha256s=(baseline.result_sha256, fault.result_sha256),
                    source_group_ids=groups,
                    support_source_key="support_"
                    + sha256_hex(
                        canonical_json_bytes([fault.dataset, fault.modality, entity, kind, groups])
                    ),
                    interval=fault.interval,
                    statistic="observed_span_status_fraction"
                    if fault.modality == "trace"
                    else "observed_log_severity_fraction",
                    severity_basis=fault.severity_basis,
                    baseline_known_denominator=b.known_denominator if b else 0,
                    fault_known_denominator=f.known_denominator if f else 0,
                    baseline_fraction=b.error_fraction if b else None,
                    fault_fraction=f.error_fraction if f else None,
                    observed_error_count=f.error_count if f else 0,
                    facts=tuple(sorted(facts, key=lambda v: v.predicate_kind)),
                    limitations=tuple(sorted(set(baseline.limitations) | set(fault.limitations))),
                    result_sha256="0" * 64,
                )
            )
        )
    return tuple(rows)


def build_repair_fixed_error_facts(
    *,
    observations: tuple[ErrorObservation, ...],
    expected_observation_sha256s: tuple[str, ...],
    source_records: tuple[tuple[Record, ...], ...],
    dataset: Literal["rcabench", "aiops25"],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
) -> tuple[RepairFixedErrorFacts, ...]:
    """Replay all four B/F modality projections; raw-file permission stays outside."""
    context = (
        dataset,
        time_contract,
        candidate_universe,
        frozen_candidates,
        expected_time_contract_sha256,
        expected_observation_sha256s,
    )
    saved = snapshot(context, _ROOTS)
    with ExitStack() as guards:
        guards.callback(require_unchanged, context, _ROOTS, saved)
        if any(
            type(v) is not tuple or len(v) != 4
            for v in (observations, expected_observation_sha256s, source_records)
        ):
            raise ValueError("fixed errors require complete four B/F modality tuples")
        for observation, records in zip(observations, source_records, strict=True):
            guards.callback(
                require_unchanged,
                (observation, records),
                _ROOTS,
                snapshot((observation, records), _ROOTS),
            )
        require_pin(expected_time_contract_sha256)
        if sha256_hex(canonical(time_contract, _ROOTS)) != expected_time_contract_sha256:
            raise ValueError("fixed error time pin mismatch")
        contract = CaseTimeContract.model_validate_json(canonical(time_contract, _ROOTS))
        universe = CandidateUniverse.model_validate_json(canonical(candidate_universe, _ROOTS))
        frozen = FrozenCandidateUniverse.model_validate_json(canonical(frozen_candidates, _ROOTS))
        working = (contract, universe, frozen)
        guards.callback(require_unchanged, working, _ROOTS, snapshot(working, _ROOTS))
        require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
        if frozen.opaque_case_id != contract.opaque_case_id or dataset not in {
            "rcabench",
            "aiops25",
        }:
            raise ValueError("fixed error case/dataset mismatch")
        slots: dict[tuple[str, str | None], ErrorObservation] = {}
        expected_slots = {(m, w) for m in ("trace", "log") for w in ("baseline", "fault")}
        for observation, pin, records in zip(
            observations, expected_observation_sha256s, source_records, strict=True
        ):
            if type(observation) is not ErrorObservation or type(records) is not tuple:
                raise ValueError("fixed error sources require exact types")
            require_pin(pin)
            fresh = ErrorObservation.model_validate_json(canonical(observation, _ROOTS))
            guards.callback(require_unchanged, fresh, _ROOTS, snapshot(fresh, _ROOTS))
            if fresh.dataset != dataset:
                raise ValueError("fixed error dataset changed")
            window = (
                "baseline"
                if fresh.interval == contract.baseline_window
                else "fault"
                if fresh.interval == contract.fault_window
                else None
            )
            slot = (fresh.modality, window)
            if slot not in expected_slots or slot in slots:
                raise ValueError("fixed error window/modality partition mismatch")
            rebuilt = _summarize(
                records=records, dataset=dataset, interval=fresh.interval, modality=fresh.modality
            )
            guards.callback(require_unchanged, rebuilt, _ROOTS, snapshot(rebuilt, _ROOTS))
            if rebuilt.result_sha256 != pin or canonical(rebuilt, _ROOTS) != canonical(
                fresh, _ROOTS
            ):
                raise ValueError("fixed error differs from independent source-record replay")
            slots[slot] = rebuilt
        if set(slots) != expected_slots:
            raise ValueError("fixed error B/F projections incomplete")
        return tuple(
            row
            for modality in ("log", "trace")
            for row in _error_values(
                slots[(modality, "baseline")],
                slots[(modality, "fault")],
                frozen.candidate_service_keys,
            )
        )
