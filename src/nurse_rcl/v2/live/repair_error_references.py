"""Complete single-copy error sources with replay-only derived references.

One atomic call admits all four raw projections under one finite case budget.
Members and groups are rebuilt, not trusted from cached summaries. Independent
observation pins must be supplied by the enclosing raw-file/adapter boundary;
a self-consistent bundle digest does not establish actual source authority.
Legacy observation and fixed-leaf meanings are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Final, Literal, cast

from pydantic import model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse, require_frozen_candidate_universe
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.live.repair_errors import (
    ERROR_SEMANTICS_SHA256,
    ErrorObservation,
    LogErrorRecord,
    Record,
    TraceErrorRecord,
    _summarize,
)
from nurse_rcl.v2.live.repair_fixed_facts import (
    REPAIR_FIXED_SEMANTICS_SHA256,
    RepairErrorFactGroup,
    RepairErrorFactSummary,
    RepairFixedErrorFacts,
    _error_values,
)
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_CAPACITY_SHA256,
    RepairModel,
    canonical,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.observation_schemas import CandidateUniverse, CaseTimeContract, TimeInterval

REPAIR_ERROR_SOURCES_SCHEMA_VERSION: Final = "nurse.v2.live-repair-error-sources/1"
REPAIR_REFERENCED_ERROR_FACTS_SCHEMA_VERSION: Final = (
    "nurse.v2.live-repair-referenced-error-facts/1"
)
REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-error-references/1",
            errors=ERROR_SEMANTICS_SHA256,
            facts=REPAIR_FIXED_SEMANTICS_SHA256,
            capacity=REPAIR_CAPACITY_SHA256,
            representation="four_complete_BF_trace_log_record_tables_no_derived_member_copies",
            replay="every_record_every_member_every_group_against_independent_observation_pins",
            exit_guards="original_and_fresh_whole_context_and_transient_expansion_both_exits",
            authority="no_score_or_raw_file_authority_no_cross_call_cache",
        )
    )
)
_SLOTS = (("baseline", "trace"), ("baseline", "log"), ("fault", "trace"), ("fault", "log"))


class RepairErrorProjection(RepairModel):
    window: Literal["baseline", "fault"]
    modality: Literal["trace", "log"]
    interval: TimeInterval
    records: tuple[TraceErrorRecord | LogErrorRecord, ...]


class RepairErrorSources(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-error-sources/1"] = (
        REPAIR_ERROR_SOURCES_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    projections: tuple[RepairErrorProjection, ...]
    result_sha256: SHA256

    @model_validator(mode="after")
    def require_partition(self) -> Self:
        if self.semantics_sha256 != REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256:
            raise ValueError("error reference semantics mismatch")
        if tuple((p.window, p.modality) for p in self.projections) != _SLOTS:
            raise ValueError("error references require complete ordered four-slot partition")
        for part in self.projections:
            record_type = TraceErrorRecord if part.modality == "trace" else LogErrorRecord
            if any(type(r) is not record_type for r in part.records):
                raise ValueError("error reference modality/type mismatch")
            if any(
                not part.interval.start_unix_ns <= r.timestamp_unix_ns < part.interval.end_unix_ns
                for r in part.records
            ):
                raise ValueError("error reference record outside window")
        return self


class RepairReferencedErrorFacts(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-referenced-error-facts/1"] = (
        REPAIR_REFERENCED_ERROR_FACTS_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    source_bundle_sha256: SHA256
    observation_sha256s: tuple[str, ...]
    facts: tuple[RepairFixedErrorFacts, ...]
    dynamic_queries_used: Literal[0] = 0
    score_authority: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (
    RepairErrorSources,
    RepairErrorProjection,
    RepairReferencedErrorFacts,
    RepairErrorFactSummary,
    RepairErrorFactGroup,
    RepairFixedErrorFacts,
    CaseTimeContract,
    CandidateUniverse,
    FrozenCandidateUniverse,
    ErrorObservation,
    TraceErrorRecord,
    LogErrorRecord,
)


def _digest_unsigned(value: object) -> str:
    document = cast(dict[str, object], plain(value))
    document.pop("result_sha256")
    return sha256_hex(canonical_json_bytes(document))


def build_repair_error_sources(
    *,
    source_records: tuple[tuple[Record, ...], ...],
    dataset: Literal["rcabench", "aiops25"],
    time_contract: CaseTimeContract,
) -> RepairErrorSources:
    """Representation producer only; consumers independently replay all sources."""
    context = (source_records, dataset, time_contract)
    saved = snapshot(context, _ROOTS)
    with ExitStack() as guards:
        guards.callback(require_unchanged, context, _ROOTS, saved)
        return _error_source_values(
            source_records=source_records, dataset=dataset, time_contract=time_contract,
        )


def _error_source_values(
    *, source_records: tuple[tuple[Record, ...], ...],
    dataset: Literal["rcabench", "aiops25"], time_contract: CaseTimeContract,
) -> RepairErrorSources:
    """Shared four-slot representation, without a lifetime mutation monitor."""
    if type(source_records) is not tuple or len(source_records) != 4:
        raise ValueError("error sources require all four record tuples")
    parts = tuple(
        RepairErrorProjection(
            window=cast(Literal["baseline", "fault"], window),
            modality=cast(Literal["trace", "log"], modality),
            interval=time_contract.baseline_window
            if window == "baseline" else time_contract.fault_window,
            records=records,
        )
        for (window, modality), records in zip(_SLOTS, source_records, strict=True)
    )
    draft = RepairErrorSources(
        semantics_sha256=REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256, dataset=dataset,
        opaque_case_id=time_contract.opaque_case_id, projections=parts, result_sha256="0" * 64,
    )
    return draft.model_copy(update={"result_sha256": _digest_unsigned(draft)})


def _summary_values(observation: ErrorObservation) -> RepairErrorFactSummary:
    return RepairErrorFactSummary(
        dataset=observation.dataset,
        modality=observation.modality,
        interval=observation.interval,
        result_sha256=observation.result_sha256,
        severity_basis=observation.severity_basis,
        limitations=observation.limitations,
        groups=tuple(
            RepairErrorFactGroup(
                entity_key=g.entity_key,
                span_kind=g.span_kind,
                evidence_group_id=g.evidence_group_id,
                observed_count=g.observed_count,
                known_denominator=g.known_denominator,
                error_count=g.error_count,
                error_fraction=g.error_fraction,
            )
            for g in observation.groups
        ),
    )


def _replay_projection(
    part: RepairErrorProjection,
    dataset: Literal["rcabench", "aiops25"],
    expected_pin: str,
) -> RepairErrorFactSummary:
    # This temporary expansion is not a new case/budget or retained audit input.
    # All four raw projections were admitted together by the outer snapshot.
    rebuilt = _summarize(
        records=part.records, dataset=dataset, interval=part.interval, modality=part.modality
    )
    saved = snapshot(rebuilt, _ROOTS)
    with ExitStack() as guards:
        guards.callback(require_unchanged, rebuilt, _ROOTS, saved)
        if rebuilt.result_sha256 != expected_pin or _digest_unsigned(rebuilt) != expected_pin:
            raise ValueError("error projection differs from independent complete-source replay")
        return _summary_values(rebuilt)


@contextmanager
def _replayed_referenced_errors(
    *,
    sources: RepairErrorSources,
    expected_sources_sha256: str,
    expected_observation_sha256s: tuple[str, ...],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
) -> Iterator[tuple[RepairReferencedErrorFacts, tuple[RepairErrorFactSummary, ...]]]:
    """Atomic four-projection replay. No observation hash inside sources grants trust."""
    context = (
        sources,
        expected_sources_sha256,
        expected_observation_sha256s,
        time_contract,
        expected_time_contract_sha256,
        candidate_universe,
        frozen_candidates,
    )
    saved = snapshot(context, _ROOTS)
    with ExitStack() as guards:
        guards.callback(require_unchanged, context, _ROOTS, saved)
        if type(sources) is not RepairErrorSources:
            raise ValueError("exact current error source bundle required")
        for pin in (
            expected_sources_sha256,
            expected_time_contract_sha256,
            *expected_observation_sha256s,
        ):
            require_pin(pin)
        if (
            type(expected_observation_sha256s) is not tuple
            or len(expected_observation_sha256s) != 4
        ):
            raise ValueError("four independent error observation pins required")
        if (
            sources.result_sha256 != expected_sources_sha256
            or _digest_unsigned(sources) != expected_sources_sha256
        ):
            raise ValueError("error source bundle pin mismatch")
        if sha256_hex(canonical(time_contract, _ROOTS)) != expected_time_contract_sha256:
            raise ValueError("error reference time pin mismatch")
        fresh = RepairErrorSources.model_validate_json(canonical(sources, _ROOTS))
        contract = CaseTimeContract.model_validate_json(canonical(time_contract, _ROOTS))
        universe = CandidateUniverse.model_validate_json(canonical(candidate_universe, _ROOTS))
        frozen = FrozenCandidateUniverse.model_validate_json(canonical(frozen_candidates, _ROOTS))
        working = (fresh, contract, universe, frozen, expected_observation_sha256s)
        guards.callback(require_unchanged, working, _ROOTS, snapshot(working, _ROOTS))
        require_frozen_candidate_universe(frozen=frozen, candidate_universe=universe)
        if not fresh.opaque_case_id == contract.opaque_case_id == frozen.opaque_case_id:
            raise ValueError("error reference case mismatch")
        summaries = []
        for part, pin in zip(fresh.projections, expected_observation_sha256s, strict=True):
            interval = (
                contract.baseline_window if part.window == "baseline" else contract.fault_window
            )
            if part.interval != interval:
                raise ValueError("error reference interval mismatch")
            summary = _replay_projection(part, fresh.dataset, pin)
            guards.callback(require_unchanged, summary, _ROOTS, snapshot(summary, _ROOTS))
            summaries.append(summary)
        result = _referenced_error_values(
            sources=fresh, summaries=tuple(summaries),
            observation_sha256s=expected_observation_sha256s,
            candidate_service_keys=frozen.candidate_service_keys,
        )
        guards.callback(require_unchanged, result, _ROOTS, snapshot(result, _ROOTS))
        yield result, tuple(summaries)


def _referenced_error_values(
    *, sources: RepairErrorSources, summaries: tuple[RepairErrorFactSummary, ...],
    observation_sha256s: tuple[str, ...], candidate_service_keys: tuple[str, ...],
) -> RepairReferencedErrorFacts:
    """Shared four-slot facts; admission/replay duties stay with each caller."""
    if len(summaries) != 4 or tuple(s.result_sha256 for s in summaries) != observation_sha256s:
        raise ValueError("error fact summary collection/pins differ")
    for summary, part in zip(summaries, sources.projections, strict=True):
        if summary.dataset != sources.dataset or summary.modality != part.modality or (
            summary.interval != part.interval
        ):
            raise ValueError("error fact summary window/modality binding differs")
    facts = tuple(
        row for b, f in ((1, 3), (0, 2))
        for row in _error_values(summaries[b], summaries[f], candidate_service_keys)
    )
    draft = RepairReferencedErrorFacts(
        semantics_sha256=REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256,
        source_bundle_sha256=sources.result_sha256,
        observation_sha256s=observation_sha256s, facts=facts, result_sha256="0" * 64,
    )
    return draft.model_copy(update={"result_sha256": _digest_unsigned(draft)})


def build_repair_referenced_error_facts(
    *,
    sources: RepairErrorSources,
    expected_sources_sha256: str,
    expected_observation_sha256s: tuple[str, ...],
    time_contract: CaseTimeContract,
    expected_time_contract_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
) -> RepairReferencedErrorFacts:
    """Legacy public result; the scoped replay also serves the current scorer."""
    with _replayed_referenced_errors(
        sources=sources,
        expected_sources_sha256=expected_sources_sha256,
        expected_observation_sha256s=expected_observation_sha256s,
        time_contract=time_contract,
        expected_time_contract_sha256=expected_time_contract_sha256,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
    ) as (result, _):
        return result
