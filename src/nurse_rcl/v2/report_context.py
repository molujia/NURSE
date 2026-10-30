"""Immutable post-ranking context exposed to NURSE V2 report generation."""

from __future__ import annotations

import hmac
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_audit import (
    CaseAuditLineageIndex,
    CaseRankingReplayBundle,
    ImmutableCaseAuditPackage,
    replay_case_audit,
)
from nurse_rcl.v2.diagnostic_schemas import (
    DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
    DiagnosticHypothesis,
    LifecycleStatus,
    PredicateResult,
)
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult
from nurse_rcl.v2.event_strength import EventStrengthResult
from nurse_rcl.v2.family_aggregation import FamilyAggregationResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.hypothesis_guard import GuardFinalDisposition, HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    Observation,
    SemanticFocusSet,
    TimeInterval,
)
from nurse_rcl.v2.propagation_scoring import PropagationScoringResult
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.ranking_schemas import RuleSet, ScoreDecomposition, TopKArtifact
from nurse_rcl.v2.reference_ranking import ReferenceRankingResult
from nurse_rcl.v2.role_compiler import RoleCompilationResult
from nurse_rcl.v2.rule_scoring import RuleScoringResult
from nurse_rcl.v2.runtime_schemas import RunMode
from nurse_rcl.v2.temporal_condensation import (
    OnsetInterval,
    RelationSCCCondensation,
    TemporalPrecedence,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

FROZEN_REPORT_CONTEXT_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-context/3"] = (
    "nurse.v2.frozen-report-context/3"
)
FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-payload/1"] = (
    "nurse.v2.frozen-report-payload/1"
)

_QUANTUM = Decimal("0.000000000001")
_MAX_REPORT_JSON_BYTES = 64 * 1024 * 1024
_MAX_REPORT_JSON_DEPTH = 64
_MAX_REPORT_JSON_NODES = 1_000_000


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _quantized_delta(left: float | str, right: float | str) -> str:
    value = Decimal(str(left)) - Decimal(str(right))
    return format(value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def _parse_utc_seconds(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a real UTC second timestamp") from error
    return parsed.replace(tzinfo=timezone.utc)


def _require_bounded_json(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    node_count = 0
    while stack:
        item, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_REPORT_JSON_NODES:
            raise ValueError("report JSON node capacity exceeded")
        if depth > _MAX_REPORT_JSON_DEPTH:
            raise ValueError("report JSON nesting capacity exceeded")
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError("report JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in cast(list[object], item))
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("report JSON contains a non-standard value")


def _load_canonical_object(value: str, *, label: str) -> dict[str, object]:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty canonical JSON string")
    if len(value.encode("utf-8")) > _MAX_REPORT_JSON_BYTES:
        raise ValueError(f"{label} exceeds the canonical JSON size limit")
    try:
        decoded = json.loads(
            value,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} is not bounded strict JSON") from error
    _require_bounded_json(decoded)
    if type(decoded) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    if canonical_json_bytes(decoded).decode("utf-8") != value:
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, object], decoded)


class ReportHypothesisFact(StrictFrozenModel):
    """Guarded hypothesis facts safe for report generation.

    The submitted free-form record is deliberately absent.  An unparseable
    rejected record contributes only its digest and deterministic guard codes.
    """

    guard_result_id: str = Field(pattern=r"^hgr_[0-9a-f]{24}$")
    submitted_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    hypothesis_id: str | None = Field(default=None, pattern=r"^hyp_[0-9a-f]{24}$")
    candidate_service_key: str | None = Field(default=None, min_length=1)
    mechanism_key: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    mechanism_definition_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    origin: str | None = Field(default=None, min_length=1)
    origin_version: str | None = Field(default=None, min_length=1)
    lifecycle_status: LifecycleStatus | None
    final_disposition: GuardFinalDisposition
    guard_decision_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    guard_reason_codes: tuple[str, ...] = Field(min_length=1, max_length=4)
    observation_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    time_intervals: tuple[TimeInterval, ...]
    necessary_predicate_result_ids: tuple[str, ...]
    optional_predicate_result_ids: tuple[str, ...]
    refuting_predicate_result_ids: tuple[str, ...]
    alternative_hypothesis_ids: tuple[str, ...]
    verification_request_ids: tuple[str, ...]
    unknown_reasons: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    hypothesis_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score_impact_id: str | None = Field(default=None, pattern=r"^sci_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def require_closed_shape(self) -> Self:
        if len(self.guard_decision_ids) != len(self.guard_reason_codes):
            raise ValueError("hypothesis guard decisions and reasons differ in length")
        canonical_fields = (
            self.observation_ids,
            self.relation_ids,
            self.event_group_ids,
            self.necessary_predicate_result_ids,
            self.optional_predicate_result_ids,
            self.refuting_predicate_result_ids,
            self.alternative_hypothesis_ids,
            self.verification_request_ids,
            self.unknown_reasons,
            self.provenance_ids,
        )
        if any(values != tuple(sorted(set(values))) for values in canonical_fields):
            raise ValueError("report hypothesis references must be sorted and unique")
        parsed_fields = (
            self.hypothesis_id,
            self.candidate_service_key,
            self.mechanism_key,
            self.mechanism_definition_sha256,
            self.origin,
            self.origin_version,
            self.lifecycle_status,
            self.hypothesis_sha256,
        )
        if self.hypothesis_id is None:
            if any(value is not None for value in parsed_fields) or any(
                (
                    self.observation_ids,
                    self.relation_ids,
                    self.event_group_ids,
                    self.time_intervals,
                    self.necessary_predicate_result_ids,
                    self.optional_predicate_result_ids,
                    self.refuting_predicate_result_ids,
                    self.alternative_hypothesis_ids,
                    self.verification_request_ids,
                    self.unknown_reasons,
                    self.provenance_ids,
                )
            ):
                raise ValueError("unparsed report hypothesis cannot expose submitted claims")
            if self.final_disposition != "rejected" or self.score_impact_id is not None:
                raise ValueError("unparsed report hypothesis must be rejected and score-neutral")
        elif (
            any(value is None for value in parsed_fields[1:])
            or not self.time_intervals
            or not self.provenance_ids
        ):
            raise ValueError("parsed report hypothesis is missing typed fields")
        else:
            rebuilt = DiagnosticHypothesis.model_validate(
                {
                    "schema_version": DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
                    "hypothesis_id": self.hypothesis_id,
                    "opaque_case_id": self.opaque_case_id,
                    "candidate_service_key": self.candidate_service_key,
                    "mechanism_key": self.mechanism_key,
                    "mechanism_definition_sha256": self.mechanism_definition_sha256,
                    "origin": self.origin,
                    "origin_version": self.origin_version,
                    "observation_ids": self.observation_ids,
                    "relation_ids": self.relation_ids,
                    "event_group_ids": self.event_group_ids,
                    "time_intervals": tuple(
                        item.model_dump(mode="json") for item in self.time_intervals
                    ),
                    "necessary_predicate_result_ids": (self.necessary_predicate_result_ids),
                    "optional_predicate_result_ids": self.optional_predicate_result_ids,
                    "refuting_predicate_result_ids": self.refuting_predicate_result_ids,
                    "alternative_hypothesis_ids": self.alternative_hypothesis_ids,
                    "verification_request_ids": self.verification_request_ids,
                    "unknown_reasons": self.unknown_reasons,
                    "provenance_ids": self.provenance_ids,
                    "lifecycle_status": self.lifecycle_status,
                    "hypothesis_sha256": self.hypothesis_sha256,
                }
            )
            if not rebuilt.verify():
                raise ValueError("report hypothesis differs from its content digest")
        if (self.score_impact_id is not None) != (self.final_disposition == "accepted"):
            raise ValueError("only an accepted report hypothesis may expose score impact")
        return self


class ReportCompetitionDelta(StrictFrozenModel):
    """Content-addressed Top-1 versus competitor difference, not prose."""

    competition_id: str = Field(pattern=r"^cmp_[0-9a-f]{24}$")
    leading_service_key: str = Field(min_length=1)
    competing_service_key: str = Field(min_length=1)
    competing_rank: int = Field(ge=2)
    leading_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    competing_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_score_gap: str = Field(pattern=r"^-?\d+\.\d{12}$")
    event_evidence_gap: str = Field(pattern=r"^-?\d+\.\d{12}$")
    propagation_gap: str = Field(pattern=r"^-?\d+\.\d{12}$")
    rule_gap: str = Field(pattern=r"^-?\d+\.\d{12}$")
    history_gap: str = Field(pattern=r"^-?\d+\.\d{12}$")
    evidence_tied: bool
    ordering_decided_by_tie_break: bool
    leading_unique_event_group_ids: tuple[str, ...]
    competing_unique_event_group_ids: tuple[str, ...]
    leading_accepted_hypothesis_ids: tuple[str, ...]
    competing_accepted_hypothesis_ids: tuple[str, ...]
    competition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_difference(self) -> Self:
        if self.leading_service_key == self.competing_service_key:
            raise ValueError("competition requires two different candidates")
        for values in (
            self.leading_unique_event_group_ids,
            self.competing_unique_event_group_ids,
            self.leading_accepted_hypothesis_ids,
            self.competing_accepted_hypothesis_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("competition references must be sorted and unique")
        if self.ordering_decided_by_tie_break != self.evidence_tied:
            raise ValueError("stable tie-break is decisive exactly for an evidence tie")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"competition_id", "competition_sha256"},
            warnings="error",
        )
        digest = _digest(unsigned)
        return self.competition_id == f"cmp_{digest[:24]}" and hmac.compare_digest(
            self.competition_sha256,
            digest,
        )


def _build_report_hypothesis_facts(
    bundle: CaseRankingReplayBundle,
) -> tuple[ReportHypothesisFact, ...]:
    return _report_hypothesis_facts(bundle.evidence.time_contract.opaque_case_id,
        bundle.evidence.role_guard_results)


def _report_hypothesis_facts(
    opaque_case_id: str, guards: Sequence[HypothesisGuardResult],
) -> tuple[ReportHypothesisFact, ...]:
    """Pure projection; callers retain their independent source replay gates."""
    facts: list[ReportHypothesisFact] = []
    for guard in guards:
        hypothesis = guard.parsed_hypothesis
        decisions = guard.decisions
        facts.append(
            ReportHypothesisFact(
                guard_result_id=guard.guard_result_id,
                submitted_record_sha256=guard.submitted_record_sha256,
                opaque_case_id=opaque_case_id,
                hypothesis_id=None if hypothesis is None else hypothesis.hypothesis_id,
                candidate_service_key=(
                    None if hypothesis is None else hypothesis.candidate_service_key
                ),
                mechanism_key=None if hypothesis is None else hypothesis.mechanism_key,
                mechanism_definition_sha256=(
                    None if hypothesis is None else hypothesis.mechanism_definition_sha256
                ),
                origin=None if hypothesis is None else hypothesis.origin,
                origin_version=None if hypothesis is None else hypothesis.origin_version,
                lifecycle_status=None if hypothesis is None else hypothesis.lifecycle_status,
                final_disposition=guard.final_disposition,
                guard_decision_ids=tuple(item.guard_decision_id for item in decisions),
                guard_reason_codes=tuple(item.reason_code for item in decisions),
                observation_ids=() if hypothesis is None else hypothesis.observation_ids,
                relation_ids=() if hypothesis is None else hypothesis.relation_ids,
                event_group_ids=() if hypothesis is None else hypothesis.event_group_ids,
                time_intervals=() if hypothesis is None else hypothesis.time_intervals,
                necessary_predicate_result_ids=(
                    () if hypothesis is None else hypothesis.necessary_predicate_result_ids
                ),
                optional_predicate_result_ids=(
                    () if hypothesis is None else hypothesis.optional_predicate_result_ids
                ),
                refuting_predicate_result_ids=(
                    () if hypothesis is None else hypothesis.refuting_predicate_result_ids
                ),
                alternative_hypothesis_ids=(
                    () if hypothesis is None else hypothesis.alternative_hypothesis_ids
                ),
                verification_request_ids=(
                    () if hypothesis is None else hypothesis.verification_request_ids
                ),
                unknown_reasons=() if hypothesis is None else hypothesis.unknown_reasons,
                provenance_ids=() if hypothesis is None else hypothesis.provenance_ids,
                hypothesis_sha256=(None if hypothesis is None else hypothesis.hypothesis_sha256),
                score_impact_id=(
                    None if guard.score_impact is None else guard.score_impact.score_impact_id
                ),
            )
        )
    return tuple(sorted(facts, key=lambda item: item.guard_result_id))


def _accepted_hypothesis_ids_by_candidate(
    facts: Sequence[ReportHypothesisFact],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {}
    for fact in facts:
        if (
            fact.final_disposition == "accepted"
            and fact.hypothesis_id is not None
            and fact.candidate_service_key is not None
        ):
            result.setdefault(fact.candidate_service_key, set()).add(fact.hypothesis_id)
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _build_competition_delta(
    *,
    leader: ScoreDecomposition,
    competitor: ScoreDecomposition,
    competing_rank: int,
    evidence_tied: bool,
    accepted_hypotheses: dict[str, tuple[str, ...]],
) -> ReportCompetitionDelta:
    leader_events = set(leader.event_group_ids)
    competitor_events = set(competitor.event_group_ids)
    unsigned: dict[str, object] = {
        "leading_service_key": leader.candidate_service_key,
        "competing_service_key": competitor.candidate_service_key,
        "competing_rank": competing_rank,
        "leading_score_sha256": leader.score_sha256,
        "competing_score_sha256": competitor.score_sha256,
        "total_score_gap": _quantized_delta(
            leader.quantized_total_score,
            competitor.quantized_total_score,
        ),
        "event_evidence_gap": _quantized_delta(
            leader.quantized_event_evidence_score,
            competitor.quantized_event_evidence_score,
        ),
        "propagation_gap": _quantized_delta(
            leader.quantized_propagation_score,
            competitor.quantized_propagation_score,
        ),
        "rule_gap": _quantized_delta(leader.rule_score, competitor.rule_score),
        "history_gap": _quantized_delta(leader.history_score, competitor.history_score),
        "evidence_tied": evidence_tied,
        "ordering_decided_by_tie_break": evidence_tied,
        "leading_unique_event_group_ids": tuple(sorted(leader_events - competitor_events)),
        "competing_unique_event_group_ids": tuple(sorted(competitor_events - leader_events)),
        "leading_accepted_hypothesis_ids": accepted_hypotheses.get(
            leader.candidate_service_key,
            (),
        ),
        "competing_accepted_hypothesis_ids": accepted_hypotheses.get(
            competitor.candidate_service_key,
            (),
        ),
    }
    digest = _digest(unsigned)
    return ReportCompetitionDelta.model_validate(
        {
            **unsigned,
            "competition_id": f"cmp_{digest[:24]}",
            "competition_sha256": digest,
        }
    )


def _build_competition_deltas(
    ranking: ReferenceRankingResult,
    facts: Sequence[ReportHypothesisFact],
) -> tuple[ReportCompetitionDelta, ...]:
    entries = ranking.top_k.entries
    if len(entries) < 2:
        return ()
    score_by_service = {item.candidate_service_key: item for item in ranking.scores}
    accepted_hypotheses = _accepted_hypothesis_ids_by_candidate(facts)
    leader_entry = entries[0]
    leader = score_by_service[leader_entry.service_entity_key]
    return tuple(
        _build_competition_delta(
            leader=leader,
            competitor=score_by_service[entry.service_entity_key],
            competing_rank=entry.rank,
            evidence_tied=(entry.evidence_tie_group_id == leader_entry.evidence_tie_group_id),
            accepted_hypotheses=accepted_hypotheses,
        )
        for entry in entries[1:]
    )


class FrozenReportPayload(StrictFrozenModel):
    """Typed report-only projection of a verified case-ranking replay bundle."""

    schema_version: Literal["nurse.v2.frozen-report-payload/1"] = (
        FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION
    )
    time_contract: CaseTimeContract
    scope_interval: TimeInterval
    candidate_universe: CandidateUniverse
    focus_set: SemanticFocusSet
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    event_strength: EventStrengthResult
    mechanism_library: MechanismLibrary
    hypotheses: tuple[ReportHypothesisFact, ...]
    predicates: tuple[PredicateResult, ...]
    query_sessions: tuple[CaseQuerySessionResult, ...]
    role_compilation: RoleCompilationResult
    graph_snapshots: tuple[GraphSnapshot, ...]
    condensations: tuple[RelationSCCCondensation, ...]
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    family_aggregation: FamilyAggregationResult
    propagation_scoring: PropagationScoringResult
    rule_set: RuleSet
    rule_scoring: RuleScoringResult
    history_evaluation: HistoryEvaluationResult
    ranking_result: ReferenceRankingResult
    competition_deltas: tuple[ReportCompetitionDelta, ...]
    lineage_index: CaseAuditLineageIndex
    degradation_reason_codes: tuple[str, ...]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def opaque_case_id(self) -> str:
        return self.time_contract.opaque_case_id

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self.event_grouping.observations

    @property
    def event_groups(self) -> tuple[EventGroup, ...]:
        return self.event_grouping.event_groups

    @property
    def scores(self) -> tuple[ScoreDecomposition, ...]:
        return self.ranking_result.scores

    @property
    def top_k(self) -> TopKArtifact:
        return self.ranking_result.top_k

    @model_validator(mode="after")
    def require_canonical_projection(self) -> Self:
        ordered = (
            (self.hypotheses, "guard_result_id"),
            (self.predicates, "predicate_result_id"),
            (self.query_sessions, "sequence_number"),
            (self.graph_snapshots, "snapshot_sha256"),
            (self.condensations, "condensation_id"),
            (self.onsets, "onset_id"),
            (self.precedences, "precedence_id"),
            (self.competition_deltas, "competing_rank"),
        )
        for values, key in ordered:
            if values != tuple(sorted(values, key=lambda item: getattr(item, key))):
                raise ValueError(f"report payload {key} records must be canonically ordered")
        if self.degradation_reason_codes != tuple(sorted(set(self.degradation_reason_codes))):
            raise ValueError("report degradation reason codes must be sorted and unique")
        self._require_reference_closure()
        return self

    def _require_reference_closure(self) -> None:
        opaque_case_id = self.opaque_case_id
        case_scoped_ids = (
            self.candidate_universe.opaque_case_id,
            self.focus_set.opaque_case_id,
            self.entity_registry.opaque_case_id,
            self.event_grouping.opaque_case_id,
            self.event_strength.opaque_case_id,
            self.role_compilation.opaque_case_id,
            self.family_aggregation.opaque_case_id,
            self.propagation_scoring.opaque_case_id,
            self.rule_scoring.opaque_case_id,
            self.ranking_result.opaque_case_id,
            self.lineage_index.opaque_case_id,
            *(item.opaque_case_id for item in self.hypotheses),
            *(item.opaque_case_id for item in self.query_sessions),
            *(item.opaque_case_id for item in self.graph_snapshots),
            *(item.opaque_case_id for item in self.condensations),
            *(item.opaque_case_id for item in self.onsets),
            *(item.opaque_case_id for item in self.precedences),
        )
        if any(value != opaque_case_id for value in case_scoped_ids):
            raise ValueError("report payload crosses opaque case scopes")
        candidate_keys = tuple(
            item.service_entity_key for item in self.candidate_universe.candidates
        )
        candidate_set = set(candidate_keys)
        score_keys = tuple(item.candidate_service_key for item in self.scores)
        if (
            self.focus_set.candidate_universe_sha256 != self.candidate_universe.universe_sha256
            or self.ranking_result.candidate_universe_sha256
            != self.candidate_universe.universe_sha256
            or self.history_evaluation.candidate_universe_sha256
            != self.candidate_universe.universe_sha256
            or set(score_keys) != candidate_set
            or len(score_keys) != len(candidate_keys)
        ):
            raise ValueError("report payload candidate and ranking scopes differ")
        channel_candidate_sets = (
            {item.candidate_service_key for item in self.family_aggregation.candidates},
            {item.candidate_service_key for item in self.propagation_scoring.candidate_scores},
            {item.candidate_service_key for item in self.rule_scoring.candidates},
            {item.candidate_service_key for item in self.history_evaluation.candidate_scores},
        )
        if any(values != candidate_set for values in channel_candidate_sets):
            raise ValueError("report payload ranking channels do not cover C exactly")
        ranking_channel_bindings = (
            (
                self.ranking_result.family_aggregation_result_sha256,
                self.family_aggregation.result_sha256,
            ),
            (
                self.ranking_result.propagation_scoring_result_sha256,
                self.propagation_scoring.result_sha256,
            ),
            (
                self.ranking_result.propagation_authoritative_input_sha256,
                self.propagation_scoring.authoritative_input_sha256,
            ),
            (
                self.ranking_result.rule_scoring_result_sha256,
                self.rule_scoring.result_sha256,
            ),
            (
                self.ranking_result.rule_authoritative_input_sha256,
                self.rule_scoring.authoritative_input_sha256,
            ),
            (
                self.ranking_result.history_evaluation_result_sha256,
                self.history_evaluation.result_sha256,
            ),
            (
                self.ranking_result.history_authoritative_input_sha256,
                self.history_evaluation.authoritative_input_sha256,
            ),
        )
        if any(
            ranking_value != channel_value
            for ranking_value, channel_value in ranking_channel_bindings
        ):
            raise ValueError("report ranking references a different evidence channel")
        observation_ids = {item.observation_id for item in self.observations}
        event_group_ids = {item.event_group_id for item in self.event_groups}
        predicate_ids = {item.predicate_result_id for item in self.predicates}
        mechanism_sha256_by_key = {
            item.mechanism_key: item.definition_sha256
            for item in self.mechanism_library.definitions
        }
        mechanism_keys = set(mechanism_sha256_by_key)
        hypothesis_ids = {
            item.hypothesis_id for item in self.hypotheses if item.hypothesis_id is not None
        }
        guard_ids = {item.guard_result_id for item in self.hypotheses}
        relation_ids = {
            *(item.claim_id for item in self.event_grouping.association_claims),
            *(edge.edge_id for snapshot in self.graph_snapshots for edge in snapshot.edges),
            *(
                edge_id
                for condensation in self.condensations
                for edge_id in condensation.selected_edge_ids
            ),
            *(
                edge.condensed_edge_id
                for condensation in self.condensations
                for edge in condensation.condensed_edges
            ),
            *(item.relation_id for item in self.role_compilation.proposals),
            *(item.relation_id for item in self.role_compilation.assertions),
        }
        verification_request_ids = {
            item.verification_request_id
            for session in self.query_sessions
            for result in (session.verification_result,)
            if result is not None
            for collection in (
                result.requests,
                result.request_counts,
                result.decisions,
                result.results,
            )
            for item in collection
        }
        score_impact_ids = {
            provenance_id
            for assertion in self.role_compilation.assertions
            for provenance_id in assertion.provenance_ids
            if len(provenance_id) == 28
            and provenance_id.startswith("sci_")
            and all(character in "0123456789abcdef" for character in provenance_id[4:])
        }
        if len(predicate_ids) != len(self.predicates):
            raise ValueError("report predicate IDs must be unique")
        if len(guard_ids) != len(self.hypotheses):
            raise ValueError("report guard IDs must be unique")
        for fact in self.hypotheses:
            if fact.candidate_service_key is not None and (
                fact.candidate_service_key not in candidate_set
            ):
                raise ValueError("report hypothesis references a candidate outside C")
            if fact.mechanism_key is not None and fact.mechanism_key not in mechanism_keys:
                raise ValueError("report hypothesis references an unknown mechanism")
            if fact.mechanism_key is not None and (
                fact.mechanism_definition_sha256 != mechanism_sha256_by_key[fact.mechanism_key]
            ):
                raise ValueError("report hypothesis references a different mechanism definition")
            if any(
                interval.start_unix_ns < self.scope_interval.start_unix_ns
                or interval.end_unix_ns > self.scope_interval.end_unix_ns
                for interval in fact.time_intervals
            ):
                raise ValueError("report hypothesis interval falls outside the case scope")
            if not set(fact.observation_ids) <= observation_ids:
                raise ValueError("report hypothesis references an unknown observation")
            if not set(fact.relation_ids) <= relation_ids:
                raise ValueError("report hypothesis references an unknown relation")
            if not set(fact.event_group_ids) <= event_group_ids:
                raise ValueError("report hypothesis references an unknown event group")
            fact_predicates = {
                *fact.necessary_predicate_result_ids,
                *fact.optional_predicate_result_ids,
                *fact.refuting_predicate_result_ids,
            }
            if not fact_predicates <= predicate_ids:
                raise ValueError("report hypothesis references an unknown predicate")
            if not set(fact.alternative_hypothesis_ids) <= hypothesis_ids:
                raise ValueError("report hypothesis references an unknown alternative")
            if not set(fact.verification_request_ids) <= verification_request_ids:
                raise ValueError("report hypothesis references an unknown verification request")
            if fact.score_impact_id is not None and fact.score_impact_id not in score_impact_ids:
                raise ValueError("report hypothesis references an unknown score impact")
        if not set(self.role_compilation.guard_result_ids) <= guard_ids:
            raise ValueError("role compilation references an unavailable report hypothesis")
        base_event_group_ids = {item.event_group_id for item in self.event_strength.group_strengths}
        if (
            self.event_strength.event_grouping_result_sha256 != self.event_grouping.result_sha256
            or self.event_strength.role_compilation_id != self.role_compilation.role_compilation_id
            or self.event_strength.role_compilation_result_sha256
            != self.role_compilation.result_sha256
            or base_event_group_ids != event_group_ids
            or any(
                item.candidate_service_key not in candidate_set
                or item.event_group_id not in event_group_ids
                for item in self.event_strength.candidate_event_adjustments
            )
        ):
            raise ValueError("report event strength differs from its frozen evidence inputs")
        if (
            self.family_aggregation.event_grouping_result_sha256
            != self.event_grouping.result_sha256
            or self.family_aggregation.event_strength_result_id
            != self.event_strength.event_strength_result_id
            or self.family_aggregation.event_strength_result_sha256
            != self.event_strength.result_sha256
        ):
            raise ValueError("report family aggregation differs from its frozen event inputs")
        if any(not set(item.observation_ids) <= observation_ids for item in self.predicates):
            raise ValueError("report predicate references an unknown observation")
        if any(not set(item.event_group_ids) <= event_group_ids for item in self.scores):
            raise ValueError("report score references an unknown event group")
        expected_deltas = _build_competition_deltas(
            self.ranking_result,
            self.hypotheses,
        )
        if self.competition_deltas != expected_deltas:
            raise ValueError("report competition deltas differ from the frozen ranking")

    def verify(self) -> bool:
        try:
            if type(self) is not FrozenReportPayload:
                return False
            rebuilt = FrozenReportPayload.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            if not (
                rebuilt.candidate_universe.verify()
                and rebuilt.focus_set.verify()
                and rebuilt.entity_registry.verify()
                and rebuilt.event_grouping.verify()
                and rebuilt.event_strength.verify()
                and rebuilt.mechanism_library.verify()
                and rebuilt.role_compilation.verify()
                and rebuilt.family_aggregation.verify()
                and rebuilt.propagation_scoring.verify_structure()
                and rebuilt.rule_set.verify()
                and rebuilt.rule_scoring.verify_structure()
                and rebuilt.history_evaluation.verify_structure()
                and rebuilt.ranking_result.verify_structure()
                and rebuilt.lineage_index.verify()
                and all(item.verify() for item in rebuilt.predicates)
                and all(item.verify() for item in rebuilt.query_sessions)
                and all(item.verify() for item in rebuilt.graph_snapshots)
                and all(item.verify() for item in rebuilt.condensations)
                and all(item.verify() for item in rebuilt.onsets)
                and all(item.verify() for item in rebuilt.precedences)
                and all(item.verify() for item in rebuilt.competition_deltas)
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"payload_sha256"},
                warnings="error",
            )
            return hmac.compare_digest(rebuilt.payload_sha256, _digest(unsigned))
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


def _deduplicated_predicates(bundle: CaseRankingReplayBundle) -> tuple[PredicateResult, ...]:
    by_id: dict[str, PredicateResult] = {}
    for predicate in (*bundle.evidence.role_predicates, *bundle.rule_predicates):
        previous = by_id.setdefault(predicate.predicate_result_id, predicate)
        if previous != predicate:
            raise ValueError("predicate ID collision across ranking channels")
    return tuple(sorted(by_id.values(), key=lambda item: item.predicate_result_id))


def _build_report_payload(
    bundle: CaseRankingReplayBundle,
    lineage: CaseAuditLineageIndex,
) -> FrozenReportPayload:
    hypotheses = _build_report_hypothesis_facts(bundle)
    unsigned: dict[str, object] = {
        "schema_version": FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION,
        "time_contract": bundle.evidence.time_contract.model_dump(mode="json"),
        "scope_interval": bundle.evidence.scope_interval.model_dump(mode="json"),
        "candidate_universe": bundle.evidence.candidate_universe.model_dump(mode="json"),
        "focus_set": bundle.evidence.focus_set.model_dump(mode="json"),
        "entity_registry": bundle.evidence.entity_registry.model_dump(mode="json"),
        "event_grouping": bundle.evidence.event_grouping.model_dump(mode="json"),
        "event_strength": bundle.evidence.event_strength.model_dump(mode="json"),
        "mechanism_library": bundle.evidence.mechanism_library.model_dump(mode="json"),
        "hypotheses": tuple(item.model_dump(mode="json") for item in hypotheses),
        "predicates": tuple(
            item.model_dump(mode="json") for item in _deduplicated_predicates(bundle)
        ),
        "query_sessions": tuple(
            item.model_dump(mode="json") for item in bundle.evidence.query_sessions
        ),
        "role_compilation": bundle.evidence.role_compilation.model_dump(mode="json"),
        "graph_snapshots": tuple(
            item.model_dump(mode="json") for item in bundle.evidence.graph_snapshots
        ),
        "condensations": tuple(
            item.model_dump(mode="json") for item in bundle.evidence.condensations
        ),
        "onsets": tuple(item.model_dump(mode="json") for item in bundle.evidence.onsets),
        "precedences": tuple(item.model_dump(mode="json") for item in bundle.evidence.precedences),
        "family_aggregation": bundle.family_aggregation.model_dump(mode="json"),
        "propagation_scoring": bundle.propagation_scoring.model_dump(mode="json"),
        "rule_set": bundle.rule_set.model_dump(mode="json"),
        "rule_scoring": bundle.rule_scoring.model_dump(mode="json"),
        "history_evaluation": bundle.history_evaluation.model_dump(mode="json"),
        "ranking_result": bundle.ranking_result.model_dump(mode="json"),
        "competition_deltas": tuple(
            item.model_dump(mode="json")
            for item in _build_competition_deltas(bundle.ranking_result, hypotheses)
        ),
        "lineage_index": lineage.model_dump(mode="json"),
        "degradation_reason_codes": bundle.degradation_reason_codes,
    }
    payload = FrozenReportPayload.model_validate({**unsigned, "payload_sha256": _digest(unsigned)})
    if not payload.verify():  # pragma: no cover - construction invariant
        raise AssertionError("frozen report payload verification failed")
    return payload


class FrozenReportContext(StrictFrozenModel):
    """Canonical-at-rest boundary consumed by deterministic or Agent reports."""

    schema_version: Literal["nurse.v2.frozen-report-context/3"] = (
        FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    experiment_version: str = Field(min_length=1)
    run_mode: RunMode
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_json: str = Field(min_length=2)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    stage: Literal["CONTEXT_FROZEN"] = "CONTEXT_FROZEN"
    ranking_mutable: Literal[False] = False
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_payload(self) -> Self:
        _load_canonical_object(self.payload_json, label="frozen report payload")
        _parse_utc_seconds(self.frozen_at_utc, label="report context freeze time")
        return self

    @property
    def payload(self) -> FrozenReportPayload:
        return FrozenReportPayload.model_validate(
            _load_canonical_object(self.payload_json, label="frozen report payload")
        )

    def verify(self) -> bool:
        try:
            if type(self) is not FrozenReportContext:
                return False
            rebuilt = FrozenReportContext.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            payload = rebuilt.payload
            if not payload.verify() or not hmac.compare_digest(
                rebuilt.payload_sha256,
                payload.payload_sha256,
            ):
                return False
            if (
                rebuilt.opaque_case_id != payload.opaque_case_id
                or rebuilt.source_ranking_bundle_sha256
                != payload.lineage_index.replay_bundle_sha256
                or rebuilt.source_ranking_result_sha256 != payload.ranking_result.result_sha256
                or rebuilt.source_top_k_sha256 != payload.top_k.ranking_sha256
                or rebuilt.source_lineage_sha256 != payload.lineage_index.lineage_sha256
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"context_sha256"},
                warnings="error",
            )
            return hmac.compare_digest(rebuilt.context_sha256, _digest(unsigned))
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


def _exact_ranked_audit(
    audit_package: ImmutableCaseAuditPackage,
    *,
    expected_audit_sha256: str,
) -> tuple[CaseRankingReplayBundle, CaseAuditLineageIndex]:
    if type(audit_package) is not ImmutableCaseAuditPackage:
        raise ValueError("report context requires the exact ImmutableCaseAuditPackage type")
    receipt = replay_case_audit(
        audit_package,
        expected_audit_sha256=expected_audit_sha256,
    )
    if receipt.terminal_state != "SEALED" or receipt.ranking_result is None:
        raise ValueError("report context requires a successful sealed ranking audit")
    bundle = audit_package.replay_bundle
    lineage = audit_package.lineage_index
    if not lineage.verify_against(bundle):  # pragma: no cover - package verification invariant
        raise AssertionError("verified audit lost its ranking lineage")
    return bundle, lineage


def freeze_report_context_from_audit(
    *,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
    frozen_at_utc: str,
) -> FrozenReportContext:
    """Freeze a report-only projection after externally anchored rank replay."""

    bundle, lineage = _exact_ranked_audit(
        audit_package,
        expected_audit_sha256=expected_audit_sha256,
    )
    sealed = audit_package.sealed_audit_package
    if _parse_utc_seconds(
        frozen_at_utc,
        label="report context freeze time",
    ) < _parse_utc_seconds(sealed.sealed_at_utc, label="ranked audit seal time"):
        raise ValueError("report context must be frozen after the ranked audit")
    payload = _build_report_payload(bundle, lineage)
    payload_json = canonical_json_bytes(payload.model_dump(mode="json", warnings="error")).decode(
        "utf-8"
    )
    unsigned: dict[str, object] = {
        "schema_version": FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
        "opaque_case_id": audit_package.opaque_case_id,
        "experiment_version": sealed.experiment_version,
        "run_mode": sealed.run_mode,
        "source_audit_sha256": audit_package.audit_sha256,
        "source_ranking_bundle_sha256": bundle.bundle_sha256,
        "source_ranking_result_sha256": bundle.ranking_result.result_sha256,
        "source_top_k_sha256": bundle.top_k.ranking_sha256,
        "source_lineage_sha256": lineage.lineage_sha256,
        "payload_json": payload_json,
        "payload_sha256": payload.payload_sha256,
        "frozen_at_utc": frozen_at_utc,
        "stage": "CONTEXT_FROZEN",
        "ranking_mutable": False,
    }
    context = FrozenReportContext.model_validate({**unsigned, "context_sha256": _digest(unsigned)})
    if not context.verify():  # pragma: no cover - construction invariant
        raise AssertionError("frozen report context verification failed")
    return context


def require_frozen_report_context(
    context: FrozenReportContext,
    *,
    audit_package: ImmutableCaseAuditPackage,
    expected_audit_sha256: str,
) -> FrozenReportContext:
    """Rebuild a report context from its external audit anchor and require identity."""

    if type(context) is not FrozenReportContext or not context.verify():
        raise ValueError("frozen report context failed exact verification")
    expected = freeze_report_context_from_audit(
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        frozen_at_utc=context.frozen_at_utc,
    )
    if not hmac.compare_digest(context.context_sha256, expected.context_sha256) or (
        context != expected
    ):
        raise ValueError("frozen report context differs from its source audit package")
    return FrozenReportContext.model_validate(context.model_dump(mode="json", warnings="error"))


__all__ = [
    "FROZEN_REPORT_CONTEXT_SCHEMA_VERSION",
    "FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION",
    "FrozenReportContext",
    "FrozenReportPayload",
    "ReportCompetitionDelta",
    "ReportHypothesisFact",
    "freeze_report_context_from_audit",
    "require_frozen_report_context",
]
