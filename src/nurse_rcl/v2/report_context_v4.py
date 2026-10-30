"""Independent report projection of a completely replayed /2 ranking audit.

The report is a projection, never a scoring or source qualification shortcut.
Pure preflight helpers are shared with the paired reader; all source authority
still comes from complete replay under an independently supplied audit pin.
"""

from __future__ import annotations

import hmac
import json
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import report_context as legacy
from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2
from nurse_rcl.v2.case_audit_v2 import (
    CaseRankingReplayBundleV2,
    ImmutableCaseAuditPackageV2,
    _audit_document,
    _audit_roots,
    _exact_field,
    _inspect,
    _json_shape,
    _json_with_defaults,
)
from nurse_rcl.v2.case_scoring_selection import CaseScoringSelection
from nurse_rcl.v2.diagnostic_schemas import (
    DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
    DiagnosticHypothesis,
    PredicateResult,
)
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult
from nurse_rcl.v2.event_strength import EventStrengthResult
from nurse_rcl.v2.family_aggregation import FamilyAggregationResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.no_role_compilation import (
    NoRoleCompilationResult,
    _json_bytes,
    _reject_json_number,
    _scan_json,
    _unique_object,
)
from nurse_rcl.v2.no_role_evidence import (
    NoRoleEventStrengthResult,
    NoRoleFamilyAggregationResult,
    _finite_json_float,
)
from nurse_rcl.v2.no_role_propagation import NoRolePropagationScoringResult
from nurse_rcl.v2.no_role_ranking import NoRoleReferenceRankingResult
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

FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-payload/2"] = (
    "nurse.v2.frozen-report-payload/2"
)
FROZEN_REPORT_CONTEXT_SCHEMA_VERSION: Literal["nurse.v2.frozen-report-context/4"] = (
    "nurse.v2.frozen-report-context/4"
)
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)


def _report_roots() -> tuple[type[BaseModel], ...]:
    return (
        *_audit_roots(),
        legacy.FrozenReportPayload,
        legacy.FrozenReportContext,
        FrozenReportPayloadV2,
        FrozenReportContextV4,
    )


def _report_snapshot(value: object) -> tuple[object, tuple[object, ...]]:
    return _inspect(value, _additional_roots=_report_roots())


def _report_unchanged(value: object, original: tuple[object, tuple[object, ...]]) -> None:
    try:
        unchanged = _report_snapshot(value) == original
    except _ERRORS:
        unchanged = False
    if not unchanged:
        raise ValueError("report original content or representation changed during reading")


def _sorted_ids(values: list[Any], *, label: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"report {label} must be sorted and unique")


def _hypothesis_domains(fact: dict[str, Any]) -> None:
    if len(fact["guard_decision_ids"]) != len(fact["guard_reason_codes"]):
        raise ValueError("report guard decisions and reasons differ in length")
    reference_fields = (
        "observation_ids",
        "relation_ids",
        "event_group_ids",
        "necessary_predicate_result_ids",
        "optional_predicate_result_ids",
        "refuting_predicate_result_ids",
        "alternative_hypothesis_ids",
        "verification_request_ids",
        "unknown_reasons",
        "provenance_ids",
    )
    for name in reference_fields:
        _sorted_ids(fact[name], label=name)
    parsed_fields = (
        "hypothesis_id",
        "candidate_service_key",
        "mechanism_key",
        "mechanism_definition_sha256",
        "origin",
        "origin_version",
        "lifecycle_status",
        "hypothesis_sha256",
    )
    if fact["hypothesis_id"] is None:
        if (
            any(fact[name] is not None for name in parsed_fields)
            or any(fact[name] for name in (*reference_fields, "time_intervals"))
            or fact["final_disposition"] != "rejected"
        ):
            raise ValueError("unparsed report hypothesis cannot expose submitted claims")
    else:
        hypothesis = {
            "schema_version": DIAGNOSTIC_HYPOTHESIS_SCHEMA_VERSION,
            **{
                name: fact[name]
                for name in DiagnosticHypothesis.model_fields
                if name != "schema_version"
            },
        }
        if not _json_shape(hypothesis, DiagnosticHypothesis):
            raise ValueError("report hypothesis lost its original declared domain")
        intervals = fact["time_intervals"]
        if intervals != sorted(
            intervals, key=lambda row: (row["start_unix_ns"], row["end_unix_ns"])
        ):
            raise ValueError("report hypothesis intervals must be canonically ordered")
        predicate_roles = [
            set(fact[name])
            for name in (
                "necessary_predicate_result_ids",
                "optional_predicate_result_ids",
                "refuting_predicate_result_ids",
            )
        ]
        if any(
            predicate_roles[left] & predicate_roles[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("report hypothesis predicate roles must be disjoint")
    if (fact["score_impact_id"] is not None) != (fact["final_disposition"] == "accepted"):
        raise ValueError("only accepted report hypotheses may expose score impact")


def _embedded_json(raw: str, *, scalar: bool = False) -> object:
    encoded = _json_bytes(raw)
    _scan_json(encoded)
    plain = json.loads(
        encoded.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_float=_finite_json_float,
        parse_constant=_reject_json_number,
    )
    _report_snapshot(plain)
    if scalar and (type(plain) is dict or type(plain) is list):
        raise ValueError("verification field value must be a scalar")
    return plain


def _query_scalar_domains(sessions: list[dict[str, Any]]) -> None:
    for session in sessions:
        batch = session["verification_result"]
        if batch is not None:
            for derived in batch["derived_observations"]:
                for field in derived["values"]:
                    _embedded_json(field["value_json"], scalar=True)


def _source_embedded_domains(bundle: dict[str, Any]) -> None:
    """Inspect named JSON fields and the query ledger of code-owned sources."""
    evidence = bundle["evidence"]
    for guard in evidence["role_guard_results"]:
        _embedded_json(guard["submitted_record_json"])
    _query_indexes(evidence)


def _query_indexes(
    payload: dict[str, Any],
) -> tuple[set[str], dict[str, Any], dict[str, Any]]:
    """Index cumulative records by real query identity, without creating Observations."""
    _query_scalar_domains(payload["query_sessions"])
    requests: dict[str, Any] = {}
    ledger: set[str] = set()
    results: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    for session in payload["query_sessions"]:
        batch = session["verification_result"]
        if batch is None:
            continue
        for collection, key in (
            ("requests", "verification_request_id"),
            ("request_counts", "verification_request_id"),
            ("results", "verification_request_id"),
            ("derived_observations", "derived_observation_id"),
        ):
            ids = [row[key] for row in batch[collection]]
            if ids != sorted(ids):
                raise ValueError("report query batch records must be canonically ordered")
        for collection, key in (
            ("requests", "verification_request_id"),
            ("request_counts", "verification_request_id"),
            ("results", "verification_result_id"),
            ("decisions", "verification_request_id"),
            ("derived_observations", "derived_observation_id"),
        ):
            ids = [row[key] for row in batch[collection]]
            if len(ids) != len(set(ids)):
                raise ValueError("report query batch record IDs must be unique")
        typed_ids = {row["verification_request_id"] for row in batch["requests"]}
        ledger_ids = {row["verification_request_id"] for row in batch["request_counts"]}
        decisions = {row["verification_request_id"]: row for row in batch["decisions"]}
        if (
            not typed_ids <= ledger_ids
            or ledger_ids != decisions.keys()
            or ledger_ids != {row["verification_request_id"] for row in batch["results"]}
            or any(not decisions[key]["permission_failure"] for key in ledger_ids - typed_ids)
        ):
            raise ValueError("report query request ledger is not closed")
        # A rejected malformed request is an auditable reference, but cannot
        # provide an executable request or qualify a derived observation.
        ledger.update(ledger_ids)
        for collection, field, index in (
            ("requests", "verification_request_id", requests),
            ("results", "verification_result_id", results),
            ("derived_observations", "derived_observation_id", derived),
        ):
            for row in batch[collection]:
                previous = index.setdefault(row[field], row)
                if previous != row:
                    raise ValueError("report query ID collision across cumulative sessions")
    for row in results.values():
        if (
            row["verification_request_id"] not in ledger
            or not set(row["derived_observation_ids"]) <= derived.keys()
        ):
            raise ValueError("report query result has an unknown request or derived observation")
        if any(
            derived[key]["verification_request_id"] != row["verification_request_id"]
            for key in row["derived_observation_ids"]
        ):
            raise ValueError("report query result and derived observation requests differ")
    for row in derived.values():
        if row["verification_request_id"] not in requests:
            raise ValueError("report derived observation has an unknown verification request")
    return ledger, results, derived


def _competition_domains(payload: dict[str, Any]) -> None:
    """Compare the original competition fields without hashing or model hooks."""
    entries = payload["ranking_result"]["top_k"]["entries"]
    deltas = payload["competition_deltas"]
    if len(deltas) != max(0, len(entries) - 1):
        raise ValueError("report competition must cover all frozen Top-K competitors")
    scores = {row["candidate_service_key"]: row for row in payload["ranking_result"]["scores"]}
    for delta, entry in zip(deltas, entries[1:], strict=True):
        leading_key, competing_key = entries[0]["service_entity_key"], entry["service_entity_key"]
        leader, competitor = scores[leading_key], scores[competing_key]
        tied = entries[0]["evidence_tie_group_id"] == entry["evidence_tie_group_id"]
        expected = {
            "leading_service_key": leading_key,
            "competing_service_key": competing_key,
            "competing_rank": entry["rank"],
            "leading_score_sha256": leader["score_sha256"],
            "competing_score_sha256": competitor["score_sha256"],
            "evidence_tied": tied,
            "ordering_decided_by_tie_break": tied,
            "leading_unique_event_group_ids": sorted(
                set(leader["event_group_ids"]) - set(competitor["event_group_ids"])
            ),
            "competing_unique_event_group_ids": sorted(
                set(competitor["event_group_ids"]) - set(leader["event_group_ids"])
            ),
        }
        for label, key in (("leading", leading_key), ("competing", competing_key)):
            expected[f"{label}_accepted_hypothesis_ids"] = sorted(
                {
                    fact["hypothesis_id"]
                    for fact in payload["hypotheses"]
                    if fact["final_disposition"] == "accepted"
                    and fact["candidate_service_key"] == key
                }
            )
        for label, score_field in (
            ("total_score_gap", "quantized_total_score"),
            ("event_evidence_gap", "quantized_event_evidence_score"),
            ("propagation_gap", "quantized_propagation_score"),
            ("rule_gap", "rule_score"),
            ("history_gap", "history_score"),
        ):
            expected[label] = legacy._quantized_delta(leader[score_field], competitor[score_field])
        if any(delta[name] != value for name, value in expected.items()):
            raise ValueError("report competition differs from the frozen ranking")


def _payload_domains(payload: dict[str, Any], *, version: int) -> None:
    """Original report pure domains, with a separate /2 query reference namespace.

    Called only on complete detached exact JSON shapes. No encoders, model
    validators, property access, replay or qualification occurs in this helper.
    """
    for name, field in (
        ("hypotheses", "guard_result_id"),
        ("predicates", "predicate_result_id"),
        ("query_sessions", "sequence_number"),
        ("graph_snapshots", "snapshot_sha256"),
        ("condensations", "condensation_id"),
        ("onsets", "onset_id"),
        ("precedences", "precedence_id"),
        ("competition_deltas", "competing_rank"),
    ):
        ids = [row[field] for row in payload[name]]
        if ids != sorted(ids):
            raise ValueError("report payload records must be canonically ordered")
        if name in ("hypotheses", "predicates") and len(ids) != len(set(ids)):
            raise ValueError("report guard and predicate IDs must be unique")
    _sorted_ids(payload["degradation_reason_codes"], label="degradation reasons")
    grouping = payload["event_grouping"]
    ranking = payload["ranking_result"]
    case_id = payload["time_contract"]["opaque_case_id"]
    singles = (
        "candidate_universe",
        "focus_set",
        "entity_registry",
        "event_grouping",
        "event_strength",
        "role_compilation",
        "family_aggregation",
        "propagation_scoring",
        "rule_scoring",
        "ranking_result",
        "lineage_index",
    )
    scoped = [payload[name] for name in singles]
    scoped.extend(
        row
        for name in (
            "hypotheses",
            "query_sessions",
            "graph_snapshots",
            "condensations",
            "onsets",
            "precedences",
        )
        for row in payload[name]
    )
    if any(row["opaque_case_id"] != case_id for row in scoped):
        raise ValueError("report payload crosses opaque case scopes")
    candidates = {row["service_entity_key"] for row in payload["candidate_universe"]["candidates"]}
    universe_hash = payload["candidate_universe"]["universe_sha256"]
    if any(
        payload[name]["candidate_universe_sha256"] != universe_hash
        for name in ("focus_set", "ranking_result", "history_evaluation")
    ):
        raise ValueError("report payload candidate and ranking scopes differ")
    for rows in (
        ranking["scores"],
        payload["family_aggregation"]["candidates"],
        payload["propagation_scoring"]["candidate_scores"],
        payload["rule_scoring"]["candidates"],
        payload["history_evaluation"]["candidate_scores"],
    ):
        if {row["candidate_service_key"] for row in rows} != candidates or len(rows) != len(
            candidates
        ):
            raise ValueError("report ranking channels must cover C exactly")
    for field, channel, channel_field in (
        ("family_aggregation_result_sha256", "family_aggregation", "result_sha256"),
        ("propagation_scoring_result_sha256", "propagation_scoring", "result_sha256"),
        (
            "propagation_authoritative_input_sha256",
            "propagation_scoring",
            "authoritative_input_sha256",
        ),
        ("rule_scoring_result_sha256", "rule_scoring", "result_sha256"),
        ("rule_authoritative_input_sha256", "rule_scoring", "authoritative_input_sha256"),
        ("history_evaluation_result_sha256", "history_evaluation", "result_sha256"),
        ("history_authoritative_input_sha256", "history_evaluation", "authoritative_input_sha256"),
    ):
        if ranking[field] != payload[channel][channel_field]:
            raise ValueError("report ranking references a different evidence channel")
    observations = {row["observation_id"] for row in grouping["observations"]}
    events = {row["event_group_id"] for row in grouping["event_groups"]}
    predicates = {row["predicate_result_id"] for row in payload["predicates"]}
    mechanisms = {
        row["mechanism_key"]: row["definition_sha256"]
        for row in payload["mechanism_library"]["definitions"]
    }
    facts = payload["hypotheses"]
    hypotheses = {row["hypothesis_id"] for row in facts if row["hypothesis_id"] is not None}
    guards = {row["guard_result_id"] for row in facts}
    roles = payload["role_compilation"]
    relations = {
        *(row["claim_id"] for row in grouping["association_claims"]),
        *(edge["edge_id"] for graph in payload["graph_snapshots"] for edge in graph["edges"]),
        *(key for item in payload["condensations"] for key in item["selected_edge_ids"]),
        *(
            edge["condensed_edge_id"]
            for item in payload["condensations"]
            for edge in item["condensed_edges"]
        ),
        *(row["relation_id"] for row in (*roles["proposals"], *roles["assertions"])),
    }
    requests, results, derived = _query_indexes(payload)
    legal_observations = observations | (set(derived) if version == 2 else set())
    score_impacts = {
        key
        for row in roles["assertions"]
        for key in row["provenance_ids"]
        if len(key) == 28
        and key.startswith("sci_")
        and all(c in "0123456789abcdef" for c in key[4:])
    }
    if version == 2:
        seen: set[str] = set()
        for collection_ids in (
            observations,
            events,
            predicates,
            hypotheses,
            guards,
            relations,
            set(requests),
            set(results),
            set(derived),
        ):
            if seen & collection_ids:
                raise ValueError("report record ID collision across source collections")
            seen.update(collection_ids)
        for row in derived.values():
            if (
                row["opaque_case_id"] != case_id
                or row["candidate_service_key"] not in candidates
                or not set(row["source_observation_ids"]) <= observations
                or not set(row["source_relation_ids"]) <= relations
            ):
                raise ValueError("report derived observation is outside its frozen sources")
    scope = payload["scope_interval"]
    for fact in facts:
        _hypothesis_domains(fact)
        candidate, mechanism = fact["candidate_service_key"], fact["mechanism_key"]
        if candidate is not None and candidate not in candidates:
            raise ValueError("report hypothesis references a candidate outside C")
        if mechanism is not None and (
            mechanism not in mechanisms
            or fact["mechanism_definition_sha256"] != mechanisms[mechanism]
        ):
            raise ValueError("report hypothesis references an unknown or different mechanism")
        if any(
            row["start_unix_ns"] < scope["start_unix_ns"]
            or row["end_unix_ns"] > scope["end_unix_ns"]
            for row in fact["time_intervals"]
        ):
            raise ValueError("report hypothesis interval falls outside case scope")
        for name, legal in (
            ("observation_ids", legal_observations),
            ("relation_ids", relations),
            ("event_group_ids", events),
            ("necessary_predicate_result_ids", predicates),
            ("optional_predicate_result_ids", predicates),
            ("refuting_predicate_result_ids", predicates),
            ("alternative_hypothesis_ids", hypotheses),
            ("verification_request_ids", set(requests)),
        ):
            if not set(fact[name]) <= legal:
                raise ValueError(f"report hypothesis references unknown {name}")
        if fact["score_impact_id"] is not None and fact["score_impact_id"] not in score_impacts:
            raise ValueError("report hypothesis references an unknown score impact")
    if not set(roles["guard_result_ids"]) <= guards:
        raise ValueError("role compilation references an unavailable report hypothesis")
    event, family = payload["event_strength"], payload["family_aggregation"]
    if (
        event["event_grouping_result_sha256"] != grouping["result_sha256"]
        or event["role_compilation_id"] != roles["role_compilation_id"]
        or event["role_compilation_result_sha256"] != roles["result_sha256"]
        or {row["event_group_id"] for row in event["group_strengths"]} != events
        or any(
            row["candidate_service_key"] not in candidates or row["event_group_id"] not in events
            for row in event["candidate_event_adjustments"]
        )
    ):
        raise ValueError("report event strength differs from its frozen evidence inputs")
    if (
        family["event_grouping_result_sha256"] != grouping["result_sha256"]
        or family["event_strength_result_id"] != event["event_strength_result_id"]
        or family["event_strength_result_sha256"] != event["result_sha256"]
    ):
        raise ValueError("report family aggregation differs from its frozen event inputs")
    pending = list(payload["predicates"])
    while pending:
        row = pending.pop()
        if not set(row["observation_ids"]) <= legal_observations or (
            version == 2 and not set(row["verification_result_ids"]) <= results.keys()
        ):
            raise ValueError("report predicate references unknown observation or query result")
        pending.extend(row["operand_states"])
    if any(not set(row["event_group_ids"]) <= events for row in ranking["scores"]):
        raise ValueError("report score references an unknown event group")
    for delta in payload["competition_deltas"]:
        if (
            delta["leading_service_key"] == delta["competing_service_key"]
            or delta["ordering_decided_by_tie_break"] != delta["evidence_tied"]
        ):
            raise ValueError("report competition differs from its candidate or tie domain")
        for name in (
            "leading_unique_event_group_ids",
            "competing_unique_event_group_ids",
            "leading_accepted_hypothesis_ids",
            "competing_accepted_hypothesis_ids",
        ):
            _sorted_ids(delta[name], label=name)
    _competition_domains(payload)
    if version == 2:
        selection = payload["scoring_selection"]
        accepted = bool(selection["eligible_guard_result_ids"])
        if payload["scoring_path"] != ("accepted_roles" if accepted else "no_eligible_roles"):
            raise ValueError("report scoring path differs from its selection")
        for field, old, new in (
            ("role_compilation", RoleCompilationResult, NoRoleCompilationResult),
            ("event_strength", EventStrengthResult, NoRoleEventStrengthResult),
            ("family_aggregation", FamilyAggregationResult, NoRoleFamilyAggregationResult),
            (
                "propagation_scoring",
                PropagationScoringResult,
                NoRolePropagationScoringResult,
            ),
            ("ranking_result", ReferenceRankingResult, NoRoleReferenceRankingResult),
        ):
            cls = old if accepted else new
            if payload[field]["schema_version"] != cls.model_fields["schema_version"].default:
                raise ValueError("report result types cross scoring paths")
        if (
            selection["opaque_case_id"] != case_id
            or payload["lineage_index"]["scoring_selection_sha256"] != selection["selection_sha256"]
        ):
            raise ValueError("report selection differs from case or lineage")
        eligible = {f["guard_result_id"] for f in facts if f["final_disposition"] == "accepted"}
        if (
            set(selection["eligible_guard_result_ids"]) != eligible
            or set(selection["excluded_guard_result_ids"]) != guards - eligible
        ):
            raise ValueError("report hypotheses differ from complete scoring selection")


def _context_domains(context: dict[str, Any], payload: dict[str, Any]) -> None:
    """Compare explicit source identities; digest recomputation remains separate."""
    for field, expected in (
        ("opaque_case_id", payload["time_contract"]["opaque_case_id"]),
        ("payload_sha256", payload["payload_sha256"]),
        ("source_ranking_bundle_sha256", payload["lineage_index"]["replay_bundle_sha256"]),
        ("source_ranking_result_sha256", payload["ranking_result"]["result_sha256"]),
        ("source_top_k_sha256", payload["ranking_result"]["top_k"]["ranking_sha256"]),
        ("source_lineage_sha256", payload["lineage_index"]["lineage_sha256"]),
    ):
        if context[field] != expected:
            raise ValueError("report context differs from its payload source identity")
    if (
        context["schema_version"] == FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
        and context["source_scoring_selection_sha256"]
        != payload["scoring_selection"]["selection_sha256"]
    ):
        raise ValueError("report context differs from its scoring selection")


def _report_document(raw: str | bytes | bytearray, cls: type[BaseModel]) -> dict[str, Any]:
    plain = _audit_document(raw, cls)
    expanded = _json_with_defaults(plain, cls)
    if cls is FrozenReportPayloadV2 or cls is legacy.FrozenReportPayload:
        _payload_domains(expanded, version=2 if cls is FrozenReportPayloadV2 else 1)
    elif cls is FrozenReportContextV4 or cls is legacy.FrozenReportContext:
        legacy._parse_utc_seconds(expanded["frozen_at_utc"], label="report context freeze time")
        payload = _report_document(
            expanded["payload_json"],
            FrozenReportPayloadV2 if cls is FrozenReportContextV4 else legacy.FrozenReportPayload,
        )
        _context_domains(expanded, payload)
    else:
        raise ValueError("report JSON requires a fixed code-owned report contract")
    return plain


def _validate_report_json(cls: type[Any], raw_input: str | bytes | bytearray, **kwargs: Any) -> Any:
    if cls is not FrozenReportPayloadV2 and cls is not FrozenReportContextV4:
        raise ValueError("report JSON requires an exact new report class")
    raw = _json_bytes(raw_input)
    try:
        _report_document(raw, cls)
        return super(cls, cls).model_validate_json(raw, **kwargs)
    finally:
        if raw != _json_bytes(raw_input):
            raise ValueError("report JSON changed during validation")


class FrozenReportPayloadV2(StrictFrozenModel):
    """All report facts from one complete /2 audit, retaining real result types."""

    schema_version: Literal["nurse.v2.frozen-report-payload/2"] = (
        FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION
    )
    time_contract: CaseTimeContract
    scope_interval: TimeInterval
    candidate_universe: CandidateUniverse
    focus_set: SemanticFocusSet
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    event_strength: EventStrengthResult | NoRoleEventStrengthResult
    mechanism_library: MechanismLibrary
    hypotheses: tuple[legacy.ReportHypothesisFact, ...]
    predicates: tuple[PredicateResult, ...]
    query_sessions: tuple[CaseQuerySessionResult, ...]
    role_compilation: RoleCompilationResult | NoRoleCompilationResult
    graph_snapshots: tuple[GraphSnapshot, ...]
    condensations: tuple[RelationSCCCondensation, ...]
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    family_aggregation: FamilyAggregationResult | NoRoleFamilyAggregationResult
    propagation_scoring: PropagationScoringResult | NoRolePropagationScoringResult
    rule_set: RuleSet
    rule_scoring: RuleScoringResult
    history_evaluation: HistoryEvaluationResult
    ranking_result: ReferenceRankingResult | NoRoleReferenceRankingResult
    competition_deltas: tuple[legacy.ReportCompetitionDelta, ...]
    lineage_index: CaseAuditLineageIndexV2
    degradation_reason_codes: tuple[str, ...]
    scoring_selection: CaseScoringSelection
    scoring_path: Literal["accepted_roles", "no_eligible_roles"]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        return cast(Self, _validate_report_json(cls, json_data, **kwargs))

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        return _exact_field(cls, value, info)

    @model_validator(mode="after")
    def canonical_competition(self) -> Self:
        if self.competition_deltas != legacy._build_competition_deltas(
            cast(Any, self.ranking_result), self.hypotheses
        ):
            raise ValueError("report competition differs from its frozen ranking")
        return self

    @model_validator(mode="wrap")
    @classmethod
    def safe_state(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        # Declared after the after-validator so its entire handler, including
        # the final encoder, remains inside the original-input finally guard.
        original = _report_snapshot(value)
        try:
            plain = original[0]
            if not _json_shape(plain, FrozenReportPayloadV2):
                raise ValueError("report payload requires exact declared fields and domains")
            _payload_domains(_json_with_defaults(plain, FrozenReportPayloadV2), version=2)
            return handler(value)
        finally:
            _report_unchanged(value, original)

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

    def verify(self) -> bool:
        try:
            if type(self) is not FrozenReportPayloadV2:
                return False
            original = _report_snapshot(self)
            try:
                _payload_domains(cast(dict[str, Any], original[0]), version=2)
                rebuilt = FrozenReportPayloadV2.model_validate_json(
                    canonical_json_bytes(original[0])
                )
                if rebuilt.competition_deltas != legacy._build_competition_deltas(
                    cast(Any, rebuilt.ranking_result), rebuilt.hypotheses
                ):
                    return False
                structural = (
                    rebuilt.propagation_scoring,
                    rebuilt.rule_scoring,
                    rebuilt.history_evaluation,
                    rebuilt.ranking_result,
                )
                checks = (
                    rebuilt.candidate_universe.verify,
                    rebuilt.focus_set.verify,
                    rebuilt.entity_registry.verify,
                    rebuilt.event_grouping.verify,
                    rebuilt.event_strength.verify,
                    rebuilt.mechanism_library.verify,
                    rebuilt.role_compilation.verify,
                    rebuilt.family_aggregation.verify,
                    rebuilt.rule_set.verify,
                    rebuilt.lineage_index.verify,
                    rebuilt.scoring_selection.verify,
                    *(item.verify for item in rebuilt.predicates),
                    *(item.verify for item in rebuilt.query_sessions),
                    *(item.verify for item in rebuilt.graph_snapshots),
                    *(item.verify for item in rebuilt.condensations),
                    *(item.verify for item in rebuilt.onsets),
                    *(item.verify for item in rebuilt.precedences),
                    *(item.verify for item in rebuilt.competition_deltas),
                )
                if not all(item.verify_structure() for item in structural) or not all(
                    check() for check in checks
                ):
                    return False
                unsigned = dict(cast(dict[str, Any], original[0]))
                digest = unsigned.pop("payload_sha256")
                return hmac.compare_digest(digest, sha256_hex(canonical_json_bytes(unsigned)))
            finally:
                _report_unchanged(self, original)
        except _ERRORS:
            return False


class FrozenReportContextV4(StrictFrozenModel):
    """Canonical report context retaining the immutable source's true version."""

    schema_version: Literal["nurse.v2.frozen-report-context/4"] = (
        FROZEN_REPORT_CONTEXT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    experiment_version: str = Field(min_length=1)
    run_mode: RunMode
    source_audit_schema_version: Literal["nurse.v2.immutable-case-audit-package/2"] = (
        "nurse.v2.immutable-case-audit-package/2"
    )
    source_ranking_bundle_schema_version: Literal["nurse.v2.case-ranking-replay-bundle/2"] = (
        "nurse.v2.case-ranking-replay-bundle/2"
    )
    source_lineage_schema_version: Literal["nurse.v2.case-audit-lineage-index/2"] = (
        "nurse.v2.case-audit-lineage-index/2"
    )
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scoring_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_json: str = Field(min_length=2)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    stage: Literal["CONTEXT_FROZEN"] = "CONTEXT_FROZEN"
    ranking_mutable: Literal[False] = False
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        return cast(Self, _validate_report_json(cls, json_data, **kwargs))

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        return _exact_field(cls, value, info)

    @model_validator(mode="after")
    def canonical_payload(self) -> Self:
        plain = _report_document(self.payload_json, FrozenReportPayloadV2)
        if canonical_json_bytes(plain).decode("utf-8") != self.payload_json:
            raise ValueError("report payload must contain canonical JSON")
        return self

    @model_validator(mode="wrap")
    @classmethod
    def safe_state(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        # Wrap the complete Python/constructor validation, not a normalized
        # replacement or only the JSON entry point's immutable byte copy.
        original = _report_snapshot(value)
        try:
            state = original[0]
            if not _json_shape(state, FrozenReportContextV4):
                raise ValueError("report context requires exact declared fields and domains")
            plain = cast(dict[str, Any], state)
            legacy._parse_utc_seconds(plain["frozen_at_utc"], label="report context freeze time")
            payload = _report_document(plain["payload_json"], FrozenReportPayloadV2)
            _context_domains(_json_with_defaults(plain, FrozenReportContextV4), payload)
            return handler(value)
        finally:
            _report_unchanged(value, original)

    @property
    def payload(self) -> FrozenReportPayloadV2:
        if type(self) is not FrozenReportContextV4:
            raise ValueError("report payload requires the exact context identity")
        original = _report_snapshot(self)
        try:
            return FrozenReportPayloadV2.model_validate_json(
                cast(dict[str, Any], original[0])["payload_json"]
            )
        finally:
            _report_unchanged(self, original)

    def verify(self) -> bool:
        try:
            if type(self) is not FrozenReportContextV4:
                return False
            original = _report_snapshot(self)
            try:
                state = cast(dict[str, Any], original[0])
                payload_state = _report_document(state["payload_json"], FrozenReportPayloadV2)
                _context_domains(state, payload_state)
                legacy._parse_utc_seconds(
                    state["frozen_at_utc"], label="report context freeze time"
                )
                rebuilt = FrozenReportContextV4.model_validate_json(canonical_json_bytes(state))
                payload = rebuilt.payload
                if not payload.verify() or any(
                    left != right
                    for left, right in (
                        (rebuilt.payload_sha256, payload.payload_sha256),
                        (rebuilt.opaque_case_id, payload.opaque_case_id),
                        (
                            rebuilt.source_ranking_bundle_sha256,
                            payload.lineage_index.replay_bundle_sha256,
                        ),
                        (
                            rebuilt.source_ranking_result_sha256,
                            payload.ranking_result.result_sha256,
                        ),
                        (rebuilt.source_top_k_sha256, payload.top_k.ranking_sha256),
                        (rebuilt.source_lineage_sha256, payload.lineage_index.lineage_sha256),
                        (
                            rebuilt.source_scoring_selection_sha256,
                            payload.scoring_selection.selection_sha256,
                        ),
                    )
                ):
                    return False
                unsigned = dict(state)
                digest = unsigned.pop("context_sha256")
                return hmac.compare_digest(digest, sha256_hex(canonical_json_bytes(unsigned)))
            finally:
                _report_unchanged(self, original)
        except _ERRORS:
            return False


def _build_report_payload_v2(
    bundle: CaseRankingReplayBundleV2, lineage: CaseAuditLineageIndexV2
) -> FrozenReportPayloadV2:
    facts = legacy._build_report_hypothesis_facts(cast(Any, bundle))
    values: dict[str, Any] = {
        name: getattr(bundle.evidence, name)
        for name in (
            "time_contract",
            "scope_interval",
            "candidate_universe",
            "focus_set",
            "entity_registry",
            "event_grouping",
            "event_strength",
            "mechanism_library",
            "query_sessions",
            "role_compilation",
            "graph_snapshots",
            "condensations",
            "onsets",
            "precedences",
            "scoring_selection",
            "scoring_path",
        )
    }
    values.update(
        {
            name: getattr(bundle, name)
            for name in (
                "family_aggregation",
                "propagation_scoring",
                "rule_set",
                "rule_scoring",
                "history_evaluation",
                "ranking_result",
                "degradation_reason_codes",
            )
        }
    )
    values.update(
        schema_version=FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION,
        hypotheses=facts,
        predicates=legacy._deduplicated_predicates(cast(Any, bundle)),
        competition_deltas=legacy._build_competition_deltas(
            cast(Any, bundle.ranking_result), facts
        ),
        lineage_index=lineage,
    )
    unsigned = _report_snapshot(values)[0]
    values["payload_sha256"] = sha256_hex(canonical_json_bytes(unsigned))
    payload = FrozenReportPayloadV2.model_validate(values)
    if not payload.verify():
        raise ValueError("frozen report payload failed structural verification")
    return payload


def freeze_report_context_from_audit_v2(
    *,
    audit_package: ImmutableCaseAuditPackageV2,
    expected_audit_sha256: str,
    frozen_at_utc: str,
) -> FrozenReportContextV4:
    """Freeze after complete source replay; never change the sealed input."""
    from nurse_rcl.v2.case_audit_read import read_case_audit

    if type(audit_package) is not ImmutableCaseAuditPackageV2:
        raise ValueError("new report freezing requires the exact /2 audit")
    if type(frozen_at_utc) is not str:
        raise ValueError("freeze time requires exact UTC second text")
    frozen = legacy._parse_utc_seconds(frozen_at_utc, label="report context freeze time")
    original = _report_snapshot(audit_package)
    try:
        _source_embedded_domains(
            _audit_document(
                cast(dict[str, Any], original[0])["replay_bundle_json"], CaseRankingReplayBundleV2
            )
        )
        view = read_case_audit(
            audit_package=audit_package, expected_audit_sha256=expected_audit_sha256
        )
        bundle = cast(CaseRankingReplayBundleV2, view.replay_bundle)
        lineage = cast(CaseAuditLineageIndexV2, view.lineage_index)
        sealed = audit_package.sealed_audit_package
        if frozen < legacy._parse_utc_seconds(sealed.sealed_at_utc, label="ranked audit seal time"):
            raise ValueError("report context must be frozen after the ranked audit")
        payload = _build_report_payload_v2(bundle, lineage)
        payload_json = canonical_json_bytes(_report_snapshot(payload)[0]).decode("utf-8")
        values = dict(
            schema_version=FROZEN_REPORT_CONTEXT_SCHEMA_VERSION,
            opaque_case_id=view.opaque_case_id,
            experiment_version=sealed.experiment_version,
            run_mode=sealed.run_mode,
            source_audit_schema_version="nurse.v2.immutable-case-audit-package/2",
            source_ranking_bundle_schema_version="nurse.v2.case-ranking-replay-bundle/2",
            source_lineage_schema_version="nurse.v2.case-audit-lineage-index/2",
            source_audit_sha256=view.audit_sha256,
            source_ranking_bundle_sha256=bundle.bundle_sha256,
            source_ranking_result_sha256=bundle.ranking_result.result_sha256,
            source_top_k_sha256=bundle.top_k.ranking_sha256,
            source_lineage_sha256=lineage.lineage_sha256,
            source_scoring_selection_sha256=bundle.evidence.scoring_selection.selection_sha256,
            payload_json=payload_json,
            payload_sha256=payload.payload_sha256,
            frozen_at_utc=frozen_at_utc,
            stage="CONTEXT_FROZEN",
            ranking_mutable=False,
        )
        context = FrozenReportContextV4.model_validate(
            {**values, "context_sha256": sha256_hex(canonical_json_bytes(values))}
        )
        if not context.verify():
            raise ValueError("frozen report context failed verification")
        return FrozenReportContextV4.model_validate_json(
            canonical_json_bytes(_report_snapshot(context)[0])
        )
    finally:
        _report_unchanged(audit_package, original)


__all__ = [
    "FROZEN_REPORT_CONTEXT_SCHEMA_VERSION",
    "FROZEN_REPORT_PAYLOAD_SCHEMA_VERSION",
    "FrozenReportContextV4",
    "FrozenReportPayloadV2",
    "freeze_report_context_from_audit_v2",
]
