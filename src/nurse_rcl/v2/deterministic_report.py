"""Deterministic, report-only SRE projection over a frozen NURSE V2 rank."""

from __future__ import annotations

import hmac
import math
from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from functools import wraps
from inspect import signature
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar, cast

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_audit import ImmutableCaseAuditPackage
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.case_audit_v2 import _inspect, _json_shape, _model_state
from nurse_rcl.v2.case_scoring_selection import _exact
from nurse_rcl.v2.report_context import (
    FrozenReportContext,
    FrozenReportPayload,
    ReportCompetitionDelta,
    require_frozen_report_context,
)
from nurse_rcl.v2.report_context_v4 import (
    FrozenReportPayloadV2,
    _report_snapshot,
    _report_unchanged,
)
from nurse_rcl.v2.report_source_read import (
    SupportedReportContext,
    _preflight_report_sources,
    require_report_source_read_view,
)
from nurse_rcl.v2.runtime_schemas import RunMode
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION: Literal["nurse.v2.deterministic-sre-report/1"] = (
    "nurse.v2.deterministic-sre-report/1"
)

StatementType: TypeAlias = Literal["observation", "inference", "recommendation"]
EvidenceSufficiency: TypeAlias = Literal[
    "sufficient_for_priority",
    "limited",
    "mixed",
    "insufficient",
]
MitigationActionClass: TypeAlias = Literal[
    "reversible_workload_recovery",
    "reversible_resource_adjustment",
    "reversible_traffic_control",
    "reversible_dependency_failover",
    "reversible_fault_containment",
]

_MAX_VERIFICATION_ACTIONS: Literal[32] = 32
_MAX_CONDITIONAL_MITIGATIONS: Literal[8] = 8
_MAX_EXACT_MODEL_DEPTH: Literal[64] = 64
_MAX_EXACT_MODEL_NODES: Literal[1_000_000] = 1_000_000
_QUANTUM = Decimal("0.000000000001")

_VerificationActionSeed: TypeAlias = tuple[str, str, str, tuple[str, ...]]
_MitigationSeed: TypeAlias = tuple[
    str,
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
]
_SupportedReportPayload: TypeAlias = FrozenReportPayload | FrozenReportPayloadV2
_CallParameters = ParamSpec("_CallParameters")
_CallResult = TypeVar("_CallResult")
_LocalReportDomains: TypeAlias = Callable[[type[BaseModel], dict[str, Any]], None]
_ReportInputSnapshot: TypeAlias = tuple[object, tuple[object, ...]]


def _report_input_snapshot(
    value: object,
    *,
    model_type: type[BaseModel],
    max_nodes: int,
    local_domains: _LocalReportDomains | None = None,
) -> _ReportInputSnapshot:
    """Snapshot one report-only root within its original exact-model budget.

    These static report model closures have no mapping-valued fields, so the
    shared walk's node count matches the original report counters (which also
    count mapping keys). Source roots retain their separate 500,000-node walk.
    No serializer, model validator, property or digest computation runs here.
    """
    if type(value) is not model_type:
        raise ValueError("report input requires its exact declared root type")
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(model_type,),
        label="report input",
        max_depth=_MAX_EXACT_MODEL_DEPTH,
        max_nodes=max_nodes,
        max_string_length=None,
        provenance_id_limit=None,
    )
    shape: list[object] = []

    def visit(current: object) -> object:
        kind = type(current)
        if kind is str or kind is int or kind is float or kind is bool or current is None:
            # Report projection can expand beyond the sources' 64 MiB budget.
            # Preserve scalar safety without adding a report/string byte cap.
            if kind is str:
                value = cast(str, current)
                try:
                    for start in range(0, len(value), 4096):
                        value[start : start + 4096].encode("utf-8")
                except UnicodeError:
                    raise ValueError("report input requires valid UTF-8 strings") from None
            elif kind is float and not math.isfinite(cast(float, current)):
                raise ValueError("report input requires finite numbers")
            shape.append((kind, repr(current)) if kind is float else kind)
            return current
        if kind is tuple or kind is list:
            rows = cast(Sequence[object], current)
            shape.extend((kind, len(rows)))
            return [visit(item) for item in rows]
        # Authorization came from the static root closure above, never kind.
        cls = cast(type[BaseModel], kind)
        state = _model_state(current, cls)
        marked = object.__getattribute__(current, "__pydantic_fields_set__")
        if any(
            field.is_required() and name not in marked for name, field in cls.model_fields.items()
        ):
            raise ValueError("report required fields must retain their field markers")
        if any(not _exact(item, cls.model_fields[name].annotation) for name, item in state.items()):
            raise ValueError("report model fields require exact declared types")
        shape.extend((kind, tuple(sorted(marked)), tuple(state)))
        document = {name: visit(item) for name, item in state.items()}
        _report_local_domains(cls, document)
        if local_domains is not None:
            local_domains(cls, document)
        return document

    plain = visit(value)
    if not _json_shape(plain, model_type):
        raise ValueError("report model fields violate their existing declared domains")
    return plain, tuple(shape)


def _require_report_order(values: list[Any], *, label: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _report_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Only local structural domains; original semantic validators stay authoritative."""
    fields: tuple[str, ...] = ()
    if cls is LabeledReportStatement or cls is ReportLimit:
        fields = ("context_reference_ids",)
    elif cls is CandidateEvidenceStatus:
        fields = (
            "supporting_event_group_ids",
            "supporting_hypothesis_ids",
            "supporting_predicate_result_ids",
            "refuting_predicate_result_ids",
            "conflicting_assertion_ids",
            "missing_reason_codes",
            "unknown_predicate_result_ids",
        )
    elif cls is ReportRankedCandidate:
        fields = (
            "event_group_ids",
            "propagation_path_ids",
            "rule_ids",
            "history_neighbor_ids",
            "accepted_hypothesis_ids",
        )
    elif cls is ReportUncertaintySummary or cls is DeterministicSREReport:
        fields = ("degradation_reason_codes",)
    elif cls is ReportVerificationAction:
        fields = ("verification_request_ids", "evidence_reference_ids")
    elif cls is ConditionalMitigation:
        fields = ("precondition_predicate_result_ids", "evidence_reference_ids")
    for name in fields:
        _require_report_order(state[name], label=name)
    if cls is not DeterministicSREReport:
        return
    ranked = state["top_k"]
    if [item["rank"] for item in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError("report Top-K ranks must be complete and ordered")
    if len({item["service_entity_key"] for item in ranked}) != len(ranked):
        raise ValueError("report Top-K candidates must be unique")
    for name, key, limit in (
        ("verification_actions", "action_id", _MAX_VERIFICATION_ACTIONS),
        ("conditional_mitigations", "mitigation_id", _MAX_CONDITIONAL_MITIGATIONS),
    ):
        if len(state[name]) > limit:
            raise ValueError(f"report {name} exceed the existing capacity limit")
        _require_report_order([item[key] for item in state[name]], label=name)
    limits = state["limits"]
    _require_report_order([item["limit_id"] for item in limits], label="report limit IDs")
    if len({item["limit_code"] for item in limits}) != len(limits):
        raise ValueError("report limit codes must be unique")


def _guard_report_call(
    *,
    roots: tuple[tuple[str, type[BaseModel]], ...] = (),
    root_node_limits: tuple[int, ...] = (),
    local_domains: _LocalReportDomains | None = None,
    proposal_type: type[BaseModel] | None = None,
    proposal_limit: int = 0,
    proposal_node_limit: int = 100_000,
    failure_codes: tuple[str, ...] = (),
) -> Callable[[Callable[_CallParameters, _CallResult]], Callable[_CallParameters, _CallResult]]:
    """Purely inspect every caller root before a report callback, then check every exit.

    Root classes are static code-owned declarations at the nine call sites. Neither
    detached snapshots nor a previously read view can authorize a public call.
    """
    node_limits = root_node_limits or (_MAX_EXACT_MODEL_NODES,) * len(roots)
    if len(node_limits) != len(roots):
        raise ValueError("report root budgets must match the static root declarations")

    def decorate(
        function: Callable[_CallParameters, _CallResult],
    ) -> Callable[_CallParameters, _CallResult]:
        call_signature = signature(function)

        @wraps(function)
        def guarded(*args: _CallParameters.args, **kwargs: _CallParameters.kwargs) -> _CallResult:
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            context, package = values["context"], values["audit_package"]
            pin = values["expected_context_sha256"]
            if pin is None:
                if (
                    type(context) is not FrozenReportContext
                    or type(package) is not ImmutableCaseAuditPackage
                ):
                    raise ValueError("new report sources require an independent context digest")
                # Legacy omitted-pin calls keep their original replay authority.
                # This value is only a pure preflight consistency check; it is
                # never supplied to the independently pinned source reader.
                pin = cast(dict[str, Any], _report_snapshot(context)[0])["context_sha256"]
            try:
                context_original, audit_original, _ = _preflight_report_sources(
                    context=context,
                    audit_package=package,
                    expected_audit_sha256=values["expected_audit_sha256"],
                    expected_context_sha256=pin,
                )
            except ValueError as error:
                if error.args == ("independent audit digest does not match the original source",):
                    raise ValueError(
                        "external audit digest does not match the original source"
                    ) from error
                raise
            originals: list[tuple[object, type[BaseModel], int, _ReportInputSnapshot]] = []
            for (name, cls), node_limit in zip(roots, node_limits, strict=True):
                label = {
                    "report": "deterministic report",
                    "validation": "report grounding validation",
                    "audit": "report generation audit",
                    "failure": "report generation failure audit",
                }[name]
                try:
                    original = _report_input_snapshot(
                        values[name],
                        model_type=cls,
                        max_nodes=node_limit,
                        local_domains=local_domains,
                    )
                except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as error:
                    raise ValueError(f"{label} failed exact verification: {error}") from error
                originals.append((values[name], cls, node_limit, original))
            if proposal_type is not None:
                proposals = values["proposals"]
                if type(proposals) is not tuple:
                    raise TypeError("report proposals must be a concrete tuple")
                if not proposals:
                    raise ValueError("report requires at least one sentence proposal")
                if len(proposals) > proposal_limit:
                    raise ValueError(f"report accepts at most {proposal_limit} sentence proposals")
                if any(type(proposal) is not proposal_type for proposal in proposals):
                    raise ValueError(f"report requires exact {proposal_type.__name__} objects")
                try:
                    for proposal in proposals:
                        original = _report_input_snapshot(
                            proposal,
                            model_type=proposal_type,
                            max_nodes=proposal_node_limit,
                            local_domains=local_domains,
                        )
                        originals.append((proposal, proposal_type, proposal_node_limit, original))
                except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"report requires exact {proposal_type.__name__} objects: {error}"
                    ) from error
            if proposal_type is not None and tuple(
                object.__getattribute__(item, "__dict__")["position"] for item in proposals
            ) != tuple(range(1, len(proposals) + 1)):
                raise ValueError("report proposal positions must be complete and ordered")
            if failure_codes:
                code, detail = values["failure_code"], values["failure_detail"]
                if type(code) is not str or code not in failure_codes:
                    raise ValueError("report generation failure code is invalid")
                if type(detail) is not str or not detail:
                    raise ValueError("report generation failure detail must be a non-empty string")
                if len(detail.encode("utf-8")) > 16_384:
                    raise ValueError("report generation failure detail exceeds 16384 UTF-8 bytes")
            scalars = (
                tuple(values[name] for name in ("failure_code", "failure_detail"))
                if failure_codes
                else ()
            )
            scalar_original = _inspect(scalars)
            try:
                return function(*args, **kwargs)
            finally:
                try:
                    _report_unchanged(context, context_original)
                finally:
                    try:
                        _report_unchanged(package, audit_original)
                    finally:
                        unchanged = True
                        # Check every original root even when another has changed.
                        for value, cls, node_limit, original in originals:
                            try:
                                current = _report_input_snapshot(
                                    value,
                                    model_type=cls,
                                    max_nodes=node_limit,
                                    local_domains=local_domains,
                                )
                                if current != original:
                                    unchanged = False
                            except (
                                AttributeError,
                                KeyError,
                                RecursionError,
                                TypeError,
                                ValueError,
                            ):
                                unchanged = False
                        try:
                            if _inspect(scalars) != scalar_original:
                                unchanged = False
                        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
                            unchanged = False
                        if not unchanged:
                            raise ValueError(
                                "report additional input content or representation "
                                "changed during reading"
                            )

        return guarded

    return decorate


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _quantized_delta(left: float | str, right: float | str) -> str:
    value = Decimal(str(left)) - Decimal(str(right))
    return format(value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def _has_exact_model_fields(value: object) -> bool:
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    active_container_ids: set[int] = set()
    node_count = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active_container_ids.discard(id(item))
            continue
        node_count += 1
        if node_count > _MAX_EXACT_MODEL_NODES or depth > _MAX_EXACT_MODEL_DEPTH:
            return False
        children: tuple[object, ...]
        if isinstance(item, BaseModel):
            declared = set(type(item).model_fields)
            if set(item.__dict__) != declared or item.__pydantic_extra__:
                return False
            children = tuple(getattr(item, field) for field in declared)
        elif isinstance(item, dict):
            children = tuple(child for key, entry in item.items() for child in (key, entry))
        elif isinstance(item, list | tuple | set | frozenset):
            children = tuple(item)
        else:
            continue
        identity = id(item)
        if identity in active_container_ids:
            return False
        active_container_ids.add(identity)
        stack.append((item, depth, True))
        stack.extend((child, depth + 1, False) for child in children)
    return True


class LabeledReportStatement(StrictFrozenModel):
    """A fixed-template statement whose variable facts are frozen references."""

    statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    statement_type: StatementType
    template_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    context_reference_ids: tuple[str, ...] = Field(min_length=1)
    statement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_references(self) -> Self:
        if self.context_reference_ids != tuple(sorted(set(self.context_reference_ids))):
            raise ValueError("statement references must be sorted and unique")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(
            mode="json",
            exclude={"statement_id", "statement_sha256"},
            warnings="error",
        )
        digest = _digest(unsigned)
        return self.statement_id == f"stm_{digest[:24]}" and hmac.compare_digest(
            self.statement_sha256,
            digest,
        )


class CandidateEvidenceStatus(StrictFrozenModel):
    """Non-compensating evidence categories for one candidate."""

    supporting_event_group_ids: tuple[str, ...]
    supporting_hypothesis_ids: tuple[str, ...]
    supporting_predicate_result_ids: tuple[str, ...]
    refuting_predicate_result_ids: tuple[str, ...]
    conflicting_assertion_ids: tuple[str, ...]
    missing_reason_codes: tuple[str, ...]
    unknown_predicate_result_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_categories(self) -> Self:
        values = (
            self.supporting_event_group_ids,
            self.supporting_hypothesis_ids,
            self.supporting_predicate_result_ids,
            self.refuting_predicate_result_ids,
            self.conflicting_assertion_ids,
            self.missing_reason_codes,
            self.unknown_predicate_result_ids,
        )
        if any(items != tuple(sorted(set(items))) for items in values):
            raise ValueError("candidate evidence categories must be sorted and unique")
        return self


class ReportRankedCandidate(StrictFrozenModel):
    """One unchanged Top-K row with its complete transparent score terms."""

    rank: int = Field(ge=1, strict=True)
    service_entity_key: str = Field(min_length=1)
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quantized_total_score: str = Field(pattern=r"^-?\d+\.\d{12}$")
    quantized_event_evidence_score: str = Field(pattern=r"^\d+\.\d{12}$")
    quantized_propagation_score: str = Field(pattern=r"^\d+\.\d{12}$")
    rule_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False, strict=True)
    history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    event_group_ids: tuple[str, ...]
    propagation_path_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    history_neighbor_ids: tuple[str, ...]
    accepted_hypothesis_ids: tuple[str, ...]
    evidence: CandidateEvidenceStatus
    evidence_tie_group_id: str = Field(pattern=r"^tie_[0-9a-f]{24}$")
    stable_tie_break_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tie_break_is_evidence: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_references(self) -> Self:
        for values in (
            self.event_group_ids,
            self.propagation_path_ids,
            self.rule_ids,
            self.history_neighbor_ids,
            self.accepted_hypothesis_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("ranked report references must be sorted and unique")
        if not set(self.evidence.supporting_event_group_ids) <= set(self.event_group_ids):
            raise ValueError("supporting events must belong to retained score event groups")
        if self.evidence.supporting_hypothesis_ids != self.accepted_hypothesis_ids:
            raise ValueError("ranked candidate evidence must retain accepted hypotheses")
        return self


class TopOneBasis(StrictFrozenModel):
    """Why the frozen ranker prioritized Top-1, without a causal claim."""

    service_entity_key: str = Field(min_length=1)
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quantized_total_score: str = Field(pattern=r"^-?\d+\.\d{12}$")
    claim_strength: Literal["ranking_priority_not_confirmed_cause"] = (
        "ranking_priority_not_confirmed_cause"
    )
    evidence_sufficiency: EvidenceSufficiency
    evidence: CandidateEvidenceStatus
    statement: LabeledReportStatement


class ReportCompetitionSummary(StrictFrozenModel):
    """One exact Top-1 comparison copied from frozen competition evidence."""

    competition_id: str = Field(pattern=r"^cmp_[0-9a-f]{24}$")
    source_competition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    leading_service_key: str = Field(min_length=1)
    competing_service_key: str = Field(min_length=1)
    competing_rank: int = Field(ge=2, strict=True)
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
    statement: LabeledReportStatement

    @model_validator(mode="after")
    def require_honest_tie(self) -> Self:
        if self.evidence_tied != self.ordering_decided_by_tie_break:
            raise ValueError("competition tie-break must be identified as non-evidence")
        return self


class ReportUncertaintySummary(StrictFrozenModel):
    """Case-level uncertainty retained without a compensating confidence score."""

    evidence_sufficiency: EvidenceSufficiency
    has_evidence_tie: bool
    has_unknown_evidence: bool
    has_conflicting_evidence: bool
    has_missing_evidence: bool
    degradation_reason_codes: tuple[str, ...]
    statement: LabeledReportStatement

    @model_validator(mode="after")
    def require_canonical_degradation(self) -> Self:
        if self.degradation_reason_codes != tuple(sorted(set(self.degradation_reason_codes))):
            raise ValueError("uncertainty degradation codes must be sorted and unique")
        return self


class ReportPredicateOutcome(StrictFrozenModel):
    """One possible result of a proposed read-only predicate recheck."""

    truth_value: Literal["false", "true", "unknown"]
    interpretation_key: Literal[
        "hypothesis_condition_not_observed",
        "hypothesis_condition_observed",
        "hypothesis_condition_unresolved",
    ]

    @model_validator(mode="after")
    def require_matching_interpretation(self) -> Self:
        expected = {
            "false": "hypothesis_condition_not_observed",
            "true": "hypothesis_condition_observed",
            "unknown": "hypothesis_condition_unresolved",
        }[self.truth_value]
        if self.interpretation_key != expected:
            raise ValueError("predicate outcome interpretation contradicts its truth value")
        return self


class ReportVerificationAction(StrictFrozenModel):
    """A read-only recommendation that can never rewrite the frozen rank."""

    action_id: str = Field(pattern=r"^vfa_[0-9a-f]{24}$")
    target_service_key: str = Field(min_length=1)
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    predicate_result_id: str = Field(pattern=r"^prd_[0-9a-f]{24}$")
    verification_request_ids: tuple[str, ...]
    action_class: Literal["read_only_diagnostic_check"] = "read_only_diagnostic_check"
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    possible_outcomes: tuple[ReportPredicateOutcome, ...] = Field(
        min_length=3,
        max_length=3,
    )
    do_not_execute_automatically: Literal[True] = True
    does_not_modify_sealed_ranking: Literal[True] = True
    statement: LabeledReportStatement
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_bounded_read_only_action(self) -> Self:
        if self.verification_request_ids != tuple(sorted(set(self.verification_request_ids))):
            raise ValueError("verification request references must be sorted and unique")
        if self.evidence_reference_ids != tuple(sorted(set(self.evidence_reference_ids))):
            raise ValueError("verification evidence references must be sorted and unique")
        if self.predicate_result_id not in self.evidence_reference_ids:
            raise ValueError("verification action must retain its predicate evidence")
        if tuple(item.truth_value for item in self.possible_outcomes) != (
            "false",
            "true",
            "unknown",
        ):
            raise ValueError("verification action must retain all three possible outcomes")
        if self.possible_outcomes != _predicate_outcomes():
            raise ValueError("verification action outcome interpretations must be fixed")
        if self.statement.statement_type != "recommendation":
            raise ValueError("verification action must remain a recommendation")
        if self.statement.template_key != "read_only_recheck_with_three_valued_outcome":
            raise ValueError("verification action must use the fixed read-only template")
        if self.statement.context_reference_ids != self.evidence_reference_ids:
            raise ValueError("verification statement must retain its action evidence")
        return self

    def verify(self) -> bool:
        if not self.statement.verify():
            return False
        unsigned = self.model_dump(
            mode="json",
            exclude={"action_id", "action_sha256"},
            warnings="error",
        )
        digest = _digest(unsigned)
        return self.action_id == f"vfa_{digest[:24]}" and hmac.compare_digest(
            self.action_sha256,
            digest,
        )


class ConditionalMitigation(StrictFrozenModel):
    """A non-executable mitigation class conditioned on frozen evidence."""

    mitigation_id: str = Field(pattern=r"^mit_[0-9a-f]{24}$")
    target_service_key: str = Field(min_length=1)
    hypothesis_id: str = Field(pattern=r"^hyp_[0-9a-f]{24}$")
    mechanism_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    action_class: MitigationActionClass
    precondition_predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    possible_outcomes: tuple[Literal["improves", "no_change", "worsens_or_unknown"], ...] = Field(
        min_length=3, max_length=3
    )
    operator_approval_required: Literal[True] = True
    reversible_action_only: Literal[True] = True
    do_not_execute_automatically: Literal[True] = True
    statement: LabeledReportStatement
    mitigation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_safe_conditional_action(self) -> Self:
        if self.precondition_predicate_result_ids != tuple(
            sorted(set(self.precondition_predicate_result_ids))
        ):
            raise ValueError("mitigation preconditions must be sorted and unique")
        if self.evidence_reference_ids != tuple(sorted(set(self.evidence_reference_ids))):
            raise ValueError("mitigation evidence references must be sorted and unique")
        if self.hypothesis_id not in self.evidence_reference_ids or not set(
            self.precondition_predicate_result_ids
        ) <= set(self.evidence_reference_ids):
            raise ValueError("mitigation must retain hypothesis and predicate evidence")
        if self.possible_outcomes != (
            "improves",
            "no_change",
            "worsens_or_unknown",
        ):
            raise ValueError("mitigation must disclose all bounded outcome classes")
        if self.statement.statement_type != "recommendation":
            raise ValueError("conditional mitigation must remain a recommendation")
        expected_action = _MITIGATION_ACTION_BY_MECHANISM.get(self.mechanism_key)
        if expected_action is None or self.action_class != expected_action:
            raise ValueError("mitigation action class contradicts its mechanism mapping")
        if (
            self.statement.template_key
            != "operator_approved_reversible_mitigation_if_evidence_holds"
        ):
            raise ValueError("conditional mitigation must use the fixed safety template")
        if self.statement.context_reference_ids != self.evidence_reference_ids:
            raise ValueError("mitigation statement must retain its action evidence")
        return self

    def verify(self) -> bool:
        if not self.statement.verify():
            return False
        unsigned = self.model_dump(
            mode="json",
            exclude={"mitigation_id", "mitigation_sha256"},
            warnings="error",
        )
        digest = _digest(unsigned)
        return self.mitigation_id == f"mit_{digest[:24]}" and hmac.compare_digest(
            self.mitigation_sha256,
            digest,
        )


class ReportLimit(StrictFrozenModel):
    """An explicit evidence or claim boundary attached to the report."""

    limit_id: str = Field(pattern=r"^lmt_[0-9a-f]{24}$")
    limit_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    severity: Literal["information", "warning"]
    context_reference_ids: tuple[str, ...] = Field(min_length=1)
    statement: LabeledReportStatement
    limit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_limit(self) -> Self:
        if self.context_reference_ids != tuple(sorted(set(self.context_reference_ids))):
            raise ValueError("report limit references must be sorted and unique")
        if self.statement.statement_type not in {"observation", "inference"}:
            raise ValueError("report limit cannot be an action recommendation")
        if self.statement.context_reference_ids != self.context_reference_ids:
            raise ValueError("limit statement must retain its frozen references")
        return self

    def verify(self) -> bool:
        if not self.statement.verify():
            return False
        unsigned = self.model_dump(
            mode="json",
            exclude={"limit_id", "limit_sha256"},
            warnings="error",
        )
        digest = _digest(unsigned)
        return self.limit_id == f"lmt_{digest[:24]}" and hmac.compare_digest(
            self.limit_sha256,
            digest,
        )


class DeterministicSREReport(StrictFrozenModel):
    """Content-addressed deterministic fallback report over one frozen context."""

    schema_version: Literal["nurse.v2.deterministic-sre-report/1"] = (
        DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION
    )
    report_id: str = Field(pattern=r"^drr_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ranking_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_mode: RunMode
    run_mode_statement: LabeledReportStatement
    degradation_reason_codes: tuple[str, ...]
    top_k: tuple[ReportRankedCandidate, ...] = Field(min_length=1)
    top_k_statement: LabeledReportStatement
    top_1_basis: TopOneBasis
    competition: tuple[ReportCompetitionSummary, ...]
    uncertainty: ReportUncertaintySummary
    verification_action_limit: Literal[32] = _MAX_VERIFICATION_ACTIONS
    verification_actions_truncated_count: int = Field(ge=0)
    verification_actions: tuple[ReportVerificationAction, ...]
    conditional_mitigation_limit: Literal[8] = _MAX_CONDITIONAL_MITIGATIONS
    conditional_mitigations_truncated_count: int = Field(ge=0)
    conditional_mitigations: tuple[ConditionalMitigation, ...]
    limits: tuple[ReportLimit, ...] = Field(min_length=4)
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_top_k(self) -> Self:
        if tuple(item.rank for item in self.top_k) != tuple(range(1, len(self.top_k) + 1)):
            raise ValueError("report Top-K ranks must be complete and ordered")
        if len({item.service_entity_key for item in self.top_k}) != len(self.top_k):
            raise ValueError("report Top-K candidates must be unique")
        leader = self.top_k[0]
        if (
            self.top_1_basis.service_entity_key != leader.service_entity_key
            or self.top_1_basis.score_sha256 != leader.score_sha256
            or self.top_1_basis.quantized_total_score != leader.quantized_total_score
            or self.top_1_basis.evidence != leader.evidence
        ):
            raise ValueError("Top-1 basis must match the first frozen Top-K row")
        if len(self.competition) != len(self.top_k) - 1:
            raise ValueError("report competition must cover every non-leading Top-K row")
        for competition, candidate in zip(
            self.competition,
            self.top_k[1:],
            strict=True,
        ):
            if (
                competition.leading_service_key != leader.service_entity_key
                or competition.competing_service_key != candidate.service_entity_key
                or competition.competing_rank != candidate.rank
            ):
                raise ValueError("report competition must match frozen Top-K order")
        if self.degradation_reason_codes != tuple(sorted(set(self.degradation_reason_codes))):
            raise ValueError("report degradation codes must be sorted and unique")
        if self.degradation_reason_codes != self.uncertainty.degradation_reason_codes:
            raise ValueError("report uncertainty must retain degradation codes")
        if self.verification_actions != tuple(
            sorted(self.verification_actions, key=lambda item: item.action_id)
        ) or len({item.action_id for item in self.verification_actions}) != len(
            self.verification_actions
        ):
            raise ValueError("verification actions must be canonically unique")
        if self.conditional_mitigations != tuple(
            sorted(self.conditional_mitigations, key=lambda item: item.mitigation_id)
        ) or len({item.mitigation_id for item in self.conditional_mitigations}) != len(
            self.conditional_mitigations
        ):
            raise ValueError("conditional mitigations must be canonically unique")
        if (
            len(self.verification_actions) > self.verification_action_limit
            or len(self.conditional_mitigations) > self.conditional_mitigation_limit
        ):
            raise ValueError("report actions exceed their declared limits")
        accepted_by_candidate = {
            item.service_entity_key: set(item.accepted_hypothesis_ids) for item in self.top_k
        }
        if any(
            item.hypothesis_id not in accepted_by_candidate.get(item.target_service_key, set())
            for item in self.verification_actions
        ):
            raise ValueError("verification action is outside accepted Top-K hypotheses")
        if any(
            item.target_service_key != leader.service_entity_key
            or item.hypothesis_id not in set(leader.accepted_hypothesis_ids)
            for item in self.conditional_mitigations
        ):
            raise ValueError("mitigation is outside accepted Top-1 hypotheses")
        if self.limits != tuple(sorted(self.limits, key=lambda item: item.limit_id)):
            raise ValueError("report limits must be canonically ordered")
        if len({item.limit_id for item in self.limits}) != len(self.limits) or len(
            {item.limit_code for item in self.limits}
        ) != len(self.limits):
            raise ValueError("report limits must have unique identities and codes")
        required_limits = {
            "ranking_score_not_probability",
            "ranking_not_causal_confirmation",
            "frozen_decision_window",
            "report_cannot_change_ranking",
        }
        if not required_limits <= {item.limit_code for item in self.limits}:
            raise ValueError("report omits a mandatory limitation")
        _require_derived_report_consistency(self)
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not DeterministicSREReport or not _has_exact_model_fields(self):
                return False
            rebuilt = DeterministicSREReport.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            statements = (
                rebuilt.run_mode_statement,
                rebuilt.top_k_statement,
                rebuilt.top_1_basis.statement,
                rebuilt.uncertainty.statement,
                *(item.statement for item in rebuilt.competition),
            )
            if (
                any(not item.verify() for item in statements)
                or any(not item.verify() for item in rebuilt.verification_actions)
                or any(not item.verify() for item in rebuilt.conditional_mitigations)
                or any(not item.verify() for item in rebuilt.limits)
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"report_id", "report_sha256"},
                warnings="error",
            )
            digest = _digest(unsigned)
            return rebuilt.report_id == f"drr_{digest[:24]}" and hmac.compare_digest(
                rebuilt.report_sha256,
                digest,
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _accepted_hypotheses_by_candidate(
    payload: _SupportedReportPayload,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {}
    for fact in payload.hypotheses:
        if (
            fact.final_disposition == "accepted"
            and fact.candidate_service_key is not None
            and fact.hypothesis_id is not None
        ):
            result.setdefault(fact.candidate_service_key, set()).add(fact.hypothesis_id)
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _build_statement(
    *,
    statement_type: StatementType,
    template_key: str,
    context_reference_ids: Sequence[str],
) -> LabeledReportStatement:
    unsigned: dict[str, object] = {
        "statement_type": statement_type,
        "template_key": template_key,
        "context_reference_ids": tuple(sorted(set(context_reference_ids))),
    }
    digest = _digest(unsigned)
    return LabeledReportStatement.model_validate(
        {
            **unsigned,
            "statement_id": f"stm_{digest[:24]}",
            "statement_sha256": digest,
        }
    )


def _candidate_evidence(
    payload: _SupportedReportPayload,
    *,
    service_entity_key: str,
    supporting_event_group_ids: Sequence[str],
) -> CandidateEvidenceStatus:
    facts = tuple(
        item
        for item in payload.hypotheses
        if item.candidate_service_key == service_entity_key and item.hypothesis_id is not None
    )
    predicates = {item.predicate_result_id: item for item in payload.predicates}
    accepted = tuple(item for item in facts if item.final_disposition == "accepted")
    supporting_predicates: set[str] = set()
    refuting_predicates: set[str] = set()
    unknown_predicates: set[str] = set()
    missing_reasons = {
        *payload.degradation_reason_codes,
        *(reason for item in facts for reason in item.unknown_reasons),
    }
    for fact in facts:
        support_ids = {
            *fact.necessary_predicate_result_ids,
            *fact.optional_predicate_result_ids,
        }
        for predicate_id in support_ids:
            predicate = predicates[predicate_id]
            if predicate.truth_value == "true" and fact.final_disposition == "accepted":
                supporting_predicates.add(predicate_id)
            elif predicate.truth_value == "false" and (
                predicate_id in fact.necessary_predicate_result_ids
            ):
                refuting_predicates.add(predicate_id)
            elif predicate.truth_value == "unknown":
                unknown_predicates.add(predicate_id)
        for predicate_id in fact.refuting_predicate_result_ids:
            predicate = predicates[predicate_id]
            if predicate.truth_value == "true":
                refuting_predicates.add(predicate_id)
            elif predicate.truth_value == "unknown":
                unknown_predicates.add(predicate_id)
    conflicting_assertions = tuple(
        item.assertion_id
        for item in payload.role_compilation.assertions
        if item.candidate_service_key == service_entity_key
        and item.lifecycle_status == "conflicted"
    )
    if not supporting_event_group_ids:
        missing_reasons.add("no_score_contributing_event_group")
    if not accepted:
        missing_reasons.add("no_guard_accepted_hypothesis")
    if not supporting_predicates:
        missing_reasons.add("no_verified_predicate_support")
    return CandidateEvidenceStatus(
        supporting_event_group_ids=tuple(sorted(set(supporting_event_group_ids))),
        supporting_hypothesis_ids=tuple(
            sorted(item.hypothesis_id for item in accepted if item.hypothesis_id is not None)
        ),
        supporting_predicate_result_ids=tuple(sorted(supporting_predicates)),
        refuting_predicate_result_ids=tuple(sorted(refuting_predicates)),
        conflicting_assertion_ids=tuple(sorted(conflicting_assertions)),
        missing_reason_codes=tuple(sorted(missing_reasons)),
        unknown_predicate_result_ids=tuple(sorted(unknown_predicates)),
    )


def _evidence_sufficiency(
    *,
    candidate: ReportRankedCandidate,
    evidence: CandidateEvidenceStatus,
    degradation_reason_codes: Sequence[str],
) -> EvidenceSufficiency:
    if not (
        evidence.supporting_event_group_ids
        or candidate.propagation_path_ids
        or candidate.rule_ids
        or candidate.history_neighbor_ids
    ):
        return "insufficient"
    if evidence.refuting_predicate_result_ids or evidence.conflicting_assertion_ids:
        return "mixed"
    if (
        not evidence.supporting_hypothesis_ids
        or evidence.unknown_predicate_result_ids
        or evidence.missing_reason_codes
        or degradation_reason_codes
    ):
        return "limited"
    return "sufficient_for_priority"


def _require_derived_report_consistency(report: DeterministicSREReport) -> None:
    leader = report.top_k[0]
    expected_sufficiency = _evidence_sufficiency(
        candidate=leader,
        evidence=leader.evidence,
        degradation_reason_codes=report.degradation_reason_codes,
    )
    if (
        report.top_1_basis.evidence_sufficiency != expected_sufficiency
        or report.uncertainty.evidence_sufficiency != expected_sufficiency
    ):
        raise ValueError("report uncertainty evidence sufficiency differs from Top-1 evidence")
    expected_uncertainty = (
        any(item.evidence_tied for item in report.competition),
        any(item.evidence.unknown_predicate_result_ids for item in report.top_k),
        any(item.evidence.conflicting_assertion_ids for item in report.top_k),
        any(item.evidence.missing_reason_codes for item in report.top_k),
    )
    actual_uncertainty = (
        report.uncertainty.has_evidence_tie,
        report.uncertainty.has_unknown_evidence,
        report.uncertainty.has_conflicting_evidence,
        report.uncertainty.has_missing_evidence,
    )
    if actual_uncertainty != expected_uncertainty:
        raise ValueError("report uncertainty flags contradict retained evidence")

    expected_run_mode_statement = _build_statement(
        statement_type="observation",
        template_key="frozen_run_mode",
        context_reference_ids=(report.context_sha256, report.payload_sha256),
    )
    expected_top_k_statement = _build_statement(
        statement_type="observation",
        template_key="frozen_top_k_and_score_components",
        context_reference_ids=(
            report.source_top_k_sha256,
            *(item.score_sha256 for item in report.top_k),
        ),
    )
    evidence = leader.evidence
    uncertainty_references = (
        report.source_top_k_sha256,
        *report.degradation_reason_codes,
        *evidence.missing_reason_codes,
        *evidence.unknown_predicate_result_ids,
        *evidence.conflicting_assertion_ids,
        *(
            predicate_id
            for candidate in report.top_k
            for predicate_id in candidate.evidence.unknown_predicate_result_ids
        ),
        *(
            assertion_id
            for candidate in report.top_k
            for assertion_id in candidate.evidence.conflicting_assertion_ids
        ),
        *(
            reason_code
            for candidate in report.top_k
            for reason_code in candidate.evidence.missing_reason_codes
        ),
    )
    expected_uncertainty_statement = _build_statement(
        statement_type="observation",
        template_key="frozen_evidence_uncertainty_status",
        context_reference_ids=uncertainty_references,
    )
    if report.uncertainty.statement != expected_uncertainty_statement:
        raise ValueError("report uncertainty statement contradicts retained evidence")
    expected_top_one_statement = _build_statement(
        statement_type="inference",
        template_key="top1_is_ranking_priority_not_confirmed_cause",
        context_reference_ids=(
            report.source_top_k_sha256,
            leader.score_sha256,
            *evidence.supporting_event_group_ids,
            *evidence.supporting_hypothesis_ids,
            *evidence.supporting_predicate_result_ids,
            *evidence.refuting_predicate_result_ids,
            *evidence.conflicting_assertion_ids,
            *evidence.unknown_predicate_result_ids,
        ),
    )
    if (
        report.run_mode_statement != expected_run_mode_statement
        or report.top_k_statement != expected_top_k_statement
        or report.top_1_basis.statement != expected_top_one_statement
    ):
        raise ValueError("report statements contradict their structured frozen references")

    for competition, candidate in zip(report.competition, report.top_k[1:], strict=True):
        evidence_tied = candidate.evidence_tie_group_id == leader.evidence_tie_group_id
        leading_unique_events = tuple(
            sorted(set(leader.event_group_ids) - set(candidate.event_group_ids))
        )
        competing_unique_events = tuple(
            sorted(set(candidate.event_group_ids) - set(leader.event_group_ids))
        )
        source_unsigned: dict[str, object] = {
            "leading_service_key": leader.service_entity_key,
            "competing_service_key": candidate.service_entity_key,
            "competing_rank": candidate.rank,
            "leading_score_sha256": leader.score_sha256,
            "competing_score_sha256": candidate.score_sha256,
            "total_score_gap": _quantized_delta(
                leader.quantized_total_score,
                candidate.quantized_total_score,
            ),
            "event_evidence_gap": _quantized_delta(
                leader.quantized_event_evidence_score,
                candidate.quantized_event_evidence_score,
            ),
            "propagation_gap": _quantized_delta(
                leader.quantized_propagation_score,
                candidate.quantized_propagation_score,
            ),
            "rule_gap": _quantized_delta(leader.rule_score, candidate.rule_score),
            "history_gap": _quantized_delta(leader.history_score, candidate.history_score),
            "evidence_tied": evidence_tied,
            "ordering_decided_by_tie_break": evidence_tied,
            "leading_unique_event_group_ids": leading_unique_events,
            "competing_unique_event_group_ids": competing_unique_events,
            "leading_accepted_hypothesis_ids": leader.accepted_hypothesis_ids,
            "competing_accepted_hypothesis_ids": candidate.accepted_hypothesis_ids,
        }
        source_digest = _digest(source_unsigned)
        template_key = (
            "evidence_tie_ordered_by_non_evidence_hash"
            if evidence_tied
            else "top1_competition_score_difference"
        )
        expected_statement = _build_statement(
            statement_type="inference",
            template_key=template_key,
            context_reference_ids=(
                f"cmp_{source_digest[:24]}",
                leader.score_sha256,
                candidate.score_sha256,
                *leading_unique_events,
                *competing_unique_events,
                *leader.accepted_hypothesis_ids,
                *candidate.accepted_hypothesis_ids,
            ),
        )
        expected_values = (
            f"cmp_{source_digest[:24]}",
            source_digest,
            source_unsigned["total_score_gap"],
            source_unsigned["event_evidence_gap"],
            source_unsigned["propagation_gap"],
            source_unsigned["rule_gap"],
            source_unsigned["history_gap"],
            evidence_tied,
            evidence_tied,
            leading_unique_events,
            competing_unique_events,
            leader.accepted_hypothesis_ids,
            candidate.accepted_hypothesis_ids,
            expected_statement,
        )
        actual_values = (
            competition.competition_id,
            competition.source_competition_sha256,
            competition.total_score_gap,
            competition.event_evidence_gap,
            competition.propagation_gap,
            competition.rule_gap,
            competition.history_gap,
            competition.evidence_tied,
            competition.ordering_decided_by_tie_break,
            competition.leading_unique_event_group_ids,
            competition.competing_unique_event_group_ids,
            competition.leading_accepted_hypothesis_ids,
            competition.competing_accepted_hypothesis_ids,
            competition.statement,
        )
        if actual_values != expected_values:
            raise ValueError("report competition contradicts frozen Top-K score evidence")

    score_by_service = {item.service_entity_key: item.score_sha256 for item in report.top_k}
    for action in report.verification_actions:
        expected_references = tuple(
            sorted(
                {
                    score_by_service[action.target_service_key],
                    action.hypothesis_id,
                    action.predicate_result_id,
                    *action.verification_request_ids,
                }
            )
        )
        if action.evidence_reference_ids != expected_references:
            raise ValueError(
                "report verification action evidence contradicts its frozen Top-K inputs"
            )

    expected_limits = _build_limits(
        context_sha256=report.context_sha256,
        payload_sha256=report.payload_sha256,
        source_top_k_sha256=report.source_top_k_sha256,
        degradation_reason_codes=report.degradation_reason_codes,
        evidence_tie_competition_ids=tuple(
            item.competition_id for item in report.competition if item.evidence_tied
        ),
        verification_actions_truncated_count=report.verification_actions_truncated_count,
        conditional_mitigations_truncated_count=(report.conditional_mitigations_truncated_count),
        has_conditional_mitigations=bool(report.conditional_mitigations),
    )
    if report.limits != expected_limits:
        raise ValueError("report limits contradict their conditional report state")


def _build_top_one_basis(
    payload: _SupportedReportPayload,
    top_k: tuple[ReportRankedCandidate, ...],
) -> TopOneBasis:
    leader = top_k[0]
    evidence = leader.evidence
    sufficiency = _evidence_sufficiency(
        candidate=leader,
        evidence=evidence,
        degradation_reason_codes=payload.degradation_reason_codes,
    )
    return TopOneBasis(
        service_entity_key=leader.service_entity_key,
        score_sha256=leader.score_sha256,
        quantized_total_score=leader.quantized_total_score,
        evidence_sufficiency=sufficiency,
        evidence=evidence,
        statement=_build_statement(
            statement_type="inference",
            template_key="top1_is_ranking_priority_not_confirmed_cause",
            context_reference_ids=(
                payload.top_k.ranking_sha256,
                leader.score_sha256,
                *evidence.supporting_event_group_ids,
                *evidence.supporting_hypothesis_ids,
                *evidence.supporting_predicate_result_ids,
                *evidence.refuting_predicate_result_ids,
                *evidence.conflicting_assertion_ids,
                *evidence.unknown_predicate_result_ids,
            ),
        ),
    )


def _build_competition_summary(
    delta: ReportCompetitionDelta,
) -> ReportCompetitionSummary:
    template_key = (
        "evidence_tie_ordered_by_non_evidence_hash"
        if delta.evidence_tied
        else "top1_competition_score_difference"
    )
    return ReportCompetitionSummary(
        competition_id=delta.competition_id,
        source_competition_sha256=delta.competition_sha256,
        leading_service_key=delta.leading_service_key,
        competing_service_key=delta.competing_service_key,
        competing_rank=delta.competing_rank,
        total_score_gap=delta.total_score_gap,
        event_evidence_gap=delta.event_evidence_gap,
        propagation_gap=delta.propagation_gap,
        rule_gap=delta.rule_gap,
        history_gap=delta.history_gap,
        evidence_tied=delta.evidence_tied,
        ordering_decided_by_tie_break=delta.ordering_decided_by_tie_break,
        leading_unique_event_group_ids=delta.leading_unique_event_group_ids,
        competing_unique_event_group_ids=delta.competing_unique_event_group_ids,
        leading_accepted_hypothesis_ids=(delta.leading_accepted_hypothesis_ids),
        competing_accepted_hypothesis_ids=(delta.competing_accepted_hypothesis_ids),
        statement=_build_statement(
            statement_type="inference",
            template_key=template_key,
            context_reference_ids=(
                delta.competition_id,
                delta.leading_score_sha256,
                delta.competing_score_sha256,
                *delta.leading_unique_event_group_ids,
                *delta.competing_unique_event_group_ids,
                *delta.leading_accepted_hypothesis_ids,
                *delta.competing_accepted_hypothesis_ids,
            ),
        ),
    )


def _build_uncertainty(
    payload: _SupportedReportPayload,
    *,
    top_one_basis: TopOneBasis,
    top_k: Sequence[ReportRankedCandidate],
) -> ReportUncertaintySummary:
    evidence = top_one_basis.evidence
    tied = any(item.evidence_tied for item in payload.competition_deltas)
    unknown = any(item.evidence.unknown_predicate_result_ids for item in top_k)
    conflicting = any(item.evidence.conflicting_assertion_ids for item in top_k)
    missing = any(item.evidence.missing_reason_codes for item in top_k)
    references = (
        payload.top_k.ranking_sha256,
        *payload.degradation_reason_codes,
        *evidence.missing_reason_codes,
        *evidence.unknown_predicate_result_ids,
        *evidence.conflicting_assertion_ids,
        *(
            predicate_id
            for candidate in top_k
            for predicate_id in candidate.evidence.unknown_predicate_result_ids
        ),
        *(
            assertion_id
            for candidate in top_k
            for assertion_id in candidate.evidence.conflicting_assertion_ids
        ),
        *(
            reason_code
            for candidate in top_k
            for reason_code in candidate.evidence.missing_reason_codes
        ),
    )
    return ReportUncertaintySummary(
        evidence_sufficiency=top_one_basis.evidence_sufficiency,
        has_evidence_tie=tied,
        has_unknown_evidence=unknown,
        has_conflicting_evidence=conflicting,
        has_missing_evidence=missing,
        degradation_reason_codes=payload.degradation_reason_codes,
        statement=_build_statement(
            statement_type="observation",
            template_key="frozen_evidence_uncertainty_status",
            context_reference_ids=references,
        ),
    )


def _predicate_outcomes() -> tuple[ReportPredicateOutcome, ...]:
    return (
        ReportPredicateOutcome(
            truth_value="false",
            interpretation_key="hypothesis_condition_not_observed",
        ),
        ReportPredicateOutcome(
            truth_value="true",
            interpretation_key="hypothesis_condition_observed",
        ),
        ReportPredicateOutcome(
            truth_value="unknown",
            interpretation_key="hypothesis_condition_unresolved",
        ),
    )


def _build_verification_action(
    *,
    target_service_key: str,
    score_sha256: str,
    hypothesis_id: str,
    predicate_result_id: str,
    verification_request_ids: Sequence[str],
) -> ReportVerificationAction:
    evidence_reference_ids = tuple(
        sorted(
            {
                score_sha256,
                hypothesis_id,
                predicate_result_id,
                *verification_request_ids,
            }
        )
    )
    statement = _build_statement(
        statement_type="recommendation",
        template_key="read_only_recheck_with_three_valued_outcome",
        context_reference_ids=evidence_reference_ids,
    )
    unsigned: dict[str, object] = {
        "target_service_key": target_service_key,
        "hypothesis_id": hypothesis_id,
        "predicate_result_id": predicate_result_id,
        "verification_request_ids": tuple(sorted(set(verification_request_ids))),
        "action_class": "read_only_diagnostic_check",
        "evidence_reference_ids": evidence_reference_ids,
        "possible_outcomes": tuple(item.model_dump(mode="json") for item in _predicate_outcomes()),
        "do_not_execute_automatically": True,
        "does_not_modify_sealed_ranking": True,
        "statement": statement.model_dump(mode="json"),
    }
    digest = _digest(unsigned)
    return ReportVerificationAction.model_validate(
        {**unsigned, "action_id": f"vfa_{digest[:24]}", "action_sha256": digest}
    )


def _build_verification_actions(
    payload: _SupportedReportPayload,
    *,
    top_k: Sequence[ReportRankedCandidate],
) -> tuple[tuple[ReportVerificationAction, ...], int]:
    scores = {item.service_entity_key: item.score_sha256 for item in top_k}
    retained: list[_VerificationActionSeed] = []
    retained_keys: set[_VerificationActionSeed] = set()
    truncated_count = 0
    for fact in payload.hypotheses:
        if (
            fact.final_disposition != "accepted"
            or fact.candidate_service_key not in scores
            or fact.hypothesis_id is None
        ):
            continue
        predicate_ids = tuple(
            sorted(
                {
                    *fact.necessary_predicate_result_ids,
                    *fact.optional_predicate_result_ids,
                    *fact.refuting_predicate_result_ids,
                }
            )
        )
        for predicate_id in predicate_ids:
            seed: _VerificationActionSeed = (
                fact.candidate_service_key,
                fact.hypothesis_id,
                predicate_id,
                fact.verification_request_ids,
            )
            if seed in retained_keys:
                continue
            if len(retained) < _MAX_VERIFICATION_ACTIONS:
                retained.append(seed)
                retained_keys.add(seed)
            else:
                truncated_count += 1
    actions = tuple(
        sorted(
            (
                _build_verification_action(
                    target_service_key=target_service_key,
                    score_sha256=scores[target_service_key],
                    hypothesis_id=hypothesis_id,
                    predicate_result_id=predicate_result_id,
                    verification_request_ids=verification_request_ids,
                )
                for (
                    target_service_key,
                    hypothesis_id,
                    predicate_result_id,
                    verification_request_ids,
                ) in retained
            ),
            key=lambda item: item.action_id,
        )
    )
    return actions, truncated_count


_MITIGATION_ACTION_BY_MECHANISM: dict[str, MitigationActionClass] = {
    "workload_unavailability": "reversible_workload_recovery",
    "local_resource_or_execution_pressure": "reversible_resource_adjustment",
    "incoming_load_pressure": "reversible_traffic_control",
    "received_dependency_failure": "reversible_dependency_failover",
    "local_storage_or_network_failure": "reversible_fault_containment",
    "unclassified_local_anomaly": "reversible_fault_containment",
}


def _build_mitigation(
    *,
    target_service_key: str,
    hypothesis_id: str,
    mechanism_key: str,
    precondition_predicate_result_ids: Sequence[str],
    evidence_reference_ids: Sequence[str],
) -> ConditionalMitigation:
    action_class = _MITIGATION_ACTION_BY_MECHANISM[mechanism_key]
    references = tuple(
        sorted(
            {
                hypothesis_id,
                *precondition_predicate_result_ids,
                *evidence_reference_ids,
            }
        )
    )
    statement = _build_statement(
        statement_type="recommendation",
        template_key="operator_approved_reversible_mitigation_if_evidence_holds",
        context_reference_ids=references,
    )
    unsigned: dict[str, object] = {
        "target_service_key": target_service_key,
        "hypothesis_id": hypothesis_id,
        "mechanism_key": mechanism_key,
        "action_class": action_class,
        "precondition_predicate_result_ids": tuple(sorted(set(precondition_predicate_result_ids))),
        "evidence_reference_ids": references,
        "possible_outcomes": ("improves", "no_change", "worsens_or_unknown"),
        "operator_approval_required": True,
        "reversible_action_only": True,
        "do_not_execute_automatically": True,
        "statement": statement.model_dump(mode="json"),
    }
    digest = _digest(unsigned)
    return ConditionalMitigation.model_validate(
        {
            **unsigned,
            "mitigation_id": f"mit_{digest[:24]}",
            "mitigation_sha256": digest,
        }
    )


def _build_conditional_mitigations(
    payload: _SupportedReportPayload,
    *,
    top_one_basis: TopOneBasis,
) -> tuple[tuple[ConditionalMitigation, ...], int]:
    leader_key = top_one_basis.service_entity_key
    predicates = {item.predicate_result_id: item for item in payload.predicates}
    retained: list[_MitigationSeed] = []
    retained_keys: set[_MitigationSeed] = set()
    truncated_count = 0
    for fact in payload.hypotheses:
        if (
            fact.final_disposition != "accepted"
            or fact.candidate_service_key != leader_key
            or fact.hypothesis_id is None
            or fact.mechanism_key is None
            or fact.mechanism_key not in _MITIGATION_ACTION_BY_MECHANISM
            or not fact.necessary_predicate_result_ids
            or any(
                predicates[item].truth_value != "true"
                for item in fact.necessary_predicate_result_ids
            )
        ):
            continue
        seed: _MitigationSeed = (
            leader_key,
            fact.hypothesis_id,
            fact.mechanism_key,
            fact.necessary_predicate_result_ids,
            (
                top_one_basis.score_sha256,
                *fact.observation_ids,
                *fact.event_group_ids,
            ),
        )
        if seed in retained_keys:
            continue
        if len(retained) < _MAX_CONDITIONAL_MITIGATIONS:
            retained.append(seed)
            retained_keys.add(seed)
        else:
            truncated_count += 1
    mitigations = tuple(
        sorted(
            (
                _build_mitigation(
                    target_service_key=target_service_key,
                    hypothesis_id=hypothesis_id,
                    mechanism_key=mechanism_key,
                    precondition_predicate_result_ids=precondition_predicate_result_ids,
                    evidence_reference_ids=evidence_reference_ids,
                )
                for (
                    target_service_key,
                    hypothesis_id,
                    mechanism_key,
                    precondition_predicate_result_ids,
                    evidence_reference_ids,
                ) in retained
            ),
            key=lambda item: item.mitigation_id,
        )
    )
    return mitigations, truncated_count


def _build_limit(
    *,
    limit_code: str,
    severity: Literal["information", "warning"],
    statement_type: Literal["observation", "inference"],
    context_reference_ids: Sequence[str],
) -> ReportLimit:
    references = tuple(sorted(set(context_reference_ids)))
    statement = _build_statement(
        statement_type=statement_type,
        template_key=f"limit_{limit_code}",
        context_reference_ids=references,
    )
    unsigned: dict[str, object] = {
        "limit_code": limit_code,
        "severity": severity,
        "context_reference_ids": references,
        "statement": statement.model_dump(mode="json"),
    }
    digest = _digest(unsigned)
    return ReportLimit.model_validate(
        {**unsigned, "limit_id": f"lmt_{digest[:24]}", "limit_sha256": digest}
    )


def _build_limits(
    *,
    context_sha256: str,
    payload_sha256: str,
    source_top_k_sha256: str,
    degradation_reason_codes: Sequence[str],
    evidence_tie_competition_ids: Sequence[str],
    verification_actions_truncated_count: int,
    conditional_mitigations_truncated_count: int,
    has_conditional_mitigations: bool,
) -> tuple[ReportLimit, ...]:
    base_references = (context_sha256, source_top_k_sha256)
    limits = [
        _build_limit(
            limit_code="ranking_score_not_probability",
            severity="warning",
            statement_type="inference",
            context_reference_ids=base_references,
        ),
        _build_limit(
            limit_code="ranking_not_causal_confirmation",
            severity="warning",
            statement_type="inference",
            context_reference_ids=base_references,
        ),
        _build_limit(
            limit_code="frozen_decision_window",
            severity="information",
            statement_type="observation",
            context_reference_ids=(context_sha256, payload_sha256),
        ),
        _build_limit(
            limit_code="report_cannot_change_ranking",
            severity="information",
            statement_type="observation",
            context_reference_ids=base_references,
        ),
        _build_limit(
            limit_code="deterministic_template_not_agent_analysis",
            severity="information",
            statement_type="observation",
            context_reference_ids=(context_sha256,),
        ),
    ]
    if evidence_tie_competition_ids:
        limits.append(
            _build_limit(
                limit_code="tie_break_not_evidence",
                severity="warning",
                statement_type="inference",
                context_reference_ids=(
                    source_top_k_sha256,
                    *evidence_tie_competition_ids,
                ),
            )
        )
    if degradation_reason_codes:
        limits.append(
            _build_limit(
                limit_code="degraded_evidence_scope",
                severity="warning",
                statement_type="observation",
                context_reference_ids=(
                    payload_sha256,
                    *degradation_reason_codes,
                ),
            )
        )
    if verification_actions_truncated_count:
        limits.append(
            _build_limit(
                limit_code="verification_actions_truncated",
                severity="warning",
                statement_type="observation",
                context_reference_ids=(context_sha256,),
            )
        )
    if conditional_mitigations_truncated_count:
        limits.append(
            _build_limit(
                limit_code="conditional_mitigations_truncated",
                severity="warning",
                statement_type="observation",
                context_reference_ids=(context_sha256,),
            )
        )
    if not has_conditional_mitigations:
        limits.append(
            _build_limit(
                limit_code="no_safe_conditional_mitigation",
                severity="warning",
                statement_type="inference",
                context_reference_ids=(context_sha256,),
            )
        )
    return tuple(sorted(limits, key=lambda item: item.limit_id))


def _build_top_k(payload: _SupportedReportPayload) -> tuple[ReportRankedCandidate, ...]:
    scores = {item.candidate_service_key: item for item in payload.scores}
    family_aggregations = {
        item.candidate_service_key: item for item in payload.family_aggregation.candidates
    }
    accepted = _accepted_hypotheses_by_candidate(payload)
    rows: list[ReportRankedCandidate] = []
    for entry in payload.top_k.entries:
        score = scores.get(entry.service_entity_key)
        if score is None or score.score_sha256 != entry.score_sha256:
            raise ValueError("report Top-K score does not match frozen ranking")
        family_aggregation = family_aggregations.get(entry.service_entity_key)
        if family_aggregation is None:
            raise ValueError("report Top-K candidate lacks frozen family aggregation")
        selected_families = set(family_aggregation.selected_family_keys)
        supporting_event_group_ids = tuple(
            sorted(
                event_group_id
                for maximum in family_aggregation.family_maxima
                if maximum.primary_signal_family in selected_families
                for event_group_id in maximum.contributing_event_group_ids
            )
        )
        if not set(supporting_event_group_ids) <= set(score.event_group_ids):
            raise ValueError("report supporting events are absent from the frozen score")
        evidence = _candidate_evidence(
            payload,
            service_entity_key=entry.service_entity_key,
            supporting_event_group_ids=supporting_event_group_ids,
        )
        rows.append(
            ReportRankedCandidate(
                rank=entry.rank,
                service_entity_key=entry.service_entity_key,
                score_sha256=score.score_sha256,
                quantized_total_score=score.quantized_total_score,
                quantized_event_evidence_score=(score.quantized_event_evidence_score),
                quantized_propagation_score=score.quantized_propagation_score,
                rule_score=score.rule_score,
                history_score=score.history_score,
                event_group_ids=score.event_group_ids,
                propagation_path_ids=score.propagation_path_ids,
                rule_ids=score.rule_ids,
                history_neighbor_ids=score.history_neighbor_ids,
                accepted_hypothesis_ids=accepted.get(entry.service_entity_key, ()),
                evidence=evidence,
                evidence_tie_group_id=entry.evidence_tie_group_id,
                stable_tie_break_sha256=entry.stable_tie_break_sha256,
                tie_break_is_evidence=False,
            )
        )
    return tuple(rows)


def _build_report_from_verified_context(
    context: SupportedReportContext,
) -> DeterministicSREReport:
    payload = context.payload
    top_k = _build_top_k(payload)
    top_one_basis = _build_top_one_basis(payload, top_k)
    competition = tuple(_build_competition_summary(item) for item in payload.competition_deltas)
    uncertainty = _build_uncertainty(
        payload,
        top_one_basis=top_one_basis,
        top_k=top_k,
    )
    verification_actions, verification_actions_truncated_count = _build_verification_actions(
        payload, top_k=top_k
    )
    conditional_mitigations, conditional_mitigations_truncated_count = (
        _build_conditional_mitigations(payload, top_one_basis=top_one_basis)
    )
    run_mode_statement = _build_statement(
        statement_type="observation",
        template_key="frozen_run_mode",
        context_reference_ids=(context.context_sha256, context.payload_sha256),
    )
    top_k_statement = _build_statement(
        statement_type="observation",
        template_key="frozen_top_k_and_score_components",
        context_reference_ids=(
            payload.top_k.ranking_sha256,
            *(item.score_sha256 for item in top_k),
        ),
    )
    limits = _build_limits(
        context_sha256=context.context_sha256,
        payload_sha256=payload.payload_sha256,
        source_top_k_sha256=payload.top_k.ranking_sha256,
        degradation_reason_codes=payload.degradation_reason_codes,
        evidence_tie_competition_ids=tuple(
            item.competition_id for item in competition if item.evidence_tied
        ),
        verification_actions_truncated_count=verification_actions_truncated_count,
        conditional_mitigations_truncated_count=(conditional_mitigations_truncated_count),
        has_conditional_mitigations=bool(conditional_mitigations),
    )
    unsigned: dict[str, object] = {
        "schema_version": DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION,
        "opaque_case_id": context.opaque_case_id,
        "context_sha256": context.context_sha256,
        "payload_sha256": context.payload_sha256,
        "source_audit_sha256": context.source_audit_sha256,
        "source_ranking_result_sha256": context.source_ranking_result_sha256,
        "source_top_k_sha256": context.source_top_k_sha256,
        "run_mode": context.run_mode,
        "run_mode_statement": run_mode_statement.model_dump(mode="json"),
        "degradation_reason_codes": payload.degradation_reason_codes,
        "top_k": tuple(item.model_dump(mode="json") for item in top_k),
        "top_k_statement": top_k_statement.model_dump(mode="json"),
        "top_1_basis": top_one_basis.model_dump(mode="json"),
        "competition": tuple(item.model_dump(mode="json") for item in competition),
        "uncertainty": uncertainty.model_dump(mode="json"),
        "verification_action_limit": _MAX_VERIFICATION_ACTIONS,
        "verification_actions_truncated_count": (verification_actions_truncated_count),
        "verification_actions": tuple(
            item.model_dump(mode="json") for item in verification_actions
        ),
        "conditional_mitigation_limit": _MAX_CONDITIONAL_MITIGATIONS,
        "conditional_mitigations_truncated_count": (conditional_mitigations_truncated_count),
        "conditional_mitigations": tuple(
            item.model_dump(mode="json") for item in conditional_mitigations
        ),
        "limits": tuple(item.model_dump(mode="json") for item in limits),
        "ranking_mutable": False,
        "adds_ranking_facts": False,
    }
    digest = _digest(unsigned)
    report = DeterministicSREReport.model_validate(
        {**unsigned, "report_id": f"drr_{digest[:24]}", "report_sha256": digest}
    )
    if not report.verify():  # pragma: no cover - construction invariant
        raise AssertionError("deterministic SRE report verification failed")
    return report


@_guard_report_call()
def build_deterministic_sre_report(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> DeterministicSREReport:
    """Build a report only after exact replay of its external audit anchor."""

    verified: SupportedReportContext
    if expected_context_sha256 is None:
        verified = require_frozen_report_context(
            cast(FrozenReportContext, context),
            audit_package=cast(ImmutableCaseAuditPackage, audit_package),
            expected_audit_sha256=expected_audit_sha256,
        )
    else:
        verified = require_report_source_read_view(
            context=context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
            expected_context_sha256=expected_context_sha256,
        ).context
    return _build_report_from_verified_context(verified)


@_guard_report_call(roots=(("report", DeterministicSREReport),))
def require_deterministic_sre_report(
    report: DeterministicSREReport,
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> DeterministicSREReport:
    """Require exact report identity against an externally anchored context."""

    if type(report) is not DeterministicSREReport or not report.verify():
        raise ValueError("deterministic SRE report failed exact verification")
    expected = build_deterministic_sre_report(
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    if not hmac.compare_digest(report.report_sha256, expected.report_sha256) or (
        report != expected
    ):
        raise ValueError("deterministic SRE report differs from its frozen report context")
    return DeterministicSREReport.model_validate(report.model_dump(mode="json", warnings="error"))


__all__: Sequence[str] = (
    "DETERMINISTIC_SRE_REPORT_SCHEMA_VERSION",
    "CandidateEvidenceStatus",
    "ConditionalMitigation",
    "DeterministicSREReport",
    "LabeledReportStatement",
    "ReportCompetitionSummary",
    "ReportLimit",
    "ReportPredicateOutcome",
    "ReportRankedCandidate",
    "ReportUncertaintySummary",
    "ReportVerificationAction",
    "TopOneBasis",
    "build_deterministic_sre_report",
    "require_deterministic_sre_report",
)
