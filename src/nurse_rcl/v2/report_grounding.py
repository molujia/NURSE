"""Deterministic grounding boundary for NURSE V2 report sentences."""

from __future__ import annotations

import hmac
import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.deterministic_report import (
    DeterministicSREReport,
    LabeledReportStatement,
    _guard_report_call,
    _require_report_order,
    require_deterministic_sre_report,
)
from nurse_rcl.v2.report_source_read import SupportedReportContext

REPORT_GROUNDING_CATALOG_SCHEMA_VERSION: Literal["nurse.v2.report-grounding-catalog/1"] = (
    "nurse.v2.report-grounding-catalog/1"
)
REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION: Literal["nurse.v2.report-grounding-validation/1"] = (
    "nurse.v2.report-grounding-validation/1"
)
# 3 fixed anchors + 99 competitors + 1 uncertainty + 32 verification
# actions + 8 mitigations + every one of the 10 declared deterministic limits.
MAX_REPORT_GROUNDING_SENTENCES: Literal[153] = 153

StatementType: TypeAlias = Literal["observation", "inference", "recommendation"]
SafeClaimStrength: TypeAlias = Literal[
    "frozen_observation",
    "ranking_inference",
    "conditional_recommendation",
]
RequestedClaimStrength: TypeAlias = Literal[
    "frozen_observation",
    "ranking_inference",
    "conditional_recommendation",
    "confirmed_cause",
    "guaranteed_outcome",
]
GroundingOwnerKind: TypeAlias = Literal[
    "run_mode",
    "top_k",
    "top_1_basis",
    "competition",
    "uncertainty",
    "verification_action",
    "conditional_mitigation",
    "limit",
]
GroundingDisposition: TypeAlias = Literal["accepted", "weakened", "rejected"]
GroundingViolationCode: TypeAlias = Literal[
    "unsupported_statement_anchor",
    "statement_type_mismatch",
    "template_key_mismatch",
    "claim_strength_mismatch",
    "unsupported_surface_text",
    "causal_overstatement",
    "guaranteed_outcome_overstatement",
]

_MAX_EXACT_MODEL_DEPTH: Literal[64] = 64
_MAX_EXACT_MODEL_NODES: Literal[100_000] = 100_000


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


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


def _safe_strength(statement_type: StatementType) -> SafeClaimStrength:
    if statement_type == "observation":
        return "frozen_observation"
    if statement_type == "inference":
        return "ranking_inference"
    return "conditional_recommendation"


def _requested_strength_is_compatible(
    statement_type: StatementType,
    requested_strength: RequestedClaimStrength,
) -> bool:
    if requested_strength == _safe_strength(statement_type):
        return True
    return (statement_type, requested_strength) in {
        ("inference", "confirmed_cause"),
        ("recommendation", "guaranteed_outcome"),
    }


_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")


def _identifier_surface(value: str) -> str:
    """Render a bounded data literal without permitting sentence injection."""

    normalized = unicodedata.normalize("NFKC", value)
    if normalized == value and _SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        return f"`{value}`"
    return f"`opaque-string sha256={_digest(value)}`"


def _joined_identifiers(values: Sequence[str]) -> str:
    if not values:
        return "none"
    rendered = ", ".join(_identifier_surface(value) for value in values)
    if len(rendered) <= 512:
        return rendered
    return f"{len(values)} frozen identifiers (sha256={_digest(tuple(values))})"


def _normalized_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_CAUSAL_OVERSTATEMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:is|are|was|were)\s+(?:the\s+)?(?:confirmed\s+)?root cause\b"),
    re.compile(r"\b(?:confirm(?:s|ed)?|prov(?:e|es|ed))\b.{0,48}\bcaus(?:e|es|ed|al)\b"),
    re.compile(r"(?:确定|确认|证实).{0,16}(?:导致|根因)"),
    re.compile(r"根因\s*(?:是|为)"),
)
_CAUSAL_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:false|not|never|unconfirmed|unproven|cannot|can't|doesn't|didn't)\b"
        r".{0,64}\b(?:root cause|caus(?:e|es|ed|al))\b"
    ),
    re.compile(r"(?:尚未|未能|无法|不能|没有|并非|不是|不).{0,24}(?:证实|确认|确定|导致|根因)"),
)
_GUARANTEED_OUTCOME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwill\s+(?:fix|resolve|eliminate|prevent)\b"),
    re.compile(r"\bguarantee(?:s|d)?\b.{0,48}\b(?:fix|resolution|recovery|outcome)\b"),
    re.compile(r"(?:一定会|必然|保证).{0,16}(?:解决|修复|恢复|消除|避免)"),
    re.compile(r"彻底解决"),
)
_GUARANTEED_OUTCOME_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:not|never|cannot|can't|doesn't|didn't)\b.{0,64}"
        r"\b(?:fix|resolve|eliminate|prevent|guarantee)\b"
    ),
    re.compile(
        r"(?:尚未|未能|无法|不能|没有|并非|不是|不).{0,24}(?:保证|解决|修复|恢复|消除|避免)"
    ),
)


def _has_unnegated_match(
    value: str,
    *,
    positive_patterns: Sequence[re.Pattern[str]],
    negative_patterns: Sequence[re.Pattern[str]],
) -> bool:
    return any(pattern.search(value) for pattern in positive_patterns) and not any(
        pattern.search(value) for pattern in negative_patterns
    )


def _overstatement_violations(
    proposal: ReportSentenceProposal,
    *,
    canonical_surface_text: str,
) -> tuple[GroundingViolationCode, ...]:
    codes: set[GroundingViolationCode] = set()
    normalized = _normalized_surface(proposal.surface_text)
    is_canonical_surface = normalized == _normalized_surface(canonical_surface_text)
    if not is_canonical_surface and _has_unnegated_match(
        normalized,
        positive_patterns=_CAUSAL_OVERSTATEMENT_PATTERNS,
        negative_patterns=_CAUSAL_NEGATION_PATTERNS,
    ):
        codes.add("causal_overstatement")
    if not is_canonical_surface and _has_unnegated_match(
        normalized,
        positive_patterns=_GUARANTEED_OUTCOME_PATTERNS,
        negative_patterns=_GUARANTEED_OUTCOME_NEGATION_PATTERNS,
    ):
        codes.add("guaranteed_outcome_overstatement")
    return tuple(sorted(codes))


class ReportGroundingAnchor(StrictFrozenModel):
    """One safe sentence form bound to a verified deterministic statement."""

    canonical_position: int = Field(ge=1, strict=True)
    statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    statement_type: StatementType
    template_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    owner_kind: GroundingOwnerKind
    owner_id: str = Field(min_length=1)
    allowed_claim_strength: SafeClaimStrength
    context_reference_ids: tuple[str, ...] = Field(min_length=1)
    canonical_surface_text: str = Field(min_length=1, max_length=4096)
    anchor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_fields(self) -> Self:
        if self.context_reference_ids != tuple(sorted(set(self.context_reference_ids))):
            raise ValueError("grounding anchor references must be sorted and unique")
        if self.allowed_claim_strength != _safe_strength(self.statement_type):
            raise ValueError("grounding anchor strength contradicts its statement type")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ReportGroundingAnchor or not _has_exact_model_fields(self):
                return False
            rebuilt = ReportGroundingAnchor.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(mode="json", exclude={"anchor_sha256"}, warnings="error")
            return hmac.compare_digest(rebuilt.anchor_sha256, _digest(unsigned))
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


class ReportGroundingCatalog(StrictFrozenModel):
    """Content-addressed set of every statement the frozen report can support."""

    schema_version: Literal["nurse.v2.report-grounding-catalog/1"] = (
        REPORT_GROUNDING_CATALOG_SCHEMA_VERSION
    )
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchors: tuple[ReportGroundingAnchor, ...] = Field(
        min_length=1, max_length=MAX_REPORT_GROUNDING_SENTENCES
    )
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_order(self) -> Self:
        if tuple(item.canonical_position for item in self.anchors) != tuple(
            range(1, len(self.anchors) + 1)
        ):
            raise ValueError("grounding catalog positions must be complete and ordered")
        if len({item.statement_id for item in self.anchors}) != len(self.anchors):
            raise ValueError("grounding catalog statement IDs must be unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ReportGroundingCatalog or not _has_exact_model_fields(self):
                return False
            rebuilt = ReportGroundingCatalog.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self or any(not item.verify() for item in rebuilt.anchors):
                return False
            unsigned = rebuilt.model_dump(mode="json", exclude={"catalog_sha256"}, warnings="error")
            return hmac.compare_digest(rebuilt.catalog_sha256, _digest(unsigned))
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


class ReportSentenceProposal(StrictFrozenModel):
    """Untrusted structured sentence request from a future report component."""

    position: int = Field(ge=1, strict=True)
    statement_type: StatementType
    anchor_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    requested_template_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    requested_claim_strength: RequestedClaimStrength
    surface_text: str = Field(min_length=1, max_length=4096)


class GroundedSentenceDecision(StrictFrozenModel):
    """Safe output for exactly one source sentence position."""

    decision_id: str = Field(pattern=r"^gdc_[0-9a-f]{24}$")
    position: int = Field(ge=1, strict=True)
    source_sentence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: GroundingDisposition
    violation_codes: tuple[GroundingViolationCode, ...]
    requested_anchor_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    grounded_statement_id: str | None = Field(default=None, pattern=r"^stm_[0-9a-f]{24}$")
    grounded_context_reference_ids: tuple[str, ...]
    effective_statement_type: StatementType | None = None
    effective_template_key: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,95}$")
    effective_claim_strength: SafeClaimStrength | None = None
    safe_surface_text: str | None = Field(default=None, min_length=1, max_length=4096)
    does_not_modify_sealed_ranking: Literal[True] = True
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_decision(self) -> Self:
        if self.violation_codes != tuple(sorted(set(self.violation_codes))):
            raise ValueError("grounding violation codes must be sorted and unique")
        if self.grounded_context_reference_ids != tuple(
            sorted(set(self.grounded_context_reference_ids))
        ):
            raise ValueError("grounded context references must be sorted and unique")
        if self.disposition == "accepted":
            if self.violation_codes:
                raise ValueError("accepted sentence cannot retain violations")
            if (
                None
                in (
                    self.grounded_statement_id,
                    self.effective_statement_type,
                    self.effective_template_key,
                    self.effective_claim_strength,
                    self.safe_surface_text,
                )
                or not self.grounded_context_reference_ids
            ):
                raise ValueError("accepted sentence must retain its complete grounding")
        elif self.disposition == "weakened":
            if not self.violation_codes or not set(self.violation_codes) <= {
                "causal_overstatement",
                "guaranteed_outcome_overstatement",
            }:
                raise ValueError("weakened sentence must retain only overstatement violations")
            if (
                None
                in (
                    self.grounded_statement_id,
                    self.effective_statement_type,
                    self.effective_template_key,
                    self.effective_claim_strength,
                    self.safe_surface_text,
                )
                or not self.grounded_context_reference_ids
            ):
                raise ValueError("weakened sentence must retain safe complete grounding")
        elif self.disposition == "rejected":
            if not self.violation_codes:
                raise ValueError("rejected sentence must retain at least one violation")
            if (
                any(
                    item is not None
                    for item in (
                        self.grounded_statement_id,
                        self.effective_statement_type,
                        self.effective_template_key,
                        self.effective_claim_strength,
                        self.safe_surface_text,
                    )
                )
                or self.grounded_context_reference_ids
            ):
                raise ValueError("rejected sentence cannot retain ungrounded output")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not GroundedSentenceDecision or not _has_exact_model_fields(self):
                return False
            rebuilt = GroundedSentenceDecision.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(
                mode="json", exclude={"decision_id", "decision_sha256"}, warnings="error"
            )
            digest = _digest(unsigned)
            return rebuilt.decision_id == f"gdc_{digest[:24]}" and hmac.compare_digest(
                rebuilt.decision_sha256, digest
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


class ReportGroundingValidation(StrictFrozenModel):
    """Content-addressed grounding outcome that cannot change the frozen rank."""

    schema_version: Literal["nurse.v2.report-grounding-validation/1"] = (
        REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION
    )
    validation_id: str = Field(pattern=r"^rgv_[0-9a-f]{24}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: tuple[GroundedSentenceDecision, ...] = Field(
        min_length=1, max_length=MAX_REPORT_GROUNDING_SENTENCES
    )
    accepted_count: int = Field(ge=0, strict=True)
    weakened_count: int = Field(ge=0, strict=True)
    rejected_count: int = Field(ge=0, strict=True)
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_decisions(self) -> Self:
        if tuple(item.position for item in self.decisions) != tuple(
            range(1, len(self.decisions) + 1)
        ):
            raise ValueError("grounding decisions must preserve every ordered position")
        if len({item.decision_id for item in self.decisions}) != len(self.decisions):
            raise ValueError("grounding decisions must have unique identities")
        expected_counts = (
            sum(item.disposition == "accepted" for item in self.decisions),
            sum(item.disposition == "weakened" for item in self.decisions),
            sum(item.disposition == "rejected" for item in self.decisions),
        )
        if expected_counts != (
            self.accepted_count,
            self.weakened_count,
            self.rejected_count,
        ):
            raise ValueError("grounding counts contradict sentence decisions")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ReportGroundingValidation or not _has_exact_model_fields(self):
                return False
            rebuilt = ReportGroundingValidation.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self or any(not item.verify() for item in rebuilt.decisions):
                return False
            unsigned = rebuilt.model_dump(
                mode="json", exclude={"validation_id", "validation_sha256"}, warnings="error"
            )
            digest = _digest(unsigned)
            return rebuilt.validation_id == f"rgv_{digest[:24]}" and hmac.compare_digest(
                rebuilt.validation_sha256, digest
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


def _grounding_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    if cls is GroundedSentenceDecision:
        for name in ("violation_codes", "grounded_context_reference_ids"):
            _require_report_order(state[name], label=name)
    elif cls is ReportGroundingValidation:
        decisions = state["decisions"]
        if [item["position"] for item in decisions] != list(range(1, len(decisions) + 1)):
            raise ValueError("grounding decisions must preserve every ordered position")
        if len({item["decision_id"] for item in decisions}) != len(decisions):
            raise ValueError("grounding decisions must have unique identities")


def _anchor(
    *,
    position: int,
    statement: LabeledReportStatement,
    owner_kind: GroundingOwnerKind,
    owner_id: str,
    canonical_surface_text: str,
) -> ReportGroundingAnchor:
    unsigned: dict[str, object] = {
        "canonical_position": position,
        "statement_id": statement.statement_id,
        "statement_type": statement.statement_type,
        "template_key": statement.template_key,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "allowed_claim_strength": _safe_strength(statement.statement_type),
        "context_reference_ids": statement.context_reference_ids,
        "canonical_surface_text": canonical_surface_text,
    }
    return ReportGroundingAnchor.model_validate({**unsigned, "anchor_sha256": _digest(unsigned)})


_LIMIT_SURFACES: dict[str, str] = {
    "ranking_score_not_probability": "Ranking scores are not root-cause probabilities.",
    "ranking_not_causal_confirmation": (
        "The frozen ranking does not confirm real-world causality."
    ),
    "frozen_decision_window": "The report uses only the frozen decision-time evidence window.",
    "report_cannot_change_ranking": "Report generation cannot change the sealed ranking.",
    "deterministic_template_not_agent_analysis": (
        "This statement is a deterministic template projection, not Agent analysis."
    ),
    "tie_break_not_evidence": "A stable tie-break is ordering metadata, not diagnostic evidence.",
    "degraded_evidence_scope": "The frozen evidence scope records at least one degradation.",
    "verification_actions_truncated": (
        "Verification actions were truncated at the declared deterministic limit."
    ),
    "conditional_mitigations_truncated": (
        "Conditional mitigations were truncated at the declared deterministic limit."
    ),
    "no_safe_conditional_mitigation": (
        "The frozen evidence supports no allowlisted conditional mitigation."
    ),
}


def _build_catalog_from_verified_report(
    report: DeterministicSREReport,
) -> ReportGroundingCatalog:
    seeds: list[tuple[LabeledReportStatement, GroundingOwnerKind, str, str]] = []
    seeds.append(
        (
            report.run_mode_statement,
            "run_mode",
            report.report_id,
            f"Run mode is {report.run_mode}; it comes from the frozen report context.",
        )
    )
    seeds.append(
        (
            report.top_k_statement,
            "top_k",
            report.report_id,
            (
                f"Frozen Top-K contains {len(report.top_k)} ranked candidates; its order "
                f"and scores are bound to sha256={report.source_top_k_sha256}."
            ),
        )
    )
    leader = report.top_1_basis
    seeds.append(
        (
            leader.statement,
            "top_1_basis",
            leader.score_sha256,
            (
                f"{_identifier_surface(leader.service_entity_key)} has the highest frozen "
                "ranking score "
                f"({leader.quantized_total_score}); it is the ranking priority, not a "
                "confirmed root cause."
            ),
        )
    )
    for competition in report.competition:
        if competition.evidence_tied:
            text = (
                f"{_identifier_surface(competition.leading_service_key)} and "
                f"{_identifier_surface(competition.competing_service_key)} are "
                f"evidence-tied; {_identifier_surface(competition.leading_service_key)} "
                "appears first only because of "
                "a non-evidence stable tie-break."
            )
        else:
            text = (
                f"{_identifier_surface(competition.leading_service_key)} ranks ahead of "
                f"{_identifier_surface(competition.competing_service_key)} by total score gap "
                f"{competition.total_score_gap}; this is a ranking comparison, not causal "
                "confirmation."
            )
        seeds.append(
            (
                competition.statement,
                "competition",
                competition.competition_id,
                text,
            )
        )
    uncertainty = report.uncertainty
    seeds.append(
        (
            uncertainty.statement,
            "uncertainty",
            report.report_id,
            (
                f"Leading evidence sufficiency is {uncertainty.evidence_sufficiency}; "
                f"tie={str(uncertainty.has_evidence_tie).lower()}, "
                f"unknown={str(uncertainty.has_unknown_evidence).lower()}, "
                f"conflicting={str(uncertainty.has_conflicting_evidence).lower()}, "
                f"missing={str(uncertainty.has_missing_evidence).lower()}, and degradation "
                f"codes are {_joined_identifiers(uncertainty.degradation_reason_codes)}."
            ),
        )
    )
    for action in report.verification_actions:
        seeds.append(
            (
                action.statement,
                "verification_action",
                action.action_id,
                (
                    f"For {_identifier_surface(action.target_service_key)}, perform only the "
                    "read-only check for predicate "
                    f"{_identifier_surface(action.predicate_result_id)}; false, true, and "
                    "unknown remain "
                    "possible, and the sealed ranking will not change."
                ),
            )
        )
    for mitigation in report.conditional_mitigations:
        seeds.append(
            (
                mitigation.statement,
                "conditional_mitigation",
                mitigation.mitigation_id,
                (
                    "If predicates "
                    f"{_joined_identifiers(mitigation.precondition_predicate_result_ids)} "
                    "hold, an operator may consider "
                    f"{_identifier_surface(mitigation.action_class)} for "
                    f"{_identifier_surface(mitigation.target_service_key)}; approval is "
                    "required, only reversible action is allowed, and it must not run "
                    "automatically."
                ),
            )
        )
    for limit_item in report.limits:
        seeds.append(
            (
                limit_item.statement,
                "limit",
                limit_item.limit_id,
                _LIMIT_SURFACES[limit_item.limit_code],
            )
        )
    anchors = tuple(
        _anchor(
            position=position,
            statement=statement,
            owner_kind=owner_kind,
            owner_id=owner_id,
            canonical_surface_text=surface,
        )
        for position, (statement, owner_kind, owner_id, surface) in enumerate(seeds, start=1)
    )
    unsigned: dict[str, object] = {
        "schema_version": REPORT_GROUNDING_CATALOG_SCHEMA_VERSION,
        "report_sha256": report.report_sha256,
        "context_sha256": report.context_sha256,
        "source_top_k_sha256": report.source_top_k_sha256,
        "anchors": tuple(item.model_dump(mode="json") for item in anchors),
    }
    catalog = ReportGroundingCatalog.model_validate(
        {**unsigned, "catalog_sha256": _digest(unsigned)}
    )
    if not catalog.verify():  # pragma: no cover - construction invariant
        raise AssertionError("report grounding catalog verification failed")
    return catalog


@_guard_report_call(roots=(("report", DeterministicSREReport),))
def build_report_grounding_catalog(
    *,
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> ReportGroundingCatalog:
    """Build safe sentence anchors after exact report/context/audit verification."""

    verified = require_deterministic_sre_report(
        report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    return _build_catalog_from_verified_report(verified)


def _accepted_decision(
    proposal: ReportSentenceProposal,
    anchor: ReportGroundingAnchor,
) -> GroundedSentenceDecision:
    source_digest = _digest(proposal.model_dump(mode="json", warnings="error"))
    unsigned: dict[str, object] = {
        "position": proposal.position,
        "source_sentence_sha256": source_digest,
        "disposition": "accepted",
        "violation_codes": (),
        "requested_anchor_statement_id": proposal.anchor_statement_id,
        "grounded_statement_id": anchor.statement_id,
        "grounded_context_reference_ids": anchor.context_reference_ids,
        "effective_statement_type": anchor.statement_type,
        "effective_template_key": anchor.template_key,
        "effective_claim_strength": anchor.allowed_claim_strength,
        "safe_surface_text": anchor.canonical_surface_text,
        "does_not_modify_sealed_ranking": True,
    }
    digest = _digest(unsigned)
    return GroundedSentenceDecision.model_validate(
        {
            **unsigned,
            "decision_id": f"gdc_{digest[:24]}",
            "decision_sha256": digest,
        }
    )


def _rejected_decision(
    proposal: ReportSentenceProposal,
    violation_code: GroundingViolationCode,
) -> GroundedSentenceDecision:
    source_digest = _digest(proposal.model_dump(mode="json", warnings="error"))
    unsigned: dict[str, object] = {
        "position": proposal.position,
        "source_sentence_sha256": source_digest,
        "disposition": "rejected",
        "violation_codes": (violation_code,),
        "requested_anchor_statement_id": proposal.anchor_statement_id,
        "grounded_statement_id": None,
        "grounded_context_reference_ids": (),
        "effective_statement_type": None,
        "effective_template_key": None,
        "effective_claim_strength": None,
        "safe_surface_text": None,
        "does_not_modify_sealed_ranking": True,
    }
    digest = _digest(unsigned)
    return GroundedSentenceDecision.model_validate(
        {
            **unsigned,
            "decision_id": f"gdc_{digest[:24]}",
            "decision_sha256": digest,
        }
    )


def _weakened_decision(
    proposal: ReportSentenceProposal,
    anchor: ReportGroundingAnchor,
    violation_codes: Sequence[GroundingViolationCode],
) -> GroundedSentenceDecision:
    source_digest = _digest(proposal.model_dump(mode="json", warnings="error"))
    unsigned: dict[str, object] = {
        "position": proposal.position,
        "source_sentence_sha256": source_digest,
        "disposition": "weakened",
        "violation_codes": tuple(sorted(set(violation_codes))),
        "requested_anchor_statement_id": proposal.anchor_statement_id,
        "grounded_statement_id": anchor.statement_id,
        "grounded_context_reference_ids": anchor.context_reference_ids,
        "effective_statement_type": anchor.statement_type,
        "effective_template_key": anchor.template_key,
        "effective_claim_strength": anchor.allowed_claim_strength,
        "safe_surface_text": anchor.canonical_surface_text,
        "does_not_modify_sealed_ranking": True,
    }
    digest = _digest(unsigned)
    return GroundedSentenceDecision.model_validate(
        {
            **unsigned,
            "decision_id": f"gdc_{digest[:24]}",
            "decision_sha256": digest,
        }
    )


@_guard_report_call(
    roots=(("report", DeterministicSREReport),),
    proposal_type=ReportSentenceProposal,
    proposal_limit=MAX_REPORT_GROUNDING_SENTENCES,
)
def validate_report_sentences(
    *,
    proposals: tuple[ReportSentenceProposal, ...],
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> ReportGroundingValidation:
    """Accept exact canonical sentences from an externally verified report."""

    if type(proposals) is not tuple:
        raise TypeError("report grounding proposals must be a concrete tuple")
    proposals_tuple = proposals
    if not proposals_tuple:
        raise ValueError("report grounding requires at least one sentence proposal")
    if len(proposals_tuple) > MAX_REPORT_GROUNDING_SENTENCES:
        raise ValueError(
            f"report grounding accepts at most {MAX_REPORT_GROUNDING_SENTENCES} sentence proposals"
        )
    exact_proposals: list[ReportSentenceProposal] = []
    for proposal in proposals_tuple:
        if type(proposal) is not ReportSentenceProposal or not _has_exact_model_fields(proposal):
            raise ValueError("report grounding requires exact ReportSentenceProposal objects")
        rebuilt_proposal = ReportSentenceProposal.model_validate(
            proposal.model_dump(mode="json", warnings="error")
        )
        if rebuilt_proposal != proposal:
            raise ValueError("report grounding requires exact ReportSentenceProposal objects")
        exact_proposals.append(rebuilt_proposal)
    proposals_tuple = tuple(exact_proposals)
    if tuple(item.position for item in proposals_tuple) != tuple(
        range(1, len(proposals_tuple) + 1)
    ):
        raise ValueError("report sentence positions must be complete and ordered")
    catalog = build_report_grounding_catalog(
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    by_statement = {item.statement_id: item for item in catalog.anchors}
    decisions: list[GroundedSentenceDecision] = []
    for proposal in proposals_tuple:
        anchor = by_statement.get(proposal.anchor_statement_id)
        violation: GroundingViolationCode | None = None
        if anchor is None:
            violation = "unsupported_statement_anchor"
        elif proposal.statement_type != anchor.statement_type:
            violation = "statement_type_mismatch"
        elif proposal.requested_template_key != anchor.template_key:
            violation = "template_key_mismatch"
        elif not _requested_strength_is_compatible(
            anchor.statement_type,
            proposal.requested_claim_strength,
        ):
            violation = "claim_strength_mismatch"
        if violation is not None:
            decisions.append(_rejected_decision(proposal, violation))
        else:
            if anchor is None:  # pragma: no cover - narrowed by the fixed guard order
                raise AssertionError("grounding anchor unexpectedly disappeared")
            overstatements = _overstatement_violations(
                proposal,
                canonical_surface_text=anchor.canonical_surface_text,
            )
            if overstatements:
                decisions.append(_weakened_decision(proposal, anchor, overstatements))
            elif proposal.requested_claim_strength != anchor.allowed_claim_strength:
                decisions.append(_rejected_decision(proposal, "claim_strength_mismatch"))
            elif _normalized_surface(proposal.surface_text) != _normalized_surface(
                anchor.canonical_surface_text
            ):
                decisions.append(_rejected_decision(proposal, "unsupported_surface_text"))
            else:
                decisions.append(_accepted_decision(proposal, anchor))
    decisions_tuple = tuple(decisions)
    request_digest = _digest(
        tuple(item.model_dump(mode="json", warnings="error") for item in proposals_tuple)
    )
    unsigned: dict[str, object] = {
        "schema_version": REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION,
        "report_sha256": report.report_sha256,
        "context_sha256": context.context_sha256,
        "source_top_k_sha256": report.source_top_k_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "source_request_sha256": request_digest,
        "decisions": tuple(item.model_dump(mode="json") for item in decisions_tuple),
        "accepted_count": sum(item.disposition == "accepted" for item in decisions_tuple),
        "weakened_count": sum(item.disposition == "weakened" for item in decisions_tuple),
        "rejected_count": sum(item.disposition == "rejected" for item in decisions_tuple),
        "ranking_mutable": False,
        "adds_ranking_facts": False,
    }
    digest = _digest(unsigned)
    result = ReportGroundingValidation.model_validate(
        {
            **unsigned,
            "validation_id": f"rgv_{digest[:24]}",
            "validation_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("report grounding validation verification failed")
    return result


@_guard_report_call(
    roots=(("validation", ReportGroundingValidation), ("report", DeterministicSREReport)),
    root_node_limits=(_MAX_EXACT_MODEL_NODES, 1_000_000),
    local_domains=_grounding_local_domains,
    proposal_type=ReportSentenceProposal,
    proposal_limit=MAX_REPORT_GROUNDING_SENTENCES,
)
def require_report_grounding_validation(
    validation: ReportGroundingValidation,
    *,
    proposals: tuple[ReportSentenceProposal, ...],
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> ReportGroundingValidation:
    """Rebuild grounding from external anchors and require exact identity."""

    if type(validation) is not ReportGroundingValidation or not validation.verify():
        raise ValueError("report grounding validation failed exact verification")
    expected = validate_report_sentences(
        proposals=proposals,
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    if (
        not hmac.compare_digest(validation.validation_sha256, expected.validation_sha256)
        or validation != expected
    ):
        raise ValueError("report grounding validation differs from rebuilt grounding validation")
    return ReportGroundingValidation.model_validate(
        validation.model_dump(mode="json", warnings="error")
    )


__all__: Sequence[str] = (
    "REPORT_GROUNDING_CATALOG_SCHEMA_VERSION",
    "REPORT_GROUNDING_VALIDATION_SCHEMA_VERSION",
    "GroundedSentenceDecision",
    "ReportGroundingAnchor",
    "ReportGroundingCatalog",
    "ReportGroundingValidation",
    "ReportSentenceProposal",
    "build_report_grounding_catalog",
    "require_report_grounding_validation",
    "validate_report_sentences",
)
