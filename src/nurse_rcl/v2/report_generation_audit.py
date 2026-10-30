"""Audit-only capture of raw report violations and deterministic fallbacks."""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.deterministic_report import (
    DeterministicSREReport,
    _guard_report_call,
    _require_report_order,
    require_deterministic_sre_report,
)
from nurse_rcl.v2.report_grounding import (
    MAX_REPORT_GROUNDING_SENTENCES,
    GroundingViolationCode,
    ReportGroundingValidation,
    ReportSentenceProposal,
    RequestedClaimStrength,
    StatementType,
    _grounding_local_domains,
    validate_report_sentences,
)
from nurse_rcl.v2.report_source_read import SupportedReportContext

REPORT_GENERATION_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.report-generation-audit/1"] = (
    "nurse.v2.report-generation-audit/1"
)
REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION: Literal[
    "nurse.v2.report-generation-failure-audit/1"
] = "nurse.v2.report-generation-failure-audit/1"

ViolationDisposition: TypeAlias = Literal["weakened", "rejected"]
ReportGenerationFailureCode: TypeAlias = Literal[
    "generation_unavailable",
    "malformed_response",
    "grounding_failed",
    "budget_exhausted",
]

_MAX_EXACT_MODEL_DEPTH: Literal[64] = 64
_MAX_EXACT_MODEL_NODES: Literal[100_000] = 100_000
_FAILURE_CODES = (
    "generation_unavailable",
    "malformed_response",
    "grounding_failed",
    "budget_exhausted",
)


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


def _exact_proposals(
    proposals: tuple[ReportSentenceProposal, ...],
) -> tuple[ReportSentenceProposal, ...]:
    if type(proposals) is not tuple:
        raise TypeError("report generation audit proposals must be a concrete tuple")
    if not proposals:
        raise ValueError("report generation audit requires at least one proposal")
    if len(proposals) > MAX_REPORT_GROUNDING_SENTENCES:
        raise ValueError(
            f"report generation audit accepts at most {MAX_REPORT_GROUNDING_SENTENCES} proposals"
        )
    exact: list[ReportSentenceProposal] = []
    for proposal in proposals:
        if type(proposal) is not ReportSentenceProposal or not _has_exact_model_fields(proposal):
            raise ValueError("report generation audit requires exact proposal objects")
        rebuilt = ReportSentenceProposal.model_validate(
            proposal.model_dump(mode="json", warnings="error")
        )
        if rebuilt != proposal:
            raise ValueError("report generation audit requires exact proposal objects")
        exact.append(rebuilt)
    if tuple(item.position for item in exact) != tuple(range(1, len(exact) + 1)):
        raise ValueError("report generation audit proposal positions must be complete and ordered")
    return tuple(exact)


class RawReportViolation(StrictFrozenModel):
    """One raw, untrusted proposal retained only for violation auditing."""

    position: int = Field(ge=1, strict=True)
    raw_proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_surface_text: str = Field(min_length=1, max_length=4096)
    requested_statement_type: StatementType
    requested_anchor_statement_id: str = Field(pattern=r"^stm_[0-9a-f]{24}$")
    requested_template_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    requested_claim_strength: RequestedClaimStrength
    disposition: ViolationDisposition
    violation_codes: tuple[GroundingViolationCode, ...] = Field(min_length=1)
    safe_replacement_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    audit_only: Literal[True] = True
    violation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_violation(self) -> Self:
        if self.violation_codes != tuple(sorted(set(self.violation_codes))):
            raise ValueError("raw report violation codes must be sorted and unique")
        if (self.disposition == "weakened") != (self.safe_replacement_sha256 is not None):
            raise ValueError("only weakened violations have a safe replacement")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not RawReportViolation or not _has_exact_model_fields(self):
                return False
            rebuilt = RawReportViolation.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            proposal = ReportSentenceProposal(
                position=rebuilt.position,
                statement_type=rebuilt.requested_statement_type,
                anchor_statement_id=rebuilt.requested_anchor_statement_id,
                requested_template_key=rebuilt.requested_template_key,
                requested_claim_strength=rebuilt.requested_claim_strength,
                surface_text=rebuilt.raw_surface_text,
            )
            if not hmac.compare_digest(
                rebuilt.raw_proposal_sha256,
                _digest(proposal.model_dump(mode="json", warnings="error")),
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json", exclude={"violation_sha256"}, warnings="error"
            )
            return hmac.compare_digest(rebuilt.violation_sha256, _digest(unsigned))
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


class ReportGenerationAudit(StrictFrozenModel):
    """Content-addressed raw violation audit with a safe grounding result."""

    schema_version: Literal["nurse.v2.report-generation-audit/1"] = (
        REPORT_GENERATION_AUDIT_SCHEMA_VERSION
    )
    audit_id: str = Field(pattern=r"^rga_[0-9a-f]{24}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounding_validation: ReportGroundingValidation
    raw_violations: tuple[RawReportViolation, ...] = Field(
        max_length=MAX_REPORT_GROUNDING_SENTENCES
    )
    total_sentence_count: int = Field(ge=1, le=MAX_REPORT_GROUNDING_SENTENCES, strict=True)
    raw_violation_count: int = Field(ge=0, le=MAX_REPORT_GROUNDING_SENTENCES, strict=True)
    weakened_count: int = Field(ge=0, le=MAX_REPORT_GROUNDING_SENTENCES, strict=True)
    rejected_count: int = Field(ge=0, le=MAX_REPORT_GROUNDING_SENTENCES, strict=True)
    raw_capture_precedes_grounding: Literal[True] = True
    audit_only: Literal[True] = True
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if tuple(item.position for item in self.raw_violations) != tuple(
            sorted({item.position for item in self.raw_violations})
        ):
            raise ValueError("raw report violations must be ordered and position-unique")
        if self.raw_violation_count != len(self.raw_violations):
            raise ValueError("raw report violation count contradicts retained records")
        if self.raw_violation_count != self.weakened_count + self.rejected_count:
            raise ValueError("raw report violation disposition counts are inconsistent")
        if (
            self.total_sentence_count != len(self.grounding_validation.decisions)
            or self.weakened_count != self.grounding_validation.weakened_count
            or self.rejected_count != self.grounding_validation.rejected_count
        ):
            raise ValueError("raw audit counts contradict grounding validation")
        if (
            self.source_request_sha256 != self.grounding_validation.source_request_sha256
            or self.raw_capture_sha256 != self.source_request_sha256
            or self.report_sha256 != self.grounding_validation.report_sha256
            or self.context_sha256 != self.grounding_validation.context_sha256
            or self.source_top_k_sha256 != self.grounding_validation.source_top_k_sha256
        ):
            raise ValueError("raw audit anchors contradict grounding validation")
        decisions = {item.position: item for item in self.grounding_validation.decisions}
        for violation in self.raw_violations:
            decision = decisions.get(violation.position)
            if decision is None:
                raise ValueError("raw violation does not reference a grounding decision")
            expected_replacement_sha256 = (
                _digest(decision.safe_surface_text)
                if decision.safe_surface_text is not None
                else None
            )
            if (
                violation.raw_proposal_sha256 != decision.source_sentence_sha256
                or violation.disposition != decision.disposition
                or violation.violation_codes != decision.violation_codes
                or violation.safe_replacement_sha256 != expected_replacement_sha256
            ):
                raise ValueError("raw violation contradicts its grounding decision")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ReportGenerationAudit or not _has_exact_model_fields(self):
                return False
            rebuilt = ReportGenerationAudit.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if (
                rebuilt != self
                or not rebuilt.grounding_validation.verify()
                or any(not item.verify() for item in rebuilt.raw_violations)
            ):
                return False
            unsigned = rebuilt.model_dump(
                mode="json", exclude={"audit_id", "audit_sha256"}, warnings="error"
            )
            digest = _digest(unsigned)
            return rebuilt.audit_id == f"rga_{digest[:24]}" and hmac.compare_digest(
                rebuilt.audit_sha256, digest
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


class ReportGenerationFailureAudit(StrictFrozenModel):
    """A report failure bound to the verified deterministic fallback report."""

    schema_version: Literal["nurse.v2.report-generation-failure-audit/1"] = (
        REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION
    )
    failure_id: str = Field(pattern=r"^rgf_[0-9a-f]{24}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_top_k_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_code: ReportGenerationFailureCode
    failure_detail_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fallback_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uses_deterministic_fallback: Literal[True] = True
    audit_only: Literal[True] = True
    ranking_mutable: Literal[False] = False
    adds_ranking_facts: Literal[False] = False
    failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_fallback_identity(self) -> Self:
        if self.fallback_report_sha256 != self.report_sha256:
            raise ValueError("report failure must bind the verified deterministic report")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not ReportGenerationFailureAudit or not _has_exact_model_fields(self):
                return False
            rebuilt = ReportGenerationFailureAudit.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(
                mode="json", exclude={"failure_id", "failure_sha256"}, warnings="error"
            )
            digest = _digest(unsigned)
            return rebuilt.failure_id == f"rgf_{digest[:24]}" and hmac.compare_digest(
                rebuilt.failure_sha256, digest
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


def _generation_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    _grounding_local_domains(cls, state)
    if cls is RawReportViolation:
        _require_report_order(state["violation_codes"], label="raw report violation codes")
    elif cls is ReportGenerationAudit:
        _require_report_order(
            [item["position"] for item in state["raw_violations"]],
            label="raw report violation positions",
        )


def _raw_violation(
    proposal: ReportSentenceProposal,
    *,
    validation: ReportGroundingValidation,
) -> RawReportViolation | None:
    decision = validation.decisions[proposal.position - 1]
    if decision.disposition == "accepted":
        return None
    if decision.disposition not in {"weakened", "rejected"}:
        raise AssertionError("unknown grounding disposition")
    safe_replacement_sha256 = (
        _digest(decision.safe_surface_text) if decision.safe_surface_text is not None else None
    )
    unsigned: dict[str, object] = {
        "position": proposal.position,
        "raw_proposal_sha256": decision.source_sentence_sha256,
        "raw_surface_text": proposal.surface_text,
        "requested_statement_type": proposal.statement_type,
        "requested_anchor_statement_id": proposal.anchor_statement_id,
        "requested_template_key": proposal.requested_template_key,
        "requested_claim_strength": proposal.requested_claim_strength,
        "disposition": decision.disposition,
        "violation_codes": decision.violation_codes,
        "safe_replacement_sha256": safe_replacement_sha256,
        "audit_only": True,
    }
    return RawReportViolation.model_validate({**unsigned, "violation_sha256": _digest(unsigned)})


@_guard_report_call(
    roots=(("report", DeterministicSREReport),),
    proposal_type=ReportSentenceProposal,
    proposal_limit=MAX_REPORT_GROUNDING_SENTENCES,
)
def capture_and_validate_report_generation(
    *,
    proposals: tuple[ReportSentenceProposal, ...],
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> ReportGenerationAudit:
    """Capture raw input first, then apply the safe deterministic grounding boundary."""

    exact = _exact_proposals(proposals)
    raw_payload = tuple(item.model_dump(mode="json", warnings="error") for item in exact)
    raw_capture_sha256 = _digest(raw_payload)
    grounding = validate_report_sentences(
        proposals=exact,
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    if not hmac.compare_digest(raw_capture_sha256, grounding.source_request_sha256):
        raise AssertionError("raw capture and grounding request digest diverged")
    raw_violations = tuple(
        violation
        for proposal in exact
        if (violation := _raw_violation(proposal, validation=grounding)) is not None
    )
    unsigned: dict[str, object] = {
        "schema_version": REPORT_GENERATION_AUDIT_SCHEMA_VERSION,
        "report_sha256": report.report_sha256,
        "context_sha256": context.context_sha256,
        "source_top_k_sha256": report.source_top_k_sha256,
        "source_request_sha256": grounding.source_request_sha256,
        "raw_capture_sha256": raw_capture_sha256,
        "grounding_validation": grounding.model_dump(mode="json"),
        "raw_violations": tuple(item.model_dump(mode="json") for item in raw_violations),
        "total_sentence_count": len(exact),
        "raw_violation_count": len(raw_violations),
        "weakened_count": grounding.weakened_count,
        "rejected_count": grounding.rejected_count,
        "raw_capture_precedes_grounding": True,
        "audit_only": True,
        "ranking_mutable": False,
        "adds_ranking_facts": False,
    }
    digest = _digest(unsigned)
    result = ReportGenerationAudit.model_validate(
        {**unsigned, "audit_id": f"rga_{digest[:24]}", "audit_sha256": digest}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("report generation audit verification failed")
    return result


@_guard_report_call(
    roots=(("audit", ReportGenerationAudit), ("report", DeterministicSREReport)),
    root_node_limits=(_MAX_EXACT_MODEL_NODES, 1_000_000),
    local_domains=_generation_local_domains,
    proposal_type=ReportSentenceProposal,
    proposal_limit=MAX_REPORT_GROUNDING_SENTENCES,
)
def require_report_generation_audit(
    audit: ReportGenerationAudit,
    *,
    proposals: tuple[ReportSentenceProposal, ...],
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
) -> ReportGenerationAudit:
    """Rebuild a raw-generation audit against all frozen external anchors."""

    if type(audit) is not ReportGenerationAudit or not audit.verify():
        raise ValueError("report generation audit failed exact verification")
    expected = capture_and_validate_report_generation(
        proposals=proposals,
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    if not hmac.compare_digest(audit.audit_sha256, expected.audit_sha256) or audit != expected:
        raise ValueError("report generation audit differs from rebuilt report generation audit")
    return ReportGenerationAudit.model_validate(audit.model_dump(mode="json", warnings="error"))


@_guard_report_call(
    roots=(("report", DeterministicSREReport),),
    failure_codes=_FAILURE_CODES,
)
def record_report_generation_failure(
    *,
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
    failure_code: ReportGenerationFailureCode,
    failure_detail: str,
) -> ReportGenerationFailureAudit:
    """Bind a report-stage failure to the already frozen deterministic fallback."""

    if type(failure_detail) is not str or not failure_detail:
        raise ValueError("report generation failure detail must be a non-empty string")
    if len(failure_detail.encode("utf-8")) > 16_384:
        raise ValueError("report generation failure detail exceeds 16384 UTF-8 bytes")
    verified = require_deterministic_sre_report(
        report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    unsigned: dict[str, object] = {
        "schema_version": REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION,
        "report_sha256": verified.report_sha256,
        "context_sha256": verified.context_sha256,
        "source_top_k_sha256": verified.source_top_k_sha256,
        "failure_code": failure_code,
        "failure_detail_sha256": _digest(failure_detail),
        "fallback_report_sha256": verified.report_sha256,
        "uses_deterministic_fallback": True,
        "audit_only": True,
        "ranking_mutable": False,
        "adds_ranking_facts": False,
    }
    digest = _digest(unsigned)
    result = ReportGenerationFailureAudit.model_validate(
        {
            **unsigned,
            "failure_id": f"rgf_{digest[:24]}",
            "failure_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("report generation failure audit verification failed")
    return result


@_guard_report_call(
    roots=(("failure", ReportGenerationFailureAudit), ("report", DeterministicSREReport)),
    root_node_limits=(_MAX_EXACT_MODEL_NODES, 1_000_000),
    failure_codes=_FAILURE_CODES,
)
def require_report_generation_failure_audit(
    failure: ReportGenerationFailureAudit,
    *,
    report: DeterministicSREReport,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str | None = None,
    failure_code: ReportGenerationFailureCode,
    failure_detail: str,
) -> ReportGenerationFailureAudit:
    """Rebuild a report failure against the frozen deterministic fallback."""

    if type(failure) is not ReportGenerationFailureAudit or not failure.verify():
        raise ValueError("report generation failure audit failed exact verification")
    expected = record_report_generation_failure(
        report=report,
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        failure_code=failure_code,
        failure_detail=failure_detail,
    )
    if (
        not hmac.compare_digest(failure.failure_sha256, expected.failure_sha256)
        or failure != expected
    ):
        raise ValueError("report generation failure audit differs from rebuilt failure audit")
    return ReportGenerationFailureAudit.model_validate(
        failure.model_dump(mode="json", warnings="error")
    )


__all__: Sequence[str] = (
    "REPORT_GENERATION_AUDIT_SCHEMA_VERSION",
    "REPORT_GENERATION_FAILURE_AUDIT_SCHEMA_VERSION",
    "RawReportViolation",
    "ReportGenerationAudit",
    "ReportGenerationFailureAudit",
    "capture_and_validate_report_generation",
    "record_report_generation_failure",
    "require_report_generation_audit",
    "require_report_generation_failure_audit",
)
