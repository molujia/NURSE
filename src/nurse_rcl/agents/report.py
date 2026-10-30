"""Verify User Manager reports and build deterministic degraded fallbacks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.verified_context import VerifiedRCLContextArtifact
from nurse_rcl.models import EvidencePath, RankedCandidate, ReportRecord
from nurse_rcl.schemas import StrictFrozenModel

REPORT_ARTIFACT_VERSION: Literal["verified_agent_report_v1"] = "verified_agent_report_v1"
ReportVerificationCode = Literal[
    "report_accepted",
    "reject_context_hash",
    "reject_top_k_order",
    "reject_ranked_candidate_mutation",
    "reject_unknown_evidence",
    "reject_evidence_path",
    "reject_unsupported_recommendation",
]


class ReportRecommendation(StrictFrozenModel):
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    support: Literal["evidence_backed", "unsupported"]


class AgentReportSubmission(StrictFrozenModel):
    verified_context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: tuple[RankedCandidate, ...] = Field(min_length=1, max_length=5)
    evidence_paths: tuple[EvidencePath, ...]
    cited_evidence_ids: tuple[str, ...]
    narrative: str = Field(min_length=1)
    recommendations: tuple[ReportRecommendation, ...]


class ReportVerificationDecision(StrictFrozenModel):
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    code: ReportVerificationCode
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"decision_hash"})
        return self.decision_hash == sha256_hex(canonical_json_bytes(unsigned))


class VerifiedAgentReportArtifact(StrictFrozenModel):
    report_version: Literal["verified_agent_report_v1"] = REPORT_ARTIFACT_VERSION
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_top_k: tuple[RankedCandidate, ...] = Field(min_length=1, max_length=5)
    report: ReportRecord
    recommendations: tuple[ReportRecommendation, ...]
    source: Literal["agent_verified", "deterministic_fallback"]
    verification_code: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    submission_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.report.verified_context_hash != self.context_hash:
            return False
        if self.report.top_k_services != tuple(
            candidate.service for candidate in self.frozen_top_k
        ):
            return False
        if self.source == "agent_verified":
            if (
                self.report.degraded
                or self.verification_code != "report_accepted"
                or self.submission_hash is None
            ):
                return False
        elif not self.report.degraded or self.verification_code == "report_accepted":
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


def agent_report_submission_schema() -> dict[str, Any]:
    """Return the strict structured arguments accepted by ``write_report``."""

    return AgentReportSubmission.model_json_schema()


def verify_agent_report(
    *,
    context: VerifiedRCLContextArtifact,
    submission: AgentReportSubmission,
) -> ReportVerificationDecision:
    """Verify every rank-bearing and evidence-bearing report field in fixed order."""

    if not context.verify():
        raise ValueError("Verified RCL context hash verification failed")
    if submission.verified_context_hash != context.context_hash:
        return _decision(context=context, submission=submission, code="reject_context_hash")

    submitted_services = tuple(candidate.service for candidate in submission.top_k)
    frozen_services = tuple(candidate.service for candidate in context.context.top_k)
    if submitted_services != frozen_services:
        return _decision(
            context=context,
            submission=submission,
            code="reject_top_k_order",
        )
    if submission.top_k != context.context.top_k:
        return _decision(
            context=context,
            submission=submission,
            code="reject_ranked_candidate_mutation",
        )

    known_evidence = _known_evidence_ids(context)
    cited_evidence = set(submission.cited_evidence_ids)
    cited_evidence.update(
        evidence_id
        for recommendation in submission.recommendations
        for evidence_id in recommendation.evidence_ids
    )
    if not cited_evidence.issubset(known_evidence):
        return _decision(
            context=context,
            submission=submission,
            code="reject_unknown_evidence",
        )
    if submission.evidence_paths != context.context.evidence_paths:
        return _decision(
            context=context,
            submission=submission,
            code="reject_evidence_path",
        )
    if any(
        recommendation.support == "evidence_backed"
        and not recommendation.evidence_ids
        for recommendation in submission.recommendations
    ):
        return _decision(
            context=context,
            submission=submission,
            code="reject_unsupported_recommendation",
        )
    return _decision(context=context, submission=submission, code="report_accepted")


def finalize_user_manager_report(
    *,
    context: VerifiedRCLContextArtifact,
    submission: AgentReportSubmission | None,
    degradation_code: str | None = None,
) -> VerifiedAgentReportArtifact:
    """Freeze a verified agent report or a deterministic context-only fallback."""

    if not context.verify():
        raise ValueError("Verified RCL context hash verification failed")
    if submission is not None and degradation_code is not None:
        raise ValueError("degradation_code is only valid when no submission exists")

    source: Literal["agent_verified", "deterministic_fallback"]
    verification_code: str
    submission_hash: str | None
    recommendations: tuple[ReportRecommendation, ...]
    if submission is not None:
        decision = verify_agent_report(context=context, submission=submission)
        submission_hash = decision.submission_hash
        if decision.accepted:
            source = "agent_verified"
            verification_code = decision.code
            narrative = submission.narrative
            recommendations = submission.recommendations
        else:
            source = "deterministic_fallback"
            verification_code = decision.code
            narrative = _fallback_narrative(context=context, code=decision.code)
            recommendations = _fallback_recommendations()
    else:
        if degradation_code is None:
            raise ValueError("missing report submission requires a degradation_code")
        _require_stable_code(degradation_code)
        if degradation_code == "report_accepted":
            raise ValueError("fallback degradation code cannot be report_accepted")
        source = "deterministic_fallback"
        verification_code = degradation_code
        submission_hash = None
        narrative = _fallback_narrative(context=context, code=degradation_code)
        recommendations = _fallback_recommendations()

    report = ReportRecord(
        case_id=context.context.case_id,
        verified_context_hash=context.context_hash,
        top_k_services=tuple(candidate.service for candidate in context.context.top_k),
        narrative=narrative,
        degraded=source == "deterministic_fallback",
    )
    unsigned = {
        "report_version": REPORT_ARTIFACT_VERSION,
        "context_hash": context.context_hash,
        "context_artifact_hash": context.artifact_hash,
        "frozen_top_k": [
            candidate.model_dump(mode="json") for candidate in context.context.top_k
        ],
        "report": report.model_dump(mode="json"),
        "recommendations": [
            recommendation.model_dump(mode="json")
            for recommendation in recommendations
        ],
        "source": source,
        "verification_code": verification_code,
        "submission_hash": submission_hash,
    }
    result = VerifiedAgentReportArtifact(
        context_hash=context.context_hash,
        context_artifact_hash=context.artifact_hash,
        frozen_top_k=context.context.top_k,
        report=report,
        recommendations=recommendations,
        source=source,
        verification_code=verification_code,
        submission_hash=submission_hash,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("verified agent report artifact verification failed")
    return result


def _decision(
    *,
    context: VerifiedRCLContextArtifact,
    submission: AgentReportSubmission,
    code: ReportVerificationCode,
) -> ReportVerificationDecision:
    submission_hash = sha256_hex(
        canonical_json_bytes(submission.model_dump(mode="json"))
    )
    accepted = code == "report_accepted"
    unsigned = {
        "context_hash": context.context_hash,
        "submission_hash": submission_hash,
        "accepted": accepted,
        "code": code,
    }
    result = ReportVerificationDecision(
        context_hash=context.context_hash,
        submission_hash=submission_hash,
        accepted=accepted,
        code=code,
        decision_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("report verification decision hash failed")
    return result


def _known_evidence_ids(context: VerifiedRCLContextArtifact) -> set[str]:
    return {
        evidence_id
        for candidate in context.context.top_k
        for evidence_id in candidate.evidence_ids
    } | {
        evidence_id
        for path in context.context.evidence_paths
        for evidence_id in path.evidence_ids
    }


def _fallback_narrative(*, context: VerifiedRCLContextArtifact, code: str) -> str:
    ranking = "; ".join(
        f"{candidate.rank}. {candidate.service} (score={candidate.score:.17g})"
        for candidate in context.context.top_k
    )
    paths = "; ".join(
        f"{path.source_service}->{path.target_service}"
        for path in context.context.evidence_paths
    ) or "none"
    return (
        f"Deterministic NURSE report for {context.context.case_id}. "
        f"Frozen Top-K: {ranking}. Evidence paths: {paths}. "
        f"Agent report degradation: {code}. Any operational recommendation not explicitly "
        "backed by the frozen evidence is unsupported."
    )


def _fallback_recommendations() -> tuple[ReportRecommendation, ...]:
    return (
        ReportRecommendation(
            text="Verify the frozen evidence before taking a mitigation action.",
            evidence_ids=(),
            support="unsupported",
        ),
    )


def _require_stable_code(code: str) -> None:
    if (
        not code
        or len(code) > 64
        or not code[0].islower()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
    ):
        raise ValueError("degradation_code must be a stable lowercase identifier")
