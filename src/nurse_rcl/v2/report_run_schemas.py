"""Private single-run policy and audit contracts; integrity never establishes authority."""

from __future__ import annotations

import hmac
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.report_audit_json import _measure_preflighted, preflight_json_tree
from nurse_rcl.v2.report_composition_proposals import ReportCompositionResponse
from nurse_rcl.v2.report_evidence_schemas import ReportEvidenceProjection
from nurse_rcl.v2.report_mechanical_schemas import ReportCompositionMechanicalValidation
from nurse_rcl.v2.report_presentation_schemas import ReportCompositionPresentation
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

REPORT_RUN_POLICY_SCHEMA_VERSION: Literal["nurse.v2.report-run-policy/1"] = (
    "nurse.v2.report-run-policy/1"
)
REPORT_RUN_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.report-run-audit/1"] = (
    "nurse.v2.report-run-audit/1"
)
REPORT_RUNNER_VERSION: Literal["nurse-v2-report-composition-runner/1"] = (
    "nurse-v2-report-composition-runner/1"
)
_SHA: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_WireHex: TypeAlias = Annotated[str, Field(max_length=524288, pattern=r"^(?:[0-9a-f]{2})*$")]
_PromptHex: TypeAlias = Annotated[str, Field(max_length=1048576, pattern=r"^(?:[0-9a-f]{2})*$")]
ReportRunReason: TypeAlias = Literal[
    "report_evidence_unavailable",
    "input_budget_exceeded",
    "request_limit",
    "reported_input_budget_exceeded",
    "reported_output_budget_exceeded",
    "no_deliverable_claims",
    "partially_blocked",
    "report_composition_schema",
    "timeout",
    "transport_error",
    "response_type",
    "response_limit",
    "response_shape",
    "provider_refusal",
    "unrequested_tool",
    "incomplete_response",
    "json_limit",
    "json_encoding",
    "json_non_finite",
    "json_cycle",
    "json_type",
    "json_duplicate_key",
    "json_syntax",
    "json_object_required",
]


class _RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ReportRunPolicy(_RunModel):
    schema_version: Literal["nurse.v2.report-run-policy/1"] = REPORT_RUN_POLICY_SCHEMA_VERSION
    configuration_version: str = Field(default="default/1", min_length=1, max_length=128)
    model: Literal["deepseek-v4-flash-ga-260731"] = "deepseek-v4-flash-ga-260731"
    temperature: Literal[0] = 0
    max_report_requests: Literal[1] = 1
    input_token_limit: int = Field(default=8000, gt=0)
    output_token_limit: int = Field(default=2000, gt=0, le=65536)
    estimator_version: Literal["request-utf8-bytes-plus-256/1"] = "request-utf8-bytes-plus-256/1"

    @field_validator(
        "temperature",
        "max_report_requests",
        "input_token_limit",
        "output_token_limit",
        mode="before",
    )
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("report run policy requires exact integer configuration")
        return value

    @model_validator(mode="after")
    def versioned_budget(self) -> Self:
        if self.configuration_version == "default/1" and (
            self.input_token_limit != 8000 or self.output_token_limit != 2000
        ):
            raise ValueError("report run policy overrides require nondefault configuration_version")
        return self

    @property
    def sha256(self) -> str:
        if type(self) is not ReportRunPolicy:
            raise ValueError("report run policy requires exact ReportRunPolicy type")
        preflight_exact_pydantic_tree(
            self, root_model_types=(ReportRunPolicy,), label="report run policy"
        )
        original = _bytes(self)
        rebuilt = ReportRunPolicy.model_validate_json(original)
        if not hmac.compare_digest(original, _bytes(rebuilt)):
            raise ValueError("report run policy snapshot changes exact canonical bytes")
        return sha256_hex(original)


def _bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json", warnings="error"))


class ReportRunAudit(_RunModel):
    """Private receipt. verify checks exact structure/hash closure, not source truth.

    Transport failures without retained bytes are only attested by an independent
    audit pin. Offline replay cannot prove that a timeout or transport event occurred.
    """

    schema_version: Literal["nurse.v2.report-run-audit/1"] = REPORT_RUN_AUDIT_SCHEMA_VERSION
    runner_version: Literal["nurse-v2-report-composition-runner/1"] = REPORT_RUNNER_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    sources_sha256: _SHA
    expected_audit_sha256: _SHA
    expected_context_sha256: _SHA
    expected_identity_manifest_sha256: _SHA
    expected_evidence_policy_sha256: _SHA
    policy: ReportRunPolicy
    expected_policy_sha256: _SHA
    projection: ReportEvidenceProjection = Field(repr=False)
    prompt_version: Literal["nurse-v2-report-composition-prompt/1"] = (
        "nurse-v2-report-composition-prompt/1"
    )
    prompt_payload_hex: _PromptHex | None = Field(default=None, repr=False)
    prompt_sha256: _SHA | None = None
    request_hex: _WireHex | None = Field(default=None, repr=False)
    request_sha256: _SHA | None = None
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    exchange_attempted: bool = False
    exchange_status: Literal["not_attempted", "response_ready", "failed"] = "not_attempted"
    exchange_failure_code: ReportRunReason | None = None
    structural_status: Literal["not_parsed", "valid", "invalid"] = "not_parsed"
    status: Literal["full", "partial", "deterministic_fallback"] = "deterministic_fallback"
    reason_code: ReportRunReason | None = None
    raw_response_hex: _WireHex | None = Field(default=None, repr=False)
    response_sha256: _SHA | None = None
    received_bytes: int | None = Field(default=None, ge=0)
    parsed_response: ReportCompositionResponse | None = Field(default=None, repr=False)
    parsed_response_sha256: _SHA | None = None
    usage_status: Literal["reported", "missing", "invalid", "unavailable"] = "unavailable"
    actual_input_tokens: int | None = Field(default=None, ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)
    raw_usage_json: str | None = Field(default=None, max_length=262144, repr=False)
    mechanical_validation: ReportCompositionMechanicalValidation | None = Field(
        default=None, repr=False
    )
    presentation: ReportCompositionPresentation | None = Field(default=None, repr=False)
    audit_sha256: _SHA

    @model_validator(mode="after")
    def receipt_structure(self) -> Self:
        if self.opaque_case_id != self.projection.foundation.opaque_case_id:
            raise ValueError("report audit case identity differs from its foundation")
        for raw, digest in (
            (self.prompt_payload_hex, self.prompt_sha256),
            (self.request_hex, self.request_sha256),
            (self.raw_response_hex, self.response_sha256),
        ):
            if (raw is None) != (digest is None):
                raise ValueError("report audit byte/hash pairs must agree")
            if raw is not None and sha256_hex(bytes.fromhex(raw)) != digest:
                raise ValueError("report audit byte/hash mismatch")
        if self.request_hex is not None:
            if self.prompt_payload_hex is None or self.estimated_input_tokens != (
                len(self.request_hex) // 2 + 256
            ):
                raise ValueError("report audit request estimate or prompt missing")
        elif self.estimated_input_tokens is not None or self.exchange_attempted:
            raise ValueError("report audit unprepared request cannot be estimated or attempted")
        if self.exchange_attempted != (self.exchange_status != "not_attempted"):
            raise ValueError("report audit attempt status mismatch")
        if (self.exchange_failure_code is not None) != (self.exchange_status == "failed"):
            raise ValueError("report audit exchange failure mismatch")
        if self.raw_response_hex is not None and (
            not self.exchange_attempted or self.received_bytes != len(self.raw_response_hex) // 2
        ):
            raise ValueError("report audit wire length mismatch")
        if self.usage_status == "reported":
            if self.actual_input_tokens is None or self.actual_output_tokens is None:
                raise ValueError("reported usage requires actual counts")
        elif self.actual_input_tokens is not None or self.actual_output_tokens is not None:
            raise ValueError("unreported usage must not invent counts")
        if (self.parsed_response is not None) != (self.structural_status == "valid"):
            raise ValueError("report audit domain parse status mismatch")
        if (self.parsed_response is None) != (self.parsed_response_sha256 is None):
            raise ValueError("report audit parsed response hash missing")
        if self.parsed_response is not None and (
            sha256_hex(_bytes(self.parsed_response)) != self.parsed_response_sha256
        ):
            raise ValueError("report audit parsed response hash mismatch")
        if (self.presentation is None) != (self.mechanical_validation is None):
            raise ValueError("report audit presentation and mechanics must coexist")
        if self.mechanical_validation is not None:
            if self.parsed_response is None or self.structural_status != "valid":
                raise ValueError("report audit mechanics requires a valid retained response")
            if self.mechanical_validation.response_sha256 != self.parsed_response_sha256:
                raise ValueError("report audit mechanical response hash mismatch")
            # Match two retained copies only; this grants neither source nor semantic authority.
            checked_claims = self.mechanical_validation.claims
            response_claims = self.parsed_response.claims
            if len(checked_claims) != len(response_claims) or any(
                not hmac.compare_digest(_bytes(checked.proposal), _bytes(proposed))
                for checked, proposed in zip(checked_claims, response_claims, strict=True)
            ):
                raise ValueError("report audit mechanical claims differ from retained response")
        if self.presentation is not None and (
            self.mechanical_validation is None
            or self.status != self.presentation.status
            or _bytes(self.presentation.foundation) != _bytes(self.projection.foundation)
            or _bytes(self.presentation.mechanical_validation) != _bytes(self.mechanical_validation)
        ):
            raise ValueError("report audit presentation closure mismatch")
        if self.presentation is None and self.status != "deterministic_fallback":
            raise ValueError("report audit delivery requires a presentation")
        return self

    def verify(self) -> bool:
        try:
            measured = _preflight_audit(self)
            serialized = self.model_dump(mode="json", warnings="error")
            original = canonical_json_bytes(serialized)
            if len(original) != measured:
                return False
            rebuilt = ReportRunAudit.model_validate_json(original)
            unsigned = {k: v for k, v in serialized.items() if k != "audit_sha256"}
            return (
                hmac.compare_digest(original, _bytes(rebuilt))
                and hmac.compare_digest(
                    self.audit_sha256, sha256_hex(canonical_json_bytes(unsigned))
                )
                and self.projection.verify()
                and hmac.compare_digest(self.policy.sha256, self.expected_policy_sha256)
                and (self.mechanical_validation is None or self.mechanical_validation.verify())
                and (self.presentation is None or self.presentation.verify())
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return False


def _preflight_audit(audit: ReportRunAudit) -> int:
    """Bound retained fields before any serializer, hash or hex allocation."""
    if type(audit) is not ReportRunAudit:
        raise ValueError("report audit requires exact ReportRunAudit type")
    preflight_json_tree(audit, root_model_types=(ReportRunAudit,))
    for name, limit in (
        ("raw_response_hex", 524288),
        ("request_hex", 524288),
        ("prompt_payload_hex", 1048576),
        ("raw_usage_json", 262144),
    ):
        value = object.__getattribute__(audit, name)
        if value is not None and (type(value) is not str or len(value) > limit):
            raise ValueError("report audit retained bytes exceed bounds")
    return _measure_preflighted(audit, root_model_types=(ReportRunAudit,))
