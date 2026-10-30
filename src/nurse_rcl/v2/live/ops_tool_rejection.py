"""Finite public diagnostics. Exception text never supplies model-facing content."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel

RejectionReason = Literal[
    "card_kind_mismatch",
    "invalid_section",
    "invalid_parameters",
    "reference_not_available",
    "request_binding_mismatch",
    "unclassified_failure",
]


class OpsToolDiagnostic(StrictFrozenModel):
    reason_code: RejectionReason
    retry_hint: Literal["change_request", "unknown"]
    authority: Literal["navigation_not_evidence"] = "navigation_not_evidence"

    @model_validator(mode="after")
    def retry_classification(self) -> Self:
        expected = "unknown" if self.reason_code == "unclassified_failure" else "change_request"
        if self.retry_hint != expected:
            raise ValueError("diagnostic retry classification differs")
        return self


class OpsToolRejection(ValueError):
    def __init__(self, reason: str):
        self.request_binding_sha256: str | None = None
        self.diagnostic = OpsToolDiagnostic.model_validate(
            dict(
                reason_code=reason,
                retry_hint="change_request",
            )
        )
        super().__init__(self.diagnostic.reason_code)
