"""Local, bounded failure diagnostics; never serialize exception messages."""

from __future__ import annotations

from typing import Literal, cast, get_args

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError

FailureStage = Literal[
    "investigation_input", "investigation_provider", "investigation_tool",
    "investigation_worker", "report_assembly", "report_projection",
    "report_provider", "report_validation", "report_delivery",
    "case_worker", "timing_finish", "timing_close", "timing_checkpoint",
]
FailureBoundary = Literal[
    "json_nodes", "json_bytes", "json_depth", "expanded_nodes", "expanded_depth",
    "facts_nodes", "metadata_nodes", "storage_nodes", "report_input_nodes",
    "logical_bytes", "logical_depth",
    "checkpoint_source_cards_bytes", "checkpoint_source_json_bytes",
    "checkpoint_cards_bytes", "checkpoint_combined_cards_bytes", "checkpoint_descriptor_bytes",
    "checkpoint_selection_bytes", "checkpoint_directories_bytes", "checkpoint_prefixes_bytes",
    "checkpoint_remainder_bytes", "checkpoint_logical_bytes", "checkpoint_file_bytes",
]


class OpsFailureDetail(StrictFrozenModel):
    stage: FailureStage
    failure_type: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    boundary: FailureBoundary | None = None
    observed: int | None = Field(default=None, ge=0, strict=True)
    limit: int | None = Field(default=None, ge=0, strict=True)
    observed_is_lower_bound: bool | None = None

    @model_validator(mode="after")
    def complete_measurement(self) -> Self:
        fields = (self.boundary, self.observed, self.limit, self.observed_is_lower_bound)
        if not (all(v is None for v in fields) or all(v is not None for v in fields)):
            raise ValueError("failure measurement must be complete or explicitly unknown")
        return self


def failure_detail(stage: FailureStage, exc: BaseException) -> OpsFailureDetail:
    from nurse_rcl.v2.live.ops_sre_checkpoint_ledger import CheckpointCommitError

    if isinstance(exc, CheckpointCommitError):
        cause = failure_detail(stage, exc.trusted_cause)
        return cause.model_copy(update={"failure_type": type(exc).__name__})
    if isinstance(exc, InvestigationStartupFailure):
        return exc.detail
    if isinstance(exc, ExchangeBoundaryError) and exc.boundary in get_args(FailureBoundary) and (
        type(exc.observed) is int and type(exc.limit) is int
        and type(exc.observed_is_lower_bound) is bool
    ):
        return OpsFailureDetail(stage=stage, failure_type=type(exc).__name__,
            boundary=cast(FailureBoundary, exc.boundary), observed=exc.observed,
            limit=exc.limit, observed_is_lower_bound=exc.observed_is_lower_bound)
    return OpsFailureDetail(stage=stage, failure_type=type(exc).__name__)


class InvestigationStartupFailure(RuntimeError):
    """Supervisor-authenticated failure; no made-up result or conversation."""

    def __init__(self, detail: OpsFailureDetail, *, receipt_sha256: str | None,
                 secondary: tuple[OpsFailureDetail, ...] = ()) -> None:
        super().__init__("investigation_startup_unavailable")
        self.detail = detail
        self.receipt_sha256 = receipt_sha256
        self._nurse_cleanup_failures = secondary


def secondary_failures(exc: BaseException) -> tuple[OpsFailureDetail, ...]:
    """Only diagnostics attached by trusted cleanup code, never exception text."""
    rows = getattr(exc, "_nurse_cleanup_failures", ())
    if type(rows) is not tuple or len(rows) > 16 or any(
        type(row) is not OpsFailureDetail for row in rows
    ):
        return ()
    return cast(tuple[OpsFailureDetail, ...], rows)


def retain_secondary(primary: BaseException, stage: FailureStage, exc: BaseException) -> None:
    rows = secondary_failures(primary)
    detail = failure_detail(stage, exc)
    if detail not in rows and len(rows) < 16:
        primary._nurse_cleanup_failures = (*rows, detail)  # type: ignore[attr-defined]
