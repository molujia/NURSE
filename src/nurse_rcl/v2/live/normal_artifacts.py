"""Normal execution contracts, distinct from legacy independent-replay receipts."""

from typing import Final, Literal

from pydantic import Field

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_measurements import RepairSignalMeasurement
from nurse_rcl.v2.live.repair_queries import RepairQueryFeedback, RepairQuerySelection
from nurse_rcl.v2.live.repair_query_session import RepairQueryCompletedStep
from nurse_rcl.v2.live.repair_ranking import RepairRanking

NORMAL_POLICY: Final = "live-repair-normal-rcl/1"


class NormalAssurance(StrictFrozenModel):
    """State the checks actually performed, without claiming an independent audit."""

    verification_level: Literal["boundary_checked"] = "boundary_checked"
    full_recompute: Literal["not_performed"] = "not_performed"
    interpretability_status: Literal["source_bound_traceable"] = "source_bound_traceable"


class NormalPreparationReceipt(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-rcl-preparation-receipt/1"] = (
        "nurse.v2.normal-rcl-preparation-receipt/1"
    )
    execution_sha256: SHA256
    execution_code_sha256: SHA256
    input_sha256: SHA256
    preparation_key_sha256: SHA256
    manifest_sha256: SHA256


class NormalQueryJournalEntry(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-rcl-query-journal/1"] = (
        "nurse.v2.normal-rcl-query-journal/1"
    )
    sequence: int = Field(ge=0, le=26, strict=True)
    previous_sha256: SHA256 | None
    catalog_sha256: SHA256
    phase: Literal["initial", "reserved", "measured", "completed", "stopped", "failed"]
    selection: RepairQuerySelection | None = None
    measurement: RepairSignalMeasurement | None = None
    feedback: RepairQueryFeedback | None = None
    failure_reason: Literal["measurement_integrity_error", "completion_integrity_error"] | None = (
        None
    )
    entry_sha256: SHA256


class NormalQuerySession(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-rcl-query-session/1"] = (
        "nurse.v2.normal-rcl-query-session/1"
    )
    semantics_sha256: SHA256
    catalog_sha256: SHA256
    journal_sha256: SHA256
    steps: tuple[RepairQueryCompletedStep, ...]
    submitted_request_keys: tuple[str, ...]
    pending_selection: RepairQuerySelection | None
    pending_measurement: RepairSignalMeasurement | None
    paid_count: int = Field(ge=0, le=8, strict=True)
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"] | None
    failure_reason: Literal["measurement_integrity_error", "completion_integrity_error"] | None
    session_sha256: SHA256


class NormalEvidence(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-rcl-evidence/1"] = "nurse.v2.normal-rcl-evidence/1"
    policy: Literal["live-repair-normal-rcl/1"] = "live-repair-normal-rcl/1"
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    session_sha256: SHA256
    session: NormalQuerySession
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    pre_ranking_semantics: Literal[False] = False
    result_sha256: SHA256
