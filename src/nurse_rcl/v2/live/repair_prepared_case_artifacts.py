"""Explicit preparation-backed identities; never relabel raw-owner receipts."""

from typing import Final, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.repair_case_artifacts import (
    REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
    _RepairCaseAuditFields,
    _RepairCaseClosureFields,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationBinding
from nurse_rcl.v2.live.repair_report import _RepairReportFields
from nurse_rcl.v2.live.repair_report_projection import _RepairProjectionFields

REPAIR_PREPARED_POLICY: Final = "live-repair-prepared-case/1"
REPAIR_PREPARED_AUDIT_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            policy=REPAIR_PREPARED_POLICY,
            score_semantics=REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
            source="independent_preparation_receipt_current_full_source_pins_and_new_scope",
            replay="complete_B_F_sources_catalog_errors_raw_projected_graph_journal_and_scores",
        )
    )
)


class RepairPreparedCaseAudit(_RepairCaseAuditFields):
    schema_version: Literal["nurse.v2.live-repair-prepared-case-audit/1"] = (
        "nurse.v2.live-repair-prepared-case-audit/1"
    )
    source_validation_policy: Literal["live-repair-prepared-case/1"] = REPAIR_PREPARED_POLICY
    preparation: RepairPreparationBinding


class RepairPreparedCaseClosure(_RepairCaseClosureFields):
    schema_version: Literal["nurse.v2.live-repair-prepared-case-closure/1"] = (
        "nurse.v2.live-repair-prepared-case-closure/1"
    )
    execution_policy: Literal["live-repair-prepared-case/1"] = REPAIR_PREPARED_POLICY
    preparation: RepairPreparationBinding


class RepairPreparedReportProjection(_RepairProjectionFields):
    schema_version: Literal["nurse.v2.live-repair-prepared-report-projection/1"] = (
        "nurse.v2.live-repair-prepared-report-projection/1"
    )
    preparation: RepairPreparationBinding


class RepairPreparedReportRun(_RepairReportFields):
    schema_version: Literal["nurse.v2.live-repair-prepared-report/1"] = (
        "nurse.v2.live-repair-prepared-report/1"
    )
    projection: RepairPreparedReportProjection
    preparation: RepairPreparationBinding
    interpretability_status: Literal["preparation_bound_source_and_score_replayed"] = (
        "preparation_bound_source_and_score_replayed"
    )
