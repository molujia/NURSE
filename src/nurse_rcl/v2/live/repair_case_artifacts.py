"""Case-owner data identities, distinct from per-call raw reassembly receipts.

Deserializing these values grants no source, score, report or success authority.
The owning executor must independently establish every bound identity/check.
"""

from __future__ import annotations

from typing import Final, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_query_session import RepairQuerySession
from nurse_rcl.v2.live.repair_ranking import REPAIR_RANKING_SEMANTICS_SHA256, RepairRanking
from nurse_rcl.v2.live.repair_safety import REPAIR_CAPACITY_SHA256, RepairModel

REPAIR_CASE_POLICY: Final = "live-repair-case-owner/1"
REPAIR_CASE_AUDIT_SCHEMA_VERSION: Final = "nurse.v2.live-repair-case-audit/1"
REPAIR_CASE_CLOSURE_SCHEMA_VERSION: Final = "nurse.v2.live-repair-case-closure/1"
REPAIR_CASE_AUDIT_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version=REPAIR_CASE_AUDIT_SCHEMA_VERSION,
            source_validation_policy=REPAIR_CASE_POLICY,
            replay_scope="same_owner_complete_sources_and_scores",
            source="whole_raw_owner_complete_file_content_pins_and_all_B_F_blocks",
            replay="fresh_catalog_fixed_errors_graph_paid_journal_and_all_scores",
            ranking=REPAIR_RANKING_SEMANTICS_SHA256,
            capacity=REPAIR_CAPACITY_SHA256,
            authority="data_only_independent_owner_verification_required",
        )
    )
)


class _RepairCaseAuditFields(RepairModel):
    semantics_sha256: SHA256
    profile: Literal["v2-live-evidence-repair"] = "v2-live-evidence-repair"
    replay_scope: Literal["same_owner_complete_sources_and_scores"] = (
        "same_owner_complete_sources_and_scores"
    )
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    session_sha256: SHA256
    session: RepairQuerySession
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    pre_ranking_semantics: Literal[False] = False
    result_sha256: SHA256


class RepairCaseAudit(_RepairCaseAuditFields):
    schema_version: Literal["nurse.v2.live-repair-case-audit/1"] = REPAIR_CASE_AUDIT_SCHEMA_VERSION
    source_validation_policy: Literal["live-repair-case-owner/1"] = REPAIR_CASE_POLICY


class _RepairCaseClosureFields(RepairModel):
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    execution_sha256: SHA256
    code_sha256: SHA256
    input_sha256: SHA256
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    journal_sha256: SHA256
    session_sha256: SHA256
    audit_sha256: SHA256
    ranking_sha256: SHA256
    report_sha256: SHA256
    request_sha256: SHA256
    response_sha256: SHA256
    verification_sha256: SHA256
    exit_checks_sha256: SHA256
    publication_sha256: SHA256
    timing_identity_sha256: SHA256
    timing_checkpoint_sha256: SHA256
    result_sha256: SHA256


class RepairCaseClosure(_RepairCaseClosureFields):
    schema_version: Literal["nurse.v2.live-repair-case-closure/1"] = (
        REPAIR_CASE_CLOSURE_SCHEMA_VERSION
    )
    execution_policy: Literal["live-repair-case-owner/1"] = REPAIR_CASE_POLICY
