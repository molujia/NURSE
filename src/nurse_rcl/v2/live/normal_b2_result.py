"""Owned B2 evidence: query-priority semantics, unchanged transparent score core."""

from typing import Literal

from pydantic import Field

from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalEvidence
from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession, score_normal_hybrid_tools
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_semantic_owner import NormalHybridSemanticRun
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_ranking import RepairRanking


class NormalB2Evidence(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-b2-evidence/1"] = "nurse.v2.normal-b2-evidence/1"
    profile: Literal["v2-live-hybrid-normal"] = "v2-live-hybrid-normal"
    semantic_scope: Literal["query_priorities_only"] = "query_priorities_only"
    pre_ranking_semantics: Literal[True] = True
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    paired_base_evidence_sha256: SHA256
    semantic_result_sha256: SHA256
    semantic_status: Literal["completed", "followup_fallback", "first_fallback"]
    failure_code: str | None
    selected_round: Literal[1, 2] | None
    selected_proposal_ids: tuple[str, ...]
    baseline_fallback: bool
    session_sha256: SHA256
    session: NormalHybridToolSession | None
    logical_tools_used: int = Field(ge=0, le=8)
    card_count: int = Field(ge=0, le=8)
    verification_count: int = Field(ge=0, le=8)
    reused_count: int = Field(ge=0, le=8)
    executed_count: int = Field(ge=0, le=8)
    physical_union_count: int = Field(ge=0, le=16)
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    result_sha256: SHA256


def score_normal_b2(
    *, data: RepairPreparationData, semantic_run: NormalHybridSemanticRun,
    paired_base_evidence: NormalEvidence, expected_semantic_result_sha256: str,
    expected_base_evidence_sha256: str,
) -> NormalB2Evidence:
    """Private admitted inputs, not a loader or a model-supplied scoring interface."""
    for pin in (expected_base_evidence_sha256, expected_semantic_result_sha256):
        require_pin(pin)
    base = paired_base_evidence
    if type(base) is not NormalEvidence or type(semantic_run) is not NormalHybridSemanticRun or (
        base.result_sha256 != expected_base_evidence_sha256
        or semantic_run.result_sha256 != expected_semantic_result_sha256
        or _seal(semantic_run, "result_sha256") != semantic_run
        or base.source_descriptor_sha256 != data.source_descriptor_sha256
        or base.source_files_sha256 != data.source_files_sha256
        or base.catalog_sha256 != data.catalog.catalog_sha256
        or base.candidate_freeze_sha256 != data.candidate_freeze_sha256
        or base.opaque_case_id != data.catalog.opaque_case_id
        or base.dataset != data.diagnostics.dataset
        or base.graph_manifest_sha256 != data.graph.manifest.manifest_sha256
        or base.session_sha256 != base.session.session_sha256
        or base.session.pending_selection is not None
        or base.session.failure_reason is not None
        or base.session.stop_reason is None
    ):
        raise ValueError("B2 independent base/semantic/source identity differs")
    session, selection, first = semantic_run.session, semantic_run.selection, semantic_run.first
    if _seal(first, "receipt_sha256") != first or (
        first.owner_identity_sha256 != semantic_run.owner_identity_sha256
    ):
        raise ValueError("B2 first receipt identity differs")
    fallback = semantic_run.status == "first_fallback"
    failure = first.failure_code
    if fallback:
        if session is not None or selection is not None or first.compilation is not None or (
            first.response is not None or failure is None
        ):
            raise ValueError("B2 first fallback is not an empty failed branch")
        ranking = base.ranking
    else:
        if session is None or selection is None or first.compilation is None or (
            first.response is None or first.failure_code is not None
            or session.paired_base_session_sha256 != base.session_sha256
            or selection.first_compilation != first.compilation
            or _seal(selection, "selection_sha256") != selection
        ):
            raise ValueError("B2 semantic selection/session binding differs")
        selected = (selection.first_compilation if selection.selected_round == 1
                    else selection.followup_compilation)
        if selected is None or (
            selection.selected_proposal_ids != selected.accepted_proposal_ids
            or (selection.status == "selected_second") != (selection.selected_round == 2)
            or (selection.status == "fallback") != (semantic_run.status == "followup_fallback")
            or (selection.failure_code is not None) != (selection.status == "fallback")
        ):
            raise ValueError("B2 selected snapshot differs")
        # Selected hypotheses do not delete valid measurements or adjust scores.
        # All actual branch evidence goes through the same deterministic scorer.
        ranking = score_normal_hybrid_tools(data=data, session=session).ranking
        failure = selection.failure_code
    if ranking.candidate_service_keys != data.catalog.candidate_service_keys:
        raise ValueError("B2 scoring changed the complete candidate universe")
    return _seal(NormalB2Evidence(
        dataset=data.diagnostics.dataset, opaque_case_id=data.catalog.opaque_case_id,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
        catalog_sha256=data.catalog.catalog_sha256,
        paired_base_evidence_sha256=expected_base_evidence_sha256,
        semantic_result_sha256=expected_semantic_result_sha256,
        semantic_status=semantic_run.status, failure_code=failure,
        selected_round=None if selection is None else selection.selected_round,
        selected_proposal_ids=() if selection is None else selection.selected_proposal_ids,
        baseline_fallback=fallback,
        session_sha256=base.session_sha256 if session is None else session.session_sha256,
        session=session, logical_tools_used=0 if session is None else session.paid_count,
        card_count=0 if session is None else session.card_count,
        verification_count=0 if session is None else session.verification_count,
        reused_count=0 if session is None else session.reused_count,
        executed_count=0 if session is None else session.executed_count,
        physical_union_count=(base.session.paid_count if session is None
                              else session.physical_union_count),
        graph_manifest_sha256=data.graph.manifest.manifest_sha256, graph=data.graph,
        ranking=ranking, result_sha256="0" * 64,
    ), "result_sha256")
