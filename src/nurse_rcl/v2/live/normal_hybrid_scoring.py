"""Explicit query-priority B2 evidence; all scores come from the shared core."""

from typing import Literal

from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance
from nurse_rcl.v2.live.normal_hybrid_queries import NormalHybridQuerySession
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_scoring import _score_ranking
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_ranking import RepairRanking


class NormalHybridEvidence(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-rcl-evidence/1"] = (
        "nurse.v2.normal-hybrid-rcl-evidence/1"
    )
    policy: Literal["live-early-semantic-normal-rcl/1"] = "live-early-semantic-normal-rcl/1"
    semantic_scope: Literal["query_priorities"] = "query_priorities"
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    session_sha256: SHA256
    session: NormalHybridQuerySession
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    pre_ranking_semantics: Literal[True] = True
    result_sha256: SHA256


def score_normal_hybrid_case(
    *, data: RepairPreparationData, session: NormalHybridQuerySession,
) -> NormalHybridEvidence:
    """Private owner-admitted completed branch, not a raw model response API."""
    if type(session) is not NormalHybridQuerySession or (
        session.catalog_sha256 != data.catalog.catalog_sha256
        or session.paid_count != len(session.steps)
        or session.reused_count + session.executed_count != session.paid_count
        or session.submitted_request_keys != tuple(sorted({
            s.feedback.request_key for s in session.steps
        }))
        or session.terminal_choice.selected_request_key is not None
        or session.terminal_choice.baseline_selection.stop_reason != session.stop_reason
    ):
        raise ValueError("hybrid scoring requires the bound complete terminal session/catalog")
    ranking = _score_ranking(data=data, session=session)
    return _seal(NormalHybridEvidence(
        dataset=data.diagnostics.dataset, opaque_case_id=data.catalog.opaque_case_id,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
        catalog_sha256=data.catalog.catalog_sha256, session_sha256=session.session_sha256,
        session=session, graph_manifest_sha256=data.graph.manifest.manifest_sha256,
        graph=data.graph, ranking=ranking, result_sha256="0" * 64,
    ), "result_sha256")
