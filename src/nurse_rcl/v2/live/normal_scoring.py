"""The existing full RCL numerical pipeline, executed once on admitted inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_hybrid_queries import NormalHybridQuerySession
    from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession

from nurse_rcl.v2.live.normal_artifacts import NormalEvidence, NormalQuerySession
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.repair_error_references import _referenced_error_values, _summary_values
from nurse_rcl.v2.live.repair_fixed_facts import _signal_values
from nurse_rcl.v2.live.repair_local_scoring import _compile_local_values
from nurse_rcl.v2.live.repair_mechanisms import _compile_values
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_propagation import _compile_propagation_values
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_ranking import RepairRanking, _compile_ranking_values
from nurse_rcl.v2.live.repair_strengths import (
    _compile_strength_terms,
    _fixed_scalar_terms,
    _paid_scalar_terms,
)


def score_normal_case(
    *, data: RepairPreparationData, session: NormalQuerySession,
) -> NormalEvidence:
    if type(session) is not NormalQuerySession or (
        session.catalog_sha256 != data.catalog.catalog_sha256
        or session.pending_selection is not None or session.pending_measurement is not None
        or session.failure_reason is not None or session.stop_reason is None
    ):
        raise ValueError("normal scoring requires the bound complete terminal session/catalog")
    ranking = _score_ranking(data=data, session=session)
    return _seal(NormalEvidence(
        dataset=data.diagnostics.dataset, opaque_case_id=data.catalog.opaque_case_id,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
        catalog_sha256=data.catalog.catalog_sha256, session_sha256=session.session_sha256,
        session=session, graph_manifest_sha256=data.graph.manifest.manifest_sha256,
        graph=data.graph, ranking=ranking, result_sha256="0" * 64,
    ), "result_sha256")


def _score_ranking(
    *, data: RepairPreparationData,
    session: NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession,
) -> RepairRanking:
    """One numerical core; the normal/hybrid owner admits its versioned session."""
    with query_step("production_scoring"):
        fixed, terms = [], []
        with _module_span("fixed_facts"):
            for source in data.sources:
                fact = _signal_values(source)
                static = _fixed_scalar_terms(source, fact)
                fixed.append(fact)
                terms.extend(static)
                terms.extend(_paid_scalar_terms(
                    static[0], formal_scoring_eligible=source.quality.formal_scoring_eligible,
                    session=session,
                ))
        with _module_span("error_facts"):
            summaries = tuple(_summary_values(o) for o in data.diagnostics.errors)
            errors = _referenced_error_values(
                sources=data.error_sources, summaries=summaries,
                observation_sha256s=data.error_observation_sha256s,
                candidate_service_keys=data.catalog.candidate_service_keys,
            )
        mechanisms = _compile_values(
            catalog=data.catalog, session=session, signal_facts=tuple(fixed), error_facts=errors,
        )
        strengths = _compile_strength_terms(
            mechanisms=mechanisms, session=session, scalar_terms=tuple(terms),
            error_summaries=summaries,
        )
        with _module_span("local_scoring"):
            local = _compile_local_values(strengths)
        propagation = _compile_propagation_values(
            local=local, session=session, graph=data.graph.graph_pair, clock=data.clock,
        )
        return _compile_ranking_values(propagation)
