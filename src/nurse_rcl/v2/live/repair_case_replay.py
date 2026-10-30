"""Same-owner production scoring and independent complete-source final replay.

Only pure numerical functions are shared. The candidate audit never enters the
reconstruction algorithm: it participates only in the final whole-value check.
No raw reassembly, paid measurement, network call or persistent trust is added.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live.repair_case_artifacts import (
    REPAIR_CASE_AUDIT_SEMANTICS_SHA256,
    RepairCaseAudit,
)
from nurse_rcl.v2.live.repair_case_guards import _CaseInputs
from nurse_rcl.v2.live.repair_error_references import (
    RepairErrorSources,
    _replayed_referenced_errors,
    build_repair_error_sources,
)
from nurse_rcl.v2.live.repair_fixed_facts import (
    RepairErrorFactSummary,
    RepairFixedSignalFacts,
    _signal_values,
)
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph, _compile_graph, _source_descriptor
from nurse_rcl.v2.live.repair_local_scoring import _compile_local_values
from nurse_rcl.v2.live.repair_mechanisms import _compile_values
from nurse_rcl.v2.live.repair_module_timing import _module_scope
from nurse_rcl.v2.live.repair_prepared_case_artifacts import (
    REPAIR_PREPARED_AUDIT_SEMANTICS_SHA256,
    RepairPreparedCaseAudit,
)
from nurse_rcl.v2.live.repair_propagation import _compile_propagation_values
from nurse_rcl.v2.live.repair_queries import (
    REPAIR_QUERY_SEMANTICS_SHA256,
    RepairQueryCatalog,
    _request_values,
    _seal,
    _time_need_values,
    _validate_catalog,
)
from nurse_rcl.v2.live.repair_query_session import _ROOTS as SESSION_ROOTS
from nurse_rcl.v2.live.repair_query_session import RepairQuerySession, _read_bytes, _read_journal
from nurse_rcl.v2.live.repair_ranking import _compile_ranking_values
from nurse_rcl.v2.live.repair_safety import canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_sources import _replayed_repair_measurement_source
from nurse_rcl.v2.live.repair_strengths import (
    RepairStrengthTerm,
    _compile_strength_terms,
    _fixed_scalar_terms,
    _paid_scalar_terms,
)
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution, freeze_clock_resolution

CaseAudit = RepairCaseAudit | RepairPreparedCaseAudit
_ROOTS = (
    *SESSION_ROOTS,
    RepairCaseAudit,
    RepairPreparedCaseAudit,
    RepairErrorSources,
    RepairErrorFactSummary,
)


def _protect(guards: ExitStack, value: Any) -> Any:
    guards.callback(require_unchanged, value, _ROOTS, snapshot(value, _ROOTS))
    return value


def _consume(guards: ExitStack, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    values = (args, tuple(sorted(kwargs.items())))
    before = snapshot(values, _ROOTS)
    try:
        result = function(*args, **kwargs)
    finally:
        require_unchanged(values, _ROOTS, before)
    return _protect(guards, result)


def _terminal(session: RepairQuerySession, catalog: RepairQueryCatalog) -> None:
    if (
        type(session) is not RepairQuerySession
        or session != _seal(session, "session_sha256")
        or session.catalog_sha256 != catalog.catalog_sha256
        or session.pending_selection is not None
        or session.pending_measurement is not None
        or session.failure_reason is not None
        or session.stop_reason is None
    ):
        raise ValueError("case scoring requires the complete terminal session")


def _complete_scalar_terms(
    values: tuple[tuple[tuple[RepairStrengthTerm, ...], bool], ...],
    session: RepairQuerySession,
) -> tuple[RepairStrengthTerm, ...]:
    return tuple(
        term
        for static, eligible in values
        for term in (
            *static,
            *_paid_scalar_terms(static[0], formal_scoring_eligible=eligible, session=session),
        )
    )


@contextmanager
def _score_scope(
    inputs: _CaseInputs,
    *,
    catalog: RepairQueryCatalog,
    session: RepairQuerySession,
    graph: RepairLiveGraph,
    clock: FrozenClockResolution,
    fixed: tuple[RepairFixedSignalFacts, ...],
    errors: RepairErrorSources,
    scalar_values: tuple[tuple[tuple[RepairStrengthTerm, ...], bool], ...],
) -> Iterator[CaseAudit]:
    """Common pure score composition with all intermediate exit responsibilities."""
    args = inputs.prepared.runtime.arguments
    with ExitStack() as guards:
        _protect(guards, (catalog, session, graph, clock, fixed, scalar_values))
        # The complete four-slot source has its own original whole-case
        # admission, including context, in _replayed_referenced_errors below.
        # Do not invent a larger raw-plus-derived aggregate capacity root.
        _protect(guards, errors)
        _terminal(session, catalog)
        error_facts, summaries = guards.enter_context(
            _replayed_referenced_errors(
                sources=errors,
                expected_sources_sha256=inputs.prepared.error_sources.result_sha256,
                expected_observation_sha256s=inputs.prepared.error_observation_sha256s,
                time_contract=args["time_contract"],
                expected_time_contract_sha256=args["expected_time_contract_sha256"],
                candidate_universe=args["candidate_universe"],
                frozen_candidates=args["frozen_candidates"],
            )
        )
        mechanisms = _consume(
            guards,
            _compile_values,
            catalog=catalog,
            session=session,
            signal_facts=fixed,
            error_facts=error_facts,
        )
        terms = _consume(guards, _complete_scalar_terms, scalar_values, session)
        strengths = _consume(
            guards,
            _compile_strength_terms,
            mechanisms=mechanisms,
            session=session,
            scalar_terms=terms,
            error_summaries=summaries,
        )
        local = _consume(guards, _compile_local_values, strengths)
        propagation = _consume(
            guards,
            _compile_propagation_values,
            local=local,
            session=session,
            graph=graph.graph_pair,
            clock=clock,
        )
        ranking = _consume(guards, _compile_ranking_values, propagation)
        audit_type = RepairCaseAudit if inputs.preparation is None else RepairPreparedCaseAudit
        binding: dict[str, Any] = (
            {} if inputs.preparation is None else {"preparation": inputs.preparation}
        )
        result = _seal(
            audit_type(
                **binding,
                semantics_sha256=(
                    REPAIR_CASE_AUDIT_SEMANTICS_SHA256
                    if inputs.preparation is None
                    else REPAIR_PREPARED_AUDIT_SEMANTICS_SHA256
                ),
                dataset=graph.dataset,
                opaque_case_id=catalog.opaque_case_id,
                source_descriptor_sha256=inputs.prepared.source_descriptor_sha256,
                source_files_sha256=inputs.prepared.source_files_sha256,
                candidate_freeze_sha256=inputs.prepared.candidate_freeze_sha256,
                catalog_sha256=catalog.catalog_sha256,
                session_sha256=session.session_sha256,
                session=session,
                graph_manifest_sha256=graph.manifest.manifest_sha256,
                graph=graph,
                ranking=ranking,
                result_sha256="0" * 64,
            ),
            "result_sha256",
        )
        _protect(guards, result)
        yield result


def _build_owned_case_audit(inputs: _CaseInputs, session: RepairQuerySession) -> CaseAudit:
    inputs.check_full_method()
    args = inputs.prepared.runtime.arguments
    with ExitStack() as guards:
        guards.callback(inputs.check_full_method)
        _protect(guards, session)
        fixed, scalar_values = [], []
        for source in args["sources"]:
            # Raw scalar roots retain their per-source limits, never one giant
            # combined payload. Complete derived evidence retains whole-case caps.
            fact = _consume(guards, _signal_values, source)
            static = tuple(_consume(guards, _fixed_scalar_terms, source, fact))
            fixed.append(fact)
            scalar_values.append((static, source.quality.formal_scoring_eligible))
        with _score_scope(
            inputs,
            catalog=inputs.prepared.runtime.catalog,
            session=session,
            graph=inputs.prepared.graph,
            clock=args["clock_resolution"],
            fixed=tuple(fixed),
            errors=inputs.prepared.error_sources,
            scalar_values=tuple(scalar_values),
        ) as result:
            return result


@contextmanager
def _rebuild_case_evidence(inputs: _CaseInputs) -> Iterator[CaseAudit]:
    """Independent source-derived values; intentionally accepts no production audit."""
    inputs.check_full_method()
    runtime = inputs.prepared.runtime
    args = runtime.arguments
    data = inputs._raw
    with ExitStack() as guards:
        guards.callback(inputs.check_full_method)
        graph = _protect(
            guards,
            _compile_graph(data.legacy, inputs._source_root, _source_descriptor(data.legacy)),
        )
        contract = data.legacy.sources.adapter.time_contract
        clock = _protect(
            guards,
            freeze_clock_resolution(
                time_contract=contract,
                expected_time_contract_sha256=sha256_hex(canonical(contract, _ROOTS)),
                resolution_status="resolved",
                provenance_ids=("declared_adapter_clock_uncertainty",),
            ),
        )
        if canonical(graph, _ROOTS) != canonical(inputs.prepared.graph, _ROOTS) or canonical(
            clock, _ROOTS
        ) != canonical(args["clock_resolution"], _ROOTS):
            raise ValueError("case graph/clock differs from raw reconstruction replay")
        resolutions = {
            r.observation.observation_id: r for r in data.legacy.entity_registry.resolutions
        }
        sources = args["sources"]
        if len(sources) != len(resolutions) or {
            s.observation.observation_id for s in sources
        } != set(resolutions):
            raise ValueError("case replay requires the complete source collection")
        requests, facts, scalar_values = [], [], []
        for source, pin in zip(sources, args["expected_source_sha256s"], strict=True):
            with _replayed_repair_measurement_source(
                source=source,
                expected_source_sha256=pin,
                time_contract=contract,
                expected_time_contract_sha256=args["expected_time_contract_sha256"],
                store=args["store"],
            ) as fresh:
                resolution = resolutions[fresh.observation.observation_id]
                expected_candidate = (
                    resolution.binding.service_entity_key
                    if resolution.binding.status == "bound"
                    else None
                )
                if (
                    fresh.candidate_service_key != expected_candidate
                    or fresh.observed_entity_key != resolution.observed_entity_key
                    or fresh.observation.entity_binding_id is not None
                    or canonical(fresh.observation, _ROOTS)
                    != canonical(
                        resolution.observation.model_copy(update={"entity_binding_id": None}),
                        _ROOTS,
                    )
                ):
                    raise ValueError("case replay source differs from raw entity binding")
                request = _protect(guards, _request_values(fresh))
                fact = _protect(guards, _signal_values(fresh))
                static = _protect(guards, tuple(_fixed_scalar_terms(fresh, fact)))
                requests.append(request)
                facts.append(fact)
                scalar_values.append((static, fresh.quality.formal_scoring_eligible))
        # Only compact derived terms survive the per-source scope, never a
        # second complete raw/source collection. No future episodes are computed.
        ordered = tuple(sorted(requests, key=lambda row: row.request_key))
        catalog = _protect(
            guards,
            _seal(
                RepairQueryCatalog(
                    semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256,
                    opaque_case_id=contract.opaque_case_id,
                    candidate_freeze_sha256=data.legacy.frozen_candidates.freeze_sha256,
                    registry_sha256=data.legacy.entity_registry.registry_sha256,
                    time_contract_sha256=args["expected_time_contract_sha256"],
                    candidate_service_keys=data.legacy.frozen_candidates.candidate_service_keys,
                    requests=ordered,
                    graph_pair_sha256=graph.graph_pair.pair_sha256,
                    clock_resolution_sha256=clock.resolution_sha256,
                    time_needs=_time_need_values(ordered, graph.graph_pair, clock),
                    catalog_sha256="0" * 64,
                ),
                "catalog_sha256",
            ),
        )
        _validate_catalog(catalog)
        if canonical(catalog, _ROOTS) != canonical(runtime.catalog, _ROOTS):
            raise ValueError("case catalog differs from complete source replay")
        entries = _protect(guards, _read_journal(runtime.root))
        names = tuple(sorted(path.name for path in runtime.root.iterdir()))
        expected_names = tuple(
            sorted(("head", "lock", *(f"{e.sequence:03d}.json" for e in entries)))
        )
        if names != expected_names:
            raise ValueError("case replay journal has unexpected files")
        head = _read_bytes(runtime.root / "head", 256)

        def journal_unchanged() -> None:
            if (
                tuple(sorted(path.name for path in runtime.root.iterdir())) != names
                or _read_bytes(runtime.root / "head", 256) != head
                or canonical(_read_journal(runtime.root), _ROOTS) != canonical(entries, _ROOTS)
            ):
                raise ValueError("case replay journal changed during consumption")

        guards.callback(journal_unchanged)
        session = _protect(guards, runtime._replay(catalog, entries))
        errors = _protect(
            guards,
            build_repair_error_sources(
                source_records=tuple(error.source_records for error in data.errors),
                dataset=graph.dataset,
                time_contract=contract,
            ),
        )
        if canonical(errors, _ROOTS) != canonical(inputs.prepared.error_sources, _ROOTS):
            raise ValueError("case errors differ from complete raw record replay")
        with _score_scope(
            inputs,
            catalog=catalog,
            session=session,
            graph=graph,
            clock=clock,
            fixed=tuple(facts),
            errors=errors,
            scalar_values=tuple(scalar_values),
        ) as result:
            yield result


@_module_scope("independent_replay", "fresh_exit", role="verification")
@contextmanager
def _replay_case_evidence(
    inputs: _CaseInputs, *, candidate_audit: CaseAudit
) -> Iterator[CaseAudit]:
    expected_type = RepairCaseAudit if inputs.preparation is None else RepairPreparedCaseAudit
    if type(candidate_audit) is not expected_type:
        raise ValueError("case replay requires the exact current audit")
    with ExitStack() as guards:
        _protect(guards, candidate_audit)
        with _rebuild_case_evidence(inputs) as rebuilt:
            if canonical(candidate_audit, _ROOTS) != canonical(rebuilt, _ROOTS):
                raise ValueError("case audit differs from complete source and score replay")
            yield rebuilt
