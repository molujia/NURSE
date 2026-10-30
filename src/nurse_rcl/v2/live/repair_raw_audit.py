"""Whole-file repair assembly, durable query input and complete score replay.

Prepared contexts are disposable data, never a trust token. Every audit build
and independent replay reopens all declared case files and reconstructs scalar,
error, candidate and graph inputs. No user-authored score/graph/source bundle is
accepted in place of these files. Audit reading alone confers no authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterConfiguration
from nurse_rcl.v2.aiops25_inventory import AIOPS25_PARQUET_FILES, AuthorizedAIOps25Queue
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.dynamic_verification.source_store import DirectorySourceBlockStore
from nurse_rcl.v2.live.aiops25_sources import read_aiops25_live_sources
from nurse_rcl.v2.live.repair_assembly import (
    REPAIR_DIAGNOSTIC_SEMANTICS_SHA256,
    RepairDiagnosticInputs,
    assemble_aiops25_repair_diagnostics,
    assemble_rcabench_repair_diagnostics,
)
from nurse_rcl.v2.live.repair_error_references import RepairErrorSources, build_repair_error_sources
from nurse_rcl.v2.live.repair_graph import (
    REPAIR_GRAPH_SEMANTICS_SHA256,
    RepairLiveGraph,
    _compile_graph,
    _source_descriptor,
)
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_preparation_guard import (
    _check_all,
    _guard_live_sources,
    _guard_repair_assembly,
)
from nurse_rcl.v2.live.repair_queries import _CatalogChecks, _replayed_repair_query_catalog
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_query_session import RepairQueryRuntime, RepairQuerySession
from nurse_rcl.v2.live.repair_ranking import (
    REPAIR_RANKING_SEMANTICS_SHA256,
    RepairRanking,
    build_repair_ranking,
)
from nurse_rcl.v2.live.repair_safety import RepairModel, canonical, require_unchanged, snapshot
from nurse_rcl.v2.live.repair_sources import (
    RepairMeasurementSource,
    build_repair_measurement_source,
)
from nurse_rcl.v2.live.repair_timing import _preparation_scope
from nurse_rcl.v2.live.sources import file_sha256, read_rcabench_live_sources
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterConfiguration
from nurse_rcl.v2.rcabench_inventory import RCABENCH_PARQUET_FILES
from nurse_rcl.v2.temporal_condensation import freeze_clock_resolution

REPAIR_RAW_AUDIT_SCHEMA_VERSION: Final = "nurse.v2.live-repair-raw-audit/1"
REPAIR_RAW_AUDIT_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-raw-audit/1",
            ranking=REPAIR_RANKING_SEMANTICS_SHA256,
            assembly=REPAIR_DIAGNOSTIC_SEMANTICS_SHA256,
            graph=REPAIR_GRAPH_SEMANTICS_SHA256,
            source="reopen_complete_declared_case_files_adapter_assembly_graph_all_scalars_four_error_slots",
            pin="independent_source_descriptor_and_terminal_session",
            ledger="read_only_replay_no_advance_no_budget_reset",
            files="no_symlinks_both_exit_full_content_pins_including_absent_sources",
            authority="complete_replay_required_serialization_is_not_a_receipt",
        )
    )
)


@dataclass(frozen=True, slots=True)
class PreparedRepairCase:
    runtime: RepairQueryRuntime
    error_sources: RepairErrorSources
    error_observation_sha256s: tuple[str, ...]
    graph: RepairLiveGraph
    source_descriptor_sha256: str
    source_files_sha256: str
    candidate_freeze_sha256: str
    raw_file_authority: Literal[False] = False


@dataclass(frozen=True, slots=True)
class _PreparationChecks:
    """Private lexical checks and raw inputs; old callers keep the full check."""

    catalog: _CatalogChecks
    raw: Callable[[], None]
    data: RepairDiagnosticInputs

    def __call__(self) -> None:
        self.catalog()


class RepairRawAudit(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-raw-audit/1"] = REPAIR_RAW_AUDIT_SCHEMA_VERSION
    semantics_sha256: SHA256
    profile: Literal["v2-live-evidence-repair"] = "v2-live-evidence-repair"
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    session_sha256: SHA256
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    pre_ranking_semantics: Literal[False] = False
    raw_file_and_score_replayed: Literal[True] = True
    result_sha256: SHA256


_ROOTS = (
    RepairRawAudit,
    RepairLiveGraph,
    RepairErrorSources,
    RepairMeasurementSource,
    RCABenchAdapterConfiguration,
    AIOps25AdapterConfiguration,
    AuthorizedAIOps25Queue,
)


def _file_pins(root: Path, dataset: str) -> tuple[tuple[str, str | None], ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("repair source root must be a real case directory")
    names = (
        ("env.json", *RCABENCH_PARQUET_FILES)
        if dataset == "rcabench"
        else (".prepared.json", "env.json", *AIOPS25_PARQUET_FILES)
    )
    rows = []
    for name in names:
        path = root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("repair source file must not be a symlink or special file")
        rows.append((name, file_sha256(path) if path.is_file() else None))
    return tuple(rows)


@_preparation_scope("raw_input_prepare", "raw_input_exit")
@contextmanager
def _fixed_repair_scope(
    *,
    source_root: Path,
    dataset: Literal["rcabench", "aiops25"],
    opaque_case_id: str,
    configuration: RCABenchAdapterConfiguration | AIOps25AdapterConfiguration,
    expected_source_descriptor_sha256: str,
    store_root: Path,
    ledger_root: Path,
    source_case_id: str | None = None,
    authorized_queue: AuthorizedAIOps25Queue | None = None,
) -> Iterator[tuple[RepairPreparationData, _PreparationChecks]]:
    if type(dataset) is not str or dataset not in ("rcabench", "aiops25"):
        raise ValueError("repair source dataset must be explicit")
    if (
        dataset == "rcabench"
        and (
            type(configuration) is not RCABenchAdapterConfiguration
            or source_case_id is not None
            or authorized_queue is not None
        )
    ) or (
        dataset == "aiops25"
        and (
            type(configuration) is not AIOps25AdapterConfiguration
            or type(source_case_id) is not str
            or type(authorized_queue) is not AuthorizedAIOps25Queue
        )
    ):
        raise ValueError("repair source configuration or authorization differs from dataset")
    require_pin(expected_source_descriptor_sha256)
    root = Path(source_root)
    files = _file_pins(root, dataset)
    config_context = (
        configuration,
        authorized_queue,
        source_case_id,
        opaque_case_id,
        dataset,
        expected_source_descriptor_sha256,
        str(root),
        str(store_root),
        str(ledger_root),
    )
    with ExitStack() as guards:
        raw_checks: list[Callable[[], None]] = []

        def protect(value: object) -> None:
            check = partial(require_unchanged, value, _ROOTS, snapshot(value, _ROOTS))
            raw_checks.append(check)
            guards.callback(check)

        protect(config_context)

        def unchanged_files() -> None:
            if _file_pins(root, dataset) != files:
                raise ValueError("repair source files changed during complete score replay")

        guards.callback(unchanged_files)
        raw_checks.append(unchanged_files)
        if dataset == "rcabench":
            with _module_span("preparation_read"):
                source = read_rcabench_live_sources(
                    root,
                    opaque_case_id=opaque_case_id,
                    configuration=cast(RCABenchAdapterConfiguration, configuration),
                )
            if source.descriptor()["descriptor_sha256"] != expected_source_descriptor_sha256:
                raise ValueError("repair source differs from independent descriptor pin")
            with _module_span("preparation_guard"):
                raw_checks.append(guards.enter_context(_guard_live_sources(source)))
            with _module_span("preparation_assembly"):
                data = assemble_rcabench_repair_diagnostics(source)
            if data.legacy.sources is not source:
                raise ValueError("repair assembly replaced the independently read source")
        else:
            with _module_span("preparation_read"):
                source25 = read_aiops25_live_sources(
                    root,
                    opaque_case_id=opaque_case_id,
                    source_case_id=cast(str, source_case_id),
                    authorized_queue=cast(AuthorizedAIOps25Queue, authorized_queue),
                    configuration=cast(AIOps25AdapterConfiguration, configuration),
                )
            if source25.descriptor()["descriptor_sha256"] != expected_source_descriptor_sha256:
                raise ValueError("repair source differs from independent descriptor pin")
            with _module_span("preparation_guard"):
                raw_checks.append(guards.enter_context(_guard_live_sources(source25)))
            with _module_span("preparation_assembly"):
                data = assemble_aiops25_repair_diagnostics(source25)
            if data.legacy.sources is not source25:
                raise ValueError("repair assembly replaced the independently read source")
        # These values were assembled directly from freshly opened full files;
        # no caller-provided diagnostics or rewritten adapter identities enter.
        legacy = data.legacy
        with _module_span("preparation_guard"):
            raw_checks.append(guards.enter_context(_guard_repair_assembly(data)))
        with _module_span("preparation_graph"):
            graph = _compile_graph(legacy, root, _source_descriptor(legacy))
        canonical(graph, _ROOTS)
        protect(graph)
        contract = legacy.sources.adapter.time_contract
        contract_pin = sha256_hex(canonical_json_bytes(contract.model_dump(mode="json")))
        clock = freeze_clock_resolution(
            time_contract=contract,
            expected_time_contract_sha256=contract_pin,
            resolution_status="resolved",
            provenance_ids=("declared_adapter_clock_uncertainty",),
        )
        blocks = Path(store_root)
        if blocks.is_symlink():
            raise ValueError("repair block store may not be a symlink")
        blocks.mkdir(parents=True, exist_ok=True)
        store = DirectorySourceBlockStore(root=str(blocks.resolve(strict=True)))
        repairs = {r.signal_id: r for r in data.signals}
        if set(repairs) != {s.signal_id for s in legacy.signals} or len(repairs) != len(
            legacy.signals
        ):
            raise ValueError("repair source signal collection is incomplete")
        with _module_span("preparation_sources"):
            sources = tuple(
                sorted(
                    (
                        build_repair_measurement_source(
                            signal=s,
                            repair=repairs[s.signal_id],
                            time_contract=contract,
                            dataset=dataset,
                            store=store,
                        )
                        for s in legacy.signals
                    ),
                    key=lambda s: s.signal_id,
                )
            )
        for signal_source in sources:
            protect(signal_source)
        arguments: dict[str, Any] = dict(
            sources=sources,
            expected_source_sha256s=tuple(s.source_sha256 for s in sources),
            time_contract=contract,
            expected_time_contract_sha256=contract_pin,
            candidate_universe=legacy.candidate_universe,
            frozen_candidates=legacy.frozen_candidates,
            entity_registry=legacy.entity_registry,
            store=store,
            graph_input_manifest=graph.manifest,
            expected_graph_input_manifest_sha256=graph.manifest.manifest_sha256,
            graph_pair=graph.graph_pair,
            expected_graph_pair_sha256=graph.graph_pair.pair_sha256,
            clock_resolution=clock,
            expected_clock_resolution_sha256=clock.resolution_sha256,
        )
        with _module_span("preparation_catalog"):
            catalog, check_catalog = guards.enter_context(
                _replayed_repair_query_catalog(**arguments)
            )
        with _module_span("preparation_errors"):
            errors = build_repair_error_sources(
                source_records=tuple(e.source_records for e in data.errors),
                dataset=dataset,
                time_contract=contract,
            )
        protect(errors)
        prepared = RepairPreparationData(
            diagnostics=data,
            sources=sources,
            catalog=catalog,
            clock=clock,
            error_sources=errors,
            error_observation_sha256s=tuple(e.result_sha256 for e in data.errors),
            graph=graph,
            source_descriptor_sha256=expected_source_descriptor_sha256,
            source_files_sha256=sha256_hex(canonical_json_bytes(files)),
            candidate_freeze_sha256=legacy.frozen_candidates.freeze_sha256,
        )
        retained_checks = tuple(raw_checks)
        yield (
            prepared,
            _PreparationChecks(
                catalog=check_catalog,
                raw=lambda: _check_all(retained_checks),
                data=data,
            ),
        )


def _bind_preparation_runtime(
    data: RepairPreparationData,
    *,
    store_root: Path,
    ledger_root: Path,
) -> PreparedRepairCase:
    """Bind this invocation's handles only; never prepare or spend query budget."""
    if type(data) is not RepairPreparationData:
        raise ValueError("runtime binding requires exact fixed preparation data")
    legacy, graph = data.diagnostics.legacy, data.graph
    contract = legacy.sources.adapter.time_contract
    contract_pin = sha256_hex(canonical_json_bytes(contract.model_dump(mode="json")))
    runtime = RepairQueryRuntime(
        catalog=data.catalog,
        expected_catalog_sha256=data.catalog.catalog_sha256,
        ledger_root=ledger_root,
        sources=data.sources,
        expected_source_sha256s=tuple(s.source_sha256 for s in data.sources),
        time_contract=contract,
        expected_time_contract_sha256=contract_pin,
        candidate_universe=legacy.candidate_universe,
        frozen_candidates=legacy.frozen_candidates,
        entity_registry=legacy.entity_registry,
        store=DirectorySourceBlockStore(root=str(require_plain_path(store_root))),
        graph_input_manifest=graph.manifest,
        expected_graph_input_manifest_sha256=graph.manifest.manifest_sha256,
        graph_pair=graph.graph_pair,
        expected_graph_pair_sha256=graph.graph_pair.pair_sha256,
        clock_resolution=data.clock,
        expected_clock_resolution_sha256=data.clock.resolution_sha256,
    )
    return PreparedRepairCase(
        runtime=runtime,
        error_sources=data.error_sources,
        error_observation_sha256s=data.error_observation_sha256s,
        graph=graph,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
    )


@contextmanager
def _prepared_repair_scope(
    **raw_arguments: Any,
) -> Iterator[tuple[PreparedRepairCase, _PreparationChecks]]:
    with _fixed_repair_scope(**raw_arguments) as (data, checks):
        yield (
            _bind_preparation_runtime(
                data,
                store_root=raw_arguments["store_root"],
                ledger_root=raw_arguments["ledger_root"],
            ),
            checks,
        )


@contextmanager
def _prepared_from_raw(**raw_arguments: Any) -> Iterator[PreparedRepairCase]:
    with _prepared_repair_scope(**raw_arguments) as (prepared, _):
        yield prepared


def prepare_repair_case(**raw_arguments: Any) -> PreparedRepairCase:
    """Prepare a complete dynamic input; performs no query and grants no audit trust."""
    with _prepared_from_raw(**raw_arguments) as prepared:
        return prepared


def run_repair_queries_from_raw(
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    **raw_arguments: Any,
) -> RepairQuerySession:
    """One full raw/catalog invocation, feedback-first steps, no reusable trust handle."""
    # Acquired after preparation, but released only after the raw owner exits.
    with ExitStack() as lock_owner, ExitStack() as outputs:  # noqa: SIM117
        with _prepared_repair_scope(**raw_arguments) as (prepared, check_catalog):
            runtime = prepared.runtime
            created = lock_owner.enter_context(runtime._locked_ledger())
            # Runtime bindings outlive raw exits under the same ledger lock.
            with query_step("runtime_inputs"):
                check_runtime = lock_owner.enter_context(runtime._protected_inputs())

            def unchanged() -> None:
                with query_step("input_guard"), ExitStack() as checks:
                    checks.callback(check_catalog)
                    checks.callback(check_runtime)

            def emit(row: dict[str, Any]) -> None:
                try:
                    if progress is not None:
                        progress(row)
                finally:
                    unchanged()

            emit(
                dict(
                    event="query_inputs_ready",
                    candidates=len(runtime.catalog.candidate_service_keys),
                )
            )
            session = None
            for index in range(9):
                session = runtime._advance_values(
                    runtime.catalog,
                    unchanged,
                    created=created and index == 0,
                )
                outputs.callback(
                    require_unchanged,
                    session,
                    (RepairQuerySession,),
                    snapshot(session, (RepairQuerySession,)),
                )
                emit(
                    dict(
                        event="query_progress",
                        completed=session.paid_count,
                        total=8,
                        stop_reason=session.stop_reason,
                        failure_reason=session.failure_reason,
                    )
                )
                if session.stop_reason or session.failure_reason:
                    break
            if session is None or session.failure_reason or not session.stop_reason:
                raise ValueError("whole-case verification did not terminate within eight queries")
    return session


@contextmanager
def _replayed_raw_audit(
    *, expected_session_sha256: str, **raw_arguments: Any
) -> Iterator[RepairRawAudit]:
    require_pin(expected_session_sha256)
    with _prepared_from_raw(**raw_arguments) as prepared:
        ranking = build_repair_ranking(
            runtime=prepared.runtime,
            expected_session_sha256=expected_session_sha256,
            error_sources=prepared.error_sources,
            expected_error_sources_sha256=prepared.error_sources.result_sha256,
            expected_error_observation_sha256s=prepared.error_observation_sha256s,
        )
        result = RepairRawAudit(
            semantics_sha256=REPAIR_RAW_AUDIT_SEMANTICS_SHA256,
            dataset=prepared.graph.dataset,
            opaque_case_id=ranking.opaque_case_id,
            source_descriptor_sha256=prepared.source_descriptor_sha256,
            source_files_sha256=prepared.source_files_sha256,
            candidate_freeze_sha256=prepared.candidate_freeze_sha256,
            catalog_sha256=prepared.runtime.catalog.catalog_sha256,
            session_sha256=expected_session_sha256,
            graph_manifest_sha256=prepared.graph.manifest.manifest_sha256,
            graph=prepared.graph,
            ranking=ranking,
            result_sha256="0" * 64,
        )
        document = cast(dict[str, object], plain(result))
        document.pop("result_sha256")
        result = result.model_copy(
            update={"result_sha256": sha256_hex(canonical_json_bytes(document))}
        )
        before = snapshot(result, _ROOTS)
        try:
            yield result
        finally:
            require_unchanged(result, _ROOTS, before)


def build_repair_raw_audit(*, expected_session_sha256: str, **raw_arguments: Any) -> RepairRawAudit:
    with _replayed_raw_audit(
        expected_session_sha256=expected_session_sha256, **raw_arguments
    ) as result:
        return result


def require_repair_raw_audit(
    *,
    audit: RepairRawAudit,
    expected_audit_sha256: str,
    expected_session_sha256: str,
    **raw_arguments: Any,
) -> RepairRawAudit:
    if type(audit) is not RepairRawAudit:
        raise ValueError("exact repair raw audit required")
    before = snapshot(audit, _ROOTS)
    try:
        require_pin(expected_audit_sha256)
        rebuilt = build_repair_raw_audit(
            expected_session_sha256=expected_session_sha256, **raw_arguments
        )
        if rebuilt.result_sha256 != expected_audit_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(audit, _ROOTS):
            raise ValueError("repair raw audit differs from complete source and score replay")
        return rebuilt
    finally:
        require_unchanged(audit, _ROOTS, before)
