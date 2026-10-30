"""Fixed-data construction from one pinned case; no query, score or replay pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterConfiguration
from nurse_rcl.v2.aiops25_inventory import AIOPS25_PARQUET_FILES, AuthorizedAIOps25Queue
from nurse_rcl.v2.candidate_freeze import require_frozen_candidate_universe
from nurse_rcl.v2.dynamic_verification.source_store import DirectorySourceBlockStore
from nurse_rcl.v2.live import repair_preparation_codec as codec
from nurse_rcl.v2.live.aiops25_sources import _read_aiops25_source_inputs
from nurse_rcl.v2.live.repair_assembly import (
    assemble_aiops25_repair_diagnostics,
    assemble_rcabench_repair_diagnostics,
)
from nurse_rcl.v2.live.repair_error_references import _error_source_values
from nurse_rcl.v2.live.repair_graph import _compile_graph, _source_descriptor
from nurse_rcl.v2.live.repair_module_timing import _module_call, _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationData,
    RepairPreparationKey,
)
from nurse_rcl.v2.live.repair_queries import (
    REPAIR_QUERY_SEMANTICS_SHA256,
    RepairQueryCatalog,
    _request_values,
    _seal,
    _time_need_values,
    _validate_catalog,
)
from nurse_rcl.v2.live.repair_sources import _measurement_source_values
from nurse_rcl.v2.live.sources import _read_rcabench_source_inputs
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterConfiguration
from nurse_rcl.v2.rcabench_inventory import RCABENCH_PARQUET_FILES
from nurse_rcl.v2.temporal_condensation import freeze_clock_resolution


@_module_call("preparation_read")
def _pinned_input_bytes(root: Path, key: RepairPreparationKey) -> dict[str, bytes]:
    names = (
        ("env.json", *RCABENCH_PARQUET_FILES) if key.dataset == "rcabench"
        else (".prepared.json", "env.json", *AIOPS25_PARQUET_FILES)
    )
    if tuple(name for name, _ in key.source_files) != names:
        raise ValueError("normal fixed preparation source inventory differs")
    require_plain_path(root)
    buffers: dict[str, bytes] = {}
    remaining = 8 * 1024**3  # Bound private input bytes by the existing 8-GiB worker ceiling.
    for name, pin in key.source_files:
        path = root / name
        if path.is_symlink():
            raise ValueError("normal fixed preparation source is a symlink")
        if pin is None:
            if path.exists():
                raise ValueError("normal fixed preparation absent source appeared")
            continue
        size = require_plain_path(path).stat().st_size
        if size > remaining:
            raise ValueError("normal fixed input bytes exceed the worker memory ceiling")
        raw = codec.read_member(path, limit=size)
        if sha256_hex(raw) != pin:
            raise ValueError("normal fixed preparation source differs from independent pin")
        remaining -= len(raw)
        buffers[name] = raw
    return buffers


@_module_call("preparation_assembly")
def construct_normal_fixed_inputs(
    *, key: RepairPreparationKey, raw_arguments: dict[str, Any], block_root: Path,
) -> RepairPreparationData:
    root = Path(raw_arguments["source_root"])
    buffers = _pinned_input_bytes(root, key)
    if key.dataset == "rcabench":
        source = _read_rcabench_source_inputs(
            root, opaque_case_id=key.opaque_case_id,
            configuration=cast(RCABenchAdapterConfiguration, raw_arguments["configuration"]),
            input_bytes=buffers,
        )
        diagnostics = assemble_rcabench_repair_diagnostics(source)
    else:
        source25 = _read_aiops25_source_inputs(
            root, opaque_case_id=key.opaque_case_id,
            source_case_id=raw_arguments["source_case_id"],
            authorized_queue=cast(AuthorizedAIOps25Queue, raw_arguments["authorized_queue"]),
            configuration=cast(AIOps25AdapterConfiguration, raw_arguments["configuration"]),
            input_bytes=buffers,
        )
        diagnostics = assemble_aiops25_repair_diagnostics(source25)
    del buffers
    legacy = diagnostics.legacy
    if legacy.sources.descriptor()["descriptor_sha256"] != key.source_descriptor_sha256:
        raise ValueError("normal fixed source descriptor differs from independent input")
    with _module_span("preparation_graph"):
        graph = _compile_graph(legacy, root, _source_descriptor(legacy))
    contract = legacy.sources.adapter.time_contract
    contract_pin = sha256_hex(canonical_json_bytes(contract.model_dump(mode="json")))
    clock = freeze_clock_resolution(
        time_contract=contract, expected_time_contract_sha256=contract_pin,
        resolution_status="resolved", provenance_ids=("declared_adapter_clock_uncertainty",),
    )
    block_root.mkdir(mode=0o700)
    store = DirectorySourceBlockStore(root=str(block_root.resolve(strict=True)))
    repairs = {item.signal_id: item for item in diagnostics.signals}
    if len(repairs) != len(diagnostics.signals) or set(repairs) != {
        item.signal_id for item in legacy.signals
    }:
        raise ValueError("normal fixed signal collection differs")
    with _module_span("preparation_sources"):
        sources = tuple(sorted((
            _measurement_source_values(
                signal=signal, repair=repairs[signal.signal_id], time_contract=contract,
                dataset=key.dataset, store=store,
            ) for signal in legacy.signals
        ), key=lambda item: item.signal_id))
    frozen, registry = legacy.frozen_candidates, legacy.entity_registry
    require_frozen_candidate_universe(frozen=frozen, candidate_universe=legacy.candidate_universe)
    if not registry.verify() or registry.opaque_case_id != contract.opaque_case_id or (
        registry.decision_time_unix_ns != contract.decision_time_unix_ns
        or frozen.opaque_case_id != contract.opaque_case_id
    ):
        raise ValueError("normal fixed catalog registry/case differs")
    resolutions = {r.observation.observation_id: r for r in registry.resolutions}
    if len(sources) != len(resolutions) or {
        s.observation.observation_id for s in sources
    } != set(resolutions):
        raise ValueError("normal fixed catalog source collection differs")
    for item in sources:
        resolution = resolutions[item.observation.observation_id]
        expected_candidate = (
            resolution.binding.service_entity_key if resolution.binding.status == "bound" else None
        )
        if (
            item.candidate_service_key != expected_candidate
            or item.observed_entity_key != resolution.observed_entity_key
            or item.observation
            != resolution.observation.model_copy(update={"entity_binding_id": None})
        ):
            raise ValueError("normal fixed catalog source binding differs")
    if not graph.graph_pair.verify_inputs(
        time_contract=contract, candidate_universe=legacy.candidate_universe,
        frozen_candidates=frozen, entity_registry=registry, graph_input_manifest=graph.manifest,
        expected_graph_input_manifest_sha256=graph.manifest.manifest_sha256,
        expected_minimum_coverage_fraction=0.8,
    ):
        raise ValueError("normal fixed graph input binding differs")
    requests = tuple(sorted((_request_values(s) for s in sources), key=lambda r: r.request_key))
    catalog = _seal(RepairQueryCatalog(
        semantics_sha256=REPAIR_QUERY_SEMANTICS_SHA256, opaque_case_id=contract.opaque_case_id,
        candidate_freeze_sha256=frozen.freeze_sha256, registry_sha256=registry.registry_sha256,
        time_contract_sha256=contract_pin, candidate_service_keys=frozen.candidate_service_keys,
        requests=requests, graph_pair_sha256=graph.graph_pair.pair_sha256,
        clock_resolution_sha256=clock.resolution_sha256,
        time_needs=_time_need_values(requests, graph.graph_pair, clock), catalog_sha256="0" * 64,
    ), "catalog_sha256")
    _validate_catalog(catalog)
    with _module_span("preparation_errors"):
        errors = _error_source_values(
            source_records=tuple(error.source_records for error in diagnostics.errors),
            dataset=key.dataset, time_contract=contract,
        )
    return RepairPreparationData(
        diagnostics=diagnostics, sources=sources, catalog=catalog, graph=graph, clock=clock,
        error_sources=errors,
        error_observation_sha256s=tuple(error.result_sha256 for error in diagnostics.errors),
        source_descriptor_sha256=key.source_descriptor_sha256,
        source_files_sha256=sha256_hex(canonical_json_bytes(key.source_files)),
        candidate_freeze_sha256=frozen.freeze_sha256,
    )
