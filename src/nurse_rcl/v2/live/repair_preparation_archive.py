"""Per-case preparation persistence. Data survives; validation authority does not."""

from __future__ import annotations

import ctypes
import fcntl
import os
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterConfiguration
from nurse_rcl.v2.aiops25_inventory import AuthorizedAIOps25Queue
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.dynamic_verification.source_store import DirectorySourceBlockStore
from nurse_rcl.v2.live import repair_raw_audit as raw
from nurse_rcl.v2.live.repair_assembly import REPAIR_DIAGNOSTIC_SEMANTICS_SHA256
from nurse_rcl.v2.live.repair_driver import _fsync, publish_json
from nurse_rcl.v2.live.repair_module_timing import _module_call, _module_span
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationBinding,
    RepairPreparationData,
    RepairPreparationKey,
    RepairPreparationManifest,
    RepairPreparationMember,
    RepairPreparationReceipt,
    RepairPreparationRun,
    RepairPreparationTable,
)
from nurse_rcl.v2.live.repair_preparation_codec import (
    _ROOTS,
    member_digest,
    read_member,
    read_preparation_data,
    write_preparation_data,
)
from nurse_rcl.v2.live.repair_preparation_guard import (
    _check_all,
    _guard_live_sources,
    _guard_repair_assembly,
)
from nurse_rcl.v2.live.repair_queries import REPAIR_QUERY_SEMANTICS_SHA256, _validate_catalog
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_CAPACITY_SHA256,
    REPAIR_MAX_JSON_BYTES,
    canonical,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.live.repair_timing import _preparation_call
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterConfiguration

# Finite code dependency inventory, separately frozen from execution/report code.
# Preparation modules and every numerical/validation dependency executed by them
# belong here; report and ranking implementations do not.
PREPARATION_DEPENDENCIES = tuple(
    sorted(
        "src/nurse_rcl/" + name + ".py"
        for name in (
            "artifacts",
            "schemas",
            "leakage",
            "v2/aiops25_adapter",
            "v2/aiops25_inventory",
            "v2/rcabench_adapter",
            "v2/rcabench_inventory",
            "v2/candidate_freeze",
            "v2/candidate_universe",
            "v2/event_grouping",
            "v2/focus_selection",
            "v2/graph_snapshots",
            "v2/diagnostic_schemas",
            "v2/hypothesis_guard",
            "v2/mechanism_library",
            "v2/observation_quality",
            "v2/observation_schemas",
            "v2/signal_transforms",
            "v2/temporal_condensation",
            "v2/temporal_entity_registry",
            "v2/propagation_scoring",
            "v2/safe_serialization",
            "v2/report_audit_json",
            "v2/exchange_json",
            "v2/dynamic_verification/contracts",
            "v2/dynamic_verification/safety",
            "v2/dynamic_verification/source_store",
            "v2/dynamic_verification/requests",
            "v2/live/sources",
            "v2/live/aiops25_sources",
            "v2/live/assembly",
            "v2/live/staged_case",
            "v2/live/graph",
            "v2/live/repair_graph",
            "v2/live/repair_assembly",
            "v2/live/repair_sources",
            "v2/live/repair_observations",
            "v2/live/repair_lifecycle",
            "v2/live/repair_errors",
            "v2/live/repair_error_references",
            "v2/live/repair_fixed_facts",
            "v2/live/repair_queries",
            "v2/live/repair_measurements",
            "v2/live/repair_temporal",
            "v2/live/repair_preparation_guard",
            "v2/live/repair_raw_audit",
            "v2/live/repair_preparation_artifacts",
            "v2/live/repair_preparation_codec",
            "v2/live/repair_preparation_archive",
            "v2/live/repair_safety",
            "v2/live/repair_projection",
        )
    )
)
_SEMANTICS = sha256_hex(
    canonical_json_bytes(
        dict(
            format="nurse.v2.live-repair-preparation-manifest/1",
            assembly=REPAIR_DIAGNOSTIC_SEMANTICS_SHA256,
            catalog=REPAIR_QUERY_SEMANTICS_SHA256,
            capacity=REPAIR_CAPACITY_SHA256,
            q_base=8,
            q_max=12,
        )
    )
)
_ARTIFACT_ROOTS = (RepairPreparationKey, RepairPreparationManifest, RepairPreparationReceipt)


def _input_fields(arguments: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "source_root",
        "dataset",
        "opaque_case_id",
        "configuration",
        "expected_source_descriptor_sha256",
        "store_root",
        "ledger_root",
        "source_case_id",
        "authorized_queue",
    }
    if type(arguments) is not dict or set(arguments) - allowed:
        raise ValueError("preparation requires exact raw arguments")
    dataset, config = arguments["dataset"], arguments["configuration"]
    queue, source_id = arguments.get("authorized_queue"), arguments.get("source_case_id")
    if (
        (
            dataset == "rcabench"
            and (
                type(config) is not RCABenchAdapterConfiguration
                or queue is not None
                or source_id is not None
            )
        )
        or (
            dataset == "aiops25"
            and (
                type(config) is not AIOps25AdapterConfiguration
                or type(queue) is not AuthorizedAIOps25Queue
                or type(source_id) is not str
            )
        )
        or type(dataset) is not str
        or dataset not in ("rcabench", "aiops25")
    ):
        raise ValueError("preparation dataset/configuration/authorization differs")
    config_raw = canonical(config, (RCABenchAdapterConfiguration, AIOps25AdapterConfiguration))
    return dict(
        dataset=dataset,
        opaque_case_id=arguments["opaque_case_id"],
        source_descriptor_sha256=require_pin(arguments["expected_source_descriptor_sha256"]),
        source_files=raw._file_pins(require_plain_path(arguments["source_root"]), dataset),
        configuration_sha256=sha256_hex(config_raw),
        authorized_queue_sha256=None
        if queue is None
        else sha256_hex(canonical(queue, (AuthorizedAIOps25Queue,))),
        source_case_id_sha256=None
        if source_id is None
        else sha256_hex(canonical_json_bytes(source_id)),
    )


def preparation_key(
    *,
    raw_arguments: dict[str, Any],
    input_sha256: str,
    code_root: Path,
) -> RepairPreparationKey:
    require_pin(input_sha256)
    root = require_plain_path(code_root)
    body = dict(
        schema_version="nurse.v2.live-repair-preparation-key/1",
        **_input_fields(raw_arguments),
        input_sha256=input_sha256,
        preparation_code=tuple(
            (name, file_sha256(require_plain_path(root / name)))
            for name in PREPARATION_DEPENDENCIES
        ),
        semantics_sha256=_SEMANTICS,
        q_base=8,
        q_max=12,
    )
    return RepairPreparationKey(**body, key_sha256=sha256_hex(canonical_json_bytes(body)))


def _require_key(key: RepairPreparationKey, arguments: dict[str, Any]) -> None:
    if type(key) is not RepairPreparationKey:
        raise ValueError("preparation requires an exact independently bound key")
    RepairPreparationKey.model_validate_json(canonical(key, _ARTIFACT_ROOTS))
    expected = preparation_key(
        raw_arguments=arguments,
        input_sha256=key.input_sha256,
        code_root=Path(__file__).resolve().parents[4],
    )
    if key != expected:
        raise ValueError("preparation key differs from current sources/code/configuration")


def _collections(data: RepairPreparationData) -> tuple[tuple[str, int, str], ...]:
    legacy = data.diagnostics.legacy
    rows = (
        ("signals", tuple(s.signal_id for s in legacy.signals)),
        ("diagnostics", tuple(s.signal_id for s in data.diagnostics.signals)),
        ("sources", tuple(s.source_sha256 for s in data.sources)),
        ("candidates", legacy.frozen_candidates.candidate_service_keys),
        ("errors", data.error_observation_sha256s),
        ("tables", tuple(t.descriptor() for t in legacy.sources.tables)),
    )
    return tuple(
        (name, len(values), sha256_hex(canonical_json_bytes(values))) for name, values in rows
    )


def _tables(data: RepairPreparationData) -> tuple[RepairPreparationTable, ...]:
    return tuple(
        RepairPreparationTable(
            name=f"tables/{index:06d}.arrow",
            rows=item.table.num_rows,
            schema_sha256=sha256_hex(item.table.schema.serialize().to_pybytes()),
        )
        for index, item in enumerate(data.diagnostics.legacy.sources.tables)
    )


@contextmanager
def _guard_preparation_data(data: RepairPreparationData) -> Iterator[Callable[[], None]]:
    if type(data) is not RepairPreparationData or type(data.sources) is not tuple:
        raise ValueError("preparation data representation differs")
    with ExitStack() as guards:
        checks = [
            guards.enter_context(_guard_live_sources(data.diagnostics.legacy.sources)),
            guards.enter_context(_guard_repair_assembly(data.diagnostics)),
        ]
        # Preserve each established capacity root rather than combine all data.
        for value in (*data.sources, data.catalog, data.graph, data.clock, data.error_sources):
            check = partial(require_unchanged, value, _ROOTS, snapshot(value, _ROOTS))
            checks.append(check)
            guards.callback(check)
        shape = (
            data.diagnostics,
            data.sources,
            data.catalog,
            data.graph,
            data.clock,
            data.error_sources,
        )

        def headers() -> object:
            return (
                data.error_observation_sha256s,
                data.source_descriptor_sha256,
                data.source_files_sha256,
                data.candidate_freeze_sha256,
            )

        header_before = snapshot(headers(), _ROOTS)

        def binding() -> None:
            current = (
                data.diagnostics,
                data.sources,
                data.catalog,
                data.graph,
                data.clock,
                data.error_sources,
            )
            if any(a is not b for a, b in zip(shape, current, strict=True)):
                raise ValueError("preparation fixed-data bindings changed")
            require_unchanged(headers(), _ROOTS, header_before)

        checks.append(binding)
        guards.callback(binding)
        yield lambda: _check_all(tuple(checks))


def _require_data(data: RepairPreparationData, key: RepairPreparationKey) -> None:
    legacy = data.diagnostics.legacy
    if (
        data.diagnostics.dataset != key.dataset
        or legacy.sources.descriptor()["descriptor_sha256"] != key.source_descriptor_sha256
        or data.source_descriptor_sha256 != key.source_descriptor_sha256
        or data.source_files_sha256 != sha256_hex(canonical_json_bytes(key.source_files))
        or data.candidate_freeze_sha256 != legacy.frozen_candidates.freeze_sha256
        or data.catalog.candidate_freeze_sha256 != data.candidate_freeze_sha256
        or data.catalog.opaque_case_id != key.opaque_case_id
        or data.catalog.registry_sha256 != legacy.entity_registry.registry_sha256
        or data.catalog.time_contract_sha256
        != sha256_hex(canonical(legacy.sources.adapter.time_contract, _ROOTS))
        or data.catalog.graph_pair_sha256 != data.graph.graph_pair.pair_sha256
        or data.catalog.clock_resolution_sha256 != data.clock.resolution_sha256
        or tuple(s.signal_id for s in data.sources)
        != tuple(sorted(s.signal_id for s in legacy.signals))
        or tuple(s.signal_id for s in data.diagnostics.signals)
        != tuple(s.signal_id for s in legacy.signals)
        or data.error_observation_sha256s != tuple(e.result_sha256 for e in data.diagnostics.errors)
        or (legacy.q_base, legacy.q_max) != (key.q_base, key.q_max)
    ):
        raise ValueError("preparation complete collection/source binding differs")
    _validate_catalog(data.catalog)


def _member_inventory(root: Path) -> tuple[RepairPreparationMember, ...]:
    rows = []
    require_plain_path(root)
    for path in sorted(root.rglob("*")):
        require_plain_path(path)
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("preparation contains a nonregular member")
        name = path.relative_to(root).as_posix()
        if name == "manifest.json":
            continue
        size, digest = member_digest(path)
        rows.append(
            RepairPreparationMember(
                name=name,
                size_bytes=size,
                sha256=digest,
            )
        )
    return tuple(rows)


def _check_members(root: Path, manifest: RepairPreparationManifest) -> None:
    if _member_inventory(root) != manifest.members:
        raise ValueError("preparation member inventory/content differs from manifest pin")
    actual_dirs = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    expected_dirs = {str(Path(m.name).parent) for m in manifest.members if "/" in m.name}
    if actual_dirs != expected_dirs:
        raise ValueError("preparation directory inventory differs")


@_module_call("preparation_publish")
def _publish_directory(source: Path, target: Path) -> None:
    # Linux-only server execution: RENAME_NOREPLACE closes the empty-directory
    # race which ordinary rename cannot. Never fall back to overwriting rename.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


@_preparation_call("raw_input_prepare")
@_module_call("preparation_build")
def _build_preparation_archive(
    *,
    archive_root: Path,
    expected_key: RepairPreparationKey,
    raw_arguments: dict[str, Any],
    receipt_path: Path,
    execution_sha256: str,
    execution_code_sha256: str,
) -> tuple[RepairPreparationData, RepairPreparationManifest, RepairPreparationReceipt]:
    _require_key(expected_key, raw_arguments)
    require_pin(execution_sha256)
    require_pin(execution_code_sha256)
    parent = require_plain_path(archive_root.parent)
    require_plain_path(receipt_path.parent)
    if (
        archive_root.name != expected_key.key_sha256
        or archive_root.exists()
        or archive_root.is_symlink()
    ):
        raise ValueError("preparation archive already exists or has wrong key")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("preparation receipt already exists")
    # Partial content is deliberately retained on failure. A receipt is required
    # even if a crash occurs after atomic directory publication.
    lock = os.open(
        parent / (expected_key.key_sha256 + ".lock"),
        os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if archive_root.exists():
            raise ValueError("preparation archive was concurrently published")
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
        started = time.monotonic()
        arguments = {**raw_arguments, "store_root": temporary / "blocks"}
        with raw._fixed_repair_scope(**arguments) as (data, _):
            _require_data(data, expected_key)
            with _module_span("preparation_encode"):
                write_preparation_data(temporary, data)
            members = _member_inventory(temporary)
            collections = _collections(data)
        # Every original preparation/catalog/fresh exit has now passed. No
        # method authority survived the scope above, only the fixed data.
        manifest = RepairPreparationManifest(
            key=expected_key,
            members=members,
            collections=collections,
            tables=_tables(data),
            build_seconds=time.monotonic() - started,
        )
        with _guard_preparation_data(data):
            _require_key(expected_key, raw_arguments)
            _require_data(data, expected_key)
            _check_members(temporary, manifest)
            publish_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        for directory in (p for p in temporary.iterdir() if p.is_dir()):
            _fsync(directory)
        _fsync(temporary)
        if archive_root.exists() or archive_root.is_symlink():
            raise ValueError("preparation publication destination appeared")
        _publish_directory(temporary, archive_root)
        _fsync(parent)
        receipt = RepairPreparationReceipt(
            execution_sha256=execution_sha256,
            execution_code_sha256=execution_code_sha256,
            input_sha256=expected_key.input_sha256,
            preparation_key_sha256=expected_key.key_sha256,
            manifest_sha256=sha256_hex(canonical(manifest, _ARTIFACT_ROOTS)),
        )
        _check_members(archive_root, manifest)
        publish_json(receipt_path, receipt.model_dump(mode="json"))
        return data, manifest, receipt
    finally:
        os.close(lock)


@_preparation_call("raw_input_prepare")
@_module_call("preparation_load")
def _load_preparation_archive(
    *,
    archive_root: Path,
    expected_key: RepairPreparationKey,
    expected_manifest_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    raw_arguments: dict[str, Any],
) -> RepairPreparationData:
    require_pin(expected_manifest_sha256)
    require_pin(expected_receipt_sha256)
    _require_key(expected_key, raw_arguments)
    root = require_plain_path(archive_root)
    if root.name != expected_key.key_sha256:
        raise ValueError("preparation archive path/key mismatch")
    receipt_bytes = read_member(receipt_path, limit=16384)
    manifest_bytes = read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES)
    if sha256_hex(receipt_bytes) != expected_receipt_sha256:
        raise ValueError("preparation independent receipt pin differs")
    if sha256_hex(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("preparation independent manifest pin differs")
    receipt = RepairPreparationReceipt.model_validate_json(receipt_bytes)
    manifest = RepairPreparationManifest.model_validate_json(manifest_bytes)
    if (
        canonical(receipt, _ARTIFACT_ROOTS) != receipt_bytes
        or canonical(manifest, _ARTIFACT_ROOTS) != manifest_bytes
        or manifest.key != expected_key
        or receipt.preparation_key_sha256 != expected_key.key_sha256
        or receipt.manifest_sha256 != expected_manifest_sha256
        or receipt.input_sha256 != expected_key.input_sha256
    ):
        raise ValueError("preparation receipt/manifest provenance differs")

    def receipt_unchanged() -> None:
        if read_member(receipt_path, limit=16384) != receipt_bytes:
            raise ValueError("preparation receipt changed while loading")

    def manifest_unchanged() -> None:
        if read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES) != manifest_bytes:
            raise ValueError("preparation manifest changed while loading")

    _check_members(root, manifest)
    with ExitStack() as checks:
        checks.callback(_check_members, root, manifest)
        checks.callback(_require_key, expected_key, raw_arguments)
        checks.callback(manifest_unchanged)
        checks.callback(receipt_unchanged)
        data = read_preparation_data(root)
        checks.enter_context(_guard_preparation_data(data))
        _require_data(data, expected_key)
        if _collections(data) != manifest.collections or _tables(data) != manifest.tables:
            raise ValueError("preparation restored collection inventory differs")
        store = DirectorySourceBlockStore(root=str(root / "blocks"))
        for source in data.sources:
            for block in source.source_blocks:
                if len(store.read(block.sha256)) != block.size_bytes:
                    raise ValueError("preparation source block length differs")
    return data


def _require_preparation_files(
    run: RepairPreparationRun,
    binding: RepairPreparationBinding,
    raw_arguments: dict[str, Any],
) -> None:
    """Publication/disclosure barrier, not a reconstructed-source trust receipt."""
    if type(run) is not RepairPreparationRun or type(binding) is not RepairPreparationBinding:
        raise ValueError("preparation publication requires exact pinned bindings")
    before = snapshot((run, binding), (RepairPreparationRun, RepairPreparationBinding))
    try:
        _require_key(run.key, raw_arguments)
        root = require_plain_path(Path(run.archive_root))
        receipt_bytes = read_member(Path(run.receipt_path), limit=16384)
        manifest_bytes = read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES)
        if (
            root.name != run.key.key_sha256
            or binding.preparation_key_sha256 != run.key.key_sha256
            or sha256_hex(receipt_bytes) != binding.receipt_sha256
            or sha256_hex(manifest_bytes) != binding.manifest_sha256
        ):
            raise ValueError("preparation receipt or manifest changed after binding")
        manifest = RepairPreparationManifest.model_validate_json(manifest_bytes)
        if manifest.key != run.key:
            raise ValueError("preparation manifest key changed")
        _check_members(root, manifest)
    finally:
        require_unchanged((run, binding), (RepairPreparationRun, RepairPreparationBinding), before)
