"""Private complete-case lifetime and explicit decision/full-method barriers.

Only the raw owning call constructs these values. They cannot authorize another
invocation. Unconsumed source payload mutation may wait until a full barrier;
decision inputs and source/store bindings are actually rechecked each time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from nurse_rcl.v2.dynamic_verification.source_store import (
    require_store_unchanged,
    store_snapshot,
)
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationBinding,
    RepairPreparationManifest,
    RepairPreparationRun,
)
from nurse_rcl.v2.live.repair_query_session import _ROOTS as SESSION_ROOTS
from nurse_rcl.v2.live.repair_query_session import RepairQueryRuntime
from nurse_rcl.v2.live.repair_raw_audit import (
    PreparedRepairCase,
    _PreparationChecks,
    _prepared_repair_scope,
)
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_MAX_JSON_BYTES,
    canonical,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource


class _CaseInputs:
    """Non-serializable lexical implementation detail, not a public capability."""

    __slots__ = ("_active", "_decision", "_full", "_raw", "_source_root", "preparation", "prepared")

    def __init__(
        self,
        prepared: PreparedRepairCase,
        checks: _PreparationChecks,
        source_root: Path,
        *,
        preparation: RepairPreparationBinding | None = None,
    ) -> None:
        self.prepared = prepared
        self._raw = checks.data
        self._source_root = source_root
        self._active = True
        if preparation is not None and type(preparation) is not RepairPreparationBinding:
            raise ValueError("case preparation binding requires exact current type")
        self.preparation = preparation
        roots = (*SESSION_ROOTS, RepairPreparationBinding)
        runtime = prepared.runtime
        sources = runtime.arguments["sources"]
        store = runtime.arguments["store"]
        saved_store = store_snapshot(store)

        def context() -> object:
            if (
                self.prepared is not prepared
                or type(prepared) is not PreparedRepairCase
                or prepared.runtime is not runtime
                or type(runtime) is not RepairQueryRuntime
                or type(runtime.arguments) is not dict
                or runtime.arguments.get("sources") is not sources
                or type(sources) is not tuple
                or runtime.arguments.get("store") is not store
                or self._raw is not checks.data
                or self._source_root is not source_root
                or self.preparation is not preparation
            ):
                raise ValueError("case-owned input binding changed")
            bindings = []
            for source in sources:
                if type(source) is not RepairMeasurementSource:
                    raise ValueError("case-owned source representation changed")
                # Headers/handles/pins only: never traverse unused sample payloads
                # at a decision boundary. Full checks still cover every field.
                bindings.append(
                    (
                        id(source),
                        source.signal_id,
                        source.source_sha256,
                        source.dataset,
                        source.observed_entity_key,
                        source.candidate_service_key,
                        source.source_file_sha256s,
                        source.source_blocks,
                    )
                )
            return (
                runtime.catalog,
                runtime.pin,
                str(runtime.root),
                tuple(bindings),
                tuple(
                    (key, value)
                    for key, value in sorted(runtime.arguments.items())
                    if key not in {"sources", "store"}
                ),
                id(prepared.graph),
                id(prepared.error_sources),
                prepared.error_observation_sha256s,
                prepared.source_descriptor_sha256,
                prepared.source_files_sha256,
                prepared.candidate_freeze_sha256,
                prepared.raw_file_authority,
                preparation,
            )

        before = snapshot(context(), roots)

        def decision() -> None:
            # All established guards run even when the first one fails.
            with ExitStack() as exit_checks:
                exit_checks.callback(require_store_unchanged, store, saved_store)
                exit_checks.callback(checks.catalog.check_decision)
                exit_checks.callback(lambda: require_unchanged(context(), roots, before))

        def full() -> None:
            with ExitStack() as exit_checks:
                exit_checks.callback(checks.raw)
                exit_checks.callback(checks.catalog)
                exit_checks.callback(decision)

        self._decision: Callable[[], None] = decision
        self._full: Callable[[], None] = full

    @_module_call("input_guard")
    def check_decision(self) -> None:
        if not self._active:
            raise ValueError("case-owned input scope is closed")
        self._decision()

    @_module_call("input_guard")
    def check_full_method(self) -> None:
        if not self._active:
            raise ValueError("case-owned input scope is closed")
        self._full()


@contextmanager
def _case_inputs_scope(**raw_arguments: Any) -> Iterator[_CaseInputs]:
    # The existing preparation decorator times only actual enter/raw exits.
    # Method barriers below run in the caller's method account, not preparation.
    with _prepared_repair_scope(**raw_arguments) as (prepared, checks):
        owned = _CaseInputs(prepared, checks, Path(raw_arguments["source_root"]))
        try:
            with ExitStack() as method_checks:
                method_checks.callback(owned.check_full_method)
                yield owned
        finally:
            owned._active = False


@contextmanager
def _prepared_case_inputs_scope(
    *,
    preparation: RepairPreparationRun,
    raw_arguments: dict[str, Any],
    execution_sha256: str,
    execution_code_sha256: str,
) -> Iterator[_CaseInputs]:
    from nurse_rcl.artifacts import sha256_hex
    from nurse_rcl.v2.live import repair_preparation_archive as archive
    from nurse_rcl.v2.live.repair_preparation_codec import (
        read_member,
        require_serialized_preparation,
    )
    from nurse_rcl.v2.live.repair_queries import _CatalogChecks
    from nurse_rcl.v2.live.repair_raw_audit import _bind_preparation_runtime

    if type(preparation) is not RepairPreparationRun:
        raise ValueError("prepared case requires an exact pinned preparation plan")
    root, receipt_path = Path(preparation.archive_root), Path(preparation.receipt_path)
    before = snapshot(preparation, (RepairPreparationRun,))
    with ExitStack() as guards:
        guards.callback(require_unchanged, preparation, (RepairPreparationRun,), before)
        common: dict[str, Any] = dict(
            archive_root=root,
            expected_key=preparation.key,
            raw_arguments=raw_arguments,
            receipt_path=receipt_path,
        )
        if preparation.mode == "build":
            data, manifest, receipt = archive._build_preparation_archive(
                **common,
                execution_sha256=execution_sha256,
                execution_code_sha256=execution_code_sha256,
            )
            manifest_pin = sha256_hex(canonical(manifest, archive._ARTIFACT_ROOTS))
            receipt_pin = sha256_hex(canonical(receipt, archive._ARTIFACT_ROOTS))
        else:
            if preparation.manifest_sha256 is None or preparation.receipt_sha256 is None:
                raise ValueError("prepared load lacks independent pins")
            manifest_pin, receipt_pin = preparation.manifest_sha256, preparation.receipt_sha256
            data = archive._load_preparation_archive(
                **common,
                expected_manifest_sha256=manifest_pin,
                expected_receipt_sha256=receipt_pin,
            )
            manifest = RepairPreparationManifest.model_validate_json(
                read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES)
            )
        binding = RepairPreparationBinding(
            preparation_key_sha256=preparation.key.key_sha256,
            manifest_sha256=manifest_pin,
            receipt_sha256=receipt_pin,
        )

        def external() -> None:
            with ExitStack() as checks:
                checks.callback(archive._require_key, preparation.key, raw_arguments)
                checks.callback(archive._check_members, root, manifest)

                def pins() -> None:
                    if (
                        sha256_hex(canonical(manifest, archive._ARTIFACT_ROOTS)) != manifest_pin
                        or sha256_hex(read_member(receipt_path, limit=16384)) != receipt_pin
                        or sha256_hex(
                            read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES)
                        )
                        != manifest_pin
                    ):
                        raise ValueError("preparation receipt or manifest changed during method")

                checks.callback(pins)
                checks.callback(require_unchanged, preparation, (RepairPreparationRun,), before)

        # Establish a new method-owned baseline only after pinned preparation
        # exits. No prior callable/check survives serialization or cold scope.
        external()
        guards.callback(external)
        check_data = guards.enter_context(archive._guard_preparation_data(data))
        require_serialized_preparation(data, manifest)
        archive._require_data(data, preparation.key)
        catalog_before = snapshot(data.catalog, SESSION_ROOTS)
        catalog = _CatalogChecks(
            check_data, lambda: require_unchanged(data.catalog, SESSION_ROOTS, catalog_before)
        )
        prepared = _bind_preparation_runtime(
            data,
            store_root=root / "blocks",
            ledger_root=raw_arguments["ledger_root"],
        )
        owned = _CaseInputs(
            prepared,
            _PreparationChecks(catalog=catalog, raw=external, data=data.diagnostics),
            Path(raw_arguments["source_root"]),
            preparation=binding,
        )
        try:
            with ExitStack() as checks:
                checks.callback(owned.check_full_method)
                yield owned
        finally:
            owned._active = False
            catalog._active = False
