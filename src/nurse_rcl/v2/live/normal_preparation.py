"""Once-per-ingress admission of complete fixed preparation, not a replay lease.

The pinned build is trusted as a project-produced data snapshot. Runtime-only
code changes do not re-sign its build identity or demand raw reassembly.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa  # type: ignore[import-untyped]

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import AIOps25AdapterConfiguration
from nurse_rcl.v2.aiops25_inventory import AuthorizedAIOps25Queue
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live import repair_preparation_codec as codec
from nurse_rcl.v2.live.normal_artifacts import NormalPreparationReceipt
from nurse_rcl.v2.live.normal_fixed_inputs import construct_normal_fixed_inputs
from nurse_rcl.v2.live.repair_driver import _fsync, publish_json
from nurse_rcl.v2.live.repair_module_timing import _module_call, _module_span
from nurse_rcl.v2.live.repair_preparation_archive import (
    _SEMANTICS,
    _collections,
    _member_inventory,
    _publish_directory,
    _require_data,
    _tables,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import (
    RepairPreparationData,
    RepairPreparationKey,
    RepairPreparationManifest,
    RepairPreparationMember,
    RepairPreparationReceipt,
)
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterConfiguration
from nurse_rcl.v2.report_audit_json import MAX_REPORT_AUDIT_BYTES


class _DigestStream(io.RawIOBase):
    """Hash the exact forward-only bytes handed to the Arrow stream decoder."""

    def __init__(self, handle: BinaryIO, member: RepairPreparationMember) -> None:
        self.handle, self.member = handle, member
        self.size = 0
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        remaining = self.member.size_bytes - self.size
        raw = self.handle.read(min(size, remaining + 1) if size >= 0 else remaining + 1)
        self.size += len(raw)
        self.digest.update(raw)
        if self.size > self.member.size_bytes:
            raise ValueError("preparation member exceeds pinned size")
        return raw

    def finish(self) -> None:
        if self.read(1):
            raise ValueError("preparation Arrow has unconsumed trailing bytes")
        if self.size != self.member.size_bytes or self.digest.hexdigest() != self.member.sha256:
            raise ValueError("preparation member digest differs from independent pin")


class _PinnedMembers:
    def __init__(self, root: Path, manifest: RepairPreparationManifest) -> None:
        self.root = require_plain_path(root)
        self.members = {m.name: m for m in manifest.members}
        self.read_names: set[str] = set()
        actual, directories = set(), set()
        for path in root.rglob("*"):
            require_plain_path(path)
            mode = path.stat().st_mode
            name = path.relative_to(root).as_posix()
            if stat.S_ISDIR(mode):
                directories.add(name)
            elif stat.S_ISREG(mode):
                actual.add(name)
                if name != "manifest.json":
                    member = self.members.get(name)
                    if member is None or path.stat().st_size != member.size_bytes:
                        raise ValueError("preparation member inventory/size differs")
            else:
                raise ValueError("preparation member is not a regular file")
        expected_dirs = {
            str(p)
            for name in self.members
            for p in Path(name).parents
            if str(p) != "."
        }
        if actual != {*self.members, "manifest.json"} or directories != expected_dirs:
            raise ValueError("preparation member/directory inventory differs")

    def _member(self, name: str) -> RepairPreparationMember:
        if name not in self.members or name in self.read_names:
            raise ValueError("preparation member absent or already consumed")
        self.read_names.add(name)
        return self.members[name]

    def json(self, name: str) -> dict[str, Any]:
        member = self._member(name)
        raw = codec.read_member(self.root / name, limit=REPAIR_MAX_JSON_BYTES)
        if len(raw) != member.size_bytes or sha256_hex(raw) != member.sha256:
            raise ValueError("preparation JSON differs from independent pin")
        return codec._decode_json_member(raw)

    def arrow(self, name: str) -> Any:
        member = self._member(name)
        with codec._member_stream(self.root / name, limit=member.size_bytes) as handle:
            stream = _DigestStream(handle, member)
            try:
                with pa.ipc.open_stream(stream) as reader:
                    table = reader.read_all()
                stream.finish()
            except pa.ArrowException as error:
                raise ValueError("preparation Arrow decoding failed") from error
        return table


def _require_compatible_key(key: RepairPreparationKey, arguments: dict[str, Any]) -> None:
    """Match the frozen input contract, not the current runner's source-code hash."""
    if type(key) is not RepairPreparationKey or key.semantics_sha256 != _SEMANTICS:
        raise ValueError("normal preparation has an incompatible data contract")
    dataset = arguments["dataset"]
    configuration = arguments["configuration"]
    queue, source_id = arguments.get("authorized_queue"), arguments.get("source_case_id")
    if dataset == "rcabench":
        if type(configuration) is not RCABenchAdapterConfiguration or queue is not None or (
            source_id is not None
        ):
            raise ValueError("normal preparation RCABench configuration/authorization differs")
    elif dataset == "aiops25":
        if type(configuration) is not AIOps25AdapterConfiguration or (
            type(queue) is not AuthorizedAIOps25Queue or type(source_id) is not str
        ):
            raise ValueError("normal preparation AIOps25 configuration/authorization differs")
    else:
        raise ValueError("normal preparation dataset is unsupported")
    if (
        key.dataset != dataset
        or key.opaque_case_id != arguments["opaque_case_id"]
        or key.source_descriptor_sha256 != arguments["expected_source_descriptor_sha256"]
        or key.configuration_sha256
        != sha256_hex(canonical_json_bytes(configuration.model_dump(mode="json")))
        or key.authorized_queue_sha256
        != (
            None if queue is None
            else sha256_hex(canonical_json_bytes(queue.model_dump(mode="json")))
        )
        or key.source_case_id_sha256
        != (None if source_id is None else sha256_hex(canonical_json_bytes(source_id)))
    ):
        raise ValueError("normal preparation input identity differs from pinned build")


@_module_call("preparation_read")
def load_normal_preparation(
    *,
    archive_root: Path,
    expected_key: RepairPreparationKey,
    raw_arguments: dict[str, Any],
    receipt_path: Path,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> RepairPreparationData:
    """Admit independently pinned fixed data once, preserving its old build identity."""
    _require_compatible_key(expected_key, raw_arguments)
    require_pin(expected_manifest_sha256)
    require_pin(expected_receipt_sha256)
    root = require_plain_path(archive_root)
    if root.name != expected_key.key_sha256:
        raise ValueError("normal preparation archive path/key mismatch")
    receipt_bytes = codec.read_member(receipt_path, limit=16384)
    manifest_bytes = codec.read_member(root / "manifest.json", limit=REPAIR_MAX_JSON_BYTES)
    if sha256_hex(receipt_bytes) != expected_receipt_sha256 or (
        sha256_hex(manifest_bytes) != expected_manifest_sha256
    ):
        raise ValueError("normal preparation receipt/manifest pin mismatch")
    receipt: RepairPreparationReceipt | NormalPreparationReceipt
    receipt_type = json.loads(receipt_bytes).get("schema_version")
    if receipt_type == "nurse.v2.normal-rcl-preparation-receipt/1":
        receipt = NormalPreparationReceipt.model_validate_json(receipt_bytes)
    else:
        receipt = RepairPreparationReceipt.model_validate_json(receipt_bytes)
    manifest = RepairPreparationManifest.model_validate_json(manifest_bytes)
    if (
        manifest.key != expected_key
        or receipt.preparation_key_sha256 != expected_key.key_sha256
        or receipt.manifest_sha256 != expected_manifest_sha256
        or receipt.input_sha256 != expected_key.input_sha256
    ):
        raise ValueError("normal preparation build provenance mismatch")
    members = _PinnedMembers(root, manifest)
    data = codec._decode_preparation_data(members.json, members.arrow)
    if members.read_names != {n for n in members.members if not n.startswith("blocks/")}:
        raise ValueError("normal preparation decoded member domain differs")
    _require_data(data, expected_key)
    if _collections(data) != manifest.collections or _tables(data) != manifest.tables:
        raise ValueError("normal preparation restored inventory differs")
    expected_blocks = set()
    for source in data.sources:
        for block in source.source_blocks:
            name = "blocks/" + block.sha256 + ".json"
            member = members.members.get(name)
            if member is None or member.sha256 != block.sha256 or (
                member.size_bytes != block.size_bytes or member.size_bytes > MAX_REPORT_AUDIT_BYTES
            ):
                raise ValueError("normal preparation block reference differs from pinned member")
            expected_blocks.add(name)
    if expected_blocks != {n for n in members.members if n.startswith("blocks/")}:
        raise ValueError("normal preparation complete block domain differs")
    return data


def build_normal_preparation(
    *, archive_root: Path, expected_key: RepairPreparationKey,
    raw_arguments: dict[str, Any], receipt_path: Path,
    execution_sha256: str, execution_code_sha256: str,
) -> tuple[RepairPreparationData, RepairPreparationManifest, NormalPreparationReceipt]:
    """Construct fixed data once and publish without claiming legacy guarded exits."""
    _require_compatible_key(expected_key, raw_arguments)
    require_pin(execution_sha256)
    require_pin(execution_code_sha256)
    parent = require_plain_path(archive_root.parent)
    require_plain_path(receipt_path.parent)
    if archive_root.name != expected_key.key_sha256:
        raise ValueError("normal preparation archive key differs")
    if archive_root.exists() or archive_root.is_symlink() or (
        receipt_path.exists() or receipt_path.is_symlink()
    ):
        raise ValueError("normal preparation archive/receipt already exists")
    descriptor = os.open(parent / (expected_key.key_sha256 + ".lock"),
                         os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if archive_root.exists() or receipt_path.exists():
            raise ValueError("normal preparation publication already exists")
        temporary = Path(tempfile.mkdtemp(prefix=".normal-partial-", dir=parent))
        started = time.monotonic()
        data = construct_normal_fixed_inputs(
            key=expected_key, raw_arguments=raw_arguments, block_root=temporary / "blocks",
        )
        _require_data(data, expected_key)
        with _module_span("preparation_encode"):
            codec.write_preparation_data(temporary, data)
        manifest = RepairPreparationManifest(
            key=expected_key, members=_member_inventory(temporary),
            collections=_collections(data), tables=_tables(data),
            build_seconds=time.monotonic() - started,
        )
        with _module_span("preparation_publish"):
            publish_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
            for directory in (p for p in temporary.iterdir() if p.is_dir()):
                _fsync(directory)
            _fsync(temporary)
            _publish_directory(temporary, archive_root)
            _fsync(parent)
        receipt = NormalPreparationReceipt(
            execution_sha256=execution_sha256, execution_code_sha256=execution_code_sha256,
            input_sha256=expected_key.input_sha256, preparation_key_sha256=expected_key.key_sha256,
            manifest_sha256=sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json"))),
        )
        publish_json(receipt_path, receipt.model_dump(mode="json"))
        return data, manifest, receipt
    finally:
        os.close(descriptor)
