"""Opaque inference identities with evaluator-only reversible mappings."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

EVALUATOR_IDENTITY_SCHEMA_VERSION: Literal[
    "nurse.v2.evaluator-identity-map/1"
] = "nurse.v2.evaluator-identity-map/1"
INFERENCE_IDENTITY_SCHEMA_VERSION: Literal[
    "nurse.v2.inference-identity-manifest/1"
] = "nurse.v2.inference-identity-manifest/1"
IDENTITY_AUDIT_SCHEMA_VERSION: Literal[
    "nurse.v2.identity-import-audit/1"
] = "nurse.v2.identity-import-audit/1"

OpaqueIdMethod = Literal["random_uuid_v4", "hmac_sha256_128"]
UuidFactory = Callable[[], uuid.UUID]


class EvaluatorIdentitySource(StrictFrozenModel):
    """Private source fields accepted only inside the isolated evaluator."""

    dataset: DatasetName
    original_case_id: str = Field(min_length=1)
    incident_group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    answer_services: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_answers(self) -> Self:
        if self.answer_services != tuple(sorted(set(self.answer_services))):
            raise ValueError("answer services must be sorted and unique")
        if any(not value.strip() for value in self.answer_services):
            raise ValueError("answer services cannot be blank")
        return self


class EvaluatorIdentityRecord(EvaluatorIdentitySource):
    """One reversible evaluator-side binding."""

    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    method: OpaqueIdMethod


class EvaluatorIdentityMap(StrictFrozenModel):
    """Private immutable identity and answer map."""

    schema_version: Literal["nurse.v2.evaluator-identity-map/1"] = (
        EVALUATOR_IDENTITY_SCHEMA_VERSION
    )
    method: OpaqueIdMethod
    source_count: int = Field(ge=0)
    records: tuple[EvaluatorIdentityRecord, ...]
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        expected_order = tuple(
            sorted(self.records, key=lambda item: (item.dataset, item.original_case_id))
        )
        if self.records != expected_order or self.source_count != len(self.records):
            return False
        originals = {(record.dataset, record.original_case_id) for record in self.records}
        opaque_ids = {record.opaque_case_id for record in self.records}
        if len(originals) != len(self.records) or len(opaque_ids) != len(self.records):
            return False
        if any(record.method != self.method for record in self.records):
            return False
        unsigned = self.model_dump(mode="json", exclude={"mapping_sha256"})
        return self.mapping_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class InferenceIdentityRecord(StrictFrozenModel):
    """Public-to-inference identity with no reversible or answer fields."""

    dataset: DatasetName
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    method: OpaqueIdMethod


class InferenceIdentityManifest(StrictFrozenModel):
    """Only identity information permitted to cross into inference."""

    schema_version: Literal["nurse.v2.inference-identity-manifest/1"] = (
        INFERENCE_IDENTITY_SCHEMA_VERSION
    )
    method: OpaqueIdMethod
    case_count: int = Field(ge=0)
    rcabench_case_count: int = Field(ge=0)
    aiops25_case_count: int = Field(ge=0)
    records: tuple[InferenceIdentityRecord, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        expected_order = tuple(
            sorted(self.records, key=lambda item: (item.dataset, item.opaque_case_id))
        )
        if self.records != expected_order or self.case_count != len(self.records):
            return False
        if len({record.opaque_case_id for record in self.records}) != len(self.records):
            return False
        if any(record.method != self.method for record in self.records):
            return False
        if self.rcabench_case_count != sum(
            record.dataset == "rcabench" for record in self.records
        ):
            return False
        if self.aiops25_case_count != sum(
            record.dataset == "aiops25" for record in self.records
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"manifest_sha256"})
        return self.manifest_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def generate_opaque_case_id(
    *,
    dataset: DatasetName,
    original_case_id: str,
    method: OpaqueIdMethod,
    hmac_secret: bytes | None = None,
    uuid_factory: UuidFactory = uuid.uuid4,
) -> str:
    """Generate a random UUIDv4 or secret-keyed HMAC identity."""

    if not original_case_id:
        raise ValueError("original case ID cannot be empty")
    if method == "random_uuid_v4":
        generated = uuid_factory()
        if generated.version != 4 or generated.variant != uuid.RFC_4122:
            raise ValueError("UUID factory must return an RFC 4122 UUIDv4")
        return f"opq_{generated.hex}"
    if method == "hmac_sha256_128":
        if hmac_secret is None or len(hmac_secret) < 32:
            raise ValueError("opaque HMAC secret must be at least 32 bytes")
        message = canonical_json_bytes(
            {
                "domain": "nurse.v2.opaque-case-id/1",
                "dataset": dataset,
                "original_case_id": original_case_id,
            }
        )
        digest = hmac.new(hmac_secret, message, hashlib.sha256).hexdigest()
        return f"opq_{digest[:32]}"
    raise ValueError(f"unsupported opaque ID method: {method}")


def build_identity_artifacts(
    sources: Sequence[EvaluatorIdentitySource],
    *,
    method: OpaqueIdMethod,
    hmac_secret: bytes | None = None,
    uuid_factory: UuidFactory = uuid.uuid4,
) -> tuple[EvaluatorIdentityMap, InferenceIdentityManifest]:
    """Build private reversible and nonreversible inference artifacts together."""

    ordered_sources = tuple(
        sorted(sources, key=lambda item: (item.dataset, item.original_case_id))
    )
    source_keys = tuple(
        (item.dataset, item.original_case_id) for item in ordered_sources
    )
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source case identities must be unique")
    records = tuple(
        EvaluatorIdentityRecord(
            **item.model_dump(mode="python"),
            opaque_case_id=generate_opaque_case_id(
                dataset=item.dataset,
                original_case_id=item.original_case_id,
                method=method,
                hmac_secret=hmac_secret,
                uuid_factory=uuid_factory,
            ),
            method=method,
        )
        for item in ordered_sources
    )
    if len({record.opaque_case_id for record in records}) != len(records):
        raise ValueError("opaque case identities must be unique")
    evaluator_unsigned: dict[str, object] = {
        "schema_version": EVALUATOR_IDENTITY_SCHEMA_VERSION,
        "method": method,
        "source_count": len(records),
        "records": [record.model_dump(mode="json") for record in records],
    }
    evaluator_map = EvaluatorIdentityMap.model_validate(
        {
            **evaluator_unsigned,
            "mapping_sha256": sha256_hex(canonical_json_bytes(evaluator_unsigned)),
        }
    )
    inference_manifest = inference_manifest_from_evaluator(evaluator_map)
    if not evaluator_map.verify() or not inference_manifest.verify():
        raise AssertionError("identity artifact verification failed")
    return evaluator_map, inference_manifest


def inference_manifest_from_evaluator(
    evaluator_map: EvaluatorIdentityMap,
) -> InferenceIdentityManifest:
    """Project the private map into the only fields available to inference."""

    if not evaluator_map.verify():
        raise ValueError("evaluator identity map verification failed")
    records = tuple(
        sorted(
            (
                InferenceIdentityRecord(
                    dataset=record.dataset,
                    opaque_case_id=record.opaque_case_id,
                    method=record.method,
                )
                for record in evaluator_map.records
            ),
            key=lambda item: (item.dataset, item.opaque_case_id),
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": INFERENCE_IDENTITY_SCHEMA_VERSION,
        "method": evaluator_map.method,
        "case_count": len(records),
        "rcabench_case_count": sum(
            record.dataset == "rcabench" for record in records
        ),
        "aiops25_case_count": sum(record.dataset == "aiops25" for record in records),
        "records": [record.model_dump(mode="json") for record in records],
    }
    manifest = InferenceIdentityManifest.model_validate(
        {
            **unsigned,
            "manifest_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not manifest.verify():  # pragma: no cover - construction invariant
        raise AssertionError("inference identity manifest verification failed")
    return manifest


def reverse_identity(
    evaluator_map: EvaluatorIdentityMap,
    opaque_case_id: str,
) -> EvaluatorIdentityRecord:
    """Resolve an opaque identity only from a verified evaluator-side map."""

    if not evaluator_map.verify():
        raise ValueError("evaluator identity map verification failed")
    for record in evaluator_map.records:
        if record.opaque_case_id == opaque_case_id:
            return record
    raise KeyError(opaque_case_id)


def write_evaluator_identity_map(
    path: Path,
    evaluator_map: EvaluatorIdentityMap,
) -> None:
    """Write a private 0600 immutable evaluator artifact."""

    output = Path(path)
    data = canonical_json_bytes(evaluator_map.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable evaluator identity map already exists")
        if not evaluator_map.verify():
            raise ValueError("evaluator identity map verification failed")
        _require_private_mode(output)
        return
    if not evaluator_map.verify():
        raise ValueError("evaluator identity map verification failed")
    _atomic_write_private(output, data)
    _require_private_mode(output)


def load_evaluator_identity_map(path: Path) -> EvaluatorIdentityMap:
    """Load a verified private map, rejecting permissive POSIX permissions."""

    source = Path(path)
    _require_private_mode(source)
    try:
        identity_map = EvaluatorIdentityMap.model_validate_json(source.read_bytes())
    except Exception as error:
        raise ValueError("invalid evaluator identity map") from error
    if not identity_map.verify():
        raise ValueError("evaluator identity map verification failed")
    return identity_map


def write_inference_identity_manifest(
    path: Path,
    manifest: InferenceIdentityManifest,
) -> None:
    """Write an immutable inference-safe manifest."""

    if not manifest.verify():
        raise ValueError("inference identity manifest verification failed")
    output = Path(path)
    data = canonical_json_bytes(manifest.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable inference identity manifest already exists")
        return
    atomic_write_bytes(output, data)


def load_incident_group_index(path: Path, *, dataset: DatasetName) -> dict[str, str]:
    """Load and verify case/group membership without exposing it to inference."""

    payload = _load_hashed_json(path, hash_field="manifest_sha256")
    if payload.get("dataset") != dataset:
        raise ValueError("incident-group dataset mismatch")
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("incident groups must be an array")
    index: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("incident group must be an object")
        group_id = group.get("group_id")
        case_ids = group.get("case_ids")
        if not isinstance(group_id, str) or not isinstance(case_ids, list):
            raise ValueError("incident group has invalid identity fields")
        for case_id in case_ids:
            if not isinstance(case_id, str) or not case_id or case_id in index:
                raise ValueError("incident-group case IDs must be unique")
            index[case_id] = group_id
    if payload.get("case_count") != len(index):
        raise ValueError("incident-group case count mismatch")
    return index


def build_server_identity_sources(
    *,
    rcabench_inventory: Path,
    aiops25_inventory: Path,
    rcabench_groups: Mapping[str, str],
    aiops25_groups: Mapping[str, str],
    rcabench_data_root: Path,
    aiops25_work_root: Path,
) -> tuple[EvaluatorIdentitySource, ...]:
    """Read answers only inside the evaluator process and form private sources."""

    rca_inventory = _load_hashed_json(
        rcabench_inventory,
        hash_field="inventory_sha256",
    )
    aiops_inventory = _load_hashed_json(
        aiops25_inventory,
        hash_field="inventory_sha256",
    )
    sources: list[EvaluatorIdentitySource] = []
    for case_id in _inventory_case_ids(
        rca_inventory,
        dataset="rcabench",
        expected_status="valid",
        expected_count=1422,
    ):
        group_id = rcabench_groups.get(case_id)
        if group_id is None:
            raise ValueError(f"rcabench/{case_id}: no incident group")
        injection = _load_json_object(Path(rcabench_data_root) / case_id / "injection.json")
        ground_truth = injection.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"rcabench/{case_id}: invalid evaluator answer mapping")
        answers = _answer_services(ground_truth.get("service"), context=case_id)
        sources.append(
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id=case_id,
                incident_group_id=group_id,
                answer_services=answers,
            )
        )
    for case_id in _inventory_case_ids(
        aiops_inventory,
        dataset="aiops25",
        expected_status="included",
        expected_count=318,
    ):
        group_id = aiops25_groups.get(case_id)
        if group_id is None:
            raise ValueError(f"aiops25/{case_id}: no incident group")
        injection = _load_json_object(
            Path(aiops25_work_root) / "canonical" / case_id / "injection.json"
        )
        answers = _answer_services(injection.get("target_services"), context=case_id)
        sources.append(
            EvaluatorIdentitySource(
                dataset="aiops25",
                original_case_id=case_id,
                incident_group_id=group_id,
                answer_services=answers,
            )
        )
    return tuple(sorted(sources, key=lambda item: (item.dataset, item.original_case_id)))


def build_identity_import_audit(
    *,
    evaluator_map: EvaluatorIdentityMap,
    inference_manifest: InferenceIdentityManifest,
    evaluator_path: Path,
    inference_path: Path,
    excluded_aiops25_node_case_ids: Sequence[str],
) -> dict[str, object]:
    """Prove isolation and population counts without emitting mapping values."""

    if not evaluator_map.verify() or not inference_manifest.verify():
        raise ValueError("identity artifacts must verify before audit")
    inference_bytes = Path(inference_path).read_bytes()
    original_ids = tuple(record.original_case_id for record in evaluator_map.records)
    answers = tuple(
        answer for record in evaluator_map.records for answer in record.answer_services
    )
    mapped_originals = set(original_ids)
    excluded_node_set = set(excluded_aiops25_node_case_ids)
    return {
        "schema_version": IDENTITY_AUDIT_SCHEMA_VERSION,
        "method": evaluator_map.method,
        "mapping_exclusively_evaluator": True,
        "opaque_id_count": len(evaluator_map.records),
        "unique_opaque_id_count": len(
            {record.opaque_case_id for record in evaluator_map.records}
        ),
        "inference_contains_original_ids": any(
            value.encode("utf-8") in inference_bytes for value in original_ids
        ),
        "inference_contains_answers": any(
            value.encode("utf-8") in inference_bytes for value in set(answers)
        ),
        "inference_contains_original_paths": b"original_path" in inference_bytes
        or b"source_path" in inference_bytes,
        "evaluator_mapping": {
            "path": Path(evaluator_path).as_posix(),
            "mode": f"{stat.S_IMODE(Path(evaluator_path).stat().st_mode):04o}",
            "artifact_sha256": _sha256_file(Path(evaluator_path)),
            "mapping_sha256": evaluator_map.mapping_sha256,
        },
        "inference_manifest": {
            "path": Path(inference_path).as_posix(),
            "artifact_sha256": _sha256_file(Path(inference_path)),
            "manifest_sha256": inference_manifest.manifest_sha256,
        },
        "datasets": {
            "rcabench": {
                "case_count": inference_manifest.rcabench_case_count,
            },
            "aiops25": {
                "case_count": inference_manifest.aiops25_case_count,
                "excluded_node_cases": len(excluded_node_set),
                "excluded_node_mappings": len(excluded_node_set.intersection(mapped_originals)),
            },
        },
    }


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    data = canonical_json_bytes(dict(payload))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable identity audit already exists")
        return
    atomic_write_bytes(output, data)


def _atomic_write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _require_private_mode(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("evaluator identity map must be a regular file")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("evaluator identity map must have mode 0600")


def _load_hashed_json(path: Path, *, hash_field: str) -> dict[str, object]:
    payload = _load_json_object(path)
    value = payload.get(hash_field)
    if not isinstance(value, str) or not _is_sha256(value):
        raise ValueError(f"{path}: invalid {hash_field}")
    unsigned = dict(payload)
    del unsigned[hash_field]
    if value != sha256_hex(canonical_json_bytes(unsigned)):
        raise ValueError(f"{path}: {hash_field} mismatch")
    return payload


def _inventory_case_ids(
    inventory: Mapping[str, object],
    *,
    dataset: DatasetName,
    expected_status: str,
    expected_count: int,
) -> tuple[str, ...]:
    if inventory.get("dataset") != dataset:
        raise ValueError("inventory dataset mismatch")
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("inventory entries must be an array")
    case_ids = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != expected_status:
            continue
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("inventory entry has invalid case ID")
        case_ids.append(case_id)
    if len(case_ids) != expected_count or len(case_ids) != len(set(case_ids)):
        raise ValueError("inventory identity population mismatch")
    return tuple(sorted(case_ids))


def _excluded_aiops25_node_ids(inventory_path: Path) -> tuple[str, ...]:
    inventory = _load_hashed_json(inventory_path, hash_field="inventory_sha256")
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("inventory entries must be an array")
    case_ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("inventory entry must be an object")
        if entry.get("status") == "excluded" and entry.get("instance_type") == "node":
            case_id = entry.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("excluded node entry has invalid case ID")
            case_ids.append(case_id)
    return tuple(sorted(case_ids))


def _answer_services(value: object, *, context: str) -> tuple[str, ...]:
    answers: tuple[str, ...]
    if isinstance(value, str):
        answers = (value,)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        answers = tuple(cast(list[str], value))
    else:
        raise ValueError(f"{context}: invalid evaluator answer services")
    normalized = tuple(sorted({answer.strip() for answer in answers if answer.strip()}))
    if not normalized:
        raise ValueError(f"{context}: evaluator answer services cannot be empty")
    return normalized


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create isolated opaque identities")
    parser.add_argument("--rcabench-inventory", type=Path, required=True)
    parser.add_argument("--aiops25-inventory", type=Path, required=True)
    parser.add_argument("--rcabench-groups", type=Path, required=True)
    parser.add_argument("--aiops25-groups", type=Path, required=True)
    parser.add_argument("--rcabench-data-root", type=Path, required=True)
    parser.add_argument("--aiops25-work-root", type=Path, required=True)
    parser.add_argument("--evaluator-output", type=Path, required=True)
    parser.add_argument("--inference-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rcabench_groups = load_incident_group_index(
        args.rcabench_groups,
        dataset="rcabench",
    )
    aiops25_groups = load_incident_group_index(
        args.aiops25_groups,
        dataset="aiops25",
    )
    if args.evaluator_output.exists():
        evaluator_map = load_evaluator_identity_map(args.evaluator_output)
        inference_manifest = inference_manifest_from_evaluator(evaluator_map)
    else:
        sources = build_server_identity_sources(
            rcabench_inventory=args.rcabench_inventory,
            aiops25_inventory=args.aiops25_inventory,
            rcabench_groups=rcabench_groups,
            aiops25_groups=aiops25_groups,
            rcabench_data_root=args.rcabench_data_root,
            aiops25_work_root=args.aiops25_work_root,
        )
        evaluator_map, inference_manifest = build_identity_artifacts(
            sources,
            method="random_uuid_v4",
        )
        write_evaluator_identity_map(args.evaluator_output, evaluator_map)
    write_inference_identity_manifest(args.inference_output, inference_manifest)
    audit = build_identity_import_audit(
        evaluator_map=evaluator_map,
        inference_manifest=inference_manifest,
        evaluator_path=args.evaluator_output,
        inference_path=args.inference_output,
        excluded_aiops25_node_case_ids=_excluded_aiops25_node_ids(
            args.aiops25_inventory
        ),
    )
    write_immutable_json(args.audit_output, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
