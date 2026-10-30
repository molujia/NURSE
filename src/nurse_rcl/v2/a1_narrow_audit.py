"""Rebuildable changed-file and configuration audit for A1/V1-repaired."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.experiment_registry import load_experiment_registry
from nurse_rcl.versioning import assert_frozen_snapshot

A1_NARROW_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.a1-narrow-audit/1"] = "nurse.v2.a1-narrow-audit/1"

AllowedA1Change = Literal[
    "candidate_fallback",
    "entity_binding",
    "topology_missing_detection",
]
ConfigurationValue: TypeAlias = str | tuple[str, ...] | None
ChangeClassification = Literal[
    "candidate_path_repair",
    "identity_metadata",
    "lineage_metadata",
]
ContractName = Literal[
    "detectors",
    "formulas",
    "prompts",
    "semantic_behavior",
    "weights",
]

_A0_MANIFEST_PATH: Literal["configs/v2/a0_frozen_manifest.json"] = (
    "configs/v2/a0_frozen_manifest.json"
)
_EXPERIMENT_REGISTRY_PATH: Literal["configs/v2/experiment_versions.json"] = (
    "configs/v2/experiment_versions.json"
)
_A1_RUNTIME_FILES = (
    "src/nurse_rcl/v2/paper_v1_repaired/__init__.py",
    "src/nurse_rcl/v2/paper_v1_repaired/candidates.py",
)
_ALLOWED_A1_CHANGES: tuple[AllowedA1Change, ...] = (
    "candidate_fallback",
    "entity_binding",
    "topology_missing_detection",
)
_EXPECTED_CHANGED_CONFIGURATION_FIELDS = frozenset(
    {
        "allowed_changes",
        "derived_from",
        "id",
        "immutable_manifest",
        "implementation_version",
        "kind",
        "localization_pipeline",
        "method_namespace",
        "name",
        "status",
    }
)
_CONFIGURATION_CLASSIFICATIONS: dict[str, ChangeClassification] = {
    "allowed_changes": "candidate_path_repair",
    "derived_from": "lineage_metadata",
    "id": "identity_metadata",
    "immutable_manifest": "lineage_metadata",
    "implementation_version": "identity_metadata",
    "kind": "identity_metadata",
    "localization_pipeline": "candidate_path_repair",
    "method_namespace": "identity_metadata",
    "name": "identity_metadata",
    "status": "identity_metadata",
}
_UNCHANGED_CONTRACT_ANCHORS: dict[ContractName, tuple[str, ...]] = {
    "detectors": ("a0_group:source",),
    "formulas": ("a0_group:source",),
    "prompts": ("a0_group:prompts",),
    "semantic_behavior": (
        "a0_group:prompts",
        "a0_group:source",
        "shared_config:credential_requirement",
        "shared_config:report_pipeline",
        "shared_config:semantic_stage",
    ),
    "weights": ("a0_group:configuration", "a0_group:source"),
}


class A1NarrowAuditMismatch(RuntimeError):
    """Raised when the committed audit no longer matches the repository state."""


class ChangedFile(StrictFrozenModel):
    """One A1 runtime file and its permitted candidate-path scope."""

    path: str = Field(pattern=r"^src/nurse_rcl/v2/paper_v1_repaired/.+\.py$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_changes: tuple[AllowedA1Change, ...]

    @model_validator(mode="after")
    def require_canonical_scope(self) -> Self:
        if self.allowed_changes != _ALLOWED_A1_CHANGES:
            raise ValueError("A1 runtime files must declare the complete narrow repair scope")
        return self


class ConfigurationFieldChange(StrictFrozenModel):
    """One explicit A0-to-A1 experiment-registry field difference."""

    field: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    before: ConfigurationValue
    after: ConfigurationValue
    classification: ChangeClassification

    @model_validator(mode="after")
    def require_actual_change(self) -> Self:
        if self.before == self.after:
            raise ValueError("configuration field change must change its value")
        return self


class SharedConfigurationField(StrictFrozenModel):
    """One A0/A1 field whose value must remain equal."""

    field: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: ConfigurationValue


class ProtectedA0Group(StrictFrozenModel):
    """One frozen A0 tree pin used as an unchanged-behavior anchor."""

    name: Literal["configuration", "environment", "prompts", "source"]
    file_count: int = Field(ge=0)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class UnchangedContract(StrictFrozenModel):
    """A behavior class anchored to immutable A0 groups and shared fields."""

    contract: ContractName
    anchors: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_anchors(self) -> Self:
        if self.anchors != tuple(sorted(set(self.anchors))):
            raise ValueError("unchanged-contract anchors must be sorted and unique")
        return self


class A1NarrowAudit(StrictFrozenModel):
    """Content-addressed proof that A1 changes only its candidate path."""

    schema_version: Literal["nurse.v2.a1-narrow-audit/1"] = A1_NARROW_AUDIT_SCHEMA_VERSION
    from_variant: Literal["A0"] = "A0"
    to_variant: Literal["A1"] = "A1"
    a0_manifest_path: Literal["configs/v2/a0_frozen_manifest.json"] = _A0_MANIFEST_PATH
    a0_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_registry_path: Literal["configs/v2/experiment_versions.json"] = (
        _EXPERIMENT_REGISTRY_PATH
    )
    experiment_registry_version: str = Field(min_length=1)
    changed_files: tuple[ChangedFile, ...]
    configuration_field_changes: tuple[ConfigurationFieldChange, ...]
    shared_configuration_fields: tuple[SharedConfigurationField, ...]
    protected_a0_groups: tuple[ProtectedA0Group, ...]
    unchanged_contracts: tuple[UnchangedContract, ...]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_membership(self) -> Self:
        for keys in (
            tuple(item.path for item in self.changed_files),
            tuple(item.field for item in self.configuration_field_changes),
            tuple(item.field for item in self.shared_configuration_fields),
            tuple(item.name for item in self.protected_a0_groups),
            tuple(item.contract for item in self.unchanged_contracts),
        ):
            if keys != tuple(sorted(set(keys))):
                raise ValueError("A1 audit collections must be sorted and unique")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"audit_sha256"})
        return self.audit_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _repository_file(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative_path:
        raise ValueError(f"unsafe repository-relative path: {relative_path!r}")
    path = (root / Path(*pure.parts)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"repository path escapes root: {relative_path!r}")
    return path


def _canonical_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _configuration_value(value: object) -> ConfigurationValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"unsupported experiment configuration value: {value!r}")


def _changed_files(root: Path) -> tuple[ChangedFile, ...]:
    runtime_root = _repository_file(root, "src/nurse_rcl/v2/paper_v1_repaired")
    actual_paths = tuple(
        sorted(path.relative_to(root).as_posix() for path in runtime_root.rglob("*.py"))
    )
    if actual_paths != _A1_RUNTIME_FILES:
        raise A1NarrowAuditMismatch(
            "A1 runtime changed-file membership differs from the declared narrow repair"
        )
    return tuple(
        ChangedFile(
            path=relative_path,
            content_sha256=_canonical_text_sha256(_repository_file(root, relative_path)),
            allowed_changes=_ALLOWED_A1_CHANGES,
        )
        for relative_path in actual_paths
    )


def _configuration_diff(
    root: Path,
) -> tuple[
    str,
    tuple[ConfigurationFieldChange, ...],
    tuple[SharedConfigurationField, ...],
]:
    registry = load_experiment_registry(_repository_file(root, _EXPERIMENT_REGISTRY_PATH))
    before = registry.require("A0").model_dump(mode="python")
    after = registry.require("A1").model_dump(mode="python")
    fields = tuple(sorted(set(before) | set(after)))
    changed_fields = frozenset(field for field in fields if before.get(field) != after.get(field))
    if changed_fields != _EXPECTED_CHANGED_CONFIGURATION_FIELDS:
        raise A1NarrowAuditMismatch(
            "A0-to-A1 configuration field diff exceeds or omits the declared boundary"
        )
    changes = tuple(
        ConfigurationFieldChange(
            field=field,
            before=_configuration_value(before.get(field)),
            after=_configuration_value(after.get(field)),
            classification=_CONFIGURATION_CLASSIFICATIONS[field],
        )
        for field in sorted(changed_fields)
    )
    shared = tuple(
        SharedConfigurationField(
            field=field,
            value=_configuration_value(before.get(field)),
        )
        for field in fields
        if field not in changed_fields
    )
    return registry.registry_version, changes, shared


def _protected_groups(
    root: Path,
) -> tuple[str, tuple[ProtectedA0Group, ...]]:
    manifest_path = _repository_file(root, _A0_MANIFEST_PATH)
    assert_frozen_snapshot(root, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = payload["groups"]
    if not isinstance(groups, dict) or set(groups) != {
        "configuration",
        "environment",
        "prompts",
        "source",
    }:
        raise A1NarrowAuditMismatch("A0 manifest group membership is invalid")
    pins = tuple(
        ProtectedA0Group(
            name=name,
            file_count=groups[name]["file_count"],
            tree_sha256=groups[name]["tree_sha256"],
        )
        for name in sorted(groups)
    )
    manifest_sha256 = sha256_hex(canonical_json_bytes(payload))
    return manifest_sha256, pins


def _unchanged_contracts() -> tuple[UnchangedContract, ...]:
    return tuple(
        UnchangedContract(contract=contract, anchors=tuple(sorted(anchors)))
        for contract, anchors in sorted(_UNCHANGED_CONTRACT_ANCHORS.items())
    )


def build_a1_narrow_audit(repository_root: Path) -> A1NarrowAudit:
    """Rebuild A1's exact source/config diff over the verified A0 snapshot."""

    root = repository_root.resolve(strict=True)
    a0_manifest_sha256, protected_groups = _protected_groups(root)
    registry_version, changes, shared = _configuration_diff(root)
    unsigned: dict[str, object] = {
        "schema_version": A1_NARROW_AUDIT_SCHEMA_VERSION,
        "from_variant": "A0",
        "to_variant": "A1",
        "a0_manifest_path": _A0_MANIFEST_PATH,
        "a0_manifest_sha256": a0_manifest_sha256,
        "experiment_registry_path": _EXPERIMENT_REGISTRY_PATH,
        "experiment_registry_version": registry_version,
        "changed_files": [item.model_dump(mode="json") for item in _changed_files(root)],
        "configuration_field_changes": [item.model_dump(mode="json") for item in changes],
        "shared_configuration_fields": [item.model_dump(mode="json") for item in shared],
        "protected_a0_groups": [item.model_dump(mode="json") for item in protected_groups],
        "unchanged_contracts": [item.model_dump(mode="json") for item in _unchanged_contracts()],
    }
    audit = A1NarrowAudit.model_validate(
        {
            **unsigned,
            "audit_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not audit.verify():  # pragma: no cover - construction invariant
        raise AssertionError("A1 narrow audit construction failed")
    return audit


def verify_a1_narrow_audit(
    repository_root: Path,
    audit: A1NarrowAudit,
) -> A1NarrowAudit:
    """Reject stale, incomplete, or self-inconsistent A1 audit artifacts."""

    if not audit.verify():
        raise A1NarrowAuditMismatch("A1 narrow audit hash failed verification")
    expected = build_a1_narrow_audit(repository_root)
    if audit != expected:
        raise A1NarrowAuditMismatch("A1 narrow audit does not match repository state")
    return audit


def load_a1_narrow_audit(
    repository_root: Path,
    audit_path: Path,
) -> A1NarrowAudit:
    """Load and rebuild-verify the committed A1 narrow-diff audit."""

    root = repository_root.resolve(strict=True)
    resolved_audit = audit_path.resolve(strict=True)
    if not resolved_audit.is_relative_to(root):
        raise ValueError("A1 audit path must remain inside the repository")
    payload = json.loads(resolved_audit.read_text(encoding="utf-8"))
    audit = A1NarrowAudit.model_validate(payload)
    return verify_a1_narrow_audit(root, audit)
