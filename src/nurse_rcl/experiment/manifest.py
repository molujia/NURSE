"""Immutable run manifests and hard version-separated storage layouts."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

RUN_MANIFEST_SCHEMA_VERSION: Literal[1] = 1
MethodNamespace = Literal["v1", "v2"]


class ModelIdentity(StrictFrozenModel):
    provider: Literal["ark_openai_compatible"]
    endpoint: Literal["https://ark.cn-beijing.volces.com/api/v3"]
    llm_model: Literal["deepseek-v4-flash-ga-260731"]
    embedding_model: Literal["sentence-transformers/all-MiniLM-L6-v2"]
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VersionPin(StrictFrozenModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(min_length=1, max_length=128)


class RunConcurrency(StrictFrozenModel):
    case_workers: int = Field(ge=1)
    llm_in_flight: int = Field(ge=1)

    @model_validator(mode="after")
    def require_bounded_llm_concurrency(self) -> Self:
        if self.llm_in_flight != 1:
            raise ValueError("fixed V1 requires exactly one LLM in flight")
        return self


class RunFingerprints(StrictFrozenModel):
    source: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompts: str = Field(pattern=r"^[0-9a-f]{64}$")
    schemas: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunManifest(StrictFrozenModel):
    schema_version: Literal[1] = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    created_at_unix: int = Field(ge=0)
    method_namespace: MethodNamespace
    method_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    dataset: DatasetName
    configuration_json: str = Field(min_length=2)
    model_identity: ModelIdentity
    prompt_versions: tuple[VersionPin, ...] = Field(min_length=1)
    schema_versions: tuple[VersionPin, ...] = Field(min_length=1)
    fingerprints: RunFingerprints
    concurrency: RunConcurrency
    cross_run_llm_cache: bool

    @model_validator(mode="after")
    def require_consistent_immutable_identity(self) -> Self:
        if self.method_version == "paper_v1" and self.method_namespace != "v1":
            raise ValueError("paper_v1 must use the v1 method namespace")
        if self.method_namespace == "v1" and self.method_version != "paper_v1":
            raise ValueError("the v1 method namespace is reserved for paper_v1")
        expected_prefix = f"{self.method_namespace}-{self.dataset}-"
        pattern = rf"^{re.escape(expected_prefix)}\d{{8}}T\d{{6}}Z-[0-9a-f]{{32}}$"
        if re.fullmatch(pattern, self.run_id) is None:
            raise ValueError("run ID prefix must match method namespace and dataset")
        try:
            configuration = json.loads(self.configuration_json)
        except (TypeError, ValueError) as error:
            raise ValueError("configuration_json must be canonical JSON") from error
        if canonical_json_bytes(configuration).decode("utf-8") != self.configuration_json:
            raise ValueError("configuration_json must use canonical serialization")
        _require_sorted_unique_pins(self.prompt_versions, name="prompt")
        _require_sorted_unique_pins(self.schema_versions, name="schema")
        if self.method_namespace == "v1" and self.cross_run_llm_cache:
            raise ValueError("fixed V1 prohibits cross-run LLM caching")
        return self


class RunManifestArtifact(StrictFrozenModel):
    manifest: RunManifest
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = RunManifest.model_validate(self.manifest.model_dump(mode="json"))
        except (TypeError, ValueError):
            return False
        if validated != self.manifest:
            return False
        fingerprints = self.manifest.fingerprints
        if fingerprints.configuration != sha256_hex(
            self.manifest.configuration_json.encode("utf-8")
        ):
            return False
        if fingerprints.model != _model_hash(self.manifest.model_identity):
            return False
        if fingerprints.prompts != _pins_hash(self.manifest.prompt_versions):
            return False
        if fingerprints.schemas != _pins_hash(self.manifest.schema_versions):
            return False
        return self.manifest_hash == _model_hash(self.manifest)


class RunStorageLayout(StrictFrozenModel):
    run_root: Path
    cache_root: Path
    history_root: Path


def allocate_run_id(
    *,
    method_namespace: MethodNamespace,
    dataset: DatasetName,
) -> str:
    """Allocate a filesystem-safe UTC/UUID run identity in one version namespace."""

    if method_namespace not in {"v1", "v2"}:
        raise ValueError("method_namespace must be v1 or v2")
    if dataset not in {"rcabench", "aiops25"}:
        raise ValueError("dataset must be rcabench or aiops25")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{method_namespace}-{dataset}-{timestamp}-{uuid.uuid4().hex}"


def create_run_manifest(
    *,
    run_id: str,
    created_at_unix: int,
    method_namespace: MethodNamespace,
    method_version: str,
    dataset: DatasetName,
    source_fingerprint: str,
    configuration: Mapping[str, Any],
    dataset_fingerprint: str,
    dependency_fingerprint: str,
    model_identity: ModelIdentity,
    prompt_versions: Mapping[str, str],
    schema_versions: Mapping[str, str],
    concurrency: RunConcurrency,
    cross_run_llm_cache: bool,
) -> RunManifestArtifact:
    """Freeze all run identities and their reproducibility fingerprints."""

    configuration_json = canonical_json_bytes(dict(configuration)).decode("utf-8")
    prompt_pins = _pins(prompt_versions)
    schema_pins = _pins(schema_versions)
    fingerprints = RunFingerprints(
        source=source_fingerprint,
        configuration=sha256_hex(configuration_json.encode("utf-8")),
        dataset=dataset_fingerprint,
        dependencies=dependency_fingerprint,
        model=_model_hash(model_identity),
        prompts=_pins_hash(prompt_pins),
        schemas=_pins_hash(schema_pins),
    )
    manifest = RunManifest(
        run_id=run_id,
        created_at_unix=created_at_unix,
        method_namespace=method_namespace,
        method_version=method_version,
        dataset=dataset,
        configuration_json=configuration_json,
        model_identity=model_identity,
        prompt_versions=prompt_pins,
        schema_versions=schema_pins,
        fingerprints=fingerprints,
        concurrency=concurrency,
        cross_run_llm_cache=cross_run_llm_cache,
    )
    result = RunManifestArtifact(
        manifest=manifest,
        manifest_hash=_model_hash(manifest),
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("run manifest artifact verification failed")
    return result


def resolve_run_layout(
    *,
    storage_root: Path,
    artifact: RunManifestArtifact,
) -> RunStorageLayout:
    """Resolve physically separated run, cache, and history paths without creating them."""

    if not artifact.verify():
        raise ValueError("run manifest verification failed")
    root = storage_root.resolve(strict=False)
    manifest = artifact.manifest
    namespace = manifest.method_namespace
    dataset = manifest.dataset
    return RunStorageLayout(
        run_root=root / "runs" / namespace / dataset / manifest.run_id,
        cache_root=root / "caches" / namespace / dataset,
        history_root=root / "history" / namespace / dataset,
    )


def _pins(versions: Mapping[str, str]) -> tuple[VersionPin, ...]:
    return tuple(
        VersionPin(name=name, version=version)
        for name, version in sorted(versions.items())
    )


def _pins_hash(pins: tuple[VersionPin, ...]) -> str:
    return sha256_hex(
        canonical_json_bytes([pin.model_dump(mode="json") for pin in pins])
    )


def _require_sorted_unique_pins(
    pins: tuple[VersionPin, ...],
    *,
    name: str,
) -> None:
    names = tuple(pin.name for pin in pins)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError(f"{name} version pins must be sorted and unique")


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
