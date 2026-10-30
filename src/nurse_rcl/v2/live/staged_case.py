"""Inference-only reader of a pinned, opaque, label-free case export.

This module never opens evaluator mappings or resolves original case paths.
Raw telemetry remains private adapter input, not an outbound model payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.live.sources import (
    RCABenchLiveSources, _VALUE_COLUMNS, file_sha256, read_rcabench_live_sources,
)
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.rcabench_adapter import RCABenchAdapterConfiguration

TELEMETRY_FILES = tuple(sorted(
    f'{window}_{kind}.parquet' for window in ('normal', 'abnormal') for kind in _VALUE_COLUMNS
))
ENV_FIELDS = ('NORMAL_START', 'NORMAL_END', 'ABNORMAL_START', 'ABNORMAL_END', 'TIMEZONE')


class StagedFile(StrictFrozenModel):
    name: str
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    export_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size_bytes: int = Field(ge=0)
    rows: int | None = Field(default=None, ge=0)


class StagedCase(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-staged-case/1'] = 'nurse.v2.live-staged-case/1'
    dataset: Literal['rcabench'] = 'rcabench'
    opaque_case_id: str = Field(pattern=r'^opq_[0-9a-f]{32}$')
    inference_artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    files: tuple[StagedFile, ...]
    missing_source_files: tuple[str, ...]
    stage_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    def verify(self) -> bool:
        names = tuple(item.name for item in self.files)
        allowed = {'env.json', *TELEMETRY_FILES}
        return (
            names == tuple(sorted(set(names)))
            and 'env.json' in names
            and set(names) <= allowed
            and self.missing_source_files == tuple(sorted(allowed - set(names)))
            and self.stage_sha256 == sha256_hex(canonical_json_bytes(
                self.model_dump(mode='json', exclude={'stage_sha256'})))
        )


def require_plain_path(path: Path) -> Path:
    """Reject symlink components, including ancestors, before resolving a path."""
    absolute = Path(path).absolute()
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise ValueError('symlink is not an authorized live source')
    return absolute.resolve(strict=True)


def read_inference_identity(path: Path, expected_sha256: str) -> InferenceIdentityManifest:
    source = require_plain_path(path)
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('identity manifest exceeds the bounded import size')
    data = source.read_bytes()
    if sha256_hex(data) != expected_sha256:
        raise ValueError('inference identity artifact pin mismatch')
    manifest = InferenceIdentityManifest.model_validate_json(data)
    if not manifest.verify():
        raise ValueError('invalid inference identity manifest')
    return manifest


def load_staged_rcabench_case(
    root: Path, *, expected_stage_sha256: str, inference_manifest_path: Path,
    expected_inference_artifact_sha256: str, configuration: RCABenchAdapterConfiguration,
) -> RCABenchLiveSources:
    """Verify independent stage/identity pins, then use the original source reader."""
    source = require_plain_path(root)
    manifest = read_inference_identity(inference_manifest_path, expected_inference_artifact_sha256)
    receipt = require_plain_path(source / 'stage.json')
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError('stage receipt exceeds the bounded import size')
    data = receipt.read_bytes()
    stage = StagedCase.model_validate_json(data)
    if (not stage.verify() or stage.stage_sha256 != expected_stage_sha256
            or stage.inference_artifact_sha256 != expected_inference_artifact_sha256
            or source.name != stage.opaque_case_id
            or not any(record.dataset == 'rcabench' and record.opaque_case_id == stage.opaque_case_id
                       for record in manifest.records)):
        raise ValueError('stage identity or independent pin mismatch')

    def check_files() -> None:
        if {item.name for item in source.iterdir()} != {'stage.json', *(item.name for item in stage.files)}:
            raise ValueError('stage contains missing or undeclared files')
        if receipt.read_bytes() != data:
            raise ValueError('stage receipt changed during read')
        for item in stage.files:
            path = require_plain_path(source / item.name)
            if path.stat().st_size != item.size_bytes or file_sha256(path) != item.export_sha256:
                raise ValueError('staged source does not match the independent receipt')

    check_files()
    try:
        return read_rcabench_live_sources(source, opaque_case_id=stage.opaque_case_id,
                                         configuration=configuration)
    finally:
        check_files()
