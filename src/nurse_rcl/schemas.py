"""Strict data contracts separating inference telemetry from evaluation labels."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

DatasetName = Literal["rcabench", "aiops25"]
TelemetryModality = Literal["metrics", "normal_traces", "fault_traces", "logs"]

_LABEL_PATH_PATTERN = re.compile(r"groundtruth|injection|answerkey", re.IGNORECASE)


class StrictFrozenModel(BaseModel):
    """Base contract for immutable artifacts that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TelemetryReference(StrictFrozenModel):
    modality: TelemetryModality
    path: Path

    @field_validator("path")
    @classmethod
    def reject_label_bearing_path(cls, path: Path) -> Path:
        collapsed = re.sub(r"[-_\s]", "", path.as_posix())
        if _LABEL_PATH_PATTERN.search(collapsed):
            raise ValueError("label-bearing paths are forbidden during inference")
        if not path.is_absolute():
            raise ValueError("telemetry paths must be absolute")
        return path

    @model_validator(mode="after")
    def require_authorized_root(self, info: ValidationInfo) -> Self:
        context = info.context if isinstance(info.context, dict) else {}
        configured_roots = context.get("allowed_roots", ())
        path = self.path.resolve(strict=False)
        roots = tuple(Path(root).resolve(strict=False) for root in configured_roots)
        if not roots or not any(path.is_relative_to(root) for root in roots):
            raise ValueError("telemetry path is outside authorized telemetry roots")
        return self


class InferenceCase(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    dataset: DatasetName
    t0: int
    telemetry: tuple[TelemetryReference, ...] = Field(min_length=1)


class InferenceManifest(StrictFrozenModel):
    schema_version: Literal[1] = 1
    dataset: DatasetName
    method_version: Literal["paper_v1"]
    cases: tuple[InferenceCase, ...]


class EvaluationCase(StrictFrozenModel):
    case_id: str = Field(min_length=1)
    label_path: Path | None = None
    label_field: str | None = None
    excluded: bool = False
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def require_label_or_exclusion(self) -> Self:
        if self.excluded:
            if not self.exclusion_reason:
                raise ValueError("excluded evaluation cases require an exclusion reason")
        elif self.label_path is None or not self.label_field:
            raise ValueError("included evaluation cases require a label reference")
        return self


class EvaluationManifest(StrictFrozenModel):
    schema_version: Literal[1] = 1
    dataset: DatasetName
    cases: tuple[EvaluationCase, ...]
