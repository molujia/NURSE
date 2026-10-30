"""Unsupervised V2 experiment-configuration validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.experiment_registry import ExperimentVersionRegistry

EXPERIMENT_CONFIGURATION_SCHEMA_VERSION: Literal[
    "nurse.v2.experiment-configuration/1"
] = "nurse.v2.experiment-configuration/1"

_ALLOWED_INFERENCE_FIELDS = frozenset(
    {
        "candidate_directory",
        "dataset",
        "decision_time",
        "entities",
        "evidence_cards",
        "graph_snapshot",
        "mechanism_library",
        "observations",
        "opaque_case_id",
        "verification_results",
    }
)
_ANSWER_BEARING_TOKENS = (
    "answer",
    "groundtruth",
    "injection",
    "labelpath",
    "originalcaseid",
    "originalpath",
    "rootcause",
)
_UNSUPERVISED_PARAMETER_SOURCES = frozenset(
    {"hand_specified", "label_free_statistics"}
)


class LearningConfiguration(StrictFrozenModel):
    regime: str = Field(min_length=1, max_length=64)
    benchmark_labels_for_training: bool
    benchmark_labels_for_model_selection: bool

    @model_validator(mode="after")
    def require_unsupervised_learning(self) -> Self:
        if (
            self.regime != "unsupervised"
            or self.benchmark_labels_for_training
            or self.benchmark_labels_for_model_selection
        ):
            raise ValueError(
                "benchmark labels cannot train, fit, or select a NURSE V2 method"
            )
        return self


class ScorerConfiguration(StrictFrozenModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(min_length=1, max_length=128)
    parameter_source: str = Field(min_length=1, max_length=64)
    artifact_path: str | None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def reject_label_fitted_artifacts(self) -> Self:
        if self.parameter_source not in _UNSUPERVISED_PARAMETER_SOURCES:
            raise ValueError("label-fitted scorer artifacts are forbidden")
        if (self.artifact_path is None) != (self.artifact_sha256 is None):
            raise ValueError("scorer artifact path and SHA-256 must be declared together")
        return self


class ExperimentConfiguration(StrictFrozenModel):
    schema_version: Literal["nurse.v2.experiment-configuration/1"] = (
        EXPERIMENT_CONFIGURATION_SCHEMA_VERSION
    )
    variant_id: str = Field(min_length=2, max_length=64)
    learning: LearningConfiguration
    scorer: ScorerConfiguration
    inference_payload_fields: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_safe_canonical_inference_fields(self) -> Self:
        fields = self.inference_payload_fields
        if fields != tuple(sorted(fields)) or len(fields) != len(set(fields)):
            raise ValueError("inference payload fields must be sorted and unique")
        for field_name in fields:
            collapsed = re.sub(r"[^a-z0-9]", "", field_name.lower())
            if any(token in collapsed for token in _ANSWER_BEARING_TOKENS):
                raise ValueError(f"answer-bearing inference field is forbidden: {field_name}")
            if field_name not in _ALLOWED_INFERENCE_FIELDS:
                raise ValueError(f"unapproved inference payload field: {field_name}")
        return self


def validate_experiment_configuration(
    payload: Mapping[str, Any],
    *,
    registry: ExperimentVersionRegistry,
) -> ExperimentConfiguration:
    """Validate a configuration and bind it to a declared experiment variant."""

    configuration = ExperimentConfiguration.model_validate(payload)
    try:
        registry.require(configuration.variant_id)
    except KeyError as error:
        raise ValueError(
            f"undeclared experiment variant: {configuration.variant_id}"
        ) from error
    return configuration
