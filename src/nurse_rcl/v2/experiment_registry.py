"""Strict experiment-version registry for NURSE comparison boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

EXPERIMENT_REGISTRY_SCHEMA_VERSION: Literal[
    "nurse.v2.experiment-version-registry/1"
] = "nurse.v2.experiment-version-registry/1"

VariantKind = Literal[
    "historical_baseline",
    "repair_baseline",
    "small_model",
    "report_only",
    "hybrid",
    "ablation",
    "external_baseline",
]
VariantStatus = Literal["frozen", "planned", "selection_pending"]
MethodNamespace = Literal["v1", "v2", "external"]
SemanticStage = Literal["none", "pre_ranking", "native"]
CredentialRequirement = Literal["none", "ark_api_key2", "native_pending"]

_REQUIRED_VARIANTS = frozenset(
    {
        "A0",
        "A1",
        "B0",
        "B1",
        "B2",
        "B2-no-propagation",
        "B2-no-role-adjustment",
        "S-simple-statistical",
        "S-strong-external",
    }
)
_A1_ALLOWED_CHANGES = (
    "candidate_fallback",
    "entity_binding",
    "topology_missing_detection",
)


class ResearchContractPin(StrictFrozenModel):
    path: str = Field(pattern=r"^configs/v2/[a-z0-9_]+\.json$")
    hash_mode: Literal["sha256-canonical-json-v1"] = "sha256-canonical-json-v1"
    sha256: str = Field(pattern=r"^[0-9A-Fa-f]{64}$")


class ExperimentVariant(StrictFrozenModel):
    id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Z][A-Za-z0-9-]*$")
    name: str = Field(min_length=1, max_length=128)
    kind: VariantKind
    status: VariantStatus
    method_namespace: MethodNamespace
    implementation_version: str = Field(min_length=1, max_length=128)
    localization_pipeline: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    semantic_stage: SemanticStage
    report_pipeline: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    credential_requirement: CredentialRequirement
    derived_from: str | None = None
    ranking_equivalent_to: str | None = None
    ablation_of: str | None = None
    allowed_changes: tuple[str, ...] = ()
    disabled_modules: tuple[str, ...] = ()
    immutable_manifest: str | None = None

    @model_validator(mode="after")
    def require_canonical_scopes(self) -> Self:
        for field_name, values in (
            ("allowed_changes", self.allowed_changes),
            ("disabled_modules", self.disabled_modules),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be sorted and unique")
        if self.kind == "external_baseline":
            if self.method_namespace != "external":
                raise ValueError("external baselines must use the external namespace")
        elif self.method_namespace == "external":
            raise ValueError("only external baselines may use the external namespace")
        if self.kind == "ablation" and self.ablation_of is None:
            raise ValueError("ablation variants require ablation_of")
        return self


class ExperimentVersionRegistry(StrictFrozenModel):
    schema_version: Literal["nurse.v2.experiment-version-registry/1"] = (
        EXPERIMENT_REGISTRY_SCHEMA_VERSION
    )
    registry_version: str = Field(pattern=r"^v[0-9]+\.[0-9]+-[a-z0-9-]+$")
    research_contract: ResearchContractPin
    variants: tuple[ExperimentVariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_comparison_boundaries(self) -> Self:
        ids = tuple(variant.id for variant in self.variants)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("variant IDs must be sorted and unique")
        by_id = {variant.id: variant for variant in self.variants}
        missing = sorted(_REQUIRED_VARIANTS - by_id.keys())
        if missing:
            raise ValueError(f"required experiment variants are missing: {', '.join(missing)}")

        for variant in self.variants:
            for field_name in ("derived_from", "ranking_equivalent_to", "ablation_of"):
                referenced_id = getattr(variant, field_name)
                if referenced_id is None:
                    continue
                if referenced_id not in by_id:
                    raise ValueError(
                        f"{variant.id}.{field_name} references unknown variant {referenced_id}"
                    )
                if referenced_id == variant.id:
                    raise ValueError(f"{variant.id}.{field_name} cannot reference itself")

        self._require_fixed_boundaries(by_id)
        return self

    @staticmethod
    def _require_fixed_boundaries(by_id: dict[str, ExperimentVariant]) -> None:
        a0 = by_id["A0"]
        if (
            a0.kind != "historical_baseline"
            or a0.status != "frozen"
            or a0.method_namespace != "v1"
            or a0.immutable_manifest != "configs/v2/a0_frozen_manifest.json"
            or a0.allowed_changes
        ):
            raise ValueError("A0 must remain the immutable V1-frozen baseline")

        a1 = by_id["A1"]
        if (
            a1.kind != "repair_baseline"
            or a1.derived_from != "A0"
            or a1.allowed_changes != _A1_ALLOWED_CHANGES
        ):
            raise ValueError("A1 must remain the narrow candidate-path repair of A0")

        b0 = by_id["B0"]
        if (
            b0.kind != "small_model"
            or b0.semantic_stage != "none"
            or b0.report_pipeline != "deterministic_v2"
            or b0.credential_requirement != "none"
        ):
            raise ValueError("B0 must remain the credential-free deterministic baseline")

        b1 = by_id["B1"]
        b2 = by_id["B2"]
        if (
            b1.kind != "report_only"
            or b1.derived_from != "B0"
            or b1.ranking_equivalent_to != "B0"
            or b1.semantic_stage != "none"
            or b1.report_pipeline != "shared_v2_report_agent"
        ):
            raise ValueError("B1 must preserve B0 ranking and add only the shared report Agent")
        if (
            b2.kind != "hybrid"
            or b2.derived_from != "B0"
            or b2.ranking_equivalent_to is not None
            or b2.semantic_stage != "pre_ranking"
            or b2.report_pipeline != b1.report_pipeline
        ):
            raise ValueError("B2 must add only verified pre-ranking semantics and shared reporting")
        if b1.credential_requirement != "ark_api_key2" or b2.credential_requirement != (
            "ark_api_key2"
        ):
            raise ValueError("B1 and B2 must use only the declared ARK_API_KEY2 boundary")

        required_ablations = {
            "B2-no-propagation": ("propagation",),
            "B2-no-role-adjustment": ("verified_role_adjustment",),
        }
        for variant_id, disabled_modules in required_ablations.items():
            variant = by_id[variant_id]
            if (
                variant.kind != "ablation"
                or variant.derived_from != "B2"
                or variant.ablation_of != "B2"
                or variant.disabled_modules != disabled_modules
            ):
                raise ValueError(f"{variant_id} must remain a named core B2 ablation")

        for variant_id in ("S-simple-statistical", "S-strong-external"):
            if by_id[variant_id].kind != "external_baseline":
                raise ValueError(f"{variant_id} must remain an external baseline")

    def require(self, variant_id: str) -> ExperimentVariant:
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise KeyError(f"undeclared experiment variant: {variant_id}")


def load_experiment_registry(path: Path) -> ExperimentVersionRegistry:
    """Load and strictly validate one versioned comparison registry."""

    resolved_path = path.resolve(strict=True)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    registry = ExperimentVersionRegistry.model_validate(payload)
    repository_root = resolved_path.parents[2]
    contract_path = (repository_root / registry.research_contract.path).resolve(strict=True)
    if not contract_path.is_relative_to(repository_root):
        raise ValueError("research contract path escapes repository root")
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = sha256_hex(canonical_json_bytes(contract_payload))
    if contract_hash != registry.research_contract.sha256.lower():
        raise ValueError("research contract hash mismatch")
    return registry
