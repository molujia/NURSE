"""Pinned one-case server admission. Metadata only; no provider or raw scanning."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned, b0_configuration
from nurse_rcl.v2.live.normal_preparation import _require_compatible_key
from nurse_rcl.v2.live.ops_contracts import (
    OpsEmpiricalProfile,
    OpsSreActionProfile,
    OpsSreAtomicSelectorProfile,
    OpsSreBalancedProfile,
    OpsSreBriefProfile,
    OpsSreComparableProfile,
    OpsSreConditionProfile,
    OpsSreDirectedProfile,
    OpsSreEntryProfile,
    OpsSreFeedbackFactProfile,
    OpsSreFocusedProfile,
    OpsSreNavigationProfile,
    OpsSreObservableProfile,
    OpsSreProfile,
    OpsSreProfileBase,
    OpsSreQuestionProfile,
    OpsSreReceiptReuseProfile,
    OpsSreRecoveryProfile,
    OpsSreReplanningProfile,
    OpsSreSingleOwnerProfile,
    OpsSreTypedPurposeProfile,
    OpsSreUsedSourceProfile,
    OpsSreWorkbenchProfile,
    OpsSreWorkingProfile,
)
from nurse_rcl.v2.live.ops_contracts import OpsProfile as OpsProfile
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationRun
from nurse_rcl.v2.live.repair_stage import RepairCaseInput
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import read_inference_identity, require_plain_path
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config


def runtime_environment() -> dict[str, str]:
    """Numerical environment only; not credentials, model, prompt or agent mode."""
    return dict(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        **{name: version(name) for name in ("numpy", "scipy", "pyarrow", "pydantic", "httpx")},
        **{
            name: os.environ.get(name, "")
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    )


def plain_location(path: Path) -> Path:
    """Resolve a destination without requiring an unused historical path to exist."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("isolated locations must be absolute and traversal-free")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("isolated locations cannot contain symlinks")
    return path.resolve(strict=False)


def code_inventory(root: Path) -> dict[str, str]:
    root = require_plain_path(root)
    paths: list[Path] = []
    for folder in ("src", "third_party/hermes_runtime"):
        base = require_plain_path(root / folder)
        if not base.is_dir():
            raise ValueError("missing frozen source directory")
        paths.extend(base.rglob("*.py"))
    paths.extend((root / "src/nurse_rcl/v2/live/ops_skills").glob("*.md"))
    paths.extend(
        root / name
        for name in (
            "configs/v2/ops_isolated_profile.json",
            "configs/v2/ops_isolated_empirical_profile.json",
            "configs/v2/ops_sre_profile.json",
            "configs/v2/ops_sre_working_profile.json",
            "configs/v2/ops_sre_brief_profile.json",
            "configs/v2/ops_sre_condition_profile.json",
            "configs/v2/ops_sre_recovery_profile.json",
            "configs/v2/ops_sre_comparable_profile.json",
            "configs/v2/ops_sre_navigation_profile.json",
            "configs/v2/ops_sre_workbench_profile.json",
            "configs/v2/ops_sre_used_source_profile.json",
            "configs/v2/ops_sre_receipt_reuse_profile.json",
            "configs/v2/ops_sre_balanced_triage_profile.json",
            "configs/v2/ops_sre_source_replanning_profile.json",
            "configs/v2/ops_sre_question_profile.json",
            "configs/v2/ops_sre_action_profile.json",
            "configs/v2/ops_sre_directed_profile.json",
            "configs/v2/ops_sre_entry_profile.json",
            "configs/v2/ops_sre_observable_profile.json",
            "configs/v2/ops_sre_single_owner_profile.json",
            "configs/v2/ops_sre_typed_purpose_profile.json",
            "configs/v2/ops_sre_atomic_selector_profile.json",
            "configs/v2/ops_sre_feedback_fact_profile.json",
            "configs/v2/ops_sre_focused_profile.json",
            "pyproject.toml",
            "uv.lock",
            "third_party/hermes_runtime/SOURCE_MANIFEST.json",
            "scripts/dev/run_ops_isolated.sh",
        )
    )
    return {
        p.relative_to(root).as_posix(): file_sha256(require_plain_path(p)) for p in sorted(paths)
    }


class OpsCaseReference(StrictFrozenModel):
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    input_path: str
    input_sha256: SHA256


class OpsSreCoreReference(StrictFrozenModel):
    plan_path: str
    plan_sha256: SHA256
    case_result_sha256: SHA256


class OpsSinglePlan(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-single-plan/1"] = "nurse.v2.ops-single-plan/1"
    code_root: str
    code_files: dict[str, SHA256]
    environment: dict[str, str]
    profile: (
        OpsProfile
        | OpsEmpiricalProfile
        | OpsSreProfile
        | OpsSreWorkingProfile
        | OpsSreBriefProfile
        | OpsSreConditionProfile
        | OpsSreRecoveryProfile
        | OpsSreComparableProfile
        | OpsSreNavigationProfile
        | OpsSreWorkbenchProfile
        | OpsSreUsedSourceProfile
        | OpsSreReceiptReuseProfile
        | OpsSreBalancedProfile
        | OpsSreReplanningProfile
        | OpsSreQuestionProfile
        | OpsSreActionProfile
        | OpsSreDirectedProfile
        | OpsSreEntryProfile
        | OpsSreObservableProfile
        | OpsSreSingleOwnerProfile
        | OpsSreTypedPurposeProfile
        | OpsSreAtomicSelectorProfile
        | OpsSreFeedbackFactProfile
        | OpsSreFocusedProfile
    )
    mode: Literal["core-only", "with-agent"]
    case: OpsCaseReference
    preparation: RepairPreparationRun
    identity_manifest_path: str
    identity_artifact_sha256: SHA256
    parent_plan_path: str
    parent_plan_sha256: SHA256
    output_root: str
    source_core: OpsSreCoreReference | None = None

    @model_validator(mode="after")
    def source_core_profile(self) -> Self:
        if self.source_core is not None and (
            not isinstance(self.profile, OpsSreProfileBase) or self.mode != "with-agent"
        ):
            raise ValueError("source core import requires an explicit new SRE investigation")
        return self

    @property
    def code_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.code_files))

    @property
    def core_configuration_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(b0_configuration()))

    @property
    def environment_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.environment))


@dataclass(frozen=True)
class OpsAdmittedPlan:
    plan: OpsSinglePlan
    plan_sha256: str
    identity_manifest: InferenceIdentityManifest
    raw_arguments: dict[str, Any]

    @property
    def state_root(self) -> Path:
        return Path(self.plan.output_root) / self.plan.case.opaque_case_id


def read_ops_plan(path: Path, pin: str) -> OpsSinglePlan:
    raw = _pinned(path, pin)
    _metadata(raw)  # Duplicate keys, depth and capacity before typed admission.
    return OpsSinglePlan.model_validate_json(raw)


def require_ops_code(plan: OpsSinglePlan) -> None:
    root = require_plain_path(Path(plan.code_root))
    if not Path(__file__).resolve().is_relative_to(root) or code_inventory(root) != plan.code_files:
        raise ValueError("isolated frozen code or skill content differs")
    if runtime_environment() != plan.environment:
        raise ValueError("isolated execution environment differs")
    empirical = isinstance(plan.profile, OpsEmpiricalProfile)
    if isinstance(plan.profile, OpsSreProfileBase):
        filename = {
            "hermes-sre-tools/4": "ops_sre_profile.json",
            "hermes-sre-tools/5": "ops_sre_working_profile.json",
            "hermes-sre-tools/6": "ops_sre_brief_profile.json",
            "hermes-sre-tools/7": "ops_sre_condition_profile.json",
            "hermes-sre-tools/8": "ops_sre_recovery_profile.json",
            "hermes-sre-tools/9": "ops_sre_comparable_profile.json",
            "hermes-sre-tools/10": "ops_sre_navigation_profile.json",
            "hermes-sre-tools/11": "ops_sre_workbench_profile.json",
            "hermes-sre-tools/12": "ops_sre_used_source_profile.json",
            "hermes-sre-tools/13": "ops_sre_receipt_reuse_profile.json",
            "hermes-sre-tools/14": "ops_sre_balanced_triage_profile.json",
            "hermes-sre-tools/15": "ops_sre_source_replanning_profile.json",
            "hermes-sre-tools/16": "ops_sre_question_profile.json",
            "hermes-sre-tools/17": "ops_sre_action_profile.json",
            "hermes-sre-tools/18": "ops_sre_directed_profile.json",
            "hermes-sre-tools/19": "ops_sre_entry_profile.json",
            "hermes-sre-tools/20": "ops_sre_observable_profile.json",
            "hermes-sre-tools/21": "ops_sre_single_owner_profile.json",
            "hermes-sre-tools/22": "ops_sre_typed_purpose_profile.json",
            "hermes-sre-tools/23": "ops_sre_atomic_selector_profile.json",
            "hermes-sre-tools/25": "ops_sre_focused_profile.json",
        }[plan.profile.investigation_profile_id]
        if isinstance(plan.profile, OpsSreFeedbackFactProfile):
            filename = "ops_sre_feedback_fact_profile.json"
        registered = type(plan.profile).model_validate_json(
            (root / "configs/v2" / filename).read_bytes()
        )
        if registered != plan.profile:
            raise ValueError("SRE registered profile differs")
        return
    profile = (OpsEmpiricalProfile if empirical else OpsProfile).model_validate_json(
        (
            root
            / "configs/v2"
            / ("ops_isolated_empirical_profile.json" if empirical else "ops_isolated_profile.json")
        ).read_bytes(),
    )
    if profile != plan.profile:
        raise ValueError("isolated registered profile differs")


def protected_locations(plan: OpsSinglePlan) -> tuple[Path, ...]:
    spec = RepairCaseInput.model_validate_json(
        _pinned(Path(plan.case.input_path), plan.case.input_sha256)
    )
    parent = _metadata(_pinned(Path(plan.parent_plan_path), plan.parent_plan_sha256))
    paths = [Path(spec.source_root), Path(plan.preparation.archive_root), Path(plan.code_root)]
    if type(parent.get("output_root")) is str:
        paths.append(Path(parent["output_root"]))
    return tuple(plain_location(path) for path in paths)


def load_ops_plan(path: Path, pin: str) -> OpsAdmittedPlan:
    plan = read_ops_plan(path, pin)
    require_ops_code(plan)
    manifest = read_inference_identity(
        Path(plan.identity_manifest_path), plan.identity_artifact_sha256
    )
    case, preparation = plan.case, plan.preparation
    spec = RepairCaseInput.model_validate_json(_pinned(Path(case.input_path), case.input_sha256))
    if (spec.dataset, spec.opaque_case_id) != (case.dataset, case.opaque_case_id) or not any(
        (r.dataset, r.opaque_case_id) == (case.dataset, case.opaque_case_id)
        for r in manifest.records
    ):
        raise ValueError("isolated case outside pinned inference scope")
    key = preparation.key
    if (key.dataset, key.opaque_case_id, key.input_sha256, key.source_files) != (
        case.dataset,
        case.opaque_case_id,
        case.input_sha256,
        spec.source_files,
    ) or Path(preparation.archive_root).name != key.key_sha256:
        raise ValueError("isolated complete case/preparation identity differs")
    parent = _metadata(_pinned(Path(plan.parent_plan_path), plan.parent_plan_sha256))
    parent_cases = parent.get("cases", [parent.get("case")])
    if case.model_dump(mode="json") not in parent_cases:
        raise ValueError("isolated case not present in frozen input lineage")
    out = require_plain_path(Path(plan.output_root))
    protected = [Path(spec.source_root), Path(preparation.archive_root), Path(plan.code_root)]
    if type(parent.get("output_root")) is str:
        protected.append(Path(parent["output_root"]))
    for other in protected:
        fixed = plain_location(other)
        if out.is_relative_to(fixed) or fixed.is_relative_to(out):
            raise ValueError("isolated output overlaps protected input or historical output")
    configuration: Any
    extra: dict[str, Any] = {}
    if case.dataset == "rcabench":
        configuration = build_rcabench_adapter_config(window_seconds=240)
        if (
            spec.stage_sha256 is None
            or spec.source_case_id is not None
            or (spec.authorized_queue is not None)
        ):
            raise ValueError("isolated RCABench requires opaque staged input")
    else:
        configuration = build_aiops25_adapter_config(window_seconds=480)
        queue = spec.authorized_queue
        if (
            spec.stage_sha256 is not None
            or queue is None
            or not queue.verify()
            or (spec.source_case_id is None or spec.source_case_id not in queue.case_ids)
        ):
            raise ValueError("isolated AIOps25 requires an audited included case queue")
        extra = dict(source_case_id=spec.source_case_id, authorized_queue=queue)
    if configuration.configuration_sha256 != spec.configuration_sha256:
        raise ValueError("isolated adapter configuration differs")
    raw = dict(
        source_root=require_plain_path(Path(spec.source_root)),
        dataset=case.dataset,
        opaque_case_id=case.opaque_case_id,
        configuration=configuration,
        expected_source_descriptor_sha256=spec.source_descriptor_sha256,
        store_root=Path(preparation.archive_root) / "blocks",
        ledger_root=out / case.opaque_case_id / "core/queries",
        **extra,
    )
    _require_compatible_key(key, raw)
    return OpsAdmittedPlan(plan, pin, manifest, raw)
