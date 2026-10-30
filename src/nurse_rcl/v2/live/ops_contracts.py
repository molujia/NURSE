"""Versioned data-only contracts for deterministic RCL and read-only investigation.

These schemas confer no execution authority. The core owns its actual consumption
record; investigation cannot mark new evidence as a formal scoring input.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, TruthValue
from nurse_rcl.v2.live.normal_artifacts import NormalEvidence

NodeKind = Literal["boundary", "symptoms", "dependency", "timing", "mechanisms", "scoring"]
Nonempty = Annotated[str, Field(min_length=1, max_length=4096)]


class OpsProfileBase(StrictFrozenModel):
    core_method_id: Literal["normal-b0-ops-core/1"] = "normal-b0-ops-core/1"
    action_limit: Literal[8] = 8
    decision_limit: Literal[8] = 8
    report_limit: Literal[1] = 1
    attempt_limit: Literal[5] = 5
    llm_in_flight: Literal[1] = 1
    api_key_env: Literal["ARK_API_KEY2"] = "ARK_API_KEY2"
    base_url: Literal["https://ark.cn-beijing.volces.com/api/v3"] = (
        "https://ark.cn-beijing.volces.com/api/v3"
    )
    model: Literal["deepseek-v4-flash-ga-260731"] = "deepseek-v4-flash-ga-260731"


class OpsProfile(OpsProfileBase):
    schema_version: Literal["nurse.v2.ops-isolated-profile/1"] = "nurse.v2.ops-isolated-profile/1"
    profile_id: Literal["v2-ops-isolated/1"] = "v2-ops-isolated/1"
    investigation_profile_id: Literal["hermes-readonly-ops/1"] = "hermes-readonly-ops/1"


class OpsEmpiricalProfile(OpsProfileBase):
    schema_version: Literal["nurse.v2.ops-isolated-profile/2"] = "nurse.v2.ops-isolated-profile/2"
    profile_id: Literal["v2-ops-isolated/2"] = "v2-ops-isolated/2"
    investigation_profile_id: Literal["hermes-readonly-ops/2"] = "hermes-readonly-ops/2"
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    decision_protocol: Literal["nurse.v2.ops-decision-exchange/3"] = (
        "nurse.v2.ops-decision-exchange/3"
    )
    report_protocol: Literal["nurse.v2.ops-report-exchange/3"] = "nurse.v2.ops-report-exchange/3"
    output_policy: Literal["explicit_max_tokens_65536/1"] = "explicit_max_tokens_65536/1"


class OpsSreProfileBase(OpsProfileBase):
    profile_id: Literal["v2-ops-sre/1"] = "v2-ops-sre/1"
    usage_policy_id: Literal["ops-empirical-input-cost/1"] = "ops-empirical-input-cost/1"
    output_policy: Literal["explicit_max_tokens_65536/1"] = "explicit_max_tokens_65536/1"


class OpsSreProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/1"] = "nurse.v2.sre-profile/1"
    investigation_profile_id: Literal["hermes-sre-tools/4"] = "hermes-sre-tools/4"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/5"] = (
        "nurse.v2.sre-decision-exchange/5"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/7"] = "nurse.v2.sre-report-exchange/7"


class OpsSreWorkingProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/2"] = "nurse.v2.sre-profile/2"
    investigation_profile_id: Literal["hermes-sre-tools/5"] = "hermes-sre-tools/5"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/6"] = (
        "nurse.v2.sre-decision-exchange/6"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/8"] = "nurse.v2.sre-report-exchange/8"


class OpsSreBriefProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/3"] = "nurse.v2.sre-profile/3"
    investigation_profile_id: Literal["hermes-sre-tools/6"] = "hermes-sre-tools/6"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/7"] = (
        "nurse.v2.sre-decision-exchange/7"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/9"] = "nurse.v2.sre-report-exchange/9"


class OpsSreConditionProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/4"] = "nurse.v2.sre-profile/4"
    investigation_profile_id: Literal["hermes-sre-tools/7"] = "hermes-sre-tools/7"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/8"] = (
        "nurse.v2.sre-decision-exchange/8"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/10"] = "nurse.v2.sre-report-exchange/10"


class OpsSreRecoveryProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/5"] = "nurse.v2.sre-profile/5"
    investigation_profile_id: Literal["hermes-sre-tools/8"] = "hermes-sre-tools/8"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/9"] = (
        "nurse.v2.sre-decision-exchange/9"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/11"] = "nurse.v2.sre-report-exchange/11"


class OpsSreComparableProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/6"] = "nurse.v2.sre-profile/6"
    investigation_profile_id: Literal["hermes-sre-tools/9"] = "hermes-sre-tools/9"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/10"] = (
        "nurse.v2.sre-decision-exchange/10"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/12"] = "nurse.v2.sre-report-exchange/12"


class OpsSreNavigationProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/7"] = "nurse.v2.sre-profile/7"
    investigation_profile_id: Literal["hermes-sre-tools/10"] = "hermes-sre-tools/10"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/11"] = (
        "nurse.v2.sre-decision-exchange/11"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/13"] = "nurse.v2.sre-report-exchange/13"


class OpsSreWorkbenchProfile(OpsSreProfileBase):
    schema_version: Literal["nurse.v2.sre-profile/8"] = "nurse.v2.sre-profile/8"
    investigation_profile_id: Literal["hermes-sre-tools/11"] = "hermes-sre-tools/11"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/12"] = (
        "nurse.v2.sre-decision-exchange/12"
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/14"] = "nurse.v2.sre-report-exchange/14"


class OpsSreUsedSourceProfile(OpsSreWorkbenchProfile):
    # Pydantic enforces each new identity at runtime; only the inherited static
    # Literal assignments are incompatible with the preceding profile version.
    schema_version: Literal["nurse.v2.sre-profile/9"] = "nurse.v2.sre-profile/9"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/2"] = "v2-ops-sre/2"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/12"] = "hermes-sre-tools/12"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/13"] = (
        "nurse.v2.sre-decision-exchange/13"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/15"] = "nurse.v2.sre-report-exchange/15"  # type: ignore[assignment]


class OpsSreReceiptReuseProfile(OpsSreUsedSourceProfile):
    schema_version: Literal["nurse.v2.sre-profile/10"] = "nurse.v2.sre-profile/10"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/3"] = "v2-ops-sre/3"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/13"] = "hermes-sre-tools/13"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/14"] = (
        "nurse.v2.sre-decision-exchange/14"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/16"] = "nurse.v2.sre-report-exchange/16"  # type: ignore[assignment]


class OpsSreBalancedProfile(OpsSreReceiptReuseProfile):
    schema_version: Literal["nurse.v2.sre-profile/11"] = "nurse.v2.sre-profile/11"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/4"] = "v2-ops-sre/4"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/14"] = "hermes-sre-tools/14"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/15"] = (
        "nurse.v2.sre-decision-exchange/15"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/17"] = "nurse.v2.sre-report-exchange/17"  # type: ignore[assignment]


class OpsSreReplanningProfile(OpsSreBalancedProfile):
    schema_version: Literal["nurse.v2.sre-profile/12"] = "nurse.v2.sre-profile/12"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/5"] = "v2-ops-sre/5"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/15"] = "hermes-sre-tools/15"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/16"] = (
        "nurse.v2.sre-decision-exchange/16"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/18"] = "nurse.v2.sre-report-exchange/18"  # type: ignore[assignment]


class OpsSreQuestionProfile(OpsSreReplanningProfile):
    schema_version: Literal["nurse.v2.sre-profile/13"] = "nurse.v2.sre-profile/13"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/6"] = "v2-ops-sre/6"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/16"] = "hermes-sre-tools/16"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/17"] = (
        "nurse.v2.sre-decision-exchange/17"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/19"] = "nurse.v2.sre-report-exchange/19"  # type: ignore[assignment]


class OpsSreActionProfile(OpsSreQuestionProfile):
    schema_version: Literal["nurse.v2.sre-profile/14"] = "nurse.v2.sre-profile/14"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/7"] = "v2-ops-sre/7"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/17"] = "hermes-sre-tools/17"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/18"] = (
        "nurse.v2.sre-decision-exchange/18"  # type: ignore[assignment]
    )
    report_protocol: Literal["nurse.v2.sre-report-exchange/20"] = "nurse.v2.sre-report-exchange/20"  # type: ignore[assignment]


class OpsSreDirectedProfile(OpsSreProfileBase):
    """Decision-only development stage; not a completed full-method profile."""

    schema_version: Literal["nurse.v2.sre-profile/15"] = "nurse.v2.sre-profile/15"
    profile_id: Literal["v2-ops-sre/8"] = "v2-ops-sre/8"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/18"] = "hermes-sre-tools/18"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/19"] = (
        "nurse.v2.sre-decision-exchange/19"
    )
    report_protocol: Literal["disabled_until_sre_report_stage/1"] = (
        "disabled_until_sre_report_stage/1"
    )
    report_limit: Literal[0] = 0  # type: ignore[assignment]


class OpsSreEntryProfile(OpsSreProfileBase):
    """Typed selection is available only in a separately approved decision stage."""

    schema_version: Literal["nurse.v2.sre-profile/16"] = "nurse.v2.sre-profile/16"
    profile_id: Literal["v2-ops-sre/9"] = "v2-ops-sre/9"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/19"] = "hermes-sre-tools/19"
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/20"] = (
        "nurse.v2.sre-decision-exchange/20"
    )
    report_protocol: Literal["disabled_until_sre_report_stage/1"] = (
        "disabled_until_sre_report_stage/1"
    )
    report_limit: Literal[0] = 0  # type: ignore[assignment]


class OpsSreObservableProfile(OpsSreEntryProfile):
    """Typed purposes cannot enter the ordinary full-case or report owner."""

    schema_version: Literal["nurse.v2.sre-profile/17"] = "nurse.v2.sre-profile/17"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/10"] = "v2-ops-sre/10"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/20"] = "hermes-sre-tools/20"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/21"] = (
        "nurse.v2.sre-decision-exchange/21"  # type: ignore[assignment]
    )


class OpsSreSingleOwnerProfile(OpsSreEntryProfile):
    """Single lifecycle writer; bounded decision stage only, never report admission."""

    schema_version: Literal["nurse.v2.sre-profile/18"] = "nurse.v2.sre-profile/18"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/11"] = "v2-ops-sre/11"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/21"] = "hermes-sre-tools/21"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/22"] = (
        "nurse.v2.sre-decision-exchange/22"  # type: ignore[assignment]
    )


class OpsSreTypedPurposeProfile(OpsSreEntryProfile):
    """Typed immediate purpose, no second lifecycle or general case entry."""

    schema_version: Literal["nurse.v2.sre-profile/19"] = "nurse.v2.sre-profile/19"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/12"] = "v2-ops-sre/12"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/22"] = "hermes-sre-tools/22"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/23"] = (
        "nurse.v2.sre-decision-exchange/23"  # type: ignore[assignment]
    )


class OpsSreAtomicSelectorProfile(OpsSreEntryProfile):
    """Atomic native selection; no ordinary case or report admission."""

    schema_version: Literal["nurse.v2.sre-profile/20"] = "nurse.v2.sre-profile/20"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/13"] = "v2-ops-sre/13"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/23"] = "hermes-sre-tools/23"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/24"] = (
        "nurse.v2.sre-decision-exchange/24"  # type: ignore[assignment]
    )


class OpsSreFeedbackFactProfile(OpsSreAtomicSelectorProfile):
    """Explicit source-faithful presentation, with unchanged atomic action identity."""

    schema_version: Literal["nurse.v2.sre-profile/21"] = "nurse.v2.sre-profile/21"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/14"] = "v2-ops-sre/14"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/25"] = (
        "nurse.v2.sre-decision-exchange/25"  # type: ignore[assignment]
    )


class OpsSreFocusedProfile(OpsSreAtomicSelectorProfile):
    """Bounded focused evidence navigation with separately versioned supply contracts."""

    schema_version: Literal["nurse.v2.sre-profile/22"] = "nurse.v2.sre-profile/22"  # type: ignore[assignment]
    profile_id: Literal["v2-ops-sre/15"] = "v2-ops-sre/15"  # type: ignore[assignment]
    investigation_profile_id: Literal["hermes-sre-tools/25"] = "hermes-sre-tools/25"  # type: ignore[assignment]
    decision_protocol: Literal["nurse.v2.sre-decision-exchange/26"] = (
        "nurse.v2.sre-decision-exchange/26"  # type: ignore[assignment]
    )


SreProfileId = Literal[
    "hermes-sre-tools/4",
    "hermes-sre-tools/5",
    "hermes-sre-tools/6",
    "hermes-sre-tools/7",
    "hermes-sre-tools/8",
    "hermes-sre-tools/9",
    "hermes-sre-tools/10",
    "hermes-sre-tools/11",
    "hermes-sre-tools/12",
    "hermes-sre-tools/13",
    "hermes-sre-tools/14",
    "hermes-sre-tools/15",
    "hermes-sre-tools/16",
    "hermes-sre-tools/17",
    "hermes-sre-tools/18",
    "hermes-sre-tools/19",
    "hermes-sre-tools/20",
    "hermes-sre-tools/21",
    "hermes-sre-tools/22",
    "hermes-sre-tools/23",
    "hermes-sre-tools/25",
]


def resolve_sre_profile(
    *, sre: bool, working: bool, profile: SreProfileId | None
) -> SreProfileId | None:
    """New callers select an explicit identity; old boolean entry points keep meaning."""
    if working and not sre:
        raise ValueError("working policy requires SRE")
    if profile is not None:
        if (
            not sre
            or profile
            not in {
                "hermes-sre-tools/4",
                "hermes-sre-tools/5",
                "hermes-sre-tools/6",
                "hermes-sre-tools/7",
                "hermes-sre-tools/8",
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
                "hermes-sre-tools/18",
                "hermes-sre-tools/19",
                "hermes-sre-tools/20",
                "hermes-sre-tools/21",
                "hermes-sre-tools/22",
                "hermes-sre-tools/23",
                "hermes-sre-tools/25",
            }
            or (working and profile != "hermes-sre-tools/5")
        ):
            raise ValueError("explicit SRE profile conflicts with legacy policy")
        return profile
    return "hermes-sre-tools/5" if working else "hermes-sre-tools/4" if sre else None


class OpsCoreIdentity(StrictFrozenModel):
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: Nonempty
    case_snapshot_id: SHA256
    core_method_id: Literal["normal-b0-ops-core/1"] = "normal-b0-ops-core/1"
    configuration_sha256: SHA256
    environment_sha256: SHA256


def snapshot_id(evidence: NormalEvidence) -> str:
    """Identify admitted frozen inputs, excluding any execution or investigation."""
    return sha256_hex(
        canonical_json_bytes(
            {
                name: getattr(evidence, name)
                for name in (
                    "dataset",
                    "opaque_case_id",
                    "source_descriptor_sha256",
                    "source_files_sha256",
                    "candidate_freeze_sha256",
                    "catalog_sha256",
                    "graph_manifest_sha256",
                )
            }
        )
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_object(value: str) -> str:
    if len(value.encode("utf-8")) > 1024 * 1024:
        raise ValueError("node JSON exceeds bounded projection capacity")
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object)
        if type(parsed) is not dict:
            raise ValueError("node JSON must be an object")
        return canonical_json_bytes(parsed).decode("utf-8")
    except (RecursionError, OverflowError) as exc:
        raise ValueError("node JSON exceeds nesting capacity") from exc


class OpsNodeEvidence(StrictFrozenModel):
    schema_version: Literal["nurse.v2.ops-node-evidence/1"] = "nurse.v2.ops-node-evidence/1"
    case_snapshot_id: SHA256
    node: NodeKind
    tool_version: Nonempty
    parameters_json: str
    evidence_ids: tuple[Nonempty, ...]
    source_refs: tuple[Nonempty, ...]
    unit: Nonempty | None
    window_ns: tuple[int, int] | None
    coverage: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    availability: Literal["available", "partial", "unavailable"]
    truth_value: TruthValue
    limitations: tuple[Nonempty, ...]
    payload_json: str

    @field_validator("parameters_json", "payload_json")
    @classmethod
    def canonical_objects(cls, value: str) -> str:
        return _canonical_object(value)

    @model_validator(mode="after")
    def unknown_is_not_normal(self) -> Self:
        if self.window_ns is not None and self.window_ns[0] >= self.window_ns[1]:
            raise ValueError("node window must have positive width")
        if self.availability == "unavailable" and (
            self.truth_value != "unknown" or not self.limitations
        ):
            raise ValueError("unavailable node requires unknown and explicit limitations")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence IDs")
        return self


class OpsInvestigationAction(StrictFrozenModel):
    action_index: int = Field(ge=1, le=8, strict=True)
    skill_id: Nonempty
    tool: Nonempty
    parameters_json: str
    question: Nonempty

    @field_validator("parameters_json")
    @classmethod
    def canonical_parameters(cls, value: str) -> str:
        return _canonical_object(value)


class OpsEmpiricalInvestigationAction(OpsInvestigationAction):
    """No prose quota; enclosing wire retains independent resource protection."""

    question: str = Field(min_length=1, strict=True)


class OpsInvestigationEvidence(StrictFrozenModel):
    action_index: int = Field(ge=1, le=8, strict=True)
    role: Literal["supplementary"] = "supplementary"
    evidence: OpsNodeEvidence
    origin: Literal["measured", "core_reuse", "read_card", "rejected"]
    reused_core_evidence_ids: tuple[Nonempty, ...]

    @model_validator(mode="after")
    def reuse_requires_references(self) -> Self:
        if (self.origin == "core_reuse") != bool(self.reused_core_evidence_ids):
            raise ValueError("core reuse requires explicit references and no other origin")
        return self


class OpsReportClaim(StrictFrozenModel):
    """A proposed claim, not an accepted fact merely because it cites an ID."""

    text: Nonempty
    evidence_ids: tuple[Nonempty, ...] = Field(min_length=1)
    candidate_service_key: Nonempty | None
    assertion: Literal["observed", "supported", "contradicted", "unknown", "hypothesis"]


class OpsReportDraft(StrictFrozenModel):
    core_result_digest: SHA256
    formal_grounds: tuple[OpsReportClaim, ...]
    supplementary_findings: tuple[OpsReportClaim, ...]
    limitations: tuple[Nonempty, ...]
    conditional_next_steps: tuple[Nonempty, ...]


class OpsChannelStatus(StrictFrozenModel):
    core: Literal["pending", "running", "completed", "failed", "interrupted"]
    investigation: Literal[
        "pending",
        "running",
        "completed",
        "partial",
        "failed",
        "unavailable",
        "disabled",
    ]
    report: Literal["pending", "running", "completed", "partial", "unavailable", "disabled"]
