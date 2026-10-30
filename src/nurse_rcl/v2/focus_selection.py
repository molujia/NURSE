"""Neutral pre-semantic E0 scoring and deterministic focus-set selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult, PrimarySignalFamily
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    CandidateUniverse,
    FocusDecision,
    Observation,
    SemanticFocusSet,
    build_focus_set,
)

NEUTRAL_EVIDENCE_SCORE_SCHEMA_VERSION: Literal["nurse.v2.neutral-evidence-score/1"] = (
    "nurse.v2.neutral-evidence-score/1"
)
REQUIRED_NEIGHBOR_SCHEMA_VERSION: Literal["nurse.v2.required-one-hop-neighbor/1"] = (
    "nurse.v2.required-one-hop-neighbor/1"
)
FOCUS_EXPANSION_DECISION_SCHEMA_VERSION: Literal["nurse.v2.focus-expansion-decision/1"] = (
    "nurse.v2.focus-expansion-decision/1"
)
FOCUS_SELECTION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.focus-selection-result/1"] = (
    "nurse.v2.focus-selection-result/1"
)
FOCUS_POLICY_VERSION: Literal["nurse-v2-neutral-focus-policy/1"] = "nurse-v2-neutral-focus-policy/1"

ExpansionReason: TypeAlias = Literal[
    "lifecycle",
    "call_failure",
    "log_error",
    "required_one_hop_neighbor",
]
ExpansionOutcome: TypeAlias = Literal[
    "already_selected_base",
    "selected_expansion",
    "q_max_truncation",
]

_NEUTRAL_ROLE_ADJUSTMENT = 0.5
_DEFAULT_FAMILY_GAMMA = 0.25
_EXPANSION_REASON_PRIORITY: dict[ExpansionReason, int] = {
    "lifecycle": 0,
    "call_failure": 1,
    "log_error": 2,
    "required_one_hop_neighbor": 3,
}


class NeutralFamilyScore(StrictFrozenModel):
    """Strongest pre-semantic evidence retained for one signal family."""

    signal_family: PrimarySignalFamily
    event_group_ids: tuple[str, ...] = Field(min_length=1)
    raw_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    neutral_strength: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_neutral_role_adjustment(self) -> Self:
        if self.event_group_ids != tuple(sorted(set(self.event_group_ids))):
            raise ValueError("neutral family event groups must be sorted and unique")
        if self.neutral_strength != self.raw_strength * _NEUTRAL_ROLE_ADJUSTMENT:
            raise ValueError("neutral family strength must disable semantic role effects")
        return self


class NeutralEvidenceScore(StrictFrozenModel):
    """Auditable E0 for one candidate before semantic support or refutation."""

    schema_version: Literal["nurse.v2.neutral-evidence-score/1"] = (
        NEUTRAL_EVIDENCE_SCORE_SCHEMA_VERSION
    )
    service_entity_key: str = Field(min_length=1)
    family_scores: tuple[NeutralFamilyScore, ...]
    e0: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_families(self) -> Self:
        expected = tuple(sorted(self.family_scores, key=lambda item: item.signal_family))
        if self.family_scores != expected:
            raise ValueError("neutral family scores must be canonically ordered")
        if len({item.signal_family for item in self.family_scores}) != len(self.family_scores):
            raise ValueError("neutral evidence cannot repeat a signal family")
        return self

    def verify(self, *, family_gamma: float = _DEFAULT_FAMILY_GAMMA) -> bool:
        try:
            validated = NeutralEvidenceScore.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            expected_e0 = _aggregate_top_two(
                tuple(
                    (item.signal_family, item.neutral_strength) for item in validated.family_scores
                ),
                family_gamma=family_gamma,
            )
            unsigned = validated.model_dump(mode="json", exclude={"score_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.e0 == expected_e0
            and validated.score_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        )


class RequiredOneHopNeighbor(StrictFrozenModel):
    """Deterministic graph proof that one neighbor needs detailed context."""

    schema_version: Literal["nurse.v2.required-one-hop-neighbor/1"] = (
        REQUIRED_NEIGHBOR_SCHEMA_VERSION
    )
    neighbor_expansion_id: str = Field(pattern=r"^nbx_[0-9a-f]{24}$")
    anchor_service_key: str = Field(min_length=1)
    neighbor_service_key: str = Field(min_length=1)
    relation_id: str = Field(min_length=1)
    path_quality: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    expansion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_distinct_canonical_relation(self) -> Self:
        if self.anchor_service_key == self.neighbor_service_key:
            raise ValueError("one-hop neighbor endpoints must be distinct")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("neighbor provenance IDs must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            validated = RequiredOneHopNeighbor.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(
                mode="json",
                exclude={"neighbor_expansion_id", "expansion_sha256"},
            )
            digest = sha256_hex(canonical_json_bytes(unsigned))
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.expansion_sha256 == digest
            and validated.neighbor_expansion_id == f"nbx_{digest[:24]}"
        )


def build_required_one_hop_neighbor(
    *,
    anchor_service_key: str,
    neighbor_service_key: str,
    relation_id: str,
    path_quality: float,
    provenance_ids: Sequence[str],
) -> RequiredOneHopNeighbor:
    """Build one content-addressed neighbor expansion proof."""

    unsigned: dict[str, object] = {
        "schema_version": REQUIRED_NEIGHBOR_SCHEMA_VERSION,
        "anchor_service_key": anchor_service_key,
        "neighbor_service_key": neighbor_service_key,
        "relation_id": relation_id,
        "path_quality": path_quality,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    result = RequiredOneHopNeighbor.model_validate(
        {
            **unsigned,
            "neighbor_expansion_id": f"nbx_{digest[:24]}",
            "expansion_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("required neighbor verification failed")
    return result


class FocusExpansionDecision(StrictFrozenModel):
    """Deduplicated evidence and outcome for one expansion-affected candidate."""

    schema_version: Literal["nurse.v2.focus-expansion-decision/1"] = (
        FOCUS_EXPANSION_DECISION_SCHEMA_VERSION
    )
    service_entity_key: str = Field(min_length=1)
    selection_reasons: tuple[ExpansionReason, ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    strongest_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    outcome: ExpansionOutcome
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_expansion_evidence(self) -> Self:
        if self.selection_reasons != tuple(sorted(set(self.selection_reasons))):
            raise ValueError("expansion reasons must be sorted and unique")
        if self.source_ids != tuple(sorted(set(self.source_ids))):
            raise ValueError("expansion source IDs must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            validated = FocusExpansionDecision.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"decision_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self and validated.decision_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


class FocusSelectionResult(StrictFrozenModel):
    """Complete E0, expansion, truncation, and frozen-Q audit."""

    schema_version: Literal["nurse.v2.focus-selection-result/1"] = (
        FOCUS_SELECTION_RESULT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-neutral-focus-policy/1"] = FOCUS_POLICY_VERSION
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    neutral_role_adjustment: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    family_gamma: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    neutral_scores: tuple[NeutralEvidenceScore, ...] = Field(min_length=1)
    expansion_decisions: tuple[FocusExpansionDecision, ...]
    focus_set: SemanticFocusSet
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = FocusSelectionResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or validated.neutral_role_adjustment != 0.5:
                return False
            if validated.neutral_scores != tuple(
                sorted(
                    validated.neutral_scores,
                    key=lambda item: item.service_entity_key,
                )
            ) or any(
                not item.verify(family_gamma=validated.family_gamma)
                for item in validated.neutral_scores
            ):
                return False
            if validated.expansion_decisions != tuple(
                sorted(
                    validated.expansion_decisions,
                    key=lambda item: item.service_entity_key,
                )
            ) or any(not item.verify() for item in validated.expansion_decisions):
                return False
            if len({item.service_entity_key for item in validated.expansion_decisions}) != len(
                validated.expansion_decisions
            ):
                return False
            if not validated.focus_set.verify():
                return False
            candidate_keys = tuple(item.service_entity_key for item in validated.neutral_scores)
            decision_keys = tuple(item.service_entity_key for item in validated.focus_set.decisions)
            if candidate_keys != decision_keys:
                return False
            focus_by_service = {
                item.service_entity_key: item for item in validated.focus_set.decisions
            }
            for expansion in validated.expansion_decisions:
                focus = focus_by_service[expansion.service_entity_key]
                expected = {
                    "already_selected_base": focus.selected and focus.selection_phase == "base",
                    "selected_expansion": focus.selected
                    and focus.selection_phase == "expanded"
                    and focus.selection_reasons == expansion.selection_reasons,
                    "q_max_truncation": not focus.selected
                    and focus.exclusion_reason == "q_max_truncation",
                }[expansion.outcome]
                if not expected:
                    return False
            unsigned = validated.model_dump(mode="json", exclude={"result_sha256"})
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return validated.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))


@dataclass
class _ExpansionAccumulator:
    strength_by_reason: dict[ExpansionReason, float] = field(default_factory=dict)
    source_ids: set[str] = field(default_factory=set)

    def add(
        self,
        *,
        reason: ExpansionReason,
        strength: float,
        source_ids: Sequence[str],
    ) -> None:
        self.strength_by_reason[reason] = max(
            strength,
            self.strength_by_reason.get(reason, 0.0),
        )
        self.source_ids.update(source_ids)

    @property
    def strongest_strength(self) -> float:
        return max(self.strength_by_reason.values())

    @property
    def reasons(self) -> tuple[ExpansionReason, ...]:
        return tuple(sorted(self.strength_by_reason))


def _aggregate_top_two(
    family_values: Sequence[tuple[str, float]],
    *,
    family_gamma: float,
) -> float:
    strongest = sorted(family_values, key=lambda item: (-item[1], item[0]))[:2]
    if not strongest:
        return 0.0
    first = strongest[0][1]
    if len(strongest) == 1:
        return first
    second = strongest[1][1]
    return first + family_gamma * (1.0 - first) * second


def _build_neutral_score(
    *,
    candidate: CandidateEntry,
    observations: dict[str, Observation],
    event_groups: dict[str, EventGroup],
    family_gamma: float,
) -> NeutralEvidenceScore:
    candidate_observation_ids = set(candidate.observation_ids)
    family_groups: dict[PrimarySignalFamily, list[tuple[str, float]]] = {}
    for event_group_id in candidate.event_group_ids:
        group = event_groups[event_group_id]
        mapped_observation_ids = {
            assignment.observation_id
            for assignment in group.member_family_assignments
            if assignment.assignment_basis == "exact_frozen_mapping"
            and assignment.primary_signal_family == group.primary_signal_family
        }
        strengths = tuple(
            observation.coverage_quality * observation.anomaly_strength
            for observation_id in group.observation_ids
            if observation_id in candidate_observation_ids
            and observation_id in mapped_observation_ids
            for observation in (observations[observation_id],)
            if observation.evidence_state == "abnormal"
            and observation.formal_scoring_eligible
            and observation.anomaly_strength is not None
        )
        if strengths and max(strengths) > 0.0:
            family_groups.setdefault(group.primary_signal_family, []).append(
                (event_group_id, max(strengths))
            )
    family_scores = tuple(
        NeutralFamilyScore(
            signal_family=signal_family,
            event_group_ids=tuple(sorted(item[0] for item in values)),
            raw_strength=max(item[1] for item in values),
            neutral_strength=(max(item[1] for item in values) * _NEUTRAL_ROLE_ADJUSTMENT),
        )
        for signal_family, values in sorted(family_groups.items())
    )
    e0 = _aggregate_top_two(
        tuple((item.signal_family, item.neutral_strength) for item in family_scores),
        family_gamma=family_gamma,
    )
    unsigned: dict[str, object] = {
        "schema_version": NEUTRAL_EVIDENCE_SCORE_SCHEMA_VERSION,
        "service_entity_key": candidate.service_entity_key,
        "family_scores": [item.model_dump(mode="json") for item in family_scores],
        "e0": e0,
    }
    return NeutralEvidenceScore.model_validate(
        {
            **unsigned,
            "score_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )


def _validate_candidate_evidence_references(
    *,
    candidate_universe: CandidateUniverse,
    event_grouping: EventGroupingResult,
) -> None:
    observations = {item.observation_id for item in event_grouping.observations}
    event_groups = {item.event_group_id: item for item in event_grouping.event_groups}
    for candidate in candidate_universe.candidates:
        candidate_observations = set(candidate.observation_ids)
        if not candidate_observations <= observations:
            raise ValueError("candidate references an unknown grouped observation")
        if not set(candidate.event_group_ids) <= set(event_groups):
            raise ValueError("candidate references an unknown event group")
        groups_for_candidate = tuple(
            event_groups[event_group_id] for event_group_id in candidate.event_group_ids
        )
        grouped_candidate_observations = {
            observation_id
            for group in groups_for_candidate
            for observation_id in group.observation_ids
            if observation_id in candidate_observations
        }
        if grouped_candidate_observations != candidate_observations:
            raise ValueError("candidate observations and event groups are inconsistent")
        if any(
            not (set(group.observation_ids) & candidate_observations)
            for group in groups_for_candidate
        ):
            raise ValueError("candidate event group has no attributed observation")


def _automatic_expansion_evidence(
    *,
    neutral_scores: Sequence[NeutralEvidenceScore],
    candidate_universe: CandidateUniverse,
    event_grouping: EventGroupingResult,
) -> dict[str, _ExpansionAccumulator]:
    observations = {item.observation_id: item for item in event_grouping.observations}
    event_groups = {item.event_group_id: item for item in event_grouping.event_groups}
    candidates = {item.service_entity_key: item for item in candidate_universe.candidates}
    expansion: dict[str, _ExpansionAccumulator] = {}
    for score in neutral_scores:
        candidate = candidates[score.service_entity_key]
        candidate_observations = set(candidate.observation_ids)
        for family in score.family_scores:
            reason: ExpansionReason | None = None
            if family.signal_family == "lifecycle":
                reason = "lifecycle"
            elif family.signal_family == "dependency_failure":
                reason = "call_failure"
            elif family.signal_family == "error" and any(
                observations[observation_id].modality == "logs"
                for event_group_id in family.event_group_ids
                for observation_id in event_groups[event_group_id].observation_ids
                if observation_id in candidate_observations
            ):
                reason = "log_error"
            if reason is not None:
                expansion.setdefault(
                    score.service_entity_key,
                    _ExpansionAccumulator(),
                ).add(
                    reason=reason,
                    strength=family.raw_strength,
                    source_ids=family.event_group_ids,
                )
    return expansion


def _build_expansion_decision(
    *,
    service_entity_key: str,
    evidence: _ExpansionAccumulator,
    outcome: ExpansionOutcome,
) -> FocusExpansionDecision:
    unsigned: dict[str, object] = {
        "schema_version": FOCUS_EXPANSION_DECISION_SCHEMA_VERSION,
        "service_entity_key": service_entity_key,
        "selection_reasons": evidence.reasons,
        "source_ids": tuple(sorted(evidence.source_ids)),
        "strongest_strength": evidence.strongest_strength,
        "outcome": outcome,
    }
    return FocusExpansionDecision.model_validate(
        {
            **unsigned,
            "decision_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )


def build_neutral_focus_selection(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    event_grouping: EventGroupingResult,
    required_one_hop_neighbors: Sequence[RequiredOneHopNeighbor] = (),
    q_base: int = 8,
    q_max: int = 12,
    family_gamma: float = _DEFAULT_FAMILY_GAMMA,
) -> FocusSelectionResult:
    """Compute E0, freeze Q, and retain every expansion/truncation reason."""

    require_frozen_candidate_universe(
        frozen=frozen_candidates,
        candidate_universe=candidate_universe,
    )
    if not event_grouping.verify():
        raise ValueError("event grouping failed verification")
    if event_grouping.opaque_case_id != candidate_universe.opaque_case_id:
        raise ValueError("event grouping belongs to a different candidate case")
    if q_base < 0:
        raise ValueError("q_base cannot be negative")
    if q_max < 1:
        raise ValueError("q_max must be positive")
    if q_base > q_max:
        raise ValueError("q_base cannot exceed q_max")
    if not 0.0 <= family_gamma <= 1.0:
        raise ValueError("family gamma must be within zero and one")
    _validate_candidate_evidence_references(
        candidate_universe=candidate_universe,
        event_grouping=event_grouping,
    )

    observations = {item.observation_id: item for item in event_grouping.observations}
    event_groups = {item.event_group_id: item for item in event_grouping.event_groups}
    neutral_scores = tuple(
        _build_neutral_score(
            candidate=candidate,
            observations=observations,
            event_groups=event_groups,
            family_gamma=family_gamma,
        )
        for candidate in candidate_universe.candidates
    )
    base_order = tuple(
        sorted(
            neutral_scores,
            key=lambda item: (-item.e0, item.service_entity_key),
        )
    )
    base_keys = {item.service_entity_key for item in base_order[: min(q_base, len(base_order))]}
    expansion = _automatic_expansion_evidence(
        neutral_scores=neutral_scores,
        candidate_universe=candidate_universe,
        event_grouping=event_grouping,
    )
    candidate_keys = set(frozen_candidates.candidate_service_keys)
    neighbors_by_id: dict[str, RequiredOneHopNeighbor] = {}
    for neighbor in required_one_hop_neighbors:
        if not neighbor.verify():
            raise ValueError("required one-hop neighbor failed verification")
        if (
            neighbor.anchor_service_key not in candidate_keys
            or neighbor.neighbor_service_key not in candidate_keys
        ):
            raise ValueError("neighbor endpoint is outside frozen candidates")
        if neighbor.anchor_service_key not in base_keys:
            raise ValueError("neighbor anchor is outside neutral base focus")
        neighbors_by_id.setdefault(neighbor.neighbor_expansion_id, neighbor)
    for neighbor in sorted(
        neighbors_by_id.values(),
        key=lambda item: item.neighbor_expansion_id,
    ):
        expansion.setdefault(
            neighbor.neighbor_service_key,
            _ExpansionAccumulator(),
        ).add(
            reason="required_one_hop_neighbor",
            strength=neighbor.path_quality,
            source_ids=(neighbor.neighbor_expansion_id, neighbor.relation_id),
        )

    expansion_order = tuple(
        sorted(
            (
                (service_entity_key, evidence)
                for service_entity_key, evidence in expansion.items()
                if service_entity_key not in base_keys
            ),
            key=lambda item: (
                -item[1].strongest_strength,
                min(_EXPANSION_REASON_PRIORITY[reason] for reason in item[1].strength_by_reason),
                item[0],
            ),
        )
    )
    remaining_capacity = max(0, q_max - len(base_keys))
    expanded_keys = {
        service_entity_key for service_entity_key, _ in expansion_order[:remaining_capacity]
    }
    focus_decisions = []
    expansion_decisions = []
    for candidate in candidate_universe.candidates:
        service_entity_key = candidate.service_entity_key
        evidence = expansion.get(service_entity_key)
        if service_entity_key in base_keys:
            focus_decisions.append(
                FocusDecision(
                    service_entity_key=service_entity_key,
                    selected=True,
                    selection_phase="base",
                    selection_reasons=("neutral_evidence",),
                    exclusion_reason=None,
                )
            )
            if evidence is not None:
                expansion_decisions.append(
                    _build_expansion_decision(
                        service_entity_key=service_entity_key,
                        evidence=evidence,
                        outcome="already_selected_base",
                    )
                )
        elif service_entity_key in expanded_keys and evidence is not None:
            focus_decisions.append(
                FocusDecision(
                    service_entity_key=service_entity_key,
                    selected=True,
                    selection_phase="expanded",
                    selection_reasons=evidence.reasons,
                    exclusion_reason=None,
                )
            )
            expansion_decisions.append(
                _build_expansion_decision(
                    service_entity_key=service_entity_key,
                    evidence=evidence,
                    outcome="selected_expansion",
                )
            )
        elif evidence is not None:
            focus_decisions.append(
                FocusDecision(
                    service_entity_key=service_entity_key,
                    selected=False,
                    selection_phase="none",
                    selection_reasons=(),
                    exclusion_reason="q_max_truncation",
                )
            )
            expansion_decisions.append(
                _build_expansion_decision(
                    service_entity_key=service_entity_key,
                    evidence=evidence,
                    outcome="q_max_truncation",
                )
            )
        else:
            focus_decisions.append(
                FocusDecision(
                    service_entity_key=service_entity_key,
                    selected=False,
                    selection_phase="none",
                    selection_reasons=(),
                    exclusion_reason="not_selected_by_policy",
                )
            )
    focus_set = build_focus_set(
        candidate_universe=candidate_universe,
        q_base=q_base,
        q_max=q_max,
        decisions=focus_decisions,
    )
    ordered_expansion_decisions = tuple(
        sorted(expansion_decisions, key=lambda item: item.service_entity_key)
    )
    unsigned: dict[str, object] = {
        "schema_version": FOCUS_SELECTION_RESULT_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "policy_version": FOCUS_POLICY_VERSION,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "event_grouping_result_sha256": event_grouping.result_sha256,
        "neutral_role_adjustment": _NEUTRAL_ROLE_ADJUSTMENT,
        "family_gamma": family_gamma,
        "neutral_scores": [item.model_dump(mode="json") for item in neutral_scores],
        "expansion_decisions": [
            item.model_dump(mode="json") for item in ordered_expansion_decisions
        ],
        "focus_set": focus_set.model_dump(mode="json"),
    }
    result = FocusSelectionResult.model_validate(
        {
            **unsigned,
            "result_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("focus selection result verification failed")
    return result
