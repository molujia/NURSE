"""Compact C directory and decision-time-bounded detailed Q context."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.event_grouping import EventGroup, EventGroupingResult
from nurse_rcl.v2.focus_selection import FocusSelectionResult
from nurse_rcl.v2.observation_schemas import (
    CandidateEntry,
    CandidateEvidenceState,
    CandidateUniverse,
    EntityBinding,
    EvidenceCard,
    FocusSelectionReason,
    Observation,
    build_evidence_card,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

COMPACT_CANDIDATE_DIRECTORY_SCHEMA_VERSION: Literal["nurse.v2.compact-candidate-directory/1"] = (
    "nurse.v2.compact-candidate-directory/1"
)
DETAILED_EVIDENCE_CARD_SCHEMA_VERSION: Literal["nurse.v2.detailed-evidence-card/1"] = (
    "nurse.v2.detailed-evidence-card/1"
)
SEMANTIC_CONTEXT_BUNDLE_SCHEMA_VERSION: Literal["nurse.v2.semantic-context-bundle/1"] = (
    "nurse.v2.semantic-context-bundle/1"
)

FocusPhase: TypeAlias = Literal["base", "expanded", "none"]
FocusExclusionReason: TypeAlias = Literal[
    "not_selected_by_policy",
    "q_max_truncation",
]


class CompactCandidateDirectoryEntry(StrictFrozenModel):
    """One compact C member without detailed telemetry references."""

    service_entity_key: str = Field(min_length=1)
    evidence_state: CandidateEvidenceState
    neutral_e0: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    in_initial_focus: bool
    focus_selection_phase: FocusPhase
    focus_selection_reasons: tuple[FocusSelectionReason, ...]
    focus_exclusion_reason: FocusExclusionReason | None
    has_initial_detailed_card: bool

    @model_validator(mode="after")
    def require_consistent_focus_projection(self) -> Self:
        if self.focus_selection_reasons != tuple(sorted(set(self.focus_selection_reasons))):
            raise ValueError("compact focus reasons must be sorted and unique")
        if self.has_initial_detailed_card != self.in_initial_focus:
            raise ValueError("initial detailed-card availability must match initial Q")
        if self.in_initial_focus:
            if (
                self.focus_selection_phase == "none"
                or not self.focus_selection_reasons
                or self.focus_exclusion_reason is not None
            ):
                raise ValueError("focused directory entry requires phase and reasons")
            if self.focus_selection_phase == "base" and self.focus_selection_reasons != (
                "neutral_evidence",
            ):
                raise ValueError("base directory entry must use neutral evidence only")
            if (
                self.focus_selection_phase == "expanded"
                and "neutral_evidence" in self.focus_selection_reasons
            ):
                raise ValueError("expanded directory entry cannot claim neutral base selection")
        elif (
            self.focus_selection_phase != "none"
            or self.focus_selection_reasons
            or self.focus_exclusion_reason is None
        ):
            raise ValueError("unfocused directory entry requires one exclusion reason")
        return self


class CompactCandidateDirectory(StrictFrozenModel):
    """Content-addressed compact projection of every member of frozen C."""

    schema_version: Literal["nurse.v2.compact-candidate-directory/1"] = (
        COMPACT_CANDIDATE_DIRECTORY_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    focus_selection_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_time_unix_ns: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    selected_count: int = Field(ge=0)
    entries: tuple[CompactCandidateDirectoryEntry, ...] = Field(min_length=1)
    directory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = CompactCandidateDirectory.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self:
                return False
            expected = tuple(sorted(validated.entries, key=lambda item: item.service_entity_key))
            if validated.entries != expected:
                return False
            if len({item.service_entity_key for item in validated.entries}) != len(
                validated.entries
            ):
                return False
            if validated.candidate_count != len(validated.entries):
                return False
            if validated.selected_count != sum(item.in_initial_focus for item in validated.entries):
                return False
            unsigned = validated.model_dump(mode="json", exclude={"directory_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated.directory_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class DetailedEvidenceCard(StrictFrozenModel):
    """Typed observation, event, and binding payload for one allowed C member."""

    schema_version: Literal["nurse.v2.detailed-evidence-card/1"] = (
        DETAILED_EVIDENCE_CARD_SCHEMA_VERSION
    )
    card: EvidenceCard
    observations: tuple[Observation, ...]
    event_groups: tuple[EventGroup, ...]
    entity_bindings: tuple[EntityBinding, ...]
    detailed_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = DetailedEvidenceCard.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated.card.verify():
                return False
            if validated.observations != tuple(
                sorted(validated.observations, key=lambda item: item.observation_id)
            ):
                return False
            if validated.event_groups != tuple(
                sorted(validated.event_groups, key=lambda item: item.event_group_id)
            ):
                return False
            if validated.entity_bindings != tuple(
                sorted(validated.entity_bindings, key=lambda item: item.binding_id)
            ):
                return False
            observation_ids = tuple(item.observation_id for item in validated.observations)
            event_group_ids = tuple(item.event_group_id for item in validated.event_groups)
            binding_ids = tuple(item.binding_id for item in validated.entity_bindings)
            if len(observation_ids) != len(set(observation_ids)):
                return False
            if len(event_group_ids) != len(set(event_group_ids)):
                return False
            if len(binding_ids) != len(set(binding_ids)):
                return False
            if validated.card.observation_ids != observation_ids:
                return False
            if validated.card.event_group_ids != event_group_ids:
                return False
            if validated.card.entity_binding_ids != binding_ids:
                return False
            if validated.card.quality_flags != tuple(
                sorted(
                    {
                        flag
                        for observation in validated.observations
                        for flag in observation.quality_flags
                    }
                )
            ):
                return False
            observation_id_set = set(observation_ids)
            binding_id_set = set(binding_ids)
            decision_time = validated.card.decision_time_unix_ns
            case_id = validated.card.opaque_case_id
            service_key = validated.card.service_entity_key
            if any(
                observation.opaque_case_id != case_id
                or observation.decision_time_unix_ns != decision_time
                or observation.interval.end_unix_ns > decision_time
                or observation.entity_binding_id not in binding_id_set
                for observation in validated.observations
            ):
                return False
            if any(
                group.opaque_case_id != case_id
                or group.interval.end_unix_ns > decision_time
                or not observation_id_set.intersection(group.observation_ids)
                for group in validated.event_groups
            ):
                return False
            for binding in validated.entity_bindings:
                if (
                    binding.opaque_case_id != case_id
                    or binding.valid_interval.end_unix_ns > decision_time
                ):
                    return False
                if binding.status == "bound":
                    if binding.service_entity_key != service_key:
                        return False
                elif binding.status == "conflicted":
                    if service_key not in binding.conflict_service_keys:
                        return False
                else:
                    return False
            binding_by_id = {binding.binding_id: binding for binding in validated.entity_bindings}
            if any(
                binding_by_id[observation.entity_binding_id].status != "bound"
                or binding_by_id[observation.entity_binding_id].service_entity_key != service_key
                for observation in validated.observations
                if observation.entity_binding_id is not None
            ):
                return False
            unsigned = validated.model_dump(
                mode="json",
                exclude={"detailed_card_sha256"},
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return validated.detailed_card_sha256 == sha256_hex(canonical_json_bytes(unsigned))


class SemanticContextBundle(StrictFrozenModel):
    """Frozen initial semantic context whose detailed payload is exactly Q."""

    schema_version: Literal["nurse.v2.semantic-context-bundle/1"] = (
        SEMANTIC_CONTEXT_BUNDLE_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    focus_selection_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_grouping_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_directory: CompactCandidateDirectory
    detailed_evidence_cards: tuple[DetailedEvidenceCard, ...]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = SemanticContextBundle.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated.candidate_directory.verify():
                return False
            if validated.candidate_directory.opaque_case_id != validated.opaque_case_id:
                return False
            if (
                validated.candidate_directory.candidate_freeze_sha256
                != validated.candidate_freeze_sha256
                or validated.candidate_directory.focus_selection_result_sha256
                != validated.focus_selection_result_sha256
            ):
                return False
            if validated.detailed_evidence_cards != tuple(
                sorted(
                    validated.detailed_evidence_cards,
                    key=lambda item: item.card.service_entity_key,
                )
            ):
                return False
            if any(not item.verify() for item in validated.detailed_evidence_cards):
                return False
            card_service_keys = tuple(
                item.card.service_entity_key for item in validated.detailed_evidence_cards
            )
            if len(card_service_keys) != len(set(card_service_keys)):
                return False
            expected_card_keys = tuple(
                item.service_entity_key
                for item in validated.candidate_directory.entries
                if item.has_initial_detailed_card
            )
            if card_service_keys != expected_card_keys:
                return False
            if any(
                item.card.opaque_case_id != validated.opaque_case_id
                or item.card.candidate_universe_sha256
                != validated.candidate_directory.candidate_universe_sha256
                or item.card.decision_time_unix_ns
                != validated.candidate_directory.decision_time_unix_ns
                or item.card.access_reason != "focus_q"
                for item in validated.detailed_evidence_cards
            ):
                return False
            unsigned = validated.model_dump(mode="json", exclude={"bundle_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated.bundle_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _binding_index(registry: TemporalEntityRegistry) -> dict[str, EntityBinding]:
    result: dict[str, EntityBinding] = {}
    for binding in (
        *registry.timeline_bindings,
        *(resolution.binding for resolution in registry.resolutions),
    ):
        previous = result.setdefault(binding.binding_id, binding)
        if previous != binding:
            raise ValueError("entity registry reuses a binding ID for different bindings")
    return result


def _binding_mentions_candidate(binding: EntityBinding, service_key: str) -> bool:
    if binding.status == "bound":
        return binding.service_entity_key == service_key
    if binding.status == "conflicted":
        return service_key in binding.conflict_service_keys
    return False


def _validate_evidence_sources(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> None:
    require_frozen_candidate_universe(
        frozen=frozen_candidates,
        candidate_universe=candidate_universe,
    )
    if not entity_registry.verify():
        raise ValueError("entity registry failed verification")
    if not event_grouping.verify():
        raise ValueError("event grouping failed verification")
    if not (
        candidate_universe.opaque_case_id
        == frozen_candidates.opaque_case_id
        == entity_registry.opaque_case_id
        == event_grouping.opaque_case_id
    ):
        raise ValueError("semantic context inputs belong to different cases")
    if entity_registry.decision_time_unix_ns != event_grouping.decision_time_unix_ns:
        raise ValueError("semantic context decision times differ")
    registry_observations = tuple(
        resolution.observation for resolution in entity_registry.resolutions
    )
    if registry_observations != event_grouping.observations:
        raise ValueError("registry and grouping observation membership differs")

    binding_by_id = _binding_index(entity_registry)
    resolution_by_observation_id = {
        item.observation.observation_id: item for item in entity_registry.resolutions
    }
    event_group_by_id = {item.event_group_id: item for item in event_grouping.event_groups}
    for candidate in candidate_universe.candidates:
        try:
            candidate_bindings = tuple(
                binding_by_id[binding_id] for binding_id in candidate.binding_ids
            )
        except KeyError as error:
            raise ValueError("candidate references an unknown entity binding") from error
        if any(
            not _binding_mentions_candidate(binding, candidate.service_entity_key)
            for binding in candidate_bindings
        ):
            raise ValueError("candidate references another service binding")
        try:
            retained_resolutions = tuple(
                resolution_by_observation_id[observation_id]
                for observation_id in candidate.observation_ids
            )
        except KeyError as error:
            raise ValueError("candidate references an unknown observation") from error
        if any(
            resolution.binding.status != "bound"
            or resolution.binding.service_entity_key != candidate.service_entity_key
            or resolution.binding.binding_id not in candidate.binding_ids
            for resolution in retained_resolutions
        ):
            raise ValueError("candidate observation is not bound to that candidate")
        try:
            candidate_groups = tuple(
                event_group_by_id[event_group_id] for event_group_id in candidate.event_group_ids
            )
        except KeyError as error:
            raise ValueError("candidate references an unknown event group") from error
        retained_observation_ids = set(candidate.observation_ids)
        if any(
            not retained_observation_ids.intersection(group.observation_ids)
            for group in candidate_groups
        ):
            raise ValueError("candidate event group lacks a candidate observation")


def _validate_context_inputs(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    focus_selection: FocusSelectionResult,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> None:
    _validate_evidence_sources(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    if not focus_selection.verify():
        raise ValueError("focus selection failed verification")
    if not (
        focus_selection.opaque_case_id
        == focus_selection.focus_set.opaque_case_id
        == candidate_universe.opaque_case_id
    ):
        raise ValueError("semantic context inputs belong to different cases")
    if focus_selection.candidate_freeze_sha256 != frozen_candidates.freeze_sha256:
        raise ValueError("focus selection does not match candidate freeze")
    if focus_selection.event_grouping_result_sha256 != event_grouping.result_sha256:
        raise ValueError("focus selection does not match event grouping")
    if focus_selection.focus_set.candidate_universe_sha256 != candidate_universe.universe_sha256:
        raise ValueError("focus set does not match candidate universe")


def _build_compact_directory(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    focus_selection: FocusSelectionResult,
    decision_time_unix_ns: int,
) -> CompactCandidateDirectory:
    candidate_by_key = {item.service_entity_key: item for item in candidate_universe.candidates}
    score_by_key = {item.service_entity_key: item for item in focus_selection.neutral_scores}
    decision_by_key = {
        item.service_entity_key: item for item in focus_selection.focus_set.decisions
    }
    expected_keys = set(frozen_candidates.candidate_service_keys)
    if set(candidate_by_key) != expected_keys:
        raise ValueError("candidate directory does not exactly cover frozen C")
    if set(score_by_key) != expected_keys or set(decision_by_key) != expected_keys:
        raise ValueError("focus projection does not exactly cover frozen C")
    entries = tuple(
        CompactCandidateDirectoryEntry(
            service_entity_key=service_key,
            evidence_state=candidate_by_key[service_key].evidence_state,
            neutral_e0=score_by_key[service_key].e0,
            in_initial_focus=decision_by_key[service_key].selected,
            focus_selection_phase=decision_by_key[service_key].selection_phase,
            focus_selection_reasons=decision_by_key[service_key].selection_reasons,
            focus_exclusion_reason=decision_by_key[service_key].exclusion_reason,
            has_initial_detailed_card=decision_by_key[service_key].selected,
        )
        for service_key in frozen_candidates.candidate_service_keys
    )
    unsigned: dict[str, object] = {
        "schema_version": COMPACT_CANDIDATE_DIRECTORY_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "focus_selection_result_sha256": focus_selection.result_sha256,
        "decision_time_unix_ns": decision_time_unix_ns,
        "candidate_count": len(entries),
        "selected_count": sum(item.in_initial_focus for item in entries),
        "entries": [item.model_dump(mode="json") for item in entries],
    }
    directory = CompactCandidateDirectory.model_validate(
        {
            **unsigned,
            "directory_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not directory.verify():  # pragma: no cover - construction invariant
        raise AssertionError("compact candidate directory verification failed")
    return directory


def _detailed_card_payload(
    *,
    candidate: CandidateEntry,
    candidate_universe: CandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    access_reason: Literal["focus_q", "approved_expansion"],
) -> DetailedEvidenceCard:
    binding_by_id = _binding_index(entity_registry)
    bindings = tuple(
        sorted(
            (binding_by_id[binding_id] for binding_id in candidate.binding_ids),
            key=lambda item: item.binding_id,
        )
    )
    candidate_binding_ids = set(candidate.binding_ids)
    observations = tuple(
        sorted(
            (
                resolution.observation
                for resolution in entity_registry.resolutions
                if resolution.binding.binding_id in candidate_binding_ids
                and resolution.binding.status == "bound"
                and resolution.binding.service_entity_key == candidate.service_entity_key
            ),
            key=lambda item: item.observation_id,
        )
    )
    observation_ids = {item.observation_id for item in observations}
    groups = tuple(
        sorted(
            (
                group
                for group in event_grouping.event_groups
                if observation_ids.intersection(group.observation_ids)
            ),
            key=lambda item: item.event_group_id,
        )
    )
    quality_flags = tuple(
        sorted({flag for observation in observations for flag in observation.quality_flags})
    )
    card = build_evidence_card(
        candidate_universe=candidate_universe,
        service_entity_key=candidate.service_entity_key,
        access_reason=access_reason,
        decision_time_unix_ns=entity_registry.decision_time_unix_ns,
        observation_ids=tuple(item.observation_id for item in observations),
        event_group_ids=tuple(item.event_group_id for item in groups),
        entity_binding_ids=tuple(item.binding_id for item in bindings),
        quality_flags=quality_flags,
    )
    unsigned: dict[str, object] = {
        "schema_version": DETAILED_EVIDENCE_CARD_SCHEMA_VERSION,
        "card": card.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in observations],
        "event_groups": [item.model_dump(mode="json") for item in groups],
        "entity_bindings": [item.model_dump(mode="json") for item in bindings],
    }
    detailed = DetailedEvidenceCard.model_validate(
        {
            **unsigned,
            "detailed_card_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not detailed.verify():  # pragma: no cover - construction invariant
        raise AssertionError("detailed evidence card verification failed")
    return detailed


def build_detailed_evidence_card(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    service_entity_key: str,
    access_reason: Literal["focus_q", "approved_expansion"],
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> DetailedEvidenceCard:
    """Build one legal detailed card for a frozen C member."""

    _validate_evidence_sources(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    candidates = {item.service_entity_key: item for item in candidate_universe.candidates}
    candidate = candidates.get(service_entity_key)
    if candidate is None:
        raise ValueError("detailed-card candidate is outside frozen C")
    return _detailed_card_payload(
        candidate=candidate,
        candidate_universe=candidate_universe,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        access_reason=access_reason,
    )


def build_semantic_context_bundle(
    *,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    focus_selection: FocusSelectionResult,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
) -> SemanticContextBundle:
    """Build the initial semantic context with full C and detailed Q only."""

    _validate_context_inputs(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        focus_selection=focus_selection,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
    )
    directory = _build_compact_directory(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        focus_selection=focus_selection,
        decision_time_unix_ns=entity_registry.decision_time_unix_ns,
    )
    candidate_by_key = {item.service_entity_key: item for item in candidate_universe.candidates}
    cards = tuple(
        _detailed_card_payload(
            candidate=candidate_by_key[service_key],
            candidate_universe=candidate_universe,
            entity_registry=entity_registry,
            event_grouping=event_grouping,
            access_reason="focus_q",
        )
        for service_key in focus_selection.focus_set.selected_service_keys
    )
    unsigned: dict[str, object] = {
        "schema_version": SEMANTIC_CONTEXT_BUNDLE_SCHEMA_VERSION,
        "opaque_case_id": candidate_universe.opaque_case_id,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "focus_selection_result_sha256": focus_selection.result_sha256,
        "entity_registry_sha256": entity_registry.registry_sha256,
        "event_grouping_result_sha256": event_grouping.result_sha256,
        "candidate_directory": directory.model_dump(mode="json"),
        "detailed_evidence_cards": [item.model_dump(mode="json") for item in cards],
    }
    bundle = SemanticContextBundle.model_validate(
        {
            **unsigned,
            "bundle_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not bundle.verify():  # pragma: no cover - construction invariant
        raise AssertionError("semantic context bundle verification failed")
    return bundle
