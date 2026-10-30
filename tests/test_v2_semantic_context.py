from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest

from nurse_rcl.v2.candidate_freeze import freeze_candidate_universe
from nurse_rcl.v2.candidate_universe import build_complete_candidate_universe
from nurse_rcl.v2.event_grouping import (
    build_event_grouping_result,
    build_frozen_signal_family_mapping,
)
from nurse_rcl.v2.focus_selection import build_neutral_focus_selection
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.opaque_ids import EvaluatorIdentitySource, build_identity_artifacts
from nurse_rcl.v2.temporal_entity_registry import (
    ObservationEntityInput,
    build_entity_metadata_record,
    build_temporal_entity_registry,
)
from tests.test_v2_candidate_universe import DECISION_TIME, rich_fixture
from tests.test_v2_inference_leakage import FIXED_UUID, typed_observation


def context_module() -> ModuleType:
    try:
        return importlib.import_module("nurse_rcl.v2.semantic_context")
    except ModuleNotFoundError:
        pytest.fail("compact C/Q semantic context is not implemented", pytrace=False)


def build_rich_context(*, q_base: int, reverse_inputs: bool = False) -> tuple[Any, ...]:
    registry, grouping = rich_fixture(reverse_inputs=reverse_inputs)
    universe = build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
    )
    frozen = freeze_candidate_universe(universe)
    focus = build_neutral_focus_selection(
        candidate_universe=universe,
        frozen_candidates=frozen,
        event_grouping=grouping,
        q_base=q_base,
        q_max=9,
    )
    bundle = context_module().build_semantic_context_bundle(
        candidate_universe=universe,
        frozen_candidates=frozen,
        focus_selection=focus,
        entity_registry=registry,
        event_grouping=grouping,
    )
    return registry, grouping, universe, frozen, focus, bundle


def test_compact_directory_covers_all_c_while_detailed_cards_cover_only_q() -> None:
    _, _, universe, frozen, focus, bundle = build_rich_context(q_base=6)

    assert bundle.verify()
    assert bundle.candidate_freeze_sha256 == frozen.freeze_sha256
    assert bundle.candidate_directory.candidate_count == universe.candidate_count == 9
    assert tuple(item.service_entity_key for item in bundle.candidate_directory.entries) == (
        frozen.candidate_service_keys
    )
    selected = focus.focus_set.selected_service_keys
    assert (
        tuple(item.card.service_entity_key for item in bundle.detailed_evidence_cards) == selected
    )

    entry_by_service = {
        item.service_entity_key: item for item in bundle.candidate_directory.entries
    }
    score_by_service = {item.service_entity_key: item.e0 for item in focus.neutral_scores}
    for candidate in universe.candidates:
        entry = entry_by_service[candidate.service_entity_key]
        assert entry.evidence_state == candidate.evidence_state
        assert entry.neutral_e0 == score_by_service[candidate.service_entity_key]
        assert entry.in_initial_focus == (candidate.service_entity_key in selected)
        assert entry.has_initial_detailed_card == entry.in_initial_focus
        assert {
            "binding_ids",
            "observation_ids",
            "event_group_ids",
            "observations",
            "event_groups",
            "entity_bindings",
        }.isdisjoint(entry.model_dump(mode="json"))

    assert all(
        not entry.has_initial_detailed_card
        for entry in bundle.candidate_directory.entries
        if entry.service_entity_key not in selected
    )


def test_detailed_q_cards_preserve_normal_zero_unknown_and_conflict_without_guessing() -> None:
    _, _, _, _, _, bundle = build_rich_context(q_base=9)
    cards = {item.card.service_entity_key: item for item in bundle.detailed_evidence_cards}

    assert [item.evidence_state for item in cards["service:normal"].observations] == [
        "verified_normal"
    ]
    assert [item.evidence_state for item in cards["service:zero"].observations] == ["zero"]
    assert [item.evidence_state for item in cards["service:unknown"].observations] == ["unknown"]
    assert cards["service:unobserved"].observations == ()
    for service_key in ("service:conflict-a", "service:conflict-b"):
        assert cards[service_key].observations == ()
        assert cards[service_key].entity_bindings
        assert all(binding.status == "conflicted" for binding in cards[service_key].entity_bindings)
        assert all(
            service_key in binding.conflict_service_keys
            for binding in cards[service_key].entity_bindings
        )

    for detailed in bundle.detailed_evidence_cards:
        assert detailed.verify()
        assert detailed.card.access_reason == "focus_q"
        assert detailed.card.observation_ids == tuple(
            item.observation_id for item in detailed.observations
        )
        assert detailed.card.event_group_ids == tuple(
            item.event_group_id for item in detailed.event_groups
        )
        assert detailed.card.entity_binding_ids == tuple(
            item.binding_id for item in detailed.entity_bindings
        )
        assert all(
            item.interval.end_unix_ns <= detailed.card.decision_time_unix_ns
            for item in (*detailed.observations, *detailed.event_groups)
        )
        assert all(
            item.valid_interval.end_unix_ns <= detailed.card.decision_time_unix_ns
            for item in detailed.entity_bindings
        )
        observation_ids = {item.observation_id for item in detailed.observations}
        assert all(
            observation_ids.intersection(group.observation_ids) for group in detailed.event_groups
        )


def test_context_construction_is_canonical_and_rejects_unverified_inputs() -> None:
    *_, canonical = build_rich_context(q_base=6)
    *_, permuted = build_rich_context(q_base=6, reverse_inputs=True)
    assert permuted == canonical

    registry, grouping, universe, frozen, focus, _ = build_rich_context(q_base=6)
    with pytest.raises(ValueError, match="focus selection failed verification"):
        context_module().build_semantic_context_bundle(
            candidate_universe=universe,
            frozen_candidates=frozen,
            focus_selection=focus.model_copy(update={"result_sha256": "0" * 64}),
            entity_registry=registry,
            event_grouping=grouping,
        )

    with pytest.raises(ValueError, match="event grouping failed verification"):
        context_module().build_semantic_context_bundle(
            candidate_universe=universe,
            frozen_candidates=frozen,
            focus_selection=focus,
            entity_registry=registry,
            event_grouping=grouping.model_copy(update={"decision_time_unix_ns": DECISION_TIME + 1}),
        )

    detailed = canonical.detailed_evidence_cards[-1]
    observation = detailed.observations[0]
    post_decision = observation.model_copy(
        update={
            "interval": TimeInterval(
                start_unix_ns=DECISION_TIME,
                end_unix_ns=DECISION_TIME + 1,
            )
        }
    )
    assert detailed.model_copy(update={"observations": (post_decision,)}).verify() is False


def test_single_card_builder_rejects_mismatched_frozen_evidence_sources() -> None:
    registry, grouping, universe, frozen, _, _ = build_rich_context(q_base=6)
    empty_grouping = build_event_grouping_result(
        opaque_case_id=grouping.opaque_case_id,
        decision_time_unix_ns=grouping.decision_time_unix_ns,
        observations=(),
        association_claims=(),
        family_mapping=grouping.family_mapping,
        association_policy=grouping.association_policy,
    )

    with pytest.raises(ValueError, match="observation membership differs"):
        context_module().build_detailed_evidence_card(
            candidate_universe=universe,
            frozen_candidates=frozen,
            service_entity_key="service:supported",
            access_reason="approved_expansion",
            entity_registry=registry,
            event_grouping=empty_grouping,
        )


def test_semantic_context_models_pass_the_typed_inference_boundary() -> None:
    leakage = importlib.import_module("nurse_rcl.v2.inference_leakage")
    _, identity_manifest = build_identity_artifacts(
        (
            EvaluatorIdentitySource(
                dataset="rcabench",
                original_case_id="private-case-id",
                incident_group_id="grp_" + "a" * 24,
                answer_services=("private-answer",),
            ),
        ),
        method="random_uuid_v4",
        uuid_factory=lambda: FIXED_UUID,
    )
    opaque_case_id = identity_manifest.records[0].opaque_case_id
    observation = typed_observation(opaque_case_id)
    metadata = build_entity_metadata_record(
        opaque_case_id=opaque_case_id,
        metadata_source="deployment",
        entity_kind="service",
        observed_entity_key="service:checkout",
        relation="explicit_service_binding",
        service_entity_key="service:checkout",
        valid_interval=TimeInterval(start_unix_ns=0, end_unix_ns=2),
        source_fingerprint="a" * 64,
        provenance_ids=("deployment:service",),
    )
    registry = build_temporal_entity_registry(
        opaque_case_id=opaque_case_id,
        decision_time_unix_ns=2,
        metadata_records=(metadata,),
        observations=(
            ObservationEntityInput(
                observation=observation,
                entity_kind="service",
                observed_entity_key="service:checkout",
            ),
        ),
    )
    grouping = build_event_grouping_result(
        opaque_case_id=opaque_case_id,
        decision_time_unix_ns=2,
        observations=tuple(item.observation for item in registry.resolutions),
        association_claims=(),
        family_mapping=build_frozen_signal_family_mapping(
            mapping_version="semantic-context-test/1",
            entries=(),
        ),
    )
    universe = build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=2),
    )
    frozen = freeze_candidate_universe(universe)
    focus = build_neutral_focus_selection(
        candidate_universe=universe,
        frozen_candidates=frozen,
        event_grouping=grouping,
        q_base=1,
        q_max=1,
    )
    bundle = context_module().build_semantic_context_bundle(
        candidate_universe=universe,
        frozen_candidates=frozen,
        focus_selection=focus,
        entity_registry=registry,
        event_grouping=grouping,
    )
    payload = {
        "dataset": "rcabench",
        "opaque_case_id": opaque_case_id,
        "decision_time": 2,
        "candidate_directory": bundle.candidate_directory,
        "entities": registry,
        "evidence_cards": bundle.detailed_evidence_cards,
    }

    audit = leakage.audit_inference_boundary(
        payload,
        identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=identity_manifest.manifest_sha256,
    )

    assert audit.verify()
    assert {
        "CompactCandidateDirectory",
        "CompactCandidateDirectoryEntry",
        "DetailedEvidenceCard",
        "EvidenceCard",
    } <= set(audit.public_model_types)
