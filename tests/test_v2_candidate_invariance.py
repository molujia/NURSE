from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import candidate_universe as candidate_module
from nurse_rcl.v2 import case_state as state_module
from nurse_rcl.v2.candidate_freeze import (
    evaluate_candidate_scope,
    freeze_candidate_universe,
)
from nurse_rcl.v2.context_expansion import (
    build_card_expansion_policy,
    build_card_expansion_request,
    execute_card_expansion_batch,
)
from nurse_rcl.v2.focus_selection import build_neutral_focus_selection
from nurse_rcl.v2.observation_schemas import TimeInterval
from tests.test_v2_candidate_universe import (
    DECISION_TIME,
    OPAQUE_CASE_ID,
    build_registry_and_grouping,
    entity_input,
    metadata_record,
    observation,
    rich_fixture,
)
from tests.test_v2_semantic_context import build_rich_context


def _candidate_keys(universe: Any) -> tuple[str, ...]:
    return tuple(item.service_entity_key for item in universe.candidates)


def _candidate_universe_for_observation_state(
    *,
    evidence_state: str,
    formally_scorable: bool,
) -> Any:
    item = observation(
        suffix="1",
        evidence_state=evidence_state,
        formally_scorable=formally_scorable,
    )
    registry, grouping = build_registry_and_grouping(
        records=(
            metadata_record(
                suffix="1",
                entity="pod:thresholded",
                service="service:thresholded",
            ),
        ),
        observation_inputs=(entity_input(item, entity="pod:thresholded"),),
    )
    return candidate_module.build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
    )


def _observations_ready_state(
    *,
    entity_artifact_sha256: str,
    observation_artifact_sha256: str,
    logical_request_ids: Sequence[str] = (),
) -> Any:
    initial = state_module.build_initial_case_state(
        opaque_case_id=OPAQUE_CASE_ID,
        experiment_version="B0/v2-small",
        upstream_hashes={"input": "1" * 64, "configuration": "2" * 64},
        produced_artifact_hashes={"discovery": "3" * 64},
        created_at_utc="2026-09-06T01:00:00Z",
    )
    opaque_bound = state_module.build_case_state_transition(
        previous=initial,
        to_state="OPAQUE_BOUND",
        upstream_hashes=initial.checkpoint.produced_artifact_hashes,
        produced_artifact_hashes={"identity": "4" * 64},
        logical_request_ids=(),
        created_at_utc="2026-09-06T01:00:01Z",
    )
    return state_module.build_case_state_transition(
        previous=opaque_bound,
        to_state="OBSERVATIONS_READY",
        upstream_hashes={
            **opaque_bound.checkpoint.produced_artifact_hashes,
            "entities": entity_artifact_sha256,
        },
        produced_artifact_hashes={"observations": observation_artifact_sha256},
        logical_request_ids=logical_request_ids,
        created_at_utc="2026-09-06T01:00:02Z",
    )


def _audit_artifact_hashes(*, registry: Any, grouping: Any) -> dict[str, str]:
    names: Sequence[str] = (
        "input",
        "configuration",
        "entities",
        "observations",
        "candidate_universe",
        "focus_set",
        "hypotheses",
        "graph",
        "scores",
        "top_k",
        "history",
        "rules",
        "calls",
        "queries",
        "costs",
        "degradation",
    )
    values = {name: format(index + 1, "064x") for index, name in enumerate(names)}
    values["entities"] = registry.registry_sha256
    values["observations"] = grouping.result_sha256
    return values


def _rehash(document: dict[str, Any], hash_field: str) -> None:
    unsigned = {key: value for key, value in document.items() if key != hash_field}
    document[hash_field] = sha256_hex(canonical_json_bytes(unsigned))


def _diagnostic_document(document: dict[str, Any]) -> dict[str, Any]:
    names = (
        "schema_version",
        "opaque_case_id",
        "failure_code",
        "scope_interval",
        "observation_contract_sha256",
        "metadata_record_count",
        "timeline_binding_count",
        "observation_resolution_count",
        "unbound_observation_count",
        "conflicted_binding_count",
        "candidate_count",
        "prediction_service_keys",
        "semantic_request_permitted",
        "semantic_call_count",
        "empty_semantic_calls_sha256",
    )
    return {name: document[name] for name in names}


def _rehash_failure_document(document: dict[str, Any]) -> None:
    state = document["case_state_entry"]
    package = document["sealed_audit_package"]
    _rehash(state["checkpoint"], "checkpoint_sha256")
    _rehash(state, "entry_sha256")
    _rehash(package, "package_sha256")
    _rehash(document, "outcome_sha256")


def _build_empty_failure() -> tuple[Any, Any, Any]:
    registry, grouping = build_registry_and_grouping(
        records=(),
        observation_inputs=(),
    )
    outcome = candidate_module.seal_empty_candidate_universe_failure(
        previous_case_state=_observations_ready_state(
            entity_artifact_sha256=registry.registry_sha256,
            observation_artifact_sha256=grouping.result_sha256,
        ),
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
        audit_artifact_hashes=_audit_artifact_hashes(
            registry=registry,
            grouping=grouping,
        ),
        created_at_utc="2026-09-06T01:00:03Z",
        sealed_at_utc="2026-09-06T01:00:04Z",
    )
    return outcome, registry, grouping


def test_alarm_and_topology_are_not_candidate_construction_inputs() -> None:
    parameters = set(
        inspect.signature(candidate_module.build_complete_candidate_universe).parameters
    )
    assert parameters == {"entity_registry", "event_grouping", "scope_interval"}

    registry, grouping = rich_fixture()
    universe = candidate_module.build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
    )
    candidate_keys = set(_candidate_keys(universe))
    alarm_members = {"service:supported"}
    topology_members = {"service:supported", "service:normal"}
    frozen = freeze_candidate_universe(universe)

    assert alarm_members < candidate_keys
    assert topology_members < candidate_keys
    assert "service:unobserved" in candidate_keys
    assert "service:conflict-a" in candidate_keys
    for filtered_keys in (alarm_members, topology_members):
        decision = evaluate_candidate_scope(
            frozen=frozen,
            candidate_service_keys=tuple(filtered_keys),
            operation="candidate_transform",
        )
        assert decision.accepted is False
        assert decision.reason_code == "candidate_removed_from_frozen_universe"


def test_anomaly_threshold_state_changes_evidence_but_not_candidate_membership() -> None:
    retained = _candidate_universe_for_observation_state(
        evidence_state="abnormal",
        formally_scorable=True,
    )
    suppressed = _candidate_universe_for_observation_state(
        evidence_state="verified_normal",
        formally_scorable=True,
    )

    assert _candidate_keys(retained) == _candidate_keys(suppressed) == ("service:thresholded",)
    assert retained.candidates[0].evidence_state == "supported"
    assert suppressed.candidates[0].evidence_state == "zero_support"


def test_q_limits_change_context_detail_without_changing_frozen_c() -> None:
    registry, grouping = rich_fixture()
    universe = candidate_module.build_complete_candidate_universe(
        entity_registry=registry,
        event_grouping=grouping,
        scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
    )
    frozen = freeze_candidate_universe(universe)

    narrow = build_neutral_focus_selection(
        candidate_universe=universe,
        frozen_candidates=frozen,
        event_grouping=grouping,
        q_base=0,
        q_max=1,
    )
    wide = build_neutral_focus_selection(
        candidate_universe=universe,
        frozen_candidates=frozen,
        event_grouping=grouping,
        q_base=9,
        q_max=9,
    )

    assert narrow.focus_set.selected_count == 0
    assert wide.focus_set.selected_count == universe.candidate_count
    assert tuple(item.service_entity_key for item in narrow.focus_set.decisions) == (
        frozen.candidate_service_keys
    )
    assert tuple(item.service_entity_key for item in wide.focus_set.decisions) == (
        frozen.candidate_service_keys
    )
    assert universe.verify() and frozen.verify()


def test_token_refusal_leaves_c_and_compact_directory_complete() -> None:
    registry, grouping, universe, frozen, _, bundle = build_rich_context(q_base=6)
    before = universe.model_dump(mode="json")
    request = build_card_expansion_request(
        opaque_case_id=bundle.opaque_case_id,
        candidate_freeze_sha256=bundle.candidate_freeze_sha256,
        semantic_context_bundle_sha256=bundle.bundle_sha256,
        service_entity_key="service:unknown",
        query_kind="candidate_detailed_evidence_card",
        decision_time_unix_ns=bundle.candidate_directory.decision_time_unix_ns,
    )

    result = execute_card_expansion_batch(
        requests=(request,),
        policy=build_card_expansion_policy(query_limit=8, token_limit=1),
        semantic_context=bundle,
        candidate_universe=universe,
        frozen_candidates=frozen,
        entity_registry=registry,
        event_grouping=grouping,
    )

    assert result.decisions[0].reason_code == "token_budget_exhausted"
    assert result.returned_cards == ()
    assert universe.model_dump(mode="json") == before
    assert frozen.candidate_service_keys == _candidate_keys(universe)
    assert (
        tuple(item.service_entity_key for item in bundle.candidate_directory.entries)
        == frozen.candidate_service_keys
    )


def test_empty_candidate_universe_seals_auditable_failure_without_semantic_request() -> None:
    outcome, _, _ = _build_empty_failure()

    assert outcome.verify()
    assert outcome.failure_code == "empty_candidate_universe"
    assert outcome.candidate_count == 0
    assert outcome.prediction_service_keys == ()
    assert outcome.semantic_request_permitted is False
    assert outcome.semantic_call_count == 0
    assert outcome.case_state_entry.checkpoint.state == "FAILURE_SEALED"
    assert outcome.case_state_entry.failure_code == "empty_candidate_universe"
    assert outcome.case_state_entry.checkpoint.logical_request_ids == ()
    assert outcome.sealed_audit_package.terminal_state == "FAILURE_SEALED"
    assert outcome.sealed_audit_package.run_mode == "failure"
    assert outcome.sealed_audit_package.prediction_service_keys == ()
    assert outcome.sealed_audit_package.verify()
    assert (
        outcome.sealed_audit_package.artifact_hashes["candidate_universe"]
        == outcome.diagnostic_sha256
    )
    assert (
        outcome.sealed_audit_package.artifact_hashes["calls"] == outcome.empty_semantic_calls_sha256
    )
    assert outcome.case_state_entry.checkpoint.produced_artifact_hashes == {
        "candidate_scope_failure": outcome.diagnostic_sha256
    }
    assert (
        outcome.case_state_entry.checkpoint.upstream_hashes["entities"]
        == (outcome.sealed_audit_package.artifact_hashes["entities"])
    )
    assert (
        outcome.case_state_entry.checkpoint.upstream_hashes["observations"]
        == (outcome.sealed_audit_package.artifact_hashes["observations"])
    )
    assert outcome.metadata_record_count == 0
    assert outcome.observation_resolution_count == 0
    assert outcome.model_copy(update={"metadata_record_count": 1}).verify() is False


def test_empty_candidate_failure_schema_is_registered_explicitly() -> None:
    from nurse_rcl.v2.schema_compatibility import CURRENT_SCHEMA_VERSIONS

    assert CURRENT_SCHEMA_VERSIONS["empty_candidate_universe_failure"] == (
        candidate_module.EMPTY_CANDIDATE_FAILURE_SCHEMA_VERSION
    )


def test_empty_candidate_failure_rejects_a_preexisting_semantic_request() -> None:
    registry, grouping = build_registry_and_grouping(
        records=(),
        observation_inputs=(),
    )

    with pytest.raises(ValueError, match="preexisting semantic request"):
        candidate_module.seal_empty_candidate_universe_failure(
            previous_case_state=_observations_ready_state(
                entity_artifact_sha256=registry.registry_sha256,
                observation_artifact_sha256=grouping.result_sha256,
                logical_request_ids=("lrq_" + "a" * 24,),
            ),
            entity_registry=registry,
            event_grouping=grouping,
            scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
            audit_artifact_hashes=_audit_artifact_hashes(
                registry=registry,
                grouping=grouping,
            ),
            created_at_utc="2026-09-06T01:00:03Z",
            sealed_at_utc="2026-09-06T01:00:04Z",
        )


def test_empty_candidate_failure_rejects_conflicting_audit_input_hashes() -> None:
    registry, grouping = build_registry_and_grouping(
        records=(),
        observation_inputs=(),
    )
    artifact_hashes = _audit_artifact_hashes(registry=registry, grouping=grouping)
    artifact_hashes["observations"] = "f" * 64

    with pytest.raises(ValueError, match="observations artifact hash"):
        candidate_module.seal_empty_candidate_universe_failure(
            previous_case_state=_observations_ready_state(
                entity_artifact_sha256=registry.registry_sha256,
                observation_artifact_sha256=grouping.result_sha256,
            ),
            entity_registry=registry,
            event_grouping=grouping,
            scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
            audit_artifact_hashes=artifact_hashes,
            created_at_utc="2026-09-06T01:00:03Z",
            sealed_at_utc="2026-09-06T01:00:04Z",
        )


def test_empty_candidate_failure_rejects_rehashed_terminal_diagnostic_drift() -> None:
    outcome, _, _ = _build_empty_failure()
    document = outcome.model_dump(mode="json")
    document["case_state_entry"]["checkpoint"]["produced_artifact_hashes"] = {
        "candidate_scope_failure": "e" * 64
    }
    _rehash_failure_document(document)

    with pytest.raises(ValidationError, match="terminal state does not bind"):
        candidate_module.EmptyCandidateUniverseFailure.model_validate(document)


def test_empty_candidate_failure_rejects_rehashed_noncanonical_empty_calls() -> None:
    outcome, _, _ = _build_empty_failure()
    document = outcome.model_dump(mode="json")
    document["empty_semantic_calls_sha256"] = "f" * 64
    document["diagnostic_sha256"] = sha256_hex(canonical_json_bytes(_diagnostic_document(document)))
    document["case_state_entry"]["checkpoint"]["produced_artifact_hashes"] = {
        "candidate_scope_failure": document["diagnostic_sha256"]
    }
    document["sealed_audit_package"]["artifact_hashes"]["candidate_universe"] = document[
        "diagnostic_sha256"
    ]
    document["sealed_audit_package"]["artifact_hashes"]["calls"] = "f" * 64
    _rehash_failure_document(document)

    with pytest.raises(ValidationError, match="canonical empty semantic call set"):
        candidate_module.EmptyCandidateUniverseFailure.model_validate(document)


def test_empty_candidate_failure_rejects_rehashed_audit_upstream_drift() -> None:
    outcome, _, _ = _build_empty_failure()
    document = outcome.model_dump(mode="json")
    document["sealed_audit_package"]["artifact_hashes"]["observations"] = "f" * 64
    _rehash_failure_document(document)

    with pytest.raises(ValidationError, match="audit package conflicts with terminal upstream"):
        candidate_module.EmptyCandidateUniverseFailure.model_validate(document)


def test_empty_candidate_failure_rejects_rehashed_loss_of_entity_lineage() -> None:
    outcome, _, _ = _build_empty_failure()
    document = outcome.model_dump(mode="json")
    del document["case_state_entry"]["checkpoint"]["upstream_hashes"]["entities"]
    _rehash_failure_document(document)

    with pytest.raises(ValidationError, match="missing frozen candidate input lineage"):
        candidate_module.EmptyCandidateUniverseFailure.model_validate(document)


def test_empty_candidate_failure_rejects_registry_replacement_after_state_binding() -> None:
    nonempty_registry, nonempty_grouping = build_registry_and_grouping(
        records=(
            metadata_record(
                suffix="1",
                entity="pod:real",
                service="service:real",
            ),
        ),
        observation_inputs=(),
    )
    empty_registry, empty_grouping = build_registry_and_grouping(
        records=(),
        observation_inputs=(),
    )
    assert nonempty_grouping == empty_grouping

    with pytest.raises(ValueError, match="entity registry conflicts with case-state"):
        candidate_module.seal_empty_candidate_universe_failure(
            previous_case_state=_observations_ready_state(
                entity_artifact_sha256=nonempty_registry.registry_sha256,
                observation_artifact_sha256=nonempty_grouping.result_sha256,
            ),
            entity_registry=empty_registry,
            event_grouping=empty_grouping,
            scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
            audit_artifact_hashes=_audit_artifact_hashes(
                registry=empty_registry,
                grouping=empty_grouping,
            ),
            created_at_utc="2026-09-06T01:00:03Z",
            sealed_at_utc="2026-09-06T01:00:04Z",
        )


def test_empty_candidate_failure_path_rejects_a_nonempty_candidate_scope() -> None:
    registry, grouping = rich_fixture()

    with pytest.raises(ValueError, match="candidate universe is not empty"):
        candidate_module.seal_empty_candidate_universe_failure(
            previous_case_state=_observations_ready_state(
                entity_artifact_sha256=registry.registry_sha256,
                observation_artifact_sha256=grouping.result_sha256,
            ),
            entity_registry=registry,
            event_grouping=grouping,
            scope_interval=TimeInterval(start_unix_ns=0, end_unix_ns=DECISION_TIME),
            audit_artifact_hashes=_audit_artifact_hashes(
                registry=registry,
                grouping=grouping,
            ),
            created_at_utc="2026-09-06T01:00:03Z",
            sealed_at_utc="2026-09-06T01:00:04Z",
        )
