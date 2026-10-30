"""Post-ranking immutable report-context projection for NURSE V2."""

from __future__ import annotations

from decimal import Decimal
from functools import cache

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.case_audit import seal_case_audit
from tests.test_v2_case_audit import build_real_replay_bundle


@cache
def _sealed_package(*, experiment_version: str = "B0/v2-small") -> object:
    bundle = build_real_replay_bundle()
    return seal_case_audit(
        replay_bundle=bundle,
        experiment_version=experiment_version,
        run_mode="full",
        sealed_at_utc="2026-09-10T11:59:00Z",
    )


@cache
def _frozen_context() -> object:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit

    package = _sealed_package()
    return freeze_report_context_from_audit(
        audit_package=package,
        expected_audit_sha256=package.audit_sha256,
        frozen_at_utc="2026-09-10T12:00:00Z",
    )


def test_freeze_report_context_requires_and_replays_a_ranked_audit() -> None:
    package = _sealed_package()
    bundle = package.replay_bundle

    context = _frozen_context()

    assert context.schema_version == "nurse.v2.frozen-report-context/3"
    assert context.stage == "CONTEXT_FROZEN"
    assert context.ranking_mutable is False
    assert context.source_audit_sha256 == package.audit_sha256
    assert context.source_ranking_bundle_sha256 == bundle.bundle_sha256
    assert context.payload.ranking_result == bundle.ranking_result
    assert context.payload.observations == bundle.evidence.event_grouping.observations
    assert len(context.payload.competition_deltas) == len(bundle.top_k.entries) - 1
    assert context.verify()


def test_report_payload_preserves_event_strength_evidence() -> None:
    bundle = _sealed_package().replay_bundle

    assert _frozen_context().payload.event_strength == bundle.evidence.event_strength


def test_report_hypothesis_facts_preserve_complete_parsed_structure() -> None:
    bundle = _sealed_package().replay_bundle
    facts_by_guard_id = {
        fact.guard_result_id: fact for fact in _frozen_context().payload.hypotheses
    }
    parsed_guards = tuple(
        guard for guard in bundle.evidence.role_guard_results if guard.parsed_hypothesis is not None
    )

    assert parsed_guards
    for guard in parsed_guards:
        hypothesis = guard.parsed_hypothesis
        assert hypothesis is not None
        fact = facts_by_guard_id[guard.guard_result_id]
        assert fact.mechanism_definition_sha256 == hypothesis.mechanism_definition_sha256
        assert fact.origin_version == hypothesis.origin_version
        assert fact.time_intervals == hypothesis.time_intervals
        assert fact.hypothesis_sha256 == hypothesis.hypothesis_sha256
        assert fact.provenance_ids == hypothesis.provenance_ids


def test_unparsed_report_hypothesis_cannot_expose_provenance() -> None:
    from nurse_rcl.v2.report_context import ReportHypothesisFact

    document = _frozen_context().payload.hypotheses[0].model_dump(mode="json")
    for field_name in (
        "hypothesis_id",
        "candidate_service_key",
        "mechanism_key",
        "mechanism_definition_sha256",
        "origin",
        "origin_version",
        "lifecycle_status",
        "hypothesis_sha256",
    ):
        if field_name in document:
            document[field_name] = None
    for field_name in (
        "observation_ids",
        "relation_ids",
        "event_group_ids",
        "time_intervals",
        "necessary_predicate_result_ids",
        "optional_predicate_result_ids",
        "refuting_predicate_result_ids",
        "alternative_hypothesis_ids",
        "verification_request_ids",
        "unknown_reasons",
    ):
        if field_name in document:
            document[field_name] = ()
    document.update(
        final_disposition="rejected",
        provenance_ids=("private:must-not-survive",),
        score_impact_id=None,
    )

    with pytest.raises(ValidationError, match="cannot expose submitted claims"):
        ReportHypothesisFact.model_validate(document)


def test_freeze_report_context_rejects_wrong_external_audit_anchor() -> None:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit

    package = _sealed_package()

    with pytest.raises(ValueError, match="external audit digest"):
        freeze_report_context_from_audit(
            audit_package=package,
            expected_audit_sha256="f" * 64,
            frozen_at_utc="2026-09-10T12:00:00Z",
        )


def test_freeze_report_context_rejects_an_unsealed_ranking_bundle() -> None:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit

    with pytest.raises(ValueError, match="exact ImmutableCaseAuditPackage"):
        freeze_report_context_from_audit(
            audit_package=build_real_replay_bundle(),  # type: ignore[arg-type]
            expected_audit_sha256="f" * 64,
            frozen_at_utc="2026-09-10T12:00:00Z",
        )


def test_report_context_cannot_be_frozen_before_the_ranked_audit() -> None:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit

    package = _sealed_package()

    with pytest.raises(ValueError, match="after the ranked audit"):
        freeze_report_context_from_audit(
            audit_package=package,
            expected_audit_sha256=package.audit_sha256,
            frozen_at_utc="2026-09-10T11:58:59Z",
        )


def test_report_payload_is_canonical_at_rest_and_omits_submitted_free_text() -> None:
    context = _frozen_context()
    first = context.payload
    original_ranking = first.ranking_result

    assert first is not context.payload
    assert "submitted_record_json" not in context.payload_json
    assert first.hypotheses
    with pytest.raises(ValidationError, match="frozen_instance"):
        first.ranking_result = build_real_replay_bundle(source_strength=0.4).ranking_result
    object.__setattr__(
        first,
        "ranking_result",
        build_real_replay_bundle(source_strength=0.4).ranking_result,
    )

    assert context.payload.ranking_result == original_ranking
    assert context.verify()


@pytest.mark.parametrize(
    ("limit_name", "limit", "payload_json", "message"),
    (
        ("_MAX_REPORT_JSON_BYTES", 8, '{"value":"too-large"}', "size limit"),
        ("_MAX_REPORT_JSON_DEPTH", 1, '{"outer":{"inner":0}}', "nesting capacity"),
        ("_MAX_REPORT_JSON_NODES", 2, '{"values":[0,1]}', "node capacity"),
    ),
)
def test_report_context_rejects_resource_exhausting_canonical_payloads(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    payload_json: str,
    message: str,
) -> None:
    from nurse_rcl.v2 import report_context

    context = _frozen_context()
    document = context.model_dump(mode="json")
    document["payload_json"] = payload_json
    monkeypatch.setattr(report_context, limit_name, limit, raising=False)

    with pytest.raises(ValueError, match=message):
        report_context.FrozenReportContext.model_validate(document)


def test_competition_deltas_are_exactly_derived_from_frozen_top_k() -> None:
    context = _frozen_context()
    payload = context.payload
    scores = {item.candidate_service_key: item for item in payload.scores}
    leader = scores[payload.top_k.entries[0].service_entity_key]

    for delta, entry in zip(
        payload.competition_deltas,
        payload.top_k.entries[1:],
        strict=True,
    ):
        competitor = scores[entry.service_entity_key]
        assert delta.competing_rank == entry.rank
        assert Decimal(delta.total_score_gap) == (
            Decimal(leader.quantized_total_score) - Decimal(competitor.quantized_total_score)
        )
        assert delta.verify()


def test_competition_deltas_include_guard_accepted_hypotheses() -> None:
    payload = _frozen_context().payload
    accepted_by_candidate: dict[str, tuple[str, ...]] = {}
    for candidate_service_key in {
        fact.candidate_service_key
        for fact in payload.hypotheses
        if fact.final_disposition == "accepted"
        and fact.candidate_service_key is not None
        and fact.hypothesis_id is not None
    }:
        accepted_by_candidate[candidate_service_key] = tuple(
            sorted(
                fact.hypothesis_id
                for fact in payload.hypotheses
                if fact.final_disposition == "accepted"
                and fact.candidate_service_key == candidate_service_key
                and fact.hypothesis_id is not None
            )
        )

    leader_key = payload.top_k.entries[0].service_entity_key
    assert accepted_by_candidate
    for delta in payload.competition_deltas:
        assert delta.leading_accepted_hypothesis_ids == accepted_by_candidate.get(
            leader_key,
            (),
        )
        assert delta.competing_accepted_hypothesis_ids == accepted_by_candidate.get(
            delta.competing_service_key,
            (),
        )


def _resign_context(context: object, **updates: object) -> object:
    from nurse_rcl.v2.report_context import FrozenReportContext

    unsigned = context.model_dump(mode="json", exclude={"context_sha256"})
    unsigned.update(updates)
    return FrozenReportContext.model_validate(
        {**unsigned, "context_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def _resign_payload(payload: object, **updates: object) -> object:
    candidate = payload.model_copy(update=updates)
    unsigned = candidate.model_dump(mode="json", exclude={"payload_sha256"})
    return candidate.model_copy(
        update={"payload_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def _resign_context_payload(context: object, payload: object) -> object:
    return _resign_context(
        context,
        payload_json=canonical_json_bytes(payload.model_dump(mode="json")).decode("utf-8"),
        payload_sha256=payload.payload_sha256,
        source_ranking_result_sha256=payload.ranking_result.result_sha256,
        source_top_k_sha256=payload.top_k.ranking_sha256,
    )


@cache
def _foreign_history_evaluation() -> object:
    from tests.test_v2_history_scoring import (
        _query,
        _trusted_components,
        history_scoring_module,
    )

    bundle = _sealed_package().replay_bundle
    *_, trusted_source_bundle, library = _trusted_components("report-foreign")
    return history_scoring_module().evaluate_history(
        frozen_candidates=bundle.evidence.frozen_candidates,
        mode="hint",
        library=library,
        query_embedding=_query(),
        trusted_source_bundle=trusted_source_bundle,
    )


@pytest.mark.parametrize(
    "channel_name",
    (
        "family_aggregation",
        "propagation_scoring",
        "rule_scoring",
        "history_evaluation",
    ),
)
def test_resigned_payload_rejects_a_valid_foreign_ranking_channel(
    channel_name: str,
) -> None:
    context = _frozen_context()
    payload = context.payload
    alternate_bundle = build_real_replay_bundle(source_strength=0.4)
    alternate_channel = (
        _foreign_history_evaluation()
        if channel_name == "history_evaluation"
        else getattr(alternate_bundle, channel_name)
    )
    original_channel = getattr(payload, channel_name)

    assert alternate_channel.verify()
    assert alternate_channel.result_sha256 != original_channel.result_sha256
    forged_payload = _resign_payload(payload, **{channel_name: alternate_channel})
    forged_context = _resign_context_payload(context, forged_payload)

    assert (forged_payload.verify(), forged_context.verify()) == (False, False)


def test_resigned_payload_binds_family_to_its_declared_event_inputs() -> None:
    from nurse_rcl.v2.family_aggregation import FamilyAggregationResult
    from nurse_rcl.v2.reference_ranking import ReferenceRankingResult

    context = _frozen_context()
    payload = context.payload
    alternate_evidence = build_real_replay_bundle(source_strength=0.4).evidence
    family_unsigned = payload.family_aggregation.model_dump(
        mode="json",
        exclude={"family_aggregation_result_id", "result_sha256"},
    )
    family_unsigned.update(
        {
            "event_grouping_result_sha256": alternate_evidence.event_grouping.result_sha256,
            "event_strength_result_id": (
                alternate_evidence.event_strength.event_strength_result_id
            ),
            "event_strength_result_sha256": alternate_evidence.event_strength.result_sha256,
        }
    )
    family_digest = sha256_hex(canonical_json_bytes(family_unsigned))
    detached_family = FamilyAggregationResult.model_validate(
        {
            **family_unsigned,
            "family_aggregation_result_id": "far_" + family_digest[:24],
            "result_sha256": family_digest,
        }
    )

    assert detached_family.verify()
    ranking_unsigned = payload.ranking_result.model_dump(
        mode="json",
        exclude={"reference_ranking_result_id", "result_sha256"},
    )
    ranking_unsigned["family_aggregation_result_sha256"] = detached_family.result_sha256
    authority_document = {
        field_name: ranking_unsigned[field_name]
        for field_name in (
            "candidate_freeze_sha256",
            "family_aggregation_result_sha256",
            "propagation_scoring_result_sha256",
            "propagation_authoritative_input_sha256",
            "rule_scoring_result_sha256",
            "rule_authoritative_input_sha256",
            "history_evaluation_result_sha256",
            "history_authoritative_input_sha256",
        )
    }
    ranking_unsigned["authoritative_input_sha256"] = sha256_hex(
        canonical_json_bytes(
            {
                "candidate_freeze_sha256": ranking_unsigned["candidate_freeze_sha256"],
                "candidate_universe_sha256": ranking_unsigned["candidate_universe_sha256"],
                "candidate_service_keys": tuple(
                    item.candidate_service_key for item in payload.ranking_result.scores
                ),
                "config_sha256": payload.ranking_result.config.config_sha256,
                "authority": authority_document,
            }
        )
    )
    ranking_digest = sha256_hex(canonical_json_bytes(ranking_unsigned))
    rebound_ranking = ReferenceRankingResult.model_validate(
        {
            **ranking_unsigned,
            "reference_ranking_result_id": "rrank_" + ranking_digest[:24],
            "result_sha256": ranking_digest,
        }
    )

    assert rebound_ranking.verify_structure()
    forged_payload = _resign_payload(
        payload,
        family_aggregation=detached_family,
        ranking_result=rebound_ranking,
    )
    forged_context = _resign_context_payload(context, forged_payload)

    assert (forged_payload.verify(), forged_context.verify()) == (False, False)


@pytest.mark.parametrize(
    ("field_name", "forged_reference"),
    (
        ("relation_ids", "assoc_" + "f" * 24),
        ("verification_request_ids", "vrq_" + "f" * 24),
        ("score_impact_id", "sci_" + "f" * 24),
    ),
)
def test_resigned_hypothesis_cannot_reference_unknown_report_evidence(
    field_name: str,
    forged_reference: str,
) -> None:
    payload = _frozen_context().payload
    fact = next(
        item
        for item in payload.hypotheses
        if field_name != "score_impact_id" or item.score_impact_id is not None
    )
    current = getattr(fact, field_name)
    forged_value = (
        forged_reference
        if field_name == "score_impact_id"
        else tuple(sorted({*current, forged_reference}))
    )
    forged_fact = fact.model_copy(update={field_name: forged_value})
    hypotheses = tuple(
        forged_fact if item.guard_result_id == fact.guard_result_id else item
        for item in payload.hypotheses
    )

    assert not _resign_payload(payload, hypotheses=hypotheses).verify()


@pytest.mark.parametrize(
    ("collection_name", "id_field", "digest_field"),
    (
        ("onsets", "onset_id", "onset_sha256"),
        ("precedences", "precedence_id", "precedence_sha256"),
    ),
)
def test_resigned_temporal_fact_cannot_cross_report_case_scope(
    collection_name: str,
    id_field: str,
    digest_field: str,
) -> None:
    payload = _frozen_context().payload
    collection = getattr(payload, collection_name)
    original = collection[0]
    unsigned = original.model_dump(mode="json", exclude={id_field, digest_field})
    unsigned["opaque_case_id"] = "opq_" + "f" * 32
    digest = sha256_hex(canonical_json_bytes(unsigned))
    prefix = "onset" if collection_name == "onsets" else "precedence"
    crossed = type(original).model_validate(
        {**unsigned, id_field: f"{prefix}_{digest[:24]}", digest_field: digest}
    )
    crossed_collection = tuple(
        sorted(
            (crossed, *collection[1:]),
            key=lambda item: getattr(item, id_field),
        )
    )

    assert not _resign_payload(payload, **{collection_name: crossed_collection}).verify()


def test_resigned_cross_case_and_competition_contradictions_do_not_verify() -> None:
    context = _frozen_context()
    wrong_case = _resign_context(context, opaque_case_id="opq_" + "f" * 32)
    assert not wrong_case.verify()

    payload = context.payload.model_dump(mode="json", exclude={"payload_sha256"})
    competition = dict(payload["competition_deltas"][0])
    competition_unsigned = {
        key: value
        for key, value in competition.items()
        if key not in {"competition_id", "competition_sha256"}
    }
    competition_unsigned["total_score_gap"] = "999.000000000000"
    competition_digest = sha256_hex(canonical_json_bytes(competition_unsigned))
    payload["competition_deltas"][0] = {
        **competition_unsigned,
        "competition_id": "cmp_" + competition_digest[:24],
        "competition_sha256": competition_digest,
    }
    payload_digest = sha256_hex(canonical_json_bytes(payload))
    payload["payload_sha256"] = payload_digest
    forged = _resign_context(
        context,
        payload_json=canonical_json_bytes(payload).decode("utf-8"),
        payload_sha256=payload_digest,
    )
    assert not forged.verify()


def test_external_audit_anchor_rejects_a_fully_resigned_source_substitution() -> None:
    from nurse_rcl.v2.report_context import (
        freeze_report_context_from_audit,
        require_frozen_report_context,
    )

    package = _sealed_package()
    alternate_bundle = build_real_replay_bundle(source_strength=0.4)
    alternate_package = seal_case_audit(
        replay_bundle=alternate_bundle,
        experiment_version="B0/v2-small",
        run_mode="full",
        sealed_at_utc="2026-09-10T11:59:00Z",
    )
    alternate = freeze_report_context_from_audit(
        audit_package=alternate_package,
        expected_audit_sha256=alternate_package.audit_sha256,
        frozen_at_utc="2026-09-10T12:00:00Z",
    )
    substituted = _resign_context(
        alternate,
        source_audit_sha256=package.audit_sha256,
    )

    assert substituted.verify()
    with pytest.raises(ValueError, match="source audit package"):
        require_frozen_report_context(
            substituted,
            audit_package=package,
            expected_audit_sha256=package.audit_sha256,
        )


def test_b0_b1_and_b2_use_one_report_payload_boundary() -> None:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit

    bundle = build_real_replay_bundle()
    contexts = []
    for experiment_version in (
        "B0/v2-small",
        "B1/v2-small+report",
        "B2/v2-hybrid",
    ):
        package = seal_case_audit(
            replay_bundle=bundle,
            experiment_version=experiment_version,
            run_mode="full",
            sealed_at_utc="2026-09-10T11:59:00Z",
        )
        contexts.append(
            freeze_report_context_from_audit(
                audit_package=package,
                expected_audit_sha256=package.audit_sha256,
                frozen_at_utc="2026-09-10T12:00:00Z",
            )
        )

    assert len({item.payload_json for item in contexts}) == 1
    assert len({item.source_ranking_result_sha256 for item in contexts}) == 1
    assert len({item.source_top_k_sha256 for item in contexts}) == 1
    assert tuple(item.experiment_version for item in contexts) == (
        "B0/v2-small",
        "B1/v2-small+report",
        "B2/v2-hybrid",
    )


def test_failure_sealed_audit_has_no_report_context() -> None:
    from nurse_rcl.v2.report_context import freeze_report_context_from_audit
    from tests.test_v2_candidate_invariance import _build_empty_failure

    failure, _, _ = _build_empty_failure()
    package = seal_case_audit(failure_diagnostic=failure)

    with pytest.raises(ValueError, match="successful sealed ranking audit"):
        freeze_report_context_from_audit(
            audit_package=package,
            expected_audit_sha256=package.audit_sha256,
            frozen_at_utc="2026-09-10T12:00:00Z",
        )
