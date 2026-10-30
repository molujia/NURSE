"""Invocation-local live evidence; no old receipt is forged or granted new authority.

Only the public build/replay boundaries grant source authority. Private numeric
cores consume objects freshly replayed in this call, under both-exit snapshots.
"""
from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import event_strength as event
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2 import family_aggregation as family
from nurse_rcl.v2 import role_compiler as roles
from nurse_rcl.v2.diagnostic_schemas import RoleAssertion
from nurse_rcl.v2.live import scoring_selection as selection
from nurse_rcl.v2.no_role_compilation import _representation
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

LIVE_SCORING_CAPACITY_PROFILE = 'live-scoring-6m-256m/1'
LIVE_SCORING_MAX_NODES = 6_000_000
LIVE_SCORING_MAX_JSON_BYTES = 256 * 1024 * 1024


class LiveScoringEvidence(StrictFrozenModel):
    """Explicit live source-bound numeric evidence, not yet a ranking or audit."""
    schema_version: Literal['nurse.v2.live-scoring-evidence/1'] = 'nurse.v2.live-scoring-evidence/1'
    capacity_profile: Literal['live-scoring-6m-256m/1'] = 'live-scoring-6m-256m/1'
    opaque_case_id: str = Field(pattern=r'^opq_[0-9a-f]{32}$')
    selection: selection.LiveScoringSelection
    scoring_path: Literal['accepted_roles', 'no_eligible_roles']
    role_policy_sha256: str
    event_policy_sha256: str
    family_policy_sha256: str
    aggregator_version: str
    aggregator_definition_sha256: str
    role_proposals: tuple[roles.RoleBindingProposal, ...]
    role_assertions: tuple[RoleAssertion, ...]
    group_strengths: tuple[event.EventGroupBaseStrength, ...]
    candidate_event_adjustments: tuple[event.CandidateEventRoleAdjustment, ...]
    retained_assertion_ids: tuple[str, ...]
    unbound_assertion_ids: tuple[str, ...]
    retained_event_group_ids: tuple[str, ...]
    aggregated_event_group_ids: tuple[str, ...]
    unbound_event_group_ids: tuple[str, ...]
    unbound_event_group_audits: tuple[family.UnboundEventGroupAudit, ...]
    family_candidates: tuple[family.CandidateFamilyAggregation, ...]
    provenance_ids: tuple[str, ...]
    result_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def _preflight(value: object) -> tuple[object, int]:
    preflight_exact_pydantic_tree(value,
        root_model_types=(*selection._ROOTS, LiveScoringEvidence, roles.RoleCompilationResult),
        label='live scoring', max_nodes=LIVE_SCORING_MAX_NODES, max_depth=64,
        max_string_length=None, provenance_id_limit=None)
    size = selection._measure(value)
    if size > LIVE_SCORING_MAX_JSON_BYTES:
        raise ValueError('live scoring exceeds the serialization byte budget')
    return legacy._plain(value), size


def _snapshot(value: object) -> tuple[bytes, bytes]:
    plain, size = _preflight(value)
    raw = canonical_json_bytes(plain)
    del plain
    if len(raw) != size:
        raise ValueError('live scoring encoding differs from byte measurement')
    return raw, _representation(value)


def _unchanged(value: object, before: tuple[bytes, bytes]) -> None:
    try:
        now = _snapshot(value)
        same = all(hmac.compare_digest(a, b) for a, b in zip(before, now, strict=True))
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
        same = False
    if not same:
        raise ValueError('live scoring intermediate content or representation changed')


def _compile_families(**kwargs: Any) -> dict[str, object]:
    return family._compile_family_parts(
        **kwargs, implementation=family._AGGREGATOR_IMPLEMENTATIONS[family.REFERENCE_KEY])


def _check_evidence(value: LiveScoringEvidence, sources: selection.LiveScoringSources) -> None:
    """Retain the old event/family closure checks under the explicit live parent."""
    envelope = sources.envelope
    definition = family.resolve_declared_family_aggregator(family.REFERENCE_KEY)
    if (value.opaque_case_id != sources.time_contract.opaque_case_id
            or value.selection.opaque_case_id != value.opaque_case_id
            or value.role_policy_sha256 != roles.ROLE_COMPILATION_POLICY_SHA256
            or value.event_policy_sha256 != event.EVENT_STRENGTH_POLICY_SHA256
            or value.family_policy_sha256 != family.FAMILY_AGGREGATION_POLICY_SHA256
            or value.aggregator_version != definition.aggregator_version
            or value.aggregator_definition_sha256 != definition.definition_sha256):
        raise ValueError('live evidence case or fixed policy differs')
    accepted = bool(value.selection.eligible_guard_result_ids)
    if (value.scoring_path != ('accepted_roles' if accepted else 'no_eligible_roles')
            or (not accepted and (value.role_proposals or value.role_assertions))):
        raise ValueError('live evidence role path differs from complete selection')
    for records, key in (
        (value.role_proposals, lambda row: row.proposal_id),
        (value.role_assertions, lambda row: row.assertion_id),
        (value.group_strengths, lambda row: row.event_group_id),
        (value.candidate_event_adjustments,
            lambda row: (row.candidate_service_key, row.event_group_id)),
        (value.family_candidates, lambda row: row.candidate_service_key),
        (value.unbound_event_group_audits, lambda row: row.event_group_id),
    ):
        keys = tuple(key(row) for row in records)
        if keys != tuple(sorted(set(keys))) or any(not row.verify() for row in records):
            raise ValueError('live evidence child identity or canonical order differs')
    for ids in (value.retained_assertion_ids, value.unbound_assertion_ids,
                value.retained_event_group_ids, value.aggregated_event_group_ids,
                value.unbound_event_group_ids, value.provenance_ids):
        if ids != tuple(sorted(set(ids))):
            raise ValueError('live evidence IDs must be canonical and unique')
    if value.retained_assertion_ids != tuple(row.assertion_id for row in value.role_assertions):
        raise ValueError('live evidence must retain every compiled role assertion')
    bases = {row.strength_id: row for row in value.group_strengths}
    expected_events = {row.event_group_id for row in envelope.event_grouping.event_groups}
    if {row.event_group_id for row in value.group_strengths} != expected_events:
        raise ValueError('live evidence must retain every source event')
    legal_bindings = {(row.service_entity_key, key)
        for row in envelope.candidate_universe.candidates for key in row.event_group_ids}
    if {(row.candidate_service_key, row.event_group_id)
            for row in value.candidate_event_adjustments} != legal_bindings:
        raise ValueError('live evidence must retain every candidate-event binding')
    bound = tuple(key for row in value.candidate_event_adjustments for key in row.retained_assertion_ids)
    unbound = set(value.unbound_assertion_ids)
    if (len(bound) != len(set(bound)) or set(bound) & unbound
            or set(bound) | unbound != set(value.retained_assertion_ids)):
        raise ValueError('live assertion partition does not close')
    for adjustment in value.candidate_event_adjustments:
        base = bases.get(adjustment.base_strength_id)
        if base is None or base.event_group_id != adjustment.event_group_id:
            raise ValueError('live event adjustment base lineage differs')
        expected = None if base.raw_strength is None else base.raw_strength * adjustment.role_adjustment
        if adjustment.adjusted_strength != expected:
            raise ValueError('live event adjustment differs from shared numeric formula')
    candidate_keys = tuple(row.candidate_service_key for row in value.family_candidates)
    if candidate_keys != envelope.frozen_candidates.candidate_service_keys:
        raise ValueError('live scoring must preserve the complete frozen candidate universe')
    aggregated = set(value.aggregated_event_group_ids)
    unbound_events = set(value.unbound_event_group_ids)
    family_events = {key for row in value.family_candidates
        for maximum in row.family_maxima for key in maximum.retained_event_group_ids}
    if (set(value.retained_event_group_ids) != expected_events
            or aggregated & unbound_events or aggregated | unbound_events != expected_events
            or aggregated != family_events
            or aggregated != {row.event_group_id for row in value.candidate_event_adjustments}
            or {row.event_group_id for row in value.unbound_event_group_audits} != unbound_events):
        raise ValueError('live family event partition does not close')
    if any(row.aggregator_version != value.aggregator_version or
           row.aggregator_definition_sha256 != value.aggregator_definition_sha256
           for row in value.family_candidates):
        raise ValueError('live family aggregation declaration differs')
    provenance = {key for row in value.family_candidates for key in row.provenance_ids}
    provenance.update(key for row in value.unbound_event_group_audits for key in row.provenance_ids)
    if set(value.provenance_ids) != provenance:
        raise ValueError('live family provenance does not close')


def _compile_evidence(fresh: selection.LiveScoringSources,
                      replay: selection.LiveScoringReplay) -> LiveScoringEvidence:
    envelope = fresh.envelope
    retained: list[tuple[object, tuple[bytes, bytes]]] = []

    def keep(value: Any) -> Any:
        retained.append((value, _snapshot(value)))
        return value

    try:
        if replay.eligible_guards:
            guards = roles._plain_sequence(replay.eligible_guards, limit=4096, name='live guards')
            roles._plain_sequence(replay.predicates, limit=4096, name='live predicates')
            proposals = keep(roles._default_role_binding_proposals(
                guards=guards, event_grouping=envelope.event_grouping))
            roles._plain_sequence(proposals, limit=16384, name='live role proposals')
            unsigned = roles._compile_role_records_core(
                proposal_values=proposals,
                guards_by_id={g.guard_result_id: g for g in replay.eligible_guards},
                predicates_by_id={p.predicate_result_id: p for p in replay.predicates},
                candidate_universe=envelope.candidate_universe, entity_registry=envelope.entity_registry,
                event_grouping=envelope.event_grouping, mechanism_library=envelope.mechanism_library,
                sessions={})
            identifier, digest = roles._stable_id('rcr', unsigned)
            compiled = keep(roles.RoleCompilationResult.model_validate(
                {**unsigned, 'role_compilation_id': identifier, 'result_sha256': digest}))
            if not compiled.verify():
                raise ValueError('live role core produced invalid records')
            assertions = compiled.assertions
            scoring_path = 'accepted_roles'
        else:
            # Empty roles are explicit in this NEW parent schema. Never construct
            # an old accepted-role/no-role receipt with a live source identity.
            proposals, assertions = (), ()
            scoring_path = 'no_eligible_roles'
            if replay.selection.eligible_guard_result_ids:
                raise ValueError('live no-role selection differs')
        bases = keep(event._compile_base_strengths(envelope.event_grouping))
        if any(not row.verify() for row in bases):
            raise ValueError('live event base core produced invalid records')
        adjustments, unbound_assertions = event._compile_event_adjustments(
            bases=bases, candidate_universe=envelope.candidate_universe, assertions=assertions)
        keep(adjustments)
        if any(not row.verify() for row in adjustments):
            raise ValueError('live event adjustment core produced invalid records')
        parts = keep(_compile_families(event_grouping=envelope.event_grouping,
            group_strengths=bases, candidate_event_adjustments=adjustments,
            candidate_universe=envelope.candidate_universe))
        definition = family.resolve_declared_family_aggregator(family.REFERENCE_KEY)
        unsigned_result = dict(
            schema_version='nurse.v2.live-scoring-evidence/1',
            capacity_profile=LIVE_SCORING_CAPACITY_PROFILE,
            opaque_case_id=fresh.time_contract.opaque_case_id, selection=replay.selection,
            scoring_path=scoring_path, role_policy_sha256=roles.ROLE_COMPILATION_POLICY_SHA256,
            event_policy_sha256=event.EVENT_STRENGTH_POLICY_SHA256,
            family_policy_sha256=family.FAMILY_AGGREGATION_POLICY_SHA256,
            aggregator_version=definition.aggregator_version,
            aggregator_definition_sha256=definition.definition_sha256,
            role_proposals=proposals, role_assertions=assertions,
            group_strengths=bases, candidate_event_adjustments=adjustments,
            retained_assertion_ids=tuple(sorted(a.assertion_id for a in assertions)),
            unbound_assertion_ids=unbound_assertions,
            family_candidates=parts['candidates'],
            **{key: value for key, value in parts.items() if key != 'candidates'})
        _preflight((selection._values(fresh), unsigned_result))
        result = LiveScoringEvidence.model_validate({**unsigned_result,
            'result_sha256': sha256_hex(selection._encoded(unsigned_result))})
        keep(result)
        _check_evidence(result, fresh)
        _preflight((selection._values(fresh), result))
        return result
    finally:
        # Check every generated owner even when a downstream stage raised.
        changed = False
        for value, before in reversed(retained):
            try:
                _unchanged(value, before)
            except ValueError:
                changed = True
        if changed:
            raise ValueError('live scoring intermediate content or representation changed')


def build_live_scoring_evidence(*, sources: selection.LiveScoringSources,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, selection: selection.LiveScoringSelection,
        expected_selection_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveScoringEvidence:
    """Replay the full original case, then consume E evidence in the same call."""
    # Import alias avoids the public argument name shadowing the module.
    from nurse_rcl.v2.live.scoring_selection import _run
    from nurse_rcl.v2.live.scoring_selection import LiveScoringSelection
    if type(selection) is not LiveScoringSelection:
        raise ValueError('live scoring requires an exact current-version selection')
    return cast(LiveScoringEvidence, _run(sources=sources, expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256,
        expected_binding_sha256=expected_binding_sha256, selection=selection,
        expected_selection_sha256=expected_selection_sha256, progress=progress,
        _consume=_compile_evidence))
