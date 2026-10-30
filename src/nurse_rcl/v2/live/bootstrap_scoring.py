"""Complete, invocation-local bootstrap scoring over independently pinned sources.

The explicit live parent is not a legacy ranking authorization. A graph manifest
retains its complete producer records; the live producer must separately replay
the pinned telemetry files. This scorer never claims raw-file authenticity from
manifest hashes, nor treats an observation window as a measured onset.
"""
from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import graph_snapshots as graph
from nurse_rcl.v2 import case_scoring_selection as legacy
from nurse_rcl.v2 import history_scoring as history
from nurse_rcl.v2 import propagation_scoring as propagation
from nurse_rcl.v2 import ranking_schemas, reference_ranking as ranking
from nurse_rcl.v2 import role_compiler as roles, rule_scoring
from nurse_rcl.v2 import temporal_condensation as temporal
from nurse_rcl.v2.live import scoring, scoring_selection as selection_module
from nurse_rcl.v2.no_role_compilation import _representation
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree


class LiveBootstrapContext(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-bootstrap-context/1'] = 'nurse.v2.live-bootstrap-context/1'
    profile: Literal['v2-live-bootstrap'] = 'v2-live-bootstrap'
    dynamic_verification: Literal[False] = False
    pre_ranking_semantics: Literal[False] = False
    history_mode: Literal['off'] = 'off'
    explicit_rules_enabled: Literal[False] = False
    graph_input_manifest: graph.FrozenGraphInputManifest
    graph_input_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    propagation_config: propagation.PropagationScoringConfig
    scoring_config: ranking.ReferenceScoringConfig


class LiveBootstrapScoring(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-bootstrap-scoring/1'] = 'nurse.v2.live-bootstrap-scoring/1'
    capacity_profile: Literal['live-scoring-6m-256m/1'] = 'live-scoring-6m-256m/1'
    profile: Literal['v2-live-bootstrap'] = 'v2-live-bootstrap'
    opaque_case_id: str = Field(pattern=r'^opq_[0-9a-f]{32}$')
    context_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence: scoring.LiveScoringEvidence
    graph_pair: graph.GraphSnapshotPair
    condensations: tuple[temporal.RelationSCCCondensation, ...]
    onsets: tuple[temporal.OnsetInterval, ...]
    precedences: tuple[temporal.TemporalPrecedence, ...]
    temporal_status: Literal['unknown_no_verified_onset'] = 'unknown_no_verified_onset'
    role_compilation: roles.RoleCompilationResult | None
    propagation_relation_bindings: tuple[propagation.PropagationRelationBinding, ...]
    propagation_path_audits: tuple[propagation.PropagationPathAudit, ...]
    propagation_target_scores: tuple[propagation.PropagationTargetScore, ...]
    propagation_scores: tuple[propagation.CandidatePropagationScore, ...]
    rule_scoring: rule_scoring.RuleScoringResult
    history_evaluation: history.HistoryEvaluationResult
    scores: tuple[ranking_schemas.ScoreDecomposition, ...]
    evidence_tie_groups: tuple[ranking_schemas.EvidenceTieGroup, ...]
    ordered_service_keys: tuple[str, ...]
    top_k: ranking_schemas.TopKArtifact
    result_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def _preflight(value: object) -> tuple[object, int]:
    preflight_exact_pydantic_tree(value,
        root_model_types=(*selection_module._ROOTS, LiveBootstrapContext, LiveBootstrapScoring,
                          propagation.PropagationScoringResult),
        label='live bootstrap scoring', max_nodes=scoring.LIVE_SCORING_MAX_NODES,
        max_depth=64, max_string_length=None, provenance_id_limit=None)
    size = selection_module._measure(value)
    if size > scoring.LIVE_SCORING_MAX_JSON_BYTES:
        raise ValueError('live bootstrap scoring exceeds the serialization byte budget')
    return legacy._plain(value), size


def _snapshot(value: object) -> tuple[bytes, bytes]:
    plain, size = _preflight(value)
    raw = canonical_json_bytes(plain)
    del plain
    if len(raw) != size:
        raise ValueError('live bootstrap scoring encoding differs from byte measurement')
    return raw, _representation(value)


def _unchanged(value: object, before: tuple[bytes, bytes]) -> None:
    try:
        same = all(hmac.compare_digest(a, b) for a, b in zip(before, _snapshot(value), strict=True))
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
        same = False
    if not same:
        raise ValueError('live bootstrap content or representation changed')


def live_bootstrap_context_bytes(context: LiveBootstrapContext) -> bytes:
    if type(context) is not LiveBootstrapContext:
        raise ValueError('live bootstrap requires an exact context')
    return _snapshot(context)[0]


def build_live_bootstrap_context(*, graph_input_manifest: graph.FrozenGraphInputManifest,
        expected_graph_input_manifest_sha256: str) -> LiveBootstrapContext:
    before = _snapshot(graph_input_manifest)
    try:
        manifest = graph._require_pinned_graph_input_manifest(graph_input_manifest,
            expected_graph_input_manifest_sha256=expected_graph_input_manifest_sha256)
        return LiveBootstrapContext(graph_input_manifest=manifest,
            graph_input_manifest_sha256=expected_graph_input_manifest_sha256,
            propagation_config=propagation.build_propagation_scoring_config(),
            scoring_config=ranking.build_reference_scoring_config(history_mode='off'))
    finally:
        _unchanged(graph_input_manifest, before)


def _rank(*, fresh: selection_module.LiveScoringSources,
        replay: selection_module.LiveScoringReplay, context: LiveBootstrapContext,
        evidence: scoring.LiveScoringEvidence, keep: Callable[[Any], Any]) -> dict[str, Any]:
    envelope = fresh.envelope
    frozen = envelope.frozen_candidates
    pair = keep(graph.build_graph_snapshot_pair(time_contract=fresh.time_contract,
        candidate_universe=envelope.candidate_universe, frozen_candidates=frozen,
        entity_registry=envelope.entity_registry, graph_input_manifest=context.graph_input_manifest,
        expected_graph_input_manifest_sha256=context.graph_input_manifest_sha256))
    relation_types: tuple[graph.GraphRelationType, ...] = (
        'caller_load_transmission', 'received_dependency_failure')
    condensations = keep(tuple(temporal.build_relation_scc_condensation(
        graph_snapshot=pair.fault_snapshot, relation_type=relation)
        for relation in relation_types))
    # No bootstrap measurement establishes an onset. The complete original graph
    # remains retained, with coverage unknown where the producer cannot prove it.
    compilation = None
    relations: tuple[propagation.PropagationRelationBinding, ...] = ()
    paths: tuple[propagation.PropagationPathAudit, ...] = ()
    targets: tuple[propagation.PropagationTargetScore, ...] = ()
    if replay.eligible_guards:
        # Recreate only the small numeric role record, not its source validation.
        # Its assertions must exactly equal this invocation's evidence channel.
        unsigned = roles._compile_role_records_core(proposal_values=evidence.role_proposals,
            guards_by_id={g.guard_result_id: g for g in replay.eligible_guards},
            predicates_by_id={p.predicate_result_id: p for p in replay.predicates},
            candidate_universe=envelope.candidate_universe, entity_registry=envelope.entity_registry,
            event_grouping=envelope.event_grouping, mechanism_library=envelope.mechanism_library,
            sessions={})
        identifier, digest = roles._stable_id('rcr', unsigned)
        compilation = keep(roles.RoleCompilationResult.model_validate(
            {**unsigned, 'role_compilation_id': identifier, 'result_sha256': digest}))
        if (not compilation.verify() or compilation.assertions != evidence.role_assertions
                or compilation.proposals != evidence.role_proposals):
            raise ValueError('live propagation roles differ from fresh evidence')
        relations = keep(propagation._compile_relation_bindings(
            role_compilation=compilation, guard_results=replay.eligible_guards))
        # Every constituent comes from this call's complete source replay or the
        # original graph/role producer above. This private carrier never escapes.
        verified = propagation._VerifiedInputs(frozen_candidates=frozen, condensations=condensations,
            event_grouping=envelope.event_grouping, onsets=(), precedences=(),
            role_compilation=compilation, role_proposals=evidence.role_proposals,
            role_guard_results=replay.eligible_guards,
            role_predicates=tuple(sorted(replay.predicates, key=lambda p: p.predicate_result_id)),
            candidate_universe=envelope.candidate_universe, entity_registry=envelope.entity_registry,
            mechanism_library=envelope.mechanism_library, scope_interval=envelope.scope_interval,
            relation_bindings=relations, config=context.propagation_config)
        propagated = keep(propagation._score_propagation_from_verified(verified))
        paths, targets, propagation_scores = (propagated.path_audits, propagated.target_scores,
                                             propagated.candidate_scores)
    else:
        propagation_scores = keep(propagation._build_candidate_scores(
            candidate_keys=frozen.candidate_service_keys, source_bundles=(), target_bundles=(),
            target_scores=(), path_audits=()))
    rules = keep(rule_scoring.score_explicit_rules(frozen_candidates=frozen,
        rule_set=ranking_schemas.build_rule_set(rules=()), assignments=(), predicates=()))
    historical = keep(history.evaluate_history(frozen_candidates=frozen, mode='off'))
    # Optional explicit R and H channels are off; this does not disable B0
    # mechanism rules, whose evidence/role effects are already in E and P.
    channel_rows = (evidence.family_candidates, propagation_scores, rules.candidates,
                    historical.candidate_scores)
    if any(tuple(row.candidate_service_key for row in channel) != frozen.candidate_service_keys
           for channel in channel_rows):
        raise ValueError('live ranking channels must each retain complete C in canonical order')
    scores = keep(tuple(ranking._build_candidate_decomposition(
        candidate_service_key=f.candidate_service_key, event_evidence_score=f.evidence_strength,
        propagation_score=p.propagation_score, rule_score=r.rule_score, history_score=h.history_score,
        explicit_rules_enabled=False, config=context.scoring_config,
        event_group_ids=tuple(key for maximum in f.family_maxima for key in maximum.retained_event_group_ids),
        propagation_path_ids=tuple(path.path_audit_id for path in paths
                                  if path.candidate_service_key == f.candidate_service_key),
        rule_ids=tuple(row.contribution_id for row in r.contributions),
        history_neighbor_ids=tuple(row.selected_neighbor_id for row in h.contributions))
        for f, p, r, h in zip(*channel_rows, strict=True)))
    ties = keep(ranking._build_evidence_tie_groups(scores))
    order = ranking._ordered_service_keys(scores=scores,
        tie_break_seed_sha256=context.scoring_config.tie_break_seed_sha256)
    top_k = keep(ranking_schemas.build_top_k(opaque_case_id=frozen.opaque_case_id,
        candidate_universe_sha256=frozen.candidate_universe_sha256, scores=scores,
        tie_groups=ties, ordered_service_keys=order, k=context.scoring_config.top_k,
        tie_break_seed_sha256=context.scoring_config.tie_break_seed_sha256))
    return dict(graph_pair=pair, condensations=condensations, onsets=(), precedences=(),
        temporal_status='unknown_no_verified_onset', role_compilation=compilation,
        propagation_relation_bindings=relations, propagation_path_audits=paths,
        propagation_target_scores=targets, propagation_scores=propagation_scores,
        rule_scoring=rules, history_evaluation=historical, scores=scores,
        evidence_tie_groups=ties, ordered_service_keys=order, top_k=top_k)


def _compile_bootstrap(*, fresh: selection_module.LiveScoringSources,
        replay: selection_module.LiveScoringReplay, context: LiveBootstrapContext,
        context_sha256: str,
        _consume: Callable[[selection_module.LiveScoringSources, LiveBootstrapScoring], Any] | None = None,
        ) -> Any:
    retained: list[tuple[object, tuple[bytes, bytes]]] = []

    def keep(value: Any) -> Any:
        retained.append((value, _snapshot(value)))
        return value

    try:
        evidence = keep(scoring._compile_evidence(fresh, replay))
        parts = _rank(fresh=fresh, replay=replay, context=context, evidence=evidence, keep=keep)
        unsigned = dict(schema_version='nurse.v2.live-bootstrap-scoring/1',
            capacity_profile=scoring.LIVE_SCORING_CAPACITY_PROFILE, profile='v2-live-bootstrap',
            opaque_case_id=fresh.time_contract.opaque_case_id,
            context_sha256=context_sha256, evidence=evidence, **parts)
        _preflight((selection_module._values(fresh), context, unsigned))
        result = keep(LiveBootstrapScoring.model_validate({**unsigned,
            'result_sha256': sha256_hex(selection_module._encoded(unsigned))}))
        if (result.onsets or result.precedences or not result.top_k.verify()
                or any(not row.verify() for row in result.scores)
                or any(not row.verify() for row in result.propagation_scores)):
            raise ValueError('live bootstrap ranking closure failed')
        _preflight((selection_module._values(fresh), context, result))
        return result if _consume is None else _consume(fresh, result)
    finally:
        changed = False
        for value, before in reversed(retained):
            try:
                _unchanged(value, before)
            except ValueError:
                changed = True
        if changed:
            raise ValueError('live bootstrap intermediate content or representation changed')


def build_live_bootstrap_scoring(*, sources: selection_module.LiveScoringSources,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, selection: selection_module.LiveScoringSelection,
        expected_selection_sha256: str, context: LiveBootstrapContext,
        expected_context_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None) -> LiveBootstrapScoring:
    return cast(LiveBootstrapScoring, _run_bootstrap(sources=sources,
        expected_source_records_sha256=expected_source_records_sha256,
        expected_rule_set_sha256=expected_rule_set_sha256,
        expected_binding_sha256=expected_binding_sha256, selection=selection,
        expected_selection_sha256=expected_selection_sha256, context=context,
        expected_context_sha256=expected_context_sha256, progress=progress))


def _run_bootstrap(*, sources: selection_module.LiveScoringSources,
        expected_source_records_sha256: str, expected_rule_set_sha256: str,
        expected_binding_sha256: str, selection: selection_module.LiveScoringSelection,
        expected_selection_sha256: str, context: LiveBootstrapContext,
        expected_context_sha256: str,
        progress: Callable[[dict[str, Any]], None] | None = None,
        _consume: Callable[[selection_module.LiveScoringSources, LiveBootstrapScoring], Any] | None = None,
        ) -> Any:
    if type(context) is not LiveBootstrapContext or type(selection) is not selection_module.LiveScoringSelection:
        raise ValueError('live bootstrap requires exact context and selection')
    selection_module._pin(expected_context_sha256)
    _preflight((selection_module._values(sources), selection, context))
    original = _snapshot(context)
    try:
        if not hmac.compare_digest(sha256_hex(original[0]), expected_context_sha256):
            raise ValueError('live bootstrap independent context pin mismatch')
        fresh_context = LiveBootstrapContext.model_validate_json(original[0])
        fresh_before = _snapshot(fresh_context)
        try:
            if (fresh_before[0] != original[0]
                    or not fresh_context.propagation_config.verify()
                    or not fresh_context.scoring_config.verify()
                    or fresh_context.scoring_config.history_mode != 'off'):
                raise ValueError('live bootstrap context differs from fixed policy')

            def consume(fresh: selection_module.LiveScoringSources,
                        replay: selection_module.LiveScoringReplay) -> Any:
                return _compile_bootstrap(fresh=fresh, replay=replay, context=fresh_context,
                    context_sha256=expected_context_sha256, _consume=_consume)

            return cast(LiveBootstrapScoring, selection_module._run(sources=sources,
                expected_source_records_sha256=expected_source_records_sha256,
                expected_rule_set_sha256=expected_rule_set_sha256,
                expected_binding_sha256=expected_binding_sha256, selection=selection,
                expected_selection_sha256=expected_selection_sha256, progress=progress,
                _consume=consume))
        finally:
            _unchanged(fresh_context, fresh_before)
    finally:
        _unchanged(context, original)
