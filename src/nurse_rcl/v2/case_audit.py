"""Immutable, typed, replayable case audits for NURSE V2."""

from __future__ import annotations

import hmac
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeVar, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    FrozenCandidateUniverse,
    require_frozen_candidate_universe,
)
from nurse_rcl.v2.candidate_universe import (
    EmptyCandidateUniverseFailure,
    build_complete_candidate_universe,
)
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.event_strength import EventStrengthResult
from nurse_rcl.v2.family_aggregation import FamilyAggregationResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    SemanticFocusSet,
    TimeInterval,
)
from nurse_rcl.v2.propagation_scoring import PropagationScoringResult
from nurse_rcl.v2.query_session import (
    CaseQuerySessionCoordinator,
    CaseQuerySessionResult,
    InMemoryCaseQuerySessionHeadStore,
    require_case_query_session_replay,
)
from nurse_rcl.v2.ranking_schemas import HistoryMode, RuleSet, TopKArtifact
from nurse_rcl.v2.reference_ranking import (
    RankingInputAuthority,
    ReferenceRankingResult,
    ReferenceScoringConfig,
    rank_reference_candidates,
)
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.rule_scoring import RuleCandidateAssignment, RuleScoringResult
from nurse_rcl.v2.runtime_schemas import (
    AttemptLedger,
    CaseCheckpoint,
    CostRecord,
    RunMode,
    SealedAuditPackage,
    SemanticCall,
    build_sealed_audit_package,
)
from nurse_rcl.v2.semantic_context import DetailedEvidenceCard, SemanticContextBundle
from nurse_rcl.v2.temporal_condensation import (
    OnsetInterval,
    RelationSCCCondensation,
    TemporalPrecedence,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION: Literal["nurse.v2.case-evidence-replay-bundle/1"] = (
    "nurse.v2.case-evidence-replay-bundle/1"
)
CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION: Literal["nurse.v2.case-ranking-replay-bundle/1"] = (
    "nurse.v2.case-ranking-replay-bundle/1"
)
CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION: Literal["nurse.v2.case-audit-lineage-index/1"] = (
    "nurse.v2.case-audit-lineage-index/1"
)
IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.immutable-case-audit-package/1"] = (
    "nurse.v2.immutable-case-audit-package/1"
)

_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _exact_model(value: object, model_type: type[_ModelT], *, label: str) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact {model_type.__name__} type")
    try:
        rebuilt = model_type.model_validate(
            model_type.model_dump(value, mode="json", warnings="error")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} failed exact schema validation") from error
    if type(rebuilt) is not model_type or rebuilt != value:
        raise ValueError(f"{label} is not a canonical exact model")
    return rebuilt


def _exact_models(
    values: object,
    model_type: type[_ModelT],
    *,
    label: str,
    sort_key: str,
) -> tuple[_ModelT, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError(f"{label} must use an exact tuple or list")
    sequence = cast(tuple[object, ...] | list[object], values)
    rebuilt = tuple(_exact_model(item, model_type, label=f"{label} item") for item in sequence)
    return tuple(sorted(rebuilt, key=lambda item: cast(str, getattr(item, sort_key))))


def _verify_model(value: object, model_type: type[_ModelT], *, label: str) -> _ModelT:
    rebuilt = _exact_model(value, model_type, label=label)
    verify = getattr(rebuilt, "verify", None)
    if not callable(verify) or verify() is not True:
        raise ValueError(f"{label} failed content verification")
    return rebuilt


def _clone_authority(authority: object) -> RankingInputAuthority:
    if type(authority) is not RankingInputAuthority:
        raise ValueError("ranking authority must use the exact RankingInputAuthority type")
    return RankingInputAuthority(
        **{
            item.name: object.__getattribute__(authority, item.name)
            for item in fields(RankingInputAuthority)
        }
    )


def _evidence_unsigned(bundle: CaseEvidenceReplayBundle) -> dict[str, object]:
    return bundle.model_dump(mode="json", exclude={"evidence_sha256"}, warnings="error")


def _detailed_card_closes_to_evidence(
    card: DetailedEvidenceCard,
    *,
    evidence: CaseEvidenceReplayBundle,
) -> bool:
    observation_by_id = {item.observation_id: item for item in evidence.event_grouping.observations}
    event_group_by_id = {item.event_group_id: item for item in evidence.event_grouping.event_groups}
    binding_by_id = {
        item.binding_id: item
        for item in (
            *evidence.entity_registry.timeline_bindings,
            *(item.binding for item in evidence.entity_registry.resolutions),
        )
    }
    return (
        card.verify()
        and card.card.opaque_case_id == evidence.time_contract.opaque_case_id
        and card.card.candidate_universe_sha256 == evidence.candidate_universe.universe_sha256
        and card.card.service_entity_key in set(evidence.frozen_candidates.candidate_service_keys)
        and all(observation_by_id.get(item.observation_id) == item for item in card.observations)
        and all(event_group_by_id.get(item.event_group_id) == item for item in card.event_groups)
        and all(binding_by_id.get(item.binding_id) == item for item in card.entity_bindings)
    )


def _verified_query_runtime(
    evidence: CaseEvidenceReplayBundle,
) -> tuple[
    dict[str, CaseQuerySessionResult],
    CaseQuerySessionCoordinator | None,
]:
    context = evidence.semantic_context
    policy = evidence.card_expansion_policy
    if (context is None) != (policy is None):
        raise ValueError("semantic context and card expansion policy must be sealed together")
    if context is not None:
        if type(context) is not SemanticContextBundle or not context.verify():
            raise ValueError("semantic context failed exact verification")
        directory = context.candidate_directory
        if (
            context.opaque_case_id != evidence.time_contract.opaque_case_id
            or context.candidate_freeze_sha256 != evidence.frozen_candidates.freeze_sha256
            or context.entity_registry_sha256 != evidence.entity_registry.registry_sha256
            or context.event_grouping_result_sha256 != evidence.event_grouping.result_sha256
            or directory.candidate_universe_sha256 != evidence.candidate_universe.universe_sha256
            or tuple(item.service_entity_key for item in directory.entries)
            != evidence.frozen_candidates.candidate_service_keys
        ):
            raise ValueError("semantic context is not closed to the frozen evidence")
        focus_by_service = {item.service_entity_key: item for item in evidence.focus_set.decisions}
        for entry in directory.entries:
            focus = focus_by_service.get(entry.service_entity_key)
            if focus is None or (
                entry.in_initial_focus,
                entry.focus_selection_phase,
                entry.focus_selection_reasons,
                entry.focus_exclusion_reason,
            ) != (
                focus.selected,
                focus.selection_phase,
                focus.selection_reasons,
                focus.exclusion_reason,
            ):
                raise ValueError("semantic context does not project the sealed focus set")
        if any(
            not _detailed_card_closes_to_evidence(item, evidence=evidence)
            for item in context.detailed_evidence_cards
        ):
            raise ValueError("semantic context card is not closed to the frozen evidence")
    if policy is not None and (type(policy) is not CardExpansionPolicy or not policy.verify()):
        raise ValueError("card expansion policy failed exact verification")

    head_store = InMemoryCaseQuerySessionHeadStore()
    coordinator = CaseQuerySessionCoordinator(head_store=head_store)
    prior_sha256: str | None = None
    for session in evidence.query_sessions:
        if (
            type(session) is not CaseQuerySessionResult
            or not session.verify()
            or session.opaque_case_id != evidence.time_contract.opaque_case_id
            or session.candidate_freeze_sha256 != evidence.frozen_candidates.freeze_sha256
            or not head_store.compare_and_swap(
                opaque_case_id=session.opaque_case_id,
                candidate_freeze_sha256=session.candidate_freeze_sha256,
                expected_session_sha256=prior_sha256,
                new_session=session,
            )
        ):
            raise ValueError("query sessions do not form one authoritative chain")
        if session.card_expansion_result is not None:
            if context is None or policy is None:
                raise ValueError("card query session is missing its frozen replay inputs")
            if any(
                not _detailed_card_closes_to_evidence(item, evidence=evidence)
                for item in session.card_expansion_result.returned_cards
            ):
                raise ValueError("expanded card is not closed to the frozen evidence")
        require_case_query_session_replay(
            session=session,
            candidate_universe=evidence.candidate_universe,
            frozen_candidates=evidence.frozen_candidates,
            entity_registry=evidence.entity_registry,
            event_grouping=evidence.event_grouping,
            semantic_context=context,
            card_expansion_policy=policy,
        )
        prior_sha256 = session.session_sha256

    session_by_sha256 = {item.session_sha256: item for item in evidence.query_sessions}
    final_session = evidence.query_sessions[-1] if evidence.query_sessions else None
    mapping: dict[str, CaseQuerySessionResult] = {}
    for guard in evidence.role_guard_results:
        provenance_ids = {item for decision in guard.decisions for item in decision.provenance_ids}
        if guard.score_impact is not None:
            provenance_ids.update(guard.score_impact.provenance_ids)
        matched_session_hashes = provenance_ids & set(session_by_sha256)
        if len(matched_session_hashes) > 1:
            raise ValueError("guard references multiple query session heads")
        hypothesis = guard.parsed_hypothesis
        required_predicate_ids = (
            ()
            if hypothesis is None
            else (
                *hypothesis.necessary_predicate_result_ids,
                *hypothesis.optional_predicate_result_ids,
                *hypothesis.refuting_predicate_result_ids,
            )
        )
        predicate_by_id = {item.predicate_result_id: item for item in evidence.role_predicates}
        verification_result_ids = {
            result_id
            for predicate_id in required_predicate_ids
            for result_id in predicate_by_id[predicate_id].verification_result_ids
        }
        verification_request_ids = (
            set() if hypothesis is None else set(hypothesis.verification_request_ids)
        )
        if (verification_result_ids or verification_request_ids) and not matched_session_hashes:
            raise ValueError("verified guard is missing its authoritative query session")
        if matched_session_hashes:
            session = session_by_sha256[next(iter(matched_session_hashes))]
            if final_session is None or session != final_session:
                raise ValueError("final guards must reference the sealed query-session head")
            verification = session.verification_result
            if verification is None:
                raise ValueError("guard references a query session without verification results")
            if not verification_request_ids <= {
                item.verification_request_id for item in verification.results
            } or not verification_result_ids <= {
                item.verification_result_id for item in verification.results
            }:
                raise ValueError("guard query references are absent from the sealed session")
            mapping[guard.guard_result_id] = session
    return mapping, (coordinator if mapping else None)


class CaseEvidenceReplayBundle(StrictFrozenModel):
    """Exact upstream evidence required to replay E and P channels."""

    schema_version: Literal["nurse.v2.case-evidence-replay-bundle/1"] = (
        CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION
    )
    time_contract: CaseTimeContract
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    focus_set: SemanticFocusSet
    semantic_context: SemanticContextBundle | None
    card_expansion_policy: CardExpansionPolicy | None
    query_sessions: tuple[CaseQuerySessionResult, ...]
    mechanism_library: MechanismLibrary
    scope_interval: TimeInterval
    role_predicates: tuple[PredicateResult, ...]
    role_guard_results: tuple[HypothesisGuardResult, ...]
    role_proposals: tuple[RoleBindingProposal, ...]
    role_compilation: RoleCompilationResult
    event_strength: EventStrengthResult
    graph_snapshots: tuple[GraphSnapshot, ...] = Field(min_length=1)
    condensations: tuple[RelationSCCCondensation, ...] = Field(min_length=1)
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_sequences(self) -> Self:
        ordered_fields = (
            (self.role_predicates, "predicate_result_id"),
            (self.query_sessions, "sequence_number"),
            (self.role_guard_results, "guard_result_id"),
            (self.role_proposals, "proposal_id"),
            (self.graph_snapshots, "snapshot_sha256"),
            (self.condensations, "condensation_id"),
            (self.onsets, "onset_id"),
            (self.precedences, "precedence_id"),
        )
        for values, key in ordered_fields:
            if values != tuple(sorted(values, key=lambda item: getattr(item, key))):
                raise ValueError(f"case evidence {key} records must be canonically ordered")
        if len({item.sequence_number for item in self.query_sessions}) != len(self.query_sessions):
            raise ValueError("query session sequence numbers must be unique")
        return self

    def verify_inputs(self) -> bool:
        try:
            rebuilt = CaseEvidenceReplayBundle.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if type(self) is not CaseEvidenceReplayBundle or rebuilt != self:
                return False
            if rebuilt.evidence_sha256 != _digest(_evidence_unsigned(rebuilt)):
                return False
            opaque_case_id = rebuilt.time_contract.opaque_case_id
            if any(
                value != opaque_case_id
                for value in (
                    rebuilt.entity_registry.opaque_case_id,
                    rebuilt.event_grouping.opaque_case_id,
                    rebuilt.candidate_universe.opaque_case_id,
                    rebuilt.frozen_candidates.opaque_case_id,
                    rebuilt.focus_set.opaque_case_id,
                    rebuilt.role_compilation.opaque_case_id,
                    rebuilt.event_strength.opaque_case_id,
                    *(item.opaque_case_id for item in rebuilt.graph_snapshots),
                    *(item.opaque_case_id for item in rebuilt.condensations),
                )
            ):
                return False
            if (
                rebuilt.time_contract.decision_time_unix_ns
                != rebuilt.entity_registry.decision_time_unix_ns
                or rebuilt.time_contract.decision_time_unix_ns
                != rebuilt.event_grouping.decision_time_unix_ns
            ):
                return False
            if not (
                rebuilt.entity_registry.verify()
                and rebuilt.event_grouping.verify()
                and rebuilt.candidate_universe.verify()
                and rebuilt.frozen_candidates.verify()
                and rebuilt.focus_set.verify()
                and rebuilt.mechanism_library.verify()
                and rebuilt.role_compilation.verify()
                and rebuilt.event_strength.verify()
            ):
                return False
            registry_observations = tuple(
                sorted(
                    (item.observation for item in rebuilt.entity_registry.resolutions),
                    key=lambda item: item.observation_id,
                )
            )
            if registry_observations != rebuilt.event_grouping.observations:
                return False
            replayed_universe = build_complete_candidate_universe(
                entity_registry=rebuilt.entity_registry,
                event_grouping=rebuilt.event_grouping,
                scope_interval=rebuilt.scope_interval,
            )
            if replayed_universe != rebuilt.candidate_universe:
                return False
            require_frozen_candidate_universe(
                frozen=rebuilt.frozen_candidates,
                candidate_universe=rebuilt.candidate_universe,
            )
            if (
                rebuilt.focus_set.candidate_universe_sha256
                != rebuilt.candidate_universe.universe_sha256
                or set(rebuilt.focus_set.selected_service_keys)
                - set(rebuilt.frozen_candidates.candidate_service_keys)
            ):
                return False
            query_sessions_by_guard_result_id, query_session_coordinator = _verified_query_runtime(
                rebuilt
            )
            if any(not item.verify() for item in rebuilt.role_predicates):
                return False
            predicate_by_id = {item.predicate_result_id: item for item in rebuilt.role_predicates}
            if len(predicate_by_id) != len(rebuilt.role_predicates):
                return False
            for guard in rebuilt.role_guard_results:
                if not guard.verify():
                    return False
                hypothesis = guard.parsed_hypothesis
                if hypothesis is not None and hypothesis.opaque_case_id != opaque_case_id:
                    return False
                required_ids = (
                    ()
                    if hypothesis is None
                    else (
                        *hypothesis.necessary_predicate_result_ids,
                        *hypothesis.optional_predicate_result_ids,
                        *hypothesis.refuting_predicate_result_ids,
                    )
                )
                if any(item not in predicate_by_id for item in required_ids):
                    return False
                guard_predicates = tuple(
                    predicate_by_id[item] for item in sorted(set(required_ids))
                )
                if not guard.verify_inputs(
                    submitted_record=json.loads(guard.submitted_record_json),
                    predicates=guard_predicates,
                    candidate_universe=rebuilt.candidate_universe,
                    frozen_candidates=rebuilt.frozen_candidates,
                    entity_registry=rebuilt.entity_registry,
                    event_grouping=rebuilt.event_grouping,
                    mechanism_library=rebuilt.mechanism_library,
                    scope_interval=rebuilt.scope_interval,
                    query_session=query_sessions_by_guard_result_id.get(guard.guard_result_id),
                    query_session_coordinator=query_session_coordinator,
                    semantic_context=rebuilt.semantic_context,
                    card_expansion_policy=rebuilt.card_expansion_policy,
                ):
                    return False
            if not rebuilt.role_compilation.verify_inputs(
                proposals=rebuilt.role_proposals,
                guard_results=rebuilt.role_guard_results,
                predicates=rebuilt.role_predicates,
                candidate_universe=rebuilt.candidate_universe,
                frozen_candidates=rebuilt.frozen_candidates,
                entity_registry=rebuilt.entity_registry,
                event_grouping=rebuilt.event_grouping,
                mechanism_library=rebuilt.mechanism_library,
                scope_interval=rebuilt.scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=rebuilt.semantic_context,
                card_expansion_policy=rebuilt.card_expansion_policy,
            ):
                return False
            if not rebuilt.event_strength.verify_inputs(
                role_compilation=rebuilt.role_compilation,
                proposals=rebuilt.role_proposals,
                guard_results=rebuilt.role_guard_results,
                predicates=rebuilt.role_predicates,
                candidate_universe=rebuilt.candidate_universe,
                frozen_candidates=rebuilt.frozen_candidates,
                entity_registry=rebuilt.entity_registry,
                event_grouping=rebuilt.event_grouping,
                mechanism_library=rebuilt.mechanism_library,
                scope_interval=rebuilt.scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=rebuilt.semantic_context,
                card_expansion_policy=rebuilt.card_expansion_policy,
            ):
                return False
            graph_by_sha = {item.snapshot_sha256: item for item in rebuilt.graph_snapshots}
            if len(graph_by_sha) != len(rebuilt.graph_snapshots):
                return False
            for graph in rebuilt.graph_snapshots:
                if not graph.verify() or (
                    graph.observation_contract_sha256
                    != rebuilt.candidate_universe.observation_contract_sha256
                    or graph.candidate_universe_sha256 != rebuilt.candidate_universe.universe_sha256
                    or graph.candidate_freeze_sha256 != rebuilt.frozen_candidates.freeze_sha256
                    or graph.entity_registry_sha256 != rebuilt.entity_registry.registry_sha256
                    or graph.node_service_keys != rebuilt.frozen_candidates.candidate_service_keys
                ):
                    return False
            for condensation in rebuilt.condensations:
                selected_graph = graph_by_sha.get(condensation.graph_snapshot_sha256)
                if selected_graph is None or not condensation.verify_inputs(
                    graph_snapshot=selected_graph
                ):
                    return False
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return True


def build_case_evidence_replay_bundle(
    *,
    time_contract: CaseTimeContract,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    focus_set: SemanticFocusSet,
    semantic_context: SemanticContextBundle | None,
    card_expansion_policy: CardExpansionPolicy | None,
    query_sessions: Sequence[CaseQuerySessionResult],
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    role_predicates: Sequence[PredicateResult],
    role_guard_results: Sequence[HypothesisGuardResult],
    role_proposals: Sequence[RoleBindingProposal],
    role_compilation: RoleCompilationResult,
    event_strength: EventStrengthResult,
    graph_snapshots: Sequence[GraphSnapshot],
    condensations: Sequence[RelationSCCCondensation],
    onsets: Sequence[OnsetInterval],
    precedences: Sequence[TemporalPrecedence],
) -> CaseEvidenceReplayBundle:
    unsigned: dict[str, object] = {
        "schema_version": CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION,
        "time_contract": _exact_model(
            time_contract, CaseTimeContract, label="time contract"
        ).model_dump(mode="json"),
        "entity_registry": _verify_model(
            entity_registry, TemporalEntityRegistry, label="entity registry"
        ).model_dump(mode="json"),
        "event_grouping": _verify_model(
            event_grouping, EventGroupingResult, label="event grouping"
        ).model_dump(mode="json"),
        "candidate_universe": _verify_model(
            candidate_universe, CandidateUniverse, label="candidate universe"
        ).model_dump(mode="json"),
        "frozen_candidates": _verify_model(
            frozen_candidates, FrozenCandidateUniverse, label="candidate freeze"
        ).model_dump(mode="json"),
        "focus_set": _verify_model(focus_set, SemanticFocusSet, label="focus set").model_dump(
            mode="json"
        ),
        "semantic_context": (
            None
            if semantic_context is None
            else _verify_model(
                semantic_context,
                SemanticContextBundle,
                label="semantic context",
            ).model_dump(mode="json")
        ),
        "card_expansion_policy": (
            None
            if card_expansion_policy is None
            else _verify_model(
                card_expansion_policy,
                CardExpansionPolicy,
                label="card expansion policy",
            ).model_dump(mode="json")
        ),
        "query_sessions": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                query_sessions,
                CaseQuerySessionResult,
                label="query sessions",
                sort_key="sequence_number",
            )
        ),
        "mechanism_library": _verify_model(
            mechanism_library, MechanismLibrary, label="mechanism library"
        ).model_dump(mode="json"),
        "scope_interval": _exact_model(
            scope_interval, TimeInterval, label="scope interval"
        ).model_dump(mode="json"),
        "role_predicates": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                role_predicates,
                PredicateResult,
                label="role predicates",
                sort_key="predicate_result_id",
            )
        ),
        "role_guard_results": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                role_guard_results,
                HypothesisGuardResult,
                label="role guard results",
                sort_key="guard_result_id",
            )
        ),
        "role_proposals": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                role_proposals,
                RoleBindingProposal,
                label="role proposals",
                sort_key="proposal_id",
            )
        ),
        "role_compilation": _verify_model(
            role_compilation, RoleCompilationResult, label="role compilation"
        ).model_dump(mode="json"),
        "event_strength": _verify_model(
            event_strength, EventStrengthResult, label="event strength"
        ).model_dump(mode="json"),
        "graph_snapshots": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                graph_snapshots,
                GraphSnapshot,
                label="graph snapshots",
                sort_key="snapshot_sha256",
            )
        ),
        "condensations": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                condensations,
                RelationSCCCondensation,
                label="SCC condensations",
                sort_key="condensation_id",
            )
        ),
        "onsets": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                onsets,
                OnsetInterval,
                label="onsets",
                sort_key="onset_id",
            )
        ),
        "precedences": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                precedences,
                TemporalPrecedence,
                label="precedences",
                sort_key="precedence_id",
            )
        ),
    }
    result = CaseEvidenceReplayBundle.model_validate(
        {**unsigned, "evidence_sha256": _digest(unsigned)}
    )
    if not result.verify_inputs():
        raise ValueError("case evidence replay bundle failed input closure verification")
    return result


class CaseRankingReplayBundle(StrictFrozenModel):
    """Exact E/P/R/H inputs and stored reference ranking for one case."""

    schema_version: Literal["nurse.v2.case-ranking-replay-bundle/1"] = (
        CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION
    )
    evidence: CaseEvidenceReplayBundle
    family_aggregation: FamilyAggregationResult
    propagation_scoring: PropagationScoringResult
    rule_set: RuleSet
    rule_assignments: tuple[RuleCandidateAssignment, ...]
    rule_predicates: tuple[PredicateResult, ...]
    rule_scoring: RuleScoringResult
    history_mode: HistoryMode
    history_evaluation: HistoryEvaluationResult
    ranking_authority: RankingInputAuthority
    scoring_config: ReferenceScoringConfig
    ranking_result: ReferenceRankingResult
    top_k: TopKArtifact
    semantic_calls: tuple[SemanticCall, ...]
    attempt_ledgers: tuple[AttemptLedger, ...]
    costs: tuple[CostRecord, ...]
    checkpoints: tuple[CaseCheckpoint, ...]
    degradation_reason_codes: tuple[str, ...]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_initial_history_contract(self) -> Self:
        if self.history_mode != "off":
            raise ValueError("case audit /1 currently requires replayable history_mode=off")
        if self.degradation_reason_codes != tuple(sorted(set(self.degradation_reason_codes))):
            raise ValueError("degradation reason codes must be sorted and unique")
        if self.semantic_calls != tuple(
            sorted(self.semantic_calls, key=lambda item: item.semantic_call_id)
        ):
            raise ValueError("semantic calls must be canonically ordered")
        if self.attempt_ledgers != tuple(
            sorted(self.attempt_ledgers, key=lambda item: item.logical_request_id)
        ):
            raise ValueError("attempt ledgers must be canonically ordered")
        if self.costs != tuple(sorted(self.costs, key=lambda item: item.cost_sha256)):
            raise ValueError("cost records must be canonically ordered")
        if self.checkpoints != tuple(sorted(self.checkpoints, key=lambda item: item.sequence)):
            raise ValueError("case checkpoints must be canonically ordered")
        unique_fields = (
            (
                tuple(item.semantic_call_id for item in self.semantic_calls),
                "semantic call IDs",
            ),
            (
                tuple(item.logical_request_id for item in self.attempt_ledgers),
                "logical request IDs",
            ),
            (tuple(item.cost_sha256 for item in self.costs), "cost record digests"),
            (tuple(item.sequence for item in self.checkpoints), "checkpoint sequences"),
        )
        for values, label in unique_fields:
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self

    def verify_inputs(self) -> bool:
        try:
            if type(self) is not CaseRankingReplayBundle:
                return False
            rebuilt = CaseRankingReplayBundle.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(mode="json", exclude={"bundle_sha256"}, warnings="error")
            if rebuilt.bundle_sha256 != _digest(unsigned):
                return False
            evidence = rebuilt.evidence
            if not evidence.verify_inputs():
                return False
            query_sessions_by_guard_result_id, query_session_coordinator = _verified_query_runtime(
                evidence
            )
            if not rebuilt.family_aggregation.verify_inputs(
                event_strength=evidence.event_strength,
                aggregator_version=rebuilt.family_aggregation.aggregator_version,
                role_compilation=evidence.role_compilation,
                proposals=evidence.role_proposals,
                guard_results=evidence.role_guard_results,
                predicates=evidence.role_predicates,
                candidate_universe=evidence.candidate_universe,
                frozen_candidates=evidence.frozen_candidates,
                entity_registry=evidence.entity_registry,
                event_grouping=evidence.event_grouping,
                mechanism_library=evidence.mechanism_library,
                scope_interval=evidence.scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=evidence.semantic_context,
                card_expansion_policy=evidence.card_expansion_policy,
            ):
                return False
            if not rebuilt.propagation_scoring.verify_inputs(
                expected_result_sha256=rebuilt.propagation_scoring.result_sha256,
                expected_authoritative_input_sha256=(
                    rebuilt.propagation_scoring.authoritative_input_sha256
                ),
                frozen_candidates=evidence.frozen_candidates,
                condensations=evidence.condensations,
                event_grouping=evidence.event_grouping,
                onsets=evidence.onsets,
                precedences=evidence.precedences,
                role_compilation=evidence.role_compilation,
                role_proposals=evidence.role_proposals,
                role_guard_results=evidence.role_guard_results,
                role_predicates=evidence.role_predicates,
                candidate_universe=evidence.candidate_universe,
                entity_registry=evidence.entity_registry,
                mechanism_library=evidence.mechanism_library,
                scope_interval=evidence.scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=evidence.semantic_context,
                card_expansion_policy=evidence.card_expansion_policy,
            ):
                return False
            if not rebuilt.rule_scoring.verify_trusted(
                expected_result_sha256=rebuilt.rule_scoring.result_sha256,
                expected_authoritative_input_sha256=(
                    rebuilt.rule_scoring.authoritative_input_sha256
                ),
                frozen_candidates=evidence.frozen_candidates,
                rule_set=rebuilt.rule_set,
                assignments=rebuilt.rule_assignments,
                predicates=rebuilt.rule_predicates,
            ):
                return False
            if not rebuilt.history_evaluation.verify_trusted(
                expected_result_sha256=rebuilt.history_evaluation.result_sha256,
                expected_authoritative_input_sha256=(
                    rebuilt.history_evaluation.authoritative_input_sha256
                ),
                frozen_candidates=evidence.frozen_candidates,
                mode=rebuilt.history_mode,
            ):
                return False
            if type(rebuilt.ranking_authority) is not RankingInputAuthority:
                return False
            if not (
                rebuilt.family_aggregation.verify()
                and rebuilt.propagation_scoring.verify_structure()
                and rebuilt.rule_set.verify()
                and rebuilt.rule_scoring.verify_structure()
                and rebuilt.history_evaluation.verify_structure()
                and rebuilt.scoring_config.verify()
                and rebuilt.ranking_result.verify_structure()
                and rebuilt.top_k.verify()
            ):
                return False
            recomputed = rank_reference_candidates(
                frozen_candidates=evidence.frozen_candidates,
                family_aggregation=rebuilt.family_aggregation,
                propagation_scoring=rebuilt.propagation_scoring,
                rule_scoring=rebuilt.rule_scoring,
                history_evaluation=rebuilt.history_evaluation,
                authority=rebuilt.ranking_authority,
                config=rebuilt.scoring_config,
            )
            if (
                recomputed != rebuilt.ranking_result
                or rebuilt.top_k != recomputed.top_k
                or rebuilt.ranking_result.top_k != rebuilt.top_k
            ):
                return False
            if any(not item.verify() for item in rebuilt.semantic_calls):
                return False
            if any(not item.verify() for item in rebuilt.attempt_ledgers):
                return False
            if any(not item.verify() for item in rebuilt.costs):
                return False
            if any(not item.verify() for item in rebuilt.checkpoints):
                return False
            opaque_case_id = evidence.time_contract.opaque_case_id
            if any(item.opaque_case_id != opaque_case_id for item in rebuilt.semantic_calls):
                return False
            if any(item.opaque_case_id != opaque_case_id for item in rebuilt.checkpoints):
                return False
            embedded_ledger_ids = tuple(
                item.attempt_ledger.logical_request_id for item in rebuilt.semantic_calls
            )
            if len(embedded_ledger_ids) != len(set(embedded_ledger_ids)):
                return False
            expected_ledgers = tuple(
                sorted(
                    (item.attempt_ledger for item in rebuilt.semantic_calls),
                    key=lambda item: item.logical_request_id,
                )
            )
            if rebuilt.attempt_ledgers != expected_ledgers:
                return False
            embedded_costs = {item.cost.cost_sha256: item.cost for item in rebuilt.semantic_calls}
            expected_costs = tuple(
                sorted(embedded_costs.values(), key=lambda item: item.cost_sha256)
            )
            if rebuilt.costs != expected_costs:
                return False
            ledger_ids = {item.logical_request_id for item in rebuilt.attempt_ledgers}
            if any(not set(item.logical_request_ids) <= ledger_ids for item in rebuilt.checkpoints):
                return False
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return True


def build_case_ranking_replay_bundle(
    *,
    evidence: CaseEvidenceReplayBundle,
    family_aggregation: FamilyAggregationResult,
    propagation_scoring: PropagationScoringResult,
    rule_set: RuleSet,
    rule_assignments: Sequence[RuleCandidateAssignment],
    rule_predicates: Sequence[PredicateResult],
    rule_scoring: RuleScoringResult,
    history_mode: HistoryMode,
    history_evaluation: HistoryEvaluationResult,
    ranking_authority: RankingInputAuthority,
    scoring_config: ReferenceScoringConfig,
    ranking_result: ReferenceRankingResult,
    top_k: TopKArtifact,
    semantic_calls: Sequence[SemanticCall],
    attempt_ledgers: Sequence[AttemptLedger],
    costs: Sequence[CostRecord],
    checkpoints: Sequence[CaseCheckpoint],
    degradation_reason_codes: Sequence[str],
) -> CaseRankingReplayBundle:
    if type(degradation_reason_codes) not in {tuple, list} or any(
        type(item) is not str or not item for item in degradation_reason_codes
    ):
        raise ValueError("degradation reason codes must use a plain string sequence")
    exact_evidence = _exact_model(
        evidence, CaseEvidenceReplayBundle, label="case evidence replay bundle"
    )
    if not exact_evidence.verify_inputs():
        raise ValueError("case evidence replay bundle failed verification")
    unsigned: dict[str, object] = {
        "schema_version": CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION,
        "evidence": exact_evidence.model_dump(mode="json"),
        "family_aggregation": _verify_model(
            family_aggregation, FamilyAggregationResult, label="family aggregation"
        ).model_dump(mode="json"),
        "propagation_scoring": _exact_model(
            propagation_scoring,
            PropagationScoringResult,
            label="propagation scoring",
        ).model_dump(mode="json"),
        "rule_set": _verify_model(rule_set, RuleSet, label="rule set").model_dump(mode="json"),
        "rule_assignments": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                rule_assignments,
                RuleCandidateAssignment,
                label="rule assignments",
                sort_key="assignment_id",
            )
        ),
        "rule_predicates": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                rule_predicates,
                PredicateResult,
                label="rule predicates",
                sort_key="predicate_result_id",
            )
        ),
        "rule_scoring": _exact_model(
            rule_scoring, RuleScoringResult, label="rule scoring"
        ).model_dump(mode="json"),
        "history_mode": history_mode,
        "history_evaluation": _exact_model(
            history_evaluation,
            HistoryEvaluationResult,
            label="history evaluation",
        ).model_dump(mode="json"),
        "ranking_authority": {
            item.name: object.__getattribute__(_clone_authority(ranking_authority), item.name)
            for item in fields(RankingInputAuthority)
        },
        "scoring_config": _verify_model(
            scoring_config, ReferenceScoringConfig, label="scoring config"
        ).model_dump(mode="json"),
        "ranking_result": _exact_model(
            ranking_result, ReferenceRankingResult, label="ranking result"
        ).model_dump(mode="json"),
        "top_k": _verify_model(top_k, TopKArtifact, label="Top-K").model_dump(mode="json"),
        "semantic_calls": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                semantic_calls,
                SemanticCall,
                label="semantic calls",
                sort_key="semantic_call_id",
            )
        ),
        "attempt_ledgers": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                attempt_ledgers,
                AttemptLedger,
                label="attempt ledgers",
                sort_key="logical_request_id",
            )
        ),
        "costs": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                costs,
                CostRecord,
                label="cost records",
                sort_key="cost_sha256",
            )
        ),
        "checkpoints": tuple(
            item.model_dump(mode="json")
            for item in _exact_models(
                checkpoints,
                CaseCheckpoint,
                label="case checkpoints",
                sort_key="sequence",
            )
        ),
        "degradation_reason_codes": tuple(sorted(set(degradation_reason_codes))),
    }
    result = CaseRankingReplayBundle.model_validate(
        {**unsigned, "bundle_sha256": _digest(unsigned)}
    )
    if not result.verify_inputs():
        raise ValueError("case ranking replay bundle failed input closure verification")
    return result


class CaseAuditLineageNode(StrictFrozenModel):
    """One mechanically derived record and its bidirectional lineage edges."""

    record_id: str = Field(min_length=1)
    record_kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: str = Field(min_length=1)
    score_affecting: bool
    source_record_ids: tuple[str, ...]
    consumer_record_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_canonical_edges(self) -> Self:
        for values, label in (
            (self.source_record_ids, "source record IDs"),
            (self.consumer_record_ids, "consumer record IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
            if self.record_id in values:
                raise ValueError("lineage node cannot reference itself")
        return self


class CaseAuditLineageIndex(StrictFrozenModel):
    """Closed source/consumer graph rebuilt only from a verified replay bundle."""

    schema_version: Literal["nurse.v2.case-audit-lineage-index/1"] = (
        CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    replay_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[CaseAuditLineageNode, ...] = Field(min_length=1)
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_nodes(self) -> Self:
        if self.nodes != tuple(sorted(self.nodes, key=lambda item: item.record_id)):
            raise ValueError("lineage nodes must be canonically ordered")
        if len({item.record_id for item in self.nodes}) != len(self.nodes):
            raise ValueError("lineage record IDs must be unique")
        return self

    def verify(self) -> bool:
        try:
            if type(self) is not CaseAuditLineageIndex:
                return False
            rebuilt = CaseAuditLineageIndex.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(mode="json", exclude={"lineage_sha256"}, warnings="error")
            if rebuilt.lineage_sha256 != _digest(unsigned):
                return False
            node_by_id = {item.record_id: item for item in rebuilt.nodes}
            if len(node_by_id) != len(rebuilt.nodes):
                return False
            for node in rebuilt.nodes:
                for source_id in node.source_record_ids:
                    source = node_by_id.get(source_id)
                    if source is None or node.record_id not in source.consumer_record_ids:
                        return False
                for consumer_id in node.consumer_record_ids:
                    consumer = node_by_id.get(consumer_id)
                    if consumer is None or node.record_id not in consumer.source_record_ids:
                        return False
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def verify_against(self, replay_bundle: CaseRankingReplayBundle) -> bool:
        """Rebuild the full graph so a newly signed forged index still fails."""

        try:
            if type(replay_bundle) is not CaseRankingReplayBundle or not self.verify():
                return False
            expected = build_case_audit_lineage_index(replay_bundle)
        except (AttributeError, TypeError, ValueError):
            return False
        return self == expected


@dataclass(frozen=True)
class _PendingLineageRecord:
    record_kind: str
    record: StrictFrozenModel
    disposition: str
    source_record_ids: tuple[str, ...]


def _candidate_record_id(service_key: str) -> str:
    return f"candidate:{service_key}"


def _graph_record_id(snapshot_sha256: str) -> str:
    return f"graph:{snapshot_sha256}"


def _top_k_entry_record_id(*, rank: int, service_key: str) -> str:
    return f"topk:{rank}:{service_key}"


def _canonical_record_sha256(record: StrictFrozenModel) -> str:
    return _digest(record.model_dump(mode="json", warnings="error"))


def _add_pending_lineage_record(
    records: dict[str, _PendingLineageRecord],
    *,
    record_id: str,
    record_kind: str,
    record: StrictFrozenModel,
    disposition: str,
    source_record_ids: Sequence[str] = (),
) -> None:
    pending = _PendingLineageRecord(
        record_kind=record_kind,
        record=record,
        disposition=disposition,
        source_record_ids=tuple(sorted(set(source_record_ids))),
    )
    previous = records.setdefault(record_id, pending)
    if previous != pending:
        raise ValueError(f"conflicting lineage record ID: {record_id}")


def _predicate_records(
    predicates: Sequence[PredicateResult],
) -> tuple[PredicateResult, ...]:
    by_id: dict[str, PredicateResult] = {}
    stack = list(predicates)
    while stack:
        predicate = stack.pop()
        previous = by_id.setdefault(predicate.predicate_result_id, predicate)
        if previous != predicate:
            raise ValueError("conflicting predicate result ID in lineage input")
        stack.extend(predicate.operand_states)
    return tuple(sorted(by_id.values(), key=lambda item: item.predicate_result_id))


def _compile_case_audit_lineage_records(
    bundle: CaseRankingReplayBundle,
) -> dict[str, _PendingLineageRecord]:
    evidence = bundle.evidence
    records: dict[str, _PendingLineageRecord] = {}
    event_groups = evidence.event_grouping.event_groups
    event_group_ids = {item.event_group_id for item in event_groups}
    observation_ids = {item.observation_id for item in evidence.event_grouping.observations}
    candidate_ids = {
        item.service_entity_key: _candidate_record_id(item.service_entity_key)
        for item in evidence.candidate_universe.candidates
    }

    for observation in evidence.event_grouping.observations:
        _add_pending_lineage_record(
            records,
            record_id=observation.observation_id,
            record_kind="observation",
            record=observation,
            disposition=observation.evidence_state,
        )
    for event_group in event_groups:
        _add_pending_lineage_record(
            records,
            record_id=event_group.event_group_id,
            record_kind="event_group",
            record=event_group,
            disposition=event_group.family_assignment_status,
            source_record_ids=event_group.observation_ids,
        )
    for candidate_scope in evidence.candidate_universe.candidates:
        _add_pending_lineage_record(
            records,
            record_id=candidate_ids[candidate_scope.service_entity_key],
            record_kind="candidate_scope",
            record=candidate_scope,
            disposition=candidate_scope.evidence_state,
            source_record_ids=(
                *candidate_scope.observation_ids,
                *candidate_scope.event_group_ids,
            ),
        )

    for ledger in bundle.attempt_ledgers:
        ledger_disposition = (
            "succeeded"
            if ledger.primary_response_sha256 is not None
            else "exhausted"
            if ledger.exhausted
            else "pending"
        )
        _add_pending_lineage_record(
            records,
            record_id=ledger.logical_request_id,
            record_kind="attempt_ledger",
            record=ledger,
            disposition=ledger_disposition,
        )
    for cost in bundle.costs:
        _add_pending_lineage_record(
            records,
            record_id=f"cost:{cost.cost_sha256}",
            record_kind="cost_record",
            record=cost,
            disposition=cost.source,
        )
    semantic_call_id_by_sha256 = {
        item.call_sha256: item.semantic_call_id for item in bundle.semantic_calls
    }
    for semantic_call in bundle.semantic_calls:
        _add_pending_lineage_record(
            records,
            record_id=semantic_call.semantic_call_id,
            record_kind="semantic_call",
            record=semantic_call,
            disposition=semantic_call.status,
            source_record_ids=(
                semantic_call.attempt_ledger.logical_request_id,
                f"cost:{semantic_call.cost.cost_sha256}",
            ),
        )
    for checkpoint in bundle.checkpoints:
        _add_pending_lineage_record(
            records,
            record_id=f"checkpoint:{checkpoint.checkpoint_sha256}",
            record_kind="case_checkpoint",
            record=checkpoint,
            disposition=checkpoint.state,
            source_record_ids=(
                *checkpoint.logical_request_ids,
                *(
                    semantic_call_id_by_sha256[item]
                    for item in checkpoint.produced_artifact_hashes.values()
                    if item in semantic_call_id_by_sha256
                ),
            ),
        )

    semantic_context_id: str | None = None
    if evidence.semantic_context is not None:
        semantic_context_id = f"semantic_context:{evidence.semantic_context.bundle_sha256}"
        _add_pending_lineage_record(
            records,
            record_id=semantic_context_id,
            record_kind="semantic_context",
            record=evidence.semantic_context,
            disposition="frozen",
            source_record_ids=tuple(candidate_ids.values()),
        )
    card_policy_id: str | None = None
    if evidence.card_expansion_policy is not None:
        card_policy_id = f"card_policy:{evidence.card_expansion_policy.policy_sha256}"
        _add_pending_lineage_record(
            records,
            record_id=card_policy_id,
            record_kind="card_expansion_policy",
            record=evidence.card_expansion_policy,
            disposition="frozen",
        )
    query_session_id_by_sha256 = {
        item.session_sha256: item.query_session_id for item in evidence.query_sessions
    }
    for session in evidence.query_sessions:
        session_sources: list[str] = []
        card_batch = session.card_expansion_result
        if card_batch is not None:
            for card_decision in card_batch.decisions:
                card_request = card_decision.request
                request_sources = [candidate_ids[card_request.service_entity_key]]
                if semantic_context_id is not None:
                    request_sources.append(semantic_context_id)
                if card_policy_id is not None:
                    request_sources.append(card_policy_id)
                _add_pending_lineage_record(
                    records,
                    record_id=card_request.expansion_request_id,
                    record_kind="card_expansion_request",
                    record=card_request,
                    disposition="requested",
                    source_record_ids=request_sources,
                )
            card_id_by_sha256: dict[str, str] = {}
            for card in card_batch.returned_cards:
                card_id = f"evidence_card:{card.detailed_card_sha256}"
                card_id_by_sha256[card.detailed_card_sha256] = card_id
                _add_pending_lineage_record(
                    records,
                    record_id=card_id,
                    record_kind="expanded_evidence_card",
                    record=card,
                    disposition=card.card.access_reason,
                    source_record_ids=(
                        candidate_ids[card.card.service_entity_key],
                        *(item.observation_id for item in card.observations),
                        *(item.event_group_id for item in card.event_groups),
                    ),
                )
            for card_decision in card_batch.decisions:
                decision_sources = [card_decision.request.expansion_request_id]
                if card_policy_id is not None:
                    decision_sources.append(card_policy_id)
                if card_decision.detailed_card_sha256 is not None:
                    decision_sources.append(card_id_by_sha256[card_decision.detailed_card_sha256])
                _add_pending_lineage_record(
                    records,
                    record_id=card_decision.expansion_decision_id,
                    record_kind="card_expansion_decision",
                    record=card_decision,
                    disposition=card_decision.status,
                    source_record_ids=decision_sources,
                )
            card_batch_id = f"card_batch:{card_batch.result_sha256}"
            _add_pending_lineage_record(
                records,
                record_id=card_batch_id,
                record_kind="card_expansion_batch",
                record=card_batch,
                disposition="completed",
                source_record_ids=(
                    *(item.expansion_decision_id for item in card_batch.decisions),
                    *card_id_by_sha256.values(),
                ),
            )
            session_sources.append(card_batch_id)
        verification = session.verification_result
        if verification is not None:
            verification_policy_id = f"verification_policy:{verification.policy.policy_sha256}"
            verification_scope_id = f"verification_scope:{verification.scope.scope_sha256}"
            _add_pending_lineage_record(
                records,
                record_id=verification_policy_id,
                record_kind="verification_policy",
                record=verification.policy,
                disposition="frozen",
            )
            _add_pending_lineage_record(
                records,
                record_id=verification_scope_id,
                record_kind="verification_scope",
                record=verification.scope,
                disposition="frozen",
                source_record_ids=(verification_policy_id, *candidate_ids.values()),
            )
            for verification_request in verification.requests:
                _add_pending_lineage_record(
                    records,
                    record_id=verification_request.verification_request_id,
                    record_kind="verification_request",
                    record=verification_request,
                    disposition="requested",
                    source_record_ids=(
                        verification_policy_id,
                        verification_scope_id,
                        candidate_ids[verification_request.candidate_service_key],
                        *(
                            item
                            for item in verification_request.observation_ids
                            if item in observation_ids
                        ),
                    ),
                )
            for derived in verification.derived_observations:
                _add_pending_lineage_record(
                    records,
                    record_id=derived.derived_observation_id,
                    record_kind="derived_verification_observation",
                    record=derived,
                    disposition="derived",
                    source_record_ids=(
                        derived.verification_request_id,
                        *(
                            item
                            for item in derived.source_observation_ids
                            if item in observation_ids
                        ),
                    ),
                )
            for result in verification.results:
                _add_pending_lineage_record(
                    records,
                    record_id=result.verification_result_id,
                    record_kind="verification_result",
                    record=result,
                    disposition=result.status,
                    source_record_ids=(
                        *(
                            (result.verification_request_id,)
                            if result.verification_request_id in records
                            else ()
                        ),
                        *result.derived_observation_ids,
                    ),
                )
            for execution_decision in verification.decisions:
                _add_pending_lineage_record(
                    records,
                    record_id=execution_decision.execution_decision_id,
                    record_kind="verification_execution_decision",
                    record=execution_decision,
                    disposition=execution_decision.reason_code,
                    source_record_ids=(
                        *(
                            (execution_decision.verification_request_id,)
                            if execution_decision.verification_request_id in records
                            else ()
                        ),
                        execution_decision.verification_result_id,
                        verification_policy_id,
                        verification_scope_id,
                    ),
                )
            verification_batch_id = f"verification_batch:{verification.batch_sha256}"
            _add_pending_lineage_record(
                records,
                record_id=verification_batch_id,
                record_kind="verification_batch",
                record=verification,
                disposition="completed",
                source_record_ids=(
                    *(item.execution_decision_id for item in verification.decisions),
                    *(item.verification_result_id for item in verification.results),
                ),
            )
            session_sources.append(verification_batch_id)
        if session.prior_session_sha256 is not None:
            session_sources.append(query_session_id_by_sha256[session.prior_session_sha256])
        _add_pending_lineage_record(
            records,
            record_id=session.query_session_id,
            record_kind="case_query_session",
            record=session,
            disposition=session.phase,
            source_record_ids=session_sources,
        )

    predicates = _predicate_records((*evidence.role_predicates, *bundle.rule_predicates))
    predicate_ids = {item.predicate_result_id for item in predicates}
    for predicate in predicates:
        _add_pending_lineage_record(
            records,
            record_id=predicate.predicate_result_id,
            record_kind="predicate_result",
            record=predicate,
            disposition=predicate.truth_value,
            source_record_ids=(
                *(item.predicate_result_id for item in predicate.operand_states),
                *(item for item in predicate.observation_ids if item in observation_ids),
                *(item for item in predicate.verification_result_ids if item in records),
            ),
        )

    hypothesis_by_id = {
        item.parsed_hypothesis.hypothesis_id: item.parsed_hypothesis
        for item in evidence.role_guard_results
        if item.parsed_hypothesis is not None
    }
    semantic_call_record_ids = {item.semantic_call_id for item in bundle.semantic_calls}
    for hypothesis in hypothesis_by_id.values():
        _add_pending_lineage_record(
            records,
            record_id=hypothesis.hypothesis_id,
            record_kind="diagnostic_hypothesis",
            record=hypothesis,
            disposition=hypothesis.lifecycle_status,
            source_record_ids=(
                candidate_ids[hypothesis.candidate_service_key],
                *(item for item in hypothesis.event_group_ids if item in event_group_ids),
                *(
                    item
                    for item in (
                        *hypothesis.necessary_predicate_result_ids,
                        *hypothesis.optional_predicate_result_ids,
                        *hypothesis.refuting_predicate_result_ids,
                    )
                    if item in predicate_ids
                ),
                *(item for item in hypothesis.verification_request_ids if item in records),
                *(item for item in hypothesis.provenance_ids if item in semantic_call_record_ids),
                *(
                    semantic_call_id_by_sha256[item]
                    for item in hypothesis.provenance_ids
                    if item in semantic_call_id_by_sha256
                ),
            ),
        )

    score_impact_by_guard_id: dict[str, str] = {}
    for guard in evidence.role_guard_results:
        parsed_hypothesis = guard.parsed_hypothesis
        guard_sources: tuple[str, ...] = ()
        if parsed_hypothesis is not None:
            guard_sources = (
                parsed_hypothesis.hypothesis_id,
                *(item for item in parsed_hypothesis.event_group_ids if item in event_group_ids),
                *(
                    item
                    for item in (
                        *parsed_hypothesis.necessary_predicate_result_ids,
                        *parsed_hypothesis.optional_predicate_result_ids,
                        *parsed_hypothesis.refuting_predicate_result_ids,
                    )
                    if item in predicate_ids
                ),
                *(item for item in parsed_hypothesis.verification_request_ids if item in records),
            )
        previous_decision_id: str | None = None
        for guard_decision in guard.decisions:
            sources: tuple[str, ...] = guard_sources
            if previous_decision_id is not None:
                sources = (*sources, previous_decision_id)
            _add_pending_lineage_record(
                records,
                record_id=guard_decision.guard_decision_id,
                record_kind="guard_decision",
                record=guard_decision,
                disposition=guard_decision.disposition,
                source_record_ids=sources,
            )
            previous_decision_id = guard_decision.guard_decision_id
        if guard.score_impact is not None:
            impact = guard.score_impact
            score_impact_by_guard_id[guard.guard_result_id] = impact.score_impact_id
            _add_pending_lineage_record(
                records,
                record_id=impact.score_impact_id,
                record_kind="score_impact_compilation",
                record=impact,
                disposition="compiled",
                source_record_ids=(
                    impact.hypothesis_id,
                    *(item.guard_decision_id for item in guard.decisions),
                    *(item for item in impact.event_group_ids if item in event_group_ids),
                    *(
                        item
                        for item in (
                            *impact.necessary_predicate_result_ids,
                            *impact.optional_predicate_result_ids,
                            *impact.refuting_predicate_result_ids,
                        )
                        if item in predicate_ids
                    ),
                    *(item for item in impact.verification_request_ids if item in records),
                ),
            )
        result_sources = tuple(item.guard_decision_id for item in guard.decisions)
        if guard.parsed_hypothesis is not None:
            result_sources = (guard.parsed_hypothesis.hypothesis_id, *result_sources)
        if guard.score_impact is not None:
            result_sources = (*result_sources, guard.score_impact.score_impact_id)
        _add_pending_lineage_record(
            records,
            record_id=guard.guard_result_id,
            record_kind="hypothesis_guard_result",
            record=guard,
            disposition=guard.final_disposition,
            source_record_ids=result_sources,
        )

    proposals_by_id = {item.proposal_id: item for item in evidence.role_proposals}
    proposal_ids = set(proposals_by_id)
    for proposal in evidence.role_proposals:
        proposal_sources: tuple[str, ...] = (
            proposal.guard_result_id,
            proposal.event_group_id,
        )
        impact_id = score_impact_by_guard_id.get(proposal.guard_result_id)
        if impact_id is not None:
            proposal_sources = (*proposal_sources, impact_id)
        _add_pending_lineage_record(
            records,
            record_id=proposal.proposal_id,
            record_kind="role_binding_proposal",
            record=proposal,
            disposition="proposed",
            source_record_ids=proposal_sources,
        )
    for assertion in evidence.role_compilation.assertions:
        _add_pending_lineage_record(
            records,
            record_id=assertion.assertion_id,
            record_kind="role_assertion",
            record=assertion,
            disposition=assertion.lifecycle_status,
            source_record_ids=(
                assertion.event_group_id,
                *(item for item in assertion.predicate_result_ids if item in predicate_ids),
                *(item for item in assertion.provenance_ids if item in proposal_ids),
                *(
                    guard.guard_result_id
                    for guard in evidence.role_guard_results
                    if guard.parsed_hypothesis is not None
                    and guard.parsed_hypothesis.hypothesis_id == assertion.hypothesis_id
                ),
            ),
        )

    graph_ids_by_sha: dict[str, str] = {}
    for graph in evidence.graph_snapshots:
        graph_id = _graph_record_id(graph.snapshot_sha256)
        graph_ids_by_sha[graph.snapshot_sha256] = graph_id
        _add_pending_lineage_record(
            records,
            record_id=graph_id,
            record_kind="graph_snapshot",
            record=graph,
            disposition=f"{graph.window_kind}:{graph.coverage.coverage_state}",
            source_record_ids=tuple(candidate_ids.values()),
        )
    for condensation in evidence.condensations:
        _add_pending_lineage_record(
            records,
            record_id=condensation.condensation_id,
            record_kind="scc_condensation",
            record=condensation,
            disposition="retained",
            source_record_ids=(graph_ids_by_sha[condensation.graph_snapshot_sha256],),
        )
    for onset in evidence.onsets:
        _add_pending_lineage_record(
            records,
            record_id=onset.onset_id,
            record_kind="onset_interval",
            record=onset,
            disposition="bounded",
            source_record_ids=tuple(
                item for item in onset.provenance_ids if item in observation_ids
            ),
        )
    for precedence in evidence.precedences:
        _add_pending_lineage_record(
            records,
            record_id=precedence.precedence_id,
            record_kind="temporal_precedence",
            record=precedence,
            disposition=precedence.truth_value,
            source_record_ids=tuple(
                item
                for item in (
                    precedence.source_onset_id,
                    precedence.target_onset_id,
                )
                if item is not None
            ),
        )

    for strength in evidence.event_strength.group_strengths:
        _add_pending_lineage_record(
            records,
            record_id=strength.strength_id,
            record_kind="event_group_base_strength",
            record=strength,
            disposition=strength.status,
            source_record_ids=(
                strength.event_group_id,
                *strength.retained_observation_ids,
            ),
        )
    for adjustment in evidence.event_strength.candidate_event_adjustments:
        _add_pending_lineage_record(
            records,
            record_id=adjustment.adjustment_id,
            record_kind="candidate_event_role_adjustment",
            record=adjustment,
            disposition=("computed" if adjustment.adjusted_strength is not None else "unknown"),
            source_record_ids=(
                adjustment.base_strength_id,
                *adjustment.retained_assertion_ids,
            ),
        )

    adjustment_by_candidate_event = {
        (item.candidate_service_key, item.event_group_id): item.adjustment_id
        for item in evidence.event_strength.candidate_event_adjustments
    }
    family_by_candidate: dict[str, str] = {}
    for family_candidate in bundle.family_aggregation.candidates:
        for maximum in family_candidate.family_maxima:
            _add_pending_lineage_record(
                records,
                record_id=maximum.family_maximum_id,
                record_kind="signal_family_maximum",
                record=maximum,
                disposition=maximum.status,
                source_record_ids=tuple(
                    adjustment_by_candidate_event[(maximum.candidate_service_key, event_group_id)]
                    for event_group_id in maximum.retained_event_group_ids
                    if (
                        maximum.candidate_service_key,
                        event_group_id,
                    )
                    in adjustment_by_candidate_event
                ),
            )
        family_by_candidate[family_candidate.candidate_service_key] = (
            family_candidate.candidate_aggregation_id
        )
        _add_pending_lineage_record(
            records,
            record_id=family_candidate.candidate_aggregation_id,
            record_kind="candidate_family_aggregation",
            record=family_candidate,
            disposition=family_candidate.evidence_status,
            source_record_ids=(
                candidate_ids[family_candidate.candidate_service_key],
                *(item.family_maximum_id for item in family_candidate.family_maxima),
            ),
        )

    assertion_ids = {item.assertion_id for item in evidence.role_compilation.assertions}
    binding_ids_by_assertion: dict[str, list[str]] = {}
    for binding in bundle.propagation_scoring.relation_bindings:
        binding_ids_by_assertion.setdefault(binding.assertion_id, []).append(binding.binding_id)
        _add_pending_lineage_record(
            records,
            record_id=binding.binding_id,
            record_kind="propagation_relation_binding",
            record=binding,
            disposition=binding.role,
            source_record_ids=(binding.assertion_id,),
        )
    for path in bundle.propagation_scoring.path_audits:
        for hop in path.hops:
            _add_pending_lineage_record(
                records,
                record_id=hop.hop_audit_id,
                record_kind="propagation_hop_audit",
                record=hop,
                disposition=hop.coverage_status,
                source_record_ids=(path.condensation_id,),
            )
        path_assertion_ids = (
            *path.source_assertion_ids,
            *path.target_assertion_ids,
        )
        _add_pending_lineage_record(
            records,
            record_id=path.path_audit_id,
            record_kind="propagation_path_audit",
            record=path,
            disposition=path.selection_status,
            source_record_ids=(
                candidate_ids[path.candidate_service_key],
                path.source_event_group_id,
                path.target_event_group_id,
                path.condensation_id,
                *(item.hop_audit_id for item in path.hops),
                *(item for item in path_assertion_ids if item in assertion_ids),
                *(
                    binding_id
                    for assertion_id in path_assertion_ids
                    for binding_id in binding_ids_by_assertion.get(assertion_id, ())
                ),
                *(
                    item
                    for item in (
                        path.source_onset_id,
                        path.target_onset_id,
                        path.precedence_id,
                    )
                    if item is not None
                ),
                *(item for item in path.source_binding_observation_ids),
                *(item for item in path.target_binding_observation_ids),
            ),
        )
    propagation_by_candidate: dict[str, str] = {}
    for target in bundle.propagation_scoring.target_scores:
        _add_pending_lineage_record(
            records,
            record_id=target.target_score_id,
            record_kind="propagation_target_score",
            record=target,
            disposition=target.status,
            source_record_ids=(
                candidate_ids[target.candidate_service_key],
                target.target_event_group_id,
                *target.retained_path_audit_ids,
            ),
        )
    for propagation_candidate in bundle.propagation_scoring.candidate_scores:
        propagation_by_candidate[propagation_candidate.candidate_service_key] = (
            propagation_candidate.candidate_score_id
        )
        _add_pending_lineage_record(
            records,
            record_id=propagation_candidate.candidate_score_id,
            record_kind="candidate_propagation_score",
            record=propagation_candidate,
            disposition=propagation_candidate.reason_code,
            source_record_ids=(
                candidate_ids[propagation_candidate.candidate_service_key],
                *propagation_candidate.retained_target_score_ids,
            ),
        )

    rules_by_key = {item.rule_key: item for item in bundle.rule_set.rules}
    for rule in bundle.rule_set.rules:
        _add_pending_lineage_record(
            records,
            record_id=rule.rule_sha256,
            record_kind="scoring_rule",
            record=rule,
            disposition="enabled" if rule.enabled else "disabled",
            source_record_ids=tuple(
                item for item in rule.predicate_result_ids if item in predicate_ids
            ),
        )
    for assignment in bundle.rule_assignments:
        _add_pending_lineage_record(
            records,
            record_id=assignment.assignment_id,
            record_kind="rule_candidate_assignment",
            record=assignment,
            disposition="assigned",
            source_record_ids=(
                candidate_ids[assignment.candidate_service_key],
                assignment.rule_sha256,
            ),
        )
    rule_by_candidate: dict[str, str] = {}
    for rule_candidate in bundle.rule_scoring.candidates:
        for rule_contribution in rule_candidate.contributions:
            matched_rule = rules_by_key.get(rule_contribution.rule_key)
            rule_source: tuple[str, ...] = (
                () if matched_rule is None else (matched_rule.rule_sha256,)
            )
            _add_pending_lineage_record(
                records,
                record_id=rule_contribution.contribution_id,
                record_kind="rule_contribution",
                record=rule_contribution,
                disposition=(
                    f"{rule_contribution.activation_status}:selected"
                    if rule_contribution.selected_for_direction_max
                    else rule_contribution.activation_status
                ),
                source_record_ids=(
                    rule_contribution.assignment_id,
                    *rule_source,
                    *(
                        item
                        for item in rule_contribution.predicate_result_ids
                        if item in predicate_ids
                    ),
                ),
            )
        rule_by_candidate[rule_candidate.candidate_service_key] = (
            rule_candidate.candidate_rule_score_id
        )
        _add_pending_lineage_record(
            records,
            record_id=rule_candidate.candidate_rule_score_id,
            record_kind="candidate_rule_score",
            record=rule_candidate,
            disposition="computed",
            source_record_ids=(
                candidate_ids[rule_candidate.candidate_service_key],
                *(item.contribution_id for item in rule_candidate.contributions),
            ),
        )

    history_neighbor_ids = {item.audit_id for item in bundle.history_evaluation.neighbor_audits}
    for neighbor in bundle.history_evaluation.neighbor_audits:
        _add_pending_lineage_record(
            records,
            record_id=neighbor.audit_id,
            record_kind="history_neighbor_audit",
            record=neighbor,
            disposition=neighbor.status,
        )
    history_by_candidate: dict[str, str] = {}
    for history_candidate in bundle.history_evaluation.candidate_scores:
        for history_contribution in history_candidate.contributions:
            _add_pending_lineage_record(
                records,
                record_id=history_contribution.contribution_id,
                record_kind="history_score_contribution",
                record=history_contribution,
                disposition=history_contribution.mapping_status,
                source_record_ids=(
                    candidate_ids[history_contribution.candidate_service_key],
                    *(
                        (history_contribution.selected_neighbor_id,)
                        if history_contribution.selected_neighbor_id in history_neighbor_ids
                        else ()
                    ),
                ),
            )
        history_by_candidate[history_candidate.candidate_service_key] = (
            history_candidate.candidate_history_score_id
        )
        _add_pending_lineage_record(
            records,
            record_id=history_candidate.candidate_history_score_id,
            record_kind="candidate_history_score",
            record=history_candidate,
            disposition="computed",
            source_record_ids=(
                candidate_ids[history_candidate.candidate_service_key],
                *(item.contribution_id for item in history_candidate.contributions),
            ),
        )

    score_by_sha = {item.score_sha256: item for item in bundle.ranking_result.scores}
    for score in bundle.ranking_result.scores:
        _add_pending_lineage_record(
            records,
            record_id=score.score_sha256,
            record_kind="score_decomposition",
            record=score,
            disposition="computed",
            source_record_ids=(
                family_by_candidate[score.candidate_service_key],
                propagation_by_candidate[score.candidate_service_key],
                rule_by_candidate[score.candidate_service_key],
                history_by_candidate[score.candidate_service_key],
            ),
        )
    for tie_group in bundle.ranking_result.evidence_tie_groups:
        _add_pending_lineage_record(
            records,
            record_id=tie_group.tie_group_id,
            record_kind="evidence_tie_group",
            record=tie_group,
            disposition="evidence_tie",
            source_record_ids=tie_group.member_score_sha256,
        )
    for entry in bundle.top_k.entries:
        entry_id = _top_k_entry_record_id(
            rank=entry.rank,
            service_key=entry.service_entity_key,
        )
        if entry.score_sha256 not in score_by_sha:
            raise ValueError("Top-K entry references an unknown score decomposition")
        _add_pending_lineage_record(
            records,
            record_id=entry_id,
            record_kind="top_k_entry",
            record=entry,
            disposition="ranked",
            source_record_ids=(entry.score_sha256, entry.evidence_tie_group_id),
        )
    return records


def build_case_audit_lineage_index(
    replay_bundle: CaseRankingReplayBundle,
) -> CaseAuditLineageIndex:
    """Compile complete lineage without accepting caller-provided provenance."""

    bundle = _exact_model(
        replay_bundle,
        CaseRankingReplayBundle,
        label="case ranking replay bundle",
    )
    if not bundle.verify_inputs():
        raise ValueError("case ranking replay bundle failed verification")
    pending = _compile_case_audit_lineage_records(bundle)
    consumers: dict[str, set[str]] = {record_id: set() for record_id in pending}
    for record_id, item in pending.items():
        for source_id in item.source_record_ids:
            if source_id not in pending:
                raise ValueError(
                    f"lineage record {record_id} references unknown source {source_id}"
                )
            consumers[source_id].add(record_id)
    nodes = tuple(
        CaseAuditLineageNode(
            record_id=record_id,
            record_kind=item.record_kind,
            record_sha256=_canonical_record_sha256(item.record),
            disposition=item.disposition,
            score_affecting=True,
            source_record_ids=item.source_record_ids,
            consumer_record_ids=tuple(sorted(consumers[record_id])),
        )
        for record_id, item in sorted(pending.items())
    )
    unsigned: dict[str, object] = {
        "schema_version": CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
        "opaque_case_id": bundle.evidence.time_contract.opaque_case_id,
        "replay_bundle_sha256": bundle.bundle_sha256,
        "nodes": tuple(item.model_dump(mode="json") for item in nodes),
    }
    result = CaseAuditLineageIndex.model_validate({**unsigned, "lineage_sha256": _digest(unsigned)})
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("case audit lineage index verification failed")
    return result


_MAX_AUDIT_JSON_BYTES = 64 * 1024 * 1024
_MAX_AUDIT_JSON_DEPTH = 64
_MAX_AUDIT_JSON_NODES = 1_000_000


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _require_bounded_json(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    node_count = 0
    while stack:
        item, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_AUDIT_JSON_NODES:
            raise ValueError("audit JSON node capacity exceeded")
        if depth > _MAX_AUDIT_JSON_DEPTH:
            raise ValueError("audit JSON nesting capacity exceeded")
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError("audit JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in cast(list[object], item))
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("audit JSON contains a non-standard value")


def _load_canonical_json_object(value: str, *, label: str) -> dict[str, Any]:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty canonical JSON string")
    if len(value.encode("utf-8")) > _MAX_AUDIT_JSON_BYTES:
        raise ValueError(f"{label} exceeds the canonical JSON size limit")
    try:
        document = json.loads(value, parse_constant=_reject_json_constant)
        _require_bounded_json(document)
        canonical = canonical_json_bytes(document).decode("utf-8")
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} is not bounded canonical JSON") from error
    if type(document) is not dict or canonical != value:
        raise ValueError(f"{label} must be one canonical JSON object")
    return cast(dict[str, Any], document)


def _canonical_model_json(value: StrictFrozenModel) -> str:
    return canonical_json_bytes(value.model_dump(mode="json", warnings="error")).decode("utf-8")


def _artifact_sequence_sha256(values: Sequence[StrictFrozenModel]) -> str:
    return _digest(tuple(item.model_dump(mode="json", warnings="error") for item in values))


def _success_audit_artifact_hashes(
    bundle: CaseRankingReplayBundle,
) -> dict[str, str]:
    evidence = bundle.evidence
    return {
        "input": _digest(evidence.time_contract.model_dump(mode="json")),
        "configuration": bundle.scoring_config.config_sha256,
        "entities": evidence.entity_registry.registry_sha256,
        "observations": evidence.event_grouping.result_sha256,
        "candidate_universe": _digest(
            {
                "candidate_universe_sha256": evidence.candidate_universe.universe_sha256,
                "candidate_freeze_sha256": evidence.frozen_candidates.freeze_sha256,
            }
        ),
        "focus_set": evidence.focus_set.focus_sha256,
        "hypotheses": _digest(
            {
                "guard_results": tuple(
                    item.model_dump(mode="json") for item in evidence.role_guard_results
                ),
                "role_compilation": evidence.role_compilation.model_dump(mode="json"),
            }
        ),
        "graph": _digest(
            {
                "snapshots": tuple(
                    item.model_dump(mode="json") for item in evidence.graph_snapshots
                ),
                "condensations": tuple(
                    item.model_dump(mode="json") for item in evidence.condensations
                ),
                "onsets": tuple(item.model_dump(mode="json") for item in evidence.onsets),
                "precedences": tuple(item.model_dump(mode="json") for item in evidence.precedences),
            }
        ),
        "scores": bundle.ranking_result.result_sha256,
        "top_k": bundle.top_k.ranking_sha256,
        "history": bundle.history_evaluation.result_sha256,
        "rules": bundle.rule_scoring.result_sha256,
        "calls": _artifact_sequence_sha256(bundle.semantic_calls),
        "queries": _artifact_sequence_sha256(evidence.query_sessions),
        "costs": _artifact_sequence_sha256(bundle.costs),
        "degradation": _digest(bundle.degradation_reason_codes),
    }


def _success_provenance_partitions(
    bundle: CaseRankingReplayBundle,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    accepted: set[str] = set()
    rejected: set[str] = set()
    conflicted: set[str] = set()
    unknown: set[str] = set()
    for guard in bundle.evidence.role_guard_results:
        hypothesis = guard.parsed_hypothesis
        if hypothesis is not None:
            hypothesis_target = {
                "accepted": accepted,
                "rejected": rejected,
                "conflicted": conflicted,
                "proposed": unknown,
            }[hypothesis.lifecycle_status]
            hypothesis_target.add(hypothesis.hypothesis_id)
        target = accepted if guard.final_disposition == "accepted" else rejected
        target.add(guard.guard_result_id)
        for guard_decision in guard.decisions:
            (accepted if guard_decision.disposition == "accepted" else rejected).add(
                guard_decision.guard_decision_id
            )
    unknown.update(item.proposal_id for item in bundle.evidence.role_proposals)
    for assertion in bundle.evidence.role_compilation.assertions:
        target = {
            "accepted": accepted,
            "rejected": rejected,
            "conflicted": conflicted,
            "proposed": unknown,
        }[assertion.lifecycle_status]
        target.add(assertion.assertion_id)
    for session in bundle.evidence.query_sessions:
        accepted.add(session.query_session_id)
        card_result = session.card_expansion_result
        if card_result is not None:
            for card_decision in card_result.decisions:
                target = accepted if card_decision.status in {"completed", "reused"} else rejected
                target.update(
                    (
                        card_decision.request.expansion_request_id,
                        card_decision.expansion_decision_id,
                    )
                )
        verification = session.verification_result
        if verification is not None:
            result_by_request_id = {
                item.verification_request_id: item for item in verification.results
            }
            for execution_decision in verification.decisions:
                result = result_by_request_id[execution_decision.verification_request_id]
                target = (
                    accepted
                    if execution_decision.reason_code == "completed"
                    and result.status == "completed"
                    else rejected
                )
                target.update(
                    (
                        execution_decision.verification_request_id,
                        execution_decision.verification_result_id,
                        execution_decision.execution_decision_id,
                    )
                )
                if target is accepted:
                    target.update(result.derived_observation_ids)
    for call in bundle.semantic_calls:
        if call.status in {"success", "partial"}:
            accepted.add(call.semantic_call_id)
        else:
            rejected.add(call.semantic_call_id)
        if call.status == "partial":
            unknown.add(f"partial_semantic:{call.semantic_call_id}")
    unknown.update(
        item.predicate_result_id
        for item in (*bundle.evidence.role_predicates, *bundle.rule_predicates)
        if item.truth_value == "unknown"
    )
    unknown.update(f"degradation:{item}" for item in bundle.degradation_reason_codes)
    return tuple(map(tuple, map(sorted, (accepted, rejected, conflicted, unknown))))  # type: ignore[return-value]


def _build_success_sealed_summary(
    *,
    bundle: CaseRankingReplayBundle,
    experiment_version: str,
    run_mode: RunMode,
    sealed_at_utc: str,
) -> SealedAuditPackage:
    accepted, rejected, conflicted, unknown = _success_provenance_partitions(bundle)
    return build_sealed_audit_package(
        opaque_case_id=bundle.evidence.time_contract.opaque_case_id,
        experiment_version=experiment_version,
        terminal_state="SEALED",
        artifact_hashes=_success_audit_artifact_hashes(bundle),
        accepted_provenance_ids=accepted,
        rejected_provenance_ids=rejected,
        conflicted_provenance_ids=conflicted,
        unknown_provenance_ids=unknown,
        prediction_service_keys=tuple(item.service_entity_key for item in bundle.top_k.entries),
        run_mode=run_mode,
        sealed_at_utc=sealed_at_utc,
    )


class ImmutableCaseAuditPackage(StrictFrozenModel):
    """Canonical-at-rest terminal case audit with no mutable nested payloads."""

    schema_version: Literal["nurse.v2.immutable-case-audit-package/1"] = (
        IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    terminal_state: Literal["SEALED", "FAILURE_SEALED"]
    replay_bundle_json: str | None
    lineage_index_json: str | None
    failure_diagnostic_json: str | None
    sealed_audit_package_json: str
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_terminal_payloads(self) -> Self:
        _load_canonical_json_object(
            self.sealed_audit_package_json,
            label="sealed audit package",
        )
        if self.terminal_state == "SEALED":
            if self.replay_bundle_json is None or self.lineage_index_json is None:
                raise ValueError("successful audit requires replay and lineage JSON")
            if self.failure_diagnostic_json is not None:
                raise ValueError("successful audit cannot contain a failure diagnostic")
            _load_canonical_json_object(
                self.replay_bundle_json,
                label="case ranking replay bundle",
            )
            _load_canonical_json_object(
                self.lineage_index_json,
                label="case audit lineage index",
            )
        else:
            if self.replay_bundle_json is not None or self.lineage_index_json is not None:
                raise ValueError("failure audit cannot contain a ranking replay")
            if self.failure_diagnostic_json is None:
                raise ValueError("failure audit requires a real diagnostic")
            _load_canonical_json_object(
                self.failure_diagnostic_json,
                label="failure diagnostic",
            )
        return self

    @property
    def replay_bundle_or_none(self) -> CaseRankingReplayBundle | None:
        if self.replay_bundle_json is None:
            return None
        return CaseRankingReplayBundle.model_validate(
            _load_canonical_json_object(
                self.replay_bundle_json,
                label="case ranking replay bundle",
            )
        )

    @property
    def replay_bundle(self) -> CaseRankingReplayBundle:
        value = self.replay_bundle_or_none
        if value is None:
            raise ValueError("failure audit has no ranking replay bundle")
        return value

    @property
    def lineage_index_or_none(self) -> CaseAuditLineageIndex | None:
        if self.lineage_index_json is None:
            return None
        return CaseAuditLineageIndex.model_validate(
            _load_canonical_json_object(
                self.lineage_index_json,
                label="case audit lineage index",
            )
        )

    @property
    def lineage_index(self) -> CaseAuditLineageIndex:
        value = self.lineage_index_or_none
        if value is None:
            raise ValueError("failure audit has no ranking lineage index")
        return value

    @property
    def failure_diagnostic(self) -> EmptyCandidateUniverseFailure | None:
        if self.failure_diagnostic_json is None:
            return None
        return EmptyCandidateUniverseFailure.model_validate(
            _load_canonical_json_object(
                self.failure_diagnostic_json,
                label="failure diagnostic",
            )
        )

    @property
    def sealed_audit_package(self) -> SealedAuditPackage:
        return SealedAuditPackage.model_validate(
            _load_canonical_json_object(
                self.sealed_audit_package_json,
                label="sealed audit package",
            )
        )

    def verify(self) -> bool:
        try:
            if type(self) is not ImmutableCaseAuditPackage:
                return False
            rebuilt = ImmutableCaseAuditPackage.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self:
                return False
            unsigned = rebuilt.model_dump(mode="json", exclude={"audit_sha256"}, warnings="error")
            if not hmac.compare_digest(rebuilt.audit_sha256, _digest(unsigned)):
                return False
            sealed = rebuilt.sealed_audit_package
            if not sealed.verify() or sealed.opaque_case_id != rebuilt.opaque_case_id:
                return False
            if rebuilt.terminal_state == "SEALED":
                bundle = rebuilt.replay_bundle
                lineage = rebuilt.lineage_index
                if (
                    sealed.terminal_state != "SEALED"
                    or not bundle.verify_inputs()
                    or not lineage.verify_against(bundle)
                    or bundle.evidence.time_contract.opaque_case_id != rebuilt.opaque_case_id
                    or any(
                        checkpoint.experiment_version != sealed.experiment_version
                        for checkpoint in bundle.checkpoints
                    )
                ):
                    return False
                expected = _build_success_sealed_summary(
                    bundle=bundle,
                    experiment_version=sealed.experiment_version,
                    run_mode=sealed.run_mode,
                    sealed_at_utc=sealed.sealed_at_utc,
                )
                return sealed == expected
            diagnostic = rebuilt.failure_diagnostic
            return (
                sealed.terminal_state == "FAILURE_SEALED"
                and diagnostic is not None
                and diagnostic.verify()
                and diagnostic.opaque_case_id == rebuilt.opaque_case_id
                and diagnostic.prediction_service_keys == ()
                and diagnostic.sealed_audit_package == sealed
            )
        except (AttributeError, KeyError, RecursionError, TypeError, ValueError):
            return False


class CaseAuditReplayReceipt(StrictFrozenModel):
    """Freshly verified terminal result returned by deterministic audit replay."""

    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    terminal_state: Literal["SEALED", "FAILURE_SEALED"]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_service_keys: tuple[str, ...]
    ranking_result: ReferenceRankingResult | None
    top_k: TopKArtifact | None
    failure_diagnostic: EmptyCandidateUniverseFailure | None

    @model_validator(mode="after")
    def require_terminal_result_shape(self) -> Self:
        if self.terminal_state == "SEALED":
            if self.ranking_result is None or self.top_k is None:
                raise ValueError("successful replay requires ranking and Top-K")
            if self.failure_diagnostic is not None or not self.prediction_service_keys:
                raise ValueError("successful replay cannot contain a failure diagnostic")
        elif (
            self.ranking_result is not None
            or self.top_k is not None
            or self.failure_diagnostic is None
            or self.prediction_service_keys
        ):
            raise ValueError("failure replay must retain one diagnostic and no prediction")
        return self


def seal_case_audit(
    *,
    replay_bundle: CaseRankingReplayBundle | None = None,
    failure_diagnostic: EmptyCandidateUniverseFailure | None = None,
    experiment_version: str | None = None,
    run_mode: RunMode | None = None,
    sealed_at_utc: str | None = None,
) -> ImmutableCaseAuditPackage:
    """Seal exactly one verified success or exact real empty-C failure."""

    if (replay_bundle is None) == (failure_diagnostic is None):
        raise ValueError("exactly one replay bundle or failure diagnostic is required")
    if replay_bundle is not None:
        bundle = _exact_model(
            replay_bundle,
            CaseRankingReplayBundle,
            label="case ranking replay bundle",
        )
        if not bundle.verify_inputs():
            raise ValueError("case ranking replay bundle failed verification")
        if not experiment_version or sealed_at_utc is None:
            raise ValueError("successful audit requires experiment version and seal time")
        if any(
            checkpoint.experiment_version != experiment_version for checkpoint in bundle.checkpoints
        ):
            raise ValueError("checkpoint experiment version differs from the sealed audit")
        effective_run_mode: RunMode = "full" if run_mode is None else run_mode
        if effective_run_mode == "failure":
            raise ValueError("successful audit cannot use failure run mode")
        lineage = build_case_audit_lineage_index(bundle)
        sealed = _build_success_sealed_summary(
            bundle=bundle,
            experiment_version=experiment_version,
            run_mode=effective_run_mode,
            sealed_at_utc=sealed_at_utc,
        )
        unsigned: dict[str, object] = {
            "schema_version": IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
            "opaque_case_id": bundle.evidence.time_contract.opaque_case_id,
            "terminal_state": "SEALED",
            "replay_bundle_json": _canonical_model_json(bundle),
            "lineage_index_json": _canonical_model_json(lineage),
            "failure_diagnostic_json": None,
            "sealed_audit_package_json": _canonical_model_json(sealed),
        }
    else:
        diagnostic = _exact_model(
            failure_diagnostic,
            EmptyCandidateUniverseFailure,
            label="empty candidate universe failure",
        )
        if not diagnostic.verify():
            raise ValueError("empty candidate universe failure failed verification")
        if experiment_version is not None or run_mode is not None or sealed_at_utc is not None:
            raise ValueError("failure audit metadata must come from the sealed diagnostic")
        unsigned = {
            "schema_version": IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
            "opaque_case_id": diagnostic.opaque_case_id,
            "terminal_state": "FAILURE_SEALED",
            "replay_bundle_json": None,
            "lineage_index_json": None,
            "failure_diagnostic_json": _canonical_model_json(diagnostic),
            "sealed_audit_package_json": _canonical_model_json(diagnostic.sealed_audit_package),
        }
    result = ImmutableCaseAuditPackage.model_validate(
        {**unsigned, "audit_sha256": _digest(unsigned)}
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("immutable case audit package verification failed")
    return result


def _recompute_ranking(bundle: CaseRankingReplayBundle) -> ReferenceRankingResult:
    return rank_reference_candidates(
        frozen_candidates=bundle.evidence.frozen_candidates,
        family_aggregation=bundle.family_aggregation,
        propagation_scoring=bundle.propagation_scoring,
        rule_scoring=bundle.rule_scoring,
        history_evaluation=bundle.history_evaluation,
        authority=bundle.ranking_authority,
        config=bundle.scoring_config,
    )


def replay_case_audit(
    package: ImmutableCaseAuditPackage,
    *,
    expected_audit_sha256: str,
) -> CaseAuditReplayReceipt:
    """Verify the external anchor, then replay E/P/R/H or the real failure."""

    if type(expected_audit_sha256) is not str or not hmac.compare_digest(
        expected_audit_sha256,
        package.audit_sha256 if type(package) is ImmutableCaseAuditPackage else "",
    ):
        raise ValueError("external audit digest does not match the sealed package")
    exact_package = _exact_model(
        package,
        ImmutableCaseAuditPackage,
        label="immutable case audit package",
    )
    if not exact_package.verify():
        raise ValueError("immutable case audit package failed verification")
    if exact_package.terminal_state == "FAILURE_SEALED":
        diagnostic = exact_package.failure_diagnostic
        if diagnostic is None:  # pragma: no cover - package invariant
            raise AssertionError("verified failure audit lost its diagnostic")
        return CaseAuditReplayReceipt(
            opaque_case_id=exact_package.opaque_case_id,
            terminal_state="FAILURE_SEALED",
            audit_sha256=exact_package.audit_sha256,
            prediction_service_keys=(),
            ranking_result=None,
            top_k=None,
            failure_diagnostic=diagnostic,
        )
    bundle = exact_package.replay_bundle
    recomputed = _recompute_ranking(bundle)
    if recomputed != bundle.ranking_result or recomputed.top_k != bundle.top_k:
        raise ValueError("replayed ranking differs from the sealed ranking")
    return CaseAuditReplayReceipt(
        opaque_case_id=exact_package.opaque_case_id,
        terminal_state="SEALED",
        audit_sha256=exact_package.audit_sha256,
        prediction_service_keys=tuple(item.service_entity_key for item in recomputed.top_k.entries),
        ranking_result=recomputed,
        top_k=recomputed.top_k,
        failure_diagnostic=None,
    )


__all__ = [
    "CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION",
    "CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION",
    "CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION",
    "IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION",
    "CaseAuditLineageIndex",
    "CaseAuditLineageNode",
    "CaseAuditReplayReceipt",
    "CaseEvidenceReplayBundle",
    "CaseRankingReplayBundle",
    "ImmutableCaseAuditPackage",
    "build_case_audit_lineage_index",
    "build_case_evidence_replay_bundle",
    "build_case_ranking_replay_bundle",
    "replay_case_audit",
    "seal_case_audit",
]
