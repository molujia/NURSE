"""Explicit graph-aware role path; legacy role output and numeric rules stay intact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.competition_manifest import (
    CompetitionBinding,
    CompetitionDeclaration,
    CompetitionManifest,
)
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import DiagnosticHypothesis, PredicateResult, RoleAssertion
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import (
    ROLE_COMPILATION_POLICY_SHA256,
    RoleBindingProposal,
    _compile_event_role_records,
    _plain_sequence,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

COMPETITION_ROLE_COMPILATION_SCHEMA_VERSION: Literal["nurse.v2.competition-role-compilation/1"] = (
    "nurse.v2.competition-role-compilation/1"
)
COMPETITION_ROLE_POLICY_VERSION: Literal["nurse-v2-competition-role-policy/1"] = (
    "nurse-v2-competition-role-policy/1"
)
COMPETITION_ROLE_POLICY_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "version": COMPETITION_ROLE_POLICY_VERSION,
            "numeric_and_evidence_policy": ROLE_COMPILATION_POLICY_SHA256,
            "alternatives": "pinned_nonrecursive_directed_manifest",
            "unresolved_endpoint": "no_scoring_authority",
        }
    )
)


class CompetitionRoleCompilation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.competition-role-compilation/1"] = (
        COMPETITION_ROLE_COMPILATION_SCHEMA_VERSION
    )
    compilation_id: str = Field(pattern=r"^crc_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-competition-role-policy/1"] = COMPETITION_ROLE_POLICY_VERSION
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    competition_manifest: CompetitionManifest
    competition_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guard_result_ids: tuple[str, ...] = Field(max_length=4096)
    proposals: tuple[RoleBindingProposal, ...] = Field(max_length=16384)
    assertions: tuple[RoleAssertion, ...] = Field(max_length=32768)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Check internal closure; source authenticity needs verify_inputs()."""
        try:
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(CompetitionRoleCompilation,),
                label="competition result",
                max_string_length=None,
                provenance_id_limit=None,
            )
            value = CompetitionRoleCompilation.model_validate(
                self.model_dump(mode="python", warnings="error")
            )
            if value != self or self.policy_sha256 != COMPETITION_ROLE_POLICY_SHA256:
                return False
            graph = self.competition_manifest
            if (
                not graph.verify()
                or graph.manifest_sha256 != self.competition_manifest_sha256
                or graph.opaque_case_id != self.opaque_case_id
            ):
                return False
            if self.guard_result_ids != tuple(sorted(set(self.guard_result_ids))):
                return False
            if self.proposals != tuple(sorted(self.proposals, key=lambda p: p.proposal_id)):
                return False
            if self.assertions != tuple(sorted(self.assertions, key=lambda a: a.assertion_id)):
                return False
            if len({p.proposal_id for p in self.proposals}) != len(self.proposals):
                return False
            if len({a.assertion_id for a in self.assertions}) != len(self.assertions):
                return False
            core_candidates = {
                h.hypothesis_id: h.candidate_service_key for h in graph.core_hypotheses
            }
            if any(
                not p.verify()
                or p.guard_result_id not in self.guard_result_ids
                or p.hypothesis_id not in core_candidates
                for p in self.proposals
            ) or any(
                not a.verify() or core_candidates.get(a.hypothesis_id) != a.candidate_service_key
                for a in self.assertions
            ):
                return False
            digest = sha256_hex(
                canonical_json_bytes(
                    self.model_dump(mode="json", exclude={"compilation_id", "result_sha256"})
                )
            )
            return self.result_sha256 == digest and self.compilation_id == f"crc_{digest[:24]}"
        except (AttributeError, TypeError, ValueError, RecursionError):
            return False

    def verify_inputs(
        self,
        *,
        competition_manifest: CompetitionManifest,
        expected_manifest_sha256: str,
        response_sha256: str,
        prompt_sha256: str,
        declarations: Sequence[CompetitionDeclaration],
        bindings: Sequence[CompetitionBinding],
        core_hypotheses: Sequence[DiagnosticHypothesis],
        proposals: Sequence[RoleBindingProposal],
        guard_results: Sequence[HypothesisGuardResult],
        predicates: Sequence[PredicateResult],
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
        query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
        query_session_coordinator: CaseQuerySessionCoordinator | None = None,
        semantic_context: SemanticContextBundle | None = None,
        card_expansion_policy: CardExpansionPolicy | None = None,
    ) -> bool:
        try:
            if type(self) is not CompetitionRoleCompilation:
                return False
            preflight_exact_pydantic_tree(
                self,
                root_model_types=(CompetitionRoleCompilation,),
                label="competition result",
                max_string_length=None,
                provenance_id_limit=None,
            )
            return CompetitionRoleCompilation.verify(self) and self == compile_competition_roles(
                competition_manifest=competition_manifest,
                expected_manifest_sha256=expected_manifest_sha256,
                response_sha256=response_sha256,
                prompt_sha256=prompt_sha256,
                declarations=declarations,
                bindings=bindings,
                core_hypotheses=core_hypotheses,
                proposals=proposals,
                guard_results=guard_results,
                predicates=predicates,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
                mechanism_library=mechanism_library,
                scope_interval=scope_interval,
                query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
                query_session_coordinator=query_session_coordinator,
                semantic_context=semantic_context,
                card_expansion_policy=card_expansion_policy,
            )
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


def compile_competition_roles(
    *,
    competition_manifest: CompetitionManifest,
    expected_manifest_sha256: str,
    response_sha256: str,
    prompt_sha256: str,
    declarations: Sequence[CompetitionDeclaration],
    bindings: Sequence[CompetitionBinding],
    core_hypotheses: Sequence[DiagnosticHypothesis],
    proposals: Sequence[RoleBindingProposal],
    guard_results: Sequence[HypothesisGuardResult],
    predicates: Sequence[PredicateResult],
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    query_sessions_by_guard_result_id: Mapping[str, CaseQuerySessionResult] | None = None,
    query_session_coordinator: CaseQuerySessionCoordinator | None = None,
    semantic_context: SemanticContextBundle | None = None,
    card_expansion_policy: CardExpansionPolicy | None = None,
) -> CompetitionRoleCompilation:
    """Replay graph source bindings, then run unchanged predicate/role authority gates.

    All pins, declarations and conversion bindings come from trusted prior compiler
    state, not a model's claim that its own hashes are authoritative. A mapped core
    may still be rejected by guards; only replayed accepted guards are consumed here.
    """
    if type(competition_manifest) is not CompetitionManifest:
        raise ValueError("competition manifest requires an exact typed source")
    # Check every data input before invoking ANY verifier or serializer. The
    # coordinator is trusted runtime authority, not model-owned artifact data.
    preflight_exact_pydantic_tree(
        (
            competition_manifest,
            expected_manifest_sha256,
            response_sha256,
            prompt_sha256,
            declarations,
            bindings,
            core_hypotheses,
            proposals,
            guard_results,
            predicates,
            candidate_universe,
            frozen_candidates,
            entity_registry,
            event_grouping,
            mechanism_library,
            scope_interval,
            query_sessions_by_guard_result_id,
            semantic_context,
            card_expansion_policy,
        ),
        root_model_types=(
            CompetitionManifest,
            RoleBindingProposal,
            HypothesisGuardResult,
            PredicateResult,
            CandidateUniverse,
            FrozenCandidateUniverse,
            TemporalEntityRegistry,
            EventGroupingResult,
            MechanismLibrary,
            TimeInterval,
            CaseQuerySessionResult,
            SemanticContextBundle,
            CardExpansionPolicy,
        ),
        label="competition inputs",
        max_string_length=None,
        provenance_id_limit=None,
    )
    preflight_exact_pydantic_tree(
        competition_manifest,
        root_model_types=(CompetitionManifest,),
        label="competition manifest",
        max_string_length=None,
        provenance_id_limit=None,
    )
    if (
        type(expected_manifest_sha256) is not str
        or competition_manifest.manifest_sha256 != expected_manifest_sha256
        or type(candidate_universe) is not CandidateUniverse
        or not competition_manifest.verify_inputs(
            opaque_case_id=candidate_universe.opaque_case_id,
            response_sha256=response_sha256,
            prompt_sha256=prompt_sha256,
            declarations=declarations,
            bindings=bindings,
            core_hypotheses=core_hypotheses,
        )
    ):
        raise ValueError("competition manifest does not match pinned source replay")
    core_by_id = {h.hypothesis_id: h for h in competition_manifest.core_hypotheses}
    for item in _plain_sequence(guard_results, limit=4096, name="competition guards"):
        if type(item) is not HypothesisGuardResult:
            raise ValueError("competition requires an exact guard result")
        preflight_exact_pydantic_tree(
            item,
            root_model_types=(HypothesisGuardResult,),
            label="competition guard",
            max_string_length=None,
            provenance_id_limit=None,
        )
        hypothesis = item.parsed_hypothesis
        if hypothesis is None or core_by_id.get(hypothesis.hypothesis_id) != hypothesis:
            raise ValueError("competition guard core is not in the source manifest")
    alternative_sets: dict[str, set[str]] = {}
    for edge in competition_manifest.edges:
        if edge.source_hypothesis_id is not None and edge.target_hypothesis_id is not None:
            alternative_sets.setdefault(edge.source_hypothesis_id, set()).add(
                edge.target_hypothesis_id
            )
    records = _compile_event_role_records(
        proposals=proposals,
        guard_results=guard_results,
        predicates=predicates,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_sessions_by_guard_result_id=query_sessions_by_guard_result_id,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
        alternatives_by_hypothesis_id={k: tuple(sorted(v)) for k, v in alternative_sets.items()},
    )
    unsigned = {
        **records,
        "schema_version": COMPETITION_ROLE_COMPILATION_SCHEMA_VERSION,
        "policy_version": COMPETITION_ROLE_POLICY_VERSION,
        "policy_sha256": COMPETITION_ROLE_POLICY_SHA256,
        "competition_manifest": competition_manifest.model_dump(mode="python"),
        "competition_manifest_sha256": competition_manifest.manifest_sha256,
    }
    digest = sha256_hex(canonical_json_bytes(unsigned))
    result = CompetitionRoleCompilation.model_validate(
        {**unsigned, "compilation_id": f"crc_{digest[:24]}", "result_sha256": digest}
    )
    if not result.verify():
        raise ValueError("competition compilation failed internal verification")
    return result
