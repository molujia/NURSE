"""Private semantic compilation receipts and internal/source verification.

Internal closure is not source authenticity. Ranking consumers must replay from
independently pinned source inputs before using any semantic role effects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.competition_manifest import (
    CompetitionManifest,
)
from nurse_rcl.v2.competition_roles import CompetitionRoleCompilation
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateResult,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.hypothesis_guard import (
    HypothesisGuardResult,
)
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.role_compiler import (
    RoleBindingProposal,
    local_event_relation_id,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_conversion import (
    SEMANTIC_COMPILER_VERSION,
    _core,
    _raw_hypotheses,
    _role_proposal,
    _source_bindings,
    _source_declarations,
)
from nurse_rcl.v2.semantic_permissions import SemanticInspection, inspect_semantic_response
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_role_scope import (
    SEMANTIC_ROLE_SCOPE_POLICY_VERSION,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

SEMANTIC_COMPILATION_SCHEMA_VERSION: Literal["nurse.v2.semantic-compilation/1"] = (
    "nurse.v2.semantic-compilation/1"
)


class SemanticCompilationDecision(StrictFrozenModel):
    section: Literal["hypotheses", "queries"]
    index: int = Field(ge=0)
    local_id: str | None
    disposition: Literal["eligible", "denied"]
    reason_code: str


class SemanticRoleScopeDecision(StrictFrozenModel):
    local_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    role_index: int = Field(ge=0, lt=16)
    relation_id: str = Field(pattern=r"^(?:assoc|lrel)_[0-9a-f]{24}$")
    reason_code: Literal[
        "eligible_for_compilation", "no_true_event_support", "hypothesis_not_accepted"
    ]


def _decisions(inspection: SemanticInspection) -> tuple[SemanticCompilationDecision, ...]:
    return tuple(
        SemanticCompilationDecision(
            section=d.section,
            index=d.index,
            local_id=d.local_id,
            disposition=d.disposition,
            reason_code=d.reason_code,
        )
        for d in inspection.decisions
    )


class SemanticCompilation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.semantic-compilation/1"] = SEMANTIC_COMPILATION_SCHEMA_VERSION
    compiler_version: Literal["nurse-v2-semantic-compiler/1"] = SEMANTIC_COMPILER_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    raw_response_hex: str = Field(max_length=524288, repr=False)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_payload_hex: str = Field(max_length=524288, repr=False)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structurally_valid: bool
    envelope_error: str | None
    decisions: tuple[SemanticCompilationDecision, ...] = Field(max_length=40)
    core_hypotheses: tuple[DiagnosticHypothesis, ...] = Field(max_length=32)
    predicates: tuple[PredicateResult, ...] = Field(max_length=512)
    guard_results: tuple[HypothesisGuardResult, ...] = Field(max_length=32)
    role_scope_policy: Literal["true-original-event-role-witness/1"] = (
        SEMANTIC_ROLE_SCOPE_POLICY_VERSION
    )
    role_scope_decisions: tuple[SemanticRoleScopeDecision, ...] = Field(max_length=32768)
    competition_manifest: CompetitionManifest
    role_compilation: CompetitionRoleCompilation | None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Internal closure only; authentic effects require independent input replay."""
        try:
            _preflight(self)
            value = SemanticCompilation.model_validate(
                self.model_dump(mode="python", warnings="error")
            )
            if value != self:
                return False
            raw = bytes.fromhex(self.raw_response_hex)
            prompt = SemanticPrompt(bytes.fromhex(self.prompt_payload_hex))
            inspection = inspect_semantic_response(
                raw, prompt=prompt, expected_prompt_sha256=self.prompt_sha256
            )
            if (
                inspection.response_sha256 != self.response_sha256
                or inspection.structurally_valid != self.structurally_valid
                or inspection.envelope_error != self.envelope_error
                or _decisions(inspection) != self.decisions
            ):
                return False
            graph = self.competition_manifest
            if (
                not graph.verify()
                or graph.response_sha256 != self.response_sha256
                or graph.prompt_sha256 != self.prompt_sha256
                or graph.opaque_case_id != self.opaque_case_id
                or graph.core_hypotheses != self.core_hypotheses
            ):
                return False
            cores = {h.hypothesis_id: h for h in self.core_hypotheses}
            if any(not p.verify() for p in self.predicates):
                return False
            if self.predicates != tuple(
                sorted(self.predicates, key=lambda p: p.predicate_result_id)
            ):
                return False
            if len({p.predicate_result_id for p in self.predicates}) != len(self.predicates):
                return False
            if {p.predicate_result_id for p in self.predicates} != {
                pid
                for h in self.core_hypotheses
                for pid in (
                    *h.necessary_predicate_result_ids,
                    *h.optional_predicate_result_ids,
                    *h.refuting_predicate_result_ids,
                )
            }:
                return False
            if self.guard_results != tuple(
                sorted(self.guard_results, key=lambda g: g.guard_result_id)
            ):
                return False
            if len({g.guard_result_id for g in self.guard_results}) != len(self.guard_results):
                return False
            if any(
                not g.verify()
                or g.parsed_hypothesis is None
                or cores.get(g.parsed_hypothesis.hypothesis_id) != g.parsed_hypothesis
                for g in self.guard_results
            ):
                return False
            if {
                g.parsed_hypothesis.hypothesis_id for g in self.guard_results if g.parsed_hypothesis
            } != set(cores):
                return False
            if not _source_receipts_close(self, inspection, prompt):
                return False
            if not _role_receipts_close(self, inspection):
                return False
            if self.role_compilation is not None and (
                not self.role_compilation.verify()
                or self.role_compilation.competition_manifest != graph
                or self.role_compilation.guard_result_ids
                != tuple(
                    sorted(
                        g.guard_result_id
                        for g in self.guard_results
                        if g.final_disposition == "accepted"
                    )
                )
            ):
                return False
            digest = sha256_hex(
                canonical_json_bytes(self.model_dump(mode="json", exclude={"result_sha256"}))
            )
            return self.result_sha256 == digest
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False

    def verify_inputs(
        self,
        *,
        raw_response: bytes,
        prompt: SemanticPrompt,
        expected_prompt_sha256: str,
        candidate_universe: CandidateUniverse,
        frozen_candidates: FrozenCandidateUniverse,
        entity_registry: TemporalEntityRegistry,
        event_grouping: EventGroupingResult,
        mechanism_library: MechanismLibrary,
        scope_interval: TimeInterval,
    ) -> bool:
        from nurse_rcl.v2.semantic_compilation import compile_semantic_response

        try:
            _preflight(self)
            return SemanticCompilation.verify(self) and self == compile_semantic_response(
                raw_response=raw_response,
                prompt=prompt,
                expected_prompt_sha256=expected_prompt_sha256,
                candidate_universe=candidate_universe,
                frozen_candidates=frozen_candidates,
                entity_registry=entity_registry,
                event_grouping=event_grouping,
                mechanism_library=mechanism_library,
                scope_interval=scope_interval,
            )
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


def _preflight(value: object) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(SemanticCompilation,),
        label="semantic compilation",
        max_string_length=None,
        provenance_id_limit=None,
    )


def _source_receipts_close(
    result: SemanticCompilation, inspection: SemanticInspection, prompt: SemanticPrompt
) -> bool:
    """Close retained source declarations and plans without claiming source truth."""
    assert inspection.raw_response is not None
    raw_items = _raw_hypotheses(inspection.raw_response, inspection)
    graph = result.competition_manifest
    core_by_id = {h.hypothesis_id: h for h in result.core_hypotheses}
    cores = {
        b.local_id: core_by_id[b.hypothesis_id]
        for b in graph.bindings
        if b.hypothesis_id is not None
    }
    failures = {b.local_id for b in graph.bindings if b.reason_code == "compilation_failed"}
    if set(cores) | failures != {i.proposal_id for i in inspection.eligible_hypotheses}:
        return False
    if graph.declarations != _source_declarations(raw_items) or graph.bindings != _source_bindings(
        raw_items, inspection, cores, failures
    ):
        return False
    data = decode_json_object(prompt.payload_json, ExchangeLimits())
    if data["opaque_case_id"] != result.opaque_case_id:
        return False
    library = MechanismLibrary.model_validate_json(
        encode_json(data["mechanism_library"], ExchangeLimits())
    )
    predicates_by_id = {p.predicate_result_id: p for p in result.predicates}
    original_provenance = {
        o["observation_id"]: o["provenance_ids"]
        for card in data["detailed_evidence_cards"]
        for o in card["observations"]
    }
    relation_prefix = "predicate-plan-relation:"
    for item in inspection.eligible_hypotheses:
        core = cores.get(item.proposal_id)
        if core is None:
            continue  # A claimed conversion failure still requires verify_inputs().
        predicates = tuple(
            predicates_by_id[pid]
            for pid in (
                *core.necessary_predicate_result_ids,
                *core.optional_predicate_result_ids,
                *core.refuting_predicate_result_ids,
            )
        )
        by_plan = {(p.predicate_kind, p.requirement): p for p in predicates}
        if len(by_plan) != len(predicates) or set(by_plan) != {
            (p.predicate_kind, p.requirement) for p in item.predicate_plans
        }:
            return False
        for plan in item.predicate_plans:
            predicate = by_plan[(plan.predicate_kind, plan.requirement)]
            if predicate.observation_ids != tuple(sorted(set(plan.observation_ids))):
                return False
            # Source provenance may itself contain this prefix; do not mistake
            # that original data for a new compiler-issued relation reference.
            expected_relations = {f"{relation_prefix}{r}" for r in plan.relation_ids} | {
                p
                for oid in plan.observation_ids
                for p in original_provenance[oid]
                if p.startswith(relation_prefix)
            }
            if predicate.verification_result_ids or expected_relations != {
                p for p in predicate.provenance_ids if p.startswith(relation_prefix)
            }:
                return False
        if core != _core(
            item,
            predicates,
            case_id=result.opaque_case_id,
            library=library,
            response_sha256=result.response_sha256,
            prompt_sha256=result.prompt_sha256,
        ):
            return False
    return True


def _role_receipts_close(result: SemanticCompilation, inspection: SemanticInspection) -> bool:
    """Match raw role occurrences to receipts and the resulting compiler proposals.

    This checks internal closure, not witness truth; only verify_inputs() can
    independently replay the latter from the full frozen source state.
    """
    occurrences = tuple(
        (item, index, relation)
        for item in inspection.eligible_hypotheses
        for index, role in enumerate(item.role_proposals)
        for relation in role.relation_ids
        or (
            local_event_relation_id(
                candidate_service_key=item.candidate_service_key,
                event_group_id=role.event_group_id,
            ),
        )
    )
    if tuple((i.proposal_id, n, r) for i, n, r in occurrences) != tuple(
        (d.local_id, d.role_index, d.relation_id) for d in result.role_scope_decisions
    ):
        return False
    core_by_id = {h.hypothesis_id: h for h in result.core_hypotheses}
    cores = {
        b.local_id: core_by_id[b.hypothesis_id]
        for b in result.competition_manifest.bindings
        if b.hypothesis_id is not None
    }
    guards = {
        g.parsed_hypothesis.hypothesis_id: g
        for g in result.guard_results
        if g.parsed_hypothesis is not None
    }
    if len(guards) != len(result.guard_results):
        return False
    expected: dict[str, RoleBindingProposal] = {}
    for (item, index, relation), receipt in zip(
        occurrences, result.role_scope_decisions, strict=True
    ):
        core = cores.get(item.proposal_id)
        guard = None if core is None else guards.get(core.hypothesis_id)
        accepted = guard is not None and guard.final_disposition == "accepted"
        if (receipt.reason_code == "hypothesis_not_accepted") == accepted:
            return False
        if receipt.reason_code == "eligible_for_compilation":
            assert core is not None and guard is not None
            proposal = _role_proposal(item, index, relation, core, guard, result.response_sha256)
            expected[proposal.proposal_id] = proposal
    if not expected:
        return result.role_compilation is None
    return result.role_compilation is not None and result.role_compilation.proposals == tuple(
        expected[key] for key in sorted(expected)
    )
