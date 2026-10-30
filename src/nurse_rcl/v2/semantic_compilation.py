"""Compile one pinned semantic response; no transport, retries or model truth.

This initial-source compiler consumes original frozen observations. Query-derived
evidence requires a separately replayed query session, not an invented observation
ID or a model-authored measurement. Raw responses remain private audit material.
"""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.competition_manifest import (
    build_competition_manifest,
)
from nurse_rcl.v2.competition_roles import compile_competition_roles
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateResult,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.hypothesis_guard import (
    HypothesisGuardResult,
    _validated_inputs,
    execute_hypothesis_guard,
)
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.predicate_plan_evaluator import evaluate_predicate_plan
from nurse_rcl.v2.role_compiler import (
    RoleBindingProposal,
    local_event_relation_id,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_compilation_artifact import (
    SEMANTIC_COMPILATION_SCHEMA_VERSION,
    SemanticCompilation,
    SemanticRoleScopeDecision,
    _decisions,
)
from nurse_rcl.v2.semantic_conversion import (
    SEMANTIC_COMPILER_VERSION,
    _core,
    _raw_hypotheses,
    _role_proposal,
    _source_bindings,
    _source_declarations,
)
from nurse_rcl.v2.semantic_permissions import inspect_semantic_response
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_role_scope import (
    SEMANTIC_ROLE_SCOPE_POLICY_VERSION,
    has_true_role_witness,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry


def compile_semantic_response(
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
) -> SemanticCompilation:
    """Inspect and compile once from original observations, retaining unusable items.

    Expected prompt identity comes from trusted pre-call state, never from the
    response. No accepted hypothesis or score is selected by model preference.
    """
    common: dict[str, Any] = dict(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
    )
    preflight_exact_pydantic_tree(
        (expected_prompt_sha256, common),
        root_model_types=(
            CandidateUniverse,
            FrozenCandidateUniverse,
            TemporalEntityRegistry,
            EventGroupingResult,
            MechanismLibrary,
            TimeInterval,
        ),
        label="semantic sources",
        max_string_length=None,
        provenance_id_limit=None,
    )
    _validated_inputs(**common)
    inspection = inspect_semantic_response(
        raw_response, prompt=prompt, expected_prompt_sha256=expected_prompt_sha256
    )
    if inspection.raw_response is None or inspection.response_sha256 is None:
        raise ValueError("semantic_response_not_retainable")
    data = decode_json_object(prompt.payload_json, ExchangeLimits())
    directory = data["candidate_directory"]
    if (
        data["opaque_case_id"] != candidate_universe.opaque_case_id
        or directory["candidate_universe_sha256"] != candidate_universe.universe_sha256
        or directory["candidate_freeze_sha256"] != frozen_candidates.freeze_sha256
        or data["scope_interval"] != scope_interval.model_dump(mode="json")
        or data["mechanism_library"] != mechanism_library.model_dump(mode="json")
    ):
        raise ValueError("semantic_prompt_source_mismatch")
    predicates_by_id: dict[str, PredicateResult] = {}
    cores: dict[str, DiagnosticHypothesis] = {}
    guards: dict[str, HypothesisGuardResult] = {}
    failures: set[str] = set()
    for item in inspection.eligible_hypotheses:
        try:
            predicates = tuple(
                evaluate_predicate_plan(plan=plan, event_grouping=event_grouping)
                for plan in item.predicate_plans
            )
            core = _core(
                item,
                predicates,
                case_id=candidate_universe.opaque_case_id,
                library=mechanism_library,
                response_sha256=inspection.response_sha256,
                prompt_sha256=inspection.prompt_sha256,
            )
        except (TypeError, ValueError):
            failures.add(item.proposal_id)
            continue
        cores[item.proposal_id] = core
        predicates_by_id.update({p.predicate_result_id: p for p in predicates})
        guards[item.proposal_id] = execute_hypothesis_guard(
            submitted_record=core.model_dump(mode="json"), predicates=predicates, **common
        )
    raw_items = _raw_hypotheses(raw_response, inspection)
    bindings = _source_bindings(raw_items, inspection, cores, failures)
    declarations = _source_declarations(raw_items)
    ordered_cores = tuple(sorted(cores.values(), key=lambda h: h.hypothesis_id))
    manifest = build_competition_manifest(
        opaque_case_id=candidate_universe.opaque_case_id,
        response_sha256=inspection.response_sha256,
        prompt_sha256=inspection.prompt_sha256,
        declarations=declarations,
        bindings=tuple(bindings),
        core_hypotheses=ordered_cores,
    )
    proposals: list[RoleBindingProposal] = []
    role_scope_decisions: list[SemanticRoleScopeDecision] = []
    for item in inspection.eligible_hypotheses:
        guard = guards.get(item.proposal_id)
        role_core = cores.get(item.proposal_id)
        for role_index, role in enumerate(item.role_proposals):
            for relation in role.relation_ids or (
                local_event_relation_id(
                    candidate_service_key=item.candidate_service_key,
                    event_group_id=role.event_group_id,
                ),
            ):
                if role_core is None or guard is None or guard.final_disposition != "accepted":
                    role_scope_decisions.append(
                        SemanticRoleScopeDecision(
                            local_id=item.proposal_id,
                            role_index=role_index,
                            relation_id=relation,
                            reason_code="hypothesis_not_accepted",
                        )
                    )
                    continue
                proposal = _role_proposal(
                    item, role_index, relation, role_core, guard, inspection.response_sha256
                )
                supported = has_true_role_witness(
                    hypothesis=role_core,
                    proposal=proposal,
                    predicates=tuple(predicates_by_id.values()),
                    event_grouping=event_grouping,
                    entity_registry=entity_registry,
                )
                role_scope_decisions.append(
                    SemanticRoleScopeDecision(
                        local_id=item.proposal_id,
                        role_index=role_index,
                        relation_id=relation,
                        reason_code="eligible_for_compilation"
                        if supported
                        else "no_true_event_support",
                    )
                )
                if supported:
                    proposals.append(proposal)
    ordered_predicates = tuple(
        sorted(predicates_by_id.values(), key=lambda p: p.predicate_result_id)
    )
    # No accepted role proposal is an auditable no-effect outcome, not a fake
    # legacy compilation or a reason to draw another model response.
    role_result = None
    if proposals:
        role_result = compile_competition_roles(
            competition_manifest=manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            response_sha256=inspection.response_sha256,
            prompt_sha256=inspection.prompt_sha256,
            declarations=declarations,
            bindings=tuple(bindings),
            core_hypotheses=ordered_cores,
            proposals=tuple(proposals),
            guard_results=tuple(g for g in guards.values() if g.final_disposition == "accepted"),
            predicates=ordered_predicates,
            **common,
        )
    unsigned = dict(
        schema_version=SEMANTIC_COMPILATION_SCHEMA_VERSION,
        compiler_version=SEMANTIC_COMPILER_VERSION,
        opaque_case_id=candidate_universe.opaque_case_id,
        raw_response_hex=raw_response.hex(),
        response_sha256=inspection.response_sha256,
        prompt_payload_hex=prompt.payload_json.hex(),
        prompt_sha256=inspection.prompt_sha256,
        structurally_valid=inspection.structurally_valid,
        envelope_error=inspection.envelope_error,
        decisions=tuple(d.model_dump(mode="python") for d in _decisions(inspection)),
        core_hypotheses=tuple(h.model_dump(mode="python") for h in ordered_cores),
        predicates=tuple(p.model_dump(mode="python") for p in ordered_predicates),
        guard_results=tuple(
            g.model_dump(mode="python")
            for g in sorted(guards.values(), key=lambda g: g.guard_result_id)
        ),
        role_scope_policy=SEMANTIC_ROLE_SCOPE_POLICY_VERSION,
        role_scope_decisions=tuple(d.model_dump(mode="python") for d in role_scope_decisions),
        competition_manifest=manifest.model_dump(mode="python"),
        role_compilation=None if role_result is None else role_result.model_dump(mode="python"),
    )
    result = SemanticCompilation.model_validate(
        {**unsigned, "result_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not result.verify():
        raise ValueError("semantic_compilation_integrity_failure")
    return result
