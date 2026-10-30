"""Compile a complete final snapshot from independently replayed query evidence.

No transport, round selection, retries, or new query execution occurs here. The
immutable result is a private in-process receipt, not a durable artifact schema
or self-authenticating score authorization. Consumers must replay its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.competition_manifest import CompetitionManifest, build_competition_manifest
from nurse_rcl.v2.competition_roles import CompetitionRoleCompilation, compile_competition_roles
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateResult,
    build_diagnostic_hypothesis,
)
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult, execute_hypothesis_guard
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_predicate_evaluator import _evaluate_query_predicate_plan
from nurse_rcl.v2.query_role_scope import (
    QUERY_ROLE_SCOPE_POLICY_VERSION,
    has_true_query_role_witness,
)
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator, CaseQuerySessionResult
from nurse_rcl.v2.role_compiler import RoleBindingProposal, local_event_relation_id
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_compilation_artifact import (
    SemanticCompilationDecision,
    SemanticRoleScopeDecision,
    _decisions,
)
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.semantic_conversion import (
    _raw_hypotheses,
    _role_proposal,
    _source_bindings,
    _source_declarations,
)
from nurse_rcl.v2.semantic_followup import build_semantic_followup_prompt
from nurse_rcl.v2.semantic_followup_permissions import inspect_semantic_followup_response
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_proposals import SemanticHypothesisProposal
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import VerificationExecutionPolicy

SEMANTIC_FOLLOWUP_COMPILER_VERSION = "nurse-v2-semantic-followup-compiler/1"


@dataclass(frozen=True, slots=True)
class SemanticFollowupCompilation:
    """Private immutable receipt; only external input replay establishes authenticity."""

    opaque_case_id: str
    initial_response_sha256: str
    initial_prompt_sha256: str
    query_session_sha256: str
    raw_response_hex: str = field(repr=False)
    response_sha256: str
    prompt_payload_hex: str = field(repr=False)
    prompt_sha256: str
    structurally_valid: bool
    envelope_error: str | None
    decisions: tuple[SemanticCompilationDecision, ...]
    core_hypotheses: tuple[DiagnosticHypothesis, ...] = field(repr=False)
    predicates: tuple[PredicateResult, ...] = field(repr=False)
    guard_results: tuple[HypothesisGuardResult, ...] = field(repr=False)
    role_scope_decisions: tuple[SemanticRoleScopeDecision, ...]
    competition_manifest: CompetitionManifest = field(repr=False)
    role_compilation: CompetitionRoleCompilation | None = field(repr=False)
    compiler_version: str = SEMANTIC_FOLLOWUP_COMPILER_VERSION
    role_scope_policy: str = QUERY_ROLE_SCOPE_POLICY_VERSION

    def verify_inputs(self, **inputs: Any) -> bool:
        """Recompute from the public compiler's external inputs, never stored self-pins."""
        if type(self) is not SemanticFollowupCompilation:
            return False
        try:
            preflight_exact_pydantic_tree(
                tuple(
                    object.__getattribute__(self, f.name)
                    for f in fields(SemanticFollowupCompilation)
                ),
                root_model_types=(
                    DiagnosticHypothesis,
                    PredicateResult,
                    HypothesisGuardResult,
                    SemanticCompilationDecision,
                    SemanticRoleScopeDecision,
                    CompetitionManifest,
                    CompetitionRoleCompilation,
                ),
                label="semantic follow-up receipt",
                max_string_length=None,
                provenance_id_limit=None,
            )
            return self == compile_semantic_followup_response(**inputs)
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


def _query_core(
    item: SemanticHypothesisProposal,
    predicates: tuple[PredicateResult, ...],
    *,
    case_id: str,
    library: MechanismLibrary,
    response_sha256: str,
    prompt_sha256: str,
    session: CaseQuerySessionResult,
    outcomes: list[dict[str, Any]],
) -> DiagnosticHypothesis:
    # These outcomes come only from the replay-built prompt. Card requests are
    # never verification IDs. A cited visible derived ID also carries its actual
    # request lineage even when the model omits the optional semantic alias.
    request_ids = {
        q["request_id"]
        for q in outcomes
        if q["query_id"] in item.query_ids and q["kind"] == "verification"
    }
    batch = session.verification_result
    if batch is not None:
        request_ids.update(
            d.verification_request_id
            for d in batch.derived_observations
            if d.derived_observation_id in item.observation_ids
        )
    return build_diagnostic_hypothesis(
        opaque_case_id=case_id,
        candidate_service_key=item.candidate_service_key,
        mechanism_key=item.mechanism_key,
        mechanism_definition_sha256=library.lookup(item.mechanism_key).definition_sha256,
        origin="semantic_agent",
        origin_version=SEMANTIC_FOLLOWUP_COMPILER_VERSION,
        observation_ids=item.observation_ids,
        relation_ids=item.relation_ids,
        event_group_ids=item.event_group_ids,
        time_intervals=tuple(
            TimeInterval(start_unix_ns=t.start_unix_ns, end_unix_ns=t.end_unix_ns)
            for t in item.time_intervals
        ),
        necessary_predicate_result_ids=tuple(
            p.predicate_result_id for p in predicates if p.requirement == "necessary"
        ),
        optional_predicate_result_ids=tuple(
            p.predicate_result_id for p in predicates if p.requirement == "optional"
        ),
        refuting_predicate_result_ids=tuple(
            p.predicate_result_id for p in predicates if p.requirement == "refuting"
        ),
        alternative_hypothesis_ids=(),
        verification_request_ids=tuple(sorted(request_ids)),
        unknown_reasons=item.unknown_reasons,
        provenance_ids=(
            response_sha256,
            prompt_sha256,
            session.session_sha256,
            f"semantic-proposal:{item.proposal_id}",
        ),
        lifecycle_status="proposed",
    )


def compile_semantic_followup_response(
    *,
    followup_raw_response: bytes,
    expected_followup_prompt_sha256: str,
    mechanism_library: MechanismLibrary,
    query_session_coordinator: CaseQuerySessionCoordinator,
    query_session: CaseQuerySessionResult,
    expected_session_sha256: str,
    expected_response_sha256: str,
    raw_response: bytes,
    prompt: SemanticPrompt,
    expected_prompt_sha256: str,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    semantic_context: SemanticContextBundle,
    card_expansion_policy: CardExpansionPolicy,
    verification_policy: VerificationExecutionPolicy,
    scope_interval: TimeInterval,
) -> SemanticFollowupCompilation:
    """Rebuild all evidence authority before compiling the final raw response once."""
    if type(followup_raw_response) is not bytes or type(mechanism_library) is not MechanismLibrary:
        raise ValueError("invalid_semantic_followup_compilation_input")
    # Preflight these additional inputs before the builder may invoke any source
    # verifier; the builder itself globally preflights every remaining input.
    try:
        preflight_exact_pydantic_tree(
            (mechanism_library, expected_followup_prompt_sha256),
            root_model_types=(MechanismLibrary,),
            label="follow-up compilation inputs",
            max_string_length=None,
            provenance_id_limit=None,
        )
    except AttributeError:
        raise ValueError("invalid_semantic_followup_compilation_model_state") from None
    followup = build_semantic_followup_prompt(
        query_session_coordinator=query_session_coordinator,
        query_session=query_session,
        expected_session_sha256=expected_session_sha256,
        expected_response_sha256=expected_response_sha256,
        raw_response=raw_response,
        prompt=prompt,
        expected_prompt_sha256=expected_prompt_sha256,
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
        verification_policy=verification_policy,
        scope_interval=scope_interval,
    )
    inspection = inspect_semantic_followup_response(
        followup_raw_response,
        prompt=followup,
        expected_prompt_sha256=expected_followup_prompt_sha256,
    )
    if inspection.raw_response is None or inspection.response_sha256 is None:
        raise ValueError("semantic_response_not_retainable")
    data = decode_json_object(followup.payload_json, ExchangeLimits())
    if not mechanism_library.verify() or data["mechanism_library"] != mechanism_library.model_dump(
        mode="json"
    ):
        raise ValueError("semantic_prompt_source_mismatch")
    common: dict[str, Any] = dict(
        candidate_universe=candidate_universe,
        frozen_candidates=frozen_candidates,
        entity_registry=entity_registry,
        event_grouping=event_grouping,
        mechanism_library=mechanism_library,
        scope_interval=scope_interval,
        query_session_coordinator=query_session_coordinator,
        semantic_context=semantic_context,
        card_expansion_policy=card_expansion_policy,
    )
    predicates_by_id: dict[str, PredicateResult] = {}
    cores: dict[str, DiagnosticHypothesis] = {}
    guards: dict[str, HypothesisGuardResult] = {}
    failures: set[str] = set()
    for item in inspection.eligible_hypotheses:
        try:
            predicates = tuple(
                _evaluate_query_predicate_plan(
                    plan=plan, event_grouping=event_grouping, query_session=query_session
                )
                for plan in item.predicate_plans
            )
            core = _query_core(
                item,
                predicates,
                case_id=candidate_universe.opaque_case_id,
                library=mechanism_library,
                response_sha256=inspection.response_sha256,
                prompt_sha256=inspection.prompt_sha256,
                session=query_session,
                outcomes=data["query_outcomes"],
            )
        except (TypeError, ValueError):
            failures.add(item.proposal_id)
            continue
        cores[item.proposal_id] = core
        predicates_by_id.update({p.predicate_result_id: p for p in predicates})
        guards[item.proposal_id] = execute_hypothesis_guard(
            submitted_record=core.model_dump(mode="json"),
            predicates=predicates,
            query_session=query_session,
            **common,
        )
    raw_items = _raw_hypotheses(followup_raw_response, inspection)
    bindings = _source_bindings(raw_items, inspection, cores, failures)
    declarations = _source_declarations(raw_items)
    ordered_cores = tuple(sorted(cores.values(), key=lambda h: h.hypothesis_id))
    manifest = build_competition_manifest(
        opaque_case_id=candidate_universe.opaque_case_id,
        response_sha256=inspection.response_sha256,
        prompt_sha256=inspection.prompt_sha256,
        declarations=declarations,
        bindings=bindings,
        core_hypotheses=ordered_cores,
    )
    proposals: list[RoleBindingProposal] = []
    scope_decisions: list[SemanticRoleScopeDecision] = []
    for item in inspection.eligible_hypotheses:
        guard, role_core = guards.get(item.proposal_id), cores.get(item.proposal_id)
        for index, role in enumerate(item.role_proposals):
            for relation in role.relation_ids or (
                local_event_relation_id(
                    candidate_service_key=item.candidate_service_key,
                    event_group_id=role.event_group_id,
                ),
            ):
                reason = "hypothesis_not_accepted"
                if (
                    role_core is not None
                    and guard is not None
                    and guard.final_disposition == "accepted"
                ):
                    proposal = _role_proposal(
                        item, index, relation, role_core, guard, inspection.response_sha256
                    )
                    supported = has_true_query_role_witness(
                        hypothesis=role_core,
                        proposal=proposal,
                        predicates=tuple(predicates_by_id.values()),
                        event_grouping=event_grouping,
                        entity_registry=entity_registry,
                        query_session=query_session,
                    )
                    reason = "eligible_for_compilation" if supported else "no_true_event_support"
                    if supported:
                        proposals.append(proposal)
                scope_decisions.append(
                    SemanticRoleScopeDecision.model_validate(
                        dict(
                            local_id=item.proposal_id,
                            role_index=index,
                            relation_id=relation,
                            reason_code=reason,
                        )
                    )
                )
    ordered_predicates = tuple(
        sorted(predicates_by_id.values(), key=lambda p: p.predicate_result_id)
    )
    accepted = tuple(g for g in guards.values() if g.final_disposition == "accepted")
    role_result = None
    if proposals:
        role_result = compile_competition_roles(
            competition_manifest=manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            response_sha256=inspection.response_sha256,
            prompt_sha256=inspection.prompt_sha256,
            declarations=declarations,
            bindings=bindings,
            core_hypotheses=ordered_cores,
            proposals=tuple(proposals),
            guard_results=accepted,
            predicates=ordered_predicates,
            query_sessions_by_guard_result_id={g.guard_result_id: query_session for g in accepted},
            **common,
        )
    return SemanticFollowupCompilation(
        opaque_case_id=candidate_universe.opaque_case_id,
        initial_response_sha256=expected_response_sha256,
        initial_prompt_sha256=expected_prompt_sha256,
        query_session_sha256=expected_session_sha256,
        raw_response_hex=followup_raw_response.hex(),
        response_sha256=inspection.response_sha256,
        prompt_payload_hex=followup.payload_json.hex(),
        prompt_sha256=inspection.prompt_sha256,
        structurally_valid=inspection.structurally_valid,
        envelope_error=inspection.envelope_error,
        decisions=_decisions(inspection),
        core_hypotheses=ordered_cores,
        predicates=ordered_predicates,
        guard_results=tuple(sorted(guards.values(), key=lambda g: g.guard_result_id)),
        role_scope_decisions=tuple(scope_decisions),
        competition_manifest=manifest,
        role_compilation=role_result,
    )
