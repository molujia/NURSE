"""Pure conversion of inspected model proposals to versioned records.

These helpers never decide truth or execute queries. Their outputs need frozen
source verification before they can authorize ranking effects.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import ValidationError

from nurse_rcl.v2.competition_manifest import (
    CompetitionBinding,
    CompetitionBindingReason,
    CompetitionDeclaration,
)
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticHypothesis,
    PredicateResult,
    build_diagnostic_hypothesis,
)
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.role_compiler import RoleBindingProposal, build_role_binding_proposal
from nurse_rcl.v2.semantic_permissions import SemanticInspection
from nurse_rcl.v2.semantic_proposals import SemanticHypothesisProposal

SEMANTIC_COMPILER_VERSION: Literal["nurse-v2-semantic-compiler/1"] = "nurse-v2-semantic-compiler/1"


def _raw_hypotheses(raw: bytes, inspection: SemanticInspection) -> list[dict[str, Any]]:
    if inspection.envelope_error is not None:
        return []
    document = decode_json_object(raw, ExchangeLimits())
    return [item for item in document["hypotheses"] if type(item) is dict]


def _source_declarations(raw_items: list[dict[str, Any]]) -> tuple[CompetitionDeclaration, ...]:
    edges: set[tuple[str, str]] = set()
    for raw in raw_items:
        try:
            item = SemanticHypothesisProposal.model_validate_json(
                encode_json(raw, ExchangeLimits())
            )
        except ValidationError:
            continue  # Malformed records remain verbatim in the private receipt.
        edges.update(
            (item.proposal_id, target)
            for target in item.alternative_proposal_ids
            if target != item.proposal_id
        )
    return tuple(
        CompetitionDeclaration(source_local_id=a, target_local_id=b) for a, b in sorted(edges)
    )


def _core(
    item: SemanticHypothesisProposal,
    predicates: tuple[PredicateResult, ...],
    *,
    case_id: str,
    library: MechanismLibrary,
    response_sha256: str,
    prompt_sha256: str,
) -> DiagnosticHypothesis:
    return build_diagnostic_hypothesis(
        opaque_case_id=case_id,
        candidate_service_key=item.candidate_service_key,
        mechanism_key=item.mechanism_key,
        mechanism_definition_sha256=library.lookup(item.mechanism_key).definition_sha256,
        origin="semantic_agent",
        origin_version=SEMANTIC_COMPILER_VERSION,
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
        verification_request_ids=(),
        unknown_reasons=item.unknown_reasons,
        provenance_ids=(response_sha256, prompt_sha256, f"semantic-proposal:{item.proposal_id}"),
        lifecycle_status="proposed",
    )


def _source_bindings(
    raw_items: list[dict[str, Any]],
    inspection: SemanticInspection,
    cores: dict[str, DiagnosticHypothesis],
    failures: set[str],
) -> tuple[CompetitionBinding, ...]:
    local_ids = sorted(
        {
            raw["proposal_id"]
            for raw in raw_items
            if type(raw.get("proposal_id")) is str
            and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", raw["proposal_id"])
        }
    )
    decisions_by_id = {d.local_id: d for d in inspection.decisions if d.section == "hypotheses"}
    bindings = []
    for local_id in local_ids:
        bound_core = cores.get(local_id)
        decision = decisions_by_id.get(local_id)
        reason: CompetitionBindingReason = (
            "compiled" if bound_core is not None else "item_schema_invalid"
        )
        if local_id in failures:
            reason = "compilation_failed"
        elif bound_core is None and decision is not None:
            reason = (
                "duplicate_local_id"
                if decision.reason_code == "duplicate_local_id"
                else "permission_denied"
            )
        bindings.append(
            CompetitionBinding(
                local_id=local_id,
                hypothesis_id=None if bound_core is None else bound_core.hypothesis_id,
                reason_code=reason,
            )
        )
    return tuple(bindings)


def _role_proposal(
    item: SemanticHypothesisProposal,
    role_index: int,
    relation: str,
    core: DiagnosticHypothesis,
    guard: HypothesisGuardResult,
    response_sha256: str,
) -> RoleBindingProposal:
    role = item.role_proposals[role_index]
    return build_role_binding_proposal(
        guard_result_id=guard.guard_result_id,
        hypothesis_id=core.hypothesis_id,
        event_group_id=role.event_group_id,
        relation_id=relation,
        interval=TimeInterval(
            start_unix_ns=role.interval.start_unix_ns,
            end_unix_ns=role.interval.end_unix_ns,
        ),
        role=role.role,
        provenance_ids=(response_sha256, f"semantic-proposal:{item.proposal_id}"),
    )
