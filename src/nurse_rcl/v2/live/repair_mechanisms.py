"""Complete repair mechanism conditions over replayed fixed and paid evidence.

This is a current-version gate, never a forged legacy magnitude predicate or
guard receipt. A permitted role still needs event/relation/strength compilation
before scoring. The outer scorer owns independent raw-file/adapter replay; the
pins supplied here must come from that boundary. No caller-authored feedback,
subset of candidates, weights, or unreserved future measurements are accepted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import Field

if TYPE_CHECKING:
    from nurse_rcl.v2.live.normal_artifacts import NormalQuerySession
    from nurse_rcl.v2.live.normal_hybrid_queries import NormalHybridQuerySession
    from nurse_rcl.v2.live.normal_hybrid_tools import NormalHybridToolSession

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.diagnostic_schemas import DiagnosticRole
from nurse_rcl.v2.dynamic_verification.contracts import SHA256, TruthValue
from nurse_rcl.v2.dynamic_verification.safety import plain, require_pin
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACT_SHA256
from nurse_rcl.v2.live.repair_error_references import (
    REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256,
    RepairErrorSources,
    RepairReferencedErrorFacts,
    _replayed_referenced_errors,
)
from nurse_rcl.v2.live.repair_fixed_facts import (
    REPAIR_FIXED_SEMANTICS_SHA256,
    RepairErrorFactSummary,
    RepairFixedErrorFacts,
    RepairFixedSignalFacts,
    build_repair_fixed_signal_facts,
)
from nurse_rcl.v2.live.repair_measurements import RepairMeasuredPredicate
from nurse_rcl.v2.live.repair_module_timing import _module_call
from nurse_rcl.v2.live.repair_queries import REPAIR_QUERY_SEMANTICS_SHA256, RepairQueryCatalog
from nurse_rcl.v2.live.repair_query_session import (
    RepairQueryRuntime,
    RepairQuerySession,
    _read_journal,
)
from nurse_rcl.v2.live.repair_safety import (
    REPAIR_CAPACITY_SHA256,
    RepairModel,
    canonical,
    require_unchanged,
    snapshot,
)
from nurse_rcl.v2.mechanism_library import build_mechanism_library

REPAIR_MECHANISM_SCHEMA_VERSION: Final = "nurse.v2.live-repair-mechanism-evidence/1"
REPAIR_MECHANISM_SEMANTICS_SHA256: Final = sha256_hex(
    canonical_json_bytes(
        dict(
            version="nurse-v2-live-repair-mechanism-consumption/1",
            capacity=REPAIR_CAPACITY_SHA256,
            applicability=_PREDICATE_EVIDENCE_CONTRACT_SHA256,
            fixed=REPAIR_FIXED_SEMANTICS_SHA256,
            errors=REPAIR_ERROR_REFERENCE_SEMANTICS_SHA256,
            dynamic=REPAIR_QUERY_SEMANTICS_SHA256,
            mechanisms=build_mechanism_library().library_sha256,
            composition="support_nonempty_three_valued_OR_normal_nonempty_three_valued_AND",
            dynamic_precedence="same_source_same_mechanism_same_kind_paid_leaf_supersedes_fixed_leaf",
            complete="all_frozen_C_all_templates_all_conditions_all_applicable_and_unused_evidence",
            permission="all_necessary_true_only_optional_and_refuting_remain_for_role_compilation",
            authority="observable_mechanism_gate_not_raw_file_event_role_or_score_authority",
        )
    )
)


class RepairConditionLeaf(RepairModel):
    evidence_id: SHA256
    dynamic_predicate_sha256: SHA256 | None
    measurement_field: str
    truth_value: TruthValue
    reason_codes: tuple[str, ...]


class RepairNonApplicableEvidence(RepairModel):
    evidence_id: SHA256
    reason: Literal["outside_registered_predicate_evidence_domain"] = (
        "outside_registered_predicate_evidence_domain"
    )


class RepairMechanismCondition(RepairModel):
    condition_id: str
    candidate_service_key: str
    mechanism_key: str
    predicate_kind: str
    requirement: Literal["necessary", "optional", "refuting"]
    original_evidence_ids: tuple[str, ...]
    leaves: tuple[RepairConditionLeaf, ...]
    non_applicable: tuple[RepairNonApplicableEvidence, ...]
    operator: Literal["or", "and"]
    truth_value: TruthValue
    reason_codes: tuple[str, ...]


class RepairMechanismDecision(RepairModel):
    mechanism_id: str
    candidate_service_key: str
    mechanism_key: str
    mechanism_definition_sha256: SHA256
    condition_ids: tuple[str, ...]
    disposition: Literal["accepted", "rejected", "unknown"]
    permitted_roles: tuple[DiagnosticRole, ...]
    support_evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    causal_status: Literal["observable_contract_only"] = "observable_contract_only"


class RepairMechanismEvidence(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-mechanism-evidence/1"] = (
        REPAIR_MECHANISM_SCHEMA_VERSION
    )
    semantics_sha256: SHA256
    opaque_case_id: str
    catalog_sha256: SHA256
    session_sha256: SHA256
    candidate_service_keys: tuple[str, ...]
    signal_facts: tuple[RepairFixedSignalFacts, ...]
    error_facts: RepairReferencedErrorFacts
    dynamic_predicates: tuple[RepairMeasuredPredicate, ...]
    conditions: tuple[RepairMechanismCondition, ...]
    mechanisms: tuple[RepairMechanismDecision, ...]
    unbound_evidence_ids: tuple[str, ...]
    dynamic_queries_used: int = Field(ge=0, le=8)
    score_authority: Literal[False] = False
    result_sha256: SHA256


_ROOTS = (
    RepairMechanismEvidence,
    RepairMechanismCondition,
    RepairMechanismDecision,
    RepairConditionLeaf,
    RepairNonApplicableEvidence,
    RepairFixedSignalFacts,
    RepairReferencedErrorFacts,
    RepairFixedErrorFacts,
    RepairMeasuredPredicate,
    RepairQuerySession,
    RepairQueryCatalog,
    RepairErrorSources,
)


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(plain(value)))


def _compose(values: tuple[TruthValue, ...], operator: Literal["or", "and"]) -> TruthValue:
    if not values:
        return "unknown"
    decisive = "true" if operator == "or" else "false"
    if decisive in values:
        return cast(TruthValue, decisive)
    if "unknown" in values:
        return "unknown"
    return "false" if operator == "or" else "true"


@_module_call("mechanisms")
def _compile_values(
    *,
    catalog: RepairQueryCatalog,
    session: (
        RepairQuerySession | NormalQuerySession | NormalHybridQuerySession | NormalHybridToolSession
    ),
    signal_facts: tuple[RepairFixedSignalFacts, ...],
    error_facts: RepairReferencedErrorFacts,
) -> RepairMechanismEvidence:
    """Pure condition/permission projection; owning caller guards all inputs."""
    facts: tuple[RepairFixedSignalFacts | RepairFixedErrorFacts, ...] = (
        *signal_facts,
        *error_facts.facts,
    )
    ids = tuple(f.result_sha256 for f in facts)
    if len(set(ids)) != len(ids):
        raise ValueError("repair mechanism duplicate evidence identity")
    dynamic = tuple(
        sorted(
            (p for step in session.steps for p in step.feedback.predicates),
            key=lambda p: p.result_sha256,
        )
    )
    dynamic_by_key = {(p.observation_id, p.mechanism_key, p.predicate_kind): p for p in dynamic}
    if len(dynamic_by_key) != len(dynamic):
        raise ValueError("repair mechanism has repeated dynamic source conditions")
    by_observation = {f.observation_id: f for f in signal_facts}
    for p in dynamic:
        source = by_observation.get(p.observation_id)
        if (
            source is None
            or p.source_sha256 != source.source_sha256
            or p.candidate_service_key != source.candidate_service_key
            or p.support_source_key != source.support_source_key
            or p.interval != source.interval
        ):
            raise ValueError("repair mechanism dynamic leaf differs from fixed source binding")
    library = build_mechanism_library()
    conditions, mechanisms = [], []
    for candidate in catalog.candidate_service_keys:
        bound = tuple(
            sorted(
                (f for f in facts if f.candidate_service_key == candidate),
                key=lambda f: f.result_sha256,
            )
        )
        original_ids = tuple(f.result_sha256 for f in bound)
        for definition in library.definitions:
            current = []
            for requirement in ("necessary", "optional", "refuting"):
                for kind in getattr(definition, requirement + "_predicate_kinds"):
                    leaves, unused = [], []
                    for f in bound:
                        fixed = next((v for v in f.facts if v.predicate_kind == kind), None)
                        if fixed is None:
                            unused.append(RepairNonApplicableEvidence(evidence_id=f.result_sha256))
                            continue
                        measured = (
                            dynamic_by_key.get((f.observation_id, definition.mechanism_key, kind))
                            if type(f) is RepairFixedSignalFacts
                            else None
                        )
                        value = measured if measured is not None else fixed
                        leaves.append(
                            RepairConditionLeaf(
                                evidence_id=f.result_sha256,
                                dynamic_predicate_sha256=measured.result_sha256
                                if measured
                                else None,
                                measurement_field=value.measurement_field,
                                truth_value=value.truth_value,
                                reason_codes=value.reason_codes,
                            )
                        )
                    operator: Literal["or", "and"] = "and" if requirement == "refuting" else "or"
                    truth = _compose(tuple(leaf.truth_value for leaf in leaves), operator)
                    unsigned = dict(
                        candidate_service_key=candidate,
                        mechanism_key=definition.mechanism_key,
                        predicate_kind=kind,
                        requirement=requirement,
                        original_evidence_ids=original_ids,
                        leaves=tuple(leaves),
                        non_applicable=tuple(unused),
                        operator=operator,
                        truth_value=truth,
                        reason_codes=("no_applicable_evidence",)
                        if not leaves
                        else (
                            "nonempty_three_valued_" + operator,
                            "fixed_and_paid_leaves_retained",
                        ),
                    )
                    c = RepairMechanismCondition.model_validate(
                        dict(unsigned, condition_id="repair_condition_" + _digest(unsigned))
                    )
                    current.append(c)
            necessary = tuple(c for c in current if c.requirement == "necessary")
            support = _compose(tuple(c.truth_value for c in necessary), "and")
            disposition = {"true": "accepted", "false": "rejected", "unknown": "unknown"}[support]
            unsigned_row = dict(
                candidate_service_key=candidate,
                mechanism_key=definition.mechanism_key,
                mechanism_definition_sha256=definition.definition_sha256,
                condition_ids=tuple(c.condition_id for c in current),
                disposition=disposition,
                permitted_roles=definition.allowed_roles if support == "true" else (),
                support_evidence_ids=tuple(
                    sorted(
                        {
                            leaf.evidence_id
                            for c in necessary
                            for leaf in c.leaves
                            if leaf.truth_value == "true"
                        }
                    )
                ),
                reason_codes=("all_necessary_conditions_true_event_role_binding_required",)
                if support == "true"
                else ("necessary_predicate_" + support,),
            )
            mechanisms.append(
                RepairMechanismDecision.model_validate(
                    dict(unsigned_row, mechanism_id="repair_mechanism_" + _digest(unsigned_row))
                )
            )
            conditions.extend(current)
    draft = RepairMechanismEvidence(
        semantics_sha256=REPAIR_MECHANISM_SEMANTICS_SHA256,
        opaque_case_id=catalog.opaque_case_id,
        catalog_sha256=catalog.catalog_sha256,
        session_sha256=session.session_sha256,
        candidate_service_keys=catalog.candidate_service_keys,
        signal_facts=signal_facts,
        error_facts=error_facts,
        dynamic_predicates=dynamic,
        conditions=tuple(conditions),
        mechanisms=tuple(mechanisms),
        unbound_evidence_ids=tuple(
            sorted(f.result_sha256 for f in facts if f.candidate_service_key is None)
        ),
        dynamic_queries_used=(len(session.steps) if session.schema_version ==
                              "nurse.v2.normal-hybrid-tool-session/1" else session.paid_count),
        result_sha256="0" * 64,
    )
    values = cast(dict[str, object], plain(draft))
    values.pop("result_sha256")
    return draft.model_copy(update={"result_sha256": _digest(values)})


@contextmanager
def _replayed_mechanism_context(
    *,
    runtime: RepairQueryRuntime,
    expected_session_sha256: str,
    error_sources: RepairErrorSources,
    expected_error_sources_sha256: str,
    expected_error_observation_sha256s: tuple[str, ...],
) -> Iterator[
    tuple[RepairMechanismEvidence, RepairQuerySession, tuple[RepairErrorFactSummary, ...]]
]:
    """Replay the complete catalog, durable paid prefix and all fixed errors.

    No episode is computed for an unreserved source. The terminal session pin
    prevents an old or still-changing query prefix from becoming a frozen gate.
    Single-source scalar bounds and complete four-slot error bounds are retained;
    no case is truncated or split into separately ranked candidate subsets.
    """
    if type(runtime) is not RepairQueryRuntime or type(error_sources) is not RepairErrorSources:
        raise ValueError("repair mechanisms require exact current runtime and error sources")
    require_pin(expected_session_sha256)
    originals = (
        error_sources,
        expected_error_sources_sha256,
        expected_error_observation_sha256s,
        expected_session_sha256,
    )
    with ExitStack() as guards:
        guards.callback(require_unchanged, originals, _ROOTS, snapshot(originals, _ROOTS))
        with runtime._operation() as (catalog, _):
            session = runtime._replay(catalog, _read_journal(runtime.root))
            guards.callback(require_unchanged, session, _ROOTS, snapshot(session, _ROOTS))
            if (
                session.session_sha256 != expected_session_sha256
                or session.pending_selection is not None
                or session.failure_reason is not None
                or session.stop_reason is None
            ):
                raise ValueError(
                    "repair mechanism session is not the pinned terminal completed prefix"
                )
            args = runtime.arguments
            if any(s.dataset != error_sources.dataset for s in args["sources"]):
                raise ValueError("repair mechanism scalar/error dataset mismatch")
            errors, summaries = guards.enter_context(
                _replayed_referenced_errors(
                    sources=error_sources,
                    expected_sources_sha256=expected_error_sources_sha256,
                    expected_observation_sha256s=expected_error_observation_sha256s,
                    time_contract=args["time_contract"],
                    expected_time_contract_sha256=args["expected_time_contract_sha256"],
                    candidate_universe=args["candidate_universe"],
                    frozen_candidates=args["frozen_candidates"],
                )
            )
            guards.callback(require_unchanged, errors, _ROOTS, snapshot(errors, _ROOTS))
            fixed = tuple(
                build_repair_fixed_signal_facts(
                    source=s,
                    expected_source_sha256=pin,
                    time_contract=args["time_contract"],
                    expected_time_contract_sha256=args["expected_time_contract_sha256"],
                    store=args["store"],
                )
                for s, pin in zip(args["sources"], args["expected_source_sha256s"], strict=True)
            )
            guards.callback(require_unchanged, fixed, _ROOTS, snapshot(fixed, _ROOTS))
            result = _compile_values(
                catalog=catalog, session=session, signal_facts=fixed, error_facts=errors
            )
            # Check the full output, not each candidate/condition against a reset budget.
            canonical(result, _ROOTS)
            guards.callback(require_unchanged, result, _ROOTS, snapshot(result, _ROOTS))
            yield result, session, summaries


def build_repair_mechanism_evidence(
    *,
    runtime: RepairQueryRuntime,
    expected_session_sha256: str,
    error_sources: RepairErrorSources,
    expected_error_sources_sha256: str,
    expected_error_observation_sha256s: tuple[str, ...],
) -> RepairMechanismEvidence:
    """Unchanged public artifact; no cached trust escapes this invocation."""
    with _replayed_mechanism_context(
        runtime=runtime,
        expected_session_sha256=expected_session_sha256,
        error_sources=error_sources,
        expected_error_sources_sha256=expected_error_sources_sha256,
        expected_error_observation_sha256s=expected_error_observation_sha256s,
    ) as (result, _, _):
        return result


def require_repair_mechanism_evidence(
    *,
    evidence: RepairMechanismEvidence,
    expected_evidence_sha256: str,
    **source_arguments: Any,
) -> RepairMechanismEvidence:
    """Independent reconstruction rejects omission/forgery even after rehashing."""
    if type(evidence) is not RepairMechanismEvidence:
        raise ValueError("repair mechanism replay requires its exact current artifact")
    before = snapshot(evidence, _ROOTS)
    try:
        require_pin(expected_evidence_sha256)
        rebuilt = build_repair_mechanism_evidence(**source_arguments)
        if rebuilt.result_sha256 != expected_evidence_sha256 or canonical(
            rebuilt, _ROOTS
        ) != canonical(evidence, _ROOTS):
            raise ValueError("repair mechanism evidence differs from complete independent replay")
        return rebuilt
    finally:
        require_unchanged(evidence, _ROOTS, before)
