"""Source-replayed, byte-budgeted evidence for the post-ranking Report Agent."""

from __future__ import annotations

import hmac
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.case_audit import ImmutableCaseAuditPackage
from nurse_rcl.v2.case_audit_read import SupportedCaseAudit
from nurse_rcl.v2.case_audit_v2 import ImmutableCaseAuditPackageV2
from nurse_rcl.v2.deterministic_report import (
    DeterministicSREReport,
    _report_input_snapshot,
    build_deterministic_sre_report,
)
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.inference_leakage import audit_inference_boundary
from nurse_rcl.v2.observation_schemas import TimeInterval
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_context import (
    FrozenReportContext,
    FrozenReportPayload,
    ReportHypothesisFact,
)
from nurse_rcl.v2.report_context_v4 import (
    FrozenReportContextV4,
    FrozenReportPayloadV2,
    _report_unchanged,
)
from nurse_rcl.v2.report_evidence_schemas import (
    ClaimStrength,
    EvidenceKind,
    EvidenceScalar,
    OmissionReason,
    ReportEvidenceBinding,
    ReportEvidenceFact,
    ReportEvidenceOmission,
    ReportEvidencePolicy,
    ReportEvidenceProjection,
    ReportEvidenceValue,
    ReportEvidenceView,
    ScopeBasis,
)
from nurse_rcl.v2.report_source_read import (
    SupportedReportContext,
    _preflight_report_sources,
    require_report_source_read_view,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_OmissionKey = tuple[EvidenceKind, tuple[str, ...], OmissionReason]
_SupportedReportPayload: TypeAlias = FrozenReportPayload | FrozenReportPayloadV2
_InputSnapshot: TypeAlias = tuple[object, tuple[object, ...]]
_INPUT_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)


def _source_types(
    context: SupportedReportContext, audit_package: SupportedCaseAudit
) -> tuple[type[BaseModel], type[BaseModel]]:
    # Exact code-owned pairs only: an untrusted runtime class never grants authority.
    if type(context) is FrozenReportContext and type(audit_package) is ImmutableCaseAuditPackage:
        return FrozenReportContext, ImmutableCaseAuditPackage
    if (
        type(context) is FrozenReportContextV4
        and type(audit_package) is ImmutableCaseAuditPackageV2
    ):
        return FrozenReportContextV4, ImmutableCaseAuditPackageV2
    raise ValueError("report requires audit /1 + context /3 or audit /2 + context /4")


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _snapshot(value: _ModelT) -> _ModelT:
    original_bytes = canonical_json_bytes(value.model_dump(mode="json", warnings="error"))
    snapshot = type(value).model_validate_json(original_bytes)
    snapshot_bytes = canonical_json_bytes(snapshot.model_dump(mode="json", warnings="error"))
    if not hmac.compare_digest(original_bytes, snapshot_bytes):
        raise ValueError("report evidence snapshot changes exact canonical source bytes")
    return snapshot


def _evidence_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Existing evidence structure, over detached exact data before model callbacks.

    Integrity hashes and source replay remain with the original validators. These
    local checks neither qualify a source nor compute a projected fact or budget.
    """
    if cls is TimeInterval:
        if state["end_unix_ns"] <= state["start_unix_ns"]:
            raise ValueError("time interval end must be greater than start")
    elif cls is ReportEvidencePolicy:
        if state["configuration_version"] == "default/1" and (
            state["max_view_bytes"] != 16384 or state["max_detail_facts"] != 64
        ):
            raise ValueError("policy overrides require a nondefault configuration_version")
    elif cls is ReportEvidenceFact:
        targets = state["target_service_keys"]
        if len(targets) != len(set(targets)):
            raise ValueError("fact targets must be unique")
        names = [value["name"] for value in state["values"]]
        if len(names) != len(set(names)):
            raise ValueError("fact value names must be unique")
        for field in ("status_codes", "related_fact_refs"):
            if state[field] != sorted(set(state[field])):
                raise ValueError("fact codes and references must be sorted and unique")
    elif cls is ReportEvidenceOmission:
        targets = state["target_service_keys"]
        if len(targets) != len(set(targets)):
            raise ValueError("omission targets must be unique")
    elif cls is ReportEvidenceBinding:
        for field in ("source_fields", "same_source_ids"):
            if state[field] != sorted(set(state[field])):
                raise ValueError("source fields and identities must be sorted and unique")
    elif cls is ReportEvidenceView:
        top_k, facts = state["top_k"], state["facts"]
        if len(top_k) != len(set(top_k)):
            raise ValueError("view Top-K must be unique")
        refs = [fact["fact_ref"] for fact in facts]
        if refs != [f"f{i}" for i in range(1, len(refs) + 1)]:
            raise ValueError("view aliases must be complete and ordered")
        mandatory = state["mandatory_fact_refs"]
        if mandatory != refs[: len(mandatory)]:
            raise ValueError("mandatory facts must form a nonempty ordered prefix")
        for fact in facts:
            if not set(fact["related_fact_refs"]) <= set(refs):
                raise ValueError("view fact references must have closure")
            if not set(fact["target_service_keys"]) <= set(top_k):
                raise ValueError("view fact targets must be in Top-K")
        omission_keys = [
            (row["kind"], tuple(row["target_service_keys"]), row["reason_code"])
            for row in state["omissions"]
        ]
        if omission_keys != sorted(set(omission_keys)):
            raise ValueError("view omissions must be grouped and sorted")
    elif cls is ReportEvidenceProjection:
        view, policy = state["view"], state["policy"]
        if (state["status"] == "ready") != (view is not None):
            raise ValueError("ready projection must have a view")
        if view is None:
            if (
                state["source_bindings"]
                or state["mandatory_view_bytes"] <= policy["max_view_bytes"]
            ):
                raise ValueError("fallback requires exhausted mandatory budget and no bindings")
        else:
            if state["mandatory_view_bytes"] > policy["max_view_bytes"]:
                raise ValueError("ready projection must fit its mandatory byte budget")
            if [row["fact_ref"] for row in state["source_bindings"]] != [
                fact["fact_ref"] for fact in view["facts"]
            ]:
                raise ValueError("bindings must cover exactly the selected facts")
            foundation = state["foundation"]
            if view["top_k"] != [row["service_entity_key"] for row in foundation["top_k"]]:
                raise ValueError("view Top-K differs from foundation")
            if (
                view["context_sha256"] != foundation["context_sha256"]
                or view["opaque_case_id"] != foundation["opaque_case_id"]
            ):
                raise ValueError("view source identity differs from projection")


def _preflight_sources(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    identity_manifest: InferenceIdentityManifest,
    policy: ReportEvidencePolicy,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    expected_identity_manifest_sha256: str,
    expected_policy_sha256: str,
    extra_roots: tuple[tuple[object, type[BaseModel]], ...] = (),
) -> tuple[_InputSnapshot, _InputSnapshot, tuple[_InputSnapshot, ...]]:
    context_type, audit_type = _source_types(context, audit_package)
    roots: tuple[tuple[object, type[BaseModel]], ...] = (
        (context, context_type),
        (audit_package, audit_type),
        (identity_manifest, InferenceIdentityManifest),
        (policy, ReportEvidencePolicy),
        *extra_roots,
    )
    for value, expected_type in roots:
        if type(value) is not expected_type:
            raise ValueError(f"report evidence requires exact {expected_type.__name__} type")
    for pin in (
        expected_audit_sha256,
        expected_context_sha256,
        expected_identity_manifest_sha256,
        expected_policy_sha256,
    ):
        if type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
            raise ValueError(
                "report evidence pins require exact 64-character lowercase hex strings"
            )
    # Complete every root and pin before any serializer, property, or replay.
    for value, expected_type in roots:
        preflight_exact_pydantic_tree(
            value,
            root_model_types=(expected_type,),
            label="report evidence source",
            max_string_length=None,
        )
    context_original, audit_original, _ = _preflight_report_sources(
        context=context,
        audit_package=audit_package,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
    )
    # These are report/manifest roots, not source JSON: retain the original per-root
    # 500,000-node contract without importing the sources' 64 MiB byte budget.
    additional = tuple(
        _report_input_snapshot(
            value, model_type=cls, max_nodes=500_000, local_domains=_evidence_local_domains
        )
        for value, cls in roots[2:]
    )
    return context_original, audit_original, additional


@contextmanager
def _preserve_evidence_sources(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    identity_manifest: InferenceIdentityManifest,
    policy: ReportEvidencePolicy,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    expected_identity_manifest_sha256: str,
    expected_policy_sha256: str,
    extra_roots: tuple[tuple[object, type[BaseModel]], ...] = (),
) -> Iterator[None]:
    context_original, audit_original, additional = _preflight_sources(
        context=context,
        audit_package=audit_package,
        identity_manifest=identity_manifest,
        policy=policy,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        extra_roots=extra_roots,
    )
    roots: tuple[tuple[object, type[BaseModel]], ...] = (
        (identity_manifest, InferenceIdentityManifest),
        (policy, ReportEvidencePolicy),
        *extra_roots,
    )
    try:
        yield
    finally:
        unchanged = True
        # Inspect every original even when another root or the callback has failed.
        for source_value, original in (
            (context, context_original),
            (audit_package, audit_original),
        ):
            try:
                _report_unchanged(source_value, original)
            except _INPUT_ERRORS:
                unchanged = False
        for (value, cls), original in zip(roots, additional, strict=True):
            try:
                if (
                    _report_input_snapshot(
                        value,
                        model_type=cls,
                        max_nodes=500_000,
                        local_domains=_evidence_local_domains,
                    )
                    != original
                ):
                    unchanged = False
            except _INPUT_ERRORS:
                unchanged = False
        if not unchanged:
            raise ValueError(
                "report evidence original content or representation changed during reading"
            )


@dataclass(frozen=True)
class _Draft:
    key: str
    fact: ReportEvidenceFact
    binding: ReportEvidenceBinding
    related_keys: tuple[str, ...] = ()


def _draft(
    *,
    key: str,
    kind: EvidenceKind,
    source: BaseModel,
    source_id: str,
    source_kind: str,
    source_fields: tuple[str, ...],
    targets: tuple[str, ...] = (),
    values: dict[str, EvidenceScalar] | None = None,
    codes: tuple[str, ...] = (),
    intervals: tuple[TimeInterval, ...] = (),
    same_source_ids: tuple[str, ...] = (),
    related: tuple[str, ...] = (),
    strength: ClaimStrength = "ranking_inference",
    scope: ScopeBasis = "frozen_assessment",
) -> _Draft:
    if not set(source_fields) <= set(type(source).model_fields):
        raise AssertionError("source bindings must name actual source fields")
    return _Draft(
        key=key,
        fact=ReportEvidenceFact(
            fact_ref="f1",
            kind=kind,
            target_service_keys=targets,
            intervals=intervals,
            values=tuple(ReportEvidenceValue(name=k, value=v) for k, v in (values or {}).items()),
            status_codes=tuple(sorted(set(codes))),
            allowed_claim_strength=strength,
            scope_basis=scope,
        ),
        binding=ReportEvidenceBinding(
            fact_ref="f1",
            source_kind=source_kind,
            source_id=source_id,
            source_sha256=_digest(source.model_dump(mode="json", warnings="error")),
            source_fields=tuple(sorted(set(source_fields))),
            same_source_ids=tuple(sorted(set(same_source_ids))),
        ),
        related_keys=related,
    )


def _scalar_fields(source: BaseModel, names: tuple[str, ...]) -> dict[str, EvidenceScalar]:
    values: dict[str, EvidenceScalar] = {}
    for name in names:
        value = getattr(source, name)
        if type(value) not in (str, int, float, bool, type(None)):
            raise AssertionError("selected field is not a scalar")
        values[name] = value
    return values


def _predicate_index(payload: _SupportedReportPayload) -> dict[str, PredicateResult]:
    """Index every nested record once; frozen payload roots need not be flattened."""
    records: dict[str, PredicateResult] = {}
    canonical_records: dict[str, bytes] = {}
    pending = list(payload.predicates)
    while pending:
        record = pending.pop()
        identifier = record.predicate_result_id
        canonical = canonical_json_bytes(record.model_dump(mode="json", warnings="error"))
        if identifier in records:
            if not hmac.compare_digest(canonical_records[identifier], canonical):
                raise ValueError("report evidence predicate ID collision")
            continue
        records[identifier] = record
        canonical_records[identifier] = canonical
        pending.extend(record.operand_states)
    return {identifier: records[identifier] for identifier in sorted(records)}


def _parents(payload: _SupportedReportPayload) -> dict[str, tuple[ReportHypothesisFact, ...]]:
    result: dict[str, list[ReportHypothesisFact]] = {}
    predicates = _predicate_index(payload)
    for hypothesis in payload.hypotheses:
        pending = list(
            (
                *hypothesis.necessary_predicate_result_ids,
                *hypothesis.optional_predicate_result_ids,
                *hypothesis.refuting_predicate_result_ids,
            )
        )
        visited: set[str] = set()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            result.setdefault(pid, []).append(hypothesis)
            pending.extend(p.predicate_result_id for p in predicates[pid].operand_states)
    return {key: tuple(value) for key, value in result.items()}


@dataclass(frozen=True)
class _PredicateScope:
    hypotheses: tuple[ReportHypothesisFact, ...]
    assessed_targets: frozenset[str]
    # Child -> direct logical parents, separately for each declared candidate.
    logical_parents: dict[str, frozenset[str]]


def _predicate_scopes(
    foundation: DeterministicSREReport,
    payload: _SupportedReportPayload,
    predicates: dict[str, PredicateResult],
) -> dict[str, _PredicateScope]:
    parents = _parents(payload)
    assessed: dict[str, set[str]] = {}
    for candidate in foundation.top_k:
        evidence = candidate.evidence
        for pid in (
            *evidence.supporting_predicate_result_ids,
            *evidence.refuting_predicate_result_ids,
            *evidence.unknown_predicate_result_ids,
        ):
            assessed.setdefault(pid, set()).add(candidate.service_entity_key)
    logical: dict[str, dict[str, set[str]]] = {}
    for pid, predicate in predicates.items():
        for hypothesis in parents.get(pid, ()):
            target = hypothesis.candidate_service_key
            if target is None:
                continue
            for operand in predicate.operand_states:
                logical.setdefault(operand.predicate_result_id, {}).setdefault(target, set()).add(
                    pid
                )
    return {
        pid: _PredicateScope(
            hypotheses=parents.get(pid, ()),
            assessed_targets=frozenset(assessed.get(pid, ())),
            logical_parents={
                target: frozenset(ids) for target, ids in logical.get(pid, {}).items()
            },
        )
        for pid in predicates
    }


def _mandatory_predicate_targets(
    foundation: DeterministicSREReport,
    predicates: dict[str, PredicateResult],
    scopes: dict[str, _PredicateScope],
) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
    ranks = {c.service_entity_key: c.rank for c in foundation.top_k}
    required = {
        pid
        for candidate in foundation.top_k
        for pid in (
            *candidate.evidence.refuting_predicate_result_ids,
            *candidate.evidence.unknown_predicate_result_ids,
        )
    }
    for pid, predicate in predicates.items():
        # Operand state is not an independent frozen assessment or evidence weight.
        if set(scopes[pid].logical_parents) & ranks.keys() and (
            predicate.truth_value == "unknown"
            or (predicate.requirement == "necessary" and predicate.truth_value == "false")
            or (predicate.requirement == "refuting" and predicate.truth_value == "true")
        ):
            required.add(pid)
    ancestors: dict[str, set[str]] = {}
    pending = list(required)
    while pending:
        pid = pending.pop()
        # Shared records retain all Top-K roles and all corresponding logical paths.
        for target, parent_ids in scopes[pid].logical_parents.items():
            if target not in ranks:
                continue
            for parent_id in parent_ids:
                ancestors.setdefault(parent_id, set()).add(target)
                if parent_id not in required:
                    required.add(parent_id)
                    pending.append(parent_id)
    targets = {
        pid: tuple(
            sorted(
                (
                    scopes[pid].assessed_targets
                    | {
                        h.candidate_service_key
                        for h in scopes[pid].hypotheses
                        if h.candidate_service_key is not None
                    }
                )
                & ranks.keys(),
                key=ranks.__getitem__,
            )
        )
        for pid in required
    }
    return targets, ancestors


def _predicate_draft(
    predicate: PredicateResult,
    *,
    targets: tuple[str, ...],
    parents: tuple[ReportHypothesisFact, ...],
    adverse: bool,
    predicate_scope: _PredicateScope,
    mandatory_ids: set[str],
    ancestor_targets: set[str] | None = None,
) -> _Draft:
    fields = ("predicate_kind", "truth_value", "requirement", "reason", "logical_operator")
    values = _scalar_fields(predicate, fields)
    codes = {f"guard_{parent.final_disposition}" for parent in parents}
    for code, name, candidates in (
        ("foundation_assessed_predicate", "foundation_assessed", predicate_scope.assessed_targets),
        ("nested_operand_state", "nested_operand", set(predicate_scope.logical_parents)),
        ("logical_ancestor_state", "logical_ancestor", ancestor_targets or set()),
    ):
        role_targets = tuple(t for t in targets if t in candidates)
        if role_targets:
            codes.add(code)
        for index, target in enumerate(role_targets, 1):
            values[f"{name}_candidate_{index}"] = target
    parent_ids = {
        pid for target in targets for pid in predicate_scope.logical_parents.get(target, ())
    }
    if not adverse:
        # A unique measured scalar and its unit remain distinct named values.
        for index, measured in enumerate(predicate.measured_values, 1):
            values[f"measurement_{index}_name"] = measured.name
            values[f"measurement_{index}_value"] = measured.value
            values[f"measurement_{index}_unit"] = measured.unit
    return _draft(
        key=f"predicate:{predicate.predicate_result_id}",
        kind="predicate",
        source=predicate,
        source_id=predicate.predicate_result_id,
        source_kind="predicate",
        source_fields=(
            *fields,
            "operand_states",
            "observation_ids",
            "provenance_ids",
            "measured_values",
            "verification_result_ids",
        ),
        targets=targets,
        values=values,
        codes=tuple(sorted(codes)),
        intervals=tuple(
            sorted(
                {
                    (i.start_unix_ns, i.end_unix_ns): i
                    for parent in parents
                    for i in parent.time_intervals
                }.values(),
                key=lambda i: (i.start_unix_ns, i.end_unix_ns),
            )
        ),
        same_source_ids=tuple(
            sorted(
                {
                    *predicate.observation_ids,
                    *predicate.verification_result_ids,
                    *parent_ids,
                    *(operand.predicate_result_id for operand in predicate.operand_states),
                    *(parent.guard_result_id for parent in parents),
                }
            )
        ),
        related=(
            *(f"evidence:{target}" for target in targets),
            # Optional parents may not survive budgeting. Only mandatory aliases are safe.
            *(f"predicate:{pid}" for pid in sorted(parent_ids & mandatory_ids)),
        ),
        scope="declared_hypothesis" if parents else "bound_observation",
        strength="ranking_inference" if parents else "frozen_observation",
    )


def _mandatory_drafts(
    foundation: DeterministicSREReport,
    payload: _SupportedReportPayload,
) -> tuple[list[_Draft], set[str]]:
    drafts: list[_Draft] = []
    predicates = _predicate_index(payload)
    scopes = _predicate_scopes(foundation, payload, predicates)
    mandatory_targets, ancestor_targets = _mandatory_predicate_targets(
        foundation, predicates, scopes
    )
    mandatory_ids = set(mandatory_targets)
    rank_fields = (
        "rank",
        "quantized_total_score",
        "quantized_event_evidence_score",
        "quantized_propagation_score",
        "rule_score",
        "history_score",
        "tie_break_is_evidence",
    )
    ties = {t.tie_group_id: t for t in payload.ranking_result.evidence_tie_groups}
    for candidate in foundation.top_k:
        values = _scalar_fields(candidate, rank_fields)
        values["tie_group_member_count"] = len(
            ties[candidate.evidence_tie_group_id].member_service_keys
        )
        drafts.append(
            _draft(
                key=f"ranking:{candidate.service_entity_key}",
                kind="ranking",
                source=candidate,
                source_id=candidate.score_sha256,
                source_kind="ranked_candidate",
                source_fields=(*rank_fields, "evidence_tie_group_id"),
                targets=(candidate.service_entity_key,),
                values=values,
                same_source_ids=(candidate.evidence_tie_group_id,),
            )
        )
    count_fields = {
        "supporting_event_group_ids": "supporting_event_group_count",
        "supporting_hypothesis_ids": "supporting_hypothesis_count",
        "supporting_predicate_result_ids": "supporting_predicate_count",
        "refuting_predicate_result_ids": "refuting_predicate_count",
        "conflicting_assertion_ids": "conflicting_assertion_count",
        "missing_reason_codes": "missing_reason_count",
        "unknown_predicate_result_ids": "unknown_predicate_count",
    }
    for candidate in foundation.top_k:
        evidence = candidate.evidence
        adverse_ids = {
            pid
            for pid, targets in mandatory_targets.items()
            if candidate.service_entity_key in targets
        }
        drafts.append(
            _draft(
                key=f"evidence:{candidate.service_entity_key}",
                kind="candidate_evidence",
                source=evidence,
                source_kind="candidate_evidence",
                source_id=candidate.service_entity_key,
                source_fields=tuple(count_fields),
                targets=(candidate.service_entity_key,),
                values={
                    name: len(getattr(evidence, field)) for field, name in count_fields.items()
                },
                codes=evidence.missing_reason_codes,
                same_source_ids=evidence.supporting_event_group_ids,
                related=tuple(
                    sorted(
                        [
                            *(f"predicate:{pid}" for pid in adverse_ids),
                            *(f"role:{rid}" for rid in evidence.conflicting_assertion_ids),
                        ]
                    )
                ),
            )
        )
    basis = foundation.top_1_basis
    drafts.append(
        _draft(
            key="top_1_sufficiency",
            kind="uncertainty",
            source=basis,
            source_kind="top_one_basis",
            source_id=basis.score_sha256,
            source_fields=("evidence_sufficiency", "claim_strength"),
            targets=(basis.service_entity_key,),
            values=_scalar_fields(basis, ("evidence_sufficiency",)),
        )
    )
    competition_fields = (
        "competing_rank",
        "total_score_gap",
        "event_evidence_gap",
        "propagation_gap",
        "rule_gap",
        "history_gap",
        "evidence_tied",
        "ordering_decided_by_tie_break",
    )
    for competition in foundation.competition:
        drafts.append(
            _draft(
                key=f"competition:{competition.competition_id}",
                kind="competition",
                source=competition,
                source_kind="competition",
                source_id=competition.competition_id,
                source_fields=competition_fields,
                targets=(competition.leading_service_key, competition.competing_service_key),
                values=_scalar_fields(competition, competition_fields),
            )
        )
    run_fields = (
        "run_mode",
        "verification_action_limit",
        "verification_actions_truncated_count",
        "conditional_mitigation_limit",
        "conditional_mitigations_truncated_count",
    )
    drafts.append(
        _draft(
            key="run",
            kind="uncertainty",
            source=foundation,
            source_kind="deterministic_report",
            source_id=foundation.report_id,
            source_fields=(*run_fields, "degradation_reason_codes"),
            values=_scalar_fields(foundation, run_fields),
            codes=foundation.degradation_reason_codes,
        )
    )
    uncertainty_fields = (
        "evidence_sufficiency",
        "has_evidence_tie",
        "has_unknown_evidence",
        "has_conflicting_evidence",
        "has_missing_evidence",
    )
    drafts.append(
        _draft(
            key="uncertainty",
            kind="uncertainty",
            source=foundation.uncertainty,
            source_kind="uncertainty",
            source_id=foundation.report_id,
            source_fields=(*uncertainty_fields, "degradation_reason_codes"),
            values=_scalar_fields(foundation.uncertainty, uncertainty_fields),
            codes=foundation.uncertainty.degradation_reason_codes,
        )
    )
    for limit in sorted(foundation.limits, key=lambda item: (item.limit_code, item.limit_id)):
        drafts.append(
            _draft(
                key=f"limit:{limit.limit_id}",
                kind="limit",
                source=limit,
                source_kind="report_limit",
                source_id=limit.limit_id,
                source_fields=("limit_code", "severity"),
                values={"severity": limit.severity},
                codes=(limit.limit_code,),
            )
        )
    ranks = {c.service_entity_key: c.rank for c in foundation.top_k}
    for pid in sorted(mandatory_targets, key=lambda pid: (ranks[mandatory_targets[pid][0]], pid)):
        targets = mandatory_targets[pid]
        drafts.append(
            _predicate_draft(
                predicates[pid],
                targets=targets,
                parents=tuple(
                    p for p in scopes[pid].hypotheses if p.candidate_service_key in targets
                ),
                adverse=True,
                predicate_scope=scopes[pid],
                mandatory_ids=mandatory_ids,
                ancestor_targets=ancestor_targets.get(pid),
            )
        )
    assertions = {a.assertion_id: a for a in payload.role_compilation.assertions}
    for candidate in foundation.top_k:
        for rid in candidate.evidence.conflicting_assertion_ids:
            assertion = assertions[rid]
            guards = tuple(
                p for p in payload.hypotheses if p.hypothesis_id == assertion.hypothesis_id
            )
            drafts.append(
                _draft(
                    key=f"role:{rid}",
                    kind="candidate_evidence",
                    source=assertion,
                    source_kind="role_assertion",
                    source_id=rid,
                    source_fields=(
                        "role",
                        "lifecycle_status",
                        "interval",
                        "hypothesis_id",
                        "event_group_id",
                        "relation_id",
                    ),
                    targets=(candidate.service_entity_key,),
                    intervals=(assertion.interval,),
                    values={"role": assertion.role, "lifecycle_status": assertion.lifecycle_status},
                    codes=tuple(f"guard_{g.final_disposition}" for g in guards),
                    same_source_ids=(
                        assertion.event_group_id,
                        assertion.relation_id,
                        *(g.guard_result_id for g in guards),
                    ),
                    related=(f"evidence:{candidate.service_entity_key}",),
                    scope="declared_hypothesis",
                )
            )
    return drafts, mandatory_ids


def _optional_drafts(
    foundation: DeterministicSREReport,
    payload: _SupportedReportPayload,
    adverse_ids: set[str],
) -> tuple[list[_Draft], Counter[_OmissionKey]]:
    drafts: list[_Draft] = []
    omissions: Counter[_OmissionKey] = Counter()
    ranks = {c.service_entity_key: c.rank for c in foundation.top_k}
    resolutions = {r.observation.observation_id: r for r in payload.entity_registry.resolutions}
    bound_targets: dict[str, str] = {}
    for observation in payload.observations:
        resolution = resolutions.get(observation.observation_id)
        if (
            resolution is None
            or resolution.binding.status != "bound"
            or resolution.binding.service_entity_key is None
        ):
            omissions[("observation", (), "unbound_observation")] += 1
            continue
        target = resolution.binding.service_entity_key
        bound_targets[observation.observation_id] = target
        if target not in ranks:
            omissions[("observation", (target,), "outside_top_k")] += 1
            continue
        fields: tuple[str, ...] = (
            "modality",
            "signal_family",
            "signal_type",
            "unit",
            "evidence_state",
            "anomaly_strength",
            "coverage_quality",
        )
        values = _scalar_fields(observation, fields)
        for prefix, summary in (
            ("baseline", observation.baseline_summary),
            ("fault", observation.fault_summary),
        ):
            for name in (
                "record_state",
                "valid_point_count",
                "coverage_fraction",
                "median",
                "event_count",
            ):
                values[f"{prefix}_{name}"] = getattr(summary, name)
        drafts.append(
            _draft(
                key=f"observation:{observation.observation_id}",
                kind="observation",
                source=observation,
                source_kind="observation",
                source_id=observation.observation_id,
                source_fields=(
                    *fields,
                    "baseline_summary",
                    "fault_summary",
                    "interval",
                    "quality_flags",
                    "entity_binding_id",
                    "source_fingerprint",
                ),
                targets=(target,),
                values=values,
                codes=observation.quality_flags,
                intervals=(observation.interval,),
                related=(f"evidence:{target}",),
                same_source_ids=(
                    observation.source_fingerprint,
                    resolution.resolution_id,
                    *(
                        e.event_group_id
                        for e in payload.event_groups
                        if observation.observation_id in e.observation_ids
                    ),
                ),
                strength="frozen_observation",
                scope="bound_observation",
            )
        )
    predicates = _predicate_index(payload)
    for hypothesis in payload.hypotheses:
        hypothesis_target = hypothesis.candidate_service_key
        if hypothesis_target is not None and hypothesis_target not in ranks:
            omissions[("hypothesis", (hypothesis_target,), "outside_top_k")] += 1
            continue
        fields = ("mechanism_key", "final_disposition", "lifecycle_status")
        values = (
            _scalar_fields(hypothesis, fields)
            if hypothesis_target is not None
            else {"rejected_record_count": 1}
        )
        for category in ("necessary", "optional", "refuting"):
            ids = getattr(hypothesis, f"{category}_predicate_result_ids")
            if hypothesis_target is not None:
                for truth in ("true", "false", "unknown"):
                    values[f"{category}_{truth}_count"] = sum(
                        predicates[pid].truth_value == truth for pid in ids
                    )
        drafts.append(
            _draft(
                key=f"hypothesis:{hypothesis.guard_result_id}",
                kind="hypothesis",
                source=hypothesis,
                source_kind="report_hypothesis",
                source_id=hypothesis.guard_result_id,
                source_fields=(
                    *fields,
                    "guard_reason_codes",
                    "unknown_reasons",
                    "necessary_predicate_result_ids",
                    "optional_predicate_result_ids",
                    "refuting_predicate_result_ids",
                    "time_intervals",
                ),
                targets=() if hypothesis_target is None else (hypothesis_target,),
                values=values,
                codes=(*hypothesis.guard_reason_codes, *hypothesis.unknown_reasons),
                intervals=hypothesis.time_intervals,
                same_source_ids=hypothesis.event_group_ids,
                related=() if hypothesis_target is None else (f"evidence:{hypothesis_target}",),
                scope="declared_hypothesis",
            )
        )
    scopes = _predicate_scopes(foundation, payload, predicates)
    for predicate in predicates.values():
        pid = predicate.predicate_result_id
        if pid in adverse_ids:
            continue
        declared = scopes[pid].hypotheses
        scoped = {p.candidate_service_key for p in declared if p.candidate_service_key is not None}
        if not scoped:
            resolved = {bound_targets.get(oid) for oid in predicate.observation_ids}
            # A predicate's aggregate measurements cannot be attributed from only
            # some of its observations, or split across unrelated service targets.
            scoped = (
                {t for t in resolved if t is not None}
                if (len(resolved) == 1 and None not in resolved)
                else set()
            )
        targets = tuple(sorted(scoped & ranks.keys(), key=ranks.__getitem__))
        if not targets:
            reason: OmissionReason = "outside_top_k" if scoped else "unscoped_predicate"
            omissions[("predicate", tuple(sorted(scoped)), reason)] += 1
            continue
        drafts.append(
            _predicate_draft(
                predicate,
                targets=targets,
                parents=tuple(p for p in declared if p.candidate_service_key in targets),
                adverse=False,
                predicate_scope=scopes[pid],
                mandatory_ids=adverse_ids,
            )
        )
    for action in foundation.verification_actions:
        values = _scalar_fields(
            action,
            ("action_class", "do_not_execute_automatically", "does_not_modify_sealed_ranking"),
        )
        values["status"] = "proposed"
        drafts.append(
            _draft(
                key=f"verification:{action.action_id}",
                kind="verification",
                source=action,
                source_kind="verification_action",
                source_id=action.action_id,
                source_fields=(
                    "target_service_key",
                    "action_class",
                    "possible_outcomes",
                    "do_not_execute_automatically",
                    "does_not_modify_sealed_ranking",
                    "predicate_result_id",
                    "verification_request_ids",
                ),
                targets=(action.target_service_key,),
                values=values,
                codes=tuple(outcome.interpretation_key for outcome in action.possible_outcomes),
                related=(f"evidence:{action.target_service_key}",),
                same_source_ids=(
                    action.hypothesis_id,
                    action.predicate_result_id,
                    *action.verification_request_ids,
                ),
                strength="conditional_recommendation",
            )
        )
    for mitigation in foundation.conditional_mitigations:
        fields = (
            "action_class",
            "operator_approval_required",
            "reversible_action_only",
            "do_not_execute_automatically",
        )
        values = _scalar_fields(mitigation, fields)
        values["status"] = "proposed"
        drafts.append(
            _draft(
                key=f"mitigation:{mitigation.mitigation_id}",
                kind="mitigation",
                source=mitigation,
                source_kind="conditional_mitigation",
                source_id=mitigation.mitigation_id,
                source_fields=(*fields, "possible_outcomes", "precondition_predicate_result_ids"),
                targets=(mitigation.target_service_key,),
                values=values,
                codes=mitigation.possible_outcomes,
                related=(f"evidence:{mitigation.target_service_key}",),
                same_source_ids=(
                    mitigation.hypothesis_id,
                    *mitigation.precondition_predicate_result_ids,
                ),
                strength="conditional_recommendation",
            )
        )
    drafts.sort(
        key=lambda d: (
            d.fact.kind,
            min((ranks[t] for t in d.fact.target_service_keys), default=0),
            d.binding.source_id,
        )
    )
    return drafts, omissions


def _view_and_bindings(
    *,
    foundation: DeterministicSREReport,
    policy: ReportEvidencePolicy,
    mandatory: list[_Draft],
    selected: list[_Draft],
    omissions: Counter[_OmissionKey],
) -> tuple[ReportEvidenceView, tuple[ReportEvidenceBinding, ...]]:
    drafts = [*mandatory, *selected]
    aliases = {draft.key: f"f{i}" for i, draft in enumerate(drafts, 1)}
    facts = tuple(
        d.fact.model_copy(
            update={
                "fact_ref": aliases[d.key],
                "related_fact_refs": tuple(sorted({aliases[key] for key in d.related_keys})),
            }
        )
        for d in drafts
    )
    unsigned = dict(
        schema_version="nurse.v2.report-evidence-view/1",
        opaque_case_id=foundation.opaque_case_id,
        context_sha256=foundation.context_sha256,
        policy_sha256=policy.sha256,
        top_k=tuple(c.service_entity_key for c in foundation.top_k),
        facts=facts,
        mandatory_fact_refs=tuple(aliases[d.key] for d in mandatory),
        omissions=tuple(
            ReportEvidenceOmission(
                kind=kind, target_service_keys=targets, reason_code=reason, count=count
            )
            for (kind, targets, reason), count in sorted(omissions.items())
            if count
        ),
    )
    # Construct from typed records, then hash the canonical JSON representation.
    view = ReportEvidenceView.model_validate({**unsigned, "view_sha256": "0" * 64})
    digest = _digest(view.model_dump(mode="json", exclude={"view_sha256"}))
    view = view.model_copy(update={"view_sha256": digest})
    bindings = tuple(d.binding.model_copy(update={"fact_ref": aliases[d.key]}) for d in drafts)
    return view, bindings


def _project_verified(
    *,
    foundation: DeterministicSREReport,
    payload: _SupportedReportPayload,
    policy: ReportEvidencePolicy,
) -> ReportEvidenceProjection:
    mandatory, adverse_ids = _mandatory_drafts(foundation, payload)
    optional, omissions = _optional_drafts(foundation, payload, adverse_ids)
    pending_reason: OmissionReason = (
        "detail_count_budget" if policy.max_detail_facts == 0 else "detail_byte_budget"
    )
    for detail in optional:
        omissions[(detail.fact.kind, detail.fact.target_service_keys, pending_reason)] += 1
    mandatory_view, bindings = _view_and_bindings(
        foundation=foundation,
        policy=policy,
        mandatory=mandatory,
        selected=[],
        omissions=omissions,
    )
    view: ReportEvidenceView | None = mandatory_view
    mandatory_bytes = len(canonical_json_bytes(mandatory_view.model_dump(mode="json")))
    status: Literal["ready", "mandatory_budget_exhausted"] = "ready"
    selected: list[_Draft] = []
    if mandatory_bytes > policy.max_view_bytes:
        status = "mandatory_budget_exhausted"
        view = None
        bindings = ()
    elif policy.max_detail_facts:
        for index, detail in enumerate(optional):
            pending_key = (detail.fact.kind, detail.fact.target_service_keys, pending_reason)
            proposed_omissions = omissions.copy()
            proposed_omissions[pending_key] -= 1
            if len(selected) + 1 == policy.max_detail_facts:
                for remaining in optional[index + 1 :]:
                    proposed_omissions[
                        (remaining.fact.kind, remaining.fact.target_service_keys, pending_reason)
                    ] -= 1
                    proposed_omissions[
                        (
                            remaining.fact.kind,
                            remaining.fact.target_service_keys,
                            "detail_count_budget",
                        )
                    ] += 1
            trial, trial_bindings = _view_and_bindings(
                foundation=foundation,
                policy=policy,
                mandatory=mandatory,
                selected=[*selected, detail],
                omissions=proposed_omissions,
            )
            if len(canonical_json_bytes(trial.model_dump(mode="json"))) <= policy.max_view_bytes:
                selected.append(detail)
                view, bindings, omissions = trial, trial_bindings, proposed_omissions
                if len(selected) == policy.max_detail_facts:
                    break
    result = ReportEvidenceProjection(
        policy=policy,
        foundation=foundation,
        view=view,
        source_bindings=bindings,
        status=status,
        mandatory_view_bytes=mandatory_bytes,
        projection_sha256="0" * 64,
    )
    return result.model_copy(
        update={
            "projection_sha256": _digest(
                result.model_dump(mode="json", exclude={"projection_sha256"})
            )
        }
    )


def build_report_evidence_projection(
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    policy: ReportEvidencePolicy,
    expected_policy_sha256: str,
) -> ReportEvidenceProjection:
    """Replay independent source pins before computing any public evidence projection."""
    with _preserve_evidence_sources(
        context=context,
        audit_package=audit_package,
        identity_manifest=identity_manifest,
        policy=policy,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
    ):
        context, audit_package = _snapshot(context), _snapshot(audit_package)
        identity_manifest, policy = _snapshot(identity_manifest), _snapshot(policy)
        for actual, expected in (
            (context.context_sha256, expected_context_sha256),
            (audit_package.audit_sha256, expected_audit_sha256),
            (identity_manifest.manifest_sha256, expected_identity_manifest_sha256),
            (policy.sha256, expected_policy_sha256),
        ):
            if not hmac.compare_digest(actual, expected):
                raise ValueError("report evidence source digest does not match its external pin")
        source = require_report_source_read_view(
            context=context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
            expected_context_sha256=expected_context_sha256,
        )
        audit_inference_boundary(
            source.context
            if type(source.context) is FrozenReportContext
            else (source.payload.entity_registry, source.payload.candidate_universe),
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        )
        foundation = build_deterministic_sre_report(
            context=source.context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
            expected_context_sha256=expected_context_sha256,
        )
        projection = _project_verified(foundation=foundation, payload=source.payload, policy=policy)
        if projection.view is not None:
            audit_inference_boundary(
                projection.view,
                identity_manifest=identity_manifest,
                expected_identity_manifest_sha256=expected_identity_manifest_sha256,
            )
        return projection


def require_report_evidence_projection(
    projection: ReportEvidenceProjection,
    *,
    context: SupportedReportContext,
    audit_package: SupportedCaseAudit,
    expected_audit_sha256: str,
    expected_context_sha256: str,
    identity_manifest: InferenceIdentityManifest,
    expected_identity_manifest_sha256: str,
    policy: ReportEvidencePolicy,
    expected_policy_sha256: str,
) -> ReportEvidenceProjection:
    """Reject even self-rehashed projections that differ from independent source replay."""
    if type(projection) is not ReportEvidenceProjection:
        raise ValueError("requires exact ReportEvidenceProjection type")
    with _preserve_evidence_sources(
        context=context,
        audit_package=audit_package,
        identity_manifest=identity_manifest,
        policy=policy,
        expected_audit_sha256=expected_audit_sha256,
        expected_context_sha256=expected_context_sha256,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        extra_roots=((projection, ReportEvidenceProjection),),
    ):
        # Verify before a snapshot can normalize a Literal or numeric representation.
        if not projection.verify():
            raise ValueError("report evidence projection failed original integrity verification")
        provided_bytes = canonical_json_bytes(projection.model_dump(mode="json", warnings="error"))
        rebuilt = build_report_evidence_projection(
            context=context,
            audit_package=audit_package,
            expected_audit_sha256=expected_audit_sha256,
            expected_context_sha256=expected_context_sha256,
            identity_manifest=identity_manifest,
            expected_identity_manifest_sha256=expected_identity_manifest_sha256,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
        )
        rebuilt_bytes = canonical_json_bytes(rebuilt.model_dump(mode="json", warnings="error"))
        if not hmac.compare_digest(provided_bytes, rebuilt_bytes):
            raise ValueError("report evidence projection differs from rebuilt source projection")
        return rebuilt
