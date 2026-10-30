"""Complete /2 audit references and independently derived replay dependencies.

``score_affecting`` means a value or branch premise actually consumed while
replaying all candidate scores and Top-K. It does not mean positive contribution,
nonzero increment, or causation. Structural/hash retention alone is not a score
dependency. In particular selection and rejected guards remain audit sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_audit as legacy
from nurse_rcl.v2.case_audit import CaseAuditLineageNode

if TYPE_CHECKING:
    from nurse_rcl.v2.case_audit_v2 import CaseRankingReplayBundleV2


CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION: Literal["nurse.v2.case-audit-lineage-index/2"] = (
    "nurse.v2.case-audit-lineage-index/2"
)


class CaseAuditLineageIndexV2(StrictFrozenModel):
    """Independent full-source index, qualified against the exact /2 bundle."""

    schema_version: Literal["nurse.v2.case-audit-lineage-index/2"] = (
        CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    replay_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[CaseAuditLineageNode, ...] = Field(min_length=1)
    scoring_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        from nurse_rcl.v2.case_audit_v2 import _validate_audit_json

        return _validate_audit_json(cls, json_data, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        from nurse_rcl.v2.case_audit_v2 import _audit_snapshot

        _audit_snapshot(value)
        return value

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        from nurse_rcl.v2.case_audit_v2 import _exact_field

        return _exact_field(cls, value, info)

    @model_validator(mode="after")
    def canonical_nodes(self) -> Self:
        keys = tuple(node.record_id for node in self.nodes)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("lineage nodes must be sorted and unique")
        return self

    def verify(self) -> bool:
        from nurse_rcl.v2.case_audit_v2 import (
            _ERRORS,
            _audit_model_bytes,
            _audit_snapshot,
            _audit_unchanged,
        )

        try:
            if type(self) is not CaseAuditLineageIndexV2:
                return False
            original = _audit_snapshot(self)
            try:
                raw = canonical_json_bytes(original[0])
                rebuilt = CaseAuditLineageIndexV2.model_validate_json(raw)
                if _audit_model_bytes(rebuilt) != raw:
                    return False
                unsigned = cast(dict[str, Any], original[0]).copy()
                stored = unsigned.pop("lineage_sha256")
                if sha256_hex(canonical_json_bytes(unsigned)) != stored:
                    return False
                nodes = {node.record_id: node for node in rebuilt.nodes}
                return all(
                    all(
                        source in nodes and node.record_id in nodes[source].consumer_record_ids
                        for source in node.source_record_ids
                    )
                    and all(
                        consumer in nodes and node.record_id in nodes[consumer].source_record_ids
                        for consumer in node.consumer_record_ids
                    )
                    for node in rebuilt.nodes
                )
            finally:
                _audit_unchanged(self, original)
        except _ERRORS:
            return False

    def verify_against(self, replay_bundle: CaseRankingReplayBundleV2) -> bool:
        from nurse_rcl.v2.case_audit_v2 import (
            _ERRORS,
            CaseRankingReplayBundleV2,
            _audit_model_bytes,
            _audit_snapshot,
            _audit_unchanged,
        )

        try:
            if (
                type(self) is not CaseAuditLineageIndexV2
                or type(replay_bundle) is not CaseRankingReplayBundleV2
            ):
                return False
            original = _audit_snapshot((self, replay_bundle))
            try:
                if not self.verify():
                    return False
                expected = build_case_audit_lineage_index_v2(replay_bundle)
                return _audit_model_bytes(self) == _audit_model_bytes(expected)
            finally:
                _audit_unchanged((self, replay_bundle), original)
        except _ERRORS:
            return False


@dataclass(frozen=True, slots=True)
class _Record:
    kind: str
    value: object
    disposition: str
    digest: str


def _build_verified_lineage(bundle: CaseRankingReplayBundleV2) -> CaseAuditLineageIndexV2:
    """Pure compiler used only after the caller's complete /2 qualification gate.

    The legacy helper is a value-record collector: no /1 construction, schema
    rewrite, verifier, or eligibility proof is used. Its reference edges are
    supplemented below; its old all-true flags are never consumed.
    """
    from nurse_rcl.v2.case_audit_v2 import _audit_model_bytes

    evidence = bundle.evidence
    pending = legacy._compile_case_audit_lineage_records(cast(Any, bundle))
    records: dict[str, _Record] = {}
    references: dict[str, set[str]] = {}
    dependencies: dict[str, set[str]] = {}
    by_digest: dict[str, str] = {}
    by_object: dict[int, str] = {}

    def register(key: str, kind: str, value: object, disposition: str) -> str:
        digest = sha256_hex(_audit_model_bytes(value))
        row = _Record(kind, value, disposition, digest)
        previous = records.setdefault(key, row)
        if (
            previous.digest != digest
            or previous.kind != kind
            or previous.disposition != disposition
        ):
            raise ValueError("conflicting complete lineage record identity")
        by_digest.setdefault(digest, key)
        by_object[id(value)] = key
        references.setdefault(key, set())
        dependencies.setdefault(key, set())
        return key

    for key, row in pending.items():
        register(key, row.record_kind, row.record, row.disposition)
        references[key].update(row.source_record_ids)

    # These fields are the actual independent result IDs on both scoring paths.
    role_id = register(
        evidence.role_compilation.role_compilation_id,
        "role_compilation_result",
        evidence.role_compilation,
        evidence.scoring_path,
    )
    event_id = register(
        evidence.event_strength.event_strength_result_id,
        "event_strength_result",
        evidence.event_strength,
        "computed",
    )
    family_id = register(
        bundle.family_aggregation.family_aggregation_result_id,
        "family_aggregation_result",
        bundle.family_aggregation,
        "computed",
    )
    propagation_id = register(
        bundle.propagation_scoring.scoring_result_id,
        "propagation_scoring_result",
        bundle.propagation_scoring,
        evidence.scoring_path,
    )
    ranking_id = register(
        bundle.ranking_result.reference_ranking_result_id,
        "reference_ranking_result",
        bundle.ranking_result,
        "computed",
    )
    selection = evidence.scoring_selection
    selection_id = register(
        f"scoring_selection:{selection.selection_sha256}",
        "scoring_selection",
        selection,
        evidence.scoring_path,
    )
    selection_sources = set(selection.eligible_guard_result_ids)
    session_ids = {
        session.session_sha256: session.query_session_id for session in evidence.query_sessions
    }
    selection_sources.update(
        session_ids[binding.session_sha256] for binding in selection.guard_query_session_bindings
    )
    references[selection_id].update(selection_sources)

    visited: set[int] = set()

    def children(value: object) -> list[BaseModel]:
        kind = type(value)
        if kind is tuple or kind is list:
            return [model for item in cast(tuple[object, ...], value) for model in children(item)]
        if kind is dict:
            return [
                model
                for item in cast(dict[str, object], value).values()
                for model in children(item)
            ]
        # All objects have already passed the closed exact-type tree preflight.
        if isinstance(value, BaseModel):
            return [value]
        return []

    def visit(value: BaseModel) -> str:
        key = by_object.get(id(value))
        if key is None:
            digest = sha256_hex(_audit_model_bytes(value))
            key = by_digest.get(digest)
            if key is None:
                schema = object.__getattribute__(value, "__dict__").get("schema_version")
                kind = (
                    schema.split("/")[0].removeprefix("nurse.v2.").replace("-", "_")
                    if type(schema) is str
                    else "retained_value"
                )
                key = register(f"{kind}:{digest}", kind, value, "retained")
            by_object[id(value)] = key
        if id(value) in visited:
            return key
        visited.add(id(value))
        for child in children(object.__getattribute__(value, "__dict__")):
            child_key = visit(child)
            if child_key != key and key != selection_id:
                references[key].add(child_key)
        return key

    visit(bundle)
    frozen_id = visit(evidence.frozen_candidates)
    time_id = visit(evidence.time_contract)
    universe_id = visit(evidence.candidate_universe)
    registry_id = visit(evidence.entity_registry)
    grouping_id = visit(evidence.event_grouping)
    library_id = visit(evidence.mechanism_library)
    config_id = visit(bundle.scoring_config)
    top_id = visit(bundle.top_k)
    rule_result_id = visit(bundle.rule_scoring)
    history_result_id = visit(bundle.history_evaluation)
    authority_digest = sha256_hex(_audit_model_bytes(bundle.ranking_authority))
    authority_id = register(
        f"ranking_input_authority:{authority_digest}",
        "ranking_input_authority",
        bundle.ranking_authority,
        "pinned",
    )

    # Audit reference closure includes source aggregates and all original guards.
    references[role_id].update(
        (selection_id, universe_id, frozen_id, registry_id, grouping_id, library_id)
    )
    references[event_id].update((role_id, grouping_id))
    references[family_id].update((event_id, frozen_id))
    references[propagation_id].update((role_id, frozen_id, grouping_id))
    references[propagation_id].update(
        visit(row)
        for row in (
            *evidence.graph_snapshots,
            *evidence.condensations,
            *evidence.onsets,
            *evidence.precedences,
        )
    )
    references[ranking_id].update(
        (family_id, propagation_id, rule_result_id, history_result_id, config_id, frozen_id)
    )
    references[ranking_id].add(authority_id)
    references[visit(bundle)].add(authority_id)
    references[authority_id].update(
        (family_id, propagation_id, rule_result_id, history_result_id, frozen_id)
    )
    references[frozen_id].add(universe_id)
    references[grouping_id].add(registry_id)

    # Start a separate relation, using only the existing value/branch dependencies
    # (never the structural/hash references added by visit).
    dependency_kinds = {
        "observation",
        "event_group",
        "candidate_scope",
        "predicate_result",
        "diagnostic_hypothesis",
        "guard_decision",
        "score_impact_compilation",
        "hypothesis_guard_result",
        "role_binding_proposal",
        "role_assertion",
        "graph_snapshot",
        "scc_condensation",
        "onset_interval",
        "temporal_precedence",
        "event_group_base_strength",
        "candidate_event_role_adjustment",
        "signal_family_maximum",
        "candidate_family_aggregation",
        "propagation_relation_binding",
        "propagation_hop_audit",
        "propagation_path_audit",
        "propagation_target_score",
        "candidate_propagation_score",
        "scoring_rule",
        "rule_candidate_assignment",
        "rule_contribution",
        "candidate_rule_score",
        "history_neighbor_audit",
        "history_score_contribution",
        "candidate_history_score",
        "score_decomposition",
        "evidence_tie_group",
        "top_k_entry",
        "verification_request",
        "derived_verification_observation",
        "verification_result",
    }
    for key, row in pending.items():
        if row.record_kind in dependency_kinds:
            dependencies[key].update(
                source
                for source in row.source_record_ids
                if records[source].kind != "semantic_call"
            )
    for guard in evidence.role_guard_results:
        if guard.final_disposition != "accepted":
            dependencies[guard.guard_result_id].clear()
        hypothesis = guard.parsed_hypothesis
        if hypothesis is not None:
            alternatives = {key for key in hypothesis.alternative_hypothesis_ids if key in records}
            references[hypothesis.hypothesis_id].update(alternatives)
            # Alternative IDs retain audit references, not rejected content
            # dependencies. Accepted alternatives are reached through role inputs.
    for binding in selection.guard_query_session_bindings:
        session_id = session_ids[binding.session_sha256]
        dependencies[binding.guard_result_id].add(session_id)
        references[binding.guard_result_id].add(session_id)
    # Accepted guards can be neutralized or have zero strength and still decide
    # role branches. The rejected pool is retained solely for audit completeness.
    dependencies[role_id].update(selection.eligible_guard_result_ids)
    dependencies[role_id].update(row.proposal_id for row in evidence.role_proposals)
    if selection.eligible_guard_result_ids:
        dependencies[role_id].update((universe_id, frozen_id, registry_id, library_id))
    for assertion in evidence.role_compilation.assertions:
        dependencies[assertion.assertion_id].add(role_id)
        # Neutralization carries the conflicting assertion/guard provenance;
        # their complete original accepted inputs decide the neutral branch.
        for source in assertion.provenance_ids:
            if source in records and records[source].kind in {
                "role_assertion",
                "role_binding_proposal",
                "hypothesis_guard_result",
            }:
                dependencies[assertion.assertion_id].add(source)
    dependencies[event_id].add(role_id)
    dependencies[family_id].add(event_id)
    dependencies[propagation_id].add(role_id)
    # A graph consulted to establish that no path exists is still a branch
    # dependency. The scorer enumerates paths only for a positive source/target
    # role pair of that relation; retained unrelated graphs remain audit-only.
    assertions_by_id = {
        assertion.assertion_id: assertion for assertion in evidence.role_compilation.assertions
    }
    relation_roles: dict[str, set[str]] = {}
    for relation_binding in bundle.propagation_scoring.relation_bindings:
        dependencies[propagation_id].add(relation_binding.binding_id)
        assertion = assertions_by_id[relation_binding.assertion_id]
        if (
            assertion.lifecycle_status == "accepted"
            and assertion.strength > 0.0
            and (assertion.role != "source_support" or assertion.score_direction == "promote")
        ):
            relation_roles.setdefault(relation_binding.relation_type, set()).add(
                relation_binding.role
            )
    for condensation in evidence.condensations:
        if relation_roles.get(condensation.relation_type) == {"source_support", "symptom_support"}:
            dependencies[propagation_id].update(
                (condensation.condensation_id, visit(bundle.propagation_scoring.config))
            )
    dependencies[ranking_id].update(
        (family_id, propagation_id, rule_result_id, history_result_id, config_id, frozen_id)
    )
    for score in bundle.ranking_result.scores:
        dependencies[score.score_sha256].add(ranking_id)
    for adjustment in evidence.event_strength.candidate_event_adjustments:
        dependencies[adjustment.adjustment_id].add(role_id)
    for strength in evidence.event_strength.group_strengths:
        # Base-strength replay consumes each retained observation, including
        # state/quality used to decide an unknown or excluded measurement branch.
        group = next(
            item
            for item in evidence.event_grouping.event_groups
            if item.event_group_id == strength.event_group_id
        )
        dependencies[strength.strength_id].update(group.observation_ids)
    for key, record in records.items():
        if record.kind in {"observation", "event_group", "candidate_scope"}:
            dependencies[key].update((time_id, registry_id))
    dependencies[frozen_id].add(universe_id)
    dependencies[universe_id].update(
        legacy._candidate_record_id(row.service_entity_key)
        for row in evidence.candidate_universe.candidates
    )
    dependencies[top_id].update((ranking_id, config_id))
    dependencies[top_id].update(
        legacy._top_k_entry_record_id(rank=entry.rank, service_key=entry.service_entity_key)
        for entry in bundle.top_k.entries
    )

    # Complete references also carry every actual numerical/branch relation.
    for key, sources in dependencies.items():
        sources.discard(key)
        references[key].update(sources)
    consumers: dict[str, set[str]] = {key: set() for key in records}
    for key, sources in references.items():
        sources.discard(key)
        for source in sources:
            if source not in records:
                raise ValueError(f"complete lineage references an absent record: {source}")
            consumers[source].add(key)
    affecting: set[str] = set()
    stack = [*(row.score_sha256 for row in bundle.ranking_result.scores), top_id]
    while stack:
        key = stack.pop()
        if key not in affecting:
            affecting.add(key)
            stack.extend(dependencies[key])
    nodes = tuple(
        CaseAuditLineageNode(
            record_id=key,
            record_kind=row.kind,
            record_sha256=row.digest,
            disposition=row.disposition,
            score_affecting=key in affecting,
            source_record_ids=tuple(sorted(references[key])),
            consumer_record_ids=tuple(sorted(consumers[key])),
        )
        for key, row in sorted(records.items())
    )
    unsigned = {
        "schema_version": CASE_AUDIT_LINEAGE_INDEX_SCHEMA_VERSION,
        "opaque_case_id": evidence.time_contract.opaque_case_id,
        "replay_bundle_sha256": bundle.bundle_sha256,
        "scoring_selection_sha256": selection.selection_sha256,
        "nodes": nodes,
    }
    from nurse_rcl.v2.case_audit_v2 import _audit_snapshot

    digest = sha256_hex(canonical_json_bytes(_audit_snapshot(unsigned)[0]))
    return CaseAuditLineageIndexV2.model_validate({**unsigned, "lineage_sha256": digest})


def build_case_audit_lineage_index_v2(
    replay_bundle: CaseRankingReplayBundleV2,
) -> CaseAuditLineageIndexV2:
    from nurse_rcl.v2.case_audit_v2 import (
        CaseRankingReplayBundleV2,
        _audit_snapshot,
        _audit_unchanged,
        _fresh,
    )

    if type(replay_bundle) is not CaseRankingReplayBundleV2:
        raise ValueError("complete lineage requires an exact /2 replay bundle")
    original = _audit_snapshot(replay_bundle)
    try:
        bundle = _fresh(replay_bundle, CaseRankingReplayBundleV2, canonical_json_bytes(original[0]))
        if not bundle.verify_inputs():
            raise ValueError("complete lineage source failed full replay")
        return _build_verified_lineage(bundle)
    finally:
        _audit_unchanged(replay_bundle, original)
