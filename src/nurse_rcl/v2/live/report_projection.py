"""Explicit live projection: complete private membership, compact bounded public wire."""
from __future__ import annotations

from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import report_evidence
from nurse_rcl.v2.deterministic_report import DeterministicSREReport
from nurse_rcl.v2.live.report_compact import (
    LiveFactGroup, LiveFactMember, decode_compact_view, encode_compact_view,
    group_verified_view, verify_group_partition,
)
from nurse_rcl.v2.report_evidence_schemas import (
    ReportEvidenceBinding, ReportEvidenceOmission, ReportEvidencePolicy,
    ReportEvidenceValue, ReportEvidenceView,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree


def _hash(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


class LiveGroupedReportProjection(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-grouped-report-projection/1'] = 'nurse.v2.live-grouped-report-projection/1'
    policy: ReportEvidencePolicy
    foundation: DeterministicSREReport
    view: ReportEvidenceView | None
    source_bindings: tuple[ReportEvidenceBinding, ...]
    groups: tuple[LiveFactGroup, ...]
    public_wire_json: str | None
    status: Literal['ready', 'mandatory_budget_exhausted']
    mandatory_view_bytes: int
    original_fact_count: int
    projection_sha256: str

    def verify(self) -> bool:
        try:
            if type(self) is not LiveGroupedReportProjection:
                return False
            preflight_exact_pydantic_tree(self, root_model_types=(LiveGroupedReportProjection,),
                label='live grouped report', max_nodes=500_000, max_string_length=None)
            raw = canonical_json_bytes(self.model_dump(mode='json'))
            if canonical_json_bytes(type(self).model_validate_json(raw).model_dump(mode='json')) != raw:
                return False
            if self.projection_sha256 != _hash(self.model_dump(mode='json', exclude={'projection_sha256'})):
                return False
            if not self.foundation.verify() or self.original_fact_count < 1 or self.mandatory_view_bytes < 1:
                return False
            if self.view is None:
                return (self.status == 'mandatory_budget_exhausted' and not self.source_bindings
                    and not self.groups and self.public_wire_json is None
                    and self.mandatory_view_bytes > self.policy.max_view_bytes)
            if self.status != 'ready' or self.public_wire_json is None or not self.view.verify():
                return False
            if not (self.mandatory_view_bytes <= len(self.public_wire_json.encode()) <= self.policy.max_view_bytes):
                return False
            if (self.view.policy_sha256 != self.policy.sha256
                    or self.view.context_sha256 != self.foundation.context_sha256
                    or self.view.opaque_case_id != self.foundation.opaque_case_id
                    or self.view.top_k != tuple(c.service_entity_key for c in self.foundation.top_k)):
                return False
            if decode_compact_view(self.public_wire_json.encode(), expected_view_sha256=self.view.view_sha256) != self.view:
                return False
            verify_group_partition(self.view, self.source_bindings, self.groups, self.original_fact_count)
            return True
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


def _project_verified(*, foundation: DeterministicSREReport, payload: Any,
                      policy: ReportEvidencePolicy) -> LiveGroupedReportProjection:
    """Only consume this inside the original source/score replay and mutation guard."""
    mandatory, adverse = report_evidence._mandatory_drafts(foundation, payload)
    optional, omissions = report_evidence._optional_drafts(foundation, payload, adverse)
    reason: Literal['detail_count_budget', 'detail_byte_budget'] = (
        'detail_byte_budget' if policy.max_detail_facts else 'detail_count_budget')
    for draft in optional:
        omissions[(draft.fact.kind, draft.fact.target_service_keys, reason)] += 1
    original, bindings = report_evidence._view_and_bindings(foundation=foundation, policy=policy,
        mandatory=mandatory, selected=[], omissions=omissions)
    protected = tuple(sorted({pid for m in foundation.conditional_mitigations
                              for pid in m.precondition_predicate_result_ids}))
    view, bindings, groups = group_verified_view(original, bindings, protected_source_ids=protected)
    original_count = len(original.facts)
    wire = encode_compact_view(view)
    mandatory_size = len(wire)
    selected = 0
    if mandatory_size > policy.max_view_bytes:
        result = LiveGroupedReportProjection(policy=policy, foundation=foundation, view=None,
            source_bindings=(), groups=(), public_wire_json=None, status='mandatory_budget_exhausted',
            mandatory_view_bytes=mandatory_size, original_fact_count=original_count, projection_sha256='0'*64)
    else:
        # Mandatory groups are immutable. Optional records are singletons and can
        # refer only to mandatory keys, as in the unchanged shared projector.
        old_alias = {draft.key: f'f{i}' for i,draft in enumerate(mandatory, 1)}
        new_alias = {member.original_fact_ref:g.fact_ref for g in groups for member in g.members}
        for index, draft in enumerate(optional):
            if selected == policy.max_detail_facts:
                break
            proposed = omissions.copy()
            proposed[(draft.fact.kind, draft.fact.target_service_keys, reason)] -= 1
            if selected + 1 == policy.max_detail_facts:
                for remaining in optional[index+1:]:
                    key = (remaining.fact.kind, remaining.fact.target_service_keys)
                    proposed[(*key, reason)] -= 1
                    proposed[(*key, 'detail_count_budget')] += 1
            ref = f'f{len(view.facts)+1}'
            old_ref = f'f{original_count+1}'
            related = tuple(sorted({old_alias[key] for key in draft.related_keys}))
            original_fact = draft.fact.model_copy(update={'fact_ref':old_ref, 'related_fact_refs':related})
            original_binding = draft.binding.model_copy(update={'fact_ref':old_ref})
            fact = original_fact.model_copy(update=dict(fact_ref=ref,
                related_fact_refs=tuple(sorted({new_alias[r] for r in related})),
                values=(*original_fact.values, ReportEvidenceValue(name='source_member_count', value=1))))
            member = LiveFactMember(original_fact_ref=old_ref, source_kind=original_binding.source_kind,
                source_id=original_binding.source_id, source_sha256=original_binding.source_sha256,
                binding_sha256=_hash(original_binding.model_dump(mode='json')))
            group = LiveFactGroup(fact_ref=ref, members=(member,), original_related_fact_refs=related,
                original_display_sha256=_hash(original_fact.model_dump(mode='json', exclude={'fact_ref'})))
            trial = view.model_copy(update=dict(facts=(*view.facts, fact),
                omissions=tuple(ReportEvidenceOmission(kind=kind, target_service_keys=targets,
                    reason_code=omitted, count=count) for (kind,targets,omitted),count in sorted(proposed.items()) if count)))
            trial = trial.model_copy(update={'view_sha256':_hash(trial.model_dump(mode='json', exclude={'view_sha256'}))})
            trial_wire = encode_compact_view(trial)
            if len(trial_wire) <= policy.max_view_bytes:
                view, wire, omissions = trial, trial_wire, proposed
                bindings = (*bindings, original_binding.model_copy(update={'fact_ref':ref}))
                groups = (*groups, group)
                original_count += 1
                selected += 1
        result = LiveGroupedReportProjection(policy=policy, foundation=foundation, view=view,
            source_bindings=bindings, groups=groups, public_wire_json=wire.decode(), status='ready',
            mandatory_view_bytes=mandatory_size, original_fact_count=original_count, projection_sha256='0'*64)
    result = result.model_copy(update={'projection_sha256':_hash(result.model_dump(mode='json', exclude={'projection_sha256'}))})
    if not result.verify():
        raise ValueError('live grouped projection failed integrity or membership')
    return result
