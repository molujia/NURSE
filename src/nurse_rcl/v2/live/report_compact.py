"""Invocation-local exact-display groups and reversible public wire encoding.

These pure helpers establish representation integrity, never source authority.
The enclosing live report must replay its complete, independently pinned audit.
"""
from __future__ import annotations

from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object
from nurse_rcl.v2.report_evidence_schemas import (
    ReportEvidenceBinding, ReportEvidenceValue, ReportEvidenceView,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

WIRE_VERSION = 'nurse.v2.live-report-wire/1'
FACT_COLUMNS = ('fact_ref', 'kind', 'target_service_keys', 'intervals', 'values',
                'status_codes', 'related_fact_refs', 'allowed_claim_strength', 'scope_basis')
OMISSION_COLUMNS = ('kind', 'target_service_keys', 'count', 'reason_code')
_META = ('opaque_case_id', 'context_sha256', 'policy_sha256', 'top_k',
         'mandatory_fact_refs', 'view_sha256')


def _hash(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


class LiveFactMember(StrictFrozenModel):
    original_fact_ref: str
    source_kind: str
    source_id: str
    source_sha256: str
    binding_sha256: str


class LiveFactGroup(StrictFrozenModel):
    schema_version: Literal['nurse.v2.live-report-fact-group/1'] = 'nurse.v2.live-report-fact-group/1'
    fact_ref: str
    original_display_sha256: str
    original_related_fact_refs: tuple[str, ...]
    members: tuple[LiveFactMember, ...]


def group_verified_view(view: ReportEvidenceView, bindings: tuple[ReportEvidenceBinding, ...],
                        *, protected_source_ids: tuple[str, ...] = ()) -> tuple[
                            ReportEvidenceView, tuple[ReportEvidenceBinding, ...], tuple[LiveFactGroup, ...]]:
    """Group only identical public predicates; retain every private member identity."""
    if type(view) is not ReportEvidenceView or type(bindings) is not tuple:
        raise ValueError('exact grouping input required')
    # This is the explicit live complete-input ceiling, not an expanded legacy view limit.
    preflight_exact_pydantic_tree((view, bindings),
        root_model_types=(ReportEvidenceView, ReportEvidenceBinding), label='live report grouping',
        max_nodes=6_000_000, max_string_length=None, max_depth=64)
    from nurse_rcl.v2.live.scoring_selection import _measure
    _measure((view, bindings))
    data = view.model_dump(mode='json')
    if _hash({k:v for k,v in data.items() if k != 'view_sha256'}) != view.view_sha256:
        raise ValueError('original view integrity differs')
    if tuple(b.fact_ref for b in bindings) != tuple(f.fact_ref for f in view.facts):
        raise ValueError('group bindings must exactly cover facts')
    buckets: dict[bytes, list[int]] = {}
    mandatory = set(view.mandatory_fact_refs)
    protected = set(protected_source_ids)
    for i, (fact, binding) in enumerate(zip(view.facts, bindings, strict=True)):
        if any(v.name == 'source_member_count' for v in fact.values):
            raise ValueError('group count name already present in source')
        display = fact.model_dump(mode='json', exclude={'fact_ref'})
        eligible = fact.kind == 'predicate' and binding.source_kind == 'predicate' and binding.source_id not in protected
        key = canonical_json_bytes([display, fact.fact_ref in mandatory,
                                     None if eligible else fact.fact_ref])
        buckets.setdefault(key, []).append(i)
    alias = {view.facts[i].fact_ref: f'f{number}'
             for number, indices in enumerate(buckets.values(), 1) for i in indices}
    facts, new_bindings, groups = [], [], []
    for number, indices in enumerate(buckets.values(), 1):
        original = view.facts[indices[0]]
        ref = f'f{number}'
        group_bindings = tuple(bindings[i] for i in indices)
        members = tuple(LiveFactMember(original_fact_ref=b.fact_ref, source_kind=b.source_kind,
            source_id=b.source_id, source_sha256=b.source_sha256,
            binding_sha256=_hash(b.model_dump(mode='json'))) for b in group_bindings)
        groups.append(LiveFactGroup(fact_ref=ref,
            original_display_sha256=_hash(original.model_dump(mode='json', exclude={'fact_ref'})),
            original_related_fact_refs=original.related_fact_refs, members=members))
        facts.append(original.model_copy(update=dict(fact_ref=ref,
            related_fact_refs=tuple(sorted({alias[r] for r in original.related_fact_refs})),
            values=(*original.values, ReportEvidenceValue(name='source_member_count', value=len(indices))))))
        if len(indices) == 1:
            new_bindings.append(group_bindings[0].model_copy(update={'fact_ref': ref}))
        else:
            digest = _hash([m.binding_sha256 for m in members])
            new_bindings.append(ReportEvidenceBinding(fact_ref=ref, source_kind='live_predicate_group',
                source_id='lpg_' + digest[:24], source_sha256=digest,
                source_fields=tuple(sorted({name for b in group_bindings for name in b.source_fields})),
                same_source_ids=tuple(sorted({identity for b in group_bindings
                    for identity in (b.source_id, *b.same_source_ids)}))))
    grouped = view.model_copy(update=dict(facts=tuple(facts),
        mandatory_fact_refs=tuple(dict.fromkeys(alias[r] for r in view.mandatory_fact_refs))))
    grouped = grouped.model_copy(update={'view_sha256': _hash(grouped.model_dump(mode='json', exclude={'view_sha256'}))})
    if not grouped.verify():
        raise ValueError('grouped view failed integrity')
    verify_group_partition(grouped, tuple(new_bindings), tuple(groups), len(view.facts))
    return grouped, tuple(new_bindings), tuple(groups)


def verify_group_partition(view: ReportEvidenceView, bindings: tuple[ReportEvidenceBinding, ...],
                           groups: tuple[LiveFactGroup, ...], original_count: int) -> None:
    """Verify reversible displayed state and closed membership, not source authenticity."""
    refs = tuple(f.fact_ref for f in view.facts)
    if tuple(g.fact_ref for g in groups) != refs or tuple(b.fact_ref for b in bindings) != refs:
        raise ValueError('group partition aliases differ')
    members = [m.original_fact_ref for g in groups for m in g.members]
    if len(members) != original_count or len(set(members)) != original_count or set(members) != {
        f'f{i}' for i in range(1, original_count + 1)}:
        raise ValueError('group partition lost or duplicated original facts')
    aliases = {m.original_fact_ref:g.fact_ref for g in groups for m in g.members}
    for fact, binding, group in zip(view.facts, bindings, groups, strict=True):
        counts = [v.value for v in fact.values if v.name == 'source_member_count']
        if counts != [len(group.members)] or not group.members or type(counts[0]) is not int:
            raise ValueError('group member count differs')
        if not set(group.original_related_fact_refs) <= aliases.keys() or fact.related_fact_refs != tuple(
                sorted({aliases[r] for r in group.original_related_fact_refs})):
            raise ValueError('group logical relationships differ')
        original = fact.model_dump(mode='json', exclude={'fact_ref'})
        original['values'] = [v for v in original['values'] if v['name'] != 'source_member_count']
        original['related_fact_refs'] = list(group.original_related_fact_refs)
        if _hash(original) != group.original_display_sha256:
            raise ValueError('group display does not preserve original state')
        if len(group.members) > 1:
            digest = _hash([m.binding_sha256 for m in group.members])
            if fact.kind != 'predicate' or any(m.source_kind != 'predicate' for m in group.members) or (
                    binding.source_kind != 'live_predicate_group' or binding.source_sha256 != digest
                    or binding.source_id != 'lpg_' + digest[:24]
                    or not {m.source_id for m in group.members} <= set(binding.same_source_ids)):
                raise ValueError('group private source membership differs')
        else:
            member = group.members[0]
            restored = binding.model_copy(update={'fact_ref': member.original_fact_ref})
            if (binding.source_id != member.source_id or binding.source_kind != member.source_kind
                    or binding.source_sha256 != member.source_sha256
                    or _hash(restored.model_dump(mode='json')) != member.binding_sha256):
                raise ValueError('singleton source binding differs')


def encode_compact_view(view: ReportEvidenceView) -> bytes:
    """Column encoding: indexes always select exact typed scalar atoms, never coercions."""
    if type(view) is not ReportEvidenceView or not view.verify():
        raise ValueError('compact view integrity failed')
    atoms: list[Any] = []
    index: dict[bytes, int] = {}
    def atom(value: Any) -> int:
        key = canonical_json_bytes(value)
        if key not in index:
            index[key] = len(atoms)
            atoms.append(value)
        return index[key]
    rows = []
    for fact in view.facts:
        rows.append([fact.fact_ref, atom(fact.kind), [atom(t) for t in fact.target_service_keys],
            [[atom(i.start_unix_ns), atom(i.end_unix_ns)] for i in fact.intervals],
            [[atom(v.name), atom(v.value)] for v in fact.values], [atom(c) for c in fact.status_codes],
            list(fact.related_fact_refs), atom(fact.allowed_claim_strength), atom(fact.scope_basis)])
    omissions = [[atom(o.kind), [atom(t) for t in o.target_service_keys], o.count, atom(o.reason_code)]
                 for o in view.omissions]
    data = view.model_dump(mode='json', include=set(_META))
    data.update(schema_version=WIRE_VERSION, fact_columns=list(FACT_COLUMNS),
        omission_columns=list(OMISSION_COLUMNS), atoms=atoms, rows=rows, omissions=omissions)
    return canonical_json_bytes(data)


def decode_compact_view(raw: bytes, *, expected_view_sha256: str) -> ReportEvidenceView:
    """Strict reversible wire with no unused dictionary data or hidden side channel."""
    try:
        data = decode_json_object(raw, ExchangeLimits(max_bytes=262144, max_nodes=100_000))
        if set(data) != {*_META, 'schema_version', 'fact_columns', 'omission_columns', 'atoms', 'rows', 'omissions'}:
            raise ValueError('compact fields differ')
        if (data['schema_version'] != WIRE_VERSION or data['fact_columns'] != list(FACT_COLUMNS)
                or data['omission_columns'] != list(OMISSION_COLUMNS)
                or data['view_sha256'] != expected_view_sha256):
            raise ValueError('compact identity or columns differ')
        atoms = data['atoms']
        if type(atoms) is not list or any(type(v) not in (str, int, float, bool, type(None)) for v in atoms):
            raise ValueError('compact atoms must be exact scalars')
        def at(i: Any) -> Any:
            if type(i) is not int or not 0 <= i < len(atoms):
                raise ValueError('compact index invalid')
            return atoms[i]
        facts = []
        for row in data['rows']:
            if type(row) is not list or len(row) != len(FACT_COLUMNS):
                raise ValueError('compact row differs')
            facts.append(dict(fact_ref=row[0], kind=at(row[1]),
                target_service_keys=[at(i) for i in row[2]],
                intervals=[dict(start_unix_ns=at(a), end_unix_ns=at(b)) for a,b in row[3]],
                values=[dict(name=at(a), value=at(b)) for a,b in row[4]],
                status_codes=[at(i) for i in row[5]], related_fact_refs=row[6],
                allowed_claim_strength=at(row[7]), scope_basis=at(row[8])))
        omissions = [dict(kind=at(a), target_service_keys=[at(i) for i in b], count=c, reason_code=at(d))
                     for a,b,c,d in data['omissions']]
        values = {k:data[k] for k in _META}
        values.update(schema_version='nurse.v2.report-evidence-view/1', facts=facts, omissions=omissions)
        view = ReportEvidenceView.model_validate_json(canonical_json_bytes(values))
        if not view.verify() or encode_compact_view(view) != raw:
            raise ValueError('compact wire is not canonical reversible evidence')
        return view
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('invalid compact report wire') from error
