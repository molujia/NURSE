"""Explicit presentation/source identities, generated from admitted long records."""

from __future__ import annotations

import re
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_references import encode_references

DERIVED_REPRESENTATIONS = {'condition_evidence_brief', 'historical_evidence_summary'}
BINDING_FIELD = 'feedback_representation_bindings'


def binding_index(bindings: list[dict[str, Any]] | None) -> dict[str, dict[str, str]] | None:
    if bindings is None:
        return None  # Original packets keep their original validation path.
    if type(bindings) is not list:
        raise ValueError('feedback representation bindings shape differs')
    result = {}
    for row in bindings:
        if type(row) is not dict or set(row) != {
            'short_record_sha256', 'source_record_ref', 'delivered_record_ref'
        } or any(type(value) is not str for value in row.values()):
            raise ValueError('feedback representation binding fields differ')
        digest = row['short_record_sha256']
        if (re.fullmatch(r'[0-9a-f]{64}', digest) is None or digest in result
                or any(re.fullmatch(r'H[1-9][0-9]{0,3}', row[key]) is None
                       for key in ('source_record_ref', 'delivered_record_ref'))):
            raise ValueError('feedback representation binding identity differs')
        result[digest] = row
    return result


def representation_bindings(
    context: dict[str, Any], *, references: dict[str, Any],
    case_id: str, snapshot: str, index: int,
) -> list[dict[str, str]]:
    """No source reads/issuance: map already-validated bodies using the same codec."""
    from nurse_rcl.v2.live.ops_sre_feedback_facts import visible_records

    book = SreActionReferenceBook.from_state(
        references, snapshot=snapshot, run_id=references['run_id'])
    before = canonical_json_bytes(book.to_state())
    rows = {}
    for item in visible_records(context):
        record = item['record']
        if record.get('representation') not in DERIVED_REPRESENTATIONS:
            continue
        source = record.get('record_sha256')
        if type(source) is not str or re.fullmatch(r'[0-9a-f]{64}', source) is None:
            raise ValueError('feedback original source identity is not a digest')
        short = encode_references(record, book=book, visible_at=index,
                                  case_id=case_id, limits=CONTEXT_LIMITS)
        identity = sha256_hex(canonical_json_bytes(short))
        relation = dict(short_record_sha256=identity,
            source_record_ref=book.issue('digest', source, visible_at=index),
            delivered_record_ref=book.issue('digest', item['source_binding']['record_sha256'],
                                            visible_at=index))
        if identity in rows and rows[identity] != relation:
            raise ValueError('feedback representation identity is ambiguous')
        rows[identity] = relation
    if canonical_json_bytes(book.to_state()) != before:
        raise ValueError('feedback representation binding attempted new references')
    result = [rows[key] for key in sorted(rows)]
    binding_index(result)
    return result
