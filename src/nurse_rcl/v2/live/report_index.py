"""Exact readable navigation copies alongside the complete compact evidence wire."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.live import report_guidance as guided
from nurse_rcl.v2.live.report_compact import decode_compact_view

INDEXED_PROMPT_VERSION = 'nurse-v2-live-compact-report-prompt/3'
INDEXED_INSTRUCTIONS = guided.GUIDED_INSTRUCTIONS + '''
The readable_index is an exact navigation copy of all ranking, candidate_evidence,
competition, uncertainty and limit facts, in their original order. It retains every
public field; values is a named scalar mapping instead of a name/value pair list.
Use it to read the exact alias, kind, targets, scope and assertion field/value owner.
It is NOT additional evidence, independent corroboration, or extra scoring weight.
The complete evidence_view remains authoritative for every fact, including facts
not duplicated in the index. Decode those rows when needed; never infer their kind
or values from nearby aliases. Do not confuse candidate summaries with hypotheses.

Use short, unique claim_id values such as c1, c2, c3 (1 to 32 characters, matching
^[a-z][a-z0-9_]{0,31}$). Never concatenate service identifiers into a claim_id.
Verification is a recommendation, not an inference or a performed action: use
statement_type=recommendation and requested_claim_strength=conditional_recommendation,
and retain all
required safeguards. Verification outcomes describe conditional true/false/unknown
branches, not measurements already obtained. Mitigation is also a conditional
recommendation and must satisfy every original prerequisite. Omit unsupported claims.
'''
_KINDS = frozenset({'ranking', 'candidate_evidence', 'competition', 'uncertainty', 'limit'})
_LIMITS = ExchangeLimits(max_bytes=524288, max_nodes=100000)


def _readable_index(wire: dict[str, Any]) -> list[dict[str, Any]]:
    view = decode_compact_view(canonical_json_bytes(wire), expected_view_sha256=wire['view_sha256'])
    return [dict(fact.model_dump(mode='json'), values={v.name: v.value for v in fact.values})
            for fact in view.facts if fact.kind in _KINDS]


@dataclass(frozen=True, slots=True)
class LiveIndexedPrompt:
    payload_json: bytes

    def _data(self) -> dict[str, Any]:
        if type(self) is not LiveIndexedPrompt:
            raise ValueError('exact indexed prompt required')
        raw = object.__getattribute__(self, 'payload_json')
        data = decode_json_object(raw, _LIMITS)
        if (set(data) != {'prompt_version', 'instructions', 'output_schema_json', 'data'}
                or data['prompt_version'] != INDEXED_PROMPT_VERSION
                or data['instructions'] != INDEXED_INSTRUCTIONS
                or type(data['data']) is not dict
                or set(data['data']) != {'prompt_version', 'evidence_view', 'readable_index'}
                or data['data']['prompt_version'] != INDEXED_PROMPT_VERSION
                or canonical_json_bytes(data) != raw):
            raise ValueError('indexed prompt shape differs')
        public = data['data']
        base = dict(data, prompt_version=guided.GUIDED_PROMPT_VERSION,
                    instructions=guided.GUIDED_INSTRUCTIONS,
                    data=dict(prompt_version=guided.GUIDED_PROMPT_VERSION,
                              evidence_view=public['evidence_view']))
        # Preserve the complete old wire/schema checks, including unused atoms.
        guided.LiveGuidedPrompt(encode_json(base, _LIMITS))._data()
        expected = _readable_index(public['evidence_view'])
        # Python equality would silently equate bool/int and int/float values.
        if canonical_json_bytes(public['readable_index']) != canonical_json_bytes(expected):
            raise ValueError('readable index differs from complete wire')
        if len(canonical_json_bytes(public)) > 80000:
            raise ExchangeBoundaryError('json_limit')
        return data

    @property
    def sha256(self) -> str:
        self._data()
        return sha256_hex(self.payload_json)

    def messages(self) -> list[dict[str, str]]:
        data = self._data()
        return [dict(role='system', content=data['instructions']+'\nOutput schema:\n'+data['output_schema_json']),
                dict(role='user', content=canonical_json_bytes(data['data']).decode())]


def index_prompt(prompt: guided.LiveGuidedPrompt) -> LiveIndexedPrompt:
    data = prompt._data()
    public = dict(data['data'], prompt_version=INDEXED_PROMPT_VERSION,
                  readable_index=_readable_index(data['data']['evidence_view']))
    indexed = dict(data, prompt_version=INDEXED_PROMPT_VERSION, instructions=INDEXED_INSTRUCTIONS, data=public)
    result = LiveIndexedPrompt(encode_json(indexed, _LIMITS))
    result._data()
    return result
