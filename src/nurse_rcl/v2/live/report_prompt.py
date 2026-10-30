"""Distinct compact-wire prompt; existing shared report prompts stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.inference_leakage import audit_inference_boundary
from nurse_rcl.v2.live.report_compact import decode_compact_view
from nurse_rcl.v2.live.report_projection import LiveGroupedReportProjection
from nurse_rcl.v2.opaque_ids import InferenceIdentityManifest
from nurse_rcl.v2.report_composition_proposals import report_composition_output_schema
from nurse_rcl.v2.report_prompt import _INSTRUCTIONS

VERSION = 'nurse-v2-live-compact-report-prompt/1'
INSTRUCTIONS = _INSTRUCTIONS + '''
The evidence_view is nurse.v2.live-report-wire/1, a reversible column encoding.
fact_columns gives the exact meaning of each position of each row in rows.
Row positions: 0=fact_ref (literal, cite this fN alias), 1=kind atom index,
2=target_service_keys atom indexes, 3=intervals as [start_unix_ns atom index,
end_unix_ns atom index] pairs, 4=values as [name atom index,value atom index] pairs,
5=status_codes atom indexes, 6=related_fact_refs (literal fN aliases),
7=allowed_claim_strength atom index, 8=scope_basis atom index.
All atom indexes are zero-based into atoms. Preserve the atom's JSON scalar type.
omission_columns describes omissions rows: kind atom index, target atom indexes,
literal count, reason_code atom index. Top-K and mandatory_fact_refs are literal.
source_member_count is the number of exact-display source records in a group,
NOT independent corroboration, independent measurements, confidence or score weight.
All grouped members have identical displayed state, scope and original logical context.
Never infer hidden member details. Source membership is retained privately for replay.
Cite visible group aliases, not numeric atom indexes or private source identifiers.
'''


@dataclass(frozen=True, slots=True)
class LiveCompactPrompt:
    payload_json: bytes

    @property
    def sha256(self) -> str:
        self._data()
        return sha256_hex(self.payload_json)

    def _data(self) -> dict[str, Any]:
        if type(self) is not LiveCompactPrompt:
            raise ValueError('exact compact prompt required')
        raw = object.__getattribute__(self, 'payload_json')
        data = decode_json_object(raw, ExchangeLimits(max_bytes=524288, max_nodes=100000))
        if (set(data) != {'prompt_version','instructions','output_schema_json','data'}
                or data['prompt_version'] != VERSION or data['instructions'] != INSTRUCTIONS
                or data['output_schema_json'] != canonical_json_bytes(report_composition_output_schema()).decode()
                or type(data['data']) is not dict
                or set(data['data']) != {'prompt_version','evidence_view'}
                or data['data']['prompt_version'] != VERSION or canonical_json_bytes(data) != raw):
            raise ValueError('compact prompt shape differs')
        view = data['data']['evidence_view']
        decode_compact_view(canonical_json_bytes(view), expected_view_sha256=view['view_sha256'])
        return data

    def messages(self) -> list[dict[str, str]]:
        data = self._data()
        return [dict(role='system', content=data['instructions']+'\nOutput schema:\n'+data['output_schema_json']),
                dict(role='user', content=canonical_json_bytes(data['data']).decode())]


def _prompt_for_verified_projection(projection: LiveGroupedReportProjection, *,
        identity_manifest: InferenceIdentityManifest, expected_identity_manifest_sha256: str) -> LiveCompactPrompt:
    if not projection.verify() or projection.view is None or projection.public_wire_json is None:
        raise ValueError('compact projection unavailable or invalid')
    # Inspect both the original public facts and the actual wire's exact decoded
    # meaning; canonical encoding refuses extra/unused atoms and unknown columns.
    audit_inference_boundary(projection.view, identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256)
    wire = projection.public_wire_json.encode()
    decoded = decode_compact_view(wire, expected_view_sha256=projection.view.view_sha256)
    if decoded != projection.view:
        raise ValueError('wire differs from source-backed grouped view')
    audit_inference_boundary(decoded, identity_manifest=identity_manifest,
        expected_identity_manifest_sha256=expected_identity_manifest_sha256)
    prompt = LiveCompactPrompt(encode_json(dict(prompt_version=VERSION, instructions=INSTRUCTIONS,
        output_schema_json=canonical_json_bytes(report_composition_output_schema()).decode(),
        data=dict(prompt_version=VERSION, evidence_view=decode_json_object(wire,
            ExchangeLimits(max_bytes=262144, max_nodes=100000)))),
        ExchangeLimits(max_bytes=524288, max_nodes=100000)))
    prompt._data()
    return prompt
