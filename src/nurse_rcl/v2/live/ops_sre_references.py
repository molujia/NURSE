"""Bound, append-only model handles; originals stay in the local owner state.

This adapter grants no tool/evidence authority. Owners must admit the original
context and re-check resolved parameters against the original tool registry.
No free text, arbitrary JSON string or unknown field is rewritten.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import ExchangeLimits, encode_json

REFERENCE_SCHEMA = "nurse.v2.sre-reference-book/1"
PREFIXES = dict(service="S", card="C", request="Q", onset="O", evidence="E", source="P", digest="H")
MAX_REFERENCES = 4096
STATE_LIMITS = ExchangeLimits(4_194_304, 32, 100_000)
VIEW_LIMITS = ExchangeLimits(4_194_304)

# Explicit identity positions, not suffix guesses or regex replacement of text.
FIELD_KINDS = {
    **dict.fromkeys(
        (
            "candidate_service_key",
            "candidate_service_keys",
            "target_service_keys",
            "left_candidate_service_key",
            "right_candidate_service_key",
            "caller_service_key",
            "callee_service_key",
            "source_service_key",
            "target_service_key",
            "source_candidate",
            "target_candidate",
            "targets",
            "entities",
            "scope",
            "formal_top_k",
        ),
        "service",
    ),
    **dict.fromkeys(
        ("card_id", "card_ids", "visible_card_ids", "receipt_card_id", "critical_record_ids"),
        "card",
    ),
    **dict.fromkeys(("request_key", "request_keys"), "request"),
    **dict.fromkeys(("onset_id", "source_onset_id", "target_onset_id"), "onset"),
    **dict.fromkeys(
        (
            "evidence_id",
            "evidence_ids",
            "original_evidence_ids",
            "support_evidence_ids",
            "reused_core_evidence_ids",
            "refuting_evidence_ids",
        ),
        "evidence",
    ),
    **dict.fromkeys(
        (
            "source_ref",
            "source_refs",
            "support_source_key",
            "provenance_ids",
            "canonical_series_key",
        ),
        "source",
    ),
    **dict.fromkeys(
        (
            "measurement_sha256",
            "dynamic_predicate_sha256",
            "predicate_sha256",
            "mechanism_definition_sha256",
            "policy_sha256",
            "semantics_sha256",
            "source_sha256",
            "result_sha256",
            "input_sha256",
            "complete_text_sha256",
            "record_sha256",
            "latest_record_refs",
            "delivered_record_refs",
            "request_sha256",
            "catalog_sha256",
            "resolution_sha256",
            "time_contract_sha256",
            "scope_content_sha256",
        ),
        "digest",
    ),
}
ARRAY_FIELDS = frozenset(k for k in FIELD_KINDS if k.endswith(("_ids", "_keys", "_refs"))) | {
    "targets",
    "entities",
    "scope",
    "formal_top_k",
}


def _index(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 9:
        raise ValueError("reference visibility index differs")


class SreReferenceBook:
    """One owner's local table. Publication visibility is checked independently."""

    schema_version = REFERENCE_SCHEMA
    prefixes: ClassVar[dict[str, str]] = PREFIXES

    def __init__(self, *, snapshot: str, run_id: str):
        require_pin(snapshot)
        require_pin(run_id)
        self.snapshot, self.run_id = snapshot, run_id
        self._rows: list[list[Any]] = []
        self._originals: dict[tuple[str, str], str] = {}
        self._handles: dict[str, tuple[str, str, int]] = {}
        self._counts = dict.fromkeys(self.prefixes, 0)

    def issue(self, kind: str, original: str, *, visible_at: int) -> str:
        _index(visible_at)
        if kind not in self.prefixes or type(original) is not str or not 1 <= len(original) <= 2048:
            raise ValueError("reference kind or original differs")
        previous = self._originals.get((kind, original))
        if previous is not None:
            self.resolve(previous, kind, visible_at=visible_at)
            return previous
        if self._rows and visible_at < self._rows[-1][3]:
            raise ValueError("reference issuance order differs")
        if len(self._rows) >= MAX_REFERENCES:
            raise ValueError("reference table resource limit")
        self._counts[kind] += 1
        handle = self.prefixes[kind] + str(self._counts[kind])
        self._rows.append([kind, handle, original, visible_at])
        self._originals[(kind, original)] = handle
        self._handles[handle] = (kind, original, visible_at)
        return handle

    def resolve(self, handle: str, kind: str, *, visible_at: int) -> str:
        _index(visible_at)
        if (
            kind not in self.prefixes
            or type(handle) is not str
            or not re.fullmatch(
                re.escape(self.prefixes[kind]) + r"[1-9][0-9]{0,3}",
                handle,
            )
        ):
            raise ValueError("reference type differs")
        if handle not in self._handles:
            raise ValueError("reference is not issued")
        registered_kind, original, first = self._handles[handle]
        if registered_kind != kind:
            raise ValueError("reference type differs")
        if first > visible_at:
            raise ValueError("reference is not visible at this request")
        return original

    def to_state(self) -> dict[str, Any]:
        value = dict(
            schema_version=self.schema_version,
            snapshot=self.snapshot,
            run_id=self.run_id,
            entries=[list(row) for row in self._rows],
        )
        value["state_sha256"] = sha256_hex(encode_json(value, STATE_LIMITS))
        encode_json(value, STATE_LIMITS)
        return value

    @classmethod
    def from_state(cls, state: dict[str, Any], *, snapshot: str, run_id: str) -> Self:
        encode_json(state, STATE_LIMITS)
        if (
            type(state) is not dict
            or set(state)
            != {
                "schema_version",
                "snapshot",
                "run_id",
                "entries",
                "state_sha256",
            }
            or (state["schema_version"], state["snapshot"], state["run_id"])
            != (
                cls.schema_version,
                snapshot,
                run_id,
            )
        ):
            raise ValueError("reference state identity differs")
        if state["state_sha256"] != sha256_hex(
            canonical_json_bytes({k: v for k, v in state.items() if k != "state_sha256"})
        ):
            raise ValueError("reference state digest differs")
        rows = state["entries"]
        if type(rows) is not list or len(rows) > MAX_REFERENCES:
            raise ValueError("reference table shape differs")
        result = cls(snapshot=snapshot, run_id=run_id)
        for row in rows:
            if type(row) is not list or len(row) != 4:
                raise ValueError("reference row shape differs")
            kind, handle, original, first = row
            if (
                type(kind) is not str
                or type(original) is not str
                or ((kind, original) in result._originals)
                or result.issue(kind, original, visible_at=first) != handle
            ):
                raise ValueError("reference collision or issuance order differs")
        return result


def _translate(
    value: dict[str, Any],
    *,
    book: SreReferenceBook,
    visible_at: int,
    case_id: str,
    encode: bool,
    limits: ExchangeLimits,
    native_facts: bool = False,
    brief_fields: bool = False,
) -> dict[str, Any]:
    from nurse_rcl.v2.live.ops_provider import _require_outbound

    _index(visible_at)
    # Match the accepted JSON wire domain (including tuple-backed model fields)
    # before traversing registered positions; never leave unvisited containers.
    value = json.loads(encode_json(value, limits))
    _require_outbound(value, case=case_id, snapshot=book.snapshot)
    # Transactional extension: failed encoding never leaves half-issued handles.
    working = type(book).from_state(book.to_state(), snapshot=book.snapshot, run_id=book.run_id)
    convert = working.issue if encode else working.resolve
    kinds = dict(FIELD_KINDS)
    if brief_fields:
        kinds["source_record_sha256"] = "digest"

    def field(name: str, item: Any) -> Any:
        kind = kinds[name]
        if item is None:
            return None
        if type(item) is list:
            if name not in ARRAY_FIELDS or any(type(child) is not str for child in item):
                raise ValueError("reference array type differs")
            return [
                convert(kind, child, visible_at=visible_at)
                if encode
                else convert(child, kind, visible_at=visible_at)
                for child in item
            ]
        if (
            type(item) is dict
            and name in {"evidence_ids", "source_refs"}
            and (
                item.get("section") == name
                and {"section", "count", "sha256"} <= item.keys()
                and set(item) <= {"section", "count", "sha256", "values", "omitted_detail"}
            )
        ):
            return {k: field(name, v) if k == "values" else visit(v) for k, v in item.items()}
        if type(item) is not str or name in ARRAY_FIELDS:
            raise ValueError("reference field type differs")
        return (
            convert(kind, item, visible_at=visible_at)
            if encode
            else convert(item, kind, visible_at=visible_at)
        )

    def native_values(values: dict[str, Any]) -> dict[str, Any]:
        # These exact native tree slots represent ReportEvidenceValue scalars,
        # including the existing canonical scalar-list encoding. Other same-name
        # fields keep the ordinary codec behavior.
        return {
            key: visit(dict(name=key, value=value))["value"]
            for key, value in sorted(values.items())
        }

    def native_node(node: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in node.items():
            if key == "children":
                result[key] = [native_node(n) for n in child]
            elif key == "value_defaults":
                result[key] = native_values(child)
            elif key == "fact":
                result[key] = {
                    k: native_values(v) if k == "values" else visit(v) for k, v in child.items()
                }
            else:
                result[key] = visit(child)
        return result

    def native_packet(packet: dict[str, Any]) -> dict[str, Any]:
        from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore

        NativeFactStore.from_packet(packet)  # Exact schema before interpreting any slots.
        rows = [
            {
                key: (
                    [native_node(n) for n in child]
                    if key == "tree"
                    else native_values(child)
                    if key == "value_defaults"
                    else visit(child)
                )
                for key, child in record.items()
            }
            for record in packet["records"]
        ]
        converted = dict(packet, records=rows)
        NativeFactStore.from_packet(converted)
        return converted

    def visit(item: Any) -> Any:
        if type(item) is list:
            return [visit(child) for child in item]
        if type(item) is not dict:
            return item
        if native_facts and item.get("schema_version") == "nurse.v2.sre-fact-records/2":
            return native_packet(item)
        assertion = set(item) == {"fact_ref", "field", "value"} or (
            brief_fields and set(item) == {"evidence_ref", "field", "value"}
        )
        if assertion and item["field"] in kinds:
            translated = visit(dict(name=item["field"], value=item["value"]))
            return dict(item, value=translated["value"])
        if set(item) == {"name", "value"} and item["name"] in kinds:
            name, scalar = item["name"], item["value"]
            if name in ARRAY_FIELDS and type(scalar) is str:
                decoded = json.loads(scalar)
                if type(decoded) is not list or canonical_json_bytes(decoded).decode() != scalar:
                    raise ValueError("reference scalar-list encoding differs")
                _require_outbound({name: decoded}, case=case_id, snapshot=book.snapshot)
                translated = field(name, decoded)
                _require_outbound({name: translated}, case=case_id, snapshot=book.snapshot)
                return dict(name=name, value=canonical_json_bytes(translated).decode())
            return dict(name=name, value=field(name, scalar))
        result = {}
        for key, child in sorted(item.items()):
            if key in {"parameter_contracts", "tools", "skill_tools", "skills", "output_schema"}:
                # Documentation examples describe fields; they are not issued IDs.
                # All text is still scanned before and after this transformation.
                result[key] = child
            elif key == "scope" and type(child) is str:
                result[key] = child  # Resource/failure classification, not entity scope.
            elif key == "sections_state" and type(child) is dict:
                result[key] = {
                    field("candidate_service_key", k): visit(v) for k, v in sorted(child.items())
                }
            elif key in kinds:
                result[key] = field(key, child)
            else:
                result[key] = visit(child)
        return result

    result = visit(value)
    encode_json(result, limits)
    _require_outbound(result, case=case_id, snapshot=book.snapshot)
    if encode:
        working.to_state()  # A successful translation must remain persistable/reversible.
        book._rows, book._originals = working._rows, working._originals
        book._handles, book._counts = working._handles, working._counts
    return dict(result)


def encode_references(
    value: dict[str, Any],
    *,
    book: SreReferenceBook,
    visible_at: int,
    case_id: str,
    limits: ExchangeLimits = VIEW_LIMITS,
) -> dict[str, Any]:
    return _translate(
        value, book=book, visible_at=visible_at, case_id=case_id, encode=True, limits=limits
    )


def decode_references(
    value: dict[str, Any],
    *,
    book: SreReferenceBook,
    visible_at: int,
    case_id: str,
    limits: ExchangeLimits = VIEW_LIMITS,
) -> dict[str, Any]:
    return _translate(
        value, book=book, visible_at=visible_at, case_id=case_id, encode=False, limits=limits
    )
