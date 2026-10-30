"""Lossless signal descriptors and legal navigation, without scoring authority.

Only the owner can bind this display to admitted sources. This module operates
on the bounded scientific wire, never on telemetry or saved future B0 results.
"""

from __future__ import annotations

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import decode_json_object, encode_json
from nurse_rcl.v2.live.normal_semantic_compact_exchange import _at, _row, _validate_wire
from nurse_rcl.v2.live.normal_semantic_wire import (
    NEED_COLUMNS,
    SIGNAL_COLUMNS,
    WIRE_LIMITS,
    WIRE_VERSION,
)
from nurse_rcl.v2.mechanism_library import build_mechanism_library

INDEXED_WIRE_VERSION = "nurse.v2.normal-semantic-indexed-wire/1"
DESCRIPTOR_COLUMNS = (
    "metric_name", "family", "modality", "unit", "evidence_state", "quality_flags",
    "formal_scoring_eligible", "lifecycle_meaning", "lifecycle_zero_count", "lifecycle_eligible",
)
INDEXED_SIGNAL_COLUMNS = (
    "reference_id", "candidate", "descriptor_index", "baseline_summary", "fault_summary",
    "fact_set",
)


def _copy(wire: dict[str, Any]) -> dict[str, Any]:
    return decode_json_object(encode_json(wire, WIRE_LIMITS), WIRE_LIMITS)


def encode_indexed_wire(wire: dict[str, Any]) -> dict[str, Any]:
    """Keep one row per signal; deduplicate only exact descriptor JSON bytes."""
    result = _copy(wire)
    _validate_wire(result)
    descriptors: list[list[Any]] = []
    indices: dict[bytes, int] = {}
    signals = []
    for row in result["signals"]:
        signal = _row(SIGNAL_COLUMNS, row)
        descriptor = [signal[key] for key in DESCRIPTOR_COLUMNS]
        identity = canonical_json_bytes(descriptor)
        if identity not in indices:
            indices[identity] = len(descriptors)
            descriptors.append(descriptor)
        signal["descriptor_index"] = indices[identity]
        signals.append([signal[key] for key in INDEXED_SIGNAL_COLUMNS])
    result.update(schema_version=INDEXED_WIRE_VERSION, signals=signals,
                  signal_columns=list(INDEXED_SIGNAL_COLUMNS),
                  signal_descriptor_columns=list(DESCRIPTOR_COLUMNS),
                  signal_descriptors=descriptors)
    encode_json(result, WIRE_LIMITS)
    return result


def decode_indexed_wire(encoded: dict[str, Any]) -> dict[str, Any]:
    """Reject noncanonical dictionaries as well as invalid scientific values."""
    result = _copy(encoded)
    if result.get("schema_version") != INDEXED_WIRE_VERSION or (
        result.get("signal_columns") != list(INDEXED_SIGNAL_COLUMNS)
        or result.pop("signal_descriptor_columns", None) != list(DESCRIPTOR_COLUMNS)
    ):
        raise ValueError("invalid indexed semantic wire")
    descriptors = result.pop("signal_descriptors", None)
    if type(descriptors) is not list or type(result.get("signals")) is not list:
        raise ValueError("invalid semantic descriptor table")
    signals = []
    for row in result["signals"]:
        signal = _row(INDEXED_SIGNAL_COLUMNS, row)
        signal.update(_row(DESCRIPTOR_COLUMNS, _at(descriptors, signal.pop("descriptor_index"))))
        signals.append([signal[key] for key in SIGNAL_COLUMNS])
    result.update(schema_version=WIRE_VERSION, signals=signals, signal_columns=list(SIGNAL_COLUMNS))
    _validate_wire(result)
    # Also rejects unused/duplicate/reordered dictionary entries and extra fields.
    if canonical_json_bytes(encode_indexed_wire(result)) != canonical_json_bytes(encoded):
        raise ValueError("semantic descriptor dictionary differs")
    return result


def require_indexed_wire(encoded: dict[str, Any], original_wire: dict[str, Any]) -> dict[str, Any]:
    """Compare full decoded science to the owner's original, not a supplied hash."""
    decoded = decode_indexed_wire(encoded)
    if encode_json(decoded, WIRE_LIMITS) != encode_json(original_wire, WIRE_LIMITS):
        raise ValueError("indexed wire differs from owned wire")
    return decoded


def build_action_index(wire: dict[str, Any]) -> list[dict[str, Any]]:
    """List each queryable need once, in C/library/request/need order.

    This is initial navigation, not a fixed execution sequence or a claim that
    a mechanism holds. The existing compiler and feedback scheduler still decide.
    """
    value = _copy(wire)
    _validate_wire(value)
    signals = {row[0]: _row(SIGNAL_COLUMNS, row) for row in value["signals"]}
    definitions = [_row(NEED_COLUMNS, row) for row in value["need_definitions"]]
    mechanisms = [d.mechanism_key for d in build_mechanism_library().definitions]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request, reference, needs in value["requests"]:
        signal = signals[reference]
        by_mechanism: dict[str, list[dict[str, Any]]] = {}
        for key, index in needs:
            need = definitions[index]
            if need["queryable"] is not True:
                continue
            mechanism = need["mechanism_key"]
            if mechanism not in mechanisms:
                raise ValueError("queryable need mechanism not registered")
            by_mechanism.setdefault(mechanism, []).append(dict(
                need_key=key, predicate_kind=need["predicate_kind"],
                requirement=need["requirement"], initial_tier=need["initial_tier"],
            ))
        for mechanism, legal in by_mechanism.items():
            groups.setdefault((signal["candidate"], mechanism), []).append(dict(
                request_key=request, signal_reference=reference,
                metric_name=signal["metric_name"], family=signal["family"], unit=signal["unit"],
                needs=legal,
            ))
    return [dict(candidate_service_key=candidate, mechanism_key=mechanism,
                 options=groups[candidate, mechanism])
            for candidate, _ in value["candidates"] for mechanism in mechanisms
            if (candidate, mechanism) in groups]


def require_action_index(index: list[dict[str, Any]], wire: dict[str, Any]) -> None:
    if encode_json(index, WIRE_LIMITS) != encode_json(build_action_index(wire), WIRE_LIMITS):
        raise ValueError("semantic action index differs from owned wire")
