"""Offline mABC PA with NURSE's explicit service-sequence matching protocol.

References must be independently supplied; this module neither constructs reference
paths from predictions nor invokes a model. Provenance records the supplier's claim
of independence, not a certification of reference quality. Only a complete reference
can be scored. Exact matching, direction conversion and per-case deduplication are
NURSE protocol choices; the un-clipped formula is PA = (Pc - 0.2 * Pi) / Pt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

PREDICTION_SCHEMA_VERSION = "nurse.v2.report-path-predictions/1"
REFERENCE_SCHEMA_VERSION = "nurse.v2.path-reference/1"
RESULT_SCHEMA_VERSION = "nurse.v2.path-accuracy/1"
AGGREGATE_SCHEMA_VERSION = "nurse.v2.path-accuracy-aggregate/1"
TAU = 0.2
_PREDICTION_STATUSES = {"available", "abstained", "missing", "invalid", "unavailable"}
_REFERENCE_STATUSES = {"complete", "unavailable", "incomplete", "not_applicable"}
_EXCLUDED_STATUSES = {"zero_reference_paths", "not_applicable"}
_FAILED_STATUSES = {
    "reference_unavailable",
    "reference_incomplete",
    "prediction_missing",
    "prediction_invalid",
    "prediction_unavailable",
}
_PIN_FIELDS = ("core_result_digest", "context_sha256", "request_sha256", "native_sha256")
_FATAL_COLLECTION_ISSUES = {
    "native_not_decodable",
    "native_response_schema_mismatch",
    "paths_missing",
    "paths_not_array",
    "abstention_field_missing",
    "empty_prediction_requires_abstention_reason",
    "unrecoverable_path_sequence",
}
_INVALID_PATH_ISSUES = {"unknown_service", "repeated_service", "not_a_root_to_symptom_path"}
ServicePath = tuple[str, ...]


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return bool(value)


def _choice(value: Any, choices: set[str], label: str) -> str:
    value = _text(value, label)
    if value not in choices:
        raise ValueError(f"unsupported {label}: {value}")
    return str(value)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _json_digest(value: Any) -> str:
    try:
        return sha256_hex(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("input must contain finite, serializable JSON data") from error


def _sequence(value: Any, label: str, *, reference: bool = False) -> ServicePath:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list of service keys")
    sequence = tuple(_text(service, f"{label} service key") for service in value)
    if reference and (len(sequence) < 2 or len(set(sequence)) != len(sequence)):
        raise ValueError(f"{label} must be a simple path with at least two services")
    return sequence


def _prediction_paths(prediction: dict[str, Any]) -> dict[ServicePath, bool]:
    rows = prediction.get("paths")
    if not isinstance(rows, list):
        raise ValueError("prediction paths must be a list")
    paths: dict[ServicePath, bool] = {}
    for row in rows:
        row = _object(row, "prediction path")
        sequence = _sequence(row.get("service_sequence"), "prediction service_sequence")
        invalid = _bool(row.get("invalid_for_scoring"), "invalid_for_scoring")
        issues = row.get("audit_issues")
        if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
            raise ValueError("prediction audit_issues must be a list of strings")
        # Identifiable malformed paths remain errors, even if a producer forgot its flag.
        invalid = invalid or len(sequence) < 2 or len(sequence) != len(set(sequence))
        invalid = invalid or bool(_INVALID_PATH_ISSUES.intersection(issues))
        # An otherwise valid duplicate must not erase another declaration's invalidity.
        paths[sequence] = paths.get(sequence, False) or invalid
    return paths


def _recognizable_raw_path(row: Any) -> bool:
    """Mirror the exporter's recoverability rule, without inferring missing services."""
    if type(row) is not dict:
        return False
    sequence = row.get("service_sequence")
    return type(sequence) is list and bool(sequence) and all(
        type(service) is str and bool(service) and service == service.strip()
        and len(service) <= 512 for service in sequence
    )


def _validate_prediction_projection(prediction: dict[str, Any]) -> None:
    """Require all recoverable declarations to survive the raw-to-canonical projection.

    This is internal consistency, not source authentication. Without the original
    native bytes and frozen alias/context owner chain, this standalone calculator
    cannot prove alias mappings or detect a coherent rewrite of both raw data and
    pins. Callers must verify those pins against their independently frozen inputs.
    """
    status = prediction["status"]
    issues = prediction.get("audit_issues", [])
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError("prediction collection audit_issues must be a list of strings")
    if status in {"available", "abstained"} and _FATAL_COLLECTION_ISSUES.intersection(issues):
        raise ValueError("prediction status contradicts fatal collection audit issues")
    if "raw_abstention_reason" in prediction:
        raw_reason = prediction["raw_abstention_reason"]
        normalized = raw_reason.strip() if isinstance(raw_reason, str) else None
        if normalized != prediction["abstention_reason"]:
            raise ValueError("raw abstention reason and normalized projection disagree")

    raw = prediction["raw_predicted_paths"]
    rows = prediction["paths"]
    if status == "unavailable" and prediction["native_sha256"] is not None:
        raise ValueError("unavailable status requires absent native input")
    if status in {"missing", "unavailable"} and (raw is not None or rows):
        raise ValueError("missing/unavailable status cannot hide a raw collection or projection")
    if not isinstance(raw, list):
        if rows or status in {"available", "abstained"}:
            raise ValueError("raw prediction collection cannot support this projection or status")
        return
    if not raw and status == "available":
        raise ValueError("an empty prediction must explicitly abstain; available requires paths")
    recognizable = [row for row in raw if _recognizable_raw_path(row)]
    if len(recognizable) != len(raw) and status != "invalid":
        raise ValueError("unrecoverable raw prediction requires invalid collection status")
    if len(recognizable) != len(rows):
        raise ValueError("projection must retain every recognizable raw declaration exactly once")

    aliases: dict[str, str] = {}
    for original, projected in zip(recognizable, rows, strict=True):
        model_sequence = original["service_sequence"]
        if projected.get("model_service_sequence") != model_sequence:
            raise ValueError("raw service_sequence and model_service_sequence projection disagree")
        for field in ("path_id", "edges", "rationale"):
            if field not in projected or _json_digest(projected[field]) != _json_digest(
                original.get(field)
            ):
                raise ValueError(f"raw {field} was changed or reordered in the projection")
        canonical = projected["service_sequence"]
        if len(canonical) != len(model_sequence):
            raise ValueError("raw and canonical projection sequence lengths disagree")
        for model_key, canonical_key in zip(model_sequence, canonical, strict=True):
            if model_key in aliases and aliases[model_key] != canonical_key:
                raise ValueError("projection maps the same service alias inconsistently")
            aliases[model_key] = canonical_key


def _validate_prediction(prediction: Any) -> dict[ServicePath, bool]:
    prediction = _object(prediction, "prediction")
    if prediction.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("unsupported prediction schema_version")
    _text(prediction.get("opaque_case_id"), "prediction opaque_case_id")
    _choice(prediction.get("direction"), {"root_to_symptom"}, "prediction direction")
    _choice(prediction.get("granularity"), {"service"}, "prediction granularity")
    status = _choice(prediction.get("status"), _PREDICTION_STATUSES, "prediction status")
    _bool(prediction.get("is_mock"), "prediction is_mock")
    for field in _PIN_FIELDS:
        value = prediction.get(field)
        if field in {"request_sha256", "native_sha256"} and value is None:
            if field not in prediction or status != "unavailable":
                raise ValueError(f"prediction {field} can be null only when unavailable")
        else:
            _hash(value, f"prediction {field}")
    if "raw_predicted_paths" not in prediction or "abstention_reason" not in prediction:
        raise ValueError("prediction must preserve raw_predicted_paths and abstention_reason")
    seal = _hash(prediction.get("artifact_sha256"), "prediction artifact_sha256")
    unsigned = {key: value for key, value in prediction.items() if key != "artifact_sha256"}
    if seal != _json_digest(unsigned):
        raise ValueError("prediction artifact_sha256 seal mismatch")
    paths = _prediction_paths(prediction)
    reason = prediction["abstention_reason"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("prediction abstention_reason must be a string or null")
    if status == "abstained":
        if paths:
            raise ValueError("abstained prediction must have empty paths")
        _text(reason, "abstained prediction reason")
    _validate_prediction_projection(prediction)
    return paths


def _validate_reference(reference: Any) -> set[ServicePath]:
    reference = _object(reference, "reference")
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported reference schema_version")
    _text(reference.get("opaque_case_id"), "reference opaque_case_id")
    direction = _choice(
        reference.get("direction"), {"root_to_symptom", "symptom_to_root"}, "reference direction"
    )
    _choice(reference.get("granularity"), {"service"}, "reference granularity")
    status = _choice(reference.get("status"), _REFERENCE_STATUSES, "reference status")
    is_mock = _bool(reference.get("is_mock"), "reference is_mock")
    provenance = _object(reference.get("provenance"), "reference provenance")
    kind = _choice(provenance.get("kind"), {"mock", "independent_reference"}, "provenance kind")
    _text(provenance.get("description"), "reference provenance description")
    if is_mock != (kind == "mock"):
        raise ValueError("reference is_mock must agree with provenance kind")
    rows = reference.get("paths")
    if not isinstance(rows, list):
        raise ValueError("reference paths must be a list of service sequences")
    paths = set()
    for row in rows:
        sequence = _sequence(row, "reference path", reference=True)
        paths.add(sequence if direction == "root_to_symptom" else tuple(reversed(sequence)))
    if status in {"not_applicable", "unavailable"} and paths:
        raise ValueError(f"reference status {status} requires empty paths")
    return paths


def _lists(paths: Iterable[ServicePath]) -> list[list[str]]:
    return [list(path) for path in sorted(paths)]


def _score_status(prediction_status: str, reference_status: str, pt: int | None) -> str:
    if reference_status == "not_applicable":
        return "not_applicable"
    if reference_status != "complete":
        return f"reference_{reference_status}"
    if pt == 0:
        return "zero_reference_paths"
    if prediction_status not in {"available", "abstained"}:
        return f"prediction_{prediction_status}"
    return "scored"


def evaluate_path_accuracy(
    prediction: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Validate one sealed export and independent reference, then score if defined.

    ``paths`` in a reference is a list of service-key lists. All services must already
    use the event's canonical service identity; this API never guesses instance maps.
    Incomplete/unavailable references and failed predictions yield ``pa=None`` with
    an explicit status. Scorable empty predictions require ``abstained`` with a
    nonempty reason. Structurally invalid or mismatched inputs raise ValueError.
    """
    predicted = _validate_prediction(prediction)
    truth = _validate_reference(reference)
    if prediction["opaque_case_id"] != reference["opaque_case_id"]:
        raise ValueError("prediction and reference opaque_case_id mismatch")
    if prediction["is_mock"] != reference["is_mock"]:
        raise ValueError("prediction and reference is_mock mismatch")
    reference_digest = _json_digest(reference)
    reference_status = reference["status"]
    prediction_status = prediction["status"]
    status = _score_status(prediction_status, reference_status, len(truth))
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "opaque_case_id": prediction["opaque_case_id"],
        "direction": "root_to_symptom",
        "granularity": "service",
        "tau": TAU,
        "Pc": None,
        "Pi": None,
        "Pt": len(truth) if reference_status == "complete" else None,
        "pa": None,
        "matched": [],
        "incorrect": [],
        "missed": [],
        "prediction_status": prediction_status,
        "reference_status": reference_status,
        "status": status,
        "expected_for_scoring": status not in _EXCLUDED_STATUSES,
        "is_mock": prediction["is_mock"],
        "input_pins": {
            "prediction_artifact_sha256": prediction["artifact_sha256"],
            "reference_sha256": reference_digest,
            **{field: prediction[field] for field in _PIN_FIELDS},
        },
    }
    if status == "scored":
        matched = {path for path, invalid in predicted.items() if path in truth and not invalid}
        incorrect = set(predicted) - matched
        result.update(
            status="scored",
            Pc=len(matched),
            Pi=len(incorrect),
            pa=(len(matched) - TAU * len(incorrect)) / len(truth),
            matched=_lists(matched),
            incorrect=_lists(incorrect),
            missed=_lists(truth - matched),
        )
    return result


def _count(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _validate_result(result: Any) -> None:
    result = _object(result, "case result")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported case result schema_version")
    _text(result.get("opaque_case_id"), "result opaque_case_id")
    _bool(result.get("is_mock"), "result is_mock")
    _choice(result.get("direction"), {"root_to_symptom"}, "result direction")
    _choice(result.get("granularity"), {"service"}, "result granularity")
    status = _choice(
        result.get("status"), {"scored"} | _EXCLUDED_STATUSES | _FAILED_STATUSES, "result status"
    )
    expected = _bool(result.get("expected_for_scoring"), "expected_for_scoring")
    if expected != (status not in _EXCLUDED_STATUSES):
        raise ValueError("case result expected_for_scoring contradicts its status")
    if type(result.get("tau")) not in {float, int} or result["tau"] != TAU:
        raise ValueError("case result tau must be 0.2")
    prediction_status = _choice(
        result.get("prediction_status"), _PREDICTION_STATUSES, "result prediction_status"
    )
    reference_status = _choice(
        result.get("reference_status"), _REFERENCE_STATUSES, "result reference_status"
    )
    pt = result.get("Pt")
    if reference_status == "complete":
        _count(pt, "Pt")
    elif pt is not None:
        raise ValueError("Pt must be null without a complete reference")
    if status != _score_status(prediction_status, reference_status, pt):
        raise ValueError("result status contradicts prediction/reference status and Pt")
    _json_digest(result)
    if status != "scored":
        if any(result.get(key) is not None for key in ("Pc", "Pi", "pa")):
            raise ValueError("unscorable case result cannot contain metrics")
        if any(result.get(key) != [] for key in ("matched", "incorrect", "missed")):
            raise ValueError("unscorable case result cannot contain match decisions")
        return
    pc = _count(result.get("Pc"), "Pc")
    pi = _count(result.get("Pi"), "Pi")
    pt = _count(result.get("Pt"), "Pt", positive=True)
    pa = result.get("pa")
    if type(pa) not in {float, int} or not math.isfinite(pa):
        raise ValueError("scored case result pa must be finite")
    if pc > pt or not math.isclose(pa, (pc - TAU * pi) / pt, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("case result counts and PA formula disagree")
    sequences: dict[str, set[ServicePath]] = {}
    for key, size in (("matched", pc), ("incorrect", pi), ("missed", pt - pc)):
        rows = result.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"result {key} must be a list")
        paths = {_sequence(row, f"result {key}", reference=key != "incorrect") for row in rows}
        if len(rows) != size or len(paths) != size:
            raise ValueError(f"result {key} must contain exactly its unique count")
        sequences[key] = paths
    if sequences["matched"] & (sequences["incorrect"] | sequences["missed"]):
        raise ValueError("matched result paths cannot also be incorrect or missed")


def aggregate_path_accuracy(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one condition's distinct cases without hiding expected-case failures.

    Full-batch ``macro_pa`` and ``micro_pa`` are null if any expected case failed.
    ``scorable_subset`` always explicitly names the available diagnostic subset.
    Pt=0 and not-applicable cases are excluded, and mock/empirical mixing is refused.
    """
    rows = list(results)
    seen: set[str] = set()
    mocks: set[bool] = set()
    for result in rows:
        _validate_result(result)
        case_id = result["opaque_case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case in path-accuracy aggregate: {case_id}")
        seen.add(case_id)
        mocks.add(result["is_mock"])
    if len(mocks) > 1:
        raise ValueError("cannot mix mock and empirical path-accuracy results")
    scored = [result for result in rows if result["status"] == "scored"]
    excluded = sum(result["status"] in _EXCLUDED_STATUSES for result in rows)
    expected = len(rows) - excluded
    failed = expected - len(scored)
    pc = sum(result["Pc"] for result in scored)
    pi = sum(result["Pi"] for result in scored)
    pt = sum(result["Pt"] for result in scored)
    macro = sum(result["pa"] for result in scored) / len(scored) if scored else None
    micro = (pc - TAU * pi) / pt if pt else None
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "incomplete" if failed else "complete" if scored else "no_scorable_cases",
        "tau": TAU,
        "is_mock": next(iter(mocks)) if mocks else None,
        "macro_pa": macro if not failed else None,
        "micro_pa": micro if not failed else None,
        "scorable_subset": {"Pc": pc, "Pi": pi, "Pt": pt, "macro_pa": macro, "micro_pa": micro},
        "coverage": {
            "total_cases": len(rows),
            "expected_cases": expected,
            "scored_cases": len(scored),
            "failed_cases": failed,
            "excluded_cases": excluded,
            "scored_fraction": len(scored) / expected if expected else None,
        },
        "status_counts": dict(sorted(Counter(result["status"] for result in rows).items())),
        "case_ids": sorted(seen),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    _json_digest(value)
    return _object(value, "input")


def main(argv: list[str] | None = None) -> int:
    """Calculate a single case from two local JSON artifacts, without model access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.resolve() in {
            args.predictions.resolve(),
            args.reference.resolve(),
        }:
            raise ValueError("output must not overwrite a prediction or reference input")
        result = evaluate_path_accuracy(_read_json(args.predictions), _read_json(args.reference))
        encoded = canonical_json_bytes(result) + b"\n"
        if args.output is None:
            print(encoded.decode("utf-8"), end="")
        else:
            args.output.write_bytes(encoded)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
