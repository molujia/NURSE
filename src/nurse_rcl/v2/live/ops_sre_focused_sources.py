"""Pure source-bound attribution and bounded comparisons of already admitted cards.

Only registered metadata is inspected. No source records, samples, measurements,
rankings, providers or question-state transitions are accessed here.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

ATTRIBUTION_SCHEMA = "nurse.v2.sre-source-attribution/1"
RELATION_SCHEMA = "nurse.v2.sre-directed-source-relation/2"
BINDING_POLICY = "sre-focused-source-binding/1"
REGISTRY_SCHEMA = "nurse.v2.sre-source-registry/1"
FIELDS = (
    "metric_name",
    "signal_family",
    "modality",
    "statistic",
    "aggregation",
    "unit",
    "baseline_window",
    "fault_window",
    "role",
    "span_kind",
    "denominator",
    "coverage",
)
_REGISTERED = {
    ("sre-prepared-summary/1", "prepared_summary"),
    *(
        ("ops-symptoms/1", kind)
        for kind in (
            "fixed_summary",
            "paid_episode",
            "supplementary_episode",
            "error_summary",
        )
    ),
}
_MISSING = object()


def _get(value: Any, name: str) -> Any:
    return value.get(name, _MISSING) if type(value) is dict else getattr(value, name, _MISSING)


def _at(value: Any, *path: str) -> Any:
    for part in path:
        if value is _MISSING:
            break
        value = _get(value, part)
    return value


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return deepcopy(value)


def _state(value: Any) -> str:
    if value is _MISSING:
        return "not_supplied"
    if value is None:
        return "supplied_null"
    if type(value) is str and value == "unknown":
        return "supplied_unknown"
    return "supplied_value"


def _cell(value: Any = _MISSING, *paths: str) -> dict[str, Any]:
    return dict(
        state=_state(value),
        source_fields=list(paths),
        **({} if value is _MISSING else {"value": _json(value)}),
    )


def _same(left: Any, right: Any) -> bool:
    return canonical_json_bytes(_json(left)) == canonical_json_bytes(_json(right))


def _merge(fields: dict[str, Any], name: str, value: Any, path: str) -> None:
    if value is _MISSING:
        return
    existing = fields[name]
    if existing["state"] == "supplied_value" and _state(value) == "supplied_value":
        if not _same(existing["value"], value):
            raise ValueError("focused source binding field conflicts")
        existing["source_fields"].append(path)
    elif existing["state"] == "not_supplied":
        fields[name] = _cell(value, path)


def _hint(*values: Any) -> Any:
    supplied = [value for value in values if value is not _MISSING and value is not None]
    if supplied and (
        any(type(value) is not str or not value for value in supplied) or len(set(supplied)) != 1
    ):
        raise ValueError("focused source binding identities conflict")
    return supplied[0] if supplied else _MISSING


def _target(row: dict[str, Any], body: dict[str, Any]) -> Any:
    targets = row.get("targets", [])
    if type(targets) is not list or len(targets) > 1:
        raise ValueError("focused source binding target differs")
    return _hint(
        targets[0] if targets else _MISSING,
        _get(body, "candidate_service_key"),
        _at(row, "parameters", "candidate_service_key"),
    )


def _record_fields(row: dict[str, Any]) -> dict[str, Any]:
    body = row["payload"]
    fields = {name: _cell(_get(body, name), "payload." + name) for name in FIELDS}
    fields["unit"] = _cell(_get(row, "unit"), "unit")
    fields["coverage"] = _cell(_get(row, "coverage"), "coverage")
    # A publication role or supplementary authority is not a telemetry role.
    fields["role"] = _cell(_get(body, "telemetry_role"), "payload.telemetry_role")
    return fields


def _check_interval(row: dict[str, Any], interval: Any) -> None:
    window = row.get("window_ns")
    if interval is not _MISSING and interval is not None and window is not None:
        bounds = [_get(interval, "start_unix_ns"), _get(interval, "end_unix_ns")]
        if _MISSING in bounds or not _same(window, bounds):
            raise ValueError("focused source binding record window conflicts")


def _scalar_binding(
    row: dict[str, Any],
    fields: dict[str, Any],
    requests: Any,
    sources: Any,
) -> dict[str, Any]:
    body = row["payload"]
    target = _target(row, body)
    request_key = _hint(
        _at(row, "parameters", "request_key"),
        _get(body, "request_key"),
        _at(body, "feedback", "request_key"),
    )
    observation_id = _hint(_at(row, "parameters", "observation_id"), _get(body, "observation_id"))
    source_refs = row.get("source_refs", [])
    matched = [source for source in sources if _get(source, "source_sha256") in source_refs]
    if len(matched) > 1:
        raise ValueError("focused source binding is not unique")
    if not matched:
        known = any(_get(request, "request_key") == request_key for request in requests) or any(
            _at(source, "observation", "observation_id") == observation_id for source in sources
        )
        if source_refs and known:
            raise ValueError("focused source binding source conflicts")
        return dict(status="not_supplied")
    source = matched[0]
    source_id = _get(source, "source_sha256")
    obs = _get(source, "observation")
    actual_observation = _get(obs, "observation_id")
    actual_target = _get(source, "candidate_service_key")
    if (observation_id is not _MISSING and observation_id != actual_observation) or (
        target is not _MISSING and target != actual_target
    ):
        raise ValueError("focused source binding observation or subject conflicts")
    source_files = _get(source, "source_file_sha256s")
    if source_files is not _MISSING and not set(source_refs) <= {source_id, *source_files}:
        raise ValueError("focused source binding provenance conflicts")
    matches = [request for request in requests if _get(request, "source_sha256") == source_id]
    # The legacy summary publisher's request_key is a source-directory hint,
    # not an executed request. It cannot resolve a non-unique source chain.
    if body.get("kind") == "prepared_summary" and len(matches) > 1:
        raise ValueError("focused source binding request is not unique")
    matches = [
        request
        for request in matches
        if request_key is _MISSING or _get(request, "request_key") == request_key
    ]
    if len(matches) > 1:
        raise ValueError("focused source binding request is not unique")
    if not matches:
        if request_key is not _MISSING:
            raise ValueError("focused source binding request conflicts")
        return dict(status="not_supplied")
    request = matches[0]
    if (
        _get(request, "observation_id") != actual_observation
        or _get(request, "candidate_service_key") != actual_target
        or _get(request, "signal_id") != _get(source, "signal_id")
    ):
        raise ValueError("focused source binding registered chain conflicts")
    if actual_target is None or actual_target is _MISSING:
        return dict(status="not_supplied")
    _check_interval(row, _get(source, "query_interval"))
    for name, value, path in (
        ("metric_name", _get(source, "metric_name"), "registered_source.metric_name"),
        (
            "signal_family",
            _get(obs, "signal_family"),
            "registered_source.observation.signal_family",
        ),
        ("modality", _get(obs, "modality"), "registered_source.observation.modality"),
        ("aggregation", _get(obs, "aggregation"), "registered_source.observation.aggregation"),
        ("unit", _get(obs, "unit"), "registered_source.observation.unit"),
        (
            "baseline_window",
            _at(source, "time_contract", "baseline_window"),
            "registered_source.time_contract.baseline_window",
        ),
        (
            "fault_window",
            _at(source, "time_contract", "fault_window"),
            "registered_source.time_contract.fault_window",
        ),
        (
            "coverage",
            _at(source, "quality", "coverage_quality"),
            "registered_source.quality.coverage_quality",
        ),
    ):
        _merge(fields, name, value, path)
    return dict(
        status="bound",
        request_key=_get(request, "request_key"),
        observation_id=actual_observation,
        source_sha256=source_id,
    )


def _error_binding(
    row: dict[str, Any],
    fields: dict[str, Any],
    observations: Any,
) -> dict[str, Any]:
    facts = row["payload"].get("facts", {})
    if type(facts) is not dict:
        raise ValueError("focused source binding error facts differ")
    _target(row, facts)
    for name in ("modality", "statistic", "span_kind"):
        fields[name] = _cell(_get(facts, name), "payload.facts." + name)
    fields["coverage"] = _cell(_get(facts, "sampling_coverage"), "payload.facts.sampling_coverage")
    names = ("baseline_known_denominator", "fault_known_denominator")
    values = {name: facts[name] for name in names if name in facts}
    states = [_state(_get(facts, name)) for name in names]
    state = next(
        (
            state
            for state in ("not_supplied", "supplied_unknown", "supplied_null")
            if state in states
        ),
        "supplied_value",
    )
    fields["denominator"] = dict(
        state=state,
        value=deepcopy(values),
        source_fields=["payload.facts." + name for name in names],
    )
    pins = facts.get("observation_sha256s")
    if pins is None:
        return dict(status="not_supplied")
    if (
        type(pins) not in (list, tuple)
        or len(pins) != 2
        or len(set(pins)) != 2
        or list(row.get("source_refs", [])) != list(pins)
    ):
        raise ValueError("focused source binding error provenance conflicts")
    matched = [[obs for obs in observations if _get(obs, "result_sha256") == pin] for pin in pins]
    if any(len(group) > 1 for group in matched):
        raise ValueError("focused source binding error observation is not unique")
    if any(not group for group in matched):
        return dict(status="not_supplied")
    before, after = (group[0] for group in matched)
    if any(_get(obs, "modality") != facts.get("modality") for obs in (before, after)):
        raise ValueError("focused source binding error modality conflicts")
    _check_interval(row, _get(after, "interval"))
    if "interval" in facts and not _same(facts["interval"], _get(after, "interval")):
        raise ValueError("focused source binding error window conflicts")
    _merge(
        fields,
        "baseline_window",
        _get(before, "interval"),
        "registered_error_observations[baseline].interval",
    )
    _merge(
        fields,
        "fault_window",
        _get(after, "interval"),
        "registered_error_observations[fault].interval",
    )
    return dict(status="bound", observation_sha256s=list(pins))


def enrich_semantic(
    row: dict[str, Any],
    requests: Any,
    observations: Any,
    sources: Any,
) -> dict[str, Any]:
    """Keep the original card intact; append only uniquely bound attribution."""
    result = deepcopy(row)
    body = result.get("payload", {})
    if (
        result.get("node") != "symptoms"
        or type(body) is not dict
        or (result.get("tool_version"), body.get("kind")) not in _REGISTERED
    ):
        return result
    result.pop("source_attribution", None)
    fields = _record_fields(result)
    binding = (
        _error_binding(result, fields, observations)
        if body["kind"] == "error_summary"
        else _scalar_binding(result, fields, requests, sources)
    )
    binding.update(
        card_id=result["card_id"],
        record_sha256=sha256_hex(canonical_json_bytes(result)),
        targets=deepcopy(result.get("targets", [])),
        source_refs=deepcopy(result.get("source_refs", [])),
    )
    result["source_attribution"] = dict(
        schema_version=ATTRIBUTION_SCHEMA,
        authority="attribution_of_same_record_not_new_evidence",
        binding=binding,
        fields=fields,
    )
    if "source_attribution" in row and not _same(
        row["source_attribution"],
        result["source_attribution"],
    ):
        raise ValueError("focused source binding attribution conflicts")
    return result


def _comparison_fields(row: dict[str, Any]) -> dict[str, Any]:
    attribution = row.get("source_attribution")
    if attribution is None:
        return _record_fields(row)
    if (
        type(attribution) is not dict
        or attribution.get("schema_version") != ATTRIBUTION_SCHEMA
        or type(attribution.get("fields")) is not dict
        or set(attribution["fields"]) != set(FIELDS)
    ):
        raise ValueError("focused comparison attribution schema differs")
    for cell in attribution["fields"].values():
        if (
            type(cell) is not dict
            or cell.get("state")
            not in {
                "not_supplied",
                "supplied_value",
                "supplied_null",
                "supplied_unknown",
            }
            or type(cell.get("source_fields")) is not list
        ):
            raise ValueError("focused comparison field state differs")
    return cast(dict[str, Any], attribution["fields"])


def _record_role(row: dict[str, Any]) -> dict[str, Any]:
    kind = row["payload"].get("kind")
    roles = dict(
        prepared_summary="window_summary",
        fixed_summary="window_summary",
        paid_episode="episode_measurement",
        supplementary_episode="episode_measurement",
        error_summary="observed_record_summary",
    )
    return _cell(roles.get(kind, _MISSING), "payload.kind")


def focused_source_relation(rows: list[dict[str, Any]], scope: list[str]) -> dict[str, Any]:
    """Compare supplied dimensions, never diagnose, measure or combine support."""
    if (
        type(rows) is not list
        or not 1 <= len(rows) <= 2
        or type(scope) is not list
        or not 1 <= len(scope) <= 2
        or any(type(key) is not str for key in scope)
        or len(set(scope)) != len(scope)
        or len({row["card_id"] for row in rows}) != len(rows)
        or any(
            row.get("node") != "symptoms"
            or len(row.get("targets", [])) != 1
            or not set(row["targets"]) <= set(scope)
            for row in rows
        )
    ):
        raise ValueError("focused comparison source bounds differ")
    missing = [
        candidate for candidate in scope if not any(row["targets"] == [candidate] for row in rows)
    ]
    sides = [_comparison_fields(row) for row in rows]
    sides = [
        dict(fields, record_role=_record_role(row)) for fields, row in zip(sides, rows, strict=True)
    ]
    dimensions = []
    for name in (*FIELDS, "record_role"):
        left = sides[0][name]
        right = sides[1][name] if len(sides) == 2 else _cell()
        known = left["state"] == right["state"] == "supplied_value"
        status = (
            "same"
            if known and _same(left["value"], right["value"])
            else ("different" if known else "not_supplied")
        )
        dimensions.append(
            dict(field=name, status=status, left=deepcopy(left), right=deepcopy(right))
        )
    status = (
        "unknown"
        if missing or len(rows) != 2
        else "incomparable"
        if any(item["status"] == "different" for item in dimensions)
        else "unknown"
        if any(item["status"] == "not_supplied" for item in dimensions)
        else "comparable"
    )
    return dict(
        schema_version=RELATION_SCHEMA,
        authority="source_relationship_not_new_evidence",
        status=status,
        dimensions=dimensions,
        missing_candidate_service_keys=missing,
        score_authority=False,
        limitations=[
            "comparison_not_independence_or_causality",
            "unknown_not_absent_normal_or_answered",
            "existing_sources_only_no_implicit_measurement",
        ],
    )


def focused_source_binding(
    snapshot: str,
    parameters: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    """Seal the actual attribution-bearing records and comparison policy."""
    return sha256_hex(
        canonical_json_bytes(
            dict(
                policy=BINDING_POLICY,
                attribution_schema=ATTRIBUTION_SCHEMA,
                relation_schema=RELATION_SCHEMA,
                snapshot=snapshot,
                parameters=parameters,
                source_rows=rows,
            )
        )
    )


_REQUEST_FIELDS = {
    "request_key",
    "source_sha256",
    "observation_id",
    "candidate_service_key",
    "signal_id",
    "canonical_series_key",
}
_SOURCE_FIELDS = {
    "source_sha256",
    "signal_id",
    "candidate_service_key",
    "metric_name",
    "source_file_sha256s",
    "observation",
    "time_contract",
    "query_interval",
    "quality",
}
_OBSERVATION_FIELDS = {"observation_id", "signal_family", "modality", "unit", "aggregation"}
_ERROR_FIELDS = {"result_sha256", "modality", "interval"}


def _pick_metadata(value: Any, names: set[str]) -> dict[str, Any]:
    result = {}
    for name in sorted(names):
        field = _get(value, name)
        if field is _MISSING:
            raise ValueError("focused source registry required metadata is missing")
        result[name] = _json(field)
    return result


def _registry_fields(value: Any, names: set[str]) -> None:
    if type(value) is not dict or set(value) != names:
        raise ValueError("focused source registry fields differ")


def _registry_text(value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or not 1 <= len(value) <= 2048:
        raise ValueError("focused source registry identity type differs")


def _registry_pin(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("focused source registry pin differs")


def _registry_interval(value: Any) -> None:
    _registry_fields(value, {"start_unix_ns", "end_unix_ns"})
    if (
        any(type(bound) is not int or not 0 <= bound < 10**30 for bound in value.values())
        or value["start_unix_ns"] >= value["end_unix_ns"]
    ):
        raise ValueError("focused source registry interval differs")


def validate_source_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Validate a metadata packet; its enclosing signed ledger supplies authority.

    The existing checkpoint component guard bounds this non-outbound packet.
    Fixed fields prevent arbitrary nested JSON, raw records or recursive refs.
    """
    _registry_fields(
        registry,
        {
            "schema_version",
            "registry_sha256",
            "requests",
            "observations",
            "sources",
        },
    )
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError("focused source registry schema differs")
    _registry_pin(registry["registry_sha256"])
    if any(type(registry[name]) is not list for name in ("requests", "observations", "sources")):
        raise ValueError("focused source registry collections differ")
    sources = {}
    observation_ids = set()
    for source in registry["sources"]:
        _registry_fields(source, _SOURCE_FIELDS)
        _registry_pin(source["source_sha256"])
        _registry_text(source["signal_id"])
        _registry_text(source["candidate_service_key"], optional=True)
        _registry_text(source["metric_name"], optional=True)
        files = source["source_file_sha256s"]
        if type(files) is not list or not files:
            raise ValueError("focused source registry provenance differs")
        for pin in files:
            _registry_pin(pin)
        if files != sorted(set(files)):
            raise ValueError("focused source registry provenance is not unique")
        observation = source["observation"]
        _registry_fields(observation, _OBSERVATION_FIELDS)
        for value in observation.values():
            _registry_text(value)
        if source["source_sha256"] in sources or observation["observation_id"] in observation_ids:
            raise ValueError("focused source registry source identity is not unique")
        observation_ids.add(observation["observation_id"])
        sources[source["source_sha256"]] = source
        contract = source["time_contract"]
        _registry_fields(contract, {"baseline_window", "fault_window"})
        for interval in (*contract.values(), source["query_interval"]):
            _registry_interval(interval)
        if (
            contract["baseline_window"]["end_unix_ns"] > contract["fault_window"]["start_unix_ns"]
            or source["query_interval"] != contract["fault_window"]
        ):
            raise ValueError("focused source registry named windows differ")
        _registry_fields(source["quality"], {"coverage_quality"})
        coverage = source["quality"]["coverage_quality"]
        if (
            coverage is not None
            and coverage != "unknown"
            and (
                type(coverage) not in (int, float)
                or not math.isfinite(coverage)
                or not 0 <= coverage <= 1
            )
        ):
            raise ValueError("focused source registry coverage differs")
    request_ids = set()
    for request in registry["requests"]:
        _registry_fields(request, _REQUEST_FIELDS)
        for name in ("request_key", "observation_id", "signal_id", "canonical_series_key"):
            _registry_text(request[name])
        _registry_text(request["candidate_service_key"], optional=True)
        _registry_pin(request["source_sha256"])
        if request["request_key"] in request_ids:
            raise ValueError("focused source registry request identity is not unique")
        request_ids.add(request["request_key"])
        source = sources.get(request["source_sha256"])
        if (
            source is None
            or any(
                request[name] != source[name]
                for name in (
                    "candidate_service_key",
                    "signal_id",
                )
            )
            or request["observation_id"] != source["observation"]["observation_id"]
        ):
            raise ValueError("focused source registry request chain differs")
    error_ids = set()
    for observation in registry["observations"]:
        _registry_fields(observation, _ERROR_FIELDS)
        _registry_pin(observation["result_sha256"])
        if (
            observation["result_sha256"] in error_ids
            or type(observation["modality"]) is not str
            or observation["modality"] not in {"trace", "log"}
        ):
            raise ValueError("focused source registry error identity or modality differs")
        error_ids.add(observation["result_sha256"])
        _registry_interval(observation["interval"])
    unsigned = {key: value for key, value in registry.items() if key != "registry_sha256"}
    try:
        digest = sha256_hex(canonical_json_bytes(unsigned))
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("focused source registry canonical JSON differs") from None
    if digest != registry["registry_sha256"]:
        raise ValueError("focused source registry canonical hash differs")
    return deepcopy(registry)


def source_registry(data: Any) -> dict[str, Any]:
    """Copy a fixed metadata allowlist from the owner's already-loaded objects."""
    requests = [
        _pick_metadata(request, _REQUEST_FIELDS) for request in _at(data, "catalog", "requests")
    ]
    observations = [
        _pick_metadata(observation, _ERROR_FIELDS)
        for observation in _at(data, "diagnostics", "errors")
    ]
    sources = []
    scalar_fields = {"source_sha256", "signal_id", "candidate_service_key", "metric_name"}
    for source in _get(data, "sources"):
        metadata = _pick_metadata(source, scalar_fields)
        metadata.update(
            source_file_sha256s=list(_get(source, "source_file_sha256s")),
            observation=_pick_metadata(_get(source, "observation"), _OBSERVATION_FIELDS),
            time_contract=_pick_metadata(
                _get(source, "time_contract"),
                {
                    "baseline_window",
                    "fault_window",
                },
            ),
            query_interval=_json(_get(source, "query_interval")),
            quality=_pick_metadata(_get(source, "quality"), {"coverage_quality"}),
        )
        sources.append(metadata)
    packet = dict(
        schema_version=REGISTRY_SCHEMA,
        requests=requests,
        observations=observations,
        sources=sources,
    )
    packet["registry_sha256"] = sha256_hex(canonical_json_bytes(packet))
    return validate_source_registry(packet)


def enrich_registered(row: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """Reproduce attribution using only the signed checkpoint's metadata packet."""
    admitted = validate_source_registry(registry)
    return enrich_semantic(
        row,
        requests=admitted["requests"],
        observations=admitted["observations"],
        sources=admitted["sources"],
    )
