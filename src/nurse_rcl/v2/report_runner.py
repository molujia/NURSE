"""One bounded embedded-Hermes exchange and independently pinned in-memory replay.

No transport settings, network client, retry ledger, persistence, or crash resume live here.
Offline replay recomputes all domain decisions; absent-wire transport events are attested
only by the external audit pin, not proven by replay.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.deterministic_report import _report_input_snapshot
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, decode_json_object, encode_json
from nurse_rcl.v2.hermes_exchange import (
    ExchangeAttempt,
    PreparedExchange,
    execute_once,
    prepare_exchange,
)
from nurse_rcl.v2.hermes_response import _valid_usage, parse_response
from nurse_rcl.v2.report_composition_proposals import parse_report_composition_response
from nurse_rcl.v2.report_evidence import build_report_evidence_projection
from nurse_rcl.v2.report_mechanical import validate_report_composition_mechanics
from nurse_rcl.v2.report_presentation import (
    _presentation_local_domains,
    _same_document,
    build_report_composition_presentation,
)
from nurse_rcl.v2.report_prompt import (
    ReportSources,
    _bytes,
    _preflight_inputs,
    _preserve_sources,
    _prompt_from_projection,
    _snapshot_model,
    _snapshot_sources,
    _source_kwargs,
    _sources_bytes,
)
from nurse_rcl.v2.report_run_schemas import ReportRunAudit, ReportRunPolicy, _preflight_audit
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_INPUT_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)


def _runtime_local_domains(cls: type[BaseModel], state: dict[str, Any]) -> None:
    """Existing receipt structure only; hashes and replay retain their original gates."""
    _presentation_local_domains(cls, state)
    if cls is ReportRunPolicy:
        if state["configuration_version"] == "default/1" and (
            state["input_token_limit"] != 8000 or state["output_token_limit"] != 2000
        ):
            raise ValueError("report run policy overrides require nondefault configuration_version")
    elif cls is ReportRunAudit:
        projection = state["projection"]
        if state["opaque_case_id"] != projection["foundation"]["opaque_case_id"]:
            raise ValueError("report audit case identity differs from its foundation")
        for raw_name, digest_name in (
            ("prompt_payload_hex", "prompt_sha256"),
            ("request_hex", "request_sha256"),
            ("raw_response_hex", "response_sha256"),
        ):
            if (state[raw_name] is None) != (state[digest_name] is None):
                raise ValueError("report audit byte/hash pairs must agree")
        request = state["request_hex"]
        if request is not None:
            if state["prompt_payload_hex"] is None or state["estimated_input_tokens"] != (
                len(request) // 2 + 256
            ):
                raise ValueError("report audit request estimate or prompt missing")
        elif state["estimated_input_tokens"] is not None or state["exchange_attempted"]:
            raise ValueError("report audit unprepared request cannot be estimated or attempted")
        if state["exchange_attempted"] != (state["exchange_status"] != "not_attempted"):
            raise ValueError("report audit attempt status mismatch")
        if (state["exchange_failure_code"] is not None) != (state["exchange_status"] == "failed"):
            raise ValueError("report audit exchange failure mismatch")
        raw = state["raw_response_hex"]
        if raw is not None and (
            not state["exchange_attempted"] or state["received_bytes"] != len(raw) // 2
        ):
            raise ValueError("report audit wire length mismatch")
        if state["usage_status"] == "reported":
            if state["actual_input_tokens"] is None or state["actual_output_tokens"] is None:
                raise ValueError("reported usage requires actual counts")
        elif state["actual_input_tokens"] is not None or state["actual_output_tokens"] is not None:
            raise ValueError("unreported usage must not invent counts")
        response, mechanical, presentation = (
            state["parsed_response"],
            state["mechanical_validation"],
            state["presentation"],
        )
        if (response is not None) != (state["structural_status"] == "valid"):
            raise ValueError("report audit domain parse status mismatch")
        if (response is None) != (state["parsed_response_sha256"] is None):
            raise ValueError("report audit parsed response hash missing")
        if (presentation is None) != (mechanical is None):
            raise ValueError("report audit presentation and mechanics must coexist")
        if mechanical is not None:
            if response is None or state["structural_status"] != "valid":
                raise ValueError("report audit mechanics requires a valid retained response")
            if mechanical["response_sha256"] != state["parsed_response_sha256"]:
                raise ValueError("report audit mechanical response hash mismatch")
            checked, proposed = mechanical["claims"], response["claims"]
            if len(checked) != len(proposed) or any(
                not _same_document(item["proposal"], claim)
                for item, claim in zip(checked, proposed, strict=True)
            ):
                raise ValueError("report audit mechanical claims differ from retained response")
        if presentation is not None and (
            mechanical is None
            or state["status"] != presentation["status"]
            or not _same_document(presentation["foundation"], projection["foundation"])
            or not _same_document(presentation["mechanical_validation"], mechanical)
        ):
            raise ValueError("report audit presentation closure mismatch")
        if presentation is None and state["status"] != "deterministic_fallback":
            raise ValueError("report audit delivery requires a presentation")


def _guard_report_runtime(
    *,
    roots: tuple[tuple[str, type[BaseModel]], ...],
    pins: tuple[str, ...],
    local_domains: Callable[[type[BaseModel], dict[str, Any]], None] = _runtime_local_domains,
    root_preflights: tuple[tuple[str, Callable[[Any], object]], ...] = (),
) -> Callable[[Callable[_Parameters, _Result]], Callable[_Parameters, _Result]]:
    """Keep every caller-owned root protected from the first encoder to the last callback."""

    def decorate(function: Callable[_Parameters, _Result]) -> Callable[_Parameters, _Result]:
        call_signature = signature(function)

        @wraps(function)
        def guarded(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            values = call_signature.bind(*args, **kwargs).arguments
            sources = values["sources"]
            additional = tuple((values[name], cls) for name, cls in roots)
            external_pins = tuple(values[name] for name in pins)
            try:
                _preflight_inputs(sources, additional, external_pins)
            except (AttributeError, KeyError, RecursionError, TypeError) as error:
                raise ValueError("report runtime input preflight failed") from error

            def snapshot(name: str, value: object, cls: type[BaseModel]) -> object:
                # Retain each report root's original narrower capacity contract.
                if cls is ReportRunAudit:
                    _preflight_audit(cast(ReportRunAudit, value))
                elif cls is ReportRunPolicy:
                    preflight_exact_pydantic_tree(
                        value, root_model_types=(ReportRunPolicy,), label="report run policy"
                    )
                for root_name, preflight in root_preflights:
                    if root_name == name:
                        preflight(value)
                return _report_input_snapshot(
                    value, model_type=cls, max_nodes=500_000, local_domains=local_domains
                )

            originals = tuple(snapshot(name, values[name], cls) for name, cls in roots)
            # This complete paired source gate is pure and precedes all callbacks.
            # It also preserves the ReportSources slots and all independent source pins.
            with _preserve_sources(sources, pins=external_pins):
                try:
                    return function(*args, **kwargs)
                finally:
                    unchanged = True
                    failure: Exception | None = None
                    for (name, cls), original in zip(roots, originals, strict=True):
                        try:
                            if snapshot(name, values[name], cls) != original:
                                unchanged = False
                        except _INPUT_ERRORS as error:
                            unchanged = False
                            if failure is None:
                                failure = error
                    if not unchanged:
                        changed_error = ValueError(
                            "report runtime source bytes changed or original "
                            "policy/audit/artifact representation changed during replay"
                        )
                        if failure is not None:
                            raise changed_error from failure
                        raise changed_error

        return guarded

    return decorate


@_guard_report_runtime(roots=(("policy", ReportRunPolicy),), pins=("expected_policy_sha256",))
def _prepare(
    sources: ReportSources,
    policy: ReportRunPolicy,
    expected_policy_sha256: str,
    source_bytes: bytes,
) -> tuple[dict[str, Any], PreparedExchange | None]:
    if not hmac.compare_digest(policy.sha256, expected_policy_sha256):
        raise ValueError("report run policy differs from independent pin")
    projection = build_report_evidence_projection(**_source_kwargs(sources))
    fields: dict[str, Any] = dict(
        opaque_case_id=projection.foundation.opaque_case_id,
        sources_sha256=sha256_hex(source_bytes),
        expected_audit_sha256=sources.expected_audit_sha256,
        expected_context_sha256=sources.expected_context_sha256,
        expected_identity_manifest_sha256=sources.expected_identity_manifest_sha256,
        expected_evidence_policy_sha256=sources.expected_policy_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        projection=projection,
    )
    if projection.view is None:
        fields["reason_code"] = "report_evidence_unavailable"
        return fields, None
    try:
        prompt = _prompt_from_projection(sources, projection)
        fields.update(prompt_payload_hex=prompt.payload_json.hex(), prompt_sha256=prompt.sha256)
        request = prepare_exchange(
            model=policy.model,
            messages=prompt.messages(),
            max_output_tokens=policy.output_token_limit,
            temperature=policy.temperature,
        )
    except ExchangeBoundaryError as exc:
        if exc.code != "json_limit":
            raise
        fields["reason_code"] = "request_limit"
        return fields, None
    estimate = len(request.payload_bytes) + 256
    fields.update(
        request_hex=request.payload_bytes.hex(),
        request_sha256=request.sha256,
        estimated_input_tokens=estimate,
    )
    if estimate > policy.input_token_limit:
        fields["reason_code"] = "input_budget_exceeded"
    return fields, request


def _usage_fields(raw: bytes | None, request: PreparedExchange) -> dict[str, Any]:
    """Read bounded cost metadata even when a refusal/truncation invalidates output."""
    if raw is None:
        return {}
    try:
        data = decode_json_object(raw, request.limits)
    except ExchangeBoundaryError:
        return {}
    usage = data.get("usage")
    if usage is None:
        return {"usage_status": "missing"}
    fields: dict[str, Any] = {
        "usage_status": "reported" if _valid_usage(usage) else "invalid",
        "raw_usage_json": encode_json(usage, request.limits).decode("utf-8"),
    }
    if fields["usage_status"] == "reported":
        # Prompt total includes cache tokens: do not mistake uncached tokens for input cost.
        fields.update(
            actual_input_tokens=usage["prompt_tokens"],
            actual_output_tokens=usage["completion_tokens"],
        )
    return fields


@_guard_report_runtime(roots=(("policy", ReportRunPolicy),), pins=())
def _apply_attempt(
    fields: dict[str, Any],
    attempt: ExchangeAttempt,
    request: PreparedExchange,
    sources: ReportSources,
    policy: ReportRunPolicy,
) -> None:
    fields.update(
        exchange_attempted=True,
        exchange_status=attempt.status,
        exchange_failure_code=attempt.failure_code,
        raw_response_hex=attempt.raw_response.hex() if attempt.raw_response is not None else None,
        response_sha256=attempt.response_sha256,
        received_bytes=attempt.received_bytes,
    )
    fields.update(_usage_fields(attempt.raw_response, request))
    if attempt.response is None:
        fields["reason_code"] = attempt.failure_code
        return
    try:
        response = parse_report_composition_response(attempt.response.output_json)
    except ExchangeBoundaryError as exc:
        fields.update(structural_status="invalid", reason_code=exc.code)
        return
    response_digest = sha256_hex(_bytes(response))
    fields.update(
        structural_status="valid", parsed_response=response, parsed_response_sha256=response_digest
    )
    if fields.get("usage_status") == "reported":
        for count, limit, reason in (
            (
                fields["actual_input_tokens"],
                policy.input_token_limit,
                "reported_input_budget_exceeded",
            ),
            (
                fields["actual_output_tokens"],
                policy.output_token_limit,
                "reported_output_budget_exceeded",
            ),
        ):
            if count > limit:
                fields["reason_code"] = reason
                return
    arguments = dict(
        response=response,
        expected_response_sha256=response_digest,
        projection=fields["projection"],
        **_source_kwargs(sources),
    )
    mechanical = validate_report_composition_mechanics(**arguments)
    presentation = build_report_composition_presentation(validation=mechanical, **arguments)
    fields.update(
        mechanical_validation=mechanical,
        presentation=presentation,
        status=presentation.status,
        reason_code={
            "full": None,
            "partial": "partially_blocked",
            "deterministic_fallback": "no_deliverable_claims",
        }[presentation.status],
    )


def _seal(fields: dict[str, Any]) -> ReportRunAudit:
    # Validators allocate retained bytes and serialize nested models. Bound the
    # entire exact provisional tree first; this unvalidated object never escapes.
    provisional = ReportRunAudit.model_construct(**fields, audit_sha256="0" * 64)
    measured = _preflight_audit(provisional)
    audit = ReportRunAudit(**fields, audit_sha256="0" * 64)
    if len(canonical_json_bytes(audit.model_dump(mode="json", warnings="error"))) != measured:
        raise ValueError("report audit canonical measurement mismatch")
    digest = sha256_hex(
        canonical_json_bytes(
            audit.model_dump(mode="json", warnings="error", exclude={"audit_sha256"})
        )
    )
    return audit.model_copy(update={"audit_sha256": digest})


@_guard_report_runtime(roots=(("policy", ReportRunPolicy),), pins=("pin",))
def _check_unchanged(
    sources: ReportSources,
    policy: ReportRunPolicy,
    pin: str,
    source_bytes: bytes,
    policy_bytes: bytes,
) -> None:
    _preflight_inputs(sources, ((policy, ReportRunPolicy),), (pin,))
    if not hmac.compare_digest(source_bytes, _sources_bytes(sources)) or not hmac.compare_digest(
        policy_bytes, _bytes(policy)
    ):
        raise ValueError("report run original source or policy bytes changed")


@_guard_report_runtime(roots=(("policy", ReportRunPolicy),), pins=("expected_policy_sha256",))
def run_report_composition(
    *,
    sources: ReportSources,
    policy: ReportRunPolicy,
    expected_policy_sha256: str,
    complete: Callable[[bytes], bytes],
) -> ReportRunAudit:
    """Validate every source before invoking transport at most once; never repair output."""
    _preflight_inputs(sources, ((policy, ReportRunPolicy),), (expected_policy_sha256,))
    preflight_exact_pydantic_tree(
        policy, root_model_types=(ReportRunPolicy,), label="report run policy"
    )
    source_bytes, policy_bytes = _sources_bytes(sources), _bytes(policy)
    try:
        snapshot, policy_snapshot = _snapshot_sources(sources), _snapshot_model(policy)
        fields, request = _prepare(snapshot, policy_snapshot, expected_policy_sha256, source_bytes)
        if request is not None and fields.get("reason_code") is None:
            attempt = execute_once(request, complete)
            _apply_attempt(fields, attempt, request, snapshot, policy_snapshot)
        return _seal(fields)
    finally:
        _check_unchanged(sources, policy, expected_policy_sha256, source_bytes, policy_bytes)


def _offline_attempt(audit: ReportRunAudit, request: PreparedExchange) -> ExchangeAttempt:
    if not audit.exchange_attempted:
        raise ValueError("report replay requires the pinned original attempt")
    if audit.raw_response_hex is None:
        code, count = audit.exchange_failure_code, audit.received_bytes
        allowed = (code in ("timeout", "transport_error", "response_type") and count is None) or (
            code == "response_limit" and type(count) is int and count > request.limits.max_bytes
        )
        if not allowed:
            raise ValueError("report replay has impossible absent-wire transport facts")
        # This is a pinned attestation, not evidence that the transport event occurred.
        return ExchangeAttempt("failed", request.sha256, True, code, None, None, None, count)
    raw = bytes.fromhex(audit.raw_response_hex)
    try:
        response = parse_response(raw, output_tool_name=None, limits=request.limits)
    except ExchangeBoundaryError as exc:
        return ExchangeAttempt(
            "failed", request.sha256, True, exc.code, None, raw, sha256_hex(raw), len(raw)
        )
    return ExchangeAttempt(
        "response_ready", request.sha256, True, None, response, raw, sha256_hex(raw), len(raw)
    )


@_guard_report_runtime(
    roots=(("policy", ReportRunPolicy), ("audit", ReportRunAudit)),
    pins=("expected_policy_sha256", "expected_audit_sha256"),
)
def replay_report_composition(
    *,
    audit: ReportRunAudit,
    expected_audit_sha256: str,
    sources: ReportSources,
    policy: ReportRunPolicy,
    expected_policy_sha256: str,
) -> ReportRunAudit:
    """Rebuild a complete audit from independent pins and retained bytes, without callbacks."""
    _preflight_inputs(
        sources,
        ((policy, ReportRunPolicy), (audit, ReportRunAudit)),
        (expected_policy_sha256, expected_audit_sha256),
    )
    # Bound new wire fields before serialization, hashing or hex decoding allocates copies.
    _preflight_audit(audit)
    preflight_exact_pydantic_tree(
        policy, root_model_types=(ReportRunPolicy,), label="report run policy"
    )
    if not hmac.compare_digest(audit.audit_sha256, expected_audit_sha256) or not audit.verify():
        raise ValueError("report audit fails exact integrity or independent audit pin")
    audit_bytes = _bytes(audit)
    original = _snapshot_model(audit)
    source_bytes, policy_bytes = _sources_bytes(sources), _bytes(policy)
    try:
        snapshot, policy_snapshot = _snapshot_sources(sources), _snapshot_model(policy)
        fields, request = _prepare(snapshot, policy_snapshot, expected_policy_sha256, source_bytes)
        if request is not None and fields.get("reason_code") is None:
            attempt = _offline_attempt(original, request)
            _apply_attempt(fields, attempt, request, snapshot, policy_snapshot)
        rebuilt = _seal(fields)
        if not hmac.compare_digest(audit_bytes, _bytes(rebuilt)):
            raise ValueError("report audit differs from independently rebuilt replay")
        return rebuilt
    finally:
        _check_unchanged(sources, policy, expected_policy_sha256, source_bytes, policy_bytes)
        _preflight_audit(audit)
        if not hmac.compare_digest(audit_bytes, _bytes(audit)):
            raise ValueError("report original audit bytes changed during replay")
