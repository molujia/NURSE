"""Explicit disabled-thinking report protocol, not a legacy schema relaxation.

PreparedExchange is only the detached byte/limit carrier. The legacy executor
still rejects these bytes. Source authority belongs to the live report producer.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nurse_rcl.agents.ark import ARK_MODEL
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
    require_limits,
)
from nurse_rcl.v2.hermes_exchange import (
    ExchangeAttempt,
    PreparedExchange,
    _validate_payload,
    prepare_exchange,
)
from nurse_rcl.v2.hermes_response import ParsedExchangeResponse, parse_response
from nurse_rcl.v2.live.normal_b2_report_prompt import (
    B2_REPORT_EXCHANGE_VERSION,
    B2_REPORT_INSTRUCTIONS,
    B2_REPORT_PROMPT_VERSION,
    NormalB2ReportPrompt,
)
from nurse_rcl.v2.live.repair_report_prompt import (
    REPAIR_EXCHANGE_VERSION,
    REPAIR_INSTRUCTIONS,
    REPAIR_PROMPT_VERSION,
    RepairReportPrompt,
)
from nurse_rcl.v2.live.report_guidance import (
    GUIDED_INSTRUCTIONS,
    GUIDED_PROMPT_VERSION,
    LiveGuidedPrompt,
)
from nurse_rcl.v2.live.report_index import (
    INDEXED_INSTRUCTIONS,
    INDEXED_PROMPT_VERSION,
    LiveIndexedPrompt,
)
from nurse_rcl.v2.live.report_prompt import INSTRUCTIONS, VERSION, LiveCompactPrompt
from nurse_rcl.v2.report_composition_proposals import (
    parse_report_composition_response,
    report_composition_output_schema,
)

REPORT_EXCHANGE_VERSION = 'nurse.v2.live-report-disabled-exchange/1'
REPORT_CONFIGURATION_VERSION = 'live-bootstrap-report/3'
GUIDED_REPORT_EXCHANGE_VERSION = 'nurse.v2.live-report-disabled-exchange/2'
GUIDED_REPORT_CONFIGURATION_VERSION = 'live-bootstrap-report/4'
INDEXED_REPORT_EXCHANGE_VERSION = 'nurse.v2.live-report-disabled-exchange/3'
INDEXED_REPORT_CONFIGURATION_VERSION = 'live-bootstrap-report/5'
REPORT_PROTOCOLS = (
    REPORT_EXCHANGE_VERSION, GUIDED_REPORT_EXCHANGE_VERSION, INDEXED_REPORT_EXCHANGE_VERSION,
    REPAIR_EXCHANGE_VERSION, B2_REPORT_EXCHANGE_VERSION,
)


def _parse_protocol_response(raw: bytes, *, output_tool_name: str | None,
                             limits: ExchangeLimits, protocol: str) -> ParsedExchangeResponse:
    parsed = parse_response(raw, output_tool_name=output_tool_name, limits=limits)
    if protocol in (
        INDEXED_REPORT_EXCHANGE_VERSION, REPAIR_EXCHANGE_VERSION, B2_REPORT_EXCHANGE_VERSION,
    ):
        # Structure only: no evidence/semantic acceptance and no output repair.
        parse_report_composition_response(parsed.output_json)
    return parsed


def validate_report_payload(
    payload: dict[str, Any], *, protocol: str = REPORT_EXCHANGE_VERSION,
) -> None:
    """Validate only protocol/shape; this does not confer source authority."""
    if type(protocol) is not str or protocol not in REPORT_PROTOCOLS:
        raise ExchangeBoundaryError('invalid_request')
    if payload.get('thinking') != {'type': 'disabled'}:
        raise ExchangeBoundaryError('invalid_request')
    base = {key: value for key, value in payload.items() if key != 'thinking'}
    if (_validate_payload(base) is not None or base['model'] != ARK_MODEL
            or base['max_tokens'] != 4096 or base['temperature'] != 0):
        raise ExchangeBoundaryError('invalid_request')
    messages = base['messages']
    if len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
        raise ExchangeBoundaryError('invalid_request')
    data = decode_json_object(messages[1]['content'].encode(), ExchangeLimits())
    contracts: dict[str, tuple[
        type[LiveCompactPrompt] | type[LiveGuidedPrompt] | type[LiveIndexedPrompt]
        | type[RepairReportPrompt] | type[NormalB2ReportPrompt], str, str,
    ]] = {
        REPORT_EXCHANGE_VERSION: (LiveCompactPrompt, VERSION, INSTRUCTIONS),
        GUIDED_REPORT_EXCHANGE_VERSION: (
            LiveGuidedPrompt, GUIDED_PROMPT_VERSION, GUIDED_INSTRUCTIONS,
        ),
        INDEXED_REPORT_EXCHANGE_VERSION: (
            LiveIndexedPrompt, INDEXED_PROMPT_VERSION, INDEXED_INSTRUCTIONS,
        ),
        REPAIR_EXCHANGE_VERSION: (RepairReportPrompt, REPAIR_PROMPT_VERSION, REPAIR_INSTRUCTIONS),
        B2_REPORT_EXCHANGE_VERSION: (
            NormalB2ReportPrompt, B2_REPORT_PROMPT_VERSION, B2_REPORT_INSTRUCTIONS,
        ),
    }
    prompt_type, version, instructions = contracts[protocol]
    prompt = prompt_type(encode_json(dict(prompt_version=version,
        instructions=instructions,
        output_schema_json=canonical_json_bytes(report_composition_output_schema()).decode(),
        data=data), ExchangeLimits()))
    if prompt.messages() != messages:
        raise ExchangeBoundaryError('invalid_request')


def prepare_report_exchange(
    *, messages: object, protocol: str = REPORT_EXCHANGE_VERSION,
) -> PreparedExchange:
    """Use embedded HERMES builder, then explicitly version the canonical payload."""
    base = prepare_exchange(model=ARK_MODEL, messages=messages, max_output_tokens=4096,
                            temperature=0.0)
    payload = decode_json_object(base.payload_bytes, base.limits)
    payload['thinking'] = {'type': 'disabled'}
    validate_report_payload(payload, protocol=protocol)
    return PreparedExchange(encode_json(payload, base.limits), base.limits)


def execute_report_once(
    request: PreparedExchange, complete: Callable[[bytes], bytes],
    *, protocol: str = REPORT_EXCHANGE_VERSION,
) -> ExchangeAttempt:
    """One exchange, same response parser; durable retries remain in the bridge."""
    if type(request) is not PreparedExchange:
        raise ExchangeBoundaryError('invalid_request')
    if not callable(complete):
        raise ExchangeBoundaryError('invalid_transport')
    require_limits(request.limits)
    limits = ExchangeLimits(
        request.limits.max_bytes, request.limits.max_depth, request.limits.max_nodes,
    )
    raw_request = request.payload_bytes
    payload = decode_json_object(raw_request, limits)
    validate_report_payload(payload, protocol=protocol)
    if encode_json(payload, limits) != raw_request:
        raise ExchangeBoundaryError('noncanonical_request')
    request_hash = sha256_hex(raw_request)

    def failed(code: str, raw: bytes | None = None, received: int | None = None) -> ExchangeAttempt:
        return ExchangeAttempt('failed', request_hash, True, code, None, raw,
                               sha256_hex(raw) if raw is not None else None, received)

    try:
        raw = complete(raw_request)
    except TimeoutError:
        return failed('timeout')
    except Exception:
        return failed('transport_error')
    if type(raw) is not bytes:
        return failed('response_type')
    if len(raw) > limits.max_bytes:
        return failed('response_limit', received=len(raw))
    try:
        parsed = _parse_protocol_response(
            raw, output_tool_name=None, limits=limits, protocol=protocol,
        )
    except ExchangeBoundaryError as error:
        return failed(error.code, raw, len(raw))
    return ExchangeAttempt('response_ready', request_hash, True, None, parsed, raw,
                           sha256_hex(raw), len(raw))
