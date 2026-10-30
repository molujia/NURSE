"""Durable bytes→fixed ARK bridge for trusted, source-validated live runners.

The runner supplies the independently pinned request after domain/source and
outbound-projection validation. This layer is NOT a source-authority factory:
it enforces exact request identity, approved cases, fixed provider, wire shape,
one in-flight call, and lifetime attempts. Domain acceptance remains downstream.
"""

from __future__ import annotations

import fcntl
import hmac
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.agents.ark import (
    ARK_MODEL,
    ArkChatCompletionsClient,
    ArkResponseResourceError,
    ArkSettings,
)
from nurse_rcl.agents.retry import (
    AttemptRecord,
    AttemptResult,
    AttemptUsage,
    CircuitBreaker,
    FatalProviderRequestError,
    RequestExecution,
    RetryableParsingError,
    RetryExhaustedError,
    execute_llm_request,
)
from nurse_rcl.artifacts import (
    ArtifactEnvelope,
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_hex,
)
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.hermes_exchange import ADAPTER_VERSION, _validate_payload
from nurse_rcl.v2.hermes_response import ParsedExchangeResponse, _valid_usage, parse_response
from nurse_rcl.v2.live.normal_b2_report_prompt import B2_REPORT_EXCHANGE_VERSION
from nurse_rcl.v2.live.ops_empirical_policy import (
    DECISION_PROTOCOL,
    LEDGER_BYTES,
    REPORT_PROTOCOL,
    SRE_ACTION_DECISION_PROTOCOL,
    SRE_ACTION_REPORT_PROTOCOL,
    SRE_ATOMIC_REPORT_PROTOCOL,
    SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
    SRE_BALANCED_DECISION_PROTOCOL,
    SRE_BALANCED_REPORT_PROTOCOL,
    SRE_BRIEF_DECISION_PROTOCOL,
    SRE_BRIEF_REPORT_PROTOCOL,
    SRE_COMPARABLE_DECISION_PROTOCOL,
    SRE_COMPARABLE_REPORT_PROTOCOL,
    SRE_CONDITION_REPORT_PROTOCOL,
    SRE_DECISION_PROTOCOLS,
    SRE_DIRECTED_DECISION_PROTOCOL,
    SRE_ENTRY_DECISION_PROTOCOL,
    SRE_FEEDBACK_DECISION_PROTOCOL,
    SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
    SRE_FOCUSED_DECISION_PROTOCOL,
    SRE_GPLUS_REPORT_PROTOCOL,
    SRE_LEGACY_ATOMIC_REPORT_PROTOCOL,
    SRE_NATIVE_REPORT_PROTOCOL,
    SRE_NAVIGATION_DECISION_PROTOCOL,
    SRE_NAVIGATION_REPORT_PROTOCOL,
    SRE_OBSERVABLE_DECISION_PROTOCOL,
    SRE_PRIOR_ATOMIC_REPORT_PROTOCOL,
    SRE_PROTOCOLS,
    SRE_QUESTION_DECISION_PROTOCOL,
    SRE_QUESTION_REPORT_PROTOCOL,
    SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
    SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
    SRE_RECOVERY_DECISION_PROTOCOL,
    SRE_RECOVERY_REPORT_PROTOCOL,
    SRE_REPLANNING_DECISION_PROTOCOL,
    SRE_REPLANNING_REPORT_PROTOCOL,
    SRE_REPORT_PROTOCOL,
    SRE_REPORT_PROTOCOLS,
    SRE_SINGLE_OWNER_DECISION_PROTOCOL,
    SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
    SRE_TYPED_REPORT_PROTOCOL,
    SRE_USED_SOURCE_DECISION_PROTOCOL,
    SRE_USED_SOURCE_REPORT_PROTOCOL,
    SRE_WORKBENCH_DECISION_PROTOCOL,
    SRE_WORKBENCH_REPORT_PROTOCOL,
    SRE_WORKING_DECISION_PROTOCOL,
    USAGE_POLICY,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    PROTOCOLS as EMPIRICAL_PROTOCOLS,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    RESPONSE_LIMITS as EMPIRICAL_RESPONSE_LIMITS,
)
from nurse_rcl.v2.live.repair_module_timing import _module_span
from nurse_rcl.v2.live.repair_report_prompt import REPAIR_EXCHANGE_VERSION
from nurse_rcl.v2.live.report_exchange import (
    REPORT_PROTOCOLS,
    validate_report_payload,
)
from nurse_rcl.v2.live.report_exchange import (
    _parse_protocol_response as _parse_report_protocol_response,
)

if TYPE_CHECKING:
    from nurse_rcl.v2.live.ops_sre_turn_validation import BoundWorkbenchTurn


def _parse_protocol_response(
    raw: bytes,
    *,
    output_tool_name: str | None,
    limits: ExchangeLimits,
    protocol: str,
    workbench_turn: BoundWorkbenchTurn | None = None,
) -> ParsedExchangeResponse:
    if protocol in (SRE_TYPED_REPORT_PROTOCOL, SRE_GPLUS_REPORT_PROTOCOL):
        from nurse_rcl.v2.live.ops_sre_typed_report_provider import parse_typed_receipt

        if output_tool_name is not None:
            raise ValueError("typed report has no tool response channel")
        return parse_typed_receipt(raw)

    from nurse_rcl.v2.live.ops_agent_decision import parse_decision
    from nurse_rcl.v2.live.ops_exchange import OPS_DECISION_EXCHANGE_VERSION
    from nurse_rcl.v2.live.ops_report_exchange import OPS_REPORT_EXCHANGE_VERSION
    from nurse_rcl.v2.live.ops_report_schema import parse_ops_report

    if protocol in {
        SRE_WORKBENCH_DECISION_PROTOCOL,
        SRE_USED_SOURCE_DECISION_PROTOCOL,
        SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
        SRE_BALANCED_DECISION_PROTOCOL,
        SRE_REPLANNING_DECISION_PROTOCOL,
        SRE_QUESTION_DECISION_PROTOCOL,
        SRE_ACTION_DECISION_PROTOCOL,
        SRE_DIRECTED_DECISION_PROTOCOL,
        SRE_ENTRY_DECISION_PROTOCOL,
        SRE_OBSERVABLE_DECISION_PROTOCOL,
        SRE_SINGLE_OWNER_DECISION_PROTOCOL,
        SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
        SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
        SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
        SRE_FOCUSED_DECISION_PROTOCOL,
    }:
        from nurse_rcl.v2.live.ops_sre_turn_validation import validate_bound_workbench_response

        if workbench_turn is None:
            raise ValueError("workbench response requires frozen request binding")
        return validate_bound_workbench_response(raw, workbench_turn)

    if protocol in SRE_PROTOCOLS:
        from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision
        from nurse_rcl.v2.live.ops_sre_report import parse_sre_report

        parsed = parse_response(raw, output_tool_name=None, limits=limits)
        if protocol == SRE_LEGACY_ATOMIC_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
                parse_legacy_expression_response,
            )

            parse_legacy_expression_response(parsed.output_json)
            return parsed
        if protocol == SRE_PRIOR_ATOMIC_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
                parse_prior_expression_response,
            )

            parse_prior_expression_response(parsed.output_json)
            return parsed
        if protocol == SRE_ATOMIC_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
                parse_expression_response,
            )

            parse_expression_response(parsed.output_json)
            return parsed
        if protocol in {
            SRE_BRIEF_DECISION_PROTOCOL,
            SRE_FEEDBACK_DECISION_PROTOCOL,
            SRE_RECOVERY_DECISION_PROTOCOL,
            SRE_COMPARABLE_DECISION_PROTOCOL,
            SRE_NAVIGATION_DECISION_PROTOCOL,
            SRE_WORKBENCH_DECISION_PROTOCOL,
            SRE_USED_SOURCE_DECISION_PROTOCOL,
            SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
            SRE_BALANCED_DECISION_PROTOCOL,
            SRE_REPLANNING_DECISION_PROTOCOL,
            SRE_QUESTION_DECISION_PROTOCOL,
            SRE_ACTION_DECISION_PROTOCOL,
            SRE_DIRECTED_DECISION_PROTOCOL,
        }:
            from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision
            from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

            (
                parse_feedback_decision
                if protocol
                in {
                    SRE_FEEDBACK_DECISION_PROTOCOL,
                    SRE_RECOVERY_DECISION_PROTOCOL,
                    SRE_COMPARABLE_DECISION_PROTOCOL,
                    SRE_NAVIGATION_DECISION_PROTOCOL,
                    SRE_WORKBENCH_DECISION_PROTOCOL,
                    SRE_USED_SOURCE_DECISION_PROTOCOL,
                    SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
                    SRE_BALANCED_DECISION_PROTOCOL,
                    SRE_REPLANNING_DECISION_PROTOCOL,
                    SRE_QUESTION_DECISION_PROTOCOL,
                    SRE_ACTION_DECISION_PROTOCOL,
                    SRE_DIRECTED_DECISION_PROTOCOL,
                }
                else parse_brief_decision
            )(parsed.output_json)
        elif protocol in {
            SRE_ATOMIC_REPORT_PROTOCOL,
            SRE_BRIEF_REPORT_PROTOCOL,
            SRE_CONDITION_REPORT_PROTOCOL,
            SRE_RECOVERY_REPORT_PROTOCOL,
            SRE_COMPARABLE_REPORT_PROTOCOL,
            SRE_NAVIGATION_REPORT_PROTOCOL,
            SRE_WORKBENCH_REPORT_PROTOCOL,
            SRE_USED_SOURCE_REPORT_PROTOCOL,
            SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
            SRE_BALANCED_REPORT_PROTOCOL,
            SRE_REPLANNING_REPORT_PROTOCOL,
            SRE_QUESTION_REPORT_PROTOCOL,
            SRE_ACTION_REPORT_PROTOCOL,
        }:
            from nurse_rcl.v2.live.ops_sre_brief_claims import parse_brief_report

            parse_brief_report(
                parsed.output_json,
                comparable_evidence=protocol
                in {
                    SRE_ATOMIC_REPORT_PROTOCOL,
                    SRE_COMPARABLE_REPORT_PROTOCOL,
                    SRE_NAVIGATION_REPORT_PROTOCOL,
                    SRE_WORKBENCH_REPORT_PROTOCOL,
                    SRE_USED_SOURCE_REPORT_PROTOCOL,
                    SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
                    SRE_BALANCED_REPORT_PROTOCOL,
                    SRE_REPLANNING_REPORT_PROTOCOL,
                    SRE_QUESTION_REPORT_PROTOCOL,
                    SRE_ACTION_REPORT_PROTOCOL,
                },
                condition_explanations=protocol
                in {
                    SRE_ATOMIC_REPORT_PROTOCOL,
                    SRE_CONDITION_REPORT_PROTOCOL,
                    SRE_RECOVERY_REPORT_PROTOCOL,
                    SRE_COMPARABLE_REPORT_PROTOCOL,
                    SRE_NAVIGATION_REPORT_PROTOCOL,
                    SRE_WORKBENCH_REPORT_PROTOCOL,
                    SRE_USED_SOURCE_REPORT_PROTOCOL,
                    SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
                    SRE_BALANCED_REPORT_PROTOCOL,
                    SRE_REPLANNING_REPORT_PROTOCOL,
                    SRE_QUESTION_REPORT_PROTOCOL,
                    SRE_ACTION_REPORT_PROTOCOL,
                },
            )
        elif protocol in SRE_DECISION_PROTOCOLS:
            parse_sre_decision(parsed.output_json)
        else:
            parse_sre_report(parsed.output_json)
        return parsed
    if protocol in (OPS_DECISION_EXCHANGE_VERSION, DECISION_PROTOCOL):
        parsed = parse_response(raw, output_tool_name=None, limits=limits)
        parse_decision(parsed.output_json, empirical=protocol == DECISION_PROTOCOL)
        return parsed
    if protocol in (OPS_REPORT_EXCHANGE_VERSION, REPORT_PROTOCOL):
        parsed = parse_response(raw, output_tool_name=None, limits=limits)
        parse_ops_report(parsed.output_json, empirical=protocol == REPORT_PROTOCOL)
        return parsed
    # Driver/query owners import this bridge; load the optional normal semantic
    # adapter only at a transport call, not while initializing that owner graph.
    from nurse_rcl.v2.live.normal_semantic import parse_normal_semantic_response
    from nurse_rcl.v2.live.normal_semantic_compact_exchange import COMPACT_SEMANTIC_EXCHANGE_VERSION
    from nurse_rcl.v2.live.normal_semantic_disabled_exchange import (
        DISABLED_SEMANTIC_EXCHANGE_VERSION,
    )
    from nurse_rcl.v2.live.normal_semantic_exchange import SEMANTIC_EXCHANGE_VERSION
    from nurse_rcl.v2.live.normal_semantic_followup import HYBRID_SEMANTIC_EXCHANGE_VERSION
    from nurse_rcl.v2.live.normal_semantic_indexed_exchange import INDEXED_SEMANTIC_EXCHANGE_VERSION

    if protocol in (
        SEMANTIC_EXCHANGE_VERSION,
        COMPACT_SEMANTIC_EXCHANGE_VERSION,
        HYBRID_SEMANTIC_EXCHANGE_VERSION,
        DISABLED_SEMANTIC_EXCHANGE_VERSION,
        INDEXED_SEMANTIC_EXCHANGE_VERSION,
    ):
        parsed = parse_response(raw, output_tool_name=output_tool_name, limits=limits)
        parse_normal_semantic_response(parsed.output_json)
        return parsed
    return _parse_report_protocol_response(
        raw,
        output_tool_name=output_tool_name,
        limits=limits,
        protocol=protocol,
    )


class LiveTransportError(RuntimeError):
    """Static local error code; never include provider text or credentials."""


class NativePartialResponseError(LiveTransportError):
    """A saved native stop cannot be converted into an automatic continuation."""


@contextmanager
def live_ark_lock(path: Path) -> Iterator[None]:
    """One lock for sending, offline recovery and consistent usage snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LiveTransportError("llm_in_flight") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class LiveRequestBinding(StrictFrozenModel):
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    variant: Literal[
        "v2-live-bootstrap",
        "v2-live-evidence-repair",
        "B1",
        "B2",
        "v2-ops-isolated/1",
        "v2-ops-isolated/2",
        "v2-ops-sre/1",
        "v2-ops-sre/2",
        "v2-ops-sre/3",
        "v2-ops-sre/4",
        "v2-ops-sre/5",
        "v2-ops-sre/6",
        "v2-ops-sre/7",
        "v2-ops-sre/8",
        "v2-ops-sre/9",
        "v2-ops-sre/10",
        "v2-ops-sre/11",
        "v2-ops-sre/12",
        "v2-ops-sre/13",
        "v2-ops-sre/14",
        "v2-ops-sre/15",
        "v2-ops-sre-report/1",
        "v2-ops-sre-typed-report/1",
        "v2-ops-sre-gplus-report/1",
    ]
    phase: Literal[
        "report",
        "semantic_first",
        "semantic_followup",
        "investigation",
        "investigation_correction",
    ]
    decision_index: int | None = Field(default=None, ge=1, le=8, strict=True)
    correction_index: Literal[1] | None = None
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def legal_phase(self) -> Self:
        if self.variant in {
            "v2-ops-sre-report/1",
            "v2-ops-sre-typed-report/1",
            "v2-ops-sre-gplus-report/1",
        }:
            if (
                self.phase != "report"
                or self.decision_index is not None
                or (self.correction_index is not None)
            ):
                raise ValueError("report-only binding requires report without decision indices")
            return self
        if self.variant in {
            "v2-ops-sre/9",
            "v2-ops-sre/10",
            "v2-ops-sre/11",
            "v2-ops-sre/12",
            "v2-ops-sre/13",
            "v2-ops-sre/14",
            "v2-ops-sre/15",
        } and self.phase not in {
            "investigation",
            "investigation_correction",
        }:
            raise ValueError("entry variant is decision-stage only; report is disabled")
        if self.variant in {
            "v2-ops-sre/2",
            "v2-ops-sre/3",
            "v2-ops-sre/4",
            "v2-ops-sre/5",
            "v2-ops-sre/6",
            "v2-ops-sre/7",
            "v2-ops-sre/8",
            "v2-ops-sre/9",
            "v2-ops-sre/10",
            "v2-ops-sre/11",
            "v2-ops-sre/12",
            "v2-ops-sre/13",
            "v2-ops-sre/14",
            "v2-ops-sre/15",
        }:
            if self.phase == "investigation_correction":
                if self.decision_index is None or self.correction_index != 1:
                    raise ValueError("correction requires original decision and unique index")
            elif self.phase in {"report", "investigation"}:
                if self.correction_index is not None or (
                    (self.phase == "investigation") != (self.decision_index is not None)
                ):
                    raise ValueError("new SRE request phase/index differs")
            else:
                raise ValueError("new SRE request phase differs")
            return self
        if self.correction_index is not None or self.phase == "investigation_correction":
            raise ValueError("correction requires new SRE variant")
        if self.variant in ("v2-ops-isolated/1", "v2-ops-isolated/2", "v2-ops-sre/1"):
            if self.phase not in {"report", "investigation"} or (
                (self.phase == "investigation") != (self.decision_index is not None)
            ):
                raise ValueError("isolated request requires report or indexed investigation")
            return self
        if self.decision_index is not None or self.phase == "investigation":
            raise ValueError("investigation index requires isolated variant")
        if self.variant != "B2" and self.phase != "report":
            raise ValueError("only B2 permits semantic ranking requests")
        return self

    @property
    def slot(self) -> str:
        # Exclude content hashes: changing a context cannot buy a new request
        # in the same case/version/phase. Content drift is rejected in _load.
        parts: list[Any] = [self.opaque_case_id, self.variant, self.phase]
        if (
            self.variant
            in {
                "v2-ops-sre/2",
                "v2-ops-sre/3",
                "v2-ops-sre/4",
                "v2-ops-sre/5",
                "v2-ops-sre/6",
                "v2-ops-sre/7",
                "v2-ops-sre/8",
                "v2-ops-sre/9",
                "v2-ops-sre/10",
                "v2-ops-sre/11",
                "v2-ops-sre/12",
                "v2-ops-sre/13",
                "v2-ops-sre/14",
                "v2-ops-sre/15",
            }
            and self.phase == "investigation_correction"
        ):
            parts.append(self.correction_index)
        elif self.variant in (
            "v2-ops-isolated/1",
            "v2-ops-isolated/2",
            "v2-ops-sre/1",
            "v2-ops-sre/2",
            "v2-ops-sre/3",
            "v2-ops-sre/4",
            "v2-ops-sre/5",
            "v2-ops-sre/6",
            "v2-ops-sre/7",
            "v2-ops-sre/8",
            "v2-ops-sre/9",
            "v2-ops-sre/10",
            "v2-ops-sre/11",
            "v2-ops-sre/12",
            "v2-ops-sre/13",
            "v2-ops-sre/14",
            "v2-ops-sre/15",
        ):
            parts.append(self.decision_index)
        return sha256_hex(canonical_json_bytes(parts))


class _StoredAttempt(StrictFrozenModel):
    number: int = Field(ge=1, le=5)
    record: AttemptRecord | None = None
    response_json: str | None = None
    response_sha256: str | None = None
    interrupted: bool = False
    latency_known: bool = True

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.record is not None and self.record.attempt_number != self.number:
            raise ValueError("attempt record number drift")
        if self.response_json is None:
            if self.response_sha256 is not None:
                raise ValueError("response hash without receipt")
        elif sha256_hex(self.response_json.encode("utf-8")) != self.response_sha256:
            raise ValueError("response receipt hash drift")
        return self


class _Ledger(StrictFrozenModel):
    schema_version: Literal["nurse.v2.live-ark-ledger/1"] = "nurse.v2.live-ark-ledger/1"
    run_identity_sha256: str
    binding: LiveRequestBinding
    attempts: tuple[_StoredAttempt, ...] = Field(default=(), max_length=5)
    response_reuses: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def contiguous(self) -> Self:
        if tuple(a.number for a in self.attempts) != tuple(range(1, len(self.attempts) + 1)):
            raise ValueError("noncontiguous live attempts")
        for item in self.attempts[:-1]:
            if item.record is None or item.record.outcome != "retry":
                raise ValueError("only final attempt may be pending or terminal")
        return self


class _EmpiricalStoredAttempt(_StoredAttempt):
    http_status_code: int | None = Field(default=None, ge=400, le=599, strict=True)

    @model_validator(mode="after")
    def http_receipt(self) -> Self:
        if self.http_status_code is not None and self.response_json is None:
            raise ValueError("HTTP status requires the retained response")
        return self


class _EmpiricalLedger(_Ledger):
    attempts: tuple[_EmpiricalStoredAttempt, ...] = Field(default=(), max_length=5)


def read_attempt_ledger(path: Path, *, empirical: bool) -> _Ledger:
    model = _EmpiricalLedger if empirical else _Ledger
    return model.model_validate(_read(path, "live_ark_request", empirical=empirical))


def _usage(raw: bytes) -> AttemptUsage:
    value = decode_json_object(raw, ExchangeLimits()).get("usage")
    if type(value) is not dict or not _valid_usage(value):
        return AttemptUsage.unavailable()
    cached = (value.get("prompt_tokens_details") or {}).get("cached_tokens")
    counts = (value["prompt_tokens"], value["completion_tokens"], value.get("total_tokens"), cached)
    return AttemptUsage("reported" if all(c is not None for c in counts) else "partial", *counts)


def _empirical_usage(raw: bytes) -> AttemptUsage:
    from nurse_rcl.v2.live.ops_usage import usage_parts

    parts = usage_parts(decode_json_object(raw, EMPIRICAL_RESPONSE_LIMITS).get("usage"))
    values = tuple(
        parts[k] for k in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")
    )
    known = sum(v is not None for v in values)
    return AttemptUsage(
        "reported" if known == 4 else "partial" if known else "unavailable", *values
    )


def _native_partial(raw: bytes) -> bool:
    value = decode_json_object(raw, EMPIRICAL_RESPONSE_LIMITS)
    choices = value.get("choices")
    return bool(
        type(choices) is list
        and len(choices) == 1
        and type(choices[0]) is dict
        and choices[0].get("finish_reason") in ("length", "content_filter")
    )


def recover_empirical_pending(
    ledger: _Ledger,
    *,
    protocol: str,
    workbench_turn: BoundWorkbenchTurn | None = None,
) -> _Ledger:
    """Pure repair from a saved response; caller holds the shared lock and saves."""
    if protocol not in EMPIRICAL_PROTOCOLS:
        raise ValueError("empirical recovery protocol differs")
    if (
        protocol
        in {
            SRE_WORKBENCH_DECISION_PROTOCOL,
            SRE_USED_SOURCE_DECISION_PROTOCOL,
            SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
            SRE_BALANCED_DECISION_PROTOCOL,
            SRE_REPLANNING_DECISION_PROTOCOL,
            SRE_QUESTION_DECISION_PROTOCOL,
            SRE_ACTION_DECISION_PROTOCOL,
            SRE_DIRECTED_DECISION_PROTOCOL,
            SRE_ENTRY_DECISION_PROTOCOL,
            SRE_OBSERVABLE_DECISION_PROTOCOL,
            SRE_SINGLE_OWNER_DECISION_PROTOCOL,
            SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
            SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
            SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
            SRE_FOCUSED_DECISION_PROTOCOL,
        }
        and workbench_turn is None
    ):
        raise ValueError("workbench recovery requires frozen request binding")
    if not ledger.attempts or ledger.attempts[-1].record is not None:
        return ledger
    pending = ledger.attempts[-1]
    if protocol in (SRE_TYPED_REPORT_PROTOCOL, SRE_GPLUS_REPORT_PROTOCOL):
        from nurse_rcl.v2.live.ops_sre_typed_report_provider import interrupted_typed_record

        record = interrupted_typed_record(
            pending.number,
            None if pending.response_json is None else pending.response_json.encode(),
            getattr(pending, "http_status_code", None),
        )
        fixed = pending.model_copy(
            update=dict(record=record, interrupted=True, latency_known=False)
        )
        return ledger.model_copy(update={"attempts": (*ledger.attempts[:-1], fixed)})
    valid = partial = False
    usage = AttemptUsage.unavailable()
    if pending.response_json is not None:
        raw = pending.response_json.encode()
        usage = _empirical_usage(raw)
        partial = _native_partial(raw)
        try:
            _parse_protocol_response(
                raw,
                output_tool_name=None,
                limits=EMPIRICAL_RESPONSE_LIMITS,
                protocol=protocol,
                workbench_turn=workbench_turn,
            )
            valid = True
        except ExchangeBoundaryError:
            pass
    status = getattr(pending, "http_status_code", None)
    if status is not None:
        retryable = status in (408, 429) or status >= 500
        record = AttemptRecord(
            attempt_number=pending.number,
            category="transport" if retryable else "configuration",
            outcome="fatal" if not retryable else "exhausted" if pending.number == 5 else "retry",
            latency_seconds=0.0,
            usage=usage,
            status_code=status,
            error_type="HTTPStatusError",
            delay_seconds=0.0 if retryable and pending.number < 5 else None,
        )
        fixed = pending.model_copy(
            update=dict(record=record, interrupted=True, latency_known=False)
        )
        return ledger.model_copy(update={"attempts": (*ledger.attempts[:-1], fixed)})
    record = AttemptRecord(
        attempt_number=pending.number,
        category="success" if valid else "parse" if pending.response_json else "transport",
        outcome="success"
        if valid
        else "fatal"
        if partial
        else "exhausted"
        if pending.number == 5
        else "retry",
        latency_seconds=0.0,
        usage=usage,
        delay_seconds=0.0 if not valid and not partial and pending.number < 5 else None,
        error_type=None
        if valid
        else "NativePartialResponseError"
        if partial
        else "OutcomeUnknownAfterInterruption"
        if pending.response_json is None
        else "InterruptedMalformedReceipt",
    )
    fixed = pending.model_copy(update=dict(record=record, interrupted=True, latency_known=False))
    return ledger.model_copy(update={"attempts": (*ledger.attempts[:-1], fixed)})


def _write(path: Path, kind: str, value: dict[str, Any]) -> None:
    envelope = ArtifactEnvelope.create(kind=kind, payload=value)
    atomic_write_bytes(path, canonical_json_bytes(envelope.model_dump(mode="json")))
    # The shared writer fsyncs file contents; also persist the directory entry
    # before making a network request, including after an atomic replacement.
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path: Path, kind: str, *, empirical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > (LEDGER_BYTES if empirical else 4_194_304):
        raise ValueError("invalid live ledger file")
    raw = path.read_bytes()
    if empirical:
        from nurse_rcl.v2.exchange_json import _integer, _nonfinite, _unique_object
        from nurse_rcl.v2.report_audit_json import scan_audit_json

        # Reuse only the lexical storage scan, not source replay or an audit.
        # Embedded individual responses are still checked under RESPONSE_LIMITS.
        scan_audit_json(raw)
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_int=_integer,
            parse_constant=_nonfinite,
        )
    else:
        document = decode_json_object(raw, ExchangeLimits(4_194_304, 64, 100_000))
    envelope = ArtifactEnvelope.model_validate(document)
    if envelope.kind != kind or not envelope.verify():
        raise ValueError("corrupt live ledger envelope")
    return envelope.payload


def _require_payload_case(
    request: dict[str, Any],
    opaque_case_id: str,
    *,
    limits: ExchangeLimits | None = None,
) -> None:
    """Bind every user-evidence case reference, including nested Report views.

    This is only an identity gate. The source-backed prompt producer still owns
    schema, label leakage, time and provenance validation before this bridge.
    """
    found = False
    for message in request["messages"]:
        if message["role"] != "user":
            continue
        try:
            data = decode_json_object(
                message["content"].encode("utf-8"),
                limits or ExchangeLimits(),
            )
        except (ExchangeBoundaryError, UnicodeError):
            raise ValueError("live payload case requires bounded JSON evidence") from None
        pending: list[Any] = [data]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                for key, value in item.items():
                    if key == "opaque_case_id":
                        if type(value) is not str or value != opaque_case_id:
                            raise ValueError("live payload case differs from request binding")
                        found = True
                    pending.append(value)
            elif type(item) is list:
                pending.extend(item)
    if not found:
        raise ValueError("live payload case identity missing")


class LiveArkBridge:
    def __init__(
        self,
        *,
        root: Path,
        lock_path: Path,
        settings: ArkSettings,
        code_sha256: str,
        config_sha256: str,
        authorized_case_ids: tuple[str, ...],
        request_protocol: str = ADAPTER_VERSION,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        from nurse_rcl.v2.live.normal_semantic_compact_exchange import (
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
        )
        from nurse_rcl.v2.live.normal_semantic_disabled_exchange import (
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
        )
        from nurse_rcl.v2.live.normal_semantic_exchange import SEMANTIC_EXCHANGE_VERSION
        from nurse_rcl.v2.live.normal_semantic_followup import HYBRID_SEMANTIC_EXCHANGE_VERSION
        from nurse_rcl.v2.live.normal_semantic_indexed_exchange import (
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
        )
        from nurse_rcl.v2.live.ops_exchange import OPS_DECISION_EXCHANGE_VERSION
        from nurse_rcl.v2.live.ops_report_exchange import OPS_REPORT_EXCHANGE_VERSION

        if type(request_protocol) is not str or request_protocol not in (
            ADAPTER_VERSION,
            OPS_DECISION_EXCHANGE_VERSION,
            OPS_REPORT_EXCHANGE_VERSION,
            *EMPIRICAL_PROTOCOLS,
            *REPORT_PROTOCOLS,
            SEMANTIC_EXCHANGE_VERSION,
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
            HYBRID_SEMANTIC_EXCHANGE_VERSION,
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
        ):
            raise ValueError("unregistered live request protocol")
        self.request_protocol = request_protocol
        if request_protocol in (SRE_TYPED_REPORT_PROTOCOL, SRE_GPLUS_REPORT_PROTOCOL):
            from nurse_rcl.v2.live.ops_sre_typed_report_provider import TypedReportTransport

            if type(transport) is not TypedReportTransport or (
                transport.code != code_sha256
                or transport.execution != config_sha256
                or transport.protocol != request_protocol
            ):
                raise ValueError("typed report requires exact bound recording transport")
        for pin in (code_sha256, config_sha256):
            if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
                raise ValueError("invalid live run identity pin")
        if type(settings) is not ArkSettings or not authorized_case_ids:
            raise ValueError("invalid live provider or authorization")
        for case in authorized_case_ids:
            LiveRequestBinding(
                opaque_case_id=case,
                variant="B1",
                phase="report",
                input_sha256=code_sha256,
                context_sha256=code_sha256,
                prompt_schema_sha256=code_sha256,
                request_sha256=code_sha256,
            )
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(lock_path).resolve()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.allowed = frozenset(authorized_case_ids)
        self.sleep = sleep
        self.identity = dict(
            schema_version="nurse.v2.live-ark-run/1",
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            authorized_case_ids=sorted(self.allowed),
            settings=settings.audit_record(),
            llm_in_flight=1,
            max_attempts=5,
        )
        if request_protocol in (
            OPS_DECISION_EXCHANGE_VERSION,
            OPS_REPORT_EXCHANGE_VERSION,
            *EMPIRICAL_PROTOCOLS,
            *REPORT_PROTOCOLS,
            SEMANTIC_EXCHANGE_VERSION,
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
            HYBRID_SEMANTIC_EXCHANGE_VERSION,
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
        ):
            self.identity.update(
                schema_version="nurse.v2.live-ark-run/2", request_protocol=request_protocol
            )
        if request_protocol in EMPIRICAL_PROTOCOLS:
            self.identity["usage_policy_id"] = USAGE_POLICY
        self.identity_sha256 = sha256_hex(canonical_json_bytes(self.identity))
        with self._lock():
            path = self.root / "run-identity.json"
            if path.exists():
                if _read(path, "live_ark_run") != self.identity:
                    raise ValueError("live run identity changed; resume forbidden")
            else:
                _write(path, "live_ark_run", self.identity)
        self.client = ArkChatCompletionsClient(
            settings,
            transport=transport,
            **(
                {"response_limit_bytes": EMPIRICAL_RESPONSE_LIMITS.max_bytes}
                if request_protocol in EMPIRICAL_PROTOCOLS
                else {}
            ),
        )

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with live_ark_lock(self.lock_path):
            yield

    def _load(self, binding: LiveRequestBinding) -> _Ledger:
        path = self.root / f"request-{binding.slot}.json"
        if not path.exists():
            model = _EmpiricalLedger if self.request_protocol in EMPIRICAL_PROTOCOLS else _Ledger
            return model(run_identity_sha256=self.identity_sha256, binding=binding)
        ledger = read_attempt_ledger(path, empirical=self.request_protocol in EMPIRICAL_PROTOCOLS)
        if ledger.binding != binding or ledger.run_identity_sha256 != self.identity_sha256:
            raise ValueError("live request binding drift")
        return ledger

    def _save(self, ledger: _Ledger) -> None:
        model = _EmpiricalLedger if self.request_protocol in EMPIRICAL_PROTOCOLS else _Ledger
        validated = model.model_validate(ledger.model_dump(mode="python"))
        _write(
            self.root / f"request-{validated.binding.slot}.json",
            "live_ark_request",
            validated.model_dump(mode="json"),
        )

    def complete(self, binding: LiveRequestBinding, payload: bytes) -> bytes:
        from nurse_rcl.v2.live.normal_semantic_compact_exchange import (
            COMPACT_LIMITS,
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
            validate_compact_semantic_payload,
        )
        from nurse_rcl.v2.live.normal_semantic_exchange import (
            SEMANTIC_EXCHANGE_VERSION,
            validate_normal_semantic_payload,
        )
        from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
        from nurse_rcl.v2.live.ops_exchange import (
            OPS_DECISION_EXCHANGE_VERSION,
            OPS_EMPIRICAL_PROMPT_SHA256,
            OPS_PROMPT_SHA256,
            validate_ops_decision_payload,
        )
        from nurse_rcl.v2.live.ops_report_exchange import (
            OPS_EMPIRICAL_REPORT_PROMPT_SHA256,
            OPS_REPORT_EXCHANGE_VERSION,
            OPS_REPORT_PROMPT_SHA256,
            validate_ops_report_payload,
        )
        from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS

        if type(binding) is not LiveRequestBinding or type(payload) is not bytes:
            raise ValueError("invalid live request types")
        binding = LiveRequestBinding.model_validate(binding.model_dump(mode="python"))
        if binding.opaque_case_id not in self.allowed:
            raise ValueError("case not authorized for external ARK")
        if not hmac.compare_digest(sha256_hex(payload), binding.request_sha256):
            raise ValueError("live outbound payload differs from independent request pin")
        from nurse_rcl.v2.live.normal_semantic_disabled_exchange import (
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
            validate_disabled_semantic_payload,
        )
        from nurse_rcl.v2.live.normal_semantic_followup import (
            HYBRID_SEMANTIC_EXCHANGE_VERSION,
            validate_hybrid_semantic_payload,
        )
        from nurse_rcl.v2.live.normal_semantic_indexed_exchange import (
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
            validate_indexed_semantic_payload,
        )

        compact_semantic = self.request_protocol in (
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
            HYBRID_SEMANTIC_EXCHANGE_VERSION,
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
        )
        empirical = self.request_protocol in EMPIRICAL_PROTOCOLS
        typed_report = self.request_protocol in (
            SRE_TYPED_REPORT_PROTOCOL,
            SRE_GPLUS_REPORT_PROTOCOL,
        )
        strict_accounting = typed_report or self.request_protocol in {
            SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
            SRE_FOCUSED_DECISION_PROTOCOL,
        }
        expected_recorded_variant = {
            SRE_TYPED_REPORT_PROTOCOL: "v2-ops-sre-typed-report/1",
            SRE_GPLUS_REPORT_PROTOCOL: "v2-ops-sre-gplus-report/1",
        }.get(self.request_protocol)
        if (
            binding.variant in {"v2-ops-sre-typed-report/1", "v2-ops-sre-gplus-report/1"}
        ) != typed_report or (typed_report and binding.variant != expected_recorded_variant):
            raise ValueError("typed report-only binding/protocol differs")
        if binding.variant == "v2-ops-sre-report/1" and (
            self.request_protocol != SRE_ATOMIC_REPORT_PROTOCOL
        ):
            raise ValueError("report-only binding cannot borrow another protocol")
        response_limits = EMPIRICAL_RESPONSE_LIMITS if empirical else ExchangeLimits()
        if strict_accounting:
            from nurse_rcl.v2.live.ops_sre_typed_report_provider import require_typed_accounting

        parse_usage: Callable[[bytes], AttemptUsage]
        if typed_report:
            from nurse_rcl.v2.live.ops_sre_typed_report_provider import (
                typed_attempt_usage,
            )

            parse_usage = typed_attempt_usage
        else:
            parse_usage = _empirical_usage if empirical else _usage
        isolated_decision = self.request_protocol in (
            OPS_DECISION_EXCHANGE_VERSION,
            DECISION_PROTOCOL,
            *SRE_DECISION_PROTOCOLS,
        )
        isolated_report = self.request_protocol in (
            OPS_REPORT_EXCHANGE_VERSION,
            REPORT_PROTOCOL,
            *SRE_REPORT_PROTOCOLS,
        )
        isolated = isolated_decision or isolated_report
        if (
            binding.variant
            in (
                "v2-ops-isolated/1",
                "v2-ops-isolated/2",
                "v2-ops-sre/1",
            )
            and not isolated
        ):
            raise ValueError("isolated variant cannot borrow a legacy protocol")
        request_limits = (
            REPORT_LIMITS
            if isolated_report
            else CONTEXT_LIMITS
            if isolated
            else (COMPACT_LIMITS if compact_semantic else ExchangeLimits())
        )
        request = decode_json_object(payload, request_limits)
        expected_variant = "v2-ops-isolated/2" if empirical else "v2-ops-isolated/1"
        if typed_report:
            from nurse_rcl.v2.live.ops_sre_typed_report import bind_typed_report

            if self.request_protocol == SRE_GPLUS_REPORT_PROTOCOL:
                from nurse_rcl.v2.live.ops_sre_gplus_report import bind_gplus_report

                expected_binding = bind_gplus_report(payload)
            else:
                expected_binding = bind_typed_report(payload)
            if binding != expected_binding:
                raise ValueError("typed report exact frozen binding differs")
            tool = None
        elif self.request_protocol in SRE_PROTOCOLS:
            from nurse_rcl.v2.live.ops_sre_exchange import bind_sre_decision
            from nurse_rcl.v2.live.ops_sre_report_exchange import bind_sre_report

            if self.request_protocol in {
                SRE_ATOMIC_REPORT_PROTOCOL,
                SRE_BRIEF_REPORT_PROTOCOL,
                SRE_CONDITION_REPORT_PROTOCOL,
                SRE_RECOVERY_REPORT_PROTOCOL,
                SRE_COMPARABLE_REPORT_PROTOCOL,
                SRE_NAVIGATION_REPORT_PROTOCOL,
                SRE_WORKBENCH_REPORT_PROTOCOL,
                SRE_USED_SOURCE_REPORT_PROTOCOL,
                SRE_RECEIPT_REUSE_REPORT_PROTOCOL,
                SRE_BALANCED_REPORT_PROTOCOL,
                SRE_REPLANNING_REPORT_PROTOCOL,
                SRE_QUESTION_REPORT_PROTOCOL,
                SRE_ACTION_REPORT_PROTOCOL,
            }:
                from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_report

                expected_binding = bind_brief_report(payload)
            elif self.request_protocol in {
                SRE_BRIEF_DECISION_PROTOCOL,
                SRE_FEEDBACK_DECISION_PROTOCOL,
                SRE_RECOVERY_DECISION_PROTOCOL,
                SRE_COMPARABLE_DECISION_PROTOCOL,
                SRE_NAVIGATION_DECISION_PROTOCOL,
                SRE_WORKBENCH_DECISION_PROTOCOL,
                SRE_USED_SOURCE_DECISION_PROTOCOL,
                SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
                SRE_BALANCED_DECISION_PROTOCOL,
                SRE_REPLANNING_DECISION_PROTOCOL,
                SRE_QUESTION_DECISION_PROTOCOL,
                SRE_ACTION_DECISION_PROTOCOL,
                SRE_DIRECTED_DECISION_PROTOCOL,
                SRE_ENTRY_DECISION_PROTOCOL,
                SRE_OBSERVABLE_DECISION_PROTOCOL,
                SRE_SINGLE_OWNER_DECISION_PROTOCOL,
                SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
                SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
                SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
                SRE_FOCUSED_DECISION_PROTOCOL,
            }:
                from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision

                if binding.decision_index is None:
                    raise ValueError("brief investigation index missing")
                expected_binding = bind_brief_decision(
                    payload,
                    opaque_case_id=binding.opaque_case_id,
                    case_snapshot_id=binding.input_sha256,
                    decision_index=binding.decision_index,
                )
            elif self.request_protocol == SRE_NATIVE_REPORT_PROTOCOL:
                from nurse_rcl.v2.live.ops_sre_native_exchange import bind_native_report

                expected_binding = bind_native_report(payload)
            elif self.request_protocol == SRE_REPORT_PROTOCOL:
                expected_binding = bind_sre_report(payload)
            else:
                if binding.decision_index is None:
                    raise ValueError("SRE investigation index missing")
                expected_binding = bind_sre_decision(
                    payload,
                    opaque_case_id=binding.opaque_case_id,
                    case_snapshot_id=binding.input_sha256,
                    decision_index=binding.decision_index,
                    working=self.request_protocol == SRE_WORKING_DECISION_PROTOCOL,
                )
            if binding != expected_binding:
                raise ValueError("SRE request binding differs")
            tool = None
        elif isolated_report:
            report_context = validate_ops_report_payload(request, empirical=empirical)
            report_values = (
                report_context
                if isinstance(report_context, dict)
                else dict(
                    core_result_digest=report_context.core_result_digest,
                    context_sha256=report_context.context_sha256,
                )
            )
            if (
                binding.variant != expected_variant
                or binding.phase != "report"
                or (
                    binding.input_sha256 != report_values["core_result_digest"]
                    or binding.context_sha256 != report_values["context_sha256"]
                    or binding.prompt_schema_sha256
                    != (
                        OPS_EMPIRICAL_REPORT_PROMPT_SHA256
                        if empirical
                        else OPS_REPORT_PROMPT_SHA256
                    )
                )
            ):
                raise ValueError("isolated report binding differs")
            tool = None
        elif isolated_decision:
            body = validate_ops_decision_payload(request, empirical=empirical)
            if (
                binding.variant != expected_variant
                or binding.phase != "investigation"
                or (
                    binding.decision_index != body["context"]["decision_index"]
                    or binding.input_sha256 != body["case_snapshot_id"]
                    or binding.context_sha256
                    != sha256_hex(
                        encode_json(
                            body["context"],
                            CONTEXT_LIMITS,
                        )
                    )
                    or binding.prompt_schema_sha256
                    != (OPS_EMPIRICAL_PROMPT_SHA256 if empirical else OPS_PROMPT_SHA256)
                )
            ):
                raise ValueError("isolated request binding differs")
            tool = None
        elif self.request_protocol in (
            SEMANTIC_EXCHANGE_VERSION,
            COMPACT_SEMANTIC_EXCHANGE_VERSION,
            HYBRID_SEMANTIC_EXCHANGE_VERSION,
            DISABLED_SEMANTIC_EXCHANGE_VERSION,
            INDEXED_SEMANTIC_EXCHANGE_VERSION,
        ):
            if self.request_protocol == INDEXED_SEMANTIC_EXCHANGE_VERSION:
                phase = validate_indexed_semantic_payload(request)
            elif self.request_protocol == DISABLED_SEMANTIC_EXCHANGE_VERSION:
                phase = validate_disabled_semantic_payload(request)
            elif self.request_protocol == HYBRID_SEMANTIC_EXCHANGE_VERSION:
                phase = validate_hybrid_semantic_payload(request)
            else:
                phase = (
                    validate_compact_semantic_payload(request)
                    if compact_semantic
                    else validate_normal_semantic_payload(request)
                )
            if binding.variant != "B2" or binding.phase != phase:
                raise ValueError("normal semantic protocol requires its bound B2 phase")
            tool = None
        elif self.request_protocol in REPORT_PROTOCOLS:
            expected_variant = (
                "B2"
                if self.request_protocol == B2_REPORT_EXCHANGE_VERSION
                else "v2-live-evidence-repair"
                if self.request_protocol == REPAIR_EXCHANGE_VERSION
                else "v2-live-bootstrap"
            )
            if binding.variant != expected_variant or binding.phase != "report":
                raise ValueError("report protocol requires its independently registered variant")
            validate_report_payload(request, protocol=self.request_protocol)
            tool = None
        else:
            if binding.variant == "v2-live-evidence-repair":
                raise ValueError("repair variant requires its registered report protocol")
            tool = _validate_payload(request)
        if request["model"] != ARK_MODEL or encode_json(request, request_limits) != payload:
            raise ValueError("live request model or canonical wire identity invalid")
        _require_payload_case(request, binding.opaque_case_id, limits=request_limits)
        workbench_turn = None
        if self.request_protocol in {
            SRE_WORKBENCH_DECISION_PROTOCOL,
            SRE_USED_SOURCE_DECISION_PROTOCOL,
            SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
            SRE_BALANCED_DECISION_PROTOCOL,
            SRE_REPLANNING_DECISION_PROTOCOL,
            SRE_QUESTION_DECISION_PROTOCOL,
            SRE_ACTION_DECISION_PROTOCOL,
            SRE_DIRECTED_DECISION_PROTOCOL,
            SRE_ENTRY_DECISION_PROTOCOL,
            SRE_OBSERVABLE_DECISION_PROTOCOL,
            SRE_SINGLE_OWNER_DECISION_PROTOCOL,
            SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
            SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
            SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
            SRE_FOCUSED_DECISION_PROTOCOL,
        }:
            from nurse_rcl.v2.live.ops_sre_turn_validation import bind_workbench_turn

            workbench_turn = bind_workbench_turn(payload, binding)
        with self._lock():
            ledger = self._load(binding)

            def update_last(**changes: Any) -> None:
                nonlocal ledger
                model = _EmpiricalStoredAttempt if empirical else _StoredAttempt
                last = model.model_validate({**ledger.attempts[-1].model_dump(), **changes})
                ledger = ledger.model_copy(update={"attempts": (*ledger.attempts[:-1], last)})
                self._save(ledger)

            # A persisted receipt wins over a missing final bookkeeping write.
            # Without a receipt, network outcome is unknown and the slot is spent.
            if empirical:
                recovered = recover_empirical_pending(
                    ledger, protocol=self.request_protocol, workbench_turn=workbench_turn
                )
                if recovered != ledger:
                    ledger = recovered
                    self._save(ledger)
            if strict_accounting:
                # Adopted feedback and recorded reports: unknown spend cannot resume.
                for saved_attempt in ledger.attempts:
                    if saved_attempt.record is not None:
                        require_typed_accounting(
                            None
                            if saved_attempt.response_json is None
                            else saved_attempt.response_json.encode()
                        )
            if ledger.attempts and ledger.attempts[-1].record is None:
                pending = ledger.attempts[-1]
                valid = False
                usage = AttemptUsage.unavailable()
                if pending.response_json is not None:
                    raw = pending.response_json.encode("utf-8")
                    usage = _usage(raw)
                    try:
                        _parse_protocol_response(
                            raw,
                            output_tool_name=tool,
                            limits=ExchangeLimits(),
                            protocol=self.request_protocol,
                            workbench_turn=workbench_turn,
                        )
                        valid = True
                    except ExchangeBoundaryError:
                        pass
                record = AttemptRecord(
                    attempt_number=pending.number,
                    category="success" if valid else "transport",
                    outcome="success" if valid else "exhausted" if pending.number == 5 else "retry",
                    latency_seconds=0.0,
                    usage=usage,
                    delay_seconds=0.0 if not valid and pending.number < 5 else None,
                    error_type=None
                    if valid
                    else (
                        "OutcomeUnknownAfterInterruption"
                        if pending.response_json is None
                        else "InterruptedMalformedReceipt"
                    ),
                )
                update_last(record=record, interrupted=True, latency_known=False)
            if ledger.attempts:
                final = ledger.attempts[-1]
                assert final.record is not None
                if final.record.outcome == "success":
                    assert final.response_json is not None
                    raw = final.response_json.encode("utf-8")
                    _parse_protocol_response(
                        raw,
                        output_tool_name=tool,
                        limits=response_limits,
                        protocol=self.request_protocol,
                        workbench_turn=workbench_turn,
                    )
                    ledger = ledger.model_copy(
                        update={"response_reuses": ledger.response_reuses + 1}
                    )
                    self._save(ledger)
                    return raw
                if final.record.outcome in ("fatal", "exhausted"):
                    if final.record.error_type == "NativePartialResponseError":
                        raise NativePartialResponseError("native_partial")
                    if final.record.error_type == "ArkResponseResourceError":
                        raise LiveTransportError("response_resource_limit")
                    raise LiveTransportError(final.record.outcome)
                if typed_report:
                    raise LiveTransportError("typed_saved_retry_no_resend")

            def reserve(number: int) -> None:
                nonlocal ledger
                model = _EmpiricalStoredAttempt if empirical else _StoredAttempt
                ledger = ledger.model_copy(
                    update={"attempts": (*ledger.attempts, model(number=number))}
                )
                self._save(ledger)

            def attempt(number: int) -> AttemptResult[bytes]:
                try:
                    with _module_span("ark_attempt"):
                        value = self.client.complete(request)
                except httpx.HTTPStatusError as error:
                    if empirical:
                        failure_raw = None
                        try:
                            value = error.response.json()
                            if type(value) is dict:
                                failure_raw = encode_json(
                                    self.settings.redact_artifact(value), response_limits
                                )
                        except (ValueError, UnicodeError):
                            pass  # Non-JSON failure: attempt retained, usage unknown.
                        if failure_raw is not None:
                            update_last(
                                response_json=failure_raw.decode(),
                                response_sha256=sha256_hex(failure_raw),
                                http_status_code=error.response.status_code,
                            )
                    raise
                raw = encode_json(self.settings.redact_artifact(value), response_limits)
                update_last(response_json=raw.decode("utf-8"), response_sha256=sha256_hex(raw))
                usage = parse_usage(raw)
                if empirical and _native_partial(raw):
                    raise NativePartialResponseError("native_partial")
                try:
                    _parse_protocol_response(
                        raw,
                        output_tool_name=tool,
                        limits=response_limits,
                        protocol=self.request_protocol,
                        workbench_turn=workbench_turn,
                    )
                except ExchangeBoundaryError as error:
                    raise RetryableParsingError(error.code, usage=usage) from None
                return AttemptResult(raw, usage)

            def record_result(record: AttemptRecord, value: bytes | None) -> None:
                if empirical and ledger.attempts[-1].response_json is not None:
                    record = replace(
                        record, usage=parse_usage(ledger.attempts[-1].response_json.encode())
                    )
                update_last(record=record)
                if strict_accounting:
                    # Actual original retry callback: preserve record FIRST. An
                    # exception here prevents both sleep and the next reservation.
                    require_typed_accounting(
                        None
                        if ledger.attempts[-1].response_json is None
                        else ledger.attempts[-1].response_json.encode()
                    )

            def retry_sleep(seconds: float) -> None:
                with _module_span("retry_wait"):
                    self.sleep(seconds)

            try:
                execution: RequestExecution[bytes] = execute_llm_request(
                    attempt,
                    sleep=retry_sleep,
                    checkpoint=lambda _: None,
                    circuit_breaker=CircuitBreaker(),
                    initial_records=tuple(
                        a.record for a in ledger.attempts if a.record is not None
                    ),
                    reserve_attempt=reserve,
                    record_attempt=record_result,
                )
            except (FatalProviderRequestError, RetryExhaustedError) as error:
                if isinstance(error, FatalProviderRequestError) and (
                    error.record.error_type == "NativePartialResponseError"
                ):
                    raise NativePartialResponseError("native_partial") from None
                if isinstance(error, FatalProviderRequestError) and (
                    error.record.error_type == ArkResponseResourceError.__name__
                ):
                    raise LiveTransportError("response_resource_limit") from None
                raise LiveTransportError(
                    "fatal" if isinstance(error, FatalProviderRequestError) else "exhausted"
                ) from None
            return execution.value

    def receipt(self, binding: LiveRequestBinding) -> dict[str, Any]:
        with self._lock():
            ledger = self._load(binding)
            return dict(
                ledger=ledger.model_dump(mode="json"),
                attempt_count=len(ledger.attempts),
                response_reuses=ledger.response_reuses,
                outcome_unknown_count=sum(
                    a.response_json is None
                    and (
                        a.record is None or a.record.error_type == "OutcomeUnknownAfterInterruption"
                    )
                    for a in ledger.attempts
                ),
                cost=None,
            )  # No provider price/usage must never become zero cost.

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> LiveArkBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
