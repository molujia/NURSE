"""Private outbound owner for admitted, single-case isolated investigations.

The case runner owns source admission. This finite wire check is not source
authentication or a sandbox for arbitrary Python callers.
"""

from __future__ import annotations

import re
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from nurse_rcl.agents import ark
from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.exchange_json import (
    ExchangeBoundaryError,
    ExchangeLimits,
    decode_json_object,
    encode_json,
)
from nurse_rcl.v2.hermes_response import parse_response
from nurse_rcl.v2.inference_leakage import _is_forbidden_field_name
from nurse_rcl.v2.live.ark_bridge import (
    LiveArkBridge,
    LiveRequestBinding,
    LiveTransportError,
    NativePartialResponseError,
    _Ledger,
    _read,
    _usage,
    _write,
    live_ark_lock,
    read_attempt_ledger,
    recover_empirical_pending,
)
from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS, DECISION_LIMITS, parse_decision
from nurse_rcl.v2.live.ops_contracts import SreProfileId, resolve_sre_profile
from nurse_rcl.v2.live.ops_empirical_policy import (
    DECISION_PROTOCOL,
    PROTOCOLS,
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
    SRE_DECISION_PROTOCOL,
    SRE_DECISION_PROTOCOLS,
    SRE_DIRECTED_DECISION_PROTOCOL,
    SRE_ENTRY_DECISION_PROTOCOL,
    SRE_FEEDBACK_DECISION_PROTOCOL,
    SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
    SRE_FOCUSED_DECISION_PROTOCOL,
    SRE_NATIVE_REPORT_PROTOCOL,
    SRE_NAVIGATION_DECISION_PROTOCOL,
    SRE_NAVIGATION_REPORT_PROTOCOL,
    SRE_OBSERVABLE_DECISION_PROTOCOL,
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
    SRE_USED_SOURCE_DECISION_PROTOCOL,
    SRE_USED_SOURCE_REPORT_PROTOCOL,
    SRE_VARIANT,
    SRE_WORKBENCH_DECISION_PROTOCOL,
    SRE_WORKBENCH_REPORT_PROTOCOL,
    SRE_WORKING_DECISION_PROTOCOL,
    USAGE_POLICY,
)
from nurse_rcl.v2.live.ops_empirical_policy import (
    RESPONSE_LIMITS as EMPIRICAL_RESPONSE_LIMITS,
)
from nurse_rcl.v2.live.ops_exchange import (
    OPS_DECISION_EXCHANGE_VERSION,
    bind_empirical_decision,
    prepare_ops_decision,
)
from nurse_rcl.v2.live.ops_report_context import OpsReportContext
from nurse_rcl.v2.live.ops_report_exchange import (
    OPS_REPORT_EXCHANGE_VERSION,
    bind_empirical_report,
    prepare_ops_report,
)
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS, RESPONSE_LIMITS, parse_ops_report
from nurse_rcl.v2.live.ops_usage import summarize_attempts
from nurse_rcl.v2.live.repair_driver import publish_json

KEY_PATH = Path("/home/wangrunzhou/.config/nurse/ark.env")
LOCK_PATH = Path("/home/wangrunzhou/NURSE/development/live-rcl-20260918/locks/ark-in-flight.lock")
_OPAQUE = re.compile(r"opq_[0-9a-f]{32}")
_RAW_CASE = re.compile(r"(?i)(?:\bts\d+-[a-z0-9_-]+|\b[0-9a-f]{8}-\d{3}\b)")
_PRIVATE = {
    "sourceroot",
    "sourcecaseid",
    "evaluator",
    "labels",
    "labelpath",
    "apikey",
    "arkapikey",
    "arkapikey2",
    "identitymanifestpath",
    "inputpath",
}
# These names describe a dependency endpoint or a record path, NOT a label or
# source filesystem. They occur in code-owned node/report schemas.
_PUBLIC_NAMES = {"target_service_key", "target_service_keys", "source_path"}


def usage_summary(
    root: Path,
    *,
    execution_sha256: str,
    empirical: bool = False,
    confirmed_no_send: bool = False,
    sre: bool = False,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
    feedback_fact: bool = False,
) -> dict[str, Any]:
    """Read latest durable slots, not duplicate historical receipts or cached responses."""
    profile = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    if feedback_fact and not (empirical and sre and profile == "hermes-sre-tools/23"):
        raise ValueError("feedback usage requires explicit atomic SRE presentation")
    if sre and not empirical:
        raise ValueError("SRE requires empirical usage policy")
    if working and not sre:
        raise ValueError("working usage requires SRE")
    if empirical:
        with live_ark_lock(LOCK_PATH):
            return _empirical_summary(
                root,
                execution_sha256,
                confirmed_no_send,
                sre=sre,
                working=working,
                sre_profile=profile,
                feedback_fact=feedback_fact,
            )
    counts: dict[str, list[int | None]] = {
        key: [] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    }
    attempts = logical = unknown = latency_unknown = 0
    latency = 0.0
    for folder, cap in (("ark-investigation", 8), ("ark-report", 1)):
        paths = sorted((root / folder).glob("request-*.json"))
        if not paths:
            continue
        identity = _read(root / folder / "run-identity.json", "live_ark_run")
        if identity["config_sha256"] != execution_sha256 or len(paths) > cap:
            raise ValueError("isolated usage identity or request limit differs")
        identity_pin = sha256_hex(canonical_json_bytes(identity))
        for path in paths:
            ledger = _Ledger.model_validate(_read(path, "live_ark_request"))
            if ledger.run_identity_sha256 != identity_pin or path.name != (
                f"request-{ledger.binding.slot}.json"
            ):
                raise ValueError("isolated usage ledger binding differs")
            logical += 1
            for attempt in ledger.attempts:
                attempts += 1
                record = attempt.record
                usage = (
                    record.usage
                    if record is not None
                    else (_usage(attempt.response_json.encode()) if attempt.response_json else None)
                )
                values = {key: None if usage is None else getattr(usage, key) for key in counts}
                for key, value in values.items():
                    counts[key].append(value)
                unknown += any(
                    values[k] is None
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                if record is None or not attempt.latency_known:
                    latency_unknown += 1
                else:
                    latency += record.latency_seconds
    return dict(
        logical_requests=logical,
        attempts=attempts,
        unknown_attempts=unknown,
        **{
            key: None if any(v is None for v in values) else sum(v for v in values if v is not None)
            for key, values in counts.items()
        },
        reported_token_subtotals={
            key: sum(v for v in values if v is not None) for key, values in counts.items()
        },
        known_attempt_seconds=latency,
        unknown_latency_attempts=latency_unknown,
        cost=None,
    )


def _empirical_summary(
    root: Path,
    execution: str,
    confirmed_no_send: bool,
    *,
    sre: bool = False,
    working: bool = False,
    sre_profile: SreProfileId | None = None,
    feedback_fact: bool = False,
) -> dict[str, Any]:
    profile = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
    focused = profile == "hermes-sre-tools/25"
    atomic_selector = profile == "hermes-sre-tools/23" or focused
    if feedback_fact and not (sre and atomic_selector and not focused):
        raise ValueError("feedback usage requires explicit atomic SRE presentation")
    typed_purpose = profile == "hermes-sre-tools/22"
    single_owner = profile == "hermes-sre-tools/21"
    observable = (
        profile == "hermes-sre-tools/20" or single_owner or typed_purpose or atomic_selector
    )
    entry = profile == "hermes-sre-tools/19" or observable
    directed = profile == "hermes-sre-tools/18"
    brief = profile in {
        "hermes-sre-tools/6",
        "hermes-sre-tools/7",
        "hermes-sre-tools/8",
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
        "hermes-sre-tools/18",
    }
    comparable = profile == "hermes-sre-tools/9"
    used_source = profile == "hermes-sre-tools/12"
    question_led = profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"} or directed or entry
    replanning = profile == "hermes-sre-tools/15" or question_led
    balanced = profile == "hermes-sre-tools/14" or replanning
    receipt_reuse = profile == "hermes-sre-tools/13" or balanced
    workbench = profile == "hermes-sre-tools/11"
    navigation = profile == "hermes-sre-tools/10"
    recovery = profile == "hermes-sre-tools/8"
    condition = profile == "hermes-sre-tools/7"
    working = profile == "hermes-sre-tools/5"
    rows = []
    latency, latency_unknown, reuses = 0.0, 0, 0
    common_identity = None
    for folder, phase, cap, protocol in (
        (
            "ark-investigation",
            "investigation",
            9 if used_source or receipt_reuse else 8,
            "nurse.v2.sre-decision-exchange/26"
            if focused
            else "nurse.v2.sre-decision-exchange/25"
            if feedback_fact
            else "nurse.v2.sre-decision-exchange/24"
            if atomic_selector
            else "nurse.v2.sre-decision-exchange/23"
            if typed_purpose
            else "nurse.v2.sre-decision-exchange/22"
            if single_owner
            else SRE_OBSERVABLE_DECISION_PROTOCOL
            if observable
            else SRE_ENTRY_DECISION_PROTOCOL
            if entry
            else SRE_DIRECTED_DECISION_PROTOCOL
            if directed
            else SRE_ACTION_DECISION_PROTOCOL
            if profile == "hermes-sre-tools/17"
            else SRE_QUESTION_DECISION_PROTOCOL
            if question_led
            else SRE_REPLANNING_DECISION_PROTOCOL
            if replanning
            else SRE_BALANCED_DECISION_PROTOCOL
            if balanced
            else SRE_RECEIPT_REUSE_DECISION_PROTOCOL
            if receipt_reuse
            else SRE_USED_SOURCE_DECISION_PROTOCOL
            if used_source
            else SRE_WORKBENCH_DECISION_PROTOCOL
            if workbench
            else SRE_NAVIGATION_DECISION_PROTOCOL
            if navigation
            else SRE_COMPARABLE_DECISION_PROTOCOL
            if comparable
            else SRE_RECOVERY_DECISION_PROTOCOL
            if recovery
            else SRE_FEEDBACK_DECISION_PROTOCOL
            if condition
            else SRE_BRIEF_DECISION_PROTOCOL
            if brief
            else SRE_WORKING_DECISION_PROTOCOL
            if working
            else SRE_DECISION_PROTOCOL
            if sre
            else DECISION_PROTOCOL,
        ),
        (
            "ark-report",
            "report",
            0 if directed or entry else 1,
            "disabled_until_sre_report_stage/1"
            if directed or entry
            else SRE_ACTION_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/17"
            else SRE_QUESTION_REPORT_PROTOCOL
            if question_led
            else SRE_REPLANNING_REPORT_PROTOCOL
            if replanning
            else SRE_BALANCED_REPORT_PROTOCOL
            if balanced
            else SRE_RECEIPT_REUSE_REPORT_PROTOCOL
            if receipt_reuse
            else SRE_USED_SOURCE_REPORT_PROTOCOL
            if used_source
            else SRE_WORKBENCH_REPORT_PROTOCOL
            if workbench
            else SRE_NAVIGATION_REPORT_PROTOCOL
            if navigation
            else SRE_COMPARABLE_REPORT_PROTOCOL
            if comparable
            else SRE_RECOVERY_REPORT_PROTOCOL
            if recovery
            else SRE_CONDITION_REPORT_PROTOCOL
            if condition
            else SRE_BRIEF_REPORT_PROTOCOL
            if brief
            else SRE_NATIVE_REPORT_PROTOCOL
            if working
            else SRE_REPORT_PROTOCOL
            if sre
            else REPORT_PROTOCOL,
        ),
    ):
        paths = sorted((root / folder).glob("request-*.json"))
        if not paths:
            continue
        identity = _read(root / folder / "run-identity.json", "live_ark_run")
        if (
            identity.get("config_sha256") != execution
            or len(paths) > cap
            or (
                identity.get("request_protocol") != protocol
                or identity.get("usage_policy_id") != USAGE_POLICY
                or len(identity.get("authorized_case_ids", ())) != 1
                or identity.get("max_attempts") != 5
                or identity.get("llm_in_flight") != 1
            )
        ):
            raise ValueError("empirical usage identity differs")
        common = {k: v for k, v in identity.items() if k != "request_protocol"}
        if common_identity is not None and common_identity != common:
            raise ValueError("empirical channel identities differ")
        common_identity = common
        identity_pin = sha256_hex(canonical_json_bytes(identity))
        for path in paths:
            ledger = read_attempt_ledger(path, empirical=True)
            binding = ledger.binding
            if (
                ledger.run_identity_sha256 != identity_pin
                or path.name != (f"request-{binding.slot}.json")
                or (
                    binding.variant
                    != (
                        "v2-ops-sre/15"
                        if focused
                        else "v2-ops-sre/14"
                        if feedback_fact
                        else "v2-ops-sre/13"
                        if atomic_selector
                        else "v2-ops-sre/12"
                        if typed_purpose
                        else "v2-ops-sre/11"
                        if single_owner
                        else "v2-ops-sre/10"
                        if observable
                        else "v2-ops-sre/9"
                        if entry
                        else "v2-ops-sre/8"
                        if directed
                        else "v2-ops-sre/7"
                        if profile == "hermes-sre-tools/17"
                        else "v2-ops-sre/6"
                        if question_led
                        else "v2-ops-sre/5"
                        if replanning
                        else "v2-ops-sre/4"
                        if balanced
                        else "v2-ops-sre/3"
                        if receipt_reuse
                        else "v2-ops-sre/2"
                        if used_source
                        else SRE_VARIANT
                        if sre
                        else "v2-ops-isolated/2"
                    )
                    or binding.phase
                    not in (
                        {"investigation", "investigation_correction"}
                        if (used_source or receipt_reuse) and phase == "investigation"
                        else {phase}
                    )
                    or [binding.opaque_case_id] != identity["authorized_case_ids"]
                )
            ):
                raise ValueError("empirical usage ledger binding differs")
            reuses += ledger.response_reuses
            for attempt in ledger.attempts:
                usage = (
                    None
                    if attempt.response_json is None
                    else decode_json_object(
                        attempt.response_json.encode(), EMPIRICAL_RESPONSE_LIMITS
                    ).get("usage")
                )
                rows.append(
                    dict(
                        phase=binding.phase,
                        slot=binding.correction_index or binding.decision_index or 1,
                        attempt=attempt.number,
                        usage=usage,
                    )
                )
                if attempt.record is None or not attempt.latency_known:
                    latency_unknown += 1
                else:
                    latency += attempt.record.latency_seconds
    summary = summarize_attempts(rows, confirmed_no_send=confirmed_no_send)
    return dict(
        summary,
        usage_policy_id=USAGE_POLICY,
        known_attempt_seconds=latency,
        unknown_latency_attempts=latency_unknown,
        response_reuses=reuses,
        attempt_usage=rows,
        paid_execution_sha256=execution,
        spend_scope="execution_lifetime_including_reused_paid_responses",
        input_target_exceeded=summary["reported_input_tokens"] > 100000,
        cost=None,
    )


def _saved_wire(
    root: Path,
    binding: LiveRequestBinding,
    *,
    code: str,
    execution: str,
    protocol: str,
    read_only: bool = False,
    frozen_wire: bytes | None = None,
) -> bytes | None:
    with live_ark_lock(LOCK_PATH) if protocol in PROTOCOLS else nullcontext():
        return _saved_wire_locked(
            root,
            binding,
            code=code,
            execution=execution,
            protocol=protocol,
            read_only=read_only,
            frozen_wire=frozen_wire,
        )


def _saved_wire_locked(
    root: Path,
    binding: LiveRequestBinding,
    *,
    code: str,
    execution: str,
    protocol: str,
    read_only: bool = False,
    frozen_wire: bytes | None = None,
) -> bytes | None:
    """No provider construction or credential access, including the receipt/write gap."""
    workbench_turn = None
    if protocol in {
        SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
        SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
        SRE_FOCUSED_DECISION_PROTOCOL,
        SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
        SRE_SINGLE_OWNER_DECISION_PROTOCOL,
        SRE_ENTRY_DECISION_PROTOCOL,
        SRE_OBSERVABLE_DECISION_PROTOCOL,
        SRE_DIRECTED_DECISION_PROTOCOL,
        SRE_WORKBENCH_DECISION_PROTOCOL,
        SRE_USED_SOURCE_DECISION_PROTOCOL,
        SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
        SRE_BALANCED_DECISION_PROTOCOL,
        SRE_REPLANNING_DECISION_PROTOCOL,
        SRE_QUESTION_DECISION_PROTOCOL,
        SRE_ACTION_DECISION_PROTOCOL,
    }:
        from nurse_rcl.v2.live.ops_sre_turn_validation import bind_workbench_turn

        if frozen_wire is None:
            raise ValueError("workbench frozen request missing")
        workbench_turn = bind_workbench_turn(frozen_wire, binding)
    empirical = protocol in PROTOCOLS
    is_report = protocol in (OPS_REPORT_EXCHANGE_VERSION, REPORT_PROTOCOL, *SRE_REPORT_PROTOCOLS)
    folder = "ark-report" if is_report else "ark-investigation"
    path = root / folder / f"request-{binding.slot}.json"
    if not path.exists():
        return None
    identity = _read(root / folder / "run-identity.json", "live_ark_run")
    if (
        identity["code_sha256"],
        identity["config_sha256"],
        identity["request_protocol"],
        identity["authorized_case_ids"],
    ) != (code, execution, protocol, [binding.opaque_case_id]):
        raise ValueError("isolated saved provider identity differs")
    if empirical and identity.get("usage_policy_id") != USAGE_POLICY:
        raise ValueError("empirical saved usage policy differs")
    ledger = read_attempt_ledger(path, empirical=empirical)
    if ledger.binding != binding or ledger.run_identity_sha256 != sha256_hex(
        canonical_json_bytes(identity)
    ):
        raise ValueError("isolated saved request binding differs")
    if not ledger.attempts:
        return None
    if empirical:
        fixed = recover_empirical_pending(ledger, protocol=protocol, workbench_turn=workbench_turn)
        if fixed != ledger:
            ledger = fixed
            if not read_only:
                _write(path, "live_ark_request", ledger.model_dump(mode="json"))
    last = ledger.attempts[-1]
    if empirical and last.record is not None:
        if last.record.error_type == "NativePartialResponseError":
            raise NativePartialResponseError("native_partial")
        if last.record.error_type == "ArkResponseResourceError":
            raise LiveTransportError("response_resource_limit")
        if last.record.outcome in ("fatal", "exhausted"):
            raise LiveTransportError(last.record.outcome)
    if last.response_json is None or (last.record is not None and last.record.outcome != "success"):
        return None
    raw = last.response_json.encode()
    try:
        if workbench_turn is not None:
            from nurse_rcl.v2.live.ops_sre_turn_validation import validate_bound_workbench_response

            parsed = validate_bound_workbench_response(raw, workbench_turn)
        else:
            parsed = parse_response(
                raw,
                output_tool_name=None,
                limits=EMPIRICAL_RESPONSE_LIMITS if empirical else ExchangeLimits(),
            )
        if protocol in {
            SRE_ATOMIC_SELECTOR_DECISION_PROTOCOL,
            SRE_FEEDBACK_FACT_DECISION_PROTOCOL,
            SRE_FOCUSED_DECISION_PROTOCOL,
            SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
            SRE_SINGLE_OWNER_DECISION_PROTOCOL,
            SRE_ENTRY_DECISION_PROTOCOL,
            SRE_OBSERVABLE_DECISION_PROTOCOL,
            SRE_WORKBENCH_DECISION_PROTOCOL,
            SRE_DIRECTED_DECISION_PROTOCOL,
            SRE_USED_SOURCE_DECISION_PROTOCOL,
            SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
            SRE_BALANCED_DECISION_PROTOCOL,
            SRE_REPLANNING_DECISION_PROTOCOL,
            SRE_QUESTION_DECISION_PROTOCOL,
            SRE_ACTION_DECISION_PROTOCOL,
        }:
            pass  # Already checked against this exact frozen request.
        elif protocol == SRE_ATOMIC_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import (
                parse_expression_response,
            )

            parse_expression_response(parsed.output_json)
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
        elif protocol in {
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
                }
                else parse_brief_decision
            )(parsed.output_json)
        elif protocol in SRE_REPORT_PROTOCOLS:
            from nurse_rcl.v2.live.ops_sre_report import parse_sre_report

            parse_sre_report(parsed.output_json)
        elif protocol in SRE_DECISION_PROTOCOLS:
            from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision

            parse_sre_decision(parsed.output_json)
        elif is_report:
            parse_ops_report(parsed.output_json, empirical=empirical)
        else:
            parse_decision(parsed.output_json, empirical=empirical)
    except ExchangeBoundaryError:
        if (
            workbench_turn is not None
            and last.record is not None
            and last.record.outcome == "success"
        ):
            raise ValueError("workbench saved response differs from frozen request") from None
        return None
    if empirical and not read_only:
        ledger = ledger.model_copy(update={"response_reuses": ledger.response_reuses + 1})
        _write(path, "live_ark_request", ledger.model_dump(mode="json"))
    return raw


def saved_ops_report(
    root: Path,
    context: OpsReportContext,
    *,
    code_sha256: str,
    execution_sha256: str,
    empirical: bool = False,
    frozen_wire: bytes | None = None,
) -> bytes | None:
    """Recover an already-paid response only; missing data never authorizes a request."""
    if not context.formal_projection_available:
        return None
    if empirical:
        if frozen_wire is None:
            return None
        binding = bind_empirical_report(frozen_wire)
        if (binding.opaque_case_id, binding.input_sha256, binding.context_sha256) != (
            context.opaque_case_id,
            context.core_result_digest,
            context.context_sha256,
        ):
            raise ValueError("saved report context binding differs")
    else:
        _, binding = prepare_ops_report(context)
    raw = _saved_wire(
        root,
        binding,
        code=code_sha256,
        execution=execution_sha256,
        protocol=REPORT_PROTOCOL if empirical else OPS_REPORT_EXCHANGE_VERSION,
    )
    if raw is None:
        return None
    parsed = parse_response(raw, output_tool_name=None, limits=REPORT_LIMITS)
    return encode_json(
        parse_ops_report(parsed.output_json, empirical=empirical).model_dump(mode="json"),
        EMPIRICAL_RESPONSE_LIMITS if empirical else RESPONSE_LIMITS,
    )


def _require_outbound(value: object, *, case: str, snapshot: str) -> None:
    pending = [(value, 0, False)]
    visited = 0
    while pending:
        item, depth, json_field = pending.pop()
        visited += 1
        if depth > 64 or visited > 100_000:
            raise ValueError("outbound nested content exceeds boundary")
        if type(item) is dict:
            # D14 public acknowledgment status, not an answer/ground-truth label.
            # Only this exact typed record is exempt from the broad name match;
            # every value, nested key and free-text field is still scanned below.
            feedback_status = (
                set(item)
                in (
                    {"action_index", "answer_status", "summary", "remaining_question"},
                    {
                        "action_index",
                        "answer_status",
                        "summary",
                        "remaining_question",
                        "content_relation",
                    },
                    {
                        "action_index",
                        "answer_status",
                        "summary",
                        "remaining_question",
                        "content_relation",
                        "question_id",
                    },
                )
                and type(item["action_index"]) is int
                and 1 <= item["action_index"] <= 8
                and type(item["answer_status"]) is str
                and item["answer_status"]
                in {"answered", "partial", "unresolved", "no_observation", "deferred"}
                and type(item["summary"]) is str
                and (
                    "question_id" not in item
                    or (
                        type(item["question_id"]) is str
                        and re.fullmatch(r"iq_[a-z0-9_]{1,32}", item["question_id"]) is not None
                    )
                )
                and (item["remaining_question"] is None or type(item["remaining_question"]) is str)
                and (
                    "content_relation" not in item
                    or item["content_relation"]
                    in {
                        "new_content",
                        "no_new_content",
                        "none_delivered",
                    }
                )
            )
            for key, child in item.items():
                normalized = "".join(
                    c for c in unicodedata.normalize("NFKC", key).casefold() if c.isalnum()
                )
                fixed_false_flag = key == "is_root_cause_probability" and child is False
                if normalized in _PRIVATE or (
                    key not in _PUBLIC_NAMES
                    and not fixed_false_flag
                    and not (feedback_status and key == "answer_status")
                    and _is_forbidden_field_name(key)
                ):
                    raise ValueError("outbound private field")
                if (normalized == "casesnapshotid" and child != snapshot) or (
                    normalized == "opaquecaseid" and child != case
                ):
                    raise ValueError("outbound case or snapshot differs")
                pending.append(
                    (
                        child,
                        depth + 1,
                        key
                        in {
                            "payload_json",
                            "parameters_json",
                            "content",
                        },
                    )
                )
        elif isinstance(item, list | tuple):
            pending.extend((child, depth + 1, False) for child in item)
        elif type(item) is str:
            normalized = unicodedata.normalize("NFKC", item).strip()
            if any(found != case for found in _OPAQUE.findall(normalized)) or (
                _RAW_CASE.search(normalized)
                or normalized.startswith(("/", "\\\\", "file:"))
                or re.match(r"^[a-zA-Z]:[\\/]", normalized)
            ):
                raise ValueError("outbound private path or case identity")
            # Node payloads, tool messages and model proposals contain nested
            # JSON strings. Decode them too; never inspect only the envelope.
            if json_field and item.lstrip().startswith(("{", "[")):
                nested = decode_json_object(
                    ('{"nested":' + item + "}").encode(),
                    CONTEXT_LIMITS,
                )["nested"]
                pending.append((nested, depth + 1, False))


class OpsLiveProvider:
    """KEY2 loaded lazily; two protocols share one lifetime-attempt bridge lock."""

    def __init__(
        self,
        *,
        root: Path,
        code_sha256: str,
        execution_sha256: str,
        approved_execution_sha256: str | None,
        opaque_case_id: str,
        case_snapshot_id: str,
        empirical: bool = False,
        sre: bool = False,
        working: bool = False,
        sre_profile: SreProfileId | None = None,
    ) -> None:
        if approved_execution_sha256 != execution_sha256:
            raise ValueError("isolated ARK requires explicit execution approval")
        if sre and not empirical:
            raise ValueError("SRE requires empirical transport policy")
        if working and not sre:
            raise ValueError("working transport requires SRE")
        for pin in (code_sha256, execution_sha256, case_snapshot_id):
            require_pin(pin)
        if _OPAQUE.fullmatch(opaque_case_id) is None:
            raise ValueError("invalid isolated provider case")
        self.root, self.code, self.execution = root, code_sha256, execution_sha256
        self.case, self.snapshot = opaque_case_id, case_snapshot_id
        self.empirical = empirical
        self.sre = sre
        self.sre_profile = resolve_sre_profile(sre=sre, working=working, profile=sre_profile)
        if self.sre_profile == "hermes-sre-tools/19" and not self._allows_bounded_entry():
            raise ValueError("entry profile requires the separately approved bounded stage")
        if self.sre_profile == "hermes-sre-tools/20" and not self._allows_bounded_observable():
            raise ValueError("observable profile requires the separately approved bounded stage")
        if self.sre_profile == "hermes-sre-tools/21" and not self._allows_bounded_single_owner():
            raise ValueError("single-owner profile requires the separately approved bounded stage")
        if self.sre_profile == "hermes-sre-tools/22" and not self._allows_bounded_typed_purpose():
            raise ValueError("typed-purpose profile requires the separately approved bounded stage")
        if self.sre_profile == "hermes-sre-tools/23" and not self._allows_bounded_atomic_selector():
            raise ValueError(
                "atomic-selector profile requires the separately approved bounded stage"
            )
        if self.sre_profile == "hermes-sre-tools/25" and not self._allows_bounded_focused():
            raise ValueError("focused profile requires the separately approved bounded stage")
        self.working = self.sre_profile == "hermes-sre-tools/5"

    def _allows_bounded_focused(self) -> bool:
        return False

    def _allows_bounded_atomic_selector(self) -> bool:
        return False

    def _allows_bounded_feedback_fact(self) -> bool:
        return False

    def _allows_bounded_typed_purpose(self) -> bool:
        return False

    def _allows_bounded_entry(self) -> bool:
        return False

    def _allows_bounded_observable(self) -> bool:
        return False

    def _allows_bounded_single_owner(self) -> bool:
        return False

    def _send(
        self,
        payload: bytes,
        binding: LiveRequestBinding,
        protocol: str,
        limits: ExchangeLimits,
    ) -> bytes:
        request = decode_json_object(payload, limits)
        body = decode_json_object(request["messages"][1]["content"].encode(), limits)
        if body.get("protocol") != protocol:
            raise ValueError("provider frozen protocol differs")
        context = body["context"]
        outbound = context
        if protocol == SRE_FOCUSED_DECISION_PROTOCOL and (
            self.sre_profile != "hermes-sre-tools/25" or not self._allows_bounded_focused()
        ):
            raise ValueError("focused decisions require the separately granted focused provider")
        if protocol == "nurse.v2.sre-decision-exchange/25":
            from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import (
                bind_entry_decision,
                entry_context_from_wire,
            )

            if not self._allows_bounded_feedback_fact():
                raise ValueError(
                    "feedback presentation requires separately granted bounded provider"
                )
            if binding.decision_index is None or binding != bind_entry_decision(
                payload, case_id=self.case, snapshot=self.snapshot, index=binding.decision_index
            ):
                raise ValueError("feedback presentation provider binding differs")
            context = entry_context_from_wire(
                payload,
                references=None,
                case_id=self.case,
                snapshot=self.snapshot,
                index=binding.decision_index,
            )
            # Include the complete new packet/check and correction, not only the old context.
            outbound = body
        elif protocol in {
            SRE_FOCUSED_DECISION_PROTOCOL,
            "nurse.v2.sre-decision-exchange/24",
            SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
            SRE_SINGLE_OWNER_DECISION_PROTOCOL,
            SRE_ENTRY_DECISION_PROTOCOL,
            SRE_OBSERVABLE_DECISION_PROTOCOL,
            SRE_BRIEF_DECISION_PROTOCOL,
            SRE_DIRECTED_DECISION_PROTOCOL,
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
        }:
            from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire

            assert binding.decision_index is not None
            context = brief_context_from_wire(
                payload,
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=binding.decision_index,
            )
            outbound = context
            if (
                protocol
                in {
                    SRE_FOCUSED_DECISION_PROTOCOL,
                    "nurse.v2.sre-decision-exchange/24",
                    SRE_TYPED_PURPOSE_DECISION_PROTOCOL,
                    SRE_SINGLE_OWNER_DECISION_PROTOCOL,
                    SRE_ENTRY_DECISION_PROTOCOL,
                    SRE_OBSERVABLE_DECISION_PROTOCOL,
                    SRE_USED_SOURCE_DECISION_PROTOCOL,
                    SRE_DIRECTED_DECISION_PROTOCOL,
                    SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
                    SRE_BALANCED_DECISION_PROTOCOL,
                    SRE_REPLANNING_DECISION_PROTOCOL,
                    SRE_QUESTION_DECISION_PROTOCOL,
                    SRE_ACTION_DECISION_PROTOCOL,
                }
                and "correction" in body
            ):
                outbound = dict(context=context, correction=body["correction"])
        elif protocol in SRE_DECISION_PROTOCOLS:
            from nurse_rcl.v2.live.ops_sre_encoding import expand
            from nurse_rcl.v2.live.ops_sre_exchange import sre_context_from_wire

            assert binding.decision_index is not None
            context = sre_context_from_wire(
                payload,
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=binding.decision_index,
                working=self.working,
            )
            outbound = expand(context)
        if protocol == SRE_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_report_exchange import sre_report_body

            context = sre_report_body(payload)["context"]
        if protocol == SRE_NATIVE_REPORT_PROTOCOL:
            from nurse_rcl.v2.live.ops_sre_native_exchange import native_report_body

            native, store = native_report_body(payload)
            context = native["context"]["evidence"]
            outbound = dict(metadata=native["context"], fact_records=store.to_packet())
        if protocol in {
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
            from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_report_body

            brief_body, brief_store = brief_report_body(payload)
            if protocol in {
                SRE_CONDITION_REPORT_PROTOCOL,
                SRE_RECOVERY_REPORT_PROTOCOL,
                SRE_COMPARABLE_REPORT_PROTOCOL,
                SRE_NAVIGATION_REPORT_PROTOCOL,
                SRE_WORKBENCH_REPORT_PROTOCOL,
            } and brief_body["context"]["investigation_input_schema"] != (
                "nurse.v2.sre-investigation-input/12"
                if protocol == SRE_WORKBENCH_REPORT_PROTOCOL
                else "nurse.v2.sre-investigation-input/11"
                if protocol == SRE_NAVIGATION_REPORT_PROTOCOL
                else "nurse.v2.sre-investigation-input/10"
                if protocol == SRE_COMPARABLE_REPORT_PROTOCOL
                else "nurse.v2.sre-investigation-input/9"
                if protocol == SRE_RECOVERY_REPORT_PROTOCOL
                else "nurse.v2.sre-investigation-input/8"
            ):
                raise ValueError("provider rejects diagnostic investigation material")
            context = brief_body["context"]["evidence"]
            outbound = dict(metadata=brief_body["context"], fact_records=brief_store.to_packet())
        if protocol in (REPORT_PROTOCOL, SRE_REPORT_PROTOCOL):
            from nurse_rcl.v2.live.ops_context_view import expand_report_facts

            outbound = {k: v for k, v in context.items() if k != "fact_tables"}
            outbound["facts"] = [f.model_dump(mode="json") for f in expand_report_facts(context)]
        _require_outbound(outbound, case=self.case, snapshot=self.snapshot)
        if protocol in (OPS_REPORT_EXCHANGE_VERSION, REPORT_PROTOCOL, *SRE_REPORT_PROTOCOLS) and (
            context["opaque_case_id"] != self.case or context["case_snapshot_id"] != self.snapshot
        ):
            raise ValueError("outbound report case differs")
        saved = _saved_wire(
            self.root,
            binding,
            code=self.code,
            execution=self.execution,
            protocol=protocol,
            frozen_wire=payload,
        )
        if saved is not None:
            return saved
        settings = ark.load_ark_settings_from_key_file(KEY_PATH)
        if settings.redact_artifact(request) != request:
            raise ValueError("outbound contains credential material")
        for folder in ("outbound", "receipts"):
            (self.root / folder).mkdir(parents=True, exist_ok=True, mode=0o700)
        publish_json(
            self.root / "outbound" / f"request-{binding.slot}.json",
            dict(
                binding=binding.model_dump(mode="json"),
                request=request,
                payload_bytes=len(payload),
                request_protocol=protocol,
                execution_sha256=self.execution,
            ),
        )
        channel = (
            "ark-report"
            if protocol in (OPS_REPORT_EXCHANGE_VERSION, REPORT_PROTOCOL, *SRE_REPORT_PROTOCOLS)
            else ("ark-investigation")
        )
        with LiveArkBridge(
            root=self.root / channel,
            lock_path=LOCK_PATH,
            settings=settings,
            code_sha256=self.code,
            config_sha256=self.execution,
            authorized_case_ids=(self.case,),
            request_protocol=protocol,
        ) as bridge:
            try:
                return bridge.complete(binding, payload)
            finally:
                receipt = bridge.receipt(binding)
                pin = sha256_hex(canonical_json_bytes(receipt))
                publish_json(self.root / "receipts" / f"request-{binding.slot}-{pin}.json", receipt)

    def saved_decision(self, index: int, frozen_wire: bytes) -> bytes | None:
        """Read the profile-bound correction receipt without another send."""
        if self.sre_profile not in {
            "hermes-sre-tools/12",
            "hermes-sre-tools/13",
            "hermes-sre-tools/14",
            "hermes-sre-tools/15",
            "hermes-sre-tools/16",
            "hermes-sre-tools/17",
        }:
            raise ValueError("saved correction requires D21/D22/D23 profile")
        from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision
        from nurse_rcl.v2.live.ops_sre_turn_validation import (
            bind_workbench_turn,
            validate_bound_workbench_response,
        )

        binding = bind_brief_decision(
            frozen_wire,
            opaque_case_id=self.case,
            case_snapshot_id=self.snapshot,
            decision_index=index,
        )
        if binding.phase != "investigation_correction":
            raise ValueError("saved correction phase differs")
        raw = _saved_wire(
            self.root,
            binding,
            code=self.code,
            execution=self.execution,
            protocol=SRE_ACTION_DECISION_PROTOCOL
            if self.sre_profile == "hermes-sre-tools/17"
            else SRE_QUESTION_DECISION_PROTOCOL
            if self.sre_profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"}
            else SRE_REPLANNING_DECISION_PROTOCOL
            if self.sre_profile == "hermes-sre-tools/15"
            else SRE_BALANCED_DECISION_PROTOCOL
            if self.sre_profile == "hermes-sre-tools/14"
            else SRE_RECEIPT_REUSE_DECISION_PROTOCOL
            if self.sre_profile == "hermes-sre-tools/13"
            else SRE_USED_SOURCE_DECISION_PROTOCOL,
            read_only=True,
            frozen_wire=frozen_wire,
        )
        return (
            None
            if raw is None
            else validate_bound_workbench_response(
                raw, bind_workbench_turn(frozen_wire, binding)
            ).output_json
        )

    def decision(self, index: int, context: dict[str, Any] | bytes) -> bytes:
        if self.sre_profile == "hermes-sre-tools/18":
            raise ValueError("directed decisions require the bounded stage provider")
        if self.sre_profile in {
            "hermes-sre-tools/6",
            "hermes-sre-tools/7",
            "hermes-sre-tools/8",
            "hermes-sre-tools/9",
            "hermes-sre-tools/10",
            "hermes-sre-tools/11",
            "hermes-sre-tools/12",
            "hermes-sre-tools/13",
            "hermes-sre-tools/14",
            "hermes-sre-tools/15",
            "hermes-sre-tools/16",
            "hermes-sre-tools/17",
        }:
            from nurse_rcl.v2.live.ops_sre_brief_context import parse_brief_decision
            from nurse_rcl.v2.live.ops_sre_brief_exchange import (
                bind_brief_decision,
                brief_context_from_wire,
            )

            if not isinstance(context, bytes):
                raise ValueError("brief provider requires owner-frozen decision wire")
            binding = bind_brief_decision(
                context,
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=index,
            )
            protocol = (
                SRE_ACTION_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/17"
                else SRE_QUESTION_DECISION_PROTOCOL
                if self.sre_profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"}
                else SRE_REPLANNING_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/15"
                else SRE_BALANCED_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/14"
                else SRE_RECEIPT_REUSE_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/13"
                else SRE_USED_SOURCE_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/12"
                else SRE_WORKBENCH_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/11"
                else SRE_NAVIGATION_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/10"
                else SRE_COMPARABLE_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/9"
                else SRE_RECOVERY_DECISION_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/8"
                else SRE_FEEDBACK_DECISION_PROTOCOL
                if self.sre_profile
                in {
                    "hermes-sre-tools/7",
                    "hermes-sre-tools/8",
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                }
                else SRE_BRIEF_DECISION_PROTOCOL
            )
            raw = self._send(context, binding, protocol, CONTEXT_LIMITS)
            if protocol in {
                SRE_WORKBENCH_DECISION_PROTOCOL,
                SRE_USED_SOURCE_DECISION_PROTOCOL,
                SRE_RECEIPT_REUSE_DECISION_PROTOCOL,
                SRE_BALANCED_DECISION_PROTOCOL,
                SRE_REPLANNING_DECISION_PROTOCOL,
                SRE_QUESTION_DECISION_PROTOCOL,
                SRE_ACTION_DECISION_PROTOCOL,
            }:
                from nurse_rcl.v2.live.ops_sre_turn_validation import (
                    bind_workbench_turn,
                    validate_bound_workbench_response,
                )

                return validate_bound_workbench_response(
                    raw, bind_workbench_turn(context, binding)
                ).output_json
            parsed = parse_response(raw, output_tool_name=None, limits=EMPIRICAL_RESPONSE_LIMITS)
            from nurse_rcl.v2.live.ops_sre_feedback_context import parse_feedback_decision

            parser = (
                parse_feedback_decision
                if self.sre_profile
                in {
                    "hermes-sre-tools/7",
                    "hermes-sre-tools/8",
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                }
                else parse_brief_decision
            )
            parser(
                parsed.output_json,
                context=brief_context_from_wire(
                    context,
                    opaque_case_id=self.case,
                    case_snapshot_id=self.snapshot,
                    decision_index=index,
                ),
            )
            return parsed.output_json
        if self.sre:
            from nurse_rcl.v2.live.ops_sre_context import parse_sre_decision
            from nurse_rcl.v2.live.ops_sre_exchange import (
                bind_sre_decision,
                sre_context_from_wire,
            )

            if not isinstance(context, bytes):
                raise ValueError("SRE provider requires owner-frozen decision wire")
            binding = bind_sre_decision(
                context,
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=index,
                working=self.working,
            )
            raw = self._send(
                context,
                binding,
                SRE_WORKING_DECISION_PROTOCOL if self.working else SRE_DECISION_PROTOCOL,
                CONTEXT_LIMITS,
            )
            parsed = parse_response(raw, output_tool_name=None, limits=EMPIRICAL_RESPONSE_LIMITS)
            parse_sre_decision(
                parsed.output_json,
                context=sre_context_from_wire(
                    context,
                    opaque_case_id=self.case,
                    case_snapshot_id=self.snapshot,
                    decision_index=index,
                    working=self.working,
                ),
            )
            return parsed.output_json
        if isinstance(context, bytes):
            if not self.empirical:
                raise ValueError("frozen wire requires empirical protocol")
            payload = context
            binding = bind_empirical_decision(
                payload,
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=index,
            )
        else:
            payload, binding = prepare_ops_decision(
                opaque_case_id=self.case,
                case_snapshot_id=self.snapshot,
                decision_index=index,
                context=context,
                empirical=self.empirical,
            )
        raw = self._send(
            payload,
            binding,
            DECISION_PROTOCOL if self.empirical else OPS_DECISION_EXCHANGE_VERSION,
            CONTEXT_LIMITS,
        )
        parsed = parse_response(raw, output_tool_name=None, limits=CONTEXT_LIMITS)
        return encode_json(
            parse_decision(parsed.output_json, empirical=self.empirical),
            EMPIRICAL_RESPONSE_LIMITS if self.empirical else DECISION_LIMITS,
        )

    def report(self, context: OpsReportContext | bytes) -> bytes:
        if self.sre_profile in {
            "hermes-sre-tools/18",
            "hermes-sre-tools/19",
            "hermes-sre-tools/20",
            "hermes-sre-tools/21",
            "hermes-sre-tools/22",
            "hermes-sre-tools/23",
            "hermes-sre-tools/25",
        }:
            raise ValueError("directed report is not admitted in the decision stage")
        if self.sre_profile in {
            "hermes-sre-tools/6",
            "hermes-sre-tools/7",
            "hermes-sre-tools/8",
            "hermes-sre-tools/9",
            "hermes-sre-tools/10",
            "hermes-sre-tools/11",
            "hermes-sre-tools/12",
            "hermes-sre-tools/13",
            "hermes-sre-tools/14",
            "hermes-sre-tools/15",
            "hermes-sre-tools/16",
            "hermes-sre-tools/17",
        }:
            from nurse_rcl.v2.live.ops_sre_brief_claims import parse_brief_report
            from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_report

            if not isinstance(context, bytes):
                raise ValueError("brief provider requires owner-frozen report wire")
            binding = bind_brief_report(context)
            if binding.opaque_case_id != self.case:
                raise ValueError("brief frozen report case differs")
            protocol = (
                SRE_ACTION_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/17"
                else SRE_QUESTION_REPORT_PROTOCOL
                if self.sre_profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"}
                else SRE_REPLANNING_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/15"
                else SRE_BALANCED_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/14"
                else SRE_RECEIPT_REUSE_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/13"
                else SRE_USED_SOURCE_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/12"
                else SRE_WORKBENCH_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/11"
                else SRE_NAVIGATION_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/10"
                else SRE_COMPARABLE_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/9"
                else SRE_RECOVERY_REPORT_PROTOCOL
                if self.sre_profile == "hermes-sre-tools/8"
                else SRE_CONDITION_REPORT_PROTOCOL
                if self.sre_profile
                in {
                    "hermes-sre-tools/7",
                    "hermes-sre-tools/8",
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                }
                else SRE_BRIEF_REPORT_PROTOCOL
            )
            raw = self._send(context, binding, protocol, REPORT_LIMITS)
            parsed = parse_response(raw, output_tool_name=None, limits=EMPIRICAL_RESPONSE_LIMITS)
            parse_brief_report(
                parsed.output_json,
                comparable_evidence=self.sre_profile
                in {
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                    "hermes-sre-tools/14",
                    "hermes-sre-tools/15",
                    "hermes-sre-tools/16",
                    "hermes-sre-tools/17",
                },
                condition_explanations=self.sre_profile
                in {
                    "hermes-sre-tools/7",
                    "hermes-sre-tools/8",
                    "hermes-sre-tools/9",
                    "hermes-sre-tools/10",
                    "hermes-sre-tools/11",
                    "hermes-sre-tools/12",
                    "hermes-sre-tools/13",
                    "hermes-sre-tools/14",
                    "hermes-sre-tools/15",
                    "hermes-sre-tools/16",
                    "hermes-sre-tools/17",
                },
            )
            return parsed.output_json
        if self.sre:
            from nurse_rcl.v2.live.ops_sre_report import parse_sre_report
            from nurse_rcl.v2.live.ops_sre_report_exchange import bind_sre_report

            if not isinstance(context, bytes):
                raise ValueError("SRE provider requires owner-frozen report wire")
            if self.working:
                from nurse_rcl.v2.live.ops_sre_native_exchange import bind_native_report

                binding = bind_native_report(context)
            else:
                binding = bind_sre_report(context)
            if binding.opaque_case_id != self.case:
                raise ValueError("SRE frozen report case differs")
            raw = self._send(
                context,
                binding,
                SRE_NATIVE_REPORT_PROTOCOL if self.working else SRE_REPORT_PROTOCOL,
                REPORT_LIMITS,
            )
            parsed = parse_response(raw, output_tool_name=None, limits=EMPIRICAL_RESPONSE_LIMITS)
            parse_sre_report(parsed.output_json)
            return parsed.output_json
        if isinstance(context, bytes):
            if not self.empirical:
                raise ValueError("frozen wire requires empirical protocol")
            payload, binding = context, bind_empirical_report(context)
            if binding.opaque_case_id != self.case:
                raise ValueError("frozen report case differs")
        else:
            payload, binding = prepare_ops_report(context, empirical=self.empirical)
        raw = self._send(
            payload,
            binding,
            REPORT_PROTOCOL if self.empirical else OPS_REPORT_EXCHANGE_VERSION,
            REPORT_LIMITS,
        )
        parsed = parse_response(raw, output_tool_name=None, limits=REPORT_LIMITS)
        return encode_json(
            parse_ops_report(parsed.output_json, empirical=self.empirical).model_dump(mode="json"),
            EMPIRICAL_RESPONSE_LIMITS if self.empirical else RESPONSE_LIMITS,
        )


def saved_sre_report(
    root: Path,
    frozen_wire: bytes,
    *,
    code_sha256: str,
    execution_sha256: str,
    working: bool = False,
    read_only: bool = False,
    sre_profile: SreProfileId | None = None,
) -> bytes | None:
    """Keyless recovery of exactly the durable report owner's frozen request."""
    from nurse_rcl.v2.live.ops_sre_report import parse_sre_report
    from nurse_rcl.v2.live.ops_sre_report_exchange import bind_sre_report

    profile = resolve_sre_profile(sre=True, working=working, profile=sre_profile)
    if profile in {
        "hermes-sre-tools/6",
        "hermes-sre-tools/7",
        "hermes-sre-tools/8",
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
    }:
        from nurse_rcl.v2.live.ops_sre_brief_claims import parse_brief_report
        from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_report

        binding = bind_brief_report(frozen_wire)
        protocol = (
            SRE_ACTION_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/17"
            else SRE_QUESTION_REPORT_PROTOCOL
            if profile in {"hermes-sre-tools/16", "hermes-sre-tools/17"}
            else SRE_REPLANNING_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/15"
            else SRE_BALANCED_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/14"
            else SRE_RECEIPT_REUSE_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/13"
            else SRE_USED_SOURCE_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/12"
            else SRE_WORKBENCH_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/11"
            else SRE_NAVIGATION_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/10"
            else SRE_COMPARABLE_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/9"
            else SRE_RECOVERY_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/8"
            else SRE_CONDITION_REPORT_PROTOCOL
            if profile == "hermes-sre-tools/7"
            else SRE_BRIEF_REPORT_PROTOCOL
        )
    elif profile == "hermes-sre-tools/5":
        from nurse_rcl.v2.live.ops_sre_native_exchange import bind_native_report

        binding = bind_native_report(frozen_wire)
        protocol = SRE_NATIVE_REPORT_PROTOCOL
    else:
        binding = bind_sre_report(frozen_wire)
        protocol = SRE_REPORT_PROTOCOL
    raw = _saved_wire(
        root,
        binding,
        code=code_sha256,
        execution=execution_sha256,
        protocol=protocol,
        read_only=read_only,
    )
    if raw is None:
        return None
    parsed = parse_response(raw, output_tool_name=None, limits=EMPIRICAL_RESPONSE_LIMITS)
    if profile in {
        "hermes-sre-tools/6",
        "hermes-sre-tools/7",
        "hermes-sre-tools/8",
        "hermes-sre-tools/9",
        "hermes-sre-tools/10",
        "hermes-sre-tools/11",
        "hermes-sre-tools/12",
        "hermes-sre-tools/13",
        "hermes-sre-tools/14",
        "hermes-sre-tools/15",
        "hermes-sre-tools/16",
        "hermes-sre-tools/17",
    }:
        parse_brief_report(
            parsed.output_json,
            comparable_evidence=profile
            in {
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            },
            condition_explanations=profile
            in {
                "hermes-sre-tools/7",
                "hermes-sre-tools/8",
                "hermes-sre-tools/9",
                "hermes-sre-tools/10",
                "hermes-sre-tools/11",
                "hermes-sre-tools/12",
                "hermes-sre-tools/13",
                "hermes-sre-tools/14",
                "hermes-sre-tools/15",
                "hermes-sre-tools/16",
                "hermes-sre-tools/17",
            },
        )
    else:
        parse_sre_report(parsed.output_json)
    return parsed.output_json
