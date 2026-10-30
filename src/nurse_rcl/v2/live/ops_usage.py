"""Pure reported-usage views for empirical cost analysis, not send admission.

The owner supplies one identity-checked inventory from existing durable attempts,
under the transport lock. This module has no file, credential or network access,
no remaining balance and no dependency on the historical hard-budget ledger.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from math import ceil
from statistics import median
from typing import Any

from nurse_rcl.v2.dynamic_verification.safety import require_pin
from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus


def usage_parts(raw_usage: object) -> dict[str, Any]:
    """Parse reported parts independently; never infer input from output/total.

An inconsistent cache subset makes input uncertain (as in the old component),
not discounted. Invalid output/total does not poison independently valid input.
Raw usage is retained so invalid or conflicting fields are not lost or repaired.
"""
    usage = raw_usage if type(raw_usage) is dict else {}
    issues: list[str] = []

    def part(mapping: dict[str, Any], names: tuple[str, ...], label: str) -> int | None:
        values = [mapping[key] for key in names if key in mapping]
        if not values:
            issues.append(label + "_missing")
        elif any(type(value) is not int or value < 0 for value in values):
            issues.append(label + "_invalid")
        elif len(set(values)) != 1:
            issues.append(label + "_conflict")
        else:
            return int(values[0])
        return None

    input_count = part(usage, ("prompt_tokens", "input_tokens"), "input")
    output_count = part(usage, ("completion_tokens", "output_tokens"), "output")
    total_count = part(usage, ("total_tokens",), "total")
    cached_values = {}
    for key in ("prompt_tokens_details", "input_tokens_details"):
        if key not in usage:
            continue
        details = usage[key]
        if type(details) is not dict:
            issues.append("cached_invalid")
        elif "cached_tokens" in details:
            cached_values[key] = details["cached_tokens"]
    cached_count = part(cached_values, tuple(cached_values), "cached")
    if input_count is not None and cached_count is not None and cached_count > input_count:
        issues.append("cached_exceeds_input")
    if any(code in issues for code in (
        "cached_invalid", "cached_conflict", "cached_exceeds_input",
    )):
        input_count = cached_count = None
    if total_count is not None:
        if input_count is not None and output_count is not None:
            if total_count != input_count + output_count:
                issues.append("total_inconsistent")
                total_count = None
        elif input_count is not None and total_count < input_count:
            issues.append("total_below_input")
            total_count = None
        elif output_count is not None and total_count < output_count:
            issues.append("total_below_output")
            total_count = None
    return dict(input_tokens=input_count, output_tokens=output_count,
                total_tokens=total_count, cached_tokens=cached_count,
                usage_issues=tuple(sorted(set(issues))), raw_usage=deepcopy(raw_usage))


def summarize_attempts(
    attempts: Iterable[Mapping[str, Any]], *, confirmed_no_send: bool = False,
) -> dict[str, Any]:
    """Derive cost once per (phase, slot, attempt); outcomes never filter rows.

Rows carry RAW ``usage`` (None if unavailable). Identity/content validation of
saved responses is the caller's responsibility, not replaced by this aggregation.
An empty inventory alone cannot establish zero spend: the trusted owner must
confirm a terminal execution with no sends. Re-reading has no side effects.
"""
    identities: set[tuple[str, int, int]] = set()
    slots: set[tuple[str, int]] = set()
    fields = ("input", "output", "total", "cached")
    counts: dict[str, list[int | None]] = {key: [] for key in fields}
    for row in attempts:
        phase, slot, number = row.get("phase"), row.get("slot"), row.get("attempt")
        if phase not in (
            "investigation", "investigation_correction", "report"
        ) or type(slot) is not int or (
            not 1 <= slot <= (8 if phase == "investigation" else 1)
        ) or type(number) is not int or not 1 <= number <= 5:
            raise ValueError("empirical attempt identity differs")
        identity = (phase, slot, number)
        if identity in identities:
            raise ValueError("duplicate empirical attempt identity")
        identities.add(identity)
        slots.add((phase, slot))
        parts = usage_parts(row.get("usage"))
        for field in fields:
            counts[field].append(parts[field + "_tokens"])
    if type(confirmed_no_send) is not bool or (confirmed_no_send and identities):
        raise ValueError("no-send confirmation conflicts with attempt inventory")
    has_inventory = bool(identities) or confirmed_no_send
    result: dict[str, Any] = dict(attempts=len(identities), logical_requests=len(slots))
    for field, values in counts.items():
        subtotal = sum(value for value in values if value is not None)
        unknown = sum(value is None for value in values)
        result["reported_" + field + "_tokens"] = subtotal
        result["unknown_" + field + "_attempts"] = unknown
        result[field + "_tokens"] = subtotal if has_inventory and not unknown else None
    result["input_usage_complete"] = has_inventory and not result["unknown_input_attempts"]
    result["usage_complete"] = has_inventory and not any(
        result["unknown_" + field + "_attempts"] for field in ("input", "output", "total")
    )
    return result


def summarize_cost(
    cases: Iterable[Mapping[str, Any]], *, planned_case_ids: Iterable[str],
    dataset: str, version: str,
) -> dict[str, Any]:
    """Aggregate admitted per-execution attempt views, never select on rank/labels.

    Caller verifies saved execution identities before creating these projections.
    Lifetime attempts include paid responses reused during recovery. Optional
    incremental IDs describe only the invocation being measured; their absence
    means unknown, not zero. This is a pure derived view, never a second ledger.
    """
    planned = tuple(planned_case_ids)
    if not planned or any(type(k) is not str or not k for k in planned) or (
        len(set(planned)) != len(planned) or dataset not in {"rcabench", "aiops25"}
        or type(version) is not str or not version
    ):
        raise ValueError("cost planned scope differs")
    terminal_states = {"completed", "completed_partial_report", "core_completed_report_unavailable",
                       "failed", "interrupted"}
    seen, executions = set(), set()
    summaries, terminal, successful = [], [], []
    increments: list[int | None] = []
    reuses = 0
    for row in cases:
        key = row.get("case_id")
        if key not in planned or key in seen or (row.get("dataset"), row.get("version")) != (
            dataset, version,
        ):
            raise ValueError("cost case scope differs or duplicates")
        seen.add(key)
        if row["case_snapshot_id"] is not None:
            require_pin(row["case_snapshot_id"])
        else:
            # A failure before preparation has a pinned input key, not a made-up snapshot.
            require_pin(row["preparation_key_sha256"])
            if row["channels"]["core"] == "completed":
                raise ValueError("completed core cost requires snapshot identity")
        execution = row["execution_sha256"]
        require_pin(execution)
        if execution in executions:
            raise ValueError("duplicate paid execution source")
        executions.add(execution)
        state = row.get("status")
        if state not in terminal_states | {"running", "pending"}:
            raise ValueError("cost case status differs")
        channels = OpsChannelStatus.model_validate(row["channels"])
        attempts = tuple(row["attempts"])
        summary = summarize_attempts(attempts,
                                     confirmed_no_send=row.get("confirmed_no_send", False))
        summaries.append(summary)
        if state in terminal_states:
            terminal.append(summary)
            if state == "completed" and all(
                value == "completed" for value in channels.model_dump().values()
            ):
                successful.append(summary)
        reused = row.get("response_reuses", 0)
        if type(reused) is not int or reused < 0:
            raise ValueError("response reuse count differs")
        reuses += reused
        selected = row.get("incremental_attempt_ids")
        if selected is None:
            increments.append(None)
            continue
        selected = tuple(tuple(identity) for identity in selected)
        by_id = {(a["phase"], a["slot"], a["attempt"]): a for a in attempts}
        if len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
            raise ValueError("incremental attempt identity differs")
        increments.append(summarize_attempts(
            (by_id[key] for key in selected), confirmed_no_send=not selected)["input_tokens"])
    count = len(planned)
    complete = len(terminal) == count
    known = [s["input_tokens"] for s in terminal if s["input_usage_complete"]]
    known.sort()
    full = complete and len(known) == count
    success_known = [s["input_tokens"] for s in successful if s["input_usage_complete"]]
    result: dict[str, Any] = dict(
        dataset=dataset, version=version, planned_case_ids=planned,
        planned_cases=count, started_cases=len(seen), completed_cases=len(terminal),
        failed_cases=len(terminal) - len(successful), successful_cases=len(successful),
        failure_rate=(len(terminal) - len(successful)) / count if complete else None,
        input_complete_cases=len(known),
        usage_complete_cases=sum(s["usage_complete"] for s in terminal),
        input_mean=sum(known) / count if full else None,
        input_target_met=sum(known) / count < 100000 if full else None,
        successful_input_mean=sum(success_known) / len(successful)
        if successful and len(success_known) == len(successful) else None,
        input_distribution=dict(count=len(known), median=median(known) if known else None,
            p95=known[ceil(0.95 * len(known)) - 1] if known else None,
            maximum=max(known) if known else None, p95_method="nearest_rank"),
        incremental_input_tokens=sum(v for v in increments if v is not None)
        if len(increments) == count and all(v is not None for v in increments) else None,
        paid_execution_sources=sorted(executions), response_reuses=reuses,
        spend_scope="execution_lifetime_including_reused_paid_responses",
        attempts=sum(s["attempts"] for s in summaries),
    )
    for field in ("input", "output", "total", "cached"):
        for name in ("reported_" + field + "_tokens", "unknown_" + field + "_attempts"):
            result[name] = sum(s[name] for s in summaries)
    result["known_input_lower_bound_per_planned_case"] = result["reported_input_tokens"] / count
    return result
