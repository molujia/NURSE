"""Versioned early-semantic query branch over owned normal preparation.

The selector sees only the fixed catalog, admitted intents and its own feedback.
Paired B0 receipts are consulted after a choice and a durable logical charge.
No source replay, scoring, ARK, or legacy selection relabelling occurs here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, Field, TypeAdapter

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalQuerySession
from nurse_rcl.v2.live.normal_queries import (
    _consume_blocks,
    _feedback_binding,
    _measurement_binding,
    _read_bytes,
    _seal,
)
from nurse_rcl.v2.live.normal_tool_budget import open_normal_branch_budget
from nurse_rcl.v2.live.repair_driver import _fsync
from nurse_rcl.v2.live.repair_measurements import RepairSignalMeasurement, _measure_values
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_preparation_codec import _decode_json_member
from nurse_rcl.v2.live.repair_queries import (
    RepairQueryCatalog,
    RepairQueryFeedback,
    RepairQuerySelection,
    _feedback_values,
    _select_values,
)
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_safety import REPAIR_MAX_JSON_BYTES
from nurse_rcl.v2.live.staged_case import require_plain_path

_M = TypeVar("_M", bound=BaseModel)


class HybridQueryIntent(StrictFrozenModel):
    """A compiled query preference, never a measurement or execution authority."""

    request_key: str = Field(pattern=r"^request_[0-9a-f]{64}$")
    need_keys: tuple[str, ...] = Field(min_length=1)


class HybridQueryChoice(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-query-choice/1"] = (
        "nurse.v2.normal-hybrid-query-choice/1"
    )
    intents_sha256: SHA256
    baseline_selection: RepairQuerySelection
    selected_request_key: str | None
    semantic_need_keys: tuple[str, ...]
    choice_sha256: SHA256


class HybridQueryCompletedStep(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-query-step/1"] = (
        "nurse.v2.normal-hybrid-query-step/1"
    )
    branch_identity_sha256: SHA256
    sequence: int = Field(ge=0, le=7, strict=True)
    choice: HybridQueryChoice
    measurement: RepairSignalMeasurement
    feedback: RepairQueryFeedback
    origin: Literal["paired_base", "executed"]
    step_sha256: SHA256


class NormalHybridQuerySession(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-query-session/1"] = (
        "nurse.v2.normal-hybrid-query-session/1"
    )
    branch_identity_sha256: SHA256
    paired_base_session_sha256: SHA256
    catalog_sha256: SHA256
    intents_sha256: SHA256
    steps: tuple[HybridQueryCompletedStep, ...]
    submitted_request_keys: tuple[str, ...]
    paid_count: int = Field(ge=0, le=8, strict=True)
    reused_count: int = Field(ge=0, le=8, strict=True)
    executed_count: int = Field(ge=0, le=8, strict=True)
    physical_union_count: int = Field(ge=0, le=16, strict=True)
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"]
    terminal_choice: HybridQueryChoice
    session_sha256: SHA256


def _intent_identity(catalog: RepairQueryCatalog, intents: tuple[HybridQueryIntent, ...]) -> str:
    requests = {r.request_key: r for r in catalog.requests}
    if type(intents) is not tuple or len(intents) > 8 or len({
        i.request_key for i in intents
    }) != len(intents):
        raise ValueError("hybrid intents must be unique and bounded")
    for intent in intents:
        request = requests.get(intent.request_key)
        if type(intent) is not HybridQueryIntent or request is None or (
            request.candidate_service_key not in catalog.candidate_service_keys
            or not intent.need_keys or len(set(intent.need_keys)) != len(intent.need_keys)
            or not set(intent.need_keys) <= {n.need_key for n in request.needs if n.queryable}
        ):
            raise ValueError("hybrid intent lacks a legal queryable catalog need")
    return sha256_hex(canonical_json_bytes(dict(
        schema_version="nurse.v2.normal-hybrid-query-intents/1",
        catalog_sha256=catalog.catalog_sha256,
        intents=[i.model_dump(mode="json") for i in intents],
    )))


def _choose_values(
    catalog: RepairQueryCatalog, feedback: tuple[RepairQueryFeedback, ...],
    submitted: tuple[str, ...], intents: tuple[HybridQueryIntent, ...],
) -> HybridQueryChoice:
    identity = _intent_identity(catalog, intents)
    baseline = _select_values(catalog, feedback, submitted)
    requests = {r.request_key: r for r in catalog.requests}
    true_conditions = {
        need.condition_key for item in feedback
        for need, predicate in zip(requests[item.request_key].needs, item.predicates, strict=True)
        if predicate.truth_value == "true"
    }
    desired = {i.request_key: set(i.need_keys) for i in intents}
    unresolved = {
        r.request_key: tuple(n.need_key for n in r.needs
                             if n.need_key in desired.get(r.request_key, set())
                             and n.condition_key not in true_conditions)
        for r in catalog.requests
    }
    # The original tier, especially real temporal tier zero, remains primary.
    # No future measurement, B0 outcome, model number or rank enters this key.
    selected = min(
        (r for r in baseline.requests if r.eligible),
        key=lambda r: (cast(int, r.demand_tier), not bool(unresolved[r.request_key]),
                       *cast(tuple[int, int, int, str, str], r.priority)[1:]),
        default=None,
    )
    return _seal(HybridQueryChoice(
        intents_sha256=identity, baseline_selection=baseline,
        selected_request_key=None if selected is None else selected.request_key,
        semantic_need_keys=() if selected is None else unresolved[selected.request_key],
        choice_sha256="0" * 64,
    ), "choice_sha256")


def _publish_value(path: Path, value: BaseModel) -> None:
    raw = canonical_json_bytes(value.model_dump(mode="json"))
    if len(raw) > REPAIR_MAX_JSON_BYTES:
        raise ValueError("hybrid receipt exceeds capacity")
    descriptor, name = tempfile.mkstemp(prefix=".hybrid-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_step(path: Path, step: HybridQueryCompletedStep) -> None:
    _publish_value(path, step)


def _load(path: Path, cls: type[_M], field: str) -> _M:
    raw = _read_bytes(path, REPAIR_MAX_JSON_BYTES)
    _decode_json_member(raw)
    value = cls.model_validate_json(raw)
    if _seal(value, field) != value or canonical_json_bytes(value.model_dump(mode="json")) != raw:
        raise ValueError("hybrid saved receipt pin differs")
    return value


def _choice_binding(
    choice: HybridQueryChoice, catalog: RepairQueryCatalog, intent_pin: str,
    steps: tuple[HybridQueryCompletedStep, ...],
) -> None:
    baseline = choice.baseline_selection
    keys = tuple(sorted(s.feedback.request_key for s in steps))
    if (
        _seal(choice, "choice_sha256") != choice
        or _seal(baseline, "result_sha256") != baseline
        or choice.intents_sha256 != intent_pin
        or baseline.catalog_sha256 != catalog.catalog_sha256
        or baseline.submitted_request_keys != keys or baseline.paid_count != len(steps)
        or baseline.remaining_budget != 8 - len(steps)
        or baseline.feedback_sha256s != tuple(s.feedback.result_sha256 for s in steps)
        or tuple(r.request_key for r in baseline.requests) != tuple(
            r.request_key for r in catalog.requests
        )
    ):
        raise ValueError("hybrid choice prefix/identity differs")
    key = choice.selected_request_key
    if key is None:
        if baseline.stop_reason is None or any(r.eligible for r in baseline.requests) or (
            (baseline.stop_reason == "query_budget_exhausted") != (len(steps) == 8)
        ):
            raise ValueError("hybrid stop differs from bound eligible catalog")
    elif baseline.stop_reason is not None or key in keys or len(steps) >= 8 or not any(
        r.request_key == key and r.eligible for r in baseline.requests
    ):
        raise ValueError("hybrid selected request is not eligible")


def _step_binding(
    step: HybridQueryCompletedStep, data: RepairPreparationData, identity: str,
    intent_pin: str, steps: tuple[HybridQueryCompletedStep, ...], base: NormalQuerySession,
) -> None:
    _choice_binding(step.choice, data.catalog, intent_pin, steps)
    key = step.choice.selected_request_key
    if step.sequence != len(steps) or step.branch_identity_sha256 != identity or (
        key is None or step.feedback.request_key != key
        or _seal(step, "step_sha256") != step
        or _seal(step.measurement, "measurement_sha256") != step.measurement
        or _seal(step.feedback, "result_sha256") != step.feedback
        or any(_seal(p, "result_sha256") != p for p in step.feedback.predicates)
    ):
        raise ValueError("hybrid step identity/sequence differs")
    request = next(r for r in data.catalog.requests if r.request_key == key)
    source = next(s for s in data.sources if s.source_sha256 == request.source_sha256)
    _measurement_binding(step.measurement, source, data.catalog)
    _feedback_binding(step.feedback, step.measurement, source, data.catalog, key)
    saved = next((s for s in base.steps if s.feedback.request_key == key), None)
    if (saved is not None) != (step.origin == "paired_base") or (
        saved is not None and (
            step.measurement != saved.measurement or step.feedback != saved.feedback
        )
    ):
        raise ValueError("hybrid selected reuse differs from paired base")


def run_normal_hybrid_queries(
    *, data: RepairPreparationData, ledger_root: Path, block_root: Path,
    paired_base_session: NormalQuerySession, expected_base_session_sha256: str,
    branch_identity_sha256: str, intents: tuple[HybridQueryIntent, ...],
) -> NormalHybridQuerySession:
    """Verification branch; card requests share the ledger but are not dispatched here.

    Inputs are owner-admitted preparation and B0. Recovery checks saved boundaries
    once, never repeats a measurement. A charged request without a durable result
    stops for diagnosis, including when the potential result was a B0 reuse.
    """
    TypeAdapter(SHA256).validate_python(branch_identity_sha256)
    intent_pin = _intent_identity(data.catalog, intents)
    if paired_base_session.catalog_sha256 != data.catalog.catalog_sha256:
        raise ValueError("hybrid paired base catalog identity differs")
    identity = sha256_hex(canonical_json_bytes(dict(
        schema_version="nurse.v2.normal-hybrid-query-branch/1",
        branch_identity_sha256=branch_identity_sha256, intents_sha256=intent_pin,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
    )))
    require_plain_path(ledger_root.parent)
    require_plain_path(block_root)
    ledger_root.mkdir(mode=0o700, exist_ok=True)
    require_plain_path(ledger_root)
    with open_normal_branch_budget(
        root=ledger_root / "budget", paired_base_session=paired_base_session,
        expected_base_session_sha256=expected_base_session_sha256,
        branch_identity_sha256=identity,
    ) as budget:
        steps: tuple[HybridQueryCompletedStep, ...] = ()
        allowed = {"budget", "stop.json", *(f"{i:02d}.json" for i in range(len(budget.records)))}
        if {p.name for p in ledger_root.iterdir()} - allowed:
            raise ValueError("hybrid journal has unexpected members")
        for index, (kind, recorded_key, pin) in enumerate(budget.records):
            if kind != "verification":
                raise ValueError("hybrid verification runner cannot consume a card receipt")
            path = ledger_root / f"{index:02d}.json"
            if not path.exists():
                raise ValueError("hybrid pending reservation without a result receipt")
            step = _load(path, HybridQueryCompletedStep, "step_sha256")
            _step_binding(step, data, identity, intent_pin, steps, paired_base_session)
            if step.feedback.request_key != recorded_key or pin not in (
                None, step.feedback.result_sha256,
            ):
                raise ValueError("hybrid budget result reference differs")
            if pin is None:
                budget.complete(kind="verification", query_key=recorded_key,
                                result_sha256=step.feedback.result_sha256)
            steps += (step,)
        stop_path = ledger_root / "stop.json"
        terminal = None
        if stop_path.exists() or stop_path.is_symlink():
            terminal = _load(stop_path, HybridQueryChoice, "choice_sha256")
            _choice_binding(terminal, data.catalog, intent_pin, steps)
            if terminal.selected_request_key is not None:
                raise ValueError("hybrid saved terminal is not stopped")
        admitted: set[str] = set()
        while terminal is None:
            with query_step("choice"):
                choice = _choose_values(
                    data.catalog, tuple(s.feedback for s in steps),
                    tuple(sorted(s.feedback.request_key for s in steps)), intents,
                )
            key = choice.selected_request_key
            if key is None:
                _publish_value(stop_path, choice)
                terminal = choice
                break
            if budget.reserve(kind="verification", query_key=key).status != "reserved":
                raise ValueError("hybrid next query lacks a fresh logical reservation")
            # Only now may the selected paired result become visible to B2.
            saved = next(
                (s for s in paired_base_session.steps if s.feedback.request_key == key), None,
            )
            request = next(r for r in data.catalog.requests if r.request_key == key)
            source = next(s for s in data.sources if s.source_sha256 == request.source_sha256)
            if saved is not None:
                with query_step("input_guard"):
                    measured, feedback = saved.measurement, saved.feedback
                    _measurement_binding(measured, source, data.catalog)
                    _feedback_binding(feedback, measured, source, data.catalog, key)
            else:
                with query_step("measurement"):
                    _consume_blocks(source, data, block_root, admitted)
                    measured = _measure_values(source)
                with query_step("feedback"):
                    feedback = _feedback_values(data.catalog, source, measured)
            step = _seal(HybridQueryCompletedStep(
                branch_identity_sha256=identity, sequence=len(steps), choice=choice,
                measurement=measured, feedback=feedback,
                origin="paired_base" if saved is not None else "executed", step_sha256="0" * 64,
            ), "step_sha256")
            _publish_step(ledger_root / f"{len(steps):02d}.json", step)
            budget.complete(kind="verification", query_key=key,
                            result_sha256=feedback.result_sha256)
            steps += (step,)
        stop = terminal.baseline_selection.stop_reason
        if stop is None:
            raise ValueError("hybrid branch lacks a terminal reason")
        executed = sum(s.origin == "executed" for s in steps)
        return _seal(NormalHybridQuerySession(
            branch_identity_sha256=identity,
            paired_base_session_sha256=expected_base_session_sha256,
            catalog_sha256=data.catalog.catalog_sha256, intents_sha256=intent_pin,
            steps=steps,
            submitted_request_keys=tuple(sorted(s.feedback.request_key for s in steps)),
            paid_count=budget.used_count, reused_count=len(steps) - executed,
            executed_count=executed,
            physical_union_count=paired_base_session.paid_count + executed,
            stop_reason=stop, terminal_choice=terminal, session_sha256="0" * 64,
        ), "session_sha256")
