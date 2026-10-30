"""Current fixed-card phase and feedback verification share one durable budget.

The old verification-only session is unchanged. Cards reveal fixed preparation,
not B0's later results. Verification continues to reselect after each measured
feedback. A reserved operation with no durable receipt stops, never refunds.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field, TypeAdapter

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.normal_artifacts import NormalAssurance, NormalQuerySession
from nurse_rcl.v2.live.normal_hybrid_queries import (
    HybridQueryChoice,
    HybridQueryCompletedStep,
    HybridQueryIntent,
    _choice_binding,
    _choose_values,
    _intent_identity,
    _load,
    _publish_value,
    _step_binding,
)
from nurse_rcl.v2.live.normal_queries import (
    _consume_blocks,
    _feedback_binding,
    _measurement_binding,
    _seal,
)
from nurse_rcl.v2.live.normal_scoring import _score_ranking
from nurse_rcl.v2.live.normal_semantic_context import (
    SemanticError,
    SemanticSignal,
    _project_fixed_context,
)
from nurse_rcl.v2.live.normal_tool_budget import open_normal_branch_budget
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_measurements import _measure_values
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.live.repair_queries import RepairQueryRequest, _feedback_values
from nurse_rcl.v2.live.repair_query_events import query_step
from nurse_rcl.v2.live.repair_ranking import RepairRanking
from nurse_rcl.v2.live.staged_case import require_plain_path
from nurse_rcl.v2.observation_schemas import TimeInterval


class NormalSemanticCard(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-semantic-card/1"] = "nurse.v2.normal-semantic-card/1"
    opaque_case_id: str
    catalog_sha256: SHA256
    candidate_service_key: str
    baseline_window: TimeInterval
    fault_window: TimeInterval
    signals: tuple[SemanticSignal, ...]
    errors: tuple[SemanticError, ...]
    requests: tuple[RepairQueryRequest, ...]
    scope: Literal["fixed_preparation_only"] = "fixed_preparation_only"
    score_authority: Literal[False] = False
    limitations: tuple[str, ...]
    card_sha256: SHA256


class NormalSemanticCardReceipt(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-semantic-card-receipt/1"] = (
        "nurse.v2.normal-semantic-card-receipt/1"
    )
    branch_identity_sha256: SHA256
    sequence: int = Field(ge=0, le=7, strict=True)
    card: NormalSemanticCard
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)
    receipt_sha256: SHA256


class _ToolStop(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-tool-stop/1"] = (
        "nurse.v2.normal-hybrid-tool-stop/1"
    )
    branch_identity_sha256: SHA256
    card_receipt_sha256s: tuple[SHA256, ...]
    step_sha256s: tuple[SHA256, ...]
    paid_count: int = Field(ge=0, le=8, strict=True)
    stop_reason: Literal["tool_budget_exhausted", "no_admissible_progress"]
    terminal_verification_choice: HybridQueryChoice | None
    stop_sha256: SHA256


class NormalHybridToolSession(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-tool-session/1"] = (
        "nurse.v2.normal-hybrid-tool-session/1"
    )
    branch_identity_sha256: SHA256
    paired_base_session_sha256: SHA256
    catalog_sha256: SHA256
    intents_sha256: SHA256
    cards: tuple[NormalSemanticCardReceipt, ...]
    steps: tuple[HybridQueryCompletedStep, ...]
    submitted_request_keys: tuple[str, ...]
    paid_count: int = Field(ge=0, le=8, strict=True)
    card_count: int = Field(ge=0, le=8, strict=True)
    verification_count: int = Field(ge=0, le=8, strict=True)
    reused_count: int = Field(ge=0, le=8, strict=True)
    executed_count: int = Field(ge=0, le=8, strict=True)
    physical_union_count: int = Field(ge=0, le=16, strict=True)
    stop_reason: Literal["tool_budget_exhausted", "no_admissible_progress"]
    stop: _ToolStop
    session_sha256: SHA256


def _card_values(data: RepairPreparationData, candidate: str) -> NormalSemanticCard:
    context = _project_fixed_context(data, {candidate})
    return _seal(NormalSemanticCard(
        opaque_case_id=context.opaque_case_id, catalog_sha256=context.catalog_sha256,
        candidate_service_key=candidate, baseline_window=context.baseline_window,
        fault_window=context.fault_window, signals=context.signals, errors=context.errors,
        requests=context.requests,
        limitations=("fixed_summaries_not_persistence_onset_or_health",) + (
            ("no_retained_signal_or_error_details",) if not context.signals and not context.errors
            else ()
        ), card_sha256="0" * 64,
    ), "card_sha256")


def _execute_card(
    data: RepairPreparationData, candidate: str, identity: str, sequence: int,
) -> NormalSemanticCardReceipt:
    started = monotonic()
    card = _card_values(data, candidate)
    return _seal(NormalSemanticCardReceipt(
        branch_identity_sha256=identity, sequence=sequence, card=card,
        duration_seconds=monotonic() - started, receipt_sha256="0" * 64,
    ), "receipt_sha256")


def _publish_card(path: Path, receipt: NormalSemanticCardReceipt) -> None:
    _publish_value(path, receipt)


def _stop_binding(
    stop: _ToolStop, data: RepairPreparationData, identity: str, intent_pin: str,
    cards: tuple[NormalSemanticCardReceipt, ...], steps: tuple[HybridQueryCompletedStep, ...],
) -> None:
    if stop.branch_identity_sha256 != identity or stop.paid_count != len(cards) + len(steps) or (
        stop.card_receipt_sha256s != tuple(c.receipt_sha256 for c in cards)
        or stop.step_sha256s != tuple(s.step_sha256 for s in steps)
        or _seal(stop, "stop_sha256") != stop
    ):
        raise ValueError("hybrid tool terminal prefix differs")
    choice = stop.terminal_verification_choice
    if stop.stop_reason == "tool_budget_exhausted":
        if stop.paid_count != 8 or choice is not None:
            raise ValueError("hybrid tool exhausted terminal differs")
    else:
        if stop.paid_count >= 8 or choice is None or choice.selected_request_key is not None:
            raise ValueError("hybrid tool no-progress terminal differs")
        _choice_binding(choice, data.catalog, intent_pin, steps)


def run_normal_hybrid_tools(
    *, data: RepairPreparationData, ledger_root: Path, block_root: Path,
    paired_base_session: NormalQuerySession, expected_base_session_sha256: str,
    branch_identity_sha256: str, intents: tuple[HybridQueryIntent, ...],
    card_candidates: tuple[str, ...],
) -> NormalHybridToolSession:
    """Owner-only compiled plan. Card and verification identities cannot reset slots."""
    TypeAdapter(SHA256).validate_python(branch_identity_sha256)
    intent_pin = _intent_identity(data.catalog, intents)
    if type(card_candidates) is not tuple or len(set(card_candidates)) != len(card_candidates) or (
        not set(card_candidates) <= set(data.catalog.candidate_service_keys)
        or len(card_candidates) + len(intents) > 8
        or paired_base_session.catalog_sha256 != data.catalog.catalog_sha256
    ):
        raise ValueError("hybrid card plan or paired catalog differs")
    identity = sha256_hex(canonical_json_bytes(dict(
        schema_version="nurse.v2.normal-hybrid-tool-branch/1",
        branch_identity_sha256=branch_identity_sha256, intents_sha256=intent_pin,
        card_candidates=card_candidates, catalog_sha256=data.catalog.catalog_sha256,
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
        cards: tuple[NormalSemanticCardReceipt, ...] = ()
        steps: tuple[HybridQueryCompletedStep, ...] = ()
        allowed = {"budget", "stop.json", *(f"{i:02d}.json" for i in range(len(budget.records)))}
        if {p.name for p in ledger_root.iterdir()} - allowed:
            raise ValueError("hybrid tool journal has unexpected members")
        for index, (kind, key, pin) in enumerate(budget.records):
            path = ledger_root / f"{index:02d}.json"
            if not path.exists():
                raise ValueError("hybrid pending reservation without a result receipt")
            if kind == "card":
                receipt = _load(path, NormalSemanticCardReceipt, "receipt_sha256")
                if steps or len(cards) >= len(card_candidates) or (
                    receipt.branch_identity_sha256 != identity or receipt.sequence != index
                    or receipt.card.candidate_service_key != card_candidates[len(cards)]
                    or key != card_candidates[len(cards)]
                    or canonical_json_bytes(receipt.card.model_dump(mode="json")) !=
                    canonical_json_bytes(_card_values(data, key).model_dump(mode="json"))
                ):
                    raise ValueError("hybrid saved card differs from fixed source/plan")
                result_pin = receipt.receipt_sha256
                cards += (receipt,)
            else:
                if len(cards) != len(card_candidates):
                    raise ValueError("hybrid verification cannot precede planned card phase")
                step = _load(path, HybridQueryCompletedStep, "step_sha256")
                _step_binding(step, data, identity, intent_pin, steps, paired_base_session)
                if key != step.feedback.request_key:
                    raise ValueError("hybrid verification receipt key differs")
                result_pin = step.feedback.result_sha256
                steps += (step,)
            if pin not in (None, result_pin):
                raise ValueError("hybrid tool budget result reference differs")
            if pin is None:
                budget.complete(kind=kind, query_key=key, result_sha256=result_pin)
        stop_path = ledger_root / "stop.json"
        terminal = None
        if stop_path.exists() or stop_path.is_symlink():
            terminal = _load(stop_path, _ToolStop, "stop_sha256")
            _stop_binding(terminal, data, identity, intent_pin, cards, steps)
        admitted: set[str] = set()
        while terminal is None:
            choice = None
            if budget.remaining_count == 0:
                reason: Literal["tool_budget_exhausted", "no_admissible_progress"] = (
                    "tool_budget_exhausted"
                )
            elif len(cards) < len(card_candidates):
                candidate = card_candidates[len(cards)]
                sequence = budget.used_count
                if budget.reserve(kind="card", query_key=candidate).status != "reserved":
                    raise ValueError("hybrid card lacks a fresh logical reservation")
                receipt = _execute_card(data, candidate, identity, sequence)
                _publish_card(ledger_root / f"{sequence:02d}.json", receipt)
                budget.complete(kind="card", query_key=candidate,
                                result_sha256=receipt.receipt_sha256)
                cards += (receipt,)
                continue
            else:
                with query_step("choice"):
                    choice = _choose_values(
                        data.catalog, tuple(s.feedback for s in steps),
                        tuple(sorted(s.feedback.request_key for s in steps)), intents,
                    )
                if choice.selected_request_key is None:
                    reason = "no_admissible_progress"
                else:
                    key, sequence = choice.selected_request_key, budget.used_count
                    if budget.reserve(kind="verification", query_key=key).status != "reserved":
                        raise ValueError("hybrid verification lacks a fresh logical reservation")
                    saved = next((s for s in paired_base_session.steps
                                  if s.feedback.request_key == key), None)
                    request = next(r for r in data.catalog.requests if r.request_key == key)
                    source = next(s for s in data.sources
                                  if s.source_sha256 == request.source_sha256)
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
                        origin="paired_base" if saved is not None else "executed",
                        step_sha256="0" * 64,
                    ), "step_sha256")
                    _publish_value(ledger_root / f"{sequence:02d}.json", step)
                    budget.complete(kind="verification", query_key=key,
                                    result_sha256=feedback.result_sha256)
                    steps += (step,)
                    continue
            terminal = _seal(_ToolStop(
                branch_identity_sha256=identity,
                card_receipt_sha256s=tuple(c.receipt_sha256 for c in cards),
                step_sha256s=tuple(s.step_sha256 for s in steps), paid_count=budget.used_count,
                stop_reason=reason, terminal_verification_choice=choice, stop_sha256="0" * 64,
            ), "stop_sha256")
            _publish_value(stop_path, terminal)
        executed = sum(s.origin == "executed" for s in steps)
        return _seal(NormalHybridToolSession(
            branch_identity_sha256=identity,
            paired_base_session_sha256=expected_base_session_sha256,
            catalog_sha256=data.catalog.catalog_sha256, intents_sha256=intent_pin,
            cards=cards, steps=steps,
            submitted_request_keys=tuple(sorted(s.feedback.request_key for s in steps)),
            paid_count=budget.used_count, card_count=len(cards), verification_count=len(steps),
            reused_count=len(steps) - executed, executed_count=executed,
            physical_union_count=paired_base_session.paid_count + len(cards) + executed,
            stop_reason=terminal.stop_reason, stop=terminal, session_sha256="0" * 64,
        ), "session_sha256")


class NormalHybridToolEvidence(NormalAssurance):
    schema_version: Literal["nurse.v2.normal-hybrid-tool-rcl-evidence/1"] = (
        "nurse.v2.normal-hybrid-tool-rcl-evidence/1"
    )
    semantic_scope: Literal["query_priorities"] = "query_priorities"
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    source_descriptor_sha256: SHA256
    source_files_sha256: SHA256
    candidate_freeze_sha256: SHA256
    catalog_sha256: SHA256
    session_sha256: SHA256
    session: NormalHybridToolSession
    graph_manifest_sha256: SHA256
    graph: RepairLiveGraph
    ranking: RepairRanking
    result_sha256: SHA256


def score_normal_hybrid_tools(
    *, data: RepairPreparationData, session: NormalHybridToolSession,
) -> NormalHybridToolEvidence:
    if type(session) is not NormalHybridToolSession or (
        session.catalog_sha256 != data.catalog.catalog_sha256
        or session.paid_count != len(session.steps) + len(session.cards)
        or session.card_count != len(session.cards)
        or session.verification_count != len(session.steps)
        or session.executed_count + session.reused_count != session.verification_count
        or session.stop_reason != session.stop.stop_reason
        or session.submitted_request_keys != tuple(sorted({s.feedback.request_key
                                                          for s in session.steps}))
        or _seal(session, "session_sha256") != session
    ):
        raise ValueError("hybrid tool scoring requires a complete bound session")
    _stop_binding(session.stop, data, session.branch_identity_sha256, session.intents_sha256,
                  session.cards, session.steps)
    ranking = _score_ranking(data=data, session=session)
    return _seal(NormalHybridToolEvidence(
        dataset=data.diagnostics.dataset, opaque_case_id=data.catalog.opaque_case_id,
        source_descriptor_sha256=data.source_descriptor_sha256,
        source_files_sha256=data.source_files_sha256,
        candidate_freeze_sha256=data.candidate_freeze_sha256,
        catalog_sha256=data.catalog.catalog_sha256, session_sha256=session.session_sha256,
        session=session, graph_manifest_sha256=data.graph.manifest.manifest_sha256,
        graph=data.graph, ranking=ranking, result_sha256="0" * 64,
    ), "result_sha256")
