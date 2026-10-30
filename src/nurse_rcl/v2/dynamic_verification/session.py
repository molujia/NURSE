"""Private, independently pinned sources for cumulative dynamic verification.

Input replay reconstructs raw source records but never executes a measurement.
Prepared values are fresh working data, not a caller-supplied authority token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.b0_rule_binding import B0RuleBinding, build_b0_rule_binding
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.observation_schemas import CaseTimeContract
from nurse_rcl.v2.propagation_scoring import PropagationScoringConfig
from nurse_rcl.v2.ranking_schemas import RuleSet
from nurse_rcl.v2.reference_ranking import ReferenceScoringConfig
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope
from nurse_rcl.v2.rule_scoring import RuleCandidateAssignment, RuleScoringResult
from nurse_rcl.v2.temporal_condensation import (
    FrozenClockResolution,
    RelationSCCCondensation,
    require_frozen_clock_resolution,
)

from .contracts import (
    SHA256,
    DynamicPredicateEvaluation,
    DynamicRequestCatalog,
    DynamicVerificationPolicy,
    NamedSourceWindow,
    SignalEpisodeMeasurement,
    VerificationSourceCatalog,
    VerificationSourceRecord,
    _SourceModel,
)
from .measurement import measure_signal_episodes, require_signal_episode_measurement
from .predicate_evaluation import (
    _compile_dynamic_evaluation_values,
    _prepare_dynamic_measurements,
)
from .requests import build_dynamic_request_catalog
from .safety import canonical, plain, require_pin, require_unchanged, snapshot
from .scheduler import (
    CandidatePaidCount,
    DynamicSelectionState,
    SelectionReceipt,
    select_next_request,
)
from .source_catalog import require_verification_source_catalog
from .source_store import (
    SourceBlockStore,
    read_source_window,
    require_store_unchanged,
    store_snapshot,
)


class B0DynamicInputs(_SourceModel):
    """Runtime-only frozen input domain; no final session, guard or ranking."""

    schema_version: Literal["nurse.v2.b0-dynamic-inputs/1"] = "nurse.v2.b0-dynamic-inputs/1"
    envelope: RuleHypothesisEnvelope
    time_contract: CaseTimeContract
    catalog: VerificationSourceCatalog
    policy: DynamicVerificationPolicy
    clock_resolution: FrozenClockResolution
    graph_snapshots: tuple[GraphSnapshot, ...]
    condensations: tuple[RelationSCCCondensation, ...]
    propagation_config: PropagationScoringConfig
    scoring_config: ReferenceScoringConfig
    rule_set: RuleSet
    rule_assignments: tuple[RuleCandidateAssignment, ...]
    rule_predicates: tuple[PredicateResult, ...]
    rule_scoring: RuleScoringResult
    history_evaluation: HistoryEvaluationResult


def b0_dynamic_inputs_sha256(*, inputs: B0DynamicInputs) -> str:
    """Pure bounded identity, not a proof that any source or result is valid."""
    if type(inputs) is not B0DynamicInputs:
        raise ValueError("dynamic session inputs require the exact runtime type")
    return sha256_hex(canonical(inputs, (B0DynamicInputs,)))


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedDynamicInputs:
    inputs: B0DynamicInputs
    inputs_sha256: str
    source_records: tuple[VerificationSourceRecord, ...]
    original_binding: B0RuleBinding
    request_catalog: DynamicRequestCatalog


def _replay_nonmeasurement_sources(inputs: B0DynamicInputs) -> None:
    """All these sources were pinned before the session, not by its outputs."""
    for keys in (
        tuple(row.assignment_id for row in inputs.rule_assignments),
        tuple(row.predicate_result_id for row in inputs.rule_predicates),
    ):
        if keys != tuple(sorted(set(keys))):
            raise ValueError("dynamic rule sources must be ordered and unique")
    graphs = {row.snapshot_sha256: row for row in inputs.graph_snapshots}
    if tuple(graphs) != tuple(sorted(graphs)) or len(graphs) != len(inputs.graph_snapshots):
        raise ValueError("dynamic graph sources must be ordered and unique")
    condensation_keys = tuple(row.condensation_sha256 for row in inputs.condensations)
    domains = tuple((row.graph_snapshot_sha256, row.relation_type) for row in inputs.condensations)
    if condensation_keys != tuple(sorted(set(condensation_keys))) or len(domains) != len(
        set(domains)
    ):
        raise ValueError("dynamic condensation sources must be ordered and unique")
    for row in inputs.condensations:
        graph = graphs.get(row.graph_snapshot_sha256)
        if graph is None or not row.verify_inputs(graph_snapshot=graph):
            raise ValueError("dynamic condensation differs from its complete frozen graph")
    if not inputs.propagation_config.verify() or not inputs.scoring_config.verify():
        raise ValueError("dynamic session scoring configuration failed verification")
    if inputs.scoring_config.history_mode != "off" or inputs.history_evaluation.mode != "off":
        raise ValueError("dynamic B0 sessions only support the approved history-off mode")
    frozen = inputs.envelope.frozen_candidates
    if not inputs.rule_scoring.verify_trusted(
        expected_result_sha256=inputs.rule_scoring.result_sha256,
        expected_authoritative_input_sha256=inputs.rule_scoring.authoritative_input_sha256,
        frozen_candidates=frozen,
        rule_set=inputs.rule_set,
        assignments=inputs.rule_assignments,
        predicates=inputs.rule_predicates,
    ):
        raise ValueError("dynamic session rule source replay failed")
    if not inputs.history_evaluation.verify_trusted(
        expected_result_sha256=inputs.history_evaluation.result_sha256,
        expected_authoritative_input_sha256=inputs.history_evaluation.authoritative_input_sha256,
        frozen_candidates=frozen,
        mode="off",
    ):
        raise ValueError("dynamic session history-off source replay failed")


def _records_from_blocks(
    *, inputs: B0DynamicInputs, store: SourceBlockStore
) -> tuple[VerificationSourceRecord, ...]:
    catalog = inputs.catalog
    windows = {
        series.canonical_series_key: tuple(
            NamedSourceWindow(name=block.name, window=read_source_window(store=store, block=block))
            for block in series.source_blocks
        )
        for series in catalog.series
    }
    series_by_key = {row.canonical_series_key: row for row in catalog.series}
    records = []
    for row in catalog.rows:
        if row.canonical_series_key is None or row.source_identity is None:
            continue
        series = series_by_key[row.canonical_series_key]
        identity = row.source_identity
        records.append(
            VerificationSourceRecord(
                observation_id=row.observation_id,
                observed_entity_key=identity.observed_entity_key,
                entity_kind=identity.entity_kind,
                source_fingerprint=identity.source_fingerprint,
                adapter_semantics_sha256=identity.adapter_semantics_sha256,
                source_inputs=windows[row.canonical_series_key],
                transform_result=series.transform_result,
                quality_result=series.quality_result,
            )
        )
    return tuple(records)


def _replay_dynamic_inputs(
    *, inputs: B0DynamicInputs, expected_inputs_sha256: str, store: SourceBlockStore
) -> _PreparedDynamicInputs:
    """Fresh source replay, before the coordinator can reserve a paid query."""
    if type(inputs) is not B0DynamicInputs:
        raise ValueError("dynamic session inputs require the exact runtime type")
    pin = require_pin(expected_inputs_sha256)
    roots = (B0DynamicInputs,)
    before = snapshot(inputs, roots)
    store_state = store_snapshot(store)
    fresh: B0DynamicInputs | None = None
    fresh_state: tuple[bytes, str] | None = None
    try:
        with ExitStack() as guards:
            if sha256_hex(before[0]) != pin:
                raise ValueError("dynamic session independent input pin mismatch")
            fresh = B0DynamicInputs.model_validate_json(before[0])
            if canonical(fresh, roots) != before[0]:
                raise ValueError("dynamic session input normalization is not allowed")
            fresh_state = snapshot(fresh, roots)
            _replay_nonmeasurement_sources(fresh)
            binding = build_b0_rule_binding(
                envelope=fresh.envelope,
                expected_rule_set_sha256=fresh.envelope.rule_hypothesis_set.result_sha256,
            )
            guards.callback(
                require_unchanged, binding, (B0RuleBinding,), snapshot(binding, (B0RuleBinding,))
            )
            directory = build_dynamic_request_catalog(
                envelope=fresh.envelope,
                expected_rule_set_sha256=fresh.envelope.rule_hypothesis_set.result_sha256,
                time_contract=fresh.time_contract,
                catalog=fresh.catalog,
                expected_catalog_sha256=fresh.catalog.catalog_sha256,
                policy=fresh.policy,
                expected_policy_sha256=fresh.policy.policy_sha256,
                graph_snapshots=fresh.graph_snapshots,
                expected_graph_snapshot_sha256s=tuple(
                    row.snapshot_sha256 for row in fresh.graph_snapshots
                ),
            )
            guards.callback(
                require_unchanged,
                directory,
                (DynamicRequestCatalog,),
                snapshot(directory, (DynamicRequestCatalog,)),
            )
            require_frozen_clock_resolution(
                clock_resolution=fresh.clock_resolution,
                time_contract=fresh.time_contract,
                expected_clock_resolution_sha256=fresh.clock_resolution.resolution_sha256,
            )
            sources = require_verification_source_catalog(
                envelope=fresh.envelope,
                time_contract=fresh.time_contract,
                source_records=_records_from_blocks(inputs=fresh, store=store),
                expected_source_records_sha256=fresh.catalog.source_records_sha256,
                catalog=fresh.catalog,
                expected_catalog_sha256=fresh.catalog.catalog_sha256,
                store=store,
            )
            return _PreparedDynamicInputs(
                inputs=fresh,
                inputs_sha256=pin,
                source_records=sources.source_records,
                original_binding=binding,
                request_catalog=directory,
            )
    finally:
        require_unchanged(inputs, roots, before)
        if fresh is not None and fresh_state is not None:
            require_unchanged(fresh, roots, fresh_state)
        require_store_unchanged(store, store_state)


class DynamicQueryStep(_SourceModel):
    """A completed measurement binds its actual reservation, never a future head."""

    selection: SelectionReceipt
    submitted_session_sha256: SHA256
    measurement: SignalEpisodeMeasurement
    evaluation: DynamicPredicateEvaluation


class DynamicQuerySession(_SourceModel):
    """Immutable cumulative prefix. Internal identity is not source authentication."""

    schema_version: Literal["nurse.v2.dynamic-query-session/1"] = "nurse.v2.dynamic-query-session/1"
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: SHA256
    inputs_sha256: SHA256
    catalog_sha256: SHA256
    policy_sha256: SHA256
    request_catalog: DynamicRequestCatalog
    previous_session_sha256: SHA256 | None
    initial_paid_count: int = Field(ge=0, le=8)
    steps: tuple[DynamicQueryStep, ...]
    pending_selection: SelectionReceipt | None
    evaluation: DynamicPredicateEvaluation
    submitted_request_keys: tuple[str, ...]
    completed_request_keys: tuple[str, ...]
    paid_count: int = Field(ge=0, le=8)
    failure_reason: Literal["measurement_integrity_error", "completion_integrity_error"] | None
    rejected_measurement: SignalEpisodeMeasurement | None
    stop_receipt: SelectionReceipt | None
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"] | None
    session_sha256: SHA256


def _seal_session(draft: DynamicQuerySession) -> DynamicQuerySession:
    canonical(draft, (DynamicQuerySession,))
    document = cast(dict[str, object], plain(draft))
    document.pop("session_sha256")
    return draft.model_copy(update={"session_sha256": sha256_hex(canonical_json_bytes(document))})


@contextmanager
def _session_view(
    session: DynamicQuerySession,
) -> Iterator[tuple[bytes, DynamicQuerySession, Callable[[], None]]]:
    """One full parse with original and independent work guarded on every exit."""
    if type(session) is not DynamicQuerySession:
        raise ValueError("dynamic head requires an exact session version")
    roots = (DynamicQuerySession,)
    before = snapshot(session, roots)
    fresh: DynamicQuerySession | None = None
    fresh_before: tuple[bytes, str] | None = None

    def unchanged() -> None:
        try:
            require_unchanged(session, roots, before)
        finally:
            if fresh is not None and fresh_before is not None:
                require_unchanged(fresh, roots, fresh_before)

    try:
        fresh = DynamicQuerySession.model_validate_json(before[0])
        fresh_before = snapshot(fresh, roots)
        if fresh_before[0] != before[0]:
            raise ValueError("dynamic head session identity differs")
        document = cast(dict[str, object], plain(fresh))
        document.pop("session_sha256")
        if fresh.session_sha256 != sha256_hex(canonical_json_bytes(document)):
            raise ValueError("dynamic head session identity differs")
        yield before[0], fresh, unchanged
    finally:
        unchanged()


def _session_bytes(session: DynamicQuerySession) -> bytes:
    with _session_view(session) as (raw, _, _):
        return raw


def _session_key(session: DynamicQuerySession) -> tuple[str, str, str]:
    return session.opaque_case_id, session.candidate_freeze_sha256, session.inputs_sha256


def _replace_session(previous: DynamicQuerySession, **changes: object) -> DynamicQuerySession:
    return _seal_session(
        previous.model_copy(
            update={
                "previous_session_sha256": previous.session_sha256,
                "initial_paid_count": previous.paid_count,
                **changes,
            }
        )
    )


def _reserved(previous: DynamicQuerySession, choice: SelectionReceipt) -> DynamicQuerySession:
    request = choice.selected_request
    if request is None or request.canonical_request_key in previous.submitted_request_keys:
        raise ValueError("dynamic request is absent or already submitted")
    if (
        previous.pending_selection is not None
        or previous.stop_receipt is not None
        or previous.failure_reason is not None
    ):
        raise ValueError("dynamic head cannot submit from a pending or stopped prefix")
    return _replace_session(
        previous,
        pending_selection=choice,
        submitted_request_keys=tuple(
            sorted((*previous.submitted_request_keys, request.canonical_request_key))
        ),
    )


def _completed(
    previous: DynamicQuerySession,
    result: SignalEpisodeMeasurement,
    evaluation: DynamicPredicateEvaluation,
) -> DynamicQuerySession:
    choice = previous.pending_selection
    if previous.failure_reason is not None or choice is None or choice.selected_request is None:
        raise ValueError("dynamic completion requires an outstanding reservation")
    if result.request_sha256 != choice.selected_request.request_sha256:
        raise ValueError("dynamic completion differs from its reserved request")
    return _replace_session(
        previous,
        steps=(
            *previous.steps,
            DynamicQueryStep(
                selection=choice,
                submitted_session_sha256=previous.session_sha256,
                measurement=result,
                evaluation=evaluation,
            ),
        ),
        pending_selection=None,
        evaluation=evaluation,
        completed_request_keys=tuple(
            sorted(
                (*previous.completed_request_keys, choice.selected_request.canonical_request_key)
            )
        ),
        paid_count=previous.paid_count + 1,
    )


def _stopped(previous: DynamicQuerySession, choice: SelectionReceipt) -> DynamicQuerySession:
    if (
        previous.pending_selection is not None
        or previous.stop_receipt is not None
        or previous.failure_reason is not None
    ):
        raise ValueError("dynamic stop requires an active completed prefix")
    if choice.selected_request is not None or choice.stop_reason is None:
        raise ValueError("dynamic stop requires a genuine terminal choice")
    return _replace_session(previous, stop_receipt=choice, stop_reason=choice.stop_reason)


def _failed(
    previous: DynamicQuerySession,
    reason: Literal["measurement_integrity_error", "completion_integrity_error"],
    rejected_measurement: SignalEpisodeMeasurement | None = None,
) -> DynamicQuerySession:
    if previous.pending_selection is None or previous.failure_reason is not None:
        raise ValueError("dynamic integrity failure requires a live reservation")
    # Keep the entire pending receipt and submitted set. Failure is not a normal stop.
    if reason == "measurement_integrity_error" and rejected_measurement is not None:
        raise ValueError("a failed measurement has no completed payload")
    return _replace_session(
        previous, failure_reason=reason, rejected_measurement=rejected_measurement
    )


def _strict_successor_values(
    previous: DynamicQuerySession, candidate: DynamicQuerySession, candidate_bytes: bytes
) -> bool:
    """Compare guarded values; callers own validation, never source authority."""
    if (
        previous.stop_receipt is not None
        or previous.failure_reason is not None
        or _session_key(previous) != _session_key(candidate)
    ):
        return False
    try:
        if candidate.failure_reason is not None:
            expected = _failed(previous, candidate.failure_reason, candidate.rejected_measurement)
        elif candidate.stop_receipt is not None:
            expected = _stopped(previous, candidate.stop_receipt)
        elif previous.pending_selection is not None and candidate.pending_selection is None:
            if len(candidate.steps) != len(previous.steps) + 1:
                return False
            last = candidate.steps[-1]
            expected = _completed(previous, last.measurement, last.evaluation)
        elif candidate.pending_selection is not None:
            expected = _reserved(previous, candidate.pending_selection)
        else:
            return False
        if type(expected) is not DynamicQuerySession:
            return False
        return canonical(expected, (DynamicQuerySession,)) == candidate_bytes
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _strict_successor(previous: DynamicQuerySession, candidate: DynamicQuerySession) -> bool:
    """Storage monotonicity only; independently validate both external inputs."""
    try:
        with (
            _session_view(previous) as (_, prior, _),
            _session_view(candidate) as (raw, fresh, _),
        ):
            return _strict_successor_values(prior, fresh, raw)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


class DynamicSessionHeadStore:
    """Exact process-local CAS store. No public arbitrary-head installation API.

    Store bytes rather than caller-owned model objects. Source authentication
    belongs to the coordinator/replayer; a storage CAS is not a source certificate.
    """

    __slots__ = ("_heads", "_lock")

    def __init__(self) -> None:
        self._heads: dict[tuple[str, str, str], bytes] = {}
        self._lock = RLock()

    def read_head(
        self, *, opaque_case_id: str, candidate_freeze_sha256: str, inputs_sha256: str
    ) -> DynamicQuerySession | None:
        if type(opaque_case_id) is not str:
            raise ValueError("dynamic head requires an exact case key")
        key = (opaque_case_id, require_pin(candidate_freeze_sha256), require_pin(inputs_sha256))
        with self._lock:
            raw = self._heads.get(key)
            return None if raw is None else DynamicQuerySession.model_validate_json(raw)

    def _publish(
        self, *, previous: DynamicQuerySession | None, candidate: DynamicQuerySession
    ) -> DynamicQuerySession:
        with self._lock:
            with ExitStack() as scopes:
                raw, owned, _ = scopes.enter_context(_session_view(candidate))
                if previous is None:
                    prior = None
                    fresh_previous = None
                else:
                    prior, fresh_previous, _ = scopes.enter_context(_session_view(previous))
                key = _session_key(owned)
                if self._heads.get(key) != prior:
                    raise ValueError("dynamic authoritative head changed concurrently")
                if fresh_previous is not None:
                    if not _strict_successor_values(fresh_previous, owned, raw):
                        raise ValueError("dynamic head requires an exact strict successor")
                elif (
                    owned.previous_session_sha256 is not None
                    or owned.steps
                    or owned.pending_selection is not None
                    or owned.submitted_request_keys
                    or owned.completed_request_keys
                    or owned.paid_count
                    or owned.initial_paid_count
                    or owned.stop_receipt is not None
                    or owned.stop_reason is not None
                    or owned.failure_reason is not None
                    or owned.rejected_measurement is not None
                ):
                    raise ValueError("dynamic head requires a genuine initial prefix")
            if self._heads.get(key) != prior:
                raise ValueError("dynamic authoritative head changed concurrently")
            self._heads[key] = raw
        return owned


def _evaluate_prefix(
    prepared: _PreparedDynamicInputs, measurements: tuple[SignalEpisodeMeasurement, ...]
) -> DynamicPredicateEvaluation:
    """Use only this operation's real preparation, never a prior prefix answer."""
    if type(prepared) is not _PreparedDynamicInputs:
        raise ValueError("dynamic prefix requires exact operation-local inputs")
    roots = (B0DynamicInputs, B0RuleBinding, DynamicRequestCatalog, SignalEpisodeMeasurement)
    originals = (prepared.inputs, prepared.original_binding, prepared.request_catalog, measurements)
    before = snapshot(originals, roots)
    try:
        with ExitStack() as guards:
            inputs = prepared.inputs
            directory = prepared.request_catalog
            binding = prepared.original_binding
            results = _prepare_dynamic_measurements(
                source=inputs.envelope,
                sources=inputs.catalog,
                directory=directory,
                measurements=measurements,
                expected_measurement_sha256s=tuple(row.measurement_sha256 for row in measurements),
            )
            guards.callback(require_unchanged, results, roots, snapshot(results, roots))
            return _compile_dynamic_evaluation_values(
                source=inputs.envelope,
                binding=binding,
                sources=inputs.catalog,
                directory=directory,
                resolution=inputs.clock_resolution,
                results=results,
                expected_rule_set_sha256=inputs.envelope.rule_hypothesis_set.result_sha256,
                expected_original_binding_sha256=binding.binding_sha256,
                expected_catalog_sha256=inputs.catalog.catalog_sha256,
                expected_policy_sha256=inputs.policy.policy_sha256,
                expected_request_catalog_sha256=directory.request_catalog_sha256,
            )
    finally:
        require_unchanged(originals, roots, before)


def _initial_session(prepared: _PreparedDynamicInputs) -> DynamicQuerySession:
    directory = prepared.request_catalog
    return _seal_session(
        DynamicQuerySession(
            opaque_case_id=directory.opaque_case_id,
            candidate_freeze_sha256=directory.candidate_freeze_sha256,
            inputs_sha256=prepared.inputs_sha256,
            catalog_sha256=directory.catalog_sha256,
            policy_sha256=directory.policy_sha256,
            request_catalog=directory,
            previous_session_sha256=None,
            initial_paid_count=0,
            steps=(),
            pending_selection=None,
            evaluation=_evaluate_prefix(prepared, ()),
            submitted_request_keys=(),
            completed_request_keys=(),
            paid_count=0,
            failure_reason=None,
            rejected_measurement=None,
            stop_receipt=None,
            stop_reason=None,
            session_sha256="0" * 64,
        )
    )


def _selection_state(
    prepared: _PreparedDynamicInputs, session: DynamicQuerySession
) -> DynamicSelectionState:
    directory = prepared.request_catalog
    requests = {row.canonical_request_key: row for row in directory.requests}
    evaluation = _evaluate_prefix(prepared, tuple(row.measurement for row in session.steps))
    if evaluation != session.evaluation:
        raise ValueError("dynamic head evaluation differs from its actual completed prefix")
    return DynamicSelectionState(
        inputs_sha256=prepared.inputs_sha256,
        predecessor_session_sha256=session.session_sha256,
        request_catalog=directory,
        expected_request_catalog_sha256=directory.request_catalog_sha256,
        policy=prepared.inputs.policy,
        expected_policy_sha256=prepared.inputs.policy.policy_sha256,
        evaluation=evaluation,
        expected_evaluation_sha256=evaluation.evaluation_sha256,
        submitted_request_keys=session.submitted_request_keys,
        completed_request_keys=session.completed_request_keys,
        paid_by_candidate=tuple(
            CandidatePaidCount(
                candidate_service_key=key,
                paid_count=sum(
                    requests[item].candidate_service_key == key
                    for item in session.completed_request_keys
                ),
            )
            for key in directory.candidate_service_keys
        ),
    )


def _replay_session_prefix(
    *,
    prepared: _PreparedDynamicInputs,
    session: DynamicQuerySession,
    store: SourceBlockStore,
) -> DynamicQuerySession:
    """Reconstruct every transition from frozen sources, never from a later state."""
    with _session_view(session) as (submitted_bytes, fresh, _):
        if fresh.failure_reason is not None:
            raise ValueError("failed dynamic prefix is diagnostic only, not replay authority")
        inputs = prepared.inputs
        rebuilt = _initial_session(prepared)
        for step in fresh.steps:
            choice = select_next_request(state=_selection_state(prepared, rebuilt))
            if canonical(choice, (SelectionReceipt,)) != canonical(
                step.selection, (SelectionReceipt,)
            ):
                raise ValueError("dynamic prefix replay selection differs")
            request = choice.selected_request
            if request is None:
                raise ValueError("dynamic prefix contains a completion after its stopping point")
            reserved = _reserved(rebuilt, choice)
            if reserved.session_sha256 != step.submitted_session_sha256:
                raise ValueError("dynamic prefix replay reservation differs")
            measurement = require_signal_episode_measurement(
                envelope=inputs.envelope,
                time_contract=inputs.time_contract,
                source_records=prepared.source_records,
                expected_source_records_sha256=inputs.catalog.source_records_sha256,
                request=request,
                expected_request_sha256=request.request_sha256,
                catalog=inputs.catalog,
                expected_catalog_sha256=inputs.catalog.catalog_sha256,
                policy=inputs.policy,
                expected_policy_sha256=inputs.policy.policy_sha256,
                store=store,
                measurement=step.measurement,
                expected_measurement_sha256=step.measurement.measurement_sha256,
            )
            evaluation = _evaluate_prefix(
                prepared, (*tuple(row.measurement for row in rebuilt.steps), measurement)
            )
            rebuilt = _completed(reserved, measurement, evaluation)
            if canonical(rebuilt.steps[-1], (DynamicQueryStep,)) != canonical(
                step, (DynamicQueryStep,)
            ):
                raise ValueError("dynamic prefix replay step evaluation differs")
        if fresh.pending_selection is not None or fresh.stop_receipt is not None:
            choice = select_next_request(state=_selection_state(prepared, rebuilt))
            if fresh.pending_selection is not None:
                rebuilt = _reserved(rebuilt, choice)
            else:
                rebuilt = _stopped(rebuilt, choice)
        if _session_bytes(rebuilt) != submitted_bytes:
            raise ValueError("dynamic prefix replay ledger or terminal state differs")
        return fresh


def require_dynamic_session(
    *,
    inputs: B0DynamicInputs,
    expected_inputs_sha256: str,
    store: SourceBlockStore,
    session: DynamicQuerySession,
    expected_session_sha256: str,
) -> DynamicQuerySession:
    """Offline source replay; no head-store writes, reservations or paid execution.

    A verified active/pending prefix is not a completed ranking. Failed receipts
    remain diagnostics and cannot be promoted to authoritative successful input.
    """
    roots = (B0DynamicInputs, DynamicQuerySession)
    originals = (inputs, session)
    before = snapshot(originals, roots)
    session_pin = require_pin(expected_session_sha256)
    input_pin = require_pin(expected_inputs_sha256)
    store_state = store_snapshot(store)
    working: tuple[object, ...] | None = None
    working_before: tuple[bytes, str] | None = None
    working_roots = (*roots, VerificationSourceRecord, B0RuleBinding, DynamicRequestCatalog)
    try:
        with _session_view(session) as (_, fresh, _):
            if fresh.session_sha256 != session_pin:
                raise ValueError("dynamic session independent session pin mismatch")
            if b0_dynamic_inputs_sha256(inputs=inputs) != input_pin:
                raise ValueError("dynamic session independent input pin mismatch")
            prepared = _replay_dynamic_inputs(
                inputs=inputs, expected_inputs_sha256=input_pin, store=store
            )
            working = (
                prepared.inputs,
                prepared.source_records,
                prepared.original_binding,
                prepared.request_catalog,
                fresh,
            )
            working_before = snapshot(working, working_roots)
            return _replay_session_prefix(prepared=prepared, session=fresh, store=store)
    finally:
        require_unchanged(originals, roots, before)
        if working is not None and working_before is not None:
            require_unchanged(working, working_roots, working_before)
        require_store_unchanged(store, store_state)


class DynamicSessionCoordinator:
    """Only real measurement completions may advance the cumulative paid ledger."""

    def __init__(
        self,
        *,
        inputs: B0DynamicInputs,
        expected_inputs_sha256: str,
        store: SourceBlockStore,
        head_store: DynamicSessionHeadStore,
        previous_session: DynamicQuerySession | None,
    ) -> None:
        if type(head_store) is not DynamicSessionHeadStore:
            raise ValueError("dynamic coordinator requires the exact code-owned head store")
        self._inputs = inputs
        self._pin = require_pin(expected_inputs_sha256)
        self._store = store
        self.head_store = head_store
        originals = (inputs, previous_session)
        roots = (B0DynamicInputs, DynamicQuerySession)
        before = snapshot(originals, roots)
        store_state = store_snapshot(store)
        prepared: _PreparedDynamicInputs | None = None
        working: tuple[object, ...] | None = None
        working_roots = (*roots, B0RuleBinding, VerificationSourceRecord, DynamicRequestCatalog)
        working_before: tuple[bytes, str] | None = None

        def unchanged() -> None:
            require_unchanged(originals, roots, before)
            if working is not None and working_before is not None:
                require_unchanged(working, working_roots, working_before)
            require_store_unchanged(store, store_state)

        try:
            with ExitStack() as session_scopes:
                prepared = self._prepare()
                prior: bytes | None = None
                fresh_previous: DynamicQuerySession | None = None
                if previous_session is not None:
                    prior, fresh_previous, _ = session_scopes.enter_context(
                        _session_view(previous_session)
                    )
                working = (
                    prepared.inputs,
                    prepared.original_binding,
                    prepared.source_records,
                    prepared.request_catalog,
                    fresh_previous,
                )
                working_before = snapshot(working, working_roots)
                initial = _initial_session(prepared)
                current = head_store.read_head(
                    opaque_case_id=initial.opaque_case_id,
                    candidate_freeze_sha256=initial.candidate_freeze_sha256,
                    inputs_sha256=initial.inputs_sha256,
                )
                if previous_session is None:
                    if current is not None:
                        raise ValueError(
                            "dynamic current head cannot be reset by an omitted prefix"
                        )
                    unchanged()
                    self._session = head_store._publish(previous=None, candidate=initial)
                else:
                    if current is None or _session_bytes(current) != prior:
                        raise ValueError("dynamic previous prefix is not the current head")
                    if fresh_previous is None:
                        raise ValueError("dynamic resume requires its complete prior prefix")
                    self._session = _replay_session_prefix(
                        prepared=prepared,
                        session=fresh_previous,
                        store=store,
                    )
                    self._require_current()
        finally:
            unchanged()

    @property
    def session(self) -> DynamicQuerySession:
        with _session_view(self._session) as (_, fresh, _):
            return fresh

    def _prepare(self) -> _PreparedDynamicInputs:
        return _replay_dynamic_inputs(
            inputs=self._inputs, expected_inputs_sha256=self._pin, store=self._store
        )

    def _require_current(self) -> None:
        if type(self.head_store) is not DynamicSessionHeadStore:
            raise ValueError("dynamic coordinator requires its exact head store")
        current = self.head_store.read_head(
            opaque_case_id=self._session.opaque_case_id,
            candidate_freeze_sha256=self._session.candidate_freeze_sha256,
            inputs_sha256=self._session.inputs_sha256,
        )
        if current is None or _session_bytes(current) != _session_bytes(self._session):
            raise ValueError("dynamic coordinator no longer owns the current head")

    def _commit(self, candidate: DynamicQuerySession) -> None:
        owned = self.head_store._publish(previous=self._session, candidate=candidate)
        self._session = owned

    @contextmanager
    def _operation(
        self, *extras: object
    ) -> Iterator[tuple[_PreparedDynamicInputs, Callable[[], None]]]:
        roots = (
            B0DynamicInputs,
            DynamicQuerySession,
            SelectionReceipt,
            SignalEpisodeMeasurement,
            B0RuleBinding,
            VerificationSourceRecord,
            DynamicRequestCatalog,
        )
        originals = (self._inputs, self._session, *extras)
        before = snapshot(originals, roots)
        store_state = store_snapshot(self._store)
        working: tuple[object, ...] | None = None
        working_before: tuple[bytes, str] | None = None

        def unchanged() -> None:
            require_unchanged(originals, roots, before)
            if working is not None and working_before is not None:
                require_unchanged(working, roots, working_before)
            require_store_unchanged(self._store, store_state)

        try:
            self._require_current()
            if self._session.failure_reason is not None:
                raise ValueError("dynamic session failed integrity and cannot continue")
            prepared = self._prepare()
            working = (
                prepared.inputs,
                prepared.original_binding,
                prepared.source_records,
                prepared.request_catalog,
            )
            working_before = snapshot(working, roots)
            yield prepared, unchanged
        finally:
            unchanged()

    def _record_failure(
        self,
        reason: Literal["measurement_integrity_error", "completion_integrity_error"],
        rejected_measurement: SignalEpisodeMeasurement | None = None,
    ) -> None:
        if self._session.pending_selection is not None and self._session.failure_reason is None:
            self._commit(_failed(self._session, reason, rejected_measurement))

    def current_state(self) -> DynamicSelectionState:
        with self._operation() as (prepared, _):
            if self._session.pending_selection is not None:
                raise ValueError("dynamic session has a pending reservation")
            return _selection_state(prepared, self._session)

    def execute_selected(self, *, choice: SelectionReceipt) -> SignalEpisodeMeasurement:
        with self._operation(choice) as (prepared, unchanged):
            expected = select_next_request(state=_selection_state(prepared, self._session))
            if canonical(choice, (SelectionReceipt,)) != canonical(expected, (SelectionReceipt,)):
                raise ValueError("dynamic request differs from the unique current selection")
            unchanged()
            # CAS before measurement: a stale/forked worker cannot repeat this query.
            self._commit(_reserved(self._session, choice))
        try:
            with self._operation(choice) as (prepared, _):
                request = choice.selected_request
                if request is None:
                    raise ValueError("dynamic execution requires a selected request")
                inputs = prepared.inputs
                return measure_signal_episodes(
                    envelope=inputs.envelope,
                    time_contract=inputs.time_contract,
                    source_records=prepared.source_records,
                    expected_source_records_sha256=inputs.catalog.source_records_sha256,
                    request=request,
                    expected_request_sha256=request.request_sha256,
                    catalog=inputs.catalog,
                    expected_catalog_sha256=inputs.catalog.catalog_sha256,
                    policy=inputs.policy,
                    expected_policy_sha256=inputs.policy.policy_sha256,
                    store=self._store,
                )
        except Exception:
            self._record_failure("measurement_integrity_error")
            raise

    def append_result(self, *, choice: SelectionReceipt, result: SignalEpisodeMeasurement) -> None:
        rejected: SignalEpisodeMeasurement | None = None
        try:
            with self._operation(choice, result) as (prepared, unchanged):
                # Retain only an exact, bounded, schema-valid diagnostic payload.
                # This snapshot grants no source authority; A4 below decides that.
                rejected = SignalEpisodeMeasurement.model_validate_json(
                    canonical(result, (SignalEpisodeMeasurement,))
                )
                if canonical(choice, (SelectionReceipt,)) != canonical(
                    self._session.pending_selection, (SelectionReceipt,)
                ):
                    raise ValueError("dynamic result does not match its reserved selection")
                request = choice.selected_request
                if request is None:
                    raise ValueError("dynamic completion requires a selected request")
                inputs = prepared.inputs
                verified = require_signal_episode_measurement(
                    envelope=inputs.envelope,
                    time_contract=inputs.time_contract,
                    source_records=prepared.source_records,
                    expected_source_records_sha256=inputs.catalog.source_records_sha256,
                    request=request,
                    expected_request_sha256=request.request_sha256,
                    catalog=inputs.catalog,
                    expected_catalog_sha256=inputs.catalog.catalog_sha256,
                    policy=inputs.policy,
                    expected_policy_sha256=inputs.policy.policy_sha256,
                    store=self._store,
                    measurement=result,
                    expected_measurement_sha256=result.measurement_sha256,
                )
                evaluation = _evaluate_prefix(
                    prepared, (*tuple(row.measurement for row in self._session.steps), verified)
                )
                candidate = _completed(self._session, verified, evaluation)
                unchanged()
                self._commit(candidate)
        except Exception:
            self._record_failure("completion_integrity_error", rejected)
            raise

    def seal_stop(self, *, choice: SelectionReceipt) -> DynamicQuerySession:
        with self._operation(choice) as (prepared, unchanged):
            expected = select_next_request(state=_selection_state(prepared, self._session))
            if canonical(choice, (SelectionReceipt,)) != canonical(expected, (SelectionReceipt,)):
                raise ValueError("dynamic stop differs from its actual current prefix")
            candidate = _stopped(self._session, choice)
            unchanged()
            self._commit(candidate)
        return self.session


def run_dynamic_verification(
    *,
    inputs: B0DynamicInputs,
    expected_inputs_sha256: str,
    store: SourceBlockStore,
    head_store: DynamicSessionHeadStore,
    previous_session: DynamicQuerySession | None,
) -> DynamicQuerySession:
    """Execute one selected request at a time, preserving the case-wide ledger."""
    coordinator = DynamicSessionCoordinator(
        inputs=inputs,
        expected_inputs_sha256=expected_inputs_sha256,
        store=store,
        head_store=head_store,
        previous_session=previous_session,
    )
    if coordinator.session.stop_receipt is not None:
        return coordinator.session
    while True:
        state = coordinator.current_state()
        choice = select_next_request(state=state)
        if choice.selected_request is None:
            return coordinator.seal_stop(choice=choice)
        result = coordinator.execute_selected(choice=choice)
        coordinator.append_result(choice=choice, result=result)
