"""Pure one-step selection from frozen needs and the observed evaluation prefix.

This is not an executor or a source certificate. The session must reconstruct
these inputs and A4-replay measurements; the receipt never grants score authority.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

from .contracts import (
    SHA256,
    DynamicDemand,
    DynamicExpressionNode,
    DynamicGraphTimePair,
    DynamicLocalTimePair,
    DynamicPredicateEvaluation,
    DynamicRequestCatalog,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    _SourceModel,
)
from .requests import build_dynamic_verification_policy
from .safety import canonical, plain, require_unchanged, snapshot


class CandidatePaidCount(_SourceModel):
    candidate_service_key: str = Field(min_length=1)
    paid_count: int = Field(ge=0)


class DynamicSelectionState(_SourceModel):
    """Private runtime projection; no raw samples, store, Q, ranking or future results."""

    inputs_sha256: SHA256
    predecessor_session_sha256: SHA256 | None
    request_catalog: DynamicRequestCatalog
    expected_request_catalog_sha256: SHA256
    policy: DynamicVerificationPolicy
    expected_policy_sha256: SHA256
    evaluation: DynamicPredicateEvaluation
    expected_evaluation_sha256: SHA256
    submitted_request_keys: tuple[str, ...]
    completed_request_keys: tuple[str, ...]
    paid_by_candidate: tuple[CandidatePaidCount, ...]


class DemandSelectionStatus(_SourceModel):
    demand_key: str
    status: Literal[
        "pending",
        "resolved",
        "unavailable",
        "missing_source",
        "submitted_reused",
        "no_new_information",
    ]
    tier: int | None = Field(ge=0, le=2)
    progress_request_keys: tuple[str, ...]
    expression_ids: tuple[str, ...]
    reason_codes: tuple[str, ...] = Field(min_length=1)


class RequestSelectionStatus(_SourceModel):
    request_key: str
    candidate_service_key: str | None
    eligible: bool
    demand_tier: int | None = Field(ge=0, le=2)
    candidate_paid_count: int = Field(ge=0)
    unresolved_demand_keys: tuple[str, ...]
    stable_tie_sha256: SHA256
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @property
    def priority(self) -> tuple[int, int, int, str, str] | None:
        if self.demand_tier is None:
            return None
        return (
            self.demand_tier,
            self.candidate_paid_count,
            -len(self.unresolved_demand_keys),
            self.stable_tie_sha256,
            self.request_key,
        )


class SelectionReceipt(_SourceModel):
    """Nested session receipt, not a separately registered or outbound root."""

    state_sha256: SHA256
    demands: tuple[DemandSelectionStatus, ...]
    requests: tuple[RequestSelectionStatus, ...]
    selected_request: DynamicVerificationRequest | None
    paid_count: int = Field(ge=0, le=8)
    remaining_budget: int = Field(ge=0, le=8)
    stop_reason: Literal["query_budget_exhausted", "no_admissible_progress"] | None
    receipt_sha256: SHA256

    def verify(self) -> bool:
        """Internal identity only; a session must replay selection from its actual prefix."""
        try:
            if type(self) is not SelectionReceipt:
                return False
            before = snapshot(self, (SelectionReceipt,))
            fresh = SelectionReceipt.model_validate_json(before[0])
            _validate_receipt(fresh)
            document = cast(dict[str, object], plain(fresh))
            document.pop("receipt_sha256")
            return (
                fresh.receipt_sha256 == sha256_hex(canonical_json_bytes(document))
                and canonical(fresh, (SelectionReceipt,)) == before[0]
                and snapshot(self, (SelectionReceipt,)) == before
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False


def _validate_receipt(receipt: SelectionReceipt) -> None:
    """Check internal consistency, not the authenticity of its observed prefix."""
    strategy = build_dynamic_verification_policy()
    if receipt.paid_count + receipt.remaining_budget != strategy.case_query_budget:
        raise ValueError("selection receipt budget is inconsistent")
    request_keys = tuple(row.request_key for row in receipt.requests)
    demand_keys = tuple(row.demand_key for row in receipt.demands)
    if request_keys != tuple(sorted(set(request_keys))) or demand_keys != tuple(
        sorted(set(demand_keys))
    ):
        raise ValueError("selection receipt requires unique ordered rows")
    for demand in receipt.demands:
        keys = demand.progress_request_keys
        if (
            keys != tuple(sorted(set(keys)))
            or not set(keys) <= set(request_keys)
            or (demand.status == "pending") != bool(keys)
            or (demand.tier is not None) != bool(keys)
            or demand.expression_ids != tuple(sorted(set(demand.expression_ids)))
            or demand.reason_codes != tuple(sorted(set(demand.reason_codes)))
        ):
            raise ValueError("selection receipt demand status is inconsistent")
    paid_by_candidate: dict[str, int] = {}
    for row in receipt.requests:
        relevant = tuple(
            demand for demand in receipt.demands if row.request_key in demand.progress_request_keys
        )
        tier = min((cast(int, demand.tier) for demand in relevant), default=None)
        reasons = []
        # Submission provenance is external to this receipt and must be replayed
        # by the session. Its internal consequence still must be consistent.
        if "already_submitted" in row.reason_codes:
            reasons.append("already_submitted")
            if relevant:
                raise ValueError("submitted request cannot advance another demand")
        if row.candidate_service_key is None:
            reasons.append("request_unavailable")
            if row.candidate_paid_count or relevant:
                raise ValueError("unavailable request cannot be paid or advance a demand")
        else:
            previous = paid_by_candidate.setdefault(
                row.candidate_service_key, row.candidate_paid_count
            )
            if previous != row.candidate_paid_count:
                raise ValueError("candidate coverage must agree across requests")
        if not relevant:
            reasons.append("no_admissible_demand")
        if receipt.remaining_budget == 0:
            reasons.append("query_budget_exhausted")
        if (
            row.demand_tier != tier
            or row.unresolved_demand_keys != tuple(demand.demand_key for demand in relevant)
            or row.eligible != (not reasons)
            or row.reason_codes != (tuple(sorted(reasons)) if reasons else ("admissible_progress",))
        ):
            raise ValueError("selection receipt request priority is inconsistent")
    if sum(paid_by_candidate.values()) != receipt.paid_count:
        raise ValueError("selection receipt candidate coverage differs from total paid")
    eligible = [row for row in receipt.requests if row.eligible]
    chosen = min(
        eligible, key=lambda row: cast(tuple[int, int, int, str, str], row.priority), default=None
    )
    selected = receipt.selected_request
    if chosen is None:
        if selected is not None:
            raise ValueError("selection receipt selected an ineligible request")
    elif (
        selected is None
        or not selected.verify()
        or selected.availability != "available"
        or selected.canonical_request_key != chosen.request_key
        or selected.candidate_service_key != chosen.candidate_service_key
        or selected.policy_sha256 != strategy.policy_sha256
        or chosen.stable_tie_sha256
        != sha256_hex(
            canonical_json_bytes(
                [
                    strategy.tie_seed,
                    selected.candidate_service_key,
                    selected.canonical_series_key,
                    selected.operator,
                ]
            )
        )
    ):
        raise ValueError("selection receipt does not select the unique best priority")
    expected_stop = (
        "query_budget_exhausted"
        if receipt.remaining_budget == 0
        else "no_admissible_progress"
        if chosen is None
        else None
    )
    if receipt.stop_reason != expected_stop:
        raise ValueError("selection receipt stop reason is inconsistent")


def _validate_state(state: DynamicSelectionState) -> None:
    directory, evaluation, policy = state.request_catalog, state.evaluation, state.policy
    if (
        not directory.verify()
        or not evaluation.verify()
        or not policy.verify()
        or policy != build_dynamic_verification_policy()
        or directory.request_catalog_sha256 != state.expected_request_catalog_sha256
        or evaluation.evaluation_sha256 != state.expected_evaluation_sha256
        or policy.policy_sha256 != state.expected_policy_sha256
    ):
        raise ValueError("selection independent prefix pins/integrity mismatch")
    for field in (
        "opaque_case_id",
        "candidate_freeze_sha256",
        "time_contract_sha256",
        "catalog_sha256",
        "policy_sha256",
        "source_rule_set_sha256",
        "request_catalog_sha256",
    ):
        if getattr(directory, field) != getattr(evaluation, field):
            raise ValueError("selection directory/evaluation headers differ")
    if directory.policy_sha256 != policy.policy_sha256:
        raise ValueError("selection policy differs from frozen requests")
    requests = {row.canonical_request_key: row for row in directory.requests}
    for keys in (state.submitted_request_keys, state.completed_request_keys):
        if keys != tuple(sorted(set(keys))) or not set(keys) <= set(requests):
            raise ValueError("selection ledger requires unique ordered legal request keys")
    submitted, completed = set(state.submitted_request_keys), set(state.completed_request_keys)
    if not completed <= submitted or len(completed) > policy.case_query_budget:
        raise ValueError("selection completion/submission ledger mismatch")
    if {row.request_key for row in evaluation.request_consumption} != set(requests) or {
        row.request_key
        for row in evaluation.request_consumption
        if row.measurement_sha256 is not None
    } != completed:
        raise ValueError("selection completed ledger differs from observed evaluation prefix")
    paid = tuple(
        CandidatePaidCount(
            candidate_service_key=key,
            paid_count=sum(requests[item].candidate_service_key == key for item in completed),
        )
        for key in directory.candidate_service_keys
    )
    if state.paid_by_candidate != paid or sum(row.paid_count for row in paid) != len(completed):
        raise ValueError("selection candidate coverage differs from actual unique paid requests")
    demands = {row.demand_key: row for row in directory.demands}
    if not set(evaluation.pending_demand_keys) <= set(demands):
        raise ValueError("selection pending evaluation demand is outside frozen directory")
    if {row.time_pair_key for row in evaluation.time_evaluations} != {
        row.time_pair_key for row in directory.time_pairs
    }:
        raise ValueError("selection time evaluations must cover every frozen pair")
    for row in evaluation.conditions:
        demand = demands.get(row.demand_key)
        if demand is None or (
            demand.time_pair_key is not None
            or row.seed_hypothesis_id not in demand.seed_ids
            or row.seed_predicate_id not in demand.source_predicate_ids
            or row.requirement != demand.requirement
        ):
            raise ValueError("selection condition/demand binding differs")


def _condition_progress(
    demand: DynamicDemand, nodes: tuple[DynamicExpressionNode, ...]
) -> tuple[str, ...]:
    keys: set[str] = set()
    for node in nodes:
        if node.child_ids or demand.demand_key not in node.pending_demand_keys:
            continue
        # An unknown min/max proof needs measurement; a known unsupported source
        # or denied direction cannot be made valid by submitting the same query.
        if node.kind != "summary_not_persistent" or node.truth_value != "unknown":
            continue
        if set(node.reason_codes) & {
            "temporal_resolution_unavailable",
            "summary_quality_unavailable",
            "summary_deviation_band_unavailable",
        }:
            continue
        keys.update(
            "request_" + ref.source_sha256
            for ref in node.field_references
            if ref.source_kind == "request"
        )
    return tuple(sorted(keys & set(demand.request_keys)))


def _classify_demands(state: DynamicSelectionState) -> tuple[DemandSelectionStatus, ...]:
    directory, evaluation = state.request_catalog, state.evaluation
    nodes = {row.expression_id: row for row in evaluation.expression_nodes}
    submitted = set(state.submitted_request_keys)
    available = {
        row.canonical_request_key for row in directory.requests if row.availability == "available"
    }
    conditions = {
        demand.demand_key: tuple(
            row for row in evaluation.conditions if row.demand_key == demand.demand_key
        )
        for demand in directory.demands
        if demand.time_pair_key is None
    }
    progress = {
        row.demand_key: _condition_progress(row, evaluation.expression_nodes)
        for row in directory.demands
        if row.time_pair_key is None
    }
    queryable_conditions = {
        (row.candidate_service_key, row.predicate_kind, key)
        for row in directory.demands
        if row.time_pair_key is None
        for key in progress[row.demand_key]
    }
    completed = set(state.completed_request_keys)
    summary_empty = {
        "request_" + ref.source_sha256
        for node in evaluation.expression_nodes
        if node.kind == "summary_not_persistent" and node.truth_value == "false"
        for ref in node.field_references
        if ref.source_kind == "request"
    }
    time_results = {row.time_pair_key: row for row in evaluation.time_evaluations}
    time_pairs = {row.time_pair_key: row for row in directory.time_pairs}
    results = []
    for demand in directory.demands:
        tier = 2
        reasons: tuple[str, ...]
        keys: tuple[str, ...]
        known = False
        if demand.time_pair_key is None:
            rows = conditions[demand.demand_key]
            expressions = tuple(sorted({row.expression_id for row in rows}))
            truths = {nodes[key].truth_value for key in expressions}
            keys = progress[demand.demand_key]
            known = bool(truths) and "unknown" not in truths
            if demand.requirement == "necessary" and truths != {"true"}:
                tier, reasons = 1, ("unresolved_necessary_condition",)
            else:
                reasons = ("optional_episode_detail",)
        else:
            time = time_results[demand.time_pair_key]
            pair = time_pairs[demand.time_pair_key]
            current = time.source if demand.time_side == "source" else time.target
            other = time.target if demand.time_side == "source" else time.source
            if current.demand_key != demand.demand_key:
                raise ValueError("selection time-side demand binding differs")
            expressions = (time.expression_id,)
            known = nodes[time.expression_id].truth_value != "unknown"
            reasons = ("optional_time_detail",)
            keys = demand.request_keys if current.measurement_pending else ()
            blocks = set(current.reason_codes + other.reason_codes) - {"time_measurement_pending"}
            if evaluation.clock_resolution.resolution_status != "resolved":
                blocks.add("clock_uncertainty_unresolved")
            if type(pair) is DynamicGraphTimePair:
                domains = (pair.source.domain, pair.target.domain)
            elif type(pair) is DynamicLocalTimePair:
                domains = (pair.source, pair.target)
            else:
                raise ValueError("selection requires an exact declared time pair")
            if any(domain is None for domain in domains):
                blocks.add("time_source_unavailable")
            # A frozen request exists even for a semantically unusable series.
            # Pending time sides must inherit the corresponding condition's
            # source/quality/unit eligibility, not merely its request identity.
            if any(
                key not in completed
                and not any(
                    (domain.candidate_service_key, kind, key) in queryable_conditions
                    for kind in domain.predicate_kinds
                )
                for domain in domains
                if domain is not None
                for key in domain.request_keys
            ):
                blocks.add("time_direction_or_source_not_queryable")
            if any(
                set(domain.request_keys) & summary_empty for domain in domains if domain is not None
            ):
                blocks.add("summary_proves_no_qualifying_episode")
            if blocks:
                keys, reasons = (), tuple(sorted(blocks))
            elif other.onset_bounds is not None and current.measurement_pending:
                tier, reasons = 0, ("observed_counterpart_onset",)
        potential = tuple(sorted(set(keys) & available))
        admissible = tuple(key for key in potential if key not in submitted)
        if admissible:
            status = "pending"
        elif demand.request_keys and set(demand.request_keys) <= submitted:
            status, reasons = "submitted_reused", ("already_submitted", *reasons)
        elif known:
            status = "resolved"
        elif not demand.request_keys:
            status = (
                "unavailable"
                if "legacy_condition_only" in demand.reason_codes
                else "missing_source"
            )
            reasons = demand.reason_codes
        elif not set(demand.request_keys) & available:
            status, reasons = "unavailable", ("request_unavailable",)
        else:
            status = "no_new_information"
        results.append(
            DemandSelectionStatus(
                demand_key=demand.demand_key,
                status=cast(
                    Literal[
                        "pending",
                        "resolved",
                        "unavailable",
                        "missing_source",
                        "submitted_reused",
                        "no_new_information",
                    ],
                    status,
                ),
                tier=tier if admissible else None,
                progress_request_keys=admissible,
                expression_ids=expressions,
                reason_codes=tuple(sorted(set(reasons))),
            )
        )
    return tuple(results)


def select_next_request(*, state: DynamicSelectionState) -> SelectionReceipt:
    """Choose one request; no execution, lookahead, scores, labels or model inputs."""
    roots = (DynamicSelectionState,)
    before = snapshot(state, roots)
    try:
        if type(state) is not DynamicSelectionState:
            raise ValueError("selection requires an exact observed-prefix state")
        fresh = DynamicSelectionState.model_validate_json(before[0])
        _validate_state(fresh)
        demands = _classify_demands(fresh)
        paid = {row.candidate_service_key: row.paid_count for row in fresh.paid_by_candidate}
        paid_count = len(fresh.completed_request_keys)
        remaining = fresh.policy.case_query_budget - paid_count
        rows = []
        for request in fresh.request_catalog.requests:
            relevant = tuple(
                row for row in demands if request.canonical_request_key in row.progress_request_keys
            )
            tier = min((cast(int, row.tier) for row in relevant), default=None)
            reasons = []
            if request.canonical_request_key in fresh.submitted_request_keys:
                reasons.append("already_submitted")
            if request.availability != "available":
                reasons.append("request_unavailable")
            if not relevant:
                reasons.append("no_admissible_demand")
            if remaining == 0:
                reasons.append("query_budget_exhausted")
            rows.append(
                RequestSelectionStatus(
                    request_key=request.canonical_request_key,
                    candidate_service_key=request.candidate_service_key,
                    eligible=not reasons,
                    demand_tier=tier,
                    candidate_paid_count=paid.get(request.candidate_service_key or "", 0),
                    unresolved_demand_keys=tuple(sorted({row.demand_key for row in relevant})),
                    stable_tie_sha256=sha256_hex(
                        canonical_json_bytes(
                            [
                                fresh.policy.tie_seed,
                                request.candidate_service_key,
                                request.canonical_series_key,
                                request.operator,
                            ]
                        )
                    ),
                    reason_codes=tuple(sorted(reasons)) if reasons else ("admissible_progress",),
                )
            )
        eligible = [row for row in rows if row.eligible]
        chosen = (
            min(eligible, key=lambda row: cast(tuple[int, int, int, str, str], row.priority))
            if eligible
            else None
        )
        selected = next(
            (
                row
                for row in fresh.request_catalog.requests
                if chosen is not None and row.canonical_request_key == chosen.request_key
            ),
            None,
        )
        draft = SelectionReceipt(
            state_sha256=sha256_hex(before[0]),
            demands=demands,
            requests=tuple(rows),
            selected_request=selected,
            paid_count=paid_count,
            remaining_budget=remaining,
            stop_reason="query_budget_exhausted"
            if remaining == 0
            else "no_admissible_progress"
            if selected is None
            else None,
            receipt_sha256="0" * 64,
        )
        document = cast(dict[str, object], plain(draft))
        document.pop("receipt_sha256")
        result = draft.model_copy(
            update={"receipt_sha256": sha256_hex(canonical_json_bytes(document))}
        )
        if not result.verify():
            raise ValueError("selection receipt construction integrity failed")
        return result
    finally:
        require_unchanged(state, roots, before)
