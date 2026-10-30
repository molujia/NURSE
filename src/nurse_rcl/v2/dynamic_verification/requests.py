"""Fixed A2 measurement identities. Feedback demands/selection are later layers."""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import cast

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.hypothesis_guard import _PREDICATE_EVIDENCE_CONTRACTS
from nurse_rcl.v2.observation_schemas import CaseTimeContract
from nurse_rcl.v2.rule_hypotheses import RuleHypothesisEnvelope

from .contracts import (
    DynamicDemand,
    DynamicGraphTimePair,
    DynamicLocalTimePair,
    DynamicRequestCatalog,
    DynamicTimeSide,
    DynamicVerificationPolicy,
    DynamicVerificationRequest,
    VerificationSourceCatalog,
    dynamic_demand_key,
)
from .safety import canonical, plain, require_pin, require_unchanged, snapshot


def build_dynamic_verification_policy() -> DynamicVerificationPolicy:
    draft = DynamicVerificationPolicy(policy_sha256="0" * 64)
    document = cast(dict[str, object], plain(draft))
    document.pop("policy_sha256")
    return draft.model_copy(update={"policy_sha256": sha256_hex(canonical_json_bytes(document))})


def build_dynamic_verification_requests(
    *,
    envelope: RuleHypothesisEnvelope,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    policy: DynamicVerificationPolicy,
    expected_policy_sha256: str,
) -> tuple[DynamicVerificationRequest, ...]:
    """Enumerate every frozen series without reading blocks or calculating episodes.

    The supplied pins bind the caller's frozen catalog/policy. This enumeration
    grants no execution authority: the measurement entry must replay actual blocks.
    """
    roots = (
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceCatalog,
        DynamicVerificationPolicy,
    )
    inputs = (envelope, time_contract, catalog, policy)
    if any(type(value) is not kind for value, kind in zip(inputs, roots, strict=True)):
        raise ValueError("dynamic request enumeration requires exact frozen roots")
    require_pin(expected_catalog_sha256)
    require_pin(expected_policy_sha256)
    before = snapshot(inputs, roots)
    try:
        documents = json.loads(before[0])
        source = RuleHypothesisEnvelope.model_validate_json(canonical_json_bytes(documents[0]))
        contract = CaseTimeContract.model_validate_json(canonical_json_bytes(documents[1]))
        sources = VerificationSourceCatalog.model_validate_json(canonical_json_bytes(documents[2]))
        strategy = DynamicVerificationPolicy.model_validate_json(canonical_json_bytes(documents[3]))
        if not source.verify() or not sources.verify() or not strategy.verify():
            raise ValueError("dynamic request source integrity failed")
        if canonical((source, contract, sources, strategy), roots) != before[0]:
            raise ValueError("dynamic request source canonical replay differs")
        if (
            sources.catalog_sha256 != expected_catalog_sha256
            or strategy.policy_sha256 != expected_policy_sha256
            or strategy != build_dynamic_verification_policy()
        ):
            raise ValueError("dynamic request independent catalog/policy pin mismatch")
        if (
            sources.opaque_case_id != contract.opaque_case_id
            or sources.opaque_case_id != source.candidate_universe.opaque_case_id
            or sources.candidate_freeze_sha256 != source.frozen_candidates.freeze_sha256
            or sources.time_contract_sha256 != sha256_hex(canonical(contract, (CaseTimeContract,)))
            or contract.decision_time_unix_ns != source.entity_registry.decision_time_unix_ns
        ):
            raise ValueError("dynamic request case/freeze/time binding mismatch")
        requests = []
        for series in sources.series:
            if (
                series.transform_result.windows[0].interval != contract.baseline_window
                or series.transform_result.windows[1].interval != contract.fault_window
            ):
                raise ValueError("dynamic request source windows differ from full frozen B/F")
            draft = DynamicVerificationRequest(
                opaque_case_id=sources.opaque_case_id,
                candidate_freeze_sha256=sources.candidate_freeze_sha256,
                catalog_sha256=sources.catalog_sha256,
                time_contract_sha256=sources.time_contract_sha256,
                canonical_series_key=series.canonical_series_key,
                candidate_service_key=series.candidate_service_key,
                transform_component=series.transform_component,
                baseline_window=contract.baseline_window,
                fault_window=contract.fault_window,
                decision_time_unix_ns=contract.decision_time_unix_ns,
                permission_scope=source.scope_interval,
                allowed_candidate_service_keys=source.frozen_candidates.candidate_service_keys,
                availability="available"
                if series.candidate_service_key is not None
                else "unavailable",
                reason_codes=()
                if series.candidate_service_key is not None
                else ("entity_binding_unavailable",),
                policy_sha256=strategy.policy_sha256,
                quality_configuration_sha256=sha256_hex(
                    canonical_json_bytes(plain(series.quality_result.configuration))
                ),
                request_sha256="0" * 64,
            )
            unsigned = cast(dict[str, object], plain(draft))
            unsigned.pop("request_sha256")
            requests.append(
                draft.model_copy(
                    update={"request_sha256": sha256_hex(canonical_json_bytes(unsigned))}
                )
            )
        return tuple(sorted(requests, key=lambda row: row.canonical_request_key))
    finally:
        require_unchanged(inputs, roots, before)


def _catalog_demands(
    envelope: RuleHypothesisEnvelope,
    catalog: VerificationSourceCatalog,
    requests: tuple[DynamicVerificationRequest, ...],
    pairs: tuple[DynamicLocalTimePair | DynamicGraphTimePair, ...],
) -> tuple[DynamicDemand, ...]:
    from .predicate_evaluation import _PERSISTENCE_FIELDS

    by_key: dict[str, DynamicDemand] = {}

    def retain(document: dict[str, object]) -> None:
        # Build the key from the declared semantic fields before strict construction.
        document["demand_key"] = dynamic_demand_key(document)
        row = DynamicDemand.model_validate_json(canonical_json_bytes(document))
        previous = by_key.get(row.demand_key)
        if previous is not None:
            merged = cast(dict[str, object], plain(row))
            for field in ("seed_ids", "source_predicate_ids", "observation_ids", "reason_codes"):
                merged[field] = sorted(set(getattr(previous, field) + getattr(row, field)))
            row = DynamicDemand.model_validate_json(canonical_json_bytes(merged))
        by_key[row.demand_key] = row

    predicates = {row.predicate_result_id: row for row in envelope.rule_hypothesis_set.predicates}
    observations = {row.observation_id: row for row in envelope.event_grouping.observations}
    families = {
        key: group.primary_signal_family
        for group in envelope.event_grouping.event_groups
        for key in group.observation_ids
    }
    for seed in envelope.rule_hypothesis_set.hypotheses:
        for predicate_id in (
            seed.necessary_predicate_result_ids
            + seed.optional_predicate_result_ids
            + seed.refuting_predicate_result_ids
        ):
            predicate = predicates[predicate_id]
            supported = (
                predicate.predicate_kind in _PERSISTENCE_FIELDS
                and predicate.requirement != "refuting"
            )
            contract = _PREDICATE_EVIDENCE_CONTRACTS[predicate.predicate_kind]
            rows = tuple(
                row
                for row in catalog.rows
                if (
                    supported
                    and row.candidate_service_key == seed.candidate_service_key
                    and row.observation_id in predicate.observation_ids
                    and (
                        families[row.observation_id] in contract.primary_signal_families
                        or observations[row.observation_id].signal_family
                        in contract.source_signal_families
                    )
                )
            )
            keys = {
                row.canonical_series_key for row in rows if row.canonical_series_key is not None
            }
            request_keys = tuple(
                sorted(
                    row.canonical_request_key
                    for row in requests
                    if row.canonical_series_key in keys
                )
            )
            retain(
                dict(
                    candidate_service_key=seed.candidate_service_key,
                    seed_ids=(seed.hypothesis_id,),
                    source_predicate_ids=(predicate_id,),
                    condition_key="condition_"
                    + sha256_hex(
                        canonical_json_bytes(
                            [
                                seed.candidate_service_key,
                                predicate.predicate_kind,
                                predicate.requirement,
                            ]
                        )
                    ),
                    predicate_kind=predicate.predicate_kind,
                    requirement=predicate.requirement,
                    observation_ids=tuple(sorted(row.observation_id for row in rows)),
                    request_keys=request_keys,
                    time_pair_key=None,
                    time_side=None,
                    reason_codes=("legacy_condition_only",)
                    if not supported
                    else ("frozen_condition_sources",)
                    if request_keys
                    else ("no_bound_measurement_source",),
                )
            )
    for pair in pairs:
        for side_name in ("source", "target"):
            side: DynamicTimeSide | None
            seed_ids: tuple[str, ...]
            predicate_ids: tuple[str, ...]
            kind: str
            if type(pair) is DynamicLocalTimePair:
                side = pair.source if side_name == "source" else pair.target
                candidate = side.candidate_service_key
                seed_ids = (pair.seed_hypothesis_id,)
                predicate_ids = (pair.seed_predicate_id,)
                kind = pair.predicate_kind
            elif type(pair) is DynamicGraphTimePair:
                binding = pair.source if side_name == "source" else pair.target
                side = binding.domain
                candidate = binding.candidate_service_key
                seed_ids = (binding.seed.hypothesis_id,)
                predicate_ids = tuple(
                    sorted(row.predicate_result_id for row in binding.necessary_predicates)
                )
                kind = "source_before_target"
            else:
                raise ValueError("dynamic demands require an exact declared time pair")
            retain(
                dict(
                    candidate_service_key=candidate,
                    seed_ids=seed_ids,
                    source_predicate_ids=predicate_ids,
                    condition_key=pair.condition_key,
                    predicate_kind=kind,
                    requirement="optional",
                    observation_ids=side.observation_ids if side is not None else (),
                    request_keys=side.request_keys if side is not None else (),
                    time_pair_key=pair.time_pair_key,
                    time_side=side_name,
                    reason_codes=("frozen_time_sources",)
                    if side is not None and side.request_keys
                    else ("no_bound_time_source",),
                )
            )
    return tuple(by_key[key] for key in sorted(by_key))


def build_dynamic_request_catalog(
    *,
    envelope: RuleHypothesisEnvelope,
    expected_rule_set_sha256: str,
    time_contract: CaseTimeContract,
    catalog: VerificationSourceCatalog,
    expected_catalog_sha256: str,
    policy: DynamicVerificationPolicy,
    expected_policy_sha256: str,
    graph_snapshots: tuple[GraphSnapshot, ...],
    expected_graph_snapshot_sha256s: tuple[str, ...],
) -> DynamicRequestCatalog:
    """Freeze all C's needs, never query, rank or inspect a future measurement.

    Local imports keep request identity primitives independent of the consumers
    that need them; the complete directory is a nested session input, not a root.
    """
    from .graph_time import _graph_time_pair_values, _prepare_graph_time_snapshots
    from .predicate_evaluation import _check_condition_sources, _local_time_pair_values

    roots = (
        RuleHypothesisEnvelope,
        CaseTimeContract,
        VerificationSourceCatalog,
        DynamicVerificationPolicy,
        GraphSnapshot,
        DynamicVerificationRequest,
        DynamicLocalTimePair,
        DynamicGraphTimePair,
    )
    inputs = (
        envelope,
        time_contract,
        catalog,
        policy,
        graph_snapshots,
        expected_rule_set_sha256,
        expected_catalog_sha256,
        expected_policy_sha256,
        expected_graph_snapshot_sha256s,
    )
    before = snapshot(inputs, roots)
    try:
        with ExitStack() as guards:
            if (
                type(envelope) is not RuleHypothesisEnvelope
                or type(time_contract) is not CaseTimeContract
                or type(catalog) is not VerificationSourceCatalog
                or type(policy) is not DynamicVerificationPolicy
            ):
                raise ValueError("dynamic request catalog requires exact frozen roots")
            require_pin(expected_rule_set_sha256)
            source = RuleHypothesisEnvelope.model_validate_json(canonical(envelope, roots))
            contract = CaseTimeContract.model_validate_json(canonical(time_contract, roots))
            sources = VerificationSourceCatalog.model_validate_json(canonical(catalog, roots))
            strategy = DynamicVerificationPolicy.model_validate_json(canonical(policy, roots))
            working = (source, contract, sources, strategy)
            guards.callback(require_unchanged, working, roots, snapshot(working, roots))
            if source.rule_hypothesis_set.result_sha256 != expected_rule_set_sha256:
                raise ValueError("dynamic request catalog independent rule pin mismatch")
            requests = build_dynamic_verification_requests(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                expected_catalog_sha256=expected_catalog_sha256,
                policy=strategy,
                expected_policy_sha256=expected_policy_sha256,
            )
            guards.callback(require_unchanged, requests, roots, snapshot(requests, roots))
            _check_condition_sources(source, sources, requests, ())
            graphs = _prepare_graph_time_snapshots(
                envelope=source,
                time_contract=contract,
                graph_snapshots=graph_snapshots,
                expected_graph_snapshot_sha256s=expected_graph_snapshot_sha256s,
            )
            guards.callback(require_unchanged, graphs, roots, snapshot(graphs, roots))
            graph_pairs = _graph_time_pair_values(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                requests=requests,
                graph_snapshots=graphs,
            )
            guards.callback(require_unchanged, graph_pairs, roots, snapshot(graph_pairs, roots))
            local_pairs = _local_time_pair_values(
                envelope=source,
                time_contract=contract,
                catalog=sources,
                requests=requests,
            )
            guards.callback(require_unchanged, local_pairs, roots, snapshot(local_pairs, roots))
            unsorted_pairs: tuple[DynamicLocalTimePair | DynamicGraphTimePair, ...] = (
                *local_pairs,
                *graph_pairs,
            )
            pairs = tuple(sorted(unsorted_pairs, key=lambda row: row.time_pair_key))
            ordered_graphs = tuple(sorted(graphs, key=lambda row: row.snapshot_sha256))
            draft = DynamicRequestCatalog(
                opaque_case_id=sources.opaque_case_id,
                candidate_freeze_sha256=sources.candidate_freeze_sha256,
                candidate_service_keys=source.frozen_candidates.candidate_service_keys,
                time_contract_sha256=sources.time_contract_sha256,
                source_rule_set_sha256=source.rule_hypothesis_set.result_sha256,
                catalog_sha256=sources.catalog_sha256,
                policy_sha256=strategy.policy_sha256,
                requests=requests,
                demands=_catalog_demands(source, sources, requests, pairs),
                time_pairs=pairs,
                graph_snapshots=ordered_graphs,
                request_catalog_sha256="0" * 64,
            )
            document = cast(dict[str, object], plain(draft))
            document.pop("request_catalog_sha256")
            result = draft.model_copy(
                update={"request_catalog_sha256": sha256_hex(canonical_json_bytes(document))}
            )
            if not result.verify():
                raise ValueError("dynamic request catalog construction integrity failed")
            return result
    finally:
        require_unchanged(inputs, roots, before)
