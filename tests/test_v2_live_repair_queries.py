"""Complete source catalogs and actual feedback, with no future episode lookup."""

import importlib
import importlib.util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2 import graph_snapshots as graphs
from nurse_rcl.v2.dynamic_verification.requests import build_dynamic_verification_policy
from nurse_rcl.v2.dynamic_verification.source_store import DirectorySourceBlockStore
from nurse_rcl.v2.event_grouping import AdapterAssociationSourceField
from nurse_rcl.v2.live.assembly import service_key
from nurse_rcl.v2.live.repair_assembly import assemble_rcabench_repair_diagnostics
from nurse_rcl.v2.live.repair_measurements import measure_repair_signal
from nurse_rcl.v2.live.repair_sources import build_repair_measurement_source
from nurse_rcl.v2.temporal_condensation import freeze_clock_resolution
from tests.test_v2_live_repair_assembly import measured_rcab
from tests.test_v2_live_sources import T0, load, write


def module():
    name = "nurse_rcl.v2.live.repair_queries"
    assert importlib.util.find_spec(name), "missing current-source feedback query adapter"
    return importlib.import_module(name)


def query_inputs(tmp_path, *, count=3, values=None):
    # Every original source/modality remains; only the synthetic metrics vary.
    root = measured_rcab(tmp_path)
    values = [10.0] * 2 + [20.0] * 3 + [10.0] * 11 if values is None else values
    for window, start in (("normal", T0 - 480), ("abnormal", T0)):
        write(
            root,
            f"{window}_metrics.parquet",
            [start + n * 30 for _ in range(count) for n in range(16)],
            metric=["container.cpu.usage"] * (16 * count),
            value=[10.0] * (16 * count) if window == "normal" else values * count,
            service_name=[f"service-{i}" for i in range(count) for _ in range(16)],
        )
    data = assemble_rcabench_repair_diagnostics(load(root))
    legacy = data.legacy
    contract = legacy.sources.adapter.time_contract
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    store = DirectorySourceBlockStore(root=str(blocks.resolve()))
    sources = tuple(
        sorted(
            (
                build_repair_measurement_source(
                    signal=signal,
                    repair=next(r for r in data.signals if r.signal_id == signal.signal_id),
                    time_contract=contract,
                    dataset="rcabench",
                    store=store,
                )
                for signal in legacy.signals
            ),
            key=lambda s: s.signal_id,
        )
    )
    return dict(
        sources=sources,
        expected_source_sha256s=tuple(s.source_sha256 for s in sources),
        time_contract=contract,
        expected_time_contract_sha256=sha256_hex(
            canonical_json_bytes(contract.model_dump(mode="json"))
        ),
        candidate_universe=legacy.candidate_universe,
        frozen_candidates=legacy.frozen_candidates,
        entity_registry=legacy.entity_registry,
        store=store,
    )


def catalog(api, args):
    return api.build_repair_query_catalog(**args)


def choose(api, directory, feedback=(), submitted=()):
    return api.select_repair_query(
        catalog=directory,
        expected_catalog_sha256=directory.catalog_sha256,
        feedback=feedback,
        expected_feedback_sha256s=tuple(f.result_sha256 for f in feedback),
        submitted_request_keys=submitted,
    )


def feedback_for(api, args, directory, selection):
    request = next(r for r in directory.requests if r.request_key == selection.selected_request_key)
    source = next(s for s in args["sources"] if s.source_sha256 == request.source_sha256)
    params = dict(
        source=source,
        expected_source_sha256=source.source_sha256,
        time_contract=args["time_contract"],
        expected_time_contract_sha256=args["expected_time_contract_sha256"],
        store=args["store"],
    )
    measured = measure_repair_signal(**params)
    return api.build_repair_query_feedback(
        **params,
        catalog=directory,
        expected_catalog_sha256=directory.catalog_sha256,
        measurement=measured,
        expected_measurement_sha256=measured.measurement_sha256,
    )


def test_complete_catalog_keeps_all_candidates_and_source_modalities_without_episodes(
    tmp_path, monkeypatch
):
    api = module()
    args = query_inputs(tmp_path)
    measurement = importlib.import_module("nurse_rcl.v2.live.repair_measurements")

    def forbidden(*a, **kw):
        pytest.fail("enumerating or selecting must not compute a future episode")

    monkeypatch.setattr(measurement, "_measure_values", forbidden)
    directory = catalog(api, args)
    assert directory.candidate_service_keys == args["frozen_candidates"].candidate_service_keys
    assert {r.source_sha256 for r in directory.requests} == set(args["expected_source_sha256s"])
    assert len(directory.requests) == len(args["sources"])
    receipt = choose(api, directory)
    assert receipt.selected_request_key is not None
    assert receipt.paid_count == 0 and receipt.remaining_budget == 8
    assert {r.request_key for r in receipt.requests} == {r.request_key for r in directory.requests}
    assert not hasattr(directory, "measurements")


def test_source_feedback_resolves_real_conditions_and_is_never_requeried(tmp_path):
    api = module()
    args = query_inputs(tmp_path)
    directory = catalog(api, args)
    first = choose(api, directory)
    feedback = feedback_for(api, args, directory, first)
    assert any(
        p.truth_value == "true" and p.predicate_kind == "local_pressure_observed"
        for p in feedback.predicates
    )
    second = choose(api, directory, (feedback,), (first.selected_request_key,))
    assert second.selected_request_key != first.selected_request_key
    assert second.paid_count == 1 and second.remaining_budget == 7
    assert any("already_submitted" in r.reason_codes for r in second.requests)


def test_fixed_normal_summaries_stop_without_claiming_or_measuring_future_episodes(tmp_path):
    api = module()
    args = query_inputs(tmp_path, values=[10.0] * 16)
    directory = catalog(api, args)
    receipt = choose(api, directory)
    assert receipt.selected_request_key is None
    assert receipt.stop_reason == "no_admissible_progress"
    assert receipt.paid_count == 0


@pytest.mark.parametrize(
    "attack", ["drop_source", "duplicate_source", "foreign_pin", "foreign_candidate"]
)
def test_catalog_does_not_accept_a_subcase_or_rebound_source(tmp_path, attack):
    api = module()
    args = query_inputs(tmp_path)
    if attack == "drop_source":
        args["sources"] = args["sources"][:-1]
        args["expected_source_sha256s"] = args["expected_source_sha256s"][:-1]
    elif attack == "duplicate_source":
        args["sources"] += args["sources"][:1]
        args["expected_source_sha256s"] += args["expected_source_sha256s"][:1]
    elif attack == "foreign_pin":
        args["expected_source_sha256s"] = ("0" * 64, *args["expected_source_sha256s"][1:])
    else:
        source = args["sources"][0].model_copy(
            update={"candidate_service_key": service_key("foreign")}
        )
        args["sources"] = (source, *args["sources"][1:])
    with pytest.raises(ValueError):
        catalog(api, args)


@pytest.mark.parametrize("attack", ["unsubmitted", "duplicate", "wrong_pin", "rehashed_truth"])
def test_selector_rejects_inconsistent_observed_prefix(tmp_path, attack):
    api = module()
    args = query_inputs(tmp_path)
    directory = catalog(api, args)
    first = choose(api, directory)
    feedback = feedback_for(api, args, directory, first)
    rows, submitted = (feedback,), (first.selected_request_key,)
    pins = (feedback.result_sha256,)
    if attack == "unsubmitted":
        submitted = ()
    elif attack == "duplicate":
        rows, pins = (feedback, feedback), pins * 2
    elif attack == "wrong_pin":
        pins = ("0" * 64,)
    else:
        # The feedback binds actual predicate objects; a top-level rehash is not enough.
        predicate = feedback.predicates[0].model_copy(update={"truth_value": "invalid"})
        changed = feedback.model_copy(update={"predicates": (predicate, *feedback.predicates[1:])})
        document = changed.model_dump(mode="json", exclude={"result_sha256"})
        changed = changed.model_copy(
            update={"result_sha256": sha256_hex(canonical_json_bytes(document))}
        )
        rows, pins = (changed,), (changed.result_sha256,)
    with pytest.raises(ValueError):
        api.select_repair_query(
            catalog=directory,
            expected_catalog_sha256=directory.catalog_sha256,
            feedback=rows,
            expected_feedback_sha256s=pins,
            submitted_request_keys=submitted,
        )


def with_graph(api, args, directory, *, coverage=True, resolved=True):
    receipt = choose(api, directory)
    ordered = sorted((r for r in receipt.requests if r.eligible), key=lambda r: r.priority)
    # Freeze a synthetic, fully sourced graph before measurement. Its target is
    # deliberately not the ordinary second priority, making the branch observable.
    source, target = ordered[0].candidate_service_key, ordered[-1].candidate_service_key
    contract = args["time_contract"]
    identity = dict(
        opaque_case_id=contract.opaque_case_id,
        observation_contract_sha256=args["candidate_universe"].observation_contract_sha256,
    )
    provenance = (
        AdapterAssociationSourceField(
            field_kind="export_id", raw_value="export:synthetic-query-graph"
        ),
    )
    parents = [
        graphs.build_graph_adapter_parent_record(
            **identity,
            window_kind=kind,
            interval=window,
            parent_artifact_kind="coverage_samples",
            collection_state="observed" if coverage else "unknown",
            valid_sample_count=16 if coverage else None,
            expected_sample_count=16 if coverage else None,
            source_identity_fields=provenance,
            provenance_fields=provenance,
        )
        for kind, window in (("normal", contract.baseline_window), ("fault", contract.fault_window))
    ]
    parents.append(
        graphs.build_graph_adapter_parent_record(
            **identity,
            window_kind="fault",
            interval=contract.fault_window,
            parent_artifact_kind="call_measurement",
            caller_service_key=source,
            callee_service_key=target,
            relation_type="caller_load_transmission",
            direction="caller_to_callee",
            observation_state="observed",
            request_count=16,
            error_count=2,
            latency_p95_seconds=None,
            source_identity_fields=provenance,
            provenance_fields=provenance,
        )
    )
    manifest = graphs.build_frozen_graph_input_manifest(**identity, parent_records=tuple(parents))
    pair = graphs.build_graph_snapshot_pair(
        time_contract=contract,
        candidate_universe=args["candidate_universe"],
        frozen_candidates=args["frozen_candidates"],
        entity_registry=args["entity_registry"],
        graph_input_manifest=manifest,
        expected_graph_input_manifest_sha256=manifest.manifest_sha256,
    )
    clock = freeze_clock_resolution(
        time_contract=contract,
        expected_time_contract_sha256=args["expected_time_contract_sha256"],
        resolution_status="resolved" if resolved else "unresolved",
        provenance_ids=("synthetic:query-clock",),
    )
    return (
        dict(
            **args,
            graph_input_manifest=manifest,
            expected_graph_input_manifest_sha256=manifest.manifest_sha256,
            graph_pair=pair,
            expected_graph_pair_sha256=pair.pair_sha256,
            clock_resolution=clock,
            expected_clock_resolution_sha256=clock.resolution_sha256,
        ),
        source,
        target,
    )


def test_priority_reuses_the_exact_existing_tie_policy(tmp_path):
    api = module()
    directory = catalog(api, query_inputs(tmp_path))
    receipt = choose(api, directory)
    policy = build_dynamic_verification_policy()
    for row, request in zip(receipt.requests, directory.requests, strict=True):
        assert row.stable_tie_sha256 == sha256_hex(
            canonical_json_bytes(
                [
                    policy.tie_seed,
                    request.candidate_service_key,
                    request.canonical_series_key,
                    policy.operator,
                ]
            )
        )


def test_actual_measured_feedback_changes_the_next_query_between_two_source_branches(tmp_path):
    api = module()
    outcomes = []
    for name, values in (
        ("episode", [10.0] * 2 + [20.0] * 3 + [10.0] * 11),
        ("spikes", [10.0, 20.0, 10.0, 20.0, 10.0, 20.0] + [10.0] * 10),
    ):
        folder = tmp_path / name
        folder.mkdir()
        args = query_inputs(folder, values=values)
        plain = catalog(api, args)
        args, source_candidate, target_candidate = with_graph(api, args, plain)
        directory = catalog(api, args)
        first = choose(api, directory)
        feedback = feedback_for(api, args, directory, first)
        second = choose(api, directory, (feedback,), (first.selected_request_key,))
        selected = next(r for r in second.requests if r.request_key == second.selected_request_key)
        assert (
            next(
                r for r in first.requests if r.request_key == first.selected_request_key
            ).candidate_service_key
            == source_candidate
        )
        if name == "episode":
            assert selected.candidate_service_key == target_candidate
            assert selected.demand_tier == 0
            assert "observed_counterpart_onset" in selected.reason_codes
        else:
            assert selected.candidate_service_key != target_candidate
            assert selected.demand_tier != 0
        outcomes.append((source_candidate, target_candidate, selected.candidate_service_key))
    assert outcomes[0][:2] == outcomes[1][:2]
    assert outcomes[0][2] != outcomes[1][2]


@pytest.mark.parametrize("options", [{"coverage": False}, {"resolved": False}])
def test_temporal_priority_requires_real_coverage_and_resolved_clock(tmp_path, options):
    api = module()
    args = query_inputs(tmp_path)
    args, _, _ = with_graph(api, args, catalog(api, args), **options)
    directory = catalog(api, args)
    first = choose(api, directory)
    feedback = feedback_for(api, args, directory, first)
    second = choose(api, directory, (feedback,), (first.selected_request_key,))
    assert all(r.demand_tier != 0 for r in second.requests)


def test_eight_actual_measurements_exhaust_the_original_case_budget(tmp_path):
    api = module()
    args = query_inputs(tmp_path, count=9)
    directory = catalog(api, args)
    feedback, submitted = (), ()
    for _ in range(8):
        selected = choose(api, directory, feedback, submitted)
        assert selected.selected_request_key is not None
        feedback += (feedback_for(api, args, directory, selected),)
        submitted = tuple(sorted((*submitted, selected.selected_request_key)))
    stopped = choose(api, directory, feedback, submitted)
    assert stopped.selected_request_key is None
    assert stopped.paid_count == 8 and stopped.remaining_budget == 0
    assert stopped.stop_reason == "query_budget_exhausted"
    assert len({f.request_key for f in feedback}) == 8


@pytest.mark.parametrize("raises", [False, True])
def test_working_request_projection_is_guarded_during_graph_linking(tmp_path, monkeypatch, raises):
    api = module()
    args = query_inputs(tmp_path)
    args, _, _ = with_graph(api, args, catalog(api, args))
    original = api._time_need_values

    def mutate(requests, graph, clock):
        result = original(requests, graph, clock)
        requests[0].__pydantic_fields_set__.add("unexpected_marker")
        if raises:
            raise RuntimeError("controlled graph-link error")
        return result

    monkeypatch.setattr(api, "_time_need_values", mutate)
    with pytest.raises(ValueError, match="changed|representation|field"):
        catalog(api, args)


@pytest.mark.parametrize("raises", [False, True])
def test_working_predicate_is_guarded_during_onset_feedback(tmp_path, monkeypatch, raises):
    api = module()
    args = query_inputs(tmp_path)
    directory = catalog(api, args)
    original = api._onset_bounds

    def mutate(measurement, predicate):
        result = original(measurement, predicate)
        predicate.__pydantic_fields_set__.add("unexpected_marker")
        if raises:
            raise RuntimeError("controlled onset error")
        return result

    monkeypatch.setattr(api, "_onset_bounds", mutate)
    with pytest.raises(ValueError, match="changed|representation|field"):
        feedback_for(api, args, directory, choose(api, directory))


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("target", ["original", "working"])
def test_selector_keeps_both_exit_checks(tmp_path, monkeypatch, target, raises):
    api = module()
    args = query_inputs(tmp_path)
    directory = catalog(api, args)
    original = api._select_values

    def mutate(fresh, feedback, submitted):
        result = original(fresh, feedback, submitted)
        victim = directory if target == "original" else fresh
        victim.__pydantic_fields_set__.add("unexpected_marker")
        if raises:
            raise RuntimeError("controlled selector error")
        return result

    monkeypatch.setattr(api, "_select_values", mutate)
    with pytest.raises(ValueError, match="changed|representation|field"):
        choose(api, directory)


def test_query_documents_have_independent_current_schema_identities():
    module()
    registry = importlib.import_module("nurse_rcl.v2.schema_compatibility")
    assert {
        "live_repair_query_catalog": "nurse.v2.live-repair-query-catalog/1",
        "live_repair_query_feedback": "nurse.v2.live-repair-query-feedback/1",
        "live_repair_query_selection": "nurse.v2.live-repair-query-selection/1",
    }.items() <= registry.CURRENT_SCHEMA_VERSIONS.items()
