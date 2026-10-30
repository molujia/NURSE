"""Actual feedback, single production measurements and durable normal recovery."""

import importlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live import repair_queries as values
from nurse_rcl.v2.live.repair_query_session import RepairQueryCompletedStep
from tests.live_repair_case_fixtures import make_case_pair


def api():
    name = "nurse_rcl.v2.live.normal_queries"
    assert importlib.util.find_spec(name), "missing normal incremental query execution"
    module = importlib.import_module(name)
    assert callable(getattr(module, "run_normal_queries", None))
    return module


@pytest.fixture(scope="module")
def prepared_cases(tmp_path_factory):
    from nurse_rcl.v2.live.repair_raw_audit import _fixed_repair_scope

    cases = {}

    def get(kind):
        if kind not in cases:
            root = tmp_path_factory.mktemp("normal-query-" + kind)
            dataset = "rcabench" if kind == "eight" else kind
            pair = make_case_pair(root, dataset, eight_queries=kind == "eight")
            with _fixed_repair_scope(**pair["raw_arguments"]) as (data, _):
                pass
            cases[kind] = data, pair["raw_arguments"]["store_root"], kind
        return cases[kind]

    return get


@pytest.fixture(scope="module", params=["rcabench", "aiops25", "eight"])
def prepared(prepared_cases, request):
    return prepared_cases(request.param)


@pytest.fixture(scope="module")
def active_prepared(prepared_cases):
    return prepared_cases("eight")


def test_single_pass_matches_legacy_measurements_and_recovers_without_replay(
    prepared, tmp_path, monkeypatch,
):
    module = api()
    data, blocks, kind = prepared
    from nurse_rcl.v2.dynamic_verification.source_store import DirectorySourceBlockStore
    from nurse_rcl.v2.live import repair_measurements as measures
    from nurse_rcl.v2.live import repair_query_session as legacy
    from tests.test_v2_live_repair_queries import feedback_for

    # One bounded legacy numerical reference, not a production audit dependency.
    args = dict(
        sources=data.sources, store=DirectorySourceBlockStore(root=str(blocks)),
        time_contract=data.diagnostics.legacy.sources.adapter.time_contract,
        expected_time_contract_sha256=data.catalog.time_contract_sha256,
    )
    expected = []
    submitted = ()
    for _ in range(9):
        choice = values._select_values(data.catalog, tuple(s.feedback for s in expected), submitted)
        if choice.selected_request_key is None:
            break
        source = next(s for s in data.sources if s.source_sha256 == next(
            r.source_sha256 for r in data.catalog.requests
            if r.request_key == choice.selected_request_key
        ))
        measured = measures.measure_repair_signal(
            source=source, expected_source_sha256=source.source_sha256,
            time_contract=args["time_contract"],
            expected_time_contract_sha256=args["expected_time_contract_sha256"],
            store=args["store"],
        )
        feedback = feedback_for(values, args, data.catalog, choice)
        expected.append(RepairQueryCompletedStep(
            selection=choice, measurement=measured, feedback=feedback,
        ))
        submitted = tuple(sorted((*submitted, choice.selected_request_key)))

    def forbidden(*a, **kw):
        pytest.fail("normal query execution entered legacy replay/protection")

    for name in ("build_repair_query_catalog", "build_repair_query_feedback", "snapshot"):
        monkeypatch.setattr(values, name, forbidden)
    monkeypatch.setattr(measures, "_source_replay", forbidden)
    monkeypatch.setattr(legacy, "_read_journal", forbidden)
    counts = Counter(choice=0, measurement=0, feedback=0)
    root = tmp_path / "ledger"
    read = module._read_bytes
    reads = Counter()

    def once(path, limit):
        reads[str(path)] += 1
        return read(path, limit)

    monkeypatch.setattr(module, "_read_bytes", once)
    for name, label in (("_select_values", "choice"), ("_measure_values", "measurement"),
                        ("_feedback_values", "feedback")):
        original = getattr(module, name)

        def count(*args, _original=original, _label=label, **kwargs):
            counts[_label] += 1
            if _label == "measurement":
                last = sorted(root.glob("*.json"))[-1]
                assert json.loads(last.read_bytes())["phase"] == "reserved"
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, count)
    actual = module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
    assert actual.schema_version == "nurse.v2.normal-rcl-query-session/1"
    assert actual.full_recompute == "not_performed"
    assert actual.steps == tuple(expected)
    assert actual.submitted_request_keys == submitted
    assert actual.stop_reason == choice.stop_reason
    assert not any(Path(p).parent == root for p in reads)
    assert counts == {"choice": len(expected) + 1, "measurement": len(expected),
                      "feedback": len(expected)}
    if kind == "eight":
        assert actual.paid_count == len(set(submitted)) == 8
        assert counts == {"choice": 9, "measurement": 8, "feedback": 8}
    counts.clear()
    reads.clear()
    resumed = module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
    assert resumed == actual
    assert counts == {}
    assert max(reads.values()) == 1
    assert all(json.loads(p.read_bytes())["schema_version"] ==
               "nurse.v2.normal-rcl-query-journal/1" for p in root.glob("*.json"))


@pytest.mark.parametrize("phase", ["measured", "completed"])
def test_write_failure_conserves_reservation_and_measurement(
    active_prepared, tmp_path, monkeypatch, phase,
):
    module = api()
    data, blocks, _ = active_prepared
    root = tmp_path / "ledger"
    publish = module._publish
    measurements = []
    measure = module._measure_values

    def measured(source):
        measurements.append(source.source_sha256)
        return measure(source)

    def fail(path, entry):
        if entry.phase == phase:
            raise OSError("controlled write failure")
        return publish(path, entry)

    monkeypatch.setattr(module, "_publish", fail)
    monkeypatch.setattr(module, "_measure_values", measured)
    with pytest.raises(OSError, match="controlled write"):
        module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
    assert len(measurements) == 1
    rows = [json.loads(p.read_bytes()) for p in sorted(root.glob("*.json"))]
    assert rows[1]["phase"] == "reserved"
    monkeypatch.setattr(module, "_publish", publish)
    if phase == "measured":
        with pytest.raises(ValueError, match="pending reservation"):
            module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
        assert len(measurements) == 1
    else:
        result = module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
        assert result.steps[0].measurement.source_sha256 == measurements[0]
        assert len(measurements) == result.paid_count
        assert len(set(measurements)) == len(measurements)


@pytest.mark.parametrize("damage", ["sequence", "pin", "case", "duplicate"])
def test_saved_journal_rejects_illegal_boundary(active_prepared, tmp_path, damage):
    module = api()
    data, blocks, _ = active_prepared
    root = tmp_path / "ledger"
    module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)
    path = root / "001.json"
    body = json.loads(path.read_bytes())
    if damage == "sequence":
        body["sequence"] += 1
    elif damage == "pin":
        body["entry_sha256"] = "0" * 64
    elif damage == "case":
        body["catalog_sha256"] = "0" * 64
    else:
        # Validly signed duplicate reservation is still an illegal transition.
        path = root / "002.json"
        body.update(sequence=2, previous_sha256=body["entry_sha256"])
    if damage != "pin":
        body["entry_sha256"] = sha256_hex(canonical_json_bytes({
            k: v for k, v in body.items() if k != "entry_sha256"
        }))
    path.write_bytes(canonical_json_bytes(body))
    with pytest.raises(ValueError):
        module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)


def test_pinned_selected_block_damage_stops_after_reservation(active_prepared, tmp_path):
    from shutil import copytree

    module = api()
    data, blocks, _ = active_prepared
    selected = values._select_values(data.catalog, (), ()).selected_request_key
    request = next(r for r in data.catalog.requests if r.request_key == selected)
    source = next(s for s in data.sources if s.source_sha256 == request.source_sha256)
    copied = tmp_path / "blocks"
    copytree(blocks, copied)
    (copied / (source.source_blocks[0].sha256 + ".json")).write_bytes(b"{}")
    root = tmp_path / "ledger"
    with pytest.raises(ValueError, match="block"):
        module.run_normal_queries(data=data, ledger_root=root, block_root=copied)
    rows = [json.loads(p.read_bytes()) for p in sorted(root.glob("*.json"))]
    assert [r["phase"] for r in rows] == ["initial", "reserved", "failed"]
    with pytest.raises(ValueError, match="failed"):
        module.run_normal_queries(data=data, ledger_root=root, block_root=blocks)


def test_normal_feedback_uses_actual_episode_to_choose_a_different_next_query(tmp_path):
    from tests.test_v2_live_repair_queries import catalog, query_inputs, with_graph

    module = api()
    outcomes = []
    for name, samples in (
        ("episode", [10.0] * 2 + [20.0] * 3 + [10.0] * 11),
        ("spikes", [10.0, 20.0, 10.0, 20.0, 10.0, 20.0] + [10.0] * 10),
    ):
        folder = tmp_path / name
        folder.mkdir()
        args = query_inputs(folder, values=samples)
        args, source_candidate, target = with_graph(values, args, catalog(values, args))
        directory = catalog(values, args)
        first = module._select_values(directory, (), ())
        request = next(r for r in directory.requests if r.request_key == first.selected_request_key)
        source = next(s for s in args["sources"] if s.source_sha256 == request.source_sha256)
        measurement = module._measure_values(source)
        feedback = module._feedback_values(directory, source, measurement)
        second = module._select_values(directory, (feedback,), (request.request_key,))
        selected = next(r for r in second.requests if r.request_key == second.selected_request_key)
        assert request.candidate_service_key == source_candidate
        assert (selected.candidate_service_key == target) == (name == "episode")
        assert (selected.demand_tier == 0) == (name == "episode")
        outcomes.append(selected.candidate_service_key)
    assert outcomes[0] != outcomes[1]
