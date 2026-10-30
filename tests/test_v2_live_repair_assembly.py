"""Intact Parquet cases: repair diagnostics, never a replacement ranking input."""

import importlib
import importlib.util
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nurse_rcl.v2.live.assembly import assemble_rcabench_live_inputs, service_key
from tests.test_v2_live_aiops25_assembly import measured_case
from tests.test_v2_live_aiops25_sources import load as load_aiops
from tests.test_v2_live_sources import T0, case, load, write


def module():
    name = "nurse_rcl.v2.live.repair_assembly"
    assert importlib.util.find_spec(name), "missing explicit repair diagnostic assembly"
    return importlib.import_module(name)


def measured_rcab(tmp_path):
    root = case(tmp_path / "private-answer-case")
    for window, start in (("normal", T0 - 480), ("abnormal", T0)):
        times = [start + n * 30 for n in range(16)]
        metrics = (
            "container.cpu.usage",
            "k8s.statefulset.ready_pods",
            "k8s.statefulset.desired_pods",
            "unregistered_cpu",
        )
        values = []
        for metric in metrics:
            if metric == "container.cpu.usage":
                values.extend([10.0 if window == "normal" else 1.0] * 16)
            else:
                values.extend([0.0 if window == "abnormal" and n == 3 else 1.0 for n in range(16)])
        write(
            root,
            f"{window}_metrics.parquet",
            times * len(metrics),
            metric=[m for m in metrics for _ in times],
            value=values,
            service_name=["resource-service"] * len(values),
        )
        write(
            root,
            f"{window}_traces.parquet",
            times,
            service_name=["trace-only-service"] * 16,
            trace_id=[f"{window}-t{n}" for n in range(16)],
            span_id=["s"] * 16,
            duration=[1_000] * 16,
            **{
                "attr.span_kind": ["server"] * 16,
                "attr.status_code": ["Unset"] * 16,
                "attr.http.response.status_code": [0 if window == "normal" else 500] * 16,
            },
        )
        write(
            root,
            f"{window}_logs.parquet",
            times,
            service_name=["log-only-service"] * 16,
            level=["INFO" if window == "normal" else "ERROR"] * 16,
            message=["PRIVATE RAW PROSE"] * 16,
        )
    return root


def test_complete_sources_and_old_meanings_retained(tmp_path):
    api = module()
    root = measured_rcab(tmp_path)
    sources = load(root)
    old = assemble_rcabench_live_inputs(sources, q_base=1, q_max=1)
    repaired = api.assemble_rcabench_repair_diagnostics(sources, q_base=1, q_max=1)
    assert repaired.legacy.inference_descriptor() == old.inference_descriptor()
    assert repaired.legacy.candidate_universe.candidate_count == 3
    assert [s.signal_id for s in repaired.signals] == [s.signal_id for s in old.signals]
    assert repaired.verify_sources(root)
    descriptor = repaired.diagnostic_descriptor()
    assert descriptor["schema_version"] == "nurse.v2.live-repair-diagnostic-inputs/1"
    assert descriptor["ranking_ready"] is False
    assert "PRIVATE RAW PROSE" not in json.dumps(descriptor)
    assert "private-answer-case" not in json.dumps(descriptor)


def test_positive_scale_and_short_readiness_survive_exact_metric_join(tmp_path):
    api = module()
    result = api.assemble_rcabench_repair_diagnostics(load(measured_rcab(tmp_path)))
    by_metric = {s.metric_name: s for s in result.signals if s.metric_name}
    cpu = by_metric["container.cpu.usage"]
    assert cpu.quality.source_summary.evidence_state == "unknown"
    assert cpu.quality.formal_scoring_eligible
    assert cpu.quality.signed_standardized_change == -18.0
    ready = by_metric["k8s.statefulset.ready_pods"]
    desired = by_metric["k8s.statefulset.desired_pods"]
    assert ready.quality.evidence_state == "verified_normal"
    assert ready.lifecycle.availability_support
    assert ready.lifecycle.events[0].zero_timestamps_ns == ((T0 + 90) * 10**9,)
    assert not desired.lifecycle.availability_support
    assert desired.lifecycle.formal_event_eligible
    assert not by_metric["unregistered_cpu"].quality.formal_scoring_eligible


def test_trace_and_log_records_preserve_unknown_denominator_and_all_rows(tmp_path):
    api = module()
    sources = load(measured_rcab(tmp_path))
    result = api.assemble_rcabench_repair_diagnostics(sources)
    assert len(result.errors) == 4
    errors = {(e.modality, e.interval.start_unix_ns): e for e in result.errors}
    baseline = errors[("trace", (T0 - 480) * 10**9)]
    fault = errors[("trace", T0 * 10**9)]
    assert baseline.groups[0].known_denominator == 0
    assert baseline.groups[0].unknown_count == 16
    assert fault.groups[0].error_count == fault.groups[0].known_denominator == 16
    assert fault.groups[0].span_kind == "server"
    assert fault.groups[0].entity_key == service_key("trace-only-service")
    assert all(e.input_record_count == 16 for e in result.errors)
    assert all(e.sampling_coverage == "unknown" for e in result.errors)
    assert all("message" not in r.model_dump() for e in result.errors for r in e.source_records)


def test_aiops_has_distinct_status_placeholder_and_heuristic_limits(tmp_path):
    api = module()
    root = measured_case(tmp_path)
    result = api.assemble_aiops25_repair_diagnostics(load_aiops(root))
    assert result.dataset == "aiops25"
    assert result.legacy.candidate_universe.candidate_count == 3
    assert len(result.signals) == 8
    process = next(s for s in result.signals if s.metric_name == "pod_processes")
    assert process.lifecycle.state_meaning == "process_count_zero"
    assert not process.lifecycle.availability_support
    assert all(
        g.known_denominator == 0 for e in result.errors if e.modality == "trace" for g in e.groups
    )
    assert all(
        e.severity_basis == "preprocessor_message_heuristic"
        for e in result.errors
        if e.modality == "log"
    )
    assert result.verify_sources(root)


@pytest.mark.parametrize("mutation", ["quality", "metric", "drop_signal", "drop_error", "dataset"])
def test_self_consistent_or_partial_result_is_not_source_permission(tmp_path, mutation):
    api = module()
    root = measured_rcab(tmp_path)
    result = api.assemble_rcabench_repair_diagnostics(load(root))
    cpu = next(s for s in result.signals if s.metric_name == "container.cpu.usage")
    if mutation == "quality":
        forged = cpu.quality.model_copy(update={"signed_standardized_change": 18.0})
        result = replace(
            result,
            signals=tuple(replace(s, quality=forged) if s is cpu else s for s in result.signals),
        )
    elif mutation == "metric":
        result = replace(
            result,
            signals=tuple(
                replace(s, metric_name="k8s.container.ready") if s is cpu else s
                for s in result.signals
            ),
        )
    elif mutation == "drop_signal":
        result = replace(result, signals=result.signals[1:])
    elif mutation == "drop_error":
        result = replace(result, errors=result.errors[1:])
    else:
        result = replace(result, dataset="aiops25")
    assert not result.verify_sources(root)


def test_diagnostic_identity_is_registered_without_legacy_relabelling():
    api = module()
    registry = importlib.import_module("nurse_rcl.v2.schema_compatibility")
    assert registry.CURRENT_SCHEMA_VERSIONS.get("live_repair_diagnostic_inputs") == (
        api.REPAIR_DIAGNOSTIC_SCHEMA_VERSION
    )
    assert "live_repair_diagnostic_inputs" not in registry.RECOGNIZED_LEGACY_SCHEMA_VERSIONS


def test_nonfinite_duration_unknown_severity_and_unbound_entities_remain_visible(tmp_path):
    api = module()
    root = measured_rcab(tmp_path)
    write(
        root,
        "abnormal_traces.parquet",
        [T0 + 1, T0 + 2, T0 + 3],
        trace_id=["t1", "t2", "t3"],
        span_id=["s"] * 3,
        duration=[float("nan"), -1.0, 999_000_000_000.0],
        service_name=[None] * 3,
        **{"attr.status_code": ["Error"] * 3},
    )
    write(
        root,
        "abnormal_logs.parquet",
        [T0 + 1],
        level=["UNKNOWN"],
        service_name=[None],
        message=["PRIVATE RAW PROSE"],
    )
    result = api.assemble_rcabench_repair_diagnostics(load(root))
    fault = [e for e in result.errors if e.interval.start_unix_ns == T0 * 10**9]
    trace = next(e for e in fault if e.modality == "trace")
    assert trace.input_record_count == 3
    assert trace.groups[0].excluded_count == 3
    assert trace.groups[0].entity_key.startswith("unbound_")
    log = next(e for e in fault if e.modality == "log")
    assert log.input_record_count == log.groups[0].unknown_count == 1
    assert result.verify_sources(root)


def test_source_status_change_is_detected_without_change_in_old_latency(tmp_path):
    api = module()
    root = measured_rcab(tmp_path)
    result = api.assemble_rcabench_repair_diagnostics(load(root))
    path = root / "abnormal_traces.parquet"
    table = pq.read_table(path)
    column = "attr.http.response.status_code"
    pq.write_table(
        table.set_column(
            table.column_names.index(column), column, pa.array([200] * table.num_rows)
        ),
        path,
    )
    assert not result.verify_sources(root)


def test_label_change_has_no_effect(tmp_path):
    api = module()
    root = measured_rcab(tmp_path)
    result = api.assemble_rcabench_repair_diagnostics(load(root))
    (root / "injection.json").write_text("new forbidden answer")
    assert result.verify_sources(root)


def test_dataset_entrypoints_and_legacy_types_cannot_be_swapped(tmp_path):
    api = module()
    rcab = load(measured_rcab(tmp_path))
    aiops = load_aiops(measured_case(tmp_path / "aiops"))
    with pytest.raises(ValueError, match="source"):
        api.assemble_rcabench_repair_diagnostics(aiops)
    with pytest.raises(ValueError, match="source"):
        api.assemble_aiops25_repair_diagnostics(rcab)


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("target", ["original", "fresh"])
def test_source_mutation_during_replay_fails_on_both_exits(tmp_path, monkeypatch, raises, target):
    api = module()
    root = measured_rcab(tmp_path)
    result = api.assemble_rcabench_repair_diagnostics(load(root))
    original_assemble = api.assemble_rcabench_repair_diagnostics

    def mutate(sources, **kwargs):
        rebuilt = original_assemble(sources, **kwargs)
        victim = result.legacy.sources if target == "original" else sources
        object.__setattr__(victim, "tables", victim.tables[1:])
        if raises:
            raise ValueError("induced downstream failure")
        return rebuilt

    monkeypatch.setattr(api, "assemble_rcabench_repair_diagnostics", mutate)
    assert not result.verify_sources(root)
