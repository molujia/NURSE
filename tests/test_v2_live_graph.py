import importlib
from dataclasses import replace

import pytest

from tests.test_v2_live_assembly import assemble
from tests.test_v2_live_sources import T0, case, load, write


TRACE, OTHER_TRACE = '1' * 32, '2' * 32
PARENT, CHILD, FOREIGN = '3' * 16, '4' * 16, '5' * 16


def graph(inputs, root):
    return importlib.import_module('nurse_rcl.v2.live.graph').build_live_graph(inputs, source_root=root)


def test_graph_joins_only_exact_trace_parent_ids_and_keeps_unknown_coverage(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_traces.parquet', [T0 - 60, T0 - 59, T0 - 58],
          service_name=['caller', 'callee', 'foreign'], trace_id=[TRACE, TRACE, OTHER_TRACE],
          span_id=[PARENT, CHILD, FOREIGN], parent_span_id=[None, PARENT, PARENT],
          duration=[1000000, 2000000, 3000000])
    inputs = assemble(load(root))
    result = graph(inputs, root)
    snapshot = result.snapshots.normal_snapshot
    assert len(snapshot.edges) == 1
    edge = snapshot.edges[0]
    assert edge.relation_type == 'service_call'
    assert len(result.snapshots.fault_snapshot.edges) == 0
    parents = result.manifest.parent_records
    calls = [p for p in parents if p.parent_artifact_kind == 'call_measurement']
    assert len(calls) == 1
    assert calls[0].request_count == 1
    assert calls[0].error_count is None
    assert calls[0].latency_p95_seconds == .002
    assert all(p.collection_state == 'unknown' for p in parents if p.parent_artifact_kind == 'coverage_samples')
    assert result.accounting['unmatched_parent'] == 1
    # These are actual versioned exports, not source digests disguised as export IDs.
    assert len(result.exports) == 2
    assert {field.raw_value for p in parents for field in p.provenance_fields} <= {
        export.export_id for export in result.exports}
    assert result.verify_sources(inputs, root)


def test_missing_traces_do_not_create_edges_or_verified_normal_coverage(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_logs.parquet', [T0 - 1], service_name=['svc'], level=['INFO'], message=['log'])
    result = graph(assemble(load(root)), root)
    assert not result.snapshots.normal_snapshot.edges
    assert not result.snapshots.fault_snapshot.edges
    assert all(p.collection_state == 'unknown' for p in result.manifest.parent_records)


def test_conflicting_duplicate_span_id_is_not_arbitrarily_joined(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_traces.parquet', [T0 - 60, T0 - 60, T0 - 59],
          service_name=['caller-a', 'caller-b', 'callee'], trace_id=[TRACE] * 3,
          span_id=[PARENT, PARENT, CHILD], parent_span_id=[None, None, PARENT], duration=[10, 10, 20])
    result = graph(assemble(load(root)), root)
    assert not result.snapshots.normal_snapshot.edges
    assert result.accounting['ambiguous_span_identity'] == 1


@pytest.mark.parametrize('trace,span,parent', [
    ('t', CHILD, PARENT), (TRACE, 's', PARENT), (TRACE, CHILD, 'p'),
    ('0' * 32, CHILD, PARENT), (TRACE, '0' * 16, PARENT),
])
def test_invalid_trace_identity_cannot_become_a_call(tmp_path, trace, span, parent):
    root = case(tmp_path / 'case')
    write(root, 'normal_traces.parquet', [T0 - 60, T0 - 59],
          service_name=['caller', 'callee'], trace_id=[TRACE, trace],
          span_id=[PARENT, span], parent_span_id=[None, parent], duration=[10, 20])
    result = graph(assemble(load(root)), root)
    assert not result.snapshots.normal_snapshot.edges
    assert result.accounting['invalid_or_unfinished_span'] == 1


def test_graph_requires_actual_source_replay_and_rejects_changed_export(tmp_path):
    root = case(tmp_path / 'case')
    write(root, 'normal_traces.parquet', [T0 - 60, T0 - 59],
          service_name=['caller', 'callee'], trace_id=[TRACE, TRACE],
          span_id=[PARENT, CHILD], parent_span_id=[None, PARENT], duration=[10, 20])
    inputs = assemble(load(root))
    result = graph(inputs, root)
    assert not replace(result, exports=()).verify_sources(inputs, root)
    write(root, 'normal_traces.parquet', [T0 - 60], service_name=['different'],
          trace_id=[TRACE], span_id=[PARENT], parent_span_id=[None], duration=[10])
    assert not result.verify_sources(inputs, root)
    with pytest.raises(ValueError, match='source replay'):
        graph(inputs, root)
