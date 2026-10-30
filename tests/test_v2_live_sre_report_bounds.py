"""Fixed report partitions preserve bytes/depth without a shared node quota."""

import importlib
import importlib.util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, encode_json
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS


def bounds():
    name = "nurse_rcl.v2.live.ops_sre_report_bounds"
    assert importlib.util.find_spec(name), "report partition boundary not implemented"
    return importlib.import_module(name)


def test_old_aggregate_limit_does_not_precede_report_partitioning():
    metadata = {"schema_version": "nurse.v2.sre-report-input/2",
                "steps": ["kept"] * 12000}
    facts = [{"value": index} for index in range(30000)]
    with pytest.raises(ExchangeBoundaryError, match="json_limit"):
        encode_json(dict(metadata, facts=facts), REPORT_LIMITS)
    stats = bounds().validate_report_parts(metadata, facts, layout="view")
    assert stats["fact_nodes"] == 90001
    assert stats["metadata_nodes"] < 20000
    assert stats["logical_bytes"] == len(canonical_json_bytes(dict(metadata, facts=facts)))
    assert stats["logical_depth"] == 3


@pytest.mark.parametrize("layout", ["context", "view"])
@pytest.mark.parametrize("value", [False, 0, 0.0, "0", None, '中\n"\\', [], {}])
def test_complete_byte_and_depth_accounting_matches_json(layout, value):
    metadata = {"evidence": {}} if layout == "context" else {}
    facts = [{"value": value}]
    full = {"evidence": {"facts": facts}} if layout == "context" else {"facts": facts}
    stats = bounds().validate_report_parts(metadata, facts, layout=layout)
    assert stats["logical_bytes"] == len(encode_json(full, REPORT_LIMITS))
    assert stats["logical_depth"] == (4 if layout == "context" else 3) + int(
        type(value) in (list, dict))


@pytest.mark.parametrize("field", ["facts", "metadata"])
def test_partition_node_limits_are_cumulative(field):
    facts = [{"value": 0}] * 33334 if field == "facts" else []
    metadata = {"steps": [0] * 20000} if field == "metadata" else {}
    with pytest.raises(ValueError) as exc:
        bounds().validate_report_parts(metadata, facts, layout="view")
    assert exc.value.boundary == field + "_nodes"
    assert exc.value.observed > exc.value.limit


def test_combined_bytes_remain_four_mib():
    metadata = {"text": "x" * 2100000}
    facts = [{"text": "y" * 2100000}]
    with pytest.raises(ValueError) as exc:
        bounds().validate_report_parts(metadata, facts, layout="view")
    assert exc.value.boundary == "logical_bytes"
    assert exc.value.observed > 4194304


def test_absolute_depth_counts_context_parents():
    nested = []
    for _ in range(62):
        nested = [nested]
    with pytest.raises(ValueError) as exc:
        bounds().validate_report_parts({"evidence": {}}, nested, layout="context")
    assert exc.value.boundary == "logical_depth"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"not json", {1: 2}])
def test_invalid_json_is_not_admitted_by_partitioning(value):
    with pytest.raises(ValueError):
        bounds().validate_report_parts({}, [value], layout="view")


def test_cycles_and_already_present_fact_fields_are_rejected():
    cycle = []
    cycle.append(cycle)
    for metadata, facts, layout in (({}, cycle, "view"), ({"facts": []}, [], "view"),
            ({"evidence": {"facts": []}}, [], "context"), ({}, [], "arbitrary")):
        with pytest.raises(ValueError):
            bounds().validate_report_parts(metadata, facts, layout=layout)


def test_wide_partition_rejected_without_visiting_children():
    class InvalidChild:
        pass
    with pytest.raises(ValueError) as exc:
        bounds().validate_report_parts({}, [InvalidChild()] * 100001, layout="view")
    assert exc.value.boundary == "facts_nodes"
