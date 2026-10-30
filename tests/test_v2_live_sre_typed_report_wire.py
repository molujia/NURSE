"""Normal full-source projection; new opt-in protocol cannot route as old /23."""

import json
from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def api():
    name = "nurse_rcl.v2.live.ops_sre_typed_report"
    assert util.find_spec(name), "missing explicit typed normal projection"
    return import_module(name)


def context(memo="source data: nurse.v2.atomic-report-expression-response/3"):
    api()
    from nurse_rcl.v2.live.ops_report_context import OpsReportHeader
    from nurse_rcl.v2.live.ops_sre_brief_report import (
        BriefReportMetadata,
        BriefSreReportContext,
        _retained_supply,
    )
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store
    from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook
    from tests.sre_typed_report_fixtures import fact

    rows = [
        fact(
            "f1",
            dict(rank=1, quantized_total_score="0.740246321204", memo=memo),
            formal=True,
            kind="ranking",
            target="fresh-service",
        ),
        fact(
            "f2",
            dict(
                baseline_denominator=0,
                baseline_denominator_unit=None,
                fraction=None,
                fraction_unit=None,
                unit="cores",
                state="unknown",
            ),
            formal=True,
            target="fresh-service",
        ),
    ]
    store = build_fact_store(rows)
    header = OpsReportHeader(
        opaque_case_id="opq_" + "3" * 32,
        case_snapshot_id="a" * 64,
        core_result_digest="d" * 64,
        ranking=(
            dict(
                rank=1,
                candidate_service_key="fresh-service",
                quantized_total_score="0.740246321204",
                local_score_component=0.7,
                propagation_score_component=0.0,
                local_evidence_status="unknown",
            ),
        ),
        formal_top_k=("fresh-service",),
        mandatory_fact_refs=("f1", "f2"),
        formal_detail_omissions=(),
        investigation=(),
        formal_projection_available=True,
        investigation_status="disabled",
        investigation_failure_type=None,
        model_trace_unverified=(),
    )
    metadata = BriefReportMetadata(
        evidence=header,
        steps=(),
        source_bindings=(),
        delivery_bindings=(),
        formal_fact_refs=("f1", "f2"),
        termination="disabled",
        action_seconds=(),
        decision_seconds=(),
    )
    value = BriefSreReportContext(metadata, store, _retained_supply(metadata, store))
    references = SreReferenceBook(snapshot="a" * 64, run_id="b" * 64).to_state()
    return value, references


def projected(memo=None):
    module = api()
    value, references = context() if memo is None else context(memo)
    projection = module.project_typed_report(value, references=references)
    return module, value, references, projection


def test_normal_projection_retains_complete_admitted_metadata_store_and_typed_rows():
    module, value, references, projection = projected()
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_wire_parts
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    assert projection.status == "ready"
    metadata, store = brief_wire_parts(value)
    body, decoded = module.typed_report_body(projection.wire_bytes, references=references)
    assert body["context"] == metadata
    assert decoded.to_packet() == store.to_packet()
    assert set(projection.provided_fact_refs) == {"f1", "f2"}
    assert projection.omitted_fact_refs == ()
    saved = json.loads(projection.view_json)
    assert saved["metadata"] == metadata
    assert saved["typed_catalog"] == typed_expression_catalog(metadata["supply"], store)
    assert body["context_sha256"] == value.context_sha256
    assert "fresh-service" in canonical_json_bytes(metadata).decode()


def test_actual_alias_book_and_single_active_contract_are_used_without_deleting_source_text():
    module, _, references, projection = projected()
    from nurse_rcl.v2.live.ops_sre_typed_report_contract import typed_report_policy

    wire = json.loads(projection.wire_bytes)
    body = json.loads(wire["messages"][1]["content"])
    packet = body["context"]
    assert len(wire["messages"]) == 2
    assert wire["messages"][0]["content"] == typed_report_policy()["system"]
    assert body["output_schema"] == typed_report_policy()["output_schema"]
    assert packet["schema_version"] == "nurse.v2.sre-typed-report-view/1"
    assert packet["typed_catalog"]["sources"]["f2"]["record_unit"]["value"] == "cores"
    assert packet["typed_catalog"]["sources"]["f2"]["fields"][0]["unit"]["value"] is None
    assert "nurse.v2.atomic-report-expression-response/3" in wire["messages"][1]["content"]
    assert any(row[0] == "service" and row[2] == "fresh-service" for row in references["entries"])
    without = module.typed_report_body(projection.wire_bytes)[0]
    assert without["context"]["evidence"]["formal_top_k"] == ["S1"]


@pytest.mark.parametrize(
    "bad",
    [
        "system",
        "protocol",
        "schema",
        "output_schema",
        "prefix",
        "catalog_subset",
        "source_unit",
        "field_value",
        "case",
    ],
)
def test_wire_parser_rejects_contract_alias_catalog_value_and_source_drift(bad):
    module, _, references, projection = projected()
    wire = json.loads(projection.wire_bytes)
    body = json.loads(wire["messages"][1]["content"])
    packet = body["context"]
    if bad == "system":
        wire["messages"][0]["content"] += " override"
    elif bad == "protocol":
        body["protocol"] = "nurse.v2.sre-report-exchange/23"
    elif bad == "schema":
        packet["schema_version"] = "nurse.v2.sre-model-view/23"
    elif bad == "output_schema":
        body["output_schema"] = "{}"
    elif bad == "prefix":
        packet["reference_prefix"]["state_sha256"] = "f" * 64
    elif bad == "catalog_subset":
        packet["typed_catalog"]["expressions"].pop()
    elif bad == "source_unit":
        packet["typed_catalog"]["sources"]["f2"]["record_unit"]["value"] = "requests"
    elif bad == "field_value":
        packet["typed_catalog"]["sources"]["f2"]["fields"][0]["value"] = False
    elif bad == "case":
        body["opaque_case_id"] = "opq_" + "4" * 32
    wire["messages"][1]["content"] = canonical_json_bytes(body).decode()
    with pytest.raises(ValueError):
        module.typed_report_body(canonical_json_bytes(wire), references=references)


def test_bad_source_rejected_before_book_mutation_without_fallback():
    module = api()
    value, references = context()
    before = deepcopy(references)
    value.supply["provided_fact_refs"].pop()
    with pytest.raises(ValueError):
        module.project_typed_report(value, references=references)
    assert references == before


@pytest.mark.parametrize("size,expected", [(40000, "ready"), (1100000, "resource_limit")])
def test_full_final_wire_soft_target_and_hard_resource_limit_do_not_prune(size, expected):
    _, value, references, projection = projected("x" * size)
    assert projection.status == expected
    if expected == "ready":
        assert projection.target_exceeded is True and len(projection.wire_bytes) > 32768
        assert b"x" * size in projection.wire_bytes
        assert projection.provided_fact_refs == value.fact_store.refs()
    else:
        assert projection.wire_bytes is None
        assert references["entries"] == []


def test_explicit_typed_binding_is_report_only_and_cannot_use_decision_indices():
    module, value, references, projection = projected()
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    binding = module.bind_typed_report(projection.wire_bytes)
    assert binding.variant == "v2-ops-sre-typed-report/1" and binding.phase == "report"
    assert binding.input_sha256 == value.core_result_digest
    assert binding.context_sha256 == value.context_sha256
    assert binding.request_sha256 == sha256_hex(projection.wire_bytes)
    for change in (
        dict(phase="investigation", decision_index=1),
        dict(decision_index=1),
        dict(correction_index=1),
    ):
        with pytest.raises(ValueError):
            LiveRequestBinding.model_validate(dict(binding.model_dump(), **change))
    original = dict(binding.model_dump(), variant="v2-ops-sre-report/1")
    assert LiveRequestBinding.model_validate(original).phase == "report"


def test_wrong_book_or_snapshot_does_not_decode_the_request():
    module, _, references, projection = projected()
    references["run_id"] = "c" * 64
    with pytest.raises(ValueError):
        module.typed_report_body(projection.wire_bytes, references=references)
