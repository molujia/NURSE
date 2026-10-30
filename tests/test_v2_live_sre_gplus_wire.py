"""Full normal source projection with exactly one active G+ contract."""

import json
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_report"
    assert util.find_spec(name), "missing G+ normal wire"
    return import_module(name)


def projected():
    module = api()
    from tests.test_v2_live_sre_typed_report_wire import context

    value, references = context()
    projection = module.project_gplus_report(value, references=references)
    return module, value, references, projection


def test_gplus_wire_has_one_active_contract_full_graph_and_actual_book_roundtrip():
    module, context, references, projection = projected()
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_wire_parts
    from nurse_rcl.v2.live.ops_sre_gplus_contract import gplus_report_policy

    assert projection.status == "ready"
    wire = json.loads(projection.wire_bytes)
    assert len(wire["messages"]) == 2
    assert wire["messages"][0]["content"] == gplus_report_policy()["system"]
    packet = json.loads(wire["messages"][1]["content"])["context"]
    assert "typed_catalog" not in packet
    assert packet["graph"]["schema_version"] == "nurse.v2.sre-gplus-evidence-dag/1"
    assert any(c["closure_admitted"] for c in packet["graph"]["choices"])
    body, store = module.gplus_report_body(projection.wire_bytes, references=references)
    expected, expected_store = brief_wire_parts(context)
    assert body["context"] == expected and store.to_packet() == expected_store.to_packet()
    assert body["context_sha256"] == context.context_sha256
    assert projection.omitted_fact_refs == ()
    assert (
        b"nurse.v2.atomic-report-expression-response/3" in projection.wire_bytes
    )  # retained source string, not active contract


@pytest.mark.parametrize(
    "bad", ["system", "schema", "protocol", "graph", "field", "prefix", "case"]
)
def test_gplus_wire_rejects_active_contract_graph_source_and_case_drift(bad):
    module, _, references, projection = projected()
    wire = json.loads(projection.wire_bytes)
    body = json.loads(wire["messages"][1]["content"])
    if bad == "system":
        wire["messages"][0]["content"] += " override"
    elif bad == "schema":
        body["schema_version"] = "nurse.v2.atomic-typed-report-input/1"
    elif bad == "protocol":
        body["protocol"] = "nurse.v2.sre-typed-report-exchange/1"
    elif bad == "graph":
        body["context"]["graph"]["edges"].pop()
    elif bad == "field":
        node = next(n for n in body["context"]["graph"]["nodes"] if n["kind"] == "field")
        node["data"]["value"] = "wrong"
    elif bad == "prefix":
        body["context"]["reference_prefix"]["state_sha256"] = "f" * 64
    else:
        body["opaque_case_id"] = "opq_" + "4" * 32
    wire["messages"][1]["content"] = canonical_json_bytes(body).decode()
    with pytest.raises(ValueError):
        module.gplus_report_body(canonical_json_bytes(wire), references=references)


def test_gplus_binding_is_explicit_report_only_not_a_third_investigation():
    module, context, _, projection = projected()
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    binding = module.bind_gplus_report(projection.wire_bytes)
    assert binding.variant == "v2-ops-sre-gplus-report/1"
    assert binding.input_sha256 == context.core_result_digest
    assert binding.request_sha256 == sha256_hex(projection.wire_bytes)
    assert binding.phase == "report"
    for change in [dict(decision_index=3), dict(correction_index=1), dict(phase="investigation")]:
        with pytest.raises(ValueError):
            LiveRequestBinding.model_validate(dict(binding.model_dump(), **change))


def test_gplus_bad_source_refused_before_mutating_trusted_book():
    module = api()
    from tests.test_v2_live_sre_typed_report_wire import context

    value, refs = context()
    before = canonical_json_bytes(refs)
    value.supply["provided_fact_refs"].pop()
    with pytest.raises(ValueError):
        module.project_gplus_report(value, references=refs)
    assert canonical_json_bytes(refs) == before


def test_explicit_table_graph_wire_has_lossless_versioned_body_and_source_binding():
    module = api()
    from tests.test_v2_live_sre_typed_report_wire import context

    value, refs = context()
    projection = module.project_gplus_report(value, references=refs, graph_representation="tables")
    assert projection.status == "ready"
    body, store = module.gplus_report_body(projection.wire_bytes, references=refs)
    assert body["schema_version"] == "nurse.v2.atomic-gplus-report-input/2"
    assert body["graph_representation"] == "tables"
    assert store.to_packet() == value.fact_store.to_packet()
    wire = json.loads(projection.wire_bytes)
    packet = json.loads(wire["messages"][1]["content"])["context"]
    assert packet["schema_version"] == "nurse.v2.sre-gplus-report-view/2"
    assert "fact_records" not in packet  # all exact facts already reside in graph context sources
    assert packet["graph"]["schema_version"] == "nurse.v2.sre-gplus-evidence-dag/2"
    packet["graph"]["edge_groups"][0]["rows"][0][0] = True
    inner = json.loads(wire["messages"][1]["content"])
    inner["context"] = packet
    wire["messages"][1]["content"] = canonical_json_bytes(inner).decode()
    with pytest.raises(ValueError):
        module.gplus_report_body(canonical_json_bytes(wire), references=refs)


def test_formal_projection_unavailable_has_explicit_unknown_numeric_detail():
    module = api()
    from nurse_rcl.v2.live.ops_sre_brief_report import BriefSreReportContext, _retained_supply
    from tests.test_v2_live_sre_typed_report_wire import context

    value, refs = context()
    metadata = value.metadata.model_copy(
        update={
            "evidence": value.metadata.evidence.model_copy(
                update={"formal_projection_available": False},
            )
        }
    )
    blocked = BriefSreReportContext(
        metadata, value.fact_store, _retained_supply(metadata, value.fact_store)
    )
    projection = module.project_gplus_report(blocked, references=refs)
    assert projection.status == "resource_limit" and projection.wire_bytes is None
    detail = getattr(projection, "failure_detail", None)
    assert detail is not None
    assert detail.stage == "report_projection"
    assert detail.failure_type == "FormalProjectionUnavailable"
    assert detail.observed is None and detail.limit is None and detail.boundary is None


def test_auto_retries_only_lossless_representation_if_full_packet_hits_boundary(monkeypatch):
    module = api()
    from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_wire_parts
    from tests.test_v2_live_sre_typed_report_wire import context

    value, refs = context()
    original = module._request

    def bounded_packet(body):
        if body["context"]["schema_version"] == "nurse.v2.sre-gplus-report-view/1":
            raise ExchangeBoundaryError(
                "json_limit",
                boundary="json_nodes",
                observed=100001,
                limit=100000,
                observed_is_lower_bound=True,
            )
        return original(body)

    monkeypatch.setattr(module, "_request", bounded_packet)
    projection = module.project_gplus_report(value, references=refs)
    assert projection.status == "ready"
    body, store = module.gplus_report_body(projection.wire_bytes, references=refs)
    expected, expected_store = brief_wire_parts(value)
    assert body["context"] == expected and store.to_packet() == expected_store.to_packet()
    assert body["graph_representation"] == "tables"
