"""Only synthetic metadata: typed purposes bind supply, never grade native prose."""

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from tests.test_v2_live_sre_action_entries import issued, metadata


def api():
    name = "nurse_rcl.v2.live.ops_sre_observable_scope"
    assert util.find_spec(name), "typed observable supply binder is missing"
    return import_module(name)


def table(kind="fixed_summary", metric="usage"):
    caps = metadata()
    caps["directed_sources"]["entries"][0].update(
        record_kind=kind,
        record_version="ops-symptoms/1",
        metric_name=metric,
    )
    caps, book = issued(caps)
    result = api().compile_observable_entries(
        caps,
        book=book,
        request_index=1,
        case_id="opaque-case",
    )
    return result, book


def selected(result):
    return next(e for e in result["entries"] if e["action"]["tool"] == "inspect_sources")


def declaration(entry):
    c = entry["observable_contract"]
    return dict(
        entry_ref=entry["entry_ref"],
        objective_kind=c["objective_kind"],
        metric_name=c["metric_name"],
        requested_facets=list(c["allowed_facets"]),
    )


def test_registered_metric_contract_is_source_bound_metadata_not_values():
    result, book = table()
    entry = selected(result)
    contract = entry["observable_contract"]
    assert result["schema_version"] == "nurse.v2.sre-action-entries/2"
    assert contract["objective_kind"] == "metric_summary"
    assert contract["metric_name"] == "usage"
    assert contract["allowed_facets"] == [
        "baseline_value",
        "fault_value",
        "unit",
        "coverage",
        "episode_status",
    ]
    assert contract["authority"] == "registered_supply_not_delivered_evidence"
    assert contract["binding"]["source_prefix"] == result["source_prefix"]
    assert contract["binding"]["record_sha256"] == "c" * 64
    assert contract["binding"]["targets"] == ["candidate"]
    assert book.resolve(entry["entry_ref"], "action", visible_at=1) == entry["entry_id"]
    assert "values" not in contract and "truth_value" not in contract


@pytest.mark.parametrize(
    "kind,metric,expected",
    [
        ("error_summary", None, "error_summary"),
        ("fixed_summary", None, "record_read"),
        ("unknown_type", "usage", "record_read"),
    ],
)
def test_unknown_and_error_types_never_guess_cpu(kind, metric, expected):
    result, _ = table(kind, metric)
    contract = selected(result)["observable_contract"]
    assert contract["objective_kind"] == expected
    if expected != "metric_summary":
        assert contract["metric_name"] is None


def test_error_entry_cannot_supply_declared_cpu_purpose_but_selection_is_untouched():
    result, _ = table("error_summary", None)
    entry = selected(result)
    claim = declaration(entry)
    claim.update(objective_kind="metric_summary", metric_name="usage")
    selection = dict(entry_ref=entry["entry_ref"], question="Compare CPU.")
    before = deepcopy((result, selection, claim))
    admission = api().admit_observable_test(claim, selection=selection, table=result)
    assert admission == dict(status="rejected", code="objective_not_supplied", scope=None)
    assert (result, selection, claim) == before


@pytest.mark.parametrize("bad", ["entry", "metric", "facet", "subject", "duplicate"])
def test_purpose_binding_rejects_wrong_declared_structure_not_free_prose(bad):
    result, _ = table()
    entry = selected(result)
    claim = declaration(entry)
    if bad == "entry":
        claim["entry_ref"] = "A_future"
    elif bad == "metric":
        claim["metric_name"] = "error_count"
    elif bad == "subject":
        claim["target_service_keys"] = ["another-subject"]
    elif bad == "duplicate":
        claim["requested_facets"].append(claim["requested_facets"][0])
    else:
        claim["requested_facets"] = ["saturation_confirmed"]
    assert (
        api().admit_observable_test(
            claim,
            selection=dict(entry_ref=entry["entry_ref"], question="unchanged prose"),
            table=result,
        )["status"]
        == "rejected"
    )


def test_accepted_scope_retains_exact_selected_binding_without_solving_question():
    result, _ = table()
    entry = selected(result)
    claim = declaration(entry)
    before = deepcopy(claim)
    admitted = api().admit_observable_test(
        claim,
        selection=dict(entry_ref=entry["entry_ref"], question="Wrong prose stays native"),
        table=result,
    )
    assert admitted["status"] == "accepted"
    assert admitted["scope"]["requested_facets"] == claim["requested_facets"]
    assert admitted["scope"]["binding"] == entry["observable_contract"]["binding"]
    assert "answered" not in canonical_json_bytes(admitted).decode()
    assert claim == before


def test_legacy_entry_table_stays_legacy_and_is_not_a_new_purpose_context():
    from tests.test_v2_live_sre_action_entries import compile_table

    caps, book = issued()
    old = compile_table(caps, book)
    assert old["schema_version"] == "nurse.v2.sre-action-entries/1"
    assert all("observable_contract" not in e for e in old["entries"])
    assert api().admit_observable_test(None, selection=None, table=old)["code"] == (
        "observable_context_required"
    )
