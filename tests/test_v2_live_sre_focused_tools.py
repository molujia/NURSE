"""Bounded evidence navigation over synthetic registered metadata only."""

from copy import deepcopy
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def api(part):
    name = "nurse_rcl.v2.live.ops_sre_focused_" + part
    assert util.find_spec(name), "focused evidence capability is absent"
    return import_module(name)


def row(index=0, candidate="svc-a", **changes):
    value = dict(
        card_id=f"card-{index:02}",
        node="symptoms",
        targets=[candidate],
        source_role="core",
        tool_version="ops-symptoms/1",
        parameters={},
        source_refs=["a" * 64],
        evidence_ids=["observation"],
        coverage={},
        payload=dict(
            kind="prepared_summary",
            metric_name=f"metric-{index:02}",
            signal_family="cpu",
            baseline_window=[0, 10],
            fault_window=[10, 20],
        ),
        source_attribution=dict(
            schema_version="nurse.v2.sre-source-attribution/1",
            status="registered",
            metric_name=f"metric-{index:02}",
            signal_family="cpu",
            modality="metrics",
            statistic="mean",
            unit="percent",
            baseline_window=[0, 10],
            fault_window=[10, 20],
            telemetry_role="unknown",
            span_kind=None,
            denominator="samples",
            coverage=dict(baseline=1, fault=1),
        ),
    )
    value.update(changes)
    return value


def test_catalog_exact_filters_cursor_and_action_atom_budget():
    cls = api("catalog").EvidenceCatalog
    rows = [row(i) for i in range(25)]
    catalog = cls(snapshot="1" * 64, rows=rows, requests={})
    filters = dict(candidate_service_keys=["svc-a"])
    first = catalog.browse(filters=filters, cursor=None)
    assert 0 < len(first["items"]) <= 16
    assert sum(len(item["actions"]) for item in first["items"]) <= 16
    assert first["authority"] == "navigation_not_delivered_evidence"
    assert first["next_cursor"] is not None
    target = catalog.browse(filters=dict(filters, metric_name="metric-24"), cursor=None)
    assert [item["card_id"] for item in target["items"]] == ["card-24"]
    assert all(
        item["representation"] == "navigation_reference_not_delivered_detail"
        for item in target["items"]
    )
    with pytest.raises(ValueError):
        catalog.browse(filters=dict(filters, metric_name="metric-24"), cursor=first["next_cursor"])
    state = catalog.export_state()
    restored = cls(snapshot="1" * 64, rows=rows, requests={})
    restored.restore_state(state)
    assert restored.browse(filters=filters, cursor=first["next_cursor"]) == catalog.browse(
        filters=filters, cursor=first["next_cursor"]
    )
    forged = deepcopy(state)
    forged["rows"][0]["metric_name"] = "changed"
    unsigned = {k: v for k, v in forged.items() if k != "state_sha256"}
    forged["state_sha256"] = sha256_hex(canonical_json_bytes(unsigned))
    with pytest.raises(ValueError):
        restored.restore_state(forged)


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"candidate_service_keys": []},
        {"candidate_service_keys": ["svc-a", "svc-a"]},
        {"candidate_service_keys": ["svc-a", "svc-b", "svc-c"]},
        {"candidate_service_keys": ["svc-a"], "path": "/tmp/telemetry"},
    ],
)
def test_catalog_rejects_unbounded_or_unregistered_filter_shapes(filters):
    catalog = api("catalog").EvidenceCatalog(snapshot="1" * 64, rows=[row()], requests={})
    with pytest.raises(ValueError):
        catalog.browse(filters=filters, cursor=None)


def test_catalog_restore_rejects_forged_cursor_between_actual_action_atom_pages():
    cls = api("catalog").EvidenceCatalog
    rows = [row(i) for i in range(25)]
    catalog = cls(snapshot="1" * 64, rows=rows, requests={})
    catalog.browse(filters=dict(candidate_service_keys=["svc-a"]), cursor=None)
    forged = catalog.export_state()
    token, binding = next(iter(forged["cursors"].items()))
    del forged["cursors"][token]
    binding["offset"] = 1
    forged["cursors"]["sre_evidence_cursor_" + catalog._digest(binding)] = binding
    unsigned = {k: v for k, v in forged.items() if k != "state_sha256"}
    forged["state_sha256"] = sha256_hex(canonical_json_bytes(unsigned))
    with pytest.raises(ValueError):
        cls(snapshot="1" * 64, rows=rows, requests={}).restore_state(forged)


@pytest.mark.parametrize("candidate", ["svc-a", "svc-b"])
def test_catalog_publishes_one_bound_actual_two_card_comparison_atom(candidate):
    cls = api("catalog").EvidenceCatalog
    rows = [row(0), row(1, candidate=candidate)]
    catalog = cls(snapshot="1" * 64, rows=rows, requests={})
    page = catalog.browse(
        filters=dict(candidate_service_keys=list(dict.fromkeys(["svc-a", candidate]))), cursor=None
    )
    pairs = [
        action
        for item in page["items"]
        for action in item["actions"]
        if action["tool"] == "inspect_sources" and len(action["parameters"]["card_ids"]) == 2
    ]
    assert len(pairs) == 1, "atomic focused selection cannot reach the actual two-card comparison"
    assert pairs[0]["parameters"]["card_ids"] == [r["card_id"] for r in rows]
    assert len(pairs[0]["binding_sha256"]) == 64
    assert sum(len(item["actions"]) for item in page["items"]) <= 16
    from tests.test_v2_live_sre_action_entries import issued, metadata

    caps = metadata()
    caps["evidence_catalog_targets"] = page["items"]
    caps, book = issued(caps)
    table = api("decision").compile_focused_entries(
        caps, book=book, request_index=1, case_id="opaque-case"
    )
    assert any(
        e["action"]["tool"] == "inspect_sources"
        and len(e["action"]["parameters"]["card_ids"]) == 2
        and e["source_binding"] == dict(route_binding_sha256=pairs[0]["binding_sha256"])
        for e in table["entries"]
    )


def test_new_profile_registered_without_mutating_old_profile():
    from nurse_rcl.v2.live import ops_contracts

    cls = getattr(ops_contracts, "OpsSreFocusedProfile", None)
    assert cls is not None, "focused profile is not registered"
    profile = cls()
    assert profile.profile_id == "v2-ops-sre/15"
    assert profile.investigation_profile_id == "hermes-sre-tools/25"
    assert (
        ops_contracts.OpsSreAtomicSelectorProfile().investigation_profile_id
        == "hermes-sre-tools/23"
    )
    assert (
        ops_contracts.resolve_sre_profile(sre=True, working=False, profile="hermes-sre-tools/25")
        == "hermes-sre-tools/25"
    )


def test_focused_parser_plural_navigation_is_isolated_and_past_only():
    from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
    from nurse_rcl.v2.live.ops_sre_atomic_selector_decision import selection_shape as old_shape
    from tests.test_v2_live_sre_action_entries import issued, metadata

    module = api("decision")
    caps, book = issued(metadata())
    table = module.compile_focused_entries(caps, book=book, request_index=1, case_id="opaque-case")
    service = next(r[1] for r in book._rows if r[0] == "service")
    nav = dict(
        navigation=dict(
            kind="catalog",
            catalog_kind="evidence",
            filters=dict(candidate_service_keys=[service]),
            cursor=None,
        ),
        question="Read metadata.",
    )
    module.selection_shape(nav)
    with pytest.raises(ExchangeBoundaryError):
        old_shape(nav)
    result = module.resolve_focused_selection(nav, table=table, book=book, request_index=1)
    assert result["status"] == "accepted"
    assert result["action"]["parameters"]["kind"] == "evidence"
    future = book.issue("service", "future-candidate", visible_at=2)
    nav["navigation"]["filters"]["candidate_service_keys"] = [future]
    assert (
        module.resolve_focused_selection(nav, table=table, book=book, request_index=1)["status"]
        == "rejected"
    )


def test_last_catalog_exact_actions_precede_generic_round_robin():
    from tests.test_v2_live_sre_action_entries import issued, metadata

    caps = metadata()
    target = deepcopy(caps["directed_sources"]["entries"][0])
    target["card_id"] = "late-card"
    target["entry"]["parameters"]["card_ids"] = ["late-card"]
    target["complete_entry"]["parameters"]["card_id"] = "late-card"
    caps["evidence_catalog_targets"] = [
        dict(target, actions=[target["complete_entry"], target["entry"]])
    ]
    caps, book = issued(caps)
    table = api("decision").compile_focused_entries(
        caps, book=book, request_index=1, case_id="opaque-case"
    )
    assert [
        e["action"]["parameters"].get("card_id", e["action"]["parameters"].get("card_ids"))
        for e in table["entries"][:2]
    ] == ["late-card", ["late-card"]]
    assert len(table["entries"]) <= 16
