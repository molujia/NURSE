"""Typed executable entries use issued metadata, never hidden bodies or ARK."""

# ruff: noqa: F811 -- imported real small fixture.

import importlib
import importlib.util
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_sre_references import SreReferenceBook, encode_references
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_directed_symptoms import opened
from tests.test_v2_live_sre_references import CASE, RUN, SNAPSHOT


def api():
    name = "nurse_rcl.v2.live.ops_sre_action_entries"
    assert importlib.util.find_spec(name), "executable action entry compiler is missing"
    return importlib.import_module(name)


def new_book(snapshot=SNAPSHOT):
    name = "nurse_rcl.v2.live.ops_sre_action_references"
    assert importlib.util.find_spec(name), "action reference book/2 is missing"
    return importlib.import_module(name).SreActionReferenceBook(snapshot=snapshot, run_id=RUN)


def metadata():
    """Explicit synthetic issued metadata; no fixture diagnosis or invented values."""
    return dict(
        directed_sources=dict(
            entries=[
                dict(
                    card_id="symptom",
                    record_sha256="c" * 64,
                    node="symptoms",
                    targets=["candidate"],
                    source_role="prepared_summary",
                    signal_family="cpu",
                    metric_name="usage",
                    source_refs=["source"],
                    entry=dict(
                        tool="inspect_sources",
                        parameters=dict(card_ids=["symptom"], candidate_service_keys=["candidate"]),
                    ),
                    complete_entry=dict(
                        tool="read_node",
                        parameters=dict(card_id="symptom", section="payload", cursor=None),
                    ),
                )
            ]
        ),
        investigation_routes=dict(
            entries=[
                dict(
                    skill_id="sre-scoring/14",
                    tool="read_node",
                    parameters=dict(card_id="score", section="payload", cursor=None),
                    binding_sha256="d" * 64,
                    category="scoring",
                    scope=["candidate"],
                    card_ids=["score"],
                    source_refs=["source"],
                    availability="available",
                    execution_history=[],
                    new_publication=False,
                    core_reusable=False,
                )
            ]
        ),
    )


def issued(value=None, index=1, book=None):
    value = metadata() if value is None else value
    book = new_book() if book is None else book
    encode_references(value, book=book, visible_at=index, case_id=CASE)
    return value, book


def compile_table(value, book, index=1):
    return api().compile_action_entries(value, book=book, request_index=index, case_id=CASE)


def test_action_book_is_typed_append_only_and_legacy_book_cannot_read_it():
    book = new_book()
    assert book.issue("action", "action-identity", visible_at=1) == "A1"
    assert book.issue("service", "candidate", visible_at=1) == "S1"
    restored = type(book).from_state(book.to_state(), snapshot=SNAPSHOT, run_id=RUN)
    assert restored.to_state() == book.to_state()
    assert restored.resolve("A1", "action", visible_at=1) == "action-identity"
    with pytest.raises(ValueError):
        restored.resolve("S1", "action", visible_at=1)
    with pytest.raises(ValueError):
        SreReferenceBook.from_state(book.to_state(), snapshot=SNAPSHOT, run_id=RUN)
    with pytest.raises(ValueError):
        type(book).from_state(
            SreReferenceBook(snapshot=SNAPSHOT, run_id=RUN).to_state(),
            snapshot=SNAPSHOT,
            run_id=RUN,
        )


def test_source_prefix_excludes_all_action_rows_but_binds_issuance_run_and_snapshot():
    book = new_book()
    book.issue("service", "candidate", visible_at=1)
    prefix = book.source_prefix(visible_at=1)
    book.issue("action", "old-action", visible_at=1)
    assert book.source_prefix(visible_at=1) == prefix
    book.issue("action", "later-action", visible_at=2)
    assert book.source_prefix(visible_at=2) == prefix
    book.issue("card", "later-card", visible_at=2)
    assert book.source_prefix(visible_at=1) == prefix
    assert book.source_prefix(visible_at=2) != prefix
    for kwargs in (dict(snapshot="e" * 64, run_id=RUN), dict(snapshot=SNAPSHOT, run_id="f" * 64)):
        other = type(book)(**kwargs)
        other.issue("service", "candidate", visible_at=1)
        assert other.source_prefix(visible_at=1) != prefix
    later = new_book()
    later.issue("service", "candidate", visible_at=2)
    assert later.source_prefix(visible_at=2) != prefix


def test_compiler_issues_symptom_and_score_read_without_score_symptom_combination():
    value, book = issued()
    prefix = book.source_prefix(visible_at=1)
    table = compile_table(value, book)
    assert table["schema_version"] == "nurse.v2.sre-action-entries/1"
    assert table["source_prefix"] == prefix
    assert table["shown"] == len(table["entries"]) <= 16
    assert table["authority"] == "navigation_not_delivered_evidence"
    assert any(e["action"]["tool"] == "inspect_sources" for e in table["entries"])
    for entry in table["entries"]:
        assert entry["entry_ref"].startswith("A")
        assert book.resolve(entry["entry_ref"], "action", visible_at=1) == entry["entry_id"]
        if entry["action"]["tool"] == "inspect_sources":
            assert entry["action"]["parameters"]["card_ids"] == ["symptom"]
        assert entry["action"]["skill_id"].endswith("/15")
    assert book.source_prefix(visible_at=1) == prefix
    assert compile_table(value, book) == table


@pytest.mark.parametrize("change", ["kind", "target", "unissued", "future", "foreign"])
def test_unavailable_or_wrongly_typed_symptoms_do_not_gain_an_executable_entry(change):
    value, book = issued()
    row = value["directed_sources"]["entries"][0]
    if change == "kind":
        row["node"] = "scoring"
    elif change == "target":
        row["entry"]["parameters"]["candidate_service_keys"] = ["other"]
    elif change == "future":
        book.issue("card", "future", visible_at=2)
        row["card_id"] = "future"
        row["entry"]["parameters"]["card_ids"] = ["future"]
        row["complete_entry"]["parameters"]["card_id"] = "future"
    elif change == "foreign":
        row["source_refs"] = ["other-source"]
    else:
        row["card_id"] = "unissued"
        row["entry"]["parameters"]["card_ids"] = ["unissued"]
    if change == "future":
        # A later book is not an earlier frozen book. Backdated new issuance is
        # fail-closed, not a successful earlier page with its future rows pruned.
        before = book.to_state()
        with pytest.raises(ValueError, match="issuance order"):
            compile_table(value, book)
        assert book.to_state() == before
        return
    table = compile_table(value, book)
    assert all(e["action"]["tool"] != "inspect_sources" for e in table["entries"])


def test_untyped_direct_route_cannot_bypass_the_symptom_source_contract():
    value, book = issued()
    route = deepcopy(value["investigation_routes"]["entries"][0])
    route.update(
        tool="inspect_sources",
        parameters=dict(card_ids=["score"], candidate_service_keys=["candidate"]),
    )
    value["investigation_routes"]["entries"].append(route)
    table = compile_table(value, book)
    assert not any(
        e["action"]["tool"] == "inspect_sources"
        and "score" in e["action"]["parameters"]["card_ids"]
        for e in table["entries"]
    )


def test_source_and_cursor_changes_have_distinct_entry_identity_without_mutating_old_rows():
    value, book = issued()
    first = compile_table(value, book)
    before = deepcopy(book.to_state()["entries"])
    value["investigation_routes"]["entries"][0]["parameters"]["cursor"] = "issued-cursor"
    value["investigation_routes"]["entries"][0]["binding_sha256"] = "e" * 64
    second = compile_table(value, book)
    old = next(e for e in first["entries"] if e["action"]["parameters"].get("card_id") == "score")
    new = next(e for e in second["entries"] if e["action"]["parameters"].get("card_id") == "score")
    assert (old["entry_ref"], old["entry_id"]) != (new["entry_ref"], new["entry_id"])
    assert book.to_state()["entries"][: len(before)] == before


def test_menu_is_bounded_with_honest_omission_and_no_hidden_body_read():
    value = metadata()
    template = value["directed_sources"]["entries"][0]
    value["directed_sources"]["entries"] = []
    for n in range(20):
        row = deepcopy(template)
        row["card_id"] = f"symptom-{n}"
        row["record_sha256"] = f"{n:064x}"
        row["entry"]["parameters"]["card_ids"] = [row["card_id"]]
        row["complete_entry"]["parameters"]["card_id"] = row["card_id"]
        value["directed_sources"]["entries"].append(row)
    value, book = issued(value)
    before = deepcopy(value)
    table = compile_table(value, book)
    assert table["shown"] == 16 and table["omitted"] == table["total"] - 16 > 0
    assert value == before
    assert table["navigation_kinds"] == [
        "case",
        "candidate",
        "candidate_pair",
        "dependency",
        "catalog",
    ]


def test_action_rows_share_the_original_reference_limit_and_failure_is_transactional():
    value, book = issued()
    while len(book._rows) < 4095:
        book.issue("service", f"service-{len(book._rows)}", visible_at=1)
    before = book.to_state()
    with pytest.raises(ValueError, match="resource"):
        compile_table(value, book)
    assert book.to_state() == before
    assert book.issue("action", "last", visible_at=1)
    with pytest.raises(ValueError, match="resource"):
        book.issue("action", "overflow", visible_at=1)


def test_real_issued_capability_menu_preserves_original_sources_without_dispatch(complete):
    tool = opened(complete)
    case = complete[0].catalog.opaque_case_id
    capabilities = tool.capabilities()
    book = new_book(tool.case_snapshot_id)
    encode_references(capabilities, book=book, visible_at=1, case_id=case)
    before = (canonical_json_bytes(tool.export_navigation()), complete[2].model_dump_json())
    table = api().compile_action_entries(capabilities, book=book, request_index=1, case_id=case)
    symptoms = [e for e in table["entries"] if e["action"]["tool"] == "inspect_sources"]
    assert symptoms
    for entry in symptoms:
        ids = entry["action"]["parameters"]["card_ids"]
        assert all(tool.cards[k].node == "symptoms" for k in ids)
        assert entry["source_binding"]["record_sha256"]
    assert before == (canonical_json_bytes(tool.export_navigation()), complete[2].model_dump_json())


def test_public_scope_reorders_supplied_entries_without_new_authority():
    value = metadata()
    template = value["directed_sources"]["entries"][0]
    value["directed_sources"]["entries"] = []
    for n in range(20):
        row = deepcopy(template)
        row["card_id"], row["targets"] = f"symptom-{n}", [f"candidate-{n}"]
        row["record_sha256"] = f"{n:064x}"
        row["entry"]["parameters"] = dict(
            card_ids=[row["card_id"]], candidate_service_keys=row["targets"]
        )
        row["complete_entry"]["parameters"]["card_id"] = row["card_id"]
        value["directed_sources"]["entries"].append(row)
    value, book = issued(value)
    default = compile_table(value, book)
    shown = {s for e in default["entries"] for s in e["source_context"].get("targets", [])}
    target = next(
        r["targets"][0]
        for r in value["directed_sources"]["entries"]
        if r["targets"][0] not in shown
    )
    before = deepcopy(value)
    focused = api().compile_action_entries(
        value, book=book, request_index=1, case_id=CASE, public_scope=(target, "unissued-target")
    )
    assert focused["entries"][0]["source_context"]["targets"] == [target]
    assert (focused["shown"], focused["total"], focused["omitted"]) == (
        default["shown"],
        default["total"],
        default["omitted"],
    )
    assert focused["navigation_kinds"] == default["navigation_kinds"]
    assert ("service", "unissued-target") not in book._originals
    assert value == before


def test_public_scope_comes_from_admitted_question_or_actual_latest_scope_not_old_intent():
    view = dict(
        question_state=dict(
            current_question_id="iq_live",
            questions=[
                dict(question_id="iq_old", state="open", target_service_keys=["old"]),
                dict(question_id="iq_live", state="open", target_service_keys=["current"]),
            ],
        ),
        previous_feedback=dict(scope=["actual"]),
        public_working_memory=dict(recent=[dict(target_service_keys=["unaccepted_intent"])]),
    )
    assert api().entry_public_scope(view) == ("current",)
    view["question_state"]["questions"][1]["state"] = "answered"
    assert api().entry_public_scope(view) == ("actual",)
    view["previous_feedback"] = None
    assert api().entry_public_scope(view) == ()


def delivery_index(*, kind="complete", digest="c" * 64, target="candidate", action=1):
    return dict(
        schema_version="nurse.v2.sre-delivered-record-index/1",
        authority="delivery_metadata_not_diagnostic_evidence",
        latest_action_index=action,
        latest_record_refs=[digest],
        records=[
            dict(
                card_id="symptom",
                record_sha256=digest,
                node="symptoms",
                source_role="prepared_summary",
                targets=[target],
                delivery_kind=kind,
                action_indices=[action],
            )
        ],
    )


def compile_delivery(value, book, index=None):
    return api().compile_action_entries(
        value, book=book, request_index=2, case_id=CASE, delivered_index=index
    )


def symptom_entries(table):
    return [e for e in table["entries"] if "record_sha256" in e["source_binding"]]


def test_unread_entry_states_actual_supply_without_claiming_deeper_source_access():
    value, book = issued()
    table = compile_delivery(value, book)
    for entry in symptom_entries(table):
        c = entry["source_context"]
        assert c["delivery"]["state"] == "unread"
        assert c["delivery"]["expected_record_relation"] == "not_proven"
        assert c["delivery"]["prior_action_indices"] == []
        assert "not_deeper_source_read" in c["supply_limits"]
        assert "cannot_fill_unsupplied_coverage_or_windows" in c["supply_limits"]
        assert c["delivery"]["authority"] == "delivery_metadata_not_diagnostic_evidence"
    aggregate = next(e for e in table["entries"] if e["action"]["tool"] == "inspect_sources")
    assert aggregate["source_context"]["supply_contract"] == "complete_selected_cards_aggregation"


def test_complete_same_record_is_bound_to_entry_and_remains_navigation_not_new_evidence():
    value, book = issued()
    before = deepcopy(value)
    table = compile_delivery(value, book, delivery_index())
    for entry in symptom_entries(table):
        assert entry["source_context"]["delivery"] == dict(
            authority="delivery_metadata_not_diagnostic_evidence",
            state="complete_same_record",
            prior_action_indices=[1],
            expected_record_relation="no_new_record",
        )
    assert value == before
    unread = compile_delivery(value, book)
    first = {e["action"]["tool"]: e["entry_id"] for e in symptom_entries(table)}
    second = {e["action"]["tool"]: e["entry_id"] for e in symptom_entries(unread)}
    assert first.keys() == second.keys() and all(first[k] != second[k] for k in first)
    assert table["authority"] == "navigation_not_delivered_evidence"


@pytest.mark.parametrize("change", ["partial", "brief", "source_or_window", "target", "cursor"])
def test_partial_changed_source_window_scope_or_cursor_is_not_marked_complete_reuse(change):
    value, book = issued()
    index = delivery_index()
    if change in {"partial", "brief"}:
        index = delivery_index(kind=change, digest="e" * 64)
    elif change == "source_or_window":
        index = delivery_index(digest="e" * 64)
    elif change == "target":
        index = delivery_index(target="other")
    else:
        row = value["directed_sources"]["entries"][0]
        row["complete_entry"]["parameters"]["cursor"] = "different-issued-cursor"
    table = compile_delivery(value, book, index)
    entries = symptom_entries(table)
    if change == "cursor":
        entries = [e for e in entries if e["action"]["tool"] == "read_node"]
    for entry in entries:
        state = entry["source_context"]["delivery"]
        assert state["state"] != "complete_same_record"
        assert state["expected_record_relation"] == "not_proven"
        assert state["prior_action_indices"] == []


def test_other_side_has_no_borrowed_delivery_and_future_state_is_rejected_transactionally():
    value = metadata()
    row = deepcopy(value["directed_sources"]["entries"][0])
    row.update(card_id="other-card", targets=["other"], record_sha256="e" * 64)
    row["entry"]["parameters"] = dict(card_ids=["other-card"], candidate_service_keys=["other"])
    row["complete_entry"]["parameters"]["card_id"] = "other-card"
    value["directed_sources"]["entries"].append(row)
    value, book = issued(value)
    table = compile_delivery(value, book, delivery_index())
    for entry in symptom_entries(table):
        if entry["source_context"]["targets"] == ["other"]:
            assert entry["source_context"]["delivery"]["state"] == "unread"
    before = book.to_state()
    with pytest.raises(ValueError, match="delivery"):
        compile_delivery(value, book, delivery_index(action=2))
    assert book.to_state() == before


def test_entry_guidance_describes_actual_selected_subject_not_menu_rank():
    from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_SYSTEM, entry_common_guidance

    for text in (ENTRY_SYSTEM, entry_common_guidance()["guidance"]):
        assert "menu order is not formal rank" in text
        assert "actual selected targets" in text
        assert "not a deeper source read" in text
