"""Short references are local identity adapters, never evidence or authority."""

import importlib
import importlib.util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import encode_json

SNAPSHOT, RUN, CASE = "a" * 64, "b" * 64, "opq_" + "c" * 32


def api():
    name = "nurse_rcl.v2.live.ops_sre_references"
    assert importlib.util.find_spec(name), "bound SRE references are not implemented"
    return importlib.import_module(name)


def book():
    return api().SreReferenceBook(snapshot=SNAPSHOT, run_id=RUN)


def reseal(state):
    state["state_sha256"] = sha256_hex(canonical_json_bytes(
        {k: v for k, v in state.items() if k != "state_sha256"}))
    return state


def test_references_are_typed_stable_and_append_only():
    refs = book()
    service = refs.issue("service", "opaque-service", visible_at=2)
    assert service == "S1"
    assert refs.issue("service", "opaque-service", visible_at=3) == service
    assert refs.issue("card", "opaque-service", visible_at=3) == "C1"
    assert refs.issue("service", "another-service", visible_at=3) == "S2"
    assert refs.resolve(service, "service", visible_at=2) == "opaque-service"
    with pytest.raises(ValueError, match="type"):
        refs.resolve(service, "card", visible_at=3)
    with pytest.raises(ValueError, match="visible"):
        refs.resolve(service, "service", visible_at=1)
    with pytest.raises(ValueError, match="visible"):
        refs.issue("service", "opaque-service", visible_at=1)
    with pytest.raises(ValueError, match="order"):
        refs.issue("service", "new-in-the-past", visible_at=1)


def test_reference_state_roundtrip_preserves_every_original():
    refs = book()
    for index, kind in enumerate(("service", "card", "request", "onset",
                                  "evidence", "source", "digest"), 1):
        refs.issue(kind, f"original-{index}", visible_at=index)
    state = refs.to_state()
    restored = api().SreReferenceBook.from_state(
        state, snapshot=SNAPSHOT, run_id=RUN)
    assert canonical_json_bytes(restored.to_state()) == canonical_json_bytes(state)
    assert restored.issue("service", "new-service", visible_at=8) == "S2"
    assert restored.resolve("S1", "service", visible_at=9) == "original-1"


@pytest.mark.parametrize("field", ["snapshot", "run_id"])
def test_reference_state_is_bound_to_case_and_run(field):
    refs = book()
    refs.issue("evidence", "original", visible_at=1)
    expected = dict(snapshot=SNAPSHOT, run_id=RUN)
    expected[field] = "d" * 64
    with pytest.raises(ValueError, match="identity"):
        api().SreReferenceBook.from_state(refs.to_state(), **expected)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_type", "renumber", "bool_index",
                                      "future_order", "bad_digest"])
def test_invalid_reference_tables_are_rejected_even_if_resealed(mutation):
    refs = book()
    refs.issue("service", "one", visible_at=1)
    refs.issue("service", "two", visible_at=2)
    state = refs.to_state()
    rows = state["entries"]
    if mutation == "duplicate":
        rows[1] = list(rows[0])
    elif mutation == "wrong_type":
        rows[0][0] = "card"
    elif mutation == "renumber":
        rows[0][1] = "S8"
    elif mutation == "bool_index":
        rows[0][3] = True
    elif mutation == "future_order":
        rows[0][3] = 3
    else:
        state["state_sha256"] = "0" * 64
    if mutation != "bad_digest":
        reseal(state)
    with pytest.raises(ValueError):
        api().SreReferenceBook.from_state(state, snapshot=SNAPSHOT, run_id=RUN)


def test_registered_fields_roundtrip_without_changing_text_or_scalar_types():
    refs = book()
    original = dict(candidate_service_key="svc-original", card_ids=["card-original"],
        nested=dict(request_key="request-original", source_onset_id="onset-original",
                    evidence_ids=["evidence-original"], source_refs=["source-original"]),
        sections_state={"svc-original": {"availability": "unknown"}},
        values=[dict(name="original_evidence_ids", value='["evidence-original"]')])
    original["exact"] = [False, 0, 0.0, "0", None]
    original["public_summary"] = "svc-original is a hypothesis, not a fact"
    original["unregistered"] = dict(value="card-original")
    original["measurement_sha256"] = "d" * 64
    original["case_snapshot_id"] = SNAPSHOT
    packed = api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert packed["candidate_service_key"] == "S1"
    assert packed["card_ids"] == ["C1"]
    assert packed["values"][0]["value"] == '["E1"]'
    assert list(packed["sections_state"]) == ["S1"]
    assert packed["public_summary"] == original["public_summary"]
    assert packed["unregistered"] == original["unregistered"]
    assert packed["measurement_sha256"] == "H1"
    assert packed["case_snapshot_id"] == SNAPSHOT
    decoded = api().decode_references(packed, book=refs, visible_at=1, case_id=CASE)
    assert canonical_json_bytes(decoded) == canonical_json_bytes(original)
    assert packed["exact"] == original["exact"]


def test_d23_latest_delivered_refs_share_the_index_digest_handles():
    refs = book()
    digest = "d" * 64
    original = dict(
        delivered_record_index=dict(latest_record_refs=[digest]),
        investigation_workbench=dict(
            latest_actual_result=dict(delivered_record_refs=[digest]),
        ),
    )
    packed = api().encode_references(original, book=refs, visible_at=2, case_id=CASE)
    assert packed["delivered_record_index"]["latest_record_refs"] == ["H1"]
    assert packed["investigation_workbench"]["latest_actual_result"]["delivered_record_refs"] == ["H1"]
    assert api().decode_references(packed, book=refs, visible_at=2, case_id=CASE) == original


def test_raw_alias_shaped_id_does_not_collide_with_issued_reference():
    refs = book()
    original = {"candidate_service_keys": ["ordinary", "S1"]}
    packed = api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert packed == {"candidate_service_keys": ["S1", "S2"]}
    assert api().decode_references(packed, book=refs, visible_at=1, case_id=CASE) == original


@pytest.mark.parametrize("bad", ["ts0-mysql-container-kill-9t6n24", "/private/source",
                                 "opq_" + "d" * 32])
def test_private_original_cannot_hide_behind_short_reference(bad):
    refs = book()
    before = refs.to_state()
    with pytest.raises(ValueError, match="private"):
        api().encode_references({"source_refs": [bad]}, book=refs, visible_at=1, case_id=CASE)
    assert refs.to_state() == before


def test_unknown_handle_future_handle_and_wrong_field_type_are_rejected():
    refs = book()
    refs.issue("card", "visible-card", visible_at=1)
    refs.issue("card", "future-card", visible_at=3)
    for bad in ({"card_id": "C999"}, {"card_id": "C2"},
                {"request_key": "C1"}, {"card_id": 1}, {"card_ids": [False]}):
        with pytest.raises(ValueError):
            api().decode_references(bad, book=refs, visible_at=1, case_id=CASE)


def test_json_string_values_require_canonical_registered_array_not_arbitrary_json():
    refs = book()
    original = dict(values=[dict(name="explanation", value='{"card_id":"raw"}')])
    assert api().encode_references(original, book=refs, visible_at=1, case_id=CASE) == original
    for value in ('[ "one" ]', '{"card_id":"one"}', '[1]'):
        with pytest.raises(ValueError):
            api().encode_references(dict(values=[dict(name="evidence_ids", value=value)]),
                                    book=refs, visible_at=1, case_id=CASE)


def test_reference_resource_boundary_does_not_mutate_the_book():
    refs = book()
    before = refs.to_state()
    with pytest.raises(ValueError, match="json_limit"):
        api().encode_references(dict(card_id="new", text="x" * 4_194_305),
                                book=refs, visible_at=1, case_id=CASE)
    assert refs.to_state() == before


def test_saved_private_map_is_scanned_again_when_restoring_visible_content():
    refs = book()
    refs.issue("source", "/private/source", visible_at=1)
    with pytest.raises(ValueError, match="private"):
        api().decode_references({"source_refs": ["P1"]},
                                book=refs, visible_at=1, case_id=CASE)


def test_tool_documentation_and_skill_placeholders_are_not_identity_positions():
    refs = book()
    original = dict(capabilities=dict(parameter_contracts=dict(inspect_candidate=dict(
        candidate_service_key="copy a registered candidate_service_key"))),
        skills=[dict(examples=[dict(parameters=dict(candidate_service_key="<candidate>"))])],
        previous_feedback=dict(parameters=dict(candidate_service_key="real-candidate")))
    packed = api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert packed["capabilities"] == original["capabilities"]
    assert packed["skills"] == original["skills"]
    assert packed["previous_feedback"]["parameters"]["candidate_service_key"] == "S1"
    assert len(refs.to_state()["entries"]) == 1


def test_omitted_reference_headers_and_literal_failure_scope_remain_literal():
    refs = book()
    header = dict(section="evidence_ids", count=10, sha256="d" * 64,
                  omitted_detail="exact references available via read_node section")
    original = dict(references=dict(evidence_ids=header), scope="tool_view")
    packed = api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert packed == original


def test_metadata_reference_headers_only_convert_their_visible_values():
    refs = book()
    original = dict(references=dict(evidence_ids=dict(section="evidence_ids", count=2,
        sha256="d" * 64, values=["first", "second"])))
    packed = api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert packed["references"]["evidence_ids"]["values"] == ["E1", "E2"]
    assert api().decode_references(packed, book=refs, visible_at=1, case_id=CASE) == original


@pytest.mark.parametrize("restore", [False, True])
def test_registered_scalar_list_cannot_hide_private_original(restore):
    refs = book()
    original = "/private/source"
    value = refs.issue("source", original, visible_at=1) if restore else original
    before = refs.to_state()
    payload = dict(values=[dict(name="source_refs", value=canonical_json_bytes([value]).decode())])
    translate = api().decode_references if restore else api().encode_references
    with pytest.raises(ValueError, match="private"):
        translate(payload, book=refs, visible_at=1, case_id=CASE)
    assert refs.to_state() == before


def test_encoding_cannot_commit_a_reference_map_that_exceeds_persistence_budget():
    refs = book()
    before = refs.to_state()
    original = dict(candidate_service_keys=[f"service{i:04d}" + "x" * 2037 for i in range(2035)])
    encode_json(original, api().VIEW_LIMITS)
    with pytest.raises(ValueError, match="json_limit"):
        api().encode_references(original, book=refs, visible_at=1, case_id=CASE)
    assert refs.to_state() == before


def test_report_assertion_values_use_the_registered_field_not_free_text():
    refs = book()
    original = dict(assertions=[dict(fact_ref="f1", field="candidate_service_key",
                                     value="service")])
    packed = api().encode_references(original, book=refs, visible_at=9, case_id=CASE)
    assert packed["assertions"][0]["value"] == "S1"
    assert api().decode_references(packed, book=refs, visible_at=9, case_id=CASE) == original


def test_json_tuple_containers_are_translated_before_wire_serialization():
    refs = book()
    original = dict(steps=(dict(card_id="original-card", targets=("original-service",)),))
    packed = api().encode_references(original, book=refs, visible_at=9, case_id=CASE)
    assert packed == dict(steps=[dict(card_id="C1", targets=["S1"])])
    restored = api().decode_references(packed, book=refs, visible_at=9, case_id=CASE)
    assert canonical_json_bytes(restored) == canonical_json_bytes(original)


def test_private_original_in_tuple_is_still_rejected_transactionally():
    refs = book()
    before = refs.to_state()
    with pytest.raises(ValueError, match="private"):
        api().encode_references(dict(steps=(dict(source_refs=("/private/source",)),)),
                                book=refs, visible_at=9, case_id=CASE)
    assert refs.to_state() == before
