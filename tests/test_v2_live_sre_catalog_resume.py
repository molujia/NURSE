"""Targeted coverage only: synthetic receipts do not certify model semantics."""

# ruff: noqa: F811 -- imported pytest fixtures intentionally share argument names.

import json
from copy import deepcopy

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_agent import run_ops_agent
from nurse_rcl.v2.live.ops_node_common import node_card
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from nurse_rcl.v2.live.ops_sre_brief_tools import candidate_situation
from nurse_rcl.v2.live.ops_sre_context import _delivered_records
from nurse_rcl.v2.live.ops_tools import card_id
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_action_decision import decision, note
from tests.test_v2_live_sre_facet_reading import context
from tests.test_v2_live_sre_factual_time import api
from tests.test_v2_live_sre_observable_owner import saved
from tests.test_v2_live_sre_observable_purpose import declaration
from tests.test_v2_live_sre_observable_tools import tools


@pytest.fixture(scope="module")
def qualified_complete(tmp_path_factory):
    """Two complete synthetic services keep both qualified reads in the bounded menu."""
    from unittest.mock import patch

    import pyarrow.parquet as pq

    from nurse_rcl.v2.live.normal_queries import run_normal_queries
    from nurse_rcl.v2.live.normal_scoring import score_normal_case
    from nurse_rcl.v2.live.repair_raw_audit import _fixed_repair_scope
    from tests import test_v2_live_repair_stage as stage_fixture
    from tests.live_repair_case_fixtures import make_case_pair
    from tests.test_v2_live_sources import T0, write

    original_setup = stage_fixture.setup_case

    def setup_two_services(root):
        arguments, opaque = original_setup(root)
        raw = arguments["source_root"]
        metadata = pq.read_table(raw / "normal_metrics.parquet").schema.metadata
        for window, start in (("normal", T0 - 240), ("abnormal", T0)):
            write(
                raw,
                f"{window}_metrics.parquet",
                [start + n * 30 for _ in range(2) for n in range(8)],
                metric=["container.cpu.usage"] * 16,
                value=(
                    [10.0] * 8
                    if window == "normal"
                    else [10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
                )
                * 2,
                service_name=[f"qualified-service-{i}" for i in range(2) for _ in range(8)],
                answer=["SECRET_LABEL"] * 16,
            )
        path = raw / "normal_metrics.parquet"
        pq.write_table(pq.read_table(path).replace_schema_metadata(metadata), path)
        return arguments, opaque

    root = tmp_path_factory.mktemp("catalog-qualified-sources")
    # Only fixture provisioning is changed, before staging and source admission.
    # Runtime catalogs, menu compilation, guards and owner execution remain real.
    with patch.object(stage_fixture, "setup_case", setup_two_services):
        pair = make_case_pair(root, "rcabench")
    with _fixed_repair_scope(**pair["raw_arguments"]) as (data, _):
        pass
    assert len(data.catalog.candidate_service_keys) == len(data.sources) == 2
    blocks = pair["raw_arguments"]["store_root"]
    session = run_normal_queries(data=data, ledger_root=root / "queries", block_root=blocks)
    return data, blocks, score_normal_case(data=data, session=session), None


def test_eleven_published_dependency_cards_are_not_thirty_three_edges_or_three_neighbors():
    rows = [dict(node="dependency", payload=dict(edges=[1, 2, 3])) for _ in range(11)]
    before = deepcopy(rows)
    caps = dict(investigation_situation=dict(candidates=[candidate_situation("svc-a", rows)]))
    aid = api().directory_reading_aid(caps)
    assert aid["records"][0]["category_card_counts"] == dict(dependency=11)
    assert aid["category_unit"] == "published_node_cards_not_measured_edges_or_votes"
    assert rows == before
    assert "selected_candidate" not in aid


def error_pair(complete):
    """Real node/ledger admission with explicitly synthetic, differently qualified records."""
    data = complete[0]
    tool = tools().OpsSreObservableToolbox(data=data, block_root=complete[1])
    sources = []
    for source in data.sources:
        if source.candidate_service_key not in [s.candidate_service_key for s in sources]:
            sources.append(source)
    assert len(sources) >= 2
    cards = []
    for source, count, denominator, fraction, modality, role, statistic in zip(
        sources,
        [48, 0],
        [192, 0],
        [0.25, None],
        ["log", "trace"],
        ["log", "client"],
        ["observed_log_severity_fraction", "observed_span_status_fraction"],
        strict=False,
    ):
        facts = deepcopy(context()["previous_feedback"]["semantic_record"]["payload"]["facts"])
        facts.update(
            candidate_service_key=source.candidate_service_key,
            observed_error_count=count,
            baseline_known_denominator=denominator,
            fault_known_denominator=denominator,
            baseline_fraction=fraction,
            fault_fraction=fraction,
            sampling_coverage="unknown",
            modality=modality,
            span_kind=role,
            statistic=statistic,
        )
        cards.append(
            node_card(
                snapshot=tool.case_snapshot_id,
                node="symptoms",
                parameters=dict(candidate_service_key=source.candidate_service_key),
                payload=dict(kind="error_summary", facts=facts),
                evidence_ids=(source.source_sha256,),
                source_refs=(source.source_sha256,),
                interval=data.diagnostics.legacy.sources.adapter.time_contract.fault_window,
                unit="observed_record_fraction",
                coverage=None,
                truth="unknown",
                limitations=("synthetic_fixture_not_real_case_evidence",),
            )
        )
    tool.admit_core(nodes=tuple(cards), steps=())
    return tool, cards


def test_two_to_three_fresh_owner_uses_second_qualified_receipt_and_preserves_question_prefix(
    qualified_complete, tmp_path
):
    complete = qualified_complete
    seen, native = [], []
    prototype, cards = error_pair(complete)
    ids = [card_id(c) for c in cards]
    selected = []
    core_before = canonical_json_bytes(complete[2].model_dump(mode="json"))

    def send(index, wire):
        ctx = brief_context_from_wire(
            wire,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id=prototype.case_snapshot_id,
            decision_index=index,
        )
        if index > 1:
            latest = ctx["investigation_progress"]["latest_actual_result"]
            fields = ctx["factual_reading_aid"]["latest_question_fields"]
            assert latest["action_index"] == index - 1
            assert fields[0]["question_id"] == f"iq_d{index - 1}"
            assert ctx["question_state"]["latest_action_question_id"] == f"iq_d{index - 1}"
            record = next(
                r
                for r in _delivered_records(ctx["previous_feedback"])
                if r["card_id"] == selected[index - 2]
            )
            facts = record["payload"]["facts"]
            assert facts["sampling_coverage"] == "unknown" and record["coverage"] is None
            assert record["unit"] == "observed_record_fraction"
            if index == 3:
                assert facts["observed_error_count"] == 0
                assert facts["baseline_known_denominator"] == facts["fault_known_denominator"] == 0
                assert facts["baseline_fraction"] is facts["fault_fraction"] is None
                assert (facts["modality"], facts["span_kind"], facts["statistic"]) == (
                    "trace",
                    "client",
                    "observed_span_status_fraction",
                )
                questions = ctx["question_state"]["questions"]
                assert questions[0]["question_id"] == "iq_d1"
                assert (
                    questions[0]["state"] == "answered" and questions[0]["remaining_facets"] == []
                )
                assert questions[1]["question_id"] == "iq_d2" and questions[1]["state"] == "open"
                envelope = json.loads(wire)
                body = json.loads(envelope["messages"][-1]["content"])
                coverage = next(
                    f
                    for f in body["context"]["workbench"]["factual_reading_aid"][
                        "latest_question_fields"
                    ][0]["fields"]
                    if f["facet"] == "coverage"
                )
                coverage["value"] = "known"
                envelope["messages"][-1]["content"] = canonical_json_bytes(body).decode()
                with pytest.raises(ValueError, match="factual reading aid"):
                    brief_context_from_wire(
                        canonical_json_bytes(envelope),
                        opaque_case_id=complete[0].catalog.opaque_case_id,
                        case_snapshot_id=prototype.case_snapshot_id,
                        decision_index=index,
                    )
        entries = ctx["action_entries"]["entries"]
        if index <= 2:
            # Current owner context has full source identities; model-facing
            # bindings use short IDs and separately rebuilt readable digests.
            # Select the same issued A entry in both already-frozen views.
            original = json.loads(saved(complete, tmp_path).decisions[index - 1].context_json)
            entry_ref = next(
                e["entry_ref"]
                for e in original["action_entries"]["entries"]
                if e["source_binding"].get("card_id") == ids[index - 1]
            )
            entry = next(e for e in entries if e["entry_ref"] == entry_ref)
            selected.append(entry["source_binding"]["card_id"])
        else:
            entry = next(
                e
                for e in entries
                if e["action"]["tool"] == "inspect_sources"
                and e["source_binding"]["card_id"] not in selected
            )
        value = decision(first=index == 1)
        value["actions"] = [
            dict(entry_ref=entry["entry_ref"], question="Read registered record fields.")
        ]
        value["investigation_note"] = dict(
            note(),
            observable_test=declaration(entry),
            basis_refs=[],
            target_service_keys=entry["observable_contract"]["binding"]["targets"],
        )
        if index > 1:
            value["responds_to"] = [dict(action_index=index - 1, card_ids=[])]
            value["latest_result_update"].update(
                action_index=index - 1,
                answer_status="answered",
                remaining_question=None,
                summary=(
                    "The record fields are delivered; diagnostic comparability remains unresolved."
                ),
                content_relation=ctx["investigation_progress"]["latest_actual_result"][
                    "content_relation"
                ],
            )
            value["investigation_note"]["question_updates"] = [
                dict(
                    question_id=f"iq_d{index - 1}",
                    state="answered",
                    remaining_gap="",
                    remaining_facets=[],
                    basis_refs=[],
                )
            ]
            value["unresolved"] = [
                "Different log-severity and client-span-status records do not establish "
                "matched absence."
            ]
        seen.append(index)
        native.append(canonical_json_bytes(value))
        return native[-1]

    def run(ceiling, complete_call):
        return run_ops_agent(
            root=tmp_path,
            toolbox=error_pair(complete)[0],
            code_sha256="a" * 64,
            complete=complete_call,
            empirical=True,
            sre=True,
            sre_profile="hermes-sre-tools/20",
            opaque_case_id=complete[0].catalog.opaque_case_id,
            stage_decision_limit=ceiling,
        )

    first = run(2, send)
    assert first.termination == "stage_pause" and first.action_count == 2, first.primary_failure
    prefix = [r.model_dump(mode="json") for r in saved(complete, tmp_path).decisions]
    second = run(3, send)
    assert second.termination == "stage_pause" and second.action_count == 3, second.primary_failure
    restored = saved(complete, tmp_path)
    assert [r.model_dump(mode="json") for r in restored.decisions[:2]] == prefix
    assert [r.native_response_json.encode() for r in restored.decisions] == native
    assert all(r.correction_wire_json is None for r in restored.decisions)
    assert restored.result is None and seen == [1, 2, 3]
    assert run(3, lambda *_: pytest.fail("same-ceiling owner must not resend")) == second
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == core_before
