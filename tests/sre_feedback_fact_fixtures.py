"""Synthetic JSON only. No real case, telemetry, provider or owner fixture."""

import socket
from copy import deepcopy
from importlib import import_module, util

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex

SNAPSHOT = "a" * 64
RUN_ID = "b" * 64
CASE_ID = "opq_" + "c" * 32


def api(name):
    full = "nurse_rcl.v2.live." + name
    assert util.find_spec(full), f"missing planned capability: {name}"
    return import_module(full)


@pytest.fixture(autouse=True)
def deny_effects(monkeypatch):
    counters = dict.fromkeys(("send", "settings", "execute", "stage", "load", "import_core"), 0)
    targets = (
        ("nurse_rcl.v2.hermes_exchange", "execute_once", "execute"),
        ("nurse_rcl.agents.ark", "load_ark_settings_from_key_file", "settings"),
        ("nurse_rcl.v2.live.ops_agent", "run_ops_agent", "execute"),
        ("nurse_rcl.v2.live.ops_sre_directed_stage", "run_ops_agent", "execute"),
        ("nurse_rcl.v2.live.ops_sre_directed_stage", "run_directed_stage", "stage"),
        ("nurse_rcl.v2.live.ops_sre_directed_stage", "run_stage_round", "stage"),
        ("nurse_rcl.v2.live.ops_sre_directed_stage", "_preparation", "load"),
        ("nurse_rcl.v2.live.ops_sre_core_import", "import_source_core", "import_core"),
        ("nurse_rcl.v2.live.ops_sre_core_import", "source_core_bytes", "import_core"),
        ("nurse_rcl.v2.live.ops_sre_core_import", "_source_metadata", "import_core"),
    )
    # Import before patching: otherwise a first import can retain a guarded
    # from-import alias after its source module has been restored by teardown.
    modules = {name: import_module(name) for name, _, _ in targets}
    directed = import_module("nurse_rcl.v2.live.ops_sre_directed_provider")

    def deny(kind):
        def denied(*args, **kwargs):
            counters[kind] += 1
            pytest.fail(f"source-faithful zero-ARK forbidden: {kind}")

        return denied

    monkeypatch.setattr(socket.socket, "connect", deny("send"))
    monkeypatch.setattr(socket.socket, "connect_ex", deny("send"))
    monkeypatch.setattr(socket, "create_connection", deny("send"))
    monkeypatch.setattr(httpx.Client, "send", deny("send"))
    monkeypatch.setattr(httpx.AsyncClient, "send", deny("send"))
    for module, attr, kind in targets:
        monkeypatch.setattr(modules[module], attr, deny(kind))
    monkeypatch.setattr(directed.OpsSreDirectedDecisionProvider, "decision", deny("send"))
    yield counters
    assert all(value == 0 for value in counters.values()), counters
    print("zeroark_forbidden_calls=" + canonical_json_bytes(counters).decode())


def symptom_record():
    return dict(
        card_id="synthetic-card-log",
        node="symptoms",
        tool_version="ops-symptoms/1",
        targets=["synthetic-service"],
        source_role="investigation_observation",
        source_refs=["d" * 64],
        window_ns=[1000000001, 2000000002],
        unit="observed_record_fraction",
        coverage=None,
        availability="partial",
        truth_value="true",
        limitations=["record_fraction_not_request_population"],
        payload=dict(
            kind="error_summary",
            facts=dict(
                baseline_fraction=0.0,
                fault_fraction=None,
                baseline_known_denominator=8,
                fault_known_denominator=0,
                observed_error_count=0,
                sampling_coverage="unknown",
                modality="log",
                span_kind="unknown",
                statistic="observed_log_severity_fraction",
                facts=[
                    dict(predicate_kind="local_anomaly_observed", truth_value="true"),
                    dict(predicate_kind="local_error_increase", truth_value="false"),
                    dict(predicate_kind="request_failure_increase", truth_value="unknown"),
                ],
            ),
        ),
    )


def mechanism_record():
    row = symptom_record()
    row.update(
        card_id="synthetic-card-mechanism",
        node="mechanisms",
        tool_version="ops-mechanisms/1",
        truth_value="unknown",
    )
    row["payload"] = dict(
        decision=dict(
            candidate_service_key="synthetic-service",
            mechanism_key="synthetic-mechanism",
            disposition="unknown",
            necessary_support="unknown",
        ),
        conditions=[
            dict(
                condition_id="synthetic-required",
                requirement="necessary",
                operator="and",
                truth_value="unknown",
                predicate_kind="workload_unavailability_observed",
                leaves=[],
            ),
            dict(
                condition_id="synthetic-optional",
                requirement="optional",
                operator="or",
                truth_value="true",
                predicate_kind="request_failure_increase",
                leaves=[],
            ),
        ],
        distinguishing_checks=[],
        score_authority=False,
    )
    return row


def metadata(record, indices=(1,)):
    return dict(
        card_id=record["card_id"],
        node=record["node"],
        targets=deepcopy(record["targets"]),
        source_role=record["source_role"],
        record_sha256=sha256_hex(canonical_json_bytes(record)),
        action_indices=list(indices),
        delivery_kind="brief"
        if record.get("representation") == "condition_evidence_brief"
        else "complete",
    )


def make_context(record=None):
    row = deepcopy(record if record is not None else symptom_record())
    item = metadata(row)
    return dict(
        schema_version="nurse.v2.sre-investigation-input/24",
        decision_index=2,
        case_snapshot_id=SNAPSHOT,
        actions=[dict(action_index=1, status="completed", visible_card_ids=[row["card_id"]])],
        previous_feedback=dict(
            action_index=1,
            status="completed",
            origin="read_card",
            semantic_record=row,
            critical_record_ids=[],
        ),
        critical_evidence=[
            dict(
                card_id=row["card_id"],
                record_sha256=item["record_sha256"],
                action_indices=[1],
                record_action_index=1,
                record_location="previous_feedback record with this exact digest",
            )
        ],
        delivered_record_index=dict(
            latest_action_index=1, latest_record_refs=[item["record_sha256"]], records=[item]
        ),
        public_working_memory=dict(
            recent=[
                dict(role="agent_statement", public_summary="synthetic unsupported increase claim")
            ],
            question_history=[],
        ),
    )


def make_model_context():
    from nurse_rcl.v2.live.ops_agent_decision import CONTEXT_LIMITS
    from nurse_rcl.v2.live.ops_sre_action_entries import NAVIGATION_KINDS
    from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
    from nurse_rcl.v2.live.ops_sre_references import encode_references
    from nurse_rcl.v2.live.ops_sre_workbench_model import decision_fields

    ctx = make_context()
    for field in decision_fields(ctx["schema_version"]):
        ctx.setdefault(field, {})
    ctx.update(skills=[], capabilities=dict(parameter_contracts={}))
    book = SreActionReferenceBook(snapshot=SNAPSHOT, run_id=RUN_ID)
    ctx["action_entries"] = dict(
        schema_version="nurse.v2.sre-action-entries/5",
        source_prefix=book.source_prefix(visible_at=2),
        authority="navigation_not_delivered_evidence",
        shown=0,
        total=0,
        omitted=0,
        navigation_kinds=list(NAVIGATION_KINDS),
        entries=[],
    )
    encode_references(ctx, book=book, visible_at=2, case_id=CASE_ID, limits=CONTEXT_LIMITS)
    ctx["action_entries"]["source_prefix"] = book.source_prefix(visible_at=2)
    return ctx, book.to_state(), dict(case_id=CASE_ID, snapshot=SNAPSHOT, index=2, report=False)
