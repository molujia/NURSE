"""Only registered synthetic receipt bytes; no preparation/core/tool execution."""

import json
from importlib import import_module

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.sre_m3_entry_fixtures import normal_context, receipt_context


def digest(value):
    return sha256_hex(canonical_json_bytes(value))


def material(*, note=None, blocked=False):
    from nurse_rcl.v2.live.ops_agent_state import OpsReplanningDecisionRecord
    from nurse_rcl.v2.live.ops_report_context import OpsRankRow, OpsReportContext
    from nurse_rcl.v2.live.ops_sre_feedback_fact_entry import prepare_entry_decision
    from tests.sre_feedback_fact_fixtures import CASE_ID, RUN_ID, SNAPSHOT

    second, _, _, row, record = receipt_context()
    first, refs, _ = normal_context(
        extra=dict(
            directed_sources=second["capabilities"]["directed_sources"],
            investigation_situation=dict(
                candidates=[dict(candidate_service_key="synthetic-service")]
            ),
        )
    )
    entry = next(
        e for e in first["action_entries"]["entries"] if e["action"]["tool"] == "read_node"
    )
    response = dict(
        actions=[entry["entry_ref"]],
        stop_reason=None,
        stop_basis=None,
        public_summary="Read the synthetic registered record",
        unresolved=["Coverage unresolved"],
        responds_to=[],
        latest_result_update=None,
        investigation_note=None,
    )
    if note is not None:
        contract = entry["observable_contract"]
        response["investigation_note"] = dict(
            role="symptoms",
            target_service_keys=contract["binding"]["targets"],
            why_now="Inspect the registered observed-record scope, not service health.",
            basis_refs=[],
            question_updates=[],
            stop_assessment=None,
            observable_test=dict(
                entry_ref=entry["entry_ref"],
                objective_kind=contract["objective_kind"],
                metric_name=contract["metric_name"],
                requested_facets=contract["allowed_facets"],
            ),
        )
        if note == "rejected":
            response["investigation_note"]["discriminating_value"] = "Not an atomic note field"
    wire = prepare_entry_decision(
        first, references=refs, case_id=CASE_ID, snapshot=SNAPSHOT, index=1
    )
    decision = OpsReplanningDecisionRecord(
        index=1,
        context_json=canonical_json_bytes(first).decode(),
        response_json=canonical_json_bytes(response).decode(),
        action_index=1,
        wire_json=wire.decode(),
        wire_sha256=sha256_hex(wire),
        target_exceeded=len(wire) > 16384,
        expansion_reasons=("required_semantics_or_complete_envelope",) if len(wire) > 16384 else (),
        native_response_json=canonical_json_bytes(response).decode(),
        call_started=True,
    )
    formal = OpsReportContext(
        opaque_case_id=CASE_ID,
        case_snapshot_id=SNAPSHOT,
        core_result_digest="e" * 64,
        ranking=(
            OpsRankRow(
                rank=1,
                candidate_service_key="synthetic-service",
                quantized_total_score="0.75",
                local_score_component=0.75,
                propagation_score_component=0.0,
                local_evidence_status="partial",
            ),
        ),
        formal_top_k=("synthetic-service",),
        mandatory_fact_refs=(),
        formal_detail_omissions=(),
        investigation=(),
        formal_projection_available=True,
        investigation_status="partial",
        investigation_failure_type=None,
        model_trace_unverified=(),
        facts=(),
    )
    source = dict(
        source_plan_sha256="1" * 64,
        source_code_sha256="2" * 64,
        stage_grant_sha256="3" * 64,
        stage_round_sha256="4" * 64,
        worker_invocation_id="5" * 32,
        opaque_case_id=CASE_ID,
        case_snapshot_id=SNAPSHOT,
        source_run_id=RUN_ID,
        core_result_digest=formal.core_result_digest,
        formal_context_sha256=formal.context_sha256,
        reference_state_sha256=digest(refs),
        decision_pins=[digest(decision.model_dump(mode="json"))],
        action_pins=[digest(record.model_dump(mode="json"))],
        retained_sources_sha256=digest([]),
        termination="stage_pause",
        investigation_status="partial",
    )
    decisions = (decision,)
    if blocked:
        ctx, refs, _ = normal_context(
            extra=dict(directed_sources=second["capabilities"]["directed_sources"]),
            references=refs,
            records=(record,),
            decisions=decisions,
            index=2,
        )
        entry = next(
            e for e in ctx["action_entries"]["entries"] if e["action"]["tool"] == "read_node"
        )
        value = dict(
            response,
            actions=[entry["entry_ref"]],
            responds_to=[dict(action_index=1, card_ids=[])],
            latest_result_update=dict(
                action_index=1,
                summary="The complete receipt is unchanged.",
                content_relation=ctx["investigation_progress"]["latest_actual_result"][
                    "content_relation"
                ],
            ),
        )
        wire = prepare_entry_decision(
            ctx, references=refs, case_id=CASE_ID, snapshot=SNAPSHOT, index=2
        )
        proof = dict(
            reason="already_delivered_no_new_content",
            prior_action_index=1,
            card_id=row["card_id"],
            parameters_sha256=digest(json.loads(record.action.parameters_json)),
            record_sha256=digest(row),
            source_refs=row["source_refs"],
            representation="complete_semantic_record",
        )
        decisions += (
            OpsReplanningDecisionRecord(
                index=2,
                context_json=canonical_json_bytes(ctx).decode(),
                response_json=canonical_json_bytes(value).decode(),
                action_index=None,
                wire_json=wire.decode(),
                wire_sha256=sha256_hex(wire),
                target_exceeded=len(wire) > 16384,
                expansion_reasons=("required_semantics",) if len(wire) > 16384 else (),
                native_response_json=canonical_json_bytes(value).decode(),
                call_started=True,
                blocked_read_proof=proof,
            ),
        )
        source.update(
            termination="no_progress_after_correction",
            reference_state_sha256=digest(refs),
            decision_pins=[digest(d.model_dump(mode="json")) for d in decisions],
        )
    return dict(
        formal=formal,
        decisions=decisions,
        records=(record,),
        references=refs,
        retained_sources=(),
        source=source,
        record=row,
    )


def assemble(module, args):
    source = module.AtomicReportSourceIdentity.model_validate_json(
        canonical_json_bytes(args["source"])
    )
    return module.assemble_atomic_report_context(
        **{
            k: args[k] for k in ("formal", "decisions", "records", "references", "retained_sources")
        },
        source_identity=source,
        expected_source_sha256=digest(source.model_dump(mode="json")),
    )


def response_from_wire(wire, *, blocked=False):
    from nurse_rcl.v2.live.ops_sre_atomic_report_expressions import expression_catalog
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_report_body

    body, store = brief_report_body(wire)
    choice = next(r for r in expression_catalog(body["context"]["supply"], store)["expressions"]
                  if r["claim"]["operation"] == "explain_uncertainty")
    claim = dict(
        claim_id="c1",
        expression_ref=choice["expression_ref"],
        outcomes=None,
        surface_text="Coverage remains unresolved."
        if not blocked
        else "Unsupported value 0.12345.",
    )
    return canonical_json_bytes(
        dict(
            schema_version="nurse.v2.atomic-report-expression-response/3",
            core_result_digest=body["context"]["evidence"]["core_result_digest"],
            executive_summary=[claim],
            formal_interpretations=[],
            step_findings={},
            final_synthesis=[],
            limitations=[],
            next_checks=[],
        )
    )


def file_pins(root):
    return {
        str(p.relative_to(root)): (p.stat().st_size, sha256_hex(p.read_bytes()))
        for p in root.rglob("*")
        if p.is_file() and p.name != "lock"
    }


def forbid_dispatch(monkeypatch, counters, *, read_only=False):
    def denied(kind):
        def fail(*args, **kwargs):
            counters[kind] += 1
            raise AssertionError("zero-ARK Report handoff forbidden: " + kind)

        return fail

    counters["dispatch"] = 0
    for name, owner, attr in (
        ("ops_investigation_ledger", "OpsInvestigationLedger", "dispatch"),
        ("ops_sre_atomic_selector_tools", "OpsSreAtomicSelectorInvestigationLedger", "dispatch"),
        ("ops_sre_tools", "OpsSreToolbox", "execute"),
    ):
        monkeypatch.setattr(
            getattr(import_module("nurse_rcl.v2.live." + name), owner), attr, denied("dispatch")
        )
    monkeypatch.setattr(import_module("nurse_rcl.v2.live.ops_case"), "_preparation", denied("load"))
    if read_only:
        counters["mutation"] = 0
        native = import_module("nurse_rcl.v2.live.ops_sre_native_run")
        for attr in ("_persist", "_write_bound", "_publish_saved_report"):
            monkeypatch.setattr(native, attr, denied("mutation"))
