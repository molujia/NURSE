"""Synthetic normal-entry fixtures; no telemetry, paid artifact or owner loading."""

from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
from nurse_rcl.v2.live.ops_sre_brief_exchange import prepare_brief_decision
from nurse_rcl.v2.live.ops_sre_context import sre_investigation_input
from nurse_rcl.v2.live.ops_sre_encoding import SEMANTIC_ENCODING, expand
from tests.sre_feedback_fact_fixtures import (  # noqa: F401
    CASE_ID,
    RUN_ID,
    SNAPSHOT,
    api,
    deny_effects,
)


def normal_context(*, extra=None, references=None, records=(), decisions=(), index=1):
    if references is None:
        references = SreActionReferenceBook(snapshot=SNAPSHOT, run_id=RUN_ID).to_state()
    capabilities = dict(
        investigation_profile_id="hermes-sre-tools/23",
        case_snapshot_id=SNAPSHOT,
        semantic_encoding=SEMANTIC_ENCODING,
        investigation_routes=dict(
            schema_version="nurse.v2.sre-investigation-routes/3", entries=[], recovery_feedback=None
        ),
        parameter_contracts={},
    )
    capabilities.update(extra or {})

    def make_wire(view):
        import json

        return json.loads(
            prepare_brief_decision(
                context=view,
                references=references,
                opaque_case_id=CASE_ID,
                case_snapshot_id=SNAPSHOT,
                decision_index=index,
            )
        )

    projection = sre_investigation_input(
        capabilities=capabilities,
        records=records,
        decisions=decisions,
        decision_index=index,
        make_wire=make_wire,
        entry_references=references,
        entry_case_id=CASE_ID,
    )
    assert projection.status == "ready", projection
    import json

    context = expand(json.loads(projection.view_json))
    return context, references, dict(case_id=CASE_ID, snapshot=SNAPSHOT, index=index)


def receipt_context():
    import json

    from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
    from nurse_rcl.v2.live.ops_agent_state import OpsDecisionRecord
    from nurse_rcl.v2.live.ops_contracts import OpsInvestigationEvidence, OpsNodeEvidence
    from nurse_rcl.v2.live.ops_sre_atomic_selector_tools import (
        OpsAtomicSelectorAction,
        OpsAtomicSelectorActionRecord,
    )
    from nurse_rcl.v2.live.ops_tools import card_id
    from tests.sre_feedback_fact_fixtures import symptom_record

    row = symptom_record()
    node = OpsNodeEvidence(
        case_snapshot_id=SNAPSHOT,
        node="symptoms",
        tool_version="ops-symptoms/1",
        parameters_json=canonical_json_bytes(
            dict(candidate_service_key="synthetic-service")
        ).decode(),
        evidence_ids=(),
        source_refs=tuple(row["source_refs"]),
        unit=row["unit"],
        window_ns=tuple(row["window_ns"]),
        coverage=None,
        availability="partial",
        truth_value="true",
        limitations=tuple(row["limitations"]),
        payload_json=canonical_json_bytes(row["payload"]).decode(),
    )
    row["card_id"] = card_id(node)
    params = dict(card_id=row["card_id"], section="payload", cursor=None)
    source = dict(
        row,
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        record_kind="error_summary",
        record_version="ops-symptoms/1",
        entry=dict(
            tool="inspect_sources",
            parameters=dict(card_ids=[row["card_id"]], candidate_service_keys=row["targets"]),
        ),
        complete_entry=dict(tool="read_node", parameters=params),
    )
    extra = dict(directed_sources=dict(entries=[source]))
    first, refs, _ = normal_context(extra=extra)
    entry = next(
        e for e in first["action_entries"]["entries"] if e["action"]["tool"] == "read_node"
    )
    response = dict(
        actions=[entry["entry_ref"]],
        stop_reason=None,
        stop_basis=None,
        public_summary="Read the synthetic registered record",
        unresolved=[],
        responds_to=[],
        latest_result_update=None,
        investigation_note=None,
    )
    record = OpsAtomicSelectorActionRecord(
        index=1,
        input_sha256=sha256_hex(canonical_json_bytes(response["actions"])),
        action=OpsAtomicSelectorAction(
            action_index=1,
            skill_id=entry["action"]["skill_id"],
            tool="read_node",
            parameters_json=canonical_json_bytes(params).decode(),
            question=None,
        ),
        status="completed",
        result=OpsInvestigationEvidence(
            action_index=1, evidence=node, origin="read_card", reused_core_evidence_ids=()
        ),
        model_view_json=canonical_json_bytes(
            dict(
                action_index=1,
                origin="read_card",
                semantic_record=row,
                section="payload",
                next_cursor=None,
                semantic_encoding=SEMANTIC_ENCODING,
                representation="complete_semantic_record",
            )
        ).decode(),
    )
    decision = OpsDecisionRecord(
        index=1,
        context_json=canonical_json_bytes(first).decode(),
        response_json=canonical_json_bytes(response).decode(),
        action_index=1,
    )
    context, refs, args = normal_context(
        extra=extra, references=refs, records=(record,), decisions=(decision,), index=2
    )
    assert json.loads(record.model_view_json)["semantic_record"] == row
    return context, refs, args, row, record
