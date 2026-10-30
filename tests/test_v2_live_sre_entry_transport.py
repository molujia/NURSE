"""New frozen entry wire and bounded owner, with explicit synthetic sends only."""

# ruff: noqa: F811 -- real small fixtures.

import json
from copy import deepcopy
from typing import get_args

import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_sre_context import sre_investigation_input
from nurse_rcl.v2.live.ops_sre_entry_tools import OpsSreEntryInvestigationLedger
from nurse_rcl.v2.live.ops_sre_skills import sre_skill_index
from tests.test_v2_live_ops_tools import complete, prepared_cases  # noqa: F401
from tests.test_v2_live_sre_entry_admission import opened


def project(ledger, book, case):
    result = sre_investigation_input(
        capabilities=ledger.capabilities(),
        records=ledger.records,
        decisions=(),
        decision_index=1,
        entry_references=book.to_state(),
        entry_case_id=case,
        make_wire=lambda v: dict(context=v),
    )
    assert result.status == "ready", result
    return json.loads(result.view_json)


def test_profile_registration_is_exact_and_report_disabled():
    from nurse_rcl.v2.live import ops_contracts

    profile_type = getattr(ops_contracts, "OpsSreEntryProfile", None)
    assert profile_type is not None, "entry profile/16 is missing"
    profile = profile_type()
    assert profile.schema_version == "nurse.v2.sre-profile/16"
    assert profile.profile_id == "v2-ops-sre/9"
    assert profile.investigation_profile_id == "hermes-sre-tools/19"
    assert profile.decision_protocol == "nurse.v2.sre-decision-exchange/20"
    assert profile.report_limit == 0
    with pytest.raises(ValueError):
        ops_contracts.OpsSreDirectedProfile.model_validate(profile.model_dump(mode="json"))


def test_new_input_skills_and_workbench_contain_real_entry_table(complete, tmp_path):
    tool, book, _, case = opened(complete)
    with OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        view = project(ledger, book, case)
    assert view["schema_version"] == "nurse.v2.sre-investigation-input/20"
    assert view["action_entries"]["entries"]
    assert all(row["skill_id"].endswith("/15") for row in view["skills"])
    for row in sre_skill_index(profile="hermes-sre-tools/19"):
        assert "entry_ref" in row["guidance"]
        assert any("entry_ref" in e or "navigation" in e for e in row["examples"])


def test_scoped_purpose_guidance_reaches_frozen_wire_and_fresh_reader(complete, tmp_path):
    from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
    from nurse_rcl.v2.live.ops_sre_entry_exchange import ENTRY_SYSTEM, entry_common_guidance
    from tests.test_v2_live_sre_entry_provider import frozen

    original_core = canonical_json_bytes(complete[2].model_dump(mode="json"))
    case, wire, _, _ = frozen(complete, tmp_path / "tools")
    envelope = json.loads(wire)
    assert envelope["messages"][0]["content"] == ENTRY_SYSTEM
    body = json.loads(envelope["messages"][1]["content"])
    packet = body["context"]
    restored = brief_context_from_wire(
        wire,
        opaque_case_id=case,
        case_snapshot_id=body["case_snapshot_id"],
        decision_index=1,
    )
    assert restored["common_skill_guidance"] == entry_common_guidance()
    for skill in restored["skills"]:
        assert "changes the next check, not source identity" in skill["guidance"]
        assert "Keep the stored read question's original scope" in skill["guidance"]
    # Inspect the outbound packet as well; canonical restoration alone cannot prove delivery.
    encoded = canonical_json_bytes(packet).decode()
    assert "different metrics are not interchangeable negative controls" in encoded
    assert "A broader diagnostic question belongs in unresolved" in encoded
    assert "a covered normal contrast can weaken an explanation" not in encoded
    assert restored["action_entries"]["entries"]  # Legal supply remains reachable.
    assert canonical_json_bytes(complete[2].model_dump(mode="json")) == original_core


def test_frozen_wire_short_names_and_fresh_resolution_preserve_action_contract(complete, tmp_path):
    from nurse_rcl.v2.live.ops_sre_brief_exchange import (
        bind_brief_decision,
        brief_context_from_wire,
        prepare_brief_decision,
    )
    from nurse_rcl.v2.live.ops_sre_model_view import resolve_model_response
    from nurse_rcl.v2.live.ops_sre_turn_validation import (
        bind_workbench_turn,
        validate_bound_workbench_response,
    )
    from tests.test_v2_live_ark import response
    from tests.test_v2_live_sre_action_decision import decision

    tool, book, _, case = opened(complete)
    references = book.to_state()
    with OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        result = sre_investigation_input(
            capabilities=ledger.capabilities(),
            records=(),
            decisions=(),
            decision_index=1,
            entry_references=references,
            entry_case_id=case,
            make_wire=lambda v: dict(context=v),
        )
        assert result.status == "ready", result
        view = json.loads(result.view_json)
    wire = prepare_brief_decision(
        opaque_case_id=case,
        case_snapshot_id=tool.case_snapshot_id,
        decision_index=1,
        context=view,
        references=references,
    )
    body = json.loads(json.loads(wire)["messages"][1]["content"])
    assert body["protocol"] == "nurse.v2.sre-decision-exchange/20"
    assert body["context"]["schema_version"] == "nurse.v2.sre-model-view/16"
    assert body["context"]["workbench"]["action_entries"]["entries"]
    binding = bind_brief_decision(
        wire, opaque_case_id=case, case_snapshot_id=tool.case_snapshot_id, decision_index=1
    )
    assert binding.variant == "v2-ops-sre/9"
    restored = brief_context_from_wire(
        wire,
        opaque_case_id=case,
        case_snapshot_id=tool.case_snapshot_id,
        decision_index=1,
        references=deepcopy(references),
    )
    assert canonical_json_bytes(restored) == canonical_json_bytes(view)
    value = decision(first=True)
    value["actions"] = [dict(entry_ref="A999", question="Unknown but structurally valid.")]
    parsed = validate_bound_workbench_response(
        canonical_json_bytes(response(canonical_json_bytes(value).decode())),
        bind_workbench_turn(wire, binding),
    )
    resolved = resolve_model_response(
        json.loads(parsed.output_json),
        packet=body["context"],
        references=references,
        case_id=case,
        snapshot=tool.case_snapshot_id,
        index=1,
        report=False,
    )
    assert resolved["actions"] == value["actions"]  # No semantic reference becomes a parse retry.
    service = next(r[1] for r in references["entries"] if r[0] == "service")
    value["actions"] = [
        dict(
            navigation=dict(
                kind="candidate", candidate_service_key=service, focus="symptoms", cursor=None
            ),
            question="Inspect this scope.",
        )
    ]
    resolved = resolve_model_response(
        value,
        packet=body["context"],
        references=references,
        case_id=case,
        snapshot=tool.case_snapshot_id,
        index=1,
        report=False,
    )
    assert (
        resolved["actions"] == value["actions"]
    )  # Navigation decoding belongs to action admission.
    with pytest.raises((ValueError, ExchangeBoundaryError)):
        prepare_brief_decision(
            opaque_case_id=case,
            case_snapshot_id=tool.case_snapshot_id,
            decision_index=1,
            context=view,
            references={**references, "schema_version": "nurse.v2.sre-reference-book/1"},
        )


def test_ordinary_provider_rejects_new_profile_before_credential_setup(
    complete, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_provider

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary owner must reject before provider credentials/network")

    monkeypatch.setattr(ops_provider.ark, "load_ark_settings_from_key_file", forbidden)
    with pytest.raises(ValueError, match="stage|bounded"):
        ops_provider.OpsLiveProvider(
            root=tmp_path,
            code_sha256="2" * 64,
            execution_sha256="4" * 64,
            approved_execution_sha256="4" * 64,
            opaque_case_id=complete[0].catalog.opaque_case_id,
            case_snapshot_id="5" * 64,
            empirical=True,
            sre=True,
            sre_profile="hermes-sre-tools/19",
        )


def test_entry_profile_is_registered_for_same_fixed_core_and_unknown_is_rejected():
    from nurse_rcl.v2.live.ops_sre_core_import import _SourceCoreMetadata

    registered = get_args(_SourceCoreMetadata.model_fields["profile_id"].annotation)
    assert "v2-ops-sre/9" in registered
    assert "v2-ops-sre/999" not in registered


def test_entry_binding_has_no_report_and_one_lifetime_correction_slot():
    from nurse_rcl.v2.live.ark_bridge import LiveRequestBinding

    fields = dict(
        opaque_case_id="opq_" + "1" * 32,
        variant="v2-ops-sre/9",
        input_sha256="2" * 64,
        context_sha256="3" * 64,
        prompt_schema_sha256="4" * 64,
        request_sha256="5" * 64,
    )
    main = LiveRequestBinding(**fields, phase="investigation", decision_index=1)
    other = LiveRequestBinding(**fields, phase="investigation", decision_index=2)
    first = LiveRequestBinding(
        **fields, phase="investigation_correction", decision_index=1, correction_index=1
    )
    later = LiveRequestBinding(
        **fields, phase="investigation_correction", decision_index=2, correction_index=1
    )
    assert main.slot != other.slot
    assert first.slot == later.slot
    with pytest.raises(ValueError):
        LiveRequestBinding(**fields, phase="report")


def test_entry_native_parser_requires_exact_frozen_turn(complete, tmp_path):
    from nurse_rcl.v2.live.ark_bridge import _parse_protocol_response
    from nurse_rcl.v2.live.ops_empirical_policy import RESPONSE_LIMITS
    from nurse_rcl.v2.live.ops_sre_brief_exchange import bind_brief_decision, prepare_brief_decision
    from nurse_rcl.v2.live.ops_sre_turn_validation import bind_workbench_turn
    from tests.test_v2_live_ark import response
    from tests.test_v2_live_sre_action_decision import decision

    tool, book, _, case = opened(complete)
    references = book.to_state()
    with OpsSreEntryInvestigationLedger(
        root=tmp_path, toolbox=tool, code_sha256="2" * 64
    ) as ledger:
        view = sre_investigation_input(
            capabilities=ledger.capabilities(),
            records=(),
            decisions=(),
            decision_index=1,
            entry_references=references,
            entry_case_id=case,
            make_wire=lambda v: dict(context=v),
        )
    wire = prepare_brief_decision(
        opaque_case_id=case,
        case_snapshot_id=tool.case_snapshot_id,
        decision_index=1,
        context=json.loads(view.view_json),
        references=references,
    )
    binding = bind_brief_decision(
        wire, opaque_case_id=case, case_snapshot_id=tool.case_snapshot_id, decision_index=1
    )
    value = decision(first=True)
    value["actions"] = [dict(entry_ref="A999", question="A semantic refusal, not a retry.")]
    raw = canonical_json_bytes(response(canonical_json_bytes(value).decode()))
    with pytest.raises(ValueError, match="frozen request binding"):
        _parse_protocol_response(
            raw,
            output_tool_name=None,
            limits=RESPONSE_LIMITS,
            protocol="nurse.v2.sre-decision-exchange/20",
        )
    parsed = _parse_protocol_response(
        raw,
        output_tool_name=None,
        limits=RESPONSE_LIMITS,
        protocol="nurse.v2.sre-decision-exchange/20",
        workbench_turn=bind_workbench_turn(wire, binding),
    )
    assert json.loads(parsed.output_json)["actions"] == value["actions"]
