"""New bounded stage reuses the existing runner; synthetic case/native fixtures only."""

# ruff: noqa: F811 -- fixtures.

import json

import httpx
import pytest

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2.live.ops_contracts import OpsSreEntryProfile
from nurse_rcl.v2.live.ops_sre_brief_exchange import brief_context_from_wire
from tests.test_v2_live_ark import response
from tests.test_v2_live_ops_execution import prepared_case, save  # noqa: F401
from tests.test_v2_live_ops_provider import mocked
from tests.test_v2_live_sre_action_decision import decision
from tests.test_v2_live_sre_directed_entry import approve, stage_plan


def entry_stage_plan(tmp_path, prepared):
    from nurse_rcl.v2.live.ops_plan import load_ops_plan

    path, admitted, core = stage_plan(tmp_path, prepared)
    value = admitted.plan.model_dump(mode="json")
    value["profile"] = OpsSreEntryProfile().model_dump(mode="json")
    return path, load_ops_plan(path, save(path, value)), core


def test_new_profile_cannot_enter_full_case_or_supervisor(prepared_case, tmp_path, monkeypatch):
    from nurse_rcl.v2.live import ops_case, ops_execution

    path, admitted, _ = entry_stage_plan(tmp_path, prepared_case)
    monkeypatch.setattr(
        ops_case, "_preparation", lambda *_: pytest.fail("full owner cannot prepare entry profile")
    )
    with pytest.raises(ValueError, match="stage"):
        ops_case.run_ops_case(admitted, approved_plan_sha256=admitted.plan_sha256)
    with pytest.raises(ValueError, match="stage"):
        ops_execution.run_round(
            path,
            admitted.plan_sha256,
            progress=lambda _: None,
            approved_plan_sha256=admitted.plan_sha256,
        )
    assert not admitted.state_root.exists()


def test_entry_stage_uses_frozen_core_real_receipt_saved_pause_and_disabled_report(
    prepared_case, tmp_path, monkeypatch
):
    from nurse_rcl.v2.live import ops_case, ops_provider
    from nurse_rcl.v2.live import ops_sre_directed_stage as stage

    path, admitted, core = entry_stage_plan(tmp_path, prepared_case)
    calls, previous = [], []

    def http(request):
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert body["protocol"] == "nurse.v2.sre-decision-exchange/20"
        index = body["context"]["reference_prefix"]["visible_at"]
        ctx = brief_context_from_wire(
            request.content,
            opaque_case_id=body["opaque_case_id"],
            case_snapshot_id=body["case_snapshot_id"],
            decision_index=index,
        )
        entries = [
            e
            for e in ctx["action_entries"]["entries"]
            if e["action"]["tool"] == "inspect_sources"
            and e["action"]["parameters"] not in previous
        ]
        assert entries
        previous.append(entries[0]["action"]["parameters"])
        value = decision(first=index == 1)
        value["investigation_note"] = None
        value["actions"] = [
            dict(entry_ref=entries[0]["entry_ref"], question="Inspect supplied symptom values.")
        ]
        if index > 1:
            latest = ctx["investigation_progress"]["latest_actual_result"]
            value["responds_to"] = [dict(action_index=latest["action_index"], card_ids=[])]
            value["latest_result_update"].update(
                action_index=latest["action_index"], content_relation=latest["content_relation"]
            )
        calls.append(index)
        return httpx.Response(200, json=response(canonical_json_bytes(value).decode()))

    mocked(ops_provider, tmp_path, monkeypatch, http)
    monkeypatch.setattr(ops_case, "run_ops_core", lambda **_: pytest.fail("recomputed source core"))
    first = stage.run_directed_stage(admitted, **approve(admitted, 1))
    assert first["termination"] == "stage_pause" and calls == [1]
    assert first["channels"] == dict(core="completed", investigation="partial", report="disabled")
    assert first["core_result_digest"] == core["core_result_digest"]
    core_bytes = (admitted.state_root / "core/result.json").read_bytes()
    with monkeypatch.context() as local:
        local.setattr(stage, "_preparation", lambda *_: pytest.fail("saved stage cannot prepare"))
        assert stage.run_directed_stage(admitted, **approve(admitted, 1)) == first
    second = stage.run_directed_stage(admitted, **approve(admitted, 2))
    assert second["decision_count"] == second["action_count"] == 2 and calls == [1, 2]
    assert stage.stage_status(path, admitted.plan_sha256)["decision_count"] == 2
    assert (admitted.state_root / "core/result.json").read_bytes() == core_bytes
    assert not (admitted.state_root / "report").exists()
    assert not (admitted.state_root / "case-result.json").exists()
    with pytest.raises(ValueError, match="stage approval"):
        stage.run_directed_stage(admitted, **dict(approve(admitted, 2), stage_decision_limit=3))
    assert calls == [1, 2]
