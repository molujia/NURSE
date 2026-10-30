"""Saved one-case status is not execution, restoration or a new paid allowance."""

from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_status"
    assert util.find_spec(name), "missing G+ keyless composition status"
    return import_module(name)


def fixture(tmp_path):
    from nurse_rcl.v2.live.ops_sre_gplus_case import GplusCompositionPlan

    plan = GplusCompositionPlan(
        core_plan_path=str(tmp_path / "not-read-core.json"),
        core_plan_sha256="a" * 64,
        run_root=str(tmp_path),
        investigation_decision_limit=2,
    )
    path = tmp_path / "plan.json"
    raw = canonical_json_bytes(plan.model_dump(mode="json"))
    path.write_bytes(raw)
    return path, sha256_hex(raw)


def test_unstarted_status_never_reads_core_or_loads_any_runtime(tmp_path):
    module = api()
    path, pin = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = module.inspect_gplus_case(path, pin)
    assert result["status"] == "not_started"
    assert result["new_ark_calls"] == 0 and result["preparation_load_import_dispatch"] == 0
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("bad", ["result_signature", "composition_pin"])
def test_terminal_tampering_rejected_before_source_access(tmp_path, bad):
    module = api()
    path, pin = fixture(tmp_path)
    result = dict(
        composition_sha256=pin if bad == "result_signature" else "b" * 64,
        status="failed",
        failure_type="synthetic",
    )
    result["result_sha256"] = (
        "c" * 64 if bad == "result_signature" else sha256_hex(canonical_json_bytes(result))
    )
    (tmp_path / "composition-result.json").write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError):
        module.inspect_gplus_case(path, pin)
