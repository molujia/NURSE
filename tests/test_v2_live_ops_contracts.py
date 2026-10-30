"""New identities and formal semantics must not inherit LLM authority."""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.normal_queries import run_normal_queries
from nurse_rcl.v2.live.normal_scoring import score_normal_case
from tests.test_v2_live_normal_queries import prepared_cases  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def api(name="ops_contracts"):
    qualified = "nurse_rcl.v2.live." + name
    assert importlib.util.find_spec(qualified), "missing isolated contract: " + name
    return importlib.import_module(qualified)


@pytest.fixture(scope="module")
def evidence(prepared_cases, tmp_path_factory):  # noqa: F811
    data, blocks, _ = prepared_cases("eight")
    session = run_normal_queries(
        data=data, ledger_root=tmp_path_factory.mktemp("ops-core") / "ledger", block_root=blocks,
    )
    return score_normal_case(data=data, session=session)


def identity(c, evidence, **overrides):
    fields = dict(
        dataset=evidence.dataset, opaque_case_id=evidence.opaque_case_id,
        case_snapshot_id=c.snapshot_id(evidence),
        core_method_id="normal-b0-ops-core/1",
        configuration_sha256="1" * 64, environment_sha256="2" * 64,
    )
    fields.update(overrides)
    return c.OpsCoreIdentity(**fields)


def test_profile_is_independent_of_historical_registry():
    path = ROOT / "configs/v2/ops_isolated_profile.json"
    assert path.is_file(), "missing isolated profile registry"
    c = api()
    profile = c.OpsProfile.model_validate_json(path.read_bytes())
    assert profile.profile_id == "v2-ops-isolated/1"
    assert profile.core_method_id == "normal-b0-ops-core/1"
    assert profile.investigation_profile_id == "hermes-readonly-ops/1"
    assert (profile.action_limit, profile.decision_limit, profile.report_limit,
            profile.attempt_limit, profile.llm_in_flight) == (8, 8, 1, 5, 1)
    assert profile.api_key_env == "ARK_API_KEY2"
    assert profile.model == "deepseek-v4-flash-ga-260731"
    assert profile.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    old = json.loads((ROOT / "configs/v2/experiment_versions.json").read_bytes())
    assert next(v for v in old["variants"] if v["id"] == "B2")["semantic_stage"] == "pre_ranking"
    for field, value in (("api_key_env", "ARK_API_KEY"), ("action_limit", 9),
                         ("model", "other"), ("llm_in_flight", 2)):
        with pytest.raises(ValidationError):
            c.OpsProfile.model_validate({**profile.model_dump(), field: value})


def test_core_digest_excludes_runtime_model_and_investigation(evidence):
    c, r = api(), api("ops_result")
    result = r.seal_core_result(identity=identity(c, evidence), evidence=evidence)
    body = result.model_dump(mode="json", exclude={"core_result_digest"})
    assert result.core_result_digest == sha256_hex(canonical_json_bytes(body))
    for metadata in (
        dict(elapsed_seconds=1, model="disabled", investigation=[]),
        dict(elapsed_seconds=9000, model="other", investigation=["contradiction"]),
    ):
        envelope = {"core": result.model_dump(mode="json"), "execution": metadata}
        assert r.OpsCoreResult.model_validate_json(
            canonical_json_bytes(envelope["core"]),
        ).core_result_digest == (
            result.core_result_digest
        )
        with pytest.raises(TypeError):
            r.seal_core_result(identity=identity(c, evidence), evidence=evidence, **metadata)
    assert result.evidence == evidence
    assert result.evidence is not evidence
    for field in ("configuration_sha256", "environment_sha256"):
        changed = r.seal_core_result(
            identity=identity(c, evidence, **{field: "3" * 64}), evidence=evidence,
        )
        assert changed.core_result_digest != result.core_result_digest


@pytest.mark.parametrize("field", ["case_snapshot_id", "opaque_case_id", "dataset"])
def test_core_rejects_foreign_identity(evidence, field):
    c, r = api(), api("ops_result")
    value = {"case_snapshot_id": "9" * 64, "opaque_case_id": "foreign",
             "dataset": "aiops25"}[field]
    with pytest.raises(ValueError, match="identity|snapshot|case"):
        r.seal_core_result(identity=identity(c, evidence, **{field: value}), evidence=evidence)


def test_core_rejects_nonterminal_or_altered_evidence(evidence):
    c, r = api(), api("ops_result")
    core_identity = identity(c, evidence)
    for session_change in ({"stop_reason": None}, {"catalog_sha256": "0" * 64}):
        damaged = evidence.model_copy(update={
            "session": evidence.session.model_copy(update=session_change),
        })
        with pytest.raises(ValueError):
            r.seal_core_result(identity=core_identity, evidence=damaged)
    result = r.seal_core_result(identity=core_identity, evidence=evidence)
    body = result.model_dump(mode="json")
    body["evidence"]["ranking"]["scores"][0]["total_score"] += 0.001
    with pytest.raises(ValueError, match="digest mismatch"):
        r.OpsCoreResult.model_validate_json(canonical_json_bytes(body))
    for extra in ("hybrid_intent", "investigation", "model", "elapsed_seconds"):
        with pytest.raises(ValidationError):
            c.OpsCoreIdentity.model_validate({**core_identity.model_dump(), extra: "forbidden"})


def test_node_and_investigation_contracts_preserve_unknown_and_roles():
    c = api()
    node = c.OpsNodeEvidence(
        case_snapshot_id="1" * 64, node="symptoms", tool_version="symptoms/1",
        parameters_json='{"observation_id":"o1"}', evidence_ids=("e1",),
        source_refs=("2" * 64,), unit=None, window_ns=(0, 10), coverage=None,
        availability="unavailable", truth_value="unknown", limitations=("missing_series",),
        payload_json='{"measurement":null}',
    )
    assert json.loads(node.payload_json) == {"measurement": None}
    for patch in ({"truth_value": "true"}, {"window_ns": (10, 0)},
                  {"coverage": 1.1}, {"payload_json": '{"a":1,"a":2}'},
                  {"parameters_json": '{"x": NaN}'}):
        with pytest.raises(ValidationError):
            c.OpsNodeEvidence.model_validate({**node.model_dump(), **patch})
    found = c.OpsInvestigationEvidence(
        action_index=1, evidence=node, origin="measured", reused_core_evidence_ids=(),
    )
    assert found.role == "supplementary"
    with pytest.raises(ValidationError):
        c.OpsInvestigationEvidence.model_validate({**found.model_dump(), "role": "formal"})
    with pytest.raises(ValidationError):
        c.OpsInvestigationEvidence.model_validate({**found.model_dump(), "formal_used": True})
    action = c.OpsInvestigationAction(
        action_index=1, skill_id="symptoms/1", tool="inspect_symptoms",
        parameters_json='{"observation_id":"o1"}', question="检查持续恶化是否存在",
    )
    assert action.action_index == 1
    with pytest.raises(ValidationError):
        c.OpsInvestigationAction.model_validate({**action.model_dump(), "action_index": 9})


def test_report_contract_never_accepts_a_model_ranking():
    c = api()
    report = c.OpsReportDraft(
        core_result_digest="1" * 64, formal_grounds=(), supplementary_findings=(),
        limitations=("not yet investigated",), conditional_next_steps=(),
    )
    assert "ranking" not in report.model_dump()
    with pytest.raises(ValidationError):
        c.OpsReportDraft.model_validate({**report.model_dump(), "ranking": ["replacement"]})
    state = c.OpsChannelStatus(core="completed", investigation="failed", report="unavailable")
    assert state.core == "completed"
