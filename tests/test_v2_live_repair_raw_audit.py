"""Final audit must read authentic complete files, not accept rehashed score claims."""

import importlib
import importlib.util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.test_v2_live_aiops25_assembly import measured_case
from tests.test_v2_live_aiops25_sources import load as load_aiops
from tests.test_v2_live_repair_assembly import measured_rcab
from tests.test_v2_live_sources import load


def api():
    name = "nurse_rcl.v2.live.repair_raw_audit"
    assert importlib.util.find_spec(name), "missing raw-file to complete-score repair audit"
    return importlib.import_module(name)


def arguments(tmp_path, dataset="rcabench"):
    root = measured_rcab(tmp_path) if dataset == "rcabench" else measured_case(tmp_path)
    sources = load(root) if dataset == "rcabench" else load_aiops(root)
    args = dict(
        source_root=root,
        dataset=dataset,
        opaque_case_id=sources.adapter.opaque_case_id,
        configuration=sources.adapter.configuration,
        expected_source_descriptor_sha256=sources.descriptor()["descriptor_sha256"],
        store_root=tmp_path / "blocks",
        ledger_root=tmp_path / "ledger",
    )
    if dataset == "aiops25":
        args.update(
            source_case_id=sources.source_case_id, authorized_queue=sources.authorized_queue
        )
    return args


def terminal(module, args):
    prepared = module.prepare_repair_case(**args)
    assert not prepared.raw_file_authority
    for _ in range(9):
        session = prepared.runtime.advance()
        if session.stop_reason:
            return session.session_sha256
    pytest.fail("bounded complete case did not terminate")


@pytest.mark.parametrize("dataset", ["rcabench", "aiops25"])
def test_complete_raw_case_produces_replayable_full_score_audit(tmp_path, dataset):
    module = api()
    args = arguments(tmp_path, dataset)
    pin = terminal(module, args)
    audit = module.build_repair_raw_audit(expected_session_sha256=pin, **args)
    assert audit.source_descriptor_sha256 == args["expected_source_descriptor_sha256"]
    assert audit.session_sha256 == pin
    assert audit.raw_file_and_score_replayed
    assert audit.ranking.candidate_service_keys
    assert audit.ranking.ordered_service_keys
    assert audit.graph.graph_pair == audit.ranking.propagation.graph_pair
    assert audit.graph.manifest.manifest_sha256 == audit.graph_manifest_sha256
    assert audit.ranking.propagation.local_scoring.strengths.dynamic_queries_used <= 8
    assert (
        module.require_repair_raw_audit(
            audit=audit,
            expected_audit_sha256=audit.result_sha256,
            expected_session_sha256=pin,
            **args,
        )
        == audit
    )
    assert module.RepairRawAudit.model_validate_json(audit.model_dump_json()) == audit
    raw = audit.model_dump_json()
    assert "PRIVATE RAW PROSE" not in raw and "private-answer-case" not in raw


def test_audit_never_creates_or_resets_a_query_ledger(tmp_path):
    module = api()
    args = arguments(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_repair_raw_audit(expected_session_sha256="0" * 64, **args)
    assert not args["ledger_root"].exists()


def test_source_pin_cannot_be_self_replaced_with_changed_data(tmp_path):
    module = api()
    args = arguments(tmp_path)
    pin = terminal(module, args)
    path = args["source_root"] / "abnormal_logs.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source"):
        module.build_repair_raw_audit(expected_session_sha256=pin, **args)


def test_self_signed_ranking_change_fails_complete_audit_replay(tmp_path):
    module = api()
    args = arguments(tmp_path)
    pin = terminal(module, args)
    audit = module.build_repair_raw_audit(expected_session_sha256=pin, **args)
    forged = audit.model_copy(update={"ranking": audit.ranking.model_copy(update={"scores": ()})})
    forged = forged.model_copy(
        update={
            "result_sha256": sha256_hex(
                canonical_json_bytes(forged.model_dump(mode="json", exclude={"result_sha256"}))
            )
        }
    )
    with pytest.raises(ValueError, match="replay"):
        module.require_repair_raw_audit(
            audit=forged,
            expected_audit_sha256=forged.result_sha256,
            expected_session_sha256=pin,
            **args,
        )


@pytest.mark.parametrize("exceptional", [False, True])
def test_raw_file_drift_at_both_score_exits_is_rejected(tmp_path, monkeypatch, exceptional):
    module = api()
    args = arguments(tmp_path)
    pin = terminal(module, args)
    original = module.build_repair_ranking
    reached = []

    def mutate(**kwargs):
        result = original(**kwargs)
        path = args["source_root"] / "normal_logs.parquet"
        path.write_bytes(path.read_bytes() + b"drift")
        reached.append(True)
        if exceptional:
            raise RuntimeError("controlled scoring failure")
        return result

    monkeypatch.setattr(module, "build_repair_ranking", mutate)
    with pytest.raises(ValueError, match="source.*changed"):
        module.build_repair_raw_audit(expected_session_sha256=pin, **args)
    assert reached


@pytest.mark.parametrize("target", ["scalar", "errors"])
@pytest.mark.parametrize("exceptional", [False, True])
def test_outer_raw_audit_keeps_generated_sources_protected_until_exit(
    tmp_path, monkeypatch, target, exceptional
):
    module = api()
    args = arguments(tmp_path)
    pin = terminal(module, args)
    original = module.build_repair_ranking
    reached = []

    def mutate(**kwargs):
        result = original(**kwargs)
        if target == "scalar":
            victim = kwargs["runtime"].arguments["sources"][0]
            victim.__pydantic_fields_set__.add("unexpected_after_score_marker")
        else:
            victim = kwargs["error_sources"]
            object.__setattr__(victim, "projections", list(victim.projections))
        reached.append(True)
        if exceptional:
            raise RuntimeError("controlled post-score failure")
        return result

    monkeypatch.setattr(module, "build_repair_ranking", mutate)
    with pytest.raises(ValueError, match="changed"):
        module.build_repair_raw_audit(expected_session_sha256=pin, **args)
    assert reached


def test_raw_audit_and_graph_use_independent_registered_schemas():
    module = api()
    registry = importlib.import_module("nurse_rcl.v2.schema_compatibility")
    assert (
        registry.CURRENT_SCHEMA_VERSIONS.get("live_repair_raw_audit")
        == module.REPAIR_RAW_AUDIT_SCHEMA_VERSION
    )
    assert (
        registry.CURRENT_SCHEMA_VERSIONS.get("live_repair_graph") == "nurse.v2.live-repair-graph/1"
    )
