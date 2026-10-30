"""Explicit focused report identity and legal same-record attribution facts."""

# ruff: noqa: F811 -- shared synthetic sealed-stage fixture.

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.sre_feedback_fact_fixtures import deny_effects  # noqa: F401
from tests.test_v2_live_sre_stage_handoff import no_dispatch, sealed_stage  # noqa: F401


@pytest.mark.parametrize("sealed_stage", ["focused"], indirect=True)
def test_sealed_focused_stage_has_explicit_source_storage_and_exact_gplus_wire(sealed_stage):
    from nurse_rcl.v2.live.ops_sre_brief_report import (
        decode_brief_report_storage,
        encode_brief_report_storage,
    )
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context
    from nurse_rcl.v2.live.ops_sre_gplus_report import gplus_report_body, project_gplus_report

    stage, core, binding, full = sealed_stage
    context, refs = stage_report_context(stage, **binding)
    identity = context.metadata.source_identity
    assert identity["schema_version"] == "nurse.v2.atomic-report-source/2"
    assert identity["execution_profile"] == "v2-ops-sre/15"
    assert identity["decision_protocol"] == "nurse.v2.sre-decision-exchange/26"
    assert identity["core_result_digest"] == core.core_result_digest
    assert identity["retained_sources_sha256"] == sha256_hex(canonical_json_bytes([full]))
    assert context.metadata.profile == "hermes-sre-tools/25"
    assert context.metadata.investigation_input_schema == "nurse.v2.sre-investigation-input/26"
    raw = encode_brief_report_storage(context)
    assert b'"nurse.v2.sre-report-storage/16"' in raw
    assert encode_brief_report_storage(decode_brief_report_storage(raw)) == raw
    projection = project_gplus_report(context, references=refs)
    assert projection.status == "ready", projection
    body, store = gplus_report_body(projection.wire_bytes, references=refs)
    assert body["context"]["source_identity"] == identity
    assert body["context"]["profile"] == "hermes-sre-tools/25"
    # This fixture delivered a mechanism brief: complete retained leaf facts
    # remain local, while every actual b field (and formal f) is report-visible.
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store

    provided = context.supply["provided_fact_refs"]
    assert (
        store.to_packet()
        == build_fact_store(context.fact_store.resolve(ref) for ref in provided).to_packet()
    )
    assert body["context"]["supply"] == context.supply
    assert set(context.fact_store.refs()) - set(provided)
    assert context.supply["steps"][0]["delivered_fact_refs"] == []
    assert context.supply["steps"][0]["delivered_brief_refs"]
    assert all(
        row["source_record_sha256"] == sha256_hex(canonical_json_bytes(full))
        for row in context.supply["briefs"]
    )


@pytest.mark.parametrize("sealed_stage", ["focused"], indirect=True)
@pytest.mark.parametrize(
    "damage", ["source_schema", "execution", "protocol", "metadata", "storage", "pin"]
)
def test_focused_report_rejects_resealed_mixed_identity(sealed_stage, damage):
    import json

    from nurse_rcl.v2.live.ops_sre_brief_report import (
        decode_brief_report_storage,
        encode_brief_report_storage,
    )
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context

    stage, _, binding, _ = sealed_stage
    context, _ = stage_report_context(stage, **binding)
    packet = json.loads(encode_brief_report_storage(context))
    metadata, source = packet["metadata"], packet["metadata"]["source_identity"]
    if damage == "source_schema":
        source["schema_version"] = "nurse.v2.atomic-report-source/1"
    elif damage == "execution":
        source["execution_profile"] = "v2-ops-sre/14"
    elif damage == "protocol":
        source["decision_protocol"] = "nurse.v2.sre-decision-exchange/25"
    elif damage == "metadata":
        metadata["profile"] = "hermes-sre-tools/23"
    elif damage == "storage":
        packet["schema_version"] = "nurse.v2.sre-report-storage/15"
    else:
        metadata["handoff_sha256"] = "f" * 64
    if damage != "pin":
        metadata["handoff_sha256"] = sha256_hex(canonical_json_bytes(source))
    with pytest.raises(ValueError):
        decode_brief_report_storage(canonical_json_bytes(packet))


def test_actual_attribution_is_assertable_same_record_source_with_exact_roundtrip():
    from nurse_rcl.v2.live.ops_sre_brief_report import (
        _source_facts,
        bind_retained_source,
        restore_retained_source,
    )
    from nurse_rcl.v2.live.ops_sre_fact_store import build_fact_store
    from tests.test_v2_live_sre_focused_sources import enriched

    row = enriched()
    assert "source_attribution" in row
    assert row["source_attribution"]["binding"]["status"] == "bound"
    facts = list(_source_facts(row, action_index=1, candidates={"service-a"}, focused=True))
    attribution = [item for item in facts if item.source_path.startswith("source_attribution")]
    assert attribution, "actual delivered source attribution has no legal report fact aliases"
    for item in attribution:
        assert item.action_index == 1 and item.source_refs == tuple(row["source_refs"])
        assert item.fact.target_service_keys == ("service-a",)
        assert "same_record_attribution_not_independent_evidence" in item.fact.status_codes
    store = build_fact_store(
        item.model_copy(update={"fact": item.fact.model_copy(update={"fact_ref": f"f{index}"})})
        for index, item in enumerate(facts, 1)
    )
    binding = bind_retained_source(dict(record=row, fact_refs=list(store.refs())), store)
    restored = restore_retained_source(binding, store)
    assert canonical_json_bytes(restored["record"]) == canonical_json_bytes(row)
    assert restored["fact_refs"] == list(store.refs())
    values = {
        value.value
        for item in attribution
        for value in item.fact.values
        if isinstance(value.value, str)
    }
    assert {"cpu_usage", "cores", "request-a", "observation-a", "a" * 64} <= values


@pytest.mark.parametrize("sealed_stage", ["focused_attribution"], indirect=True)
def test_sealed_actual_attribution_survives_storage_and_final_gplus_source(sealed_stage):
    from nurse_rcl.v2.live.ops_sre_brief_report import (
        decode_brief_report_storage,
        encode_brief_report_storage,
        restore_retained_source,
    )
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context
    from nurse_rcl.v2.live.ops_sre_gplus_report import gplus_report_body, project_gplus_report

    stage, _, binding, original = sealed_stage
    context, refs = stage_report_context(stage, **binding)
    restored = decode_brief_report_storage(encode_brief_report_storage(context))
    assert any(
        restore_retained_source(item, restored.fact_store)["record"] == original
        for item in restored.metadata.source_bindings
    )
    projection = project_gplus_report(restored, references=refs)
    assert projection.status == "ready", projection
    body, store = gplus_report_body(projection.wire_bytes, references=refs)
    # An EXACT delivery grants the entire actual source, including attribution;
    # unlike the separate brief fixture, no local source facts may disappear.
    assert store.to_packet() == restored.fact_store.to_packet()
    assert set(store.refs()) == set(body["context"]["supply"]["provided_fact_refs"])
    attrs = [
        item for item in store.iter_facts() if item.source_path.startswith("source_attribution")
    ]
    assert attrs and all(
        item.action_index == 1 and item.source_refs == tuple(original["source_refs"])
        for item in attrs
    )
    values = {
        value.value for item in attrs for value in item.fact.values if isinstance(value.value, str)
    }
    assert {"cpu_usage", "cores", "request-a", "observation-a", "a" * 64} <= values
    assert set(item.fact.fact_ref for item in attrs) <= set(
        body["context"]["supply"]["provided_fact_refs"]
    )


@pytest.mark.parametrize("sealed_stage", ["focused_attribution"], indirect=True)
def test_focused_pure_gplus_evaluation_checks_and_renders_actual_source(sealed_stage):
    import json

    from nurse_rcl.v2.live.ops_report_render import _text
    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context
    from nurse_rcl.v2.live.ops_sre_gplus_provider import GplusDelivery
    from nurse_rcl.v2.live.ops_sre_gplus_report import project_gplus_report
    from nurse_rcl.v2.live.ops_sre_gplus_run import _evaluate
    from nurse_rcl.v2.live.ops_sre_report_check_policy import report_check_policy
    from tests.test_v2_live_sre_gplus_run import response

    stage, _, binding, _ = sealed_stage
    context, refs = stage_report_context(stage, **binding)
    policy = report_check_policy(context)
    assert policy["schema_version"] == "nurse.v2.sre-report-check-policy/5"
    assert policy["metadata_schema"] == "nurse.v2.sre-report-context/20"
    wire = project_gplus_report(context, references=refs).wire_bytes
    assert wire is not None
    native = response(wire).encode()
    # Pure evaluator boundary only: no provider, credential, transport or owner.
    delivered = GplusDelivery(
        native=native,
        receipt={},
        origin=dict(request_sha256=sha256_hex(wire), native_sha256=sha256_hex(native)),
    )
    outputs = _evaluate(context, refs, wire, delivered)
    evidence_text = outputs["evidence.md"].decode()
    for value in (
        "cpu_usage",
        "cores",
        "request-a",
        "observation-a",
        "a" * 64,
        "source_attribution",
    ):
        assert _text(value) in evidence_text
    assert json.loads(outputs["checks.json"])
    assert b"sre-readable-pair/15" in outputs["report.md"]
    assert "stage_pause" in outputs["report.md"].decode()


@pytest.mark.parametrize("sealed_stage", ["focused_attribution"], indirect=True)
@pytest.mark.parametrize(
    "damage",
    [
        "missing_catalog",
        "catalog_hash",
        "catalog_schema",
        "catalog_snapshot",
        "profile",
        "envelope",
        "extra",
    ],
)
def test_exact_only_focused_stage_rejects_resealed_checkpoint_identity(sealed_stage, damage):
    import json

    from nurse_rcl.v2.live.ops_sre_gplus_case import stage_report_context
    from tests.test_v2_live_sre_stage_handoff import rewrite_stage

    stage, _, binding, _ = sealed_stage
    path = stage.state_root / "investigation/state.json"
    saved = json.loads(path.read_bytes())
    navigation = saved["navigation"]
    if damage == "missing_catalog":
        navigation["base"].pop("evidence_catalog")
    elif damage == "profile":
        navigation["base"]["profile"] = "hermes-sre-tools/23"
    elif damage == "envelope":
        navigation["storage_encoding"] = "sre-checkpoint-score-source/1"
    elif damage == "extra":
        navigation["unregistered"] = {}
    else:
        catalog = navigation["base"]["evidence_catalog"]
        if damage == "catalog_hash":
            catalog["state_sha256"] = "f" * 64
        else:
            catalog["schema_version" if damage == "catalog_schema" else "snapshot"] = (
                "nurse.v2.sre-evidence-catalog/0" if damage == "catalog_schema" else "f" * 64
            )
            catalog["state_sha256"] = sha256_hex(
                canonical_json_bytes(
                    {key: value for key, value in catalog.items() if key != "state_sha256"}
                )
            )
    saved["state_sha256"] = sha256_hex(
        canonical_json_bytes({key: value for key, value in saved.items() if key != "state_sha256"})
    )
    path.write_bytes(canonical_json_bytes(saved))
    rewrite_stage(stage)
    with pytest.raises(ValueError):
        stage_report_context(stage, **binding)


def test_focused_composition_uses_explicit_new_version_and_preserves_old_plan_shape():
    from nurse_rcl.v2.live import ops_sre_gplus_case as module

    cls = getattr(module, "GplusFocusedCompositionPlan", None)
    assert cls is not None, "focused composition plan is not registered"
    values = dict(
        core_plan_path="/isolated/core-plan.json",
        core_plan_sha256="a" * 64,
        run_root="/isolated/run",
        investigation_decision_limit=8,
    )
    original = module.GplusCompositionPlan(**values)
    focused = cls(**values)
    assert focused.schema_version == "nurse.v2.gplus-one-case-composition/2"
    assert focused.investigation_profile == "v2-ops-sre/15"
    assert "investigation_profile" not in original.model_dump(mode="json")
    assert module.composition_sha256(focused) != module.composition_sha256(original)
    assert (
        module.composition_grant(focused)["schema_version"] == "nurse.v2.gplus-composition-grant/2"
    )
    assert module.composition_grant(focused)["investigation_profile"] == "v2-ops-sre/15"
    assert "investigation_profile" not in module.composition_grant(original)


def test_focused_composition_status_rejects_old_child_profile(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nurse_rcl.v2.live import ops_sre_gplus_status as status
    from nurse_rcl.v2.live.ops_contracts import OpsSreFeedbackFactProfile
    from nurse_rcl.v2.live.ops_sre_gplus_case import GplusFocusedCompositionPlan, composition_sha256

    plan = GplusFocusedCompositionPlan(
        core_plan_path=str(tmp_path / "core.json"),
        core_plan_sha256="a" * 64,
        run_root=str(tmp_path),
        investigation_decision_limit=8,
    )
    path = tmp_path / "composition.json"
    path.write_bytes(canonical_json_bytes(plan.model_dump(mode="json")))
    pin = composition_sha256(plan)
    result = dict(
        composition_sha256=pin,
        channels=dict(core="failed"),
        stage_plan_sha256="b" * 64,
        report_context_sha256=None,
    )
    result["result_sha256"] = sha256_hex(canonical_json_bytes(result))
    (tmp_path / "composition-result.json").write_bytes(canonical_json_bytes(result))
    (tmp_path / "investigation-plan.json").write_bytes(b"{}")
    case = SimpleNamespace(opaque_case_id="opq_" + "c" * 32)
    core = SimpleNamespace(case=case, code_files={}, output_root=str(tmp_path / "core-results"))
    stage = SimpleNamespace(
        case=case,
        code_files={},
        source_core=SimpleNamespace(plan_sha256=plan.core_plan_sha256),
        output_root=str(tmp_path / "results"),
        profile=OpsSreFeedbackFactProfile(),
    )
    monkeypatch.setattr(
        status, "read_ops_plan", lambda path, pin: core if path.name == "core.json" else stage
    )
    with pytest.raises(ValueError, match="profile"):
        status.inspect_gplus_case(path, pin)


@pytest.mark.parametrize("damage", [None, "unknown", "snapshot", "pin"])
def test_readable_service_aliases_validate_actual_action_book_without_mutation(damage):
    from nurse_rcl.v2.live.ops_sre_action_references import SreActionReferenceBook
    from nurse_rcl.v2.live.ops_sre_report_render import service_aliases

    book = SreActionReferenceBook(snapshot="a" * 64, run_id="b" * 64)
    book.issue("service", "synthetic-service", visible_at=1)
    book.issue("action", "c" * 64, visible_at=1)
    references = book.to_state()
    if damage == "unknown":
        references["schema_version"] = "nurse.v2.sre-reference-book/99"
    elif damage == "pin":
        references["entries"][0][2] = "other-service"
    before = canonical_json_bytes(references)
    if damage is None:
        assert service_aliases(("synthetic-service",), references, snapshot="a" * 64) == {
            "synthetic-service": "服务S1"
        }
    else:
        with pytest.raises(ValueError):
            service_aliases(
                ("synthetic-service",),
                references,
                snapshot="f" * 64 if damage == "snapshot" else "a" * 64,
            )
    assert canonical_json_bytes(references) == before


def test_real_focused_composition_derives_profile15_child_from_synthetic_core(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from nurse_rcl.v2.live import ops_sre_gplus_case as module
    from nurse_rcl.v2.live.ops_contracts import OpsEmpiricalProfile
    from nurse_rcl.v2.live.ops_plan import OpsSinglePlan, read_ops_plan

    case = "opq_" + "c" * 32
    key = dict(
        schema_version="nurse.v2.live-repair-preparation-key/1",
        dataset="rcabench",
        opaque_case_id=case,
        input_sha256="a" * 64,
        source_descriptor_sha256="b" * 64,
        source_files=[],
        configuration_sha256="c" * 64,
        authorized_queue_sha256=None,
        source_case_id_sha256=None,
        q_base=8,
        q_max=12,
        preparation_code=[["synthetic.py", "d" * 64]],
        semantics_sha256="e" * 64,
    )
    key["key_sha256"] = sha256_hex(canonical_json_bytes(key))
    receipt = tmp_path / "synthetic-receipt.json"
    receipt.write_bytes(canonical_json_bytes(dict(manifest_sha256="f" * 64)))
    parent = OpsSinglePlan.model_validate_json(
        canonical_json_bytes(
            dict(
                code_root=str(tmp_path / "metadata-only-code"),
                code_files={"synthetic.py": "a" * 64},
                environment={},
                profile=OpsEmpiricalProfile().model_dump(mode="json"),
                mode="core-only",
                case=dict(
                    dataset="rcabench",
                    opaque_case_id=case,
                    input_path=str(tmp_path / "input.json"),
                    input_sha256="a" * 64,
                ),
                preparation=dict(
                    key=key,
                    archive_root=str(tmp_path / "metadata-only-archive"),
                    mode="build",
                    receipt_path=str(receipt),
                    manifest_sha256=None,
                    receipt_sha256=None,
                ),
                identity_manifest_path=str(tmp_path / "identity.json"),
                identity_artifact_sha256="b" * 64,
                parent_plan_path=str(tmp_path / "parent.json"),
                parent_plan_sha256="c" * 64,
                output_root=str(tmp_path / "core-results"),
            )
        )
    )
    core_root = tmp_path / "synthetic-core-terminal"
    core_root.mkdir()
    result_raw = canonical_json_bytes(dict(synthetic_core_terminal=True))
    (core_root / "case-result.json").write_bytes(result_raw)
    core = SimpleNamespace(plan=parent, state_root=core_root)
    run_root = tmp_path / "composed"
    run_root.mkdir()
    plan = module.GplusFocusedCompositionPlan(
        core_plan_path=str(tmp_path / "core-plan.json"),
        core_plan_sha256="d" * 64,
        run_root=str(run_root),
        investigation_decision_limit=8,
    )
    monkeypatch.setattr(
        module,
        "inspect_ops_case",
        lambda admitted: dict(
            channels=dict(core="completed", investigation="disabled", report="disabled")
        ),
    )
    admitted_children = []

    def admit_child(path, pin):
        # Only the external admission boundary is injected; canonical reading,
        # typed child validation and _derive_stage/profile selection are real.
        child = read_ops_plan(path, pin)
        admitted_children.append(child)
        return SimpleNamespace(plan=child)

    monkeypatch.setattr(module, "load_ops_plan", admit_child)
    path, pin = module._derive_stage(plan, core)
    child = read_ops_plan(path, pin)
    assert admitted_children == [child]
    assert child.profile.profile_id == "v2-ops-sre/15"
    assert child.profile.investigation_profile_id == "hermes-sre-tools/25"
    assert child.profile.report_limit == 0 and child.mode == "with-agent"
    assert child.source_core.plan_sha256 == plan.core_plan_sha256
    assert child.source_core.case_result_sha256 == sha256_hex(result_raw)
    assert child.parent_plan_path == plan.core_plan_path
    assert child.preparation.mode == "load"
    assert child.preparation.receipt_sha256 == sha256_hex(receipt.read_bytes())
    assert child.code_files == parent.code_files and child.case == parent.case


@pytest.mark.parametrize("terminal", ["completed", "completed_partial_report"])
@pytest.mark.parametrize(
    "missing",
    [
        None,
        "child_plan",
        "stage_outcome",
        "stage_round",
        "stage_plan_sha256",
        "stage_grant_sha256",
        "report_context_sha256",
        "report_state_sha256",
    ],
)
def test_successful_focused_status_requires_original_child_stage_and_report(
    tmp_path,
    monkeypatch,
    terminal,
    missing,
):
    from types import SimpleNamespace

    from nurse_rcl.v2.live import ops_sre_gplus_status as module
    from nurse_rcl.v2.live.ops_contracts import OpsSreFocusedProfile
    from nurse_rcl.v2.live.ops_sre_gplus_case import GplusFocusedCompositionPlan, composition_sha256

    plan = GplusFocusedCompositionPlan(
        core_plan_path=str(tmp_path / "core.json"),
        core_plan_sha256="a" * 64,
        run_root=str(tmp_path),
        investigation_decision_limit=8,
    )
    path = tmp_path / "composition.json"
    path.write_bytes(canonical_json_bytes(plan.model_dump(mode="json")))
    pin = composition_sha256(plan)
    case = SimpleNamespace(opaque_case_id="opq_" + "c" * 32)
    core = SimpleNamespace(
        case=case, code_files={}, output_root=str(tmp_path / "core-results"), code_sha256="1" * 64
    )
    stage = SimpleNamespace(
        case=case,
        code_files={},
        source_core=SimpleNamespace(plan_sha256=plan.core_plan_sha256),
        output_root=str(tmp_path / "results"),
        profile=OpsSreFocusedProfile(),
    )
    stage_root = tmp_path / "results" / case.opaque_case_id
    stage_root.mkdir(parents=True)
    if missing != "child_plan":
        (tmp_path / "investigation-plan.json").write_bytes(b"{}")
    if missing != "stage_outcome":
        (stage_root / "stage-outcome.json").write_bytes(b"{}")
    if missing != "stage_round":
        (stage_root / "stage-round.json").write_bytes(b"{}")
    result = dict(
        composition_sha256=pin,
        status=terminal,
        channels=dict(core="completed", investigation="partial", report="completed"),
        core_result_digest="b" * 64,
        stage_plan_sha256="c" * 64,
        stage_grant_sha256="d" * 64,
        report_context_sha256="e" * 64,
        report_state_sha256="f" * 64,
    )
    if missing in result:
        result[missing] = None
    result["result_sha256"] = sha256_hex(canonical_json_bytes(result))
    (tmp_path / "composition-result.json").write_bytes(canonical_json_bytes(result))
    # Reader boundaries stand in for already-tested saved owners; the actual
    # composition reader and its missing-original control flow remain real.
    monkeypatch.setattr(
        module, "read_ops_plan", lambda path, pin: core if path.name == "core.json" else stage
    )
    monkeypatch.setattr(module, "plan_identity", lambda plan, pin: dict(plan_sha256=pin))
    monkeypatch.setattr(
        module, "inspect_saved_case", lambda root, identity: dict(core_result_digest="b" * 64)
    )
    monkeypatch.setattr(
        module,
        "_inspect_stage",
        lambda root, identity: dict(
            approved_grant=dict(through_decision_index=8), decision_count=1
        ),
    )
    monkeypatch.setattr(
        module, "inspect_gplus_report", lambda **kwargs: dict(state_sha256="f" * 64)
    )
    if missing is None:
        assert module.inspect_gplus_case(path, pin)["status"] == terminal
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            module.inspect_gplus_case(path, pin)


def test_failed_focused_status_preserves_explicit_incomplete_terminal(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nurse_rcl.v2.live import ops_sre_gplus_status as module
    from nurse_rcl.v2.live.ops_sre_gplus_case import GplusFocusedCompositionPlan, composition_sha256

    plan = GplusFocusedCompositionPlan(
        core_plan_path=str(tmp_path / "core.json"),
        core_plan_sha256="a" * 64,
        run_root=str(tmp_path),
        investigation_decision_limit=8,
    )
    path = tmp_path / "composition.json"
    path.write_bytes(canonical_json_bytes(plan.model_dump(mode="json")))
    pin = composition_sha256(plan)
    result = dict(
        composition_sha256=pin,
        status="failed",
        channels=dict(core="failed"),
        stage_plan_sha256=None,
        report_context_sha256=None,
    )
    result["result_sha256"] = sha256_hex(canonical_json_bytes(result))
    (tmp_path / "composition-result.json").write_bytes(canonical_json_bytes(result))
    core = SimpleNamespace(
        output_root=str(tmp_path / "core-results"),
        case=SimpleNamespace(opaque_case_id="opq_" + "c" * 32),
    )
    monkeypatch.setattr(module, "read_ops_plan", lambda path, pin: core)
    assert module.inspect_gplus_case(path, pin)["status"] == "failed"
