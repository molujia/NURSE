"""Actual normal projection/provider/ledger/owner/render and keyless fresh read."""

import json
from importlib import import_module, util

import pytest

from nurse_rcl.artifacts import canonical_json_bytes


def api():
    name = "nurse_rcl.v2.live.ops_sre_gplus_run"
    assert util.find_spec(name), "missing normal G+ owner"
    return import_module(name)


def context():
    from nurse_rcl.v2.live.ops_sre_brief_report import (
        BriefSreReportContext,
        ComparableReportMetadata,
        _retained_supply,
    )
    from tests.test_v2_live_sre_typed_report_wire import context as original

    before, refs = original()
    fields = before.metadata.model_dump(
        exclude={"schema_version", "profile", "investigation_input_schema"}
    )
    metadata = ComparableReportMetadata(**fields)
    return BriefSreReportContext(
        metadata, before.fact_store, _retained_supply(metadata, before.fact_store)
    ), refs


def response(wire):
    from nurse_rcl.v2.live.ops_sre_gplus_report import gplus_report_body
    from nurse_rcl.v2.live.ops_sre_typed_report_binding import typed_expression_catalog

    body, store = gplus_report_body(wire)
    catalog = typed_expression_catalog(body["context"]["supply"], store)
    expression = next(
        r["expression"]
        for r in catalog["rows"]
        if r["expression"]["claim"]["operation"] == "explain_uncertainty"
    )
    ref = expression["claim"]["evidence_refs"][0]
    field = catalog["sources"][ref]["fields"][0]["field"]
    value = dict(
        schema_version="nurse.v2.atomic-gplus-report-expression-response/2",
        predicted_paths=[],
        path_abstention_reason="The supplied evidence does not support a path prediction.",
        executive_summary=[],
        formal_interpretations=[],
        step_findings={},
        final_synthesis=[],
        limitations=[
            dict(
                claim_id="claim_1",
                expression_ref=expression["expression_ref"],
                primary_fields=[dict(evidence_ref=ref, field=field)],
                surface_text="The supplied status remains unknown.",
                outcomes=None,
            )
        ],
        next_checks=[],
    )
    return canonical_json_bytes(value).decode()


def run(tmp_path, *, text=None):
    module = api()
    import httpx

    from nurse_rcl.agents.ark import ArkCredential, ArkSettings
    from nurse_rcl.v2.live.ops_sre_gplus_provider import GplusReportProvider
    from tests.test_v2_live_sre_typed_report_delivery import body, outer

    value, refs = context()
    root = tmp_path / "report"
    seen = []

    def complete(wire):
        def http(request):
            seen.append(request.content)
            return httpx.Response(
                200, content=body(outer(response(wire) if text is None else text))
            )

        return GplusReportProvider(
            root=root / "provider",
            wire=wire,
            references=refs,
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
            approved_execution_sha256="e" * 64,
            expected_context_sha256=value.context_sha256,
            settings=ArkSettings(ArkCredential("fixture-only-no-network-secret")),
            transport=httpx.MockTransport(http),
            lock_path=tmp_path / "in-flight.lock",
            sleep=lambda _: None,
        ).report(wire)

    state = module.run_gplus_report(
        root=root,
        context=value,
        references=refs,
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        approved_execution_sha256="e" * 64,
        complete=complete,
    )
    return module, root, state, seen, complete, value, refs


def test_normal_gplus_full_chain_and_fresh_reader_do_not_send_or_change_files(tmp_path):
    module, root, state, seen, complete, value, refs = run(tmp_path)
    assert state["status"] in {"completed", "partial"}
    assert state["semantic_certified"] is False
    assert len(seen) == 1
    assert (root / "report.md").is_file() and (root / "evidence.md").is_file()
    native = json.loads((root / "native.bin").read_bytes())
    assert native["limitations"][0]["surface_text"] == "The supplied status remains unknown."
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    loaded = module.inspect_gplus_report(
        root=root,
        expected_context_sha256=value.context_sha256,
        expected_code_sha256="c" * 64,
        expected_execution_sha256="e" * 64,
    )
    assert loaded == state
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    # Existing terminal owner is read-only even if a callback is supplied again.
    assert (
        module.run_gplus_report(
            root=root,
            context=value,
            references=refs,
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
            approved_execution_sha256="e" * 64,
            complete=complete,
        )
        == state
    )
    assert len(seen) == 1


@pytest.mark.parametrize(
    "member",
    [
        "context.json",
        "references.json",
        "request.bin",
        "native.bin",
        "binding.json",
        "resolved.json",
        "checks.json",
        "report.md",
    ],
)
def test_fresh_reader_rejects_modified_saved_member(tmp_path, member):
    module, root, _, _, _, value, _ = run(tmp_path)
    path = root / member
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        module.inspect_gplus_report(
            root=root,
            expected_context_sha256=value.context_sha256,
            expected_code_sha256="c" * 64,
            expected_execution_sha256="e" * 64,
        )


def test_bad_complete_native_is_rejected_without_quality_retry_or_prose_repair(tmp_path):
    module, root, state, seen, _, value, _ = run(tmp_path, text="wrong complete native\n")
    assert state["status"] == "rejected"
    assert len(seen) == 1 and (root / "native.bin").read_bytes() == b"wrong complete native\n"
    assert (
        module.inspect_gplus_report(
            root=root,
            expected_context_sha256=value.context_sha256,
            expected_code_sha256="c" * 64,
            expected_execution_sha256="e" * 64,
        )
        == state
    )


def test_unapproved_owner_does_not_invoke_callback(tmp_path):
    module = api()
    value, refs = context()
    seen = []
    with pytest.raises(ValueError, match="approval"):
        module.run_gplus_report(
            root=tmp_path / "report",
            context=value,
            references=refs,
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
            approved_execution_sha256=None,
            complete=lambda wire: seen.append(wire),
        )
    assert seen == [] and not (tmp_path / "report").exists()


def test_presend_resource_detail_is_saved_and_fresh_read_without_callback(tmp_path, monkeypatch):
    module = api()
    from nurse_rcl.v2.live.ops_context_view import DiagnosedInputProjection
    from nurse_rcl.v2.live.ops_failure import OpsFailureDetail

    value, refs = context()
    detail = OpsFailureDetail(
        stage="report_projection",
        failure_type="ExchangeBoundaryError",
        boundary="json_nodes",
        observed=100013,
        limit=100000,
        observed_is_lower_bound=True,
    )
    monkeypatch.setattr(
        module,
        "project_gplus_report",
        lambda *a, **k: DiagnosedInputProjection(
            status="resource_limit",
            failure_detail=detail,
        ),
    )

    def forbidden(_):
        pytest.fail("pre-send rejection invoked provider")

    root = tmp_path / "report"
    state = module.run_gplus_report(
        root=root,
        context=value,
        references=refs,
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        approved_execution_sha256="e" * 64,
        complete=forbidden,
    )
    assert state["status"] == "unavailable" and state["call_started"] is False
    assert state["schema_version"] == "nurse.v2.sre-gplus-report-run/3"
    assert state["failure_detail"] == detail.model_dump(mode="json")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert (
        module.inspect_gplus_report(
            root=root,
            expected_context_sha256=value.context_sha256,
            expected_code_sha256="c" * 64,
            expected_execution_sha256="e" * 64,
        )
        == state
    )
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_unknown_send_failure_detail_has_no_exception_message_or_retry(tmp_path):
    module = api()
    value, refs = context()
    calls = []

    def failed(_):
        calls.append(1)
        raise RuntimeError("untrusted provider text must not be persisted")

    root = tmp_path / "report"
    state = module.run_gplus_report(
        root=root,
        context=value,
        references=refs,
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        approved_execution_sha256="e" * 64,
        complete=failed,
    )
    assert state["call_started"] and state["status"] == "unavailable"
    assert state["failure_detail"]["stage"] == "report_provider"
    assert state["failure_detail"]["failure_type"] == "RuntimeError"
    assert "untrusted provider text" not in (root / "result.json").read_text()
    assert (
        module.run_gplus_report(
            root=root,
            context=value,
            references=refs,
            code_sha256="c" * 64,
            execution_sha256="e" * 64,
            approved_execution_sha256="e" * 64,
            complete=failed,
        )
        == state
    )
    assert calls == [1]


def test_legacy_presend_owner_remains_readonly_without_inventing_detail(tmp_path, monkeypatch):
    module = api()
    from nurse_rcl.v2.live.ops_context_view import OpsInputProjection

    value, refs = context()
    monkeypatch.setattr(
        module,
        "project_gplus_report",
        lambda *a, **k: OpsInputProjection(
            status="resource_limit",
        ),
    )
    root = tmp_path / "report"
    state = module.run_gplus_report(
        root=root,
        context=value,
        references=refs,
        code_sha256="c" * 64,
        execution_sha256="e" * 64,
        approved_execution_sha256="e" * 64,
        complete=lambda _: pytest.fail("legacy fixture sent"),
    )
    state["schema_version"] = "nurse.v2.sre-gplus-report-run/1"
    state.pop("failure_detail", None)
    module._save(root, state)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    loaded = module.inspect_gplus_report(
        root=root,
        expected_context_sha256=value.context_sha256,
        expected_code_sha256="c" * 64,
        expected_execution_sha256="e" * 64,
    )
    assert "failure_detail" not in loaded
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
