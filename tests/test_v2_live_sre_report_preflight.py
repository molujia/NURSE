"""Offline source proof and output isolation, without credentials or network."""

# ruff: noqa: F811 -- imported fixtures

import importlib.util
from pathlib import Path

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.test_v2_live_sre_report import (  # noqa: F401
    complete,
    context,
    history,
    prepared_cases,
    report_inputs,
)


def api():
    path = Path(__file__).parents[1] / ".audit_scripts/sre_report_storage_preflight_20260929.py"
    assert path.exists(), "saved report production preflight not implemented"
    spec = importlib.util.spec_from_file_location("storage_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return sha256_hex(raw)


def test_preflight_api_exists():
    assert callable(api().main)


@pytest.fixture
def sources(context, tmp_path, monkeypatch):
    module = api()
    source, prior = tmp_path / "source", tmp_path / "prior"
    code = tmp_path / "code"
    code.mkdir()
    (code / "formal.py").write_bytes(b"# frozen synthetic formal projector\n")
    files = {"formal.py": sha256_hex((code / "formal.py").read_bytes())}
    identity = dict(case_snapshot_id=context.evidence.case_snapshot_id,
                    opaque_case_id=context.evidence.opaque_case_id)
    members = {"core/identity.json": canonical_json_bytes(identity),
               "core/result.json": b"synthetic pinned core bytes"}
    pins = {k: [len(v), sha256_hex(v)] for k, v in members.items()}
    provenance = dict(core_result_digest=context.core_result_digest, core_members=pins,
                      source={"plan_sha256": "a" * 64})
    plans = []
    for folder in (source, prior):
        root = folder / "results" / context.evidence.opaque_case_id
        for name, raw in members.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        put(root / "source-core.json", provenance)
        plan = dict(case={"opaque_case_id": context.evidence.opaque_case_id},
            code_root=str(code), code_files=files, output_root=str(folder / "results"),
            environment={"python_version": "fixture"}, preparation={"key": "fixture"},
            profile={"core_method_id": "normal-b0-ops-core/1"}, source_core=provenance["source"])
        plans.append(put(folder / "plan.json", plan))
    old = context.model_dump(mode="json")
    old["schema_version"] = "nurse.v2.sre-report-context/1"
    prior_root = prior / "results" / context.evidence.opaque_case_id
    context_pin = put(prior_root / "report/context.json", old)
    size = (prior_root / "report/context.json").stat().st_size
    receipt = dict(identity={"plan_sha256": plans[1],
        "code_sha256": sha256_hex(canonical_json_bytes(files))},
        case_snapshot_id=context.evidence.case_snapshot_id,
        core_result_digest=context.core_result_digest, channels={"core": "completed"},
        files={**pins, "report/context.json": [size, context_pin]})
    receipt["result_sha256"] = sha256_hex(canonical_json_bytes(receipt))
    put(prior_root / "case-result.json", receipt)
    expected = dict(plan=plans[0], prior_plan=plans[1],
        code=sha256_hex(canonical_json_bytes(files)),
        snapshot=context.evidence.case_snapshot_id, core=context.core_result_digest,
        case=context.evidence.opaque_case_id,
        formal_facts=sum(f.role == "formal_record" for f in context.evidence.facts))
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "PRIOR", prior)
    monkeypatch.setattr(module, "EXPECTED", expected)
    monkeypatch.setattr(module, "CURRENT", code)
    monkeypatch.setattr(module, "PROJECTION_FILES", ("formal.py",))
    monkeypatch.setattr(module, "CORE_MEMBERS", tuple(members))
    # These sentinels apply to actual function calls, not a fabricated success.
    import socket

    import httpx

    from nurse_rcl.agents import ark

    def forbidden(*_args, **_kwargs):
        pytest.fail("offline preflight cannot load credentials or send HTTP")

    for name in ("load_ark_settings", "load_server_ark_settings",
                 "load_ark_settings_from_key_file"):
        monkeypatch.setattr(ark, name, forbidden)
    monkeypatch.setattr(ark.ArkChatCompletionsClient, "complete", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("network"))
    return module, source, prior_root


def test_saved_projection_requires_full_source_proof(sources):
    module, source, prior = sources
    inventory = module.Inventory()
    formal, proof = module.verify_sources(inventory)
    assert len(formal.facts) == module.EXPECTED["formal_facts"]
    assert all(f.role == "formal_record" for f in formal.facts)
    assert proof["projection_compatible"] is True
    assert inventory.unchanged()


@pytest.mark.parametrize("mutation", ["core", "snapshot", "source_hash", "missing_proof"])
def test_incompatible_saved_projection_is_rejected(sources, mutation):
    import json

    module, source, prior = sources
    if mutation in {"core", "snapshot"}:
        module.EXPECTED[mutation] = "0" * 64
    elif mutation == "source_hash":
        (module.CURRENT / "formal.py").write_bytes(b"changed")
    else:
        value = json.loads((prior / "case-result.json").read_bytes())
        value["files"].pop("report/context.json")
        value.pop("result_sha256")
        value["result_sha256"] = sha256_hex(canonical_json_bytes(value))
        put(prior / "case-result.json", value)
    with pytest.raises(ValueError, match="source_projection"):
        module.verify_sources(module.Inventory())


def test_output_cannot_overlap_sources_or_overwrite_different_artifacts(sources, tmp_path):
    module, source, prior = sources
    for output in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="overlap"):
            module.output_location(output)
    output = module.output_location(tmp_path / "offline-preflight")
    module.write_once(output / "context.json", b"saved")
    module.write_once(output / "context.json", b"saved")
    with pytest.raises(ValueError, match="differs"):
        module.write_once(output / "context.json", b"changed")
    assert (output / "context.json").read_bytes() == b"saved"


def test_missing_source_does_not_create_report_or_send(sources, tmp_path):
    module, source, prior = sources
    module.EXPECTED["core"] = "0" * 64
    destination = tmp_path / "offline-preflight"
    assert module.main(["--output-root", str(destination)]) == 2
    assert not (destination / "report").exists()
    assert not (destination / "context.json").exists()


def test_output_rejects_symlink_before_resolving(sources, tmp_path):
    module, _, _ = sources
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.output_location(link / "offline-preflight")
