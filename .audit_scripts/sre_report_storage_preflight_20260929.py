"""Saved-source offline preflight only. No credentials, completion, or run migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import _unique_object
from nurse_rcl.v2.live.normal_queries import _read_bytes
from nurse_rcl.v2.live.ops_agent_state import load_conversation
from nurse_rcl.v2.live.ops_report import _fact_identity
from nurse_rcl.v2.live.ops_sre_core_import import CORE_MEMBERS
from nurse_rcl.v2.live.ops_sre_ledger import OpsSreLedgerState
from nurse_rcl.v2.live.ops_sre_report_bounds import validate_report_parts
from nurse_rcl.v2.live.ops_sre_report_context import assemble_sre_report_context
from nurse_rcl.v2.live.ops_sre_report_storage import (
    decode_sre_report_storage,
    encode_sre_report_storage,
    read_legacy_sre_report_context,
)
from nurse_rcl.v2.live.sources import file_sha256
from nurse_rcl.v2.live.staged_case import require_plain_path

# The release includes this helper only for explicitly configured synthetic
# regression fixtures. No historical case or owner is a runnable default.
SOURCE = Path("/__nurse_release_unconfigured_source__")
PRIOR = Path("/__nurse_release_unconfigured_prior__")
CURRENT = Path(__file__).resolve().parents[1]
EXPECTED = {}
PROJECTION_FILES = tuple("src/nurse_rcl/" + name for name in (
    "artifacts.py", "schemas.py", "v2/exchange_json.py", "v2/report_evidence_schemas.py",
    "v2/live/ops_report_context.py", "v2/live/ops_report_schema.py", "v2/live/ops_result.py",
    "v2/live/normal_artifacts.py", "v2/live/normal_report.py", "v2/live/normal_report_context.py",
    "v2/live/repair_report_projection.py", "v2/live/repair_sources.py",
))
HISTORICAL_CONTRACT_ONLY = {
    "src/nurse_rcl/v2/live/ops_skills/sre-boundary.md",
    "src/nurse_rcl/v2/live/ops_skills/sre-scoring.md",
    "src/nurse_rcl/v2/live/ops_skills/sre-symptoms.md",
    "src/nurse_rcl/v2/live/ops_sre_exchange.py",
    "src/nurse_rcl/v2/live/ops_sre_report.py",
    "src/nurse_rcl/v2/live/ops_sre_report_exchange.py",
}


class Inventory:
    def __init__(self):
        self.files = {}

    def pin(self, path, expected=None):
        path = require_plain_path(path)
        actual = [path.stat().st_size, file_sha256(path)]
        if expected is not None and actual != expected:
            raise ValueError("blocked_source_projection: file identity differs")
        if str(path) in self.files and self.files[str(path)] != actual:
            raise ValueError("blocked_source_projection: source changed during read")
        self.files[str(path)] = actual
        return actual

    def read(self, path, limit=128 * 1024 * 1024):
        pin = self.pin(path)
        raw = _read_bytes(path, limit)
        if [len(raw), sha256_hex(raw)] != pin:
            raise ValueError("blocked_source_projection: concurrent source change")
        return raw

    def json(self, path):
        return json.loads(self.read(path), object_pairs_hook=_unique_object)

    def unchanged(self):
        return all([Path(p).stat().st_size, file_sha256(require_plain_path(Path(p)))] == value
                   for p, value in self.files.items())


def _require(condition):
    if not condition:
        raise ValueError("blocked_source_projection: incompatible or incomplete proof")


def _root(folder, plan):
    expected = folder / "results" / EXPECTED["case"]
    _require(Path(plan["output_root"]) / plan["case"]["opaque_case_id"] == expected)
    return require_plain_path(expected)


def verify_sources(inventory):
    """Read only saved metadata/receipts/code; never load telemetry or run projection."""
    plans = []
    for folder, name in ((SOURCE, "plan"), (PRIOR, "prior_plan")):
        raw = inventory.read(folder / "plan.json")
        _require(sha256_hex(raw) == EXPECTED[name])
        plan = json.loads(raw, object_pairs_hook=_unique_object)
        for member, pin in plan["code_files"].items():
            _require(not Path(member).is_absolute() and ".." not in Path(member).parts)
            _require(inventory.pin(Path(plan["code_root"]) / member)[1] == pin)
        plans.append(plan)
    source, prior = plans
    _require(sha256_hex(canonical_json_bytes(source["code_files"])) == EXPECTED["code"])
    for field in ("case", "environment", "source_core"):
        _require(source[field] == prior[field])
    _require(source["preparation"]["key"] == prior["preparation"]["key"])
    _require(source["profile"]["core_method_id"] == prior["profile"]["core_method_id"])
    changed = {k for k in source["code_files"].keys() | prior["code_files"].keys()
               if source["code_files"].get(k) != prior["code_files"].get(k)}
    _require(changed <= HISTORICAL_CONTRACT_ONLY)
    for member in PROJECTION_FILES:
        _require(member in source["code_files"] and
                 source["code_files"][member] == prior["code_files"].get(member))
        _require(inventory.pin(CURRENT / member)[1] == source["code_files"][member])
    root, previous = _root(SOURCE, source), _root(PRIOR, prior)
    provenance = inventory.json(root / "source-core.json")
    _require(provenance == inventory.json(previous / "source-core.json"))
    _require(provenance["core_result_digest"] == EXPECTED["core"])
    _require(provenance["source"] == source["source_core"])
    receipt = inventory.json(previous / "case-result.json")
    _require(receipt["result_sha256"] == sha256_hex(canonical_json_bytes(
        {k: v for k, v in receipt.items() if k != "result_sha256"})))
    _require(receipt["identity"]["plan_sha256"] == EXPECTED["prior_plan"])
    _require(receipt["identity"]["code_sha256"] == sha256_hex(
        canonical_json_bytes(prior["code_files"])))
    _require(receipt["core_result_digest"] == EXPECTED["core"] and
             receipt["case_snapshot_id"] == EXPECTED["snapshot"] and
             receipt["channels"]["core"] == "completed")
    for member in CORE_MEMBERS:
        _require(member in receipt["files"] and member in provenance["core_members"])
        _require(receipt["files"][member] == provenance["core_members"][member])
        inventory.pin(root / member, receipt["files"][member])
        inventory.pin(previous / member, receipt["files"][member])
    identity = inventory.json(root / "core/identity.json")
    _require(identity["case_snapshot_id"] == EXPECTED["snapshot"] and
             identity["opaque_case_id"] == EXPECTED["case"])
    _require("report/context.json" in receipt["files"])
    context_path = previous / "report/context.json"
    inventory.pin(context_path, receipt["files"]["report/context.json"])
    old = read_legacy_sre_report_context(inventory.read(context_path, 4_194_304))
    _require(old.core_result_digest == EXPECTED["core"] and
             old.evidence.case_snapshot_id == EXPECTED["snapshot"] and
             old.evidence.opaque_case_id == EXPECTED["case"])
    facts = tuple(f for f in old.evidence.facts if f.role == "formal_record")
    _require(len(facts) == EXPECTED["formal_facts"] and
             all(f.origin == "core_evaluation" and f.action_index is None for f in facts))
    refs = {f.fact.fact_ref for f in facts}
    _require(len(refs) == len(facts) and
             all(set(f.fact.related_fact_refs) <= refs for f in facts))
    formal = old.evidence.model_copy(update=dict(facts=facts,
        mandatory_fact_refs=tuple(r for r in old.evidence.mandatory_fact_refs if r in refs),
        investigation=(), investigation_status="disabled", investigation_failure_type=None,
        model_trace_unverified=()))
    return formal, dict(projection_compatible=True, source_plan=EXPECTED["plan"],
        prior_plan=EXPECTED["prior_plan"], formal_facts=len(facts),
        source_context_sha256=old.context_sha256,
        compared_projection_files=list(PROJECTION_FILES),
        historical_contract_only_changes=sorted(changed))


def output_location(path):
    # Inspect the requested spelling before resolve() erases symlink components.
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("offline output symlink")
    path = path.resolve(strict=False)
    for source in (SOURCE, PRIOR, CURRENT / "src"):
        source = source.resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("offline output overlaps source")
    return path


def write_once(path, raw):
    if path.exists() or path.is_symlink():
        if _read_bytes(path, 128 * 1024 * 1024) != raw:
            raise ValueError("offline output differs; refusing overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    require_plain_path(path.parent)
    atomic_write_bytes(path, raw)


def inspect_investigation(inventory):
    from nurse_rcl.v2.hermes_response import parse_response
    from nurse_rcl.v2.live.ark_bridge import read_attempt_ledger
    from nurse_rcl.v2.live.ops_empirical_policy import RESPONSE_LIMITS
    from nurse_rcl.v2.live.ops_provider import usage_summary

    root = SOURCE / "results" / EXPECTED["case"]
    for path in sorted((root / "investigation").glob("*.json")):
        inventory.pin(path)
    for path in sorted((root / "provider").rglob("*.json")):
        inventory.pin(path)
    conversation = load_conversation(root / "investigation/conversation.json",
        snapshot=EXPECTED["snapshot"], code=EXPECTED["code"], empirical=True, sre=True,
        opaque_case_id=EXPECTED["case"])
    ledger = OpsSreLedgerState.model_validate_json(
        inventory.read(root / "investigation/state.json"))
    _require(ledger.code_sha256 == EXPECTED["code"] and
             ledger.case_snapshot_id == EXPECTED["snapshot"] and
             ledger.state_sha256 == sha256_hex(canonical_json_bytes(ledger.model_dump(
                 mode="json", exclude={"state_sha256"}))))
    agent = inventory.json(root / "investigation/result.json")
    _require((agent["decision_count"], agent["action_count"], agent["termination"],
              agent["failure_type"]) == (8, 7, "model_stop", None))
    _require(len(conversation.decisions) == 8 and len(ledger.records) == 7 and
             all(r.status == "completed" for r in ledger.records))
    _require(not list((root / "provider/ark-report").glob("request-*.json")))
    _require(not (root / "report/result.json").exists())
    by_index = {r.index: r for r in conversation.decisions}
    paths = sorted((root / "provider/ark-investigation").glob("request-*.json"))
    _require(len(paths) == 8)
    for path in paths:
        request = read_attempt_ledger(path, empirical=True)
        row = by_index[request.binding.decision_index]
        _require(request.binding.request_sha256 == row.wire_sha256)
        _require(len(request.attempts) == 1 and request.attempts[0].response_json is not None)
        parsed = parse_response(request.attempts[0].response_json.encode(),
                                output_tool_name=None, limits=RESPONSE_LIMITS)
        _require(parsed.output_json.decode() == row.native_response_json)
    usage = usage_summary(root / "provider", execution_sha256=EXPECTED["plan"],
                          empirical=True, sre=True)
    _require((usage["attempts"], usage["reported_input_tokens"],
              usage["reported_output_tokens"], usage["usage_complete"]) == (8, 110515, 2131, True))
    return conversation, ledger, usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = output_location(args.output_root)
    inventory = Inventory()
    started = monotonic()

    def emit(event, stage, **values):
        print(json.dumps(dict(event=event, stage=stage, total_stages=5,
            wall_seconds=monotonic() - started, **values), ensure_ascii=False), flush=True)

    try:
        formal, proof = verify_sources(inventory)
        emit("source_projection_verified", 1, formal_facts=len(formal.facts))
        conversation, ledger, usage = inspect_investigation(inventory)
        emit("saved_investigation_verified", 2, decisions=8, actions=7, report_attempts=0)
        context = assemble_sre_report_context(formal=formal, decisions=conversation.decisions,
            records=ledger.records, investigation_status="completed", termination="model_stop")
        _require(len(context.evidence.facts) == 1100)
        value = context.model_dump(mode="json")
        facts = value["evidence"].pop("facts")
        capacity = validate_report_parts(value, facts, layout="context")
        raw = encode_sre_report_storage(context)
        write_once(output / "context.json", raw)
        restored = decode_sre_report_storage(_read_bytes(output / "context.json", 4_194_304))
        _require([_fact_identity(f) for f in context.evidence.facts] ==
                 [_fact_identity(f) for f in restored.evidence.facts])
        other = restored.model_dump(mode="json")
        other["evidence"].pop("facts")
        _require(canonical_json_bytes(value) == canonical_json_bytes(other))
        emit("shared_storage_verified", 3, **capacity, storage_bytes=len(raw), facts=1100)

        from nurse_rcl.v2.live.ops_contracts import OpsChannelStatus
        from nurse_rcl.v2.live.ops_sre_report_exchange import (
            bind_sre_report,
            sre_report_body,
            sre_report_projection,
        )
        from nurse_rcl.v2.live.ops_sre_report_run import OpsSreReportRun, _check_wire

        references = json.loads(canonical_json_bytes(conversation.reference_state))
        projection = sre_report_projection(restored, references=references)
        if projection.wire_bytes is None:
            raise ValueError("offline_report_projection_" + projection.status)
        wire = projection.wire_bytes
        binding = bind_sre_report(wire)
        long_view = sre_report_body(wire, references=references)["context"]
        state = OpsSreReportRun(code_sha256=EXPECTED["code"],
            core_result_digest=context.core_result_digest, context_sha256=context.context_sha256,
            channels=OpsChannelStatus(core="completed", investigation="completed",
                                      report="pending"),
            reference_state=references, input_wire_json=wire.decode(),
            input_wire_sha256=sha256_hex(wire), provided_fact_refs=projection.provided_fact_refs,
            provided_step_refs=tuple(s["step_ref"] for s in context.steps),
            target_exceeded=projection.target_exceeded,
            expansion_reasons=projection.expansion_reasons)
        # In-memory invariant check only, NOT a runnable new owner receipt.
        _check_wire(state, restored)
        write_once(output / "report-wire.json", wire)
        write_once(output / "report-reference-state.json", canonical_json_bytes(references))
        payload = json.loads(wire)
        body = json.loads(payload["messages"][1]["content"])
        wire_sizes = dict(wire_bytes=len(wire),
            messages_bytes=sum(len(m["content"].encode()) for m in payload["messages"]),
            system_bytes=len(payload["messages"][0]["content"].encode()),
            schema_bytes=len(body["output_schema"].encode()),
            model_context_bytes=len(canonical_json_bytes(body["context"])),
            restored_view_bytes=len(canonical_json_bytes(long_view)))
        emit("production_wire_verified", 4, **wire_sizes,
             provided_facts=len(projection.provided_fact_refs))
        _require(inventory.unchanged())
        result = dict(schema_version="nurse.offline-sre-report-preflight/1",
            source_unchanged=True, real_calls=0, offline_projection_ready=True,
            live_continuation_authorized=False, owner_executed=False,
            source_proof=proof, context_sha256=context.context_sha256, capacity=capacity,
            storage_bytes=len(raw), wire=wire_sizes,
            request_sha256=binding.request_sha256, usage=usage,
            source_reference_sha256=conversation.reference_state["state_sha256"],
            report_reference_sha256=references["state_sha256"], source_files=inventory.files,
            provided_facts=len(projection.provided_fact_refs),
            omitted_facts=len(projection.omitted_fact_refs),
            steps=len(context.steps), record_groups=len(context.record_groups))
        write_once(output / "preflight.json", canonical_json_bytes(result))
        emit("offline_preflight_complete", 5, source_unchanged=True, real_calls=0,
             offline_projection_ready=True, live_continuation_authorized=False)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        emit("offline_preflight_blocked", 0, error_type=type(exc).__name__,
             code=str(exc) if isinstance(exc, ValueError) else "incomplete_source",
             boundary=getattr(exc, "boundary", None), observed=getattr(exc, "observed", None),
             limit=getattr(exc, "limit", None), source_unchanged=inventory.unchanged(),
             real_calls=0,
             offline_projection_ready=False, live_continuation_authorized=False)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
