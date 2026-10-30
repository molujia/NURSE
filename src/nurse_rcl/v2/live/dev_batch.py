"""Fixed Dev membership and native result sealing. No labels or inference calls."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import ArtifactEnvelope, canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ark_bridge import _Ledger
from nurse_rcl.v2.live.normal_b0_case import inspect_b0_result
from nurse_rcl.v2.live.normal_b2_execution import _identity as b2_identity
from nurse_rcl.v2.live.normal_b2_execution import _member, inspect_b2_result
from nurse_rcl.v2.live.normal_case import _identity as b0_identity
from nurse_rcl.v2.live.normal_case import inspect_normal_result
from nurse_rcl.v2.live.normal_execution import B0_SCHEMA, SCHEMA, _internal, _metadata, _pinned
from nurse_rcl.v2.live.repair_driver import publish_json
from nurse_rcl.v2.split_freeze import ApprovedInferenceSplitManifest

SEAL_SCHEMA = "nurse.v2.dev-paired-prediction-seal/1"
ASSIGNED_FAILURE = "nurse.v2.dev-assigned-failure-plan/1"


def save_dev_rows(root: Path, prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One complete case per bounded record; never split telemetry or a ranking."""
    if prefix not in {"predictions", "evaluation"} or not 1 <= len(rows) <= 1740:
        raise ValueError("Dev row scope differs")
    entries = []
    for number, row in enumerate(rows):
        raw = canonical_json_bytes(row)
        _metadata(raw)  # Same bounded per-case metadata contract as readers.
        name = f"{prefix}-{number:04d}.json"
        publish_json(root / name, row)
        entries.append(dict(name=name, sha256=sha256_hex(raw), size_bytes=len(raw)))
    return entries


def read_dev_rows(root: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if type(entries) is not list or not 1 <= len(entries) <= 1740:
        raise ValueError("Dev row index length differs")
    names = []
    rows = []
    for number, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"name", "sha256", "size_bytes"}:
            raise ValueError("Dev row index fields differ")
        name = entry["name"]
        if (
            type(name) is not str
            or re.fullmatch(
                rf"(?:predictions|evaluation)-{number:04d}\.json",
                name,
            )
            is None
            or type(entry["size_bytes"]) is not int
            or not (0 < entry["size_bytes"] <= 4 * 1024 * 1024)
        ):
            raise ValueError("Dev row path/order/size differs")
        raw = _pinned(root / name, entry["sha256"], entry["size_bytes"])
        if len(raw) != entry["size_bytes"]:
            raise ValueError("Dev row size differs")
        names.append(name.split("-")[0])
        rows.append(_metadata(raw))
    if len(set(names)) != 1:
        raise ValueError("Dev row kinds differ")
    return rows


def summarize_ledgers(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    """Count all reserved attempts; missing provider usage is never zero cost."""
    usage = dict(
        known_prompt_tokens=0,
        known_completion_tokens=0,
        known_total_tokens=0,
        known_cached_prompt_tokens=0,
        unknown_attempts=0,
        unknown_cached_attempts=0,
    )
    attempts = 0
    phases: dict[str, int] = {}
    fatal_errors = []
    for ledger in ledgers:
        rows = ledger["attempts"]
        if (
            type(rows) is not list
            or len(rows) > 5
            or any(
                type(r) is not dict or type(r.get("number")) is not int or r["number"] != i
                for i, r in enumerate(rows, 1)
            )
        ):
            raise ValueError("Dev ledger lifetime attempts differ")
        phase = ledger["binding"]["phase"]
        phases[phase] = phases.get(phase, 0) + 1
        attempts += len(rows)
        for row in rows:
            record = row.get("record") or {}
            if record.get("outcome") == "fatal":
                fatal_errors.append(
                    dict(
                        phase=phase,
                        status_code=record.get("status_code"),
                        error_type=record.get("error_type"),
                    )
                )
            raw = row.get("response_json")
            response = json.loads(raw) if raw is not None else {}
            values = response.get("usage") or {}
            complete = True
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = values.get(field)
                if type(value) is int and value >= 0:
                    usage["known_" + field] += value
                else:
                    complete = False
            usage["unknown_attempts"] += not complete
            cached = (values.get("prompt_tokens_details") or {}).get("cached_tokens")
            if type(cached) is int and cached >= 0:
                prompt = values.get("prompt_tokens")
                if type(prompt) is int and cached > prompt:
                    raise ValueError("Dev cached tokens exceed input tokens")
                usage["known_cached_prompt_tokens"] += cached
            else:
                usage["unknown_cached_attempts"] += 1
    return dict(
        logical_requests=len(ledgers),
        attempts=attempts,
        usage=usage,
        phases=phases,
        money=None if ledgers else 0,
        provider_fatal_errors=fatal_errors,
    )


def load_dev_membership(path: Path, pin: str) -> tuple[tuple[str, str], ...]:
    from nurse_rcl.v2.live.dev_sample import SCHEMA as PILOT_SCHEMA
    from nurse_rcl.v2.live.dev_sample import read_pilot_selection

    raw = _pinned(path, pin)
    if _metadata(raw).get("schema_version") == PILOT_SCHEMA:
        selection = read_pilot_selection(path, pin)
        return tuple(("rcabench", opaque) for opaque in selection["opaque_case_ids"])
    split = ApprovedInferenceSplitManifest.model_validate_json(raw)
    if not split.verify():
        raise ValueError("Dev split integrity differs")
    members = tuple(
        (dataset, opaque)
        for dataset in ("rcabench", "aiops25")
        for part in split.partitions
        if part.dataset == dataset and part.partition == "dev"
        for opaque in part.opaque_case_ids
    )
    if (
        not members
        or len(members) > 1740
        or any(re.fullmatch(r"opq_[0-9a-f]{32}", i) is None for _, i in members)
    ):
        raise ValueError("Dev membership is empty or invalid")
    return members


def _reference(reference: dict[str, Any]) -> dict[str, Any]:
    if type(reference) is not dict or set(reference) != {
        "plan_path",
        "plan_sha256",
        "result_path",
        "result_sha256",
    }:
        raise ValueError("Dev native reference fields differ")
    return _metadata(_pinned(Path(reference["plan_path"]), reference["plan_sha256"]))


def _empty(status: str) -> dict[str, Any]:
    return dict(
        status=status,
        candidate_count=0,
        candidate_information="unavailable",
        candidate_service_keys=[],
        ordered_service_keys=[],
        scores=[],
        tie_groups=[],
        interpretability=dict(status="unavailable", full_recompute="not_performed"),
        explanation=dict(status="unavailable", semantic_truth="not_established"),
        cost=dict(logical_requests=None, attempts=None, usage=None, money=None),
        timing=None,
        baseline_fallback=False,
    )


def explanation_summary(report: dict[str, Any]) -> dict[str, Any]:
    presentation = report.get("presentation") or {}
    claims = presentation.get("claims", [])
    blocks = presentation.get("safe_display_blocks", [])
    return dict(
        status=report["status"],
        semantic_truth="not_established",
        exchange_attempted=report.get("exchange_attempted", False),
        displayable_claims=len(blocks),
        model_claims_total=len(claims),
        model_claims_displayable=sum(c["surface_status"] == "model_unverified" for c in claims),
        canonical_replacements=sum(b["surface_status"] == "canonical" for b in blocks),
        mechanically_rejected=sum(c["mechanical_status"] == "rejected" for c in claims),
        language_blocked=sum(
            c["surface_status"] == "blocked" and c["mechanical_status"] == "passed" for c in claims
        ),
    )


def _read_assigned_failure(
    reference: dict[str, Any],
    plan: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    from nurse_rcl.v2.live.repair_timing import DEV_TIMING_POLICY, TimingJournal

    batch = _metadata(_pinned(Path(plan["batch_path"]), plan["batch_sha256"]))
    index = plan.get("index")
    if type(index) is not int or not 1 <= index <= len(batch["rows"]):
        raise ValueError("Dev assigned failure index is outside the fixed batch")
    row = batch["rows"][index - 1]
    if type(row.get("index")) is not int or row["index"] != index:
        raise ValueError("Dev assigned failure index differs from its fixed row")
    root = Path(batch["output_root"]) / f"case-{plan['index']:04d}"
    expected_identity = sha256_hex(
        canonical_json_bytes(
            [
                plan["batch_sha256"],
                plan["index"],
                row["dataset"],
                row["opaque_case_id"],
            ]
        )
    )
    if (
        plan.get("branch") != branch
        or plan.get("output_root") != str(root)
        or any(plan["case"][k] != row[k] for k in ("dataset", "opaque_case_id"))
        or row["imported_pair"] is not None
    ):
        raise ValueError("Dev preparation failure differs from assigned case")
    if reference["result_path"] != str(root / (branch.lower() + "-failure.json")):
        raise ValueError("Dev assigned failure path differs")
    result = _metadata(_pinned(Path(reference["result_path"]), reference["result_sha256"]))
    first = _metadata((root / "timing/event-00000000.json").read_bytes())
    expired = TimingJournal._inspect_expired(
        root / "timing",
        identity=expected_identity,
        boot_id=first["boot_id"],
        expected_policy=DEV_TIMING_POLICY,
    )
    if result != dict(
        schema_version="nurse.v2.dev-assigned-failure/1",
        status="failed",
        plan_sha256=reference["plan_sha256"],
        reason="preparation_timeout",
        timing=expired,
    ):
        raise ValueError("Dev assigned failure lacks exhausted shared preparation evidence")
    view = _empty("failed")
    # This clock covers the whole paired case, not this failed branch alone.
    # Its verified totals remain in the pair's execution record.
    view.update(failure_stage="preparation", error_type="CaseBudgetExpired")
    if branch == "B0":
        view["cost"] = dict(summarize_ledgers([]), scope="B0_only")
    return view


def read_branch(
    reference: dict[str, Any],
    *,
    case: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    """Verify native sealed bytes and retain the complete ordered candidate universe."""
    plan = _reference(reference)
    if type(plan.get("case")) is not dict or any(plan["case"].get(k) != v for k, v in case.items()):
        raise ValueError("Dev native result has a different complete case")
    schema = plan.get("schema_version")
    if schema == ASSIGNED_FAILURE:
        return _read_assigned_failure(reference, plan, branch)
    if branch == "B0" and schema in (B0_SCHEMA, SCHEMA):
        identity = b0_identity(_internal(plan), reference["plan_sha256"])
        inspect = inspect_b0_result if schema == B0_SCHEMA else inspect_normal_result
        failure_schema = (
            "nurse.v2.normal-b0-failure/1"
            if schema == B0_SCHEMA
            else "nurse.v2.normal-rcl-failure/1"
        )
    elif branch == "B2" and schema == "nurse.v2.normal-b2-single-plan/1":
        identity = b2_identity(plan, reference["plan_sha256"])
        inspect = inspect_b2_result
        failure_schema = "nurse.v2.normal-b2-failure/1"
    else:
        raise ValueError("Dev native branch/schema differs")
    root = Path(plan["output_root"]) / plan["case"]["opaque_case_id"]
    result_path = Path(reference["result_path"])
    result = _metadata(_pinned(result_path, reference["result_sha256"]))
    if result.get("status") == "failed":
        if (
            result_path != root / "failure.json"
            or result.get("identity") != identity
            or (result.get("schema_version") != failure_schema or (root / "result.json").exists())
        ):
            raise ValueError("Dev failed result lacks its own bound failure")
        failed = {
            **_empty("failed"),
            "failure_stage": result.get("stage"),
            "error_type": result.get("error_type"),
        }
        if branch == "B0" and schema == B0_SCHEMA:
            failed["cost"] = dict(summarize_ledgers([]), scope="B0_only")
        return failed
    if (root / "failure.json").exists():
        raise ValueError("Dev native success conflicts with a retained failure")
    if result_path != root / "result.json" or inspect(root, identity) != result:
        raise ValueError("Dev completion differs from native receipt")
    entries = {entry["name"]: entry for entry in result["files"]}
    ranking = _metadata(_member(root, entries["ranking.json"]))
    ordered, candidates = ranking["ordered_service_keys"], ranking["candidate_service_keys"]
    if (
        len(ordered) != result["candidate_count"]
        or len(set(ordered)) != len(ordered)
        or (
            len(candidates) != len(ordered)
            or set(candidates) != set(ordered)
            or ranking["source_evidence_sha256"] != result["evidence_sha256"]
            or ranking["source_ranking_sha256"] != result["ranking_sha256"]
            or ranking["top_k_service_keys"] != ordered[: len(ranking["top_k_service_keys"])]
            or {r["candidate_service_key"] for r in ranking["scores"]} != set(candidates)
            or len(ranking["scores"]) != len(candidates)
        )
    ):
        raise ValueError("Dev complete ranking/receipt binding differs")
    pure = branch == "B0" and schema == B0_SCHEMA
    explanation: dict[str, Any] = dict(status="not_applicable", semantic_truth="not_established")
    if branch == "B2":
        report = _metadata(_member(root, entries["report.json"]))
        explanation = explanation_summary(report)
    ledgers = []
    for name, entry in entries.items():
        if not re.fullmatch(r"(?:semantic-ark|ark)/request-[0-9a-f]{64}\.json", name):
            continue
        envelope = ArtifactEnvelope.model_validate(_metadata(_member(root, entry)))
        if envelope.kind != "live_ark_request" or not envelope.verify():
            raise ValueError("Dev saved model ledger integrity differs")
        ledger = _Ledger.model_validate(envelope.payload)
        if ledger.binding.opaque_case_id != case["opaque_case_id"]:
            raise ValueError("Dev saved model ledger case differs")
        ledgers.append(ledger.model_dump(mode="json"))
    if (pure and ledgers) or len(ledgers) > (0 if pure else 1 if branch == "B0" else 3):
        raise ValueError("Dev logical request budget differs")
    cost = dict(
        summarize_ledgers(ledgers),
        scope="B0_only" if pure else "legacy_B0_plus_B1" if branch == "B0" else "B2_incremental",
    )
    return dict(
        status=result["status"],
        candidate_count=len(candidates),
        candidate_information="available",
        candidate_service_keys=candidates,
        ordered_service_keys=ordered,
        scores=ranking["scores"],
        tie_groups=ranking["tie_groups"],
        evidence_sha256=result["evidence_sha256"],
        ranking_sha256=result["ranking_sha256"],
        interpretability={
            k: result[k]
            for k in (
                "verification_level",
                "full_recompute",
                "interpretability_status",
            )
        },
        explanation=explanation,
        cost=cost,
        timing=result["timing"],
        baseline_fallback=result.get("baseline_fallback", False),
    )


def _collect(
    split_path: Path,
    split_pin: str,
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    members = load_dev_membership(split_path, split_pin)
    if (
        type(pairs) is not list
        or len(pairs) != len(members)
        or any(
            type(p) is not dict
            or set(p)
            not in (
                {"dataset", "opaque_case_id", "B0", "B2"},
                {"dataset", "opaque_case_id", "B0", "B2", "execution"},
            )
            for p in pairs
        )
    ):
        raise ValueError("Dev paired population is incomplete")
    if tuple((p["dataset"], p["opaque_case_id"]) for p in pairs) != members:
        raise ValueError("Dev paired population order, duplicate or membership differs")
    return [read_pair(pair) for pair in pairs]


def read_pair(pair: dict[str, Any]) -> dict[str, Any]:
    case = {k: pair[k] for k in ("dataset", "opaque_case_id")}
    base_plan = _reference(pair["B0"])
    b0 = read_branch(pair["B0"], case=case, branch="B0")
    if b0["status"] == "failed":
        if pair["B2"] is not None:
            raise ValueError("failed B0 must have an explicit B2 dependency failure")
        b2 = _empty("dependency_failure")
        b2["cost"] = dict(summarize_ledgers([]), scope="B2_not_started")
    else:
        if pair["B2"] is None:
            raise ValueError("successful B0 has no sealed B2 terminal")
        b2_plan = _reference(pair["B2"])
        if b2_plan.get("baseline_result_sha256") != pair["B0"]["result_sha256"] or (
            b2_plan.get("baseline_plan_sha256") != pair["B0"]["plan_sha256"]
        ):
            raise ValueError("Dev B2 is not paired with this exact B0")
        b2 = read_branch(pair["B2"], case=base_plan["case"], branch="B2")
        if b2["status"] != "failed" and (
            b0["candidate_service_keys"] != b2["candidate_service_keys"]
        ):
            raise ValueError("Dev paired candidate universe differs")
    execution = None
    if "execution" in pair:
        ref = pair["execution"]
        if type(ref) is not dict or set(ref) != {"path", "sha256"}:
            raise ValueError("Dev execution reference fields differ")
        saved = _metadata(_pinned(Path(ref["path"]), ref["sha256"]))
        if saved.get("pair") != {k: pair[k] for k in ("dataset", "opaque_case_id", "B0", "B2")}:
            raise ValueError("Dev execution record differs from native pair")
        module = saved["module_timing"]
        table = _metadata(_pinned(Path(module["path"]), module["sha256"]))
        if table["timing"] != saved["timing"]:
            raise ValueError("Dev execution timing table differs")
        execution = dict(
            timing=saved["timing"],
            branch_account_deltas=saved["branch_account_deltas"],
            full_case_method_scope="B0_plus_B2_plus_active_orchestration",
            report_inclusive_seconds=sum(
                r["inclusive_seconds"]
                for r in table.get("modules", [])
                if r["module"] == "report" and r["account"] == "method"
            ),
            module_table_complete=table["complete"],
            module_timing=module,
            nested_module_seconds_are_not_additive=True,
        )
    return dict(**case, B0=b0, B2=b2, execution=execution)


def seal_dev_pairs(
    *,
    split_path: Path,
    split_sha256: str,
    pairs: list[dict[str, Any]],
    output: Path,
) -> str:
    predictions = _collect(split_path, split_sha256, pairs)
    rows = save_dev_rows(output.parent, "predictions", predictions)
    body = dict(
        schema_version=SEAL_SCHEMA,
        split_path=str(split_path),
        split_sha256=split_sha256,
        case_count=len(predictions),
        references=pairs,
        prediction_rows=rows,
    )
    publish_json(output, body)
    return sha256_hex(canonical_json_bytes(body))


def verify_dev_seal(path: Path, pin: str) -> dict[str, Any]:
    seal = _metadata(_pinned(path, pin))
    if (
        set(seal)
        != {
            "schema_version",
            "split_path",
            "split_sha256",
            "case_count",
            "references",
            "prediction_rows",
        }
        or seal["schema_version"] != SEAL_SCHEMA
    ):
        raise ValueError("Dev seal fields differ")
    predictions = _collect(Path(seal["split_path"]), seal["split_sha256"], seal["references"])
    if (
        type(seal["case_count"]) is not int
        or seal["case_count"] != len(predictions)
        or (
            canonical_json_bytes(predictions)
            != canonical_json_bytes(read_dev_rows(path.parent, seal["prediction_rows"]))
        )
    ):
        raise ValueError("Dev sealed predictions differ from native artifacts")
    return dict(seal, predictions=predictions)
