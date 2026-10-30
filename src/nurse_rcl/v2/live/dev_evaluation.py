"""Evaluator-only paired full-ranking metrics; never imported by inference owners."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.evaluation.evaluator import CaseEvaluation, _metrics_from_cases
from nurse_rcl.v2.exposure_registry import append_exposure_events, create_exposure_event
from nurse_rcl.v2.live.assembly import service_key
from nurse_rcl.v2.live.dev_batch import read_dev_rows, save_dev_rows, verify_dev_seal
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.opaque_ids import EvaluatorIdentityMap, load_evaluator_identity_map


def execution_totals(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive I/E and native branch costs, never treating unknowns as zero."""
    output: dict[str, Any] = {}
    for dataset in ("rcabench", "aiops25"):
        strata = {}
        for imported in (True, False):
            rows = [
                row
                for row in predictions
                if row["dataset"] == dataset
                and (row["B0"]["cost"].get("scope") == "legacy_B0_plus_B1") == imported
            ]
            group: dict[str, Any] = dict(case_count=len(rows))
            parents = [r["execution"] for r in rows if r.get("execution") is not None]
            group["whole_case_timing"] = dict(
                scope="shared_parent_including_supply_not_sum_of_nested_modules",
                known_cases=len(parents),
                unknown_cases=len(rows) - len(parents),
                **{
                    "known_" + k: sum(r["timing"][k] for r in parents)
                    for k in ("preparation_seconds", "method_seconds", "end_to_end_seconds")
                },
                report_inclusive_seconds=sum(r["report_inclusive_seconds"] for r in parents),
            )
            for branch in ("B0", "B2"):
                views = [row[branch] for row in rows]
                costs = [view["cost"] for view in views]
                known = [cost for cost in costs if cost.get("usage") is not None]
                unknown = len(costs) - len(known)
                usage_keys = (
                    "known_prompt_tokens",
                    "known_completion_tokens",
                    "known_total_tokens",
                    "known_cached_prompt_tokens",
                    "unknown_attempts",
                    "unknown_cached_attempts",
                )
                cost = dict(
                    known_logical_requests=sum(c["logical_requests"] for c in known),
                    known_attempts=sum(c["attempts"] for c in known),
                    unknown_case_count=unknown,
                    usage={k: sum(c["usage"][k] for c in known) for k in usage_keys},
                    money=0 if all(c.get("money") == 0 for c in costs) else None,
                )
                times = [v["timing"] for v in views if v.get("timing") is not None]
                timing = {
                    "known_" + k: sum(t.get(k, 0) for t in times)
                    for k in ("preparation_seconds", "method_seconds", "elapsed_seconds")
                }
                timing["unknown_case_count"] = len(views) - len(times)
                explanation: dict[str, Any] = {
                    k: sum(v["explanation"].get(k, 0) for v in views)
                    for k in (
                        "model_claims_total",
                        "model_claims_displayable",
                        "canonical_replacements",
                        "mechanically_rejected",
                        "language_blocked",
                        "exchange_attempted",
                    )
                }
                explanation.update(
                    semantic_truth="not_established",
                    not_applicable=sum(
                        v["explanation"].get("status") == "not_applicable" for v in views
                    ),
                    unavailable=sum(v["explanation"].get("status") == "unavailable" for v in views),
                )
                group[branch] = dict(
                    cost=cost,
                    timing=timing,
                    timing_scope="legacy_B0_plus_B1_not_separable"
                    if imported and branch == "B0"
                    else "B0_only"
                    if branch == "B0"
                    else "B2_incremental_not_complete_method",
                    I=dict(
                        boundary_checked=sum(
                            v["interpretability"].get("verification_level") == "boundary_checked"
                            for v in views
                        ),
                        unavailable=sum(
                            v["interpretability"].get("status") == "unavailable" for v in views
                        ),
                        full_recompute="not_performed",
                    ),
                    E=explanation,
                )
            strata["imported_history" if imported else "new_execution"] = group
        output[dataset] = strata
    return output


def _case_metric(case_id: str, view: dict[str, Any], answers: tuple[str, ...]) -> CaseEvaluation:
    status = view.get("status")
    if status not in {
        "completed",
        "completed_partial_report",
        "completed_fallback",
        "failed",
        "dependency_failure",
    }:
        raise ValueError("Dev evaluation requires a terminal status")
    failed = status in {"failed", "dependency_failure"}
    ordered, candidates = view["ordered_service_keys"], view["candidate_service_keys"]
    if (
        type(ordered) is not list
        or type(candidates) is not list
        or any(type(k) is not str for k in (*ordered, *candidates))
        or len(set(ordered)) != len(ordered)
        or len(set(candidates)) != len(candidates)
        or (
            set(ordered) != set(candidates)
            or (failed and (ordered or candidates))
            or view["candidate_information"] != ("unavailable" if failed else "available")
        )
    ):
        raise ValueError("Dev ranking/failure candidate domain differs")
    truth = {service_key(answer) for answer in answers}
    first = next((i for i, key in enumerate(ordered, 1) if key in truth), None)
    body = dict(
        schema_version=1,
        case_id=case_id,
        prediction_status="failed"
        if failed
        else "success"
        if status == "completed"
        else "degraded",
        true_services=tuple(sorted(truth)),
        candidate_services=tuple(candidates),
        predicted_services=tuple(ordered),
        hit_at_1=first is not None and first <= 1,
        hit_at_3=first is not None and first <= 3,
        hit_at_5=first is not None and first <= 5,
        reciprocal_rank=0.0 if first is None else 1 / first,
        candidate_covered=bool(truth.intersection(candidates)),
        failure=failed,
    )
    return CaseEvaluation.model_validate(
        {**body, "audit_hash": sha256_hex(canonical_json_bytes(body))}
    )


def score_pairs(predictions: list[dict[str, Any]], mapping: EvaluatorIdentityMap) -> dict[str, Any]:
    """Pure evaluator calculation, using full ranks and the existing aggregate formulas."""
    if not predictions or not mapping.verify():
        raise ValueError("Dev evaluation needs valid predictions and identity map")
    by_id = {r.opaque_case_id: r for r in mapping.records}
    ids = [r["opaque_case_id"] for r in predictions]
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError("Dev predictions have duplicate or unmapped cases")
    cases: list[dict[str, Any]] = []
    for row in predictions:
        record = by_id[row["opaque_case_id"]]
        if record.dataset != row["dataset"]:
            raise ValueError("Dev prediction dataset differs from private mapping")
        cases.append(
            dict(
                dataset=record.dataset,
                opaque_case_id=record.opaque_case_id,
                original_case_id=record.original_case_id,
                **{
                    branch: _case_metric(
                        record.opaque_case_id, row[branch], record.answer_services
                    ).model_dump(mode="json")
                    for branch in ("B0", "B2")
                },
            )
        )
    datasets = {}
    for dataset in ("rcabench", "aiops25"):
        subset = [r for r in cases if r["dataset"] == dataset]
        if not subset:
            continue
        executions = [r for r in predictions if r["dataset"] == dataset]
        groups = {}
        for branch in ("B0", "B2"):
            metrics = _metrics_from_cases(
                tuple(CaseEvaluation.model_validate(r[branch]) for r in subset)
            ).model_dump(mode="json")
            groups[branch] = dict(
                metrics=metrics,
                failed_count=sum(r[branch]["failure"] for r in subset),
                fallback_count=sum(r[branch]["status"] == "completed_fallback" for r in executions),
                unavailable_candidate_count=sum(
                    r[branch]["candidate_information"] == "unavailable" for r in executions
                ),
            )
        groups["paired_delta"] = {
            name: groups["B2"]["metrics"][name] - groups["B0"]["metrics"][name]
            for name in ("acc_at_1", "acc_at_3", "acc_at_5", "mrr", "candidate_coverage")
        }
        datasets[dataset] = groups
    return dict(
        cases=cases,
        datasets=datasets,
        denominator=len(cases),
        mrr_scope="complete_ordered_ranking",
        candidate_coverage_scope="any_root_in_complete_C",
        failed_candidate_coverage="zero_in_primary_denominator_not_known_absence",
        uncertainty="descriptive_development_results_no_confirmatory_claim",
        explainability_truth="not_established",
        fit_parameters=False,
    )


def evaluate_dev(
    *,
    seal_path: Path,
    seal_sha256: str,
    evaluator_map_path: Path,
    evaluator_map_sha256: str,
    exposure_registry: Path,
    output: Path,
) -> dict[str, Any]:
    """All native predictions are checked before any private label file is opened."""
    seal = verify_dev_seal(seal_path, seal_sha256)
    split = _metadata(_pinned(Path(seal["split_path"]), seal["split_sha256"]))
    if split["identity_artifact_sha256"] != evaluator_map_sha256:
        raise ValueError("Dev evaluator map differs from the approved split")
    identity = dict(seal_sha256=seal_sha256, evaluator_map_sha256=evaluator_map_sha256)
    with (output.parent / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            saved = read_json(output)
            unsigned = {k: v for k, v in saved.items() if k != "result_sha256"}
            if saved.get("identity") != identity or saved.get("result_sha256") != sha256_hex(
                canonical_json_bytes(unsigned)
            ):
                raise ValueError("Dev saved evaluation identity differs")
            read_dev_rows(output.parent, saved["case_details"])
            return saved
        intent_path = output.with_name(output.stem + "-intent.json")
        if intent_path.exists():
            intent = read_json(intent_path)
            if intent.get("identity") != identity:
                raise ValueError("Dev evaluation contact identity differs")
        else:
            intent = dict(identity=identity, contact_at_unix=int(time.time()))
            publish_json(intent_path, intent)
        _pinned(evaluator_map_path, evaluator_map_sha256)
        mapping = load_evaluator_identity_map(evaluator_map_path)
        _pinned(evaluator_map_path, evaluator_map_sha256)
        records = {r.opaque_case_id: r for r in mapping.records}
        events = []
        for row in seal["predictions"]:
            record = records.get(row["opaque_case_id"])
            if record is None or record.dataset != row["dataset"]:
                raise ValueError("Dev case missing from approved private map")
            events.append(
                create_exposure_event(
                    dataset=record.dataset,
                    case_id=record.original_case_id,
                    incident_group_id=record.incident_group_id,
                    contact_at_unix=intent["contact_at_unix"],
                    label_access="confirmed",
                    prediction_access="confirmed",
                    configuration_use="live-dev-paired-sealed-no-fitting",
                    responsible_experiment="dev-" + seal_sha256,
                    exposure_class="development_evaluation",
                    source_artifact_hashes=[seal_sha256, evaluator_map_sha256],
                    source_paths=[str(seal_path), str(evaluator_map_path)],
                    reasons=[
                        "all_assigned_predictions_sealed_before_labels",
                        "no_inference_or_network",
                    ],
                )
            )
        with exposure_registry.with_suffix(".repair.lock").open("a") as registry_lock:
            fcntl.flock(registry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = exposure_registry.read_bytes() if exposure_registry.exists() else b""
            append_exposure_events(exposure_registry, events)
            if not exposure_registry.read_bytes().startswith(before):
                raise ValueError("Dev exposure registry prefix changed")
        scored = score_pairs(seal["predictions"], mapping)
        details = [
            dict(metrics=metric, execution=execution)
            for metric, execution in zip(
                scored.pop("cases"),
                seal["predictions"],
                strict=True,
            )
        ]
        result = dict(
            schema_version="nurse.v2.dev-paired-evaluation/1",
            identity=identity,
            **scored,
            execution_totals=execution_totals(seal["predictions"]),
            exposure_event_ids=[e.event_id for e in events],
            inference_calls=0,
            network_calls=0,
            case_details=save_dev_rows(output.parent, "evaluation", details),
        )
        result["result_sha256"] = sha256_hex(canonical_json_bytes(result))
        publish_json(output, result)
        return result
