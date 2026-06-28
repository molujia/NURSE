"""Nurse CLI runner."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .alarm_service import DEFAULT_DETECTOR_CONFIG, detect_alarm, get_or_create_alarm
from .case_order import select_case_dirs_from_order
from .closure import build_evidence_graph
from .datasets import case_dirs_for_dataset, dataset_choices, is_rcaeval_dataset
from .doc_generator import generate_system_doc
from .context_packager import package_verified_rcl_context
from .evidence_extractor import extract_candidates
from .experiment_modes import experiment_mode_choices, load_experiment_mode, parse_ranker_overrides
from .experiment_stats import write_experiment_statistics
from .labels import load_labels_for_dataset
from .matching import hit_at
from .incident_shape_prior import build_incident_shape_features, score_incident_shape_prior
from .llm_bridge import NurseLLMBridge, llm_candidates_to_evidence
from .llm_payload import AIOPS_RETAINED_BLOCKED_INFRA_PREFIXES, AIOPS_SYMPTOM_FAMILY_TOP3_RESCUE, AIOPS_SYMPTOM_PRONE_ENTRYPOINTS, RCAEVAL_TT_PRIMARY_SERVICES, build_grounded_user_manager_payload, build_rank_repair_payload, rank_repair_limits_for
from .metric_deviation import service_metric_deviation_scores
from .neural_rank_repair import apply_primary_service_slot_rescue, apply_symptom_family_top3_rescue, apply_verified_rank_repair
from .progress import format_progress_line
from .ranker import NurseRanker
from .raw_evidence_compiler import METRIC_SCORE_BACKENDS, service_candidates_from_incident
from .raw_adapters import RawIncident, load_raw_incident
from .schemas import AlarmArtifact, EvidenceCandidate, VerifiedEvidence
from .verifier import verify_candidates


REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = Path(os.environ.get("NURSE_LAB_ROOT", REPO_ROOT.parent)).expanduser().resolve()
ALARM_CACHE_ROOT = Path(os.environ.get("NURSE_ALARM_CACHE_ROOT", REPO_ROOT)).expanduser().resolve()
LLM_CHOICES = ["gpt-5-nano"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Nurse root-cause localization")
    parser.add_argument("--dataset", choices=dataset_choices(), default="rcaeval-tt")
    parser.add_argument("--case-limit", type=int, default=5)
    parser.add_argument("--llm", choices=LLM_CHOICES, default="gpt-5-nano")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "nurse" / "runs"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-doc", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle case order with --seed before applying --case-limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is set.")
    parser.add_argument("--case-manifest", default="", help="Shared case-order manifest JSON.")
    parser.add_argument("--case-ids-file", default="", help="Plain-text case id list, one case per line.")
    parser.add_argument(
        "--ranking-engine",
        choices=["closure_v1", "evidence_program_v2"],
        default="evidence_program_v2",
    )
    parser.add_argument("--optimization-round", default="", help=argparse.SUPPRESS)
    parser.add_argument("--config-out", default=None)
    parser.add_argument(
        "--experiment-mode",
        choices=experiment_mode_choices(),
        default="full",
        help="NURSE release mode. full is the default configuration.",
    )
    parser.add_argument(
        "--ranker-overrides-json",
        default="",
        help="JSON object or path overriding NurseRanker parameters for advanced local tuning.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["conservative", "balanced", "recall-heavy"],
        default="balanced",
        help="Prompt profile label recorded in LLM payloads.",
    )
    parser.add_argument(
        "--metric-score-backend",
        choices=sorted(METRIC_SCORE_BACKENDS),
        default="evidence_robust",
        help="Metric anomaly score backend for Nurse evidence and metric profile scoring.",
    )
    return parser


def _case_dirs(dataset: str) -> List[Path]:
    return case_dirs_for_dataset(dataset, LAB_ROOT)


def _selected_case_dirs(
    dataset: str,
    case_limit: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    case_manifest: str | Path | None = None,
    case_ids_file: str | Path | None = None,
) -> List[Path]:
    return select_case_dirs_from_order(
        dataset,
        lab_root=LAB_ROOT,
        manifest=case_manifest or None,
        case_ids_file=case_ids_file or None,
        case_limit=case_limit,
        seed=seed,
        shuffle=shuffle,
    )


def _first_alarm(incident: RawIncident) -> Optional[AlarmArtifact]:
    return detect_alarm(incident)


def _prediction_label_from_rcaeval_case_id(case_id: str) -> List[str]:
    parts = case_id.split("_")
    if len(parts) >= 3:
        return [parts[1]]
    return []


def _labels_for_case(dataset: str, case_id: str) -> List[str]:
    if is_rcaeval_dataset(dataset):
        return _prediction_label_from_rcaeval_case_id(case_id)
    labels = load_labels_for_dataset(dataset, case_id, LAB_ROOT)
    if labels:
        return labels
    return []


def _hit_at(ranked: List[str], labels: List[str], k: int) -> bool:
    return hit_at(ranked, labels, k)


def _metrics(predictions: Dict[str, List[str]], labels: Dict[str, List[str]], ranked_cases: int) -> Dict[str, float]:
    case_ids = sorted(labels)
    total = max(len(case_ids), 1)
    return {
        "acc_at_1": sum(_hit_at(predictions.get(cid, []), labels[cid], 1) for cid in case_ids) / total,
        "acc_at_3": sum(_hit_at(predictions.get(cid, []), labels[cid], 3) for cid in case_ids) / total,
        "acc_at_5": sum(_hit_at(predictions.get(cid, []), labels[cid], 5) for cid in case_ids) / total,
        "labeled_cases": float(len(case_ids)),
        "ranked_cases": float(ranked_cases),
    }


def _usage_tokens(usage: Dict) -> int:
    if not isinstance(usage, dict):
        return 0
    if "total_tokens" in usage:
        return int(usage.get("total_tokens", 0) or 0)
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(input_tokens or 0) + int(output_tokens or 0)


def _load_json_dict(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_release_llm_configuration(llm: str, llm_bridge: NurseLLMBridge, *, allow_no_llm: bool = False) -> None:
    if str(llm or "").strip().lower() == "none":
        if allow_no_llm:
            return
        raise ValueError("Nurse release requires an LLM; use --llm gpt-5-nano.")
    if not llm_bridge.has_credentials(llm):
        env_key = llm_bridge.required_env_key(llm)
        raise RuntimeError(
            f"Missing {env_key}. Configure an LLM key before running Nurse. "
            "For the release quick start, set OPENAI_API_KEY and run --llm gpt-5-nano."
        )

def _load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dedupe_records(records: List[Dict]) -> List[Dict]:
    latest_by_case: Dict[str, Dict] = {}
    for record in records:
        case_id = str(record.get("case_id") or "")
        if not case_id:
            continue
        if case_id in latest_by_case:
            latest_by_case.pop(case_id)
        latest_by_case[case_id] = record
    return list(latest_by_case.values())


def _is_resumable_completion(record: Dict, llm: str) -> bool:
    status = record.get("status")
    if status == "missed_alarm":
        return True
    if status != "ranked":
        return False
    if llm == "none":
        return True
    return record.get("llm_status") == "ok"


def _timing_payload(*, total_seconds: float, llm_seconds: float) -> Dict[str, float]:
    total = round(max(float(total_seconds), 0.0), 6)
    llm_time = round(max(float(llm_seconds), 0.0), 6)
    non_llm = round(max(total - llm_time, 0.0), 6)
    return {
        "total_seconds": total,
        "llm_seconds": llm_time,
        "non_llm_seconds": non_llm,
    }


def _llm_payload(*, llm: str, llm_result: Dict) -> Dict:
    usage = llm_result.get("usage") if isinstance(llm_result.get("usage"), dict) else {}
    status = str(llm_result.get("status") or "")
    enabled = llm != "none" and status not in {"disabled", "not_called"}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = _usage_tokens(usage)
    return {
        "enabled": enabled,
        "status": status,
        "model": str(llm_result.get("model") or (llm if llm != "none" else "")),
        "calls": 1 if enabled and status not in {"skipped"} else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _evidence_summary(evidence: VerifiedEvidence) -> Dict:
    claim = evidence.claim if isinstance(evidence.claim, dict) else {}
    observed = evidence.observed if isinstance(evidence.observed, dict) else {}
    return {
        "evidence_id": evidence.evidence_id,
        "entity": evidence.entity,
        "signal": evidence.signal,
        "kind": evidence.kind,
        "metric_role": claim.get("metric_role") or "",
        "weight": claim.get("evidence_program_weight")
        or claim.get("anomaly_score")
        or observed.get("matching_points")
        or observed.get("matching_rows")
        or 1.0,
        "verification_rule": evidence.verification_rule,
        "observed": {
            key: observed.get(key)
            for key in ("observed", "baseline", "matching_points", "matching_rows", "max_duration")
            if key in observed
        },
        "source": evidence.provenance.get("source", ""),
    }


def _llm_proposal_payload(
    *,
    case_id: str,
    alarm: AlarmArtifact,
    service_candidates: List[str],
    verified: List[VerifiedEvidence],
    current_ranking: List[str],
    topology_edges: Iterable[tuple[str, str]],
    prompt_profile: str = "balanced",
) -> Dict:
    evidence = [
        _evidence_summary(item)
        for item in verified
        if item.verified and item.kind != "topology"
    ]
    evidence.sort(
        key=lambda item: (
            0 if item.get("metric_role") == "source_metric" else 1,
            str(item.get("entity") or ""),
            str(item.get("signal") or ""),
        )
    )
    return {
        "incident_ref": "current-incident",
        "alarm": alarm.to_dict(),
        "service_candidates": list(service_candidates)[:80],
        "verified_evidence": evidence[:40],
        "current_ranking": list(current_ranking)[:20],
        "prompt_profile": prompt_profile,
        "topology_edges": [
            {"caller": caller, "callee": callee}
            for caller, callee in list(topology_edges)[:80]
        ],
        "instructions": {
            "rank_repair": "Preferred: compact bounded reorder proposals with promote, to_rank, supporting_evidence_ids, confidence, reason.",
            "candidates": "Optional raw telemetry checks with entity/service, signal/metric, kind, confidence, reason.",
            "hypotheses": "Optional root hypotheses using service plus supporting_evidence_ids from verified_evidence.",
            "candidate_rules": "Optional bounded SOP rules; they will be audited and cannot directly rank roots.",
        },
    }


def _unsafe_verified_llm_evidence(candidate: EvidenceCandidate) -> VerifiedEvidence:
    provenance = dict(candidate.provenance)
    provenance["candidate_source"] = "llm_proposal"
    return VerifiedEvidence(
        evidence_id=candidate.evidence_id,
        case_id=candidate.case_id,
        entity=candidate.entity,
        signal=candidate.signal,
        kind=candidate.kind,
        time_range=dict(candidate.time_range),
        claim=dict(candidate.claim),
        provenance=provenance,
        observed={"unsafe": True, "source": "llm_candidate_without_verification"},
        verification_rule="llm_guardrail_disabled",
        verified=True,
        rejection_reason="",
    )


def _append_record(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_run_state(
    *,
    run_dir: Path,
    dataset: str,
    case_limit: int,
    llm: str,
    records: List[Dict],
    predictions: Dict[str, List[str]],
    labels: Dict[str, List[str]],
    start_run: float,
    token_usage: int,
) -> None:
    records = _dedupe_records(records)
    run_metrics = _metrics(predictions, labels, sum(1 for record in records if record.get("status") == "ranked"))
    llm_status_counts = Counter(str(record.get("llm_status", "")) for record in records if record.get("llm_status"))
    summary = {
        "dataset": dataset,
        "case_limit": case_limit,
        "llm": llm,
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - start_run, 3),
        "token_usage": token_usage,
        "metrics": run_metrics,
        "nurse_top1": run_metrics["acc_at_1"],
        "llm_status_counts": dict(sorted(llm_status_counts.items())),
    }
    (run_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (run_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed_cases": [record.get("case_id") for record in records],
                "completed_count": len(records),
                "ranked_count": sum(1 for record in records if record.get("status") == "ranked"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_nurse(
    dataset: str,
    case_limit: int,
    llm: str,
    output_root: str | Path,
    *,
    generate_doc: bool = False,
    run_dir: str | Path | None = None,
    resume: bool = False,
    progress: bool = True,
    ranking_engine: str = "evidence_program_v2",
    optimization_round: str = "",
    config_out: str | Path | None = None,
    experiment_mode: str = "full",
    ranker_overrides_json: str = "",
    prompt_profile: str = "balanced",
    shuffle: bool = False,
    seed: int = 0,
    case_manifest: str | Path | None = None,
    case_ids_file: str | Path | None = None,
    allow_no_llm: bool = False,
    metric_score_backend: str = "evidence_robust",
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_dir) if run_dir else Path(output_root) / f"{dataset}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "graphs").mkdir(exist_ok=True)
    (run_dir / "alarms").mkdir(exist_ok=True)
    (run_dir / "llm_artifacts").mkdir(exist_ok=True)
    (run_dir / "contexts").mkdir(exist_ok=True)

    predictions: Dict[str, List[str]] = _load_json_dict(run_dir / "predictions.json") if resume else {}
    labels: Dict[str, List[str]] = _load_json_dict(run_dir / "labels.json") if resume else {}
    records: List[Dict] = _dedupe_records(_load_records(run_dir / "records.jsonl")) if resume else []
    if resume:
        records = [record for record in records if _is_resumable_completion(record, llm)]
        resumable_cases = {str(record.get("case_id")) for record in records}
        predictions = {case_id: value for case_id, value in predictions.items() if case_id in resumable_cases}
        labels = {case_id: value for case_id, value in labels.items() if case_id in resumable_cases}
    completed_cases = {str(record.get("case_id")) for record in records}
    llm_bridge = NurseLLMBridge()
    _require_release_llm_configuration(llm, llm_bridge, allow_no_llm=allow_no_llm)
    mode = load_experiment_mode(experiment_mode)
    ranker_overrides = parse_ranker_overrides(ranker_overrides_json)
    ranker_kwargs: Dict[str, object] = {}
    if ranking_engine == "closure_v1":
        ranker_kwargs.update({"evidence_program_weight": 0.0, "counterfactual_weight": 0.0})
    ranker_kwargs.update(mode.ranker_kwargs(ranker_overrides))
    ranker = NurseRanker(**ranker_kwargs)
    detector_config = dict(DEFAULT_DETECTOR_CONFIG)
    config = {
        "method": "Nurse",
        "dataset": dataset,
        "case_limit": case_limit,
        "llm": llm,
        "experiment_mode": mode.name,
        "experiment_mode_description": mode.description,
        "prompt_profile": prompt_profile,
        "metric_score_backend": metric_score_backend,
        "unsafe_experiment_mode": mode.unsafe,
        "ranking_engine": ranking_engine,
        "optimization_round": optimization_round,
        "detector": detector_config,
        "case_order": {
            "shuffle": bool(shuffle),
            "seed": int(seed),
            "case_manifest": str(case_manifest or ""),
            "case_ids_file": str(case_ids_file or ""),
        },
        "ranker": {
            "source_weight": ranker.source_weight,
            "generic_weight": ranker.generic_weight,
            "symptom_weight": ranker.symptom_weight,
            "support_bonus": ranker.support_bonus,
            "propagation_bonus": ranker.propagation_bonus,
            "symptom_penalty": ranker.symptom_penalty,
            "evidence_program_weight": ranker.evidence_program_weight,
            "counterfactual_weight": ranker.counterfactual_weight,
            "metric_profile_enabled": ranker.metric_profile_enabled,
            "metric_tier_size": ranker.metric_tier_size,
            "metric_closure_prior_weight": ranker.metric_closure_prior_weight,
            "metric_deviation_consensus_weight": ranker.metric_deviation_consensus_weight,
            "metric_deviation_magnitude_weight": ranker.metric_deviation_magnitude_weight,
            "metric_rank_prior_weight": ranker.metric_rank_prior_weight,
            "service_role_prior_weight": ranker.service_role_prior_weight,
            "llm_semantic_prior_weight": ranker.llm_semantic_prior_weight,
            "llm_hypothesis_confidence_scale": ranker.llm_hypothesis_confidence_scale,
            "llm_hypothesis_max_weight": ranker.llm_hypothesis_max_weight,
            "incident_shape_prior_weight": ranker.incident_shape_prior_weight,
            "incident_shape_counterfactual_weight": ranker.incident_shape_counterfactual_weight,
            "incident_shape_topology_weight": ranker.incident_shape_topology_weight,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if config_out:
        config_out_path = Path(config_out)
        config_out_path.parent.mkdir(parents=True, exist_ok=True)
        config_out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    start_run = time.perf_counter()
    previous_summary = _load_json_dict(run_dir / "summary.json") if resume else {}
    token_usage = int(previous_summary.get("token_usage", 0) or 0)
    case_dirs = _selected_case_dirs(
        dataset,
        case_limit,
        shuffle=shuffle,
        seed=seed,
        case_manifest=case_manifest,
        case_ids_file=case_ids_file,
    )
    total_cases = len(case_dirs)
    for index, case_dir in enumerate(case_dirs, start=1):
        case_started = time.perf_counter()
        llm_seconds = 0.0
        incident = load_raw_incident(case_dir, dataset=dataset)
        if resume and incident.case_id in completed_cases:
            if progress:
                print(
                    format_progress_line(
                        dataset=dataset,
                        llm=llm,
                        index=index,
                        total=total_cases,
                        case_id=incident.case_id,
                        status="resume-skip",
                    ),
                    flush=True,
                )
            continue
        labels_for_case = _labels_for_case(dataset, incident.case_id)
        if labels_for_case:
            labels[incident.case_id] = labels_for_case
        alarm = get_or_create_alarm(incident, repo_root=ALARM_CACHE_ROOT, write_cache=True)
        if alarm is None:
            total_seconds = time.perf_counter() - case_started
            llm_result = {"status": "not_called", "model": llm, "usage": {}}
            record = {
                "case_id": incident.case_id,
                "status": "missed_alarm",
                "llm_status": llm_result["status"],
                "timing": _timing_payload(total_seconds=total_seconds, llm_seconds=llm_seconds),
                "llm": _llm_payload(llm=llm, llm_result=llm_result),
            }
            records.append(record)
            _append_record(run_dir / "records.jsonl", record)
            _write_run_state(
                run_dir=run_dir,
                dataset=dataset,
                case_limit=case_limit,
                llm=llm,
                records=records,
                predictions=predictions,
                labels=labels,
                start_run=start_run,
                token_usage=token_usage,
            )
            if progress:
                print(
                    format_progress_line(
                        dataset=dataset,
                        llm=llm,
                        index=index,
                        total=total_cases,
                        case_id=incident.case_id,
                        status="missed_alarm",
                    ),
                    flush=True,
                )
            continue
        service_candidates = service_candidates_from_incident(incident, alarm)
        metric_deviation_by_metric = service_metric_deviation_scores(
            incident,
            alarm,
            service_candidates=service_candidates,
            score_backend=metric_score_backend,
        )
        metric_deviation_scores = {
            service: max(metric_scores.values())
            for service, metric_scores in metric_deviation_by_metric.items()
            if metric_scores
        }
        incident_shape_features = build_incident_shape_features(
            metric_deviation_by_metric,
            incident.topology_edges,
            service_candidates,
            dataset,
        )
        incident_shape_prior_scores = score_incident_shape_prior(incident_shape_features)
        candidates = extract_candidates(incident, alarm, metric_score_backend=metric_score_backend)
        verified = verify_candidates(incident, candidates)
        preliminary_graph = build_evidence_graph(
            case_id=incident.case_id,
            evidences=verified,
            topology_edges=incident.topology_edges,
            alert_entity=alarm.alert_entity,
        )
        preliminary_graph.metadata["dataset"] = dataset
        preliminary_graph.metadata["service_candidates"] = service_candidates
        preliminary_graph.metadata["metric_deviation_scores"] = metric_deviation_scores
        preliminary_graph.metadata["metric_deviation_by_metric"] = metric_deviation_by_metric
        preliminary_graph.metadata["incident_shape_features"] = incident_shape_features
        preliminary_graph.metadata["incident_shape_prior_scores"] = incident_shape_prior_scores
        preliminary_prediction = ranker.rank(preliminary_graph)
        bounded_llm_rank_effects = bool(
            mode.use_llm_candidates or mode.use_llm_hypotheses or mode.use_llm_rank_repair
        )
        if llm != "none" and mode.call_llm and bounded_llm_rank_effects:
            llm_started = time.perf_counter()
            llm_result = llm_bridge.propose_evidence(
                task="nurse_rank_repair" if mode.use_llm_rank_repair else "nurse_evidence_proposal",
                payload=build_rank_repair_payload(
                    alarm=alarm,
                    service_candidates=service_candidates,
                    prediction=preliminary_prediction,
                    verified_evidence=verified,
                    topology_edges=incident.topology_edges,
                    prompt_profile=prompt_profile,
                ),
                model=llm,
            )
            llm_seconds += time.perf_counter() - llm_started
            raw_llm_proposal = llm_result.get("proposal") or {
                "rank_repair": llm_result.get("rank_repair", []),
                "hypotheses": llm_result.get("hypotheses", []),
                "candidate_rules": llm_result.get("candidate_rules", []),
                "candidates": llm_result.get("candidates", []),
            }
            sanitized_llm_proposal = mode.sanitize_proposal(raw_llm_proposal)
            (run_dir / "llm_artifacts" / f"{incident.case_id}.json").write_text(
                json.dumps(
                    {
                        "case_id": incident.case_id,
                        "llm": llm,
                        "experiment_mode": mode.name,
                        "prompt_profile": prompt_profile,
                        "metric_score_backend": metric_score_backend,
                        "llm_role": "bounded_rank_repair_or_evidence_proposal",
                        "result": llm_result,
                        "sanitized_proposal": sanitized_llm_proposal,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            llm_candidates = llm_candidates_to_evidence(
                case_id=incident.case_id,
                raw_candidates=llm_result.get("candidates", []) if mode.use_llm_candidates else [],
                time_range=alarm.window,
                model=llm,
                service_candidates=service_candidates,
            )
            if mode.verify_llm_candidates:
                verified.extend(verify_candidates(incident, llm_candidates))
            else:
                verified.extend(_unsafe_verified_llm_evidence(candidate) for candidate in llm_candidates)
            token_usage += _usage_tokens(llm_result.get("usage", {}))
        else:
            llm_result = {"status": "disabled", "candidates": []}
            raw_llm_proposal = {}
            sanitized_llm_proposal = {}
        verified_count = sum(1 for item in verified if item.verified)
        rejected_count = len(verified) - verified_count
        graph = build_evidence_graph(
            case_id=incident.case_id,
            evidences=verified,
            topology_edges=incident.topology_edges,
            alert_entity=alarm.alert_entity,
        )
        graph.metadata["dataset"] = dataset
        graph.metadata["service_candidates"] = service_candidates
        graph.metadata["metric_deviation_scores"] = metric_deviation_scores
        graph.metadata["metric_deviation_by_metric"] = metric_deviation_by_metric
        graph.metadata["incident_shape_features"] = incident_shape_features
        graph.metadata["incident_shape_prior_scores"] = incident_shape_prior_scores
        graph.metadata["experiment_mode"] = mode.name
        graph.metadata["prompt_profile"] = prompt_profile
        if llm != "none" and sanitized_llm_proposal:
            graph.metadata["llm_proposal"] = sanitized_llm_proposal
            graph.metadata["llm_status"] = llm_result.get("status")
            graph.metadata["llm_raw_candidate_count"] = len(raw_llm_proposal.get("candidates", [])) if isinstance(raw_llm_proposal, dict) else 0
            graph.metadata["llm_candidate_count_used"] = len(sanitized_llm_proposal.get("candidates", [])) if isinstance(sanitized_llm_proposal, dict) else 0
            graph.metadata["llm_rank_repair_count"] = len(sanitized_llm_proposal.get("rank_repair", [])) if isinstance(sanitized_llm_proposal, dict) else 0
        prediction = ranker.rank(graph)
        rank_repair_result = None
        rank_repair_limits = rank_repair_limits_for(dataset, prompt_profile)
        if llm != "none" and mode.use_llm_rank_repair and isinstance(sanitized_llm_proposal, dict) and sanitized_llm_proposal.get("rank_repair"):
            rank_repair_result = apply_verified_rank_repair(
                original_ranking=prediction.ranked,
                proposals=sanitized_llm_proposal.get("rank_repair", []),
                verified_evidence_ids=[item.evidence_id for item in verified if item.verified],
                repair_top_k=rank_repair_limits["repair_top_k"],
                max_repairs=rank_repair_limits["max_repairs"],
                blocked_prefixes=(
                    AIOPS_RETAINED_BLOCKED_INFRA_PREFIXES
                    if dataset == "aiops25-retained-nonnode"
                    else ()
                ),
                blocked_services=(
                    AIOPS_SYMPTOM_PRONE_ENTRYPOINTS
                    if dataset == "aiops25-retained-nonnode"
                    else ()
                ),
                scores=prediction.scores,
                score_breakdown=prediction.score_breakdown,
                max_score_gap_from_top=None if dataset == "aiops25-retained-nonnode" else 0.3,
                score_gap_bypass_services=(RCAEVAL_TT_PRIMARY_SERVICES if dataset == "rcaeval-tt" else ()),
                score_gap_bypass_protected_top1_services=(RCAEVAL_TT_PRIMARY_SERVICES if dataset == "rcaeval-tt" else ()),
                score_gap_bypass_min_target_rank=2,
                min_source_evidence_for_top1_repair=(1.0 if dataset == "aiops25-retained-nonnode" else None),
                reject_symptom_penalty_at_or_below=(-1.0 if dataset == "aiops25-retained-nonnode" else None),
            )
            if rank_repair_result.verified_repairs:
                prediction.ranked = rank_repair_result.ranking
            prediction.metadata["rank_repair"] = {
                "verified_repairs": rank_repair_result.verified_repairs,
                "rejected_repairs": rank_repair_result.rejected_repairs,
            }
        if llm != "none" and mode.use_llm_rank_repair and dataset == "aiops25-retained-nonnode":
            symptom_rescue_result = apply_symptom_family_top3_rescue(
                prediction.ranked,
                symptom_family=AIOPS_SYMPTOM_FAMILY_TOP3_RESCUE,
                blocked_prefixes=AIOPS_RETAINED_BLOCKED_INFRA_PREFIXES,
                top_k=3,
            )
            if symptom_rescue_result.applied:
                prediction.ranked = symptom_rescue_result.ranking
            prediction.metadata["symptom_family_top3_rescue"] = {
                "applied": symptom_rescue_result.applied,
                "promoted_service": symptom_rescue_result.promoted_service,
                "target_rank": symptom_rescue_result.target_rank,
                "reason": symptom_rescue_result.reason,
            }
        if llm != "none" and mode.use_llm_rank_repair and dataset == "rcaeval-tt":
            slot_rescue_result = apply_primary_service_slot_rescue(
                prediction.ranked,
                primary_services=RCAEVAL_TT_PRIMARY_SERVICES,
                top_k=5,
                search_k=10,
            )
            if slot_rescue_result.applied:
                prediction.ranked = slot_rescue_result.ranking
            prediction.metadata["semantic_slot_rescue"] = {
                "applied": slot_rescue_result.applied,
                "promoted_service": slot_rescue_result.promoted_service,
                "target_rank": slot_rescue_result.target_rank,
                "reason": slot_rescue_result.reason,
            }

        verified_context = package_verified_rcl_context(
            graph=graph,
            prediction=prediction,
            dataset=dataset,
            llm_advisory=sanitized_llm_proposal,
            rejected_items=[
                {
                    "evidence_id": item.evidence_id,
                    "entity": item.entity,
                    "reason": item.rejection_reason,
                }
                for item in verified
                if not item.verified
            ],
        )
        (run_dir / "contexts" / f"{incident.case_id}.json").write_text(
            json.dumps(verified_context.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if llm != "none" and mode.call_llm and not bounded_llm_rank_effects:
            llm_started = time.perf_counter()
            user_manager_payload = build_grounded_user_manager_payload(context=verified_context)
            llm_result = llm_bridge.propose_evidence(
                task="nurse_user_manager",
                payload=user_manager_payload,
                model=llm,
            )
            llm_seconds += time.perf_counter() - llm_started
            raw_llm_proposal = llm_result.get("proposal") if isinstance(llm_result.get("proposal"), dict) else {}
            token_usage += _usage_tokens(llm_result.get("usage", {}))
            (run_dir / "llm_artifacts" / f"{incident.case_id}.json").write_text(
                json.dumps(
                    {
                        "case_id": incident.case_id,
                        "llm": llm,
                        "experiment_mode": mode.name,
                        "prompt_profile": prompt_profile,
                        "metric_score_backend": metric_score_backend,
                        "llm_role": "user_manager_grounded_explanation",
                        "payload_contract": user_manager_payload.get("output_contract"),
                        "result": llm_result,
                        "sanitized_proposal": {},
                        "rank_mutated_by_llm": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        prediction.metadata["llm_rank_mutation"] = False if not mode.use_llm_rank_repair else bool(
            rank_repair_result and rank_repair_result.verified_repairs
        )
        predictions[incident.case_id] = prediction.ranked

        (run_dir / "alarms" / f"{incident.case_id}.json").write_text(
            json.dumps(alarm.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "graphs" / f"{incident.case_id}.json").write_text(
            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "predictions" / f"{incident.case_id}.json").write_text(
            json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_seconds = time.perf_counter() - case_started
        record = {
            "case_id": incident.case_id,
            "status": "ranked",
            "top1": prediction.ranked[0] if prediction.ranked else "",
            "llm_status": llm_result.get("status"),
            "llm_reason": str(llm_result.get("reason") or "")[:500],
            "timing": _timing_payload(total_seconds=total_seconds, llm_seconds=llm_seconds),
            "llm": _llm_payload(llm=llm, llm_result=llm_result),
            "experiment_mode": mode.name,
            "unsafe_experiment_mode": mode.unsafe,
            "llm_raw_candidates": len(raw_llm_proposal.get("candidates", [])) if isinstance(raw_llm_proposal, dict) else 0,
            "llm_candidates_used": len(sanitized_llm_proposal.get("candidates", [])) if isinstance(sanitized_llm_proposal, dict) else 0,
            "llm_hypotheses_used": len(sanitized_llm_proposal.get("hypotheses", [])) if isinstance(sanitized_llm_proposal, dict) else 0,
            "llm_rank_repairs_used": len(sanitized_llm_proposal.get("rank_repair", [])) if isinstance(sanitized_llm_proposal, dict) else 0,
            "verified_rank_repairs": len(rank_repair_result.verified_repairs) if rank_repair_result else 0,
            "rejected_rank_repairs": len(rank_repair_result.rejected_repairs) if rank_repair_result else 0,
            "verified_evidence": verified_count,
            "rejected_evidence": rejected_count,
            "closure_paths": len(prediction.closure_paths),
        }
        records.append(record)
        _append_record(run_dir / "records.jsonl", record)
        _write_run_state(
            run_dir=run_dir,
            dataset=dataset,
            case_limit=case_limit,
            llm=llm,
            records=records,
            predictions=predictions,
            labels=labels,
            start_run=start_run,
            token_usage=token_usage,
        )
        if progress:
            print(
                format_progress_line(
                    dataset=dataset,
                    llm=llm,
                    index=index,
                    total=total_cases,
                    case_id=incident.case_id,
                    status=f"ranked/llm={llm_result.get('status')}",
                    top1=prediction.ranked[0] if prediction.ranked else "",
                ),
                flush=True,
            )

    _write_run_state(
        run_dir=run_dir,
        dataset=dataset,
        case_limit=case_limit,
        llm=llm,
        records=records,
        predictions=predictions,
        labels=labels,
        start_run=start_run,
        token_usage=token_usage,
    )
    write_experiment_statistics(run_dir=run_dir, method="Nurse", dataset=dataset, llm=llm)
    if generate_doc:
        generate_system_doc(run_dir, REPO_ROOT / "outputs" / "nurse" / "docs")
    return run_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_dir = run_nurse(
        args.dataset,
        args.case_limit,
        args.llm,
        args.output_root,
        generate_doc=args.generate_doc,
        run_dir=args.run_dir,
        resume=args.resume,
        progress=not args.no_progress,
        ranking_engine=args.ranking_engine,
        optimization_round=args.optimization_round,
        config_out=args.config_out,
        experiment_mode=args.experiment_mode,
        ranker_overrides_json=args.ranker_overrides_json,
        prompt_profile=args.prompt_profile,
        metric_score_backend=args.metric_score_backend,
        shuffle=args.shuffle,
        seed=args.seed,
        case_manifest=args.case_manifest,
        case_ids_file=args.case_ids_file,
    )
    print(f"[Nurse] Run directory: {run_dir}")


if __name__ == "__main__":
    main()
