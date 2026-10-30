"""Reproducible operator commands for fixed paper V1 experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from nurse_rcl.agents import (
    SEMANTIC_FACT_PROMPT_VERSION,
    USER_MANAGER_ORCHESTRATION_PROMPT_VERSION,
    USER_MANAGER_REPORT_PROMPT_VERSION,
)
from nurse_rcl.agents.ark import ArkChatCompletionsClient, load_server_ark_settings
from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.datasets.aiops25 import (
    build_aiops25_evaluation_manifest,
    build_aiops25_inference_manifest,
    prepare_aiops25,
    server_preparation_config,
)
from nurse_rcl.datasets.rcabench import (
    build_evaluation_manifest as build_rcabench_evaluation_manifest,
)
from nurse_rcl.datasets.rcabench import build_inference_manifest as build_rcabench_manifest
from nurse_rcl.deployment import deploy_release_archive, promote_release
from nurse_rcl.evaluation.evaluator import evaluate_predictions
from nurse_rcl.evaluation.outputs import write_evaluation_outputs
from nurse_rcl.experiment.launcher import RunLaunchRequest, validate_workload_placement
from nurse_rcl.experiment.manifest import (
    ModelIdentity,
    RunConcurrency,
    RunManifestArtifact,
    allocate_run_id,
    create_run_manifest,
)
from nurse_rcl.experiment.partial import (
    PartialSelectionArtifact,
    evaluate_materialized_partial,
    freeze_partial_selection,
    load_partial_selection,
    materialize_partial_predictions,
    partial_evaluation_root,
)
from nurse_rcl.experiment.runner import (
    RUNNER_SCHEMA_VERSION,
    run_production_dataset,
    run_production_partial,
)
from nurse_rcl.experiment.status import inspect_run_status
from nurse_rcl.methods.paper_v1 import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_REVISION,
    EMBEDDING_WEIGHT_SHA256,
    FINAL_RANKING_VERSION,
    HISTORY_SCHEMA_VERSION,
    VERIFIED_CONTEXT_VERSION,
)
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact
from nurse_rcl.methods.paper_v1.embedding_asset import verify_embedding_asset
from nurse_rcl.release import build_release_package
from nurse_rcl.schemas import EvaluationManifest, InferenceManifest, StrictFrozenModel
from nurse_rcl.sealing import create_prediction_seal, verify_prediction_seal

Completion = Callable[[dict[str, Any]], dict[str, Any]]
Embedder = Callable[[SemanticPayloadArtifact], tuple[float, ...]]


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one fail-closed operational command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    return handler(arguments)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate fixed paper V1 NURSE runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifests = subparsers.add_parser(
        "manifests", description="Create isolated inference and evaluation manifests"
    )
    manifests.add_argument("--dataset", required=True, choices=("rcabench", "aiops25"))
    manifests.add_argument("--data-root", required=True, type=Path)
    manifests.add_argument("--inference-output", required=True, type=Path)
    manifests.add_argument("--evaluation-output", required=True, type=Path)
    manifests.add_argument("--case-limit", type=int)
    manifests.set_defaults(handler=_command_manifests)

    prepare = subparsers.add_parser(
        "prepare-aiops25", description="Run the mandated AIOps25 converter"
    )
    prepare.add_argument("--work-root", required=True, type=Path)
    prepare.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare.set_defaults(handler=_command_prepare_aiops25)

    freeze = subparsers.add_parser(
        "freeze-run", description="Create one immutable fixed V1 run manifest"
    )
    freeze.add_argument("--dataset", required=True, choices=("rcabench", "aiops25"))
    freeze.add_argument("--inference-manifest", required=True, type=Path)
    freeze.add_argument("--telemetry-root", required=True, type=Path)
    freeze.add_argument("--method-config", required=True, type=Path)
    freeze.add_argument("--dataset-config", required=True, type=Path)
    freeze.add_argument("--dependency-lock", required=True, type=Path)
    freeze.add_argument("--source-fingerprint", required=True)
    freeze.add_argument("--case-workers", required=True, type=int)
    freeze.add_argument("--llm-in-flight", required=True, type=int)
    freeze.add_argument("--run-id")
    freeze.add_argument("--output", required=True, type=Path)
    freeze.set_defaults(handler=_command_freeze_run)

    package = subparsers.add_parser(
        "package-release", description="Build a safe content-addressed source archive"
    )
    package.add_argument("--repository-root", required=True, type=Path)
    package.add_argument("--output-root", required=True, type=Path)
    package.set_defaults(handler=_command_package_release)

    deploy = subparsers.add_parser(
        "deploy-release", description="Install and promote a verified server release"
    )
    deploy.add_argument("--archive", required=True, type=Path)
    deploy.add_argument("--deployment-root", required=True, type=Path)
    deploy.add_argument("--previous-source-hash")
    deploy.set_defaults(handler=_command_deploy_release)

    promote = subparsers.add_parser(
        "promote-release", description="Promote or roll back to an installed release"
    )
    promote.add_argument("--deployment-root", required=True, type=Path)
    promote.add_argument("--source-hash", required=True)
    promote.add_argument("--previous-source-hash")
    promote.set_defaults(handler=_command_promote_release)

    run = subparsers.add_parser("run", description="Execute the production V1 runner")
    run.add_argument("--inference-manifest", required=True, type=Path)
    run.add_argument("--telemetry-root", required=True, type=Path)
    run.add_argument("--run-manifest", required=True, type=Path)
    run.add_argument("--run-root", required=True, type=Path)
    run.add_argument("--provider", required=True, choices=("mock", "ark"))
    run.add_argument("--embedding-snapshot", type=Path)
    run.add_argument("--scope", required=True, choices=("synthetic", "sample", "full"))
    run.add_argument("--estimated-seconds", required=True, type=int)
    run.add_argument("--supervised", action="store_true")
    run.add_argument("--sample-approved", action="store_true")
    run.set_defaults(handler=_command_run)

    freeze_partial = subparsers.add_parser(
        "freeze-partial",
        description="Freeze a complete same-run Stage-A manifest-order sample",
    )
    freeze_partial.add_argument("--inference-manifest", required=True, type=Path)
    freeze_partial.add_argument("--telemetry-root", required=True, type=Path)
    freeze_partial.add_argument("--run-manifest", required=True, type=Path)
    freeze_partial.add_argument("--run-root", required=True, type=Path)
    freeze_partial.add_argument("--namespace", required=True)
    freeze_partial.add_argument("--case-limit", required=True, type=int)
    freeze_partial.set_defaults(handler=_command_freeze_partial)

    continue_partial = subparsers.add_parser(
        "continue-partial",
        description="Execute fixed V1 Stage B for a frozen partial selection",
    )
    continue_partial.add_argument("--inference-manifest", required=True, type=Path)
    continue_partial.add_argument("--telemetry-root", required=True, type=Path)
    continue_partial.add_argument("--run-manifest", required=True, type=Path)
    continue_partial.add_argument("--run-root", required=True, type=Path)
    continue_partial.add_argument("--selection", required=True, type=Path)
    continue_partial.add_argument("--provider", required=True, choices=("mock", "ark"))
    continue_partial.add_argument(
        "--scope",
        required=True,
        choices=("synthetic", "sample", "full"),
    )
    continue_partial.add_argument("--estimated-seconds", required=True, type=int)
    continue_partial.add_argument("--supervised", action="store_true")
    continue_partial.add_argument("--sample-approved", action="store_true")
    continue_partial.set_defaults(handler=_command_continue_partial)

    evaluate_partial = subparsers.add_parser(
        "evaluate-partial",
        description="Seal and evaluate a completed development-partial selection",
    )
    evaluate_partial.add_argument("--run-root", required=True, type=Path)
    evaluate_partial.add_argument("--selection", required=True, type=Path)
    evaluate_partial.add_argument("--evaluation-manifest", required=True, type=Path)
    evaluate_partial.set_defaults(handler=_command_evaluate_partial)

    seal = subparsers.add_parser("seal", description="Seal one completed prediction tree")
    seal.add_argument("--run-root", required=True, type=Path)
    seal.set_defaults(handler=_command_seal)

    evaluate = subparsers.add_parser(
        "evaluate", description="Evaluate labels only after prediction seal verification"
    )
    evaluate.add_argument("--run-root", required=True, type=Path)
    evaluate.add_argument("--evaluation-manifest", required=True, type=Path)
    evaluate.set_defaults(handler=_command_evaluate)
    return parser


def _command_manifests(arguments: argparse.Namespace) -> int:
    dataset = cast(str, arguments.dataset)
    data_root = cast(Path, arguments.data_root)
    if dataset == "rcabench":
        inference = build_rcabench_manifest(data_root)
        evaluation = build_rcabench_evaluation_manifest(data_root)
    else:
        inference = build_aiops25_inference_manifest(data_root)
        evaluation = build_aiops25_evaluation_manifest(data_root)
    case_limit = cast(int | None, arguments.case_limit)
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("--case-limit must be positive")
        inference = inference.model_copy(update={"cases": inference.cases[:case_limit]})
        selected = {case.case_id for case in inference.cases}
        evaluation = evaluation.model_copy(
            update={"cases": tuple(case for case in evaluation.cases if case.case_id in selected)}
        )
    _write_immutable_model(cast(Path, arguments.inference_output), inference)
    _write_immutable_model(cast(Path, arguments.evaluation_output), evaluation)
    _emit(
        {
            "dataset": dataset,
            "inference_cases": len(inference.cases),
            "evaluation_cases": len(evaluation.cases),
            "inference_manifest_hash": _model_hash(inference),
            "evaluation_manifest_hash": _model_hash(evaluation),
        }
    )
    return 0


def _command_prepare_aiops25(arguments: argparse.Namespace) -> int:
    config = server_preparation_config(
        work_root=cast(Path, arguments.work_root),
        python_executable=cast(Path, arguments.python),
    )
    receipt = prepare_aiops25(config)
    _emit(receipt.model_dump(mode="json"))
    return 0


def _command_freeze_run(arguments: argparse.Namespace) -> int:
    dataset = cast(str, arguments.dataset)
    inference = _load_inference_manifest(
        cast(Path, arguments.inference_manifest),
        telemetry_roots=(cast(Path, arguments.telemetry_root),),
    )
    if inference.dataset != dataset:
        raise ValueError("inference manifest dataset does not match --dataset")
    source_fingerprint = cast(str, arguments.source_fingerprint).casefold()
    if len(source_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in source_fingerprint
    ):
        raise ValueError("source fingerprint must be a lowercase SHA-256 digest")
    configuration = {
        "method": _load_yaml_mapping(cast(Path, arguments.method_config)),
        "dataset": _load_yaml_mapping(cast(Path, arguments.dataset_config)),
    }
    run_id = cast(str | None, arguments.run_id) or allocate_run_id(
        method_namespace="v1",
        dataset=cast(Any, dataset),
    )
    artifact = create_run_manifest(
        run_id=run_id,
        created_at_unix=int(time.time()),
        method_namespace="v1",
        method_version="paper_v1",
        dataset=cast(Any, dataset),
        source_fingerprint=source_fingerprint,
        configuration=configuration,
        dataset_fingerprint=_model_hash(inference),
        dependency_fingerprint=_file_sha256(cast(Path, arguments.dependency_lock)),
        model_identity=ModelIdentity(
            provider="ark_openai_compatible",
            endpoint="https://ark.cn-beijing.volces.com/api/v3",
            llm_model="deepseek-v4-flash-ga-260731",
            embedding_model=EMBEDDING_MODEL_ID,
            embedding_revision=EMBEDDING_REVISION,
            embedding_weight_sha256=EMBEDDING_WEIGHT_SHA256,
        ),
        prompt_versions={
            "semantic_fact": SEMANTIC_FACT_PROMPT_VERSION,
            "user_manager_orchestration": USER_MANAGER_ORCHESTRATION_PROMPT_VERSION,
            "user_manager_report": USER_MANAGER_REPORT_PROMPT_VERSION,
        },
        schema_versions={
            "history": str(HISTORY_SCHEMA_VERSION),
            "ranking": FINAL_RANKING_VERSION,
            "runner": str(RUNNER_SCHEMA_VERSION),
            "verified_context": VERIFIED_CONTEXT_VERSION,
        },
        concurrency=RunConcurrency(
            case_workers=cast(int, arguments.case_workers),
            llm_in_flight=cast(int, arguments.llm_in_flight),
        ),
        cross_run_llm_cache=False,
    )
    output = cast(Path, arguments.output)
    _write_immutable_model(output, artifact)
    _emit(
        {
            "run_id": artifact.manifest.run_id,
            "manifest_hash": artifact.manifest_hash,
            "output": str(output.resolve(strict=False)),
        }
    )
    return 0


def _command_package_release(arguments: argparse.Namespace) -> int:
    receipt = build_release_package(
        repository_root=cast(Path, arguments.repository_root),
        output_root=cast(Path, arguments.output_root),
    )
    _emit(receipt.model_dump(mode="json"))
    return 0


def _command_deploy_release(arguments: argparse.Namespace) -> int:
    receipt = deploy_release_archive(
        archive_path=cast(Path, arguments.archive),
        deployment_root=cast(Path, arguments.deployment_root),
        previous_source_hash=cast(str | None, arguments.previous_source_hash),
    )
    _emit(receipt.model_dump(mode="json"))
    return 0


def _command_promote_release(arguments: argparse.Namespace) -> int:
    receipt = promote_release(
        deployment_root=cast(Path, arguments.deployment_root),
        source_hash=cast(str, arguments.source_hash),
        previous_source_hash=cast(str | None, arguments.previous_source_hash),
    )
    _emit(receipt.model_dump(mode="json"))
    return 0


def _command_run(arguments: argparse.Namespace) -> int:
    inference = _load_inference_manifest(
        cast(Path, arguments.inference_manifest),
        telemetry_roots=(cast(Path, arguments.telemetry_root),),
    )
    run_manifest = RunManifestArtifact.model_validate_json(
        cast(Path, arguments.run_manifest).read_text(encoding="utf-8")
    )
    if not run_manifest.verify():
        raise ValueError("run manifest verification failed")
    if run_manifest.manifest.fingerprints.dataset != _model_hash(inference):
        raise ValueError("inference manifest fingerprint differs from the frozen run manifest")
    placement = validate_workload_placement(
        request=RunLaunchRequest(
            dataset=inference.dataset,
            scope=cast(Any, arguments.scope),
            estimated_duration_seconds=cast(int, arguments.estimated_seconds),
            supervised=cast(bool, arguments.supervised),
            sample_approved=cast(bool, arguments.sample_approved),
        )
    )
    run_root = cast(Path, arguments.run_root)
    _write_immutable_model(run_root / "placement.json", placement)
    provider = cast(str, arguments.provider)
    client: ArkChatCompletionsClient | None = None
    secrets: tuple[str, ...] = ()
    if provider == "mock":
        complete: Completion = _DeterministicMockCompletion()
        embed: Embedder = _mock_embed
    else:
        snapshot = cast(Path | None, arguments.embedding_snapshot)
        if snapshot is None:
            raise ValueError("--embedding-snapshot is required for the ARK provider")
        settings = load_server_ark_settings()
        client = ArkChatCompletionsClient(settings)
        complete = client.complete
        embed = _real_embedder(snapshot)
        secrets = (settings.credential.reveal_for_transport(),)
    try:
        result = run_production_dataset(
            inference_manifest=inference,
            run_manifest=run_manifest,
            run_root=run_root,
            complete=complete,
            embed=embed,
            secrets=secrets,
        )
    finally:
        if client is not None:
            client.close()
    _emit(
        {
            "run_id": result.run_id,
            "result_hash": result.result_hash,
            "shutdown_reason": result.schedule.shutdown_reason,
            "terminal_count": len(result.terminals),
        }
    )
    return 0


def _command_freeze_partial(arguments: argparse.Namespace) -> int:
    inference = _load_inference_manifest(
        cast(Path, arguments.inference_manifest),
        telemetry_roots=(cast(Path, arguments.telemetry_root),),
    )
    run_manifest = _load_run_manifest(cast(Path, arguments.run_manifest))
    selection = freeze_partial_selection(
        inference_manifest=inference,
        run_manifest=run_manifest,
        run_root=cast(Path, arguments.run_root),
        namespace=cast(str, arguments.namespace),
        limit=cast(int, arguments.case_limit),
    )
    _emit(
        {
            "parent_run_id": selection.parent_run_id,
            "selection_hash": selection.selection_hash,
            "sample_strategy": selection.sample_strategy,
            "sample_size": len(selection.cases),
            "scanned_case_count": selection.scanned_case_count,
            "skipped_case_count": len(selection.skipped_cases),
            "selection_path": str(
                partial_evaluation_root(
                    run_root=cast(Path, arguments.run_root),
                    namespace=selection.namespace,
                )
                / "selection.json"
            ),
        }
    )
    return 0


def _command_continue_partial(arguments: argparse.Namespace) -> int:
    inference = _load_inference_manifest(
        cast(Path, arguments.inference_manifest),
        telemetry_roots=(cast(Path, arguments.telemetry_root),),
    )
    run_manifest = _load_run_manifest(cast(Path, arguments.run_manifest))
    _require_frozen_inference_binding(
        inference=inference,
        run_manifest=run_manifest,
    )
    run_root = cast(Path, arguments.run_root)
    selection = load_partial_selection(
        selection_path=cast(Path, arguments.selection),
        inference_manifest=inference,
        run_manifest=run_manifest,
        run_root=run_root,
    )
    placement = validate_workload_placement(
        request=RunLaunchRequest(
            dataset=inference.dataset,
            scope=cast(Any, arguments.scope),
            estimated_duration_seconds=cast(int, arguments.estimated_seconds),
            supervised=cast(bool, arguments.supervised),
            sample_approved=cast(bool, arguments.sample_approved),
        )
    )
    partial_root = partial_evaluation_root(
        run_root=run_root,
        namespace=selection.namespace,
    )
    _write_immutable_model(partial_root / "placement.json", placement)
    provider = cast(str, arguments.provider)
    client: ArkChatCompletionsClient | None = None
    secrets: tuple[str, ...] = ()
    if provider == "mock":
        complete: Completion = _DeterministicMockCompletion()
    else:
        settings = load_server_ark_settings()
        client = ArkChatCompletionsClient(settings)
        complete = client.complete
        secrets = (settings.credential.reveal_for_transport(),)
    try:
        result = run_production_partial(
            inference_manifest=inference,
            run_manifest=run_manifest,
            run_root=run_root,
            selection=selection,
            complete=complete,
            secrets=secrets,
        )
    finally:
        if client is not None:
            client.close()
    _emit(
        {
            "run_id": result.run_id,
            "selection_hash": result.selection_hash,
            "result_hash": result.result_hash,
            "shutdown_reason": result.schedule.shutdown_reason,
            "terminal_count": len(result.terminals),
        }
    )
    return 0


def _command_evaluate_partial(arguments: argparse.Namespace) -> int:
    run_root = cast(Path, arguments.run_root).resolve(strict=True)
    selection = PartialSelectionArtifact.model_validate_json(
        cast(Path, arguments.selection).read_text(encoding="utf-8")
    )
    if not selection.verify():
        raise ValueError("partial selection hash verification failed")
    materialize_partial_predictions(run_root=run_root, selection=selection)
    evaluation_manifest = EvaluationManifest.model_validate_json(
        cast(Path, arguments.evaluation_manifest).read_text(encoding="utf-8")
    )
    receipt = evaluate_materialized_partial(
        run_root=run_root,
        selection=selection,
        evaluation_manifest=evaluation_manifest,
    )
    _emit(
        {
            "provisional": receipt.provisional,
            "sample_strategy": receipt.sample_strategy,
            "sample_size": receipt.sample_size,
            "metrics": receipt.metrics.model_dump(mode="json"),
            "prediction_tree_hash": receipt.prediction_tree_hash,
            "receipt_hash": receipt.receipt_hash,
        }
    )
    return 0


def _command_seal(arguments: argparse.Namespace) -> int:
    run_root = cast(Path, arguments.run_root).resolve(strict=True)
    status = inspect_run_status(run_root=run_root)
    if status.status != "completed" or not status.healthy:
        raise ValueError("prediction sealing requires a verified completed run")
    prediction_root = run_root / "predictions"
    seal_path = run_root / "evaluation" / "prediction_seal.json"
    seal = (
        verify_prediction_seal(prediction_root, seal_path)
        if seal_path.exists()
        else create_prediction_seal(prediction_root, seal_path)
    )
    _emit({"file_count": len(seal.files), "tree_hash": seal.tree_hash})
    return 0


def _command_evaluate(arguments: argparse.Namespace) -> int:
    run_root = cast(Path, arguments.run_root).resolve(strict=True)
    evaluation_manifest = EvaluationManifest.model_validate_json(
        cast(Path, arguments.evaluation_manifest).read_text(encoding="utf-8")
    )
    prediction_root = run_root / "predictions"
    result = evaluate_predictions(
        prediction_root=prediction_root,
        seal_path=run_root / "evaluation" / "prediction_seal.json",
        evaluation_manifest=evaluation_manifest,
    )
    receipt = write_evaluation_outputs(
        result=result,
        prediction_root=prediction_root,
        output_root=run_root / "evaluation" / "results",
    )
    _emit(
        {
            "dataset": result.dataset,
            "metrics": result.metrics.model_dump(mode="json"),
            "result_hash": result.result_hash,
            "output_receipt_hash": receipt.receipt_hash,
        }
    )
    return 0


class _DeterministicMockCompletion:
    """Offline provider that exercises the actual HERMES loops without a key."""

    def __init__(self) -> None:
        self._sequence = 0
        self._lock = threading.Lock()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("mock request has no messages")
        last = messages[-1]
        if isinstance(last, Mapping) and last.get("role") == "tool":
            return _mock_response(content="stage complete")
        tools = request.get("tools")
        if not isinstance(tools, list) or len(tools) != 1:
            raise ValueError("mock request must expose exactly one tool")
        function = tools[0].get("function") if isinstance(tools[0], Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str):
            raise ValueError("mock request tool has no name")
        content = last.get("content") if isinstance(last, Mapping) else None
        if not isinstance(content, str):
            raise ValueError("mock request has no JSON user document")
        document = json.loads(content)
        if not isinstance(document, dict):
            raise ValueError("mock user document must be an object")
        arguments = _mock_tool_arguments(name=name, document=document)
        with self._lock:
            self._sequence += 1
            call_id = f"mock-{self._sequence}"
        return _mock_response(name=name, arguments=arguments, call_id=call_id)


def _mock_tool_arguments(*, name: str, document: dict[str, Any]) -> dict[str, Any]:
    if name == "submit_semantic_facts":
        payload = document["semantic_payload"]
        cards = payload.get("candidates", [])
        facts: list[dict[str, Any]] = []
        if cards and cards[0].get("evidence_ids"):
            facts.append(
                {
                    "fact_type": "root_support",
                    "service": cards[0]["service"],
                    "evidence_id": cards[0]["evidence_ids"][0],
                    "related_service": None,
                    "related_evidence_id": None,
                }
            )
        return {"payload_hash": document["payload_hash"], "facts": facts}
    if name == "write_report":
        context = document["verified_rcl_context"]
        cited = sorted(
            {
                evidence_id
                for candidate in context["top_k"]
                for evidence_id in candidate["evidence_ids"]
            }
        )
        return {
            "verified_context_hash": document["verified_context_hash"],
            "top_k": context["top_k"],
            "evidence_paths": context["evidence_paths"],
            "cited_evidence_ids": cited,
            "narrative": "Deterministic offline smoke report from frozen evidence.",
            "recommendations": [],
        }
    return {
        "checkpoint_hash": document["checkpoint_hash"],
        "input_artifact_hash": document["input_artifact_hash"],
    }


def _mock_response(
    *,
    content: str | None = None,
    name: str | None = None,
    arguments: dict[str, Any] | None = None,
    call_id: str = "mock-call",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if name is not None and arguments is not None:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                },
            }
        ]
    return {
        "choices": [{"finish_reason": "tool_calls" if name else "stop", "message": message}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _mock_embed(payload: SemanticPayloadArtifact) -> tuple[float, ...]:
    digest = hashlib.sha256(canonical_json_bytes(payload.payload.model_dump(mode="json"))).digest()
    return tuple((value + 1) / 256.0 for value in digest)


def _real_embedder(snapshot_dir: Path) -> Embedder:
    verify_embedding_asset(
        snapshot_dir=snapshot_dir,
        resolved_revision=EMBEDDING_REVISION,
    )
    module = importlib.import_module("sentence_transformers")
    factory = module.SentenceTransformer
    model: Any = factory(
        str(snapshot_dir),
        device="cpu",
        local_files_only=True,
    )

    def embed(payload: SemanticPayloadArtifact) -> tuple[float, ...]:
        text = canonical_json_bytes(payload.payload.model_dump(mode="json")).decode("utf-8")
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector)

    return embed


def _load_inference_manifest(
    path: Path,
    *,
    telemetry_roots: tuple[Path, ...],
) -> InferenceManifest:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return InferenceManifest.model_validate(
        value,
        context={"allowed_roots": telemetry_roots},
    )


def _load_run_manifest(path: Path) -> RunManifestArtifact:
    artifact = RunManifestArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if not artifact.verify():
        raise ValueError("run manifest verification failed")
    return artifact


def _require_frozen_inference_binding(
    *,
    inference: InferenceManifest,
    run_manifest: RunManifestArtifact,
) -> None:
    if run_manifest.manifest.fingerprints.dataset != _model_hash(inference):
        raise ValueError("inference manifest fingerprint differs from the frozen run manifest")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    yaml_module = importlib.import_module("yaml")
    value: object = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a YAML object")
    return cast(dict[str, Any], value)


def _write_immutable_model(path: Path, model: StrictFrozenModel) -> None:
    data = canonical_json_bytes(model.model_dump(mode="json"))
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"immutable output differs: {path.name}")
        return
    atomic_write_bytes(path, data)


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(value: Mapping[str, Any]) -> None:
    print(canonical_json_bytes(dict(value)).decode("utf-8"), flush=True)


if __name__ == "__main__":  # pragma: no cover - exercised through console entry point
    raise SystemExit(main())
