"""Manifest-bound development partial continuation and evaluation artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.evaluation.evaluator import evaluate_predictions
from nurse_rcl.evaluation.outputs import write_evaluation_outputs
from nurse_rcl.experiment.checkpoints import (
    CaseRecovery,
    CheckpointStore,
    FailureTombstone,
    StageCheckpoint,
)
from nurse_rcl.experiment.manifest import RunManifestArtifact
from nurse_rcl.experiment.runner import CaseTerminalBundle, StageABundle
from nurse_rcl.experiment.scheduler import SchedulingResult
from nurse_rcl.models import EvaluationMetrics, PredictionRecord
from nurse_rcl.schemas import (
    EvaluationManifest,
    InferenceCase,
    InferenceManifest,
    StrictFrozenModel,
)
from nurse_rcl.sealing import (
    PredictionSeal,
    create_prediction_seal,
    verify_prediction_seal,
)

PARTIAL_EVALUATION_SCHEMA_VERSION: Literal[1] = 1
PARTIAL_SAMPLE_STRATEGY: Literal["completed_manifest_order_sample"] = (
    "completed_manifest_order_sample"
)


class PartialSelectedCase(StrictFrozenModel):
    """One complete Stage-A checkpoint frozen into a partial selection."""

    manifest_position: int = Field(ge=1)
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    stage_a_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_a_checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class PartialSkippedCase(StrictFrozenModel):
    """One verified failure tombstone skipped during manifest-order selection."""

    manifest_position: int = Field(ge=1)
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    failure_code: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    previous_checkpoint_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    failure_tombstone_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class PartialSelectionArtifact(StrictFrozenModel):
    """Immutable manifest-order selection for development-only evaluation."""

    schema_version: Literal[1] = PARTIAL_EVALUATION_SCHEMA_VERSION
    namespace: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    parent_run_id: str = Field(min_length=1)
    parent_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_inference_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_strategy: Literal["completed_manifest_order_sample"] = PARTIAL_SAMPLE_STRATEGY
    cases: tuple[PartialSelectedCase, ...] = Field(min_length=1)
    skipped_cases: tuple[PartialSkippedCase, ...] = ()
    scanned_case_count: int = Field(ge=1)
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_unique_cases(self) -> Self:
        identities = tuple(item.case_id for item in self.cases)
        if len(identities) != len(set(identities)):
            raise ValueError("partial selection case identities must be unique")
        skipped_identities = tuple(item.case_id for item in self.skipped_cases)
        if len(skipped_identities) != len(set(skipped_identities)):
            raise ValueError("partial skipped case identities must be unique")
        if set(identities) & set(skipped_identities):
            raise ValueError("selected and skipped partial cases must be disjoint")
        selected_positions = tuple(item.manifest_position for item in self.cases)
        skipped_positions = tuple(item.manifest_position for item in self.skipped_cases)
        if selected_positions != tuple(sorted(selected_positions)):
            raise ValueError("partial selected cases must retain manifest order")
        if skipped_positions != tuple(sorted(skipped_positions)):
            raise ValueError("partial skipped cases must retain manifest order")
        observed_positions = sorted((*selected_positions, *skipped_positions))
        if observed_positions != list(range(1, self.scanned_case_count + 1)):
            raise ValueError("partial selection must account for every scanned case")
        if selected_positions[-1] != self.scanned_case_count:
            raise ValueError("partial manifest scan must end on a selected case")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"selection_hash"})
        return self.selection_hash == sha256_hex(canonical_json_bytes(unsigned))


class PartialEvaluationReceipt(StrictFrozenModel):
    """Hash-bound proof that one partial population was sealed and evaluated."""

    schema_version: Literal[1] = PARTIAL_EVALUATION_SCHEMA_VERSION
    provisional: Literal[True] = True
    sample_strategy: Literal["completed_manifest_order_sample"] = PARTIAL_SAMPLE_STRATEGY
    sample_size: int = Field(ge=1)
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: EvaluationMetrics
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


def partial_evaluation_root(*, run_root: Path, namespace: str) -> Path:
    """Return one safe development-partial namespace below the parent run."""

    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", namespace) is None:
        raise ValueError("partial namespace is not a safe path segment")
    root = run_root.resolve(strict=False)
    candidate = (root / "evaluation" / namespace).resolve(strict=False)
    if not candidate.is_relative_to(root / "evaluation"):
        raise ValueError("partial namespace escapes the evaluation root")
    return candidate


def freeze_partial_selection(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    run_root: Path,
    namespace: str,
    limit: int,
) -> PartialSelectionArtifact:
    """Freeze the first complete cases in manifest order, binding every skip."""

    _validate_parent_binding(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        run_root=run_root,
    )
    if limit < 1:
        raise ValueError("partial case limit must be positive")
    if limit > len(inference_manifest.cases):
        raise ValueError("partial case limit exceeds the inference manifest")

    store = CheckpointStore(root=run_root, manifest=run_manifest)
    selected, skipped, scanned_case_count = _select_manifest_order_sample(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        store=store,
        limit=limit,
    )

    unsigned = {
        "schema_version": PARTIAL_EVALUATION_SCHEMA_VERSION,
        "namespace": namespace,
        "parent_run_id": run_manifest.manifest.run_id,
        "parent_manifest_hash": run_manifest.manifest_hash,
        "parent_inference_manifest_hash": _model_hash(inference_manifest),
        "sample_strategy": PARTIAL_SAMPLE_STRATEGY,
        "cases": [item.model_dump(mode="json") for item in selected],
        "skipped_cases": [item.model_dump(mode="json") for item in skipped],
        "scanned_case_count": scanned_case_count,
    }
    artifact = PartialSelectionArtifact(
        schema_version=PARTIAL_EVALUATION_SCHEMA_VERSION,
        namespace=namespace,
        parent_run_id=run_manifest.manifest.run_id,
        parent_manifest_hash=run_manifest.manifest_hash,
        parent_inference_manifest_hash=_model_hash(inference_manifest),
        sample_strategy=PARTIAL_SAMPLE_STRATEGY,
        cases=selected,
        skipped_cases=skipped,
        scanned_case_count=scanned_case_count,
        selection_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not artifact.verify():  # pragma: no cover - construction invariant
        raise AssertionError("partial selection verification failed")
    _write_immutable_model(
        partial_evaluation_root(run_root=run_root, namespace=namespace)
        / "selection.json",
        artifact,
    )
    return artifact


def load_partial_selection(
    *,
    selection_path: Path,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    run_root: Path,
) -> PartialSelectionArtifact:
    """Load and revalidate a selection against its parent and Stage-A chain."""

    artifact = PartialSelectionArtifact.model_validate_json(
        selection_path.read_text(encoding="utf-8")
    )
    if not artifact.verify():
        raise ValueError("partial selection hash verification failed")
    _validate_parent_binding(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        run_root=run_root,
    )
    if (
        artifact.parent_run_id != run_manifest.manifest.run_id
        or artifact.parent_manifest_hash != run_manifest.manifest_hash
        or artifact.parent_inference_manifest_hash != _model_hash(inference_manifest)
    ):
        raise ValueError("partial selection parent binding differs")

    store = CheckpointStore(root=run_root, manifest=run_manifest)
    expected_cases, expected_skips, expected_scan_count = _select_manifest_order_sample(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        store=store,
        limit=len(artifact.cases),
    )
    if (
        artifact.cases != expected_cases
        or artifact.skipped_cases != expected_skips
        or artifact.scanned_case_count != expected_scan_count
    ):
        raise ValueError("partial selection does not match the manifest-order sample")
    return artifact


def _select_manifest_order_sample(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    store: CheckpointStore,
    limit: int,
) -> tuple[tuple[PartialSelectedCase, ...], tuple[PartialSkippedCase, ...], int]:
    selected: list[PartialSelectedCase] = []
    skipped: list[PartialSkippedCase] = []
    for manifest_position, case in enumerate(inference_manifest.cases, start=1):
        recovery = store.recover_case(case_id=case.case_id)
        if recovery.failure is not None:
            skipped.append(
                _bind_skipped_case(
                    case=case,
                    manifest_position=manifest_position,
                    failure=recovery.failure,
                )
            )
            continue
        bundle, checkpoint = _restore_complete_stage_a_bundle_from_recovery(
            case=case,
            run_manifest=run_manifest,
            store=store,
            recovery=recovery,
        )
        selected.append(
            PartialSelectedCase(
                manifest_position=manifest_position,
                case_id=case.case_id,
                t0=case.t0,
                stage_a_artifact_hash=bundle.artifact_hash,
                stage_a_checkpoint_hash=checkpoint.checkpoint_hash,
            )
        )
        if len(selected) == limit:
            return tuple(selected), tuple(skipped), manifest_position
    raise ValueError("partial case limit exceeds complete manifest-order Stage A cases")


def _bind_skipped_case(
    *,
    case: InferenceCase,
    manifest_position: int,
    failure: FailureTombstone,
) -> PartialSkippedCase:
    return PartialSkippedCase(
        manifest_position=manifest_position,
        case_id=case.case_id,
        t0=case.t0,
        stage=failure.stage,
        failure_code=failure.failure_code,
        previous_checkpoint_hash=failure.previous_checkpoint_hash,
        failure_tombstone_hash=failure.tombstone_hash,
    )


def restore_complete_stage_a_bundle(
    *,
    case: InferenceCase,
    run_manifest: RunManifestArtifact,
    store: CheckpointStore,
) -> tuple[StageABundle, StageCheckpoint]:
    """Restore one verified Stage-A bundle without loading telemetry or an LLM."""

    recovery = store.recover_case(case_id=case.case_id)
    return _restore_complete_stage_a_bundle_from_recovery(
        case=case,
        run_manifest=run_manifest,
        store=store,
        recovery=recovery,
    )


def _restore_complete_stage_a_bundle_from_recovery(
    *,
    case: InferenceCase,
    run_manifest: RunManifestArtifact,
    store: CheckpointStore,
    recovery: CaseRecovery,
) -> tuple[StageABundle, StageCheckpoint]:
    if recovery.failure is not None:
        raise ValueError(f"case {case.case_id} has no complete Stage A bundle")
    checkpoints = tuple(
        item for item in recovery.checkpoints if item.stage == "stage_a_complete"
    )
    if len(checkpoints) != 1:
        raise ValueError(f"case {case.case_id} has no complete Stage A bundle")
    checkpoint = checkpoints[0]
    replay = store.replay_stage(
        case_id=case.case_id,
        stage="stage_a_complete",
        input_hashes=checkpoint.input_hashes,
        history_snapshot_hash=None,
    )
    output = replay.output
    if not replay.hit or output is None or set(output) != {"bundle"}:
        raise ValueError(f"case {case.case_id} Stage A checkpoint cannot be replayed")
    try:
        bundle = StageABundle.model_validate(output["bundle"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"case {case.case_id} Stage A bundle is invalid") from error
    expected_inputs = (
        run_manifest.manifest_hash,
        _model_hash(case),
        bundle.telemetry_hash,
    )
    if checkpoint.input_hashes != expected_inputs:
        raise ValueError(f"case {case.case_id} Stage A parent binding differs")
    if (
        bundle.case_id != case.case_id
        or bundle.t0 != case.t0
        or bundle.controls.run_manifest_hash != run_manifest.manifest_hash
        or not bundle.verify()
    ):
        raise ValueError(f"case {case.case_id} Stage A artifact verification failed")
    return bundle, checkpoint


def materialize_partial_predictions(
    *,
    run_root: Path,
    selection: PartialSelectionArtifact,
) -> PredictionSeal:
    """Copy exactly the completed selection into an independently sealed tree."""

    if not selection.verify():
        raise ValueError("partial selection hash verification failed")
    root = partial_evaluation_root(
        run_root=run_root,
        namespace=selection.namespace,
    )
    _require_persisted_selection(root=root, selection=selection)
    predictions = _load_completed_partial_predictions(
        root=root,
        selection=selection,
    )
    source_root = run_root.resolve(strict=True) / "predictions"
    prediction_root = root / "predictions"
    expected_names = {f"{item.case_id}.json" for item in selection.cases}
    existing_names = (
        {
            path.relative_to(prediction_root).as_posix()
            for path in prediction_root.rglob("*")
            if path.is_file()
        }
        if prediction_root.exists()
        else set()
    )
    if existing_names - expected_names:
        raise ValueError("partial prediction tree contains an unselected artifact")
    by_case = {item.case_id: item for item in predictions}
    for selected in selection.cases:
        source = source_root / f"{selected.case_id}.json"
        try:
            source_bytes = source.read_bytes()
            source_record = PredictionRecord.model_validate_json(source_bytes)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"selected terminal prediction is unreadable: {selected.case_id}"
            ) from error
        if source_record != by_case[selected.case_id]:
            raise ValueError("parent prediction differs from the partial terminal result")
        _write_immutable_bytes(
            prediction_root / f"{selected.case_id}.json",
            source_bytes,
        )
    actual_names = {
        path.relative_to(prediction_root).as_posix()
        for path in prediction_root.rglob("*")
        if path.is_file()
    }
    if actual_names != expected_names:
        raise ValueError("partial prediction tree does not exactly match the selection")
    seal_path = root / "prediction_seal.json"
    return (
        verify_prediction_seal(prediction_root, seal_path)
        if seal_path.exists()
        else create_prediction_seal(prediction_root, seal_path)
    )


def evaluate_materialized_partial(
    *,
    run_root: Path,
    selection: PartialSelectionArtifact,
    evaluation_manifest: EvaluationManifest,
) -> PartialEvaluationReceipt:
    """Verify the independent seal, then derive labels and report provisional metrics."""

    if not selection.verify():
        raise ValueError("partial selection hash verification failed")
    root = partial_evaluation_root(
        run_root=run_root,
        namespace=selection.namespace,
    )
    prediction_root = root / "predictions"
    seal = verify_prediction_seal(prediction_root, root / "prediction_seal.json")
    subset = _derive_partial_evaluation_manifest(
        selection=selection,
        evaluation_manifest=evaluation_manifest,
    )
    _write_immutable_model(root / "evaluation_manifest.json", subset)
    result = evaluate_predictions(
        prediction_root=prediction_root,
        seal_path=root / "prediction_seal.json",
        evaluation_manifest=subset,
    )
    output = write_evaluation_outputs(
        result=result,
        prediction_root=prediction_root,
        output_root=root / "results",
    )
    unsigned = {
        "schema_version": PARTIAL_EVALUATION_SCHEMA_VERSION,
        "provisional": True,
        "sample_strategy": PARTIAL_SAMPLE_STRATEGY,
        "sample_size": len(selection.cases),
        "selection_hash": selection.selection_hash,
        "prediction_tree_hash": seal.tree_hash,
        "evaluation_manifest_hash": _model_hash(subset),
        "evaluation_result_hash": result.result_hash,
        "output_receipt_hash": output.receipt_hash,
        "metrics": result.metrics.model_dump(mode="json"),
    }
    receipt = PartialEvaluationReceipt(
        schema_version=PARTIAL_EVALUATION_SCHEMA_VERSION,
        provisional=True,
        sample_strategy=PARTIAL_SAMPLE_STRATEGY,
        sample_size=len(selection.cases),
        selection_hash=selection.selection_hash,
        prediction_tree_hash=seal.tree_hash,
        evaluation_manifest_hash=_model_hash(subset),
        evaluation_result_hash=result.result_hash,
        output_receipt_hash=output.receipt_hash,
        metrics=result.metrics,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("partial evaluation receipt verification failed")
    _write_immutable_model(root / "partial_evaluation_receipt.json", receipt)
    return receipt


def _derive_partial_evaluation_manifest(
    *,
    selection: PartialSelectionArtifact,
    evaluation_manifest: EvaluationManifest,
) -> EvaluationManifest:
    included = tuple(case for case in evaluation_manifest.cases if not case.excluded)
    identities = tuple(case.case_id for case in included)
    if len(identities) != len(set(identities)):
        raise ValueError("evaluation manifest contains duplicate included case IDs")
    by_case = {case.case_id: case for case in included}
    missing = tuple(
        item.case_id for item in selection.cases if item.case_id not in by_case
    )
    if missing:
        raise ValueError("evaluation manifest is missing selected case labels")
    return EvaluationManifest(
        dataset=evaluation_manifest.dataset,
        cases=tuple(by_case[item.case_id] for item in selection.cases),
    )


def _load_completed_partial_predictions(
    *,
    root: Path,
    selection: PartialSelectionArtifact,
) -> tuple[PredictionRecord, ...]:
    marker_path = root / "PARTIAL_COMPLETE"
    result_path = root / "partial_run_result.json"
    if not marker_path.is_file() or not result_path.is_file():
        raise ValueError("partial continuation has no verified completion artifacts")
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "run_id",
            "selection_hash",
            "schedule",
            "predictions",
            "terminals",
            "result_hash",
        }:
            raise ValueError("partial result schema is invalid")
        schedule = SchedulingResult.model_validate(value["schedule"])
        predictions = tuple(
            PredictionRecord.model_validate(item) for item in value["predictions"]
        )
        terminals = tuple(
            CaseTerminalBundle.model_validate(item) for item in value["terminals"]
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("partial result is unreadable") from error
    prediction_case_ids = tuple(item.case_id for item in predictions)
    selected_case_ids = tuple(item.case_id for item in selection.cases)
    if (
        value["schema_version"] != PARTIAL_EVALUATION_SCHEMA_VERSION
        or value["run_id"] != selection.parent_run_id
        or value["selection_hash"] != selection.selection_hash
        or schedule.shutdown_reason != "completed"
        or not schedule.verify()
        or any(not item.verify() for item in terminals)
        or predictions != tuple(item.prediction for item in terminals)
        or len(prediction_case_ids) != len(selected_case_ids)
        or len(prediction_case_ids) != len(set(prediction_case_ids))
        or set(prediction_case_ids) != set(selected_case_ids)
    ):
        raise ValueError("partial result does not match the completed selection")
    unsigned = {
        "run_id": value["run_id"],
        "selection_hash": value["selection_hash"],
        "schedule": schedule.model_dump(mode="json"),
        "predictions": [item.model_dump(mode="json") for item in predictions],
        "terminal_artifact_hashes": [item.artifact_hash for item in terminals],
    }
    if value["result_hash"] != sha256_hex(canonical_json_bytes(unsigned)):
        raise ValueError("partial result hash verification failed")
    marker = {
        "schema_version": PARTIAL_EVALUATION_SCHEMA_VERSION,
        "run_id": value["run_id"],
        "selection_hash": value["selection_hash"],
        "result_hash": value["result_hash"],
        "terminal_count": len(terminals),
    }
    if marker_path.read_bytes() != canonical_json_bytes(marker):
        raise ValueError("partial completion marker binding is invalid")
    return predictions


def _require_persisted_selection(
    *,
    root: Path,
    selection: PartialSelectionArtifact,
) -> None:
    expected = canonical_json_bytes(selection.model_dump(mode="json"))
    try:
        observed = (root / "selection.json").read_bytes()
    except OSError as error:
        raise ValueError("partial selection artifact is missing") from error
    if observed != expected:
        raise ValueError("partial selection differs from its immutable artifact")


def _validate_parent_binding(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    run_root: Path,
) -> None:
    if not run_manifest.verify():
        raise ValueError("run manifest verification failed")
    if (
        inference_manifest.method_version != run_manifest.manifest.method_version
        or inference_manifest.dataset != run_manifest.manifest.dataset
    ):
        raise ValueError("inference and run manifest identities differ")
    inference_hash = _model_hash(inference_manifest)
    if run_manifest.manifest.fingerprints.dataset != inference_hash:
        raise ValueError("inference manifest fingerprint differs from the frozen run")
    bound_path = run_root.resolve(strict=False) / "inference_manifest.json"
    expected = canonical_json_bytes(inference_manifest.model_dump(mode="json"))
    try:
        observed = bound_path.read_bytes()
    except OSError as error:
        raise ValueError("parent run has no bound inference manifest") from error
    if observed != expected:
        raise ValueError("parent run inference manifest binding differs")


def _write_immutable_model(path: Path, model: StrictFrozenModel) -> None:
    data = canonical_json_bytes(model.model_dump(mode="json"))
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"immutable partial output differs: {path.name}")
        return
    atomic_write_bytes(path, data)


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"immutable partial output differs: {path.name}")
        return
    atomic_write_bytes(path, data)


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))
