"""Production manifest-to-terminal runner for the fixed paper V1 method."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from hermes_runtime.redaction import redact_sensitive_text
from hermes_runtime.runtime import AgentRunResult
from hermes_runtime.transport import ChatCompletionsTransport
from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.agents.execution import (
    CompletionFunction,
    StageToolOutput,
    run_semantic_fact_agent,
    run_user_manager_report_agent,
    run_user_manager_tool_step,
)
from nurse_rcl.agents.report import VerifiedAgentReportArtifact, finalize_user_manager_report
from nurse_rcl.agents.retry import (
    AttemptRecord,
    AttemptResult,
    AttemptUsage,
    CircuitBreaker,
    CircuitOpenError,
    FatalProviderRequestError,
    RequestExecution,
    RetryableParsingError,
    RetryableRepairError,
    RetryExhaustedError,
    execute_llm_request,
)
from nurse_rcl.agents.semantic_fact import SemanticFactSubmission
from nurse_rcl.agents.user_manager_state import (
    UserManagerCheckpoint,
    UserManagerState,
    UserManagerTool,
    apply_user_manager_tool,
    create_user_manager_checkpoint,
    restore_user_manager_checkpoint,
)
from nurse_rcl.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_hex,
    stable_identifier,
)
from nurse_rcl.datasets.telemetry import CaseTelemetry, LogTelemetryAudit, load_case_telemetry
from nurse_rcl.experiment.checkpoints import (
    CheckpointStore,
    LLMAttemptLedger,
    LLMCacheEntry,
    LLMCumulativeUsage,
)
from nurse_rcl.experiment.manifest import RunManifestArtifact
from nurse_rcl.experiment.observability import RunObserver, UsageTotals, read_progress_snapshot
from nurse_rcl.experiment.scheduler import (
    ScheduledCase,
    SchedulingResult,
    StageAResult,
    TerminalCaseResult,
    run_causal_wave_schedule,
    run_causal_wave_stage_b_schedule,
)
from nurse_rcl.methods.paper_v1.alarm import (
    AlarmDecision,
    SignalSeries,
    TimedValue,
    detect_alarm,
)
from nurse_rcl.methods.paper_v1.candidates import (
    MJ_BY_DATASET,
    CandidateSelection,
    build_candidates,
)
from nurse_rcl.methods.paper_v1.causal_history import (
    CausalHistoryIndex,
    HistoricalCase,
    HistoricalServiceScore,
    HistoryFailureTombstone,
    HistoryScoreArtifact,
    HistorySnapshot,
    compute_history_scores,
)
from nurse_rcl.methods.paper_v1.deviation import EVIDENCE_THRESHOLD, compute_metric_deviation
from nurse_rcl.methods.paper_v1.distiller import SemanticPayloadArtifact, distill_context
from nurse_rcl.methods.paper_v1.evidence_initializer import (
    PreliminaryGraphArtifact,
    initialize_preliminary_graph,
)
from nurse_rcl.methods.paper_v1.final_ranking import FinalRankingArtifact, assemble_final_ranking
from nurse_rcl.methods.paper_v1.raw_scoring import RawEPRArtifact, compute_raw_epr
from nurse_rcl.methods.paper_v1.scc_scoring import SCCScoringArtifact, score_scc_propagation
from nurse_rcl.methods.paper_v1.semantic_stage import (
    SemanticStageArtifact,
    complete_semantic_fact_stage,
    degrade_semantic_fact_stage,
)
from nurse_rcl.methods.paper_v1.topology import NormalTopology, TraceSpan, build_normal_topology
from nurse_rcl.methods.paper_v1.verified_context import (
    VerifiedRCLContextArtifact,
    freeze_verified_rcl_context,
)
from nurse_rcl.models import EvidenceRecord, FailureRecord, PredictionRecord, TimeWindow
from nurse_rcl.schemas import InferenceCase, InferenceManifest, StrictFrozenModel

if TYPE_CHECKING:
    from nurse_rcl.experiment.partial import PartialSelectionArtifact

RUNNER_SCHEMA_VERSION: Literal[1] = 1
WINDOW_SECONDS: Literal[480] = 480
MetricEmbedder = Callable[[SemanticPayloadArtifact], tuple[float, ...]]
TArtifact = TypeVar("TArtifact", bound=StrictFrozenModel)


class ProductionRunFatalError(RuntimeError):
    """Raised after the run has persisted a fatal marker and diagnostics."""


class ResumeDriftError(ValueError):
    """Raised when an existing stage cannot be replayed under the frozen inputs."""


class FrozenCaseControls(StrictFrozenModel):
    schema_version: Literal[1] = RUNNER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    dataset: Literal["rcabench", "aiops25"]
    t0: int = Field(ge=0)
    window_seconds: Literal[480] = WINDOW_SECONDS
    mj: int = Field(ge=1)
    evidence_threshold: float = EVIDENCE_THRESHOLD
    component_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    run_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_json: str = Field(min_length=2)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class AlarmContextArtifact(StrictFrozenModel):
    schema_version: Literal[1] = RUNNER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    telemetry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alarm: AlarmDecision
    evidence: tuple[EvidenceRecord, ...]
    topology: NormalTopology
    log_audit: LogTelemetryAudit
    warnings: tuple[str, ...]
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class RankingBundleArtifact(StrictFrozenModel):
    raw: RawEPRArtifact
    scc: SCCScoringArtifact
    history_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    history: HistoryScoreArtifact
    final: FinalRankingArtifact
    context: VerifiedRCLContextArtifact
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if not all(
            (
                self.raw.verify(),
                self.scc.verify(),
                self.history.verify(),
                self.final.verify(),
                self.context.verify(),
            )
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class StageABundle(StrictFrozenModel):
    schema_version: Literal[1] = RUNNER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    telemetry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    controls: FrozenCaseControls
    alarm_context: AlarmContextArtifact
    selection: CandidateSelection
    preliminary: PreliminaryGraphArtifact
    payload: SemanticPayloadArtifact
    semantic: SemanticStageArtifact
    embedding: tuple[float, ...] = Field(min_length=1)
    user_checkpoint: UserManagerCheckpoint
    degradation_codes: tuple[str, ...]
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_stage_a_chain(self) -> Self:
        if self.user_checkpoint.state != UserManagerState.FACTS_VALIDATED:
            raise ValueError("Stage A must end at FACTS_VALIDATED")
        if self.case_id != self.user_checkpoint.case_id:
            raise ValueError("Stage A case identity does not match User Manager checkpoint")
        return self

    def verify(self) -> bool:
        if not all(
            (
                self.controls.verify(),
                self.alarm_context.verify(),
                self.preliminary.verify(),
                self.payload.verify(),
                self.semantic.verify(),
                self.user_checkpoint.verify(),
            )
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class StageAFailureBundle(StrictFrozenModel):
    schema_version: Literal[1] = RUNNER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    failure: FailureRecord
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class CaseTerminalBundle(StrictFrozenModel):
    schema_version: Literal[1] = RUNNER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    t0: int = Field(ge=0)
    history_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction: PredictionRecord
    failure: FailureRecord | None = None
    ranking: RankingBundleArtifact | None = None
    context: VerifiedRCLContextArtifact | None = None
    report: VerifiedAgentReportArtifact | None = None
    user_checkpoint: UserManagerCheckpoint | None = None
    historical_case: HistoricalCase | None = None
    history_tombstone: HistoryFailureTombstone | None = None
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_one_terminal_shape(self) -> Self:
        if self.prediction.case_id != self.case_id:
            raise ValueError("terminal prediction case identity does not match")
        if self.prediction.status == "failed":
            if self.failure is None or self.history_tombstone is None:
                raise ValueError("failed terminal requires diagnostics and history tombstone")
            if any(
                item is not None
                for item in (
                    self.ranking,
                    self.context,
                    self.report,
                    self.user_checkpoint,
                    self.historical_case,
                )
            ):
                raise ValueError("failed terminal cannot contain a frozen ranking")
        elif (
            any(
                item is None
                for item in (
                    self.ranking,
                    self.context,
                    self.report,
                    self.user_checkpoint,
                    self.historical_case,
                )
            )
            or self.failure is not None
            or self.history_tombstone is not None
        ):
            raise ValueError("successful terminal requires the complete frozen chain")
        return self

    def verify(self) -> bool:
        nested = (self.ranking, self.context, self.report, self.user_checkpoint)
        if any(item is not None and not item.verify() for item in nested):
            return False
        unsigned = self.model_dump(mode="json", exclude={"artifact_hash"})
        return self.artifact_hash == sha256_hex(canonical_json_bytes(unsigned))


class LLMExhaustionSnapshot(StrictFrozenModel):
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    attempts: tuple[AttemptRecord, ...] = Field(min_length=5, max_length=5)
    cumulative_usage: LLMCumulativeUsage

    @model_validator(mode="after")
    def require_complete_exhausted_attempt_trace(self) -> Self:
        if tuple(item.attempt_number for item in self.attempts) != (1, 2, 3, 4, 5):
            raise ValueError("LLM exhaustion attempts must be one-based and complete")
        if any(item.outcome != "retry" for item in self.attempts[:-1]):
            raise ValueError("non-final exhausted attempts must be retries")
        if self.attempts[-1].outcome != "exhausted":
            raise ValueError("LLM exhaustion trace must end as exhausted")
        if self.cumulative_usage.attempts < len(self.attempts):
            raise ValueError("cumulative usage cannot omit exhausted attempts")
        return self


@dataclass(frozen=True, slots=True)
class ProductionRunResult:
    run_id: str
    inference_manifest_hash: str
    schedule: SchedulingResult
    predictions: tuple[PredictionRecord, ...]
    terminals: tuple[CaseTerminalBundle, ...]
    result_hash: str
    checkpoint_store: CheckpointStore = field(compare=False, repr=False)

    def verify(self) -> bool:
        if not self.schedule.verify() or any(not item.verify() for item in self.terminals):
            return False
        if self.predictions != tuple(item.prediction for item in self.terminals):
            return False
        unsigned = _production_result_document(
            run_id=self.run_id,
            inference_manifest_hash=self.inference_manifest_hash,
            schedule=self.schedule,
            predictions=self.predictions,
            terminals=self.terminals,
        )
        return self.result_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(frozen=True, slots=True)
class ProductionPartialResult:
    """Selection-bound Stage-B result that cannot complete the parent run."""

    run_id: str
    selection_hash: str
    schedule: SchedulingResult
    predictions: tuple[PredictionRecord, ...]
    terminals: tuple[CaseTerminalBundle, ...]
    result_hash: str
    checkpoint_store: CheckpointStore = field(compare=False, repr=False)

    def verify(self) -> bool:
        if not self.schedule.verify() or any(not item.verify() for item in self.terminals):
            return False
        if self.predictions != tuple(item.prediction for item in self.terminals):
            return False
        unsigned = _partial_result_document(
            run_id=self.run_id,
            selection_hash=self.selection_hash,
            schedule=self.schedule,
            predictions=self.predictions,
            terminals=self.terminals,
        )
        return self.result_hash == sha256_hex(canonical_json_bytes(unsigned))


@dataclass(slots=True)
class _RunMetrics:
    observer: RunObserver
    usage: UsageTotals
    stage_counts: dict[str, int]
    cache_hits: int
    lock: threading.RLock = field(default_factory=threading.RLock)

    def provider_attempt(self, *, latency_seconds: float) -> None:
        with self.lock:
            self.usage = self.usage.model_copy(
                update={
                    "attempts": self.usage.attempts + 1,
                    "latency_seconds": self.usage.latency_seconds + latency_seconds,
                }
            )

    def provider_tokens(self, usage: AttemptUsage) -> None:
        prompt = usage.prompt_tokens
        input_tokens = prompt if prompt is not None else 0
        output_tokens = usage.completion_tokens if usage.completion_tokens is not None else 0
        with self.lock:
            self.usage = self.usage.model_copy(
                update={
                    "input_tokens": self.usage.input_tokens + input_tokens,
                    "output_tokens": self.usage.output_tokens + output_tokens,
                }
            )

    def record_stage(self, *, stage: str, replayed: bool) -> None:
        with self.lock:
            if replayed:
                self.cache_hits += 1
            else:
                self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def record_cache_hit(self) -> None:
        with self.lock:
            self.cache_hits += 1

    def snapshot(self) -> UsageTotals:
        with self.lock:
            return self.usage


StageABundleValue = StageABundle | StageAFailureBundle


class _ProductionRunner:
    def __init__(
        self,
        *,
        inference_manifest: InferenceManifest,
        run_manifest: RunManifestArtifact,
        run_root: Path,
        complete: CompletionFunction,
        embed: MetricEmbedder,
        retry_sleep: Callable[[float], None],
        stop_requested: Callable[[], bool],
        secrets: Iterable[str],
    ) -> None:
        _validate_run_inputs(inference_manifest=inference_manifest, run_manifest=run_manifest)
        self._inference_manifest = inference_manifest
        self._run_manifest = run_manifest
        self._run_root = run_root.resolve(strict=False)
        self._run_root.mkdir(parents=True, exist_ok=True)
        self._inference_manifest_hash = _model_hash(inference_manifest)
        self._bind_inference_manifest()
        self._complete = complete
        self._embed = embed
        self._retry_sleep = retry_sleep
        self._stop_requested = stop_requested
        self._secrets = tuple(item for item in secrets if item)
        self._store = CheckpointStore(root=self._run_root, manifest=run_manifest)
        self._history = CausalHistoryIndex(root=self._run_root / "history")
        self._observer = RunObserver(
            root=self._run_root,
            run_id=run_manifest.manifest.run_id,
            total=len(inference_manifest.cases),
            secrets=self._secrets,
        )
        progress = read_progress_snapshot(self._run_root)
        progress_usage = UsageTotals(
            attempts=progress.llm_attempts,
            input_tokens=progress.llm_input_tokens,
            output_tokens=progress.llm_output_tokens,
            latency_seconds=progress.llm_latency_seconds,
            cost_usd=progress.llm_cost_usd,
        )
        case_ids = tuple(case.case_id for case in inference_manifest.cases)
        attempt_ledgers = self._store.verified_llm_attempt_ledgers()
        if any(ledger.case_id not in case_ids for ledger in attempt_ledgers):
            raise ResumeDriftError("LLM attempt ledger case is outside the run manifest")
        attempt_ledgers = self._store.reconcile_interrupted_llm_attempts()
        _ensure_attempt_ledger_exhaustions(
            store=self._store,
            ledgers=attempt_ledgers,
        )
        cache_entries = self._store.verified_llm_cache_entries()
        exhaustion_snapshots = _verified_llm_exhaustions(
            store=self._store,
            case_ids=case_ids,
        )
        recovered_usage = _recover_provider_usage(
            progress=progress_usage,
            entries=cache_entries,
            exhaustions=exhaustion_snapshots,
            ledgers=attempt_ledgers,
        )
        self._metrics = _RunMetrics(
            observer=self._observer,
            usage=recovered_usage,
            stage_counts=dict(progress.stage_counts),
            cache_hits=progress.cache_hits,
        )
        self._breaker = CircuitBreaker()
        restored_circuit_reason = _restored_global_circuit_reason(
            exhaustions=exhaustion_snapshots,
            ledgers=attempt_ledgers,
        )
        if restored_circuit_reason is not None:
            self._breaker.open(restored_circuit_reason)
        self._llm_semaphore = threading.BoundedSemaphore(
            run_manifest.manifest.concurrency.llm_in_flight
        )
        self._llm_request_lock = threading.RLock()
        self._stage_a: dict[str, StageABundleValue] = {}
        self._terminals: dict[str, CaseTerminalBundle] = {}
        self._snapshots: dict[str, HistorySnapshot] = {}
        self._state_lock = threading.RLock()

    @property
    def checkpoint_store(self) -> CheckpointStore:
        return self._store

    def run(self) -> ProductionRunResult:
        terminal_marker_exists = (self._run_root / "RUN_COMPLETE").exists()
        if (self._run_root / "fatal.json").exists():
            raise ProductionRunFatalError("existing_fatal_marker")
        if terminal_marker_exists:
            return self._restore_completed_result()
        self._observer.start_heartbeat()
        try:
            schedule = run_causal_wave_schedule(
                cases=tuple(
                    ScheduledCase(case_id=case.case_id, t0=case.t0)
                    for case in self._inference_manifest.cases
                ),
                case_workers=self._run_manifest.manifest.concurrency.case_workers,
                stage_a=self._run_stage_a,
                history_snapshot=self._history_snapshot,
                stage_b=self._run_stage_b,
                commit_group=self._commit_group,
                circuit_breaker=self._breaker,
                stop_requested=self._stop_requested,
            )
            if schedule.shutdown_reason == "circuit_open":
                reason = schedule.circuit_reason or "circuit_open"
                self._refresh_progress(
                    current_group=read_progress_snapshot(self._run_root).current_group
                )
                self._observer.fail(error_code="retry_exhausted", message=reason)
                raise ProductionRunFatalError(f"retry_exhausted: {reason}")
            if schedule.shutdown_reason == "completed":
                self._refresh_progress(current_group=None)
            else:
                self._observer.stop_heartbeat()
        except ProductionRunFatalError:
            raise
        except Exception as error:
            code = _fatal_error_code(error)
            message = _safe_message(error, secrets=self._secrets)
            self._refresh_progress(
                current_group=read_progress_snapshot(self._run_root).current_group
            )
            with suppress(ValueError):
                self._observer.fail(error_code=code, message=message)
            raise ProductionRunFatalError(f"{code}: {type(error).__name__}") from error

        terminals = tuple(self._terminals[item.case_id] for item in schedule.terminal_results)
        predictions = tuple(item.prediction for item in terminals)
        unsigned = _production_result_document(
            run_id=self._run_manifest.manifest.run_id,
            inference_manifest_hash=self._inference_manifest_hash,
            schedule=schedule,
            predictions=predictions,
            terminals=terminals,
        )
        result = ProductionRunResult(
            run_id=self._run_manifest.manifest.run_id,
            inference_manifest_hash=self._inference_manifest_hash,
            schedule=schedule,
            predictions=predictions,
            terminals=terminals,
            result_hash=sha256_hex(canonical_json_bytes(unsigned)),
            checkpoint_store=self._store,
        )
        if not result.verify():  # pragma: no cover - construction invariant
            raise AssertionError("production run result verification failed")
        if schedule.shutdown_reason == "completed":
            self._persist_completed_result(result)
            self._observer.complete(message="all selected cases reached terminal records")
        return result

    def run_partial(
        self,
        *,
        selection: PartialSelectionArtifact,
    ) -> ProductionPartialResult:
        """Restore selected Stage A and execute only the unchanged causal Stage B."""

        from nurse_rcl.experiment.partial import (
            load_partial_selection,
            partial_evaluation_root,
            restore_complete_stage_a_bundle,
        )

        if (self._run_root / "RUN_COMPLETE").exists():
            raise ResumeDriftError("partial continuation cannot replace a completed run")
        if (self._run_root / "fatal.json").exists():
            raise ProductionRunFatalError("existing_fatal_marker")
        partial_root = partial_evaluation_root(
            run_root=self._run_root,
            namespace=selection.namespace,
        )
        persisted_selection = load_partial_selection(
            selection_path=partial_root / "selection.json",
            inference_manifest=self._inference_manifest,
            run_manifest=self._run_manifest,
            run_root=self._run_root,
        )
        if persisted_selection != selection:
            raise ResumeDriftError("partial selection differs from its immutable artifact")
        marker_path = partial_root / "PARTIAL_COMPLETE"
        if marker_path.exists():
            return self._restore_partial_result(selection=selection, root=partial_root)
        if (partial_root / "partial_run_result.json").exists():
            raise ResumeDriftError("partial result exists without completion marker")

        stage_a_results: list[StageAResult] = []
        for selected in selection.cases:
            case = self._inference_manifest.cases[selected.manifest_position - 1]
            if case.case_id != selected.case_id:
                raise ResumeDriftError("partial manifest position binding differs")
            bundle, checkpoint = restore_complete_stage_a_bundle(
                case=case,
                run_manifest=self._run_manifest,
                store=self._store,
            )
            if (
                selected.stage_a_artifact_hash != bundle.artifact_hash
                or selected.stage_a_checkpoint_hash != checkpoint.checkpoint_hash
            ):
                raise ResumeDriftError("partial Stage A selection binding differs")
            self._remember_stage_a(bundle)
            stage_a_results.append(
                StageAResult(
                    case_id=bundle.case_id,
                    t0=bundle.t0,
                    artifact_hash=bundle.artifact_hash,
                )
            )

        self._observer.start_heartbeat()
        try:
            schedule = run_causal_wave_stage_b_schedule(
                stage_a_results=tuple(stage_a_results),
                case_workers=self._run_manifest.manifest.concurrency.case_workers,
                history_snapshot=self._history_snapshot,
                stage_b=self._run_stage_b,
                commit_group=self._commit_group,
                circuit_breaker=self._breaker,
                stop_requested=self._stop_requested,
            )
            if schedule.shutdown_reason == "circuit_open":
                reason = schedule.circuit_reason or "circuit_open"
                self._refresh_progress(
                    current_group=read_progress_snapshot(self._run_root).current_group
                )
                self._observer.fail(error_code="retry_exhausted", message=reason)
                raise ProductionRunFatalError(f"retry_exhausted: {reason}")
            self._refresh_progress(
                current_group=None
                if schedule.shutdown_reason == "completed"
                else read_progress_snapshot(self._run_root).current_group
            )
            self._observer.stop_heartbeat()
        except ProductionRunFatalError:
            raise
        except Exception as error:
            code = _fatal_error_code(error)
            message = _safe_message(error, secrets=self._secrets)
            self._refresh_progress(
                current_group=read_progress_snapshot(self._run_root).current_group
            )
            with suppress(ValueError):
                self._observer.fail(error_code=code, message=message)
            raise ProductionRunFatalError(f"{code}: {type(error).__name__}") from error

        terminals = tuple(self._terminals[item.case_id] for item in schedule.terminal_results)
        predictions = tuple(item.prediction for item in terminals)
        unsigned = _partial_result_document(
            run_id=self._run_manifest.manifest.run_id,
            selection_hash=selection.selection_hash,
            schedule=schedule,
            predictions=predictions,
            terminals=terminals,
        )
        result = ProductionPartialResult(
            run_id=self._run_manifest.manifest.run_id,
            selection_hash=selection.selection_hash,
            schedule=schedule,
            predictions=predictions,
            terminals=terminals,
            result_hash=sha256_hex(canonical_json_bytes(unsigned)),
            checkpoint_store=self._store,
        )
        if not result.verify():  # pragma: no cover - construction invariant
            raise AssertionError("partial production result verification failed")
        if schedule.shutdown_reason == "completed":
            self._persist_partial_result(result=result, root=partial_root)
            marker = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "run_id": result.run_id,
                "selection_hash": result.selection_hash,
                "result_hash": result.result_hash,
                "terminal_count": len(result.terminals),
            }
            _write_immutable_bytes(marker_path, canonical_json_bytes(marker))
        return result

    def _persist_partial_result(
        self,
        *,
        result: ProductionPartialResult,
        root: Path,
    ) -> None:
        document = _partial_result_artifact_document(result)
        _write_immutable_bytes(
            root / "partial_run_result.json",
            canonical_json_bytes(document),
        )

    def _restore_partial_result(
        self,
        *,
        selection: PartialSelectionArtifact,
        root: Path,
    ) -> ProductionPartialResult:
        path = root / "partial_run_result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("partial result must be an object")
            if set(value) != {
                "schema_version",
                "run_id",
                "selection_hash",
                "schedule",
                "predictions",
                "terminals",
                "result_hash",
            }:
                raise ValueError("partial result schema is invalid")
            if value["schema_version"] != RUNNER_SCHEMA_VERSION:
                raise ValueError("partial result schema version is invalid")
            schedule = SchedulingResult.model_validate(value["schedule"])
            predictions = tuple(
                PredictionRecord.model_validate(item) for item in value["predictions"]
            )
            terminals = tuple(
                CaseTerminalBundle.model_validate(item) for item in value["terminals"]
            )
            result = ProductionPartialResult(
                run_id=cast(str, value["run_id"]),
                selection_hash=cast(str, value["selection_hash"]),
                schedule=schedule,
                predictions=predictions,
                terminals=terminals,
                result_hash=cast(str, value["result_hash"]),
                checkpoint_store=self._store,
            )
        except (OSError, TypeError, ValueError, ValidationError) as error:
            raise ResumeDriftError("completed partial result is unreadable") from error
        if (
            result.run_id != self._run_manifest.manifest.run_id
            or result.selection_hash != selection.selection_hash
            or not result.verify()
        ):
            raise ResumeDriftError("completed partial result hash is invalid")
        marker = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "run_id": result.run_id,
            "selection_hash": result.selection_hash,
            "result_hash": result.result_hash,
            "terminal_count": len(result.terminals),
        }
        if (root / "PARTIAL_COMPLETE").read_bytes() != canonical_json_bytes(marker):
            raise ResumeDriftError("partial completion marker binding is invalid")
        return result

    def _persist_completed_result(self, result: ProductionRunResult) -> None:
        document = _production_result_artifact_document(result)
        path = self._run_root / "run_result.json"
        data = canonical_json_bytes(document)
        if path.exists():
            if path.read_bytes() != data:
                raise ResumeDriftError("completed production result changed")
            return
        atomic_write_bytes(path, data)

    def _restore_completed_result(self) -> ProductionRunResult:
        path = self._run_root / "run_result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("run result must be an object")
            if set(value) != {
                "schema_version",
                "run_id",
                "inference_manifest_hash",
                "schedule",
                "predictions",
                "terminals",
                "result_hash",
            }:
                raise ValueError("run result schema is invalid")
            if value["schema_version"] != RUNNER_SCHEMA_VERSION:
                raise ValueError("run result schema version is invalid")
            schedule = SchedulingResult.model_validate(value["schedule"])
            predictions = tuple(
                PredictionRecord.model_validate(item) for item in value["predictions"]
            )
            terminals = tuple(
                CaseTerminalBundle.model_validate(item) for item in value["terminals"]
            )
            result = ProductionRunResult(
                run_id=cast(str, value["run_id"]),
                inference_manifest_hash=cast(str, value["inference_manifest_hash"]),
                schedule=schedule,
                predictions=predictions,
                terminals=terminals,
                result_hash=cast(str, value["result_hash"]),
                checkpoint_store=self._store,
            )
        except (OSError, TypeError, ValueError, ValidationError) as error:
            raise ResumeDriftError("completed production result is unreadable") from error
        if (
            result.run_id != self._run_manifest.manifest.run_id
            or result.inference_manifest_hash != self._inference_manifest_hash
            or not result.verify()
        ):
            raise ResumeDriftError("completed production result hash is invalid")
        return result

    def _bind_inference_manifest(self) -> None:
        path = self._run_root / "inference_manifest.json"
        data = canonical_json_bytes(self._inference_manifest.model_dump(mode="json"))
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("inference manifest drift detected for existing run")
            return
        atomic_write_bytes(path, data)

    def _run_stage_a(self, scheduled: ScheduledCase) -> StageAResult:
        case = self._case(scheduled.case_id)
        recovery = self._store.recover_case(case_id=case.case_id)
        if recovery.failure is not None:
            failure_bundle = self._restore_failure_bundle(case)
            self._remember_stage_a(failure_bundle)
            return StageAResult(
                case_id=case.case_id,
                t0=case.t0,
                artifact_hash=failure_bundle.artifact_hash,
            )

        stage = "telemetry"
        try:
            telemetry = load_case_telemetry(case)
            case_hash = _model_hash(case)
            self._persist_telemetry(case=case, case_hash=case_hash, telemetry=telemetry)
            replay = self._store.replay_stage(
                case_id=case.case_id,
                stage="stage_a_complete",
                input_hashes=(
                    self._run_manifest.manifest_hash,
                    case_hash,
                    telemetry.telemetry_hash,
                ),
            )
            if replay.hit:
                output = _required_output(replay.output)
                replayed_bundle = StageABundle.model_validate(output["bundle"])
                if not replayed_bundle.verify():
                    raise ResumeDriftError("replayed Stage A bundle hash is invalid")
                self._metrics.record_stage(stage="stage_a_complete", replayed=True)
                self._remember_stage_a(replayed_bundle)
                self._refresh_progress(current_group=str(case.t0))
                return StageAResult(
                    case_id=case.case_id,
                    t0=case.t0,
                    artifact_hash=replayed_bundle.artifact_hash,
                )
            _require_stage_miss(replay.code, stage="stage_a_complete")

            checkpoint = create_user_manager_checkpoint(case_id=case.case_id)
            degradation_codes: list[str] = []
            stage = "controls_frozen"
            controls, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.FREEZE_CONTROLS,
                checkpoint=checkpoint,
                input_artifact_hash=case_hash,
                input_hashes=(
                    self._run_manifest.manifest_hash,
                    case_hash,
                    telemetry.telemetry_hash,
                ),
                artifact_type=FrozenCaseControls,
                builder=lambda: _freeze_controls(case=case, manifest=self._run_manifest),
            )
            _append_code(degradation_codes, degraded)

            stage = "scope_fixed"
            alarm_context, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.ALARM_CONTEXT_ADAPTER,
                checkpoint=checkpoint,
                input_artifact_hash=_model_hash(controls),
                input_hashes=(controls.artifact_hash, telemetry.telemetry_hash),
                artifact_type=AlarmContextArtifact,
                builder=lambda: _build_alarm_context(telemetry),
            )
            _append_code(degradation_codes, degraded)

            stage = "candidates_ready"
            selection, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.CANDIDATE_BUILDER,
                checkpoint=checkpoint,
                input_artifact_hash=_model_hash(alarm_context),
                input_hashes=(alarm_context.artifact_hash,),
                artifact_type=CandidateSelection,
                builder=lambda: build_candidates(
                    case_id=case.case_id,
                    dataset=case.dataset,
                    alarm_service=alarm_context.alarm.service,
                    topology=alarm_context.topology,
                    evidence=alarm_context.evidence,
                ),
            )
            _append_code(degradation_codes, degraded)

            stage = "prelim_graph_ready"
            preliminary, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.EVIDENCE_INITIALIZER,
                checkpoint=checkpoint,
                input_artifact_hash=_model_hash(selection),
                input_hashes=(_model_hash(selection), alarm_context.artifact_hash),
                artifact_type=PreliminaryGraphArtifact,
                builder=lambda: initialize_preliminary_graph(
                    selection=selection,
                    evidence=alarm_context.evidence,
                    topology=alarm_context.topology,
                    rank_input_fingerprint=telemetry.telemetry_hash,
                ),
            )
            _append_code(degradation_codes, degraded)

            stage = "payload_ready"
            payload, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.CONTEXT_DISTILLER,
                checkpoint=checkpoint,
                input_artifact_hash=_model_hash(preliminary),
                input_hashes=(preliminary.graph_hash,),
                artifact_type=SemanticPayloadArtifact,
                builder=lambda: distill_context(preliminary),
            )
            _append_code(degradation_codes, degraded)

            stage = "semantic_agent"
            submission, semantic_input_hash, semantic_failure = self._run_semantic_agent(
                case=case,
                payload=payload,
            )
            _append_code(degradation_codes, semantic_failure)

            stage = "facts_validated"
            semantic, checkpoint, degraded = self._run_user_stage(
                case=case,
                stage=stage,
                tool=UserManagerTool.VALIDATE_SEMANTIC_FACTS,
                checkpoint=checkpoint,
                input_artifact_hash=semantic_input_hash,
                input_hashes=(payload.payload_hash, semantic_input_hash),
                artifact_type=SemanticStageArtifact,
                builder=lambda: (
                    complete_semantic_fact_stage(
                        preliminary=preliminary,
                        payload=payload,
                        response=submission.model_dump(mode="json"),
                    )
                    if submission is not None
                    else degrade_semantic_fact_stage(
                        preliminary=preliminary,
                        payload=payload,
                        failure_code=semantic_failure or "semantic_agent_failed",
                    )
                ),
            )
            _append_code(degradation_codes, degraded)
            if semantic.degraded and semantic.degradation_code is not None:
                degradation_codes.append(semantic.degradation_code)

            embedding = tuple(float(value) for value in self._embed(payload))
            _validate_embedding(embedding)
            completed_bundle = _stage_a_bundle(
                case=case,
                telemetry=telemetry,
                controls=controls,
                alarm_context=alarm_context,
                selection=selection,
                preliminary=preliminary,
                payload=payload,
                semantic=semantic,
                embedding=embedding,
                checkpoint=checkpoint,
                degradation_codes=tuple(degradation_codes),
            )
            self._store.write_stage(
                case_id=case.case_id,
                stage="stage_a_complete",
                input_hashes=(
                    self._run_manifest.manifest_hash,
                    case_hash,
                    telemetry.telemetry_hash,
                ),
                output={"bundle": completed_bundle.model_dump(mode="json")},
            )
            self._metrics.record_stage(stage="stage_a_complete", replayed=False)
            self._remember_stage_a(completed_bundle)
            self._refresh_progress(current_group=str(case.t0))
            return StageAResult(
                case_id=case.case_id,
                t0=case.t0,
                artifact_hash=completed_bundle.artifact_hash,
            )
        except (RetryExhaustedError, FatalProviderRequestError, CircuitOpenError, ResumeDriftError):
            raise
        except Exception as error:
            code = "telemetry_validation_failed" if stage == "telemetry" else "case_pipeline_failed"
            bundle = self._case_failure(case=case, stage=stage, code=code, error=error)
            self._remember_stage_a(bundle)
            self._refresh_progress(current_group=str(case.t0))
            return StageAResult(
                case_id=case.case_id,
                t0=case.t0,
                artifact_hash=bundle.artifact_hash,
            )

    def _run_stage_b(self, result: StageAResult, history_hash: str) -> TerminalCaseResult:
        stage_a = self._stage_a[result.case_id]
        snapshot = self._snapshots[history_hash]
        if isinstance(stage_a, StageAFailureBundle):
            terminal = _failed_terminal(stage_a=stage_a, history_hash=history_hash)
            self._write_terminal_files(terminal)
            self._remember_terminal(terminal)
            self._refresh_progress(current_group=str(stage_a.t0))
            return _terminal_result(terminal)

        replay = self._store.replay_stage(
            case_id=stage_a.case_id,
            stage="terminal",
            input_hashes=(stage_a.artifact_hash, history_hash),
            history_snapshot_hash=history_hash,
        )
        if replay.hit:
            output = _required_output(replay.output)
            terminal = CaseTerminalBundle.model_validate(output["terminal"])
            if not terminal.verify():
                raise ResumeDriftError("replayed terminal bundle hash is invalid")
            self._metrics.record_stage(stage="terminal", replayed=True)
            self._write_terminal_files(terminal)
            self._remember_terminal(terminal)
            self._refresh_progress(current_group=str(stage_a.t0))
            return _terminal_result(terminal)
        _require_stage_miss(replay.code, stage="terminal")

        warnings = set(stage_a.alarm_context.warnings)
        warnings.update(stage_a.selection.warnings)
        warnings.update(stage_a.degradation_codes)
        checkpoint = stage_a.user_checkpoint
        ranking, checkpoint, ranking_degraded = self._run_user_stage(
            case=self._case(stage_a.case_id),
            stage="ranking",
            tool=UserManagerTool.EVIDENCE_RANKER,
            checkpoint=checkpoint,
            input_artifact_hash=_model_hash(stage_a.semantic),
            input_hashes=(stage_a.artifact_hash, stage_a.semantic.artifact_hash),
            history_snapshot_hash=history_hash,
            artifact_type=RankingBundleArtifact,
            builder=lambda: _rank_case(
                stage_a=stage_a,
                snapshot=snapshot,
                warnings=tuple(sorted(warnings)),
            ),
            degraded_builder=lambda code: _rank_case(
                stage_a=stage_a,
                snapshot=snapshot,
                warnings=tuple(sorted({*warnings, code})),
            ),
        )
        if ranking_degraded is not None:
            warnings.add(ranking_degraded)
        report, checkpoint, report_degraded = self._run_report_stage(
            case_id=stage_a.case_id,
            checkpoint=checkpoint,
            context=ranking.context,
        )
        if report_degraded is not None:
            warnings.add(report_degraded)
        degraded = bool(
            stage_a.degradation_codes
            or stage_a.semantic.degraded
            or ranking_degraded
            or report.source == "deterministic_fallback"
        )
        prediction = PredictionRecord(
            case_id=stage_a.case_id,
            status="degraded" if degraded else "success",
            top_k=ranking.final.ranking.top_k,
            candidate_services=tuple(
                candidate.service for candidate in ranking.final.ranking.candidates
            ),
            verified_context_hash=ranking.context.context_hash,
        )
        historical_case = HistoricalCase(
            case_id=stage_a.case_id,
            t0=stage_a.t0,
            evidence_graph_hash=stage_a.preliminary.graph_hash,
            semantic_payload_hash=stage_a.payload.payload_hash,
            embedding=stage_a.embedding,
            frozen_scores=tuple(
                HistoricalServiceScore(service=item.service, score=item.score)
                for item in ranking.final.ranking.candidates
            ),
        )
        terminal = _successful_terminal(
            stage_a=stage_a,
            history_hash=history_hash,
            prediction=prediction,
            ranking=ranking,
            report=report,
            checkpoint=checkpoint,
            historical_case=historical_case,
        )
        self._store.write_stage(
            case_id=stage_a.case_id,
            stage="terminal",
            input_hashes=(stage_a.artifact_hash, history_hash),
            history_snapshot_hash=history_hash,
            output={"terminal": terminal.model_dump(mode="json")},
        )
        self._metrics.record_stage(stage="terminal", replayed=False)
        self._write_terminal_files(terminal)
        self._remember_terminal(terminal)
        self._refresh_progress(current_group=str(stage_a.t0))
        return _terminal_result(terminal)

    def _run_user_stage(
        self,
        *,
        case: InferenceCase,
        stage: str,
        tool: UserManagerTool,
        checkpoint: UserManagerCheckpoint,
        input_artifact_hash: str,
        input_hashes: tuple[str, ...],
        artifact_type: type[TArtifact],
        builder: Callable[[], TArtifact],
        degraded_builder: Callable[[str], TArtifact] | None = None,
        history_snapshot_hash: str | None = None,
    ) -> tuple[TArtifact, UserManagerCheckpoint, str | None]:
        replay = self._store.replay_stage(
            case_id=case.case_id,
            stage=stage,
            input_hashes=input_hashes,
            history_snapshot_hash=history_snapshot_hash,
        )
        if replay.hit:
            output = _required_output(replay.output)
            artifact = artifact_type.model_validate(output["artifact"])
            restored = restore_user_manager_checkpoint(output["user_checkpoint"])
            degradation = cast(str | None, output.get("degradation_code"))
            self._metrics.record_stage(stage=stage, replayed=True)
            return artifact, restored, degradation
        _require_stage_miss(replay.code, stage=stage)

        built: list[TArtifact] = []

        def handler(_: UserManagerTool, __: dict[str, Any]) -> StageToolOutput:
            if not built:
                built.append(builder())
            artifact = built[0]
            return StageToolOutput(
                artifact_hash=_model_hash(artifact),
                result={"stage": stage, "artifact_hash": _model_hash(artifact)},
            )

        degradation_code: str | None = None
        agent_document: dict[str, Any] | None
        try:
            step = run_user_manager_tool_step(
                checkpoint=checkpoint,
                input_artifact_hash=input_artifact_hash,
                complete=self._completion(case_id=case.case_id, stage=stage),
                handler=handler,
            )
            artifact = built[0]
            updated = step.checkpoint
            agent_document = _agent_run_document(step.agent_run)
        except (
            RetryExhaustedError,
            FatalProviderRequestError,
            CircuitOpenError,
            ResumeDriftError,
        ):
            raise
        except Exception:
            degradation_code = f"user_manager_{stage}_failed"
            artifact = (
                degraded_builder(degradation_code)
                if degraded_builder is not None
                else built[0]
                if built
                else builder()
            )
            updated = apply_user_manager_tool(
                checkpoint=checkpoint,
                tool=tool,
                output_artifact_hash=_model_hash(artifact),
            )
            agent_document = None

        self._store.write_stage(
            case_id=case.case_id,
            stage=stage,
            input_hashes=input_hashes,
            history_snapshot_hash=history_snapshot_hash,
            output={
                "artifact": artifact.model_dump(mode="json"),
                "user_checkpoint": updated.model_dump(mode="json"),
                "agent_run": agent_document,
                "degradation_code": degradation_code,
            },
        )
        self._metrics.record_stage(stage=stage, replayed=False)
        self._observer.emit_event(
            kind="stage_completed",
            level="warning" if degradation_code else "info",
            message=degradation_code or "stage completed",
            case_id=case.case_id,
            stage=stage,
            usage=self._metrics.usage,
        )
        return artifact, updated, degradation_code

    def _run_semantic_agent(
        self,
        *,
        case: InferenceCase,
        payload: SemanticPayloadArtifact,
    ) -> tuple[SemanticFactSubmission | None, str, str | None]:
        replay = self._store.replay_stage(
            case_id=case.case_id,
            stage="semantic_agent",
            input_hashes=(payload.payload_hash,),
        )
        if replay.hit:
            output = _required_output(replay.output)
            raw_submission = output.get("submission")
            submission = (
                SemanticFactSubmission.model_validate(raw_submission)
                if raw_submission is not None
                else None
            )
            semantic_hash = cast(str, output["semantic_input_hash"])
            replay_failure_code = cast(str | None, output.get("failure_code"))
            self._metrics.record_stage(stage="semantic_agent", replayed=True)
            return submission, semantic_hash, replay_failure_code
        _require_stage_miss(replay.code, stage="semantic_agent")

        failure_code: str | None = None
        agent_document: dict[str, Any] | None = None
        try:
            loop = run_semantic_fact_agent(
                artifact=payload,
                complete=self._completion(case_id=case.case_id, stage="semantic_agent"),
            )
            submission = loop.submission
            semantic_hash = loop.submission_hash
            agent_document = _agent_run_document(loop.agent_run)
        except (
            RetryExhaustedError,
            FatalProviderRequestError,
            CircuitOpenError,
            ResumeDriftError,
        ):
            raise
        except Exception:
            submission = None
            failure_code = "semantic_agent_failed"
            semantic_hash = sha256_hex(
                canonical_json_bytes(
                    {"payload_hash": payload.payload_hash, "failure_code": failure_code}
                )
            )
        self._store.write_stage(
            case_id=case.case_id,
            stage="semantic_agent",
            input_hashes=(payload.payload_hash,),
            output={
                "submission": submission.model_dump(mode="json") if submission else None,
                "semantic_input_hash": semantic_hash,
                "agent_run": agent_document,
                "failure_code": failure_code,
            },
        )
        self._metrics.record_stage(stage="semantic_agent", replayed=False)
        return submission, semantic_hash, failure_code

    def _run_report_stage(
        self,
        *,
        case_id: str,
        checkpoint: UserManagerCheckpoint,
        context: VerifiedRCLContextArtifact,
    ) -> tuple[VerifiedAgentReportArtifact, UserManagerCheckpoint, str | None]:
        inputs = (context.artifact_hash, checkpoint.checkpoint_hash)
        replay = self._store.replay_stage(
            case_id=case_id,
            stage="report_written",
            input_hashes=inputs,
        )
        if replay.hit:
            output = _required_output(replay.output)
            report = VerifiedAgentReportArtifact.model_validate(output["report"])
            restored = restore_user_manager_checkpoint(output["user_checkpoint"])
            degradation = cast(str | None, output.get("degradation_code"))
            self._metrics.record_stage(stage="report_written", replayed=True)
            return report, restored, degradation
        _require_stage_miss(replay.code, stage="report_written")

        degradation_code: str | None = None
        agent_document: dict[str, Any] | None
        try:
            result = run_user_manager_report_agent(
                checkpoint=checkpoint,
                context=context,
                complete=self._completion(case_id=case_id, stage="report_written"),
            )
            report = result.report
            updated = result.checkpoint
            agent_document = _agent_run_document(result.agent_run)
        except RetryExhaustedError as error:
            if not _is_recoverable_report_exhaustion(
                stage="report_written",
                error=error,
            ):
                raise
            degradation_code = "report_agent_failed"
            report = finalize_user_manager_report(
                context=context,
                submission=None,
                degradation_code=degradation_code,
            )
            updated = apply_user_manager_tool(
                checkpoint=checkpoint,
                tool=UserManagerTool.WRITE_REPORT,
                output_artifact_hash=report.artifact_hash,
            )
            agent_document = None
        except (FatalProviderRequestError, CircuitOpenError, ResumeDriftError):
            raise
        except Exception:
            degradation_code = "report_agent_failed"
            report = finalize_user_manager_report(
                context=context,
                submission=None,
                degradation_code=degradation_code,
            )
            updated = apply_user_manager_tool(
                checkpoint=checkpoint,
                tool=UserManagerTool.WRITE_REPORT,
                output_artifact_hash=report.artifact_hash,
            )
            agent_document = None
        self._store.write_stage(
            case_id=case_id,
            stage="report_written",
            input_hashes=inputs,
            output={
                "report": report.model_dump(mode="json"),
                "user_checkpoint": updated.model_dump(mode="json"),
                "agent_run": agent_document,
                "degradation_code": degradation_code,
            },
        )
        self._metrics.record_stage(stage="report_written", replayed=False)
        return report, updated, degradation_code

    def _completion(self, *, case_id: str, stage: str) -> CompletionFunction:
        def complete(request: dict[str, Any]) -> dict[str, Any]:
            request_hash = sha256_hex(canonical_json_bytes(request))
            cache_path = self._run_root / "llm_cache" / f"{request_hash}.json"
            with self._llm_request_lock:
                self._breaker.require_closed()
                if cache_path.exists():
                    try:
                        entry = LLMCacheEntry.model_validate_json(
                            cache_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError, ValidationError) as error:
                        raise ResumeDriftError("same-run LLM cache is unreadable") from error
                    receipt = self._store.replay_llm_response(
                        entry=entry,
                        request_hash=request_hash,
                    )
                    if not receipt.hit or receipt.response is None:
                        raise ResumeDriftError("same-run LLM cache replay was rejected")
                    self._metrics.record_cache_hit()
                    return receipt.response

                exhausted = self._store.replay_stage(
                    case_id=case_id,
                    stage="llm_retry_exhausted",
                    input_hashes=(request_hash,),
                )
                if exhausted.hit:
                    snapshot = _llm_exhaustion_snapshot(
                        _required_output(exhausted.output),
                        request_hash=request_hash,
                        stage=stage,
                    )
                    replayed_error = RetryExhaustedError(snapshot.attempts)
                    if not _is_recoverable_report_exhaustion(
                        stage=stage,
                        error=replayed_error,
                    ):
                        self._breaker.open("retry_exhausted")
                    raise replayed_error
                _require_stage_miss(exhausted.code, stage="llm_retry_exhausted")
                ledger = self._store.llm_attempt_ledger(
                    request_hash=request_hash,
                    case_id=case_id,
                    stage=stage,
                )
                initial_records = (
                    tuple(item.attempt for item in ledger.results)
                    if ledger is not None
                    else ()
                )
                if initial_records and initial_records[-1].outcome == "success":
                    successful_result = ledger.results[-1] if ledger is not None else None
                    if successful_result is None or successful_result.response is None:
                        raise ResumeDriftError(
                            "successful LLM attempt result omitted its response"
                        )
                    self._store.write_llm_response(
                        request_hash=request_hash,
                        response=successful_result.response,
                        attempts=initial_records,
                        cumulative_usage=successful_result.cumulative_usage,
                    )
                    return successful_result.response

                def attempt(_: int) -> AttemptResult[dict[str, Any]]:
                    started = time.monotonic()
                    try:
                        with self._llm_semaphore:
                            raw = self._complete(request)
                    finally:
                        self._metrics.provider_attempt(
                            latency_seconds=max(0.0, time.monotonic() - started)
                        )
                    if not isinstance(raw, Mapping):
                        raise RetryableParsingError(
                            "provider response must be an object",
                            usage=AttemptUsage.unavailable(),
                        )
                    value = dict(raw)
                    usage = _provider_attempt_usage(value)
                    self._metrics.provider_tokens(usage)
                    try:
                        _validate_agent_response(request=request, response_value=value)
                    except (RetryableParsingError, RetryableRepairError) as error:
                        error.bind_usage(usage)
                        raise
                    return AttemptResult(value=value, usage=usage)

                def checkpoint(records: tuple[AttemptRecord, ...]) -> None:
                    usage_snapshot = self._metrics.snapshot()
                    exhaustion = LLMExhaustionSnapshot(
                        request_hash=request_hash,
                        stage=stage,
                        attempts=records,
                        cumulative_usage=LLMCumulativeUsage(
                            attempts=usage_snapshot.attempts,
                            input_tokens=usage_snapshot.input_tokens,
                            output_tokens=usage_snapshot.output_tokens,
                            latency_seconds=usage_snapshot.latency_seconds,
                        ),
                    )
                    self._store.write_stage(
                        case_id=case_id,
                        stage="llm_retry_exhausted",
                        input_hashes=(request_hash,),
                        output=exhaustion.model_dump(mode="json"),
                    )

                def reserve_attempt(attempt_number: int) -> None:
                    usage_snapshot = self._metrics.snapshot()
                    self._store.reserve_llm_attempt(
                        case_id=case_id,
                        request_hash=request_hash,
                        stage=stage,
                        attempt_number=attempt_number,
                        cumulative_usage=LLMCumulativeUsage(
                            attempts=usage_snapshot.attempts + 1,
                            input_tokens=usage_snapshot.input_tokens,
                            output_tokens=usage_snapshot.output_tokens,
                            latency_seconds=usage_snapshot.latency_seconds,
                        ),
                    )

                def record_attempt(
                    record: AttemptRecord,
                    response_value: dict[str, Any] | None,
                ) -> None:
                    usage_snapshot = self._metrics.snapshot()
                    self._store.write_llm_attempt_result(
                        case_id=case_id,
                        request_hash=request_hash,
                        stage=stage,
                        attempt=record,
                        cumulative_usage=LLMCumulativeUsage(
                            attempts=usage_snapshot.attempts,
                            input_tokens=usage_snapshot.input_tokens,
                            output_tokens=usage_snapshot.output_tokens,
                            latency_seconds=usage_snapshot.latency_seconds,
                        ),
                        response=response_value,
                    )

                if len(initial_records) == 5:
                    checkpoint(initial_records)
                    interrupted_error = RetryExhaustedError(initial_records)
                    if not _is_recoverable_report_exhaustion(
                        stage=stage,
                        error=interrupted_error,
                    ):
                        self._breaker.open("retry_exhausted")
                    raise interrupted_error

                request_breaker = self._breaker
                if stage == "report_written":
                    request_breaker = CircuitBreaker()
                try:
                    execution: RequestExecution[dict[str, Any]] = execute_llm_request(
                        attempt,
                        sleep=self._retry_sleep,
                        checkpoint=checkpoint,
                        circuit_breaker=request_breaker,
                        initial_records=initial_records,
                        reserve_attempt=reserve_attempt,
                        record_attempt=record_attempt,
                    )
                except FatalProviderRequestError:
                    self._breaker.open("provider_fatal")
                    raise
                except RetryExhaustedError as error:
                    if request_breaker is not self._breaker and not (
                        _is_recoverable_report_exhaustion(stage=stage, error=error)
                    ):
                        self._breaker.open("retry_exhausted")
                    raise
                usage_snapshot = self._metrics.snapshot()
                self._store.write_llm_response(
                    request_hash=request_hash,
                    response=execution.value,
                    attempts=execution.records,
                    cumulative_usage=LLMCumulativeUsage(
                        attempts=usage_snapshot.attempts,
                        input_tokens=usage_snapshot.input_tokens,
                        output_tokens=usage_snapshot.output_tokens,
                        latency_seconds=usage_snapshot.latency_seconds,
                    ),
                )
                return execution.value

        return complete

    def _persist_telemetry(
        self,
        *,
        case: InferenceCase,
        case_hash: str,
        telemetry: CaseTelemetry,
    ) -> None:
        replay = self._store.replay_stage(
            case_id=case.case_id,
            stage="telemetry_loaded",
            input_hashes=(case_hash,),
        )
        if replay.hit:
            output = _required_output(replay.output)
            persisted = CaseTelemetry.model_validate(output["telemetry"])
            if persisted != telemetry:
                raise ResumeDriftError("normalized telemetry changed within one run")
            self._metrics.record_stage(stage="telemetry_loaded", replayed=True)
            return
        _require_stage_miss(replay.code, stage="telemetry_loaded")
        self._store.write_stage(
            case_id=case.case_id,
            stage="telemetry_loaded",
            input_hashes=(case_hash,),
            output={"telemetry": telemetry.model_dump(mode="json")},
        )
        self._metrics.record_stage(stage="telemetry_loaded", replayed=False)

    def _history_snapshot(self, t0: int) -> str:
        snapshot = self._history.snapshot_for(t0=t0)
        with self._state_lock:
            self._snapshots[snapshot.snapshot_hash] = snapshot
        self._observer.emit_event(
            kind="history_snapshot",
            level="info",
            message="causal history snapshot frozen",
            stage="ranking",
            usage=self._metrics.usage,
        )
        return snapshot.snapshot_hash

    def _commit_group(
        self,
        t0: int,
        history_hash: str,
        results: tuple[TerminalCaseResult, ...],
    ) -> str:
        terminals = tuple(self._terminals[item.case_id] for item in results)
        cases = tuple(
            item.historical_case for item in terminals if item.historical_case is not None
        )
        tombstones = tuple(
            item.history_tombstone for item in terminals if item.history_tombstone is not None
        )
        artifact = self._history.commit_group(t0=t0, cases=cases, tombstones=tombstones)
        self._observer.emit_event(
            kind="history_group_committed",
            level="info",
            message=f"timestamp group {t0} committed from snapshot {history_hash}",
            stage="history_commit",
            usage=self._metrics.usage,
        )
        self._refresh_progress(current_group=str(t0))
        return artifact.group_hash

    def _case_failure(
        self,
        *,
        case: InferenceCase,
        stage: str,
        code: str,
        error: Exception,
    ) -> StageAFailureBundle:
        failure = FailureRecord(
            case_id=case.case_id,
            stage=stage,
            error_code=code,
            message=_safe_message(error, secrets=self._secrets),
        )
        self._store.write_failure_tombstone(
            case_id=case.case_id,
            stage=_stable_stage(stage),
            failure_code=code,
        )
        _write_immutable_model(self._run_root / "failures" / f"{case.case_id}.json", failure)
        unsigned = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case.case_id,
            "t0": case.t0,
            "failure": failure.model_dump(mode="json"),
        }
        bundle = StageAFailureBundle(
            case_id=case.case_id,
            t0=case.t0,
            failure=failure,
            artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )
        self._observer.emit_event(
            kind="case_stage_failed",
            level="error",
            message=failure.message,
            case_id=case.case_id,
            stage=stage,
            usage=self._metrics.usage,
        )
        return bundle

    def _restore_failure_bundle(self, case: InferenceCase) -> StageAFailureBundle:
        path = self._run_root / "failures" / f"{case.case_id}.json"
        failure = FailureRecord.model_validate_json(path.read_text(encoding="utf-8"))
        unsigned = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case.case_id,
            "t0": case.t0,
            "failure": failure.model_dump(mode="json"),
        }
        return StageAFailureBundle(
            case_id=case.case_id,
            t0=case.t0,
            failure=failure,
            artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
        )

    def _write_terminal_files(self, terminal: CaseTerminalBundle) -> None:
        _write_immutable_model(
            self._run_root / "predictions" / f"{terminal.case_id}.json",
            terminal.prediction,
        )
        if terminal.failure is not None:
            _write_immutable_model(
                self._run_root / "failures" / f"{terminal.case_id}.json",
                terminal.failure,
            )

    def _refresh_progress(self, *, current_group: str | None) -> None:
        prediction_root = self._run_root / "predictions"
        records = (
            tuple(
                PredictionRecord.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(prediction_root.glob("*.json"))
            )
            if prediction_root.exists()
            else ()
        )
        statuses = CounterLike(record.status for record in records)
        self._observer.update_progress(
            completed=len(records),
            succeeded=statuses["success"],
            degraded=statuses["degraded"],
            failed=statuses["failed"],
            current_group=current_group,
            stage_counts=self._metrics.stage_counts,
            cache_hits=self._metrics.cache_hits,
            usage=self._metrics.usage,
        )

    def _remember_stage_a(self, bundle: StageABundleValue) -> None:
        with self._state_lock:
            self._stage_a[bundle.case_id] = bundle

    def _remember_terminal(self, bundle: CaseTerminalBundle) -> None:
        with self._state_lock:
            self._terminals[bundle.case_id] = bundle

    def _case(self, case_id: str) -> InferenceCase:
        return next(case for case in self._inference_manifest.cases if case.case_id == case_id)


class CounterLike(defaultdict[str, int]):
    def __init__(self, values: Iterable[str]) -> None:
        super().__init__(int)
        for value in values:
            self[value] += 1


def run_production_dataset(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    run_root: Path,
    complete: CompletionFunction,
    embed: MetricEmbedder,
    retry_sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = lambda: False,
    secrets: Iterable[str] = (),
) -> ProductionRunResult:
    """Run every selected telemetry-only case through the production V1 path."""

    runner = _ProductionRunner(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        run_root=run_root,
        complete=complete,
        embed=embed,
        retry_sleep=retry_sleep,
        stop_requested=stop_requested,
        secrets=secrets,
    )
    return runner.run()


def run_production_partial(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
    run_root: Path,
    selection: PartialSelectionArtifact,
    complete: CompletionFunction,
    retry_sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = lambda: False,
    secrets: Iterable[str] = (),
) -> ProductionPartialResult:
    """Run only Stage B for one immutable same-run partial selection."""

    runner = _ProductionRunner(
        inference_manifest=inference_manifest,
        run_manifest=run_manifest,
        run_root=run_root,
        complete=complete,
        embed=_partial_stage_a_forbidden,
        retry_sleep=retry_sleep,
        stop_requested=stop_requested,
        secrets=secrets,
    )
    return runner.run_partial(selection=selection)


def _validate_run_inputs(
    *,
    inference_manifest: InferenceManifest,
    run_manifest: RunManifestArtifact,
) -> None:
    if not run_manifest.verify():
        raise ValueError("run manifest verification failed")
    manifest = run_manifest.manifest
    if inference_manifest.method_version != manifest.method_version:
        raise ValueError("inference and run method versions do not match")
    if inference_manifest.dataset != manifest.dataset:
        raise ValueError("inference and run datasets do not match")
    case_ids = tuple(case.case_id for case in inference_manifest.cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("inference case identities must be unique")


def _freeze_controls(*, case: InferenceCase, manifest: RunManifestArtifact) -> FrozenCaseControls:
    unsigned = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "case_id": case.case_id,
        "dataset": case.dataset,
        "t0": case.t0,
        "window_seconds": WINDOW_SECONDS,
        "mj": MJ_BY_DATASET[case.dataset],
        "evidence_threshold": EVIDENCE_THRESHOLD,
        "component_weights": [1.0, 1.0, 1.0, 1.0],
        "run_manifest_hash": manifest.manifest_hash,
        "configuration_json": manifest.manifest.configuration_json,
    }
    return FrozenCaseControls(
        case_id=case.case_id,
        dataset=case.dataset,
        t0=case.t0,
        mj=MJ_BY_DATASET[case.dataset],
        run_manifest_hash=manifest.manifest_hash,
        configuration_json=manifest.manifest.configuration_json,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _build_alarm_context(telemetry: CaseTelemetry) -> AlarmContextArtifact:
    signal_series = [
        SignalSeries(
            service=series.service,
            signal=series.signal,
            baseline=tuple(sample.value for sample in series.baseline),
            fault=tuple(
                TimedValue(timestamp=sample.timestamp_ns, value=sample.value)
                for sample in series.fault
            ),
        )
        for series in telemetry.metric_series
    ]
    alarm = detect_alarm(signal_series)
    evidence: list[EvidenceRecord] = []
    for series in telemetry.metric_series:
        deviation = compute_metric_deviation(
            baseline=(sample.value for sample in series.baseline),
            fault=(sample.value for sample in series.fault),
        )
        if not deviation.retained or deviation.anomaly_score is None:
            continue
        first_anomalous_at = _first_anomalous_timestamp(series)
        evidence_payload = {
            "case_id": telemetry.case_id,
            "telemetry_hash": telemetry.telemetry_hash,
            "service": series.service,
            "signal": series.signal,
            "window_start": telemetry.t0,
            "window_end": telemetry.t0 + WINDOW_SECONDS,
        }
        evidence.append(
            EvidenceRecord(
                evidence_id=stable_identifier("evidence", evidence_payload),
                service=series.service,
                signal=series.signal,
                kind="metric_anomaly",
                window=TimeWindow(start=telemetry.t0, end=telemetry.t0 + WINDOW_SECONDS),
                weight=deviation.anomaly_score,
                verification_rule="paper_v1_metric_deviation_z_ge_3",
                observed={
                    "numeric_provenance": deviation.model_dump(mode="json"),
                    "fault_mean": deviation.fault_mean,
                    "max_fault_distance": deviation.max_fault_distance,
                    "ks_distance": deviation.ks_distance,
                    "z_residual": deviation.z_residual,
                    "z_shift": deviation.z_shift,
                    "z_ks": deviation.z_ks,
                    "max_deviation": deviation.max_deviation,
                },
                baseline={
                    "mean": deviation.baseline_mean,
                    "median": deviation.baseline_median,
                    "mad": deviation.baseline_mad,
                    "scale": deviation.scale,
                    "valid_count": deviation.baseline_valid_count,
                },
                first_anomalous_at=first_anomalous_at,
            )
        )
    topology = build_normal_topology(
        case_id=telemetry.case_id,
        t0=telemetry.t0,
        spans=tuple(
            TraceSpan(
                case_id=telemetry.case_id,
                timestamp_ns=span.timestamp_ns,
                trace_id=span.trace_id,
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                service=span.service,
            )
            for span in telemetry.normal_spans
        ),
    )
    warnings = set[str]()
    if alarm.fallback:
        warnings.add("alarm_global_maximum_fallback")
    if any(item.rejections for item in telemetry.coverage):
        warnings.add("telemetry_rows_rejected")
    unsigned = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "case_id": telemetry.case_id,
        "telemetry_hash": telemetry.telemetry_hash,
        "alarm": alarm.model_dump(mode="json"),
        "evidence": [
            item.model_dump(mode="json")
            for item in sorted(evidence, key=lambda item: item.evidence_id)
        ],
        "topology": topology.model_dump(mode="json"),
        "log_audit": telemetry.log_audit.model_dump(mode="json"),
        "warnings": sorted(warnings),
    }
    return AlarmContextArtifact(
        case_id=telemetry.case_id,
        telemetry_hash=telemetry.telemetry_hash,
        alarm=alarm,
        evidence=tuple(sorted(evidence, key=lambda item: item.evidence_id)),
        topology=topology,
        log_audit=telemetry.log_audit,
        warnings=tuple(sorted(warnings)),
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _first_anomalous_timestamp(series: Any) -> int | None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for sample in series.fault:
        grouped[sample.timestamp_ns].append(sample.value)
    prefix: list[float] = []
    for timestamp in sorted(grouped):
        prefix.extend(grouped[timestamp])
        deviation = compute_metric_deviation(
            baseline=(sample.value for sample in series.baseline),
            fault=prefix,
        )
        if deviation.retained:
            return timestamp
    return None


def _stage_a_bundle(
    *,
    case: InferenceCase,
    telemetry: CaseTelemetry,
    controls: FrozenCaseControls,
    alarm_context: AlarmContextArtifact,
    selection: CandidateSelection,
    preliminary: PreliminaryGraphArtifact,
    payload: SemanticPayloadArtifact,
    semantic: SemanticStageArtifact,
    embedding: tuple[float, ...],
    checkpoint: UserManagerCheckpoint,
    degradation_codes: tuple[str, ...],
) -> StageABundle:
    unsigned = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "case_id": case.case_id,
        "t0": case.t0,
        "telemetry_hash": telemetry.telemetry_hash,
        "controls": controls.model_dump(mode="json"),
        "alarm_context": alarm_context.model_dump(mode="json"),
        "selection": selection.model_dump(mode="json"),
        "preliminary": preliminary.model_dump(mode="json"),
        "payload": payload.model_dump(mode="json"),
        "semantic": semantic.model_dump(mode="json"),
        "embedding": list(embedding),
        "user_checkpoint": checkpoint.model_dump(mode="json"),
        "degradation_codes": sorted(set(degradation_codes)),
    }
    return StageABundle(
        case_id=case.case_id,
        t0=case.t0,
        telemetry_hash=telemetry.telemetry_hash,
        controls=controls,
        alarm_context=alarm_context,
        selection=selection,
        preliminary=preliminary,
        payload=payload,
        semantic=semantic,
        embedding=embedding,
        user_checkpoint=checkpoint,
        degradation_codes=tuple(sorted(set(degradation_codes))),
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _rank_case(
    *,
    stage_a: StageABundle,
    snapshot: HistorySnapshot,
    warnings: tuple[str, ...],
) -> RankingBundleArtifact:
    raw = compute_raw_epr(
        preliminary=stage_a.preliminary,
        guarded=stage_a.semantic.guarded_graph,
    )
    scc = score_scc_propagation(raw=raw, guarded=stage_a.semantic.guarded_graph)
    history = compute_history_scores(
        snapshot=snapshot,
        query_embedding=stage_a.embedding,
        candidate_services=tuple(item.service for item in raw.candidates),
    )
    final = assemble_final_ranking(raw=raw, scc=scc, history=history)
    context = freeze_verified_rcl_context(
        case_id=stage_a.case_id,
        final=final,
        guarded=stage_a.semantic.guarded_graph,
        scc=scc,
        history=history,
        warnings=warnings,
    )
    unsigned = {
        "raw": raw.model_dump(mode="json"),
        "scc": scc.model_dump(mode="json"),
        "history_snapshot_hash": snapshot.snapshot_hash,
        "history": history.model_dump(mode="json"),
        "final": final.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
    }
    return RankingBundleArtifact(
        raw=raw,
        scc=scc,
        history_snapshot_hash=snapshot.snapshot_hash,
        history=history,
        final=final,
        context=context,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _successful_terminal(
    *,
    stage_a: StageABundle,
    history_hash: str,
    prediction: PredictionRecord,
    ranking: RankingBundleArtifact,
    report: VerifiedAgentReportArtifact,
    checkpoint: UserManagerCheckpoint,
    historical_case: HistoricalCase,
) -> CaseTerminalBundle:
    unsigned = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "case_id": stage_a.case_id,
        "t0": stage_a.t0,
        "history_snapshot_hash": history_hash,
        "prediction": prediction.model_dump(mode="json"),
        "failure": None,
        "ranking": ranking.model_dump(mode="json"),
        "context": ranking.context.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "user_checkpoint": checkpoint.model_dump(mode="json"),
        "historical_case": historical_case.model_dump(mode="json"),
        "history_tombstone": None,
    }
    return CaseTerminalBundle(
        case_id=stage_a.case_id,
        t0=stage_a.t0,
        history_snapshot_hash=history_hash,
        prediction=prediction,
        ranking=ranking,
        context=ranking.context,
        report=report,
        user_checkpoint=checkpoint,
        historical_case=historical_case,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _failed_terminal(*, stage_a: StageAFailureBundle, history_hash: str) -> CaseTerminalBundle:
    prediction = PredictionRecord(
        case_id=stage_a.case_id,
        status="failed",
        top_k=(),
        candidate_services=(),
        failure_code=stage_a.failure.error_code,
    )
    tombstone = HistoryFailureTombstone(
        case_id=stage_a.case_id,
        t0=stage_a.t0,
        failure_code=stage_a.failure.error_code,
    )
    unsigned = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "case_id": stage_a.case_id,
        "t0": stage_a.t0,
        "history_snapshot_hash": history_hash,
        "prediction": prediction.model_dump(mode="json"),
        "failure": stage_a.failure.model_dump(mode="json"),
        "ranking": None,
        "context": None,
        "report": None,
        "user_checkpoint": None,
        "historical_case": None,
        "history_tombstone": tombstone.model_dump(mode="json"),
    }
    return CaseTerminalBundle(
        case_id=stage_a.case_id,
        t0=stage_a.t0,
        history_snapshot_hash=history_hash,
        prediction=prediction,
        failure=stage_a.failure,
        history_tombstone=tombstone,
        artifact_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )


def _terminal_result(terminal: CaseTerminalBundle) -> TerminalCaseResult:
    return TerminalCaseResult(
        case_id=terminal.case_id,
        t0=terminal.t0,
        history_snapshot_hash=terminal.history_snapshot_hash,
        status=terminal.prediction.status,
        artifact_hash=terminal.artifact_hash,
    )


def _validate_agent_response(
    *,
    request: Mapping[str, Any],
    response_value: Mapping[str, Any],
) -> None:
    transport = ChatCompletionsTransport()
    try:
        normalized = transport.normalize_response(response_value)
    except Exception as error:
        raise RetryableParsingError("provider response normalization failed") from error
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RetryableRepairError("agent request messages are invalid")
    last_message = messages[-1]
    terminal_turn = isinstance(last_message, Mapping) and last_message.get("role") == "tool"
    if terminal_turn:
        if normalized.tool_calls:
            raise RetryableRepairError("terminal agent turn repeated a tool call")
        return
    if len(normalized.tool_calls) != 1:
        raise RetryableRepairError("agent must make exactly one tool call")
    call = normalized.tool_calls[0]
    expected_tool = _expected_tool_name(request)
    if call.name != expected_tool:
        raise RetryableRepairError("agent called a tool outside the current allowlist")
    try:
        arguments = json.loads(call.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        _validate_tool_arguments(
            tool_name=expected_tool,
            arguments=arguments,
            messages=messages,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise RetryableRepairError("agent tool arguments failed deterministic repair") from error


def _expected_tool_name(request: Mapping[str, Any]) -> str:
    tools = request.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        raise RetryableRepairError("agent request must expose one tool")
    tool = tools[0]
    function = tool.get("function") if isinstance(tool, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(name, str) or not name:
        raise RetryableRepairError("agent request tool name is invalid")
    return name


def _validate_tool_arguments(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    messages: list[Any],
) -> None:
    if tool_name == "submit_semantic_facts":
        SemanticFactSubmission.model_validate(arguments)
        return
    if tool_name == "write_report":
        from nurse_rcl.agents.report import AgentReportSubmission

        AgentReportSubmission.model_validate(arguments)
        return
    user_message = messages[-1]
    content = user_message.get("content") if isinstance(user_message, Mapping) else None
    if not isinstance(content, str):
        raise ValueError("orchestration user message is invalid")
    document = json.loads(content)
    expected = {
        "checkpoint_hash": document["checkpoint_hash"],
        "input_artifact_hash": document["input_artifact_hash"],
    }
    if arguments != expected:
        raise ValueError("orchestration arguments do not match frozen hashes")


def _agent_run_document(run: AgentRunResult) -> dict[str, Any]:
    return {
        "content": run.content,
        "messages": list(run.messages),
        "usage": asdict(run.usage),
        "turns": run.turns,
        "audit_events": [asdict(item) for item in run.audit_events],
    }


def _production_result_document(
    *,
    run_id: str,
    inference_manifest_hash: str,
    schedule: SchedulingResult,
    predictions: tuple[PredictionRecord, ...],
    terminals: tuple[CaseTerminalBundle, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "inference_manifest_hash": inference_manifest_hash,
        "schedule": schedule.model_dump(mode="json"),
        "predictions": [item.model_dump(mode="json") for item in predictions],
        "terminal_artifact_hashes": [item.artifact_hash for item in terminals],
    }


def _production_result_artifact_document(result: ProductionRunResult) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": result.run_id,
        "inference_manifest_hash": result.inference_manifest_hash,
        "schedule": result.schedule.model_dump(mode="json"),
        "predictions": [item.model_dump(mode="json") for item in result.predictions],
        "terminals": [item.model_dump(mode="json") for item in result.terminals],
        "result_hash": result.result_hash,
    }


def _partial_result_document(
    *,
    run_id: str,
    selection_hash: str,
    schedule: SchedulingResult,
    predictions: tuple[PredictionRecord, ...],
    terminals: tuple[CaseTerminalBundle, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "selection_hash": selection_hash,
        "schedule": schedule.model_dump(mode="json"),
        "predictions": [item.model_dump(mode="json") for item in predictions],
        "terminal_artifact_hashes": [item.artifact_hash for item in terminals],
    }


def _partial_result_artifact_document(
    result: ProductionPartialResult,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": result.run_id,
        "selection_hash": result.selection_hash,
        "schedule": result.schedule.model_dump(mode="json"),
        "predictions": [item.model_dump(mode="json") for item in result.predictions],
        "terminals": [item.model_dump(mode="json") for item in result.terminals],
        "result_hash": result.result_hash,
    }


def _write_immutable_model(path: Path, model: StrictFrozenModel) -> None:
    data = canonical_json_bytes(model.model_dump(mode="json"))
    if path.exists():
        if path.read_bytes() != data:
            raise ResumeDriftError(f"immutable artifact changed: {path.name}")
        return
    atomic_write_bytes(path, data)


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ResumeDriftError(f"immutable artifact changed: {path.name}")
        return
    atomic_write_bytes(path, data)


def _required_output(output: dict[str, Any] | None) -> dict[str, Any]:
    if output is None:
        raise ResumeDriftError("checkpoint replay omitted its output")
    return output


def _require_stage_miss(code: str, *, stage: str) -> None:
    if code != "stage_not_found":
        raise ResumeDriftError(f"checkpoint replay rejected {stage}: {code}")


def _model_hash(model: StrictFrozenModel) -> str:
    return sha256_hex(canonical_json_bytes(model.model_dump(mode="json")))


def _append_code(codes: list[str], code: str | None) -> None:
    if code is not None:
        codes.append(code)


def _validate_embedding(embedding: tuple[float, ...]) -> None:
    if not embedding or any(not math.isfinite(item) for item in embedding):
        raise ValueError("embedding must contain finite values")
    if math.fsum(item * item for item in embedding) <= 0.0:
        raise ValueError("embedding must have non-zero norm")


def _partial_stage_a_forbidden(_: SemanticPayloadArtifact) -> tuple[float, ...]:
    raise AssertionError("partial continuation attempted to execute Stage A")


def _stable_stage(stage: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "_" for character in stage.casefold()
    )
    normalized = normalized.strip("_") or "case_pipeline"
    return normalized[:64]


def _safe_message(error: Exception, *, secrets: tuple[str, ...]) -> str:
    text = str(error).strip() or type(error).__name__
    return redact_sensitive_text(text, secrets=secrets)[:1024]


def _fatal_error_code(error: Exception) -> str:
    if isinstance(error, RetryExhaustedError):
        return "retry_exhausted"
    if isinstance(error, FatalProviderRequestError):
        return "provider_fatal"
    if isinstance(error, ResumeDriftError):
        return "resume_drift"
    if isinstance(error, CircuitOpenError):
        return "circuit_open"
    return "runner_failed"


def _is_recoverable_report_exhaustion(
    *,
    stage: str,
    error: RetryExhaustedError,
) -> bool:
    return stage == "report_written" and bool(error.records) and all(
        record.category in {"parse", "repair"} for record in error.records
    )


def _ensure_attempt_ledger_exhaustions(
    *,
    store: CheckpointStore,
    ledgers: tuple[LLMAttemptLedger, ...],
) -> None:
    for ledger in ledgers:
        if not ledger.results or ledger.results[-1].attempt.outcome != "exhausted":
            continue
        attempts = tuple(item.attempt for item in ledger.results)
        final_usage = ledger.results[-1].cumulative_usage
        replay = store.replay_stage(
            case_id=ledger.case_id,
            stage="llm_retry_exhausted",
            input_hashes=(ledger.request_hash,),
        )
        if replay.hit:
            _llm_exhaustion_snapshot(
                _required_output(replay.output),
                request_hash=ledger.request_hash,
                stage=ledger.stage,
            )
            continue
        _require_stage_miss(replay.code, stage="llm_retry_exhausted")
        snapshot = LLMExhaustionSnapshot(
            request_hash=ledger.request_hash,
            stage=ledger.stage,
            attempts=attempts,
            cumulative_usage=final_usage,
        )
        store.write_stage(
            case_id=ledger.case_id,
            stage="llm_retry_exhausted",
            input_hashes=(ledger.request_hash,),
            output=snapshot.model_dump(mode="json"),
        )


def _restored_global_circuit_reason(
    *,
    exhaustions: tuple[LLMExhaustionSnapshot, ...],
    ledgers: tuple[LLMAttemptLedger, ...],
) -> str | None:
    if any(
        not _is_recoverable_report_exhaustion(
            stage=snapshot.stage,
            error=RetryExhaustedError(snapshot.attempts),
        )
        for snapshot in exhaustions
    ):
        return "retry_exhausted"
    if any(
        ledger.results and ledger.results[-1].attempt.outcome == "fatal"
        for ledger in ledgers
    ):
        return "provider_fatal"
    return None


def _recover_provider_usage(
    *,
    progress: UsageTotals,
    entries: tuple[LLMCacheEntry, ...],
    exhaustions: tuple[LLMExhaustionSnapshot, ...],
    ledgers: tuple[LLMAttemptLedger, ...],
) -> UsageTotals:
    cache_usage = UsageTotals()
    for entry in entries:
        snapshot = entry.cumulative_usage
        cache_usage = _later_monotonic_usage(
            cache_usage,
            UsageTotals(
                attempts=snapshot.attempts,
                input_tokens=snapshot.input_tokens,
                output_tokens=snapshot.output_tokens,
                latency_seconds=snapshot.latency_seconds,
                cost_usd=0.0,
            ),
        )
    for exhaustion in exhaustions:
        snapshot = exhaustion.cumulative_usage
        cache_usage = _later_monotonic_usage(
            cache_usage,
            UsageTotals(
                attempts=snapshot.attempts,
                input_tokens=snapshot.input_tokens,
                output_tokens=snapshot.output_tokens,
                latency_seconds=snapshot.latency_seconds,
                cost_usd=0.0,
            ),
        )
    for ledger in ledgers:
        for reservation in ledger.reservations:
            snapshot = reservation.cumulative_usage
            cache_usage = _later_monotonic_usage(
                cache_usage,
                UsageTotals(
                    attempts=snapshot.attempts,
                    input_tokens=snapshot.input_tokens,
                    output_tokens=snapshot.output_tokens,
                    latency_seconds=snapshot.latency_seconds,
                    cost_usd=0.0,
                ),
            )
        for result in ledger.results:
            snapshot = result.cumulative_usage
            cache_usage = _later_monotonic_usage(
                cache_usage,
                UsageTotals(
                    attempts=snapshot.attempts,
                    input_tokens=snapshot.input_tokens,
                    output_tokens=snapshot.output_tokens,
                    latency_seconds=snapshot.latency_seconds,
                    cost_usd=0.0,
                ),
            )
    return _later_monotonic_usage(progress, cache_usage)


def _verified_llm_exhaustions(
    *,
    store: CheckpointStore,
    case_ids: tuple[str, ...],
) -> tuple[LLMExhaustionSnapshot, ...]:
    snapshots: list[LLMExhaustionSnapshot] = []
    for case_id in case_ids:
        recovery = store.recover_case(case_id=case_id)
        checkpoint = next(
            (item for item in recovery.checkpoints if item.stage == "llm_retry_exhausted"),
            None,
        )
        if checkpoint is None:
            continue
        if len(checkpoint.input_hashes) != 1:
            raise ResumeDriftError("LLM exhaustion checkpoint request binding is invalid")
        request_hash = checkpoint.input_hashes[0]
        replay = store.replay_stage(
            case_id=case_id,
            stage="llm_retry_exhausted",
            input_hashes=(request_hash,),
        )
        if not replay.hit:
            raise ResumeDriftError("LLM exhaustion checkpoint replay was rejected")
        output = _required_output(replay.output)
        stage = output.get("stage")
        if not isinstance(stage, str):
            raise ResumeDriftError("LLM exhaustion checkpoint stage is invalid")
        snapshots.append(
            _llm_exhaustion_snapshot(
                output,
                request_hash=request_hash,
                stage=stage,
            )
        )
    return tuple(snapshots)


def _llm_exhaustion_snapshot(
    output: Mapping[str, Any],
    *,
    request_hash: str,
    stage: str,
) -> LLMExhaustionSnapshot:
    try:
        snapshot = LLMExhaustionSnapshot.model_validate(output)
    except (TypeError, ValueError, ValidationError) as error:
        raise ResumeDriftError("LLM exhaustion checkpoint is invalid") from error
    if snapshot.request_hash != request_hash or snapshot.stage != stage:
        raise ResumeDriftError("LLM exhaustion checkpoint binding changed")
    return snapshot


def _later_monotonic_usage(left: UsageTotals, right: UsageTotals) -> UsageTotals:
    left_values = (
        left.attempts,
        left.input_tokens,
        left.output_tokens,
        left.latency_seconds,
    )
    right_values = (
        right.attempts,
        right.input_tokens,
        right.output_tokens,
        right.latency_seconds,
    )
    if all(
        left_value >= right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ):
        later = left
    elif all(
        right_value >= left_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ):
        later = right
    else:
        raise ResumeDriftError("progress and LLM cache usage snapshots are not monotonic")
    return later.model_copy(update={"cost_usd": max(left.cost_usd, right.cost_usd)})


def _provider_attempt_usage(response_value: Mapping[str, Any]) -> AttemptUsage:
    raw_usage = response_value.get("usage")
    if not isinstance(raw_usage, Mapping):
        return AttemptUsage.unavailable()
    details = raw_usage.get("prompt_tokens_details")
    cached_tokens = (
        _reported_nonnegative_int(details.get("cached_tokens"))
        if isinstance(details, Mapping)
        else None
    )
    values = {
        "prompt_tokens": _reported_nonnegative_int(raw_usage.get("prompt_tokens")),
        "completion_tokens": _reported_nonnegative_int(
            raw_usage.get("completion_tokens")
        ),
        "total_tokens": _reported_nonnegative_int(raw_usage.get("total_tokens")),
        "cached_tokens": cached_tokens,
    }
    available = sum(value is not None for value in values.values())
    if available == 0:
        return AttemptUsage.unavailable()
    status: Literal["partial", "reported"] = (
        "reported" if available == len(values) else "partial"
    )
    return AttemptUsage(status=status, **values)


def _reported_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
