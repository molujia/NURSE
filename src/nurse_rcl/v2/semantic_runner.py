"""Bounded embedded-HERMES semantic flow, without HTTP, retries or score selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.hermes_exchange import ExchangeAttempt, execute_once, prepare_exchange
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.observation_schemas import CandidateUniverse, TimeInterval
from nurse_rcl.v2.query_session import CaseQuerySessionCoordinator
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.semantic_compilation import compile_semantic_response
from nurse_rcl.v2.semantic_compilation_artifact import SemanticCompilation
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.semantic_followup import SemanticFollowupPrompt, build_semantic_followup_prompt
from nurse_rcl.v2.semantic_followup_compilation import (
    SemanticFollowupCompilation,
    compile_semantic_followup_response,
)
from nurse_rcl.v2.semantic_prompt import SemanticPrompt
from nurse_rcl.v2.semantic_query_bridge import (
    SemanticQueryExecution,
    _bind_results,
    execute_semantic_queries,
)
from nurse_rcl.v2.semantic_query_plan import plan_semantic_queries
from nurse_rcl.v2.semantic_query_replay import replay_semantic_queries
from nurse_rcl.v2.semantic_run_audit import (
    SemanticRankingPolicy,
    SemanticRoundAudit,
    SemanticRunAudit,
    StopReason,
    build_semantic_run_audit,
    require_semantic_run_audit,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry
from nurse_rcl.v2.verification_executor import VerificationExecutionPolicy

_EMPTY = canonical_json_bytes(
    dict(
        schema_version="nurse.v2.semantic-proposal-response/1",
        hypotheses=[],
        queries=[],
        unknown_reasons=[],
    )
)
_SOURCE_TYPES: dict[str, type[StrictFrozenModel]] = {
    "candidate_universe": CandidateUniverse,
    "frozen_candidates": FrozenCandidateUniverse,
    "entity_registry": TemporalEntityRegistry,
    "event_grouping": EventGroupingResult,
    "mechanism_library": MechanismLibrary,
    "scope_interval": TimeInterval,
    "semantic_context": SemanticContextBundle,
    "card_expansion_policy": CardExpansionPolicy,
    "verification_policy": VerificationExecutionPolicy,
}


@dataclass(frozen=True, slots=True)
class SemanticRankingSources:
    """Caller-owned source inputs; copied and replay-validated before any transport."""

    prompt: SemanticPrompt
    expected_prompt_sha256: str
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    mechanism_library: MechanismLibrary
    scope_interval: TimeInterval
    semantic_context: SemanticContextBundle
    card_expansion_policy: CardExpansionPolicy
    verification_policy: VerificationExecutionPolicy

    def query_inputs(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "mechanism_library"
        }

    def compilation_inputs(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("semantic_context", "card_expansion_policy", "verification_policy")
        }


@dataclass(frozen=True, slots=True)
class SemanticRankingRun:
    audit: SemanticRunAudit
    first: SemanticCompilation | None
    second: SemanticFollowupCompilation | None

    @property
    def selected(self) -> SemanticCompilation | SemanticFollowupCompilation | None:
        return (
            self.second
            if self.audit.selected_round == 2
            else (self.first if self.audit.selected_round == 1 else None)
        )


def _validated_sources(
    sources: SemanticRankingSources, policy: SemanticRankingPolicy
) -> tuple[SemanticRankingSources, SemanticRankingPolicy, str]:
    if type(sources) is not SemanticRankingSources or type(policy) is not SemanticRankingPolicy:
        raise ValueError("invalid_semantic_runner_input")
    prompt = object.__getattribute__(sources, "prompt")
    if type(prompt) is not SemanticPrompt or type(prompt.payload_json) is not bytes:
        raise ValueError("invalid_semantic_runner_prompt")
    models = {name: object.__getattribute__(sources, name) for name in _SOURCE_TYPES}
    if any(type(models[name]) is not cls for name, cls in _SOURCE_TYPES.items()):
        raise ValueError("invalid_semantic_runner_source_type")
    pin = object.__getattribute__(sources, "expected_prompt_sha256")
    preflight_exact_pydantic_tree(
        (models, pin, policy),
        root_model_types=(*_SOURCE_TYPES.values(), SemanticRankingPolicy),
        label="semantic runner sources",
        max_string_length=None,
        provenance_id_limit=None,
    )
    frozen: dict[str, Any] = {
        name: cls.model_validate_json(models[name].model_dump_json())
        for name, cls in _SOURCE_TYPES.items()
    }
    copied = SemanticRankingSources(
        prompt=SemanticPrompt(prompt.payload_json), expected_prompt_sha256=pin, **frozen
    )
    copied_policy = SemanticRankingPolicy.model_validate_json(policy.model_dump_json())
    # Both paths replay their own external sources; empty proposals perform no query.
    plan_semantic_queries(raw_response=_EMPTY, **copied.query_inputs())
    compile_semantic_response(raw_response=_EMPTY, **copied.compilation_inputs())
    digest = sha256_hex(
        canonical_json_bytes(
            dict(
                prompt_payload_hex=prompt.payload_json.hex(),
                expected_prompt_sha256=pin,
                **{name: value.model_dump(mode="json") for name, value in frozen.items()},
            )
        )
    )
    return copied, copied_policy, digest


def _projection(value: Any) -> Any:
    """Only called on fresh compiler/bridge outputs, not caller-supplied authority."""
    if isinstance(value, tuple):
        return [_projection(item) for item in value]
    if hasattr(type(value), "model_fields"):
        return value.model_dump(mode="json")
    return value


def _compilation_json(value: SemanticCompilation | SemanticFollowupCompilation) -> str:
    document = (
        value.model_dump(mode="json")
        if type(value) is SemanticCompilation
        else {
            f.name: _projection(getattr(value, f.name)) for f in fields(SemanticFollowupCompilation)
        }
    )
    return canonical_json_bytes(document).decode("utf-8")


def _query_receipt_json(execution: SemanticQueryExecution) -> str:
    return canonical_json_bytes(
        dict(
            bindings=[
                {f.name: _projection(getattr(binding, f.name)) for f in fields(binding)}
                for binding in execution.bindings
            ],
            successful_query_ids=execution.successful_query_ids,
            revision_dependency_proposal_ids=execution.revision_dependency_proposal_ids,
        )
    ).decode("utf-8")


def _exchange(
    prompt: SemanticPrompt | SemanticFollowupPrompt,
    *,
    round_number: Literal[1, 2],
    policy: SemanticRankingPolicy,
    complete: Callable[[bytes], bytes],
    recorded: SemanticRoundAudit | None = None,
) -> SemanticRoundAudit:
    values: dict[str, Any] = dict(
        round_number=round_number,
        prompt_sha256=prompt.sha256,
        prompt_payload_hex=prompt.payload_json.hex(),
        request_hex=None,
        estimated_input_tokens=None,
        status="prepare_failed",
        failure_code=None,
        raw_response_hex=None,
        received_bytes=None,
        semantic_response_hex=None,
        usage_status="unavailable",
        input_tokens=None,
        output_tokens=None,
        compilation_json=None,
    )
    try:
        request = prepare_exchange(
            model=policy.model,
            messages=prompt.messages(),
            max_output_tokens=policy.output_token_limit,
            temperature=policy.temperature,
        )
    except ExchangeBoundaryError as exc:
        return SemanticRoundAudit(**(values | dict(failure_code=exc.code)))
    # Conservative byte-based estimate, not actual tokens or a bytes/4 heuristic.
    # Count the whole request (including schema) plus explicit framing allowance.
    estimated = len(request.payload_bytes) + 256
    values.update(request_hex=request.payload_bytes.hex(), estimated_input_tokens=estimated)
    if estimated > policy.input_token_limit:
        return SemanticRoundAudit(
            **(
                values
                | dict(status="budget_exhausted", failure_code="input_token_budget_exhausted")
            )
        )
    if recorded is None:
        attempt = execute_once(request, complete)
    elif recorded.raw_response_hex is not None:
        raw_wire = bytes.fromhex(recorded.raw_response_hex)
        attempt = execute_once(request, lambda _: raw_wire)
    else:
        # No fabricated response for a timeout, invalid transport return, or a
        # deliberately unretained oversized response. Such failures rely on the
        # externally pinned audit for transport history, never as score evidence.
        code = recorded.failure_code
        size = recorded.received_bytes
        if recorded.status != "failed" or not (
            (code in ("timeout", "transport_error", "response_type") and size is None)
            or (code == "response_limit" and size is not None and size > request.limits.max_bytes)
        ):
            raise ValueError("invalid_semantic_replay_failure")
        attempt = ExchangeAttempt("failed", request.sha256, True, code, None, None, None, size)
    values.update(
        status=attempt.status,
        failure_code=attempt.failure_code,
        raw_response_hex=None if attempt.raw_response is None else attempt.raw_response.hex(),
        received_bytes=attempt.received_bytes,
    )
    if attempt.response is not None:
        parsed = attempt.response
        values.update(
            semantic_response_hex=parsed.output_json.hex(), usage_status=parsed.usage_status
        )
        if parsed.usage is not None:
            values.update(
                input_tokens=parsed.usage.input_tokens, output_tokens=parsed.usage.output_tokens
            )
    return SemanticRoundAudit(**values)


def _run_semantic_ranking(
    *,
    sources: SemanticRankingSources,
    policy: SemanticRankingPolicy,
    complete: Callable[[bytes], bytes],
    query_session_coordinator: CaseQuerySessionCoordinator,
    replay_audit: SemanticRunAudit | None = None,
) -> SemanticRankingRun:
    """One bounded invocation; durable lifetime attempts/resume belong to chapter 11."""
    if type(query_session_coordinator) is not CaseQuerySessionCoordinator or not callable(complete):
        raise ValueError("invalid_semantic_runner_runtime")
    sources, policy, source_hash = _validated_sources(sources, policy)
    first: SemanticCompilation | None = None
    second: SemanticFollowupCompilation | None = None
    execution: SemanticQueryExecution | None = None
    selected: Literal[0, 1, 2] = 0
    reason: StopReason = "first_failure"
    followup_failure_code = None
    record = _exchange(
        sources.prompt,
        round_number=1,
        policy=policy,
        complete=complete,
        recorded=None if replay_audit is None else replay_audit.rounds[0],
    )
    rounds = [record]
    if record.status == "budget_exhausted":
        reason = "first_budget_exhausted"
    if record.semantic_response_hex is not None:
        raw = bytes.fromhex(record.semantic_response_hex)
        first = compile_semantic_response(raw_response=raw, **sources.compilation_inputs())
        rounds[0] = record.model_copy(update={"compilation_json": _compilation_json(first)})
        if first.structurally_valid or any(
            g.final_disposition == "accepted" for g in first.guard_results
        ):
            selected = 1
        if replay_audit is None:
            execution = execute_semantic_queries(
                query_session_coordinator=query_session_coordinator,
                raw_response=raw,
                **sources.query_inputs(),
            )
        elif replay_audit.query_session is not None:
            execution = replay_semantic_queries(
                query_session_coordinator=query_session_coordinator,
                query_session=replay_audit.query_session,
                expected_session_sha256=replay_audit.query_session.session_sha256,
                expected_response_sha256=sha256_hex(raw),
                raw_response=raw,
                **sources.query_inputs(),
            )
        else:
            plan = plan_semantic_queries(raw_response=raw, **sources.query_inputs())
            if plan.card_requests or plan.verification_requests:
                raise ValueError("semantic audit is missing query execution")
            bindings, successful, _ = _bind_results(plan, None)
            execution = SemanticQueryExecution(plan, None, bindings, successful, ())
        reason = "no_revision_dependency" if selected else "first_failure"
        if execution.session is not None and execution.revision_dependency_proposal_ids:
            followup_inputs = dict(
                query_session_coordinator=query_session_coordinator,
                query_session=execution.session,
                expected_session_sha256=execution.session.session_sha256,
                expected_response_sha256=sha256_hex(raw),
                raw_response=raw,
                **sources.query_inputs(),
            )
            try:
                followup = build_semantic_followup_prompt(**followup_inputs)
            except ExchangeBoundaryError as exc:
                reason = "second_prompt_denied"
                followup_failure_code = exc.code
            else:
                if replay_audit is not None and len(replay_audit.rounds) != 2:
                    raise ValueError("semantic audit is missing second round")
                revision = _exchange(
                    followup,
                    round_number=2,
                    policy=policy,
                    complete=complete,
                    recorded=None if replay_audit is None else replay_audit.rounds[1],
                )
                reason = (
                    "second_budget_exhausted"
                    if revision.status == "budget_exhausted"
                    else "second_failure"
                )
                if revision.semantic_response_hex is not None:
                    second = compile_semantic_followup_response(
                        followup_raw_response=bytes.fromhex(revision.semantic_response_hex),
                        expected_followup_prompt_sha256=followup.sha256,
                        mechanism_library=sources.mechanism_library,
                        **followup_inputs,
                    )
                    revision = revision.model_copy(
                        update={"compilation_json": _compilation_json(second)}
                    )
                    if second.structurally_valid:
                        selected, reason = 2, "second_snapshot_selected"
                rounds.append(revision)
    values = dict(
        opaque_case_id=sources.candidate_universe.opaque_case_id,
        sources_sha256=source_hash,
        policy=policy.model_dump(mode="json"),
        rounds=[r.model_dump(mode="json") for r in rounds],
        query_session=None
        if execution is None or execution.session is None
        else execution.session.model_dump(mode="json"),
        query_receipt_json=None if execution is None else _query_receipt_json(execution),
        selected_round=selected,
        stop_reason=reason,
    )
    if followup_failure_code is not None:
        values["followup_failure_code"] = followup_failure_code
    result = SemanticRankingRun(build_semantic_run_audit(**values), first, second)
    if replay_audit is not None and result.audit != replay_audit:
        raise ValueError("semantic audit does not match independent source replay")
    return result


def run_semantic_ranking(
    *,
    sources: SemanticRankingSources,
    policy: SemanticRankingPolicy,
    complete: Callable[[bytes], bytes],
    query_session_coordinator: CaseQuerySessionCoordinator,
) -> SemanticRankingRun:
    return _run_semantic_ranking(
        sources=sources,
        policy=policy,
        complete=complete,
        query_session_coordinator=query_session_coordinator,
    )


def replay_semantic_ranking(
    *,
    audit: SemanticRunAudit,
    expected_audit_sha256: str,
    sources: SemanticRankingSources,
    policy: SemanticRankingPolicy,
    query_session_coordinator: CaseQuerySessionCoordinator,
) -> SemanticRankingRun:
    """Recompile retained bytes from external sources, without network/query mutations."""
    require_semantic_run_audit(audit, expected_audit_sha256=expected_audit_sha256)

    def no_transport(_: bytes) -> bytes:
        raise AssertionError("semantic audit replay cannot invoke a transport")

    return _run_semantic_ranking(
        sources=sources,
        policy=policy,
        complete=no_transport,
        query_session_coordinator=query_session_coordinator,
        replay_audit=audit,
    )
