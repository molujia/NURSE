"""Current normal-mode proposals compile to guarded query preferences only.

Structural parsing is retryable at the existing bridge; deterministic rejection
is a valid model response, never a reason to sample another answer. Even an
accepted hypothesis is a testable proposal, not an accepted scoring mechanism.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.exchange_json import ExchangeBoundaryError, ExchangeLimits, decode_json_object
from nurse_rcl.v2.live.normal_hybrid_queries import HybridQueryIntent, _intent_identity
from nurse_rcl.v2.live.normal_queries import _seal
from nurse_rcl.v2.live.normal_semantic_context import (
    NormalSemanticContext,
    project_normal_semantic_context,
)
from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationData
from nurse_rcl.v2.mechanism_library import build_mechanism_library

__all__ = ["compile_normal_semantics", "parse_normal_semantic_response",
           "project_normal_semantic_context"]

ShortText = Annotated[str, Field(min_length=1, max_length=1000)]
Reference = Annotated[str, Field(min_length=1, max_length=200)]
LocalID = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")]


class SemanticHypothesis(StrictFrozenModel):
    proposal_id: LocalID
    candidate_service_key: Reference
    mechanism_key: Reference
    evidence_refs: tuple[Reference, ...] = Field(max_length=32)
    query_ids: tuple[LocalID, ...] = Field(max_length=8)
    rationale: ShortText
    unknown_reasons: tuple[ShortText, ...] = Field(max_length=16)


class VerificationSuggestion(StrictFrozenModel):
    kind: Literal["verification"]
    query_id: LocalID
    request_key: Reference
    need_keys: tuple[Reference, ...] = Field(min_length=1, max_length=32)


class CardSuggestion(StrictFrozenModel):
    kind: Literal["card"]
    query_id: LocalID
    candidate_service_key: Reference


Suggestion = Annotated[VerificationSuggestion | CardSuggestion, Field(discriminator="kind")]


class NormalSemanticResponse(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-response/1"]
    hypotheses: tuple[SemanticHypothesis, ...] = Field(max_length=32)
    queries: tuple[Suggestion, ...] = Field(max_length=8)
    unknown_reasons: tuple[ShortText, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({h.proposal_id for h in self.hypotheses}) != len(self.hypotheses) or (
            len({q.query_id for q in self.queries}) != len(self.queries)
        ):
            raise ValueError("duplicate semantic local IDs")
        return self


class SemanticRejection(StrictFrozenModel):
    item_kind: Literal["hypothesis", "query"]
    item_id: str
    reason_code: str


class NormalSemanticCompilation(StrictFrozenModel):
    schema_version: Literal["nurse.v2.normal-semantic-compilation/1"] = (
        "nurse.v2.normal-semantic-compilation/1"
    )
    context_sha256: SHA256
    response_sha256: SHA256
    round_number: Literal[1, 2]
    accepted_proposal_ids: tuple[str, ...]
    intents: tuple[HybridQueryIntent, ...]
    card_candidates: tuple[str, ...]
    revision_dependency_proposal_ids: tuple[str, ...]
    rejections: tuple[SemanticRejection, ...]
    score_authority: Literal[False] = False
    executed_queries: Literal[0] = 0
    compilation_sha256: SHA256


def parse_normal_semantic_response(raw: bytes) -> NormalSemanticResponse:
    decode_json_object(raw, ExchangeLimits())
    try:
        return NormalSemanticResponse.model_validate_json(raw)
    except (ValueError, ValidationError):
        raise ExchangeBoundaryError("invalid_normal_semantic_response") from None


def compile_normal_semantics(
    *, data: RepairPreparationData, context: NormalSemanticContext,
    response: NormalSemanticResponse, round_number: Literal[1, 2] = 1,
) -> NormalSemanticCompilation:
    """No I/O or retry. The owner supplies its private, just-projected context."""
    return _seal(NormalSemanticCompilation(**_compile_values(
        data=data, context=context, response=response, round_number=round_number,
    )), "compilation_sha256")


def _compile_values(
    *, data: RepairPreparationData, context: NormalSemanticContext,
    response: NormalSemanticResponse, round_number: Literal[1, 2],
    feedback_refs: dict[str, tuple[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Shared rejection core. Feedback owner alone supplies its paid reference map."""
    if type(context) is not NormalSemanticContext or (
        context.catalog_sha256 != data.catalog.catalog_sha256
        or context.candidate_service_keys != data.catalog.candidate_service_keys
        or context.opaque_case_id != data.catalog.opaque_case_id
        or _seal(context, "context_sha256") != context
        or type(response) is not NormalSemanticResponse or round_number not in (1, 2)
    ):
        raise ValueError("normal semantic owner context differs")
    # External structure is validated once at this boundary; no whole-case replay.
    response = parse_normal_semantic_response(
        canonical_json_bytes(response.model_dump(mode="json")),
    )
    requests = {r.request_key: r for r in data.catalog.requests}
    if any(requests.get(r.request_key) != r for r in context.requests):
        raise ValueError("normal semantic visible request differs from owned catalog")
    visible = {r.request_key for r in context.requests}
    candidates = set(context.candidate_service_keys)
    mechanisms = {d.mechanism_key for d in build_mechanism_library().definitions}
    refs = {f.reference_id: {f.candidate_service_key} for f in context.signals}
    refs.update({f.reference_id: {f.candidate_service_key} for f in context.errors})
    refs.update({e.reference_id: {e.caller, e.callee} for e in context.edges})
    feedback_refs = {} if feedback_refs is None else feedback_refs
    refs.update({key: {candidate} for key, (candidate, _) in feedback_refs.items()})
    queries = {q.query_id: q for q in response.queries}
    rejects: list[SemanticRejection] = []

    def reject(kind: Literal["hypothesis", "query"], key: str, reason: str) -> None:
        rejects.append(SemanticRejection(item_kind=kind, item_id=key, reason_code=reason))

    valid_queries: set[str] = set()
    for q in response.queries:
        if round_number == 2:
            reject("query", q.query_id, "followup_queries_forbidden")
        elif isinstance(q, CardSuggestion):
            if q.candidate_service_key not in candidates:
                reject("query", q.query_id, "candidate_outside_C")
            else:
                valid_queries.add(q.query_id)
        else:
            request = requests.get(q.request_key)
            if request is None or q.request_key not in visible:
                reject("query", q.query_id, "request_not_visible")
            elif len(set(q.need_keys)) != len(q.need_keys) or not set(q.need_keys) <= {
                n.need_key for n in request.needs if n.queryable
            }:
                reject("query", q.query_id, "need_not_queryable")
            else:
                valid_queries.add(q.query_id)
    accepted, dependencies = [], []
    used_queries: set[str] = set()
    for h in response.hypotheses:
        reason = None
        if h.candidate_service_key not in candidates:
            reason = "candidate_outside_C"
        elif h.mechanism_key not in mechanisms:
            reason = "mechanism_not_registered"
        elif any(h.candidate_service_key not in refs.get(ref, set()) for ref in h.evidence_refs):
            reason = "evidence_not_visible_for_candidate"
        elif any(feedback_refs[ref][1] not in (None, h.mechanism_key)
                 for ref in h.evidence_refs if ref in feedback_refs):
            reason = "feedback_mechanism_differs"
        elif not set(h.query_ids) <= valid_queries:
            reason = "query_not_accepted"
        elif not h.evidence_refs and not h.query_ids:
            reason = "no_evidence_or_testable_query"
        else:
            for query_id in h.query_ids:
                query = queries[query_id]
                if isinstance(query, CardSuggestion):
                    compatible = query.candidate_service_key == h.candidate_service_key
                else:
                    request = requests[query.request_key]
                    compatible = request.candidate_service_key == h.candidate_service_key and (
                        set(query.need_keys) <= {n.need_key for n in request.needs
                                               if n.mechanism_key == h.mechanism_key}
                    )
                if not compatible:
                    reason = "query_candidate_or_mechanism_differs"
                    break
        if reason is not None:
            reject("hypothesis", h.proposal_id, reason)
            continue
        accepted.append(h.proposal_id)
        used_queries.update(h.query_ids)
        if h.query_ids and h.unknown_reasons:
            dependencies.append(h.proposal_id)
    intent_needs: dict[str, set[str]] = {}
    cards = set()
    for q in response.queries:
        if q.query_id not in valid_queries:
            continue
        if q.query_id not in used_queries:
            reject("query", q.query_id, "no_accepted_hypothesis_dependency")
        elif isinstance(q, CardSuggestion):
            cards.add(q.candidate_service_key)
        else:
            intent_needs.setdefault(q.request_key, set()).update(q.need_keys)
    intents = tuple(HybridQueryIntent(request_key=k, need_keys=tuple(sorted(v)))
                    for k, v in sorted(intent_needs.items()))
    _intent_identity(data.catalog, intents)
    return dict(
        context_sha256=context.context_sha256,
        response_sha256=sha256_hex(canonical_json_bytes(response.model_dump(mode="json"))),
        round_number=round_number, accepted_proposal_ids=tuple(accepted), intents=intents,
        card_candidates=tuple(sorted(cards)), revision_dependency_proposal_ids=tuple(dependencies),
        rejections=tuple(rejects), compilation_sha256="0" * 64,
    )
