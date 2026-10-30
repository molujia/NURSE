"""Private, layered candidate-coverage diagnostics for the isolated evaluator."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.context_expansion import CardExpansionBatchResult
from nurse_rcl.v2.focus_selection import FocusSelectionResult
from nurse_rcl.v2.observation_schemas import CandidateUniverse
from nurse_rcl.v2.opaque_ids import EvaluatorIdentityMap
from nurse_rcl.v2.ranking_schemas import TopKArtifact
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

COVERAGE_DIAGNOSTICS_SCHEMA_VERSION: Literal[
    "nurse.v2.layered-candidate-coverage-diagnostics/1"
] = "nurse.v2.layered-candidate-coverage-diagnostics/1"

CoverageStage = Literal[
    "discovery",
    "binding",
    "candidate_universe",
    "initial_focus",
    "final_focus",
    "top_k",
]

_STAGE_ORDER: tuple[CoverageStage, ...] = (
    "discovery",
    "binding",
    "candidate_universe",
    "initial_focus",
    "final_focus",
    "top_k",
)


class CoverageStageDiagnostic(StrictFrozenModel):
    """Non-identifying coverage summary for one evaluator stage."""

    stage: CoverageStage
    eligible_count: int = Field(ge=0)
    answer_count: int = Field(ge=1)
    matched_answer_count: int = Field(ge=0)
    covered: bool
    eligible_service_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Validate count and any-hit semantics without retaining service names."""

        try:
            validated = CoverageStageDiagnostic.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and validated.matched_answer_count <= validated.answer_count
            and validated.matched_answer_count <= validated.eligible_count
            and validated.covered == (validated.matched_answer_count > 0)
        )


class LayeredCandidateCoverageDiagnostics(StrictFrozenModel):
    """Evaluator-only summary that is forbidden at the inference boundary."""

    schema_version: Literal[
        "nurse.v2.layered-candidate-coverage-diagnostics/1"
    ] = COVERAGE_DIAGNOSTICS_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    evaluator_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    focus_selection_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_context_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expansion_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_only: Literal[True] = True
    inference_feedback_permitted: Literal[False] = False
    answer_count: int = Field(ge=1)
    stages: tuple[CoverageStageDiagnostic, ...] = Field(min_length=6, max_length=6)
    diagnostics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        """Verify canonical stage order, privacy flags, counts, and content digest."""

        try:
            validated = LayeredCandidateCoverageDiagnostics.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self:
                return False
            if tuple(item.stage for item in validated.stages) != _STAGE_ORDER:
                return False
            if any(
                not item.verify() or item.answer_count != validated.answer_count
                for item in validated.stages
            ):
                return False
            unsigned = validated.model_dump(mode="json", exclude={"diagnostics_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return validated.diagnostics_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _service_set(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _stage_diagnostic(
    *,
    stage: CoverageStage,
    eligible_services: Iterable[str],
    answer_services: frozenset[str],
) -> CoverageStageDiagnostic:
    eligible = _service_set(eligible_services)
    matched_count = len(set(eligible).intersection(answer_services))
    return CoverageStageDiagnostic(
        stage=stage,
        eligible_count=len(eligible),
        answer_count=len(answer_services),
        matched_answer_count=matched_count,
        covered=matched_count > 0,
        eligible_service_set_sha256=sha256_hex(
            canonical_json_bytes({"service_entity_keys": eligible})
        ),
    )


def _validate_inputs(
    *,
    evaluator_map: EvaluatorIdentityMap,
    expected_evaluator_mapping_sha256: str,
    entity_registry: TemporalEntityRegistry,
    candidate_universe: CandidateUniverse,
    focus_selection: FocusSelectionResult,
    semantic_context: SemanticContextBundle,
    expansion_result: CardExpansionBatchResult | None,
    top_k: TopKArtifact,
) -> None:
    if not evaluator_map.verify():
        raise ValueError("evaluator mapping failed verification")
    if evaluator_map.mapping_sha256 != expected_evaluator_mapping_sha256:
        raise ValueError("evaluator mapping hash does not match pinned value")
    for artifact, label in (
        (entity_registry, "entity registry"),
        (candidate_universe, "candidate universe"),
        (focus_selection, "focus selection"),
        (semantic_context, "semantic context"),
        (top_k, "Top-K"),
    ):
        if not artifact.verify():
            raise ValueError(f"{label} failed verification")
    if expansion_result is not None and not expansion_result.verify():
        raise ValueError("card expansion result failed verification")

    opaque_case_id = candidate_universe.opaque_case_id
    if not (
        entity_registry.opaque_case_id
        == focus_selection.opaque_case_id
        == semantic_context.opaque_case_id
        == opaque_case_id
    ):
        raise ValueError("coverage inputs belong to different cases")
    if (
        semantic_context.entity_registry_sha256 != entity_registry.registry_sha256
        or semantic_context.candidate_directory.candidate_universe_sha256
        != candidate_universe.universe_sha256
        or semantic_context.focus_selection_result_sha256 != focus_selection.result_sha256
        or semantic_context.candidate_freeze_sha256
        != focus_selection.candidate_freeze_sha256
    ):
        raise ValueError("coverage inputs do not share one frozen inference scope")
    if expansion_result is not None and (
        expansion_result.opaque_case_id != opaque_case_id
        or expansion_result.candidate_freeze_sha256
        != semantic_context.candidate_freeze_sha256
        or expansion_result.semantic_context_bundle_sha256 != semantic_context.bundle_sha256
    ):
        raise ValueError("card expansion result belongs to another inference scope")

    candidate_keys = {
        item.service_entity_key for item in candidate_universe.candidates
    }
    focus_candidate_keys = {
        item.service_entity_key for item in focus_selection.focus_set.decisions
    }
    directory_candidate_keys = {
        item.service_entity_key for item in semantic_context.candidate_directory.entries
    }
    initial_focus_keys = set(focus_selection.focus_set.selected_service_keys)
    directory_focus_keys = {
        item.service_entity_key
        for item in semantic_context.candidate_directory.entries
        if item.in_initial_focus
    }
    detailed_card_keys = {
        item.card.service_entity_key
        for item in semantic_context.detailed_evidence_cards
    }
    if (
        focus_selection.focus_set.opaque_case_id != opaque_case_id
        or focus_selection.focus_set.candidate_universe_sha256
        != candidate_universe.universe_sha256
        or focus_candidate_keys != candidate_keys
        or directory_candidate_keys != candidate_keys
        or initial_focus_keys != directory_focus_keys
        or initial_focus_keys != detailed_card_keys
    ):
        raise ValueError("initial focus disagrees with candidate or semantic context")
    if (
        semantic_context.candidate_directory.opaque_case_id != opaque_case_id
        or semantic_context.candidate_directory.decision_time_unix_ns
        != entity_registry.decision_time_unix_ns
        or any(
            item.card.opaque_case_id != opaque_case_id
            or item.card.candidate_universe_sha256 != candidate_universe.universe_sha256
            or item.card.decision_time_unix_ns != entity_registry.decision_time_unix_ns
            for item in semantic_context.detailed_evidence_cards
        )
    ):
        raise ValueError("initial focus evidence belongs to another inference scope")
    if expansion_result is not None:
        returned_cards = {
            item.detailed_card_sha256: item for item in expansion_result.returned_cards
        }
        if any(
            item.card.service_entity_key not in candidate_keys
            or item.card.opaque_case_id != opaque_case_id
            or item.card.candidate_universe_sha256 != candidate_universe.universe_sha256
            or item.card.decision_time_unix_ns != entity_registry.decision_time_unix_ns
            for item in returned_cards.values()
        ):
            raise ValueError("expansion card belongs to another inference scope")
        for decision in expansion_result.decisions:
            if decision.detailed_card_sha256 is None:
                continue
            request = decision.request
            if (
                request.opaque_case_id != opaque_case_id
                or request.candidate_freeze_sha256
                != semantic_context.candidate_freeze_sha256
                or request.semantic_context_bundle_sha256 != semantic_context.bundle_sha256
                or request.decision_time_unix_ns
                != entity_registry.decision_time_unix_ns
                or request.service_entity_key not in candidate_keys
                or decision.detailed_card_sha256 not in returned_cards
            ):
                raise ValueError("completed expansion decision has invalid scope")
            if decision.status == "reused" and (
                request.service_entity_key not in initial_focus_keys
                or decision.detailed_card_sha256
                not in {
                    item.detailed_card_sha256
                    for item in semantic_context.detailed_evidence_cards
                }
            ):
                raise ValueError("reused expansion does not reference initial focus")
    top_k_keys = {item.service_entity_key for item in top_k.entries}
    if not top_k_keys.issubset(candidate_keys):
        raise ValueError("Top-K contains a service outside candidate universe")
    if top_k.total_candidate_count != candidate_universe.candidate_count:
        raise ValueError("Top-K candidate count differs from candidate universe")
    if (
        top_k.opaque_case_id != opaque_case_id
        or top_k.candidate_universe_sha256 != candidate_universe.universe_sha256
    ):
        raise ValueError("Top-K belongs to another inference scope")


def build_layered_candidate_coverage_diagnostics(
    *,
    evaluator_map: EvaluatorIdentityMap,
    expected_evaluator_mapping_sha256: str,
    entity_registry: TemporalEntityRegistry,
    candidate_universe: CandidateUniverse,
    focus_selection: FocusSelectionResult,
    semantic_context: SemanticContextBundle,
    expansion_result: CardExpansionBatchResult | None,
    top_k: TopKArtifact,
) -> LayeredCandidateCoverageDiagnostics:
    """Compute any-hit stage coverage without releasing answer service identities."""

    _validate_inputs(
        evaluator_map=evaluator_map,
        expected_evaluator_mapping_sha256=expected_evaluator_mapping_sha256,
        entity_registry=entity_registry,
        candidate_universe=candidate_universe,
        focus_selection=focus_selection,
        semantic_context=semantic_context,
        expansion_result=expansion_result,
        top_k=top_k,
    )
    opaque_case_id = candidate_universe.opaque_case_id
    identity_records = tuple(
        item for item in evaluator_map.records if item.opaque_case_id == opaque_case_id
    )
    if len(identity_records) != 1:
        raise ValueError("evaluator mapping must contain exactly one case identity")
    answer_services = frozenset(identity_records[0].answer_services)

    discovered = _service_set(
        item.service_entity_key
        for item in entity_registry.metadata_records
        if item.relation == "explicit_service_binding"
        and item.service_entity_key is not None
    )
    bindings = (
        *entity_registry.timeline_bindings,
        *(item.binding for item in entity_registry.resolutions),
    )
    bound = _service_set(
        item.service_entity_key
        for item in bindings
        if item.status == "bound" and item.service_entity_key is not None
    )
    candidate_keys = _service_set(
        item.service_entity_key for item in candidate_universe.candidates
    )
    initial_focus = _service_set(focus_selection.focus_set.selected_service_keys)
    expanded = () if expansion_result is None else tuple(
        item.card.service_entity_key for item in expansion_result.returned_cards
    )
    final_focus = _service_set((*initial_focus, *expanded))
    top_k_keys = _service_set(item.service_entity_key for item in top_k.entries)

    stage_sets = (
        discovered,
        bound,
        candidate_keys,
        initial_focus,
        final_focus,
        top_k_keys,
    )
    stages = tuple(
        _stage_diagnostic(
            stage=stage,
            eligible_services=eligible,
            answer_services=answer_services,
        )
        for stage, eligible in zip(_STAGE_ORDER, stage_sets, strict=True)
    )
    unsigned: dict[str, object] = {
        "schema_version": COVERAGE_DIAGNOSTICS_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "evaluator_mapping_sha256": evaluator_map.mapping_sha256,
        "entity_registry_sha256": entity_registry.registry_sha256,
        "candidate_universe_sha256": candidate_universe.universe_sha256,
        "focus_selection_result_sha256": focus_selection.result_sha256,
        "semantic_context_bundle_sha256": semantic_context.bundle_sha256,
        "expansion_result_sha256": (
            None if expansion_result is None else expansion_result.result_sha256
        ),
        "ranking_sha256": top_k.ranking_sha256,
        "evaluator_only": True,
        "inference_feedback_permitted": False,
        "answer_count": len(answer_services),
        "stages": [item.model_dump(mode="json") for item in stages],
    }
    result = LayeredCandidateCoverageDiagnostics.model_validate(
        {
            **unsigned,
            "diagnostics_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("layered candidate coverage diagnostics failed verification")
    return result
