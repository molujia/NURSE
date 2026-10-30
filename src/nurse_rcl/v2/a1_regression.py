"""Deterministic A0/A1 candidate-coverage and ranking-attribution regression."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.methods.paper_v1.alarm import canonicalize_service_name
from nurse_rcl.methods.paper_v1.candidates import (
    CandidateSelection,
)
from nurse_rcl.methods.paper_v1.candidates import (
    build_candidates as build_a0_candidates,
)
from nurse_rcl.methods.paper_v1.final_ranking import (
    CandidateComponentInput,
    RankedComponentArtifact,
    rank_component_inputs,
)
from nurse_rcl.methods.paper_v1.topology import NormalTopology
from nurse_rcl.models import ComponentValues, EvidenceRecord
from nurse_rcl.schemas import DatasetName, StrictFrozenModel
from nurse_rcl.v2.a1_narrow_audit import A1NarrowAudit
from nurse_rcl.v2.paper_v1_repaired.candidates import (
    build_candidates as build_a1_candidates,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

A0_A1_REGRESSION_SCHEMA_VERSION: Literal["nurse.v2.a0-a1-regression-archive/1"] = (
    "nurse.v2.a0-a1-regression-archive/1"
)
_RANKING_INPUT_POLICY: Literal["max_metric_weight_as_raw_e_other_components_zero"] = (
    "max_metric_weight_as_raw_e_other_components_zero"
)


class VariantRegressionOutcome(StrictFrozenModel):
    """One variant's candidate coverage and unchanged-ranker replay."""

    variant: Literal["A0", "A1"]
    selection: CandidateSelection
    eligible_metric_services: tuple[str, ...]
    eligible_metric_service_count: int = Field(ge=0)
    selected_metric_service_count: int = Field(ge=0)
    metric_service_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evaluator_target_services: tuple[str, ...] = Field(min_length=1)
    target_candidate_hits: tuple[str, ...]
    target_candidate_coverage: bool
    ranking: RankedComponentArtifact
    target_ranks: tuple[int, ...]

    @model_validator(mode="after")
    def require_recomputed_coverage(self) -> Self:
        if self.eligible_metric_services != tuple(sorted(set(self.eligible_metric_services))):
            raise ValueError("eligible metric services must be sorted and unique")
        if self.evaluator_target_services != tuple(sorted(set(self.evaluator_target_services))):
            raise ValueError("evaluator target services must be sorted and unique")
        if self.target_candidate_hits != tuple(sorted(set(self.target_candidate_hits))):
            raise ValueError("target candidate hits must be sorted and unique")
        eligible = set(self.eligible_metric_services)
        selected = set(self.selection.candidates)
        expected_selected = len(eligible & selected)
        if self.eligible_metric_service_count != len(eligible):
            raise ValueError("eligible metric service count is inconsistent")
        if self.selected_metric_service_count != expected_selected:
            raise ValueError("selected metric service count is inconsistent")
        expected_coverage = expected_selected / len(eligible) if eligible else 0.0
        if self.metric_service_coverage != expected_coverage:
            raise ValueError("metric service coverage is inconsistent")
        expected_hits = tuple(sorted(set(self.evaluator_target_services) & selected))
        if self.target_candidate_hits != expected_hits:
            raise ValueError("target candidate hits are inconsistent")
        if self.target_candidate_coverage is not bool(expected_hits):
            raise ValueError("target candidate coverage is inconsistent")
        ranking_services = tuple(item.service for item in self.ranking.candidates)
        if set(ranking_services) != selected or len(ranking_services) != len(selected):
            raise ValueError("ranking must exactly cover the selected candidates")
        expected_ranks = tuple(
            sorted(
                item.rank
                for item in self.ranking.candidates
                if item.service in self.evaluator_target_services
            )
        )
        if self.target_ranks != expected_ranks:
            raise ValueError("target ranks are inconsistent")
        if not self.ranking.verify():
            raise ValueError("variant ranking failed verification")
        return self


class SharedCandidateAttribution(StrictFrozenModel):
    """Score change for one shared candidate under the same raw components."""

    service: str = Field(min_length=1)
    a0_raw: ComponentValues
    a1_raw: ComponentValues
    raw_components_changed: bool
    a0_score: float = Field(allow_inf_nan=False)
    a1_score: float = Field(allow_inf_nan=False)
    score_delta: float = Field(allow_inf_nan=False)
    reason: Literal["unchanged", "candidate_set_changed_normalization_context"]

    @model_validator(mode="after")
    def require_exact_attribution(self) -> Self:
        changed = self.a0_raw != self.a1_raw
        if self.raw_components_changed is not changed:
            raise ValueError("shared raw-component change flag is inconsistent")
        if self.score_delta != self.a1_score - self.a0_score:
            raise ValueError("shared candidate score delta is inconsistent")
        expected_reason = (
            "unchanged"
            if self.score_delta == 0.0
            else "candidate_set_changed_normalization_context"
        )
        if self.reason != expected_reason:
            raise ValueError("shared candidate attribution reason is inconsistent")
        if changed:
            raise ValueError("A0/A1 shared raw components must remain identical")
        return self


class RankingAttribution(StrictFrozenModel):
    """Exact boundary between candidate membership and unchanged ranking logic."""

    added_candidates: tuple[str, ...]
    removed_candidates: tuple[str, ...]
    ranking_formula_changed: Literal[False] = False
    component_weights_changed: Literal[False] = False
    shared_raw_components_changed: Literal[False] = False
    a0_top1: str = Field(min_length=1)
    a1_top1: str = Field(min_length=1)
    top1_changed: bool
    target_candidate_coverage_changed: bool
    shared_candidates: tuple[SharedCandidateAttribution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_attribution(self) -> Self:
        for values in (self.added_candidates, self.removed_candidates):
            if values != tuple(sorted(set(values))):
                raise ValueError("candidate differences must be sorted and unique")
        services = tuple(item.service for item in self.shared_candidates)
        if services != tuple(sorted(set(services))):
            raise ValueError("shared candidate attributions must be sorted and unique")
        if self.top1_changed is not (self.a0_top1 != self.a1_top1):
            raise ValueError("Top-1 change flag is inconsistent")
        if any(item.raw_components_changed for item in self.shared_candidates):
            raise ValueError("shared raw components changed across A0/A1")
        return self


class A0A1RegressionArchive(StrictFrozenModel):
    """Content-addressed evaluator-only archive for the candidate repair."""

    schema_version: Literal["nurse.v2.a0-a1-regression-archive/1"] = A0_A1_REGRESSION_SCHEMA_VERSION
    evaluation_scope: Literal["isolated_evaluator_regression_only"] = (
        "isolated_evaluator_regression_only"
    )
    ranking_input_policy: Literal["max_metric_weight_as_raw_e_other_components_zero"] = (
        _RANKING_INPUT_POLICY
    )
    source_fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    a1_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str = Field(min_length=1)
    dataset: DatasetName
    a0: VariantRegressionOutcome
    a1: VariantRegressionOutcome
    ranking_attribution: RankingAttribution
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_linked_variants(self) -> Self:
        if self.a0.variant != "A0" or self.a1.variant != "A1":
            raise ValueError("regression outcomes must be ordered A0 then A1")
        for outcome in (self.a0, self.a1):
            if outcome.selection.case_id != self.case_id:
                raise ValueError("regression selection case identity is inconsistent")
            if outcome.selection.dataset != self.dataset:
                raise ValueError("regression selection dataset is inconsistent")
        attribution = self.ranking_attribution
        if attribution.a0_top1 != self.a0.ranking.candidates[0].service:
            raise ValueError("A0 Top-1 attribution is inconsistent")
        if attribution.a1_top1 != self.a1.ranking.candidates[0].service:
            raise ValueError("A1 Top-1 attribution is inconsistent")
        expected_added = tuple(
            sorted(set(self.a1.selection.candidates) - set(self.a0.selection.candidates))
        )
        expected_removed = tuple(
            sorted(set(self.a0.selection.candidates) - set(self.a1.selection.candidates))
        )
        if attribution.added_candidates != expected_added:
            raise ValueError("added-candidate attribution is inconsistent")
        if attribution.removed_candidates != expected_removed:
            raise ValueError("removed-candidate attribution is inconsistent")
        coverage_changed = self.a0.target_candidate_coverage != self.a1.target_candidate_coverage
        if attribution.target_candidate_coverage_changed is not coverage_changed:
            raise ValueError("target candidate coverage attribution is inconsistent")
        return self

    def verify(self) -> bool:
        unsigned = self.model_dump(mode="json", exclude={"archive_sha256"})
        return self.archive_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _metric_inputs(
    *,
    selection: CandidateSelection,
    records: tuple[EvidenceRecord, ...],
) -> tuple[CandidateComponentInput, ...]:
    records_by_service: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if record.kind != "metric_anomaly":
            continue
        service = canonicalize_service_name(record.service)
        records_by_service.setdefault(service, []).append(record)
    return tuple(
        CandidateComponentInput(
            service=service,
            e=max(
                (record.weight for record in records_by_service.get(service, ())),
                default=0.0,
            ),
            p=0.0,
            h=0.0,
            r=0.0,
            evidence_ids=tuple(
                sorted(record.evidence_id for record in records_by_service.get(service, ()))
            ),
        )
        for service in selection.candidates
    )


def _outcome(
    *,
    variant: Literal["A0", "A1"],
    selection: CandidateSelection,
    records: tuple[EvidenceRecord, ...],
    eligible_metric_services: tuple[str, ...],
    evaluator_target_services: tuple[str, ...],
) -> VariantRegressionOutcome:
    ranking = rank_component_inputs(inputs=_metric_inputs(selection=selection, records=records))
    selected = set(selection.candidates)
    selected_metric_count = len(selected & set(eligible_metric_services))
    target_hits = tuple(sorted(selected & set(evaluator_target_services)))
    return VariantRegressionOutcome(
        variant=variant,
        selection=selection,
        eligible_metric_services=eligible_metric_services,
        eligible_metric_service_count=len(eligible_metric_services),
        selected_metric_service_count=selected_metric_count,
        metric_service_coverage=(
            selected_metric_count / len(eligible_metric_services)
            if eligible_metric_services
            else 0.0
        ),
        evaluator_target_services=evaluator_target_services,
        target_candidate_hits=target_hits,
        target_candidate_coverage=bool(target_hits),
        ranking=ranking,
        target_ranks=tuple(
            sorted(
                item.rank
                for item in ranking.candidates
                if item.service in evaluator_target_services
            )
        ),
    )


def _ranking_attribution(
    a0: VariantRegressionOutcome,
    a1: VariantRegressionOutcome,
) -> RankingAttribution:
    if a0.ranking.ranking_version != a1.ranking.ranking_version:
        raise ValueError("A0/A1 regression cannot change the ranking formula")
    if a0.ranking.component_weights != a1.ranking.component_weights:
        raise ValueError("A0/A1 regression cannot change component weights")
    a0_by_service = {item.service: item for item in a0.ranking.candidates}
    a1_by_service = {item.service: item for item in a1.ranking.candidates}
    shared = tuple(sorted(set(a0_by_service) & set(a1_by_service)))
    shared_attribution = tuple(
        SharedCandidateAttribution(
            service=service,
            a0_raw=a0_by_service[service].components.raw,
            a1_raw=a1_by_service[service].components.raw,
            raw_components_changed=(
                a0_by_service[service].components.raw != a1_by_service[service].components.raw
            ),
            a0_score=a0_by_service[service].score,
            a1_score=a1_by_service[service].score,
            score_delta=(a1_by_service[service].score - a0_by_service[service].score),
            reason=(
                "unchanged"
                if a1_by_service[service].score == a0_by_service[service].score
                else "candidate_set_changed_normalization_context"
            ),
        )
        for service in shared
    )
    if any(item.raw_components_changed for item in shared_attribution):
        raise ValueError("A0/A1 regression changed shared raw ranking inputs")
    return RankingAttribution(
        added_candidates=tuple(sorted(set(a1.selection.candidates) - set(a0.selection.candidates))),
        removed_candidates=tuple(
            sorted(set(a0.selection.candidates) - set(a1.selection.candidates))
        ),
        ranking_formula_changed=False,
        component_weights_changed=False,
        shared_raw_components_changed=False,
        a0_top1=a0.ranking.candidates[0].service,
        a1_top1=a1.ranking.candidates[0].service,
        top1_changed=(a0.ranking.candidates[0].service != a1.ranking.candidates[0].service),
        target_candidate_coverage_changed=(
            a0.target_candidate_coverage != a1.target_candidate_coverage
        ),
        shared_candidates=shared_attribution,
    )


def run_a0_a1_regression(
    *,
    case_id: str,
    dataset: DatasetName,
    alarm_service: str,
    topology: NormalTopology,
    evidence: Iterable[EvidenceRecord],
    entity_registry: TemporalEntityRegistry,
    evaluator_target_services: tuple[str, ...],
    source_fixture_sha256: str,
    a1_audit: A1NarrowAudit,
) -> A0A1RegressionArchive:
    """Run both candidate paths and attribute downstream rank differences exactly."""

    if not a1_audit.verify():
        raise ValueError("A1 narrow audit failed verification")
    records = tuple(evidence)
    targets = tuple(
        sorted({canonicalize_service_name(service) for service in evaluator_target_services})
    )
    if not targets:
        raise ValueError("regression requires evaluator-only target services")
    eligible = tuple(
        sorted(
            {
                canonicalize_service_name(record.service)
                for record in records
                if record.kind == "metric_anomaly"
            }
        )
    )
    a0_selection = build_a0_candidates(
        case_id=case_id,
        dataset=dataset,
        alarm_service=alarm_service,
        topology=topology,
        evidence=records,
    )
    a1_selection = build_a1_candidates(
        case_id=case_id,
        dataset=dataset,
        alarm_service=alarm_service,
        topology=topology,
        evidence=records,
        entity_registry=entity_registry,
    )
    a0 = _outcome(
        variant="A0",
        selection=a0_selection,
        records=records,
        eligible_metric_services=eligible,
        evaluator_target_services=targets,
    )
    a1 = _outcome(
        variant="A1",
        selection=a1_selection,
        records=records,
        eligible_metric_services=eligible,
        evaluator_target_services=targets,
    )
    attribution = _ranking_attribution(a0, a1)
    unsigned: dict[str, object] = {
        "schema_version": A0_A1_REGRESSION_SCHEMA_VERSION,
        "evaluation_scope": "isolated_evaluator_regression_only",
        "ranking_input_policy": _RANKING_INPUT_POLICY,
        "source_fixture_sha256": source_fixture_sha256,
        "a1_audit_sha256": a1_audit.audit_sha256,
        "case_id": case_id,
        "dataset": dataset,
        "a0": a0.model_dump(mode="json"),
        "a1": a1.model_dump(mode="json"),
        "ranking_attribution": attribution.model_dump(mode="json"),
    }
    archive = A0A1RegressionArchive.model_validate(
        {
            **unsigned,
            "archive_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )
    if not archive.verify():  # pragma: no cover - construction invariant
        raise AssertionError("A0/A1 regression archive construction failed")
    return archive
