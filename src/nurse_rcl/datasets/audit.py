"""Fixed-population audits for label-isolated dataset manifests."""

from __future__ import annotations

from typing import Literal

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import (
    DatasetName,
    EvaluationManifest,
    InferenceManifest,
    StrictFrozenModel,
)

_EXPECTED_POPULATIONS: dict[DatasetName, tuple[int, int]] = {
    "rcabench": (1_422, 0),
    "aiops25": (318, 82),
}


class PopulationAudit(StrictFrozenModel):
    """Label-free proof that every fixed benchmark case is classified once."""

    schema_version: Literal[1] = 1
    dataset: DatasetName
    expected_included: int
    expected_excluded: int
    observed_inference: int
    observed_included: int
    observed_excluded: int
    classified_case_ids_sha256: str


def audit_population(
    inference: InferenceManifest,
    evaluation: EvaluationManifest,
) -> PopulationAudit:
    """Reject count, uniqueness, overlap, and classification drift."""

    if inference.dataset != evaluation.dataset:
        raise ValueError("inference/evaluation dataset mismatch")
    dataset = inference.dataset
    expected_included, expected_excluded = _EXPECTED_POPULATIONS[dataset]

    inference_ids = [case.case_id for case in inference.cases]
    included_ids = [case.case_id for case in evaluation.cases if not case.excluded]
    excluded_ids = [case.case_id for case in evaluation.cases if case.excluded]
    _reject_duplicates(inference_ids, "inference")
    _reject_duplicates(included_ids, "included evaluation")
    _reject_duplicates(excluded_ids, "excluded evaluation")

    included_set = set(included_ids)
    excluded_set = set(excluded_ids)
    overlap = included_set & excluded_set
    if overlap:
        raise ValueError(f"case is both included and excluded: {min(overlap)}")
    if set(inference_ids) != included_set:
        raise ValueError("inference/evaluation classification mismatch")

    observed_included = len(included_ids)
    observed_excluded = len(excluded_ids)
    if observed_included < expected_included or observed_excluded < expected_excluded:
        raise ValueError(
            "unclassified population: "
            f"expected {expected_included} included/{expected_excluded} excluded, "
            f"found {observed_included}/{observed_excluded}"
        )
    if observed_included != expected_included or observed_excluded != expected_excluded:
        raise ValueError(
            "unexpected population: "
            f"expected {expected_included} included/{expected_excluded} excluded, "
            f"found {observed_included}/{observed_excluded}"
        )

    population_digest = sha256_hex(
        canonical_json_bytes(
            {
                "dataset": dataset,
                "included": sorted(included_ids),
                "excluded": sorted(excluded_ids),
            }
        )
    )
    return PopulationAudit(
        dataset=dataset,
        expected_included=expected_included,
        expected_excluded=expected_excluded,
        observed_inference=len(inference_ids),
        observed_included=observed_included,
        observed_excluded=observed_excluded,
        classified_case_ids_sha256=population_digest,
    )


def _reject_duplicates(case_ids: list[str], population: str) -> None:
    seen: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            raise ValueError(f"duplicate {population} case ID: {case_id}")
        seen.add(case_id)
