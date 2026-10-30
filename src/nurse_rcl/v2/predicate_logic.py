"""Deterministic three-valued verification predicates.

Logical composition uses Strong Kleene semantics. For conjunction, any false
operand decides false, otherwise any unknown operand keeps the result unknown,
otherwise the result is true. For disjunction, any true operand decides true,
otherwise any unknown operand keeps the result unknown, otherwise the result is
false. Unknown is never coerced to a negative or normal observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from nurse_rcl.v2.diagnostic_schemas import (
    MeasuredValue,
    PredicateRequirement,
    PredicateResult,
    PredicateThreshold,
    TruthValue,
    build_predicate_result,
    derive_atomic_predicate_state,
)


def evaluate_predicate(
    *,
    predicate_kind: str,
    requirement: PredicateRequirement,
    measured_values: Sequence[MeasuredValue],
    threshold: PredicateThreshold | None,
    threshold_source: str | None,
    observation_ids: Sequence[str],
    verification_result_ids: Sequence[str],
    provenance_ids: Sequence[str],
) -> PredicateResult:
    """Evaluate one atomic threshold comparison without executing a query.

    Missing evidence, missing thresholds, and incompatible value types or units
    are explicit unknowns. Booleans never participate in numeric comparisons.
    """

    values = tuple(measured_values)

    def build(truth_value: TruthValue, reason: str) -> PredicateResult:
        return build_predicate_result(
            predicate_kind=predicate_kind,
            requirement=requirement,
            truth_value=truth_value,
            logical_operator="atomic",
            operand_states=(),
            observation_ids=observation_ids,
            verification_result_ids=verification_result_ids,
            measured_values=values,
            threshold=threshold,
            threshold_source=threshold_source,
            reason=reason,
            provenance_ids=provenance_ids,
        )

    truth_value, reason = derive_atomic_predicate_state(
        measured_values=values,
        threshold=threshold,
        threshold_source=threshold_source,
        observation_ids=observation_ids,
        verification_result_ids=verification_result_ids,
    )
    return build(truth_value, reason)


def strong_kleene_and(
    *,
    predicate_kind: str,
    requirement: PredicateRequirement,
    operands: Sequence[PredicateResult],
) -> PredicateResult:
    """Compose a nonempty conjunction: false > unknown > true."""

    return _compose(
        logical_operator="and",
        predicate_kind=predicate_kind,
        requirement=requirement,
        operands=operands,
    )


def strong_kleene_or(
    *,
    predicate_kind: str,
    requirement: PredicateRequirement,
    operands: Sequence[PredicateResult],
) -> PredicateResult:
    """Compose a nonempty disjunction: true > unknown > false."""

    return _compose(
        logical_operator="or",
        predicate_kind=predicate_kind,
        requirement=requirement,
        operands=operands,
    )


def _compose(
    *,
    logical_operator: Literal["and", "or"],
    predicate_kind: str,
    requirement: PredicateRequirement,
    operands: Sequence[PredicateResult],
) -> PredicateResult:
    if not operands:
        raise ValueError("logical composition requires at least one operand")
    by_id: dict[str, PredicateResult] = {}
    for operand in operands:
        if type(operand) is not PredicateResult or not operand.verify():
            raise ValueError("logical composition requires verified predicate operands")
        previous = by_id.setdefault(operand.predicate_result_id, operand)
        if previous != operand:
            raise ValueError("predicate operand ID collision")
    canonical = tuple(by_id[item] for item in sorted(by_id))
    truths = {item.truth_value for item in canonical}
    if logical_operator == "and":
        truth_value: TruthValue = (
            "false" if "false" in truths else "unknown" if "unknown" in truths else "true"
        )
    else:
        truth_value = "true" if "true" in truths else "unknown" if "unknown" in truths else "false"
    return build_predicate_result(
        predicate_kind=predicate_kind,
        requirement=requirement,
        truth_value=truth_value,
        logical_operator=logical_operator,
        operand_states=canonical,
        observation_ids=tuple(
            observation_id for item in canonical for observation_id in item.observation_ids
        ),
        verification_result_ids=tuple(
            result_id for item in canonical for result_id in item.verification_result_ids
        ),
        measured_values=(),
        threshold=None,
        threshold_source=None,
        reason=f"strong_kleene_{logical_operator}:{truth_value}",
        provenance_ids=tuple(
            provenance_id for item in canonical for provenance_id in item.provenance_ids
        ),
    )
