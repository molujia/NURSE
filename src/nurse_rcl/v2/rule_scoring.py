"""Explicit, candidate-scoped SRE rule scoring for NURSE V2."""

from __future__ import annotations

import hmac
from collections.abc import Iterator, Sequence
from heapq import merge
from typing import Literal, TypeVar, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.diagnostic_schemas import PredicateResult, TruthValue
from nurse_rcl.v2.ranking_schemas import RuleSet, ScoringRule
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

RULE_CANDIDATE_ASSIGNMENT_SCHEMA_VERSION: Literal["nurse.v2.rule-candidate-assignment/1"] = (
    "nurse.v2.rule-candidate-assignment/1"
)
RULE_CONTRIBUTION_SCHEMA_VERSION: Literal["nurse.v2.rule-contribution/1"] = (
    "nurse.v2.rule-contribution/1"
)
CANDIDATE_RULE_SCORE_SCHEMA_VERSION: Literal["nurse.v2.candidate-rule-score/1"] = (
    "nurse.v2.candidate-rule-score/1"
)
RULE_SCORING_RESULT_SCHEMA_VERSION: Literal["nurse.v2.rule-scoring-result/1"] = (
    "nurse.v2.rule-scoring-result/1"
)
RULE_SCORING_POLICY_VERSION: Literal["nurse-v2-explicit-sre-rule-scoring/1"] = (
    "nurse-v2-explicit-sre-rule-scoring/1"
)
RULE_SCORING_POLICY_SHA256 = sha256_hex(
    canonical_json_bytes(
        {
            "policy_version": RULE_SCORING_POLICY_VERSION,
            "activation": "enabled_and_all_referenced_predicates_true",
            "unknown_semantics": "false_precedes_unknown_precedes_true",
            "candidate_rule_score": "max(promote)-max(demote)",
            "default_directional_rules_enabled": False,
        }
    )
)

RuleActivationStatus = Literal[
    "applied",
    "disabled",
    "predicate_false",
    "predicate_unknown",
]

_MAX_INPUT_ARTIFACTS = 32_768
_MAX_PROVENANCE_IDS = 131_072
_MAX_SERIALIZED_STRING_LENGTH = 8_192
_SERIALIZATION_PREFLIGHT_MAX_DEPTH = 192
_SERIALIZATION_PREFLIGHT_MAX_NODES = 500_000
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)


class _RuleScoringResourceLimitError(ValueError):
    """Internal marker for fixed, redacted public budget failures."""


class RuleScoringBudgetExceeded(ValueError):
    """Raised when fixed rule-channel input or serialization ceilings are exceeded."""

    reason_code: Literal["input_budget_exceeded"] = "input_budget_exceeded"

    def __init__(self) -> None:
        super().__init__("rule scoring budget exceeded")


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> tuple[str, str]:
    digest = _digest(value)
    return f"{prefix}_{digest[:24]}", digest


def _authoritative_input_digest(
    *,
    candidate_freeze_sha256: str,
    rule_set_sha256: str,
    predicate_sha256s: tuple[str, ...],
    assignment_sha256s: tuple[str, ...],
) -> str:
    return _digest(
        {
            "candidate_freeze_sha256": candidate_freeze_sha256,
            "rule_set_sha256": rule_set_sha256,
            "policy_sha256": RULE_SCORING_POLICY_SHA256,
            "predicate_sha256s": predicate_sha256s,
            "assignment_sha256s": assignment_sha256s,
        }
    )


def _preflight_exact_tree(
    value: object,
    *,
    root_model_types: tuple[type[StrictFrozenModel], ...],
    label: str,
) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=root_model_types,
        label=label,
        max_depth=_SERIALIZATION_PREFLIGHT_MAX_DEPTH,
        max_nodes=_SERIALIZATION_PREFLIGHT_MAX_NODES,
        max_string_length=_MAX_SERIALIZED_STRING_LENGTH,
        provenance_id_limit=_MAX_PROVENANCE_IDS,
        resource_error_type=_RuleScoringResourceLimitError,
    )


def _preflight_model_tuple_limit(
    value: object,
    *,
    model_type: type[StrictFrozenModel],
    field_name: str,
    label: str,
) -> None:
    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or field_name not in state:
        raise ValueError(f"{label} lacks a required tuple field")
    field_value = dict.__getitem__(state, field_name)
    if type(field_value) is not tuple:
        raise ValueError(f"{label} field must use an exact tuple")
    if tuple.__len__(field_value) > _MAX_INPUT_ARTIFACTS:
        raise _RuleScoringResourceLimitError(f"{label} exceeds the declared model quantity limit")


def _same_exact_runtime_shape(value: object, rebuilt: object) -> bool:
    if type(value) is not type(rebuilt):
        return False
    if isinstance(rebuilt, StrictFrozenModel):
        try:
            left = object.__getattribute__(value, "__dict__")
            right = object.__getattribute__(rebuilt, "__dict__")
            left_extra = object.__getattribute__(value, "__pydantic_extra__")
            right_extra = object.__getattribute__(rebuilt, "__pydantic_extra__")
            left_private = object.__getattribute__(value, "__pydantic_private__")
            right_private = object.__getattribute__(rebuilt, "__pydantic_private__")
            left_fields = object.__getattribute__(value, "__pydantic_fields_set__")
            right_fields = object.__getattribute__(rebuilt, "__pydantic_fields_set__")
        except AttributeError:
            return False
        if (
            type(left) is not dict
            or type(right) is not dict
            or tuple(sorted(left)) != tuple(sorted(right))
            or left_extra is not None
            or right_extra is not None
            or left_private is not None
            or right_private is not None
            or type(left_fields) is not set
            or type(right_fields) is not set
            or left_fields != right_fields
        ):
            return False
        return all(
            _same_exact_runtime_shape(
                dict.__getitem__(left, field_name),
                dict.__getitem__(right, field_name),
            )
            for field_name in right
        )
    if type(rebuilt) is tuple:
        left_sequence = cast(tuple[object, ...], value)
        right_sequence = cast(tuple[object, ...], rebuilt)
        return tuple.__len__(left_sequence) == tuple.__len__(right_sequence) and all(
            _same_exact_runtime_shape(left_item, right_item)
            for left_item, right_item in zip(left_sequence, right_sequence, strict=True)
        )
    if type(rebuilt) is list:
        left_list = cast(list[object], value)
        right_list = cast(list[object], rebuilt)
        return list.__len__(left_list) == list.__len__(right_list) and all(
            _same_exact_runtime_shape(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    if type(rebuilt) is dict:
        left_mapping = cast(dict[object, object], value)
        right_mapping = cast(dict[object, object], rebuilt)
        return set(dict.keys(left_mapping)) == set(dict.keys(right_mapping)) and all(
            _same_exact_runtime_shape(
                dict.__getitem__(left_mapping, key),
                dict.__getitem__(right_mapping, key),
            )
            for key in dict.keys(right_mapping)
        )
    return True


def _trusted_exact_model(
    value: object,
    model_type: type[_ModelT],
    *,
    label: str,
) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError(f"{label} must use the exact model type")
    _preflight_exact_tree(
        value,
        root_model_types=(model_type,),
        label=label,
    )
    document = model_type.model_dump(value, mode="json", warnings="error")
    rebuilt = model_type.model_validate(document)
    if not _same_exact_runtime_shape(value, rebuilt):
        raise ValueError(f"{label} is not an exact canonical model")
    if model_type.model_dump(rebuilt, mode="json", warnings="error") != document:
        raise ValueError(f"{label} is not canonical")
    return rebuilt


def _plain_provenance_ids(values: object) -> tuple[str, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError("provenance IDs must use an exact tuple or list")
    sequence: Sequence[object]
    if type(values) is tuple:
        sequence = cast(tuple[object, ...], values)
    else:
        sequence = cast(list[object], values)
    count = len(sequence)
    if count > _MAX_PROVENANCE_IDS:
        raise _RuleScoringResourceLimitError("provenance IDs exceed the assignment limit")
    if count < 1:
        raise ValueError("provenance IDs size is outside the assignment limit")
    if any(
        type(item) is not str
        or str.__len__(item) < 1
        or str.__len__(item) > _MAX_SERIALIZED_STRING_LENGTH
        for item in sequence
    ):
        if any(
            type(item) is str and str.__len__(item) > _MAX_SERIALIZED_STRING_LENGTH
            for item in sequence
        ):
            raise _RuleScoringResourceLimitError("provenance IDs exceed the string length limit")
        raise ValueError("provenance IDs contain a malformed value")
    typed = cast(Sequence[str], sequence)
    if len(set(typed)) != count:
        raise ValueError("provenance IDs must be unique")
    return tuple(sorted(typed))


def _has_valid_provenance_ids(values: object) -> bool:
    if type(values) is not tuple:
        return False
    typed = cast(tuple[object, ...], values)
    if not typed or tuple.__len__(typed) > _MAX_PROVENANCE_IDS:
        return False
    if any(
        type(item) is not str
        or str.__len__(item) < 1
        or str.__len__(item) > _MAX_SERIALIZED_STRING_LENGTH
        for item in typed
    ):
        return False
    strings = cast(tuple[str, ...], typed)
    return strings == tuple(sorted(set(strings)))


def _predicate_tree_has_valid_provenance(root: PredicateResult) -> bool:
    pending = [root]
    while pending:
        current = pending.pop()
        if not _has_valid_provenance_ids(current.provenance_ids):
            return False
        pending.extend(current.operand_states)
    return True


def _plain_models(
    values: object,
    model_type: type[_ModelT],
    *,
    label: str,
) -> tuple[_ModelT, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError(f"{label} must use an exact tuple or list")
    sequence: Sequence[object]
    if type(values) is tuple:
        sequence = cast(tuple[object, ...], values)
    else:
        sequence = cast(list[object], values)
    count = len(sequence)
    if count > _MAX_INPUT_ARTIFACTS:
        raise _RuleScoringResourceLimitError(f"{label} exceeds the artifact limit")
    _preflight_exact_tree(
        sequence,
        root_model_types=(model_type,),
        label=label,
    )
    return tuple(_trusted_exact_model(item, model_type, label=label) for item in sequence)


class RuleCandidateAssignment(StrictFrozenModel):
    """Explicitly bind one declared SRE rule to one frozen candidate."""

    schema_version: Literal["nurse.v2.rule-candidate-assignment/1"] = (
        RULE_CANDIDATE_ASSIGNMENT_SCHEMA_VERSION
    )
    assignment_id: str = Field(pattern=r"^sra_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_service_key: str = Field(min_length=1)
    rule_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_assignment(self) -> Self:
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("rule assignment provenance IDs must be sorted and unique")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                RuleCandidateAssignment,
                label="rule assignment",
            )
            unsigned = RuleCandidateAssignment.model_dump(
                rebuilt,
                mode="json",
                exclude={"assignment_id", "assignment_sha256"},
            )
            identifier, digest = _stable_id("sra", unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return (
            _has_valid_provenance_ids(rebuilt.provenance_ids)
            and rebuilt.assignment_id == identifier
            and rebuilt.assignment_sha256 == digest
        )


class RuleContribution(StrictFrozenModel):
    """One retained rule application, whether effective or not."""

    schema_version: Literal["nurse.v2.rule-contribution/1"] = RULE_CONTRIBUTION_SCHEMA_VERSION
    contribution_id: str = Field(pattern=r"^rcon_[0-9a-f]{24}$")
    assignment_id: str = Field(pattern=r"^sra_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    rule_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    rule_version: str = Field(min_length=1)
    rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    direction: Literal["promote", "demote"]
    configured_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    predicate_result_ids: tuple[str, ...] = Field(min_length=1)
    predicate_truth_values: tuple[TruthValue, ...] = Field(min_length=1)
    activation_status: RuleActivationStatus
    effective_strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    selected_for_direction_max: bool
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    contribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_auditable_activation(self) -> Self:
        if self.predicate_result_ids != tuple(sorted(set(self.predicate_result_ids))):
            raise ValueError("rule predicate IDs must be sorted and unique")
        if len(self.predicate_truth_values) != len(self.predicate_result_ids):
            raise ValueError("rule predicate truths must align with predicate IDs")
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("rule contribution provenance IDs must be sorted and unique")
        expected_strength = self.configured_strength if self.activation_status == "applied" else 0.0
        if self.effective_strength != expected_strength:
            raise ValueError("rule contribution strength differs from activation status")
        if self.selected_for_direction_max and self.activation_status != "applied":
            raise ValueError("an inactive rule contribution cannot define a maximum")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                RuleContribution,
                label="rule contribution",
            )
            truths = rebuilt.predicate_truth_values
            if rebuilt.activation_status == "applied" and any(value != "true" for value in truths):
                return False
            if rebuilt.activation_status == "predicate_false" and "false" not in truths:
                return False
            if rebuilt.activation_status == "predicate_unknown" and (
                "false" in truths or "unknown" not in truths
            ):
                return False
            if not _has_valid_provenance_ids(rebuilt.provenance_ids):
                return False
            unsigned = RuleContribution.model_dump(
                rebuilt,
                mode="json",
                exclude={"contribution_id", "contribution_sha256"},
            )
            identifier, digest = _stable_id("rcon", unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return rebuilt.contribution_id == identifier and rebuilt.contribution_sha256 == digest


class CandidateRuleScore(StrictFrozenModel):
    """Auditable signed rule channel for one candidate."""

    schema_version: Literal["nurse.v2.candidate-rule-score/1"] = CANDIDATE_RULE_SCORE_SCHEMA_VERSION
    candidate_rule_score_id: str = Field(pattern=r"^crs_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    contributions: tuple[RuleContribution, ...]
    selected_promote_contribution_ids: tuple[str, ...]
    selected_demote_contribution_ids: tuple[str, ...]
    promote_max: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    demote_max: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rule_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    candidate_rule_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_empty_or_scored_channel(self) -> Self:
        if self.contributions != tuple(
            sorted(self.contributions, key=lambda item: item.contribution_id)
        ):
            raise ValueError("rule contributions must be canonically ordered")
        if self.selected_promote_contribution_ids != tuple(
            sorted(set(self.selected_promote_contribution_ids))
        ) or self.selected_demote_contribution_ids != tuple(
            sorted(set(self.selected_demote_contribution_ids))
        ):
            raise ValueError("selected rule contribution IDs must be sorted and unique")
        if self.rule_score != self.promote_max - self.demote_max:
            raise ValueError("candidate rule score differs from directional maxima")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                CandidateRuleScore,
                label="candidate rule score",
            )
            contributions = rebuilt.contributions
            if any(not RuleContribution.verify(item) for item in contributions):
                return False
            for values in (
                tuple(item.contribution_id for item in contributions),
                tuple(item.contribution_sha256 for item in contributions),
                tuple(item.assignment_id for item in contributions),
                tuple(item.rule_key for item in contributions),
            ):
                if len(values) != len(set(values)):
                    return False
            if any(
                item.candidate_service_key != rebuilt.candidate_service_key
                for item in contributions
            ):
                return False
            promotes = tuple(
                item.effective_strength
                for item in contributions
                if item.direction == "promote" and item.activation_status == "applied"
            )
            demotes = tuple(
                item.effective_strength
                for item in contributions
                if item.direction == "demote" and item.activation_status == "applied"
            )
            if rebuilt.promote_max != max(promotes, default=0.0) or rebuilt.demote_max != max(
                demotes, default=0.0
            ):
                return False
            expected_promote_ids = tuple(
                sorted(
                    item.contribution_id
                    for item in contributions
                    if item.direction == "promote"
                    and item.activation_status == "applied"
                    and item.effective_strength == rebuilt.promote_max
                )
            )
            expected_demote_ids = tuple(
                sorted(
                    item.contribution_id
                    for item in contributions
                    if item.direction == "demote"
                    and item.activation_status == "applied"
                    and item.effective_strength == rebuilt.demote_max
                )
            )
            if (
                rebuilt.selected_promote_contribution_ids != expected_promote_ids
                or rebuilt.selected_demote_contribution_ids != expected_demote_ids
                or {
                    item.contribution_id
                    for item in contributions
                    if item.selected_for_direction_max
                }
                != set((*expected_promote_ids, *expected_demote_ids))
            ):
                return False
            unsigned = CandidateRuleScore.model_dump(
                rebuilt,
                mode="json",
                exclude={
                    "candidate_rule_score_id",
                    "candidate_rule_score_sha256",
                },
            )
            identifier, digest = _stable_id("crs", unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return (
            rebuilt.candidate_rule_score_id == identifier
            and rebuilt.candidate_rule_score_sha256 == digest
        )


class RuleScoringResult(StrictFrozenModel):
    """Case-atomic result for the explicit SRE rule channel."""

    schema_version: Literal["nurse.v2.rule-scoring-result/1"] = RULE_SCORING_RESULT_SCHEMA_VERSION
    rule_scoring_result_id: str = Field(pattern=r"^rrs_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["nurse-v2-explicit-sre-rule-scoring/1"] = RULE_SCORING_POLICY_VERSION
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    default_directional_rules_enabled: Literal[False] = False
    predicate_sha256s: tuple[str, ...]
    assignment_sha256s: tuple[str, ...]
    candidates: tuple[CandidateRuleScore, ...] = Field(min_length=1)
    authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_case_result(self) -> Self:
        if self.policy_sha256 != RULE_SCORING_POLICY_SHA256:
            raise ValueError("rule scoring policy digest differs")
        if self.candidates != tuple(
            sorted(self.candidates, key=lambda item: item.candidate_service_key)
        ):
            raise ValueError("candidate rule scores must be canonically ordered")
        if len({item.candidate_service_key for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate rule scores must be unique")
        for values, name in (
            (self.predicate_sha256s, "predicate digests"),
            (self.assignment_sha256s, "assignment digests"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    def verify_structure(self) -> bool:
        try:
            rebuilt = _trusted_exact_model(
                self,
                RuleScoringResult,
                label="rule scoring result",
            )
            if any(not CandidateRuleScore.verify(item) for item in rebuilt.candidates):
                return False
            for values in (
                tuple(item.candidate_rule_score_id for item in rebuilt.candidates),
                tuple(item.candidate_rule_score_sha256 for item in rebuilt.candidates),
                tuple(
                    contribution.contribution_id
                    for candidate in rebuilt.candidates
                    for contribution in candidate.contributions
                ),
                tuple(
                    contribution.contribution_sha256
                    for candidate in rebuilt.candidates
                    for contribution in candidate.contributions
                ),
                tuple(
                    contribution.assignment_id
                    for candidate in rebuilt.candidates
                    for contribution in candidate.contributions
                ),
            ):
                if len(values) != len(set(values)):
                    return False
            if rebuilt.authoritative_input_sha256 != _authoritative_input_digest(
                candidate_freeze_sha256=rebuilt.candidate_freeze_sha256,
                rule_set_sha256=rebuilt.rule_set_sha256,
                predicate_sha256s=rebuilt.predicate_sha256s,
                assignment_sha256s=rebuilt.assignment_sha256s,
            ):
                return False
            unsigned = RuleScoringResult.model_dump(
                rebuilt,
                mode="json",
                exclude={"rule_scoring_result_id", "result_sha256"},
            )
            identifier, digest = _stable_id("rrs", unsigned)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return rebuilt.rule_scoring_result_id == identifier and rebuilt.result_sha256 == digest

    def verify(self) -> bool:
        """Compatibility alias for structural, not authoritative, verification."""

        return RuleScoringResult.verify_structure(self)

    def verify_trusted(
        self,
        *,
        expected_result_sha256: str,
        expected_authoritative_input_sha256: str,
        frozen_candidates: FrozenCandidateUniverse,
        rule_set: RuleSet,
        assignments: Sequence[RuleCandidateAssignment],
        predicates: Sequence[PredicateResult],
    ) -> bool:
        """Replay from complete trusted inputs under two external digest anchors."""

        if any(
            type(value) is not str
            or str.__len__(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                expected_result_sha256,
                expected_authoritative_input_sha256,
            )
        ):
            return False
        try:
            rebuilt = _trusted_exact_model(
                self,
                RuleScoringResult,
                label="rule scoring result",
            )
            if (
                not RuleScoringResult.verify_structure(rebuilt)
                or not hmac.compare_digest(
                    rebuilt.result_sha256,
                    expected_result_sha256,
                )
                or not hmac.compare_digest(
                    rebuilt.authoritative_input_sha256,
                    expected_authoritative_input_sha256,
                )
            ):
                return False
            replayed = score_explicit_rules(
                frozen_candidates=frozen_candidates,
                rule_set=rule_set,
                assignments=assignments,
                predicates=predicates,
            )
            replayed_document = RuleScoringResult.model_dump(
                replayed,
                mode="json",
                warnings="error",
            )
            rebuilt_document = RuleScoringResult.model_dump(
                rebuilt,
                mode="json",
                warnings="error",
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return replayed_document == rebuilt_document


def _empty_candidate_score(candidate_service_key: str) -> CandidateRuleScore:
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_RULE_SCORE_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "contributions": (),
        "selected_promote_contribution_ids": (),
        "selected_demote_contribution_ids": (),
        "promote_max": 0.0,
        "demote_max": 0.0,
        "rule_score": 0.0,
    }
    identifier, digest = _stable_id("crs", unsigned)
    return CandidateRuleScore.model_validate(
        {
            **unsigned,
            "candidate_rule_score_id": identifier,
            "candidate_rule_score_sha256": digest,
        }
    )


def build_rule_candidate_assignment(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    rule: ScoringRule,
    candidate_service_key: str,
    provenance_ids: Sequence[str],
) -> RuleCandidateAssignment:
    """Bind one verified explicit rule to one frozen candidate."""

    trusted_inputs: (
        tuple[
            FrozenCandidateUniverse,
            ScoringRule,
            tuple[str, ...],
        ]
        | None
    ) = None
    budget_exceeded = False
    try:
        provenance = _plain_provenance_ids(provenance_ids)
        if (
            type(candidate_service_key) is not str
            or str.__len__(candidate_service_key) < 1
            or str.__len__(candidate_service_key) > _MAX_SERIALIZED_STRING_LENGTH
        ):
            raise ValueError("candidate service key is malformed")
        _preflight_model_tuple_limit(
            frozen_candidates,
            model_type=FrozenCandidateUniverse,
            field_name="candidate_service_keys",
            label="candidate freeze",
        )
        _preflight_model_tuple_limit(
            rule,
            model_type=ScoringRule,
            field_name="predicate_result_ids",
            label="scoring rule",
        )
        frozen = _trusted_exact_model(
            frozen_candidates,
            FrozenCandidateUniverse,
            label="candidate freeze",
        )
        verified_rule = _trusted_exact_model(rule, ScoringRule, label="scoring rule")
        if (
            not FrozenCandidateUniverse.verify(frozen)
            or not ScoringRule.verify(verified_rule)
            or not _has_valid_provenance_ids(verified_rule.provenance_ids)
        ):
            raise ValueError("assignment authority failed verification")
        trusted_inputs = (frozen, verified_rule, provenance)
    except _RuleScoringResourceLimitError:
        budget_exceeded = True
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if budget_exceeded:
        raise RuleScoringBudgetExceeded() from None
    if trusted_inputs is None:
        raise ValueError("rule assignment input verification failed") from None
    frozen, verified_rule, provenance = trusted_inputs
    if candidate_service_key not in frozen.candidate_service_keys:
        raise ValueError("rule assignment references a candidate outside the frozen universe")
    unsigned: dict[str, object] = {
        "schema_version": RULE_CANDIDATE_ASSIGNMENT_SCHEMA_VERSION,
        "opaque_case_id": frozen.opaque_case_id,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "candidate_service_key": candidate_service_key,
        "rule_key": verified_rule.rule_key,
        "rule_sha256": verified_rule.rule_sha256,
        "provenance_ids": provenance,
    }
    identifier, digest = _stable_id("sra", unsigned)
    assignment = RuleCandidateAssignment.model_validate(
        {**unsigned, "assignment_id": identifier, "assignment_sha256": digest}
    )
    if not assignment.verify():  # pragma: no cover - construction invariant
        raise AssertionError("rule assignment verification failed")
    return assignment


def _activation_status(rule: ScoringRule, truths: Sequence[TruthValue]) -> RuleActivationStatus:
    if not rule.enabled:
        return "disabled"
    if "false" in truths:
        return "predicate_false"
    if "unknown" in truths:
        return "predicate_unknown"
    return "applied"


def _iter_bounded_merged_provenance_ids(
    *,
    assignment: RuleCandidateAssignment,
    rule: ScoringRule,
    predicates: Sequence[PredicateResult],
) -> Iterator[str]:
    streams = (
        assignment.provenance_ids,
        rule.provenance_ids,
        *(item.provenance_ids for item in predicates),
    )
    count = 0
    previous: str | None = None
    for provenance_id in merge(*streams):
        if provenance_id == previous:
            continue
        count += 1
        if count > _MAX_PROVENANCE_IDS:
            raise _RuleScoringResourceLimitError("rule contribution exceeds the provenance limit")
        previous = provenance_id
        yield provenance_id


def _merged_provenance_count(
    *,
    assignment: RuleCandidateAssignment,
    rule: ScoringRule,
    predicates: Sequence[PredicateResult],
) -> int:
    return sum(
        1
        for _ in _iter_bounded_merged_provenance_ids(
            assignment=assignment,
            rule=rule,
            predicates=predicates,
        )
    )


def _add_projected_cost(*, current: int, additional: int, limit: int) -> int:
    if additional > limit - current:
        raise _RuleScoringResourceLimitError("rule scoring output exceeds its fixed budget")
    return current + additional


def _preflight_projected_output(
    *,
    candidate_service_keys: tuple[str, ...],
    assignments: tuple[RuleCandidateAssignment, ...],
    rule_by_key: dict[str, ScoringRule],
    predicate_by_id: dict[str, PredicateResult],
) -> None:
    """Bound exact output costs before constructing any result model."""

    candidate_document_base_nodes = 1 + len(CandidateRuleScore.model_fields)
    contribution_document_base_nodes = 1 + len(RuleContribution.model_fields)
    result_document_base_nodes = 1 + len(RuleScoringResult.model_fields)
    projected_nodes = _add_projected_cost(
        current=0,
        additional=(
            result_document_base_nodes
            + len(predicate_by_id)
            + len(assignments)
            + len(candidate_service_keys) * candidate_document_base_nodes
        ),
        limit=_SERIALIZATION_PREFLIGHT_MAX_NODES,
    )
    projected_provenance_ids = 0
    promote_max_by_candidate = dict.fromkeys(candidate_service_keys, 0.0)
    demote_max_by_candidate = dict.fromkeys(candidate_service_keys, 0.0)

    for assignment in assignments:
        rule = rule_by_key[assignment.rule_key]
        predicates = tuple(predicate_by_id[value] for value in rule.predicate_result_ids)
        provenance_count = _merged_provenance_count(
            assignment=assignment,
            rule=rule,
            predicates=predicates,
        )
        projected_provenance_ids = _add_projected_cost(
            current=projected_provenance_ids,
            additional=provenance_count,
            limit=_MAX_PROVENANCE_IDS,
        )
        projected_nodes = _add_projected_cost(
            current=projected_nodes,
            additional=(
                contribution_document_base_nodes
                + 2 * len(rule.predicate_result_ids)
                + provenance_count
            ),
            limit=_SERIALIZATION_PREFLIGHT_MAX_NODES,
        )
        truths = tuple(item.truth_value for item in predicates)
        if _activation_status(rule, truths) != "applied":
            continue
        maxima = (
            promote_max_by_candidate if rule.direction == "promote" else demote_max_by_candidate
        )
        maxima[assignment.candidate_service_key] = max(
            maxima[assignment.candidate_service_key], rule.strength
        )

    for assignment in assignments:
        rule = rule_by_key[assignment.rule_key]
        predicates = tuple(predicate_by_id[value] for value in rule.predicate_result_ids)
        truths = tuple(item.truth_value for item in predicates)
        if _activation_status(rule, truths) != "applied":
            continue
        directional_max = (
            promote_max_by_candidate[assignment.candidate_service_key]
            if rule.direction == "promote"
            else demote_max_by_candidate[assignment.candidate_service_key]
        )
        if rule.strength == directional_max:
            projected_nodes = _add_projected_cost(
                current=projected_nodes,
                additional=1,
                limit=_SERIALIZATION_PREFLIGHT_MAX_NODES,
            )


def _build_contribution(
    *,
    assignment: RuleCandidateAssignment,
    rule: ScoringRule,
    predicates: Sequence[PredicateResult],
    selected_for_direction_max: bool,
) -> RuleContribution:
    truths = tuple(item.truth_value for item in predicates)
    status = _activation_status(rule, truths)
    unsigned: dict[str, object] = {
        "schema_version": RULE_CONTRIBUTION_SCHEMA_VERSION,
        "assignment_id": assignment.assignment_id,
        "candidate_service_key": assignment.candidate_service_key,
        "rule_key": rule.rule_key,
        "rule_version": rule.rule_version,
        "rule_sha256": rule.rule_sha256,
        "direction": rule.direction,
        "configured_strength": rule.strength,
        "predicate_result_ids": rule.predicate_result_ids,
        "predicate_truth_values": truths,
        "activation_status": status,
        "effective_strength": rule.strength if status == "applied" else 0.0,
        "selected_for_direction_max": selected_for_direction_max,
        "provenance_ids": tuple(
            _iter_bounded_merged_provenance_ids(
                assignment=assignment,
                rule=rule,
                predicates=predicates,
            )
        ),
    }
    identifier, digest = _stable_id("rcon", unsigned)
    contribution = RuleContribution.model_validate(
        {**unsigned, "contribution_id": identifier, "contribution_sha256": digest}
    )
    if not contribution.verify():  # pragma: no cover - construction invariant
        raise AssertionError("rule contribution verification failed")
    return contribution


def _candidate_score(
    *,
    candidate_service_key: str,
    assignments: Sequence[RuleCandidateAssignment],
    rule_by_key: dict[str, ScoringRule],
    predicate_by_id: dict[str, PredicateResult],
) -> CandidateRuleScore:
    drafts: list[
        tuple[RuleCandidateAssignment, ScoringRule, tuple[PredicateResult, ...], float]
    ] = []
    for assignment in assignments:
        rule = rule_by_key[assignment.rule_key]
        predicates = tuple(predicate_by_id[value] for value in rule.predicate_result_ids)
        status = _activation_status(rule, tuple(item.truth_value for item in predicates))
        effective = rule.strength if status == "applied" else 0.0
        drafts.append((assignment, rule, predicates, effective))
    promote_max = max(
        (
            effective
            for _, rule, _, effective in drafts
            if rule.direction == "promote" and rule.enabled
        ),
        default=0.0,
    )
    demote_max = max(
        (
            effective
            for _, rule, _, effective in drafts
            if rule.direction == "demote" and rule.enabled
        ),
        default=0.0,
    )
    contributions = tuple(
        sorted(
            (
                _build_contribution(
                    assignment=assignment,
                    rule=rule,
                    predicates=predicates,
                    selected_for_direction_max=(
                        _activation_status(
                            rule,
                            tuple(item.truth_value for item in predicates),
                        )
                        == "applied"
                        and effective
                        == (promote_max if rule.direction == "promote" else demote_max)
                    ),
                )
                for assignment, rule, predicates, effective in drafts
            ),
            key=lambda item: item.contribution_id,
        )
    )
    selected_promotes = tuple(
        item.contribution_id
        for item in contributions
        if item.direction == "promote" and item.selected_for_direction_max
    )
    selected_demotes = tuple(
        item.contribution_id
        for item in contributions
        if item.direction == "demote" and item.selected_for_direction_max
    )
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_RULE_SCORE_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "contributions": tuple(item.model_dump(mode="json") for item in contributions),
        "selected_promote_contribution_ids": selected_promotes,
        "selected_demote_contribution_ids": selected_demotes,
        "promote_max": promote_max,
        "demote_max": demote_max,
        "rule_score": promote_max - demote_max,
    }
    identifier, digest = _stable_id("crs", unsigned)
    result = CandidateRuleScore.model_validate(
        {
            **unsigned,
            "candidate_rule_score_id": identifier,
            "candidate_rule_score_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("candidate rule score verification failed")
    return result


def score_explicit_rules(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    rule_set: RuleSet,
    assignments: Sequence[RuleCandidateAssignment],
    predicates: Sequence[PredicateResult],
) -> RuleScoringResult:
    """Score only explicit, predicate-backed SRE rule assignments."""

    if type(assignments) not in {tuple, list} or type(predicates) not in {tuple, list}:
        raise ValueError("rule scoring sequences must use an exact tuple or list")
    assignment_count = (
        tuple.__len__(cast(tuple[object, ...], assignments))
        if type(assignments) is tuple
        else list.__len__(cast(list[object], assignments))
    )
    predicate_count = (
        tuple.__len__(cast(tuple[object, ...], predicates))
        if type(predicates) is tuple
        else list.__len__(cast(list[object], predicates))
    )
    if assignment_count > _MAX_INPUT_ARTIFACTS or predicate_count > _MAX_INPUT_ARTIFACTS:
        raise RuleScoringBudgetExceeded() from None
    trusted_inputs: (
        tuple[
            FrozenCandidateUniverse,
            RuleSet,
            tuple[RuleCandidateAssignment, ...],
            tuple[PredicateResult, ...],
        ]
        | None
    ) = None
    budget_exceeded = False
    try:
        _preflight_model_tuple_limit(
            frozen_candidates,
            model_type=FrozenCandidateUniverse,
            field_name="candidate_service_keys",
            label="candidate freeze",
        )
        _preflight_model_tuple_limit(
            rule_set,
            model_type=RuleSet,
            field_name="rules",
            label="rule set",
        )
        frozen = _trusted_exact_model(
            frozen_candidates,
            FrozenCandidateUniverse,
            label="candidate freeze",
        )
        rules = _trusted_exact_model(rule_set, RuleSet, label="rule set")
        assignment_values = _plain_models(
            assignments,
            RuleCandidateAssignment,
            label="rule assignments",
        )
        predicate_values = _plain_models(
            predicates,
            PredicateResult,
            label="rule predicates",
        )
        if (
            not FrozenCandidateUniverse.verify(frozen)
            or not RuleSet.verify(rules)
            or any(not RuleCandidateAssignment.verify(item) for item in assignment_values)
            or any(
                not PredicateResult.verify(item) or not _predicate_tree_has_valid_provenance(item)
                for item in predicate_values
            )
            or any(not _has_valid_provenance_ids(item.provenance_ids) for item in rules.rules)
        ):
            raise ValueError("rule scoring authority failed verification")
        trusted_inputs = (frozen, rules, assignment_values, predicate_values)
    except _RuleScoringResourceLimitError:
        budget_exceeded = True
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if budget_exceeded:
        raise RuleScoringBudgetExceeded() from None
    if trusted_inputs is None:
        raise ValueError("rule scoring input verification failed") from None
    frozen, rules, assignment_values, predicate_values = trusted_inputs
    assignment_keys = tuple(
        (item.candidate_service_key, item.rule_key) for item in assignment_values
    )
    if len(assignment_keys) != len(set(assignment_keys)):
        raise ValueError("duplicate candidate rule assignment")
    identity_fields = (
        tuple(item.rule_key for item in rules.rules),
        tuple(item.rule_sha256 for item in rules.rules),
        tuple(item.assignment_id for item in assignment_values),
        tuple(item.assignment_sha256 for item in assignment_values),
        tuple(item.predicate_result_id for item in predicate_values),
        tuple(item.predicate_sha256 for item in predicate_values),
    )
    if any(len(values) != len(set(values)) for values in identity_fields):
        raise ValueError("rule scoring input identity collision")
    predicate_by_id = {item.predicate_result_id: item for item in predicate_values}
    if len(predicate_by_id) != len(predicate_values):
        raise ValueError("duplicate rule predicate")
    rule_by_key = {item.rule_key: item for item in rules.rules}
    if any(
        item.opaque_case_id != frozen.opaque_case_id
        or item.candidate_freeze_sha256 != frozen.freeze_sha256
        or item.candidate_service_key not in frozen.candidate_service_keys
        or item.rule_key not in rule_by_key
        or item.rule_sha256 != rule_by_key[item.rule_key].rule_sha256
        for item in assignment_values
    ):
        raise ValueError("rule assignment is outside the authoritative input scope")
    assigned_rule_keys = {item.rule_key for item in assignment_values}
    if assigned_rule_keys != set(rule_by_key):
        raise ValueError("every explicit rule must have at least one candidate assignment")
    referenced_predicate_ids = {
        value for rule in rules.rules for value in rule.predicate_result_ids
    }
    if referenced_predicate_ids != set(predicate_by_id):
        raise ValueError("rule predicates must exactly cover explicit rule references")
    projected_output_budget_exceeded = False
    try:
        _preflight_projected_output(
            candidate_service_keys=frozen.candidate_service_keys,
            assignments=assignment_values,
            rule_by_key=rule_by_key,
            predicate_by_id=predicate_by_id,
        )
    except _RuleScoringResourceLimitError:
        projected_output_budget_exceeded = True
    if projected_output_budget_exceeded:
        raise RuleScoringBudgetExceeded() from None
    assignment_by_candidate: dict[str, list[RuleCandidateAssignment]] = {}
    for assignment in assignment_values:
        assignment_by_candidate.setdefault(assignment.candidate_service_key, []).append(assignment)
    candidates = tuple(
        _candidate_score(
            candidate_service_key=candidate_service_key,
            assignments=tuple(
                sorted(
                    assignment_by_candidate.get(candidate_service_key, ()),
                    key=lambda item: item.assignment_id,
                )
            ),
            rule_by_key=rule_by_key,
            predicate_by_id=predicate_by_id,
        )
        for candidate_service_key in frozen.candidate_service_keys
    )
    predicate_sha256s = tuple(sorted(item.predicate_sha256 for item in predicate_values))
    assignment_sha256s = tuple(sorted(item.assignment_sha256 for item in assignment_values))
    authoritative_input_sha256 = _authoritative_input_digest(
        candidate_freeze_sha256=frozen.freeze_sha256,
        rule_set_sha256=rules.rule_set_sha256,
        predicate_sha256s=predicate_sha256s,
        assignment_sha256s=assignment_sha256s,
    )
    unsigned: dict[str, object] = {
        "schema_version": RULE_SCORING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": frozen.opaque_case_id,
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "rule_set_sha256": rules.rule_set_sha256,
        "policy_version": RULE_SCORING_POLICY_VERSION,
        "policy_sha256": RULE_SCORING_POLICY_SHA256,
        "default_directional_rules_enabled": False,
        "predicate_sha256s": predicate_sha256s,
        "assignment_sha256s": assignment_sha256s,
        "candidates": tuple(item.model_dump(mode="json") for item in candidates),
        "authoritative_input_sha256": authoritative_input_sha256,
    }
    identifier, digest = _stable_id("rrs", unsigned)
    result = RuleScoringResult.model_validate(
        {
            **unsigned,
            "rule_scoring_result_id": identifier,
            "result_sha256": digest,
        }
    )
    if not result.verify():  # pragma: no cover - construction invariant
        raise AssertionError("rule scoring result verification failed")
    return result
