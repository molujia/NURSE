"""Authoritative deterministic reference ranking for NURSE V2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import (
    CANDIDATE_SCOPE_DECISION_SCHEMA_VERSION,
    FrozenCandidateUniverse,
    require_candidate_scope,
)
from nurse_rcl.v2.family_aggregation import (
    CandidateFamilyAggregation,
    FamilyAggregationResult,
)
from nurse_rcl.v2.history_scoring import (
    CandidateHistoryScore,
    HistoryEvaluationResult,
)
from nurse_rcl.v2.propagation_scoring import (
    CandidatePropagationScore,
    PropagationScoringResult,
)
from nurse_rcl.v2.ranking_schemas import (
    EvidenceTieGroup,
    HistoryMode,
    ScoreDecomposition,
    TopKArtifact,
    _deterministic_score_order,
    build_score_decomposition,
    build_tie_group,
    build_top_k,
)
from nurse_rcl.v2.ranking_schemas import (
    _stable_tie_break_sha256 as _schema_tie_break_sha256,
)
from nurse_rcl.v2.rule_scoring import CandidateRuleScore, RuleScoringResult

REFERENCE_SCORING_CONFIG_SCHEMA_VERSION: Literal["nurse.v2.reference-scoring-config/1"] = (
    "nurse.v2.reference-scoring-config/1"
)
REFERENCE_RANKING_RESULT_SCHEMA_VERSION: Literal["nurse.v2.reference-ranking-result/1"] = (
    "nurse.v2.reference-ranking-result/1"
)
REFERENCE_SCORER_VERSION: Literal["nurse-v2-reference-score/1"] = "nurse-v2-reference-score/1"
REFERENCE_TIE_BREAK_SEED_VERSION: Literal["nurse-v2-reference-ranking-tie-break/1"] = (
    "nurse-v2-reference-ranking-tie-break/1"
)
REFERENCE_TIE_BREAK_SEED_SHA256 = sha256_hex(
    canonical_json_bytes({"tie_break_seed_version": REFERENCE_TIE_BREAK_SEED_VERSION})
)
_MAX_RANKING_CANDIDATES = 4_096


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class RankingInputAuthority:
    """Externally frozen digests that authorize one ranking projection."""

    candidate_freeze_sha256: str
    family_aggregation_result_sha256: str
    propagation_scoring_result_sha256: str
    propagation_authoritative_input_sha256: str
    rule_scoring_result_sha256: str
    rule_authoritative_input_sha256: str
    history_evaluation_result_sha256: str
    history_authoritative_input_sha256: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _require_sha256(
                object.__getattribute__(self, field_name),
                label=field_name.replace("_", " "),
            )


class ReferenceScoringConfig(StrictFrozenModel):
    """Content-addressed fixed policy for the first V2 reference scorer."""

    schema_version: Literal["nurse.v2.reference-scoring-config/1"] = (
        REFERENCE_SCORING_CONFIG_SCHEMA_VERSION
    )
    scorer_version: Literal["nurse-v2-reference-score/1"] = REFERENCE_SCORER_VERSION
    history_mode: HistoryMode = "off"
    lambda_p: float = Field(default=0.25, ge=0.25, le=0.25, allow_inf_nan=False)
    eta_r_when_enabled: float = Field(
        default=0.10,
        ge=0.10,
        le=0.10,
        allow_inf_nan=False,
    )
    eta_h: float = Field(ge=0.0, le=0.10, allow_inf_nan=False)
    numeric_type: Literal["ieee754_binary64"] = "ieee754_binary64"
    quantization: Literal["round_half_even_12"] = "round_half_even_12"
    tie_break_seed_version: Literal["nurse-v2-reference-ranking-tie-break/1"] = (
        REFERENCE_TIE_BREAK_SEED_VERSION
    )
    tie_break_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(default=5, ge=1, le=100)
    is_root_cause_probability: Literal[False] = False
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_declared_policy(self) -> Self:
        expected_eta_h = 0.10 if self.history_mode == "score" else 0.0
        if self.eta_h != expected_eta_h:
            raise ValueError("history weight does not match the declared history mode")
        if self.tie_break_seed_sha256 != REFERENCE_TIE_BREAK_SEED_SHA256:
            raise ValueError("tie-break seed digest differs from the fixed reference seed")
        return self

    def verify(self) -> bool:
        try:
            rebuilt = ReferenceScoringConfig.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = rebuilt.model_dump(mode="json", exclude={"config_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt == self and rebuilt.config_sha256 == sha256_hex(
            canonical_json_bytes(unsigned)
        )


def _reference_config_document(
    *,
    history_mode: HistoryMode,
    eta_h: float,
    top_k: int,
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_SCORING_CONFIG_SCHEMA_VERSION,
        "scorer_version": REFERENCE_SCORER_VERSION,
        "history_mode": history_mode,
        "lambda_p": 0.25,
        "eta_r_when_enabled": 0.10,
        "eta_h": eta_h,
        "numeric_type": "ieee754_binary64",
        "quantization": "round_half_even_12",
        "tie_break_seed_version": REFERENCE_TIE_BREAK_SEED_VERSION,
        "tie_break_seed_sha256": REFERENCE_TIE_BREAK_SEED_SHA256,
        "top_k": top_k,
        "is_root_cause_probability": False,
    }


def build_reference_scoring_config(
    *,
    history_mode: HistoryMode = "off",
    top_k: int = 5,
) -> ReferenceScoringConfig:
    """Build the fixed, secret-free reference-scoring declaration."""

    eta_h = 0.10 if history_mode == "score" else 0.0
    unsigned = _reference_config_document(
        history_mode=history_mode,
        eta_h=eta_h,
        top_k=top_k,
    )
    return ReferenceScoringConfig.model_validate(
        {
            **unsigned,
            "config_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )


def _binary64(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite binary64 value")
    return 0.0 if result == 0.0 else result


def _build_candidate_decomposition(
    *,
    candidate_service_key: str,
    event_evidence_score: float,
    propagation_score: float,
    rule_score: float,
    history_score: float,
    explicit_rules_enabled: bool,
    config: ReferenceScoringConfig,
    event_group_ids: Sequence[str],
    propagation_path_ids: Sequence[str],
    rule_ids: Sequence[str],
    history_neighbor_ids: Sequence[str],
) -> ScoreDecomposition:
    """Build one formula-exact decomposition from already verified channels."""

    if type(config) is not ReferenceScoringConfig or not config.verify():
        raise ValueError("reference scoring configuration failed verification")
    event = _binary64(event_evidence_score, label="event evidence score")
    propagation = _binary64(propagation_score, label="propagation score")
    rule = _binary64(rule_score, label="rule score")
    history = _binary64(history_score, label="history score")
    if not explicit_rules_enabled and rule != 0.0:
        raise ValueError("disabled explicit-rule channel cannot contribute a rule score")
    if config.history_mode != "score" and history != 0.0:
        raise ValueError("non-score history mode cannot contribute a history score")
    eta_r = config.eta_r_when_enabled if explicit_rules_enabled else 0.0
    total = _binary64(
        (1.0 - config.lambda_p) * event
        + config.lambda_p * propagation
        + eta_r * rule
        + config.eta_h * history,
        label="total score",
    )
    return build_score_decomposition(
        candidate_service_key=candidate_service_key,
        event_evidence_score=event,
        propagation_score=propagation,
        rule_score=rule,
        history_score=history,
        lambda_p=config.lambda_p,
        eta_r=eta_r,
        eta_h=config.eta_h,
        total_score=total,
        event_group_ids=event_group_ids,
        propagation_path_ids=propagation_path_ids,
        rule_ids=rule_ids,
        history_neighbor_ids=history_neighbor_ids,
        scoring_configuration_sha256=config.config_sha256,
    )


def _stable_tie_break_sha256(
    *,
    tie_break_seed_sha256: str,
    candidate_service_key: str,
) -> str:
    return _schema_tie_break_sha256(
        tie_break_seed_sha256=tie_break_seed_sha256,
        candidate_service_key=candidate_service_key,
    )


def _ordered_service_keys(
    *,
    scores: Sequence[ScoreDecomposition],
    tie_break_seed_sha256: str,
) -> tuple[str, ...]:
    return _deterministic_score_order(
        scores=scores,
        tie_break_seed_sha256=tie_break_seed_sha256,
    )


def _stable_id(prefix: str, unsigned: object) -> tuple[str, str]:
    digest = sha256_hex(canonical_json_bytes(unsigned))
    return f"{prefix}_{digest[:24]}", digest


def _build_evidence_tie_groups(
    scores: Sequence[ScoreDecomposition],
) -> tuple[EvidenceTieGroup, ...]:
    grouped: dict[tuple[str, str, str], list[ScoreDecomposition]] = {}
    for score in scores:
        key = (
            score.quantized_total_score,
            score.quantized_event_evidence_score,
            score.quantized_propagation_score,
        )
        grouped.setdefault(key, []).append(score)
    return tuple(
        sorted(
            (build_tie_group(scores=members) for members in grouped.values()),
            key=lambda item: item.tie_group_id,
        )
    )


def _authority_document(authority: RankingInputAuthority) -> dict[str, str]:
    return {
        field_name: object.__getattribute__(authority, field_name)
        for field_name in authority.__dataclass_fields__
    }


def _ranking_authoritative_input_sha256(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    config: ReferenceScoringConfig,
    authority: RankingInputAuthority,
) -> str:
    return _ranking_authoritative_input_digest(
        candidate_freeze_sha256=frozen_candidates.freeze_sha256,
        candidate_universe_sha256=frozen_candidates.candidate_universe_sha256,
        candidate_service_keys=frozen_candidates.candidate_service_keys,
        config_sha256=config.config_sha256,
        authority_document=_authority_document(authority),
    )


def _ranking_authoritative_input_digest(
    *,
    candidate_freeze_sha256: str,
    candidate_universe_sha256: str,
    candidate_service_keys: Sequence[str],
    config_sha256: str,
    authority_document: dict[str, str],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "candidate_freeze_sha256": candidate_freeze_sha256,
                "candidate_universe_sha256": candidate_universe_sha256,
                "candidate_service_keys": tuple(candidate_service_keys),
                "config_sha256": config_sha256,
                "authority": authority_document,
            }
        )
    )


def _accepted_scope_decision_sha256(
    *,
    candidate_freeze_sha256: str,
    candidate_service_keys: Sequence[str],
) -> str:
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_SCOPE_DECISION_SCHEMA_VERSION,
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "operation": "ranking",
        "requested_candidate_keys": tuple(sorted(candidate_service_keys)),
        "duplicate_candidate_keys": (),
        "added_candidate_keys": (),
        "missing_candidate_keys": (),
        "accepted": True,
        "reason_code": "candidate_scope_accepted",
    }
    return sha256_hex(canonical_json_bytes(unsigned))


class ReferenceRankingResult(StrictFrozenModel):
    """Complete content-addressed reference rank for one frozen candidate set."""

    schema_version: Literal["nurse.v2.reference-ranking-result/1"] = (
        REFERENCE_RANKING_RESULT_SCHEMA_VERSION
    )
    reference_ranking_result_id: str = Field(pattern=r"^rrank_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1, le=_MAX_RANKING_CANDIDATES)
    candidate_scope_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ReferenceScoringConfig
    explicit_rules_enabled: bool
    effective_eta_r: float = Field(ge=0.0, le=0.10, allow_inf_nan=False)
    effective_eta_h: float = Field(ge=0.0, le=0.10, allow_inf_nan=False)
    family_aggregation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    propagation_scoring_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    propagation_authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_scoring_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_evaluation_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[ScoreDecomposition, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    evidence_tie_groups: tuple[EvidenceTieGroup, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    ordered_service_keys: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_RANKING_CANDIDATES,
    )
    top_k: TopKArtifact
    authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    is_root_cause_probability: Literal[False] = False
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_reference_rank(self) -> Self:
        if self.scores != tuple(sorted(self.scores, key=lambda item: item.candidate_service_key)):
            raise ValueError("reference scores must be canonically ordered")
        candidate_keys = tuple(item.candidate_service_key for item in self.scores)
        if (
            len(candidate_keys) != len(set(candidate_keys))
            or self.candidate_count != len(candidate_keys)
            or len(self.ordered_service_keys) != self.candidate_count
            or set(self.ordered_service_keys) != set(candidate_keys)
        ):
            raise ValueError("reference ranking must exactly cover its candidates")
        if self.evidence_tie_groups != tuple(
            sorted(self.evidence_tie_groups, key=lambda item: item.tie_group_id)
        ):
            raise ValueError("evidence tie groups must be canonically ordered")
        expected_eta_r = self.config.eta_r_when_enabled if self.explicit_rules_enabled else 0.0
        if self.effective_eta_r != expected_eta_r or self.effective_eta_h != self.config.eta_h:
            raise ValueError("effective ranking weights differ from their declared channels")
        return self

    def verify_structure(self) -> bool:
        try:
            if type(self) is not ReferenceRankingResult:
                return False
            rebuilt = ReferenceRankingResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if rebuilt != self or not rebuilt.config.verify():
                return False
            if any(type(item) is not ScoreDecomposition for item in rebuilt.scores):
                return False
            if any(not item.verify() for item in rebuilt.scores):
                return False
            for item in rebuilt.scores:
                expected_total = _binary64(
                    (1.0 - item.lambda_p) * item.event_evidence_score
                    + item.lambda_p * item.propagation_score
                    + item.eta_r * item.rule_score
                    + item.eta_h * item.history_score,
                    label="total score",
                )
                if (
                    item.total_score != expected_total
                    or item.lambda_p != rebuilt.config.lambda_p
                    or item.eta_r != rebuilt.effective_eta_r
                    or item.eta_h != rebuilt.effective_eta_h
                    or item.scoring_configuration_sha256 != rebuilt.config.config_sha256
                ):
                    return False
            expected_groups = _build_evidence_tie_groups(rebuilt.scores)
            if rebuilt.evidence_tie_groups != expected_groups:
                return False
            expected_order = _ordered_service_keys(
                scores=rebuilt.scores,
                tie_break_seed_sha256=rebuilt.config.tie_break_seed_sha256,
            )
            if rebuilt.ordered_service_keys != expected_order:
                return False
            candidate_keys = tuple(item.candidate_service_key for item in rebuilt.scores)
            if rebuilt.candidate_scope_decision_sha256 != (
                _accepted_scope_decision_sha256(
                    candidate_freeze_sha256=rebuilt.candidate_freeze_sha256,
                    candidate_service_keys=candidate_keys,
                )
            ):
                return False
            authority_document = {
                "candidate_freeze_sha256": rebuilt.candidate_freeze_sha256,
                "family_aggregation_result_sha256": (rebuilt.family_aggregation_result_sha256),
                "propagation_scoring_result_sha256": (rebuilt.propagation_scoring_result_sha256),
                "propagation_authoritative_input_sha256": (
                    rebuilt.propagation_authoritative_input_sha256
                ),
                "rule_scoring_result_sha256": rebuilt.rule_scoring_result_sha256,
                "rule_authoritative_input_sha256": (rebuilt.rule_authoritative_input_sha256),
                "history_evaluation_result_sha256": (rebuilt.history_evaluation_result_sha256),
                "history_authoritative_input_sha256": (rebuilt.history_authoritative_input_sha256),
            }
            if rebuilt.authoritative_input_sha256 != (
                _ranking_authoritative_input_digest(
                    candidate_freeze_sha256=rebuilt.candidate_freeze_sha256,
                    candidate_universe_sha256=rebuilt.candidate_universe_sha256,
                    candidate_service_keys=candidate_keys,
                    config_sha256=rebuilt.config.config_sha256,
                    authority_document=authority_document,
                )
            ):
                return False
            expected_top_k = build_top_k(
                opaque_case_id=rebuilt.opaque_case_id,
                candidate_universe_sha256=rebuilt.candidate_universe_sha256,
                scores=rebuilt.scores,
                tie_groups=rebuilt.evidence_tie_groups,
                ordered_service_keys=rebuilt.ordered_service_keys,
                k=rebuilt.config.top_k,
                tie_break_seed_sha256=rebuilt.config.tie_break_seed_sha256,
            )
            if rebuilt.top_k != expected_top_k:
                return False
            unsigned = rebuilt.model_dump(
                mode="json",
                exclude={"reference_ranking_result_id", "result_sha256"},
            )
            identifier, digest = _stable_id("rrank", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return rebuilt.reference_ranking_result_id == identifier and rebuilt.result_sha256 == digest

    def verify(self) -> bool:
        return self.verify_structure()


def _validate_authority(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    family_aggregation: FamilyAggregationResult,
    propagation_scoring: PropagationScoringResult,
    rule_scoring: RuleScoringResult,
    history_evaluation: HistoryEvaluationResult,
    authority: RankingInputAuthority,
) -> None:
    if type(authority) is not RankingInputAuthority:
        raise ValueError("ranking input authority must use the exact frozen type")
    expected = {
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "family_aggregation_result_sha256": family_aggregation.result_sha256,
        "propagation_scoring_result_sha256": propagation_scoring.result_sha256,
        "propagation_authoritative_input_sha256": (propagation_scoring.authoritative_input_sha256),
        "rule_scoring_result_sha256": rule_scoring.result_sha256,
        "rule_authoritative_input_sha256": rule_scoring.authoritative_input_sha256,
        "history_evaluation_result_sha256": history_evaluation.result_sha256,
        "history_authoritative_input_sha256": (history_evaluation.authoritative_input_sha256),
    }
    if _authority_document(authority) != expected:
        raise ValueError("ranking input authority does not match the frozen channels")


def _validate_channel_scope(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    family_aggregation: FamilyAggregationResult,
    propagation_scoring: PropagationScoringResult,
    rule_scoring: RuleScoringResult,
    history_evaluation: HistoryEvaluationResult,
    config: ReferenceScoringConfig,
) -> str:
    frozen_values = _preflight_candidate_collection(
        frozen_candidates,
        expected_type=FrozenCandidateUniverse,
        field_name="candidate_service_keys",
        item_type=str,
        label="frozen candidate universe",
    )
    family_values = _preflight_candidate_collection(
        family_aggregation,
        expected_type=FamilyAggregationResult,
        field_name="candidates",
        item_type=CandidateFamilyAggregation,
        label="family aggregation",
    )
    propagation_values = _preflight_candidate_collection(
        propagation_scoring,
        expected_type=PropagationScoringResult,
        field_name="candidate_scores",
        item_type=CandidatePropagationScore,
        label="propagation scoring",
    )
    rule_values = _preflight_candidate_collection(
        rule_scoring,
        expected_type=RuleScoringResult,
        field_name="candidates",
        item_type=CandidateRuleScore,
        label="rule scoring",
    )
    history_values = _preflight_candidate_collection(
        history_evaluation,
        expected_type=HistoryEvaluationResult,
        field_name="candidate_scores",
        item_type=CandidateHistoryScore,
        label="history evaluation",
    )
    if any(
        len(values) > _MAX_RANKING_CANDIDATES
        for values in (
            frozen_values,
            family_values,
            propagation_values,
            rule_values,
            history_values,
        )
    ):
        raise ValueError("candidate count exceeds the reference ranking limit")
    if not frozen_candidates.verify():
        raise ValueError("frozen candidate universe failed verification")
    if not family_aggregation.verify():
        raise ValueError("family aggregation failed verification")
    if not propagation_scoring.verify_structure():
        raise ValueError("propagation scoring failed verification")
    if not rule_scoring.verify_structure():
        raise ValueError("rule scoring failed verification")
    if not history_evaluation.verify_structure():
        raise ValueError("history evaluation failed verification")
    if type(config) is not ReferenceScoringConfig or not config.verify():
        raise ValueError("reference scoring configuration failed verification")
    expected_keys = frozen_candidates.candidate_service_keys
    channel_keys = (
        tuple(item.candidate_service_key for item in family_aggregation.candidates),
        propagation_scoring.candidate_service_keys,
        tuple(item.candidate_service_key for item in rule_scoring.candidates),
        history_evaluation.candidate_service_keys,
    )
    if any(keys != expected_keys for keys in channel_keys):
        raise ValueError("ranking channels do not exactly cover frozen candidates")
    if any(
        opaque_case_id != frozen_candidates.opaque_case_id
        for opaque_case_id in (
            family_aggregation.opaque_case_id,
            propagation_scoring.opaque_case_id,
            rule_scoring.opaque_case_id,
        )
    ):
        raise ValueError("ranking channels belong to another opaque case")
    if any(
        freeze_sha256 != frozen_candidates.freeze_sha256
        for freeze_sha256 in (
            propagation_scoring.candidate_freeze_sha256,
            rule_scoring.candidate_freeze_sha256,
            history_evaluation.candidate_freeze_sha256,
        )
    ):
        raise ValueError("ranking channel candidate freeze differs")
    if history_evaluation.candidate_universe_sha256 != (
        frozen_candidates.candidate_universe_sha256
    ):
        raise ValueError("history ranking channel candidate universe differs")
    if history_evaluation.mode != config.history_mode:
        raise ValueError("history result mode differs from the ranking configuration")
    return require_candidate_scope(
        frozen=frozen_candidates,
        candidate_service_keys=expected_keys,
        operation="ranking",
    ).decision_sha256


def _preflight_candidate_collection(
    value: object,
    *,
    expected_type: type[object],
    field_name: str,
    item_type: type[object],
    label: str,
) -> tuple[object, ...]:
    if type(value) is not expected_type:
        raise ValueError(f"{label} must use the exact model type")
    try:
        state = object.__getattribute__(value, "__dict__")
    except AttributeError as error:
        raise ValueError(f"{label} lacks canonical model state") from error
    if type(state) is not dict or field_name not in state:
        raise ValueError(f"{label} lacks its candidate collection")
    values = dict.__getitem__(state, field_name)
    if type(values) is not tuple:
        raise ValueError(f"{label} candidate collection must use an exact tuple")
    if any(type(item) is not item_type for item in values):
        raise ValueError(f"{label} candidate collection uses a noncanonical item type")
    return values


def rank_reference_candidates(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    family_aggregation: FamilyAggregationResult,
    propagation_scoring: PropagationScoringResult,
    rule_scoring: RuleScoringResult,
    history_evaluation: HistoryEvaluationResult,
    authority: RankingInputAuthority,
    config: ReferenceScoringConfig | None = None,
) -> ReferenceRankingResult:
    """Rank every legal member of frozen C from verified E/P/R/H channels."""

    effective_config = config or build_reference_scoring_config(
        history_mode=history_evaluation.mode
    )
    scope_decision_sha256 = _validate_channel_scope(
        frozen_candidates=frozen_candidates,
        family_aggregation=family_aggregation,
        propagation_scoring=propagation_scoring,
        rule_scoring=rule_scoring,
        history_evaluation=history_evaluation,
        config=effective_config,
    )
    _validate_authority(
        frozen_candidates=frozen_candidates,
        family_aggregation=family_aggregation,
        propagation_scoring=propagation_scoring,
        rule_scoring=rule_scoring,
        history_evaluation=history_evaluation,
        authority=authority,
    )
    family_by_key = {item.candidate_service_key: item for item in family_aggregation.candidates}
    propagation_by_key = {
        item.candidate_service_key: item for item in propagation_scoring.candidate_scores
    }
    rule_by_key = {item.candidate_service_key: item for item in rule_scoring.candidates}
    history_by_key = {
        item.candidate_service_key: item for item in history_evaluation.candidate_scores
    }
    explicit_rules_enabled = any(
        contribution.activation_status != "disabled"
        for candidate in rule_scoring.candidates
        for contribution in candidate.contributions
    )
    scores: list[ScoreDecomposition] = []
    for candidate_service_key in frozen_candidates.candidate_service_keys:
        family = family_by_key[candidate_service_key]
        propagation = propagation_by_key[candidate_service_key]
        rule = rule_by_key[candidate_service_key]
        history = history_by_key[candidate_service_key]
        event_group_ids = tuple(
            event_group_id
            for maximum in family.family_maxima
            for event_group_id in maximum.retained_event_group_ids
        )
        propagation_path_ids = tuple(
            item.path_audit_id
            for item in propagation_scoring.path_audits
            if item.candidate_service_key == candidate_service_key
        )
        scores.append(
            _build_candidate_decomposition(
                candidate_service_key=candidate_service_key,
                event_evidence_score=family.evidence_strength,
                propagation_score=propagation.propagation_score,
                rule_score=rule.rule_score,
                history_score=history.history_score,
                explicit_rules_enabled=explicit_rules_enabled,
                config=effective_config,
                event_group_ids=event_group_ids,
                propagation_path_ids=propagation_path_ids,
                rule_ids=tuple(item.contribution_id for item in rule.contributions),
                history_neighbor_ids=tuple(
                    item.selected_neighbor_id for item in history.contributions
                ),
            )
        )
    ordered_scores = tuple(sorted(scores, key=lambda item: item.candidate_service_key))
    tie_groups = _build_evidence_tie_groups(ordered_scores)
    ordered_service_keys = _ordered_service_keys(
        scores=ordered_scores,
        tie_break_seed_sha256=effective_config.tie_break_seed_sha256,
    )
    top_k = build_top_k(
        opaque_case_id=frozen_candidates.opaque_case_id,
        candidate_universe_sha256=frozen_candidates.candidate_universe_sha256,
        scores=ordered_scores,
        tie_groups=tie_groups,
        ordered_service_keys=ordered_service_keys,
        k=effective_config.top_k,
        tie_break_seed_sha256=effective_config.tie_break_seed_sha256,
    )
    authoritative_input_sha256 = _ranking_authoritative_input_sha256(
        frozen_candidates=frozen_candidates,
        config=effective_config,
        authority=authority,
    )
    unsigned: dict[str, object] = {
        "schema_version": REFERENCE_RANKING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": frozen_candidates.opaque_case_id,
        "candidate_freeze_sha256": frozen_candidates.freeze_sha256,
        "candidate_universe_sha256": frozen_candidates.candidate_universe_sha256,
        "candidate_count": frozen_candidates.candidate_count,
        "candidate_scope_decision_sha256": scope_decision_sha256,
        "config": effective_config.model_dump(mode="json"),
        "explicit_rules_enabled": explicit_rules_enabled,
        "effective_eta_r": (effective_config.eta_r_when_enabled if explicit_rules_enabled else 0.0),
        "effective_eta_h": effective_config.eta_h,
        "family_aggregation_result_sha256": family_aggregation.result_sha256,
        "propagation_scoring_result_sha256": propagation_scoring.result_sha256,
        "propagation_authoritative_input_sha256": (propagation_scoring.authoritative_input_sha256),
        "rule_scoring_result_sha256": rule_scoring.result_sha256,
        "rule_authoritative_input_sha256": rule_scoring.authoritative_input_sha256,
        "history_evaluation_result_sha256": history_evaluation.result_sha256,
        "history_authoritative_input_sha256": (history_evaluation.authoritative_input_sha256),
        "scores": tuple(item.model_dump(mode="json") for item in ordered_scores),
        "evidence_tie_groups": tuple(item.model_dump(mode="json") for item in tie_groups),
        "ordered_service_keys": ordered_service_keys,
        "top_k": top_k.model_dump(mode="json"),
        "authoritative_input_sha256": authoritative_input_sha256,
        "is_root_cause_probability": False,
    }
    identifier, digest = _stable_id("rrank", unsigned)
    result = ReferenceRankingResult.model_validate(
        {
            **unsigned,
            "reference_ranking_result_id": identifier,
            "result_sha256": digest,
        }
    )
    if not result.verify_structure():  # pragma: no cover - construction invariant
        raise AssertionError("reference ranking result verification failed")
    return result


__all__ = [
    "REFERENCE_RANKING_RESULT_SCHEMA_VERSION",
    "REFERENCE_SCORER_VERSION",
    "REFERENCE_SCORING_CONFIG_SCHEMA_VERSION",
    "REFERENCE_TIE_BREAK_SEED_SHA256",
    "REFERENCE_TIE_BREAK_SEED_VERSION",
    "RankingInputAuthority",
    "ReferenceRankingResult",
    "ReferenceScoringConfig",
    "build_reference_scoring_config",
    "rank_reference_candidates",
]
