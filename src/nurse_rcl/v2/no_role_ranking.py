"""Independent NoRole reference ranking; structural receipts confer no authority."""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar, cast

from pydantic import ConfigDict, Field, ValidationInfo, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_scoring_selection as _selection
from nurse_rcl.v2 import history_scoring as _history
from nurse_rcl.v2 import no_role_compilation as _roles
from nurse_rcl.v2 import no_role_evidence as _evidence
from nurse_rcl.v2 import no_role_propagation as _propagation
from nurse_rcl.v2 import reference_ranking as _ranking
from nurse_rcl.v2 import rule_scoring as _rules
from nurse_rcl.v2.case_scoring_selection import CaseGuardReplayInputs, CaseScoringSelection
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.no_role_compilation import (
    NO_ROLE_POLICY_SHA256,
    NO_ROLE_POLICY_VERSION,
    NoRoleCompilationResult,
)
from nurse_rcl.v2.no_role_evidence import NoRoleEventStrengthResult, NoRoleFamilyAggregationResult
from nurse_rcl.v2.no_role_propagation import NoRolePropagationScoringResult
from nurse_rcl.v2.propagation_scoring import PropagationScoringConfig
from nurse_rcl.v2.ranking_schemas import (
    EvidenceTieGroup,
    RuleSet,
    ScoreDecomposition,
    TopKArtifact,
    build_top_k,
)
from nurse_rcl.v2.reference_ranking import RankingInputAuthority, ReferenceScoringConfig
from nurse_rcl.v2.rule_scoring import RuleCandidateAssignment, RuleScoringResult
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.temporal_condensation import (
    OnsetInterval,
    RelationSCCCondensation,
    TemporalPrecedence,
)

NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.no-role-reference-ranking-result/1"
] = "nurse.v2.no-role-reference-ranking-result/1"
NO_ROLE_RANKING_INPUT_SCHEMA_VERSION = "nurse.v2.no-role-reference-ranking-input/1"
_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)
_AUTHORITY_FIELDS = (
    "candidate_freeze_sha256",
    "family_aggregation_result_sha256",
    "propagation_scoring_result_sha256",
    "propagation_authoritative_input_sha256",
    "rule_scoring_result_sha256",
    "rule_authoritative_input_sha256",
    "history_evaluation_result_sha256",
    "history_authoritative_input_sha256",
)


def _preflight(value: object, *, narrow: bool = False) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(
            NoRoleReferenceRankingResult,
            ReferenceScoringConfig,
            RuleSet,
            RuleCandidateAssignment,
            PredicateResult,
            RuleScoringResult,
            HistoryEvaluationResult,
        ),
        label="NoRole reference ranking",
        max_string_length=8192 if narrow else None,
        provenance_id_limit=131072 if narrow else None,
    )
    plain = _selection._plain(value)
    _roles._model_capacities(value)
    if (
        type(plain) is dict
        and cast(dict[str, object], plain).get("schema_version")
        == NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION
    ):
        _result_limits(cast(dict[str, Any], plain))


def _bounded_count(value: object, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("NoRole ranking integer exceeds its declared domain")


def _unique(values: list[Any]) -> None:
    if len(values) != len(set(values)):
        raise ValueError("NoRole ranking candidate or artifact IDs must be unique")


def _result_limits(plain: dict[str, Any]) -> None:
    try:
        _bounded_count(plain["candidate_count"], 4096)
        _bounded_count(plain["config"]["top_k"], 100)
        _bounded_count(plain["top_k"]["k"], 100)
        _bounded_count(plain["top_k"]["total_candidate_count"], 4096)
        if len(plain["top_k"]["entries"]) > 100:
            raise ValueError("NoRole ranking TopK exceeds its entry capacity")
        for entry in plain["top_k"]["entries"]:
            _bounded_count(entry["rank"], 4096)
        _unique(plain["ordered_service_keys"])
        _unique([row["candidate_service_key"] for row in plain["scores"]])
        _unique([row["rank"] for row in plain["top_k"]["entries"]])
        _unique([row["service_entity_key"] for row in plain["top_k"]["entries"]])
        for groups in (plain["evidence_tie_groups"], plain["top_k"]["evidence_tie_groups"]):
            _unique([row["tie_group_id"] for row in groups])
            _unique([key for row in groups for key in row["member_service_keys"]])
            _unique([key for row in groups for key in row["member_score_sha256"]])
    except (KeyError, TypeError) as error:
        raise ValueError("NoRole ranking result lacks required bounded fields") from error


def _prepare(value: object) -> tuple[dict[str, object], bytes]:
    _preflight(value)
    plain = cast(dict[str, object], _selection._plain(value))
    _selection._measure_json(plain)
    return plain, _roles._representation(value)


def _bounded_json_int(token: str) -> int:
    # Only result counts/ranks <=4096 and requested k<=100 use this reader.
    # Source timestamp integers retain their original independent domains.
    if len(token) > 4 or token.startswith("-"):
        raise ValueError("NoRole ranking JSON integer exceeds its result domain")
    return int(token)


class NoRoleReferenceRankingResult(StrictFrozenModel):
    """A complete independent ranking, distinct from source authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.no-role-reference-ranking-result/1"] = (
        NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION
    )
    reference_ranking_result_id: str = Field(pattern=r"^nrrank_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    candidate_freeze_sha256: _SHA
    candidate_universe_sha256: _SHA
    candidate_count: int = Field(ge=1, le=4096)
    candidate_scope_decision_sha256: _SHA
    config: ReferenceScoringConfig
    explicit_rules_enabled: bool
    effective_eta_r: float = Field(ge=0.0, le=0.10, allow_inf_nan=False)
    effective_eta_h: float = Field(ge=0.0, le=0.10, allow_inf_nan=False)
    family_aggregation_result_sha256: _SHA
    propagation_scoring_result_sha256: _SHA
    propagation_authoritative_input_sha256: _SHA
    rule_scoring_result_sha256: _SHA
    rule_authoritative_input_sha256: _SHA
    history_evaluation_result_sha256: _SHA
    history_authoritative_input_sha256: _SHA
    scores: tuple[ScoreDecomposition, ...] = Field(min_length=1, max_length=4096)
    evidence_tie_groups: tuple[EvidenceTieGroup, ...] = Field(min_length=1, max_length=4096)
    ordered_service_keys: tuple[str, ...] = Field(min_length=1, max_length=4096)
    top_k: TopKArtifact
    authoritative_input_sha256: _SHA
    is_root_cause_probability: Literal[False] = False
    result_sha256: _SHA
    source_records_sha256: _SHA
    scoring_selection_sha256: _SHA
    no_role_policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    no_role_policy_sha256: _SHA
    source_family_schema_version: Literal["nurse.v2.no-role-family-aggregation-result/1"] = (
        _evidence.NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION
    )
    source_propagation_schema_version: Literal["nurse.v2.no-role-propagation-scoring-result/1"] = (
        _propagation.NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
    )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if cls is not NoRoleReferenceRankingResult:
            raise ValueError("NoRole ranking JSON requires the exact result class")
        raw = _roles._json_bytes(json_data)
        try:
            text = raw.decode("utf-8", errors="strict")
            _roles._scan_json(raw)
            document = json.loads(
                text,
                object_pairs_hook=_roles._unique_object,
                parse_int=_bounded_json_int,
                parse_float=_evidence._finite_json_float,
                parse_constant=_roles._reject_json_number,
            )
            _preflight(document)
            _selection._measure_json(_selection._plain(document))
            if not _evidence._json_shape(document, cls):
                raise ValueError("NoRole ranking JSON field representation differs")
            return super().model_validate_json(
                raw, strict=True, context=context, by_alias=by_alias, by_name=by_name
            )
        finally:
            if not hmac.compare_digest(raw, _roles._json_bytes(json_data)):
                raise ValueError("NoRole ranking JSON changed during validation")

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object, info: ValidationInfo) -> object:
        if cls is not NoRoleReferenceRankingResult or (
            type(value) is not dict and type(value) is not NoRoleReferenceRankingResult
        ):
            raise ValueError("NoRole ranking requires an exact model or dictionary")
        _preflight(value)
        _selection._measure_json(_selection._plain(value))
        if type(value) is dict:
            if info.mode == "json":
                if not _evidence._json_shape(value, cls):
                    raise ValueError("NoRole ranking JSON representation differs")
                return _evidence._json_containers(value, cls)
            if not all(
                name in cls.model_fields
                and _selection._exact(child, cls.model_fields[name].annotation)
                for name, child in cast(dict[str, object], value).items()
            ):
                raise ValueError("NoRole ranking fields require exact declared types")
        return value

    @model_validator(mode="after")
    def closed_rank(self) -> Self:
        if self.no_role_policy_sha256 != NO_ROLE_POLICY_SHA256:
            raise ValueError("NoRole ranking activation policy differs")
        if self.config.history_mode != "off":
            raise ValueError("NoRole reference ranking requires history off")
        keys = tuple(row.candidate_service_key for row in self.scores)
        if keys != tuple(sorted(set(keys))) or self.candidate_count != len(keys):
            raise ValueError("NoRole ranking scores must canonically cover all candidates")
        if len(self.ordered_service_keys) != len(keys) or set(self.ordered_service_keys) != set(
            keys
        ):
            raise ValueError("NoRole ranking order must exactly cover all candidates")
        if self.evidence_tie_groups != tuple(
            sorted(self.evidence_tie_groups, key=lambda row: row.tie_group_id)
        ):
            raise ValueError("NoRole ranking ties must be canonical")
        expected = self.config.eta_r_when_enabled if self.explicit_rules_enabled else 0.0
        if self.effective_eta_r != expected or self.effective_eta_h != self.config.eta_h:
            raise ValueError("NoRole ranking effective weights differ")
        return self

    def verify_structure(self) -> bool:
        try:
            if type(self) is not NoRoleReferenceRankingResult:
                return False
            original, representation = _prepare(self)
            try:
                return _valid_result(original)
            finally:
                current, current_representation = _prepare(self)
                if original != current or representation != current_representation:
                    raise ValueError("NoRole ranking result changed during verification")
        except _ERRORS:
            return False

    def verify(self) -> bool:
        return NoRoleReferenceRankingResult.verify_structure(self)

    def verify_inputs(
        self,
        *,
        inputs: CaseGuardReplayInputs,
        selection: CaseScoringSelection,
        expected_source_records_sha256: str,
        expected_selection_sha256: str,
        role_compilation: NoRoleCompilationResult,
        expected_role_compilation_sha256: str,
        event_strength: NoRoleEventStrengthResult,
        expected_event_strength_sha256: str,
        family_aggregation: NoRoleFamilyAggregationResult,
        expected_family_aggregation_sha256: str,
        aggregator_version: str,
        graph_snapshots: tuple[GraphSnapshot, ...],
        condensations: tuple[RelationSCCCondensation, ...],
        onsets: tuple[OnsetInterval, ...],
        precedences: tuple[TemporalPrecedence, ...],
        propagation_config: PropagationScoringConfig,
        propagation_scoring: NoRolePropagationScoringResult,
        expected_propagation_sha256: str,
        rule_set: RuleSet,
        rule_assignments: tuple[RuleCandidateAssignment, ...],
        rule_predicates: tuple[PredicateResult, ...],
        rule_scoring: RuleScoringResult,
        history_evaluation: HistoryEvaluationResult,
        history_mode: Literal["off"] = "off",
        authority: RankingInputAuthority,
        config: ReferenceScoringConfig,
        expected_authoritative_input_sha256: str,
        expected_result_sha256: str,
    ) -> bool:
        try:
            if type(self) is not NoRoleReferenceRankingResult:
                return False
            values = dict(locals())
            values.pop("self")
            original = _seal(values, self)
            try:
                if (
                    original.result is None
                    or original.result["result_sha256"] != expected_result_sha256
                ):
                    return False
                rebuilt = _build(original, expected_authoritative_input_sha256)
                return hmac.compare_digest(
                    _roles._encoded(original.result), _roles._encoded(_prepare(rebuilt)[0])
                )
            finally:
                _unchanged(original, _seal(values, self))
        except _ERRORS:
            return False


def _digest(plain: dict[str, object], excluded: tuple[str, ...] = ()) -> str:
    return sha256_hex(
        _roles._encoded({key: value for key, value in plain.items() if key not in excluded})
    )


def _authority_document(
    frozen: dict[str, Any],
    config_sha: str,
    authority: dict[str, str],
    source_sha: str,
    selection_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": NO_ROLE_RANKING_INPUT_SCHEMA_VERSION,
        "candidate_freeze_sha256": frozen["freeze_sha256"],
        "candidate_universe_sha256": frozen["candidate_universe_sha256"],
        "candidate_service_keys": frozen["candidate_service_keys"],
        "config_sha256": config_sha,
        "authority": authority,
        "source_records_sha256": source_sha,
        "scoring_selection_sha256": selection_sha,
        "source_family_schema_version": _evidence.NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
        "source_propagation_schema_version": (
            _propagation.NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
        ),
    }


def _valid_result(plain: dict[str, object]) -> bool:
    encoded = _roles._encoded(plain)
    rebuilt = NoRoleReferenceRankingResult.model_validate_json(encoded)
    if not ReferenceScoringConfig.verify(rebuilt.config):
        return False
    for row in rebuilt.scores:
        expected = _ranking._build_candidate_decomposition(
            candidate_service_key=row.candidate_service_key,
            event_evidence_score=row.event_evidence_score,
            propagation_score=row.propagation_score,
            rule_score=row.rule_score,
            history_score=row.history_score,
            explicit_rules_enabled=rebuilt.explicit_rules_enabled,
            config=rebuilt.config,
            event_group_ids=row.event_group_ids,
            propagation_path_ids=row.propagation_path_ids,
            rule_ids=row.rule_ids,
            history_neighbor_ids=row.history_neighbor_ids,
        )
        if row != expected or row.propagation_score != 0.0 or row.history_score != 0.0:
            return False
    groups = _ranking._build_evidence_tie_groups(rebuilt.scores)
    order = _ranking._ordered_service_keys(
        scores=rebuilt.scores, tie_break_seed_sha256=rebuilt.config.tie_break_seed_sha256
    )
    keys = tuple(row.candidate_service_key for row in rebuilt.scores)
    scope = _ranking._accepted_scope_decision_sha256(
        candidate_freeze_sha256=rebuilt.candidate_freeze_sha256, candidate_service_keys=keys
    )
    authority = {key: cast(str, plain[key]) for key in _AUTHORITY_FIELDS}
    document = _authority_document(
        {
            "freeze_sha256": rebuilt.candidate_freeze_sha256,
            "candidate_universe_sha256": rebuilt.candidate_universe_sha256,
            "candidate_service_keys": keys,
        },
        rebuilt.config.config_sha256,
        authority,
        rebuilt.source_records_sha256,
        rebuilt.scoring_selection_sha256,
    )
    top = build_top_k(
        opaque_case_id=rebuilt.opaque_case_id,
        candidate_universe_sha256=rebuilt.candidate_universe_sha256,
        scores=rebuilt.scores,
        tie_groups=groups,
        ordered_service_keys=order,
        k=rebuilt.config.top_k,
        tie_break_seed_sha256=rebuilt.config.tie_break_seed_sha256,
    )
    digest = _digest(plain, ("reference_ranking_result_id", "result_sha256"))
    return (
        groups == rebuilt.evidence_tie_groups
        and order == rebuilt.ordered_service_keys
        and top == rebuilt.top_k
        and scope == rebuilt.candidate_scope_decision_sha256
        and _digest(document) == rebuilt.authoritative_input_sha256
        and rebuilt.result_sha256 == digest
        and rebuilt.reference_ranking_result_id == "nrrank_" + digest[:24]
        and hmac.compare_digest(encoded, _roles._encoded(_prepare(rebuilt)[0]))
    )


@dataclass(frozen=True, slots=True)
class _Seal:
    propagation: _propagation._Seal
    evidence: _evidence._Seal
    values: dict[str, Any]
    result: dict[str, object] | None
    representations: tuple[bytes, ...]


def _seal(values: dict[str, Any], result: NoRoleReferenceRankingResult | None = None) -> _Seal:
    for name, cls in (
        ("inputs", CaseGuardReplayInputs),
        ("selection", CaseScoringSelection),
        ("role_compilation", NoRoleCompilationResult),
        ("event_strength", NoRoleEventStrengthResult),
        ("family_aggregation", NoRoleFamilyAggregationResult),
        ("propagation_scoring", NoRolePropagationScoringResult),
        ("propagation_config", PropagationScoringConfig),
        ("rule_set", RuleSet),
        ("rule_scoring", RuleScoringResult),
        ("history_evaluation", HistoryEvaluationResult),
        ("config", ReferenceScoringConfig),
        ("authority", RankingInputAuthority),
    ):
        if type(values[name]) is not cls:
            raise ValueError("NoRole ranking requires exact upstream roots")
    if result is not None and type(result) is not NoRoleReferenceRankingResult:
        raise ValueError("NoRole ranking requires the exact result type")
    for name, value in values.items():
        if name.startswith("expected_"):
            _ranking._require_sha256(value, label=name)
    authority = {
        name: _ranking._require_sha256(
            object.__getattribute__(values["authority"], name), label=name
        )
        for name in _AUTHORITY_FIELDS
    }
    if type(values["history_mode"]) is not str or values["history_mode"] != "off":
        raise ValueError("NoRole ranking requires history_mode off")
    _evidence._aggregator_key(values["aggregator_version"])
    for name, item_cls in (
        ("rule_assignments", RuleCandidateAssignment),
        ("rule_predicates", PredicateResult),
        ("graph_snapshots", GraphSnapshot),
        ("condensations", RelationSCCCondensation),
        ("onsets", OnsetInterval),
        ("precedences", TemporalPrecedence),
    ):
        sequence = values[name]
        if type(sequence) is not tuple or len(sequence) > 32768:
            raise ValueError("NoRole ranking requires bounded exact input tuples")
        if any(type(row) is not item_cls for row in sequence):
            raise ValueError("NoRole ranking requires exact input artifact types")
    # Seals only preflight and capture plain values; they invoke no encoder or
    # validator. Each independent artifact retains its separate 64 MiB budget.
    prop = _propagation._seal(
        inputs=values["inputs"],
        selection=values["selection"],
        role_compilation=values["role_compilation"],
        graph_snapshots=values["graph_snapshots"],
        condensations=values["condensations"],
        onsets=values["onsets"],
        precedences=values["precedences"],
        config=values["propagation_config"],
        result=values["propagation_scoring"],
    )
    ev = _evidence._seal(
        values["inputs"],
        values["selection"],
        values["role_compilation"],
        values["event_strength"],
        values["family_aggregation"],
    )
    extras = (
        "rule_set",
        "rule_assignments",
        "rule_predicates",
        "rule_scoring",
        "history_evaluation",
        "config",
    )
    plain: dict[str, Any] = {}
    for name in extras:
        value = values[name]
        _preflight(value, narrow=name != "config")
        if name == "rule_predicates":
            # Exact, acyclic trees were checked above. Count the root and every
            # occurrence separately, with an independent budget for each tree.
            for predicate in value:
                pending = [predicate]
                records = 1
                while pending:
                    state = object.__getattribute__(pending.pop(), "__dict__")
                    operands = state["operand_states"]
                    records += len(operands)
                    if records > 4096:
                        raise ValueError("NoRole rule predicate tree exceeds its record capacity")
                    pending.extend(operands)
        if type(value) is tuple:
            for row in value:
                _selection._measure_json(_selection._plain(row))
        else:
            _selection._measure_json(_selection._plain(value))
        plain[name] = _selection._plain(value)
    if len(plain["rule_set"]["rules"]) > 32768:
        raise ValueError("NoRole ranking rule inputs exceed 32768")
    for sequence, fields in (
        (plain["rule_set"]["rules"], ("rule_key", "rule_sha256")),
        (plain["rule_assignments"], ("assignment_id", "assignment_sha256")),
        (plain["rule_predicates"], ("predicate_result_id", "predicate_sha256")),
    ):
        for field in fields:
            ids = [row[field] for row in sequence]
            if len(ids) != len(set(ids)):
                raise ValueError("NoRole ranking rule artifact IDs must be unique")
    binding_keys = [
        (row["candidate_service_key"], row["rule_key"]) for row in plain["rule_assignments"]
    ]
    if len(binding_keys) != len(set(binding_keys)):
        raise ValueError("NoRole ranking duplicate rule assignment")
    frozen = cast(dict[str, Any], prop.roles.source["frozen_candidates"])
    assert ev.result is not None and prop.result is not None
    _bounded_count(frozen["candidate_count"], 4096)
    _bounded_count(
        cast(dict[str, Any], prop.roles.source["candidate_universe"])["candidate_count"], 4096
    )
    _bounded_count(plain["history_evaluation"]["candidate_count"], 4096)
    collections = (
        frozen["candidate_service_keys"],
        cast(list[Any], ev.result["candidates"]),
        cast(list[Any], prop.result["candidate_scores"]),
        plain["rule_scoring"]["candidates"],
        plain["history_evaluation"]["candidate_scores"],
    )
    if any(not 1 <= len(rows) <= 4096 for rows in collections):
        raise ValueError("NoRole reference ranking candidate capacity is 1..4096")
    keys = frozen["candidate_service_keys"]
    _unique(keys)
    if keys != sorted(keys):
        raise ValueError("NoRole ranking frozen candidates must be canonical")
    for rows in collections[1:]:
        channel_keys = [row["candidate_service_key"] for row in rows]
        _unique(channel_keys)
        if channel_keys != keys:
            raise ValueError("NoRole ranking channels must exactly cover frozen C")
    if plain["config"]["history_mode"] != "off" or plain["history_evaluation"]["mode"] != "off":
        raise ValueError("NoRole ranking only accepts history off")
    if type(plain["config"]["top_k"]) is not int or not 1 <= plain["config"]["top_k"] <= 100:
        raise ValueError("NoRole reference top-k capacity is 1..100")
    result_plain, result_repr = (
        (None, _roles._representation(None)) if result is None else _prepare(result)
    )
    plain["authority"] = authority
    plain.update({name: value for name, value in values.items() if name.startswith("expected_")})
    plain["aggregator_version"] = values["aggregator_version"]
    plain["history_mode"] = values["history_mode"]
    return _Seal(
        prop,
        ev,
        plain,
        result_plain,
        (
            *(_roles._representation(values[name]) for name in extras),
            _roles._representation(authority),
            result_repr,
        ),
    )


def _unchanged(original: _Seal, current: _Seal) -> None:
    _propagation._unchanged(original.propagation, current.propagation)
    _evidence._unchanged(original.evidence, current.evidence)
    if (
        original.values != current.values
        or original.result != current.result
        or original.representations != current.representations
    ):
        raise ValueError("NoRole ranking original content or exact representation changed")


def _authority(original: _Seal) -> str:
    values, prop, ev = original.values, original.propagation, original.evidence
    source_pin, selection_pin, role_pin = (
        values["expected_source_records_sha256"],
        values["expected_selection_sha256"],
        values["expected_role_compilation_sha256"],
    )
    prop_input = _propagation._authority(prop, source_pin, selection_pin, role_pin)
    if prop.roles.result is None or prop.roles.result["result_sha256"] != role_pin:
        raise ValueError("NoRole ranking role result differs from its independent pin")
    if prop.roles.selection["selection_sha256"] != selection_pin:
        raise ValueError("NoRole ranking selection differs from its independent pin")
    assert ev.event is not None and ev.result is not None and prop.result is not None
    for document, excluded, pin in (
        (
            ev.event,
            ("event_strength_result_id", "result_sha256"),
            values["expected_event_strength_sha256"],
        ),
        (
            ev.result,
            ("family_aggregation_result_id", "result_sha256"),
            values["expected_family_aggregation_sha256"],
        ),
        (
            prop.result,
            ("scoring_result_id", "result_sha256"),
            values["expected_propagation_sha256"],
        ),
    ):
        if _digest(document, excluded) != pin or document["result_sha256"] != pin:
            raise ValueError("NoRole ranking independent channel pin differs")
    frozen = cast(dict[str, Any], prop.roles.source["frozen_candidates"])
    freeze_sha = _digest(frozen, ("freeze_sha256",))
    rule_input = _rules._authoritative_input_digest(
        candidate_freeze_sha256=freeze_sha,
        rule_set_sha256=_digest(values["rule_set"], ("rule_set_sha256",)),
        predicate_sha256s=tuple(
            sorted(
                _digest(row, ("predicate_result_id", "predicate_sha256"))
                for row in values["rule_predicates"]
            )
        ),
        assignment_sha256s=tuple(
            sorted(
                _digest(row, ("assignment_id", "assignment_sha256"))
                for row in values["rule_assignments"]
            )
        ),
    )
    history_binding = _history._candidate_binding_sha256(
        candidate_freeze_sha256=freeze_sha,
        candidate_universe_sha256=frozen["candidate_universe_sha256"],
        candidate_count=frozen["candidate_count"],
        candidate_service_keys=tuple(frozen["candidate_service_keys"]),
    )
    # Original history-off defaults, with no library/query/config source inputs.
    history_config_sha = _digest(
        {
            "schema_version": _history.HISTORY_SCORING_CONFIG_SCHEMA_VERSION,
            "k": 3,
            "min_similarity": 0.7,
            "time_policy": "strict_prior",
        }
    )
    history_input = _history._authoritative_input_sha256(
        candidate_binding_sha256=history_binding,
        mode="off",
        generation_authority_sha256=None,
        library_sha256=None,
        query_embedding_sha256=None,
        service_mapping_sha256s=(),
        config_sha256=history_config_sha,
    )
    actual = {
        "candidate_freeze_sha256": freeze_sha,
        "family_aggregation_result_sha256": values["expected_family_aggregation_sha256"],
        "propagation_scoring_result_sha256": values["expected_propagation_sha256"],
        "propagation_authoritative_input_sha256": prop_input,
        "rule_scoring_result_sha256": _digest(
            values["rule_scoring"], ("rule_scoring_result_id", "result_sha256")
        ),
        "rule_authoritative_input_sha256": rule_input,
        "history_evaluation_result_sha256": _digest(
            values["history_evaluation"], ("history_evaluation_result_id", "result_sha256")
        ),
        "history_authoritative_input_sha256": history_input,
    }
    if actual != values["authority"]:
        raise ValueError(
            "NoRole ranking eight independent authority pins differ from actual content"
        )
    for name, authority_field in (
        ("rule_scoring", "rule_scoring_result_sha256"),
        ("history_evaluation", "history_evaluation_result_sha256"),
    ):
        if values[name]["result_sha256"] != actual[authority_field]:
            raise ValueError("NoRole ranking channel result differs from its independent pin")
    for result, field in (
        (prop.result, "propagation"),
        (values["rule_scoring"], "rule"),
        (values["history_evaluation"], "history"),
    ):
        if result["authoritative_input_sha256"] != actual[field + "_authoritative_input_sha256"]:
            raise ValueError("NoRole ranking channel authority differs")
    config_sha = _digest(values["config"], ("config_sha256",))
    if values["config"]["config_sha256"] != config_sha or frozen["freeze_sha256"] != freeze_sha:
        raise ValueError("NoRole ranking config/freeze content hash differs")
    return _digest(_authority_document(frozen, config_sha, actual, source_pin, selection_pin))


def _fresh(plain: dict[str, Any], cls: type[_ModelT]) -> _ModelT:
    encoded = _roles._encoded(plain)
    result = cls.model_validate_json(encoded)
    if not hmac.compare_digest(encoded, _roles._encoded(_prepare(result)[0])):
        raise ValueError("NoRole ranking fresh model representation differs")
    return result


def _build(original: _Seal, authority_pin: str) -> NoRoleReferenceRankingResult:
    authority_sha = _authority(original)
    if authority_sha != authority_pin:
        raise ValueError("NoRole ranking independent input authority differs")
    values = original.values
    common = (
        values["expected_source_records_sha256"],
        values["expected_selection_sha256"],
        values["expected_role_compilation_sha256"],
    )
    family = _evidence._build_family(
        original.evidence,
        *common,
        values["expected_event_strength_sha256"],
        values["aggregator_version"],
    )
    if not _evidence._matches(
        original.evidence.result, family, values["expected_family_aggregation_sha256"]
    ):
        raise ValueError("NoRole ranking family fails complete original source replay")
    prop = _propagation._build(
        original.propagation, *common, values["authority"]["propagation_authoritative_input_sha256"]
    )
    if (
        original.propagation.result is None
        or prop.result_sha256 != values["expected_propagation_sha256"]
        or not hmac.compare_digest(
            _roles._encoded(original.propagation.result),
            _roles._encoded(_propagation._prepare(prop)[0]),
        )
    ):
        raise ValueError("NoRole ranking propagation fails complete retained source replay")
    inputs = _selection._snapshot(_roles._encoded(original.propagation.roles.source))
    frozen = inputs.frozen_candidates
    rules = _fresh(values["rule_scoring"], RuleScoringResult)
    rule_set = _fresh(values["rule_set"], RuleSet)
    assignments = tuple(_fresh(row, RuleCandidateAssignment) for row in values["rule_assignments"])
    predicates = tuple(_fresh(row, PredicateResult) for row in values["rule_predicates"])
    history = _fresh(values["history_evaluation"], HistoryEvaluationResult)
    authority = values["authority"]
    if not RuleScoringResult.verify_trusted(
        rules,
        expected_result_sha256=authority["rule_scoring_result_sha256"],
        expected_authoritative_input_sha256=authority["rule_authoritative_input_sha256"],
        frozen_candidates=frozen,
        rule_set=rule_set,
        assignments=assignments,
        predicates=predicates,
    ):
        raise ValueError("NoRole ranking rule trusted replay failed")
    if not HistoryEvaluationResult.verify_trusted(
        history,
        expected_result_sha256=authority["history_evaluation_result_sha256"],
        expected_authoritative_input_sha256=authority["history_authoritative_input_sha256"],
        frozen_candidates=frozen,
        mode="off",
    ):
        raise ValueError("NoRole ranking history-off trusted replay failed")
    config = _fresh(values["config"], ReferenceScoringConfig)
    if not ReferenceScoringConfig.verify(config):
        raise ValueError("NoRole ranking configuration failed verification")
    keys = frozen.candidate_service_keys
    if any(
        channel != keys
        for channel in (
            tuple(row.candidate_service_key for row in family.candidates),
            tuple(row.candidate_service_key for row in prop.candidate_scores),
            tuple(row.candidate_service_key for row in rules.candidates),
            tuple(row.candidate_service_key for row in history.candidate_scores),
        )
    ):
        raise ValueError("NoRole ranking channels must exactly cover frozen C")
    if any(row.opaque_case_id != frozen.opaque_case_id for row in (family, prop, rules)):
        raise ValueError("NoRole ranking channel case differs")
    if any(
        row.candidate_freeze_sha256 != frozen.freeze_sha256
        for row in (family, prop, rules, history)
    ):
        raise ValueError("NoRole ranking channel freeze differs")
    if history.candidate_universe_sha256 != frozen.candidate_universe_sha256:
        raise ValueError("NoRole ranking history universe differs")
    enabled = any(
        contribution.activation_status != "disabled"
        for row in rules.candidates
        for contribution in row.contributions
    )
    scores = tuple(
        _ranking._build_candidate_decomposition(
            candidate_service_key=key,
            event_evidence_score=family_row.evidence_strength,
            propagation_score=prop_row.propagation_score,
            rule_score=rule_row.rule_score,
            history_score=history_row.history_score,
            explicit_rules_enabled=enabled,
            config=config,
            event_group_ids=tuple(
                event
                for maximum in family_row.family_maxima
                for event in maximum.retained_event_group_ids
            ),
            propagation_path_ids=tuple(
                row.path_audit_id for row in prop.path_audits if row.candidate_service_key == key
            ),
            rule_ids=tuple(row.contribution_id for row in rule_row.contributions),
            history_neighbor_ids=tuple(
                row.selected_neighbor_id for row in history_row.contributions
            ),
        )
        for key, family_row, prop_row, rule_row, history_row in zip(
            keys,
            family.candidates,
            prop.candidate_scores,
            rules.candidates,
            history.candidate_scores,
            strict=True,
        )
    )
    groups = _ranking._build_evidence_tie_groups(scores)
    order = _ranking._ordered_service_keys(
        scores=scores, tie_break_seed_sha256=config.tie_break_seed_sha256
    )
    top = build_top_k(
        opaque_case_id=frozen.opaque_case_id,
        candidate_universe_sha256=frozen.candidate_universe_sha256,
        scores=scores,
        tie_groups=groups,
        ordered_service_keys=order,
        k=config.top_k,
        tie_break_seed_sha256=config.tie_break_seed_sha256,
    )
    unsigned: dict[str, object] = {
        "schema_version": NO_ROLE_REFERENCE_RANKING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": frozen.opaque_case_id,
        "candidate_universe_sha256": frozen.candidate_universe_sha256,
        "candidate_count": len(keys),
        "candidate_scope_decision_sha256": _ranking._accepted_scope_decision_sha256(
            candidate_freeze_sha256=frozen.freeze_sha256, candidate_service_keys=keys
        ),
        "config": config,
        "explicit_rules_enabled": enabled,
        "effective_eta_r": config.eta_r_when_enabled if enabled else 0.0,
        "effective_eta_h": config.eta_h,
        **authority,
        "scores": scores,
        "evidence_tie_groups": groups,
        "ordered_service_keys": order,
        "top_k": top,
        "authoritative_input_sha256": authority_sha,
        "is_root_cause_probability": False,
        "source_records_sha256": common[0],
        "scoring_selection_sha256": common[1],
        "no_role_policy_version": NO_ROLE_POLICY_VERSION,
        "no_role_policy_sha256": NO_ROLE_POLICY_SHA256,
        "source_family_schema_version": _evidence.NO_ROLE_FAMILY_AGGREGATION_RESULT_SCHEMA_VERSION,
        "source_propagation_schema_version": (
            _propagation.NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
        ),
    }
    digest = _digest(cast(dict[str, object], _selection._plain(unsigned)))
    return NoRoleReferenceRankingResult.model_validate(
        {
            **unsigned,
            "reference_ranking_result_id": "nrrank_" + digest[:24],
            "result_sha256": digest,
        }
    )


def no_role_ranking_input_sha256(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
    event_strength: NoRoleEventStrengthResult,
    expected_event_strength_sha256: str,
    family_aggregation: NoRoleFamilyAggregationResult,
    expected_family_aggregation_sha256: str,
    aggregator_version: str,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    propagation_config: PropagationScoringConfig,
    propagation_scoring: NoRolePropagationScoringResult,
    expected_propagation_sha256: str,
    rule_set: RuleSet,
    rule_assignments: tuple[RuleCandidateAssignment, ...],
    rule_predicates: tuple[PredicateResult, ...],
    rule_scoring: RuleScoringResult,
    history_evaluation: HistoryEvaluationResult,
    history_mode: Literal["off"] = "off",
    authority: RankingInputAuthority,
    config: ReferenceScoringConfig,
) -> str:
    """Digest preflighted actual content; it grants and caches no source eligibility."""
    values = dict(locals())
    original = _seal(values)
    try:
        return _authority(original)
    finally:
        _unchanged(original, _seal(values))


def rank_no_role_reference_candidates(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
    event_strength: NoRoleEventStrengthResult,
    expected_event_strength_sha256: str,
    family_aggregation: NoRoleFamilyAggregationResult,
    expected_family_aggregation_sha256: str,
    aggregator_version: str,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    propagation_config: PropagationScoringConfig,
    propagation_scoring: NoRolePropagationScoringResult,
    expected_propagation_sha256: str,
    rule_set: RuleSet,
    rule_assignments: tuple[RuleCandidateAssignment, ...],
    rule_predicates: tuple[PredicateResult, ...],
    rule_scoring: RuleScoringResult,
    history_evaluation: HistoryEvaluationResult,
    history_mode: Literal["off"] = "off",
    authority: RankingInputAuthority,
    config: ReferenceScoringConfig,
    expected_authoritative_input_sha256: str,
) -> NoRoleReferenceRankingResult:
    """Replay complete original sources, then apply the original binary64 formula."""
    values = dict(locals())
    original = _seal(values)
    try:
        return _build(original, expected_authoritative_input_sha256)
    finally:
        _unchanged(original, _seal(values))
