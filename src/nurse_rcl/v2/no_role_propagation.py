"""Independent NoRole propagation with complete source and retained graph gates.

The pure input digest and structural receipts confer no source authority. Build
and replay independently prove empty eligible roles before the original empty
bundle aggregation. Retained temporal records do not supply propagation credit;
their missing raw clock-resolution source is deliberately not reconstructed.
"""

from __future__ import annotations

import hmac
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict, TypeVar, cast

from pydantic import ConfigDict, Field, ValidationInfo, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import case_scoring_selection as _selection
from nurse_rcl.v2 import no_role_compilation as _roles
from nurse_rcl.v2 import no_role_evidence as _evidence
from nurse_rcl.v2 import propagation_scoring as _scoring
from nurse_rcl.v2 import temporal_condensation as _temporal
from nurse_rcl.v2.case_scoring_selection import CaseGuardReplayInputs, CaseScoringSelection
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.no_role_compilation import (
    NO_ROLE_POLICY_SHA256,
    NO_ROLE_POLICY_VERSION,
    NoRoleCompilationResult,
)
from nurse_rcl.v2.propagation_scoring import (
    CandidatePropagationScore,
    PropagationPathAudit,
    PropagationRelationBinding,
    PropagationScoringConfig,
    PropagationTargetScore,
)
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree
from nurse_rcl.v2.temporal_condensation import (
    OnsetInterval,
    RelationSCCCondensation,
    TemporalPrecedence,
)

NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION: Literal[
    "nurse.v2.no-role-propagation-scoring-result/1"
] = "nurse.v2.no-role-propagation-scoring-result/1"
NO_ROLE_PROPAGATION_INPUT_SCHEMA_VERSION = "nurse.v2.no-role-propagation-input/1"
_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)


def _preflight(value: object, *, temporal: bool = False) -> None:
    preflight_exact_pydantic_tree(
        value,
        root_model_types=(
            NoRolePropagationScoringResult,
            GraphSnapshot,
            RelationSCCCondensation,
            OnsetInterval,
            TemporalPrecedence,
        ),
        label="NoRole propagation",
        max_string_length=8192 if temporal else None,
        provenance_id_limit=131072 if temporal else None,
    )
    _selection._plain(value)
    _roles._model_capacities(value)
    if type(value) is NoRolePropagationScoringResult:
        _config_limits(object.__getattribute__(value, "config"))
    elif type(value) is PropagationScoringConfig:
        _config_limits(value)
    elif type(value) is dict:
        config = cast(dict[str, object], value).get("config")
        if type(config) is PropagationScoringConfig:
            _config_limits(config)


def _prepare(value: object) -> tuple[dict[str, object], bytes]:
    _preflight(value)
    plain = cast(dict[str, object], _selection._plain(value))
    _selection._measure_json(plain)
    return plain, _roles._representation(value)


def _bounded_json_int(token: str) -> int:
    # The ONLY integer fields in this result are the four config limits. Their
    # largest permitted value is 500000; reject impossible tokens before the
    # unbounded decimal conversion on Python 3.10.2. Source ints have no new cap.
    if len(token) > 6:
        raise ValueError("NoRole propagation JSON integer exceeds its config domain")
    return int(token)


def _config_limits(config: PropagationScoringConfig) -> None:
    state = object.__getattribute__(config, "__dict__")
    for name, maximum in (
        ("max_bundle_pairs", 32768),
        ("max_expanded_states", 100000),
        ("max_traversed_edges", 500000),
    ):
        value = state[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("NoRole propagation config integer outside its declared domain")
    if type(state["propagation_max_hops"]) is not int or state["propagation_max_hops"] not in (
        1,
        2,
        3,
        5,
    ):
        raise ValueError("NoRole propagation config hop limit differs")
    if type(state["gamma"]) is not float or state["gamma"] != _scoring.PROPAGATION_GAMMA:
        raise ValueError("NoRole propagation config gamma differs")


class NoRolePropagationScoringResult(StrictFrozenModel):
    """Internal integrity only; source authority requires independent pinned replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["nurse.v2.no-role-propagation-scoring-result/1"] = (
        NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION
    )
    scoring_result_id: str = Field(pattern=r"^npres_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    policy_version: Literal["nurse-v2-bounded-propagation/1"] = _scoring.PROPAGATION_POLICY_VERSION
    config: PropagationScoringConfig
    authoritative_input_sha256: _SHA
    candidate_freeze_sha256: _SHA
    candidate_service_keys: tuple[str, ...] = Field(min_length=1)
    event_grouping_result_sha256: _SHA
    role_compilation_id: str = Field(pattern=r"^nrc_[0-9a-f]{24}$")
    role_compilation_result_sha256: _SHA
    relation_policy_sha256: _SHA
    condensation_sha256s: tuple[str, ...]
    onset_sha256s: tuple[str, ...]
    precedence_sha256s: tuple[str, ...]
    role_assertion_sha256s: tuple[str, ...] = Field(max_length=0)
    relation_bindings: tuple[PropagationRelationBinding, ...] = Field(max_length=0)
    path_audits: tuple[PropagationPathAudit, ...] = Field(max_length=0)
    target_scores: tuple[PropagationTargetScore, ...] = Field(max_length=0)
    candidate_scores: tuple[CandidatePropagationScore, ...] = Field(min_length=1)
    result_sha256: _SHA
    source_records_sha256: _SHA
    scoring_selection_sha256: _SHA
    source_role_schema_version: Literal["nurse.v2.no-role-compilation-result/1"] = (
        _roles.NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION
    )
    no_role_policy_version: Literal["nurse-v2-no-eligible-role-replay/1"] = NO_ROLE_POLICY_VERSION
    no_role_policy_sha256: _SHA

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
        if cls is not NoRolePropagationScoringResult:
            raise ValueError("NoRole propagation JSON requires the exact result class")
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
                raise ValueError("NoRole propagation JSON field representation differs")
            return super().model_validate_json(
                raw, strict=True, context=context, by_alias=by_alias, by_name=by_name
            )
        finally:
            if not hmac.compare_digest(raw, _roles._json_bytes(json_data)):
                raise ValueError("NoRole propagation JSON changed during validation")

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object, info: ValidationInfo) -> object:
        if cls is not NoRolePropagationScoringResult or (
            type(value) is not dict and type(value) is not NoRolePropagationScoringResult
        ):
            raise ValueError("NoRole propagation requires an exact model or dictionary")
        _preflight(value)
        _selection._measure_json(_selection._plain(value))
        if type(value) is dict:
            if info.mode == "json":
                if not _evidence._json_shape(value, cls):
                    raise ValueError("NoRole propagation JSON representation differs")
                return _evidence._json_containers(value, cls)
            if not all(
                name in cls.model_fields
                and _selection._exact(child, cls.model_fields[name].annotation)
                for name, child in cast(dict[str, object], value).items()
            ):
                raise ValueError("NoRole propagation fields require exact declared types")
        return value

    @model_validator(mode="after")
    def closed_no_role_result(self) -> Self:
        if self.no_role_policy_sha256 != NO_ROLE_POLICY_SHA256:
            raise ValueError("NoRole propagation activation policy differs")
        if self.relation_policy_sha256 != _scoring.PROPAGATION_RELATION_POLICY_SHA256:
            raise ValueError("NoRole propagation relation policy differs")
        if (
            self.role_assertion_sha256s
            or self.relation_bindings
            or self.path_audits
            or self.target_scores
        ):
            raise ValueError("NoRole propagation roles and paths must remain empty")
        if self.candidate_service_keys != tuple(sorted(set(self.candidate_service_keys))):
            raise ValueError("NoRole propagation candidates must be sorted and unique")
        if (
            tuple(row.candidate_service_key for row in self.candidate_scores)
            != self.candidate_service_keys
        ):
            raise ValueError("NoRole propagation scores must exactly cover retained candidates")
        for row in self.candidate_scores:
            if (
                row.propagation_score != 0.0
                or row.reason_code != "missing_source_support"
                or row.retained_target_score_ids
                or row.top_target_service_keys
                or row.top_target_event_group_ids
                or row.top_target_contributions
            ):
                raise ValueError("NoRole propagation requires original empty-bundle values")
        for hashes in (self.condensation_sha256s, self.onset_sha256s, self.precedence_sha256s):
            if hashes != tuple(sorted(set(hashes))) or any(
                re.fullmatch(r"[0-9a-f]{64}", item) is None for item in hashes
            ):
                raise ValueError("NoRole propagation retained hashes must be canonical and unique")
        return self

    def verify_structure(self) -> bool:
        try:
            if type(self) is not NoRolePropagationScoringResult:
                return False
            original, representation = _prepare(self)
            try:
                return _valid_result(original)
            finally:
                current, current_representation = _prepare(self)
                if original != current or representation != current_representation:
                    raise ValueError("NoRole propagation result changed during verification")
        except _ERRORS:
            return False

    def verify(self) -> bool:
        return NoRolePropagationScoringResult.verify_structure(self)

    def verify_inputs(
        self,
        *,
        inputs: CaseGuardReplayInputs,
        selection: CaseScoringSelection,
        expected_source_records_sha256: str,
        expected_selection_sha256: str,
        role_compilation: NoRoleCompilationResult,
        expected_role_compilation_sha256: str,
        graph_snapshots: tuple[GraphSnapshot, ...],
        condensations: tuple[RelationSCCCondensation, ...],
        onsets: tuple[OnsetInterval, ...],
        precedences: tuple[TemporalPrecedence, ...],
        config: PropagationScoringConfig,
        expected_authoritative_input_sha256: str,
        expected_result_sha256: str,
    ) -> bool:
        try:
            if type(self) is not NoRolePropagationScoringResult:
                return False
            values = _arguments(
                inputs,
                selection,
                role_compilation,
                graph_snapshots,
                condensations,
                onsets,
                precedences,
                config,
            )
            pins = (
                expected_source_records_sha256,
                expected_selection_sha256,
                expected_role_compilation_sha256,
                expected_authoritative_input_sha256,
                expected_result_sha256,
            )
            original = _seal(**values, result=self, pins=pins)
            try:
                if (
                    original.result is None
                    or original.result["result_sha256"] != expected_result_sha256
                ):
                    return False
                rebuilt = _build(original, *pins[:4])
                return hmac.compare_digest(
                    _roles._encoded(original.result), _roles._encoded(_prepare(rebuilt)[0])
                )
            finally:
                _unchanged(original, _seal(**values, result=self))
        except _ERRORS:
            return False


def _valid_result(plain: dict[str, object]) -> bool:
    encoded = _roles._encoded(plain)
    rebuilt = NoRolePropagationScoringResult.model_validate_json(encoded)
    restored = _prepare(rebuilt)[0]
    if not PropagationScoringConfig.verify(rebuilt.config) or any(
        not CandidatePropagationScore.verify(row) for row in rebuilt.candidate_scores
    ):
        return False
    digest = _content_digest(restored, ("scoring_result_id", "result_sha256"))
    return (
        hmac.compare_digest(encoded, _roles._encoded(restored))
        and rebuilt.result_sha256 == digest
        and rebuilt.scoring_result_id == "npres_" + digest[:24]
    )


@dataclass(frozen=True, slots=True)
class _Seal:
    roles: _roles._Seal
    graphs: tuple[dict[str, object], ...]
    condensations: tuple[dict[str, object], ...]
    onsets: tuple[dict[str, object], ...]
    precedences: tuple[dict[str, object], ...]
    config: dict[str, object]
    result: dict[str, object] | None
    representations: tuple[bytes, ...]


class _Arguments(TypedDict):
    inputs: CaseGuardReplayInputs
    selection: CaseScoringSelection
    role_compilation: NoRoleCompilationResult
    graph_snapshots: tuple[GraphSnapshot, ...]
    condensations: tuple[RelationSCCCondensation, ...]
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    config: PropagationScoringConfig


def _seal(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    role_compilation: NoRoleCompilationResult,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    config: PropagationScoringConfig,
    result: NoRolePropagationScoringResult | None = None,
    pins: tuple[str, ...] = (),
) -> _Seal:
    # Reject root kinds by identity before any unknown metaclass can be compared,
    # hashed, iterated, or sent to a serializer/property/plugin.
    for value, expected in (
        (inputs, CaseGuardReplayInputs),
        (selection, CaseScoringSelection),
        (role_compilation, NoRoleCompilationResult),
        (config, PropagationScoringConfig),
    ):
        if type(value) is not expected:
            raise ValueError("NoRole propagation requires exact upstream roots")
    if result is not None and type(result) is not NoRolePropagationScoringResult:
        raise ValueError("NoRole propagation requires an exact result")
    if any(type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None for pin in pins):
        raise ValueError("NoRole propagation requires exact lowercase SHA256 pins")
    sequences = (
        (graph_snapshots, GraphSnapshot, "snapshot_sha256"),
        (condensations, RelationSCCCondensation, "condensation_id"),
        (onsets, OnsetInterval, "onset_id"),
        (precedences, TemporalPrecedence, "precedence_id"),
    )
    for sequence, cls, _ in sequences:
        if type(sequence) is not tuple or len(sequence) > 32768:
            raise ValueError("NoRole propagation requires bounded exact artifact tuples")
        if any(type(row) is not cls for row in sequence):
            raise ValueError("NoRole propagation requires exact artifact classes")
    # No encoding happens during any of these preflights or detached captures.
    roles = _roles._seal(inputs, selection, role_compilation, pins)
    _preflight(config)
    _config_limits(config)
    if result is not None:
        _preflight(result)
        _config_limits(result.config)
    for sequence, cls, _ in sequences:
        for row in sequence:
            _preflight(row, temporal=True)
            _temporal._preflight_model_cardinality(
                row, model_type=cls, label="NoRole retained graph/time"
            )
    _scoring._preflight_condensation_structure(condensations)
    for sequence, _, id_field in sequences:
        ids = tuple(object.__getattribute__(row, id_field) for row in sequence)
        if len(ids) != len(set(ids)):
            raise ValueError("NoRole propagation artifact IDs must be unique")
    plains = tuple(tuple(_prepare(row)[0] for row in sequence) for sequence, _, _ in sequences)
    return _Seal(
        roles,
        plains[0],
        plains[1],
        plains[2],
        plains[3],
        _prepare(config)[0],
        None if result is None else _prepare(result)[0],
        (
            *(_roles._representation(sequence) for sequence, _, _ in sequences),
            _roles._representation(config),
            _roles._representation(result),
        ),
    )


def _unchanged(original: _Seal, current: _Seal) -> None:
    _roles._unchanged(original.roles, current.roles)
    if (
        original.graphs != current.graphs
        or original.condensations != current.condensations
        or original.onsets != current.onsets
        or original.precedences != current.precedences
        or original.config != current.config
        or original.result != current.result
        or original.representations != current.representations
    ):
        raise ValueError("NoRole propagation original content or exact representation changed")


def _content_digest(plain: dict[str, object], excluded: tuple[str, ...]) -> str:
    return sha256_hex(
        _roles._encoded({key: value for key, value in plain.items() if key not in excluded})
    )


def _authority(original: _Seal, source_pin: str, selection_pin: str, role_pin: str) -> str:
    if original.roles.result is None:
        raise ValueError("NoRole compilation is required")
    source_sha = sha256_hex(_roles._encoded(original.roles.source))
    selection_sha = _content_digest(original.roles.selection, ("selection_sha256",))
    role_sha = _content_digest(original.roles.result, ("role_compilation_id", "result_sha256"))
    if source_sha != source_pin or selection_sha != selection_pin or role_sha != role_pin:
        raise ValueError("NoRole propagation independent source/selection/compilation pin differs")
    return sha256_hex(
        _roles._encoded(
            {
                "schema_version": NO_ROLE_PROPAGATION_INPUT_SCHEMA_VERSION,
                "source_records_sha256": source_sha,
                "scoring_selection_sha256": selection_sha,
                "role_compilation_result_sha256": role_sha,
                "graph_snapshot_sha256s": tuple(
                    sorted(_content_digest(row, ("snapshot_sha256",)) for row in original.graphs)
                ),
                "condensation_sha256s": tuple(
                    sorted(
                        _content_digest(row, ("condensation_id", "condensation_sha256"))
                        for row in original.condensations
                    )
                ),
                "onset_sha256s": tuple(
                    sorted(
                        _content_digest(row, ("onset_id", "onset_sha256"))
                        for row in original.onsets
                    )
                ),
                "precedence_sha256s": tuple(
                    sorted(
                        _content_digest(row, ("precedence_id", "precedence_sha256"))
                        for row in original.precedences
                    )
                ),
                "config_sha256": _content_digest(original.config, ("config_id", "config_sha256")),
            }
        )
    )


def _fresh(plain: dict[str, object], cls: type[_ModelT]) -> _ModelT:
    encoded = _roles._encoded(plain)
    rebuilt = cls.model_validate_json(encoded)
    if not hmac.compare_digest(encoded, _roles._encoded(_prepare(rebuilt)[0])):
        raise ValueError("NoRole propagation upstream canonical representation differs")
    return rebuilt


def _retained_gates(original: _Seal, inputs: CaseGuardReplayInputs) -> PropagationScoringConfig:
    contract, universe, frozen = (
        inputs.time_contract,
        inputs.candidate_universe,
        inputs.frozen_candidates,
    )
    case_id = contract.opaque_case_id
    graphs = tuple(_fresh(row, GraphSnapshot) for row in original.graphs)
    by_sha = {row.snapshot_sha256: row for row in graphs}
    if len(by_sha) != len(graphs):
        raise ValueError("NoRole propagation retained graph digests must be unique")
    for graph in graphs:
        window = (
            contract.baseline_window if graph.window_kind == "normal" else contract.fault_window
        )
        if not GraphSnapshot.verify(graph) or (
            graph.opaque_case_id != case_id
            or graph.interval != window
            or graph.decision_time_unix_ns != contract.decision_time_unix_ns
            or graph.observation_contract_sha256 != universe.observation_contract_sha256
            or graph.candidate_universe_sha256 != universe.universe_sha256
            or graph.candidate_freeze_sha256 != frozen.freeze_sha256
            or graph.entity_registry_sha256 != inputs.entity_registry.registry_sha256
            or graph.node_service_keys != frozen.candidate_service_keys
        ):
            raise ValueError("NoRole propagation graph fails retained source gate")
    condensations = tuple(_fresh(row, RelationSCCCondensation) for row in original.condensations)
    relations = tuple(row.relation_type for row in condensations)
    if len(relations) != len(set(relations)):
        raise ValueError("NoRole propagation permits one condensation per directed relation")
    for scc in condensations:
        selected_graph = by_sha.get(scc.graph_snapshot_sha256)
        if (
            selected_graph is None
            or scc.opaque_case_id != case_id
            or not RelationSCCCondensation.verify_inputs(scc, graph_snapshot=selected_graph)
        ):
            raise ValueError("NoRole propagation SCC fails actual retained graph replay")
    onsets = tuple(_fresh(row, OnsetInterval) for row in original.onsets)
    onset_by_id = {row.onset_id: row for row in onsets}
    contract_sha = _temporal.case_time_contract_sha256(contract)
    for onset in onsets:
        if not OnsetInterval.verify(onset) or (
            onset.opaque_case_id != case_id
            or onset.time_contract_sha256 != contract_sha
            or onset.service_key not in frozen.candidate_service_keys
            or onset.t_low_unix_ns < contract.baseline_window.start_unix_ns
            or onset.t_high_unix_ns > contract.decision_time_unix_ns
        ):
            raise ValueError("NoRole propagation onset fails retained source gate")
    precedences = tuple(_fresh(row, TemporalPrecedence) for row in original.precedences)
    pairs: set[tuple[str, str]] = set()
    for precedence in precedences:
        if not TemporalPrecedence.verify(precedence) or (
            precedence.opaque_case_id != case_id
            or precedence.time_contract_sha256 != contract_sha
            or precedence.clock_uncertainty_ns != contract.clock_uncertainty_ns
        ):
            raise ValueError("NoRole propagation precedence fails retained source gate")
        references = (precedence.source_onset_id, precedence.target_onset_id)
        if any(key is not None and key not in onset_by_id for key in references):
            raise ValueError("NoRole propagation precedence onset references must close")
        if precedence.source_onset_id is not None and precedence.target_onset_id is not None:
            pair = (precedence.source_onset_id, precedence.target_onset_id)
            if pair in pairs:
                raise ValueError("NoRole propagation precedence pairs must be unique")
            pairs.add(pair)
            if precedence.reason == "onset_outside_time_contract":
                raise ValueError("NoRole propagation retained onsets are inside the time contract")
            if precedence.clock_uncertainty_resolved:
                source, target = (onset_by_id[key] for key in pair)
                uncertainty = contract.clock_uncertainty_ns
                expected = (
                    max(0, source.t_low_unix_ns - uncertainty),
                    source.t_high_unix_ns + uncertainty,
                    max(0, target.t_low_unix_ns - uncertainty),
                    target.t_high_unix_ns + uncertainty,
                )
                actual = (
                    precedence.source_effective_low_unix_ns,
                    precedence.source_effective_high_unix_ns,
                    precedence.target_effective_low_unix_ns,
                    precedence.target_effective_high_unix_ns,
                )
                if actual != expected:
                    raise ValueError(
                        "NoRole propagation effective bounds differ from retained onsets"
                    )
        provenance = {contract_sha, precedence.clock_resolution_sha256}
        for key in references:
            if key is not None:
                provenance.update(onset_by_id[key].provenance_ids)
        if not provenance <= set(precedence.provenance_ids):
            raise ValueError("NoRole propagation precedence omits retained source provenance")
    # FrozenClockResolution is not an API input. The checks above establish only
    # retained-record consistency, never replay of the absent raw clock source.
    config = _fresh(original.config, PropagationScoringConfig)
    if not PropagationScoringConfig.verify(config):
        raise ValueError("NoRole propagation config failed content verification")
    return config


def _build(
    original: _Seal, source_pin: str, selection_pin: str, role_pin: str, authority_pin: str
) -> NoRolePropagationScoringResult:
    authority = _authority(original, source_pin, selection_pin, role_pin)
    if not hmac.compare_digest(authority, authority_pin):
        raise ValueError("NoRole propagation independent authority pin differs")
    roles = _roles._build(original.roles, source_pin, selection_pin)
    if (
        original.roles.result is None
        or roles.result_sha256 != role_pin
        or not hmac.compare_digest(
            _roles._encoded(original.roles.result), _roles._encoded(_roles._prepare(roles)[0])
        )
    ):
        raise ValueError("NoRole compilation differs from complete original source replay")
    inputs = _selection._snapshot(_roles._encoded(original.roles.source))
    config = _retained_gates(original, inputs)
    scores = _scoring._build_candidate_scores(
        candidate_keys=inputs.frozen_candidates.candidate_service_keys,
        source_bundles=(),
        target_bundles=(),
        target_scores=(),
        path_audits=(),
    )
    unsigned: dict[str, object] = {
        "schema_version": NO_ROLE_PROPAGATION_SCORING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": roles.opaque_case_id,
        "policy_version": _scoring.PROPAGATION_POLICY_VERSION,
        "config": config,
        "authoritative_input_sha256": authority,
        "candidate_freeze_sha256": roles.candidate_freeze_sha256,
        "candidate_service_keys": inputs.frozen_candidates.candidate_service_keys,
        "event_grouping_result_sha256": inputs.event_grouping.result_sha256,
        "role_compilation_id": roles.role_compilation_id,
        "role_compilation_result_sha256": roles.result_sha256,
        "relation_policy_sha256": _scoring.PROPAGATION_RELATION_POLICY_SHA256,
        "condensation_sha256s": tuple(
            sorted(cast(str, row["condensation_sha256"]) for row in original.condensations)
        ),
        "onset_sha256s": tuple(sorted(cast(str, row["onset_sha256"]) for row in original.onsets)),
        "precedence_sha256s": tuple(
            sorted(cast(str, row["precedence_sha256"]) for row in original.precedences)
        ),
        "role_assertion_sha256s": (),
        "relation_bindings": (),
        "path_audits": (),
        "target_scores": (),
        "candidate_scores": scores,
        "source_records_sha256": source_pin,
        "scoring_selection_sha256": selection_pin,
        "source_role_schema_version": _roles.NO_ROLE_COMPILATION_RESULT_SCHEMA_VERSION,
        "no_role_policy_version": NO_ROLE_POLICY_VERSION,
        "no_role_policy_sha256": NO_ROLE_POLICY_SHA256,
    }
    digest = sha256_hex(_roles._encoded(cast(dict[str, object], _selection._plain(unsigned))))
    return NoRolePropagationScoringResult.model_validate(
        {**unsigned, "scoring_result_id": "npres_" + digest[:24], "result_sha256": digest}
    )


def _arguments(
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    role_compilation: NoRoleCompilationResult,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    config: PropagationScoringConfig,
) -> _Arguments:
    return {
        "inputs": inputs,
        "selection": selection,
        "role_compilation": role_compilation,
        "graph_snapshots": graph_snapshots,
        "condensations": condensations,
        "onsets": onsets,
        "precedences": precedences,
        "config": config,
    }


def no_role_propagation_input_sha256(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    config: PropagationScoringConfig,
) -> str:
    """Freeze complete retained input content; this pure digest confers no eligibility."""
    values = _arguments(
        inputs,
        selection,
        role_compilation,
        graph_snapshots,
        condensations,
        onsets,
        precedences,
        config,
    )
    pins = (
        expected_source_records_sha256,
        expected_selection_sha256,
        expected_role_compilation_sha256,
    )
    original = _seal(**values, pins=pins)
    try:
        return _authority(original, *pins)
    finally:
        _unchanged(original, _seal(**values))


def score_no_role_propagation(
    *,
    inputs: CaseGuardReplayInputs,
    selection: CaseScoringSelection,
    expected_source_records_sha256: str,
    expected_selection_sha256: str,
    role_compilation: NoRoleCompilationResult,
    expected_role_compilation_sha256: str,
    graph_snapshots: tuple[GraphSnapshot, ...],
    condensations: tuple[RelationSCCCondensation, ...],
    onsets: tuple[OnsetInterval, ...],
    precedences: tuple[TemporalPrecedence, ...],
    config: PropagationScoringConfig,
    expected_authoritative_input_sha256: str,
) -> NoRolePropagationScoringResult:
    """Replay complete pinned sources and retained graph/time gates, then aggregate."""
    values = _arguments(
        inputs,
        selection,
        role_compilation,
        graph_snapshots,
        condensations,
        onsets,
        precedences,
        config,
    )
    pins = (
        expected_source_records_sha256,
        expected_selection_sha256,
        expected_role_compilation_sha256,
        expected_authoritative_input_sha256,
    )
    original = _seal(**values, pins=pins)
    try:
        return _build(original, *pins)
    finally:
        _unchanged(original, _seal(**values))
