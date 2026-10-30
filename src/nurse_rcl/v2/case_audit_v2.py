"""Independent /2 evidence and ranking replay over complete guard sources.

These contracts prove internal closure. External source authority belongs to the
reader's independent audit pin; a self-reported bundle hash cannot establish it.
"""

from __future__ import annotations

import hmac
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2 import no_role_compilation as _no_roles
from nurse_rcl.v2 import no_role_evidence as _no_evidence
from nurse_rcl.v2 import no_role_propagation as _no_propagation
from nurse_rcl.v2 import no_role_ranking as _no_ranking
from nurse_rcl.v2 import temporal_condensation as _temporal
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.candidate_universe import EmptyCandidateUniverseFailure
from nurse_rcl.v2.case_scoring_selection import (
    CaseGuardReplayInputs,
    CaseScoringSelection,
    ReplayedCaseScoringInputs,
    _exact,
    _measure_json,
    case_guard_replay_inputs_sha256,
    require_case_scoring_selection,
)
from nurse_rcl.v2.context_expansion import CardExpansionPolicy
from nurse_rcl.v2.diagnostic_schemas import PredicateResult
from nurse_rcl.v2.event_grouping import EventGroupingResult
from nurse_rcl.v2.event_strength import EventStrengthResult
from nurse_rcl.v2.family_aggregation import FamilyAggregationResult
from nurse_rcl.v2.graph_snapshots import GraphSnapshot
from nurse_rcl.v2.history_scoring import HistoryEvaluationResult
from nurse_rcl.v2.hypothesis_guard import HypothesisGuardResult
from nurse_rcl.v2.mechanism_library import MechanismLibrary
from nurse_rcl.v2.no_role_compilation import NoRoleCompilationResult
from nurse_rcl.v2.no_role_evidence import NoRoleEventStrengthResult, NoRoleFamilyAggregationResult
from nurse_rcl.v2.no_role_propagation import NoRolePropagationScoringResult
from nurse_rcl.v2.no_role_ranking import NoRoleReferenceRankingResult
from nurse_rcl.v2.observation_schemas import (
    CandidateUniverse,
    CaseTimeContract,
    SemanticFocusSet,
    TimeInterval,
)
from nurse_rcl.v2.propagation_scoring import PropagationScoringConfig, PropagationScoringResult
from nurse_rcl.v2.query_session import CaseQuerySessionResult
from nurse_rcl.v2.ranking_schemas import HistoryMode, RuleSet, TopKArtifact
from nurse_rcl.v2.reference_ranking import (
    RankingInputAuthority,
    ReferenceRankingResult,
    ReferenceScoringConfig,
    rank_reference_candidates,
)
from nurse_rcl.v2.role_compiler import RoleBindingProposal, RoleCompilationResult
from nurse_rcl.v2.rule_scoring import RuleCandidateAssignment, RuleScoringResult
from nurse_rcl.v2.runtime_schemas import (
    AttemptLedger,
    CaseCheckpoint,
    CostRecord,
    RunMode,
    SealedAuditPackage,
    SemanticCall,
)
from nurse_rcl.v2.safe_serialization import (
    _declared_model_type_closure,
    preflight_exact_pydantic_tree,
)
from nurse_rcl.v2.semantic_context import SemanticContextBundle
from nurse_rcl.v2.temporal_condensation import (
    OnsetInterval,
    RelationSCCCondensation,
    TemporalPrecedence,
)
from nurse_rcl.v2.temporal_entity_registry import TemporalEntityRegistry

if TYPE_CHECKING:
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2

CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION: Literal["nurse.v2.case-evidence-replay-bundle/2"] = (
    "nurse.v2.case-evidence-replay-bundle/2"
)
CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION: Literal["nurse.v2.case-ranking-replay-bundle/2"] = (
    "nurse.v2.case-ranking-replay-bundle/2"
)
_MAX_DEPTH = 64
_MAX_NODES = 500_000
_ERRORS = (AttributeError, KeyError, RecursionError, TypeError, ValueError)
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)
_AUTHORITY_TYPES = get_type_hints(RankingInputAuthority)
_EVIDENCE_ORDER = {
    "role_predicates": "predicate_result_id",
    "query_sessions": "sequence_number",
    "role_guard_results": "guard_result_id",
    "role_proposals": "proposal_id",
    "graph_snapshots": "snapshot_sha256",
    "condensations": "condensation_id",
    "onsets": "onset_id",
    "precedences": "precedence_id",
}
_RANKING_ORDER = {
    "rule_assignments": "assignment_id",
    "rule_predicates": "predicate_result_id",
    "semantic_calls": "semantic_call_id",
    "attempt_ledgers": "logical_request_id",
    "costs": "cost_sha256",
    "checkpoints": "sequence",
}


def _model_state(value: object, model_type: type[BaseModel]) -> dict[str, object]:
    """Read only storage belonging to an already allowlisted exact class."""
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or len(state) != len(model_type.model_fields):
        raise ValueError("audit model has hidden or missing state")
    if any(type(key) is not str or key not in model_type.model_fields for key in state):
        raise ValueError("audit model has noncanonical field names")
    if (
        object.__getattribute__(value, "__pydantic_extra__") is not None
        or object.__getattribute__(value, "__pydantic_private__") is not None
    ):
        raise ValueError("audit model has hidden slot state")
    marked = object.__getattribute__(value, "__pydantic_fields_set__")
    if type(marked) is not set or any(
        type(key) is not str or key not in model_type.model_fields for key in marked
    ):
        raise ValueError("audit model has noncanonical field markers")
    return cast(dict[str, object], state)


def _inspect(
    value: object, *, _additional_roots: tuple[type[BaseModel], ...] = ()
) -> tuple[object, tuple[object, ...]]:
    """Bound the whole tree without invoking serializers, properties or plugins.

    Preserve representation separately from JSON so even a legal builder list
    replaced by an equivalent tuple during replay is detected by the finally gate.
    """
    allowed = _declared_model_type_closure(
        (CaseEvidenceReplayBundleV2, CaseRankingReplayBundleV2, *_additional_roots)
    )
    shape: list[object] = []
    active: set[int] = set()
    models: list[BaseModel] = []
    model_documents: dict[int, dict[str, Any]] = {}
    scheduled = 1

    def visit(current: object, depth: int) -> object:
        nonlocal scheduled
        if depth > _MAX_DEPTH:
            raise ValueError("audit exceeds serialization depth budget")
        kind = type(current)
        if kind is str or kind is int or kind is float or kind is bool or current is None:
            if kind is float and not math.isfinite(cast(float, current)):
                raise ValueError("audit requires finite numbers")
            shape.append((kind, repr(current)) if kind is float else kind)
            return current
        model_type = next((item for item in allowed if kind is item), None)
        annotations: dict[str, object] | None = None
        if model_type is not None:
            models.append(cast(BaseModel, current))
            state = _model_state(current, model_type)
            annotations = {name: f.annotation for name, f in model_type.model_fields.items()}
            shape.append(
                (kind, tuple(sorted(object.__getattribute__(current, "__pydantic_fields_set__"))))
            )
        elif kind is RankingInputAuthority:
            # This slots-only code-owned dataclass has no plugin state or properties.
            annotations = _AUTHORITY_TYPES
            state = {name: object.__getattribute__(current, name) for name in annotations}
            shape.append(kind)
        elif kind is dict:
            state = cast(dict[str, object], current)
            shape.append(dict)
        elif kind is tuple or kind is list:
            state = {}
            shape.append(kind)
        else:
            raise ValueError("audit requires exact code-owned runtime types")
        count = (
            len(cast(Sequence[object], current)) if kind is tuple or kind is list else len(state)
        )
        scheduled += count
        if scheduled > _MAX_NODES:
            raise ValueError("audit exceeds serialization node budget")
        if id(current) in active:
            raise ValueError("audit contains a serialization cycle")
        active.add(id(current))
        shape.append(count)
        try:
            if kind is tuple or kind is list:
                return [visit(item, depth + 1) for item in cast(Sequence[object], current)]
            if any(
                type(key) is not str and type(key) is not int
                for key in cast(dict[object, object], state)
            ):
                raise ValueError("audit requires exact JSON mapping keys")
            if annotations is not None and any(
                not _exact(item, annotations[name]) for name, item in state.items()
            ):
                raise ValueError("audit model fields require exact declared types")
            # Dict order is representational state even though canonical JSON sorts it.
            shape.append(tuple(state))
            document = {name: visit(item, depth + 1) for name, item in state.items()}
            if model_type is not None:
                model_documents[id(current)] = document
            return document
        finally:
            active.remove(id(current))

    plain = visit(value, 0)
    _measure_json(plain)
    _collection_limits(plain)
    # The entire tree has now been inspected. These narrower module preflights
    # likewise read storage only and never call validators or encoders.
    for model in models:
        kind = type(model)
        state = object.__getattribute__(model, "__dict__")
        if _additional_roots:
            marked = object.__getattribute__(model, "__pydantic_fields_set__")
            if any(
                field.is_required() and name not in marked
                for name, field in kind.model_fields.items()
            ):
                raise ValueError("audit required fields must retain their field markers")
            _audit_declared_domains(model_documents[id(model)], kind)
        for name, field in kind.model_fields.items():
            _declared_limits(state[name], field.metadata)
            _declared_annotation_limits(state[name], field.annotation)
        if "candidate_count" in kind.model_fields and kind is not EmptyCandidateUniverseFailure:
            _no_ranking._bounded_count(state["candidate_count"], 4096)
        if kind is ReferenceRankingResult or kind is NoRoleReferenceRankingResult:
            _no_ranking._result_limits(model_documents[id(model)])
        if kind is CaseQuerySessionResult:
            preflight_exact_pydantic_tree(
                model,
                root_model_types=(kind,),
                label="audit query",
                max_nodes=20_000,
                max_string_length=None,
                provenance_id_limit=None,
            )
        if kind is PredicateResult:
            pending = [model]
            count = 1
            while pending:
                operands = object.__getattribute__(pending.pop(), "__dict__")["operand_states"]
                count += len(operands)
                if count > 4096:
                    raise ValueError("audit predicate exceeds per-tree record capacity")
                pending.extend(operands)
        if (
            kind is GraphSnapshot
            or kind is RelationSCCCondensation
            or kind is OnsetInterval
            or kind is TemporalPrecedence
        ):
            _no_propagation._preflight(model, temporal=True)
            _temporal._preflight_model_cardinality(
                model, model_type=cast(type[StrictFrozenModel], kind), label="audit graph/time"
            )
        if kind is PropagationScoringConfig:
            _no_propagation._config_limits(cast(PropagationScoringConfig, model))
        if (
            kind is RuleSet
            or kind is RuleCandidateAssignment
            or kind is RuleScoringResult
            or kind is HistoryEvaluationResult
        ):
            _no_ranking._preflight(model, narrow=True)
    return plain, tuple(shape)


@lru_cache(maxsize=128)
def _declared_pattern_validator(pattern: str) -> SchemaValidator:
    # Cache only code-owned expressions, never values, artifacts or qualification.
    # This primitive schema contains no model/field hooks or external callbacks.
    # Using the same engine preserves both strict terminal anchors and Unicode \d.
    return SchemaValidator(
        core_schema.str_schema(strict=True, pattern=pattern, regex_engine="rust-regex")
    )


def _declared_limits(value: Any, metadata: Sequence[object]) -> None:
    """Preserve existing field domains; no new timestamp or integer bound is added."""
    if value is None:
        return
    for constraint in metadata:
        kind = type(constraint)
        if kind is FieldInfo:
            _declared_limits(value, cast(FieldInfo, constraint).metadata)
            continue
        pattern = getattr(constraint, "pattern", None)
        if pattern is not None:
            _declared_pattern_validator(pattern).validate_python(value)
        invalid = (
            (kind is MaxLen and len(value) > cast(MaxLen, constraint).max_length)
            or (kind is MinLen and len(value) < cast(MinLen, constraint).min_length)
            or (kind is Ge and value < cast(Ge, constraint).ge)
            or (kind is Gt and value <= cast(Gt, constraint).gt)
            or (kind is Le and value > cast(Le, constraint).le)
            or (kind is Lt and value >= cast(Lt, constraint).lt)
        )
        if invalid:
            raise ValueError("audit field exceeds its existing declared domain")


def _declared_annotation_limits(value: Any, annotation: object) -> None:
    """Inspect container-element domains after the whole typed tree is exact.

    Nested models are handled by the enclosing tree walk, not revalidated here.
    Union selection remains structural and never probes a wrong branch's domain.
    """
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        _declared_limits(value, args[1:])
        _declared_annotation_limits(value, args[0])
    elif origin is UnionType or origin is Union:
        for branch in args:
            if _exact(value, branch):
                _declared_annotation_limits(value, branch)
                break
    elif origin is tuple or origin is list:
        for index, item in enumerate(value):
            expected = args[0] if origin is list or args[-1] is Ellipsis else args[index]
            _declared_annotation_limits(item, expected)
    elif origin is dict:
        for key, item in value.items():
            _declared_annotation_limits(key, args[0])
            _declared_annotation_limits(item, args[1])


def _collection_limits(value: object) -> None:
    """Check IDs and published graph/ranking capacities in detached plain state."""
    stack = [value]
    order_fields = {**_EVIDENCE_ORDER, **_RANKING_ORDER, "rules": "rule_key"}
    while stack:
        current = stack.pop()
        if type(current) is list:
            stack.extend(cast(list[object], current))
        elif type(current) is dict:
            state = cast(dict[str, Any], current)
            for name in ("candidates", "candidate_scores", "scores"):
                rows = state.get(name)
                if type(rows) is list and len(rows) > 4096:
                    raise ValueError("audit candidate records exceed the existing C capacity")
            compilation = state.get("role_compilation")
            if (
                type(compilation) is dict
                and compilation.get("schema_version")
                == RoleCompilationResult.model_fields["schema_version"].default
            ):
                _accepted_compiler_pool_limits(state)
            for name, id_field in order_fields.items():
                rows = state.get(name)
                if type(rows) is list:
                    ids = [row[id_field] for row in rows if type(row) is dict and id_field in row]
                    if len(ids) == len(rows) and len(ids) != len(set(ids)):
                        raise ValueError("audit requires unique retained record IDs")
                    if (
                        name
                        in (
                            "graph_snapshots",
                            "condensations",
                            "onsets",
                            "precedences",
                            "rule_assignments",
                            "rule_predicates",
                            "rules",
                        )
                        and len(rows) > 32768
                    ):
                        raise ValueError("audit record sequence exceeds module capacity")
            for name, key in (
                ("rules", "rule_sha256"),
                ("rule_assignments", "assignment_sha256"),
                ("rule_predicates", "predicate_sha256"),
                ("candidates", "candidate_service_key"),
                ("candidates", "service_entity_key"),
                ("candidate_scores", "candidate_service_key"),
                ("scores", "candidate_service_key"),
            ):
                rows = state.get(name)
                if type(rows) is list and all(type(row) is dict and key in row for row in rows):
                    ids = [row[key] for row in rows]
                    if len(ids) != len(set(ids)):
                        raise ValueError("audit requires unique rule and candidate record IDs")
            if "rule_predicates" in state:
                _plain_capacity(state["rule_predicates"], narrow=True)
            schema = state.get("schema_version", "")
            if schema == TopKArtifact.model_fields["schema_version"].default:
                _top_k_limits(state)
            for name in ("node_service_keys", "ordered_service_keys", "candidate_service_keys"):
                keys = state.get(name)
                if (
                    type(keys) is list
                    and all(type(key) is str for key in keys)
                    and (len(keys) > 4096 or len(set(keys)) != len(keys))
                ):
                    raise ValueError("audit candidate/node keys exceed capacity or repeat")
            stack.extend(state.values())


def _accepted_compiler_pool_limits(value: dict[str, Any]) -> None:
    # These are the original compiler's consumed-pool limits. NoRole does not
    # consume these pools, and all rejected guards remain retained on both paths.
    for name, maximum in (("role_predicates", 4096), ("role_proposals", 16384)):
        rows = value.get(name)
        if type(rows) is list and len(rows) > maximum:
            raise ValueError("audit accepted compiler pool exceeds its existing capacity")


def _top_k_limits(value: dict[str, Any]) -> None:
    # These are the original reference-ranking / N4 domains, not new limits on
    # generic source integers or timestamps.
    _no_ranking._bounded_count(value.get("k"), 100)
    _no_ranking._bounded_count(value.get("total_candidate_count"), 4096)
    entries = value.get("entries")
    if type(entries) is not list or len(entries) > 100:
        raise ValueError("audit TopK exceeds the reference entry capacity")
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("audit TopK entries must be records")
        _no_ranking._bounded_count(entry.get("rank"), 4096)
    for field in ("rank", "service_entity_key"):
        ids = [row.get(field) for row in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("audit TopK entries must have unique ranks and candidates")


def _plain_capacity(value: object, *, max_nodes: int = _MAX_NODES, narrow: bool = False) -> None:
    """Apply a module's published budget to an already inspected JSON subtree."""
    pending = [value]
    nodes = 1
    provenance = 0
    while pending:
        current = pending.pop()
        if type(current) is str and narrow and len(current) > 8192:
            raise ValueError("audit exceeds the module string capacity")
        if type(current) is dict:
            state = cast(dict[str, object], current)
            children = list(state.values())
            if narrow:
                for name, child in state.items():
                    if (
                        type(name) is str
                        and name.endswith("provenance_ids")
                        and type(child) is list
                    ):
                        provenance += len(cast(list[object], child))
                        if provenance > 131072:
                            raise ValueError("audit exceeds the module provenance capacity")
        elif type(current) is list:
            children = cast(list[object], current)
        else:
            continue
        nodes += len(children)
        if nodes > max_nodes:
            raise ValueError("audit exceeds the module node capacity")
        pending.extend(children)


def _json_model_capacity(value: dict[str, Any], annotation: type[BaseModel]) -> None:
    _audit_declared_domains(value, annotation)
    # Only a declared model count has numeric meaning; free artifact hash maps
    # may legitimately use "candidate_count" (or "schema_version") as a key.
    if (
        "candidate_count" in annotation.model_fields
        and "candidate_count" in value
        and annotation is not EmptyCandidateUniverseFailure
    ):
        _no_ranking._bounded_count(value["candidate_count"], 4096)
    if annotation is ReferenceRankingResult or annotation is NoRoleReferenceRankingResult:
        _no_ranking._result_limits(value)
    if annotation is CaseEvidenceReplayBundleV2 and _json_shape(
        value.get("role_compilation"), RoleCompilationResult, domains=False
    ):
        # A declared schema default may be omitted in JSON. Resolve only the exact
        # structural branch here; actual scoring eligibility is still replayed.
        _accepted_compiler_pool_limits(value)
    if annotation is TopKArtifact:
        _top_k_limits(value)
    if annotation is PredicateResult:
        pending = [value]
        count = 1
        while pending:
            operands = pending.pop().get("operand_states", [])
            if type(operands) is not list or any(type(row) is not dict for row in operands):
                raise ValueError("audit JSON predicates require object arrays")
            count += len(operands)
            if count > 4096:
                raise ValueError("audit JSON predicate exceeds per-tree record capacity")
            pending.extend(operands)
    if annotation is CaseQuerySessionResult:
        _plain_capacity(value, max_nodes=20_000)
    if (
        annotation is GraphSnapshot
        or annotation is RelationSCCCondensation
        or annotation is OnsetInterval
        or annotation is TemporalPrecedence
        or annotation is RuleSet
        or annotation is RuleCandidateAssignment
        or annotation is RuleScoringResult
        or annotation is HistoryEvaluationResult
    ):
        _plain_capacity(value, narrow=True)
    if annotation is GraphSnapshot or annotation is RelationSCCCondensation:
        edge_field = "edges" if annotation is GraphSnapshot else "selected_original_edges"
        edges = value.get(edge_field)
        if type(edges) is list and len(edges) > 32768:
            raise ValueError("audit JSON graph exceeds retained-edge capacity")


def _json_shape(value: object, annotation: object, *, domains: bool = True) -> bool:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        if not _json_shape(value, args[0], domains=domains):
            return False
        if domains:
            _declared_limits(value, args[1:])
        return True
    if origin is UnionType or origin is Union:
        # Match the entire structural branch first. A different branch's field
        # domain must not decide validity based on JSON object key order.
        for branch in args:
            if _json_shape(value, branch, domains=False):
                return not domains or _json_shape(value, branch)
        return False
    if origin is tuple:
        if type(value) is not list:
            return False
        rows = cast(list[object], value)
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_json_shape(row, args[0], domains=domains) for row in rows)
        return len(rows) == len(args) and all(
            _json_shape(row, expected, domains=domains)
            for row, expected in zip(rows, args, strict=True)
        )
    if origin is dict:
        return type(value) is dict and all(
            _exact(key, args[0]) and _json_shape(child, args[1], domains=domains)
            for key, child in cast(dict[object, object], value).items()
        )
    if annotation is RankingInputAuthority:
        return (
            type(value) is dict
            and set(cast(dict[str, object], value)) == set(_AUTHORITY_TYPES)
            and all(
                _exact(child, _AUTHORITY_TYPES[name])
                for name, child in cast(dict[str, object], value).items()
            )
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if type(value) is not dict:
            return False
        if any(
            field.is_required() and name not in value
            for name, field in annotation.model_fields.items()
        ):
            return False
        if domains:
            _json_model_capacity(cast(dict[str, Any], value), annotation)
        for name, child in cast(dict[str, object], value).items():
            if name not in annotation.model_fields or not _json_shape(
                child, annotation.model_fields[name].annotation, domains=domains
            ):
                return False
            if domains:
                _declared_limits(child, annotation.model_fields[name].metadata)
        return True
    return _exact(value, annotation)


def _json_with_defaults(value: Any, annotation: object) -> Any:
    """Expand only declared constant defaults in a detached, shape-checked tree.

    Keep unchanged subtrees and every existing scalar representation untouched.
    No constructors, default factories, encoders or validators run here.
    """
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _json_with_defaults(value, args[0])
    if origin is UnionType or origin is Union:
        for branch in args:
            if _json_shape(value, branch, domains=False):
                return _json_with_defaults(value, branch)
        raise ValueError("audit JSON has no exact declared branch")
    if origin is tuple:
        changed_rows: list[Any] | None = None
        for index, child in enumerate(value):
            item_type = args[0] if len(args) == 2 and args[1] is Ellipsis else args[index]
            expanded = _json_with_defaults(child, item_type)
            if expanded is not child:
                if changed_rows is None:
                    changed_rows = list(value)
                changed_rows[index] = expanded
        return value if changed_rows is None else changed_rows
    is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
    if origin is dict or is_model:
        model_type = cast(type[BaseModel], annotation)
        changed_fields: dict[str, Any] | None = None
        for name, child in value.items():
            item_type = model_type.model_fields[name].annotation if is_model else args[1]
            expanded = _json_with_defaults(child, item_type)
            if expanded is not child:
                if changed_fields is None:
                    changed_fields = dict(value)
                changed_fields[name] = expanded
        if is_model:
            for name, field in model_type.model_fields.items():
                if name not in value and not field.is_required() and field.default_factory is None:
                    if changed_fields is None:
                        changed_fields = dict(value)
                    changed_fields[name] = _inspect(field.default)[0]
        return value if changed_fields is None else changed_fields
    return value


def _validate_json(
    cls: type[_ModelT], raw_input: str | bytes | bytearray, **kwargs: Any
) -> _ModelT:
    if cls is not CaseEvidenceReplayBundleV2 and cls is not CaseRankingReplayBundleV2:
        raise ValueError("audit JSON requires an exact bundle class")
    raw = _no_roles._json_bytes(raw_input)
    try:
        _no_roles._scan_json(raw)
        plain = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_roles._unique_object,
            parse_float=_no_evidence._finite_json_float,
            parse_constant=_no_roles._reject_json_number,
        )
        _inspect(plain)
        if not _json_shape(plain, cls, domains=False):
            raise ValueError("audit JSON requires exact field representations")
        expanded = _json_with_defaults(plain, cls)
        if expanded is not plain:
            _inspect(expanded)
        if not _json_shape(expanded, cls):
            raise ValueError("audit JSON requires exact declared domains")
        validated_raw = raw if expanded is plain else canonical_json_bytes(expanded)
        return super(cls, cls).model_validate_json(validated_raw, **kwargs)
    finally:
        if raw != _no_roles._json_bytes(raw_input):
            raise ValueError("audit JSON input changed during validation")


def _capture(value: object) -> tuple[bytes, tuple[object, ...]]:
    plain, shape = _inspect(value)
    return canonical_json_bytes(plain), shape


def _unchanged(value: object, original: tuple[object, tuple[object, ...]]) -> None:
    try:
        # Compare detached plain content and exact representations without an
        # encoder after the final live snapshot, including signed zero/markers.
        current = _inspect(value)
        unchanged = current == original
    except _ERRORS:
        unchanged = False
    if not unchanged:
        raise ValueError("audit original input bytes or representation changed during replay")


def _fresh(value: _ModelT, model_type: type[_ModelT], original: bytes) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError("audit requires an exact bundle root")
    rebuilt = model_type.model_validate_json(original)
    if not hmac.compare_digest(_capture(rebuilt)[0], original):
        raise ValueError("audit schema roundtrip changed content")
    return rebuilt


def _hash_closes(value: StrictFrozenModel, field: str) -> bool:
    plain, _ = _inspect(value)
    unsigned = cast(dict[str, object], plain)
    stored = cast(str, unsigned.pop(field))
    return hmac.compare_digest(stored, sha256_hex(canonical_json_bytes(unsigned)))


def _safe_state(value: object) -> object:
    _inspect(value)
    return value


def _exact_field(cls: type[BaseModel], value: object, info: ValidationInfo) -> object:
    # JSON has no tuples or model instances. Python inputs do, and must retain them.
    if info.mode != "json" and not _exact(
        value, cls.model_fields[cast(str, info.field_name)].annotation
    ):
        raise ValueError("audit Python fields require exact declared types")
    return value


def _ordered(value: BaseModel, orders: dict[str, str], *, unique: tuple[str, ...]) -> None:
    for name, key in orders.items():
        records = getattr(value, name)
        ids = tuple(getattr(item, key) for item in records)
        if ids != tuple(sorted(ids)) or (name in unique and len(set(ids)) != len(ids)):
            raise ValueError(f"audit {name} requires canonical ordering and unique keys")


class CaseEvidenceReplayBundleV2(StrictFrozenModel):
    """Complete original evidence plus the pinned accepted-only scoring receipt."""

    schema_version: Literal["nurse.v2.case-evidence-replay-bundle/2"] = (
        CASE_EVIDENCE_REPLAY_BUNDLE_SCHEMA_VERSION
    )
    time_contract: CaseTimeContract
    entity_registry: TemporalEntityRegistry
    event_grouping: EventGroupingResult
    candidate_universe: CandidateUniverse
    frozen_candidates: FrozenCandidateUniverse
    focus_set: SemanticFocusSet
    semantic_context: SemanticContextBundle | None
    card_expansion_policy: CardExpansionPolicy | None
    query_sessions: tuple[CaseQuerySessionResult, ...]
    mechanism_library: MechanismLibrary
    scope_interval: TimeInterval
    role_predicates: tuple[PredicateResult, ...]
    role_guard_results: tuple[HypothesisGuardResult, ...]
    role_proposals: tuple[RoleBindingProposal, ...]
    role_compilation: RoleCompilationResult | NoRoleCompilationResult
    event_strength: EventStrengthResult | NoRoleEventStrengthResult
    graph_snapshots: tuple[GraphSnapshot, ...] = Field(min_length=1)
    condensations: tuple[RelationSCCCondensation, ...] = Field(min_length=1)
    onsets: tuple[OnsetInterval, ...]
    precedences: tuple[TemporalPrecedence, ...]
    scoring_selection: CaseScoringSelection
    scoring_path: Literal["accepted_roles", "no_eligible_roles"]
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return _validate_json(
            cls, json_data, strict=strict, context=context, by_alias=by_alias, by_name=by_name
        )

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        return _safe_state(value)

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        return _exact_field(cls, value, info)

    @model_validator(mode="after")
    def canonical_sequences(self) -> Self:
        _ordered(self, _EVIDENCE_ORDER, unique=("query_sessions",))
        accepted = self.scoring_path == "accepted_roles"
        if (
            accepted != bool(self.scoring_selection.eligible_guard_result_ids)
            or type(self.role_compilation)
            is not (RoleCompilationResult if accepted else NoRoleCompilationResult)
            or type(self.event_strength)
            is not (EventStrengthResult if accepted else NoRoleEventStrengthResult)
            or (not accepted and self.role_proposals != ())
        ):
            raise ValueError("audit scoring path and exact evidence branch differ")
        return self

    def verify_inputs(self) -> bool:
        """Check internal closure against a fresh copy of every retained source."""
        try:
            if type(self) is not CaseEvidenceReplayBundleV2:
                return False
            original = _inspect(self)
            try:
                rebuilt = _fresh(
                    self, CaseEvidenceReplayBundleV2, canonical_json_bytes(original[0])
                )
                if not _hash_closes(rebuilt, "evidence_sha256"):
                    return False
                scoring = _selection_replay(rebuilt)
                common = _role_inputs(rebuilt, scoring)
                return _evidence_closes(rebuilt, common)
            finally:
                _unchanged(self, original)
        except _ERRORS:
            return False


def _selection_replay(evidence: CaseEvidenceReplayBundleV2) -> ReplayedCaseScoringInputs:
    return _require_selection(_source_inputs(evidence), evidence.scoring_selection)


def _source_inputs(evidence: CaseEvidenceReplayBundleV2) -> CaseGuardReplayInputs:
    return CaseGuardReplayInputs(
        **{f.name: object.__getattribute__(evidence, f.name) for f in fields(CaseGuardReplayInputs)}
    )


def _require_selection(
    inputs: CaseGuardReplayInputs, selection: CaseScoringSelection
) -> ReplayedCaseScoringInputs:
    # The source pin is computed from independently retained evidence, never copied
    # from the selection receipt's claim. Both values are fixed before replay.
    source_pin = case_guard_replay_inputs_sha256(inputs=inputs)
    selection_pin = _content_pin(selection, "selection_sha256")
    return require_case_scoring_selection(
        selection=selection,
        expected_selection_sha256=selection_pin,
        inputs=inputs,
        expected_source_records_sha256=source_pin,
    )


def _content_pin(value: BaseModel, *excluded: str) -> str:
    plain = cast(dict[str, object], _inspect(value)[0])
    return sha256_hex(
        canonical_json_bytes({key: child for key, child in plain.items() if key not in excluded})
    )


def _no_role_inputs(evidence: CaseEvidenceReplayBundleV2) -> dict[str, Any]:
    inputs = _source_inputs(evidence)
    return dict(
        inputs=inputs,
        selection=evidence.scoring_selection,
        expected_source_records_sha256=case_guard_replay_inputs_sha256(inputs=inputs),
        expected_selection_sha256=_content_pin(evidence.scoring_selection, "selection_sha256"),
    )


def _role_inputs(
    evidence: CaseEvidenceReplayBundleV2, scoring: ReplayedCaseScoringInputs
) -> dict[str, Any]:
    mapping, coordinator = scoring.fresh_query_runtime()
    return dict(
        proposals=evidence.role_proposals,
        guard_results=scoring.eligible_guards,
        predicates=scoring.predicates,
        candidate_universe=evidence.candidate_universe,
        frozen_candidates=evidence.frozen_candidates,
        entity_registry=evidence.entity_registry,
        event_grouping=evidence.event_grouping,
        mechanism_library=evidence.mechanism_library,
        scope_interval=evidence.scope_interval,
        query_sessions_by_guard_result_id=mapping,
        query_session_coordinator=coordinator,
        semantic_context=evidence.semantic_context,
        card_expansion_policy=evidence.card_expansion_policy,
    )


def _evidence_closes(evidence: CaseEvidenceReplayBundleV2, common: dict[str, Any]) -> bool:
    # Selection already replays all thirteen source fields and every guard,
    # including excluded guards and their query provenance.
    case_id = evidence.time_contract.opaque_case_id
    if any(
        value.opaque_case_id != case_id
        for value in (
            evidence.role_compilation,
            evidence.event_strength,
            *evidence.graph_snapshots,
            *evidence.condensations,
        )
    ):
        return False
    if evidence.scoring_path == "accepted_roles":
        accepted_roles = cast(RoleCompilationResult, evidence.role_compilation)
        accepted_event = cast(EventStrengthResult, evidence.event_strength)
        if not accepted_roles.verify_inputs(**common):
            return False
        if not accepted_event.verify_inputs(role_compilation=accepted_roles, **common):
            return False
    else:
        if (
            common["guard_results"]
            or common["query_sessions_by_guard_result_id"]
            or evidence.role_proposals
        ):
            return False
        inputs = _no_role_inputs(evidence)
        roles = _no_roles.build_no_role_compilation(**inputs)
        event = _no_evidence.compile_no_role_event_strengths(
            **inputs,
            role_compilation=roles,
            expected_role_compilation_sha256=roles.result_sha256,
        )
        if (
            _capture(roles)[0] != _capture(evidence.role_compilation)[0]
            or _capture(event)[0] != _capture(evidence.event_strength)[0]
        ):
            return False
    graphs = {graph.snapshot_sha256: graph for graph in evidence.graph_snapshots}
    if len(graphs) != len(evidence.graph_snapshots):
        return False
    for graph in evidence.graph_snapshots:
        window = (
            evidence.time_contract.baseline_window
            if graph.window_kind == "normal"
            else evidence.time_contract.fault_window
        )
        if not graph.verify() or (
            graph.interval != window
            or graph.decision_time_unix_ns != evidence.time_contract.decision_time_unix_ns
            or graph.observation_contract_sha256
            != evidence.candidate_universe.observation_contract_sha256
            or graph.candidate_universe_sha256 != evidence.candidate_universe.universe_sha256
            or graph.candidate_freeze_sha256 != evidence.frozen_candidates.freeze_sha256
            or graph.entity_registry_sha256 != evidence.entity_registry.registry_sha256
            or graph.node_service_keys != evidence.frozen_candidates.candidate_service_keys
        ):
            return False
    relations = tuple(row.relation_type for row in evidence.condensations)
    if len(relations) != len(set(relations)):
        return False
    for condensation in evidence.condensations:
        selected_graph = graphs.get(condensation.graph_snapshot_sha256)
        if selected_graph is None or not condensation.verify_inputs(graph_snapshot=selected_graph):
            return False
    return _temporal_closes(evidence)


def _temporal_closes(evidence: CaseEvidenceReplayBundleV2) -> bool:
    """Retained-record consistency only: no absent raw clock source is invented."""
    contract = evidence.time_contract
    case_id = contract.opaque_case_id
    contract_sha = _temporal.case_time_contract_sha256(contract)
    by_id = {row.onset_id: row for row in evidence.onsets}
    if len(by_id) != len(evidence.onsets):
        return False
    for onset in evidence.onsets:
        if not OnsetInterval.verify(onset) or (
            onset.opaque_case_id != case_id
            or onset.time_contract_sha256 != contract_sha
            or onset.service_key not in evidence.frozen_candidates.candidate_service_keys
            or onset.t_low_unix_ns < contract.baseline_window.start_unix_ns
            or onset.t_high_unix_ns > contract.decision_time_unix_ns
        ):
            return False
    pairs: set[tuple[str, str]] = set()
    for precedence in evidence.precedences:
        if not TemporalPrecedence.verify(precedence) or (
            precedence.opaque_case_id != case_id
            or precedence.time_contract_sha256 != contract_sha
            or precedence.clock_uncertainty_ns != contract.clock_uncertainty_ns
        ):
            return False
        references = (precedence.source_onset_id, precedence.target_onset_id)
        if any(key is not None and key not in by_id for key in references):
            return False
        if precedence.source_onset_id is not None and precedence.target_onset_id is not None:
            pair = (precedence.source_onset_id, precedence.target_onset_id)
            if pair in pairs or precedence.reason == "onset_outside_time_contract":
                return False
            pairs.add(pair)
            if precedence.clock_uncertainty_resolved:
                source, target = (by_id[key] for key in pair)
                uncertainty = contract.clock_uncertainty_ns
                if (
                    precedence.source_effective_low_unix_ns,
                    precedence.source_effective_high_unix_ns,
                    precedence.target_effective_low_unix_ns,
                    precedence.target_effective_high_unix_ns,
                ) != (
                    max(0, source.t_low_unix_ns - uncertainty),
                    source.t_high_unix_ns + uncertainty,
                    max(0, target.t_low_unix_ns - uncertainty),
                    target.t_high_unix_ns + uncertainty,
                ):
                    return False
        provenance = {contract_sha, precedence.clock_resolution_sha256}
        for key in references:
            if key is not None:
                provenance.update(by_id[key].provenance_ids)
        if not provenance <= set(precedence.provenance_ids):
            return False
    return True


class CaseRankingReplayBundleV2(StrictFrozenModel):
    """Original E/P/R/H channels and exact full ranking over complete /2 evidence."""

    schema_version: Literal["nurse.v2.case-ranking-replay-bundle/2"] = (
        CASE_RANKING_REPLAY_BUNDLE_SCHEMA_VERSION
    )
    evidence: CaseEvidenceReplayBundleV2
    family_aggregation: FamilyAggregationResult | NoRoleFamilyAggregationResult
    propagation_scoring: PropagationScoringResult | NoRolePropagationScoringResult
    rule_set: RuleSet
    rule_assignments: tuple[RuleCandidateAssignment, ...]
    rule_predicates: tuple[PredicateResult, ...]
    rule_scoring: RuleScoringResult
    history_mode: HistoryMode
    history_evaluation: HistoryEvaluationResult
    ranking_authority: RankingInputAuthority
    scoring_config: ReferenceScoringConfig
    ranking_result: ReferenceRankingResult | NoRoleReferenceRankingResult
    top_k: TopKArtifact
    semantic_calls: tuple[SemanticCall, ...]
    attempt_ledgers: tuple[AttemptLedger, ...]
    costs: tuple[CostRecord, ...]
    checkpoints: tuple[CaseCheckpoint, ...]
    degradation_reason_codes: tuple[str, ...]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return _validate_json(
            cls, json_data, strict=strict, context=context, by_alias=by_alias, by_name=by_name
        )

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        return _safe_state(value)

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        return _exact_field(cls, value, info)

    @model_validator(mode="after")
    def canonical_runtime(self) -> Self:
        accepted = self.evidence.scoring_path == "accepted_roles"
        for value, expected in (
            (
                self.family_aggregation,
                FamilyAggregationResult if accepted else NoRoleFamilyAggregationResult,
            ),
            (
                self.propagation_scoring,
                PropagationScoringResult if accepted else NoRolePropagationScoringResult,
            ),
            (
                self.ranking_result,
                ReferenceRankingResult if accepted else NoRoleReferenceRankingResult,
            ),
        ):
            if type(value) is not expected:
                raise ValueError("audit ranking channels must follow the exact evidence path")
        if self.history_mode != "off":
            raise ValueError("case audit /2 requires replayable history_mode=off")
        if self.degradation_reason_codes != tuple(sorted(set(self.degradation_reason_codes))):
            raise ValueError("degradation reason codes must be sorted and unique")
        _ordered(
            self,
            _RANKING_ORDER,
            unique=("semantic_calls", "attempt_ledgers", "costs", "checkpoints"),
        )
        return self

    def verify_inputs(self) -> bool:
        try:
            if type(self) is not CaseRankingReplayBundleV2:
                return False
            original = _inspect(self)
            try:
                rebuilt = _fresh(self, CaseRankingReplayBundleV2, canonical_json_bytes(original[0]))
                if not _hash_closes(rebuilt, "bundle_sha256") or not _hash_closes(
                    rebuilt.evidence, "evidence_sha256"
                ):
                    return False
                scoring = _selection_replay(rebuilt.evidence)
                common = _role_inputs(rebuilt.evidence, scoring)
                return (
                    _evidence_closes(rebuilt.evidence, common)
                    and _ranking_closes(rebuilt, common)
                    and _runtime_closes(rebuilt)
                )
            finally:
                _unchanged(self, original)
        except _ERRORS:
            return False


def _ranking_closes(bundle: CaseRankingReplayBundleV2, common: dict[str, Any]) -> bool:
    evidence = bundle.evidence
    if evidence.scoring_path == "no_eligible_roles":
        return _no_role_ranking_closes(bundle)
    family = cast(FamilyAggregationResult, bundle.family_aggregation)
    propagation = cast(PropagationScoringResult, bundle.propagation_scoring)
    roles = cast(RoleCompilationResult, evidence.role_compilation)
    if not family.verify_inputs(
        event_strength=cast(EventStrengthResult, evidence.event_strength),
        aggregator_version=bundle.family_aggregation.aggregator_version,
        role_compilation=roles,
        **common,
    ):
        return False
    propagation_inputs = dict(common)
    for old, new in (
        ("proposals", "role_proposals"),
        ("guard_results", "role_guard_results"),
        ("predicates", "role_predicates"),
    ):
        propagation_inputs[new] = propagation_inputs.pop(old)
    if not propagation.verify_inputs(
        expected_result_sha256=bundle.propagation_scoring.result_sha256,
        expected_authoritative_input_sha256=bundle.propagation_scoring.authoritative_input_sha256,
        condensations=evidence.condensations,
        onsets=evidence.onsets,
        precedences=evidence.precedences,
        role_compilation=roles,
        config=bundle.propagation_scoring.config,
        **propagation_inputs,
    ):
        return False
    if not bundle.rule_scoring.verify_trusted(
        expected_result_sha256=bundle.rule_scoring.result_sha256,
        expected_authoritative_input_sha256=bundle.rule_scoring.authoritative_input_sha256,
        frozen_candidates=evidence.frozen_candidates,
        rule_set=bundle.rule_set,
        assignments=bundle.rule_assignments,
        predicates=bundle.rule_predicates,
    ):
        return False
    if not bundle.history_evaluation.verify_trusted(
        expected_result_sha256=bundle.history_evaluation.result_sha256,
        expected_authoritative_input_sha256=bundle.history_evaluation.authoritative_input_sha256,
        frozen_candidates=evidence.frozen_candidates,
        mode=bundle.history_mode,
    ):
        return False
    if not (
        bundle.family_aggregation.verify()
        and bundle.propagation_scoring.verify_structure()
        and bundle.rule_set.verify()
        and bundle.rule_scoring.verify_structure()
        and bundle.history_evaluation.verify_structure()
        and bundle.scoring_config.verify()
        and bundle.ranking_result.verify_structure()
        and bundle.top_k.verify()
    ):
        return False
    recomputed = rank_reference_candidates(
        frozen_candidates=evidence.frozen_candidates,
        family_aggregation=family,
        propagation_scoring=propagation,
        rule_scoring=bundle.rule_scoring,
        history_evaluation=bundle.history_evaluation,
        authority=bundle.ranking_authority,
        config=bundle.scoring_config,
    )
    # All candidates, quantization, ties and TopK must close, including the tail.
    return (
        _capture(recomputed)[0] == _capture(bundle.ranking_result)[0]
        and _capture(bundle.top_k)[0] == _capture(recomputed.top_k)[0]
    )


def _no_role_ranking_closes(bundle: CaseRankingReplayBundleV2) -> bool:
    evidence = bundle.evidence
    inputs = _no_role_inputs(evidence)
    common = dict(
        **inputs,
        role_compilation=evidence.role_compilation,
        expected_role_compilation_sha256=_content_pin(
            evidence.role_compilation, "role_compilation_id", "result_sha256"
        ),
        graph_snapshots=evidence.graph_snapshots,
        condensations=evidence.condensations,
        onsets=evidence.onsets,
        precedences=evidence.precedences,
    )
    propagation_pin = _no_propagation.no_role_propagation_input_sha256(
        **common,
        config=bundle.propagation_scoring.config,
    )
    if bundle.ranking_authority.propagation_authoritative_input_sha256 != propagation_pin:
        return False
    args = dict(
        **common,
        event_strength=evidence.event_strength,
        expected_event_strength_sha256=_content_pin(
            evidence.event_strength, "event_strength_result_id", "result_sha256"
        ),
        family_aggregation=bundle.family_aggregation,
        expected_family_aggregation_sha256=_content_pin(
            bundle.family_aggregation, "family_aggregation_result_id", "result_sha256"
        ),
        aggregator_version=bundle.family_aggregation.aggregator_version,
        propagation_config=bundle.propagation_scoring.config,
        propagation_scoring=bundle.propagation_scoring,
        expected_propagation_sha256=_content_pin(
            bundle.propagation_scoring, "scoring_result_id", "result_sha256"
        ),
        rule_set=bundle.rule_set,
        rule_assignments=bundle.rule_assignments,
        rule_predicates=bundle.rule_predicates,
        rule_scoring=bundle.rule_scoring,
        history_evaluation=bundle.history_evaluation,
        history_mode=bundle.history_mode,
        authority=bundle.ranking_authority,
        config=bundle.scoring_config,
    )
    ranking_pin = _no_ranking.no_role_ranking_input_sha256(**args)
    recomputed = _no_ranking.rank_no_role_reference_candidates(
        **args,
        expected_authoritative_input_sha256=ranking_pin,
    )
    return (
        _capture(recomputed)[0] == _capture(bundle.ranking_result)[0]
        and _capture(recomputed.top_k)[0] == _capture(bundle.top_k)[0]
    )


def _runtime_closes(bundle: CaseRankingReplayBundleV2) -> bool:
    case_id = bundle.evidence.time_contract.opaque_case_id
    for records in (
        bundle.semantic_calls,
        bundle.attempt_ledgers,
        bundle.costs,
        bundle.checkpoints,
    ):
        if any(not record.verify() for record in records):
            return False
    if any(item.opaque_case_id != case_id for item in bundle.semantic_calls) or any(
        item.opaque_case_id != case_id for item in bundle.checkpoints
    ):
        return False
    ledger_ids = tuple(item.attempt_ledger.logical_request_id for item in bundle.semantic_calls)
    if len(ledger_ids) != len(set(ledger_ids)):
        return False
    ledgers = tuple(
        sorted(
            (call.attempt_ledger for call in bundle.semantic_calls),
            key=lambda item: item.logical_request_id,
        )
    )
    costs = {call.cost.cost_sha256: call.cost for call in bundle.semantic_calls}
    if bundle.attempt_ledgers != ledgers or bundle.costs != tuple(
        costs[key] for key in sorted(costs)
    ):
        return False
    return all(
        set(checkpoint.logical_request_ids) <= set(ledger_ids) for checkpoint in bundle.checkpoints
    )


def _build_bundle(
    values: dict[str, object], model_type: type[_ModelT], *, hash_field: str, orders: dict[str, str]
) -> _ModelT:
    _inspect(values)
    for name, value in values.items():
        annotation = model_type.model_fields[name].annotation
        if get_origin(annotation) is tuple:
            if (type(value) is not tuple and type(value) is not list) or any(
                not _exact(item, get_args(annotation)[0]) for item in cast(Sequence[object], value)
            ):
                raise ValueError(f"audit builder {name} requires exact tuple or list items")
        elif not _exact(value, annotation):
            raise ValueError(f"audit builder {name} requires the exact declared root")
        _declared_limits(value, model_type.model_fields[name].metadata)
    original = _inspect(values)
    try:
        # Only preflighted builtin sequences can reach sorting. Their original
        # bytes and representations above remain part of the finally check.
        ordered = dict(values)
        for name, key in orders.items():
            ordered[name] = tuple(
                sorted(cast(Sequence[object], values[name]), key=lambda item: getattr(item, key))
            )
        if model_type is CaseEvidenceReplayBundleV2:
            inputs = CaseGuardReplayInputs(
                **cast(
                    dict[str, Any], {f.name: ordered[f.name] for f in fields(CaseGuardReplayInputs)}
                )
            )
            scoring = _require_selection(
                inputs, cast(CaseScoringSelection, ordered["scoring_selection"])
            )
            ordered["scoring_selection"] = scoring.selection
            ordered["scoring_path"] = (
                "accepted_roles" if scoring.eligible_guards else "no_eligible_roles"
            )
        if "degradation_reason_codes" in values:
            codes = cast(Sequence[str], values["degradation_reason_codes"])
            if any(not code for code in codes):
                raise ValueError("degradation reason codes must be nonempty strings")
            ordered["degradation_reason_codes"] = tuple(sorted(set(codes)))
        ordered["schema_version"] = model_type.model_fields["schema_version"].default
        unsigned = _capture(ordered)[0]
        ordered[hash_field] = sha256_hex(unsigned)
        result = model_type.model_validate_json(_capture(ordered)[0])
        if not cast(Any, result).verify_inputs():
            raise ValueError("case audit /2 failed input closure verification")
        return result
    finally:
        _unchanged(values, original)


def build_case_evidence_replay_bundle_v2(
    *,
    time_contract: CaseTimeContract,
    entity_registry: TemporalEntityRegistry,
    event_grouping: EventGroupingResult,
    candidate_universe: CandidateUniverse,
    frozen_candidates: FrozenCandidateUniverse,
    focus_set: SemanticFocusSet,
    semantic_context: SemanticContextBundle | None,
    card_expansion_policy: CardExpansionPolicy | None,
    query_sessions: Sequence[CaseQuerySessionResult],
    mechanism_library: MechanismLibrary,
    scope_interval: TimeInterval,
    role_predicates: Sequence[PredicateResult],
    role_guard_results: Sequence[HypothesisGuardResult],
    role_proposals: Sequence[RoleBindingProposal],
    role_compilation: RoleCompilationResult | NoRoleCompilationResult,
    event_strength: EventStrengthResult | NoRoleEventStrengthResult,
    graph_snapshots: Sequence[GraphSnapshot],
    condensations: Sequence[RelationSCCCondensation],
    onsets: Sequence[OnsetInterval],
    precedences: Sequence[TemporalPrecedence],
    scoring_selection: CaseScoringSelection,
) -> CaseEvidenceReplayBundleV2:
    return _build_bundle(
        locals(), CaseEvidenceReplayBundleV2, hash_field="evidence_sha256", orders=_EVIDENCE_ORDER
    )


def build_case_ranking_replay_bundle_v2(
    *,
    evidence: CaseEvidenceReplayBundleV2,
    family_aggregation: FamilyAggregationResult | NoRoleFamilyAggregationResult,
    propagation_scoring: PropagationScoringResult | NoRolePropagationScoringResult,
    rule_set: RuleSet,
    rule_assignments: Sequence[RuleCandidateAssignment],
    rule_predicates: Sequence[PredicateResult],
    rule_scoring: RuleScoringResult,
    history_mode: HistoryMode,
    history_evaluation: HistoryEvaluationResult,
    ranking_authority: RankingInputAuthority,
    scoring_config: ReferenceScoringConfig,
    ranking_result: ReferenceRankingResult | NoRoleReferenceRankingResult,
    top_k: TopKArtifact,
    semantic_calls: Sequence[SemanticCall],
    attempt_ledgers: Sequence[AttemptLedger],
    costs: Sequence[CostRecord],
    checkpoints: Sequence[CaseCheckpoint],
    degradation_reason_codes: Sequence[str],
) -> CaseRankingReplayBundleV2:
    return _build_bundle(
        locals(), CaseRankingReplayBundleV2, hash_field="bundle_sha256", orders=_RANKING_ORDER
    )


IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.immutable-case-audit-package/2"] = (
    "nurse.v2.immutable-case-audit-package/2"
)


def _audit_roots() -> tuple[type[BaseModel], ...]:
    # Code-owned closed identities. No caller-supplied class participates in dispatch.
    from nurse_rcl.v2 import case_audit as legacy
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2

    return (
        ImmutableCaseAuditPackageV2,
        CaseAuditLineageIndexV2,
        SealedAuditPackage,
        legacy.ImmutableCaseAuditPackage,
        legacy.CaseRankingReplayBundle,
        legacy.CaseAuditLineageIndex,
        EmptyCandidateUniverseFailure,
    )


def _audit_snapshot(value: object) -> tuple[object, tuple[object, ...]]:
    snapshot = _inspect(value, _additional_roots=_audit_roots())
    pending = [snapshot[0]]
    while pending:
        current = pending.pop()
        if type(current) is list:
            pending.extend(cast(list[object], current))
        elif type(current) is dict:
            state = cast(dict[str, Any], current)
            if state.get("schema_version") == "nurse.v2.case-audit-lineage-index/2":
                nodes = state.get("nodes")
                if (
                    type(nodes) is not list
                    or not nodes
                    or any(type(node) is not dict for node in nodes)
                ):
                    raise ValueError("lineage requires a nonempty exact node sequence")
                keys = [node.get("record_id") for node in nodes]
                if any(type(key) is not str for key in keys) or keys != sorted(set(keys)):
                    raise ValueError("lineage nodes must be sorted and unique before encoding")
                for node in nodes:
                    for field in ("source_record_ids", "consumer_record_ids"):
                        edges = node.get(field)
                        if (
                            type(edges) is not list
                            or any(type(edge) is not str for edge in edges)
                            or edges != sorted(set(edges))
                            or node["record_id"] in edges
                        ):
                            raise ValueError(
                                "lineage edges require sorted unique nonreflexive IDs "
                                "before encoding"
                            )
            pending.extend(state.values())
    return snapshot


def _canonical_string_ids(value: object) -> bool:
    return (
        type(value) is list
        and all(type(item) is str for item in value)
        and value == sorted(set(value))
    )


def _audit_declared_domains(state: dict[str, Any], cls: type[BaseModel]) -> None:
    """Mirror existing pure audit domains only in their exact model context.

    These checks do not hash, encode, invoke model validators or establish source
    authority. Free checkpoint maps never acquire summary or lineage semantics.
    """
    from nurse_rcl.v2 import case_audit as legacy
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2
    from nurse_rcl.v2.runtime_schemas import _REQUIRED_AUDIT_ARTIFACTS

    if cls is SealedAuditPackage:
        hashes = state.get("artifact_hashes")
        if (
            type(hashes) is not dict
            or set(hashes) != _REQUIRED_AUDIT_ARTIFACTS
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            )
        ):
            raise ValueError("audit summary requires its existing artifact keys and digests")
        for name in (
            "accepted_provenance_ids",
            "rejected_provenance_ids",
            "conflicted_provenance_ids",
            "unknown_provenance_ids",
            "prediction_service_keys",
        ):
            if not _canonical_string_ids(state.get(name)):
                raise ValueError("audit summary collections must be sorted and unique")
        failure = state.get("terminal_state") == "FAILURE_SEALED"
        if (
            failure and (state["prediction_service_keys"] or state.get("run_mode") != "failure")
        ) or (not failure and not state["prediction_service_keys"]):
            raise ValueError("audit summary requires its original terminal prediction domain")
    elif cls is legacy.CaseAuditLineageIndex or cls is CaseAuditLineageIndexV2:
        nodes = state.get("nodes")
        if (
            type(nodes) is not list
            or any(type(node) is not dict for node in nodes)
            or not _canonical_string_ids([node.get("record_id") for node in nodes])
        ):
            raise ValueError("lineage nodes must be sorted and unique before encoding")
    elif cls is legacy.CaseAuditLineageNode:
        for name in ("source_record_ids", "consumer_record_ids"):
            edges = state.get(name)
            if not _canonical_string_ids(edges) or state.get("record_id") in cast(list[str], edges):
                raise ValueError("lineage edges must be sorted, unique and nonreflexive")
    elif cls is EmptyCandidateUniverseFailure:
        if state.get("prediction_service_keys") != []:
            raise ValueError("empty candidate failure must retain an empty prediction")
    elif cls is legacy.ImmutableCaseAuditPackage:
        if state.get("terminal_state") == "SEALED":
            invalid = (
                state.get("replay_bundle_json") is None
                or state.get("lineage_index_json") is None
                or state.get("failure_diagnostic_json") is not None
            )
        else:
            invalid = (
                state.get("replay_bundle_json") is not None
                or state.get("lineage_index_json") is not None
                or state.get("failure_diagnostic_json") is None
            )
        if invalid:
            raise ValueError("audit package requires its original terminal payload domain")


def _audit_unchanged(value: object, original: tuple[object, tuple[object, ...]]) -> None:
    try:
        unchanged = _audit_snapshot(value) == original
    except _ERRORS:
        unchanged = False
    if not unchanged:
        raise ValueError("audit original input bytes or representation changed during replay")


def _audit_document(raw_input: str | bytes | bytearray, cls: type[BaseModel]) -> dict[str, Any]:
    """Inspect complete named JSON before validators, serializers, or properties."""
    raw = _no_roles._json_bytes(raw_input)
    _no_roles._scan_json(raw)
    plain = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_no_roles._unique_object,
        parse_float=_no_evidence._finite_json_float,
        parse_constant=_no_roles._reject_json_number,
    )
    _audit_snapshot(plain)
    if not _json_shape(plain, cls, domains=False):
        raise ValueError("audit JSON requires exact declared fields and representations")
    expanded = _json_with_defaults(plain, cls)
    _audit_snapshot(expanded)
    if not _json_shape(expanded, cls):
        raise ValueError("audit JSON exceeds declared domains")
    return cast(dict[str, Any], plain)


def _named_documents(state: dict[str, Any], *, version: int) -> dict[str, dict[str, Any]]:
    from nurse_rcl.v2 import case_audit as legacy
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2

    specifications: tuple[tuple[str, type[BaseModel]], ...] = (
        (
            "replay_bundle_json",
            CaseRankingReplayBundleV2 if version == 2 else legacy.CaseRankingReplayBundle,
        ),
        (
            "lineage_index_json",
            CaseAuditLineageIndexV2 if version == 2 else legacy.CaseAuditLineageIndex,
        ),
        ("failure_diagnostic_json", EmptyCandidateUniverseFailure),
        ("sealed_audit_package_json", SealedAuditPackage),
    )
    documents = {}
    for name, cls in specifications:
        raw = state[name]
        if raw is None:
            continue
        if type(raw) is not str or not raw:
            raise ValueError("named audit payload requires nonempty canonical text")
        documents[name] = _audit_document(raw, cls)
    # Every payload has passed its resource/domain checks before the first encoder.
    _audit_snapshot(documents)
    for name, document in documents.items():
        if canonical_json_bytes(document).decode("utf-8") != state[name]:
            raise ValueError("named audit payload must contain canonical JSON")
    return documents


def _preflight_audit_package(
    value: object, expected_pin: str | None = None
) -> tuple[object, tuple[object, ...]]:
    from nurse_rcl.v2 import case_audit as legacy

    kind = type(value)
    if kind is not ImmutableCaseAuditPackageV2 and kind is not legacy.ImmutableCaseAuditPackage:
        raise ValueError("audit requires an exact supported package identity")
    snapshot = _audit_snapshot(value)
    state = cast(dict[str, Any], snapshot[0])
    version = 2 if kind is ImmutableCaseAuditPackageV2 else 1
    expected_schema = cast(type[BaseModel], kind).model_fields["schema_version"].default
    if state["schema_version"] != expected_schema:
        raise ValueError("audit class identity and schema literal differ")
    if expected_pin is not None and (
        type(expected_pin) is not str
        or len(expected_pin) != 64
        or any(c not in "0123456789abcdef" for c in expected_pin)
        or not hmac.compare_digest(expected_pin, state["audit_sha256"])
    ):
        raise ValueError("external audit digest does not match the sealed package")
    try:
        _named_documents(state, version=version)
    finally:
        _audit_unchanged(value, snapshot)
    return snapshot


def _validate_audit_json(
    cls: type[_ModelT], raw_input: str | bytes | bytearray, **kwargs: Any
) -> _ModelT:
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2

    if cls is not ImmutableCaseAuditPackageV2 and cls is not CaseAuditLineageIndexV2:
        raise ValueError("audit JSON requires an exact code-owned class")
    raw = _no_roles._json_bytes(raw_input)
    try:
        plain = _audit_document(raw, cls)
        if cls is ImmutableCaseAuditPackageV2:
            _named_documents(plain, version=2)
        return super(cls, cls).model_validate_json(raw, **kwargs)
    finally:
        if raw != _no_roles._json_bytes(raw_input):
            raise ValueError("audit JSON input changed during validation")


def _audit_model_bytes(value: object) -> bytes:
    return canonical_json_bytes(_audit_snapshot(value)[0])


class ImmutableCaseAuditPackageV2(StrictFrozenModel):
    """Independent successful /2 package; complete canonical payloads are immutable strings."""

    schema_version: Literal["nurse.v2.immutable-case-audit-package/2"] = (
        IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    terminal_state: Literal["SEALED"]
    replay_bundle_json: str = Field(min_length=1)
    lineage_index_json: str = Field(min_length=1)
    failure_diagnostic_json: None
    sealed_audit_package_json: str = Field(min_length=1)
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        return _validate_audit_json(cls, json_data, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def safe_state(cls, value: object) -> object:
        snapshot = _audit_snapshot(value)
        state = cast(dict[str, Any], snapshot[0])
        if not _json_shape(state, cls):
            raise ValueError("audit package requires exact declared root fields and domains")
        if type(state) is dict and all(
            name in state
            for name in (
                "replay_bundle_json",
                "lineage_index_json",
                "failure_diagnostic_json",
                "sealed_audit_package_json",
            )
        ):
            try:
                _named_documents(state, version=2)
            finally:
                _audit_unchanged(value, snapshot)
        return value

    @field_validator("*", mode="before")
    @classmethod
    def exact_fields(cls, value: object, info: ValidationInfo) -> object:
        return _exact_field(cls, value, info)

    @property
    def replay_bundle(self) -> CaseRankingReplayBundleV2:
        original = _preflight_audit_package(self)
        try:
            return CaseRankingReplayBundleV2.model_validate_json(self.replay_bundle_json)
        finally:
            _audit_unchanged(self, original)

    @property
    def replay_bundle_or_none(self) -> CaseRankingReplayBundleV2:
        return self.replay_bundle

    @property
    def lineage_index(self) -> CaseAuditLineageIndexV2:
        from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2

        original = _preflight_audit_package(self)
        try:
            return CaseAuditLineageIndexV2.model_validate_json(self.lineage_index_json)
        finally:
            _audit_unchanged(self, original)

    @property
    def lineage_index_or_none(self) -> CaseAuditLineageIndexV2:
        return self.lineage_index

    @property
    def failure_diagnostic(self) -> None:
        _preflight_audit_package(self)
        return None

    @property
    def sealed_audit_package(self) -> SealedAuditPackage:
        original = _preflight_audit_package(self)
        try:
            return SealedAuditPackage.model_validate_json(self.sealed_audit_package_json)
        finally:
            _audit_unchanged(self, original)

    def verify(self) -> bool:
        try:
            if type(self) is not ImmutableCaseAuditPackageV2:
                return False
            original = _preflight_audit_package(self)
            try:
                _verified_audit_parts(self, original)
                return True
            finally:
                _audit_unchanged(self, original)
        except _ERRORS:
            return False


def _verified_audit_parts(
    package: ImmutableCaseAuditPackageV2, original: tuple[object, tuple[object, ...]]
) -> tuple[CaseRankingReplayBundleV2, CaseAuditLineageIndexV2, SealedAuditPackage]:
    from nurse_rcl.v2 import case_audit as legacy
    from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2, _build_verified_lineage

    state = cast(dict[str, Any], original[0])
    unsigned = {name: value for name, value in state.items() if name != "audit_sha256"}
    if not hmac.compare_digest(state["audit_sha256"], sha256_hex(canonical_json_bytes(unsigned))):
        raise ValueError("immutable audit outer hash differs")
    detached = ImmutableCaseAuditPackageV2.model_validate_json(canonical_json_bytes(state))
    if _audit_model_bytes(detached) != canonical_json_bytes(state):
        raise ValueError("audit package schema roundtrip differs")
    bundle = CaseRankingReplayBundleV2.model_validate_json(detached.replay_bundle_json)
    lineage = CaseAuditLineageIndexV2.model_validate_json(detached.lineage_index_json)
    summary = SealedAuditPackage.model_validate_json(detached.sealed_audit_package_json)
    if not bundle.verify_inputs() or not lineage.verify():
        raise ValueError("audit bundle or lineage failed verification")
    if _audit_model_bytes(lineage) != _audit_model_bytes(_build_verified_lineage(bundle)):
        raise ValueError("audit lineage differs from complete source replay")
    if (
        summary.terminal_state != "SEALED"
        or summary.run_mode == "failure"
        or summary.opaque_case_id != state["opaque_case_id"]
        or bundle.evidence.time_contract.opaque_case_id != state["opaque_case_id"]
        or any(row.experiment_version != summary.experiment_version for row in bundle.checkpoints)
    ):
        raise ValueError("audit case, terminal metadata, or checkpoint experiment differs")
    expected = legacy._build_success_sealed_summary(
        bundle=cast(Any, bundle),
        experiment_version=summary.experiment_version,
        run_mode=summary.run_mode,
        sealed_at_utc=summary.sealed_at_utc,
    )
    if _audit_model_bytes(summary) != _audit_model_bytes(expected):
        raise ValueError("audit sealed summary differs from complete source replay")
    return bundle, lineage, summary


@dataclass(frozen=True, slots=True, init=False)
class CaseAuditReplayReceiptV2:
    """Private nonpersistent result value, constructed only after complete /2 replay."""

    opaque_case_id: str
    terminal_state: Literal["SEALED"]
    audit_sha256: str
    prediction_service_keys: tuple[str, ...]
    ranking_result: ReferenceRankingResult | NoRoleReferenceRankingResult
    top_k: TopKArtifact
    failure_diagnostic: None

    def __init__(self) -> None:
        raise TypeError("audit receipts can only be created by complete audit replay")


def seal_case_audit_v2(
    *,
    replay_bundle: CaseRankingReplayBundleV2,
    experiment_version: str,
    run_mode: RunMode,
    sealed_at_utc: str,
) -> ImmutableCaseAuditPackageV2:
    from nurse_rcl.v2 import case_audit as legacy
    from nurse_rcl.v2.case_audit_lineage_v2 import _build_verified_lineage

    if type(replay_bundle) is not CaseRankingReplayBundleV2:
        raise ValueError("audit /2 sealing requires its exact ranking bundle")
    original = _audit_snapshot(replay_bundle)
    _inspect((experiment_version, run_mode, sealed_at_utc))
    if (
        type(experiment_version) is not str
        or not experiment_version
        or type(sealed_at_utc) is not str
        or not _exact(run_mode, RunMode)
        or run_mode == "failure"
    ):
        raise ValueError("successful audit requires exact experiment, run mode and seal time")
    try:
        bundle = _fresh(replay_bundle, CaseRankingReplayBundleV2, canonical_json_bytes(original[0]))
        if not bundle.verify_inputs():
            raise ValueError("audit /2 bundle failed complete replay")
        if any(row.experiment_version != experiment_version for row in bundle.checkpoints):
            raise ValueError("checkpoint experiment version differs from the sealed audit")
        lineage = _build_verified_lineage(bundle)
        summary = legacy._build_success_sealed_summary(
            bundle=cast(Any, bundle),
            experiment_version=experiment_version,
            run_mode=run_mode,
            sealed_at_utc=sealed_at_utc,
        )
        unsigned = {
            "schema_version": IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION,
            "opaque_case_id": bundle.evidence.time_contract.opaque_case_id,
            "terminal_state": "SEALED",
            "replay_bundle_json": _audit_model_bytes(bundle).decode("utf-8"),
            "lineage_index_json": _audit_model_bytes(lineage).decode("utf-8"),
            "failure_diagnostic_json": None,
            "sealed_audit_package_json": _audit_model_bytes(summary).decode("utf-8"),
        }
        _audit_snapshot(unsigned)
        return ImmutableCaseAuditPackageV2.model_validate(
            {**unsigned, "audit_sha256": sha256_hex(canonical_json_bytes(unsigned))}
        )
    finally:
        _audit_unchanged(replay_bundle, original)


def replay_case_audit_v2(
    package: ImmutableCaseAuditPackageV2, *, expected_audit_sha256: str
) -> CaseAuditReplayReceiptV2:
    if type(package) is not ImmutableCaseAuditPackageV2 or type(expected_audit_sha256) is not str:
        raise ValueError("audit /2 replay requires exact package and independent external pin")
    original = _preflight_audit_package(package, expected_audit_sha256)
    try:
        bundle, _, _ = _verified_audit_parts(package, original)
        receipt = object.__new__(CaseAuditReplayReceiptV2)
        for name, value in {
            "opaque_case_id": bundle.evidence.time_contract.opaque_case_id,
            "terminal_state": "SEALED",
            "audit_sha256": expected_audit_sha256,
            "prediction_service_keys": tuple(
                row.service_entity_key for row in bundle.top_k.entries
            ),
            "ranking_result": bundle.ranking_result,
            "top_k": bundle.top_k,
            "failure_diagnostic": None,
        }.items():
            object.__setattr__(receipt, name, value)
        return receipt
    finally:
        _audit_unchanged(package, original)
