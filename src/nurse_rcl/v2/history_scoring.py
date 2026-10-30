"""Isolated, label-free history retrieval and scoring for NURSE V2."""

from __future__ import annotations

import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeVar, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel
from nurse_rcl.v2.candidate_freeze import FrozenCandidateUniverse
from nurse_rcl.v2.ranking_schemas import HistoryMode
from nurse_rcl.v2.safe_serialization import preflight_exact_pydantic_tree

HISTORY_GENERATION_AUTHORITY_SCHEMA_VERSION: Literal["nurse.v2.history-generation-authority/1"] = (
    "nurse.v2.history-generation-authority/1"
)
HISTORY_GENERATION_BATCH_MANIFEST_SCHEMA_VERSION: Literal[
    "nurse.v2.history-generation-batch-manifest/1"
] = "nurse.v2.history-generation-batch-manifest/1"
HISTORY_GENERATION_ENTRY_MEMBERSHIP_SCHEMA_VERSION: Literal[
    "nurse.v2.history-generation-entry-membership/1"
] = "nurse.v2.history-generation-entry-membership/1"
HISTORY_GENERATION_BASE_SCORE_MEMBERSHIP_SCHEMA_VERSION: Literal[
    "nurse.v2.history-generation-base-score-membership/1"
] = "nurse.v2.history-generation-base-score-membership/1"
HISTORY_SANITIZED_BASE_SCORE_ARTIFACT_SCHEMA_VERSION: Literal[
    "nurse.v2.history-sanitized-base-score-artifact/1"
] = "nurse.v2.history-sanitized-base-score-artifact/1"
HISTORY_SANITIZED_SEMANTIC_ARTIFACT_SCHEMA_VERSION: Literal[
    "nurse.v2.history-sanitized-semantic-artifact/1"
] = "nurse.v2.history-sanitized-semantic-artifact/1"
HISTORY_SANITIZED_GRAPH_ARTIFACT_SCHEMA_VERSION: Literal[
    "nurse.v2.history-sanitized-graph-artifact/1"
] = "nurse.v2.history-sanitized-graph-artifact/1"
HISTORY_SANITIZED_TOPOLOGY_ARTIFACT_SCHEMA_VERSION: Literal[
    "nurse.v2.history-sanitized-topology-artifact/1"
] = "nurse.v2.history-sanitized-topology-artifact/1"
HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION: Literal["nurse.v2.history-base-candidate-score/1"] = (
    "nurse.v2.history-base-candidate-score/1"
)
HISTORY_LIBRARY_ENTRY_SCHEMA_VERSION: Literal["nurse.v2.history-library-entry/1"] = (
    "nurse.v2.history-library-entry/1"
)
FROZEN_HISTORY_LIBRARY_SCHEMA_VERSION: Literal["nurse.v2.frozen-history-library/1"] = (
    "nurse.v2.frozen-history-library/1"
)
HISTORY_QUERY_EMBEDDING_SCHEMA_VERSION: Literal["nurse.v2.history-query-embedding/1"] = (
    "nurse.v2.history-query-embedding/1"
)
HISTORY_SERVICE_MAPPING_SCHEMA_VERSION: Literal["nurse.v2.history-service-mapping/1"] = (
    "nurse.v2.history-service-mapping/1"
)
HISTORY_SCORING_CONFIG_SCHEMA_VERSION: Literal["nurse.v2.history-scoring-config/1"] = (
    "nurse.v2.history-scoring-config/1"
)

HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION: Literal["nurse.v2.history-score-contribution/1"] = (
    "nurse.v2.history-score-contribution/1"
)
CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION: Literal["nurse.v2.candidate-history-score/1"] = (
    "nurse.v2.candidate-history-score/1"
)
HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION: Literal["nurse.v2.history-neighbor-audit/1"] = (
    "nurse.v2.history-neighbor-audit/1"
)
HISTORY_HINT_REFERENCE_SCHEMA_VERSION: Literal["nurse.v2.history-hint-reference/1"] = (
    "nurse.v2.history-hint-reference/1"
)
HISTORY_EVALUATION_RESULT_SCHEMA_VERSION: Literal["nurse.v2.history-evaluation-result/1"] = (
    "nurse.v2.history-evaluation-result/1"
)

HistoryClaimScope = Literal[
    "history_disabled",
    "prior_history",
    "offline_similarity_only",
]
HistoryNeighborStatus = Literal[
    "self_excluded",
    "incident_group_excluded",
    "time_excluded",
    "below_similarity",
    "same_group_not_selected",
    "top_k_not_selected",
    "selected",
]
HistoryMappingStatus = Literal["mapped", "missing_mapping"]
HistorySourcePartition = Literal["dev", "approved_unlabeled_archive"]
HistoryTimePolicy = Literal["strict_prior", "offline_similarity"]

_MAX_VECTOR_DIMENSION = 4_096
_MAX_HISTORY_BASE_SCORES = 32_768
_MAX_HISTORY_ENTRIES = 4_096
_MAX_HISTORY_MAPPINGS = 32_768
_MAX_HISTORY_CANDIDATES = 32_768
_MAX_PROVENANCE_IDS = 131_072
_MAX_SERIALIZED_STRING_LENGTH = 8_192
_MAX_HISTORY_ARTIFACT_BYTES = 1_048_576
_MAX_TRUSTED_BUNDLE_TOTAL_BASE_SOURCES = _MAX_HISTORY_BASE_SCORES
_MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES = 64 * _MAX_HISTORY_ARTIFACT_BYTES
_SERIALIZATION_PREFLIGHT_MAX_DEPTH = 128
_SERIALIZATION_PREFLIGHT_MAX_NODES = 500_000
_MAX_TRUSTED_BUNDLE_PROJECTED_WORK = _SERIALIZATION_PREFLIGHT_MAX_NODES
_ModelT = TypeVar("_ModelT", bound=StrictFrozenModel)
_ExternalDocumentT = TypeVar("_ExternalDocumentT", bound=StrictFrozenModel)


class _HistoryScoringResourceLimitError(ValueError):
    """Internal marker translated to a fixed public budget failure."""


class HistoryScoringBudgetExceeded(ValueError):
    """Fixed redacted error for history input, work, or output ceilings."""

    reason_code: Literal["history_scoring_budget_exceeded"] = "history_scoring_budget_exceeded"

    def __init__(self) -> None:
        super().__init__("history scoring budget exceeded")


_DEFAULT_CONFIG_DOCUMENT: dict[str, object] = {
    "schema_version": "nurse.v2.history-scoring-config/1",
    "k": 3,
    "min_similarity": 0.7,
    "time_policy": "strict_prior",
}
_DEFAULT_CONFIG_SHA256 = sha256_hex(canonical_json_bytes(_DEFAULT_CONFIG_DOCUMENT))


def _digest(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _stable_id(prefix: str, value: object) -> tuple[str, str]:
    digest = _digest(value)
    return f"{prefix}_{digest[:24]}", digest


def _candidate_binding_sha256(
    *,
    candidate_freeze_sha256: str,
    candidate_universe_sha256: str,
    candidate_count: int,
    candidate_service_keys: tuple[str, ...],
) -> str:
    return _digest(
        {
            "candidate_freeze_sha256": candidate_freeze_sha256,
            "candidate_universe_sha256": candidate_universe_sha256,
            "candidate_count": candidate_count,
            "candidate_service_keys": candidate_service_keys,
        }
    )


def _authoritative_input_sha256(
    *,
    candidate_binding_sha256: str,
    mode: HistoryMode,
    generation_authority_sha256: str | None,
    library_sha256: str | None,
    query_embedding_sha256: str | None,
    service_mapping_sha256s: tuple[str, ...],
    config_sha256: str,
) -> str:
    return _digest(
        {
            "candidate_binding_sha256": candidate_binding_sha256,
            "mode": mode,
            "generation_authority_sha256": generation_authority_sha256,
            "library_sha256": library_sha256,
            "query_embedding_sha256": query_embedding_sha256,
            "service_mapping_sha256s": service_mapping_sha256s,
            "config_sha256": config_sha256,
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
        resource_error_type=_HistoryScoringResourceLimitError,
    )


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
        left_tuple = cast(tuple[object, ...], value)
        right_tuple = cast(tuple[object, ...], rebuilt)
        return tuple.__len__(left_tuple) == tuple.__len__(right_tuple) and all(
            _same_exact_runtime_shape(left_item, right_item)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=True)
        )
    if type(rebuilt) is list:
        left_list = cast(list[object], value)
        right_list = cast(list[object], rebuilt)
        return list.__len__(left_list) == list.__len__(right_list) and all(
            _same_exact_runtime_shape(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    if type(rebuilt) is dict:
        left_dict = cast(dict[object, object], value)
        right_dict = cast(dict[object, object], rebuilt)
        return set(dict.keys(left_dict)) == set(dict.keys(right_dict)) and all(
            _same_exact_runtime_shape(
                dict.__getitem__(left_dict, key),
                dict.__getitem__(right_dict, key),
            )
            for key in dict.keys(right_dict)
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
    _preflight_exact_tree(value, root_model_types=(model_type,), label=label)
    document = model_type.model_dump(value, mode="json", warnings="error")
    rebuilt = model_type.model_validate(document)
    if not _same_exact_runtime_shape(value, rebuilt):
        raise ValueError(f"{label} is not an exact canonical model")
    if model_type.model_dump(rebuilt, mode="json", warnings="error") != document:
        raise ValueError(f"{label} is not canonical")
    return rebuilt


def _verify_content_address(
    value: StrictFrozenModel,
    model_type: type[_ModelT],
    *,
    id_field: str | None,
    id_prefix: str | None,
    sha_field: str,
) -> bool:
    try:
        if type(value) is not model_type:
            return False
        _preflight_exact_tree(value, root_model_types=(model_type,), label="history artifact")
        document = model_type.model_dump(value, mode="json", warnings="error")
        rebuilt = model_type.model_validate(document)
        if not _same_exact_runtime_shape(value, rebuilt):
            return False
        excluded = {sha_field} if id_field is None else {id_field, sha_field}
        unsigned = model_type.model_dump(rebuilt, mode="json", exclude=excluded)
        digest = _digest(unsigned)
    except (AttributeError, TypeError, ValueError):
        return False
    if rebuilt != value or getattr(rebuilt, sha_field) != digest:
        return False
    return id_field is None or getattr(rebuilt, id_field) == f"{id_prefix}_{digest[:24]}"


def _plain_sequence(values: object, *, label: str, maximum: int) -> Sequence[object]:
    if type(values) is tuple:
        sequence: Sequence[object] = cast(tuple[object, ...], values)
    elif type(values) is list:
        sequence = cast(list[object], values)
    else:
        raise ValueError(f"{label} must use an exact tuple or list")
    if len(sequence) > maximum:
        raise _HistoryScoringResourceLimitError(f"{label} exceeds its fixed limit")
    return sequence


def _preflight_model_tuple_limit(
    value: object,
    *,
    model_type: type[StrictFrozenModel],
    field_name: str,
    maximum: int,
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
    if tuple.__len__(field_value) > maximum:
        raise _HistoryScoringResourceLimitError(
            f"{label} exceeds the declared model quantity limit"
        )


def _add_projected_cost(*, current: int, additional: int) -> int:
    if additional > _SERIALIZATION_PREFLIGHT_MAX_NODES - current:
        raise _HistoryScoringResourceLimitError("history scoring output exceeds its fixed budget")
    return current + additional


def _preflight_projected_output(
    *,
    mode: HistoryMode,
    entry_count: int,
    mapping_count: int,
    candidate_count: int,
    k: int,
) -> None:
    """Bound cross-product work and every persisted output before allocation."""

    selected_upper = min(entry_count, k)
    projected = _add_projected_cost(
        current=0,
        additional=1 + len(HistoryEvaluationResult.model_fields),
    )
    projected = _add_projected_cost(
        current=projected,
        additional=entry_count * (1 + len(HistoryNeighborAudit.model_fields)),
    )
    projected = _add_projected_cost(
        current=projected,
        additional=candidate_count * (1 + len(CandidateHistoryScore.model_fields)),
    )
    projected = _add_projected_cost(
        current=projected,
        additional=entry_count * mapping_count * candidate_count,
    )
    projected = _add_projected_cost(
        current=projected,
        additional=selected_upper,
    )
    if mode == "hint":
        _add_projected_cost(
            current=projected,
            additional=selected_upper * (1 + len(HistoryHintReference.model_fields)),
        )
    elif mode == "score":
        _add_projected_cost(
            current=projected,
            additional=(
                selected_upper * candidate_count * (1 + len(HistoryScoreContribution.model_fields))
            ),
        )


def _is_finite_nonzero_embedding(values: tuple[float, ...]) -> bool:
    if any(not math.isfinite(value) for value in values):
        return False
    try:
        norm_squared = math.fsum(value * value for value in values)
    except OverflowError:
        return False
    return math.isfinite(norm_squared) and norm_squared > 0.0


def _plain_embedding(values: object) -> tuple[float, ...]:
    sequence = _plain_sequence(
        values,
        label="history embedding",
        maximum=_MAX_VECTOR_DIMENSION,
    )
    if not sequence:
        raise ValueError("history embedding input verification failed")
    embedding: list[float] = []
    for value in sequence:
        if type(value) not in {int, float}:
            raise ValueError("history embedding input verification failed")
        converted = float(cast(int | float, value))
        if not math.isfinite(converted):
            raise ValueError("history embedding input verification failed")
        embedding.append(converted)
    result = tuple(embedding)
    if not _is_finite_nonzero_embedding(result):
        raise ValueError("history embedding input verification failed")
    return result


def _require_plain_string(value: object, *, label: str) -> str:
    if type(value) is str and str.__len__(value) > _MAX_SERIALIZED_STRING_LENGTH:
        raise _HistoryScoringResourceLimitError(f"{label} exceeds the string limit")
    if type(value) is not str or str.__len__(value) < 1:
        raise ValueError(f"{label} input verification failed")
    return value


_FORBIDDEN_HISTORY_METADATA = re.compile(
    r"(?:^|[^a-z0-9])(?:answer|answers|groundtruth|ground_truth|label|labels|"
    r"injection|fault|fault_type|original|original_id|case_path|source_path)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_WINDOWS_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_HISTORY_MAPPING_PROVENANCE_ID_V1 = re.compile(r"^mapping:[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?$")
_HISTORY_IDENTIFIER_V1 = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


def _has_forbidden_history_metadata(value: str) -> bool:
    return bool(
        _FORBIDDEN_HISTORY_METADATA.search(value)
        or _WINDOWS_PATH.match(value)
        or "\\" in value
        or value.startswith("/")
        or "://" in value
        or value.casefold().startswith("file:")
    )


def _require_history_identifier(value: object, *, label: str) -> str:
    text = _require_plain_string(value, label=label)
    if _HISTORY_IDENTIFIER_V1.fullmatch(text) is None or _has_forbidden_history_metadata(text):
        raise ValueError(f"{label} violates history identifier grammar v1")
    return text


def _preflight_json_lexical_budget(raw: bytes) -> int:
    depth = 0
    nodes = 0
    in_string = False
    escaped = False
    in_atom = False
    quoted_length = 0
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
                quoted_length += 1
            elif byte == 0x5C:
                escaped = True
                quoted_length += 1
            elif byte == 0x22:
                in_string = False
            else:
                quoted_length += 1
            if quoted_length > _MAX_SERIALIZED_STRING_LENGTH:
                raise _HistoryScoringResourceLimitError(
                    "history artifact exceeds the quoted string budget"
                )
            continue
        if byte == 0x22:
            in_string = True
            escaped = False
            in_atom = False
            quoted_length = 0
            nodes += 1
        elif byte in (0x5B, 0x7B):
            depth += 1
            in_atom = False
            nodes += 1
            if depth > _SERIALIZATION_PREFLIGHT_MAX_DEPTH:
                raise _HistoryScoringResourceLimitError(
                    "history artifact exceeds the nesting budget"
                )
        elif byte in (0x5D, 0x7D, 0x2C, 0x3A, 0x20, 0x09, 0x0A, 0x0D):
            if byte in (0x5D, 0x7D):
                depth -= 1
            in_atom = False
        elif not in_atom:
            in_atom = True
            nodes += 1
        if nodes > _SERIALIZATION_PREFLIGHT_MAX_NODES:
            raise _HistoryScoringResourceLimitError(
                "history artifact exceeds the lexical node budget"
            )
    return nodes


def _preflight_parsed_artifact_tree(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    while stack:
        current, containing_depth = stack.pop()
        nodes += 1
        if nodes > _SERIALIZATION_PREFLIGHT_MAX_NODES:
            raise _HistoryScoringResourceLimitError(
                "history artifact exceeds the parsed node budget"
            )
        if type(current) is str:
            if len(current) > _MAX_SERIALIZED_STRING_LENGTH:
                raise _HistoryScoringResourceLimitError(
                    "history artifact exceeds the parsed string budget"
                )
            continue
        if type(current) is list:
            container_depth = containing_depth + 1
            if container_depth > _SERIALIZATION_PREFLIGHT_MAX_DEPTH:
                raise _HistoryScoringResourceLimitError(
                    "history artifact exceeds the parsed nesting budget"
                )
            container_id = id(current)
            if container_id in seen_containers:
                raise ValueError("history artifact contains a repeated container")
            seen_containers.add(container_id)
            stack.extend((item, container_depth) for item in cast(list[object], current))
            continue
        if type(current) is dict:
            container_depth = containing_depth + 1
            if container_depth > _SERIALIZATION_PREFLIGHT_MAX_DEPTH:
                raise _HistoryScoringResourceLimitError(
                    "history artifact exceeds the parsed nesting budget"
                )
            container_id = id(current)
            if container_id in seen_containers:
                raise ValueError("history artifact contains a repeated container")
            seen_containers.add(container_id)
            mapping = cast(dict[object, object], current)
            stack.extend((item, container_depth) for item in dict.keys(mapping))
            stack.extend((item, container_depth) for item in dict.values(mapping))
            continue
        if type(current) not in {int, float, bool, type(None)}:
            raise ValueError("history artifact contains a non-JSON value")


def _expected_manifest_entry_key(
    *,
    dataset: DatasetName,
    source_partition: HistorySourcePartition,
    source_opaque_case_id: str,
) -> str:
    digest = _digest(
        {
            "dataset": dataset,
            "source_partition": source_partition,
            "source_opaque_case_id": source_opaque_case_id,
        }
    )
    return f"hment_{digest[:24]}"


def _expected_manifest_base_score_key(
    *,
    manifest_entry_key: str,
    source_service_key: str,
) -> str:
    digest = _digest(
        {
            "manifest_entry_key": manifest_entry_key,
            "source_service_key": source_service_key,
        }
    )
    return f"hmbs_{digest[:24]}"


def _require_sorted_unique_digests(values: tuple[str, ...], *, label: str) -> None:
    if values != tuple(sorted(set(values))) or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values
    ):
        raise ValueError(f"{label} must be sorted unique SHA-256 digests")


class _HistoryExternalSafetyContract(StrictFrozenModel):
    uses_supervision: Literal[False] = False
    has_solution_data: Literal[False] = False
    has_failure_metadata: Literal[False] = False
    has_injected_metadata: Literal[False] = False
    has_reversible_source_refs: Literal[False] = False


class HistoryGenerationBaseScoreMembership(StrictFrozenModel):
    schema_version: Literal["nurse.v2.history-generation-base-score-membership/1"] = (
        HISTORY_GENERATION_BASE_SCORE_MEMBERSHIP_SCHEMA_VERSION
    )
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    manifest_base_score_key: str = Field(pattern=r"^hmbs_[0-9a-f]{24}$")
    source_service_key: str = Field(min_length=1)
    base_history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_history_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    source_rule_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    score_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_membership_key(self) -> Self:
        _require_history_identifier(
            self.source_service_key,
            label="history manifest source service",
        )
        expected = _expected_manifest_base_score_key(
            manifest_entry_key=self.manifest_entry_key,
            source_service_key=self.source_service_key,
        )
        if self.manifest_base_score_key != expected:
            raise ValueError("history manifest base-score key is not canonical")
        return self


class HistoryGenerationEntryMembership(_HistoryExternalSafetyContract):
    schema_version: Literal["nurse.v2.history-generation-entry-membership/1"] = (
        HISTORY_GENERATION_ENTRY_MEMBERSHIP_SCHEMA_VERSION
    )
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    dataset: DatasetName
    source_partition: HistorySourcePartition
    source_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_completed_at_unix_ns: int = Field(ge=0)
    source_history_mode: Literal["off"] = "off"
    source_rule_channel_enabled: Literal[False] = False
    embedding_policy: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1, le=_MAX_VECTOR_DIMENSION)
    embedding: tuple[float, ...] = Field(min_length=1, max_length=_MAX_VECTOR_DIMENSION)
    semantic_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_scores: tuple[HistoryGenerationBaseScoreMembership, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_BASE_SCORES,
    )

    @model_validator(mode="after")
    def require_canonical_entry_membership(self) -> Self:
        _require_history_identifier(
            self.embedding_policy,
            label="history manifest embedding policy",
        )
        _require_history_identifier(
            self.embedding_model,
            label="history manifest embedding model",
        )
        expected_key = _expected_manifest_entry_key(
            dataset=self.dataset,
            source_partition=self.source_partition,
            source_opaque_case_id=self.source_opaque_case_id,
        )
        if self.manifest_entry_key != expected_key:
            raise ValueError("history manifest entry key is not canonical")
        if self.embedding_dimension != len(self.embedding) or not _is_finite_nonzero_embedding(
            self.embedding
        ):
            raise ValueError("history manifest embedding must be finite and nonzero")
        expected_scores = tuple(sorted(self.base_scores, key=lambda item: item.source_service_key))
        if self.base_scores != expected_scores:
            raise ValueError("history manifest base scores must be canonically ordered")
        if (
            len({item.manifest_base_score_key for item in self.base_scores})
            != len(self.base_scores)
            or len({item.source_service_key for item in self.base_scores}) != len(self.base_scores)
            or len({item.score_artifact_sha256 for item in self.base_scores})
            != len(self.base_scores)
            or any(item.manifest_entry_key != self.manifest_entry_key for item in self.base_scores)
        ):
            raise ValueError("history manifest base scores must be unique members")
        return self


class HistoryGenerationBatchManifest(_HistoryExternalSafetyContract):
    schema_version: Literal["nurse.v2.history-generation-batch-manifest/1"] = (
        HISTORY_GENERATION_BATCH_MANIFEST_SCHEMA_VERSION
    )
    dataset: DatasetName
    source_partition: HistorySourcePartition
    generation_policy: str = Field(min_length=1)
    source_history_mode: Literal["off"] = "off"
    source_rule_channel_enabled: Literal[False] = False
    entries: tuple[HistoryGenerationEntryMembership, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_ENTRIES,
    )

    @model_validator(mode="after")
    def require_canonical_batch(self) -> Self:
        _require_history_identifier(
            self.generation_policy,
            label="history manifest generation policy",
        )
        expected_entries = tuple(sorted(self.entries, key=lambda item: item.manifest_entry_key))
        if self.entries != expected_entries:
            raise ValueError("history manifest entries must be canonically ordered")
        if (
            len({item.manifest_entry_key for item in self.entries}) != len(self.entries)
            or len({item.source_opaque_case_id for item in self.entries}) != len(self.entries)
            or any(
                item.dataset != self.dataset or item.source_partition != self.source_partition
                for item in self.entries
            )
        ):
            raise ValueError("history manifest entries must be unique members of one batch")
        library_embedding_contracts = {
            (
                item.embedding_policy,
                item.embedding_model,
                item.embedding_dimension,
            )
            for item in self.entries
        }
        if len(library_embedding_contracts) != 1:
            raise ValueError("history manifest entries must share one embedding contract")
        return self


class HistorySanitizedBaseScoreArtifact(_HistoryExternalSafetyContract):
    schema_version: Literal["nurse.v2.history-sanitized-base-score-artifact/1"] = (
        HISTORY_SANITIZED_BASE_SCORE_ARTIFACT_SCHEMA_VERSION
    )
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    manifest_base_score_key: str = Field(pattern=r"^hmbs_[0-9a-f]{24}$")
    source_service_key: str = Field(min_length=1)
    base_history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_history_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    source_rule_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_canonical_score_identity(self) -> Self:
        _require_history_identifier(
            self.source_service_key,
            label="history sanitized score source service",
        )
        expected = _expected_manifest_base_score_key(
            manifest_entry_key=self.manifest_entry_key,
            source_service_key=self.source_service_key,
        )
        if self.manifest_base_score_key != expected:
            raise ValueError("history sanitized score key is not canonical")
        return self


class _HistorySanitizedReferenceArtifact(_HistoryExternalSafetyContract):
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    source_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    service_keys: tuple[str, ...] = Field(max_length=_MAX_HISTORY_BASE_SCORES)

    @model_validator(mode="after")
    def require_canonical_services(self) -> Self:
        if self.service_keys != tuple(sorted(set(self.service_keys))):
            raise ValueError("history sanitized service keys must be sorted and unique")
        for service_key in self.service_keys:
            _require_history_identifier(
                service_key,
                label="history sanitized service key",
            )
        return self


class HistorySanitizedSemanticArtifact(_HistorySanitizedReferenceArtifact):
    schema_version: Literal["nurse.v2.history-sanitized-semantic-artifact/1"] = (
        HISTORY_SANITIZED_SEMANTIC_ARTIFACT_SCHEMA_VERSION
    )
    semantic_signal_sha256s: tuple[str, ...] = Field(max_length=_MAX_PROVENANCE_IDS)

    @model_validator(mode="after")
    def require_canonical_signal_digests(self) -> Self:
        _require_sorted_unique_digests(
            self.semantic_signal_sha256s,
            label="history sanitized semantic signals",
        )
        return self


class HistorySanitizedGraphArtifact(_HistorySanitizedReferenceArtifact):
    schema_version: Literal["nurse.v2.history-sanitized-graph-artifact/1"] = (
        HISTORY_SANITIZED_GRAPH_ARTIFACT_SCHEMA_VERSION
    )
    graph_edge_sha256s: tuple[str, ...] = Field(max_length=_MAX_PROVENANCE_IDS)

    @model_validator(mode="after")
    def require_canonical_edge_digests(self) -> Self:
        _require_sorted_unique_digests(
            self.graph_edge_sha256s,
            label="history sanitized graph edges",
        )
        return self


class HistorySanitizedTopologyArtifact(_HistorySanitizedReferenceArtifact):
    schema_version: Literal["nurse.v2.history-sanitized-topology-artifact/1"] = (
        HISTORY_SANITIZED_TOPOLOGY_ARTIFACT_SCHEMA_VERSION
    )
    topology_relation_sha256s: tuple[str, ...] = Field(max_length=_MAX_PROVENANCE_IDS)

    @model_validator(mode="after")
    def require_canonical_relation_digests(self) -> Self:
        _require_sorted_unique_digests(
            self.topology_relation_sha256s,
            label="history sanitized topology relations",
        )
        return self


def _parse_canonical_typed_document(
    value: object,
    model_type: type[_ExternalDocumentT],
) -> tuple[_ExternalDocumentT, str]:
    if type(value) is not bytes or not value:
        raise ValueError("history typed artifact must use exact nonempty bytes")
    raw = value
    if bytes.__len__(raw) > _MAX_HISTORY_ARTIFACT_BYTES:
        raise _HistoryScoringResourceLimitError("history typed artifact exceeds its byte budget")
    _preflight_json_lexical_budget(raw)
    try:
        document = json.loads(raw.decode("utf-8"))
    except MemoryError:
        raise _HistoryScoringResourceLimitError(
            "history typed artifact parser exceeded its resource budget"
        ) from None
    except (
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        raise ValueError("history typed artifact is not canonical JSON") from None
    _preflight_parsed_artifact_tree(document)
    if type(document) is not dict:
        raise ValueError("history typed artifact must be a JSON object")
    try:
        parsed = model_type.model_validate_json(raw, strict=True)
        _preflight_exact_tree(
            parsed,
            root_model_types=(model_type,),
            label="history typed artifact",
        )
        canonical = canonical_json_bytes(
            model_type.model_dump(parsed, mode="json", warnings="error")
        )
    except MemoryError:
        raise _HistoryScoringResourceLimitError(
            "history typed artifact validation exceeded its resource budget"
        ) from None
    except (RecursionError, OverflowError, TypeError, ValueError):
        raise ValueError("history typed artifact failed strict validation") from None
    if canonical != raw:
        raise ValueError("history typed artifact is not canonical")
    return parsed, sha256_hex(raw)


class HistoryGenerationAuthority(StrictFrozenModel):
    """Content-addressed authority for one label-free, nonrecursive generation batch."""

    schema_version: Literal["nurse.v2.history-generation-authority/1"] = (
        HISTORY_GENERATION_AUTHORITY_SCHEMA_VERSION
    )
    generation_authority_id: str = Field(pattern=r"^hgauth_[0-9a-f]{24}$")
    dataset: DatasetName
    source_partition: HistorySourcePartition
    source_history_mode: Literal["off"] = "off"
    source_rule_channel_enabled: Literal[False] = False
    generation_policy: str = Field(min_length=1)
    canonical_batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_safe_generation_policy(self) -> Self:
        _require_history_identifier(
            self.generation_policy,
            label="history generation policy",
        )
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryGenerationAuthority,
            id_field="generation_authority_id",
            id_prefix="hgauth",
            sha_field="generation_authority_sha256",
        )

    def verify_trusted(
        self,
        *,
        expected_generation_authority_sha256: str,
        canonical_batch_manifest: bytes,
    ) -> bool:
        """Verify this single-batch authority against independent manifest input."""

        try:
            _build_trusted_generation_context(
                self,
                expected_generation_authority_sha256=expected_generation_authority_sha256,
                canonical_batch_manifest=canonical_batch_manifest,
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return True


@dataclass(frozen=True, slots=True)
class _TrustedGenerationContext:
    manifest: HistoryGenerationBatchManifest
    generation_authority: HistoryGenerationAuthority
    expected_generation_authority_sha256: str
    manifest_sha256: str
    entry_memberships: Mapping[str, HistoryGenerationEntryMembership]
    base_score_memberships: Mapping[
        tuple[str, str, str],
        HistoryGenerationBaseScoreMembership,
    ]


def _build_trusted_generation_context(
    generation_authority: HistoryGenerationAuthority,
    *,
    expected_generation_authority_sha256: str,
    canonical_batch_manifest: bytes,
) -> _TrustedGenerationContext:
    if (
        type(generation_authority) is not HistoryGenerationAuthority
        or type(expected_generation_authority_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_generation_authority_sha256) is None
        or type(canonical_batch_manifest) is not bytes
    ):
        raise ValueError("history trusted generation context has invalid exact inputs")
    manifest, manifest_sha256 = _parse_canonical_typed_document(
        canonical_batch_manifest,
        HistoryGenerationBatchManifest,
    )
    if not (
        HistoryGenerationAuthority.verify(generation_authority)
        and generation_authority.source_history_mode == "off"
        and generation_authority.source_rule_channel_enabled is False
        and generation_authority.dataset == manifest.dataset
        and generation_authority.source_partition == manifest.source_partition
        and generation_authority.generation_policy == manifest.generation_policy
        and generation_authority.source_history_mode == manifest.source_history_mode
        and generation_authority.source_rule_channel_enabled == manifest.source_rule_channel_enabled
        and hmac.compare_digest(
            generation_authority.generation_authority_sha256,
            expected_generation_authority_sha256,
        )
        and hmac.compare_digest(
            generation_authority.canonical_batch_manifest_sha256,
            manifest_sha256,
        )
    ):
        raise ValueError("history generation authority failed trusted verification")
    entry_memberships = {item.manifest_entry_key: item for item in manifest.entries}
    base_score_memberships = {
        (
            item.manifest_entry_key,
            score.manifest_base_score_key,
            score.source_service_key,
        ): score
        for item in manifest.entries
        for score in item.base_scores
    }
    if len(entry_memberships) != len(manifest.entries) or len(base_score_memberships) != sum(
        len(item.base_scores) for item in manifest.entries
    ):
        raise ValueError("history generation manifest memberships are not unique")
    return _TrustedGenerationContext(
        manifest=manifest,
        generation_authority=generation_authority,
        expected_generation_authority_sha256=expected_generation_authority_sha256,
        manifest_sha256=manifest_sha256,
        entry_memberships=MappingProxyType(entry_memberships),
        base_score_memberships=MappingProxyType(base_score_memberships),
    )


@dataclass(frozen=True, slots=True)
class HistoryTrustedBaseScoreSource:
    """Ephemeral raw source used to replay one persisted base-score digest."""

    source_service_key: str
    base_score_artifact: bytes

    def __post_init__(self) -> None:
        _require_history_identifier(
            self.source_service_key,
            label="history trusted source service",
        )
        if type(self.base_score_artifact) is not bytes:
            raise ValueError("history trusted base score artifact must use exact bytes")
        if bytes.__len__(self.base_score_artifact) > _MAX_HISTORY_ARTIFACT_BYTES:
            raise HistoryScoringBudgetExceeded()


@dataclass(frozen=True, slots=True)
class HistoryTrustedEntrySource:
    """Ephemeral raw source used to replay one persisted history entry."""

    entry_id: str
    base_score_sources: tuple[HistoryTrustedBaseScoreSource, ...]
    sanitized_semantic_payload: bytes
    sanitized_graph: bytes
    sanitized_topology: bytes

    def __post_init__(self) -> None:
        if (
            type(self.entry_id) is not str
            or re.fullmatch(r"hent_[0-9a-f]{24}", self.entry_id) is None
        ):
            raise ValueError("history trusted entry ID is invalid")
        if type(self.base_score_sources) is not tuple:
            raise ValueError("history trusted base score sources must use an exact tuple")
        if tuple.__len__(self.base_score_sources) > _MAX_HISTORY_BASE_SCORES:
            raise HistoryScoringBudgetExceeded()
        if any(type(item) is not HistoryTrustedBaseScoreSource for item in self.base_score_sources):
            raise ValueError("history trusted base score sources must use an exact tuple")
        for value in (
            self.sanitized_semantic_payload,
            self.sanitized_graph,
            self.sanitized_topology,
        ):
            if type(value) is not bytes:
                raise ValueError("history trusted entry artifacts must use exact bytes")
            if bytes.__len__(value) > _MAX_HISTORY_ARTIFACT_BYTES:
                raise HistoryScoringBudgetExceeded()


@dataclass(frozen=True, slots=True)
class HistoryTrustedSourceBundle:
    """Ephemeral independently anchored raw sources for full library replay."""

    generation_authority: HistoryGenerationAuthority
    expected_generation_authority_sha256: str
    canonical_batch_manifest: bytes
    entry_sources: tuple[HistoryTrustedEntrySource, ...]

    def __post_init__(self) -> None:
        if type(self.generation_authority) is not HistoryGenerationAuthority:
            raise ValueError("history trusted authority must use the exact model")
        if (
            type(self.expected_generation_authority_sha256) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.expected_generation_authority_sha256,
            )
            is None
        ):
            raise ValueError("history trusted authority digest is invalid")
        if type(self.canonical_batch_manifest) is not bytes:
            raise ValueError("history trusted batch manifest must use exact bytes")
        if bytes.__len__(self.canonical_batch_manifest) > _MAX_HISTORY_ARTIFACT_BYTES:
            raise HistoryScoringBudgetExceeded()
        if type(self.entry_sources) is not tuple:
            raise ValueError("history trusted entry sources must use an exact tuple")
        if tuple.__len__(self.entry_sources) > _MAX_HISTORY_ENTRIES:
            raise HistoryScoringBudgetExceeded()
        if any(type(item) is not HistoryTrustedEntrySource for item in self.entry_sources):
            raise ValueError("history trusted entry sources must use an exact tuple")


def _preflight_trusted_source_bundle(bundle: HistoryTrustedSourceBundle) -> None:
    """Bound aggregate replay work using only exact builtins and slotted fields."""

    if type(bundle) is not HistoryTrustedSourceBundle:
        raise ValueError("history trusted source bundle must use the exact type")
    manifest = object.__getattribute__(bundle, "canonical_batch_manifest")
    entry_sources = object.__getattribute__(bundle, "entry_sources")
    if type(manifest) is not bytes or type(entry_sources) is not tuple:
        raise ValueError("history trusted source bundle has invalid exact fields")
    manifest_size = bytes.__len__(manifest)
    entry_count = tuple.__len__(entry_sources)
    if manifest_size > _MAX_HISTORY_ARTIFACT_BYTES or entry_count > _MAX_HISTORY_ENTRIES:
        raise _HistoryScoringResourceLimitError(
            "history trusted source bundle exceeds an individual fixed limit"
        )

    total_base_sources = 0
    total_raw_bytes = manifest_size
    projected_work = 5
    if total_raw_bytes > _MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES:
        raise _HistoryScoringResourceLimitError(
            "history trusted bundle exceeds its aggregate raw-byte limit"
        )
    manifest_work = 1 + _preflight_json_lexical_budget(manifest)
    if manifest_work > _MAX_TRUSTED_BUNDLE_PROJECTED_WORK - projected_work:
        raise _HistoryScoringResourceLimitError(
            "history trusted bundle exceeds its projected-work limit"
        )
    projected_work += manifest_work
    for entry_index in range(entry_count):
        source = tuple.__getitem__(entry_sources, entry_index)
        if type(source) is not HistoryTrustedEntrySource:
            raise ValueError("history trusted entry source must use the exact type")
        base_sources = object.__getattribute__(source, "base_score_sources")
        if type(base_sources) is not tuple:
            raise ValueError("history trusted base score sources must use an exact tuple")
        base_source_count = tuple.__len__(base_sources)
        if base_source_count > _MAX_HISTORY_BASE_SCORES:
            raise _HistoryScoringResourceLimitError(
                "history trusted entry sources exceed their fixed limit"
            )
        if base_source_count > _MAX_TRUSTED_BUNDLE_TOTAL_BASE_SOURCES - total_base_sources:
            raise _HistoryScoringResourceLimitError(
                "history trusted bundle exceeds its aggregate base-source limit"
            )
        total_base_sources += base_source_count
        entry_work = 6 + 2 * base_source_count
        if entry_work > _MAX_TRUSTED_BUNDLE_PROJECTED_WORK - projected_work:
            raise _HistoryScoringResourceLimitError(
                "history trusted bundle exceeds its projected-work limit"
            )
        projected_work += entry_work

        for field_name in (
            "sanitized_semantic_payload",
            "sanitized_graph",
            "sanitized_topology",
        ):
            raw = object.__getattribute__(source, field_name)
            if type(raw) is not bytes:
                raise ValueError("history trusted entry artifacts must use exact bytes")
            raw_size = bytes.__len__(raw)
            if raw_size > _MAX_HISTORY_ARTIFACT_BYTES:
                raise _HistoryScoringResourceLimitError(
                    "history trusted entry artifact exceeds its fixed limit"
                )
            if raw_size > _MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES - total_raw_bytes:
                raise _HistoryScoringResourceLimitError(
                    "history trusted bundle exceeds its aggregate raw-byte limit"
                )
            total_raw_bytes += raw_size
            artifact_work = 1 + _preflight_json_lexical_budget(raw)
            if artifact_work > _MAX_TRUSTED_BUNDLE_PROJECTED_WORK - projected_work:
                raise _HistoryScoringResourceLimitError(
                    "history trusted bundle exceeds its projected-work limit"
                )
            projected_work += artifact_work

        for base_index in range(base_source_count):
            base_source = tuple.__getitem__(base_sources, base_index)
            if type(base_source) is not HistoryTrustedBaseScoreSource:
                raise ValueError("history trusted base score source must use the exact type")
            raw = object.__getattribute__(base_source, "base_score_artifact")
            if type(raw) is not bytes:
                raise ValueError("history trusted base score artifact must use exact bytes")
            raw_size = bytes.__len__(raw)
            if raw_size > _MAX_HISTORY_ARTIFACT_BYTES:
                raise _HistoryScoringResourceLimitError(
                    "history trusted base score artifact exceeds its fixed limit"
                )
            if raw_size > _MAX_TRUSTED_BUNDLE_TOTAL_RAW_BYTES - total_raw_bytes:
                raise _HistoryScoringResourceLimitError(
                    "history trusted bundle exceeds its aggregate raw-byte limit"
                )
            total_raw_bytes += raw_size
            artifact_work = 1 + _preflight_json_lexical_budget(raw)
            if artifact_work > _MAX_TRUSTED_BUNDLE_PROJECTED_WORK - projected_work:
                raise _HistoryScoringResourceLimitError(
                    "history trusted bundle exceeds its projected-work limit"
                )
            projected_work += artifact_work


class HistoryBaseCandidateScore(StrictFrozenModel):
    """A label-free source candidate score produced with H and R disabled."""

    schema_version: Literal["nurse.v2.history-base-candidate-score/1"] = (
        HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION
    )
    base_candidate_score_id: str = Field(pattern=r"^hbcs_[0-9a-f]{24}$")
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    manifest_base_score_key: str = Field(pattern=r"^hmbs_[0-9a-f]{24}$")
    source_service_key: str = Field(min_length=1)
    base_history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_history_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    source_rule_score: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    generation_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_score_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_candidate_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_safe_source_service(self) -> Self:
        _require_history_identifier(
            self.source_service_key,
            label="history source service",
        )
        expected_key = _expected_manifest_base_score_key(
            manifest_entry_key=self.manifest_entry_key,
            source_service_key=self.source_service_key,
        )
        if self.manifest_base_score_key != expected_key:
            raise ValueError("history persisted base-score key is not canonical")
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryBaseCandidateScore,
            id_field="base_candidate_score_id",
            id_prefix="hbcs",
            sha_field="base_candidate_score_sha256",
        )


class HistoryLibraryEntry(StrictFrozenModel):
    """One sanitized, nonrecursive, label-free historical case artifact."""

    schema_version: Literal["nurse.v2.history-library-entry/1"] = (
        HISTORY_LIBRARY_ENTRY_SCHEMA_VERSION
    )
    entry_id: str = Field(pattern=r"^hent_[0-9a-f]{24}$")
    manifest_entry_key: str = Field(pattern=r"^hment_[0-9a-f]{24}$")
    dataset: DatasetName
    source_partition: HistorySourcePartition
    source_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    source_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_completed_at_unix_ns: int = Field(ge=0)
    source_history_mode: Literal["off"] = "off"
    source_rule_channel_enabled: Literal[False] = False
    label_free: Literal[True] = True
    contains_answers: Literal[False] = False
    contains_fault_metadata: Literal[False] = False
    contains_injection_metadata: Literal[False] = False
    contains_original_identifiers_or_paths: Literal[False] = False
    generation_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sanitized_semantic_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sanitized_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sanitized_topology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_policy: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1, le=_MAX_VECTOR_DIMENSION)
    embedding: tuple[float, ...] = Field(min_length=1, max_length=_MAX_VECTOR_DIMENSION)
    base_candidate_scores: tuple[HistoryBaseCandidateScore, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_BASE_SCORES,
    )
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_label_free_entry(self) -> Self:
        _require_history_identifier(
            self.embedding_policy,
            label="history embedding policy",
        )
        _require_history_identifier(
            self.embedding_model,
            label="history embedding model",
        )
        expected_manifest_entry_key = _expected_manifest_entry_key(
            dataset=self.dataset,
            source_partition=self.source_partition,
            source_opaque_case_id=self.source_opaque_case_id,
        )
        if self.manifest_entry_key != expected_manifest_entry_key:
            raise ValueError("history persisted entry key is not canonical")
        if self.embedding_dimension != len(self.embedding):
            raise ValueError("history embedding dimension does not match")
        if not _is_finite_nonzero_embedding(self.embedding):
            raise ValueError("history embedding must be finite and nonzero")
        expected = tuple(
            sorted(self.base_candidate_scores, key=lambda item: item.source_service_key)
        )
        if self.base_candidate_scores != expected:
            raise ValueError("history base scores must be canonically ordered")
        if len({item.source_service_key for item in self.base_candidate_scores}) != len(
            self.base_candidate_scores
        ) or len({item.manifest_base_score_key for item in self.base_candidate_scores}) != len(
            self.base_candidate_scores
        ):
            raise ValueError("history base score services must be unique")
        if any(
            item.manifest_entry_key != self.manifest_entry_key
            for item in self.base_candidate_scores
        ):
            raise ValueError("history base scores do not belong to the persisted entry")
        if (
            len({item.base_candidate_score_id for item in self.base_candidate_scores})
            != len(self.base_candidate_scores)
            or len({item.base_candidate_score_sha256 for item in self.base_candidate_scores})
            != len(self.base_candidate_scores)
            or len({item.base_score_artifact_sha256 for item in self.base_candidate_scores})
            != len(self.base_candidate_scores)
        ):
            raise ValueError("history base scores must have unique identities and hashes")
        if any(
            item.generation_authority_sha256 != self.generation_authority_sha256
            for item in self.base_candidate_scores
        ):
            raise ValueError("history base scores do not share the entry generation authority")
        return self

    def verify(self) -> bool:
        return all(
            HistoryBaseCandidateScore.verify(item) for item in self.base_candidate_scores
        ) and _verify_content_address(
            self,
            HistoryLibraryEntry,
            id_field="entry_id",
            id_prefix="hent",
            sha_field="entry_sha256",
        )


class FrozenHistoryLibrary(StrictFrozenModel):
    """Dataset-specific, read-only collection of canonical history entries."""

    schema_version: Literal["nurse.v2.frozen-history-library/1"] = (
        FROZEN_HISTORY_LIBRARY_SCHEMA_VERSION
    )
    dataset: DatasetName
    source_partition: HistorySourcePartition
    embedding_policy: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1, le=_MAX_VECTOR_DIMENSION)
    generation_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    read_only: Literal[True] = True
    entries: tuple[HistoryLibraryEntry, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_ENTRIES,
    )
    library_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_library(self) -> Self:
        _require_history_identifier(
            self.embedding_policy,
            label="history embedding policy",
        )
        _require_history_identifier(
            self.embedding_model,
            label="history embedding model",
        )
        if self.entries != tuple(sorted(self.entries, key=lambda item: item.entry_id)):
            raise ValueError("history entries must be canonically ordered")
        if len({item.entry_id for item in self.entries}) != len(self.entries) or len(
            {item.entry_sha256 for item in self.entries}
        ) != len(self.entries):
            raise ValueError("history entries must have unique identities and hashes")
        source_case_keys = tuple(
            (item.dataset, item.source_opaque_case_id) for item in self.entries
        )
        if len(source_case_keys) != len(set(source_case_keys)):
            raise ValueError("history entries must have unique dataset source cases")
        if any(
            item.dataset != self.dataset
            or item.source_partition != self.source_partition
            or item.embedding_policy != self.embedding_policy
            or item.embedding_model != self.embedding_model
            or item.embedding_dimension != self.embedding_dimension
            or item.generation_authority_sha256 != self.generation_authority_sha256
            for item in self.entries
        ):
            raise ValueError("history entries do not match the frozen library contract")
        return self

    def verify(self) -> bool:
        return all(HistoryLibraryEntry.verify(item) for item in self.entries) and (
            _verify_content_address(
                self,
                FrozenHistoryLibrary,
                id_field=None,
                id_prefix=None,
                sha_field="library_sha256",
            )
        )


def _verify_trusted_base_candidate_score(
    score: HistoryBaseCandidateScore,
    source: HistoryTrustedBaseScoreSource,
    *,
    context: _TrustedGenerationContext,
    parsed_artifact: tuple[HistorySanitizedBaseScoreArtifact, str] | None = None,
) -> bool:
    if (
        type(score) is not HistoryBaseCandidateScore
        or type(source) is not HistoryTrustedBaseScoreSource
        or type(context) is not _TrustedGenerationContext
    ):
        return False
    try:
        if parsed_artifact is None:
            artifact, artifact_sha256 = _parse_canonical_typed_document(
                source.base_score_artifact,
                HistorySanitizedBaseScoreArtifact,
            )
        else:
            artifact, artifact_sha256 = parsed_artifact
            if (
                type(artifact) is not HistorySanitizedBaseScoreArtifact
                or type(artifact_sha256) is not str
                or not hmac.compare_digest(
                    artifact_sha256,
                    sha256_hex(source.base_score_artifact),
                )
            ):
                return False
        membership = context.base_score_memberships.get(
            (
                score.manifest_entry_key,
                score.manifest_base_score_key,
                score.source_service_key,
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False
    generation_authority = context.generation_authority
    return (
        membership is not None
        and HistoryBaseCandidateScore.verify(score)
        and score.source_service_key == source.source_service_key
        and artifact.manifest_entry_key == membership.manifest_entry_key
        and artifact.manifest_base_score_key == membership.manifest_base_score_key
        and artifact.source_service_key == membership.source_service_key
        and artifact.base_history_score == membership.base_history_score
        and artifact.source_history_score == membership.source_history_score
        and artifact.source_rule_score == membership.source_rule_score
        and score.manifest_entry_key == membership.manifest_entry_key
        and score.manifest_base_score_key == membership.manifest_base_score_key
        and score.source_service_key == membership.source_service_key
        and score.base_history_score == membership.base_history_score
        and score.source_history_score == membership.source_history_score
        and score.source_rule_score == membership.source_rule_score
        and hmac.compare_digest(
            score.generation_authority_sha256,
            generation_authority.generation_authority_sha256,
        )
        and hmac.compare_digest(score.base_score_artifact_sha256, artifact_sha256)
        and hmac.compare_digest(membership.score_artifact_sha256, artifact_sha256)
    )


def _verify_trusted_library_entry(
    entry: HistoryLibraryEntry,
    source: HistoryTrustedEntrySource,
    *,
    context: _TrustedGenerationContext,
    parsed_artifacts: (
        tuple[
            tuple[HistorySanitizedSemanticArtifact, str],
            tuple[HistorySanitizedGraphArtifact, str],
            tuple[HistorySanitizedTopologyArtifact, str],
        ]
        | None
    ) = None,
) -> bool:
    if (
        type(entry) is not HistoryLibraryEntry
        or type(source) is not HistoryTrustedEntrySource
        or type(context) is not _TrustedGenerationContext
        or type(source.base_score_sources) is not tuple
        or any(
            type(item) is not HistoryTrustedBaseScoreSource for item in source.base_score_sources
        )
    ):
        return False
    source_keys = tuple(item.source_service_key for item in source.base_score_sources)
    score_keys = tuple(item.source_service_key for item in entry.base_candidate_scores)
    if (
        len(source_keys) != len(set(source_keys))
        or tuple(sorted(source_keys)) != tuple(sorted(score_keys))
        or entry.entry_id != source.entry_id
    ):
        return False
    source_by_key = {item.source_service_key: item for item in source.base_score_sources}
    try:
        membership = context.entry_memberships.get(entry.manifest_entry_key)
        if parsed_artifacts is None:
            semantic, semantic_sha256 = _parse_canonical_typed_document(
                source.sanitized_semantic_payload,
                HistorySanitizedSemanticArtifact,
            )
            graph, graph_sha256 = _parse_canonical_typed_document(
                source.sanitized_graph,
                HistorySanitizedGraphArtifact,
            )
            topology, topology_sha256 = _parse_canonical_typed_document(
                source.sanitized_topology,
                HistorySanitizedTopologyArtifact,
            )
        else:
            (
                (semantic, semantic_sha256),
                (graph, graph_sha256),
                (
                    topology,
                    topology_sha256,
                ),
            ) = parsed_artifacts
            if (
                type(semantic) is not HistorySanitizedSemanticArtifact
                or type(graph) is not HistorySanitizedGraphArtifact
                or type(topology) is not HistorySanitizedTopologyArtifact
                or type(semantic_sha256) is not str
                or type(graph_sha256) is not str
                or type(topology_sha256) is not str
                or not hmac.compare_digest(
                    semantic_sha256,
                    sha256_hex(source.sanitized_semantic_payload),
                )
                or not hmac.compare_digest(graph_sha256, sha256_hex(source.sanitized_graph))
                or not hmac.compare_digest(
                    topology_sha256,
                    sha256_hex(source.sanitized_topology),
                )
            ):
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    if membership is None:
        return False
    expected_services = tuple(sorted(item.source_service_key for item in membership.base_scores))
    membership_score_keys = tuple(
        sorted(item.manifest_base_score_key for item in membership.base_scores)
    )
    persisted_score_keys = tuple(
        sorted(item.manifest_base_score_key for item in entry.base_candidate_scores)
    )
    generation_authority = context.generation_authority
    return (
        HistoryLibraryEntry.verify(entry)
        and entry.manifest_entry_key == membership.manifest_entry_key
        and entry.dataset == membership.dataset
        and entry.source_partition == membership.source_partition
        and entry.source_opaque_case_id == membership.source_opaque_case_id
        and entry.source_group_fingerprint == membership.source_group_fingerprint
        and entry.source_completed_at_unix_ns == membership.source_completed_at_unix_ns
        and entry.source_history_mode == membership.source_history_mode
        and entry.source_rule_channel_enabled == membership.source_rule_channel_enabled
        and entry.embedding_policy == membership.embedding_policy
        and entry.embedding_model == membership.embedding_model
        and entry.embedding_dimension == membership.embedding_dimension
        and entry.embedding == membership.embedding
        and membership_score_keys == persisted_score_keys
        and semantic.manifest_entry_key == membership.manifest_entry_key
        and semantic.source_opaque_case_id == membership.source_opaque_case_id
        and semantic.service_keys == expected_services
        and graph.manifest_entry_key == membership.manifest_entry_key
        and graph.source_opaque_case_id == membership.source_opaque_case_id
        and graph.service_keys == expected_services
        and topology.manifest_entry_key == membership.manifest_entry_key
        and topology.source_opaque_case_id == membership.source_opaque_case_id
        and topology.service_keys == expected_services
        and hmac.compare_digest(
            entry.generation_authority_sha256,
            generation_authority.generation_authority_sha256,
        )
        and hmac.compare_digest(entry.sanitized_semantic_payload_sha256, semantic_sha256)
        and hmac.compare_digest(entry.sanitized_graph_sha256, graph_sha256)
        and hmac.compare_digest(entry.sanitized_topology_sha256, topology_sha256)
        and hmac.compare_digest(membership.semantic_artifact_sha256, semantic_sha256)
        and hmac.compare_digest(membership.graph_artifact_sha256, graph_sha256)
        and hmac.compare_digest(membership.topology_artifact_sha256, topology_sha256)
        and all(
            _verify_trusted_base_candidate_score(
                score,
                source_by_key[score.source_service_key],
                context=context,
            )
            for score in entry.base_candidate_scores
        )
    )


def _verify_trusted_source_bundle(
    library: FrozenHistoryLibrary,
    bundle: HistoryTrustedSourceBundle,
    *,
    context: _TrustedGenerationContext | None = None,
) -> bool:
    if type(library) is not FrozenHistoryLibrary or type(bundle) is not HistoryTrustedSourceBundle:
        return False
    _preflight_trusted_source_bundle(bundle)
    try:
        if context is None:
            trusted_context = _build_trusted_generation_context(
                bundle.generation_authority,
                expected_generation_authority_sha256=(bundle.expected_generation_authority_sha256),
                canonical_batch_manifest=bundle.canonical_batch_manifest,
            )
        else:
            trusted_context = context
            if (
                type(trusted_context) is not _TrustedGenerationContext
                or bundle.generation_authority != trusted_context.generation_authority
                or not hmac.compare_digest(
                    bundle.expected_generation_authority_sha256,
                    trusted_context.expected_generation_authority_sha256,
                )
                or not hmac.compare_digest(
                    sha256_hex(bundle.canonical_batch_manifest),
                    trusted_context.manifest_sha256,
                )
            ):
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    manifest = trusted_context.manifest
    source_ids = tuple(item.entry_id for item in bundle.entry_sources)
    entry_ids = tuple(item.entry_id for item in library.entries)
    manifest_entry_keys = tuple(item.manifest_entry_key for item in manifest.entries)
    library_entry_keys = tuple(item.manifest_entry_key for item in library.entries)
    if (
        len(source_ids) != len(set(source_ids))
        or tuple(sorted(source_ids)) != tuple(sorted(entry_ids))
        or tuple(sorted(manifest_entry_keys)) != tuple(sorted(library_entry_keys))
    ):
        return False
    source_by_id = {item.entry_id: item for item in bundle.entry_sources}
    return (
        FrozenHistoryLibrary.verify(library)
        and library.dataset == manifest.dataset
        and library.source_partition == manifest.source_partition
        and library.dataset == bundle.generation_authority.dataset
        and library.source_partition == bundle.generation_authority.source_partition
        and hmac.compare_digest(
            library.generation_authority_sha256,
            bundle.generation_authority.generation_authority_sha256,
        )
        and all(
            _verify_trusted_library_entry(
                entry,
                source_by_id[entry.entry_id],
                context=trusted_context,
            )
            for entry in library.entries
        )
    )


class HistoryQueryEmbedding(StrictFrozenModel):
    """Content-addressed current-case embedding used only for frozen retrieval."""

    schema_version: Literal["nurse.v2.history-query-embedding/1"] = (
        HISTORY_QUERY_EMBEDDING_SCHEMA_VERSION
    )
    current_opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    current_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: DatasetName
    current_decision_time_unix_ns: int = Field(ge=0)
    embedding_policy: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1, le=_MAX_VECTOR_DIMENSION)
    embedding: tuple[float, ...] = Field(min_length=1, max_length=_MAX_VECTOR_DIMENSION)
    query_embedding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_finite_nonzero_embedding(self) -> Self:
        _require_history_identifier(
            self.embedding_policy,
            label="history embedding policy",
        )
        _require_history_identifier(
            self.embedding_model,
            label="history embedding model",
        )
        if self.embedding_dimension != len(self.embedding) or not _is_finite_nonzero_embedding(
            self.embedding
        ):
            raise ValueError("history query embedding must be finite and nonzero")
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryQueryEmbedding,
            id_field=None,
            id_prefix=None,
            sha_field="query_embedding_sha256",
        )


class HistoryServiceMapping(StrictFrozenModel):
    """Deterministic source-to-current service binding for one history entry."""

    schema_version: Literal["nurse.v2.history-service-mapping/1"] = (
        HISTORY_SERVICE_MAPPING_SCHEMA_VERSION
    )
    mapping_id: str = Field(pattern=r"^hmap_[0-9a-f]{24}$")
    entry_id: str = Field(pattern=r"^hent_[0-9a-f]{24}$")
    source_service_key: str = Field(min_length=1)
    current_service_key: str = Field(min_length=1)
    mapping_policy: str = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_PROVENANCE_IDS,
    )
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_provenance(self) -> Self:
        _require_history_identifier(
            self.source_service_key,
            label="history source service",
        )
        _require_history_identifier(
            self.current_service_key,
            label="history current service",
        )
        _require_history_identifier(
            self.mapping_policy,
            label="history mapping policy",
        )
        if self.provenance_ids != tuple(sorted(set(self.provenance_ids))):
            raise ValueError("history mapping provenance must be sorted and unique")
        if any(
            _HISTORY_MAPPING_PROVENANCE_ID_V1.fullmatch(item) is None
            for item in self.provenance_ids
        ):
            raise ValueError("history mapping requires opaque provenance ID grammar v1")
        if _has_forbidden_history_metadata(self.mapping_policy) or any(
            _has_forbidden_history_metadata(item) for item in self.provenance_ids
        ):
            raise ValueError("history mapping contains forbidden metadata")
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryServiceMapping,
            id_field="mapping_id",
            id_prefix="hmap",
            sha_field="mapping_sha256",
        )


class HistoryScoringConfig(StrictFrozenModel):
    """Content-addressed retrieval policy with explicit time claim semantics."""

    schema_version: Literal["nurse.v2.history-scoring-config/1"] = (
        HISTORY_SCORING_CONFIG_SCHEMA_VERSION
    )
    k: int = Field(default=3, ge=1, le=_MAX_HISTORY_ENTRIES)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0, allow_inf_nan=False)
    time_policy: HistoryTimePolicy = "strict_prior"
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryScoringConfig,
            id_field=None,
            id_prefix=None,
            sha_field="config_sha256",
        )


def build_history_generation_authority(
    *,
    dataset: DatasetName,
    source_partition: HistorySourcePartition,
    generation_policy: str,
    canonical_batch_manifest: bytes,
    source_history_mode: HistoryMode = "off",
    source_rule_channel_enabled: bool = False,
) -> HistoryGenerationAuthority:
    """Anchor one nonrecursive generation batch to complete canonical manifest bytes."""

    try:
        manifest, manifest_sha256 = _parse_canonical_typed_document(
            canonical_batch_manifest,
            HistoryGenerationBatchManifest,
        )
        if (
            dataset != manifest.dataset
            or source_partition != manifest.source_partition
            or generation_policy != manifest.generation_policy
            or source_history_mode != manifest.source_history_mode
            or source_rule_channel_enabled != manifest.source_rule_channel_enabled
        ):
            raise ValueError("history generation claims do not match the typed manifest")
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_GENERATION_AUTHORITY_SCHEMA_VERSION,
            "dataset": manifest.dataset,
            "source_partition": manifest.source_partition,
            "source_history_mode": manifest.source_history_mode,
            "source_rule_channel_enabled": manifest.source_rule_channel_enabled,
            "generation_policy": manifest.generation_policy,
            "canonical_batch_manifest_sha256": manifest_sha256,
        }
        identifier, digest = _stable_id("hgauth", unsigned)
        authority = HistoryGenerationAuthority.model_validate(
            {
                **unsigned,
                "generation_authority_id": identifier,
                "generation_authority_sha256": digest,
            }
        )
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history generation authority input verification failed") from None
    if not HistoryGenerationAuthority.verify(authority):
        raise ValueError("history generation authority input verification failed")
    return authority


def build_history_base_candidate_score(
    *,
    source_service_key: str,
    base_history_score: float,
    base_score_artifact: bytes,
    generation_authority: HistoryGenerationAuthority,
    expected_generation_authority_sha256: object = None,
    canonical_batch_manifest: object = None,
    source_history_score: float = 0.0,
    source_rule_score: float = 0.0,
) -> HistoryBaseCandidateScore:
    """Freeze one source score while proving recursive H/R channels are disabled."""

    try:
        authority = _trusted_exact_model(
            generation_authority,
            HistoryGenerationAuthority,
            label="history generation authority",
        )
        context = _build_trusted_generation_context(
            authority,
            expected_generation_authority_sha256=cast(str, expected_generation_authority_sha256),
            canonical_batch_manifest=cast(bytes, canonical_batch_manifest),
        )
        trusted_source = HistoryTrustedBaseScoreSource(
            source_service_key=source_service_key,
            base_score_artifact=base_score_artifact,
        )
        artifact, artifact_sha256 = _parse_canonical_typed_document(
            base_score_artifact,
            HistorySanitizedBaseScoreArtifact,
        )
        membership = context.base_score_memberships.get(
            (
                artifact.manifest_entry_key,
                artifact.manifest_base_score_key,
                source_service_key,
            )
        )
        if membership is None:
            raise ValueError("history score artifact does not select one manifest membership")
        if (
            base_history_score != membership.base_history_score
            or source_history_score != membership.source_history_score
            or source_rule_score != membership.source_rule_score
            or artifact.manifest_entry_key != membership.manifest_entry_key
            or artifact.manifest_base_score_key != membership.manifest_base_score_key
            or artifact.source_service_key != membership.source_service_key
            or artifact.base_history_score != membership.base_history_score
            or artifact.source_history_score != membership.source_history_score
            or artifact.source_rule_score != membership.source_rule_score
            or not hmac.compare_digest(membership.score_artifact_sha256, artifact_sha256)
        ):
            raise ValueError("history base score does not match its manifest membership")
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_BASE_CANDIDATE_SCORE_SCHEMA_VERSION,
            "manifest_entry_key": membership.manifest_entry_key,
            "manifest_base_score_key": membership.manifest_base_score_key,
            "source_service_key": membership.source_service_key,
            "base_history_score": membership.base_history_score,
            "source_history_score": membership.source_history_score,
            "source_rule_score": membership.source_rule_score,
            "generation_authority_sha256": authority.generation_authority_sha256,
            "base_score_artifact_sha256": artifact_sha256,
        }
        identifier, digest = _stable_id("hbcs", unsigned)
        result = HistoryBaseCandidateScore.model_validate(
            {
                **unsigned,
                "base_candidate_score_id": identifier,
                "base_candidate_score_sha256": digest,
            }
        )
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history base score input verification failed") from None
    if not _verify_trusted_base_candidate_score(
        result,
        trusted_source,
        context=context,
        parsed_artifact=(artifact, artifact_sha256),
    ):
        raise ValueError("history base score input verification failed")
    return result


def build_history_library_entry(
    *,
    dataset: DatasetName,
    source_partition: HistorySourcePartition,
    source_opaque_case_id: str,
    source_group_fingerprint: str,
    source_completed_at_unix_ns: int,
    embedding_policy: str,
    embedding_model: str,
    embedding: Sequence[float],
    sanitized_semantic_payload: bytes,
    sanitized_graph: bytes,
    sanitized_topology: bytes,
    base_candidate_scores: Sequence[HistoryBaseCandidateScore],
    generation_authority: HistoryGenerationAuthority,
    expected_generation_authority_sha256: object = None,
    canonical_batch_manifest: object = None,
    trusted_base_score_sources: object = None,
    source_history_mode: HistoryMode = "off",
    source_rule_channel_enabled: bool = False,
    label_free: bool = True,
    contains_answers: bool = False,
    contains_fault_metadata: bool = False,
    contains_injection_metadata: bool = False,
    contains_original_identifiers_or_paths: bool = False,
) -> HistoryLibraryEntry:
    """Build one sanitized history entry without labels or recursive channels."""

    try:
        authority = _trusted_exact_model(
            generation_authority,
            HistoryGenerationAuthority,
            label="history generation authority",
        )
        context = _build_trusted_generation_context(
            authority,
            expected_generation_authority_sha256=cast(
                str,
                expected_generation_authority_sha256,
            ),
            canonical_batch_manifest=cast(bytes, canonical_batch_manifest),
        )
        if authority.dataset != dataset or authority.source_partition != source_partition:
            raise ValueError("history generation authority does not match entry")
        membership = context.entry_memberships.get(
            _expected_manifest_entry_key(
                dataset=dataset,
                source_partition=source_partition,
                source_opaque_case_id=source_opaque_case_id,
            )
        )
        if membership is None:
            raise ValueError("history entry does not select one manifest membership")
        if type(trusted_base_score_sources) is not tuple or any(
            type(item) is not HistoryTrustedBaseScoreSource
            for item in cast(tuple[object, ...], trusted_base_score_sources)
        ):
            raise ValueError("history trusted base score sources must use an exact tuple")
        vector = _plain_embedding(embedding)
        score_values = _plain_sequence(
            base_candidate_scores,
            label="history base scores",
            maximum=_MAX_HISTORY_BASE_SCORES,
        )
        if not score_values or any(
            type(item) is not HistoryBaseCandidateScore for item in score_values
        ):
            raise ValueError("history base score input verification failed")
        scores = tuple(cast(HistoryBaseCandidateScore, item) for item in score_values)
        for item in scores:
            _preflight_exact_tree(
                item,
                root_model_types=(HistoryBaseCandidateScore,),
                label="history base score",
            )
        if any(not HistoryBaseCandidateScore.verify(item) for item in scores):
            raise ValueError("history base score input verification failed")
        if any(
            item.generation_authority_sha256 != authority.generation_authority_sha256
            for item in scores
        ):
            raise ValueError("history base score generation authority mismatch")
        semantic, semantic_sha256 = _parse_canonical_typed_document(
            sanitized_semantic_payload,
            HistorySanitizedSemanticArtifact,
        )
        graph, graph_sha256 = _parse_canonical_typed_document(
            sanitized_graph,
            HistorySanitizedGraphArtifact,
        )
        topology, topology_sha256 = _parse_canonical_typed_document(
            sanitized_topology,
            HistorySanitizedTopologyArtifact,
        )
        manifest_score_keys = tuple(item.manifest_base_score_key for item in membership.base_scores)
        persisted_score_keys = tuple(sorted(item.manifest_base_score_key for item in scores))
        expected_services = tuple(item.source_service_key for item in membership.base_scores)
        if (
            dataset != membership.dataset
            or source_partition != membership.source_partition
            or source_group_fingerprint != membership.source_group_fingerprint
            or source_completed_at_unix_ns != membership.source_completed_at_unix_ns
            or embedding_policy != membership.embedding_policy
            or embedding_model != membership.embedding_model
            or vector != membership.embedding
            or source_history_mode != membership.source_history_mode
            or source_rule_channel_enabled != membership.source_rule_channel_enabled
            or label_free is not True
            or contains_answers is not False
            or contains_fault_metadata is not False
            or contains_injection_metadata is not False
            or contains_original_identifiers_or_paths is not False
            or tuple(sorted(manifest_score_keys)) != persisted_score_keys
            or semantic.manifest_entry_key != membership.manifest_entry_key
            or semantic.source_opaque_case_id != membership.source_opaque_case_id
            or semantic.service_keys != expected_services
            or graph.manifest_entry_key != membership.manifest_entry_key
            or graph.source_opaque_case_id != membership.source_opaque_case_id
            or graph.service_keys != expected_services
            or topology.manifest_entry_key != membership.manifest_entry_key
            or topology.source_opaque_case_id != membership.source_opaque_case_id
            or topology.service_keys != expected_services
            or not hmac.compare_digest(membership.semantic_artifact_sha256, semantic_sha256)
            or not hmac.compare_digest(membership.graph_artifact_sha256, graph_sha256)
            or not hmac.compare_digest(membership.topology_artifact_sha256, topology_sha256)
        ):
            raise ValueError("history entry does not match its manifest membership")
        score_by_manifest_key = {item.manifest_base_score_key: item for item in scores}
        if len(score_by_manifest_key) != len(scores):
            raise ValueError("history base score manifest memberships must be unique")
        ordered_scores = tuple(score_by_manifest_key[key] for key in manifest_score_keys)
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_LIBRARY_ENTRY_SCHEMA_VERSION,
            "manifest_entry_key": membership.manifest_entry_key,
            "dataset": membership.dataset,
            "source_partition": membership.source_partition,
            "source_opaque_case_id": membership.source_opaque_case_id,
            "source_group_fingerprint": membership.source_group_fingerprint,
            "source_completed_at_unix_ns": membership.source_completed_at_unix_ns,
            "source_history_mode": membership.source_history_mode,
            "source_rule_channel_enabled": membership.source_rule_channel_enabled,
            "label_free": True,
            "contains_answers": False,
            "contains_fault_metadata": False,
            "contains_injection_metadata": False,
            "contains_original_identifiers_or_paths": False,
            "generation_authority_sha256": authority.generation_authority_sha256,
            "sanitized_semantic_payload_sha256": semantic_sha256,
            "sanitized_graph_sha256": graph_sha256,
            "sanitized_topology_sha256": topology_sha256,
            "embedding_policy": membership.embedding_policy,
            "embedding_model": membership.embedding_model,
            "embedding_dimension": membership.embedding_dimension,
            "embedding": membership.embedding,
            "base_candidate_scores": tuple(item.model_dump(mode="json") for item in ordered_scores),
        }
        identifier, digest = _stable_id("hent", unsigned)
        entry = HistoryLibraryEntry.model_validate(
            {**unsigned, "entry_id": identifier, "entry_sha256": digest}
        )
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history library entry input verification failed") from None
    trusted_entry_source = HistoryTrustedEntrySource(
        entry_id=entry.entry_id,
        base_score_sources=cast(
            tuple[HistoryTrustedBaseScoreSource, ...], trusted_base_score_sources
        ),
        sanitized_semantic_payload=sanitized_semantic_payload,
        sanitized_graph=sanitized_graph,
        sanitized_topology=sanitized_topology,
    )
    if not _verify_trusted_library_entry(
        entry,
        trusted_entry_source,
        context=context,
        parsed_artifacts=(
            (semantic, semantic_sha256),
            (graph, graph_sha256),
            (topology, topology_sha256),
        ),
    ):
        raise ValueError("history library entry input verification failed")
    return entry


def build_frozen_history_library(
    *,
    dataset: DatasetName,
    source_partition: HistorySourcePartition,
    embedding_policy: str,
    embedding_model: str,
    embedding_dimension: int,
    entries: Sequence[HistoryLibraryEntry],
    generation_authority: HistoryGenerationAuthority,
    trusted_source_bundle: object = None,
    read_only: bool = True,
) -> FrozenHistoryLibrary:
    """Freeze a dataset-specific library in stable entry-ID order."""

    try:
        authority = _trusted_exact_model(
            generation_authority,
            HistoryGenerationAuthority,
            label="history generation authority",
        )
        if type(trusted_source_bundle) is not HistoryTrustedSourceBundle:
            raise ValueError("history trusted source bundle must use the exact type")
        _preflight_trusted_source_bundle(trusted_source_bundle)
        context = _build_trusted_generation_context(
            authority,
            expected_generation_authority_sha256=(
                trusted_source_bundle.expected_generation_authority_sha256
            ),
            canonical_batch_manifest=(trusted_source_bundle.canonical_batch_manifest),
        )
        manifest = context.manifest
        manifest_embedding = manifest.entries[0]
        if (
            trusted_source_bundle.generation_authority != authority
            or authority.dataset != manifest.dataset
            or authority.source_partition != manifest.source_partition
            or dataset != manifest.dataset
            or source_partition != manifest.source_partition
            or embedding_policy != manifest_embedding.embedding_policy
            or embedding_model != manifest_embedding.embedding_model
            or embedding_dimension != manifest_embedding.embedding_dimension
        ):
            raise ValueError("history generation authority does not match library")
        entry_values = _plain_sequence(
            entries,
            label="history entries",
            maximum=_MAX_HISTORY_ENTRIES,
        )
        if not entry_values or any(type(item) is not HistoryLibraryEntry for item in entry_values):
            raise ValueError("history entry input verification failed")
        typed = tuple(cast(HistoryLibraryEntry, item) for item in entry_values)
        _preflight_exact_tree(
            typed,
            root_model_types=(HistoryLibraryEntry,),
            label="history library entries",
        )
        if any(not HistoryLibraryEntry.verify(item) for item in typed):
            raise ValueError("history entry input verification failed")
        if any(
            item.generation_authority_sha256 != authority.generation_authority_sha256
            for item in typed
        ):
            raise ValueError("history entry generation authority mismatch")
        manifest_entry_keys = tuple(item.manifest_entry_key for item in manifest.entries)
        persisted_entry_keys = tuple(item.manifest_entry_key for item in typed)
        if tuple(sorted(persisted_entry_keys)) != manifest_entry_keys:
            raise ValueError("history entries do not match exact manifest membership")
        ordered = tuple(sorted(typed, key=lambda item: item.entry_id))
        unsigned: dict[str, object] = {
            "schema_version": FROZEN_HISTORY_LIBRARY_SCHEMA_VERSION,
            "dataset": manifest.dataset,
            "source_partition": manifest.source_partition,
            "embedding_policy": manifest_embedding.embedding_policy,
            "embedding_model": manifest_embedding.embedding_model,
            "embedding_dimension": manifest_embedding.embedding_dimension,
            "generation_authority_sha256": authority.generation_authority_sha256,
            "read_only": read_only,
            "entries": tuple(item.model_dump(mode="json") for item in ordered),
        }
        library = FrozenHistoryLibrary.model_validate(
            {**unsigned, "library_sha256": _digest(unsigned)}
        )
        if not _verify_trusted_source_bundle(
            library,
            trusted_source_bundle,
            context=context,
        ):
            raise ValueError("history library input verification failed")
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history library input verification failed") from None
    return library


def build_history_query_embedding(
    *,
    current_opaque_case_id: str,
    current_group_fingerprint: str,
    dataset: DatasetName,
    current_decision_time_unix_ns: int,
    embedding_policy: str,
    embedding_model: str,
    embedding_dimension: int,
    embedding: Sequence[float],
) -> HistoryQueryEmbedding:
    """Freeze the current-case embedding and its retrieval authority."""

    try:
        vector = _plain_embedding(embedding)
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_QUERY_EMBEDDING_SCHEMA_VERSION,
            "current_opaque_case_id": current_opaque_case_id,
            "current_group_fingerprint": current_group_fingerprint,
            "dataset": dataset,
            "current_decision_time_unix_ns": current_decision_time_unix_ns,
            "embedding_policy": _require_history_identifier(
                embedding_policy, label="history embedding policy"
            ),
            "embedding_model": _require_history_identifier(
                embedding_model, label="history embedding model"
            ),
            "embedding_dimension": embedding_dimension,
            "embedding": vector,
        }
        query = HistoryQueryEmbedding.model_validate(
            {**unsigned, "query_embedding_sha256": _digest(unsigned)}
        )
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history query input verification failed") from None
    if not HistoryQueryEmbedding.verify(query):
        raise ValueError("history query input verification failed")
    return query


def build_history_service_mapping(
    *,
    entry_id: str,
    source_service_key: str,
    current_service_key: str,
    mapping_policy: str,
    provenance_ids: Sequence[str],
) -> HistoryServiceMapping:
    """Build one deterministic mapping into current frozen-C service space."""

    try:
        provenance_values = _plain_sequence(
            provenance_ids,
            label="history mapping provenance",
            maximum=_MAX_PROVENANCE_IDS,
        )
        if not provenance_values or any(type(item) is not str for item in provenance_values):
            raise ValueError("history mapping provenance input verification failed")
        provenance = tuple(sorted(cast(str, item) for item in provenance_values))
        if any(not item or len(item) > _MAX_SERIALIZED_STRING_LENGTH for item in provenance):
            raise ValueError("history mapping provenance input verification failed")
        mapping_policy_value = _require_history_identifier(
            mapping_policy,
            label="history mapping policy",
        )
        if _has_forbidden_history_metadata(mapping_policy_value) or any(
            _has_forbidden_history_metadata(item) for item in provenance
        ):
            raise ValueError("history mapping contains forbidden metadata")
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_SERVICE_MAPPING_SCHEMA_VERSION,
            "entry_id": entry_id,
            "source_service_key": _require_history_identifier(
                source_service_key, label="history source service"
            ),
            "current_service_key": _require_history_identifier(
                current_service_key, label="history current service"
            ),
            "mapping_policy": mapping_policy_value,
            "provenance_ids": provenance,
        }
        identifier, digest = _stable_id("hmap", unsigned)
        mapping = HistoryServiceMapping.model_validate(
            {**unsigned, "mapping_id": identifier, "mapping_sha256": digest}
        )
    except _HistoryScoringResourceLimitError:
        raise HistoryScoringBudgetExceeded() from None
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history service mapping input verification failed") from None
    if not HistoryServiceMapping.verify(mapping):
        raise ValueError("history service mapping input verification failed")
    return mapping


def build_history_scoring_config(
    *,
    k: int = 3,
    min_similarity: float = 0.7,
    time_policy: HistoryTimePolicy = "strict_prior",
) -> HistoryScoringConfig:
    """Build an explicit content-addressed retrieval configuration."""

    try:
        unsigned: dict[str, object] = {
            "schema_version": HISTORY_SCORING_CONFIG_SCHEMA_VERSION,
            "k": k,
            "min_similarity": min_similarity,
            "time_policy": time_policy,
        }
        config = HistoryScoringConfig.model_validate(
            {**unsigned, "config_sha256": _digest(unsigned)}
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history scoring config input verification failed") from None
    if not HistoryScoringConfig.verify(config):
        raise ValueError("history scoring config input verification failed")
    return config


class HistoryScoreContribution(StrictFrozenModel):
    """One selected neighbor's explicit contribution to one current candidate."""

    schema_version: Literal["nurse.v2.history-score-contribution/1"] = (
        HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION
    )
    contribution_id: str = Field(pattern=r"^hcon_[0-9a-f]{24}$")
    selected_neighbor_id: str = Field(pattern=r"^hent_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    mapping_id: str | None = Field(default=None, pattern=r"^hmap_[0-9a-f]{24}$")
    source_service_key: str | None = Field(default=None, min_length=1)
    mapping_status: HistoryMappingStatus
    similarity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    normalized_similarity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    base_history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    contribution: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    contribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_exact_mapping_contribution(self) -> Self:
        _require_history_identifier(
            self.candidate_service_key,
            label="history candidate service",
        )
        if self.source_service_key is not None:
            _require_history_identifier(
                self.source_service_key,
                label="history source service",
            )
        if self.mapping_status == "missing_mapping":
            if (
                self.mapping_id is not None
                or self.source_service_key is not None
                or self.base_history_score != 0.0
                or self.contribution != 0.0
            ):
                raise ValueError("missing history mapping cannot inject a score")
        elif self.mapping_id is None or self.source_service_key is None:
            raise ValueError("mapped history contribution requires mapping authority")
        expected = self.normalized_similarity * self.base_history_score
        if not math.isclose(self.contribution, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("history contribution does not match its explicit inputs")
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryScoreContribution,
            id_field="contribution_id",
            id_prefix="hcon",
            sha_field="contribution_sha256",
        )


class CandidateHistoryScore(StrictFrozenModel):
    """Complete history-channel score for one member of frozen C."""

    schema_version: Literal["nurse.v2.candidate-history-score/1"] = (
        CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION
    )
    candidate_history_score_id: str = Field(pattern=r"^chs_[0-9a-f]{24}$")
    candidate_service_key: str = Field(min_length=1)
    contributions: tuple[HistoryScoreContribution, ...] = Field(
        max_length=_MAX_HISTORY_ENTRIES,
    )
    history_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_history_score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_exact_candidate_total(self) -> Self:
        _require_history_identifier(
            self.candidate_service_key,
            label="history candidate service",
        )
        if any(
            item.candidate_service_key != self.candidate_service_key for item in self.contributions
        ):
            raise ValueError("history contribution belongs to another candidate")
        if len({item.contribution_id for item in self.contributions}) != len(
            self.contributions
        ) or len({item.contribution_sha256 for item in self.contributions}) != len(
            self.contributions
        ):
            raise ValueError("history contributions must have unique identities and hashes")
        neighbor_ids = tuple(item.selected_neighbor_id for item in self.contributions)
        if len(neighbor_ids) != len(set(neighbor_ids)):
            raise ValueError("candidate history contributions must be unique per neighbor")
        expected = math.fsum(item.contribution for item in self.contributions)
        if not math.isclose(self.history_score, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("candidate history score does not match contributions")
        return self

    def verify(self) -> bool:
        return all(HistoryScoreContribution.verify(item) for item in self.contributions) and (
            _verify_content_address(
                self,
                CandidateHistoryScore,
                id_field="candidate_history_score_id",
                id_prefix="chs",
                sha_field="candidate_history_score_sha256",
            )
        )


class HistoryNeighborAudit(StrictFrozenModel):
    """Selection disposition retained for every frozen library entry."""

    schema_version: Literal["nurse.v2.history-neighbor-audit/1"] = (
        HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION
    )
    audit_id: str = Field(pattern=r"^haud_[0-9a-f]{24}$")
    entry_id: str = Field(pattern=r"^hent_[0-9a-f]{24}$")
    source_group_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: HistoryNeighborStatus
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    normalized_similarity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_status_values(self) -> Self:
        similarity_excluded = self.status in {
            "self_excluded",
            "incident_group_excluded",
            "time_excluded",
        }
        if similarity_excluded != (self.similarity is None):
            raise ValueError("history audit status does not match similarity availability")
        if self.status != "selected" and self.normalized_similarity != 0.0:
            raise ValueError("unselected history entry cannot receive normalized similarity")
        if self.status == "selected" and (self.similarity is None or self.similarity < 0.0):
            raise ValueError("selected history entry requires nonnegative similarity")
        return self

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryNeighborAudit,
            id_field="audit_id",
            id_prefix="haud",
            sha_field="audit_sha256",
        )


class HistoryHintReference(StrictFrozenModel):
    """Sanitized digest-only semantic reference for current-case reverification."""

    schema_version: Literal["nurse.v2.history-hint-reference/1"] = (
        HISTORY_HINT_REFERENCE_SCHEMA_VERSION
    )
    hint_reference_id: str = Field(pattern=r"^hhint_[0-9a-f]{24}$")
    selected_neighbor_id: str = Field(pattern=r"^hent_[0-9a-f]{24}$")
    semantic_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requires_current_case_reverification: Literal[True] = True
    hint_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        return _verify_content_address(
            self,
            HistoryHintReference,
            id_field="hint_reference_id",
            id_prefix="hhint",
            sha_field="hint_reference_sha256",
        )


class HistoryEvaluationResult(StrictFrozenModel):
    """Immutable history result with a structural verification entry point."""

    schema_version: Literal["nurse.v2.history-evaluation-result/1"] = (
        HISTORY_EVALUATION_RESULT_SCHEMA_VERSION
    )
    history_evaluation_result_id: str = Field(pattern=r"^heval_[0-9a-f]{24}$")
    mode: HistoryMode
    claim_scope: HistoryClaimScope
    candidate_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1, le=_MAX_HISTORY_CANDIDATES)
    candidate_service_keys: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_CANDIDATES,
    )
    candidate_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    library_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_embedding_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    service_mapping_sha256s: tuple[str, ...] = Field(
        max_length=_MAX_HISTORY_MAPPINGS,
    )
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    neighbor_audits: tuple[HistoryNeighborAudit, ...] = Field(
        max_length=_MAX_HISTORY_ENTRIES,
    )
    selected_neighbor_ids: tuple[str, ...] = Field(
        max_length=_MAX_HISTORY_ENTRIES,
    )
    hint_references: tuple[HistoryHintReference, ...] = Field(
        max_length=_MAX_HISTORY_ENTRIES,
    )
    requires_current_case_reverification: bool
    candidate_scores: tuple[CandidateHistoryScore, ...] = Field(
        min_length=1,
        max_length=_MAX_HISTORY_CANDIDATES,
    )
    authoritative_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_mode_isolation(self) -> Self:
        for candidate_service_key in self.candidate_service_keys:
            _require_history_identifier(
                candidate_service_key,
                label="history candidate service",
            )
        if self.neighbor_audits != tuple(
            sorted(self.neighbor_audits, key=lambda item: item.entry_id)
        ):
            raise ValueError("history neighbor audits must be canonically ordered")
        if self.service_mapping_sha256s != tuple(sorted(set(self.service_mapping_sha256s))):
            raise ValueError("history service mapping digests must be sorted and unique")
        if len({item.audit_id for item in self.neighbor_audits}) != len(
            self.neighbor_audits
        ) or len({item.audit_sha256 for item in self.neighbor_audits}) != len(self.neighbor_audits):
            raise ValueError("history neighbor audits must have unique identities and hashes")
        if len({item.entry_id for item in self.neighbor_audits}) != len(self.neighbor_audits):
            raise ValueError("history neighbor audits must have unique entries")
        if len(self.selected_neighbor_ids) != len(set(self.selected_neighbor_ids)):
            raise ValueError("history selected neighbor IDs must be unique")
        if len({item.hint_reference_id for item in self.hint_references}) != len(
            self.hint_references
        ) or len({item.hint_reference_sha256 for item in self.hint_references}) != len(
            self.hint_references
        ):
            raise ValueError("history hint references must have unique identities and hashes")
        expected_selected = tuple(
            item.entry_id
            for item in sorted(
                (audit for audit in self.neighbor_audits if audit.status == "selected"),
                key=lambda item: (
                    -(item.similarity if item.similarity is not None else -1.0),
                    item.entry_id,
                ),
            )
        )
        if self.selected_neighbor_ids != expected_selected:
            raise ValueError("history selected neighbors do not match entry audits")
        selected_audits = tuple(
            audit for audit in self.neighbor_audits if audit.status == "selected"
        )
        if len({audit.source_group_fingerprint for audit in selected_audits}) != len(
            selected_audits
        ):
            raise ValueError("selected history neighbors must use distinct source groups")
        selected_similarity_total = math.fsum(
            max(audit.similarity if audit.similarity is not None else 0.0, 0.0)
            for audit in selected_audits
        )
        for audit in selected_audits:
            similarity = max(audit.similarity if audit.similarity is not None else 0.0, 0.0)
            expected_normalized = (
                similarity / selected_similarity_total if selected_similarity_total > 0.0 else 0.0
            )
            if not math.isclose(
                audit.normalized_similarity,
                expected_normalized,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("history normalized similarity does not match selected audits")
        if self.candidate_scores != tuple(
            sorted(self.candidate_scores, key=lambda item: item.candidate_service_key)
        ) or len({item.candidate_service_key for item in self.candidate_scores}) != len(
            self.candidate_scores
        ):
            raise ValueError("history candidate scores must be canonically complete")
        if len({item.candidate_history_score_id for item in self.candidate_scores}) != len(
            self.candidate_scores
        ) or len({item.candidate_history_score_sha256 for item in self.candidate_scores}) != len(
            self.candidate_scores
        ):
            raise ValueError("history candidate scores must have unique identities and hashes")
        if (
            self.candidate_count != len(self.candidate_service_keys)
            or self.candidate_service_keys != tuple(sorted(set(self.candidate_service_keys)))
            or tuple(item.candidate_service_key for item in self.candidate_scores)
            != self.candidate_service_keys
        ):
            raise ValueError("history candidate coverage does not match frozen candidates")
        if self.mode == "off" and (
            self.claim_scope != "history_disabled"
            or self.generation_authority_sha256 is not None
            or self.library_sha256 is not None
            or self.query_embedding_sha256 is not None
            or self.service_mapping_sha256s
            or self.neighbor_audits
            or self.selected_neighbor_ids
            or self.hint_references
            or self.requires_current_case_reverification
            or any(item.history_score != 0.0 for item in self.candidate_scores)
        ):
            raise ValueError("history-off result cannot consume history")
        if self.mode != "off" and (
            self.generation_authority_sha256 is None
            or self.library_sha256 is None
            or self.query_embedding_sha256 is None
        ):
            raise ValueError("enabled history result requires frozen input digests")
        if self.mode == "hint" and (
            self.claim_scope == "history_disabled"
            or not self.requires_current_case_reverification
            or tuple(item.selected_neighbor_id for item in self.hint_references)
            != self.selected_neighbor_ids
            or any(
                item.history_score != 0.0 or item.contributions for item in self.candidate_scores
            )
        ):
            raise ValueError("history hint result cannot inject scores")
        if self.mode == "score" and (
            self.claim_scope == "history_disabled"
            or self.requires_current_case_reverification
            or self.hint_references
            or any(
                tuple(contribution.selected_neighbor_id for contribution in item.contributions)
                != self.selected_neighbor_ids
                for item in self.candidate_scores
            )
        ):
            raise ValueError("history score result cannot expose hints")
        audit_by_entry = {item.entry_id: item for item in selected_audits}
        if self.mode == "score" and any(
            contribution.selected_neighbor_id not in audit_by_entry
            or not math.isclose(
                contribution.similarity,
                audit_by_entry[contribution.selected_neighbor_id].similarity or 0.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                contribution.normalized_similarity,
                audit_by_entry[contribution.selected_neighbor_id].normalized_similarity,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for score in self.candidate_scores
            for contribution in score.contributions
        ):
            raise ValueError("history contribution does not match its selected neighbor audit")
        return self

    def verify_structure(self) -> bool:
        """Verify the content address and the mode-local structural invariants."""

        try:
            if type(self) is not HistoryEvaluationResult:
                return False
            _preflight_exact_tree(
                self,
                root_model_types=(HistoryEvaluationResult,),
                label="history evaluation result",
            )
            rebuilt = HistoryEvaluationResult.model_validate(
                HistoryEvaluationResult.model_dump(self, mode="json", warnings="error")
            )
            if not _same_exact_runtime_shape(self, rebuilt):
                return False
            expected_candidate_binding_sha256 = _candidate_binding_sha256(
                candidate_freeze_sha256=rebuilt.candidate_freeze_sha256,
                candidate_universe_sha256=rebuilt.candidate_universe_sha256,
                candidate_count=rebuilt.candidate_count,
                candidate_service_keys=rebuilt.candidate_service_keys,
            )
            expected_authoritative_input_sha256 = _authoritative_input_sha256(
                candidate_binding_sha256=rebuilt.candidate_binding_sha256,
                mode=rebuilt.mode,
                generation_authority_sha256=rebuilt.generation_authority_sha256,
                library_sha256=rebuilt.library_sha256,
                query_embedding_sha256=rebuilt.query_embedding_sha256,
                service_mapping_sha256s=rebuilt.service_mapping_sha256s,
                config_sha256=rebuilt.config_sha256,
            )
            unsigned = HistoryEvaluationResult.model_dump(
                rebuilt,
                mode="json",
                exclude={"history_evaluation_result_id", "result_sha256"},
            )
            identifier, digest = _stable_id("heval", unsigned)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            rebuilt == self
            and all(HistoryNeighborAudit.verify(item) for item in rebuilt.neighbor_audits)
            and all(HistoryHintReference.verify(item) for item in rebuilt.hint_references)
            and all(CandidateHistoryScore.verify(item) for item in rebuilt.candidate_scores)
            and hmac.compare_digest(
                rebuilt.candidate_binding_sha256,
                expected_candidate_binding_sha256,
            )
            and hmac.compare_digest(
                rebuilt.authoritative_input_sha256,
                expected_authoritative_input_sha256,
            )
            and rebuilt.history_evaluation_result_id == identifier
            and rebuilt.result_sha256 == digest
        )

    def verify(self) -> bool:
        return HistoryEvaluationResult.verify_structure(self)

    def verify_trusted(
        self,
        *,
        expected_result_sha256: str,
        expected_authoritative_input_sha256: str,
        frozen_candidates: FrozenCandidateUniverse,
        mode: HistoryMode = "off",
        library: FrozenHistoryLibrary | None = None,
        query_embedding: HistoryQueryEmbedding | None = None,
        service_mappings: Sequence[HistoryServiceMapping] = (),
        config: HistoryScoringConfig | None = None,
        trusted_source_bundle: HistoryTrustedSourceBundle | None = None,
    ) -> bool:
        """Replay from the full authoritative inputs under two external anchors."""

        if type(self) is not HistoryEvaluationResult or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                expected_result_sha256,
                expected_authoritative_input_sha256,
            )
        ):
            return False
        try:
            if (
                not HistoryEvaluationResult.verify_structure(self)
                or not hmac.compare_digest(self.result_sha256, expected_result_sha256)
                or not hmac.compare_digest(
                    self.authoritative_input_sha256,
                    expected_authoritative_input_sha256,
                )
            ):
                return False
            replayed = evaluate_history(
                frozen_candidates=frozen_candidates,
                mode=mode,
                library=library,
                query_embedding=query_embedding,
                service_mappings=service_mappings,
                config=config,
                trusted_source_bundle=trusted_source_bundle,
            )
            actual_document = HistoryEvaluationResult.model_dump(
                self,
                mode="json",
                warnings="error",
            )
            replayed_document = HistoryEvaluationResult.model_dump(
                replayed,
                mode="json",
                warnings="error",
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        return actual_document == replayed_document


def _empty_candidate_score(candidate_service_key: str) -> CandidateHistoryScore:
    unsigned: dict[str, object] = {
        "schema_version": CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
        "candidate_service_key": candidate_service_key,
        "contributions": (),
        "history_score": 0.0,
    }
    identifier, digest = _stable_id("chs", unsigned)
    return CandidateHistoryScore.model_validate(
        {
            **unsigned,
            "candidate_history_score_id": identifier,
            "candidate_history_score_sha256": digest,
        }
    )


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    similarity = dot / (left_norm * right_norm)
    if not math.isfinite(similarity):  # pragma: no cover - verified nonzero finite inputs
        raise AssertionError("verified history cosine must be finite")
    return min(1.0, max(-1.0, similarity))


def _build_neighbor_audit(
    *,
    entry: HistoryLibraryEntry,
    status: HistoryNeighborStatus,
    similarity: float | None,
    normalized_similarity: float,
) -> HistoryNeighborAudit:
    unsigned: dict[str, object] = {
        "schema_version": HISTORY_NEIGHBOR_AUDIT_SCHEMA_VERSION,
        "entry_id": entry.entry_id,
        "source_group_fingerprint": entry.source_group_fingerprint,
        "status": status,
        "similarity": similarity,
        "normalized_similarity": normalized_similarity,
    }
    identifier, digest = _stable_id("haud", unsigned)
    return HistoryNeighborAudit.model_validate(
        {**unsigned, "audit_id": identifier, "audit_sha256": digest}
    )


def _select_neighbors(
    *,
    library: FrozenHistoryLibrary,
    query: HistoryQueryEmbedding,
    config: HistoryScoringConfig,
) -> tuple[tuple[HistoryLibraryEntry, ...], tuple[HistoryNeighborAudit, ...]]:
    disposition: dict[str, tuple[HistoryNeighborStatus, float | None]] = {}
    qualified: list[tuple[HistoryLibraryEntry, float]] = []
    for entry in library.entries:
        if entry.source_opaque_case_id == query.current_opaque_case_id:
            disposition[entry.entry_id] = ("self_excluded", None)
            continue
        if entry.source_group_fingerprint == query.current_group_fingerprint:
            disposition[entry.entry_id] = ("incident_group_excluded", None)
            continue
        if (
            config.time_policy == "strict_prior"
            and entry.source_completed_at_unix_ns >= query.current_decision_time_unix_ns
        ):
            disposition[entry.entry_id] = ("time_excluded", None)
            continue
        similarity = _cosine_similarity(query.embedding, entry.embedding)
        if similarity < config.min_similarity or similarity < 0.0:
            disposition[entry.entry_id] = ("below_similarity", similarity)
            continue
        qualified.append((entry, similarity))

    group_winners: list[tuple[HistoryLibraryEntry, float]] = []
    by_group: dict[str, list[tuple[HistoryLibraryEntry, float]]] = {}
    for entry, similarity in qualified:
        by_group.setdefault(entry.source_group_fingerprint, []).append((entry, similarity))
    for group_entries in by_group.values():
        ordered_group = sorted(group_entries, key=lambda item: (-item[1], item[0].entry_id))
        group_winners.append(ordered_group[0])
        for entry, similarity in ordered_group[1:]:
            disposition[entry.entry_id] = ("same_group_not_selected", similarity)

    ranked = sorted(group_winners, key=lambda item: (-item[1], item[0].entry_id))
    selected_pairs = ranked[: config.k]
    for entry, similarity in ranked[config.k :]:
        disposition[entry.entry_id] = ("top_k_not_selected", similarity)
    for entry, similarity in selected_pairs:
        disposition[entry.entry_id] = ("selected", similarity)

    selected_similarity_total = math.fsum(max(similarity, 0.0) for _, similarity in selected_pairs)
    normalized_by_entry = {
        entry.entry_id: (
            max(similarity, 0.0) / selected_similarity_total
            if selected_similarity_total > 0.0
            else 0.0
        )
        for entry, similarity in selected_pairs
    }
    audits = tuple(
        _build_neighbor_audit(
            entry=entry,
            status=disposition[entry.entry_id][0],
            similarity=disposition[entry.entry_id][1],
            normalized_similarity=normalized_by_entry.get(entry.entry_id, 0.0),
        )
        for entry in library.entries
    )
    return tuple(entry for entry, _ in selected_pairs), audits


def _build_hint_reference(entry: HistoryLibraryEntry) -> HistoryHintReference:
    unsigned: dict[str, object] = {
        "schema_version": HISTORY_HINT_REFERENCE_SCHEMA_VERSION,
        "selected_neighbor_id": entry.entry_id,
        "semantic_payload_sha256": entry.sanitized_semantic_payload_sha256,
        "requires_current_case_reverification": True,
    }
    identifier, digest = _stable_id("hhint", unsigned)
    return HistoryHintReference.model_validate(
        {
            **unsigned,
            "hint_reference_id": identifier,
            "hint_reference_sha256": digest,
        }
    )


def _build_history_contribution(
    *,
    entry: HistoryLibraryEntry,
    audit: HistoryNeighborAudit,
    candidate_service_key: str,
    mapping: HistoryServiceMapping | None,
) -> HistoryScoreContribution:
    base_score = 0.0
    source_service_key: str | None = None
    mapping_id: str | None = None
    if mapping is not None:
        mapping_id = mapping.mapping_id
        source_service_key = mapping.source_service_key
        base_score_by_service = {
            item.source_service_key: item.base_history_score for item in entry.base_candidate_scores
        }
        base_score = base_score_by_service[source_service_key]
    similarity = audit.similarity if audit.similarity is not None else 0.0
    contribution_value = audit.normalized_similarity * base_score
    unsigned: dict[str, object] = {
        "schema_version": HISTORY_SCORE_CONTRIBUTION_SCHEMA_VERSION,
        "selected_neighbor_id": entry.entry_id,
        "candidate_service_key": candidate_service_key,
        "mapping_id": mapping_id,
        "source_service_key": source_service_key,
        "mapping_status": "mapped" if mapping is not None else "missing_mapping",
        "similarity": similarity,
        "normalized_similarity": audit.normalized_similarity,
        "base_history_score": base_score,
        "contribution": contribution_value,
    }
    identifier, digest = _stable_id("hcon", unsigned)
    return HistoryScoreContribution.model_validate(
        {**unsigned, "contribution_id": identifier, "contribution_sha256": digest}
    )


def _build_candidate_history_scores(
    *,
    candidate_service_keys: tuple[str, ...],
    selected_entries: tuple[HistoryLibraryEntry, ...],
    neighbor_audits: tuple[HistoryNeighborAudit, ...],
    mappings: tuple[HistoryServiceMapping, ...],
) -> tuple[CandidateHistoryScore, ...]:
    audit_by_entry = {item.entry_id: item for item in neighbor_audits}
    mapping_by_entry_candidate = {
        (item.entry_id, item.current_service_key): item for item in mappings
    }
    scores: list[CandidateHistoryScore] = []
    for candidate_service_key in candidate_service_keys:
        contributions = tuple(
            _build_history_contribution(
                entry=entry,
                audit=audit_by_entry[entry.entry_id],
                candidate_service_key=candidate_service_key,
                mapping=mapping_by_entry_candidate.get((entry.entry_id, candidate_service_key)),
            )
            for entry in selected_entries
        )
        history_score = math.fsum(item.contribution for item in contributions)
        unsigned: dict[str, object] = {
            "schema_version": CANDIDATE_HISTORY_SCORE_SCHEMA_VERSION,
            "candidate_service_key": candidate_service_key,
            "contributions": tuple(item.model_dump(mode="json") for item in contributions),
            "history_score": min(1.0, max(0.0, history_score)),
        }
        identifier, digest = _stable_id("chs", unsigned)
        scores.append(
            CandidateHistoryScore.model_validate(
                {
                    **unsigned,
                    "candidate_history_score_id": identifier,
                    "candidate_history_score_sha256": digest,
                }
            )
        )
    return tuple(scores)


def evaluate_history(
    *,
    frozen_candidates: FrozenCandidateUniverse,
    mode: HistoryMode = "off",
    library: object | None = None,
    query_embedding: object | None = None,
    service_mappings: object = (),
    config: object | None = None,
    trusted_source_bundle: object | None = None,
) -> HistoryEvaluationResult:
    """Evaluate isolated history; the default path never loads a history library."""

    budget_exceeded = False
    trusted_common: (
        tuple[
            FrozenCandidateUniverse,
            Sequence[object],
            HistoryScoringConfig,
        ]
        | None
    ) = None
    try:
        if type(mode) is not str or mode not in {"off", "hint", "score"}:
            raise ValueError("history mode is invalid")
        _preflight_model_tuple_limit(
            frozen_candidates,
            model_type=FrozenCandidateUniverse,
            field_name="candidate_service_keys",
            maximum=_MAX_HISTORY_CANDIDATES,
            label="candidate freeze",
        )
        frozen = _trusted_exact_model(
            frozen_candidates,
            FrozenCandidateUniverse,
            label="candidate freeze",
        )
        if not FrozenCandidateUniverse.verify(frozen):
            raise ValueError("candidate freeze failed verification")
        mapping_values = _plain_sequence(
            service_mappings,
            label="history service mappings",
            maximum=_MAX_HISTORY_MAPPINGS,
        )
        if config is None:
            verified_config = build_history_scoring_config()
        else:
            verified_config = _trusted_exact_model(
                config,
                HistoryScoringConfig,
                label="history scoring config",
            )
            if not HistoryScoringConfig.verify(verified_config):
                raise ValueError("history config failed verification")
        trusted_common = (frozen, mapping_values, verified_config)
    except _HistoryScoringResourceLimitError:
        budget_exceeded = True
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if budget_exceeded:
        raise HistoryScoringBudgetExceeded() from None
    if trusted_common is None:
        raise ValueError("history scoring input verification failed") from None
    frozen, mapping_values, verified_config = trusted_common
    verified_bundle: HistoryTrustedSourceBundle | None = None

    if mode == "off":
        if (
            library is not None
            or query_embedding is not None
            or mapping_values
            or trusted_source_bundle is not None
        ):
            raise ValueError("history scoring input verification failed")
        verified_library: FrozenHistoryLibrary | None = None
        verified_query: HistoryQueryEmbedding | None = None
        ordered_mappings: tuple[HistoryServiceMapping, ...] = ()
        selected_entries: tuple[HistoryLibraryEntry, ...] = ()
        neighbor_audits: tuple[HistoryNeighborAudit, ...] = ()
        _preflight_projected_output(
            mode="off",
            entry_count=0,
            mapping_count=0,
            candidate_count=len(frozen.candidate_service_keys),
            k=verified_config.k,
        )
    else:
        trusted_enabled: (
            tuple[
                FrozenHistoryLibrary,
                HistoryQueryEmbedding,
                tuple[HistoryServiceMapping, ...],
                HistoryTrustedSourceBundle,
            ]
            | None
        ) = None
        budget_exceeded = False
        try:
            if any(type(item) is not HistoryServiceMapping for item in mapping_values):
                raise ValueError("history mapping must use exact models")
            mapping_models = tuple(cast(HistoryServiceMapping, item) for item in mapping_values)
            _preflight_exact_tree(
                mapping_models,
                root_model_types=(HistoryServiceMapping,),
                label="history service mappings",
            )
            typed_mappings = tuple(
                _trusted_exact_model(
                    item,
                    HistoryServiceMapping,
                    label="history service mapping",
                )
                for item in mapping_models
            )
            if any(not HistoryServiceMapping.verify(item) for item in typed_mappings):
                raise ValueError("history mapping failed verification")
            _preflight_model_tuple_limit(
                library,
                model_type=FrozenHistoryLibrary,
                field_name="entries",
                maximum=_MAX_HISTORY_ENTRIES,
                label="history library",
            )
            _preflight_model_tuple_limit(
                query_embedding,
                model_type=HistoryQueryEmbedding,
                field_name="embedding",
                maximum=_MAX_VECTOR_DIMENSION,
                label="history query embedding",
            )
            verified_library_value = _trusted_exact_model(
                library,
                FrozenHistoryLibrary,
                label="history library",
            )
            verified_query_value = _trusted_exact_model(
                query_embedding,
                HistoryQueryEmbedding,
                label="history query embedding",
            )
            if type(trusted_source_bundle) is not HistoryTrustedSourceBundle:
                raise ValueError("history trusted source bundle must use the exact type")
            verified_bundle = trusted_source_bundle
            _preflight_trusted_source_bundle(verified_bundle)
            if not FrozenHistoryLibrary.verify(
                verified_library_value
            ) or not HistoryQueryEmbedding.verify(verified_query_value):
                raise ValueError("history authority failed verification")
            if not _verify_trusted_source_bundle(
                verified_library_value,
                verified_bundle,
            ):
                raise ValueError("history source bundle failed verification")
            trusted_enabled = (
                verified_library_value,
                verified_query_value,
                typed_mappings,
                verified_bundle,
            )
        except _HistoryScoringResourceLimitError:
            budget_exceeded = True
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        if budget_exceeded:
            raise HistoryScoringBudgetExceeded() from None
        if trusted_enabled is None:
            raise ValueError("history scoring input verification failed") from None
        verified_library, verified_query, typed_mappings, verified_bundle = trusted_enabled
        if (
            verified_query.current_opaque_case_id != frozen.opaque_case_id
            or verified_query.dataset != verified_library.dataset
            or verified_query.embedding_policy != verified_library.embedding_policy
            or verified_query.embedding_model != verified_library.embedding_model
            or verified_query.embedding_dimension != verified_library.embedding_dimension
        ):
            raise ValueError("history scoring input verification failed")
        try:
            _preflight_projected_output(
                mode=mode,
                entry_count=len(verified_library.entries),
                mapping_count=len(typed_mappings),
                candidate_count=len(frozen.candidate_service_keys),
                k=verified_config.k,
            )
        except _HistoryScoringResourceLimitError:
            raise HistoryScoringBudgetExceeded() from None
        entry_by_id = {item.entry_id: item for item in verified_library.entries}
        source_services_by_entry = {
            item.entry_id: frozenset(
                score.source_service_key for score in item.base_candidate_scores
            )
            for item in verified_library.entries
        }
        if any(
            item.entry_id not in entry_by_id
            or item.current_service_key not in frozen.candidate_service_keys
            or item.source_service_key
            not in source_services_by_entry.get(item.entry_id, frozenset())
            for item in typed_mappings
        ):
            raise ValueError("history scoring input verification failed")
        mapping_keys = tuple((item.entry_id, item.current_service_key) for item in typed_mappings)
        if len(mapping_keys) != len(set(mapping_keys)):
            raise ValueError("history scoring input verification failed")
        if len({item.mapping_id for item in typed_mappings}) != len(typed_mappings) or len(
            {item.mapping_sha256 for item in typed_mappings}
        ) != len(typed_mappings):
            raise ValueError("history scoring input verification failed")
        ordered_mappings = tuple(sorted(typed_mappings, key=lambda item: item.mapping_id))
        selected_entries, neighbor_audits = _select_neighbors(
            library=verified_library,
            query=verified_query,
            config=verified_config,
        )

    candidate_scores = (
        _build_candidate_history_scores(
            candidate_service_keys=frozen.candidate_service_keys,
            selected_entries=selected_entries,
            neighbor_audits=neighbor_audits,
            mappings=ordered_mappings,
        )
        if mode == "score"
        else tuple(
            _empty_candidate_score(candidate_service_key)
            for candidate_service_key in frozen.candidate_service_keys
        )
    )
    hint_references = (
        tuple(_build_hint_reference(entry) for entry in selected_entries) if mode == "hint" else ()
    )
    selected_neighbor_ids = tuple(item.entry_id for item in selected_entries)
    library_sha256 = verified_library.library_sha256 if verified_library is not None else None
    generation_authority_sha256 = (
        verified_bundle.generation_authority.generation_authority_sha256
        if verified_bundle is not None
        else None
    )
    query_sha256 = verified_query.query_embedding_sha256 if verified_query is not None else None
    mapping_sha256s = tuple(sorted(item.mapping_sha256 for item in ordered_mappings))
    candidate_binding_sha256 = _candidate_binding_sha256(
        candidate_freeze_sha256=frozen.freeze_sha256,
        candidate_universe_sha256=frozen.candidate_universe_sha256,
        candidate_count=frozen.candidate_count,
        candidate_service_keys=frozen.candidate_service_keys,
    )
    authoritative_input_sha256 = _authoritative_input_sha256(
        candidate_binding_sha256=candidate_binding_sha256,
        mode=mode,
        generation_authority_sha256=generation_authority_sha256,
        library_sha256=library_sha256,
        query_embedding_sha256=query_sha256,
        service_mapping_sha256s=mapping_sha256s,
        config_sha256=verified_config.config_sha256,
    )
    unsigned: dict[str, object] = {
        "schema_version": HISTORY_EVALUATION_RESULT_SCHEMA_VERSION,
        "mode": mode,
        "claim_scope": (
            "history_disabled"
            if mode == "off"
            else (
                "prior_history"
                if verified_config.time_policy == "strict_prior"
                else "offline_similarity_only"
            )
        ),
        "candidate_freeze_sha256": frozen.freeze_sha256,
        "candidate_universe_sha256": frozen.candidate_universe_sha256,
        "candidate_count": frozen.candidate_count,
        "candidate_service_keys": frozen.candidate_service_keys,
        "candidate_binding_sha256": candidate_binding_sha256,
        "generation_authority_sha256": generation_authority_sha256,
        "library_sha256": library_sha256,
        "query_embedding_sha256": query_sha256,
        "service_mapping_sha256s": mapping_sha256s,
        "config_sha256": verified_config.config_sha256,
        "neighbor_audits": tuple(item.model_dump(mode="json") for item in neighbor_audits),
        "selected_neighbor_ids": selected_neighbor_ids,
        "hint_references": tuple(item.model_dump(mode="json") for item in hint_references),
        "requires_current_case_reverification": mode == "hint",
        "candidate_scores": tuple(item.model_dump(mode="json") for item in candidate_scores),
        "authoritative_input_sha256": authoritative_input_sha256,
    }
    identifier, digest = _stable_id("heval", unsigned)
    return HistoryEvaluationResult.model_validate(
        {
            **unsigned,
            "history_evaluation_result_id": identifier,
            "result_sha256": digest,
        }
    )


__all__ = [
    "CandidateHistoryScore",
    "FrozenHistoryLibrary",
    "HistoryBaseCandidateScore",
    "HistoryEvaluationResult",
    "HistoryGenerationAuthority",
    "HistoryHintReference",
    "HistoryLibraryEntry",
    "HistoryNeighborAudit",
    "HistoryQueryEmbedding",
    "HistoryScoreContribution",
    "HistoryScoringBudgetExceeded",
    "HistoryScoringConfig",
    "HistoryServiceMapping",
    "HistoryTrustedBaseScoreSource",
    "HistoryTrustedEntrySource",
    "HistoryTrustedSourceBundle",
    "build_frozen_history_library",
    "build_history_base_candidate_score",
    "build_history_generation_authority",
    "build_history_library_entry",
    "build_history_query_embedding",
    "build_history_scoring_config",
    "build_history_service_mapping",
    "evaluate_history",
]
