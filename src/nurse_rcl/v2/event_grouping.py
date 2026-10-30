"""Deterministic event grouping and exact semantic-family assignment for V2."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from functools import cache
from typing import Literal, TypeAlias

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.observation_schemas import (
    OBSERVATION_SCHEMA_VERSION,
    Observation,
    TimeInterval,
)

SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION: Literal["nurse.v2.signal-family-mapping-entry/1"] = (
    "nurse.v2.signal-family-mapping-entry/1"
)
SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION: Literal["nurse.v2.signal-family-mapping/1"] = (
    "nurse.v2.signal-family-mapping/1"
)
ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION: Literal["nurse.v2.association-derivation-rule/3"] = (
    "nurse.v2.association-derivation-rule/3"
)
ASSOCIATION_POLICY_SCHEMA_VERSION: Literal["nurse.v2.association-policy/3"] = (
    "nurse.v2.association-policy/3"
)
ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION: Literal[
    "nurse.v2.canonical-association-source-record/2"
] = "nurse.v2.canonical-association-source-record/2"
ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION: Literal["nurse.v2.adapter-association-identity/3"] = (
    "nurse.v2.adapter-association-identity/3"
)
OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION: Literal[
    "nurse.v2.observation-association-proof/3"
] = "nurse.v2.observation-association-proof/3"
ASSOCIATION_CLAIM_SCHEMA_VERSION: Literal["nurse.v2.event-association-claim/4"] = (
    "nurse.v2.event-association-claim/4"
)
EVENT_GROUP_SCHEMA_VERSION: Literal["nurse.v2.event-group/2"] = "nurse.v2.event-group/2"
GROUPING_DECISION_SCHEMA_VERSION: Literal["nurse.v2.event-grouping-decision/2"] = (
    "nurse.v2.event-grouping-decision/2"
)
EVENT_GROUPING_RESULT_SCHEMA_VERSION: Literal["nurse.v2.event-grouping-result/4"] = (
    "nurse.v2.event-grouping-result/4"
)
ASSOCIATION_POLICY_VERSION: Literal["nurse-v2-frozen-association-policy/3"] = (
    "nurse-v2-frozen-association-policy/3"
)
ASSOCIATION_DERIVATION_SEMANTICS_VERSION: Literal["nurse-v2-association-derivation/3"] = (
    "nurse-v2-association-derivation/3"
)
ASSOCIATION_NORMALIZATION_SEMANTICS: Literal["sha256-utf8-nfc/1"] = "sha256-utf8-nfc/1"
ASSOCIATION_FIELD_PARSER_SEMANTICS_VERSION: Literal["nurse-v2-association-field-parser/2"] = (
    "nurse-v2-association-field-parser/2"
)

PrimarySignalFamily: TypeAlias = Literal[
    "cpu",
    "memory",
    "lifecycle",
    "request_load",
    "error",
    "latency",
    "dependency_failure",
    "storage",
    "network",
    "unclassified",
]
AssociationBasis: TypeAlias = Literal[
    "shared_export_id",
    "explicit_source_relation",
    "deterministic_association_key",
]
AssociationDerivationKind: TypeAlias = Literal[
    "adapter_shared_export_field",
    "adapter_explicit_relation_field",
    "adapter_canonical_record_key_fields",
]
AssociationSourceFieldKind: TypeAlias = Literal[
    "export_id",
    "source_relation_id",
    "trace_id",
    "span_id",
    "log_record_id",
    "metric_exemplar_id",
    "counter_series_id",
]
AssociationSourceFieldParserKind: TypeAlias = Literal[
    "canonical_lower_hex",
    "prefixed_safe_opaque_id",
]
GroupAssociationBasis: TypeAlias = Literal[
    "shared_export_id",
    "explicit_source_relation",
    "deterministic_association_key",
    "singleton_no_association_evidence",
]

_FORBIDDEN_ASSOCIATION_INPUTS = (
    "entity",
    "filename",
    "llm_inference",
    "modality",
    "name",
    "service",
    "temporal_interval",
    "temporal_proximity",
)
_ASSOCIATION_SOURCE_FIELD_ORDER: tuple[AssociationSourceFieldKind, ...] = (
    "export_id",
    "source_relation_id",
    "trace_id",
    "span_id",
    "log_record_id",
    "metric_exemplar_id",
    "counter_series_id",
)
_ASSOCIATION_SOURCE_FIELD_ORDER_INDEX = {
    field_kind: index for index, field_kind in enumerate(_ASSOCIATION_SOURCE_FIELD_ORDER)
}
_SAFE_OPAQUE_TOKEN_GRAMMAR = r"[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?"
_FIELD_PARSER_SPEC_BY_KIND: dict[
    AssociationSourceFieldKind,
    tuple[AssociationSourceFieldParserKind, str, str | None, int | None, bool, bool],
] = {
    "export_id": (
        "prefixed_safe_opaque_id",
        rf"export:{_SAFE_OPAQUE_TOKEN_GRAMMAR}",
        "export",
        None,
        False,
        True,
    ),
    "source_relation_id": (
        "prefixed_safe_opaque_id",
        rf"relation:{_SAFE_OPAQUE_TOKEN_GRAMMAR}",
        "relation",
        None,
        False,
        True,
    ),
    "trace_id": (
        "canonical_lower_hex",
        "[0-9a-f]{32}",
        None,
        32,
        True,
        False,
    ),
    "span_id": (
        "canonical_lower_hex",
        "[0-9a-f]{16}",
        None,
        16,
        True,
        False,
    ),
    "log_record_id": (
        "prefixed_safe_opaque_id",
        rf"log:{_SAFE_OPAQUE_TOKEN_GRAMMAR}",
        "log",
        None,
        False,
        True,
    ),
    "metric_exemplar_id": (
        "prefixed_safe_opaque_id",
        rf"exemplar:{_SAFE_OPAQUE_TOKEN_GRAMMAR}",
        "exemplar",
        None,
        False,
        True,
    ),
    "counter_series_id": (
        "prefixed_safe_opaque_id",
        rf"series:{_SAFE_OPAQUE_TOKEN_GRAMMAR}",
        "series",
        None,
        False,
        True,
    ),
}
_RULE_SPEC_BY_BASIS: dict[
    AssociationBasis,
    tuple[
        AssociationDerivationKind,
        tuple[AssociationSourceFieldKind, ...],
        tuple[tuple[AssociationSourceFieldKind, ...], ...],
    ],
] = {
    "shared_export_id": (
        "adapter_shared_export_field",
        ("export_id",),
        (("export_id",),),
    ),
    "explicit_source_relation": (
        "adapter_explicit_relation_field",
        ("source_relation_id",),
        (("source_relation_id",),),
    ),
    "deterministic_association_key": (
        "adapter_canonical_record_key_fields",
        (
            "trace_id",
            "span_id",
            "log_record_id",
            "metric_exemplar_id",
            "counter_series_id",
        ),
        (
            ("trace_id", "span_id"),
            ("log_record_id",),
            ("metric_exemplar_id",),
            ("counter_series_id",),
        ),
    ),
}


def _stable_id(prefix: str, payload: object) -> str:
    return f"{prefix}_{sha256_hex(canonical_json_bytes(payload))[:24]}"


def _require_canonical(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


class SignalFamilyMappingEntry(StrictFrozenModel):
    """One adapter-frozen exact match from source semantics to a primary family."""

    schema_version: Literal["nurse.v2.signal-family-mapping-entry/1"] = (
        SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION
    )
    mapping_entry_id: str = Field(pattern=r"^fam_[0-9a-f]{24}$")
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_signal_family: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    primary_signal_family: PrimarySignalFamily
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_content_identity(self) -> Self:
        _require_canonical(self.provenance_ids, "mapping provenance IDs")
        if self.mapping_entry_id != _mapping_entry_id(self):
            raise ValueError("signal-family mapping entry ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = SignalFamilyMappingEntry.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _mapping_entry_id(entry: SignalFamilyMappingEntry) -> str:
    return _stable_id(
        "fam",
        entry.model_dump(mode="json", exclude={"mapping_entry_id"}),
    )


def build_signal_family_mapping_entry(
    *,
    adapter_semantics_sha256: str,
    source_signal_family: str,
    primary_signal_family: PrimarySignalFamily,
    provenance_ids: Sequence[str],
) -> SignalFamilyMappingEntry:
    """Build a content-addressed exact adapter-family mapping entry."""

    unsigned: dict[str, object] = {
        "schema_version": SIGNAL_FAMILY_MAPPING_ENTRY_SCHEMA_VERSION,
        "adapter_semantics_sha256": adapter_semantics_sha256,
        "source_signal_family": source_signal_family,
        "primary_signal_family": primary_signal_family,
        "provenance_ids": tuple(sorted(set(provenance_ids))),
    }
    return SignalFamilyMappingEntry.model_validate(
        {**unsigned, "mapping_entry_id": _stable_id("fam", unsigned)}
    )


def _mapping_entry_sort_key(entry: SignalFamilyMappingEntry) -> tuple[str, str, str]:
    return (
        entry.adapter_semantics_sha256,
        entry.source_signal_family,
        entry.mapping_entry_id,
    )


class FrozenSignalFamilyMapping(StrictFrozenModel):
    """Versioned, content-addressed exact mapping frozen by an adapter."""

    schema_version: Literal["nurse.v2.signal-family-mapping/1"] = (
        SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION
    )
    mapping_version: str = Field(min_length=1, max_length=128)
    entries: tuple[SignalFamilyMappingEntry, ...]
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_unambiguous_mapping(self) -> Self:
        if self.entries != tuple(sorted(self.entries, key=_mapping_entry_sort_key)):
            raise ValueError("signal-family mapping entries must be canonically ordered")
        keys = tuple(
            (entry.adapter_semantics_sha256, entry.source_signal_family) for entry in self.entries
        )
        if len(keys) != len(set(keys)):
            raise ValueError("signal-family mapping contains an ambiguous exact key")
        return self

    def verify(self) -> bool:
        try:
            validated = FrozenSignalFamilyMapping.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"mapping_sha256"})
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and all(entry.verify() for entry in validated.entries)
            and validated.mapping_sha256 == sha256_hex(canonical_json_bytes(unsigned))
        )


def build_frozen_signal_family_mapping(
    *,
    mapping_version: str,
    entries: Sequence[SignalFamilyMappingEntry],
) -> FrozenSignalFamilyMapping:
    """Freeze a deterministic exact mapping and reject duplicate match keys."""

    validated = tuple(
        SignalFamilyMappingEntry.model_validate(entry.model_dump(mode="json", warnings="error"))
        for entry in entries
    )
    ordered = tuple(sorted(validated, key=_mapping_entry_sort_key))
    keys = tuple((entry.adapter_semantics_sha256, entry.source_signal_family) for entry in ordered)
    if len(keys) != len(set(keys)):
        raise ValueError("signal-family mapping contains an ambiguous exact key")
    unsigned: dict[str, object] = {
        "schema_version": SIGNAL_FAMILY_MAPPING_SCHEMA_VERSION,
        "mapping_version": mapping_version,
        "entries": [entry.model_dump(mode="json") for entry in ordered],
    }
    return FrozenSignalFamilyMapping.model_validate(
        {**unsigned, "mapping_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


class AssociationSourceFieldParserSpec(StrictFrozenModel):
    """Frozen, auditable parser grammar for one typed raw identifier."""

    field_kind: AssociationSourceFieldKind
    parser_semantics_version: Literal["nurse-v2-association-field-parser/2"] = (
        ASSOCIATION_FIELD_PARSER_SEMANTICS_VERSION
    )
    parser_kind: AssociationSourceFieldParserKind
    grammar: str = Field(min_length=1)
    required_prefix: str | None = None
    canonical_hex_length: int | None = Field(default=None, gt=0)
    reject_all_zero: bool
    reject_sha256_payload: bool
    parser_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_frozen_parser(self) -> Self:
        (
            parser_kind,
            grammar,
            prefix,
            hex_length,
            reject_all_zero,
            reject_sha256_payload,
        ) = _FIELD_PARSER_SPEC_BY_KIND[self.field_kind]
        if (
            self.parser_kind != parser_kind
            or self.grammar != grammar
            or self.required_prefix != prefix
            or self.canonical_hex_length != hex_length
            or self.reject_all_zero != reject_all_zero
            or self.reject_sha256_payload != reject_sha256_payload
        ):
            raise ValueError("association source field parser is not the frozen grammar")
        unsigned = self.model_dump(mode="json", exclude={"parser_semantics_sha256"})
        if self.parser_semantics_sha256 != sha256_hex(canonical_json_bytes(unsigned)):
            raise ValueError("association source field parser hash is invalid")
        return self

    def verify(self) -> bool:
        try:
            validated = AssociationSourceFieldParserSpec.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


@cache
def _build_field_parser_spec(
    field_kind: AssociationSourceFieldKind,
) -> AssociationSourceFieldParserSpec:
    (
        parser_kind,
        grammar,
        prefix,
        hex_length,
        reject_all_zero,
        reject_sha256_payload,
    ) = _FIELD_PARSER_SPEC_BY_KIND[field_kind]
    unsigned: dict[str, object] = {
        "field_kind": field_kind,
        "parser_semantics_version": ASSOCIATION_FIELD_PARSER_SEMANTICS_VERSION,
        "parser_kind": parser_kind,
        "grammar": grammar,
        "required_prefix": prefix,
        "canonical_hex_length": hex_length,
        "reject_all_zero": reject_all_zero,
        "reject_sha256_payload": reject_sha256_payload,
    }
    return AssociationSourceFieldParserSpec.model_validate(
        {
            **unsigned,
            "parser_semantics_sha256": sha256_hex(canonical_json_bytes(unsigned)),
        }
    )


def _field_parser_semantics_sha256(
    specs: tuple[AssociationSourceFieldParserSpec, ...],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "parser_semantics_version": ASSOCIATION_FIELD_PARSER_SEMANTICS_VERSION,
                "field_parser_specs": [item.model_dump(mode="json") for item in specs],
            }
        )
    )


class AssociationDerivationRule(StrictFrozenModel):
    """Built-in rule allowing only typed adapter identity derivations."""

    schema_version: Literal["nurse.v2.association-derivation-rule/3"] = (
        ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION
    )
    derivation_rule_id: str = Field(pattern=r"^arule_[0-9a-f]{24}$")
    association_basis: AssociationBasis
    derivation_kind: AssociationDerivationKind
    producer_kind: Literal["frozen_adapter"] = "frozen_adapter"
    semantics_version: Literal["nurse-v2-association-derivation/3"] = (
        ASSOCIATION_DERIVATION_SEMANTICS_VERSION
    )
    normalization_semantics: Literal["sha256-utf8-nfc/1"] = ASSOCIATION_NORMALIZATION_SEMANTICS
    allowed_source_fields: tuple[AssociationSourceFieldKind, ...] = Field(min_length=1)
    allowed_source_field_combinations: tuple[tuple[AssociationSourceFieldKind, ...], ...] = Field(
        min_length=1
    )
    field_cardinality: Literal["exactly_one", "one_or_more"]
    field_parser_specs: tuple[AssociationSourceFieldParserSpec, ...] = Field(min_length=1)
    field_parser_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forbidden_inputs: tuple[
        Literal[
            "entity",
            "filename",
            "llm_inference",
            "modality",
            "name",
            "service",
            "temporal_interval",
            "temporal_proximity",
        ],
        ...,
    ]
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_frozen_rule(self) -> Self:
        derivation_kind, allowed_fields, allowed_combinations = _RULE_SPEC_BY_BASIS[
            self.association_basis
        ]
        cardinality = (
            "exactly_one"
            if all(len(combination) == 1 for combination in allowed_combinations)
            else "one_or_more"
        )
        parser_specs = tuple(_build_field_parser_spec(item) for item in allowed_fields)
        if (
            self.derivation_kind != derivation_kind
            or self.allowed_source_fields != allowed_fields
            or self.allowed_source_field_combinations != allowed_combinations
            or self.field_cardinality != cardinality
            or self.field_parser_specs != parser_specs
            or self.field_parser_semantics_sha256 != _field_parser_semantics_sha256(parser_specs)
            or self.forbidden_inputs != _FORBIDDEN_ASSOCIATION_INPUTS
        ):
            raise ValueError("association derivation rule is not the frozen basis rule")
        if self.derivation_semantics_sha256 != _derivation_semantics_sha256(
            basis=self.association_basis,
            derivation_kind=self.derivation_kind,
            allowed_source_fields=self.allowed_source_fields,
            allowed_source_field_combinations=self.allowed_source_field_combinations,
            field_cardinality=self.field_cardinality,
            field_parser_specs=self.field_parser_specs,
        ):
            raise ValueError("association derivation semantics hash is invalid")
        if self.derivation_rule_id != _association_derivation_rule_id(self):
            raise ValueError("association derivation rule ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = AssociationDerivationRule.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _derivation_semantics_sha256(
    *,
    basis: AssociationBasis,
    derivation_kind: AssociationDerivationKind,
    allowed_source_fields: tuple[AssociationSourceFieldKind, ...],
    allowed_source_field_combinations: tuple[tuple[AssociationSourceFieldKind, ...], ...],
    field_cardinality: str,
    field_parser_specs: tuple[AssociationSourceFieldParserSpec, ...],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "semantics_version": ASSOCIATION_DERIVATION_SEMANTICS_VERSION,
                "association_basis": basis,
                "derivation_kind": derivation_kind,
                "producer_kind": "frozen_adapter",
                "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
                "allowed_source_fields": allowed_source_fields,
                "allowed_source_field_combinations": allowed_source_field_combinations,
                "field_cardinality": field_cardinality,
                "field_parser_semantics_sha256": _field_parser_semantics_sha256(field_parser_specs),
                "field_parser_specs": [item.model_dump(mode="json") for item in field_parser_specs],
                "forbidden_inputs": _FORBIDDEN_ASSOCIATION_INPUTS,
                "identity_digest_inputs": (
                    "association_basis",
                    "derivation_kind",
                    "normalization_semantics",
                    "source_field_hashes_derived_from_canonical_source_record",
                ),
                "source_record_bindings": (
                    "observation_schema_version",
                    "opaque_case_id",
                    "observation_id",
                    "decision_time_unix_ns",
                    "parent_source_fingerprint",
                    "adapter_semantics_sha256",
                    "raw_source_fields",
                ),
            }
        )
    )


def _association_derivation_rule_id(rule: AssociationDerivationRule) -> str:
    return _stable_id(
        "arule",
        rule.model_dump(mode="json", exclude={"derivation_rule_id"}),
    )


@cache
def _build_association_derivation_rule(
    basis: AssociationBasis,
) -> AssociationDerivationRule:
    derivation_kind, allowed_fields, allowed_combinations = _RULE_SPEC_BY_BASIS[basis]
    cardinality = (
        "exactly_one"
        if all(len(combination) == 1 for combination in allowed_combinations)
        else "one_or_more"
    )
    parser_specs = tuple(_build_field_parser_spec(item) for item in allowed_fields)
    unsigned: dict[str, object] = {
        "schema_version": ASSOCIATION_DERIVATION_RULE_SCHEMA_VERSION,
        "association_basis": basis,
        "derivation_kind": derivation_kind,
        "producer_kind": "frozen_adapter",
        "semantics_version": ASSOCIATION_DERIVATION_SEMANTICS_VERSION,
        "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
        "allowed_source_fields": allowed_fields,
        "allowed_source_field_combinations": allowed_combinations,
        "field_cardinality": cardinality,
        "field_parser_specs": [item.model_dump(mode="json") for item in parser_specs],
        "field_parser_semantics_sha256": _field_parser_semantics_sha256(parser_specs),
        "forbidden_inputs": _FORBIDDEN_ASSOCIATION_INPUTS,
        "derivation_semantics_sha256": _derivation_semantics_sha256(
            basis=basis,
            derivation_kind=derivation_kind,
            allowed_source_fields=allowed_fields,
            allowed_source_field_combinations=allowed_combinations,
            field_cardinality=cardinality,
            field_parser_specs=parser_specs,
        ),
    }
    return AssociationDerivationRule.model_validate(
        {
            **unsigned,
            "derivation_rule_id": _stable_id("arule", unsigned),
        }
    )


class FrozenAssociationPolicy(StrictFrozenModel):
    """Canonical policy registry trusted by grouping and retained for replay."""

    schema_version: Literal["nurse.v2.association-policy/3"] = ASSOCIATION_POLICY_SCHEMA_VERSION
    policy_version: Literal["nurse-v2-frozen-association-policy/3"] = ASSOCIATION_POLICY_VERSION
    rules: tuple[AssociationDerivationRule, ...] = Field(min_length=3, max_length=3)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_rules(self) -> Self:
        if self.rules != tuple(sorted(self.rules, key=lambda item: item.association_basis)):
            raise ValueError("association policy rules must be canonically ordered")
        if tuple(item.association_basis for item in self.rules) != tuple(
            sorted(_RULE_SPEC_BY_BASIS)
        ):
            raise ValueError("association policy must contain every frozen basis rule once")
        return self

    def verify(self) -> bool:
        try:
            validated = FrozenAssociationPolicy.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            unsigned = validated.model_dump(mode="json", exclude={"policy_sha256"})
            canonical = build_frozen_association_policy()
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            validated == self
            and all(rule.verify() for rule in validated.rules)
            and validated.policy_sha256 == sha256_hex(canonical_json_bytes(unsigned))
            and validated == canonical
        )

    def rule_for(self, basis: AssociationBasis) -> AssociationDerivationRule:
        for rule in self.rules:
            if rule.association_basis == basis:
                return rule
        raise ValueError("association basis is absent from frozen policy")


@cache
def build_frozen_association_policy() -> FrozenAssociationPolicy:
    """Build the sole trusted association rule registry for this implementation."""

    rules = tuple(
        sorted(
            (_build_association_derivation_rule(basis) for basis in _RULE_SPEC_BY_BASIS),
            key=lambda item: item.association_basis,
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": ASSOCIATION_POLICY_SCHEMA_VERSION,
        "policy_version": ASSOCIATION_POLICY_VERSION,
        "rules": [rule.model_dump(mode="json") for rule in rules],
    }
    return FrozenAssociationPolicy.model_validate(
        {**unsigned, "policy_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )


def _is_safe_opaque_token(value: str) -> bool:
    return re.fullmatch(_SAFE_OPAQUE_TOKEN_GRAMMAR, value) is not None


def _parse_association_source_field(
    field_kind: AssociationSourceFieldKind,
    raw_value: str,
) -> str:
    normalized = unicodedata.normalize("NFC", raw_value)
    (
        parser_kind,
        grammar,
        prefix,
        hex_length,
        reject_all_zero,
        reject_sha256_payload,
    ) = _FIELD_PARSER_SPEC_BY_KIND[field_kind]
    if parser_kind == "canonical_lower_hex":
        if (
            hex_length is None
            or re.fullmatch(rf"[0-9a-f]{{{hex_length}}}", normalized) is None
            or (reject_all_zero and set(normalized) == {"0"})
        ):
            raise ValueError(
                f"{field_kind} must use canonical {hex_length} lowercase hex and cannot be all zero"
            )
        return normalized
    if prefix is None:  # pragma: no cover - frozen parser-table invariant
        raise AssertionError("opaque association parser has no required prefix")
    required = f"{prefix}:"
    if not normalized.startswith(required):
        raise ValueError(f"{field_kind} must use canonical opaque ID grammar {grammar}")
    token = normalized[len(required) :]
    if reject_sha256_payload and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", token):
        raise ValueError("raw association identifiers cannot be pre-hashed")
    if not _is_safe_opaque_token(token):
        raise ValueError(f"{field_kind} must use canonical opaque ID grammar {grammar}")
    return normalized


class AdapterAssociationSourceField(StrictFrozenModel):
    """Allowlisted raw identifier emitted by a frozen adapter extractor."""

    field_kind: AssociationSourceFieldKind
    raw_value: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_typed_raw_value(self) -> Self:
        candidate = self.raw_value.strip()
        if re.fullmatch(r"(?i)(?:sha256:)?[0-9a-f]{64}", candidate):
            raise ValueError("raw association identifiers cannot be pre-hashed")
        _parse_association_source_field(self.field_kind, self.raw_value)
        return self


class AssociationSourceFieldHash(StrictFrozenModel):
    """Replayable digest derived internally from one retained raw identifier."""

    field_kind: AssociationSourceFieldKind
    parser_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _normalized_source_field_sha256(field: AdapterAssociationSourceField) -> str:
    normalized = _parse_association_source_field(field.field_kind, field.raw_value)
    return sha256_hex(normalized.encode("utf-8"))


def _source_field_hashes(
    source_fields: tuple[AdapterAssociationSourceField, ...],
) -> tuple[AssociationSourceFieldHash, ...]:
    return tuple(
        AssociationSourceFieldHash(
            field_kind=item.field_kind,
            parser_semantics_sha256=_build_field_parser_spec(
                item.field_kind
            ).parser_semantics_sha256,
            normalized_value_sha256=_normalized_source_field_sha256(item),
        )
        for item in source_fields
    )


class CanonicalAssociationSourceRecord(StrictFrozenModel):
    """Content-addressed adapter record anchoring raw association evidence."""

    schema_version: Literal["nurse.v2.canonical-association-source-record/2"] = (
        ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION
    )
    source_record_id: str = Field(pattern=r"^asrc_[0-9a-f]{24}$")
    observation_schema_version: Literal["nurse.v2.observation/2"] = OBSERVATION_SCHEMA_VERSION
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    decision_time_unix_ns: int = Field(ge=0)
    parent_source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    association_basis: AssociationBasis
    producer_kind: Literal["frozen_adapter"] = "frozen_adapter"
    derivation_rule_id: str = Field(pattern=r"^arule_[0-9a-f]{24}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_semantics: Literal["sha256-utf8-nfc/1"] = ASSOCIATION_NORMALIZATION_SEMANTICS
    source_fields: tuple[AdapterAssociationSourceField, ...] = Field(min_length=1)
    source_field_hashes: tuple[AssociationSourceFieldHash, ...] = Field(min_length=1)
    association_source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_canonical_record(self) -> Self:
        policy = build_frozen_association_policy()
        rule = policy.rule_for(self.association_basis)
        if (
            self.producer_kind != rule.producer_kind
            or self.derivation_rule_id != rule.derivation_rule_id
            or self.derivation_semantics_sha256 != rule.derivation_semantics_sha256
            or self.normalization_semantics != rule.normalization_semantics
        ):
            raise ValueError("association source record does not use the frozen rule")
        _validate_source_fields(rule, self.source_fields)
        if self.source_field_hashes != _source_field_hashes(self.source_fields):
            raise ValueError("association source field hashes are not derived from raw fields")
        if self.association_source_record_sha256 != _association_source_record_sha256(self):
            raise ValueError("association source record hash is not content-addressed")
        if self.source_record_id != _association_source_record_id(self):
            raise ValueError("association source record ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = CanonicalAssociationSourceRecord.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _association_source_record_sha256(record: CanonicalAssociationSourceRecord) -> str:
    unsigned = record.model_dump(
        mode="json",
        exclude={"source_record_id", "association_source_record_sha256"},
    )
    return sha256_hex(canonical_json_bytes(unsigned))


def _association_source_record_id(record: CanonicalAssociationSourceRecord) -> str:
    return _stable_id(
        "asrc",
        record.model_dump(mode="json", exclude={"source_record_id"}),
    )


def build_canonical_association_source_record(
    *,
    opaque_case_id: str,
    observation_id: str,
    decision_time_unix_ns: int,
    parent_source_fingerprint: str,
    adapter_semantics_sha256: str,
    association_basis: AssociationBasis,
    source_fields: Sequence[AdapterAssociationSourceField],
) -> CanonicalAssociationSourceRecord:
    """Build the only accepted association anchor from allowlisted raw fields."""

    validated_fields = tuple(
        AdapterAssociationSourceField.model_validate(item.model_dump(mode="json", warnings="error"))
        for item in source_fields
    )
    ordered_fields = tuple(
        sorted(
            validated_fields,
            key=lambda item: _ASSOCIATION_SOURCE_FIELD_ORDER_INDEX[item.field_kind],
        )
    )
    policy = build_frozen_association_policy()
    rule = policy.rule_for(association_basis)
    _validate_source_fields(rule, ordered_fields)
    unsigned: dict[str, object] = {
        "schema_version": ASSOCIATION_SOURCE_RECORD_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "observation_id": observation_id,
        "decision_time_unix_ns": decision_time_unix_ns,
        "parent_source_fingerprint": parent_source_fingerprint,
        "adapter_semantics_sha256": adapter_semantics_sha256,
        "association_basis": association_basis,
        "producer_kind": rule.producer_kind,
        "derivation_rule_id": rule.derivation_rule_id,
        "derivation_semantics_sha256": rule.derivation_semantics_sha256,
        "normalization_semantics": rule.normalization_semantics,
        "source_fields": [item.model_dump(mode="json") for item in ordered_fields],
        "source_field_hashes": [
            item.model_dump(mode="json") for item in _source_field_hashes(ordered_fields)
        ],
    }
    record_sha256 = sha256_hex(canonical_json_bytes(unsigned))
    with_hash = {
        **unsigned,
        "association_source_record_sha256": record_sha256,
    }
    return CanonicalAssociationSourceRecord.model_validate(
        {**with_hash, "source_record_id": _stable_id("asrc", with_hash)}
    )


class AdapterAssociationIdentity(StrictFrozenModel):
    """Source-bound typed adapter derivation whose identity digest is replayable."""

    schema_version: Literal["nurse.v2.adapter-association-identity/3"] = (
        ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION
    )
    adapter_identity_id: str = Field(pattern=r"^aid_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    decision_time_unix_ns: int = Field(ge=0)
    association_basis: AssociationBasis
    derivation_kind: AssociationDerivationKind
    producer_kind: Literal["frozen_adapter"] = "frozen_adapter"
    adapter_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_rule_id: str = Field(pattern=r"^arule_[0-9a-f]{24}$")
    derivation_semantics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    association_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_semantics: Literal["sha256-utf8-nfc/1"] = ASSOCIATION_NORMALIZATION_SEMANTICS
    source_record: CanonicalAssociationSourceRecord
    association_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def source_fingerprint(self) -> str:
        return self.source_record.parent_source_fingerprint

    @property
    def source_reference_sha256(self) -> str:
        return self.source_record.association_source_record_sha256

    @property
    def source_fields(self) -> tuple[AdapterAssociationSourceField, ...]:
        return self.source_record.source_fields

    @model_validator(mode="after")
    def require_frozen_typed_derivation(self) -> Self:
        policy = build_frozen_association_policy()
        rule = policy.rule_for(self.association_basis)
        if (
            self.producer_kind != rule.producer_kind
            or self.derivation_kind != rule.derivation_kind
            or self.derivation_rule_id != rule.derivation_rule_id
            or self.derivation_semantics_sha256 != rule.derivation_semantics_sha256
            or self.association_policy_sha256 != policy.policy_sha256
            or self.normalization_semantics != rule.normalization_semantics
        ):
            raise ValueError("adapter association identity does not use the frozen rule")
        if not self.source_record.verify():
            raise ValueError("association source record failed verification")
        if (
            self.source_record.association_basis != self.association_basis
            or self.source_record.derivation_rule_id != self.derivation_rule_id
            or self.source_record.derivation_semantics_sha256 != self.derivation_semantics_sha256
            or self.source_record.adapter_semantics_sha256 != self.adapter_semantics_sha256
            or self.source_record.normalization_semantics != self.normalization_semantics
            or self.source_record.opaque_case_id != self.opaque_case_id
            or self.source_record.observation_id != self.observation_id
            or self.source_record.decision_time_unix_ns != self.decision_time_unix_ns
        ):
            raise ValueError("adapter association identity does not match its source record")
        if self.association_identity_sha256 != _association_identity_sha256(
            basis=self.association_basis,
            derivation_kind=self.derivation_kind,
            source_field_hashes=self.source_record.source_field_hashes,
        ):
            raise ValueError("adapter association identity digest is not derived from typed fields")
        if self.adapter_identity_id != _adapter_association_identity_id(self):
            raise ValueError("adapter association identity ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = AdapterAssociationIdentity.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _validate_source_fields(
    rule: AssociationDerivationRule,
    source_fields: tuple[AdapterAssociationSourceField, ...],
) -> None:
    ordered = tuple(
        sorted(
            source_fields,
            key=lambda item: _ASSOCIATION_SOURCE_FIELD_ORDER_INDEX[item.field_kind],
        )
    )
    if source_fields != ordered:
        raise ValueError("adapter association source fields must be canonically ordered")
    field_kinds = tuple(item.field_kind for item in source_fields)
    if len(field_kinds) != len(set(field_kinds)):
        raise ValueError("adapter association source fields must be unique")
    if not set(field_kinds).issubset(rule.allowed_source_fields):
        raise ValueError("association source field is forbidden for the derivation basis")
    if field_kinds not in rule.allowed_source_field_combinations:
        raise ValueError("association source field combination is not allowed for the basis")


def _association_identity_sha256(
    *,
    basis: AssociationBasis,
    derivation_kind: AssociationDerivationKind,
    source_field_hashes: tuple[AssociationSourceFieldHash, ...],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "association_basis": basis,
                "derivation_kind": derivation_kind,
                "normalization_semantics": ASSOCIATION_NORMALIZATION_SEMANTICS,
                "source_field_hashes": [
                    item.model_dump(mode="json") for item in source_field_hashes
                ],
            }
        )
    )


def _adapter_association_identity_id(identity: AdapterAssociationIdentity) -> str:
    return _stable_id(
        "aid",
        identity.model_dump(mode="json", exclude={"adapter_identity_id"}),
    )


def build_adapter_association_identity(
    *,
    observation: Observation,
    source_record: CanonicalAssociationSourceRecord,
) -> AdapterAssociationIdentity:
    """Bind one verified canonical source record to its exact observation."""

    validated_observation = Observation.model_validate(
        observation.model_dump(mode="json", warnings="error")
    )
    validated_record = CanonicalAssociationSourceRecord.model_validate(
        source_record.model_dump(mode="json", warnings="error")
    )
    if not validated_record.verify():
        raise ValueError("association source record failed verification")
    if validated_observation.association_source_record_sha256 is None:
        raise ValueError(
            "adapter association identity requires an observation source record anchor"
        )
    if (
        validated_record.association_source_record_sha256
        != validated_observation.association_source_record_sha256
    ):
        raise ValueError("association source record hash does not match observation")
    if (
        validated_record.observation_id != validated_observation.observation_id
        or validated_record.opaque_case_id != validated_observation.opaque_case_id
        or validated_record.decision_time_unix_ns != validated_observation.decision_time_unix_ns
        or validated_record.parent_source_fingerprint != validated_observation.source_fingerprint
        or validated_record.adapter_semantics_sha256
        != validated_observation.adapter_semantics_sha256
        or validated_record.observation_schema_version != validated_observation.schema_version
    ):
        raise ValueError("association source record does not exactly bind the observation")
    policy = build_frozen_association_policy()
    rule = policy.rule_for(validated_record.association_basis)
    unsigned: dict[str, object] = {
        "schema_version": ADAPTER_ASSOCIATION_IDENTITY_SCHEMA_VERSION,
        "opaque_case_id": validated_observation.opaque_case_id,
        "observation_id": validated_observation.observation_id,
        "decision_time_unix_ns": validated_observation.decision_time_unix_ns,
        "association_basis": validated_record.association_basis,
        "derivation_kind": rule.derivation_kind,
        "producer_kind": rule.producer_kind,
        "adapter_semantics_sha256": validated_observation.adapter_semantics_sha256,
        "derivation_rule_id": rule.derivation_rule_id,
        "derivation_semantics_sha256": rule.derivation_semantics_sha256,
        "association_policy_sha256": policy.policy_sha256,
        "normalization_semantics": rule.normalization_semantics,
        "source_record": validated_record.model_dump(mode="json"),
        "association_identity_sha256": _association_identity_sha256(
            basis=validated_record.association_basis,
            derivation_kind=rule.derivation_kind,
            source_field_hashes=validated_record.source_field_hashes,
        ),
    }
    return AdapterAssociationIdentity.model_validate(
        {**unsigned, "adapter_identity_id": _stable_id("aid", unsigned)}
    )


class ObservationAssociationProof(StrictFrozenModel):
    """Decision-bounded proof retaining its complete typed adapter derivation."""

    schema_version: Literal["nurse.v2.observation-association-proof/3"] = (
        OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION
    )
    proof_id: str = Field(pattern=r"^aproof_[0-9a-f]{24}$")
    evidence_time_unix_ns: int = Field(ge=0)
    adapter_identity: AdapterAssociationIdentity

    @property
    def observation_id(self) -> str:
        return self.adapter_identity.observation_id

    @property
    def association_basis(self) -> AssociationBasis:
        return self.adapter_identity.association_basis

    @property
    def association_identity_sha256(self) -> str:
        return self.adapter_identity.association_identity_sha256

    @property
    def producer_kind(self) -> Literal["frozen_adapter"]:
        return self.adapter_identity.producer_kind

    @property
    def adapter_semantics_sha256(self) -> str:
        return self.adapter_identity.adapter_semantics_sha256

    @property
    def source_fingerprint(self) -> str:
        return self.adapter_identity.source_fingerprint

    @property
    def source_reference_sha256(self) -> str:
        return self.adapter_identity.source_reference_sha256

    @property
    def source_record(self) -> CanonicalAssociationSourceRecord:
        return self.adapter_identity.source_record

    @model_validator(mode="after")
    def require_bounded_content_identity(self) -> Self:
        if self.evidence_time_unix_ns > self.adapter_identity.decision_time_unix_ns:
            raise ValueError("association evidence is post-decision")
        if not self.adapter_identity.verify():
            raise ValueError("adapter association identity failed verification")
        if self.proof_id != _observation_association_proof_id(self):
            raise ValueError("observation association proof ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = ObservationAssociationProof.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _observation_association_proof_id(proof: ObservationAssociationProof) -> str:
    return _stable_id(
        "aproof",
        proof.model_dump(mode="json", exclude={"proof_id"}),
    )


def build_observation_association_proof(
    *,
    observation: Observation,
    evidence_time_unix_ns: int,
    adapter_identity: AdapterAssociationIdentity,
) -> ObservationAssociationProof:
    """Create a proof only from an already typed, source-bound adapter identity."""

    validated_observation = Observation.model_validate(
        observation.model_dump(mode="json", warnings="error")
    )
    validated_identity = AdapterAssociationIdentity.model_validate(
        adapter_identity.model_dump(mode="json", warnings="error")
    )
    if (
        validated_identity.opaque_case_id != validated_observation.opaque_case_id
        or validated_identity.observation_id != validated_observation.observation_id
        or validated_identity.decision_time_unix_ns != validated_observation.decision_time_unix_ns
        or validated_identity.source_fingerprint != validated_observation.source_fingerprint
        or validated_identity.source_reference_sha256
        != validated_observation.association_source_record_sha256
        or validated_identity.adapter_semantics_sha256
        != validated_observation.adapter_semantics_sha256
    ):
        raise ValueError("adapter association identity does not exactly bind the observation")
    unsigned: dict[str, object] = {
        "schema_version": OBSERVATION_ASSOCIATION_PROOF_SCHEMA_VERSION,
        "evidence_time_unix_ns": evidence_time_unix_ns,
        "adapter_identity": validated_identity.model_dump(mode="json"),
    }
    return ObservationAssociationProof.model_validate(
        {**unsigned, "proof_id": _stable_id("aproof", unsigned)}
    )


class AssociationClaim(StrictFrozenModel):
    """Decision-bounded proof linking explicit observation references."""

    schema_version: Literal["nurse.v2.event-association-claim/4"] = ASSOCIATION_CLAIM_SCHEMA_VERSION
    claim_id: str = Field(pattern=r"^assoc_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    decision_time_unix_ns: int = Field(ge=0)
    evidence_time_unix_ns: int = Field(ge=0)
    association_basis: AssociationBasis
    association_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    association_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_rule_id: str = Field(pattern=r"^arule_[0-9a-f]{24}$")
    observation_ids: tuple[str, ...] = Field(min_length=1)
    proof_references: tuple[ObservationAssociationProof, ...] = Field(min_length=1)
    source_fingerprints: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_bounded_claim(self) -> Self:
        for values, name in (
            (self.observation_ids, "association observation IDs"),
            (self.source_fingerprints, "association source fingerprints"),
            (self.provenance_ids, "association provenance IDs"),
        ):
            _require_canonical(values, name)
        if self.proof_references != tuple(
            sorted(self.proof_references, key=lambda item: item.proof_id)
        ):
            raise ValueError("association proofs must be canonically ordered")
        if any(not proof.verify() for proof in self.proof_references):
            raise ValueError("association proof failed verification")
        policy = build_frozen_association_policy()
        rule = policy.rule_for(self.association_basis)
        if (
            self.association_policy_sha256 != policy.policy_sha256
            or self.derivation_rule_id != rule.derivation_rule_id
        ):
            raise ValueError("association claim does not use the frozen policy rule")
        if any(
            proof.association_basis != self.association_basis
            or proof.adapter_identity.opaque_case_id != self.opaque_case_id
            or proof.adapter_identity.decision_time_unix_ns != self.decision_time_unix_ns
            or proof.adapter_identity.association_policy_sha256 != self.association_policy_sha256
            or proof.adapter_identity.derivation_rule_id != self.derivation_rule_id
            for proof in self.proof_references
        ):
            raise ValueError("association proof does not match the claim boundary")
        identities = {proof.association_identity_sha256 for proof in self.proof_references}
        if identities != {self.association_key_sha256}:
            raise ValueError("association proofs do not derive one shared identity")
        expected_observations = tuple(
            sorted(proof.observation_id for proof in self.proof_references)
        )
        if len(expected_observations) != len(set(expected_observations)):
            raise ValueError("association observation references must be unique")
        if self.observation_ids != expected_observations:
            raise ValueError("association claim observations are not proof-covered")
        if self.source_fingerprints != tuple(
            sorted({proof.source_fingerprint for proof in self.proof_references})
        ):
            raise ValueError("association source fingerprints are not proof-derived")
        if self.provenance_ids != tuple(sorted(proof.proof_id for proof in self.proof_references)):
            raise ValueError("association provenance is not proof-derived")
        if self.evidence_time_unix_ns != max(
            proof.evidence_time_unix_ns for proof in self.proof_references
        ):
            raise ValueError("association evidence time is not proof-derived")
        if self.evidence_time_unix_ns > self.decision_time_unix_ns:
            raise ValueError("association evidence is post-decision")
        if self.claim_id != _association_claim_id(self):
            raise ValueError("association claim ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = AssociationClaim.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _association_claim_id(claim: AssociationClaim) -> str:
    return _stable_id(
        "assoc",
        claim.model_dump(mode="json", exclude={"claim_id"}),
    )


def build_association_claim(
    *,
    opaque_case_id: str,
    decision_time_unix_ns: int,
    association_basis: AssociationBasis,
    proof_references: Sequence[ObservationAssociationProof],
) -> AssociationClaim:
    """Build a claim entirely from typed, source-bound adapter proofs."""

    validated_proofs = tuple(
        ObservationAssociationProof.model_validate(proof.model_dump(mode="json", warnings="error"))
        for proof in proof_references
    )
    if not validated_proofs:
        raise ValueError("association claim requires typed proofs")
    observation_ids = tuple(proof.observation_id for proof in validated_proofs)
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("association observation references must be unique")
    ordered_proofs = tuple(sorted(validated_proofs, key=lambda item: item.proof_id))
    policy = build_frozen_association_policy()
    rule = policy.rule_for(association_basis)
    identities = {proof.association_identity_sha256 for proof in ordered_proofs}
    if len(identities) != 1:
        raise ValueError("association proofs do not derive one shared identity")
    unsigned: dict[str, object] = {
        "schema_version": ASSOCIATION_CLAIM_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "decision_time_unix_ns": decision_time_unix_ns,
        "evidence_time_unix_ns": max(proof.evidence_time_unix_ns for proof in ordered_proofs),
        "association_basis": association_basis,
        "association_key_sha256": next(iter(identities)),
        "association_policy_sha256": policy.policy_sha256,
        "derivation_rule_id": rule.derivation_rule_id,
        "observation_ids": tuple(sorted(observation_ids)),
        "proof_references": [proof.model_dump(mode="json") for proof in ordered_proofs],
        "source_fingerprints": tuple(
            sorted({proof.source_fingerprint for proof in ordered_proofs})
        ),
        "provenance_ids": tuple(sorted(proof.proof_id for proof in ordered_proofs)),
    }
    return AssociationClaim.model_validate({**unsigned, "claim_id": _stable_id("assoc", unsigned)})


class MemberSignalFamilyAssignment(StrictFrozenModel):
    """Auditable exact mapping decision for one observation."""

    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    adapter_semantics_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_signal_family: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    primary_signal_family: PrimarySignalFamily
    assignment_basis: Literal["exact_frozen_mapping", "missing_mapping_unknown_safe"]
    mapping_entry_id: str | None = Field(default=None, pattern=r"^fam_[0-9a-f]{24}$")
    provenance_ids: tuple[str, ...]

    @model_validator(mode="after")
    def require_consistent_assignment(self) -> Self:
        _require_canonical(self.provenance_ids, "family-assignment provenance IDs")
        if self.assignment_basis == "exact_frozen_mapping":
            if self.mapping_entry_id is None or not self.provenance_ids:
                raise ValueError("exact family assignment requires mapping evidence")
        elif self.mapping_entry_id is not None or self.primary_signal_family != "unclassified":
            raise ValueError("missing mapping must remain explicitly unclassified")
        return self


class EventGroup(StrictFrozenModel):
    """All retained exports of one proven event, or one unassociated singleton."""

    schema_version: Literal["nurse.v2.event-group/2"] = EVENT_GROUP_SCHEMA_VERSION
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    association_basis: GroupAssociationBasis
    association_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    association_claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...] = Field(min_length=1)
    entity_binding_ids: tuple[str, ...]
    primary_signal_family: PrimarySignalFamily
    family_assignment_status: Literal[
        "exact_mapped",
        "exact_mapped_with_unmapped_members",
        "degraded_unclassified",
    ]
    family_mapping_version: str = Field(min_length=1, max_length=128)
    family_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_family_assignments: tuple[MemberSignalFamilyAssignment, ...] = Field(min_length=1)
    interval: TimeInterval
    source_fingerprints: tuple[str, ...] = Field(min_length=1)
    provenance_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_group(self) -> Self:
        for values, name in (
            (self.association_claim_ids, "association claim IDs"),
            (self.observation_ids, "observation IDs"),
            (self.entity_binding_ids, "entity binding IDs"),
            (self.source_fingerprints, "source fingerprints"),
            (self.provenance_ids, "provenance IDs"),
        ):
            _require_canonical(values, name)
        member_ids = tuple(item.observation_id for item in self.member_family_assignments)
        if member_ids != self.observation_ids:
            raise ValueError("member family assignments must cover each group observation once")
        is_singleton = self.association_basis == "singleton_no_association_evidence"
        if is_singleton != (not self.association_claim_ids):
            raise ValueError("singleton and association-claim basis are inconsistent")
        if is_singleton and len(self.observation_ids) != 1:
            raise ValueError("no-evidence group must contain exactly one observation")
        has_unmapped_members = any(
            item.assignment_basis == "missing_mapping_unknown_safe"
            for item in self.member_family_assignments
        )
        mapped_families = {
            item.primary_signal_family
            for item in self.member_family_assignments
            if item.assignment_basis == "exact_frozen_mapping"
        }
        if len(mapped_families) > 1:
            raise ValueError("group members contain conflicting mapped primary families")
        if mapped_families:
            expected_family = next(iter(mapped_families))
            expected_status = (
                "exact_mapped_with_unmapped_members" if has_unmapped_members else "exact_mapped"
            )
        else:
            expected_family = "unclassified"
            expected_status = "degraded_unclassified"
        if self.family_assignment_status != expected_status:
            raise ValueError("group family-assignment status does not match its members")
        if self.primary_signal_family != expected_family:
            raise ValueError("group primary family does not match exact member assignments")
        if self.event_group_id != _event_group_id(self):
            raise ValueError("event group ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = EventGroup.model_validate(self.model_dump(mode="json", warnings="error"))
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _event_group_id(group: EventGroup) -> str:
    return _stable_id(
        "evt",
        group.model_dump(mode="json", exclude={"event_group_id"}),
    )


class GroupingDecision(StrictFrozenModel):
    """Auditable record of one group or singleton decision."""

    schema_version: Literal["nurse.v2.event-grouping-decision/2"] = GROUPING_DECISION_SCHEMA_VERSION
    decision_id: str = Field(pattern=r"^gdec_[0-9a-f]{24}$")
    event_group_id: str = Field(pattern=r"^evt_[0-9a-f]{24}$")
    decision: Literal["grouped_by_explicit_identity", "retained_as_singleton"]
    association_basis: GroupAssociationBasis
    association_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    association_claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_decision(self) -> Self:
        _require_canonical(self.association_claim_ids, "decision association claim IDs")
        _require_canonical(self.observation_ids, "decision observation IDs")
        singleton = self.decision == "retained_as_singleton"
        if singleton != (self.association_basis == "singleton_no_association_evidence"):
            raise ValueError("grouping decision does not match association basis")
        if singleton and (len(self.observation_ids) != 1 or self.association_claim_ids):
            raise ValueError("singleton decision must reference one unclaimed observation")
        if not singleton and not self.association_claim_ids:
            raise ValueError("non-singleton decision requires at least one association claim")
        if self.decision_id != _grouping_decision_id(self):
            raise ValueError("grouping decision ID is not content-addressed")
        return self

    def verify(self) -> bool:
        try:
            validated = GroupingDecision.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return validated == self


def _grouping_decision_id(decision: GroupingDecision) -> str:
    return _stable_id(
        "gdec",
        decision.model_dump(mode="json", exclude={"decision_id"}),
    )


class EventGroupingResult(StrictFrozenModel):
    """Replayable case-atomic event grouping result."""

    schema_version: Literal["nurse.v2.event-grouping-result/4"] = (
        EVENT_GROUPING_RESULT_SCHEMA_VERSION
    )
    opaque_case_id: str = Field(pattern=r"^opq_[0-9a-f]{32}$")
    decision_time_unix_ns: int = Field(ge=0)
    observations: tuple[Observation, ...]
    association_policy: FrozenAssociationPolicy
    association_claims: tuple[AssociationClaim, ...]
    family_mapping: FrozenSignalFamilyMapping
    event_groups: tuple[EventGroup, ...]
    decisions: tuple[GroupingDecision, ...]
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        try:
            validated = EventGroupingResult.model_validate(
                self.model_dump(mode="json", warnings="error")
            )
            if validated != self or not validated._verify_hash_and_structure():
                return False
            expected = _rebuild_event_grouping_result(validated)
        except (AttributeError, TypeError, ValueError):
            return False
        return expected == validated

    def _verify_hash_and_structure(self) -> bool:
        if self.observations != tuple(
            sorted(self.observations, key=lambda item: item.observation_id)
        ):
            return False
        if self.association_claims != tuple(
            sorted(self.association_claims, key=lambda item: item.claim_id)
        ):
            return False
        if self.event_groups != tuple(
            sorted(self.event_groups, key=lambda item: item.event_group_id)
        ):
            return False
        if self.decisions != tuple(sorted(self.decisions, key=lambda item: item.decision_id)):
            return False
        observation_ids = tuple(item.observation_id for item in self.observations)
        memberships = tuple(
            observation_id
            for group in self.event_groups
            for observation_id in group.observation_ids
        )
        if sorted(memberships) != sorted(observation_ids):
            return False
        if len(memberships) != len(set(memberships)):
            return False
        if len(self.event_groups) != len(self.decisions):
            return False
        if {(item.event_group_id, item.observation_ids) for item in self.decisions} != {
            (item.event_group_id, item.observation_ids) for item in self.event_groups
        }:
            return False
        if not self.family_mapping.verify():
            return False
        if not self.association_policy.verify():
            return False
        if any(not claim.verify() for claim in self.association_claims):
            return False
        if any(not group.verify() for group in self.event_groups):
            return False
        if any(not decision.verify() for decision in self.decisions):
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_sha256"})
        return self.result_sha256 == sha256_hex(canonical_json_bytes(unsigned))


def _family_assignment(
    observation: Observation,
    lookup: dict[tuple[str, str], SignalFamilyMappingEntry],
) -> MemberSignalFamilyAssignment:
    semantics = observation.adapter_semantics_sha256
    entry = lookup.get((semantics, observation.signal_family)) if semantics is not None else None
    if entry is None:
        return MemberSignalFamilyAssignment(
            observation_id=observation.observation_id,
            adapter_semantics_sha256=semantics,
            source_signal_family=observation.signal_family,
            primary_signal_family="unclassified",
            assignment_basis="missing_mapping_unknown_safe",
            mapping_entry_id=None,
            provenance_ids=observation.provenance_ids,
        )
    return MemberSignalFamilyAssignment(
        observation_id=observation.observation_id,
        adapter_semantics_sha256=semantics,
        source_signal_family=observation.signal_family,
        primary_signal_family=entry.primary_signal_family,
        assignment_basis="exact_frozen_mapping",
        mapping_entry_id=entry.mapping_entry_id,
        provenance_ids=entry.provenance_ids,
    )


def _group_payload(
    *,
    opaque_case_id: str,
    basis: GroupAssociationBasis,
    key_sha256: str,
    observations: tuple[Observation, ...],
    claims: tuple[AssociationClaim, ...],
    mapping: FrozenSignalFamilyMapping,
    lookup: dict[tuple[str, str], SignalFamilyMappingEntry],
) -> dict[str, object]:
    assignments = tuple(_family_assignment(item, lookup) for item in observations)
    mapped_families = {
        item.primary_signal_family
        for item in assignments
        if item.assignment_basis == "exact_frozen_mapping"
    }
    if len(mapped_families) > 1:
        raise ValueError("associated observations have conflicting primary signal families")
    has_unmapped_members = any(
        item.assignment_basis == "missing_mapping_unknown_safe" for item in assignments
    )
    if mapped_families:
        primary_family: PrimarySignalFamily = next(iter(mapped_families))
        assignment_status = (
            "exact_mapped_with_unmapped_members" if has_unmapped_members else "exact_mapped"
        )
    else:
        primary_family = "unclassified"
        assignment_status = "degraded_unclassified"
    return {
        "schema_version": EVENT_GROUP_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "association_basis": basis,
        "association_key_sha256": key_sha256,
        "association_claim_ids": tuple(sorted(claim.claim_id for claim in claims)),
        "observation_ids": tuple(item.observation_id for item in observations),
        "entity_binding_ids": tuple(
            sorted(
                {
                    item.entity_binding_id
                    for item in observations
                    if item.entity_binding_id is not None
                }
            )
        ),
        "primary_signal_family": primary_family,
        "family_assignment_status": assignment_status,
        "family_mapping_version": mapping.mapping_version,
        "family_mapping_sha256": mapping.mapping_sha256,
        "member_family_assignments": [item.model_dump(mode="json") for item in assignments],
        "interval": {
            "start_unix_ns": min(item.interval.start_unix_ns for item in observations),
            "end_unix_ns": max(item.interval.end_unix_ns for item in observations),
        },
        "source_fingerprints": tuple(
            sorted(
                {item.source_fingerprint for item in observations}
                | {fingerprint for claim in claims for fingerprint in claim.source_fingerprints}
            )
        ),
        "provenance_ids": tuple(
            sorted(
                {provenance_id for item in observations for provenance_id in item.provenance_ids}
                | {provenance_id for claim in claims for provenance_id in claim.provenance_ids}
            )
        ),
    }


def build_event_grouping_result(
    *,
    opaque_case_id: str,
    decision_time_unix_ns: int,
    observations: Sequence[Observation],
    association_claims: Sequence[AssociationClaim],
    family_mapping: FrozenSignalFamilyMapping,
    association_policy: FrozenAssociationPolicy | None = None,
) -> EventGroupingResult:
    """Group only by explicit identities; retain every other observation as a singleton."""

    policy_input = (
        build_frozen_association_policy() if association_policy is None else association_policy
    )
    validated_policy = FrozenAssociationPolicy.model_validate(
        policy_input.model_dump(mode="json", warnings="error")
    )
    if not validated_policy.verify():
        raise ValueError("association policy failed frozen verification")
    validated_mapping = FrozenSignalFamilyMapping.model_validate(
        family_mapping.model_dump(mode="json", warnings="error")
    )
    if not validated_mapping.verify():
        raise ValueError("signal-family mapping failed verification")
    validated_observations = tuple(
        Observation.model_validate(item.model_dump(mode="json", warnings="error"))
        for item in observations
    )
    observation_ids = tuple(item.observation_id for item in validated_observations)
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("observation IDs must be unique")
    ordered_observations = tuple(
        sorted(validated_observations, key=lambda item: item.observation_id)
    )
    observation_by_id = {item.observation_id: item for item in ordered_observations}
    for observation in ordered_observations:
        if observation.opaque_case_id != opaque_case_id:
            raise ValueError("observation belongs to a different opaque case")
        if observation.decision_time_unix_ns != decision_time_unix_ns:
            raise ValueError("observation decision time does not match grouping result")

    validated_claims = tuple(
        AssociationClaim.model_validate(item.model_dump(mode="json", warnings="error"))
        for item in association_claims
    )
    claim_ids = tuple(item.claim_id for item in validated_claims)
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("association claim IDs must be unique")
    ordered_claims = tuple(sorted(validated_claims, key=lambda item: item.claim_id))
    identity_by_observation: dict[str, tuple[AssociationBasis, str]] = {}
    claims_by_identity: dict[tuple[GroupAssociationBasis, str], list[AssociationClaim]] = {}
    for association in ordered_claims:
        if association.opaque_case_id != opaque_case_id:
            raise ValueError("association claim belongs to a different opaque case")
        if association.decision_time_unix_ns != decision_time_unix_ns:
            raise ValueError("association claim decision time does not match grouping result")
        if not association.verify():
            raise ValueError("association claim failed verification")
        if association.association_policy_sha256 != validated_policy.policy_sha256:
            raise ValueError("association claim policy does not match grouping result")
        claim_identity: tuple[AssociationBasis, str] = (
            association.association_basis,
            association.association_key_sha256,
        )
        claims_by_identity.setdefault(claim_identity, []).append(association)
        for proof in association.proof_references:
            observation_id = proof.observation_id
            if observation_id not in observation_by_id:
                raise ValueError("association claim references an unknown observation")
            observation = observation_by_id[observation_id]
            identity = proof.adapter_identity
            if identity.source_fingerprint != observation.source_fingerprint:
                raise ValueError("association proof source fingerprint does not match observation")
            if identity.source_reference_sha256 != observation.association_source_record_sha256:
                raise ValueError("association proof source reference does not match observation")
            if identity.adapter_semantics_sha256 != observation.adapter_semantics_sha256:
                raise ValueError("association proof adapter semantics do not match observation")
            if (
                identity.opaque_case_id != observation.opaque_case_id
                or identity.decision_time_unix_ns != observation.decision_time_unix_ns
                or identity.source_record.observation_id != observation.observation_id
                or identity.source_record.observation_schema_version != observation.schema_version
            ):
                raise ValueError("association proof case/time does not match observation")
            previous = identity_by_observation.get(observation_id)
            if previous is not None and previous != claim_identity:
                raise ValueError("observation has ambiguous or conflicting association identities")
            identity_by_observation[observation_id] = claim_identity

    grouped_ids: dict[tuple[GroupAssociationBasis, str], list[str]] = {}
    for observation in ordered_observations:
        observation_identity = identity_by_observation.get(observation.observation_id)
        if observation_identity is None:
            singleton_key = sha256_hex(
                canonical_json_bytes({"singleton_observation_id": observation.observation_id})
            )
            group_identity: tuple[GroupAssociationBasis, str] = (
                "singleton_no_association_evidence",
                singleton_key,
            )
        else:
            group_identity = observation_identity
        grouped_ids.setdefault(group_identity, []).append(observation.observation_id)

    lookup = {
        (entry.adapter_semantics_sha256, entry.source_signal_family): entry
        for entry in validated_mapping.entries
    }
    groups: list[EventGroup] = []
    decisions: list[GroupingDecision] = []
    for (basis, key_sha256), member_ids in grouped_ids.items():
        members = tuple(observation_by_id[item] for item in sorted(member_ids))
        identity_claims = tuple(
            sorted(
                claims_by_identity.get((basis, key_sha256), ()),
                key=lambda item: item.claim_id,
            )
        )
        group_unsigned = _group_payload(
            opaque_case_id=opaque_case_id,
            basis=basis,
            key_sha256=key_sha256,
            observations=members,
            claims=identity_claims,
            mapping=validated_mapping,
            lookup=lookup,
        )
        group = EventGroup.model_validate(
            {
                **group_unsigned,
                "event_group_id": _stable_id("evt", group_unsigned),
            }
        )
        decision_unsigned: dict[str, object] = {
            "schema_version": GROUPING_DECISION_SCHEMA_VERSION,
            "event_group_id": group.event_group_id,
            "decision": (
                "retained_as_singleton"
                if basis == "singleton_no_association_evidence"
                else "grouped_by_explicit_identity"
            ),
            "association_basis": basis,
            "association_key_sha256": key_sha256,
            "association_claim_ids": group.association_claim_ids,
            "observation_ids": group.observation_ids,
        }
        decisions.append(
            GroupingDecision.model_validate(
                {
                    **decision_unsigned,
                    "decision_id": _stable_id("gdec", decision_unsigned),
                }
            )
        )
        groups.append(group)

    ordered_groups = tuple(sorted(groups, key=lambda item: item.event_group_id))
    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.decision_id))
    unsigned: dict[str, object] = {
        "schema_version": EVENT_GROUPING_RESULT_SCHEMA_VERSION,
        "opaque_case_id": opaque_case_id,
        "decision_time_unix_ns": decision_time_unix_ns,
        "observations": [item.model_dump(mode="json") for item in ordered_observations],
        "association_policy": validated_policy.model_dump(mode="json"),
        "association_claims": [item.model_dump(mode="json") for item in ordered_claims],
        "family_mapping": validated_mapping.model_dump(mode="json"),
        "event_groups": [item.model_dump(mode="json") for item in ordered_groups],
        "decisions": [item.model_dump(mode="json") for item in ordered_decisions],
    }
    result = EventGroupingResult.model_validate(
        {**unsigned, "result_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
    if not result._verify_hash_and_structure():  # pragma: no cover - construction invariant
        raise AssertionError("event grouping result verification failed")
    return result


def _rebuild_event_grouping_result(result: EventGroupingResult) -> EventGroupingResult:
    return build_event_grouping_result(
        opaque_case_id=result.opaque_case_id,
        decision_time_unix_ns=result.decision_time_unix_ns,
        observations=result.observations,
        association_claims=result.association_claims,
        family_mapping=result.family_mapping,
        association_policy=result.association_policy,
    )


def replay_event_grouping_result(result: EventGroupingResult) -> EventGroupingResult:
    """Rebuild retained inputs and reject malformed or drifted derived state."""

    try:
        validated = EventGroupingResult.model_validate(
            result.model_dump(mode="json", warnings="error")
        )
        if validated != result or not validated._verify_hash_and_structure():
            raise ValueError("event grouping result structure failed verification")
        expected = _rebuild_event_grouping_result(validated)
        if expected != validated:
            raise ValueError("event grouping derived state drifted")
        return expected
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "event grouping result cannot be replayed because it does not replay exactly"
        ) from error
