"""Versioned, label-free mechanism templates for deterministic and semantic use.

The initial catalog is a code-declared allowlist, not caller-authored text.
Changing a template requires an explicit versioned catalog change; recomputing
a submitted digest never authorizes new names, summaries, or predicates.
Predicates describe observable conditions only. Their three-valued evaluation
and any score effects belong to the later verifier and compiler tasks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel
from nurse_rcl.v2.diagnostic_schemas import (
    DiagnosticRole,
    MechanismDefinition,
    build_mechanism_definition,
)

MECHANISM_LIBRARY_SCHEMA_VERSION: Literal["nurse.v2.mechanism-library/1"] = (
    "nurse.v2.mechanism-library/1"
)
INITIAL_MECHANISM_LIBRARY_VERSION: Literal["nurse-v2-mechanisms/1"] = "nurse-v2-mechanisms/1"


class _Template(NamedTuple):
    key: str
    summary: str
    necessary: tuple[str, ...]
    optional: tuple[str, ...]
    refuting: tuple[str, ...]
    roles: tuple[DiagnosticRole, ...]


# These are diagnostic possibilities, not benchmark classes or causal verdicts.
# A missing prerequisite remains unknown; a refuting condition must be observed,
# never inferred from an absent signal. No template supplies scoring weights.
_TEMPLATES = (
    _Template(
        "incoming_load_pressure",
        "Increased incoming request rate accompanies local pressure; the source remains open.",
        ("incoming_request_rate_increase", "local_pressure_observed"),
        ("incoming_load_precedes_local_pressure", "request_latency_increase"),
        ("incoming_request_rate_verified_normal",),
        ("root_support", "source_support", "symptom_support"),
    ),
    _Template(
        "local_resource_or_execution_pressure",
        "Local resource or execution pressure may explain the observed service degradation.",
        ("local_pressure_observed",),
        ("local_error_increase", "request_latency_increase"),
        ("local_resource_and_execution_verified_normal",),
        ("root_support", "source_support", "symptom_support"),
    ),
    _Template(
        "local_storage_or_network_failure",
        "Local storage or network operation failures may explain service degradation.",
        ("local_storage_or_network_failure_observed",),
        ("local_error_increase", "request_latency_increase"),
        ("local_storage_and_network_verified_normal",),
        ("root_support", "source_support", "symptom_support"),
    ),
    _Template(
        "received_dependency_failure",
        "The service observes failures on calls to a dependency; local alternatives remain open.",
        ("outgoing_dependency_failure_observed",),
        ("dependency_failure_precedes_local_symptom", "local_error_increase"),
        ("outgoing_dependency_calls_verified_normal",),
        ("symptom_support",),
    ),
    _Template(
        "unclassified_local_anomaly",
        "A local anomaly is observed without enough evidence for a more specific mechanism.",
        ("local_anomaly_observed",),
        (),
        ("local_signals_verified_normal",),
        ("symptom_support",),
    ),
    _Template(
        "workload_unavailability",
        "Observed workload unavailability may explain a loss of service availability.",
        ("workload_unavailability_observed",),
        ("lifecycle_transition_observed", "request_failure_increase"),
        ("workload_availability_verified_normal",),
        ("root_support", "source_support", "symptom_support"),
    ),
)


def _definition(template: _Template) -> MechanismDefinition:
    return build_mechanism_definition(
        mechanism_key=template.key,
        mechanism_version="1",
        summary=template.summary,
        necessary_predicate_kinds=template.necessary,
        optional_predicate_kinds=template.optional,
        refuting_predicate_kinds=template.refuting,
        allowed_roles=template.roles,
    )


def verify_declared_mechanism(definition: object) -> bool:
    """Require an exact current template, including its schema and content hash."""

    if type(definition) is not MechanismDefinition:
        return False
    if set(vars(definition)) != set(MechanismDefinition.model_fields):
        return False
    try:
        validated = MechanismDefinition.model_validate(
            definition.model_dump(mode="python", warnings="error")
        )
        if validated != definition:
            return False
        return any(
            definition == _definition(template)
            for template in _TEMPLATES
            if template.key == definition.mechanism_key
        )
    except (AttributeError, TypeError, ValueError):
        return False


class MechanismLibrary(StrictFrozenModel):
    """Complete initial catalog shared by B0 and B2, independent of any case."""

    schema_version: Literal["nurse.v2.mechanism-library/1"] = MECHANISM_LIBRARY_SCHEMA_VERSION
    library_version: Literal["nurse-v2-mechanisms/1"] = INITIAL_MECHANISM_LIBRARY_VERSION
    definitions: tuple[MechanismDefinition, ...] = Field(min_length=1)
    library_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_declared_catalog(self) -> Self:
        if not all(verify_declared_mechanism(item) for item in self.definitions):
            raise ValueError("mechanism library contains an undeclared definition")
        if tuple(item.mechanism_key for item in self.definitions) != tuple(
            sorted(template.key for template in _TEMPLATES)
        ):
            raise ValueError("mechanism library must be complete, sorted, and unique")
        unsigned = self.model_dump(mode="json", exclude={"library_sha256"}, warnings="error")
        if self.library_sha256 != sha256_hex(canonical_json_bytes(unsigned)):
            raise ValueError("mechanism library content digest does not match")
        return self

    def verify(self) -> bool:
        """Revalidate the outer contract and every child, including model_copy edits."""

        if type(self) is not MechanismLibrary:
            return False
        if set(vars(self)) != set(MechanismLibrary.model_fields):
            return False
        try:
            if not all(verify_declared_mechanism(item) for item in self.definitions):
                return False
            validated = MechanismLibrary.model_validate(
                self.model_dump(mode="python", warnings="error")
            )
            return validated == self
        except (AttributeError, TypeError, ValueError):
            return False

    def lookup(self, mechanism_key: str) -> MechanismDefinition:
        """Return a declared definition only from a verified complete catalog."""

        if not self.verify():
            raise ValueError("invalid mechanism library")
        for definition in self.definitions:
            if definition.mechanism_key == mechanism_key:
                return definition
        raise KeyError("unknown mechanism key")


def build_mechanism_library(
    *,
    definitions: Iterable[MechanismDefinition] | None = None,
) -> MechanismLibrary:
    """Build or replay the complete initial catalog without data or label inputs."""

    records = (
        tuple(_definition(template) for template in _TEMPLATES)
        if definitions is None
        else tuple(definitions)
    )
    if not all(verify_declared_mechanism(item) for item in records):
        raise ValueError("mechanism library contains an undeclared definition")
    ordered = tuple(sorted(records, key=lambda item: item.mechanism_key))
    unsigned: dict[str, object] = {
        "schema_version": MECHANISM_LIBRARY_SCHEMA_VERSION,
        "library_version": INITIAL_MECHANISM_LIBRARY_VERSION,
        "definitions": tuple(item.model_dump(mode="json") for item in ordered),
    }
    return MechanismLibrary.model_validate(
        {**unsigned, "library_sha256": sha256_hex(canonical_json_bytes(unsigned))}
    )
