"""Explicit fixed-input data and persistence receipts; no runtime authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.dynamic_verification.contracts import SHA256
from nurse_rcl.v2.live.repair_assembly import RepairDiagnosticInputs
from nurse_rcl.v2.live.repair_error_references import RepairErrorSources
from nurse_rcl.v2.live.repair_graph import RepairLiveGraph
from nurse_rcl.v2.live.repair_queries import RepairQueryCatalog
from nurse_rcl.v2.live.repair_safety import RepairModel
from nurse_rcl.v2.live.repair_sources import RepairMeasurementSource
from nurse_rcl.v2.temporal_condensation import FrozenClockResolution


@dataclass(frozen=True, slots=True)
class RepairPreparationData:
    diagnostics: RepairDiagnosticInputs
    sources: tuple[RepairMeasurementSource, ...]
    catalog: RepairQueryCatalog
    graph: RepairLiveGraph
    clock: FrozenClockResolution
    error_sources: RepairErrorSources
    error_observation_sha256s: tuple[str, ...]
    source_descriptor_sha256: str
    source_files_sha256: str
    candidate_freeze_sha256: str


class RepairPreparationKey(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-preparation-key/1"] = (
        "nurse.v2.live-repair-preparation-key/1"
    )
    dataset: Literal["rcabench", "aiops25"]
    opaque_case_id: str
    input_sha256: SHA256
    source_descriptor_sha256: SHA256
    source_files: tuple[tuple[str, str | None], ...]
    configuration_sha256: SHA256
    authorized_queue_sha256: SHA256 | None
    source_case_id_sha256: SHA256 | None
    q_base: Literal[8] = 8
    q_max: Literal[12] = 12
    preparation_code: tuple[tuple[str, str], ...]
    semantics_sha256: SHA256
    key_sha256: SHA256

    @field_validator("source_files", "preparation_code", mode="before")
    @classmethod
    def json_pairs(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and type(value) in (list, tuple):
            return tuple(
                tuple(item) if type(item) is list else item
                for item in cast(list[object] | tuple[object, ...], value)
            )
        return value

    @model_validator(mode="after")
    def complete_key(self) -> Self:
        if len({n for n, _ in self.source_files}) != len(self.source_files):
            raise ValueError("preparation key has duplicate sources")
        names = tuple(n for n, _ in self.preparation_code)
        if not names or names != tuple(sorted(set(names))):
            raise ValueError("preparation key requires ordered complete code pins")
        body = self.model_dump(mode="json", exclude={"key_sha256"})
        if self.key_sha256 != sha256_hex(canonical_json_bytes(body)):
            raise ValueError("preparation key digest mismatch")
        return self


class RepairPreparationMember(RepairModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_./-]{0,200}$")
    size_bytes: int = Field(ge=0)
    sha256: SHA256

    @model_validator(mode="after")
    def relative_member(self) -> Self:
        if any(p in ("", ".", "..") for p in self.name.split("/")):
            raise ValueError("preparation member path escapes archive")
        return self


class RepairPreparationTable(RepairModel):
    name: str = Field(pattern=r"^tables/[0-9]{6}\.arrow$")
    rows: int = Field(ge=0)
    schema_sha256: SHA256


class RepairPreparationManifest(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-preparation-manifest/1"] = (
        "nurse.v2.live-repair-preparation-manifest/1"
    )
    key: RepairPreparationKey
    members: tuple[RepairPreparationMember, ...]
    collections: tuple[tuple[str, int, str], ...]
    tables: tuple[RepairPreparationTable, ...]
    build_seconds: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("collections", mode="before")
    @classmethod
    def json_collections(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and type(value) in (list, tuple):
            return tuple(
                tuple(item) if type(item) is list else item
                for item in cast(list[object] | tuple[object, ...], value)
            )
        return value

    @model_validator(mode="after")
    def complete_members(self) -> Self:
        names = tuple(m.name for m in self.members)
        if not names or names != tuple(sorted(set(names))) or "manifest.json" in names:
            raise ValueError("preparation manifest member domain differs")
        if tuple(t.name for t in self.tables) != tuple(n for n in names if n.endswith(".arrow")):
            raise ValueError("preparation table inventory differs from members")
        return self


class RepairPreparationReceipt(RepairModel):
    schema_version: Literal["nurse.v2.live-repair-preparation-receipt/1"] = (
        "nurse.v2.live-repair-preparation-receipt/1"
    )
    execution_sha256: SHA256
    execution_code_sha256: SHA256
    input_sha256: SHA256
    preparation_key_sha256: SHA256
    manifest_sha256: SHA256
    preparation_exits_completed: Literal[True] = True


class RepairPreparationBinding(RepairModel):
    preparation_key_sha256: SHA256
    manifest_sha256: SHA256
    receipt_sha256: SHA256


class RepairPreparationRun(RepairModel):
    """Pinned plan data only. The strict executor owns whether it may be used."""

    key: RepairPreparationKey
    archive_root: str
    mode: Literal["build", "load"]
    receipt_path: str
    manifest_sha256: SHA256 | None
    receipt_sha256: SHA256 | None

    @model_validator(mode="after")
    def independent_pins(self) -> Self:
        if (
            self.mode == "build"
            and (self.manifest_sha256 is not None or self.receipt_sha256 is not None)
        ) or (
            self.mode == "load" and (self.manifest_sha256 is None or self.receipt_sha256 is None)
        ):
            raise ValueError("preparation run must independently pin both load receipts")
        return self
