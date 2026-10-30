"""Closed, read-only dispatch over the two code-owned case audit contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2 import case_audit as legacy
from nurse_rcl.v2.case_audit_lineage_v2 import CaseAuditLineageIndexV2
from nurse_rcl.v2.case_audit_v2 import (
    CaseAuditReplayReceiptV2,
    CaseRankingReplayBundleV2,
    ImmutableCaseAuditPackageV2,
    _audit_unchanged,
    _preflight_audit_package,
    replay_case_audit_v2,
)

SupportedCaseAudit: TypeAlias = legacy.ImmutableCaseAuditPackage | ImmutableCaseAuditPackageV2


@dataclass(frozen=True, slots=True, init=False)
class AuditReadView:
    """Private fresh value view; no persistence, transport or scoring authority."""

    canonical_audit_bytes: bytes
    schema_version: Literal[
        "nurse.v2.immutable-case-audit-package/1", "nurse.v2.immutable-case-audit-package/2"
    ]
    opaque_case_id: str
    terminal_state: Literal["SEALED", "FAILURE_SEALED"]
    audit_sha256: str
    replay_bundle: legacy.CaseRankingReplayBundle | CaseRankingReplayBundleV2 | None
    lineage_index: legacy.CaseAuditLineageIndex | CaseAuditLineageIndexV2 | None
    receipt: legacy.CaseAuditReplayReceipt | CaseAuditReplayReceiptV2

    def __init__(self) -> None:
        raise TypeError("audit views can only be created by complete pinned audit reading")


def read_case_audit(
    *, audit_package: SupportedCaseAudit, expected_audit_sha256: str
) -> AuditReadView:
    """Require the caller's independent source pin, then replay the exact version.

    Identity checks use ``is`` exclusively. Neither subclasses, runtime protocols,
    caller-made views nor a plugin's metaclass can supply authority to this reader.
    """
    if type(expected_audit_sha256) is not str:
        raise ValueError("audit reading requires an independent exact external digest")
    original = _preflight_audit_package(audit_package, expected_audit_sha256)
    try:
        state = cast(dict[str, Any], original[0])
        raw = canonical_json_bytes(state)
        receipt: legacy.CaseAuditReplayReceipt | CaseAuditReplayReceiptV2
        bundle: legacy.CaseRankingReplayBundle | CaseRankingReplayBundleV2 | None
        lineage: legacy.CaseAuditLineageIndex | CaseAuditLineageIndexV2 | None
        if (
            type(audit_package) is legacy.ImmutableCaseAuditPackage
            and state["schema_version"] == legacy.IMMUTABLE_CASE_AUDIT_SCHEMA_VERSION
        ):
            old_package = legacy.ImmutableCaseAuditPackage.model_validate_json(raw)
            receipt = legacy.replay_case_audit(
                old_package, expected_audit_sha256=expected_audit_sha256
            )
            bundle = old_package.replay_bundle_or_none
            lineage = old_package.lineage_index_or_none
        elif (
            type(audit_package) is ImmutableCaseAuditPackageV2
            and state["schema_version"] == "nurse.v2.immutable-case-audit-package/2"
        ):
            new_package = ImmutableCaseAuditPackageV2.model_validate_json(raw)
            receipt = replay_case_audit_v2(new_package, expected_audit_sha256=expected_audit_sha256)
            bundle = new_package.replay_bundle
            lineage = new_package.lineage_index
        else:
            raise ValueError("audit type identity and schema literal do not match")
        view = object.__new__(AuditReadView)
        values = {
            "canonical_audit_bytes": raw,
            "schema_version": state["schema_version"],
            "opaque_case_id": state["opaque_case_id"],
            "terminal_state": state["terminal_state"],
            "audit_sha256": expected_audit_sha256,
            "replay_bundle": bundle,
            "lineage_index": lineage,
            "receipt": receipt,
        }
        for name, value in values.items():
            object.__setattr__(view, name, value)
        return view
    finally:
        # No encoding happens after this final storage/representation snapshot.
        _audit_unchanged(audit_package, original)


__all__ = ["AuditReadView", "SupportedCaseAudit", "read_case_audit"]
