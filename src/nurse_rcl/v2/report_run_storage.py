"""Private canonical Report receipts, pinned loading and atomic no-clobber publish.

Parents must be trusted and protected. Leaf checks do not resist malicious parent
renames. File fsync is used, without claiming cross-platform directory power-loss
durability. A process crash can leave a hidden temp; there is no resume service.
Loading verifies integrity only: replay with independent sources/policies is still
required before treating retained effects as meaningful.
"""

from __future__ import annotations

import hmac
import json
import os
import stat
import tempfile
from pathlib import Path

from nurse_rcl.artifacts import canonical_json_bytes
from nurse_rcl.v2 import report_audit_json as audit_json
from nurse_rcl.v2.report_run_schemas import ReportRunAudit, _preflight_audit

_NATIVE_PATH = type(Path())


def _check_roots(path: Path, pin: str, audit: ReportRunAudit | None = None) -> None:
    if type(path) is not _NATIVE_PATH:
        raise ValueError("report audit storage requires an exact native Path")
    if type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("report audit storage requires an exact lowercase SHA-256 pin")
    if audit is not None and type(audit) is not ReportRunAudit:
        raise ValueError("report audit storage requires exact ReportRunAudit type")


def _snapshot(audit: ReportRunAudit) -> bytes:
    measured = _preflight_audit(audit)
    data = canonical_json_bytes(audit.model_dump(mode="json", warnings="error"))
    if len(data) != measured:
        raise ValueError("report audit canonical measurement mismatch")
    return data


def _require_integrity(audit: ReportRunAudit, pin: str) -> None:
    if not audit.verify() or not hmac.compare_digest(audit.audit_sha256, pin):
        raise ValueError("report audit integrity or independent pin mismatch")


def _regular(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and not (
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
    )


def _same_leaf(left: os.stat_result, right: os.stat_result) -> bool:
    return _regular(right) and (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_bounded(path: Path) -> bytes:
    before = os.lstat(path)
    if not _regular(before):
        raise ValueError("report audit leaf must be a regular nonlink file")
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not _same_leaf(before, opened):
            raise ValueError("report audit leaf changed while opening")
        maximum = audit_json.MAX_REPORT_AUDIT_BYTES
        if opened.st_size > maximum:
            raise ValueError("report audit exceeds byte limit")
        chunks: list[bytes] = []
        count = 0
        while count <= maximum:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
        if count > maximum:
            raise ValueError("report audit exceeds byte limit")
        after = os.fstat(fd)
        if (
            not _same_leaf(opened, after)
            or not _same_leaf(opened, os.lstat(path))
            or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("report audit leaf changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def persist_report_run_audit(
    path: Path, audit: ReportRunAudit, *, expected_audit_sha256: str
) -> None:
    """Publish one detached canonical receipt, preserving any conflicting file."""
    _check_roots(path, expected_audit_sha256, audit)
    # None must not use the read-only optional-audit branch of root validation.
    if type(audit) is not ReportRunAudit:
        raise ValueError("report audit storage requires exact ReportRunAudit type")
    try:
        data = _snapshot(audit)
        detached = ReportRunAudit.model_validate_json(data)
        if _snapshot(detached) != data:
            raise ValueError("report audit normalization changes canonical bytes")
        _require_integrity(detached, expected_audit_sha256)
        if not hmac.compare_digest(_snapshot(audit), data):
            raise ValueError("report audit original bytes changed")
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise ValueError("invalid report audit or independent pin") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        # POSIX creates a private 0600 file; Windows inherits the parent ACL.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".report-audit-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if stream.write(data) != len(data):
                raise OSError("incomplete report audit write")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                identical = hmac.compare_digest(_read_bounded(path), data)
            except (OSError, ValueError):
                identical = False
            if not identical:
                raise FileExistsError(
                    "report audit destination contains different content"
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate report audit JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite report audit JSON scalar")


def load_report_run_audit(path: Path, *, expected_audit_sha256: str) -> ReportRunAudit:
    """Load exact pinned bytes, without establishing source or semantic authority."""
    _check_roots(path, expected_audit_sha256)
    data = _read_bounded(path)
    audit_json.scan_audit_json(data)
    try:
        document = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(document) is not dict:
            raise ValueError("report audit requires a JSON object")
        measured = audit_json.measure_canonical_json(document, root_model_types=(ReportRunAudit,))
        canonical = canonical_json_bytes(document)
        if len(canonical) != measured or not hmac.compare_digest(canonical, data):
            raise ValueError("noncanonical report audit JSON")
        audit = ReportRunAudit.model_validate_json(data)
        if not hmac.compare_digest(_snapshot(audit), data):
            raise ValueError("report audit normalization changes canonical bytes")
        _require_integrity(audit, expected_audit_sha256)
        return audit
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise ValueError("invalid report audit JSON or independent pin") from None
