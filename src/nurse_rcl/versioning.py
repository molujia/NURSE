"""Version-bound snapshot verification for immutable NURSE baselines."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SNAPSHOT_SCHEMA_VERSION = "nurse.v2.frozen-snapshot/1"
SNAPSHOT_HASH_MODE = "sha256-utf8-lf-path-bound-v1"


class FrozenSnapshotMismatch(RuntimeError):
    """Raised when an immutable baseline no longer matches its manifest."""


@dataclass(frozen=True, slots=True)
class SnapshotGroup:
    """One deterministic, path-bound snapshot group."""

    file_count: int
    tree_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotVerification:
    """Verification result without mutating or repairing the frozen tree."""

    ok: bool
    verified_groups: tuple[str, ...]
    mismatched_groups: tuple[str, ...]


def _validate_pattern(pattern: str) -> None:
    path = PurePosixPath(pattern)
    if not pattern or path.is_absolute() or ".." in path.parts or "\\" in pattern:
        raise ValueError(f"snapshot pattern must be a safe POSIX relative path: {pattern!r}")


def _canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_excluded(relative_path: str, excludes: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in excludes)


def snapshot_group(
    root: Path,
    *,
    includes: Sequence[str],
    excludes: Sequence[str],
) -> SnapshotGroup:
    """Select UTF-8 text files and compute a stable path-bound tree digest."""

    resolved_root = root.resolve(strict=True)
    for pattern in (*includes, *excludes):
        _validate_pattern(pattern)

    selected: set[str] = set()
    for pattern in includes:
        for candidate in resolved_root.glob(pattern):
            if not candidate.is_file():
                continue
            resolved_candidate = candidate.resolve(strict=True)
            if not resolved_candidate.is_relative_to(resolved_root):
                raise ValueError(f"snapshot file escapes root: {candidate}")
            relative = resolved_candidate.relative_to(resolved_root).as_posix()
            if not _is_excluded(relative, excludes):
                selected.add(relative)

    paths = tuple(sorted(selected))
    tree = hashlib.sha256()
    for relative in paths:
        file_digest = _canonical_text_sha256(resolved_root / relative)
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(bytes.fromhex(file_digest))
        tree.update(b"\n")
    return SnapshotGroup(
        file_count=len(paths),
        tree_sha256=tree.hexdigest(),
        paths=paths,
    )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen snapshot manifest must be a JSON object")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported frozen snapshot schema version")
    if value.get("hash_mode") != SNAPSHOT_HASH_MODE:
        raise ValueError("unsupported frozen snapshot hash mode")
    groups = value.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("frozen snapshot manifest requires groups")
    return value


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def verify_frozen_snapshot(root: Path, manifest_path: Path) -> SnapshotVerification:
    """Verify every declared group without modifying the working tree."""

    manifest = _load_manifest(manifest_path)
    raw_groups = manifest["groups"]
    if not isinstance(raw_groups, dict):
        raise AssertionError("validated groups changed type")

    verified: list[str] = []
    mismatched: list[str] = []
    for name in sorted(raw_groups):
        raw_group = raw_groups[name]
        if not isinstance(name, str) or not isinstance(raw_group, dict):
            raise ValueError("snapshot group names and values must be objects")
        includes = _string_sequence(raw_group.get("includes"), field=f"{name}.includes")
        excludes = _string_sequence(raw_group.get("excludes"), field=f"{name}.excludes")
        expected_count = raw_group.get("file_count")
        expected_hash = raw_group.get("tree_sha256")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 0
        ):
            raise ValueError(f"{name}.file_count must be a non-negative integer")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"{name}.tree_sha256 must be lowercase SHA-256")

        actual = snapshot_group(root, includes=includes, excludes=excludes)
        if actual.file_count == expected_count and actual.tree_sha256 == expected_hash:
            verified.append(name)
        else:
            mismatched.append(name)

    return SnapshotVerification(
        ok=not mismatched,
        verified_groups=tuple(verified),
        mismatched_groups=tuple(mismatched),
    )


def assert_frozen_snapshot(root: Path, manifest_path: Path) -> SnapshotVerification:
    """Return a successful report or raise with the mismatched group names."""

    report = verify_frozen_snapshot(root, manifest_path)
    if not report.ok:
        groups = ", ".join(report.mismatched_groups)
        raise FrozenSnapshotMismatch(f"frozen snapshot mismatch: {groups}")
    return report
