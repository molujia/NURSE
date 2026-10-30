"""Recoverable hash-addressed server release deployment."""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.release import ReleaseManifest, ReleaseSafetyError, verify_release_archive
from nurse_rcl.schemas import StrictFrozenModel

DEPLOYMENT_SCHEMA_VERSION: Literal[1] = 1
PointerUpdater = Callable[[Path, Path], None]
_SHARED_ROOTS = (
    "shared/data/v1",
    "shared/caches/v1",
    "shared/history/v1",
    "shared/manifests/v1",
    "shared/runs/v1",
)


class DeploymentError(RuntimeError):
    """Raised when deployment cannot preserve release or shared-state integrity."""


class DeploymentReceipt(StrictFrozenModel):
    schema_version: Literal[1] = DEPLOYMENT_SCHEMA_VERSION
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    release_relative_path: str = Field(pattern=r"^releases/[0-9a-f]{64}$")
    shared_roots: tuple[str, ...]
    promoted: Literal[True] = True
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.release_relative_path != f"releases/{self.source_hash}":
            return False
        if self.shared_roots != _SHARED_ROOTS:
            return False
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


def deploy_release_archive(
    *,
    archive_path: Path,
    deployment_root: Path,
    pointer_updater: PointerUpdater | None = None,
    previous_source_hash: str | None = None,
) -> DeploymentReceipt:
    """Verify, install, and promote a source archive without touching shared state."""

    try:
        manifest = verify_release_archive(archive_path)
    except (OSError, ValueError, ReleaseSafetyError) as error:
        raise DeploymentError("release archive verification failed") from error
    root = deployment_root.resolve(strict=False)
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    _create_shared_roots(root)
    target = releases / manifest.source_hash
    if target.exists():
        _require_existing_release(target=target, expected=manifest)
    else:
        _install_release(
            archive_path=archive_path,
            releases_root=releases,
            target=target,
            manifest=manifest,
        )
    return promote_release(
        deployment_root=root,
        source_hash=manifest.source_hash,
        pointer_updater=pointer_updater,
        previous_source_hash=previous_source_hash,
    )


def promote_release(
    *,
    deployment_root: Path,
    source_hash: str,
    pointer_updater: PointerUpdater | None = None,
    previous_source_hash: str | None = None,
) -> DeploymentReceipt:
    """Atomically point current at one already verified release."""

    if re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise DeploymentError("source hash is invalid")
    if (
        previous_source_hash is not None
        and re.fullmatch(r"[0-9a-f]{64}", previous_source_hash) is None
    ):
        raise DeploymentError("previous source hash is invalid")
    root = deployment_root.resolve(strict=False)
    target = root / "releases" / source_hash
    try:
        manifest = _verify_release_directory(target)
    except (OSError, ValueError, ReleaseSafetyError) as error:
        raise DeploymentError("promotion target is not a verified release") from error
    if manifest.source_hash != source_hash:
        raise DeploymentError("promotion target source hash differs")
    _create_shared_roots(root)
    updater = _atomic_current_symlink if pointer_updater is None else pointer_updater
    updater(root / "current", target)
    unsigned = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "source_hash": source_hash,
        "previous_source_hash": previous_source_hash,
        "release_relative_path": f"releases/{source_hash}",
        "shared_roots": list(_SHARED_ROOTS),
        "promoted": True,
    }
    receipt = DeploymentReceipt(
        source_hash=source_hash,
        previous_source_hash=previous_source_hash,
        release_relative_path=f"releases/{source_hash}",
        shared_roots=_SHARED_ROOTS,
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("deployment receipt verification failed")
    return receipt


def _install_release(
    *,
    archive_path: Path,
    releases_root: Path,
    target: Path,
    manifest: ReleaseManifest,
) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f".{manifest.source_hash}.", dir=releases_root))
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            root_name = f"nurse-rcl-{manifest.source_hash}"
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name).relative_to(root_name)
                destination = temporary / relative
                extracted = archive.extractfile(member)
                if extracted is None:  # pragma: no cover - verified regular file
                    raise DeploymentError(f"release member is unreadable: {member.name}")
                atomic_write_bytes(destination, extracted.read())
                destination.chmod(member.mode)
        _require_existing_release(target=temporary, expected=manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _require_existing_release(*, target: Path, expected: ReleaseManifest) -> None:
    try:
        actual = _verify_release_directory(target)
    except (OSError, ValueError, ReleaseSafetyError) as error:
        raise DeploymentError("existing release is corrupt or drifted") from error
    if actual != expected:
        raise DeploymentError("existing release differs from archive")


def _verify_release_directory(root: Path) -> ReleaseManifest:
    if root.is_symlink():
        raise ReleaseSafetyError("deployed release root cannot be a symlink")
    directory = root.resolve(strict=True)
    manifest_path = directory / "release_manifest.json"
    sums_path = directory / "SHA256SUMS"
    manifest = ReleaseManifest.model_validate_json(manifest_path.read_bytes())
    if not manifest.verify():
        raise ReleaseSafetyError("deployed release manifest verification failed")
    expected_sums = {item.path: item.sha256 for item in manifest.files} | {
        "release_manifest.json": sha256_hex(manifest_path.read_bytes())
    }
    observed_sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in observed_sums:
            raise ReleaseSafetyError("deployed SHA256SUMS is invalid")
        observed_sums[relative] = digest
    if observed_sums != expected_sums:
        raise ReleaseSafetyError("deployed SHA256SUMS differs")
    expected_files = set(expected_sums) | {"SHA256SUMS"}
    observed_files: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ReleaseSafetyError("deployed release contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleaseSafetyError("deployed release contains an unsafe entry")
        observed_files.add(path.relative_to(directory).as_posix())
    if observed_files != expected_files:
        raise ReleaseSafetyError("deployed release file set differs")
    for item in manifest.files:
        path = directory / PurePosixPath(item.path)
        if not path.is_file() or path.is_symlink():
            raise ReleaseSafetyError(f"deployed release file is missing: {item.path}")
        data = path.read_bytes()
        if len(data) != item.size or sha256_hex(data) != item.sha256:
            raise ReleaseSafetyError(f"deployed release file drifted: {item.path}")
    return manifest


def _create_shared_roots(root: Path) -> None:
    for relative in _SHARED_ROOTS:
        (root / PurePosixPath(relative)).mkdir(parents=True, exist_ok=True)


def _atomic_current_symlink(current: Path, target: Path) -> None:
    if os.name != "posix":
        raise DeploymentError("current symlink promotion is supported only on Linux")
    if current.exists() and not current.is_symlink():
        raise DeploymentError("current exists but is not a symlink")
    relative_target = target.relative_to(current.parent)
    temporary = current.parent / f".current.{uuid.uuid4().hex}.tmp"
    try:
        temporary.symlink_to(relative_target, target_is_directory=True)
        os.replace(temporary, current)
    finally:
        if temporary.is_symlink():
            temporary.unlink()
