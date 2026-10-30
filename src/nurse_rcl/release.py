"""Deterministic, content-addressed source release packaging and verification."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import StrictFrozenModel

RELEASE_SCHEMA_VERSION: Literal[1] = 1
_ARCHIVE_PREFIX = "nurse-rcl-"
_FORBIDDEN_SEGMENTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".tox",
        ".venv",
        "__pycache__",
        "cache",
        "caches",
        "data",
        "datasets",
        "model_cache",
        "outputs",
        "runs",
        "tmp",
        "venv",
    }
)
_FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "ark.env",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "known_hosts",
        "ssh_config",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".key",
        ".npy",
        ".npz",
        ".onnx",
        ".p12",
        ".parquet",
        ".pem",
        ".pt",
        ".pth",
        ".safetensors",
    }
)
_PRIVATE_KEY_MARKERS = tuple(
    f"-----BEGIN {key_type}PRIVATE KEY-----".encode("ascii")
    for key_type in ("", "OPENSSH ", "RSA ", "DSA ", "EC ")
)


class ReleaseSafetyError(ValueError):
    """Raised before packaging or extraction can include an unsafe payload."""


class ReleaseFile(StrictFrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["0644", "0755"]


class ReleaseManifest(StrictFrozenModel):
    schema_version: Literal[1] = RELEASE_SCHEMA_VERSION
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[ReleaseFile, ...] = Field(min_length=1)

    def verify(self) -> bool:
        if self.files != tuple(sorted(self.files, key=lambda item: item.path)):
            return False
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            return False
        return self.source_hash == _source_hash(self.files)


class ReleasePackageReceipt(StrictFrozenModel):
    schema_version: Literal[1] = RELEASE_SCHEMA_VERSION
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_name: str = Field(
        pattern=r"^nurse-rcl-[0-9a-f]{64}\.tar\.gz$",
    )
    archive_size: int = Field(gt=0)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(gt=0)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bool:
        if self.archive_name != f"{_ARCHIVE_PREFIX}{self.source_hash}.tar.gz":
            return False
        unsigned = self.model_dump(mode="json", exclude={"receipt_hash"})
        return self.receipt_hash == sha256_hex(canonical_json_bytes(unsigned))


def build_release_package(
    *,
    repository_root: Path,
    output_root: Path,
    tracked_paths: Sequence[Path] | None = None,
) -> ReleasePackageReceipt:
    """Build one reproducible archive from safe Git-tracked source files."""

    repository = repository_root.resolve(strict=True)
    if not repository.is_dir():
        raise NotADirectoryError(repository)
    sources = (
        _git_tracked_paths(repository)
        if tracked_paths is None
        else tuple(Path(path) for path in tracked_paths)
    )
    records_and_sources = tuple(
        sorted(
            (_release_file(repository, source) for source in sources),
            key=lambda item: item[0].path,
        )
    )
    if not records_and_sources:
        raise ReleaseSafetyError("release cannot be empty")
    records = tuple(item[0] for item in records_and_sources)
    manifest = ReleaseManifest(source_hash=_source_hash(records), files=records)
    if not manifest.verify():  # pragma: no cover - construction invariant
        raise AssertionError("release manifest verification failed")

    destination = output_root.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = f"{_ARCHIVE_PREFIX}{manifest.source_hash}.tar.gz"
    archive_path = destination / archive_name
    temporary_root = Path(tempfile.mkdtemp(prefix=".nurse-release-", dir=destination))
    try:
        bundle_root = temporary_root / f"{_ARCHIVE_PREFIX}{manifest.source_hash}"
        _stage_bundle(
            bundle_root=bundle_root,
            records_and_sources=records_and_sources,
            manifest=manifest,
        )
        temporary_archive = temporary_root / archive_name
        file_modes = {item.path: int(item.mode, 8) for item in manifest.files}
        file_modes.update({"release_manifest.json": 0o644, "SHA256SUMS": 0o644})
        _write_deterministic_archive(
            bundle_root=bundle_root,
            output=temporary_archive,
            file_modes=file_modes,
        )
        archive_sha256 = _file_sha256(temporary_archive)
        archive_size = temporary_archive.stat().st_size
        if archive_path.exists():
            if (
                not archive_path.is_file()
                or archive_path.stat().st_size != archive_size
                or _file_sha256(archive_path) != archive_sha256
            ):
                raise ReleaseSafetyError("existing release archive differs")
        else:
            os.replace(temporary_archive, archive_path)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    verified = verify_release_archive(archive_path)
    manifest_bytes = canonical_json_bytes(verified.model_dump(mode="json"))
    unsigned = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "source_hash": verified.source_hash,
        "archive_name": archive_name,
        "archive_size": archive_size,
        "archive_sha256": archive_sha256,
        "manifest_sha256": sha256_hex(manifest_bytes),
        "file_count": len(verified.files),
    }
    receipt = ReleasePackageReceipt(
        source_hash=verified.source_hash,
        archive_name=archive_name,
        archive_size=archive_size,
        archive_sha256=archive_sha256,
        manifest_sha256=sha256_hex(manifest_bytes),
        file_count=len(verified.files),
        receipt_hash=sha256_hex(canonical_json_bytes(unsigned)),
    )
    if not receipt.verify():  # pragma: no cover - construction invariant
        raise AssertionError("release receipt verification failed")
    _write_immutable(
        destination / f"{archive_name}.receipt.json",
        canonical_json_bytes(receipt.model_dump(mode="json")),
    )
    return receipt


def verify_release_archive(archive_path: Path) -> ReleaseManifest:
    """Verify safe members, the content manifest, and every payload digest."""

    archive = archive_path.resolve(strict=True)
    with tarfile.open(archive, mode="r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            _validate_archive_member(member)
        files = {member.name: member for member in members if member.isfile()}
        roots = {PurePosixPath(name).parts[0] for name in files}
        if len(roots) != 1:
            raise ReleaseSafetyError("release archive must have one content root")
        root = next(iter(roots))
        match = re.fullmatch(r"nurse-rcl-([0-9a-f]{64})", root)
        if match is None:
            raise ReleaseSafetyError("release archive root is invalid")
        manifest_name = f"{root}/release_manifest.json"
        sums_name = f"{root}/SHA256SUMS"
        if manifest_name not in files or sums_name not in files:
            raise ReleaseSafetyError("release archive omits its integrity manifests")
        manifest_bytes = _read_member(stream, files[manifest_name])
        manifest = ReleaseManifest.model_validate_json(manifest_bytes)
        if not manifest.verify() or manifest.source_hash != match.group(1):
            raise ReleaseSafetyError("release source manifest verification failed")
        if manifest_bytes != canonical_json_bytes(manifest.model_dump(mode="json")):
            raise ReleaseSafetyError("release source manifest is not canonical")

        expected_names = {
            manifest_name,
            sums_name,
            *(f"{root}/{item.path}" for item in manifest.files),
        }
        if set(files) != expected_names:
            raise ReleaseSafetyError("release archive file set differs from its manifest")
        sums: dict[str, str] = {}
        for item in manifest.files:
            member = files[f"{root}/{item.path}"]
            data = _read_member(stream, member)
            if len(data) != item.size or sha256_hex(data) != item.sha256:
                raise ReleaseSafetyError(f"release file hash mismatch: {item.path}")
            expected_mode = int(item.mode, 8)
            if member.mode != expected_mode:
                raise ReleaseSafetyError(f"release file mode mismatch: {item.path}")
            sums[item.path] = item.sha256
        sums["release_manifest.json"] = sha256_hex(manifest_bytes)
        expected_sums = _sha256sums_bytes(sums)
        if _read_member(stream, files[sums_name]) != expected_sums:
            raise ReleaseSafetyError("SHA256SUMS verification failed")
    return manifest


def _release_file(repository: Path, source: Path) -> tuple[ReleaseFile, Path]:
    if source.is_symlink():
        raise ReleaseSafetyError(f"release source cannot be a symlink: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(repository):
        raise ReleaseSafetyError(f"release source is outside repository: {source}")
    relative = resolved.relative_to(repository).as_posix()
    _validate_release_path(relative)
    data = resolved.read_bytes()
    if any(marker in data for marker in _PRIVATE_KEY_MARKERS):
        raise ReleaseSafetyError(f"private key material found in release source: {relative}")
    mode: Literal["0644", "0755"] = "0755" if _is_executable(resolved, relative) else "0644"
    return (
        ReleaseFile(
            path=relative,
            size=len(data),
            sha256=sha256_hex(data),
            mode=mode,
        ),
        resolved,
    )


def _validate_release_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReleaseSafetyError(f"unsafe release path: {relative}")
    parts = tuple(part.casefold() for part in path.parts)
    name = parts[-1]
    suffix = PurePosixPath(name).suffix.casefold()
    forbidden_segments = tuple(
        part
        for part in parts
        if part in _FORBIDDEN_SEGMENTS or part.startswith(".venv")
    )
    approved_dataset_adapter = (
        parts[:3] == ("src", "nurse_rcl", "datasets")
        and suffix in {".py", ".pyi"}
        and forbidden_segments == ("datasets",)
    )
    if (
        (forbidden_segments and not approved_dataset_adapter)
        or name in _FORBIDDEN_NAMES
        or suffix in _FORBIDDEN_SUFFIXES
    ):
        raise ReleaseSafetyError(f"forbidden release path: {relative}")


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReleaseSafetyError(f"unsafe archive member: {member.name}")
    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
        raise ReleaseSafetyError(f"unsafe archive member type: {member.name}")


def _git_tracked_paths(repository: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return tuple(
        repository / relative
        for relative in completed.stdout.decode("utf-8").split("\0")
        if relative
    )


def _stage_bundle(
    *,
    bundle_root: Path,
    records_and_sources: tuple[tuple[ReleaseFile, Path], ...],
    manifest: ReleaseManifest,
) -> None:
    for record, source in records_and_sources:
        destination = bundle_root / PurePosixPath(record.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(int(record.mode, 8))
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    _write_immutable(bundle_root / "release_manifest.json", manifest_bytes)
    sums = {item.path: item.sha256 for item in manifest.files}
    sums["release_manifest.json"] = sha256_hex(manifest_bytes)
    _write_immutable(bundle_root / "SHA256SUMS", _sha256sums_bytes(sums))


def _write_deterministic_archive(
    *,
    bundle_root: Path,
    output: Path,
    file_modes: dict[str, int],
) -> None:
    root_name = bundle_root.name
    files = tuple(sorted(path for path in bundle_root.rglob("*") if path.is_file()))
    directories = {PurePosixPath(root_name)}
    for path in files:
        relative = PurePosixPath(path.relative_to(bundle_root).as_posix())
        parent = relative.parent
        while parent != PurePosixPath("."):
            directories.add(PurePosixPath(root_name) / parent)
            parent = parent.parent
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for directory in sorted(directories, key=lambda item: item.as_posix()):
            info = tarfile.TarInfo(f"{directory.as_posix()}/")
            _normalize_tar_info(info, mode=0o755)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for path in files:
            relative_name = path.relative_to(bundle_root).as_posix()
            info = tarfile.TarInfo(f"{root_name}/{relative_name}")
            info.size = path.stat().st_size
            _normalize_tar_info(info, mode=file_modes[relative_name])
            with path.open("rb") as source:
                tar.addfile(info, source)


def _normalize_tar_info(info: tarfile.TarInfo, *, mode: int) -> None:
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""


def _read_member(stream: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = stream.extractfile(member)
    if extracted is None:  # pragma: no cover - guarded by member.isfile
        raise ReleaseSafetyError(f"release member is unreadable: {member.name}")
    return extracted.read()


def _source_hash(files: tuple[ReleaseFile, ...]) -> str:
    return sha256_hex(canonical_json_bytes([item.model_dump(mode="json") for item in files]))


def _sha256sums_bytes(sums: dict[str, str]) -> bytes:
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(sums.items())).encode("utf-8")


def _is_executable(path: Path, relative: str) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR) or relative.endswith(".sh")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ReleaseSafetyError(f"immutable release output differs: {path.name}")
        return
    atomic_write_bytes(path, data)
