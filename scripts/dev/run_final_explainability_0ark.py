"""Linux pytest boundary for a fresh isolated final-delivery candidate only."""

from __future__ import annotations

import os
import re
import resource
import stat
import sys
import tempfile
from functools import wraps
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "third_party/hermes_runtime"), str(root)]
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for kind, limit in (
        (resource.RLIMIT_AS, 8 * 1024**3),
        (resource.RLIMIT_STACK, 64 * 1024**2),
        (resource.RLIMIT_NOFILE, 4096),
    ):
        resource.setrlimit(kind, (limit, limit))

    def audit(event, arguments):
        if event == "subprocess.Popen":
            executable, args, cwd, _ = arguments
            # Existing independent saved-reader regression has its own strict
            # deny_effects/forbid_dispatch guards. No arbitrary child is allowed.
            if (
                executable == sys.executable
                and type(args) is list
                and len(args) == 5
                and args[:2] == [sys.executable, "tests/sre_atomic_report_reader.py"]
                and cwd is None
                and Path(args[2]).is_dir()
                and Path(args[2]).resolve().is_relative_to(Path("/tmp"))
                and all(re.fullmatch(r"[0-9a-f]{64}", arg) for arg in args[3:])
            ):
                return
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.bind",
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
        }:
            raise RuntimeError("final 0ARK external execution forbidden")

    sys.addaudithook(audit)
    import nurse_rcl.agents.ark as ark

    def forbidden_credentials(*_, **__):
        raise RuntimeError("final 0ARK real credential loading forbidden")

    # Never let caller-supplied --basetemp point the credential exception at
    # existing files. A fixed Linux temp parent also avoids environment lookup.
    fixture_root = Path(
        tempfile.mkdtemp(prefix="nurse-final-explainability-0ark.", dir="/tmp")
    ) / "pytest"
    original_key_loader = ark.load_ark_settings_from_key_file
    key1_rejection_fixture = b"ARK_API_KEY=synthetic-key-one\n"

    @wraps(original_key_loader)
    def synthetic_key1_only(path):
        try:
            key_path = Path(path)
            if (
                not key_path.is_absolute()
                or ".." in key_path.parts
                or key_path.name != "ark.env"
            ):
                forbidden_credentials()
            relative = key_path.relative_to(fixture_root)
            current = fixture_root
            # lstat every directory so even an in-root symlink is forbidden.
            for directory in (fixture_root.parent, fixture_root):
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    forbidden_credentials()
            for part in relative.parts[:-1]:
                current /= part
                if not stat.S_ISDIR(current.lstat().st_mode):
                    forbidden_credentials()
            observed = key_path.lstat()
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_size != len(key1_rejection_fixture)
            ):
                forbidden_credentials()
            with key_path.open("rb") as handle:
                if handle.read(len(key1_rejection_fixture) + 1) != key1_rejection_fixture:
                    forbidden_credentials()
        except (OSError, TypeError, ValueError):
            forbidden_credentials()
        # The real parser must reject KEY1; do not manufacture its ValueError.
        return original_key_loader(key_path)

    ark.load_ark_settings_from_key_file = synthetic_key1_only
    ark.load_server_ark_settings = forbidden_credentials
    import pytest

    return pytest.main(
        [*sys.argv[1:], "--basetemp", str(fixture_root), "-p", "no:cacheprovider"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
