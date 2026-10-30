"""The final 0ARK harness admits only its fresh, fixed KEY1 rejection fixture."""

from __future__ import annotations

import inspect
import stat
import sys
from pathlib import Path

import pytest

from nurse_rcl.agents import ark
from scripts.dev import run_final_explainability_0ark as boundary

KEY1_BYTES = b"ARK_API_KEY=synthetic-key-one\n"


def _key(path: Path, content: bytes = KEY1_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def _run_guard(tmp_path, monkeypatch, exercise, *, original=None, arguments=()):
    calls = []
    hooks = []

    def parser(path):
        calls.append(path)
        if original is not None:
            return original(path)
        raise ValueError("original synthetic parser rejection")

    def pytest_main(arguments):
        # The fallback makes the original all-denying harness fail on the
        # missing parser behavior, rather than on a test setup error.
        basetemps = [
            Path(arguments[index + 1])
            for index, arg in enumerate(arguments[:-1])
            if arg == "--basetemp"
        ]
        fixture_root = basetemps[-1] if basetemps else tmp_path / "baseline-pytest"
        exercise(fixture_root, ark.load_ark_settings_from_key_file, calls, arguments, hooks)
        return 0

    original_key_loader = ark.load_ark_settings_from_key_file
    original_server_loader = ark.load_server_ark_settings
    with monkeypatch.context() as guarded:
        guarded.setattr(ark, "load_ark_settings_from_key_file", parser)
        guarded.setattr(
            ark,
            "load_server_ark_settings",
            lambda **_: pytest.fail("server credential loader must never run"),
        )
        guarded.setattr(boundary.resource, "setrlimit", lambda *_: None)
        guarded.setattr(sys, "addaudithook", hooks.append)
        guarded.setattr(sys, "argv", [str(boundary.__file__), *arguments])
        guarded.setattr(sys, "path", list(sys.path))
        guarded.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
        guarded.setattr(pytest, "main", pytest_main)
        assert boundary.main() == 0
    assert ark.load_ark_settings_from_key_file is original_key_loader
    assert ark.load_server_ark_settings is original_server_loader


def test_guard_helper_restores_server_loader_before_return(tmp_path, monkeypatch):
    original_server_loader = ark.load_server_ark_settings
    _run_guard(tmp_path, monkeypatch, lambda *_: None)
    assert ark.load_server_ark_settings is original_server_loader


def test_exact_private_key1_fixture_reaches_real_parser(tmp_path, monkeypatch):
    original = inspect.unwrap(ark.load_ark_settings_from_key_file)

    def exercise(root, loader, calls, *_):
        key = _key(root / "test-key1" / "ark.env")
        with pytest.raises(ark.ArkKeyFileFormatError, match="only ARK_API_KEY2 is allowed"):
            loader(key)
        assert calls == [key]

    _run_guard(tmp_path, monkeypatch, exercise, original=original)


def test_each_run_overrides_basetemp_with_a_fresh_private_pytest_root(tmp_path, monkeypatch):
    supplied = tmp_path / "existing-user-directory"
    marker = _key(supplied / "keep.txt", b"keep\n")
    observed = []

    def exercise(root, _loader, _calls, arguments, _hooks):
        assert arguments[-4:-2] == ["--basetemp", str(root)]
        assert root.name == "pytest"
        assert root.parent.name.startswith("nurse-final-explainability-0ark.")
        assert root.parent != supplied
        assert not root.exists()
        assert root.parent.is_dir() and not root.parent.is_symlink()
        assert stat.S_IMODE(root.parent.stat().st_mode) == 0o700
        observed.append(root)

    for _ in range(2):
        _run_guard(
            tmp_path,
            monkeypatch,
            exercise,
            arguments=("--basetemp", str(supplied)),
        )
    assert observed[0] != observed[1]
    assert marker.read_bytes() == b"keep\n"


@pytest.mark.parametrize("location", ["outside", "sibling", "traversal", "wrong-name"])
def test_key1_at_an_unapproved_path_never_reaches_parser(tmp_path, monkeypatch, location):
    def exercise(root, loader, calls, *_):
        if location == "outside":
            path = _key(tmp_path / "outside" / "ark.env")
        elif location == "sibling":
            path = _key(root.with_name(root.name + "-other") / "ark.env")
        elif location == "traversal":
            root.mkdir(parents=True, exist_ok=True)
            _key(root.parent / "ark.env")
            path = root / ".." / "ark.env"
        else:
            path = _key(root / "key.env")
        with pytest.raises(RuntimeError, match="credential loading forbidden"):
            loader(path)
        assert calls == []

    _run_guard(tmp_path, monkeypatch, exercise)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlinks_never_reach_parser(tmp_path, monkeypatch, kind):
    def exercise(root, loader, calls, *_):
        target = _key(root / "target" / "ark.env")
        if kind == "file":
            path = root / "ark.env"
            path.symlink_to(target)
        else:
            link = root / "linked-directory"
            link.symlink_to(target.parent, target_is_directory=True)
            path = link / "ark.env"
        with pytest.raises(RuntimeError, match="credential loading forbidden"):
            loader(path)
        assert calls == []

    _run_guard(tmp_path, monkeypatch, exercise)


@pytest.mark.parametrize(
    "content",
    [
        b"ARK_API_KEY2=synthetic-key-two\n",
        b"ARK_API_KEY=synthetic-key-two\n",
        KEY1_BYTES + b"# extra\n",
        KEY1_BYTES.rstrip(b"\n"),
        b"",
    ],
)
def test_other_content_never_reaches_parser(tmp_path, monkeypatch, content):
    def exercise(root, loader, calls, *_):
        path = _key(root / "ark.env", content)
        with pytest.raises(RuntimeError, match="credential loading forbidden"):
            loader(path)
        assert calls == []

    _run_guard(tmp_path, monkeypatch, exercise)


def test_nonregular_key_path_never_reaches_parser(tmp_path, monkeypatch):
    def exercise(root, loader, calls, *_):
        path = root / "ark.env"
        path.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="credential loading forbidden"):
            loader(path)
        assert calls == []

    _run_guard(tmp_path, monkeypatch, exercise)


def test_server_loader_remains_unconditionally_forbidden(tmp_path, monkeypatch):
    def exercise(root, _loader, calls, *_):
        _key(root / ".config" / "nurse" / "ark.env")
        with pytest.raises(RuntimeError, match="credential loading forbidden"):
            ark.load_server_ark_settings(home=root)
        assert calls == []

    _run_guard(tmp_path, monkeypatch, exercise)


@pytest.mark.parametrize(
    "event",
    ["socket.connect", "socket.getaddrinfo", "socket.bind", "os.system", "os.posix_spawn"],
)
def test_network_and_external_execution_remain_forbidden(tmp_path, monkeypatch, event):
    def exercise(_root, _loader, _calls, _arguments, hooks):
        assert len(hooks) == 1
        with pytest.raises(RuntimeError, match="external execution forbidden"):
            hooks[0](event, ())

    _run_guard(tmp_path, monkeypatch, exercise)


def test_arbitrary_subprocess_remains_forbidden(tmp_path, monkeypatch):
    def exercise(_root, _loader, _calls, _arguments, hooks):
        with pytest.raises(RuntimeError, match="external execution forbidden"):
            hooks[0]("subprocess.Popen", (sys.executable, [sys.executable, "-c", "pass"], None, {}))

    _run_guard(tmp_path, monkeypatch, exercise)
