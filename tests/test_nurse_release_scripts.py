from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_text(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_basic_release_scripts_are_present():
    expected = {
        "run_nurse_gpt_smoke.ps1",
        "run_nurse_openai_full.ps1",
        "run_nurse_openai_consistency_5cases.ps1",
        "test_nurse_llm_connectivity.ps1",
        "test_nurse_all_llm_providers.ps1",
    }
    actual = {path.name for path in (REPO_ROOT / "scripts").iterdir() if path.is_file()}

    assert actual == expected


def test_openai_release_quickstart_scripts_use_basic_nurse_paths():
    full = _script_text("run_nurse_openai_full.ps1")
    consistency = _script_text("run_nurse_openai_consistency_5cases.ps1")

    assert "OPENAI_API_KEY" in full
    assert '"matrix"' in full
    assert '"openai_full_results"' in full
    assert "OPENAI_API_KEY" in consistency
    assert '"sample"' in consistency
    assert '"gpt-5-nano"' in consistency
