"""Read-only core provenance accepts known SRE versions, never executable old agents."""

# ruff: noqa: F811 -- imported pytest fixtures.

import pytest

from nurse_rcl.v2.live.ops_sre_core_import import _source_metadata
from tests.test_v2_live_ops_execution import ops_plan, prepared_case, save  # noqa: F401


@pytest.mark.parametrize("version", range(1, 10))
def test_completed_core_metadata_accepts_known_same_core_sre_versions(
    prepared_case,
    tmp_path,
    version,
):
    path, _, value = ops_plan(tmp_path, prepared_case)
    # Agent protocol fields deliberately remain opaque/retired. This is only
    # provenance and must not make those historical protocols executable.
    value["profile"].update(
        profile_id=f"v2-ops-sre/{version}", decision_protocol="retired-not-runnable"
    )
    metadata = _source_metadata(path, save(path, value))
    assert metadata.profile_id == f"v2-ops-sre/{version}"
    assert metadata.core_method_id == "normal-b0-ops-core/1"


@pytest.mark.parametrize(
    "profile,method",
    [
        ("v2-ops-sre/999", "normal-b0-ops-core/1"),
        ("arbitrary-agent", "normal-b0-ops-core/1"),
        ("v2-ops-sre/7", "different-core/1"),
    ],
)
def test_core_provenance_still_rejects_unknown_profile_or_other_core(
    prepared_case,
    tmp_path,
    profile,
    method,
):
    path, _, value = ops_plan(tmp_path, prepared_case)
    value["profile"].update(profile_id=profile, core_method_id=method)
    with pytest.raises(ValueError):
        _source_metadata(path, save(path, value))


def test_core_provenance_keeps_exact_source_plan_pin(prepared_case, tmp_path):
    path, pin, value = ops_plan(tmp_path, prepared_case)
    value["profile"]["profile_id"] = "v2-ops-sre/7"
    save(path, value)
    with pytest.raises(ValueError):
        _source_metadata(path, pin)
