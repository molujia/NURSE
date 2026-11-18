"""Exact observation-band boundaries, provenance and legacy replay."""

import pytest

from nurse_rcl.v2.live import verified_history as h


def core(terms):
    return {
        "identity": {"opaque_case_id": "opq_" + "0" * 32},
        "core_result_digest": "a" * 64,
        "evidence": {
            "graph": {"graph_pair": {"fault_snapshot": {"decision_time_unix_ns": 7}}},
            "ranking": {
                "propagation": {
                    "local_scoring": {
                        "strengths": {"terms": terms, "mechanisms": {"mechanisms": []}}
                    }
                }
            },
        },
    }


def term(z, q, priority=1):
    return {
        "kind": "window_summary",
        "metric_name": "cpu.util",
        "candidate_service_key": "c",
        "local_strength": priority,
        "quality_factor": q,
        "calculation": {"signed_z": z},
    }


@pytest.mark.parametrize(
    "z,q,amplitude,quality",
    [
        (0.99, 0.49, "slight", "low"),
        (-1, 0.5, "moderate", "moderate"),
        (3, 0.8, "large", "high"),
        (-6, 1, "very large", "high"),
    ],
)
def test_exact_bands_and_common_text(z, q, amplitude, quality):
    data = core([term(z, q)])
    doc = h.project_core(data)
    (row,) = doc["evidence"]
    assert (row["strength_band"], row["quality_band"]) == (amplitude, quality)
    old = h.project_core(data, profile=h.LEGACY_PROFILE)
    base = h.document_text(old)
    assert h.document_text(doc, profile=h.LEGACY_PROFILE) == base
    state = "up" if z > 0 else "down"
    assert h.document_text(doc) == (
        base
        + f"\nDetail | window_summary | cpu.util | {state} | strength_band | {amplitude}"
        + f"\nDetail | window_summary | cpu.util | {state} | quality_band | {quality}"
    )
    (natural,) = h.rendered_evidence(doc)
    assert natural["source_ref"] == row["source_ref"]
    assert f"observation quality was {quality}" in natural["text"]
    assert "Detail |" not in natural["text"]


def test_representative_owns_both_bands_and_source():
    doc = h.project_core(core([term(8, 0.1, 0.2), term(1, 0.9, 0.8), term(9, 0.3, 0.8)]))
    (row,) = doc["evidence"]
    assert row["source_ref"].endswith("/terms/1")
    assert (row["strength_band"], row["quality_band"]) == ("moderate", "high")
    del row["quality_band"]
    with pytest.raises(ValueError):
        h.validate_document(doc)


@pytest.mark.parametrize("z,q", [(None, 1), (float("inf"), 1), (1, float("nan")), (1, None)])
def test_missing_or_nonfinite_is_not_binned(z, q):
    assert not h.project_core(core([term(z, q)]))["evidence"]
