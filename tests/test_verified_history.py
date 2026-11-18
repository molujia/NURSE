"""Regression checks at the frozen evidence, review, and temporal boundaries."""

import copy
import json

import pytest

from nurse_rcl.evaluation import history_library as admission
from nurse_rcl.v2.live import verified_history as history


def case_id(number):
    return f"opq_{number:032x}"


def projection(number, time):
    return {
        "schema_version": history.PROJECTION,
        "case_id": case_id(number),
        "decision_time_ns": time,
        "evidence": [
            {
                "kind": "window_summary",
                "signal": "pod.cpu.usage",
                "state": "up",
                "source_ref": f"src_{number:032x}",
            }
        ],
    }


def emit(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_core_projection_keeps_distinct_meanings_and_replayable_fields():
    def term(kind, calc, strength=0.5):
        return {
            "candidate_service_key": "internal-service",
            "quality_factor": 0.8,
            "kind": kind,
            "calculation": calc,
            "metric_name": "pod.cpu.usage",
            "local_strength": strength,
        }

    terms = [
        term("window_summary", {"signed_z": 2.0}),
        term("window_summary", {"signed_z": 4.0}, 0.7),
        term("paid_episode", {"direction": "up"}),
        term("window_summary", {"signed_z": -6.0}, 0),
    ]
    strengths = {
        "terms": terms,
        "mechanisms": {
            "mechanisms": [
                {
                    "candidate_service_key": "internal-service",
                    "mechanism_key": "local_pressure",
                    "disposition": "unknown",
                }
            ]
        },
    }
    core = {
        "core_result_digest": "c" * 64,
        "identity": {"opaque_case_id": case_id(1)},
        "unrelated_label": "hidden-truth",
        "evidence": {
            "graph": {"graph_pair": {"fault_snapshot": {"decision_time_unix_ns": 75}}},
            "ranking": {"propagation": {"local_scoring": {"strengths": strengths}}},
        },
    }
    original = copy.deepcopy(core)
    doc = history.project_core(core)
    assert core == original
    assert doc["decision_time_ns"] == 75
    assert {(r["kind"], r["state"]) for r in doc["evidence"]} == {
        ("episode", "up"),
        ("window_summary", "up"),
        ("mechanism", "unknown"),
    }
    expected_targets = {
        "window_summary": terms[1],
        "episode": terms[2],
        "mechanism": strengths["mechanisms"]["mechanisms"][0],
    }
    text = history.document_text(doc)
    assert "baseline median" in text and "sustained" in text and "undetermined" in text
    assert all(value not in text for value in ("hidden-truth", "internal-service", case_id(1)))
    for fact in doc["evidence"]:
        prefix, pointer = fact["source_ref"].split("#")
        assert prefix == "sha256:" + core["core_result_digest"]
        node = core
        for key in pointer[1:].split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        assert node is expected_targets[fact["kind"]]


def test_benchmark_receipt_is_private_and_cannot_admit_into_human_policy(tmp_path):
    doc = projection(1, 10)
    emit(tmp_path / "evidence.json", doc)
    candidate = {
        "case_id": case_id(1),
        "group_id": "g1",
        "document_path": "evidence.json",
        "available_at_ns": 20,
    }
    manifest = emit(tmp_path / "manifest.json", [candidate])
    truth = emit(tmp_path / "truth.json", {"runs": [{"pod": "label-only-pod"}]})
    sheet = admission.prepare_review(manifest)
    row = sheet["reviews"][0]
    reviews = emit(tmp_path / "reviews.json", sheet)
    with pytest.raises(ValueError):
        admission.build_library(manifest, reviews)
    with pytest.raises(ValueError):
        admission.export_labels([manifest], [reviews])
    row.update(
        status="benchmark-reference",
        truth_path=truth.name,
        truth_sha256=history.sha(truth.read_bytes()),
        truth_pointer="/runs/0",
        service="label-only-pod",
        mechanism="label-only-mechanism",
    )
    emit(reviews, sheet)
    library = admission.build_library(manifest, reviews, benchmark_reference=True)
    assert library["review_policy"] == history.BENCHMARK
    assert library["cases"][0]["document"] == doc
    assert "label-only" not in json.dumps(library)
    assert str(truth) not in json.dumps(library)
    with pytest.raises(ValueError):
        admission.build_library(manifest, reviews)
    with pytest.raises(ValueError):
        history.retrieve_document(projection(2, 50), library, "0" * 64, group_id="g2")
    row["status"] = "approved"
    emit(reviews, sheet)
    with pytest.raises(ValueError):
        admission.build_library(manifest, reviews)
    row.update(reviewer="test reviewer", reviewed_at="2004-01-01T00:00:00Z")
    emit(reviews, sheet)
    assert admission.build_library(manifest, reviews)["review_policy"] == history.HUMAN
    labels = tmp_path / "labels.json"
    assert (
        admission.main(
            [
                "export-labels",
                "--candidates",
                str(manifest),
                "--reviews",
                str(reviews),
                "--out",
                str(labels),
            ]
        )
        == 0
    )
    assert json.loads(labels.read_bytes()) == {
        "labels": [
            {
                "case_id": case_id(1),
                "service": "label-only-pod",
                "mechanism": "label-only-mechanism",
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate case or injection group"):
        admission.export_labels([manifest, manifest], [reviews, reviews])
    row["status"] = "rejected"
    emit(reviews, sheet)
    assert admission.export_labels([manifest], [reviews]) == {"labels": []}
    row["status"] = "approved"
    row["truth_pointer"] = "/runs/1"
    emit(reviews, sheet)
    with pytest.raises(ValueError):
        admission.build_library(manifest, reviews)
    row["truth_pointer"] = "/runs/0"
    emit(reviews, sheet)
    doc["evidence"][0]["state"] = "down"
    emit(tmp_path / "evidence.json", doc)
    with pytest.raises(ValueError):
        admission.build_library(manifest, reviews)


def test_retrieval_filters_query_duplicate_group_and_future_before_embedding():
    query = projection(1, 50)
    cases = []
    for number, group, timestamp in (
        (1, "same-id", 10),
        (2, "same-group", 20),
        (3, "equal-time", 50),
        (4, "future", 60),
        (5, "earlier", 49),
    ):
        cases.append(
            {
                "case_id": case_id(number),
                "group_id": group,
                "available_at_ns": timestamp,
                "document": projection(number, timestamp),
                "review_receipt_sha256": "8" * 64,
            }
        )
    library = {"schema_version": history.LIBRARY, "review_policy": history.HUMAN, "cases": cases}
    eligible = history.eligible_cases(library, query, "same-group")
    assert [case["case_id"] for case in eligible] == [case_id(5)]
    query["evidence"] = []
    receipt = history.retrieve_document(
        query, library, "0" * 64, group_id="same-group", embedder=object()
    )
    assert receipt["status"] == "no-query-evidence" and receipt["matches"] == []


def test_human_review_can_confirm_service_and_leave_mechanism_unknown(tmp_path):
    emit(tmp_path / "evidence.json", projection(1, 10))
    manifest = emit(
        tmp_path / "candidates.json",
        [
            {
                "case_id": case_id(1),
                "group_id": "injection1",
                "available_at_ns": 20,
                "document_path": "evidence.json",
            }
        ],
    )
    truth = emit(tmp_path / "truth.json", {"incident": {"affected_service": "catalog"}})
    sheet = admission.prepare_review(manifest)
    row = sheet["reviews"][0]
    row.update(
        status="approved",
        reviewer="test reviewer",
        reviewed_at="2005-05-05T00:00:00Z",
        service="catalog",
        mechanism=None,
        truth_path=truth.name,
        truth_sha256=history.sha(truth.read_bytes()),
        truth_pointer="/incident",
    )
    reviews = emit(tmp_path / "reviews.json", sheet)
    library = admission.build_library(manifest, reviews)
    assert len(library["cases"]) == 1 and "catalog" not in json.dumps(library)
    assert admission.export_labels([manifest], [reviews]) == {
        "labels": [{"case_id": case_id(1), "service": "catalog", "mechanism": None}]
    }
    row["mechanism"] = " "
    emit(reviews, sheet)
    with pytest.raises(ValueError, match="mechanism"):
        admission.build_library(manifest, reviews)
    row.update(mechanism=None, reviewer=None)
    emit(reviews, sheet)
    with pytest.raises(ValueError, match="reviewer"):
        admission.build_library(manifest, reviews)
    row.update(reviewer="test reviewer", truth_pointer="/missing")
    emit(reviews, sheet)
    with pytest.raises(ValueError, match="pointer"):
        admission.build_library(manifest, reviews)
    row.update(truth_pointer="/incident", service=None)
    emit(reviews, sheet)
    with pytest.raises(ValueError, match="service"):
        admission.build_library(manifest, reviews)


def test_reference_and_label_fields_do_not_enter_embedding():
    doc = projection(1, 20)
    text = history.document_text(doc)
    fact = doc["evidence"][0]
    assert doc["case_id"] not in text and fact["source_ref"] not in text
    fact["ground_truth"] = "private label"
    with pytest.raises(ValueError):
        history.document_text(doc)


def test_two_methods_use_identical_text_and_rule_never_opens_ark(monkeypatch):
    query = projection(1, 40)
    cases = []
    for number, state in ((2, "down"), (3, "up")):
        doc = projection(number, 20)
        doc["evidence"][0]["state"] = state
        cases.append(dict(case_id=doc["case_id"], group_id=str(number), available_at_ns=25,
                          document=doc, review_receipt_sha256="a" * 64))
    library = dict(schema_version=history.LIBRARY, review_policy=history.HUMAN, cases=cases)
    seen = []
    real_rule = history.fixed_history_scores

    def record_rule(query_text, texts):
        seen.append([query_text, *texts.values()])
        return real_rule(query_text, texts)

    def deny_network(*_args, **_kwargs):
        raise AssertionError("rule retrieval must work offline")

    monkeypatch.setattr(history, "fixed_history_scores", record_rule)
    monkeypatch.setattr(history, "make_embedder", deny_network)
    fixed = history.retrieve_document(query, library, "b" * 64, group_id="q", method="fixed-rule")

    class Recorder:
        model_id = "unit-test"

        def embed(self, texts):
            seen.append(list(texts))
            return [[0], [1], [0]]

    vector = history.retrieve_document(query, library, "b" * 64, group_id="q",
                                       method="embedding", embedder=Recorder())
    assert seen[0] == seen[1] and len(seen[0]) == 3
    assert fixed["matches"][0]["case_id"] == vector["matches"][0]["case_id"] == case_id(3)
    assert fixed["metric"] == history.RULE_DISTANCE and fixed["model"] is None
    assert fixed["provider_receipts"] == [] and fixed["matches"][0]["distance"] == 0


def test_auto_selects_only_a_configured_embedding_model(monkeypatch):
    query, doc = projection(1, 40), projection(2, 20)
    case = dict(case_id=doc["case_id"], group_id="h", available_at_ns=25,
                document=doc, review_receipt_sha256="a" * 64)
    library = dict(schema_version=history.LIBRARY, review_policy=history.HUMAN, cases=[case])
    monkeypatch.delenv("NURSE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("ARK_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("ARK_API_KEY", "unused")
    calls = []

    class Recorder:
        model_id = "text-embedding-3-small"

        def embed(self, texts):
            calls.append(list(texts))
            return [[1] for _ in texts]

    monkeypatch.setattr(history, "make_embedder", lambda model: Recorder())
    def retrieve():
        return history.retrieve_document(query, library, "b" * 64, group_id="q")

    assert retrieve()["method"] == "fixed-rule" and not calls
    monkeypatch.setenv("NURSE_EMBEDDING_MODEL", Recorder.model_id)
    result = retrieve()
    assert result["method"] == "embedding" and result["model"] == Recorder.model_id
    assert result["legacy_candidate_count"] == 1 and len(calls) == 1
