import hashlib
import itertools
import json

import pytest

from nurse_rcl.evaluation import history_experiment as experiment
from nurse_rcl.v2.live.verified_history import HUMAN, LIBRARY, PROJECTION


@pytest.mark.parametrize(("n", "r", "k"), [(3, 1, 1), (4, 2, 2), (3, 3, 5), (3, 0, 2)])
def test_analytic_uniform_baseline_matches_all_permutations(n, r, k):
    ids = [str(i) for i in range(n)]
    values = [experiment._measured(p, set(ids[:r]), k) for p in itertools.permutations(ids)]
    assert experiment._uniform(n, r, k) == pytest.approx(
        [sum(col) / len(values) for col in zip(*values, strict=True)]
    )


def orders(qid, group, evidence, pool, semantic, lexical):
    return {
        "case_id": qid, "group_id": group, "eligible_ids": pool, "query_has_evidence": evidence,
        "euclidean": semantic, "bm25": lexical, "cosine": semantic,
        "distances": list(range(len(semantic))),
        "cosine_similarities": [1.0 / (i + 1) for i in range(len(semantic))],
        "bm25_scores": list(range(len(lexical), 0, -1)),
    }


def test_paired_metrics_coverage_and_one_group_limit():
    rankings = {
        "schema": experiment.RANK_SCHEMA, "model": "synthetic", "library_sha256": "a" * 64,
        "vector_norm_range": [1.0, 1.0], "history_groups": {"h1": "a", "h2": "b"}, "queries": [
            orders("q1", "g1", True, ["h1", "h2"], ["h1", "h2"], ["h2", "h1"]),
            orders("q2", "g1", True, ["h2"], ["h2"], ["h2"]),
            orders("q3", "g2", True, [], [], []),
            orders("q4", "g3", False, ["h1"], [], []),
        ],
    }
    labels = {"labels": [
        {"case_id": cid, "service": service, "mechanism": mech}
        for cid, service, mech in [
            ("h1", "s1", "disk"), ("h2", "s2", "network"), ("q1", "s2", "disk"),
            ("q2", "s1", "disk"), ("q3", "s3", "cpu"), ("q4", "s4", "disk"),
        ]
    ]}
    result = experiment.score(rankings, labels, k=1, replicates=100)
    primary = result["relevance"]["same_mechanism"]
    assert primary["relevant_history_coverage"] == 0.5
    assert primary["queries_with_evidence"] == 3
    assert primary["positive_available_means"]["euclidean"]["ndcg_at_k"] == 0.5
    assert primary["all_query_means_zero_if_no_relevant"]["euclidean"]["ndcg_at_k"] == 0.25
    diff = primary["positive_available_paired_differences"]["bm25"]
    assert diff["mean_delta"]["mrr"] == 0.25
    assert diff["independent_groups"] == 2
    assert diff["paired_group_bootstrap_95pct"]["mrr"] == [0.0, 0.5]
    assert primary["per_query"][-1]["methods"]["uniform_random"]["ndcg_at_k"] == 1.0
    service = result["relevance"]["same_service"]["positive_available_paired_differences"]["bm25"]
    assert service["mean_delta"]["mrr"] == -0.5
    assert service["paired_group_bootstrap_95pct"]["mrr"] is None

    del labels["labels"][0]
    with pytest.raises(ValueError, match="missing private"):
        experiment.score(rankings, labels)
    rankings["queries"][0]["bm25"] = ["h1", "h1"]
    with pytest.raises(ValueError, match="exactly the eligible"):
        experiment.score(rankings, labels)


def test_equal_distances_bound_rank_dependent_conclusions():
    q = {"euclidean": ["a", "b", "c", "d"], "distances": [0.0, 1.0, 1.0, 1.0]}
    bounds = experiment._tie_sensitivity(q, {"c"}, 2)
    assert bounds == pytest.approx((True, 0.0, 1 / experiment.math.log2(3)))
    terminal = experiment._tie_sensitivity(q, {"c"}, 4)
    assert not terminal[0]
    assert terminal[1:] == pytest.approx((1 / experiment.math.log2(5), 1 / experiment.math.log2(3)))


def test_service_only_cohort_requires_explicit_cli_mode(tmp_path):
    ranks = {
        "schema": experiment.RANK_SCHEMA, "model": "test", "library_sha256": "c" * 64,
        "history_groups": {"a": "g1"}, "vector_norm_range": [1, 2],
        "queries": [orders("q", "g", True, ["a"], ["a"], ["a"])],
    }
    labels = {"labels": [
        {"case_id": "q", "service": "db"}, {"case_id": "a", "service": "db", "mechanism": None}
    ]}
    rank_path, label_path, output = [tmp_path / f"{n}.json" for n in ("r", "l", "s")]
    rank_path.write_text(json.dumps(ranks), encoding="utf-8")
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    args = ["score", "--ranks", str(rank_path), "--labels", str(label_path), "--out", str(output)]
    with pytest.raises(SystemExit):
        experiment.main(args)
    assert not output.exists()
    assert experiment.main([*args, "--service-only"]) == 0
    report = json.loads(output.read_bytes())
    assert report["primary"] == "same_service/positive_available/ndcg_at_k"
    assert report["label_scope"] == "service-only"
    assert set(report["relevance"]) == {"same_service"}
    assert report["relevance"]["same_service"]["relevant_history_coverage"] == 1.0
    del labels["labels"][0]["service"]
    with pytest.raises(ValueError):
        experiment.score(ranks, labels, service_only=True)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("distances", [0, float("nan")], "invalid distances"),
    ("distances", [0, -1], "invalid distances"),
    ("distances", [1, 0], "violate"),
    ("bm25", ["b", "a"], "tie order"),
    ("cosine_similarities", [1, 1.1], "invalid cosine"),
    ("cosine_similarities", [-1], "invalid cosine"),
])
def test_score_refuses_inconsistent_rank_artifacts(field, value, error):
    q = orders("q", "g", True, ["a", "b"], ["a", "b"], ["a", "b"])
    q["bm25_scores"] = [0.0, 0.0]
    q[field] = value
    ranks = {
        "schema": experiment.RANK_SCHEMA, "model": "synthetic", "library_sha256": "b" * 64,
        "history_groups": {"a": "g1", "b": "g2"}, "queries": [q], "vector_norm_range": [1, 2],
    }
    with pytest.raises(ValueError, match=error):
        experiment.score(ranks, {})


def test_rank_never_embeds_private_or_future_records_and_reuses_equal_text(tmp_path, monkeypatch):
    def document(index, decision, signal):
        return {
            "schema_version": PROJECTION, "case_id": f"opq_{index:032x}",
            "decision_time_ns": decision,
            "evidence": [{
                "kind": "episode", "signal": signal, "state": "up",
                "source_ref": f"src_{index:032x}",
            }],
        }

    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    near, far, future, query, query2 = [
        document(*spec) for spec in
        [(1, 1, "cpu"), (2, 2, "errors"), (3, 20, "future_token"), (4, 10, "cpu"), (5, 11, "mem")]
    ]
    lib = {
        "schema_version": LIBRARY, "review_policy": HUMAN,
        "cases": [{"case_id": doc["case_id"], "group_id": f'g-{doc["decision_time_ns"]}',
                   "available_at_ns": doc["decision_time_ns"], "review_receipt_sha256": "a" * 64,
                   "document": doc} for doc in [near, far, future]],
    }
    path = save("library.json", lib)
    save("query.json", query)
    save("query2.json", query2)
    manifest = save("queries.json", {
        "queries": [{"group_id": "query", "document_path": "query.json"},
                    {"group_id": "query2", "document_path": "query2.json"}],
    })
    sent = []

    class Embedder:
        model_id = "test"
        receipts = ()

        def embed(self, texts):
            sent.extend(texts)
            return [[4.0 if "cpu" in t else 1.0, float("errors" in t)] for t in texts]

    monkeypatch.setattr(experiment, "ArkTextEmbedder", Embedder)
    ranks_path, score_path = tmp_path / "ranks.json", tmp_path / "score.json"
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    rank_args = [
        "rank", "--library", str(path), "--queries", str(manifest), "--out", str(ranks_path)
    ]
    with pytest.raises(SystemExit):
        experiment.main([*rank_args, "--library-sha256", "0" * 64])
    assert not sent and not ranks_path.exists()
    assert experiment.main([*rank_args, "--library-sha256", pin]) == 0
    frozen = json.loads(ranks_path.read_bytes())
    assert len(sent) == 3
    assert all("future" not in s and "opq_" not in s and "src_" not in s for s in sent)
    assert frozen["queries"][0]["euclidean"] == frozen["queries"][0]["bm25"] == [
        near["case_id"], far["case_id"],
    ]
    assert frozen["queries"][0]["cosine"] == frozen["queries"][0]["euclidean"]
    assert frozen["queries"][1]["cosine"] == [near["case_id"], far["case_id"]]
    assert frozen["queries"][1]["euclidean"] == [far["case_id"], near["case_id"]]
    assert frozen["vector_norm_range"] == pytest.approx([1.0, 4.0])
    assert future["case_id"] not in frozen["history_groups"]
    annotated = [(near, "disk"), (far, "network"), (query, "disk"), (query2, "network")]
    private_path = save("labels.json", {"labels": [
        {"case_id": doc["case_id"], "service": "s1", "mechanism": mechanism}
        for doc, mechanism in annotated
    ]})
    assert experiment.main([
        "score", "--ranks", str(ranks_path), "--labels", str(private_path), "--out", str(score_path)
    ]) == 0
    result = json.loads(score_path.read_bytes())
    assert result["model"] == frozen["model"] == "test"
    assert result["vector_norm_range"] == frozen["vector_norm_range"]
    assert result["library_sha256"] == pin
    assert result["ranks_sha256"] == hashlib.sha256(ranks_path.read_bytes()).hexdigest()
    assert result["labels_sha256"] == hashlib.sha256(private_path.read_bytes()).hexdigest()
    means = result["relevance"]["same_mechanism"]["positive_available_means"]
    assert means["euclidean"]["mrr"] == 1.0
    assert means["cosine"]["mrr"] == 0.75
    save("query.json", near)
    with pytest.raises(ValueError, match="disjoint"):
        experiment.rank(path, manifest, None)
    lib["review_policy"] = "benchmark-reference-only"
    save("library.json", lib)
    with pytest.raises(ValueError, match="human-reviewed"):
        experiment.rank(path, manifest, None)
