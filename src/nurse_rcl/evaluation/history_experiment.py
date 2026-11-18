"""Rank histories from current evidence, then score frozen ranks with private reference labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from typing import Any

from nurse_rcl.v2.live.semantic_history_retrieval import (
    DEFAULT_TOP_K,
    ArkTextEmbedder,
    rank_history_vectors,
)
from nurse_rcl.v2.live.verified_history import (
    document_text,
    eligible_cases,
    load_document,
    read_library,
)

RANK_SCHEMA = "nurse.history-experiment.ranks/1"
SCORE_SCHEMA = "nurse.history-experiment.scores/1"
ORDERED_METHODS = ("euclidean", "bm25", "cosine")
METHODS = (*ORDERED_METHODS, "uniform_random")
METRICS = ("mrr", "ndcg_at_k", "recall_at_k", "hit_at_k")
CaseLabel = tuple[str, str | None]
Criteria = Callable[[CaseLabel, CaseLabel], bool]


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("case, group and label identifiers must be nonempty, trimmed strings")
    return value


def _rows(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get(key), list) or not value[key]:
        raise ValueError(f"expected a nonempty {key} array")
    if any(not isinstance(row, dict) for row in value[key]):
        raise ValueError(f"invalid row in {key}")
    return list(value[key])


def _bm25_scores(query: str, docs: Mapping[str, str]) -> dict[str, float]:
    """Okapi BM25 on the identical eligible pool and texts, with fixed k1=1.2 and b=0.75."""
    def tokens(text: str) -> list[str]:
        return re.findall(r"[^\W_]+", text.casefold())

    terms = set(tokens(query))
    counts = {cid: Counter(tokens(doc)) for cid, doc in docs.items()}
    lengths = {cid: sum(freqs.values()) for cid, freqs in counts.items()}
    mean_len = sum(lengths.values()) / len(docs) if docs else 0.0
    if not mean_len:
        return dict.fromkeys(docs, 0.0)
    idfs = {}
    for term in sorted(terms):
        hits = sum(term in freqs for freqs in counts.values())
        idfs[term] = math.log1p((len(docs) - hits + 0.5) / (hits + 0.5))
    scores = {}
    for cid, freqs in counts.items():
        norm = 1.2 * (0.25 + 0.75 * lengths[cid] / mean_len)
        scores[cid] = sum(idf * 2.2 * freqs[t] / (freqs[t] + norm) for t, idf in idfs.items())
    return scores


def rank(library_path: Path, query_manifest: Path, expected_sha256: str | None) -> dict[str, Any]:
    library, digest = read_library(library_path, expected_sha256=expected_sha256)
    if library["review_policy"] != "human-reviewed":
        raise ValueError("formal retrieval evaluation requires a human-reviewed library")
    manifest = json.loads(query_manifest.read_text(encoding="utf-8-sig"))
    queries = []
    history_texts: dict[str, str] = {}
    unique_texts: dict[str, int] = {}
    seen: set[str] = set()
    history_groups: dict[str, str] = {}
    archived_cases = {case["case_id"] for case in library["cases"]}
    archived_groups = {case["group_id"] for case in library["cases"]}
    for spec in _rows(manifest, "queries"):
        if set(spec) != {"document_path", "group_id"}:
            raise ValueError("a query specification requires only document_path and group_id")
        doc = load_document(query_manifest.parent / _identifier(spec["document_path"]))
        qid, group = _identifier(doc["case_id"]), _identifier(spec["group_id"])
        if qid in seen or doc.get("group_id") not in (None, group):
            raise ValueError("duplicate query case ID or inconsistent query group")
        seen.add(qid)
        if qid in archived_cases or group in archived_groups:
            raise ValueError("query and history cases and groups must be disjoint")
        eligible = eligible_cases(library, doc, group)
        qtext = document_text(doc)
        pool = []
        if eligible and qtext:
            unique_texts.setdefault(qtext, len(unique_texts))
        for case in eligible:
            cid, cgroup = _identifier(case["case_id"]), _identifier(case["group_id"])
            text = document_text(case["document"])
            if cid in pool or history_groups.setdefault(cid, cgroup) != cgroup:
                raise ValueError("duplicate or inconsistent candidate")
            if history_texts.setdefault(cid, text) != text:
                raise ValueError("history changed between queries")
            if qtext:
                unique_texts.setdefault(text, len(unique_texts))
            pool.append(cid)
        queries.append((qid, group, qtext, sorted(pool)))
    provider = ArkTextEmbedder()
    vectors = provider.embed(list(unique_texts))
    norms = [math.hypot(*v) for v in vectors]
    if any(not math.isfinite(norm) or norm == 0 for norm in norms):
        raise ValueError("cosine comparison requires finite, nonzero embedding norms")
    units = [tuple(x / norm for x in v) for v, norm in zip(vectors, norms, strict=True)]
    out = []
    for qid, group, qtext, pool in queries:
        docs = {cid: history_texts[cid] for cid in pool}
        lexical = _bm25_scores(qtext, docs) if qtext else {}
        lexical_order = sorted(lexical, key=lambda cid: (-lexical[cid], cid))
        if docs and qtext:
            query_unit = units[unique_texts[qtext]]
            similarity = {}
            for cid, text in docs.items():
                unit = units[unique_texts[text]]
                sim = math.fsum(a * b for a, b in zip(query_unit, unit, strict=True))
                similarity[cid] = min(1.0, max(-1.0, sim))
            cosine_order = sorted(docs, key=lambda cid: (-similarity[cid], cid))
            matches = rank_history_vectors(
                vectors[unique_texts[qtext]],
                {cid: vectors[unique_texts[text]] for cid, text in docs.items()},
                top_k=len(docs),
            )
        else:
            matches = ()
            cosine_order = []
            similarity = {}
        out.append({
            "case_id": qid, "group_id": group, "eligible_ids": pool,
            "query_has_evidence": bool(qtext),
            "query_text_sha256": hashlib.sha256(qtext.encode("utf-8")).hexdigest(),
            "euclidean": [m.case_id for m in matches], "distances": [m.distance for m in matches],
            "bm25": lexical_order, "bm25_scores": [lexical[cid] for cid in lexical_order],
            "cosine": cosine_order,
            "cosine_similarities": [similarity[cid] for cid in cosine_order],
        })
    return {
        "schema": RANK_SCHEMA, "library_sha256": digest, "model": provider.model_id,
        "receipts": [asdict(r) for r in provider.receipts], "history_groups": history_groups,
        "queries": out, "vector_norm_range": [min(norms), max(norms)] if norms else None,
    }


def _measured(order: Sequence[str], relevant: set[str], k: int) -> tuple[float, ...]:
    if not relevant or not order:
        return (0.0,) * len(METRICS)
    ranks = [idx for idx, cid in enumerate(order, start=1) if cid in relevant]
    top = [rank for rank in ranks if rank <= k]
    discount = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    dcg = sum(1 / math.log2(i + 1) for i in top)
    return 1 / ranks[0], dcg / discount, len(top) / len(relevant), float(bool(top))


def _uniform(n: int, r: int, k: int) -> tuple[float, ...]:
    """Analytic expectation under a uniform permutation without replacement."""
    if r == 0:
        return (0.0,) * len(METRICS)
    limit = min(k, n)
    survival, mrr, hit = 1.0, 0.0, 0.0
    for i in range(1, n - r + 2):
        first = survival * r / (n - i + 1)
        mrr += first / i
        if i <= limit:
            hit += first
        survival -= first
    discount = sum(1 / math.log2(i + 1) for i in range(1, limit + 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(limit, r) + 1))
    return mrr, (r / n) * discount / ideal, limit / n, min(1.0, hit)


def _tie_sensitivity(q: dict[str, Any], relevant: set[str], k: int) -> tuple[bool, float, float]:
    """Possible nDCG extrema over permutations within equal raw-Euclidean-distance blocks."""
    order, distances = q["euclidean"], q["distances"]
    crosses = k < len(distances) and distances[k - 1] == distances[k]
    if not order or not relevant:
        return crosses, 0.0, 0.0
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    lo, hi, start = 0.0, 0.0, 0
    for _, ties in groupby(zip(distances, order, strict=True), key=lambda pair: pair[0]):
        if start >= k:
            break
        ids = [cid for _, cid in ties]
        count, end = sum(cid in relevant for cid in ids), start + len(ids)
        hi += sum(1 / math.log2(i + 2) for i in range(start, min(k, start + count)))
        lo += sum(1 / math.log2(i + 2) for i in range(end - count, min(k, end)))
        start = end
    return crosses, lo / ideal, hi / ideal


def _means(values: list[tuple[float, ...]]) -> dict[str, float | None]:
    if not values:
        return dict.fromkeys(METRICS)
    return dict(zip(METRICS, (sum(col) / len(values) for col in zip(*values, strict=True)),
                    strict=True))


def _percentile(values: list[float], p: float) -> float:
    x = p * (len(values) - 1)
    lo = math.floor(x)
    return values[lo] + (x - lo) * (values[math.ceil(x)] - values[lo])


def _paired(
    groups: Sequence[str], left: list[tuple[float, ...]], right: list[tuple[float, ...]],
    replicates: int, seed: int,
) -> dict[str, Any]:
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(METRICS))
    sizes = Counter(groups)
    for group, lhs, rhs in zip(groups, left, right, strict=True):
        for i, (a, b) in enumerate(zip(lhs, rhs, strict=True)):
            sums[group][i] += a - b
    delta = {metric: sum(row[i] for row in sums.values()) / len(left) if left else None
             for i, metric in enumerate(METRICS)}
    intervals: dict[str, Any] = dict.fromkeys(METRICS)
    if len(sizes) >= 2:
        samples: list[list[float]] = [[] for _ in METRICS]
        rng, names = random.Random(seed), sorted(sizes)
        for _ in range(replicates):
            multiplicity = Counter(rng.choices(names, k=len(names)))
            n = sum(sizes[group] * count for group, count in multiplicity.items())
            for i, sample in enumerate(samples):
                sample.append(sum(sums[g][i] * c for g, c in multiplicity.items()) / n)
        for name, sample in zip(METRICS, samples, strict=True):
            sample.sort()
            intervals[name] = [_percentile(sample, 0.025), _percentile(sample, 0.975)]
    return {"mean_delta": delta, "paired_group_bootstrap_95pct": intervals,
            "independent_groups": len(sizes)}


def _validated_queries(ranks: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(ranks, dict) or ranks.get("schema") != RANK_SCHEMA:
        raise ValueError("unrecognized frozen rank schema")
    _identifier(ranks.get("model"))
    if not re.fullmatch(r"[0-9a-f]{64}", str(ranks.get("library_sha256"))):
        raise ValueError("invalid frozen library digest")
    group_map = ranks.get("history_groups")
    if not isinstance(group_map, dict):
        raise ValueError("missing candidate group mapping")
    cgroups = {_identifier(value) for value in group_map.values()}
    queries = _rows(ranks, "queries")
    qids: set[str] = set()
    for row in queries:
        qid, group = _identifier(row.get("case_id")), _identifier(row.get("group_id"))
        if qid in qids or qid in group_map or group in cgroups:
            raise ValueError("duplicate query or query/history split overlap")
        qids.add(qid)
        if not isinstance(row.get("eligible_ids"), list):
            raise ValueError("missing eligible pool")
        pool = {_identifier(cid) for cid in row["eligible_ids"]}
        if not pool.issubset(group_map) or len(pool) != len(row["eligible_ids"]):
            raise ValueError("unknown or duplicate eligible candidate")
        if type(row.get("query_has_evidence")) is not bool:
            raise ValueError("missing query evidence availability")
        rankable = pool if row["query_has_evidence"] else set()
        for method in ORDERED_METHODS:
            order = row.get(method)
            if (
                not isinstance(order, list)
                or len(order) != len(rankable)
                or any(not isinstance(cid, str) for cid in order)
                or set(order) != rankable
            ):
                raise ValueError(f"{method} order must rank exactly the eligible candidates")
        for method, field, low, high in (
            ("euclidean", "distances", 0.0, math.inf),
            ("bm25", "bm25_scores", 0.0, math.inf),
            ("cosine", "cosine_similarities", -1.0, 1.0),
        ):
            values = row.get(field)
            if (
                not isinstance(values, list) or len(values) != len(rankable)
                or any(type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v)
                       for v in values)
            ):
                raise ValueError(f"invalid {field}")
            vals = values if method == "euclidean" else [-v for v in values]
            pairs = list(zip(vals, row[method], strict=True))
            if pairs != sorted(pairs):
                raise ValueError(f"{method} ranks violate score or case ID tie order")
    bounds = ranks.get("vector_norm_range")
    if any(q["euclidean"] for q in queries):
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= bounds[1]):
            raise ValueError("invalid vector norm range")
    elif bounds is not None:
        raise ValueError("unexpected vector norm range for empty rankings")
    return queries


def score(
    ranks: dict[str, Any], labels: dict[str, Any], *, k: int = DEFAULT_TOP_K,
    replicates: int = 2000, seed: int = 1729, service_only: bool = False,
) -> dict[str, Any]:
    if type(k) is not int or k < 1 or type(replicates) is not int or replicates < 2:
        raise ValueError("k must be positive and bootstrap replicates must be at least two")
    queries = _validated_queries(ranks)
    private: dict[str, CaseLabel] = {}
    for row in _rows(labels, "labels"):
        cid = _identifier(row.get("case_id"))
        if cid in private:
            raise ValueError("duplicate private case label")
        mechanism = row.get("mechanism")
        if mechanism is None and not service_only:
            raise ValueError("full evaluation requires independent mechanism labels")
        if mechanism is not None:
            mechanism = _identifier(mechanism)
        private[cid] = (_identifier(row.get("service")), mechanism)
    criteria: dict[str, Criteria] = {"same_service": lambda a, b: a[0] == b[0]}
    if not service_only:
        criteria.update({
            "same_mechanism": lambda a, b: a[1] == b[1],
            "same_service_and_mechanism": lambda a, b: a == b,
        })
    reports = {}
    for name, matches in criteria.items():
        per_query = []
        positive_groups = []
        tie_intervals = []
        boundary_ties = 0
        all_values: dict[str, list[tuple[float, ...]]] = {m: [] for m in METHODS}
        positive: dict[str, list[tuple[float, ...]]] = {m: [] for m in METHODS}
        for q in queries:
            qid, pool = q["case_id"], q["eligible_ids"]
            if qid not in private or any(cid not in private for cid in pool):
                raise ValueError("missing private label for a ranked case")
            relevant = {cid for cid in pool if matches(private[qid], private[cid])}
            values = {m: _measured(q[m], relevant, k) for m in ORDERED_METHODS}
            values["uniform_random"] = _uniform(len(pool), len(relevant), k)
            crosses, lo, hi = _tie_sensitivity(q, relevant, k)
            boundary_ties += int(crosses)
            per_query.append({
                "case_id": qid, "group_id": q["group_id"], "eligible": len(pool),
                "relevant": len(relevant), "query_has_evidence": q["query_has_evidence"],
                "euclidean_tie_crosses_k": crosses, "euclidean_tie_ndcg_range": [lo, hi],
                "methods": {m: dict(zip(METRICS, v, strict=True)) for m, v in values.items()},
            })
            if relevant:
                positive_groups.append(q["group_id"])
                tie_intervals.append((lo, hi))
            for method, row_values in values.items():
                all_values[method].append(row_values)
                if relevant:
                    positive[method].append(row_values)
        reports[name] = {
            "queries": len(queries),
            "queries_with_candidates": sum(bool(q["eligible_ids"]) for q in queries),
            "queries_with_evidence": sum(q["query_has_evidence"] for q in queries),
            "queries_with_euclidean_tie_crossing_k": boundary_ties,
            "positive_available_euclidean_tie_ndcg_range": [
                sum(bounds) / len(tie_intervals) for bounds in zip(*tie_intervals, strict=True)
            ] if tie_intervals else None,
            "queries_with_relevant_history": len(positive_groups),
            "relevant_history_coverage": len(positive_groups) / len(queries),
            "all_query_means_zero_if_no_relevant": {m: _means(v) for m, v in all_values.items()},
            "positive_available_means": {m: _means(v) for m, v in positive.items()},
            "positive_available_paired_differences": {
                m: _paired(positive_groups, positive["euclidean"], positive[m], replicates, seed)
                for m in METHODS[1:]
            },
            "per_query": per_query,
        }
    primary = "same_service" if service_only else "same_mechanism"
    return {"schema": SCORE_SCHEMA, "primary": f"{primary}/positive_available/ndcg_at_k",
            "label_scope": "service-only" if service_only else "service-and-mechanism",
            "k": k, "library_sha256": ranks.get("library_sha256"), "seed": seed,
            "model": ranks["model"], "vector_norm_range": ranks.get("vector_norm_range"),
            "bootstrap_replicates": replicates, "relevance": reports}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    ranking = modes.add_parser("rank", help="freeze rankings without access to relevance labels")
    ranking.add_argument("--library", required=True, type=Path)
    ranking.add_argument("--library-sha256")
    ranking.add_argument("--queries", required=True, type=Path)
    scoring = modes.add_parser("score", help="join independently verified labels after ranking")
    scoring.add_argument("--ranks", required=True, type=Path)
    scoring.add_argument("--labels", required=True, type=Path)
    scoring.add_argument("--service-only", action="store_true",
                         help="score an independently labeled service-only cohort")
    scoring.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    scoring.add_argument("--bootstrap-replicates", type=int, default=2000)
    scoring.add_argument("--seed", type=int, default=1729)
    for mode in (ranking, scoring):
        mode.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("output already exists; choose a fresh path for frozen results")
        if args.mode == "rank":
            result = rank(args.library, args.queries, args.library_sha256)
        else:
            raw_ranks, raw_labels = args.ranks.read_bytes(), args.labels.read_bytes()
            result = score(json.loads(raw_ranks), json.loads(raw_labels), k=args.k,
                           replicates=args.bootstrap_replicates, seed=args.seed,
                           service_only=args.service_only)
            result["ranks_sha256"] = hashlib.sha256(raw_ranks).hexdigest()
            result["labels_sha256"] = hashlib.sha256(raw_labels).hexdigest()
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.out.open("x", encoding="utf-8") as target:
            target.write(encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
