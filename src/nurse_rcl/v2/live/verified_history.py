"""Project frozen evidence and retrieve independently admitted, earlier incidents."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nurse_rcl.v2.live.fixed_history_similarity import (
    RULE_DISTANCE,
    RULE_ID,
    HistoryMethod,
    fixed_history_scores,
)
from nurse_rcl.v2.live.semantic_history_retrieval import (
    HistoryMatch,
    TextEmbedder,
    make_embedder,
    rank_history,
    requested_embedding_model,
)

PROJECTION = "nurse.history-signal-projection/1"
PROJECTION_V2 = "nurse.history-signal-projection/2"
LEGACY_PROFILE = "direction-only"
AMPLITUDE_QUALITY_PROFILE = "amplitude-quality"
LIBRARY = "nurse.verified-history-library/1"
HUMAN = "human-reviewed"
BENCHMARK = "benchmark-reference-only"
_STRENGTHS = "/evidence/ranking/propagation/local_scoring/strengths"
_CASE = re.compile(r"opq_[0-9a-f]{32}")
_GROUP = re.compile(r"[a-zA-Z0-9_.-]{1,128}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SIGNAL = re.compile(r"[A-Za-z][A-Za-z0-9_.: /%-]{0,95}")
_SOURCE = re.compile(r"(?:src_[a-f0-9]{32,64}|sha256:[0-9a-f]{64}#/evidence/[a-zA-Z0-9_/]+)")
_STATES = {
    "episode": {"up", "down"},
    "window_summary": {"up", "down"},
    "mechanism": {"true", "false", "unknown"},
}
_MAX_FACTS = 24

JsonRecord = dict[str, Any]
_CanonicalFacts = dict[tuple[str, str, str], tuple[float, str]]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj: object) -> str:
    return sha(
        json.dumps(
            obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _matches(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def group_id_ok(value: object) -> bool:
    return _matches(value, _GROUP)


def _time(value: object) -> bool:
    return type(value) is int and value > 0


def validate_document(doc: JsonRecord) -> JsonRecord:
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema_version", "case_id", "decision_time_ns", "evidence"}
        or doc["schema_version"] not in (PROJECTION, PROJECTION_V2)
        or not _matches(doc["case_id"], _CASE)
    ):
        raise ValueError("invalid evidence projection schema or case identity")
    if not _time(doc["decision_time_ns"]) or not isinstance(doc["evidence"], list):
        raise ValueError("invalid decision time or evidence list")
    seen = set()
    for item in doc["evidence"]:
        if not isinstance(item, dict):
            raise ValueError("unexpected evidence fields")
        fields = {"kind", "signal", "state", "source_ref"}
        if doc["schema_version"] == PROJECTION_V2 and item.get("kind") == "window_summary":
            fields.update(("strength_band", "quality_band"))
            if item.get("strength_band") not in ("slight", "moderate", "large", "very large") or (
                item.get("quality_band") not in ("low", "moderate", "high")
            ):
                raise ValueError("invalid magnitude or observation quality band")
        if set(item) != fields:
            raise ValueError("unexpected evidence fields")
        if (
            not _matches(item["signal"], _SIGNAL)
            or not _matches(item["source_ref"], _SOURCE)
            or item["state"] not in _STATES.get(item["kind"], set())
        ):
            raise ValueError("invalid signal, state or source reference")
        signature = (item["kind"], item["signal"], item["state"])
        if signature in seen:
            raise ValueError("duplicate canonical fact in a projection")
        seen.add(signature)
    if len(seen) > _MAX_FACTS:
        raise ValueError("projection exceeds the canonical fact budget")
    return doc


def load_document(path: Path) -> JsonRecord:
    return validate_document(json.loads(path.read_bytes()))


def _positive(value: object) -> bool:
    if type(value) is not int and type(value) is not float:
        return False
    return math.isfinite(value) and value > 0


def _includes_bands(profile: str) -> bool:
    if profile not in (AMPLITUDE_QUALITY_PROFILE, LEGACY_PROFILE):
        raise ValueError("unknown historical evidence profile")
    return profile == AMPLITUDE_QUALITY_PROFILE


def project_core(core: JsonRecord, *, profile: str = AMPLITUDE_QUALITY_PROFILE) -> JsonRecord:
    """Use only registered measurement/mechanism fields as semantic features."""
    use_bands = _includes_bands(profile)
    evidence = core["evidence"]
    strengths = evidence["ranking"]["propagation"]["local_scoring"]["strengths"]
    core_id = core["core_result_digest"]
    if not _matches(core_id, _DIGEST):
        raise ValueError("invalid frozen core digest")
    # Keep the strongest representative per distinct statement. Reserve eight
    # slots for mechanisms so repeated unknown gates cannot drown out observations.
    observations: _CanonicalFacts = {}
    gates: _CanonicalFacts = {}

    def keep(
        store: _CanonicalFacts, kind: str, signal: str, state: str, priority: float, pointer: str
    ) -> None:
        if not _matches(signal, _SIGNAL):
            return
        key = kind, signal, state
        source = f"sha256:{core_id}#{_STRENGTHS}/{pointer}"
        if key not in store or (-priority, source) < store[key]:
            store[key] = (-priority, source)

    for index, term in enumerate(strengths["terms"]):
        strength = term["local_strength"]
        if not (
            term["candidate_service_key"]
            and _positive(term["quality_factor"])
            and _positive(strength)
        ):
            continue
        calculation = term["calculation"]
        if term["kind"] == "window_summary":
            z = calculation["signed_z"]
            if type(z) not in (int, float) or not math.isfinite(z) or not z:
                continue
            kind, state = "window_summary", "up" if z > 0 else "down"
        elif term["kind"] == "paid_episode" and calculation["direction"] in {"up", "down"}:
            kind, state = "episode", calculation["direction"]
        else:
            continue
        keep(observations, kind, term["metric_name"], state, strength, f"terms/{index}")

    for index, gate in enumerate(strengths["mechanisms"]["mechanisms"]):
        status = {"accepted": ("true", 2), "rejected": ("false", 1), "unknown": ("unknown", 0)}.get(
            gate["disposition"]
        )
        if status and gate["candidate_service_key"]:
            keep(
                gates, "mechanism", gate["mechanism_key"], *status, f"mechanisms/mechanisms/{index}"
            )

    def best(store: _CanonicalFacts, limit: int) -> list[dict[str, str]]:
        rows = sorted(store.items(), key=lambda pair: (pair[1][0], pair[0]))
        return [
            {"kind": k, "signal": s, "state": t, "source_ref": source}
            for (k, s, t), (_, source) in rows[:limit]
        ]

    mechanisms = best(gates, 8)
    rows = best(observations, _MAX_FACTS - len(mechanisms)) + mechanisms
    if use_bands:
        for row in rows:
            if row["kind"] == "window_summary":
                # Reuse the exact term selected above, including deterministic tie-breaking.
                term = strengths["terms"][int(row["source_ref"].rsplit("/", 1)[1])]
                magnitude = abs(term["calculation"]["signed_z"])
                quality = term["quality_factor"]
                row["strength_band"] = (
                    "slight"
                    if magnitude < 1
                    else "moderate"
                    if magnitude < 3
                    else "large"
                    if magnitude < 6
                    else "very large"
                )
                row["quality_band"] = (
                    "low" if quality < 0.5 else "moderate" if quality < 0.8 else "high"
                )
    return validate_document(
        {
            "schema_version": PROJECTION_V2 if use_bands else PROJECTION,
            "case_id": core["identity"]["opaque_case_id"],
            "decision_time_ns": evidence["graph"]["graph_pair"]["fault_snapshot"][
                "decision_time_unix_ns"
            ],
            "evidence": rows,
        }
    )


def _sentence(row: JsonRecord) -> str:
    signal = row["signal"].replace("_", " ").replace(".", " ")
    state = row["state"]
    if row["kind"] == "mechanism":
        meaning = {"true": "satisfied", "false": "rejected", "unknown": "undetermined"}[state]
        return f"For a candidate service, the mechanism conditions for {signal} were {meaning}."
    if row["kind"] == "episode":
        direction = "high" if state == "up" else "low"
        return f"A candidate service had a sustained episode of unusually {direction} {signal}."
    direction = "higher" if state == "up" else "lower"
    return (
        f"For a candidate service, the fault-window median of {signal} "
        f"was {direction} than its baseline median."
    )


def _ordered_rows(doc: JsonRecord) -> list[JsonRecord]:
    return sorted(
        validate_document(doc)["evidence"], key=lambda v: (v["kind"], v["signal"], v["state"])
    )


def rendered_evidence(
    doc: JsonRecord, *, profile: str = AMPLITUDE_QUALITY_PROFILE
) -> list[JsonRecord]:
    use_bands = _includes_bands(profile)
    result = []
    for row in _ordered_rows(doc):
        text = _sentence(row)
        if use_bands and "strength_band" in row:
            text += (
                f" The standardized change was {row['strength_band']}; "
                f"observation quality was {row['quality_band']}."
            )
        result.append({"text": text, "source_ref": row["source_ref"]})
    return result


def document_text(doc: JsonRecord, *, profile: str = AMPLITUDE_QUALITY_PROFILE) -> str:
    use_bands = _includes_bands(profile)
    lines = []
    for row in _ordered_rows(doc):
        lines.append(_sentence(row))
        if use_bands and "strength_band" in row:
            for attr in ("strength_band", "quality_band"):
                lines.append(
                    f"Detail | window_summary | {row['signal']} | {row['state']} | "
                    f"{attr} | {row[attr]}"
                )
    return "\n".join(lines)


def validate_library(library: JsonRecord) -> None:
    if (
        not isinstance(library, dict)
        or set(library) != {"schema_version", "review_policy", "cases"}
        or library["schema_version"] != LIBRARY
        or library["review_policy"] not in {HUMAN, BENCHMARK}
        or not isinstance(library["cases"], list)
    ):
        raise ValueError("invalid historical library format")
    ids, groups = set(), set()
    for case in library["cases"]:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "group_id",
            "available_at_ns",
            "document",
            "review_receipt_sha256",
        }:
            raise ValueError("invalid admitted case schema")
        doc = validate_document(case["document"])
        if (
            case["case_id"] != doc["case_id"]
            or not doc["evidence"]
            or not group_id_ok(case["group_id"])
            or not _matches(case["review_receipt_sha256"], _DIGEST)
            or not _time(case["available_at_ns"])
            or case["available_at_ns"] < doc["decision_time_ns"]
        ):
            raise ValueError("invalid historical admission, availability, or evidence")
        if case["case_id"] in ids or case["group_id"] in groups:
            raise ValueError("duplicate historical case or injection group")
        ids.add(case["case_id"])
        groups.add(case["group_id"])


def read_library(path: Path, expected_sha256: str | None = None) -> tuple[JsonRecord, str]:
    raw = path.read_bytes()
    checksum = sha(raw)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("historical library differs from the pinned SHA-256")
    library = json.loads(raw)
    validate_library(library)
    return library, checksum


def eligible_cases(library: JsonRecord, query: JsonRecord, group_id: str) -> list[JsonRecord]:
    if not group_id_ok(group_id):
        raise ValueError("a query requires a stable injection group")
    return [
        c
        for c in library["cases"]
        if c["case_id"] != query["case_id"]
        and c["group_id"] != group_id
        and c["available_at_ns"] < query["decision_time_ns"]
    ]


def retrieve_document(
    query: JsonRecord,
    library: JsonRecord,
    library_sha256: str,
    *,
    group_id: str,
    top_k: int = 5,
    embedder: TextEmbedder | None = None,
    method: HistoryMethod | None = None,
    embedding_model: str | None = None,
    profile: str = AMPLITUDE_QUALITY_PROFILE,
    benchmark_reference: bool = False,
) -> JsonRecord:
    validate_document(query)
    validate_library(library)
    if library["review_policy"] != HUMAN and not benchmark_reference:
        raise ValueError("runtime requires a human-reviewed historical library")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    pool = {c["case_id"]: c["document"] for c in eligible_cases(library, query, group_id)}
    configured = requested_embedding_model(embedding_model) if method != "fixed-rule" else None
    if method is None:
        method = "embedding" if configured or getattr(embedder, "model_id", None) else "fixed-rule"
    if method not in ("fixed-rule", "embedding"):
        raise ValueError("unknown historical retrieval method")
    model = None
    if method == "embedding":
        if embedder is not None:
            model = embedder
        elif configured is not None:
            model = make_embedder(configured)
        else:
            raise ValueError("embedding retrieval requires an embedding model ID")
        if configured and getattr(model, "model_id", configured) != configured:
            raise ValueError("historical embedder differs from the requested model")
    matches: tuple[HistoryMatch, ...] = ()
    if pool and query["evidence"]:
        query_text = document_text(query, profile=profile)
        documents = {cid: document_text(doc, profile=profile) for cid, doc in pool.items()}
        if model is None:
            scores = fixed_history_scores(query_text, documents)
            ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
            matches = tuple(HistoryMatch(cid, 1.0 - scores[cid]) for cid in ordered)
        else:
            matches = rank_history(query_text, documents, model, top_k=top_k)
    return {
        "schema_version": "nurse.history-retrieval/1",
        "query_id": query["case_id"],
        "query_sha256": digest(query),
        "library_sha256": library_sha256,
        "review_policy": library["review_policy"],
        "method": method,
        "evidence_profile": profile,
        "query_projection": query["schema_version"],
        "legacy_candidate_count": sum(doc["schema_version"] == PROJECTION for doc in pool.values()),
        "rule_id": RULE_ID if method == "fixed-rule" else None,
        "model": getattr(model, "model_id", configured) if model is not None else None,
        "metric": RULE_DISTANCE if method == "fixed-rule" else "euclidean",
        "eligible_count": len(pool),
        "provider_receipts": [asdict(row) for row in getattr(model, "receipts", ())]
        if matches else [],
        "status": "ranked"
        if matches
        else ("no-query-evidence" if not query["evidence"] else "empty-pool"),
        "matches": [
            {
                "case_id": m.case_id,
                "rank": i,
                "distance": m.distance,
                "evidence": rendered_evidence(pool[m.case_id], profile=profile),
            }
            for i, m in enumerate(matches, 1)
        ],
    }


def retrieve_verified_history(
    *,
    query_core: JsonRecord,
    library_path: Path,
    query_id: str,
    decision_time: int,
    query_group_id: str,
    top_k: int = 5,
    embedding_model: str | None = None,
    method: HistoryMethod | None = None,
    profile: str = AMPLITUDE_QUALITY_PROFILE,
    embedder: TextEmbedder | None = None,
    expected_library_sha256: str | None = None,
) -> JsonRecord:
    library, checksum = read_library(Path(library_path), expected_library_sha256)
    query = project_core(query_core, profile=profile)
    if query["case_id"] != query_id or query["decision_time_ns"] != decision_time:
        raise ValueError("historical retrieval query differs from the frozen incident")
    return retrieve_document(
        query, library, checksum, group_id=query_group_id, top_k=top_k, embedder=embedder,
        method=method, embedding_model=embedding_model, profile=profile,
    )
