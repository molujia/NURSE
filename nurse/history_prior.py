"""History prior helpers for NURSE ranking.

The history prior stores each finished case with a compact semantic payload and
the final score summary. New cases embed their semantic payload, retrieve the
most similar approved historical cases, and transfer the historical service
scores back into the current score space.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .matching import canonical_entity


DEFAULT_HISTORY_TOP_K = 3
DEFAULT_HISTORY_WEIGHT = 0.35
DEFAULT_HASH_DIMENSION = 384
_STRIP_KEYS = {"case_id", "trigger_time"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _canonicalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if str(key) in _STRIP_KEYS:
                continue
            items.append((str(key), _canonicalise(child)))
        return {key: child for key, child in items}
    if isinstance(value, list):
        return [_canonicalise(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalise(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def semantic_payload_to_text(payload: Mapping[str, Any] | None) -> str:
    """Serialize a semantic payload into a stable text form for embedding."""

    canonical = _canonicalise(payload or {})
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _dedupe_ids(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _hash_embedding(text: str, *, dimension: int = DEFAULT_HASH_DIMENSION) -> List[float]:
    vector = [0.0] * max(int(dimension), 1)
    tokens = _tokenize(text)
    if not tokens:
        return vector
    features = list(tokens)
    features.extend(f"{left}::{right}" for left, right in zip(tokens, tokens[1:]))
    for index, feature in enumerate(features):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, person=b"NURSEHST").digest()
        slot = int.from_bytes(digest, "big") % len(vector)
        weight = 1.0 if index < len(tokens) else 0.7
        vector[slot] += weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(size):
        l_value = float(left[index])
        r_value = float(right[index])
        dot += l_value * r_value
        left_norm += l_value * l_value
        right_norm += r_value * r_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


@dataclass
class SemanticPayloadEncoder:
    """Lightweight encoder with a sentence-transformers fast path and a fallback."""

    model_name: str = ""
    hash_dimension: int = DEFAULT_HASH_DIMENSION
    backend_name: str = field(init=False)
    embedding_dimension: int = field(init=False)
    _model: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._model = None
        self.backend_name = "hashing-fallback"
        self.embedding_dimension = max(int(self.hash_dimension), 1)
        requested_model = str(self.model_name or os.getenv("NURSE_HISTORY_EMBEDDING_MODEL") or "").strip()
        if requested_model.lower() == "none":
            requested_model = ""
        if not requested_model:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return
        try:
            self._model = SentenceTransformer(requested_model)
            probe = self._model.encode(["nurse history probe"], normalize_embeddings=True)
            vector = probe[0].tolist() if hasattr(probe[0], "tolist") else list(probe[0])
            if vector:
                self.embedding_dimension = len(vector)
                self.backend_name = f"sentence-transformers:{requested_model}"
        except Exception:
            self._model = None
            self.backend_name = "hashing-fallback"
            self.embedding_dimension = max(int(self.hash_dimension), 1)

    def encode_text(self, text: str) -> List[float]:
        if self._model is None:
            return _hash_embedding(text, dimension=self.embedding_dimension)
        try:
            vector = self._model.encode([text], normalize_embeddings=True)[0]
            if hasattr(vector, "tolist"):
                return [float(value) for value in vector.tolist()]
            return [float(value) for value in vector]
        except Exception:
            return _hash_embedding(text, dimension=self.embedding_dimension)

    def encode_payload(self, payload: Mapping[str, Any] | None) -> tuple[List[float], str]:
        text = semantic_payload_to_text(payload)
        return self.encode_text(text), text


_DEFAULT_ENCODER: SemanticPayloadEncoder | None = None


def get_default_encoder() -> SemanticPayloadEncoder:
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = SemanticPayloadEncoder()
    return _DEFAULT_ENCODER


def _canonical_service_set(service_candidates: Iterable[str] | None) -> set[str]:
    services = set()
    for candidate in service_candidates or []:
        service = canonical_entity(str(candidate))
        if service:
            services.add(service)
    return services


def _extract_candidate_set(payload: Mapping[str, Any] | None, service_candidates: Iterable[str] | None) -> set[str]:
    allowed = _canonical_service_set(service_candidates)
    if allowed:
        return allowed
    if not isinstance(payload, Mapping):
        return set()
    for key in ("service_candidates", "current_ranking"):
        raw = payload.get(key)
        if isinstance(raw, list):
            allowed.update(_canonical_service_set(raw))
    candidate_cards = payload.get("candidate_cards")
    if isinstance(candidate_cards, list):
        for card in candidate_cards:
            if not isinstance(card, Mapping):
                continue
            service = canonical_entity(str(card.get("service") or ""))
            if service:
                allowed.add(service)
    return allowed


def _history_score_map(record: Mapping[str, Any]) -> Dict[str, float]:
    raw_scores = record.get("score_summary")
    if not isinstance(raw_scores, Mapping):
        raw_scores = record.get("scores")
    if not isinstance(raw_scores, Mapping):
        return {}
    scores: Dict[str, float] = {}
    for raw_service, raw_score in raw_scores.items():
        service = canonical_entity(str(raw_service))
        if not service:
            continue
        score = _safe_float(raw_score)
        if math.isfinite(score):
            scores[service] = score
    return scores


def build_history_record(
    *,
    case_id: str,
    dataset: str,
    semantic_payload: Mapping[str, Any] | None,
    score_summary: Mapping[str, Any],
    score_breakdown: Mapping[str, Mapping[str, Any]] | None = None,
    ranked_candidates: Sequence[str] | None = None,
    encoder: SemanticPayloadEncoder | None = None,
) -> Dict[str, Any]:
    """Build a serializable history snapshot for later retrieval."""

    encoder = encoder or get_default_encoder()
    embedding, payload_text = encoder.encode_payload(semantic_payload)
    return {
        "case_id": str(case_id),
        "dataset": str(dataset),
        "semantic_rank_payload": dict(semantic_payload or {}),
        "semantic_payload_text": payload_text,
        "embedding": [float(value) for value in embedding],
        "embedding_backend": encoder.backend_name,
        "embedding_dimension": int(len(embedding)),
        "score_summary": {
            canonical_entity(str(service)): _safe_float(score)
            for service, score in dict(score_summary or {}).items()
            if canonical_entity(str(service))
        },
        "score_breakdown": {
            canonical_entity(str(service)): {
                str(part): _safe_float(value)
                for part, value in dict(parts).items()
            }
            for service, parts in dict(score_breakdown or {}).items()
            if canonical_entity(str(service))
        },
        "ranked_candidates": [str(candidate) for candidate in list(ranked_candidates or [])],
    }


def load_history_records(path: str | os.PathLike[str] | None) -> List[Dict[str, Any]]:
    """Load a history index from JSON or JSONL."""

    if path is None:
        return []
    history_path = Path(path).expanduser()
    if not str(history_path):
        return []
    if not history_path.exists():
        raise FileNotFoundError(f"history index not found: {history_path}")

    if history_path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, Mapping):
                    records.append(dict(payload))
        return records

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        if isinstance(payload.get("records"), list):
            return [dict(item) for item in payload["records"] if isinstance(item, Mapping)]
        if isinstance(payload.get("history_index"), list):
            return [dict(item) for item in payload["history_index"] if isinstance(item, Mapping)]
        return [dict(payload)]
    return []


def build_corrected_history_record(
    *,
    case_id: str,
    dataset: str,
    semantic_payload: Mapping[str, Any] | None,
    score_summary: Mapping[str, Any],
    score_breakdown: Mapping[str, Mapping[str, Any]] | None = None,
    ranked_candidates: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    encoder: SemanticPayloadEncoder | None = None,
    correction_margin: float = 1.0,
) -> Dict[str, Any]:
    """Promote benchmark labels to the top of a history record for seeded H(s)."""

    canonical_labels = _dedupe_ids(
        [
            canonical_entity(str(label))
            for label in list(labels or [])
            if canonical_entity(str(label))
        ]
    )
    base_scores = {
        canonical_entity(str(service)): _safe_float(score)
        for service, score in dict(score_summary or {}).items()
        if canonical_entity(str(service))
    }
    corrected_scores = dict(base_scores)
    corrected_breakdown: Dict[str, Dict[str, float]] = {
        canonical_entity(str(service)): {
            str(part): _safe_float(value)
            for part, value in dict(parts or {}).items()
        }
        for service, parts in dict(score_breakdown or {}).items()
        if canonical_entity(str(service))
    }

    max_score = max(corrected_scores.values(), default=0.0)
    margin = max(_safe_float(correction_margin, default=1.0), 1e-6)
    for index, service in enumerate(canonical_labels):
        previous = corrected_scores.get(service, 0.0)
        boosted = max_score + margin + float(len(canonical_labels) - index) * 1e-3
        corrected_scores[service] = boosted
        corrected_breakdown.setdefault(service, {})
        corrected_breakdown[service]["history_bootstrap_correction"] = (
            corrected_breakdown[service].get("history_bootstrap_correction", 0.0) + (boosted - previous)
        )

    corrected_ranked = sorted(corrected_scores, key=lambda service: (-corrected_scores[service], service))
    record = build_history_record(
        case_id=case_id,
        dataset=dataset,
        semantic_payload=semantic_payload,
        score_summary=corrected_scores,
        score_breakdown=corrected_breakdown,
        ranked_candidates=corrected_ranked or ranked_candidates,
        encoder=encoder,
    )
    record["history_correction"] = {
        "labels": canonical_labels,
        "original_top1": str(list(ranked_candidates or [""])[0] if ranked_candidates else ""),
        "corrected_top1": str(corrected_ranked[0] if corrected_ranked else ""),
        "correction_margin": margin,
    }
    return record


@dataclass(frozen=True)
class HistoryPriorResult:
    case_id: str
    dataset: str
    active: bool
    method: str
    embedding_backend: str
    matched_history: List[Dict[str, Any]] = field(default_factory=list)
    score_contribution: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    query_signature: str = ""
    top_k: int = 0
    history_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "active": self.active,
            "method": self.method,
            "embedding_backend": self.embedding_backend,
            "matched_history": list(self.matched_history),
            "score_contribution": dict(self.score_contribution),
            "rationale": self.rationale,
            "query_signature": self.query_signature,
            "top_k": int(self.top_k),
            "history_count": int(self.history_count),
        }


def compute_history_prior(
    *,
    case_id: str,
    dataset: str,
    current_payload: Mapping[str, Any] | None,
    history_index: Iterable[Mapping[str, Any]] | None = None,
    service_candidates: Iterable[str] | None = None,
    top_k: int = DEFAULT_HISTORY_TOP_K,
    history_weight: float = DEFAULT_HISTORY_WEIGHT,
    encoder: SemanticPayloadEncoder | None = None,
) -> HistoryPriorResult:
    """Compute the semantic history prior for the current case.

    The returned score contribution is already normalized by the similarity
    weights of the selected neighbors, then scaled by ``history_weight``.
    """

    encoder = encoder or get_default_encoder()
    checked_history = [dict(item) for item in (history_index or []) if isinstance(item, Mapping)]
    if not checked_history or history_weight <= 0.0:
        return HistoryPriorResult(
            case_id=str(case_id),
            dataset=str(dataset),
            active=False,
            method="semantic_history_similarity",
            embedding_backend=encoder.backend_name,
            rationale="No approved historical semantic payloads were supplied.",
            query_signature=semantic_payload_to_text(current_payload),
            top_k=max(int(top_k), 0),
            history_count=len(checked_history),
        )

    query_embedding, query_text = encoder.encode_payload(current_payload)
    allowed_services = _extract_candidate_set(current_payload, service_candidates)
    similarities: List[Dict[str, Any]] = []

    for item in checked_history:
        item_case_id = str(item.get("case_id") or "")
        item_dataset = str(item.get("dataset") or "")
        if item_case_id and item_case_id == str(case_id):
            continue
        if dataset and item_dataset and item_dataset != str(dataset):
            continue
        payload = item.get("semantic_rank_payload")
        if not isinstance(payload, Mapping):
            payload = item.get("semantic_payload")
        if not isinstance(payload, Mapping):
            payload = {}
        stored_text = str(item.get("semantic_payload_text") or "")
        stored_embedding = item.get("embedding")
        embedding_backend = str(item.get("embedding_backend") or "")
        if not stored_text:
            stored_text = semantic_payload_to_text(payload)
        if not isinstance(stored_embedding, list) or len(stored_embedding) != len(query_embedding):
            if embedding_backend and embedding_backend != encoder.backend_name:
                stored_embedding = encoder.encode_text(stored_text)
            else:
                stored_embedding = encoder.encode_text(stored_text)
        else:
            stored_embedding = [float(value) for value in stored_embedding]

        similarity = max(_cosine_similarity(query_embedding, stored_embedding), 0.0)
        if similarity <= 0.0:
            continue
        similarities.append(
            {
                "case_id": str(item.get("case_id") or ""),
                "dataset": str(item.get("dataset") or ""),
                "similarity": similarity,
                "semantic_payload_text": stored_text,
                "score_summary": _history_score_map(item),
            }
        )

    similarities.sort(key=lambda item: (-item["similarity"], item["case_id"]))
    selected = similarities[: max(int(top_k), 0)]
    if not selected:
        return HistoryPriorResult(
            case_id=str(case_id),
            dataset=str(dataset),
            active=False,
            method="semantic_history_similarity",
            embedding_backend=encoder.backend_name,
            rationale="Historical payload retrieval produced no positive similarity.",
            query_signature=query_text,
            top_k=max(int(top_k), 0),
            history_count=len(checked_history),
        )

    total_similarity = sum(float(item["similarity"]) for item in selected)
    if total_similarity <= 0.0:
        return HistoryPriorResult(
            case_id=str(case_id),
            dataset=str(dataset),
            active=False,
            method="semantic_history_similarity",
            embedding_backend=encoder.backend_name,
            rationale="Historical payload retrieval could not normalize similarity weights.",
            query_signature=query_text,
            top_k=max(int(top_k), 0),
            history_count=len(checked_history),
        )

    contributions: Dict[str, float] = {}
    matched_history: List[Dict[str, Any]] = []
    allowed = allowed_services or set()

    for item in selected:
        similarity_weight = float(item["similarity"]) / total_similarity
        history_scores = dict(item["score_summary"])
        service_contribution: Dict[str, float] = {}
        top_services: List[Dict[str, float]] = []
        for service, historical_score in history_scores.items():
            if allowed and service not in allowed:
                continue
            contribution = float(history_weight) * similarity_weight * float(historical_score)
            if abs(contribution) <= 1e-12:
                continue
            contributions[service] = contributions.get(service, 0.0) + contribution
            service_contribution[service] = contribution
            top_services.append(
                {
                    "service": service,
                    "historical_score": float(historical_score),
                    "weighted_contribution": float(contribution),
                }
            )
        top_services.sort(key=lambda entry: (-abs(entry["weighted_contribution"]), entry["service"]))
        matched_history.append(
            {
                "case_id": item["case_id"],
                "dataset": item["dataset"],
                "similarity": float(item["similarity"]),
                "similarity_weight": float(similarity_weight),
                "score_contribution": service_contribution,
                "top_services": top_services[:8],
            }
        )

    if not contributions:
        return HistoryPriorResult(
            case_id=str(case_id),
            dataset=str(dataset),
            active=False,
            method="semantic_history_similarity",
            embedding_backend=encoder.backend_name,
            matched_history=matched_history,
            rationale="History matches existed, but none of their services were valid current candidates.",
            query_signature=query_text,
            top_k=max(int(top_k), 0),
            history_count=len(checked_history),
        )

    return HistoryPriorResult(
        case_id=str(case_id),
        dataset=str(dataset),
        active=True,
        method="semantic_history_similarity",
        embedding_backend=encoder.backend_name,
        matched_history=matched_history,
        score_contribution={service: float(value) for service, value in contributions.items()},
        rationale="History prior transferred the top semantic neighbors' final service scores.",
        query_signature=query_text,
        top_k=max(int(top_k), 0),
        history_count=len(checked_history),
    )
