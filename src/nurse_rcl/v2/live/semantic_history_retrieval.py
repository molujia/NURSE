"""Text embeddings and exact Euclidean retrieval for post-ranking case history."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import httpx

MODEL_ID = "doubao-embedding-vision-251215"
TEXT_MODEL_ID = "doubao-embedding-text-240715"
OPENAI_MODEL_ID = "text-embedding-3-small"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
EmbeddingModel = Literal[
    "doubao-embedding-vision-251215", "doubao-embedding-text-240715", "text-embedding-3-small"
]
EMBEDDINGS_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
MULTIMODAL_URL = EMBEDDINGS_URL + "/multimodal"
DEFAULT_TOP_K = 5
_BATCH_SIZE = 32
_MAX_INPUT_BYTES = 100_000  # Ark Embeddings API limit per input.


class EmbeddingError(ValueError):
    """Configuration, transport, or response error safe to show in a CLI."""


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class HistoryMatch:
    case_id: str
    distance: float


@dataclass(frozen=True)
class EmbeddingReceipt:
    model: str
    input_count: int
    dimensions: int
    request_id: str | None


def _validate_text(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("an evidence document must contain nonempty text")
    if len(text.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise EmbeddingError("an evidence document exceeds the evidence input byte limit")


def _vector(values: object) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or not values:
        raise EmbeddingError("embedding must be a nonempty numeric vector")
    try:
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
            raise EmbeddingError("embedding contains a nonfinite or nonnumeric coordinate")
        return tuple(float(x) for x in values)
    except OverflowError as exc:
        raise EmbeddingError("embedding coordinate exceeds the numeric range") from exc


def _decode_response(response: httpx.Response, count: int, model: str) -> list[tuple[float, ...]]:
    provider = "OpenAI" if model == OPENAI_MODEL_ID else "Ark"
    if not response.is_success:
        # Error messages can echo source documents. Only report the HTTP status and
        # provider's machine-readable error category.
        category = ""
        try:
            error = response.json().get("error", {}).get("code")
            if isinstance(error, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,95}", error):
                category = f" ({error})"
        except (ValueError, AttributeError):
            pass
        raise EmbeddingError(
            f"{provider} embeddings returned HTTP {response.status_code}{category}"
        )
    try:
        body = response.json()
    except ValueError:
        raise EmbeddingError(f"{provider} embeddings returned invalid JSON") from None
    if not isinstance(body, dict) or body.get("model") != model:
        raise EmbeddingError(f"{provider} response did not confirm the required model {model}")
    rows = body.get("data")
    if model == MODEL_ID:
        if not isinstance(rows, dict) or count != 1:
            raise EmbeddingError("Ark multimodal embeddings must return a single dense vector")
        return [_vector(rows.get("embedding"))]
    if not isinstance(rows, list) or len(rows) != count:
        raise EmbeddingError(f"{provider} returned the wrong number of embeddings")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise EmbeddingError(f"{provider} returned an invalid embedding row")
        index = row.get("index")
        if type(index) is not int or not 0 <= index < count or index in by_index:
            raise EmbeddingError(
                f"{provider} returned a missing, duplicate, or invalid embedding index"
            )
        by_index[index] = _vector(row.get("embedding"))
    return [by_index[i] for i in range(count)]


def _request_id(response: httpx.Response) -> str | None:
    value = response.headers.get("x-request-id") or response.headers.get("x-tt-logid")
    return value if value and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) else None


class ArkTextEmbedder:
    """Use the Beijing Ark text embeddings endpoint. Credentials are read on demand.

    Optional ARK_EMBEDDING_ENDPOINT_ID may name a provisioned endpoint. Its
    response must still report the exact pinned model and version.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model_id = model or os.environ.get("ARK_EMBEDDING_MODEL") or MODEL_ID
        if self.model_id not in (MODEL_ID, TEXT_MODEL_ID):
            raise EmbeddingError("unsupported Ark embedding model")
        self.receipts: tuple[EmbeddingReceipt, ...] = ()

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.receipts = ()
        if isinstance(texts, str | bytes):
            raise EmbeddingError("expected a sequence of evidence documents")
        if not texts:
            return ()
        for text in texts:
            _validate_text(text)
        key = os.environ.get("ARK_API_KEY", "").strip()
        if not key:
            raise EmbeddingError("ARK_API_KEY is required for case history embeddings")
        endpoint = os.environ.get("ARK_EMBEDDING_ENDPOINT_ID", "").strip()
        if endpoint and not re.fullmatch(r"ep-[A-Za-z0-9-]+", endpoint):
            raise EmbeddingError("ARK_EMBEDDING_ENDPOINT_ID must be an Ark endpoint ID")
        vectors: list[tuple[float, ...]] = []
        receipts = []
        multimodal = self.model_id == MODEL_ID
        batch_size = 1 if multimodal else _BATCH_SIZE
        try:
            with httpx.Client(timeout=60.0, follow_redirects=False) as client:
                for start in range(0, len(texts), batch_size):
                    batch = list(texts[start : start + batch_size])
                    # The multimodal endpoint embeds all parts as one document.
                    # Separate requests keep the query and each case independent.
                    contents = [{"type": "text", "text": batch[0]}] if multimodal else batch
                    response = client.post(
                        MULTIMODAL_URL if multimodal else EMBEDDINGS_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": endpoint or self.model_id, "input": contents,
                              "encoding_format": "float"},
                    )
                    batch_vectors = _decode_response(response, len(batch), self.model_id)
                    vectors.extend(batch_vectors)
                    receipts.append(EmbeddingReceipt(
                        self.model_id, len(batch), len(batch_vectors[0]), _request_id(response)
                    ))
        except httpx.RequestError as exc:
            raise EmbeddingError(f"Ark embeddings request failed ({type(exc).__name__})") from None
        if len({len(v) for v in vectors}) != 1:
            raise EmbeddingError("Ark returned inconsistent embedding dimensions")
        self.receipts = tuple(receipts)
        return tuple(vectors)


def requested_embedding_model(explicit: str | None = None) -> EmbeddingModel | None:
    """A model name enables embeddings; provider credentials alone do not."""
    for value in (explicit, os.getenv("NURSE_EMBEDDING_MODEL"), os.getenv("ARK_EMBEDDING_MODEL")):
        if value and (model := value.strip()):
            if model not in (MODEL_ID, TEXT_MODEL_ID, OPENAI_MODEL_ID):
                raise EmbeddingError("unsupported history embedding model")
            return cast(EmbeddingModel, model)
    return None


class OpenAITextEmbedder:
    """Embed identical evidence documents using the pinned OpenAI model."""

    model_id = OPENAI_MODEL_ID

    def __init__(self) -> None:
        self.receipts: tuple[EmbeddingReceipt, ...] = ()

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.receipts = ()
        if isinstance(texts, str | bytes):
            raise EmbeddingError("expected a sequence of evidence documents")
        if not texts:
            return ()
        for text in texts:
            _validate_text(text)
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAI history embeddings")
        vectors = []
        receipts = []
        try:
            with httpx.Client(timeout=60.0, follow_redirects=False) as client:
                for start in range(0, len(texts), _BATCH_SIZE):
                    batch = list(texts[start : start + _BATCH_SIZE])
                    response = client.post(
                        OPENAI_EMBEDDINGS_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": self.model_id, "input": batch, "encoding_format": "float"},
                    )
                    result = _decode_response(response, len(batch), self.model_id)
                    if any(len(vector) != 1536 for vector in result):
                        raise EmbeddingError("OpenAI returned unexpected embedding dimensions")
                    vectors.extend(result)
                    receipts.append(EmbeddingReceipt(
                        self.model_id, len(batch), 1536, _request_id(response)
                    ))
        except httpx.RequestError as exc:
            reason = type(exc).__name__
            raise EmbeddingError(f"OpenAI embeddings request failed ({reason})") from None
        self.receipts = tuple(receipts)
        return tuple(vectors)


def make_embedder(model: str) -> ArkTextEmbedder | OpenAITextEmbedder:
    if model == OPENAI_MODEL_ID:
        return OpenAITextEmbedder()
    if model in (MODEL_ID, TEXT_MODEL_ID):
        return ArkTextEmbedder(model)
    raise EmbeddingError("unsupported history embedding model")


def rank_history_vectors(
    query: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[HistoryMatch, ...]:
    """Rank raw vectors in a single embedding space; case IDs break exact ties."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    origin = _vector(query)
    result = []
    for case_id, coordinates in candidates.items():
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("history candidates require nonempty case IDs")
        vector = _vector(coordinates)
        if len(vector) != len(origin):
            raise EmbeddingError("query and history vectors have different dimensions")
        distance = math.dist(origin, vector)
        if not math.isfinite(distance):
            raise EmbeddingError("Euclidean distance exceeds the numeric range")
        result.append(HistoryMatch(case_id, distance))
    result.sort(key=lambda row: (row.distance, row.case_id))
    return tuple(result[:top_k])


def rank_history(
    query_text: str,
    candidates: Mapping[str, str],
    embedder: TextEmbedder,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[HistoryMatch, ...]:
    """Rank eligible histories supplied by the caller using only canonical evidence text.

    The caller selects the eligible case IDs and excludes the query incident.
    IDs are local join keys. Only the text values are sent to the embedder.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    _validate_text(query_text)
    if not candidates:
        return ()
    ids = list(candidates)
    if any(not isinstance(key, str) or not key.strip() for key in ids):
        raise ValueError("history candidates require nonempty case IDs")
    texts = [query_text, *candidates.values()]
    for text in texts[1:]:
        _validate_text(text)
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise EmbeddingError("embedder returned the wrong number of vectors")
    return rank_history_vectors(vectors[0], dict(zip(ids, vectors[1:], strict=True)), top_k=top_k)
