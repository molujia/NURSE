"""Case identity, returned model identity, and item alignment matter for retrieval."""

import json

import httpx
import pytest

from nurse_rcl.v2.live import semantic_history_retrieval as retrieval


class ToyEmbedder:
    def __init__(self):
        self.documents = []

    def embed(self, texts):
        self.documents.extend(texts)
        return [(1.0, 0.0), (4.0, 0.0), (1.0, 1.0), (1.0, -1.0)]


def test_rank_uses_raw_l2_and_stable_ties_without_embedding_ids():
    provider = ToyEmbedder()
    matches = retrieval.rank_history(
        "pressure rose", {"z": "queue full", "b": "disk slow", "a": "latency fell"}, provider
    )
    assert [(x.case_id, x.distance) for x in matches] == [("a", 1.0), ("b", 1.0), ("z", 3.0)]
    assert provider.documents == ["pressure rose", "queue full", "disk slow", "latency fell"]
    assert retrieval.rank_history("pressure rose", {}, provider) == ()


def test_default_cutoff_and_complete_pool_ranking():
    candidates = {f"id-{i}": [float(i), 0] for i in range(10)}
    assert len(retrieval.rank_history_vectors([0, 0], candidates)) == 5
    assert len(retrieval.rank_history_vectors([0, 0], candidates, top_k=10)) == 10
    with pytest.raises(ValueError, match="positive"):
        retrieval.rank_history_vectors([0, 0], candidates, top_k=0)
    with pytest.raises(retrieval.EmbeddingError, match="dimensions"):
        retrieval.rank_history_vectors([0, 0], {"short": [0]})


def install_transport(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(retrieval.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setenv("ARK_API_KEY", "test-only")


def test_endpoint_id_still_requires_pinned_model_and_restores_item_order(monkeypatch):
    monkeypatch.setenv("ARK_EMBEDDING_ENDPOINT_ID", "ep-deployment-for-text")

    def handler(request):
        assert str(request.url) == retrieval.EMBEDDINGS_URL
        assert request.headers["Authorization"] == "Bearer test-only"
        assert json.loads(request.content) == {
            "model": "ep-deployment-for-text",
            "input": ["source one", "source two"],
            "encoding_format": "float",
        }
        return httpx.Response(200, headers={"x-request-id": "req-123"}, json={
            "model": retrieval.TEXT_MODEL_ID,
            "data": [{"index": 1, "embedding": [-0.2]}, {"index": 0, "embedding": [0.8]}],
        })

    install_transport(monkeypatch, handler)
    provider = retrieval.ArkTextEmbedder(retrieval.TEXT_MODEL_ID)
    assert provider.embed(["source one", "source two"]) == ((0.8,), (-0.2,))
    assert provider.receipts == (
        retrieval.EmbeddingReceipt(retrieval.TEXT_MODEL_ID, 2, 1, "req-123"),
    )
    assert provider.embed([]) == ()
    assert provider.receipts == ()


@pytest.mark.parametrize("model, rows, reason", [
    ("ep-other", [{"index": 0, "embedding": [1]}], "required model"),
    (retrieval.TEXT_MODEL_ID, [], "wrong number"),
    (retrieval.TEXT_MODEL_ID, [{"index": 2, "embedding": [1]}], "invalid embedding index"),
    (retrieval.TEXT_MODEL_ID, [{"index": False, "embedding": [1]}], "invalid embedding index"),
    (retrieval.TEXT_MODEL_ID, [{"index": 0, "embedding": ["1"]}], "coordinate"),
])
def test_rejects_invalid_provider_output(monkeypatch, model, rows, reason):
    monkeypatch.delenv("ARK_EMBEDDING_ENDPOINT_ID", raising=False)
    body = {"model": model, "data": rows}
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    with pytest.raises(retrieval.EmbeddingError, match=reason):
        retrieval.ArkTextEmbedder(retrieval.TEXT_MODEL_ID).embed(["current evidence"])


def test_multimodal_route_embeds_each_document_separately(monkeypatch):
    monkeypatch.delenv("ARK_EMBEDDING_ENDPOINT_ID", raising=False)
    submissions = []

    def handler(request):
        assert str(request.url) == retrieval.MULTIMODAL_URL
        payload = json.loads(request.content)
        assert payload["model"] == retrieval.MODEL_ID
        submissions.append(payload["input"])
        return httpx.Response(200, json={
            "model": retrieval.MODEL_ID, "data": {"embedding": [len(submissions)]},
        })

    install_transport(monkeypatch, handler)
    provider = retrieval.ArkTextEmbedder(retrieval.MODEL_ID)
    assert provider.embed(["first", "second"]) == ((1.0,), (2.0,))
    assert submissions == [[{"type": "text", "text": t}] for t in ("first", "second")]
    assert len(provider.receipts) == 2


def test_error_does_not_echo_incident_or_key(monkeypatch):
    monkeypatch.delenv("ARK_EMBEDDING_ENDPOINT_ID", raising=False)
    install_transport(monkeypatch, lambda _: httpx.Response(403, json={
        "error": {"code": "ModelNotOpen", "message": "current evidence test-only"}
    }))
    with pytest.raises(retrieval.EmbeddingError) as caught:
        retrieval.ArkTextEmbedder().embed(["current evidence"])
    assert str(caught.value) == "Ark embeddings returned HTTP 403 (ModelNotOpen)"
