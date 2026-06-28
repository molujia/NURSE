"""Reusable entity matching for service-level RCA metrics."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple


_POD_SUFFIX_RE = re.compile(r"^(?P<service>.+)-(?P<hash>[a-z0-9]{8,})-(?P<suffix>[a-z0-9]{4,8})$")
_OTELDEMO_SERVICE_RE = re.compile(r"oteldemo\.([a-z0-9-]+service)\b")
_SERVICE_ALIASES = {
    "adservice": "ad",
    "cartservice": "cart",
    "checkoutservice": "checkout",
    "currencyservice": "currency",
    "emailservice": "email",
    "paymentservice": "payment",
    "productcatalogservice": "product-catalog",
    "recommendationservice": "recommendation",
    "shippingservice": "shipping",
    "tidb-pd": "pd",
    "tidb-tidb": "tidb",
    "tidb-tikv": "tikv",
}


def _service_alias(value: str) -> str:
    return _SERVICE_ALIASES.get(value, value)


def canonical_entity(value: str) -> str:
    text = str(value or "").strip()
    if "::" in text:
        text = text.split("::", 1)[0].strip()
    if text.startswith("ts-"):
        text = text[3:]
    lowered = text.lower()
    otel_match = _OTELDEMO_SERVICE_RE.search(lowered)
    if otel_match:
        return _service_alias(otel_match.group(1))
    if lowered in _SERVICE_ALIASES:
        return _service_alias(lowered)
    match = _POD_SUFFIX_RE.match(lowered)
    if match:
        return _service_alias(match.group("service"))
    return _service_alias(lowered)


def entity_matches(prediction: str, label: str) -> bool:
    return canonical_entity(prediction) == canonical_entity(label)


def first_match(ranked: Iterable[str], labels: Iterable[str], k: int | None = None) -> Tuple[str, int]:
    label_by_canonical = {canonical_entity(label): str(label) for label in labels}
    limit = None if k is None else int(k)
    for index, item in enumerate(list(ranked)[:limit], start=1):
        normalized = canonical_entity(item)
        if normalized in label_by_canonical:
            return label_by_canonical[normalized], index
    return "", 0


def hit_at(ranked: Iterable[str], labels: Iterable[str], k: int) -> bool:
    return first_match(ranked, labels, k)[1] > 0


