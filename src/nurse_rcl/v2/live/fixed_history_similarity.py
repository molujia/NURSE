"""Fixed, label-free semantic matching over the exact inputs to the case embedder."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from typing import Literal

RULE_ID = "nurse.typed-idf-jaccard/1"
RULE_DISTANCE = "idf-weighted-jaccard-distance"
HistoryMethod = Literal["fixed-rule", "embedding"]
DEFAULT_HISTORY_METHOD: HistoryMethod = "fixed-rule"
Feature = tuple[str, ...]
_PATTERNS = (
    ("window_summary", re.compile(
        r"for a candidate service, the fault-window median of (?P<signal>.+) "
        r"was (?P<state>higher|lower) than its baseline median\."
    )),
    ("episode", re.compile(
        r"a candidate service had a sustained episode of unusually "
        r"(?P<state>high|low) (?P<signal>.+)\."
    )),
    ("mechanism", re.compile(
        r"for a candidate service, the mechanism conditions for (?P<signal>.+) "
        r"were (?P<state>satisfied|rejected|undetermined)\."
    )),
)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


def _features(text: str) -> dict[Feature, float]:
    features: dict[Feature, float] = {}

    def terms(prefix: Feature, tokens: tuple[str, ...]) -> None:
        distinct = set(tokens)
        for token in distinct:
            key = (*prefix, token)
            features[key] = max(features.get(key, 0.0), 1 / len(distinct))

    for raw in text.splitlines():
        line = " ".join(raw.casefold().split())
        if not line:
            continue
        for kind, pattern in _PATTERNS:
            match = pattern.fullmatch(line)
            if match:
                signal = _tokens(match["signal"])
                features[("state", kind, *signal, match["state"])] = 3.0
                features[("signal", kind, *signal)] = 1.0
                terms(("term", kind), signal)
                break
        else:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) == 6 and fields[0] == "detail" and all(fields):
                _, kind, signal_name, state, attribute, value = fields
                scope = ("detail", kind, " ".join(_tokens(signal_name)), state, attribute)
                features[(*scope, value)] = 3.0
                features[scope] = 1.0
                terms(("detail-term", attribute), _tokens(value))
            else:
                # Handle shared prose extensions without silently discarding their information.
                words = _tokens(line)
                if words:
                    features[("sentence", *words)] = 3.0
                    terms(("word",), words)
                    bigrams = tuple(f"{a} {b}" for a, b in pairwise(words))
                    terms(("bigram",), bigrams)
    return features


def fixed_history_scores(query_text: str, candidates: Mapping[str, str]) -> dict[str, float]:
    """Return weighted Jaccard similarities. Only eligible history estimates document frequency.

    Repetition never increases a feature's weight. Exact signed observations carry weight 3;
    unsigned signal identity and partial name matching each carry weight 1 per observation.
    The two documents are compared symmetrically with the same historical reference weights.
    """
    origin = _features(query_text)
    pool = {cid: _features(text) for cid, text in candidates.items()}
    df = Counter(key for row in pool.values() for key in row)
    size = len(pool)
    weights = {key: 1.0 + math.log((size + 1) / (df[key] + 1)) for key in origin.keys() | df}
    scores = {}
    for cid, row in pool.items():
        shared = origin.keys() & row.keys()
        union = origin.keys() | row.keys()
        numerator = math.fsum(weights[k] * min(origin[k], row[k]) for k in shared)
        denominator = math.fsum(weights[k] * max(origin.get(k, 0), row.get(k, 0)) for k in union)
        scores[cid] = numerator / denominator if denominator else 0.0
    return scores
