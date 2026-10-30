"""Lossless same-record attribution layouts over the original native facts."""

from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.live.ops_sre_brief_sources import (
    bind_retained_source as legacy_bind,
)
from nurse_rcl.v2.live.ops_sre_brief_sources import (
    restore_retained_source as legacy_restore,
)
from nurse_rcl.v2.live.ops_sre_fact_store import NativeFactStore, build_fact_store

SCHEMA = "nurse.v2.sre-retained-source/2"
PREFIX = "source_attribution"


def _attribute_store(refs: list[str], store: NativeFactStore) -> NativeFactStore:
    if any(
        not store.resolve(ref).source_path.startswith(PREFIX + ".")
        and store.resolve(ref).source_path != PREFIX
        for ref in refs
    ):
        raise ValueError("focused attribution path differs")
    return build_fact_store(
        store.resolve(ref).model_copy(
            update={
                "source_path": "payload" + store.resolve(ref).source_path[len(PREFIX) :],
            }
        )
        for ref in refs
    )


def bind_retained_source(source: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    row, refs = source["record"], source["fact_refs"]
    attribute_refs = [ref for ref in refs if store.resolve(ref).source_path.startswith(PREFIX)]
    if not attribute_refs:
        return legacy_bind(source, store)
    if "source_attribution" not in row:
        raise ValueError("focused attribution facts lack original record")
    plain = {key: value for key, value in row.items() if key != PREFIX}
    payload_refs = [ref for ref in refs if ref not in attribute_refs]
    return dict(
        schema_version=SCHEMA,
        record_sha256=sha256_hex(canonical_json_bytes(row)),
        fact_refs=list(refs),
        source=legacy_bind(dict(record=plain, fact_refs=payload_refs), store),
        attribution=legacy_bind(
            dict(record=dict(payload=row[PREFIX]), fact_refs=attribute_refs),
            _attribute_store(attribute_refs, store),
        ),
    )


def restore_retained_source(binding: dict[str, Any], store: NativeFactStore) -> dict[str, Any]:
    if "schema_version" not in binding:
        return legacy_restore(binding, store)
    if set(binding) != {
        "schema_version",
        "record_sha256",
        "fact_refs",
        "source",
        "attribution",
    } or (
        binding["schema_version"] != SCHEMA
        or type(binding["source"]) is not dict
        or type(binding["attribution"]) is not dict
        or "fact_refs" not in binding["source"]
        or "fact_refs" not in binding["attribution"]
    ):
        raise ValueError("focused retained source identity differs")
    refs = binding["fact_refs"]
    plain_refs, attribute_refs = binding["source"]["fact_refs"], binding["attribution"]["fact_refs"]
    if (
        type(refs) is not list
        or type(plain_refs) is not list
        or type(attribute_refs) is not list
        or any(type(ref) is not str for ref in refs + plain_refs + attribute_refs)
        or len(set(refs)) != len(refs)
        or set(plain_refs) & set(attribute_refs)
        or set(refs) != set(plain_refs + attribute_refs)
        or not attribute_refs
        or not set(refs) <= set(store.refs())
        or any(not store.resolve(ref).source_path.startswith(PREFIX) for ref in attribute_refs)
        or any(store.resolve(ref).source_path.startswith(PREFIX) for ref in plain_refs)
    ):
        raise ValueError("focused retained source fact closure differs")
    plain = legacy_restore(binding["source"], store)["record"]
    attribution = legacy_restore(binding["attribution"], _attribute_store(attribute_refs, store))
    if set(attribution["record"]) != {"payload"} or PREFIX in plain:
        raise ValueError("focused attribution layout differs")
    row = dict(plain, source_attribution=attribution["record"]["payload"])
    if sha256_hex(canonical_json_bytes(row)) != binding["record_sha256"]:
        raise ValueError("focused retained source digest differs")
    return dict(record=row, fact_refs=list(refs))
