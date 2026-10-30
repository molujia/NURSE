"""Report storage remains exact, versioned, bounded, and read-only for legacy."""

# ruff: noqa: F811 -- imported pytest fixtures

import importlib
import importlib.util
import json

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import encode_json
from nurse_rcl.v2.live.ops_report import _fact_identity
from nurse_rcl.v2.live.ops_report_schema import REPORT_LIMITS
from tests.test_v2_live_sre_report import (  # noqa: F401
    complete,
    context,
    history,
    prepared_cases,
    report_inputs,
)


def storage():
    name = "nurse_rcl.v2.live.ops_sre_report_storage"
    assert importlib.util.find_spec(name), "report shared storage not implemented"
    return importlib.import_module(name)


def test_storage_roundtrip_binds_all_facts_and_history(context):
    before = canonical_json_bytes(context.model_dump(mode="json"))
    raw = storage().encode_sre_report_storage(context)
    assert context.schema_version == "nurse.v2.sre-report-context/4"
    packet = json.loads(raw)
    assert set(packet) == {"schema_version", "evidence_header", "fact_records", "history"}
    assert packet["schema_version"] == "nurse.v2.sre-report-storage/1"
    assert "facts" not in packet["evidence_header"]
    restored = storage().decode_sre_report_storage(raw)
    assert storage().encode_sre_report_storage(restored) == raw
    assert [_fact_identity(f) for f in restored.evidence.facts] == [
        _fact_identity(f) for f in context.evidence.facts]
    original, recovered = json.loads(before), restored.model_dump(mode="json")
    del original["evidence"]["facts"]
    del recovered["evidence"]["facts"]
    assert canonical_json_bytes(recovered) == canonical_json_bytes(original)
    assert restored.context_sha256 == sha256_hex(raw)
    assert canonical_json_bytes(context.model_dump(mode="json")) == before


@pytest.mark.parametrize("field", ["critical", "mandatory", "observed", "group", "step"])
def test_dangling_context_references_are_not_persistable(context, field):
    value = context.model_dump(mode="json")
    if field == "critical":
        value["critical_fact_refs"].append("f999999")
    elif field == "mandatory":
        value["evidence"]["mandatory_fact_refs"].append("f999999")
    elif field == "observed":
        value["steps"][0]["observed_fact_refs"].append("f999999")
    elif field == "group":
        value["record_groups"][0]["fact_refs"].append("f999999")
    else:
        value["steps"][1]["step_ref"] = value["steps"][0]["step_ref"]
    invalid = type(context).model_validate_json(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        storage().encode_sre_report_storage(invalid)


def test_legacy_context_keeps_old_digest_but_is_not_new_storage(context):
    value = context.model_dump(mode="json")
    value["schema_version"] = "nurse.v2.sre-report-context/1"
    raw = encode_json(value, REPORT_LIMITS)
    legacy = storage().read_legacy_sre_report_context(raw)
    assert legacy.schema_version == value["schema_version"]
    assert legacy.context_sha256 == sha256_hex(raw)
    with pytest.raises(ValueError):
        storage().encode_sre_report_storage(legacy)
    with pytest.raises(ValueError):
        storage().decode_sre_report_storage(raw)


def test_storage_rejects_duplicate_keys_and_noncanonical_envelope(context):
    raw = storage().encode_sre_report_storage(context)
    duplicate = b'{"schema_version":"bad",' + raw[1:]
    for invalid in (duplicate, b" " + raw):
        with pytest.raises(ValueError):
            storage().decode_sre_report_storage(invalid)


def test_storage_rejects_private_content_and_preserves_original(context):
    before = canonical_json_bytes(context.model_dump(mode="json"))
    value = context.model_dump(mode="json")
    value["steps"][0]["planned"]["public_summary"]["text"] = "C:/private/path"
    invalid = type(context).model_validate_json(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        storage().encode_sre_report_storage(invalid)
    assert canonical_json_bytes(context.model_dump(mode="json")) == before
