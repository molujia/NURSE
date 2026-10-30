"""Pure source attribution and two-card comparisons, without data or tool execution."""

from copy import deepcopy
from importlib import import_module, util
from types import SimpleNamespace

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex


def api():
    name = "nurse_rcl.v2.live.ops_sre_focused_sources"
    assert util.find_spec(name), "focused source attribution/comparison is missing"
    return import_module(name)


BASELINE = dict(start_unix_ns=10, end_unix_ns=20)
FAULT = dict(start_unix_ns=20, end_unix_ns=30)
DIMENSIONS = {
    "metric_name": "cpu_usage",
    "signal_family": "cpu",
    "modality": "metrics",
    "statistic": "median",
    "aggregation": "instantaneous",
    "unit": "cores",
    "baseline_window": BASELINE,
    "fault_window": FAULT,
    "role": "server",
    "span_kind": "server",
    "denominator": dict(baseline_known_denominator=10, fault_known_denominator=10),
    "coverage": 0.8,
}


def scalar():
    row = dict(
        card_id="card-a",
        node="symptoms",
        tool_version="ops-symptoms/1",
        parameters=dict(request_key="request-a"),
        targets=["service-a"],
        payload=dict(
            kind="paid_episode",
            candidate_service_key="service-a",
            observation_id="observation-a",
            measurement_sha256="d" * 64,
        ),
        source_role="core_published",
        published_by_core=True,
        formal_scoring_use="not_asserted_by_publication",
        score_authority=False,
        unit="cores",
        window_ns=[20, 30],
        coverage=0.8,
        availability="available",
        truth_value="unknown",
        evidence_ids=["evidence-a"],
        source_refs=["a" * 64, "f" * 64],
        limitations=["episode_not_causality"],
    )
    request = dict(
        request_key="request-a",
        source_sha256="a" * 64,
        observation_id="observation-a",
        candidate_service_key="service-a",
        signal_id="signal-a",
        canonical_series_key="series-a",
        needs=[],
    )
    source = dict(
        source_sha256="a" * 64,
        signal_id="signal-a",
        candidate_service_key="service-a",
        metric_name="cpu_usage",
        source_file_sha256s=["f" * 64],
        observation=dict(
            observation_id="observation-a",
            signal_family="cpu",
            modality="metrics",
            unit="cores",
            aggregation="instantaneous",
        ),
        time_contract=dict(baseline_window=BASELINE, fault_window=FAULT),
        query_interval=FAULT,
        quality=dict(coverage_quality=0.8),
    )
    return row, [request], [], [source]


def enriched():
    row, requests, observations, sources = scalar()
    return api().enrich_semantic(row, requests=requests, observations=observations, sources=sources)


def pair(*, same_candidate=False):
    left = enriched()
    right = deepcopy(left)
    right["card_id"] = "card-b"
    right["targets"] = left["targets"] if same_candidate else ["service-b"]
    for row in (left, right):
        row["source_attribution"]["fields"] = {
            name: dict(state="supplied_value", value=deepcopy(value), source_fields=[name])
            for name, value in DIMENSIONS.items()
        }
    return left, right


def comparison(left, right, *, scope=None):
    return api().focused_source_relation(
        [left, right], scope or sorted(set(left["targets"] + right["targets"]))
    )


def test_exact_registry_chain_enriches_the_same_original_record_without_mutation():
    args = scalar()
    before = canonical_json_bytes(args)
    row = api().enrich_semantic(args[0], requests=args[1], observations=args[2], sources=args[3])
    original = {key: value for key, value in row.items() if key != "source_attribution"}
    assert original == args[0] and canonical_json_bytes(args) == before
    attribution = row["source_attribution"]
    assert attribution["schema_version"] == "nurse.v2.sre-source-attribution/1"
    assert attribution["binding"]["status"] == "bound"
    assert attribution["binding"]["request_key"] == "request-a"
    assert attribution["binding"]["observation_id"] == "observation-a"
    assert attribution["binding"]["card_id"] == row["card_id"]
    assert attribution["binding"]["record_sha256"] == sha256_hex(canonical_json_bytes(args[0]))
    fields = attribution["fields"]
    assert fields["metric_name"]["value"] == "cpu_usage"
    assert fields["modality"]["value"] == "metrics"
    assert fields["aggregation"]["value"] == "instantaneous"
    assert fields["fault_window"]["value"] == FAULT
    assert fields["statistic"]["state"] == "not_supplied"
    assert fields["role"]["state"] == "not_supplied"
    assert "core_published" not in canonical_json_bytes(fields).decode()


@pytest.mark.parametrize(
    "change",
    [
        "request",
        "observation",
        "target",
        "source",
        "metric",
        "unit",
        "window",
        "duplicate_request",
    ],
)
def test_supplied_chain_conflicts_are_rejected_without_picking_one_side(change):
    row, requests, observations, sources = scalar()
    if change == "request":
        row["parameters"]["request_key"] = "foreign-request"
    elif change == "observation":
        row["payload"]["observation_id"] = "foreign-observation"
    elif change == "target":
        row["targets"] = ["foreign-service"]
    elif change == "source":
        row["source_refs"] = ["b" * 64]
    elif change == "metric":
        row["payload"]["metric_name"] = "different-metric"
    elif change == "unit":
        row["unit"] = "%"
    elif change == "window":
        row["payload"]["fault_window"] = dict(start_unix_ns=21, end_unix_ns=30)
    else:
        requests.append(dict(requests[0], request_key="second-request"))
        row["parameters"] = {}
    before = canonical_json_bytes((row, requests, observations, sources))
    with pytest.raises(ValueError, match="binding"):
        api().enrich_semantic(row, requests=requests, observations=observations, sources=sources)
    assert canonical_json_bytes((row, requests, observations, sources)) == before


def test_absent_chain_keeps_unknown_fields_without_copying_a_neighbor_source():
    row, _, observations, sources = scalar()
    row["parameters"] = {}
    row["payload"].pop("observation_id")
    row["source_refs"] = []
    result = api().enrich_semantic(row, requests=[], observations=observations, sources=sources)
    fields = result["source_attribution"]["fields"]
    assert result["source_attribution"]["binding"]["status"] == "not_supplied"
    assert fields["metric_name"]["state"] == "not_supplied"
    assert fields["unit"]["value"] == "cores"


@pytest.mark.parametrize("dimension", list(DIMENSIONS))
def test_each_known_comparison_dimension_reports_the_actual_mismatch(dimension):
    left, right = pair()
    cell = right["source_attribution"]["fields"][dimension]
    cell["value"] = "different" if type(cell["value"]) is not dict else dict(changed=True)
    before = canonical_json_bytes((left, right))
    relation = comparison(left, right)
    item = next(item for item in relation["dimensions"] if item["field"] == dimension)
    assert item["status"] == "different" and relation["status"] == "incomparable"
    assert item["left"]["value"] == DIMENSIONS[dimension]
    assert item["right"]["value"] == cell["value"]
    assert canonical_json_bytes((left, right)) == before


@pytest.mark.parametrize(
    "state,value",
    [
        ("not_supplied", None),
        ("supplied_null", None),
        ("supplied_unknown", "unknown"),
    ],
)
def test_missing_null_and_unknown_never_certify_comparability(state, value):
    left, right = pair()
    right["source_attribution"]["fields"]["coverage"] = dict(
        state=state,
        source_fields=["coverage"],
        **({} if state == "not_supplied" else {"value": value}),
    )
    relation = comparison(left, right)
    item = next(item for item in relation["dimensions"] if item["field"] == "coverage")
    assert relation["status"] == "unknown" and item["status"] == "not_supplied"
    assert item["right"]["state"] == state


def test_two_cards_from_the_same_candidate_still_compare_all_dimensions():
    left, right = pair(same_candidate=True)
    assert comparison(left, right)["status"] == "comparable"
    right["source_attribution"]["fields"]["unit"]["value"] = "milliseconds"
    assert comparison(left, right)["status"] == "incomparable"


def test_one_card_or_missing_candidate_is_unknown_without_inventing_the_other_side():
    left, _ = pair()
    relation = api().focused_source_relation([left], ["service-a", "service-b"])
    assert relation["status"] == "unknown"
    assert relation["missing_candidate_service_keys"] == ["service-b"]
    assert all(item["right"]["state"] == "not_supplied" for item in relation["dimensions"])


def error_record():
    row, _, _, _ = scalar()
    row.update(
        parameters=dict(candidate_service_key="service-a", evidence_id="fact-error"),
        unit="observed_record_fraction",
        coverage=None,
        source_refs=["b" * 64, "c" * 64],
    )
    row["payload"] = dict(
        kind="error_summary",
        facts=dict(
            candidate_service_key="service-a",
            modality="trace",
            span_kind="server",
            statistic="observed_span_status_fraction",
            observation_sha256s=["b" * 64, "c" * 64],
            baseline_known_denominator=49,
            fault_known_denominator=0,
            observed_error_count=0,
            baseline_fraction=0.0,
            fault_fraction=None,
            sampling_coverage="unknown",
            interval=FAULT,
        ),
    )
    observations = [
        dict(result_sha256="b" * 64, modality="trace", interval=BASELINE),
        dict(result_sha256="c" * 64, modality="trace", interval=FAULT),
    ]
    return row, observations


def test_error_summary_uses_its_real_facts_and_bound_observation_windows():
    row, observations = error_record()
    original = deepcopy(row)
    result = api().enrich_semantic(row, requests=[], observations=observations, sources=[])
    assert row == original and result["payload"] == original["payload"]
    fields = result["source_attribution"]["fields"]
    assert fields["statistic"]["value"] == "observed_span_status_fraction"
    assert fields["span_kind"]["value"] == "server"
    assert fields["denominator"]["value"] == dict(
        baseline_known_denominator=49, fault_known_denominator=0
    )
    assert fields["coverage"]["state"] == "supplied_unknown"
    assert fields["baseline_window"]["value"] == BASELINE
    assert fields["metric_name"]["state"] == "not_supplied"
    assert fields["role"]["state"] == "not_supplied"


def test_error_binding_conflicts_reject_and_unknown_denominator_is_not_zero():
    row, observations = error_record()
    row["payload"]["facts"]["fault_known_denominator"] = None
    result = api().enrich_semantic(row, requests=[], observations=observations, sources=[])
    denominator = result["source_attribution"]["fields"]["denominator"]
    assert denominator["state"] == "supplied_null"
    assert denominator["value"]["fault_known_denominator"] is None
    row["source_refs"] = ["wrong"]
    with pytest.raises(ValueError, match="binding"):
        api().enrich_semantic(row, requests=[], observations=observations, sources=[])


def test_binding_seals_actual_attribution_and_the_new_policy_version():
    row = enriched()
    params = dict(card_ids=[row["card_id"]], candidate_service_keys=row["targets"])
    pin = api().focused_source_binding("s" * 64, params, [row])
    changed = deepcopy(row)
    changed["source_attribution"]["fields"]["metric_name"]["value"] = "other"
    assert api().focused_source_binding("s" * 64, params, [changed]) != pin
    from nurse_rcl.v2.live.ops_sre_directed_sources import directed_source_binding

    assert pin != directed_source_binding("s" * 64, params, [row])


def test_comparison_does_not_treat_episode_and_window_summary_as_same_record_role():
    left, right = pair()
    right["payload"]["kind"] = "fixed_summary"
    assert comparison(left, right)["status"] == "incomparable"


@pytest.mark.parametrize("change", ["duplicate", "too_many", "wrong_scope"])
def test_comparison_retains_the_two_distinct_card_and_scope_bounds(change):
    left, right = pair()
    rows, scope = [left, right], ["service-a", "service-b"]
    if change == "duplicate":
        rows = [left, left]
    elif change == "too_many":
        rows.append(dict(right, card_id="third"))
    else:
        scope = ["service-a"]
    with pytest.raises(ValueError):
        api().focused_source_relation(rows, scope)


def registry_api():
    module = api()
    assert all(
        hasattr(module, name)
        for name in (
            "source_registry",
            "validate_source_registry",
            "enrich_registered",
        )
    ), "metadata source registry API is missing"
    return module


def registry_data():
    row, requests, _, sources = scalar()
    error, observations = error_record()
    data = SimpleNamespace(
        catalog=SimpleNamespace(requests=requests),
        sources=sources,
        diagnostics=SimpleNamespace(errors=observations),
    )
    return data, row, error


def reseal(registry):
    unsigned = {key: value for key, value in registry.items() if key != "registry_sha256"}
    registry["registry_sha256"] = sha256_hex(canonical_json_bytes(unsigned))


def test_metadata_registry_never_traverses_or_copies_raw_records_points_or_needs():
    module = registry_api()
    data, _, _ = registry_data()

    class Forbidden:
        def __deepcopy__(self, memo):
            pytest.fail("source registry copied raw telemetry")

    data.catalog.requests[0]["needs"] = [Forbidden()]
    data.sources[0]["source_blocks"] = [Forbidden()]
    data.sources[0]["observation"]["baseline_summary"] = Forbidden()
    data.sources[0]["quality"]["source_summary"] = Forbidden()
    data.diagnostics.errors[0]["source_records"] = [Forbidden()]
    data.diagnostics.errors[0]["groups"] = [Forbidden()]
    registry = module.source_registry(data)
    assert registry["schema_version"] == "nurse.v2.sre-source-registry/1"
    assert set(registry) == {
        "schema_version",
        "registry_sha256",
        "requests",
        "observations",
        "sources",
    }
    assert registry["sources"][0]["quality"] == dict(coverage_quality=0.8)
    text = canonical_json_bytes(registry).decode()
    assert not any(
        name in text
        for name in (
            "source_records",
            "source_blocks",
            "source_summary",
            "baseline_summary",
            "needs",
            "groups",
        )
    )
    unsigned = {key: value for key, value in registry.items() if key != "registry_sha256"}
    assert registry["registry_sha256"] == sha256_hex(canonical_json_bytes(unsigned))


def test_registry_reconstructs_exact_scalar_and_error_attribution_without_loaded_data():
    module = registry_api()
    data, row, error = registry_data()
    registry = module.source_registry(data)
    before = canonical_json_bytes(registry)
    for record in (row, error):
        expected = module.enrich_semantic(
            record, data.catalog.requests, data.diagnostics.errors, data.sources
        )
        assert module.enrich_registered(record, registry) == expected
    assert canonical_json_bytes(registry) == before
    assert module.validate_source_registry(registry) == registry


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "extra_root",
        "extra_source",
        "extra_observation",
        "extra_request",
        "chain",
        "duplicate_request",
        "duplicate_source",
        "duplicate_observation",
        "boolean_time",
    ],
)
def test_registry_rejects_unknown_fields_duplicate_identities_or_inconsistent_chain(change):
    module = registry_api()
    data, row, _ = registry_data()
    registry = module.source_registry(data)
    if change == "hash":
        registry["registry_sha256"] = "0" * 64
    elif change == "extra_root":
        registry["raw"] = []
    elif change == "extra_source":
        registry["sources"][0]["source_blocks"] = []
    elif change == "extra_observation":
        registry["observations"][0]["groups"] = []
    elif change == "extra_request":
        registry["requests"][0]["needs"] = []
    elif change == "chain":
        registry["requests"][0]["observation_id"] = "foreign-observation"
    elif change == "boolean_time":
        registry["sources"][0]["query_interval"]["start_unix_ns"] = True
    else:
        key = {
            "duplicate_request": "requests",
            "duplicate_source": "sources",
            "duplicate_observation": "observations",
        }[change]
        registry[key].append(deepcopy(registry[key][0]))
    if change != "hash":
        reseal(registry)
    before = canonical_json_bytes(registry)
    with pytest.raises(ValueError, match="registry"):
        module.enrich_registered(row, registry)
    assert canonical_json_bytes(registry) == before


def test_registry_preserves_two_requests_for_one_source_without_last_write_wins():
    module = registry_api()
    data, row, _ = registry_data()
    data.catalog.requests.append(dict(data.catalog.requests[0], request_key="second-request"))
    registry = module.source_registry(data)
    assert [request["request_key"] for request in registry["requests"]] == [
        "request-a",
        "second-request",
    ]
    assert (
        module.enrich_registered(row, registry)["source_attribution"]["binding"]["request_key"]
        == "request-a"
    )
    row["parameters"] = {}
    with pytest.raises(ValueError, match="binding"):
        module.enrich_registered(row, registry)


@pytest.mark.parametrize("hint", ["request-a", "second-request"])
@pytest.mark.parametrize("saved_registry", [False, True])
def test_prepared_summary_request_hint_cannot_disambiguate_shared_source(hint, saved_registry):
    module = registry_api()
    data, row, _ = registry_data()
    data.catalog.requests.append(dict(data.catalog.requests[0], request_key="second-request"))
    # This is the legacy prepared_summaries publication shape: its payload hint
    # comes from a source->request dictionary, not an executed request. Neither
    # the first nor last registered request can authenticate this summary.
    row.update(
        tool_version="sre-prepared-summary/1",
        source_role="prepared_summary",
        published_by_core=False,
        parameters=dict(candidate_service_key="service-a", observation_id="observation-a"),
    )
    row["payload"].update(kind="prepared_summary", request_key=hint)
    original = canonical_json_bytes((row, data.catalog.requests, data.sources))
    with pytest.raises(ValueError, match="binding.*request.*unique"):
        if saved_registry:
            module.enrich_registered(row, module.source_registry(data))
        else:
            module.enrich_semantic(
                row,
                requests=data.catalog.requests,
                observations=data.diagnostics.errors,
                sources=data.sources,
            )
    assert canonical_json_bytes((row, data.catalog.requests, data.sources)) == original


def test_prepared_summary_keeps_a_unique_registered_source_request_without_rewriting_card():
    module = registry_api()
    data, row, _ = registry_data()
    row.update(
        tool_version="sre-prepared-summary/1",
        source_role="prepared_summary",
        published_by_core=False,
        parameters=dict(candidate_service_key="service-a", observation_id="observation-a"),
    )
    row["payload"].update(kind="prepared_summary", request_key="request-a")
    original = canonical_json_bytes(row)
    for enriched_row in (
        module.enrich_semantic(
            row,
            requests=data.catalog.requests,
            observations=data.diagnostics.errors,
            sources=data.sources,
        ),
        module.enrich_registered(row, module.source_registry(data)),
    ):
        assert enriched_row["source_attribution"]["binding"]["status"] == "bound"
        assert enriched_row["source_attribution"]["binding"]["request_key"] == "request-a"
        assert (
            canonical_json_bytes(
                {key: value for key, value in enriched_row.items() if key != "source_attribution"}
            )
            == original
        )
    assert canonical_json_bytes(row) == original


@pytest.mark.parametrize("kind", ["paid_episode", "supplementary_episode"])
def test_executed_episode_keeps_actual_request_disambiguation_for_shared_source(kind):
    module = registry_api()
    data, row, _ = registry_data()
    data.catalog.requests.append(dict(data.catalog.requests[0], request_key="second-request"))
    row["payload"]["kind"] = kind
    original = canonical_json_bytes(row)
    for enriched_row in (
        module.enrich_semantic(
            row,
            requests=data.catalog.requests,
            observations=data.diagnostics.errors,
            sources=data.sources,
        ),
        module.enrich_registered(row, module.source_registry(data)),
    ):
        assert enriched_row["source_attribution"]["binding"]["status"] == "bound"
        assert enriched_row["source_attribution"]["binding"]["request_key"] == "request-a"
        assert (
            canonical_json_bytes(
                {key: value for key, value in enriched_row.items() if key != "source_attribution"}
            )
            == original
        )
    assert canonical_json_bytes(row) == original


def test_new_comparison_schema_is_explicitly_versioned_and_old_relation_is_unchanged():
    left, right = pair()
    assert comparison(left, right)["schema_version"] == "nurse.v2.sre-directed-source-relation/2"
