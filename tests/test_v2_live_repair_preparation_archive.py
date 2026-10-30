"""Complete small-case persistence; no network, labels, or cached authority."""

import importlib
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from tests.live_repair_case_fixtures import make_case_pair


def api():
    name = "nurse_rcl.v2.live.repair_preparation_archive"
    assert importlib.util.find_spec(name), "missing complete preparation archive"
    return importlib.import_module(name)


def setup(tmp_path, dataset):
    module = api()
    pair = make_case_pair(tmp_path, dataset)
    key = module.preparation_key(
        raw_arguments=pair["raw_arguments"],
        input_sha256=pair["owner_stage_args"]["expected_input_sha256"],
        code_root=Path(__file__).resolve().parents[1],
    )
    parent = tmp_path / "archives"
    parent.mkdir()
    root = parent / key.key_sha256
    receipt_path = pair["owner_state"] / "preparation-receipt.json"
    args = dict(
        archive_root=root,
        expected_key=key,
        raw_arguments=pair["raw_arguments"],
        receipt_path=receipt_path,
    )
    return module, pair, args


def build(module, pair, args):
    return module._build_preparation_archive(
        **args,
        execution_sha256=pair["owner_stage_args"]["execution_sha256"],
        execution_code_sha256=sha256_hex(
            canonical_json_bytes(pair["owner_stage_args"]["code_files"])
        ),
    )


def load_arguments(args, manifest, receipt):
    return dict(
        **args,
        expected_manifest_sha256=sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json"))),
        expected_receipt_sha256=sha256_hex(canonical_json_bytes(receipt.model_dump(mode="json"))),
    )


def equal_fixed_data(cold, warm):
    assert type(cold) is type(warm)
    assert {f.name for f in fields(cold)} == {
        "diagnostics",
        "sources",
        "catalog",
        "graph",
        "clock",
        "error_sources",
        "error_observation_sha256s",
        "source_descriptor_sha256",
        "source_files_sha256",
        "candidate_freeze_sha256",
    }
    for field in fields(cold):
        if field.name != "diagnostics":
            assert getattr(cold, field.name) == getattr(warm, field.name), field.name
    assert cold.diagnostics.dataset == warm.diagnostics.dataset
    assert cold.diagnostics.signals == warm.diagnostics.signals
    assert cold.diagnostics.errors == warm.diagnostics.errors
    left, right = cold.diagnostics.legacy, warm.diagnostics.legacy
    assert type(left) is type(right)
    for field in fields(left):
        if field.name != "sources":
            assert getattr(left, field.name) == getattr(right, field.name), field.name
    assert type(left.sources) is type(right.sources)
    for field in fields(left.sources):
        if field.name != "tables":
            assert getattr(left.sources, field.name) == getattr(right.sources, field.name)
    assert len(left.sources.tables) == len(right.sources.tables)
    for a, b in zip(left.sources.tables, right.sources.tables, strict=True):
        assert a.audit == b.audit and a.missing_columns == b.missing_columns
        assert a.table.equals(b.table, check_metadata=True)
    assert left.sources.descriptor() == right.sources.descriptor()


@pytest.mark.parametrize("dataset", ["rcabench", "aiops25"])
def test_complete_cold_warm_equivalence_without_rebuilding(tmp_path, monkeypatch, dataset):
    module, pair, args = setup(tmp_path, dataset)
    from nurse_rcl.v2.live import repair_raw_audit as raw

    calls = []
    actual = getattr(raw, f"assemble_{dataset}_repair_diagnostics")

    def counted(*values, **kwargs):
        calls.append(dataset)
        return actual(*values, **kwargs)

    monkeypatch.setattr(raw, f"assemble_{dataset}_repair_diagnostics", counted)
    cold, manifest, receipt = build(module, pair, args)
    assert calls == [dataset]
    assert not pair["raw_arguments"]["ledger_root"].exists()
    assert receipt.preparation_exits_completed is True
    assert manifest.key == args["expected_key"]
    assert tuple((t.name, t.rows, t.schema_sha256) for t in manifest.tables) == tuple(
        (
            f"tables/{i:06d}.arrow",
            item.table.num_rows,
            sha256_hex(item.table.schema.serialize().to_pybytes()),
        )
        for i, item in enumerate(cold.diagnostics.legacy.sources.tables)
    )
    assert not hasattr(cold, "runtime") and not hasattr(cold, "checks")

    def no_build(*values, **kwargs):
        pytest.fail("warm path rebuilt fixed preparation")

    for name in (
        "assemble_rcabench_repair_diagnostics",
        "assemble_aiops25_repair_diagnostics",
        "_prepared_repair_scope",
        "_fixed_repair_scope",
        "_compile_graph",
        "_replayed_repair_query_catalog",
    ):
        monkeypatch.setattr(raw, name, no_build)
    warm = module._load_preparation_archive(**load_arguments(args, manifest, receipt))
    equal_fixed_data(cold, warm)
    assert warm is not cold and warm.diagnostics is not cold.diagnostics
    assert not pair["raw_arguments"]["ledger_root"].exists()


@pytest.mark.parametrize("damage", ["member", "extra", "missing", "symlink", "receipt", "source"])
def test_corrupt_hit_fails_without_rebuild(tmp_path, monkeypatch, damage):
    module, pair, args = setup(tmp_path, "rcabench")
    _, manifest, receipt = build(module, pair, args)
    bound = load_arguments(args, manifest, receipt)
    root = args["archive_root"]
    member = root / next(m.name for m in manifest.members if m.name.endswith(".arrow"))
    if damage == "member":
        member.write_bytes(b"corrupt")
    elif damage == "extra":
        (root / "unlisted.json").write_bytes(b"{}")
    elif damage == "missing":
        member.unlink()
    elif damage == "symlink":
        target = tmp_path / "other.arrow"
        member.rename(target)
        member.symlink_to(target)
    elif damage == "receipt":
        args["receipt_path"].write_bytes(b"{}")
    else:
        path = pair["raw_arguments"]["source_root"] / "env.json"
        path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(
        module, "_build_preparation_archive", lambda **_: pytest.fail("rebuilt hit")
    )
    with pytest.raises((ValueError, OSError)):
        module._load_preparation_archive(**bound)
    assert root.exists()


def test_rehashing_edited_manifest_does_not_replace_independent_pin(tmp_path):
    module, pair, args = setup(tmp_path, "rcabench")
    _, manifest, receipt = build(module, pair, args)
    bound = load_arguments(args, manifest, receipt)
    path = args["archive_root"] / "manifest.json"
    body = json.loads(path.read_bytes())
    body["build_seconds"] += 1.0
    path.write_bytes(canonical_json_bytes(body))
    with pytest.raises(ValueError, match="pin|digest|manifest"):
        module._load_preparation_archive(**bound)


def test_preparation_exit_failure_never_publishes_reusable_archive(tmp_path, monkeypatch):
    module, pair, args = setup(tmp_path, "rcabench")
    from nurse_rcl.v2.live import repair_raw_audit as raw

    actual = raw._file_pins
    calls = []

    def changed(*values):
        result = actual(*values)
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("controlled preparation exit failure")
        return result

    monkeypatch.setattr(raw, "_file_pins", changed)
    with pytest.raises(ValueError, match="controlled preparation exit"):
        build(module, pair, args)
    assert not args["receipt_path"].exists()
    assert not args["archive_root"].exists()


def test_preparation_key_binds_code_inputs_and_configuration_not_report(tmp_path):
    module, pair, args = setup(tmp_path, "rcabench")
    code = tmp_path / "prep-code"
    original = Path(__file__).resolve().parents[1]
    for name in module.PREPARATION_DEPENDENCIES:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    call = dict(
        raw_arguments=pair["raw_arguments"],
        input_sha256=pair["owner_stage_args"]["expected_input_sha256"],
        code_root=code,
    )
    before = module.preparation_key(**call)
    assert before == args["expected_key"]
    for name in ("repair_case_report.py", "repair_ranking.py"):
        path = code / "src/nurse_rcl/v2/live" / name
        assert str(path.relative_to(code)) not in module.PREPARATION_DEPENDENCIES
        path.write_text("# unrelated method/report change\n")
    assert module.preparation_key(**call) == before
    for name in module.PREPARATION_DEPENDENCIES:
        path = code / name
        saved = path.read_bytes()
        path.write_bytes(saved + b"\n")
        assert module.preparation_key(**call).key_sha256 != before.key_sha256, name
        path.write_bytes(saved)
    assert module.preparation_key(**{**call, "input_sha256": "b" * 64}) != before


def test_existing_destination_or_receipt_cannot_be_overwritten(tmp_path):
    module, pair, args = setup(tmp_path, "rcabench")
    build(module, pair, args)
    before = args["receipt_path"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        build(module, pair, args)
    assert args["receipt_path"].read_bytes() == before


def test_fixed_scope_is_data_only_and_runtime_binding_does_not_prepare(tmp_path, monkeypatch):
    module, pair, args = setup(tmp_path, "rcabench")
    from nurse_rcl.v2.live import repair_raw_audit as raw

    assert hasattr(raw, "_fixed_repair_scope"), "fixed preparation must not construct a runtime"
    assert hasattr(raw, "_bind_preparation_runtime"), "missing fresh runtime binding"
    with raw._fixed_repair_scope(**pair["raw_arguments"]) as (data, checks):
        assert type(data).__name__ == "RepairPreparationData"
        assert not hasattr(data, "runtime")
        checks.raw()
    for name in ("_fixed_repair_scope", "_compile_graph", "_replayed_repair_query_catalog"):
        monkeypatch.setattr(
            raw, name, lambda **_: pytest.fail("runtime binding rebuilt preparation")
        )
    bound = raw._bind_preparation_runtime(
        data,
        store_root=pair["raw_arguments"]["store_root"],
        ledger_root=pair["raw_arguments"]["ledger_root"],
    )
    assert bound.runtime.catalog is data.catalog
    assert bound.runtime.arguments["sources"] is data.sources
    assert not pair["raw_arguments"]["ledger_root"].exists()


@pytest.mark.parametrize("dataset", ["rcabench", "aiops25"])
def test_preparation_dependency_inventory_covers_actual_cold_calls(tmp_path, dataset):
    module, pair, args = setup(tmp_path, dataset)
    code = Path(__file__).resolve().parents[1]
    called = set()
    seen = set()
    prefix = str(code / "src") + "/"

    def trace(frame, event, arg):
        name = frame.f_code.co_filename
        if event == "call" and name.startswith(prefix) and name not in seen:
            seen.add(name)
            called.add(Path(name).relative_to(code).as_posix())

    # Explicit orchestration/diagnostic-only exclusions; no package-prefix bypass.
    excluded = {
        "src/nurse_rcl/v2/live/repair_module_timing.py",
        "src/nurse_rcl/v2/live/repair_timing.py",
        "src/nurse_rcl/v2/live/repair_query_events.py",
        "src/nurse_rcl/v2/live/repair_driver.py",
    }
    sys.setprofile(trace)
    try:
        build(module, pair, args)
    finally:
        sys.setprofile(None)
    assert called - excluded <= set(module.PREPARATION_DEPENDENCIES)


@pytest.mark.parametrize("failure", [False, True])
def test_load_checks_external_members_on_normal_and_exceptional_exits(
    tmp_path, monkeypatch, failure
):
    module, pair, args = setup(tmp_path, "rcabench")
    _, manifest, receipt = build(module, pair, args)
    actual = module.read_preparation_data

    def mutate(root):
        data = actual(root)
        args["receipt_path"].write_bytes(b"{}")
        if failure:
            raise RuntimeError("controlled decoding failure")
        return data

    monkeypatch.setattr(module, "read_preparation_data", mutate)
    with pytest.raises(ValueError, match="receipt.*changed"):
        module._load_preparation_archive(**load_arguments(args, manifest, receipt))
    assert args["archive_root"].exists()


def test_publication_does_not_replace_racing_empty_destination(tmp_path, monkeypatch):
    module, pair, args = setup(tmp_path, "rcabench")
    assert hasattr(module, "_publish_directory"), "missing atomic no-replace publication"
    actual = module._publish_directory

    def raced(source, target):
        target.mkdir()
        actual(source, target)

    monkeypatch.setattr(module, "_publish_directory", raced)
    with pytest.raises(FileExistsError):
        build(module, pair, args)
    assert not args["receipt_path"].exists()
    assert not tuple(args["archive_root"].iterdir())


def test_member_digest_rejects_symlink_swap_before_open(tmp_path, monkeypatch):
    from nurse_rcl.v2.live import repair_preparation_codec as codec

    assert hasattr(codec, "member_digest"), "missing protected streaming member hash"
    target = tmp_path / "member.arrow"
    target.write_bytes(b"original")
    outside = tmp_path / "different.arrow"
    outside.write_bytes(b"different")
    actual = codec.os.open

    def raced(path, flags, *values):
        if Path(path) == target:
            target.unlink()
            target.symlink_to(outside)
        return actual(path, flags, *values)

    monkeypatch.setattr(codec.os, "open", raced)
    with pytest.raises((ValueError, OSError)):
        codec.member_digest(target)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("representation", ["list", "string_subclass"])
def test_data_guard_rejects_header_representation_on_both_exits(tmp_path, failure, representation):
    module, pair, args = setup(tmp_path, "rcabench")
    data, _, _ = build(module, pair, args)
    with pytest.raises(ValueError, match="binding|representation"):  # noqa: SIM117
        with module._guard_preparation_data(data):
            if representation == "list":
                object.__setattr__(
                    data, "error_observation_sha256s", list(data.error_observation_sha256s)
                )
            else:

                class SameString(str):
                    pass

                object.__setattr__(
                    data, "source_descriptor_sha256", SameString(data.source_descriptor_sha256)
                )
            if failure:
                raise RuntimeError("controlled body failure")


@pytest.mark.parametrize("change", ["source", "missing", "window", "cadence", "clock"])
def test_preparation_key_misses_when_fixed_input_changes(tmp_path, change):
    module, pair, args = setup(tmp_path, "rcabench")
    raw = dict(pair["raw_arguments"])
    if change in ("source", "missing"):
        root = raw["source_root"]
        if change == "source":
            path = root / "env.json"
            path.write_bytes(path.read_bytes() + b" ")
        else:
            name = next(n for n, pin in args["expected_key"].source_files if pin is None)
            (root / name).write_bytes(b"new source")
    else:
        from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config

        field = {
            "window": "window_seconds",
            "cadence": "expected_cadence_seconds",
            "clock": "clock_uncertainty_seconds",
        }[change]
        config = raw["configuration"]
        settings = {
            n: getattr(config, n)
            for n in ("window_seconds", "expected_cadence_seconds", "clock_uncertainty_seconds")
        }
        settings[field] += 1
        raw["configuration"] = build_rcabench_adapter_config(**settings)
    assert (
        module.preparation_key(
            raw_arguments=raw,
            input_sha256=args["expected_key"].input_sha256,
            code_root=Path(__file__).resolve().parents[1],
        )
        != args["expected_key"]
    )


@pytest.mark.parametrize("partial", ["receipt_absent", "manifest_absent", "rehashed_receipt"])
def test_unpinned_or_partial_publication_is_not_a_cache_hit(tmp_path, partial):
    module, pair, args = setup(tmp_path, "rcabench")
    _, manifest, receipt = build(module, pair, args)
    bound = load_arguments(args, manifest, receipt)
    if partial == "receipt_absent":
        args["receipt_path"].unlink()
    elif partial == "manifest_absent":
        (args["archive_root"] / "manifest.json").unlink()
    else:
        changed = "b" * 64 if receipt.execution_sha256 != "b" * 64 else "c" * 64
        altered = receipt.model_copy(update={"execution_sha256": changed})
        args["receipt_path"].write_bytes(canonical_json_bytes(altered.model_dump(mode="json")))
    with pytest.raises((ValueError, OSError)):
        module._load_preparation_archive(**bound)
    assert args["archive_root"].exists()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":1}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
    ],
)
def test_archive_json_decoder_rejects_noncanonical_or_overdeep_members(tmp_path, raw):
    from nurse_rcl.v2.live import repair_preparation_codec as codec

    (tmp_path / "input.json").write_bytes(raw)
    with pytest.raises(ValueError):
        codec._read(tmp_path, "input.json")


def test_bad_restore_shape_and_capacity_rejected_before_collection_reads(tmp_path):
    from nurse_rcl.v2.live import repair_preparation_codec as codec

    (tmp_path / "core.json").write_bytes(b'{"dataset":"rcabench"}')
    with pytest.raises(ValueError, match="fields"):
        codec.read_preparation_data(tmp_path)


def test_member_path_cannot_escape_archive():
    from nurse_rcl.v2.live.repair_preparation_artifacts import RepairPreparationMember

    for name in ("a/../outside", "/absolute", "a//empty", "a/./self"):
        with pytest.raises(ValueError):
            RepairPreparationMember(name=name, size_bytes=0, sha256="a" * 64)
