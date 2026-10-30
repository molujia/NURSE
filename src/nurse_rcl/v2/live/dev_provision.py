"""PRIVATE single-case Dev supply. This process must exit before inference starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.aiops25_adapter import build_aiops25_adapter_config
from nurse_rcl.v2.aiops25_inventory import AIOps25InventoryReport, authorize_aiops25_queue
from nurse_rcl.v2.live.aiops25_sources import read_aiops25_live_sources
from nurse_rcl.v2.live.dev_batch import load_dev_membership
from nurse_rcl.v2.live.normal_execution import _metadata, _pinned
from nurse_rcl.v2.live.repair_driver import publish_json, read_json
from nurse_rcl.v2.live.repair_raw_audit import _file_pins
from nurse_rcl.v2.live.repair_stage import RepairCaseInput
from nurse_rcl.v2.live.staged_case import (
    load_staged_rcabench_case,
    read_inference_identity,
    require_plain_path,
)
from nurse_rcl.v2.opaque_ids import (
    inference_manifest_from_evaluator,
    load_evaluator_identity_map,
)
from nurse_rcl.v2.provision_live_case import provision_rcabench_live_case
from nurse_rcl.v2.rcabench_adapter import build_rcabench_adapter_config


def provision_one(
    settings: dict[str, Any],
    *,
    dataset: str,
    opaque_case_id: str,
    output: Path,
) -> dict[str, Any]:
    members = load_dev_membership(Path(settings["split_path"]), settings["split_sha256"])
    if (dataset, opaque_case_id) not in members:
        raise ValueError("requested complete case is outside frozen Dev")
    require_plain_path(output)
    identity = dict(
        dataset=dataset,
        opaque_case_id=opaque_case_id,
        supply_settings_sha256=sha256_hex(canonical_json_bytes(settings)),
    )
    receipt_path = output / "supply.json"
    if receipt_path.exists():
        retained = read_json(receipt_path)
        if retained.get("identity") != identity:
            raise ValueError("Dev supply resume identity differs")
        case = retained["case"]
        RepairCaseInput.model_validate_json(_pinned(Path(case["input_path"]), case["input_sha256"]))
        return retained
    split = _metadata(_pinned(Path(settings["split_path"]), settings["split_sha256"]))
    if split["identity_artifact_sha256"] != settings["evaluator_map_sha256"]:
        raise ValueError("Dev private map differs from frozen split")
    private = Path(settings["evaluator_map_path"])
    _pinned(private, settings["evaluator_map_sha256"])
    mapping = load_evaluator_identity_map(private)
    _pinned(private, settings["evaluator_map_sha256"])
    manifest = read_inference_identity(
        Path(settings["identity_manifest_path"]), settings["identity_artifact_sha256"]
    )
    if inference_manifest_from_evaluator(mapping) != manifest:
        raise ValueError("Dev private and inference identities disagree")
    matches = [
        r.original_case_id
        for r in mapping.records
        if (r.dataset, r.opaque_case_id) == (dataset, opaque_case_id)
    ]
    if len(matches) != 1:
        raise ValueError("Dev opaque case does not resolve uniquely")
    original = matches[0]
    del mapping  # Never forward answer-bearing objects into inference arguments.
    stage_pin, source_id, queue = None, None, None
    configuration: Any
    if dataset == "rcabench":
        configuration = build_rcabench_adapter_config(window_seconds=240)
        staging = output / "staged"
        staging.mkdir(mode=0o700, exist_ok=True)
        stage_receipt = output / "stage-receipt.json"
        if stage_receipt.exists():
            stage = read_json(stage_receipt)
            if stage.get("opaque_case_id") != opaque_case_id:
                raise ValueError("Dev saved stage identity differs")
            stage_pin = stage["stage_sha256"]
        else:
            receipt = provision_rcabench_live_case(
                source_root=Path(settings["rcabench_root"]) / original,
                evaluator_map_path=private,
                expected_evaluator_artifact_sha256=settings["evaluator_map_sha256"],
                inference_manifest_path=Path(settings["identity_manifest_path"]),
                expected_inference_artifact_sha256=settings["identity_artifact_sha256"],
                stage_root=staging,
            )
            stage_pin = receipt.stage_sha256
            publish_json(stage_receipt, dict(opaque_case_id=opaque_case_id, stage_sha256=stage_pin))
        root = staging / opaque_case_id
        source = load_staged_rcabench_case(
            root,
            expected_stage_sha256=stage_pin,
            inference_manifest_path=Path(settings["identity_manifest_path"]),
            expected_inference_artifact_sha256=settings["identity_artifact_sha256"],
            configuration=configuration,
        )
        descriptor = source.descriptor()["descriptor_sha256"]
    else:
        configuration = build_aiops25_adapter_config(window_seconds=480)
        inventory = AIOps25InventoryReport.model_validate_json(
            _pinned(
                Path(settings["inventory_path"]),
                settings["inventory_sha256"],
            )
        )
        queue = authorize_aiops25_queue(inventory, case_ids=(original,), destination="inference")
        root = Path(settings["aiops25_root"]) / original
        source_id = original
        source25 = read_aiops25_live_sources(
            root,
            source_case_id=source_id,
            opaque_case_id=opaque_case_id,
            authorized_queue=queue,
            configuration=configuration,
        )
        descriptor = source25.descriptor()["descriptor_sha256"]
    specification = RepairCaseInput.model_validate(
        dict(
            schema_version="nurse.v2.live-repair-case-input/1",
            dataset=dataset,
            opaque_case_id=opaque_case_id,
            source_root=str(root),
            stage_sha256=stage_pin,
            source_descriptor_sha256=descriptor,
            source_files=_file_pins(root, dataset),
            configuration_sha256=configuration.configuration_sha256,
            source_case_id=source_id,
            authorized_queue=queue,
        )
    )
    path = output / "input.json"
    publish_json(path, specification.model_dump(mode="json"))
    result = dict(
        schema_version="nurse.v2.dev-single-case-supply/1",
        identity=identity,
        case=dict(
            dataset=dataset,
            opaque_case_id=opaque_case_id,
            input_path=str(path),
            input_sha256=sha256_hex(path.read_bytes()),
        ),
    )
    publish_json(receipt_path, result)
    return result
