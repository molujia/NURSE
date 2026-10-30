"""Append-only evaluator-side exposure registry and historical V1 importer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator
from typing_extensions import Self

from nurse_rcl.artifacts import atomic_write_bytes, canonical_json_bytes, sha256_hex
from nurse_rcl.schemas import DatasetName, StrictFrozenModel

EXPOSURE_EVENT_SCHEMA_VERSION: Literal[
    "nurse.v2.exposure-event/1"
] = "nurse.v2.exposure-event/1"
EXPOSURE_IMPORT_AUDIT_SCHEMA_VERSION: Literal[
    "nurse.v2.exposure-import-audit/1"
] = "nurse.v2.exposure-import-audit/1"
RCABENCH_AUTHORIZED_SMOKE_CASE_IDS = (
    "ts0-mysql-container-kill-9t6n24",
    "ts0-mysql-corrupt-jkgn5j",
)
AIOPS25_AUTHORIZED_SMOKE_CASE_IDS = (
    "009be6db-313",
    "0410d710-226",
)

AccessStatus = Literal["confirmed", "not_observed", "unknown"]
ExposureClass = Literal[
    "authorized_smoke",
    "development_evaluation",
    "pipeline_contact",
    "prior_prediction",
]


class ExposureEvent(StrictFrozenModel):
    """One immutable observation that a case entered a development activity."""

    schema_version: Literal["nurse.v2.exposure-event/1"] = (
        EXPOSURE_EVENT_SCHEMA_VERSION
    )
    event_id: str = Field(pattern=r"^exp_[0-9a-f]{24}$")
    dataset: DatasetName
    case_id: str = Field(min_length=1)
    incident_group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    contact_at_unix: int = Field(ge=0)
    label_access: AccessStatus
    prediction_access: AccessStatus
    configuration_use: str = Field(min_length=1)
    responsible_experiment: str = Field(min_length=1)
    exposure_class: ExposureClass
    source_artifact_hashes: tuple[str, ...] = Field(min_length=1)
    source_paths: tuple[str, ...] = Field(min_length=1)
    reasons: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_event(self) -> Self:
        for values, field in (
            (self.source_artifact_hashes, "source_artifact_hashes"),
            (self.source_paths, "source_paths"),
            (self.reasons, "reasons"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be sorted and unique")
        if not all(_is_sha256(value) for value in self.source_artifact_hashes):
            raise ValueError("source_artifact_hashes must contain SHA-256 values")
        if self.event_id != _event_id(
            self.model_dump(mode="json", exclude={"event_id"})
        ):
            raise ValueError("exposure event ID does not match its content")
        return self

    def verify(self) -> bool:
        return self.event_id == _event_id(
            self.model_dump(mode="json", exclude={"event_id"})
        )


class ExposureCaseSummary(StrictFrozenModel):
    """Derived current exposure state; the JSONL event history stays authoritative."""

    dataset: DatasetName
    case_id: str = Field(min_length=1)
    incident_group_id: str = Field(pattern=r"^grp_[0-9a-f]{24}$")
    first_contact_at_unix: int = Field(ge=0)
    label_access: AccessStatus
    prediction_access: AccessStatus
    configuration_uses: tuple[str, ...] = Field(min_length=1)
    responsible_experiments: tuple[str, ...] = Field(min_length=1)
    exposure_classes: tuple[ExposureClass, ...] = Field(min_length=1)
    event_ids: tuple[str, ...] = Field(min_length=1)


def create_exposure_event(
    *,
    dataset: DatasetName,
    case_id: str,
    incident_group_id: str,
    contact_at_unix: int,
    label_access: AccessStatus,
    prediction_access: AccessStatus,
    configuration_use: str,
    responsible_experiment: str,
    exposure_class: ExposureClass,
    source_artifact_hashes: Sequence[str],
    source_paths: Sequence[str],
    reasons: Sequence[str] | None = None,
) -> ExposureEvent:
    """Create a content-addressed exposure event with canonical provenance."""

    reason_values = (
        tuple(reasons)
        if reasons is not None
        else (f"historical_{exposure_class}",)
    )
    unsigned: dict[str, object] = {
        "schema_version": EXPOSURE_EVENT_SCHEMA_VERSION,
        "dataset": dataset,
        "case_id": case_id,
        "incident_group_id": incident_group_id,
        "contact_at_unix": contact_at_unix,
        "label_access": label_access,
        "prediction_access": prediction_access,
        "configuration_use": configuration_use,
        "responsible_experiment": responsible_experiment,
        "exposure_class": exposure_class,
        "source_artifact_hashes": tuple(sorted(set(source_artifact_hashes))),
        "source_paths": tuple(sorted(set(source_paths))),
        "reasons": tuple(sorted(set(reason_values))),
    }
    return ExposureEvent.model_validate(
        {**unsigned, "event_id": _event_id(unsigned)}
    )


def load_exposure_events(path: Path) -> tuple[ExposureEvent, ...]:
    """Load and validate every immutable JSONL event in registry order."""

    registry = Path(path)
    if not registry.exists():
        return ()
    data = registry.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("invalid exposure registry: missing terminal newline")
    events: list[ExposureEvent] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line:
            raise ValueError(
                f"invalid exposure registry at line {line_number}: empty line"
            )
        try:
            event = ExposureEvent.model_validate_json(raw_line)
        except Exception as error:
            raise ValueError(
                f"invalid exposure registry at line {line_number}"
            ) from error
        if not event.verify() or event.event_id in seen_ids:
            raise ValueError(
                f"invalid exposure registry at line {line_number}: "
                "invalid or duplicate event"
            )
        seen_ids.add(event.event_id)
        events.append(event)
    return tuple(events)


def append_exposure_events(
    path: Path,
    events: Sequence[ExposureEvent],
) -> tuple[ExposureEvent, ...]:
    """Atomically append only unseen events while preserving the existing byte prefix."""

    registry = Path(path)
    existing_data = registry.read_bytes() if registry.exists() else b""
    existing = load_exposure_events(registry)
    existing_by_id = {event.event_id: event for event in existing}
    requested_by_id: dict[str, ExposureEvent] = {}
    for event in events:
        if not event.verify():
            raise ValueError("cannot append an invalid exposure event")
        prior = requested_by_id.get(event.event_id)
        if prior is not None and prior != event:
            raise ValueError("conflicting duplicate exposure event ID")
        requested_by_id[event.event_id] = event
    additions = tuple(
        sorted(
            (
                event
                for event_id, event in requested_by_id.items()
                if event_id not in existing_by_id
            ),
            key=_event_sort_key,
        )
    )
    if not additions:
        return existing
    appended = b"".join(
        canonical_json_bytes(event.model_dump(mode="json")) + b"\n"
        for event in additions
    )
    atomic_write_bytes(registry, existing_data + appended)
    loaded = load_exposure_events(registry)
    expected = existing + additions
    if loaded != expected or not registry.read_bytes().startswith(existing_data):
        raise AssertionError("append-only exposure registry verification failed")
    return loaded


def summarize_exposure_events(
    events: Sequence[ExposureEvent],
) -> tuple[ExposureCaseSummary, ...]:
    """Fold immutable events into one case summary without mutating the ledger."""

    by_case: dict[tuple[DatasetName, str], list[ExposureEvent]] = defaultdict(list)
    for event in events:
        if not event.verify():
            raise ValueError("cannot summarize an invalid exposure event")
        by_case[(event.dataset, event.case_id)].append(event)
    summaries: list[ExposureCaseSummary] = []
    for (dataset, case_id), case_events in by_case.items():
        group_ids = {event.incident_group_id for event in case_events}
        if len(group_ids) != 1:
            raise ValueError("one case cannot change incident groups across exposure events")
        summaries.append(
            ExposureCaseSummary(
                dataset=dataset,
                case_id=case_id,
                incident_group_id=next(iter(group_ids)),
                first_contact_at_unix=min(
                    event.contact_at_unix for event in case_events
                ),
                label_access=_merge_access(
                    event.label_access for event in case_events
                ),
                prediction_access=_merge_access(
                    event.prediction_access for event in case_events
                ),
                configuration_uses=tuple(
                    sorted({event.configuration_use for event in case_events})
                ),
                responsible_experiments=tuple(
                    sorted({event.responsible_experiment for event in case_events})
                ),
                exposure_classes=tuple(
                    sorted({event.exposure_class for event in case_events})
                ),
                event_ids=tuple(sorted(event.event_id for event in case_events)),
            )
        )
    return tuple(sorted(summaries, key=lambda item: (item.dataset, item.case_id)))


def discover_prior_v1_exposures(
    runs_root: Path,
    *,
    group_ids_by_dataset: Mapping[str, Mapping[str, str]],
) -> tuple[ExposureEvent, ...]:
    """Discover every case contact under historical V1 runs conservatively."""

    root = Path(runs_root).resolve(strict=True)
    events: list[ExposureEvent] = []
    for dataset in cast(tuple[DatasetName, ...], ("rcabench", "aiops25")):
        dataset_root = root / dataset
        if not dataset_root.is_dir() or dataset_root.is_symlink():
            raise FileNotFoundError(f"missing historical run root for {dataset}")
        group_index = group_ids_by_dataset.get(dataset)
        if group_index is None:
            raise ValueError(f"missing incident-group index for {dataset}")
        for run_root in sorted(dataset_root.iterdir(), key=lambda path: path.name):
            if not run_root.is_dir() or run_root.is_symlink():
                continue
            case_ids = _case_contact_ids(run_root)
            manifest_path = run_root / "run_manifest.json"
            if case_ids and not manifest_path.is_file():
                raise ValueError(
                    f"{run_root.name}: case contacts without a run manifest"
                )
            if not manifest_path.is_file():
                continue
            manifest, manifest_hash = _load_run_manifest(
                manifest_path,
                dataset=dataset,
                run_directory_name=run_root.name,
            )
            prediction_ids = _prediction_case_ids(run_root)
            if not prediction_ids.issubset(case_ids):
                raise ValueError(
                    f"{run_root.name}: prediction exists without a case contact"
                )
            evaluated = _evaluated_case_sources(run_root, root=root)
            if not set(evaluated).issubset(case_ids):
                raise ValueError(
                    f"{run_root.name}: evaluated case has no case contact"
                )
            configuration_use = _configuration_use(manifest)
            contact_at_unix = _required_nonnegative_int(
                manifest.get("created_at_unix"),
                field="created_at_unix",
                context=run_root.name,
            )
            run_id = _required_string(
                manifest.get("run_id"),
                field="run_id",
                context=run_root.name,
            )
            run_relative = manifest_path.relative_to(root).as_posix()
            for case_id in sorted(case_ids):
                group_id = group_index.get(case_id)
                if group_id is None:
                    raise ValueError(
                        f"{dataset}/{case_id}: outside the incident-group manifest"
                    )
                exposure_class = _exposure_class(
                    dataset=dataset,
                    case_id=case_id,
                    evaluated=case_id in evaluated,
                    has_prediction=case_id in prediction_ids,
                )
                label_access = _label_access(
                    dataset=dataset,
                    evaluated=case_id in evaluated,
                    exposure_class=exposure_class,
                )
                prediction_access = _prediction_access(
                    evaluated=case_id in evaluated,
                    exposure_class=exposure_class,
                    has_prediction=case_id in prediction_ids,
                )
                source_paths = [run_relative]
                source_hashes = [manifest_hash]
                reasons = [f"historical_{exposure_class}"]
                if dataset == "rcabench":
                    reasons.append("answer_bearing_case_identity_observed")
                evaluation_source = evaluated.get(case_id)
                if evaluation_source is not None:
                    paths, hashes = evaluation_source
                    source_paths.extend(paths)
                    source_hashes.extend(hashes)
                events.append(
                    create_exposure_event(
                        dataset=dataset,
                        case_id=case_id,
                        incident_group_id=group_id,
                        contact_at_unix=contact_at_unix,
                        label_access=label_access,
                        prediction_access=prediction_access,
                        configuration_use=configuration_use,
                        responsible_experiment=run_id,
                        exposure_class=exposure_class,
                        source_artifact_hashes=source_hashes,
                        source_paths=source_paths,
                        reasons=reasons,
                    )
                )
    return tuple(sorted(events, key=_event_sort_key))


def load_incident_group_index(
    path: Path,
    *,
    dataset: DatasetName,
) -> tuple[dict[str, str], str]:
    """Load a verified incident-group manifest into a case-to-group index."""

    payload = _load_json_object(path)
    if payload.get("dataset") != dataset:
        raise ValueError("incident-group dataset mismatch")
    manifest_hash = payload.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or not _is_sha256(manifest_hash):
        raise ValueError("incident-group manifest has no valid content hash")
    unsigned = dict(payload)
    del unsigned["manifest_sha256"]
    if manifest_hash != sha256_hex(canonical_json_bytes(unsigned)):
        raise ValueError("incident-group manifest content hash verification failed")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("incident-group manifest groups must be an array")
    index: dict[str, str] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ValueError("incident group must be an object")
        group_id = raw_group.get("group_id")
        case_ids = raw_group.get("case_ids")
        if (
            not isinstance(group_id, str)
            or not group_id.startswith("grp_")
            or not isinstance(case_ids, list)
        ):
            raise ValueError("incident group has invalid identity fields")
        for case_id in case_ids:
            if not isinstance(case_id, str) or not case_id or case_id in index:
                raise ValueError("incident-group case identities must be unique")
            index[case_id] = group_id
    if payload.get("case_count") != len(index):
        raise ValueError("incident-group case count mismatch")
    return index, manifest_hash


def build_exposure_import_audit(
    events: Sequence[ExposureEvent],
    *,
    registry_path: Path,
    group_manifest_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build compact evidence that every discovered contact entered the ledger."""

    summaries = summarize_exposure_events(events)
    registry_bytes = Path(registry_path).read_bytes()
    datasets: dict[str, object] = {}
    for dataset in cast(tuple[DatasetName, ...], ("rcabench", "aiops25")):
        dataset_events = tuple(event for event in events if event.dataset == dataset)
        dataset_summaries = tuple(
            summary for summary in summaries if summary.dataset == dataset
        )
        datasets[dataset] = {
            "event_count": len(dataset_events),
            "unique_exposed_cases": len(dataset_summaries),
            "unique_exposed_groups": len(
                {summary.incident_group_id for summary in dataset_summaries}
            ),
            "development_evaluation_cases": sum(
                "development_evaluation" in summary.exposure_classes
                for summary in dataset_summaries
            ),
            "authorized_smoke_cases": sum(
                "authorized_smoke" in summary.exposure_classes
                for summary in dataset_summaries
            ),
            "contact_without_prediction_cases": sum(
                summary.prediction_access == "not_observed"
                for summary in dataset_summaries
            ),
            "confirmed_label_access_cases": sum(
                summary.label_access == "confirmed" for summary in dataset_summaries
            ),
            "incident_group_manifest_sha256": group_manifest_hashes[dataset],
        }
    registered_ids = {
        dataset: {
            summary.case_id for summary in summaries if summary.dataset == dataset
        }
        for dataset in ("rcabench", "aiops25")
    }
    smoke_ids = {
        "rcabench": list(RCABENCH_AUTHORIZED_SMOKE_CASE_IDS),
        "aiops25": list(AIOPS25_AUTHORIZED_SMOKE_CASE_IDS),
    }
    missing_smoke = {
        dataset: sorted(set(case_ids).difference(registered_ids[dataset]))
        for dataset, case_ids in smoke_ids.items()
    }
    if any(missing_smoke.values()):
        raise ValueError(f"authorized smoke cases are missing: {missing_smoke}")
    content = [event.model_dump(mode="json") for event in events]
    return {
        "schema_version": EXPOSURE_IMPORT_AUDIT_SCHEMA_VERSION,
        "append_only_registry_verified": load_exposure_events(registry_path)
        == tuple(events),
        "registry_path": Path(registry_path).as_posix(),
        "registry_event_count": len(events),
        "registry_artifact_sha256": sha256_hex(registry_bytes),
        "registry_content_sha256": sha256_hex(canonical_json_bytes(content)),
        "authorized_smoke_cases_registered": smoke_ids,
        "unclassified_case_contacts": 0,
        "datasets": datasets,
    }


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a content artifact once and permit only byte-identical replay."""

    output = Path(path)
    data = canonical_json_bytes(dict(payload))
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError("immutable exposure audit already exists")
        return
    atomic_write_bytes(output, data)


def _event_id(unsigned: object) -> str:
    return f"exp_{sha256_hex(canonical_json_bytes(unsigned))[:24]}"


def _event_sort_key(event: ExposureEvent) -> tuple[object, ...]:
    return (
        event.contact_at_unix,
        event.dataset,
        event.case_id,
        event.responsible_experiment,
        event.event_id,
    )


def _merge_access(values: Iterable[AccessStatus]) -> AccessStatus:
    value_set = set(values)
    if "confirmed" in value_set:
        return "confirmed"
    if "unknown" in value_set:
        return "unknown"
    return "not_observed"


def _case_contact_ids(run_root: Path) -> set[str]:
    cases_root = run_root / "cases"
    if not cases_root.exists():
        return set()
    if not cases_root.is_dir() or cases_root.is_symlink():
        raise ValueError(f"{run_root.name}: invalid cases directory")
    case_ids: set[str] = set()
    for path in cases_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{run_root.name}: invalid case contact entry {path.name}")
        case_ids.add(path.name)
    return case_ids


def _prediction_case_ids(run_root: Path) -> set[str]:
    predictions_root = run_root / "predictions"
    if not predictions_root.exists():
        return set()
    if not predictions_root.is_dir() or predictions_root.is_symlink():
        raise ValueError(f"{run_root.name}: invalid predictions directory")
    predictions: set[str] = set()
    for path in predictions_root.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            continue
        predictions.add(path.stem)
    return predictions


def _load_run_manifest(
    path: Path,
    *,
    dataset: DatasetName,
    run_directory_name: str,
) -> tuple[dict[str, object], str]:
    envelope = _load_json_object(path)
    manifest = envelope.get("manifest")
    manifest_hash = envelope.get("manifest_hash")
    if not isinstance(manifest, dict) or not all(
        isinstance(key, str) for key in manifest
    ):
        raise ValueError(f"{run_directory_name}: invalid run manifest payload")
    if not isinstance(manifest_hash, str) or not _is_sha256(manifest_hash):
        raise ValueError(f"{run_directory_name}: invalid run manifest hash")
    if manifest_hash != sha256_hex(canonical_json_bytes(manifest)):
        raise ValueError(f"{run_directory_name}: run manifest hash mismatch")
    if manifest.get("dataset") != dataset:
        raise ValueError(f"{run_directory_name}: run dataset mismatch")
    return cast(dict[str, object], manifest), manifest_hash


def _configuration_use(manifest: Mapping[str, object]) -> str:
    namespace = _required_string(
        manifest.get("method_namespace"),
        field="method_namespace",
        context="run manifest",
    )
    version = _required_string(
        manifest.get("method_version"),
        field="method_version",
        context="run manifest",
    )
    fingerprints = manifest.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("run manifest has invalid fingerprints")
    configuration_hash = fingerprints.get("configuration")
    if not isinstance(configuration_hash, str) or not _is_sha256(configuration_hash):
        raise ValueError("run manifest has invalid configuration fingerprint")
    return f"{namespace}/{version}@{configuration_hash}"


def _evaluated_case_sources(
    run_root: Path,
    *,
    root: Path,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    evaluation_root = run_root / "evaluation"
    if not evaluation_root.is_dir() or evaluation_root.is_symlink():
        return {}
    sources: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for selection_path in sorted(evaluation_root.glob("*/selection.json")):
        if selection_path.is_symlink() or not selection_path.is_file():
            continue
        receipt_path = selection_path.parent / "partial_evaluation_receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            continue
        selection = _load_json_object(selection_path)
        selection_hash = _verified_embedded_hash(
            selection,
            hash_field="selection_hash",
            context=selection_path.as_posix(),
        )
        receipt = _load_json_object(receipt_path)
        receipt_hash = _verified_embedded_hash(
            receipt,
            hash_field="receipt_hash",
            context=receipt_path.as_posix(),
        )
        if receipt.get("selection_hash") != selection_hash:
            raise ValueError("evaluation receipt does not bind its selection")
        raw_cases = selection.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("evaluation selection cases must be an array")
        case_ids = []
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError("evaluation selection case must be an object")
            case_id = raw_case.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("evaluation selection has invalid case ID")
            case_ids.append(case_id)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation selection case IDs must be unique")
        if receipt.get("sample_size") != len(case_ids):
            raise ValueError("evaluation receipt sample size mismatch")
        paths = tuple(
            sorted(
                (
                    selection_path.relative_to(root).as_posix(),
                    receipt_path.relative_to(root).as_posix(),
                )
            )
        )
        hashes = tuple(sorted((selection_hash, receipt_hash)))
        for case_id in case_ids:
            prior = sources.get(case_id)
            if prior is not None and prior != (paths, hashes):
                raise ValueError("case appears in multiple evaluated selections")
            sources[case_id] = (paths, hashes)
    return sources


def _verified_embedded_hash(
    payload: Mapping[str, object],
    *,
    hash_field: str,
    context: str,
) -> str:
    value = payload.get(hash_field)
    if not isinstance(value, str) or not _is_sha256(value):
        raise ValueError(f"{context}: invalid {hash_field}")
    unsigned = dict(payload)
    del unsigned[hash_field]
    if value != sha256_hex(canonical_json_bytes(unsigned)):
        raise ValueError(f"{context}: {hash_field} mismatch")
    return value


def _exposure_class(
    *,
    dataset: DatasetName,
    case_id: str,
    evaluated: bool,
    has_prediction: bool,
) -> ExposureClass:
    if evaluated:
        return "development_evaluation"
    smoke_ids = (
        RCABENCH_AUTHORIZED_SMOKE_CASE_IDS
        if dataset == "rcabench"
        else AIOPS25_AUTHORIZED_SMOKE_CASE_IDS
    )
    if case_id in smoke_ids:
        return "authorized_smoke"
    if has_prediction:
        return "prior_prediction"
    return "pipeline_contact"


def _label_access(
    *,
    dataset: DatasetName,
    evaluated: bool,
    exposure_class: ExposureClass,
) -> AccessStatus:
    if evaluated or dataset == "rcabench":
        return "confirmed"
    if exposure_class == "authorized_smoke":
        return "unknown"
    return "not_observed"


def _prediction_access(
    *,
    evaluated: bool,
    exposure_class: ExposureClass,
    has_prediction: bool,
) -> AccessStatus:
    if evaluated or (exposure_class == "authorized_smoke" and has_prediction):
        return "confirmed"
    if has_prediction:
        return "unknown"
    return "not_observed"


def _required_string(value: object, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: invalid {field}")
    return value


def _required_nonnegative_int(value: object, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: invalid {field}")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import historical exposure events")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--rcabench-groups", type=Path, required=True)
    parser.add_argument("--aiops25-groups", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rcabench_groups, rcabench_group_hash = load_incident_group_index(
        args.rcabench_groups,
        dataset="rcabench",
    )
    aiops25_groups, aiops25_group_hash = load_incident_group_index(
        args.aiops25_groups,
        dataset="aiops25",
    )
    discovered = discover_prior_v1_exposures(
        args.runs_root,
        group_ids_by_dataset={
            "rcabench": rcabench_groups,
            "aiops25": aiops25_groups,
        },
    )
    registry = append_exposure_events(args.registry_output, discovered)
    audit = build_exposure_import_audit(
        registry,
        registry_path=args.registry_output,
        group_manifest_hashes={
            "rcabench": rcabench_group_hash,
            "aiops25": aiops25_group_hash,
        },
    )
    write_immutable_json(args.audit_output, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
