"""Prepare private review sheets, admit evidence, and replay a history query."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from nurse_rcl.v2.live.fixed_history_similarity import DEFAULT_HISTORY_METHOD
from nurse_rcl.v2.live.verified_history import (
    BENCHMARK,
    HUMAN,
    LIBRARY,
    JsonRecord,
    digest,
    group_id_ok,
    load_document,
    project_core,
    read_library,
    retrieve_document,
    sha,
    validate_library,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _write(path: Path, value: object) -> None:
    # Never overwrite a signed review sheet, an existing library, or a previous run.
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        # Atomic publication with no replacement, including concurrent invocations.
        if os.name == "nt":
            temporary.rename(path)
        else:
            path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _entries(path: Path) -> list[tuple[JsonRecord, JsonRecord]]:
    entries = _json(path)
    if not isinstance(entries, list):
        raise ValueError("candidate manifest must be an array")
    result, seen, groups = [], set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "case_id",
            "group_id",
            "available_at_ns",
            "document_path",
        }:
            raise ValueError("unexpected candidate fields")
        doc = load_document(path.parent / entry["document_path"])
        case_id = entry["case_id"]
        if (
            doc["case_id"] != case_id
            or case_id in seen
            or entry["group_id"] in groups
            or not group_id_ok(entry["group_id"])
            or type(entry["available_at_ns"]) is not int
            or entry["available_at_ns"] < doc["decision_time_ns"]
            or not doc["evidence"]
        ):
            raise ValueError("invalid, duplicated, empty, or prematurely available candidate")
        seen.add(case_id)
        groups.add(entry["group_id"])
        result.append((entry, doc))
    return result


def _identity(entry: JsonRecord, doc: JsonRecord) -> JsonRecord:
    return {
        "case_id": entry["case_id"],
        "group_id": entry["group_id"],
        "available_at_ns": entry["available_at_ns"],
        "document_sha256": digest(doc),
    }


def prepare_review(candidates: Path) -> JsonRecord:
    return {
        "schema_version": "nurse.history-private-reviews/1",
        "reviews": [
            dict(
                _identity(entry, doc),
                status="pending",
                truth_path=None,
                truth_sha256=None,
                truth_pointer=None,
                service=None,
                mechanism=None,
                reviewer=None,
                reviewed_at=None,
            )
            for entry, doc in _entries(candidates)
        ],
    }


def _ground_truth(row: JsonRecord, directory: Path) -> None:
    if not all(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("service", "truth_path", "truth_sha256", "truth_pointer")
    ):
        raise ValueError("missing independent ground-truth record or normalized service")
    # Reviewers may attest to the service even when the incident record
    # does not establish a mechanism. Keep that distinction in the private labels.
    unknown_mechanism = (
        row["status"] == "approved" and "mechanism" in row and row["mechanism"] is None
    )
    if not unknown_mechanism and not (
        isinstance(row.get("mechanism"), str) and row["mechanism"].strip()
    ):
        raise ValueError("provide a normalized mechanism or explicitly review it as null")
    source = (directory / row["truth_path"]).read_bytes()
    if sha(source) != row["truth_sha256"] or not row["truth_pointer"].startswith("/"):
        raise ValueError("ground truth checksum or pointer is invalid")
    node = json.loads(source)
    for token in row["truth_pointer"][1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid JSON Pointer escape")
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= len(node):
                raise ValueError("ground-truth array index does not exist")
            node = node[int(key)]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            raise ValueError("ground-truth pointer does not resolve")
    if not node:
        raise ValueError("empty independent ground-truth record")


def _signed(row: JsonRecord) -> None:
    if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
        raise ValueError("human approval requires the reviewer identity")
    try:
        stamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ValueError(
            "human approval requires a past, timezone-aware review timestamp"
        ) from None


def _admissions(
    candidates: Path, reviews: Path, *, benchmark_reference: bool
) -> tuple[JsonRecord, dict[str, JsonRecord]]:
    sheet = _json(reviews)
    if (
        not isinstance(sheet, dict)
        or sheet.get("schema_version") != "nurse.history-private-reviews/1"
        or not isinstance(sheet.get("reviews"), list)
    ):
        raise ValueError("invalid private review sheet")
    entries = _entries(candidates)
    rows = {row["case_id"]: row for row in sheet["reviews"]}
    if len(rows) != len(sheet["reviews"]) or rows.keys() != {e["case_id"] for e, _ in entries}:
        raise ValueError("review sheet must cover each proposed incident exactly once")
    policy = BENCHMARK if benchmark_reference else HUMAN
    required = "benchmark-reference" if benchmark_reference else "approved"
    cases = []
    used_refs = set()
    for entry, doc in entries:
        row = rows[entry["case_id"]]
        if any(row.get(k) != v for k, v in _identity(entry, doc).items()):
            raise ValueError("case facts or availability changed since review preparation")
        if row["status"] == "rejected":
            continue
        if row["status"] != required:
            raise ValueError("case has not received the required review decision")
        _ground_truth(row, reviews.parent)
        reference = row["truth_sha256"], row["truth_pointer"]
        if reference in used_refs:
            raise ValueError("historical cases share a ground-truth incident record")
        used_refs.add(reference)
        if not benchmark_reference:
            _signed(row)
        cases.append(
            {
                "case_id": entry["case_id"],
                "group_id": entry["group_id"],
                "available_at_ns": entry["available_at_ns"],
                "document": doc,
                "review_receipt_sha256": digest(row),
            }
        )
    library = {"schema_version": LIBRARY, "review_policy": policy, "cases": cases}
    validate_library(library)
    return library, rows


def build_library(
    candidates: Path, reviews: Path, *, benchmark_reference: bool = False
) -> JsonRecord:
    library, _ = _admissions(candidates, reviews, benchmark_reference=benchmark_reference)
    return library


def export_labels(candidates: list[Path], reviews: list[Path]) -> JsonRecord:
    """Export approved labels from independent review sheets for post-ranking scoring."""
    if not candidates or len(candidates) != len(reviews):
        raise ValueError("each candidate manifest requires its corresponding review sheet")
    labels = []
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_references: set[tuple[str, str]] = set()
    for manifest, review in zip(candidates, reviews, strict=True):
        library, rows = _admissions(manifest, review, benchmark_reference=False)
        for case_id, row in rows.items():
            if case_id in seen_ids or row["group_id"] in seen_groups:
                raise ValueError("duplicate case or injection group across review sheets")
            seen_ids.add(case_id)
            seen_groups.add(row["group_id"])
        for case in library["cases"]:
            row = rows[case["case_id"]]
            ref = (row["truth_sha256"], row["truth_pointer"])
            if ref in seen_references:
                raise ValueError("different cases refer to the same ground-truth incident")
            seen_references.add(ref)
            labels.append(
                {
                    "case_id": case["case_id"],
                    "service": row["service"],
                    "mechanism": row["mechanism"],
                }
            )
    return {"labels": labels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project")
    project.add_argument("--core", type=Path, required=True)
    prepare = commands.add_parser("prepare-review")
    build = commands.add_parser("build-library")
    export = commands.add_parser("export-labels")
    export.add_argument("--candidates", type=Path, required=True, action="append")
    export.add_argument("--reviews", type=Path, required=True, action="append")
    for cmd in (prepare, build):
        cmd.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--reviews", type=Path, required=True)
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--library", type=Path, required=True)
    retrieve.add_argument("--query", type=Path, required=True)
    retrieve.add_argument("--library-sha256")
    retrieve.add_argument("--group-id", required=True)
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument(
        "--method", choices=("embedding", "fixed-rule"), default=DEFAULT_HISTORY_METHOD
    )
    for cmd in (build, retrieve):
        cmd.add_argument("--benchmark-reference", action="store_true")
    for cmd in (project, prepare, build, retrieve, export):
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        if args.command == "project":
            result = project_core(_json(args.core))
        elif args.command == "prepare-review":
            result = prepare_review(args.candidates)
        elif args.command == "build-library":
            result = build_library(
                args.candidates, args.reviews, benchmark_reference=args.benchmark_reference
            )
        elif args.command == "export-labels":
            result = export_labels(args.candidates, args.reviews)
        else:
            library, checksum = read_library(args.library, args.library_sha256)
            result = retrieve_document(
                load_document(args.query),
                library,
                checksum,
                group_id=args.group_id,
                top_k=args.top_k,
                method=args.method,
                benchmark_reference=args.benchmark_reference,
            )
        _write(args.out, result)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"history-library: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
