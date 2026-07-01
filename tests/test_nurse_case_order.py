from __future__ import annotations

import json

from nurse.case_order import select_case_dirs_from_order


def test_select_case_dirs_prefers_case_ids_file_over_manifest(tmp_path, monkeypatch):
    root = tmp_path / "lab"
    case_dirs = []
    for case_id in ("case-a", "case-b", "case-c"):
        path = root / case_id
        path.mkdir(parents=True)
        case_dirs.append(path)

    monkeypatch.setattr("nurse.case_order.case_dirs_for_dataset", lambda dataset, lab_root: list(case_dirs))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"datasets": {"ds": ["case-a", "case-b"]}}), encoding="utf-8")
    case_ids_file = tmp_path / "subset.txt"
    case_ids_file.write_text("case-c\n", encoding="utf-8")

    selected = select_case_dirs_from_order(
        "ds",
        lab_root=root,
        manifest=manifest_path,
        case_ids_file=case_ids_file,
    )

    assert [path.name for path in selected] == ["case-c"]

