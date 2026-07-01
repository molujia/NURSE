from __future__ import annotations

from nurse.aiops25 import aiops25_root


def test_aiops25_root_accepts_public_aio_ps25_layout(tmp_path):
    public_root = tmp_path / "datasets" / "AIOps25" / "aiopschallenge2025-main"
    public_root.mkdir(parents=True)

    assert aiops25_root(tmp_path) == public_root


def test_aiops25_root_prefers_existing_local_chinese_layout(tmp_path):
    public_root = tmp_path / "datasets" / "AIOps25" / "aiopschallenge2025-main"
    local_root = tmp_path / "datasets" / "25挑战赛" / "aiopschallenge2025-main"
    public_root.mkdir(parents=True)
    local_root.mkdir(parents=True)

    assert aiops25_root(tmp_path) == local_root
