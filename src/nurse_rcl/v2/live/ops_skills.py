"""Versioned application skills shipped inside the package, not user installs."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from nurse_rcl.artifacts import sha256_hex
from nurse_rcl.v2.live.ops_tools import SKILL_TOOLS

_TRIGGERS = {
    "boundary": "确定窗口、候选域、模态与覆盖缺口时",
    "symptoms": "需要区分摘要异常与实际持续症状时",
    "dependency": "需要核对调用边、症状位置与依赖方向时",
    "timing": "需要用实测起点区间区分竞争解释时",
    "mechanisms": "需要核对必要条件、反证或区别性检查时",
    "scoring": "需要解释已冻结正式分项与并列处理时",
}


def load_skill(skill_id: str) -> str:
    if skill_id not in SKILL_TOOLS:
        raise ValueError("skill ID is not registered")
    node = skill_id.removeprefix("ops-").removesuffix("/1")
    return files("nurse_rcl.v2.live").joinpath("ops_skills").joinpath(node + ".md").read_text(
        encoding="utf-8",
    )


def skill_index() -> tuple[dict[str, Any], ...]:
    return tuple(dict(
        skill_id="ops-" + node + "/1", node=node, when=trigger,
        tools=SKILL_TOOLS["ops-" + node + "/1"],
        content_sha256=sha256_hex(load_skill("ops-" + node + "/1").encode("utf-8")),
    ) for node, trigger in _TRIGGERS.items())


def indexed_skill_index() -> tuple[dict[str, Any], ...]:
    """Compact /2 bundle; the legacy loader and /1 guidance remain unchanged.

    Known exact IDs need no preliminary navigation. Every navigation, read or
    refusal costs an action; read sections are data, not expressions or paths.
    """
    from nurse_rcl.v2.live.ops_indexed_tools import INDEXED_SKILL_TOOLS

    rows = []
    for node, trigger in _TRIGGERS.items():
        skill = "ops-" + node + "/2"
        guidance = files("nurse_rcl.v2.live").joinpath("ops_skills").joinpath(
            "indexed-" + node + ".md",
        ).read_text(encoding="utf-8").strip()
        rows.append(dict(skill_id=skill, node=node, when=trigger,
                         tools=INDEXED_SKILL_TOOLS[skill], guidance=guidance,
                         content_sha256=sha256_hex(guidance.encode("utf-8"))))
    return tuple(rows)
