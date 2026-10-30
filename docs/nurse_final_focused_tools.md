# NURSE 最终有限交付：工具目录

本文描述当前 focused 产品实现：已有证据解释、目标证据直达、最多双卡对照及同源归属。原验收 candidate 已完成一个已揭盲案例的产品回归，确定性 core 完成，调查完成两次决策及两次行动并合法交接，G+ 最终综合完成且报告已生成。该验收只说明单案产品流程贯通，不认证排名、自然语言语义或总体质量；RCL 评分与既有科研结论不由本交付改写。

有限出口排除两个研究专用 helper，使用独立的 580 成员代码 pin；原 candidate 的 582 成员 pin 不适用于本出口。出口的 Linux 软件验证需单独执行，见[快速开始](nurse_final_focused_quickstart.md)。文档不包含真实案例内容或后续 ARK 调用授权。

## 版本与入口

| 项目 | 当前版本 |
| --- | --- |
| 注册配置 | `configs/v2/ops_sre_focused_profile.json`，`nurse.v2.sre-profile/22` |
| 执行 / 工具 profile | `v2-ops-sre/15` / `hermes-sre-tools/25` |
| 决策协议 / 输入 | `nurse.v2.sre-decision-exchange/26` / `nurse.v2.sre-investigation-input/26` |
| 行动菜单 / evidence 目录 | `nurse.v2.sre-action-entries/6` / `nurse.v2.sre-evidence-catalog/1` |
| 同源归属 / 来源关系 | `nurse.v2.sre-source-attribution/1` / `nurse.v2.sre-directed-source-relation/2` |
| focused 组合计划 | `GplusFocusedCompositionPlan`，`nurse.v2.gplus-one-case-composition/2` |
| 最终综合 | `nurse.v2.sre-gplus-report-exchange/1` |

正常产品入口是 `python -m nurse_rcl.v2.live.ops_sre_gplus_case`，从完整 checkout 运行，见[快速开始](nurse_final_focused_quickstart.md)。focused profile 本身是调查阶段：`action_limit=8`、`decision_limit=8`、`report_limit=0`。组合入口在全新确定性 core 之后执行计划明确指定的 1–8 次自主决策上限，允许提前停止，随后进行 1 次 G+ 最终综合。本次验收的两次决策上限只适用于该次回归。`--help` 和 `--check-only` 不授予调用预算。

## 保留的工具功能

所有选择均受当前请求已签发引用、作用域、预算和账本约束。聚合、阅读、测量各自有明确行动，工具没有生产变更接口。

| 工具 | 用途 |
| --- | --- |
| `inspect_case` | 分页查看案例边界和已提供证据概览。 |
| `inspect_candidate` | 按候选与焦点聚合已有症状、机制或评分材料。 |
| `compare_candidates` | 并列查看两个不同候选的已有材料，不把排序当作因果证明。 |
| `inspect_dependency` | 查看有方向的 caller → callee 依赖材料。 |
| `browse_index` | 浏览 candidates、signals、nodes、onsets；focused 新增 evidence 目录。 |
| `read_node` | 按精确卡引用、section 和游标读取原卡内容。 |
| `measure_signal` | 显式执行已注册且绑定的 `request_key`；目录或聚合不会隐式测量。 |
| `compare_timing` | 对照已测量的起始时间；边界或依赖不足时保留 unknown。 |
| `check_mechanism` | 检查 core 已发布机制卡中的条件表，不自行补测。 |
| `inspect_sources` | 完整交付一张或两张已签发症状卡，并返回有界来源关系。 |

## 目标证据直达

使用 `browse_index(kind="evidence", filters=…, cursor=…)`。`filters` 必须包含 `candidate_service_keys`：当前已签发的 1 或 2 个不同候选；可选精确文本键只有 `metric_name`、`signal_family`、`record_kind`，不得加入路径、模糊搜索或自定义查询。文本必须非空，目录校验每项最多 512 字符。多候选目录交错呈现候选的单目标卡，首屏不代表完整覆盖。

每页最多 16 个目录项，且实际可选行动原子总数不超过 16；必要时为本页真实双卡的一个 `inspect_sources` 对照行动预留位置。行动菜单也最多 16 项，最近 evidence 页给出的精确行动优先进入菜单。目录项只提供卡引用、来源与版本等导航元数据，以及已绑定的读取、来源检查或测量入口。

目录查询或翻页各耗 1 次行动；选择目录中的 `read_node`、`inspect_sources` 或 `measure_signal` 再耗另 1 次行动。看到入口不等于已交付详情、已经测量、已经使用证据或已经回答问题。翻页只能原样复用已签发 `next_cursor`，并保持同一组 filters；游标绑定原发布前缀，不能换过滤条件、猜引用或跨请求借用未来引用。

## 最多双卡可比对照

`inspect_sources` 接受 1–2 张不同的已签发症状卡，以及 1–2 个不同的已签发候选；每张卡必须只有一个目标，且目标属于给定候选范围。同一候选的两张卡也可对照，不会自动要求它们来自不同候选。

对照保留 12 个来源语义字段：`metric_name`、`signal_family`、`modality`、`statistic`、`aggregation`、`unit`、`baseline_window`、`fault_window`、`role`、`span_kind`、`denominator`、`coverage`；另检查从记录种类得出的 `record_role`，防止把窗口摘要和 episode 测量视作同种记录。发布角色、core/supplementary 权限不能代填遥测角色，邻卡或记录级单位不能代填字段单位。

每项同时保留 `supplied_value`、`not_supplied`、`supplied_null` 或 `supplied_unknown` 状态及来源字段。两侧都提供实际值时才判定 same/different。候选缺侧、只有一张卡或任一维度未知时不能认证可比；完整双卡中有已知差异时返回 incomparable，其余不足保留 unknown。unknown 不表示不存在、正常或已回答。即使 comparable，也不证明证据独立、因果关系或候选优劣。

## 同源归属与最终解释

对已注册的窗口摘要、episode 和错误摘要，在原记录外附加 `source_attribution`。它按实际注册的 request → source → observation 链、目标、窗口和来源引用绑定；缺失保留缺失，冲突拒绝，不借相邻来源填空。原卡内容、单位、数值、窗口、覆盖率和证据权限仍是解释依据。

同源归属和来源关系只说明原记录的归属与可比约束，均无评分权限；重复路径、摘要、聚合回执和同一来源的不同展示不增加独立证据票。登记的来源元数据随冻结状态保留并校验，最终报告中的同一原记录及其归属可无损恢复，不能用新回执替换原始证据身份。

G+ 解释只可使用已提供且获准的字段、section、step 和完整上游约束；不得强制 Top1、根因或建议。机械绑定检查不认证自然语言语义正确，`semantic_certified=False`；原生说明文本保留原文，不以改写或质量重采样修补。软件测试只能验证这些边界及可回读性，不能替代真实模型说明质量或生产验收。
