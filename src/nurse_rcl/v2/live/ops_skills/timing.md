---
name: ops-timing
description: Use when competing explanations depend on measured anomaly onset ordering and uncertainty.
---

# 时序核验（ops-timing/1）

问题：源端异常能否在观测分辨率下先于目标症状？

前提：两端必须是同一快照内已完成测量给出的 onset ID。工具解析起点区间，模型不得提交时间数值、覆盖或图对象。事件发布顺序与故障观测时间无关。

工具：`read_node(card_id)` 看时序卡；必要时下一轮 `measure_signal(request_key)` 取得一个起点；有两个合法起点后 `compare_timing(source_onset_id,target_onset_id)`。缺一端不准一次隐式测两端。

判别：使用区间而非单点估计，并保留时钟不确定性、真实依赖方向和覆盖。严格先于须由工具返回 true；重叠、缺边、覆盖不足或未测为 unknown，不能按点估计强排。

支持与反证：false 只削弱此次指定先后假设，不排除所有其他机制；true 仍不证明因果。如果补查显示正式 Top-1 晚发，必须明确呈现这个疑点，保留原排名。

后续：unknown 时仅在有可用、可区分的新观测时再查；反驳时转查竞争机制或列出未决项。不得重抽同一测量以改变答案。

停止：数据分辨率或快照能力不足时写清不可判。每轮一个动作；补查不回流核心查询、停止或评分。
