---
name: ops-mechanisms
description: Use when a mechanism explanation needs its necessary conditions, contrary evidence, or distinguishing checks examined.
---

# 机制与竞争解释（ops-mechanisms/1）

问题：这个机制有哪些必要条件仍缺失，什么观测能区分替代解释？

前提：正式 mechanism 卡已发布时，读取真实条件表和区别性 request_keys；未发布不伪造完整机制结论。模型只提出问题，不能补数值、阈值或谓词。

工具：`read_node(card_id)` 看 formal decision、necessary/optional/refuting 条件；`measure_signal(request_key)` 在下一轮测一个相关序列；`check_mechanism(card_id)` 对一张正式机制卡用已经取得的证据作补充重检查，不触发新的测量。

判别：必要条件用工具的三值组合；unknown 是没证据或不够，不是正常或否定。所有必要条件为 true 也只是观测支持，不等于根因确认。optional 与 refuting 条件都要保留，不能为了维护 Top-1 删除反证。

后续：支持当前解释时优先找有区分价值的反证/竞争解释；反驳时转查尚可解释症状的机制；unknown 时仅选择能力索引可查的一个缺失条件。两种机制分别读取、分别检查，不存在一次穷举全案的组合工具。

停止：没有有效区别性观测或预算耗尽时留下未决条件。报告区分 formal_decision 与 supplementary_necessary_support；补查不追溯改写正式表、权限、分数或排名。
