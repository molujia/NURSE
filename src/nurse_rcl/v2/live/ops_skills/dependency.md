---
name: ops-dependency
description: Use when caller symptoms might be confused with a callee failure or an unobserved propagation edge.
---

# 依赖与症状方向（ops-dependency/1）

问题：是否确有这条边，症状与候选分别在调用关系的哪端？

前提：使用已发布 dependency 卡和真实已完成的 onset IDs。图边、caller/callee 身份和方向由工具给出，不靠服务名猜测。

工具：`read_node(card_id)` 阅读边、端点、覆盖和原始方向；`compare_timing(source_onset_id,target_onset_id)` 检查源端（callee）到受影响端（caller）的依赖相容症状先后。service_call 与 received_dependency_failure 含义不同。

判别：存在边不等于传播已证实；调用方超时不能确认被调用方故障。工具保留原方向，逆依赖症状检查不是将图边改向。无边、反向、覆盖不足都不能由叙述补成合法关系。

支持与反证：时间与依赖相容只提供支持，不确认因果；false 削弱指定关系，unknown 不构成反驳。必须说明窗口、时钟容差和起点区间。

后续：缺起点时下一轮选择时序或症状技能；若图不支持此路径，保留反证，调查局部机制或另一个有真实边的解释。不得输出替代排名。

停止：所需边或端点不可观测，明确局限；不连接另一案例或线上数据。每轮只执行一个已注册动作。
