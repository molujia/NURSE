# 评分材料的类型边界

- score_lineage_summary：评分摘要，只回答排名、分项、聚合配置及实际详情入口，不是 term 或 pool 明细。
- score_family：以真实 primary_family 标明家族，回答已供代表项、选择链和最终聚合状态；term_details 是入口，未读项仍未提供。
- score_term：该 term 的原计算输入、池内选择及 role_mechanism_ids；池内胜出、家族代表、最终使用分别核对。
- score_propagation：传播汇总及 path_details 入口；0 不等于已证无传播。
- score_path：该路径实供输入和缺口，不补造未读路径。

使用实际 kind 和 primary_family，不按短引用编号或 selected_family_keys 的顺序猜类型。
重复读取摘要不会自动展开子项；read_node(payload)、成对续页、转向其他工具或停止均可自主选择。
详情链接只能支持下一次提问和合法工具参数，不能当成正文事实；responds_to 回应实际交付入口的源卡。
