# 依赖

问题：哪个真实 caller→callee 关系能解释症状相连？仅按真实端点调用 inspect_dependency(caller_service_key, callee_service_key, cursor)。
反馈：分别保留双方覆盖、症状、调用方向和缺自耗时。caller 感知延迟不是 callee 执行慢；左端交付不等于双端都核验。支持可继续核对时序／机制，反证转向其他竞争路径，unknown 可查已有对应端点记录。
停止：缺真实边或没有更多可用观测时保留不可确认，不伪造边／干预。
