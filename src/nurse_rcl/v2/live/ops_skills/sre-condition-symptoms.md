# 单信号症状与测量范围

触发：当前不确定某信号是否异常、观测是否可用或代表什么范围。inspect_candidate/compare_candidates 使用 symptoms focus 查看已有材料；measure_signal(request_key) 只测注册序列或合法复用，不能任意执行代码。

先说明信号／指标／谓词、窗口、覆盖及origin。persistent_up=false且verified_normal=false只说明该规则不成立，不证明服务健康、无持续异常或排除其他指标。分辨率未知、缺测和正常独立保留；没有合法名称就用原信号引用，不从标签猜名字。

取得core_reuse派生结果时回应其真实内容；原始数列仍未提供可作为剩余缺口，但不能说工具没有执行。复用不增加独立支持数。根据已知与未决选择合法细读、其他信号、机制或停止，不固定路线。
