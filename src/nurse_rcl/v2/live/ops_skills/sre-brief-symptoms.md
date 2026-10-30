# 症状

问题：哪个已有症状与异常窗口吻合，哪些竞争解释仍可成立？inspect_candidate 的 focus="symptoms" 保留来源、单位、窗口与已测／仅摘要区别。
前提：新测只用 signals 目录中 bound=true 的真实 request_key 调 measure_signal；已测同源复用不是独立支持。
反馈：true 只支持对应谓词；false 不等于未测，也不自动等于服务正常；unknown 记录覆盖或未测原因。可用精确卡片、机制或合法双候选比较进一步区分，不隐藏反证。
停止：没有能缩小当前未知的注册测量则记录缺口，不编造观测。
