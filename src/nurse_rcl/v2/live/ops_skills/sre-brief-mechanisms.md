# 机制

问题：哪条必要条件、反证或未知能区分候选机制？inspect_candidate／compare_candidates 使用 focus="mechanisms" 先看全部条件简报。
前提：精确问题用可见 card_id 的 read_node(section="payload", cursor=null)；check_mechanism 要求 core-published mechanisms 卡及条件表，不隐式补测。
反馈：保留原判定、因果状态、每条角色和 AND/OR。OR=true 同时有 false／unknown 是合法混合，不能计票确认。若需叶子值，显式展开同一卡片；回执的真实机制与公开问题不同时不能冒称已核验原目标。
停止：必要观测不可得时保留机制未知；无需一次检查就能证明唯一根因，能缩小不确定性也有价值。
