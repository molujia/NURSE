# 机制解释与区分检查

触发：需解释正式候选何以得到支持，或检查支持与反证是否冲突。先 inspect_candidate/compare_candidates 的 mechanisms focus，看完整条件简报；精确机制已有卡片时直接 read_node，补充复核用 check_mechanism(card_id)。只用已发布合法卡片，不猜参数，不隐式测量。

逐条件保留 necessary/refuting/optional、AND/OR、真值、混合叶子计数、范围及来源。先看 resolution_state：已由决定性已知叶子确定者，不再以“补未知可翻转此条件”为目的；真正未决者再看合法检查是否能消除缺口。仍可检查其他机制、覆盖或替换测量，但说清目的。

OR为真不代表每个叶子真；AND反证为假且有未知，不意味着未知补齐后必会推翻已知假。条件成立、正式接纳、评分使用和因果确认不同。补充条件与formal_decision分开，supplementary_necessary_support不产生新的正式accepted。缺数据时保留未决，不造决定性验证。
