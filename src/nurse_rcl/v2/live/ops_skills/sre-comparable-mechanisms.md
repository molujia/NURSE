# 同机制对照与条件含义

问题：两侧在同一机制定义和窗口究竟哪里相同、哪里不同？用 compare_candidates(focus=mechanisms) 获取整对条件；单候选可 inspect_candidate，精确叶子用 read_node，补充复核用 check_mechanism。续页严格使用原参数与 cursor，不猜 ID。

对照保持 necessary/refuting/optional、AND/OR、三值、混合叶子计数与非适用。condition_meaning 只解释此条件；disposition_rule=necessary_conjunction_only 表示必要条件合取决定正式接纳，反证并非自动扣分开关。相同条件不能解释分差，不同机制不能称同类比较。

decisive_known_leaf 只在原结构、窗口与已知叶子不变时稳定；补未知不能翻转已决定的 OR/AND，替换已知叶子则不同。反证 false 不证明机制为真，unknown 不等于正常。check_mechanism 的 supplementary_support 不产生新的正式 accepted，也无正式评分权。

若已回答当前问题，选择真正有区分价值的其他来源或停止；若仍缺来源则诚实保留，不强制读完所有页或更换候选。
