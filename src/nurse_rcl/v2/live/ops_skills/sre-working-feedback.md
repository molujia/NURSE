# D12：按问题读取并核对实际反馈

inspect_candidate和compare_candidates必须提供focus：symptoms读取已有症状／测量，
mechanisms读取完整机制，scoring读取已发布评分，all读取原完整类别；续页不得改变
focus、目标或冻结范围。未选类别叫未展开，不是没有反证。其他工具参数不增加focus。

已经知道具体机制card_id时，直接read_node或合法check_mechanism。核验C22不得将
C17的条件当作C22的条件；先核对实际card_id、mechanism_key、来源和目标再解释。
候选首屏或合法导航不保证返回了问题所指机制，不用工具成功代替问题已回答。

delivery_feedback来自实际回执身份，不是新观测：repeated_complete_record表示已有
完整内容重读；first_complete_expansion表示过去只有概览，现在才完整展开；来源或
内容变化另行标注。说清本次究竟新增了什么，重读无新增时不能声称新增条件支持。

historical_evidence_summary是过去证据的条件级工作摘要：保留AND/OR、条件角色、
三值、原因和叶子三值数量，但不携带所有叶子值。OR=true不能抹掉其false／unknown
叶子。数量不是独立证据票数；需要精确历史叶子时通过原card_id显式回查，仍计动作。
上一完整回执保留实际细节。摘要不等于新测量，也不能让未展开记录获得观测权限。

下一步围绕尚未回答的具体问题选择：该检查如何支持、反驳或缩小竞争解释？可选择
症状对照、具体机制、已有依赖／时序工具、有理由重读或停止。不要仅因还有未读页
就连续翻页；不强制Top-2或某种顺序，不为耗尽动作而查询。调查始终不改变正式排名。
