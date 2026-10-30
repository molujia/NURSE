# 真实评分选择链

问题：正式分项差异由哪些实际输入和聚合规则形成？用 inspect_candidate/compare_candidates(focus=scoring)。首屏保留双侧原 rank、完整分项与差值及明细交付状态；后续同 family 成对，term/path 的精确内容经 read_node 可达。不从目录补出未读事实。

pool.selected_term_id 仅池内胜出；family.selected_term_id 是家族代表；只有 candidate.selected_family_keys 的代表值进入家族聚合。numeric 未选、unknown、依赖未用仍是需要保留的来源。使用原 config/aggregator 与保存结果，不把非线性聚合说成独立可加贡献。

role_support 的解释须该 term 的 role_mechanism_ids 及原 calculation、raw/weighted/local strength 支撑；某机制 accepted 不代表实际给该 term 根因加成。保留 metric、modality、quality、window_summary/paid_episode/state_event/error_status 粒度，不把窗口摘要说成持续异常段。

propagation 为0并非证明没有传播；结合实际 selected_path_ids、路径值、两端 witness 与 reason_codes。模型不能改正式排名。若已有材料回答问题，无需仅因还有目录就重复读取；仍允许有理由的复核或转向。
