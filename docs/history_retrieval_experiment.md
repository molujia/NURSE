# 历史案例检索与有效性实验

**研究问题。** 给定故障发生时已经可见的遥测证据，NURSE 能否把有相同故障机制的已结案案例排到前面？同一服务可能经历多种故障，同一机制也可能发生在不同服务。因此预先指定「独立标注的故障机制相同」为主要相关性定义，同时报告「根因服务相同」及「服务和机制均相同」。本文的机制标签指独立注入记录或复盘确认的故障类型。NURSE 对机制条件的检查结果属于预测端的观测特征。

## 1. 建立历史库

1. 在每个数据集内按事件发生时间划分较早的历史库和较晚的查询集。在查看检索结果前冻结分界、案例清单、故障类型词表、服务别名与跨实体映射。按原始注入或真实故障分组。同一事件的不同窗口、采集副本、重跑和同组变体只能出现在一侧。每次独立注入在历史库中占一个位置。保留自然发生的机制频率，包括与查询症状相近而原因不同的事件。
2. 对两侧使用同一版核心、窗口、适配器和文本化规则。每个案例只取当时决策截止前形成的冻结证据。原生导出保留有数值支持的窗口中位数变化、合格持续异常和机制门控状态，最多 24 条，其中机制最多 8 条。相同语义的观测取局部贡献最高的一条，并保留到该案例冻结核心的字段指针。变为固定句式的规范文本后再编码，每个完整文档对应一个向量。服务真值、案例编号、历史诊断结论、人工备注和原始文件路径都不进入向量输入。对自制适配器，应检查信号名来自指标注册表，避免带入注入器名称、工单标题或根因标签。
3. 向核验人提供原始注入记录或独立复盘、事件时间、实体到服务的映射，以及可定位到原始记录的证据。核验人核对记录属于同一事件、复核根因服务和机制、确认同组关系和可用时间，再签署或拒绝。建议两人独立标注并仲裁分歧，封存原始记录、标签词表和最终裁决。核验人看不到检索顺序。不要依据 NURSE 的排名、机制门控或报告生成参考标签。缺少独立故障类型的数据集只能回答服务层面的问题，不能把服务名当作故障类型。
4. 用 `available_at_ns` 记录当时可获取该历史材料的时间，只有严格早于查询决策时刻的材料可以入选。对前瞻部署，应使用实际核验并归档后的可用时刻。对回顾性基准，预先定义并披露采用的结案时刻或代理时刻，另存今天真实的复核时间。缺少结案时刻时，只能解释为该可用性假设下的回放。

在安装了项目依赖的环境中，以下命令分别导出某次运行的检索文档并生成私有核验表。路径为使用者自己的文件。CLI 需要新的输出路径。

```bash
python -m nurse_rcl.evaluation.history_library project --core core/result.json --out documents/incident.json
python -m nurse_rcl.evaluation.history_library prepare-review --candidates history-candidates.json --out private/history-reviews.json
```

`history-candidates.json` 是数组。每行含 `case_id`（核心使用的 `opq_` 加 32 位十六进制标识）、`group_id`、`available_at_ns` 和相对于清单的 `document_path`。核验表把案例、组、可用时间与文档摘要绑定。核验人在私有表中填写 `truth_path`、`truth_sha256`、`truth_pointer`（指向该事件的 JSON Pointer）、规范化 `service` 和 `mechanism`。同意入库时填写 `status=approved`、`reviewer` 及带时区的实际 `reviewed_at`。拒绝的填 `rejected`，未决定的保留 `pending`。入库器验证来源文件及指针，生成复核记录摘要。判断标签是否正确属于核验人的职责。

```bash
python -m nurse_rcl.evaluation.history_library build-library --candidates history-candidates.json --reviews private/history-reviews.json --out library.json
```

将人工复核表和原始真值保存在私有位置；运行时只需冻结的库、独立取得的文件 SHA‑256 及历史来源的索引。历史原文与来源按案例保留，向量只用于排序。技术联调用 `--benchmark-reference` 显式构建、读取仅经基准标注核对的库，审核行标为 `benchmark-reference`。正式评估和面向 SRE 的运行要求人工核验库。

## 2. 运行检索与独立评分

新版检索的两种方法读取逐字相同的幅度与质量文本。对每条保留的窗口摘要，标准化中位数变化绝对值按 1、3、6 切分为四档，代表证据的观测质量按 0.5、0.8 切分为三档；恰在边界的值归入较高一档。持续异常和机制门控保留各自的证据语义。正式比较应从冻结核心重新导出版本 2 投影并复核历史库。旧版库仍可用于事件调查，旧窗口摘要仅提供此前记录的字段。

没有配置嵌入模型时使用本地规则。它匹配证据种类、信号、方向，并对同一观测中的幅度档、质量档分别提取特征。完整状态或属性值匹配权重为 3，种类内的信号或属性名称匹配权重为 1，分词可部分匹配。重复特征取最大权重；历史池中的罕见特征乘以较高的 IDF，即 `1 + ln((n+1)/(df+1))`。两例对应权重的最小值之和除以最大值之和得到相似度，按 `1 − 相似度` 升序取前五例。

指定 `--embedding-model` 或 `NURSE_EMBEDDING_MODEL` 后，程序使用所选模型编码整份案例，以同一模型空间内的原始向量欧氏距离排序。豆包和 OpenAI 的凭据分别为 `ARK_API_KEY`、`OPENAI_API_KEY`。单次检索可用 `--method fixed-rule` 显式运行规则对照。两种方法均使用同一时间截止和候选池，同分时按匿名 ID 确定展示顺序。[运行说明](nurse_final_focused_quickstart.md)

```bash
python -m nurse_rcl.evaluation.history_library retrieve --query documents/query.json --group-id GROUP --library library.json --library-sha256 SHA256 --top-k 5 --out retrieval.json
python -m nurse_rcl.evaluation.history_experiment rank --embedding-model doubao-embedding-vision-251215 --library library.json --library-sha256 SHA256 --queries queries.json --out ranks-doubao.json
python -m nurse_rcl.evaluation.history_experiment rank --embedding-model text-embedding-3-small --library library.json --library-sha256 SHA256 --queries queries.json --out ranks-openai.json
python -m nurse_rcl.evaluation.history_experiment score --ranks ranks-doubao.json --labels private/labels.json --out scores-doubao.json
python -m nurse_rcl.evaluation.history_experiment score --ranks ranks-openai.json --labels private/labels.json --out scores-openai.json
```

正式实验分两次执行。排序命令读取 `queries.json`（顶层 `queries` 数组，每行只有 `document_path` 和 `group_id`），按所有合格历史案例保存完整顺序。先封存此文件及其摘要，再解封独立核验的标签。`private/labels.json` 的 `labels` 数组包含查询与历史案例的 `case_id`、`service` 和 `mechanism`。如果两侧均有独立核验表，可以用以下命令导出私有标签；无可编码证据的查询也应在独立标签清单和覆盖率分母中保留。

```bash
python -m nurse_rcl.evaluation.history_library export-labels --candidates history-candidates.json --reviews private/history-reviews.json --candidates query-candidates.json --reviews private/query-reviews.json --out private/labels.json
```

若队列只有经核实的根因服务，可用 `score --service-only`（其余参数同上）只评估同服务相关性。标准的机制实验使用事先确定的单根因、单机制标注队列。多故障或标签不确定的事件应按预定规则单独统计并披露数量。

同一数据集的两个评分文件应有相同的 `comparison_input_sha256` 与 `labels_sha256`。前者绑定库、查询文本、查询组和逐查询候选集。主表每个数据集只填写一行固定规则，将每个模型相对于该规则的配对差值及区间分别列出。

## 3. 如何判断是否有效

| 比较方法 | 保持相同的条件 | 回答的问题 |
| --- | --- | --- |
| 固定规则 | 与嵌入完全相同的文本和历史池 | 显式匹配信号及其诊断状态能找回哪些案例？ |
| 豆包嵌入 + 原始欧氏距离 | 查询、文本、历史池和预算 | 文本级语义向量能带来多少额外收益？ |
| BM25 | 完全相同的规范文本与合格候选池 | 直接匹配指标、方向和机制词已能解释多少效果？ |
| 同一向量的余弦相似度 | 复用同一批坐标，无额外嵌入调用 | 向量模长是否影响结果？若模长相等，两种排序在数学上等价。 |
| 均匀随机 | 每个查询自己的合格历史池 | 自然频率和库大小能带来多高的偶然命中？使用无放回排列的解析期望。 |

**主要终点**为所有查询上的同机制 nDCG@5 均值。相关性取 0 或 1，相关案例越靠前，得分越高。固定规则与欧氏排序分别对各自的同分组取排列期望，避免案例编号左右判断。没有同机制历史的查询计零，另报库覆盖率及存在同机制历史时的条件均值。MRR 描述第一个相关案例的位置，Recall@5 衡量找回多少相关案例，Hit@5 衡量前五名是否至少有一个。对服务、服务与机制联合匹配分别报告这些结果。

以查询所属原始事件组为单位进行配对 bootstrap，默认 2,000 次，报告嵌入相对固定规则的主指标差值和 95% 区间。正式研究应在独立先导样本上估计配对差的波动，按预定的最小有意义差异和精度确定测试规模，再冻结模型及表示。各数据集分别报告结果，并公开 BM25、随机和余弦对照。少量查询适用于工程选型；总体效果需要独立测试集验证。

**并列与困难案例。** 规范文本可能同时适用于不同真实原因。报告跨越第 5 名的精确并列数及同距组内可达到的 nDCG 范围，检查结论是否依赖编号顺序。单独查看高频/稀有机制、同服务不同机制、跨服务同机制，以及没有可用同机制历史的查询。若多数库只有几个候选，Top‑5 很容易满分；必须同时给出池规模、随机期望及首个相关案例的位置。

**结果解释。** 若同机制案例提前、覆盖率足够、且相对 BM25 的增益在并列敏感性分析中保留，可以支持该数据集和时间划分下的检索有效性。若仅优于随机，证据只支持利用遥测文本比随机选例有价值。若经常召回同服务的其他故障、跨服务的同机制案例排在后面、或多种原因被压成完全相同的文本，应报告对应失败和覆盖缺口。若欧氏与余弦等效，应把结论落在文本表示和检索流程上。

## 4. 若要进一步评价 SRE 是否获益

将“找回相关案例”与“帮助工程师作出下一步判断”作为两个实验。后者可以在同一冻结核心、行动预算、报告模型和可用历史库上，配对比较不提供历史、提供等量随机历史及提供语义检索历史三种条件。随机与语义两组使用相同的案例数和文本长度上限。预先定义区分当前候选解释所需的检查，由不知道条件的 SRE 评定建议能否利用当前事件证据区分解释，同时记录不当套用历史结论的比例。若直接测人，随机分派并平衡顺序，避免同一参与者重复作答同一事件。三组共享冻结的 RCL 排名；检索实验本身只评价案例排序，不能提供这项下游效用的结论。
