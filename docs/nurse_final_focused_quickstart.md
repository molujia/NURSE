# NURSE 有限源码交付：快速开始

## 保存报告的展示兼容

当前 G+ owner 保存 `report.md` 和 `evidence.md`，正文的自动证据链接使用 `report-evidence.md`。这是文件名兼容问题，不是新增证据或模型结论。不要向已封存的 owner 目录补文件、改链接或覆盖原件，否则会改变完整成员集校验。可在一个全新的、位于 owner 之外的展示目录建立字节相同的副本和别名：

```bash
# REPORT_OWNER 指向已结束且已核验的 gplus-report 目录。
# REPORT_VIEW 必须是 owner 之外、尚不存在的绝对目录。
test ! -e "$REPORT_VIEW"
mkdir -m 700 -- "$REPORT_VIEW"
cp -- "$REPORT_OWNER/report.md" "$REPORT_VIEW/report.md"
cp -- "$REPORT_OWNER/evidence.md" "$REPORT_VIEW/evidence.md"
cp -- "$REPORT_OWNER/evidence.md" "$REPORT_VIEW/report-evidence.md"
cmp -- "$REPORT_OWNER/report.md" "$REPORT_VIEW/report.md"
cmp -- "$REPORT_OWNER/evidence.md" "$REPORT_VIEW/report-evidence.md"
```

展示副本不参与 owner 准入，不替代原件、来源链或费用账本；`semantic_certified=False` 保持。此发布未更改历史渲染协议，后续统一文件名仍是已知的展示层改进项。

原验收 candidate 已完成一个已揭盲案例的产品回归：确定性 core 完成，调查执行两次决策和两次行动后以合法 `stage_paused` 交接，G+ 最终综合完成并生成报告，组合状态为 `completed`。这是单案产品流程贯通证据，不认证排名、报告语义或总体质量。工具及证据边界见[工具目录](nurse_final_focused_tools.md)。

本出口排除两个研究专用 helper，代码清单为 580 成员，必须使用独立计算的代码 pin。原 582 成员 candidate 的计划与验收 pin 不能直接移用。出口的软件检查须单独执行；以下命令不授予新的 ARK 调用预算。

## 1. 无需 ARK 的软件检查

在具有项目依赖与测试依赖的 Linux 环境中，从完整 checkout 的仓库根目录执行；安装方法见[仓库说明](../README.md)。项目支持 Python `>=3.10,<3.13`。测试封装使用 POSIX `resource` 和锁语义，需可写的 `/tmp`，并允许地址空间 8 GiB、栈 64 MiB、文件描述符 4096 的资源限制。Windows 或兼容桩结果不能替代 Linux 验证。

```bash
set -euo pipefail
mapfile -t selected_tests < tests/final_explainability_0ark.txt
test "${#selected_tests[@]}" -eq 45
python scripts/dev/run_final_explainability_0ark.py \
  -q --tb=short "${selected_tests[@]}"
```

以上使用 Bash 数组逐项传递[固定的 45 文件清单](../tests/final_explainability_0ark.txt)。其余测试模块用于提供导入依赖和合成夹具；不要将本门的范围自动扩展为整个 `tests/` 目录。清单不依赖历史案例的行政 freezer。

该封装设置数值库单线程，禁用 pytest 插件自动加载与缓存，拒绝网络连接、真实凭据加载及外部执行。每次运行强制使用新建的私有 `/tmp` pytest 目录；仅该目录内字节完全固定的假 KEY1 负例可进入真实解析器，以验证 KEY1 被拒绝，KEY2 和其它文件均不允许。唯一子进程例外是受守卫的 `/tmp` 合成保存结果读取回归。它不会向真实 provider 发送请求；运行结果只作为所列合成测试的软件证据。不得把历史答案、标签、原始遥测或密钥放进测试输入。

原 candidate 的软件门曾完成 45 文件、818 项、零跳过；该结果不自动适用于当前出口，出口实际结果以本节命令的完整输出为准。中断的检查不能计为通过。

## 2. 准备可审阅的 focused 组合计划

真实接入使用 `GplusFocusedCompositionPlan`：`schema_version="nurse.v2.gplus-one-case-composition/2"`、`investigation_profile="v2-ops-sre/15"`，配置是 `configs/v2/ops_sre_focused_profile.json`。从完整 checkout 使用这些配置与源文件；单独安装 wheel 不能代替计划要求的完整代码根目录。

计划必须引用当前代码与运行环境对应的新 core plan 及其 SHA-256。core 必须为 `core-only`、profile `v2-ops-isolated/2`、`source_core=null`，输出位于该 run 的 `core-results`。代码/skill/config 清单、输入 pin、推理身份清单、父计划、preparation key、source files 与 source descriptor 约束必须一致。代码改动后旧 plan 的 code pin 不再代表当前实现；不能只改一个 profile 名称沿用旧计划。

`investigation_decision_limit` 由计划明确指定为严格整数 1–8，调查可以提前停止，之后进行 1 次 G+ 最终综合。本次验收使用两次决策上限；未来运行按各自明确授权的计划预算执行。上限内每次决策至多选择一个实际行动；导航、翻页、读取、检查和显式测量分别计入行动预算，不表示工具本身各收取一笔模型费用。另有最多一次严格重复修正，不能当作额外调查配额；每个逻辑调用最多 5 次传输尝试，未知发送或计费状态不得盲重发。

这是全新 run：重新生成确定性 core，随后派生同案 focused 调查，再封存供最终综合。不能恢复旧 owner、导入历史调查/报告来冒充新交付，也不能通过重启获得新预算。已有开始或结束标记时使用状态入口核查；不重跑命令。

## 3. 仅检查计划接入

安装完成后可先查看入口参数，此命令不读取计划或调用 ARK：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:third_party/hermes_runtime
python -m nurse_rcl.v2.live.ops_sre_gplus_case --help
```

接入检查需自行准备并审阅完整组合计划；本仓库没有附带可直接启动的真实计划。将 `COMPOSITION_PLAN`、`COMPOSITION_SHA256` 设为该计划的绝对路径及其实际文件 SHA-256。以下命令只检查接入，不含执行授权：

```bash
: "${COMPOSITION_PLAN:?请设置已审阅组合计划的绝对路径}"
: "${COMPOSITION_SHA256:?请设置该计划的 SHA-256}"
python -m nurse_rcl.v2.live.ops_sre_gplus_case \
  --plan "$COMPOSITION_PLAN" \
  --plan-sha256 "$COMPOSITION_SHA256" \
  --check-only
```

`--check-only` 也要求完整、真实可访问的配置资源：规范 JSON 及独立 pin、当前冻结代码/skill/config 清单、匹配的 Python 与数值库环境、推理身份/父计划/输入元数据、兼容的 preparation 身份与 source 根目录，以及已存在且不与代码、preparation、输入或历史输出重叠的 core 输出目录。路径须满足绝对路径及无符号链接等接入约束。它执行元数据校验，不加载 preparation、不导入历史调查、不派发查询或模型调用，不代表真实数据运行已验收。

该入口在检查分支之前设置地址空间 8 GiB、栈 64 MiB、文件描述符 4096 的软/硬限制，因此 Linux 进程权限与现有硬限制必须允许这些值；导入模块前也应设置上述线程环境变量，并与 plan 中冻结环境一致。检查成功输出 `status="admitted"`、`new_ark_calls=0`、`preparation_load_import_dispatch=0`。错误应按实际缺失资源、pin 或环境差异处理，不通过编造 plan、兼容桩或绕过校验消除。

## 4. 真实运行与验收边界

本次单案回归的完整输出已在源码交付之外保全。后续真实运行需要对其精确 composition SHA-256、全新 run 与 1–8 决策上限对应的预算单独授权；执行入口还会核对 `--approved-composition-sha256` 与该 pin 完全一致。`--help`、`--check-only`、软件测试以及本次回归结论均不授予后续调用预算。

后续验收应检查目标证据是否实际交付、双卡差异与 unknown 是否被准确解释、同源是否被重复计票、G+ 最终文本是否保留作用域/窗口/单位/权限及来源闭包。机械接入、合成测试和单案流程完成不能认证报告语义，也不能证明排名或总体准确率。RCL 评分与既有科研结论不由本次有限源码交付改写。
