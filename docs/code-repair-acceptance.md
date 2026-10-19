# 隔离修复阶段三验收记录

## 实现范围

新增默认关闭的 `code_repair`、显式 `candidate_only` 请求和服务端保留 Profile。独立可写候选、不可变版本、注册开发检查、独立镜像验收、同图失败反馈、租约发布与失败保留全部接入既有 Task/Run 生命周期。没有写回原目录、合并或交付日志。

迁移为 `core_0011`，历史迁移及历史结构签名保留；新增候选、版本、attempt 表，检查及资源增加候选 / 版本归属，运行角色授权、readiness 与迁移白名单同步升级。

## 自动验证对应关系

| 边界 | 主要证据 |
| --- | --- |
| 显式候选入口、默认关闭、无授权验收拒绝、幂等、普通入口禁止 | `test_repair_contract_and_default_off`、`test_repair_entry_authority_idempotency` |
| 配额和全局 / 阶段时间必须容纳基线与一轮 gate | `test_repair_admission_reserves_baseline_and_one_gate` |
| 原目录不变、staged / unstaged / untracked 和模式保留、B→Rn | `test_repair_same_graph_versions_and_export[user-changes]` |
| W 可写、Git 挂载只读、禁止原目录回退挂载 | 候选流程挂载断言、`test_repair_rejects_mismatched_mount_or_publication[writable-fallback]` |
| hash 编辑、创建删除、忽略规则冻结、权限模式、缓存排除 | `test_hash_edits_checkpoint_create_delete_and_ignore_policy` |
| 两轮失败后第三轮通过，同一 Run / 候选 / 图，预算连续 | `test_repair_same_graph_versions_and_export[two-failures]` |
| 开发断言改恒真、修改 pytest 配置 / conftest 仍无法降低验收 | `test_repair_same_graph_versions_and_export[test-bypass]`，开发检查通过、独立验收失败 |
| 旧检查不能证明修改后的版本，最终必须重验 | `test_repair_same_graph_versions_and_export[stale-check]` |
| 普通结束、无效报告、混合批次、未完成目标、旧源码摘要 | `test_repair_gate_correction_boundaries` |
| 特殊文件、未结束进程阻止封存，允许模型修正 | 同组 `special-file` / `background` |
| 版本、检查、gate 和发布确认丢失不重放 | `test_repair_lost_confirmation_never_replays_mutation_or_acceptance` |
| 取消、失租、Worker 中断、Shell 超时、模型 / 工具 / 存储预算 | `test_repair_failures_retain_last_consistent_checkpoint` |
| 验收协议损坏终止，独立检查超时 / OOM 形成失败反馈，候选 OOM 终止 | `test_repair_control_faults_and_independent_check_feedback` |
| 清理失败、最终 checkpoint、包字节或检查摘要不一致，禁止发布成功 | 同组清理测试及 `test_repair_rejects_mismatched_mount_or_publication` |
| 实际 SIGKILL 后重启只清理、保留检查点，不重放 | `test_repair_sigkill_restart_preserves_checkpoint_without_replay` |
| 历史 Review 在新基线重新读取，不继承旧复现结论 | `test_historical_review_is_context_not_current_reproduction` |
| 完整 B 包、版本清单、真实二进制包、文本补丁及版本摘要一致 | 正式发布事务重新生成 / 比对；候选流程下载字节和 ZIP 内容断言 |
| GC 覆盖 W、独立 Git、全部版本、包、过期未登记目录 | `test_repair_gc_owns_writable_revisions_packages_and_abandoned_directories`；原有活动依赖与下载事务测试 |
| 旧分析、Review、固定验证器、Run / Task / API、Docker 清理恢复 | 完整现有测试集，包含真实数据库历史迁移与最小授权验证 |

自动测试使用真实 Git、PostgreSQL、Docker 和 SyntheticModel；HTTP 守卫禁止真实模型 HTTP，严格 runner 拒绝零用例及任何跳过。中断的开发回归不作为通过结果；以最终完整报告为准。

## 端到端产物

集成目录 `repair-examples/` 包括：`standalone`、`historical-review`、`two-failures`、`test-bypass`、`stale-check`、`budget`、`user-changes`、`baseline-passed` 和各类 `recovery-*`。完整候选样例导出基线 ZIP、各轮版本清单、文本补丁、修改文件 ZIP、检查结果与日志、attempt 反馈以及结构化 / Markdown 报告；恢复样例导出诊断并保留最后一致版本。版本引用均明确，失败产物不标为成功报告。

## 真实模型人工验收

另行启动本机既有 Qwen3.5-0.8B BF16 权重与 llama-server 后端，使用临时 `127.0.0.1:1236` 端点、32768 token 上下文和 temperature=0，执行两次真实 Worker / PostgreSQL / Docker 候选任务，未使用 SyntheticModel，且与自动测试 HTTP 守卫完全分开。

| 场景 | 观察结果 |
| --- | --- |
| 按模块合同修复 `sample.answer()` | 32 次模型调用后 `QUOTA_EXCEEDED`；模型反复猜错文件路径、重复请求开发检查，没有修改文件或提交 RepairDraft。检查预留配额生效。 |
| 明确告知根目录文件 `sample.py` | 成功读取当前源码，但重复读取并请求未授权历史上下文（已拒绝）；第 7 次模型调用后本机后端返回工具格式 `peg-native` 解析错误，Run 为 `MODEL_CALL_FAILED`。没有修改文件或提交 RepairDraft。 |

两次任务均未发布成功候选；原目录及其权限模式保持不变，全部主 / 子执行资源确认释放，保留基线包、最后一致版本、检查证据、消息轨迹和失败诊断。独立人工复核结论为 **真实模型语义验收未通过**，不能将本地小模型的工具调用冒烟当作可用修复能力。临时模型进程和两次专用数据库均已关闭。

证据分别位于 `.data/repair-real-model/`、`.data/repair-real-model-explicit-path/`：`summary.json`、`manual-review.json`、`messages.json`、`artifacts/`、`run.log`；模型启动参数及后端日志也已保留。上线前仍需按 [使用说明](code-repair.md) 通过真实模型修复需求、测试合理性与限制说明的人工复核。默认开关继续关闭。

独立验收当前提供 Python JSON 函数 RPC 适配器，示例套件只覆盖 `sample.answer()` 合同。其他应用入口需要可信适配器及对应独立套件；不能将示例通过解释为通用项目验收。

## 最终运行

- 完整回归：**808 通过，0 失败，0 错误，0 跳过**，648.44 秒。证据：`.data/integration/agent-core-test-49b0f5164d4a/`。
- 最终修复专项与完整样例导出：**43 通过，0 失败，0 错误，0 跳过**，119.37 秒。证据：`.data/integration/agent-core-test-f82ce1fa6a2c/`。
- 两次运行均通过 `core_0011` 迁移、运行角色授权、readiness 和只读 doctor，`metadata.json` 中 `new_unreleased_containers=[]`；测试数据库及卷已移除。
- 最终完整样例位于后一次运行的 `repair-examples/`。历史 Review 样例附授权前序报告；恢复样例也导出基线、版本清单 / 包、检查记录、attempt 和诊断。
- `pytest.xml`、`summary.json`、`metadata.json` 和 `integration.log` 保存计数、镜像 ID、依赖与源码摘要。最终专项元数据对应当前实现及导出代码。
