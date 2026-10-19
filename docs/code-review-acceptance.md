# 固定范围 Review 实施与验收

本次实现为既有可执行检查分析之上的独立 Review 流程。默认关闭，未进行生产部署或真实模型人工语义验收。

| 批次 | 已实现内容 | 自动验收入口 |
|---|---|---|
| 1 契约与分派 | 独立 Review 请求/报告、服务端 Profile、预算、core_0010、入口守卫、幂等与发布分派 | `tests/test_code_review.py` 请求与 API 测试；`tests/test_database.py` 迁移与权限回归 |
| 2 固定版本与差异 | index 原始副本、workspace 三模式、历史/根/合并提交、direct/唯一 merge-base、固定 tree、两侧读取、差异与 hunk ID | `test_workspace_modes_are_raw_and_original_immutable`、`test_net_change_and_unborn_index`、历史与范围测试、`test_review_target_copies_checks_and_exports` |
| 3 执行证据 | head Python 观察 runner、独立登记容器、脚本产物、日志分页、共享预算与原执行核对 | Review 端到端复现测试、`test_execution_evidence_pages_and_specific_failure_diagnostic`、恢复文件中的预算和执行边界测试 |
| 4 问题与覆盖 | 变化行和两侧证据验证、实际送达回执、hunk/逐项覆盖、同图纠错、确定性渲染和持久化复核 | `test_pagination_coverage_and_forged_evidence`、historical/static/clean/partial/partial_finding 样例、报告篡改与发布测试 |
| 5 生命周期与回归 | 取消、失租、Worker 中断/SIGKILL、Docker 断联、确认丢失、GC 与历史上下文隔离 | `tests/test_code_review_recovery.py`；阶段一及全仓库回归 |

自动验收使用真实 Git、PostgreSQL 和 Docker，模型为 SyntheticModel；测试入口拒绝真实模型 HTTP 和任何跳过用例。每次端到端任务前后比较原目录文件摘要、权限以及 `.git` 下的 index、HEAD、引用与对象文件，确认没有写回。

最终全量验收：**765 项通过，0 失败，0 错误，0 跳过**，耗时 551.97 秒。Review 专项另行运行 61 项全部通过。`core_0010` 迁移、运行角色授权、readiness 和只读 doctor 均通过；验收元数据记录的所有源码摘要与最终文件一致。

运行命令为 `python -m scripts.postgres_integration --port 55440`，Python 3.12.3。第一次全量运行发现一项历史迁移测试仍断言 `core_0009`；更新为新版本断言后重新执行完整验收，未跳过或弱化任何用例。

- [全量测试摘要](../.data/integration/agent-core-test-cf9b5ed99f60/summary.json)、[JUnit 结果](../.data/integration/agent-core-test-cf9b5ed99f60/pytest.xml)、[版本、镜像及源码摘要](../.data/integration/agent-core-test-cf9b5ed99f60/metadata.json)。
- [Review 专项 JUnit 结果](../.data/review-tests/final-review-tests.xml)。
- [20 组整理后的样例清单](../.data/review-tests/final-examples/index.json)：每组包含 JSON、Markdown、固定差异及对应复现脚本、结果和日志；已核对报告与差异的范围绑定和脚本数量。
- [detached HEAD 补充核对](../.data/review-tests/detached-head.json)、[移动引用及多 hunk 行号补充核对](../.data/review-tests/fixed-ref-and-hunks.json)。

测试数据库均已删除，最终 Docker 容器清单为空，验收元数据中的 `new_unreleased_containers` 为空。原目录不变断言、确认丢失时不重复执行及中断后的资源恢复均通过。以上记录于 2026-10-19 完成最终核对。

样例覆盖：

- 原目录 D 审查历史 C，读取及复现命中 C；错误主位置在同一图内纠正。
- clean：语义保持的改动，全部差异已读且注册检查通过，无正式问题。
- static：语义保持的改动，静态差异覆盖完整，未运行检查和复现。
- partial：未读补丁，不能宣称完整无问题；partial_finding 保留已验证位置的发现并列出另一未审查文件。
- all/staged/unstaged：实际读取、检查和观察命中正确副本；unstaged 的 untracked 仅作执行上下文。
- root/merge/range-direct/range-merge_base：导出可核对的比较元数据；missing-tests 证明不会回退到其他版本的测试。
- observed/not_observed/exception/invalid/timeout/stdout，以及恢复测试中的缺依赖、OOM、日志截断和只读挂载。

未完成的上线条件：真实模型人工语义验收。需人工确认问题实际成立、触发条件明确、属于本次改动且建议可执行。确定性测试、结构校验及模型编写脚本的观察结果均不能替代这一项；因此本次不启用默认开关，也不宣称完成生产上线验收。
