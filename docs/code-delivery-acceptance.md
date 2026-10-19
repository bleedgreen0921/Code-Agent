# 阶段四确定性验收

## 交付范围

新增显式 `auto_apply`，服务端工作区白名单默认空。沿用 Task/Run、不可变候选和注册验收，新增滚动 P/A/U→M 合并、M 的独立检查、同图回流、持久交付中断、受控容器代次、宿主原子文件事务、恢复 Worker、可信管理 CLI、正式交付报告与未恢复内容保留。数据库迁移为 `core_0012`。

`candidate_only` 没有自动升级为写回模式。原目录 index 不执行 Git 写操作。完成事务必须核对 A 报告证据、M 的真实检查、最终 checkpoint、容器释放、applied 和每个文件的确认。恢复使用原 Run 和截止时间，模型入口由持久预算与恢复专用禁止调用的模型双重约束。

## 验证矩阵

| 边界 | 自动验证 |
| --- | --- |
| 无并发变化、无修改候选、相同修改 | `test_auto_delivery[simple/no_changes/merge_equal]`；空写集记录 `applied / none` |
| 同文件不同区域合并、新版本独立检查 | `test_auto_delivery[clean_merge]`；M 与 A 的 revision 不同，M 有自己的检查 ID |
| M 失败后继续同图、滚动 P、用户再次撤销 | `test_auto_delivery[merged_failure/rolling_revert]`；同一 Run，用户改动安装核对后才更新 P |
| 清理后用户变化、同字节原子替换、容器代次、预算不归零 | `test_auto_delivery[cleanup_refresh/inode_refresh]`；模型调用保持两次，按新身份重新捕获 U |
| 纳入清单与 Git 忽略规则 | `test_auto_delivery[ignored_files]`；冻结版本的显式路径始终核对，原有忽略文件保持原样 |
| 三种差异 | B→A 候选包、P→A 集成产物、U→M 交付包分别保存并在交付证据校验时重建比对 |
| 初次 U 后最多三次刷新 | `test_auto_delivery[refresh_limit]`；第四次变化停止且不写回 |
| 文本、add/add、delete/modify、文件目录、二进制、模式冲突 | `test_merge_matrix`；内容和模式分别合并 |
| 累计用户来源与精确撤销 | `test_provenance_survives_second_install_and_unrelated_user_edit`；无关修改不授权撤销旧区间 |
| 增删、二进制、模式、新建目录、保留 index | `test_binary_add_delete_mode_and_resolved_retention` |
| 原生 checkpoint、冻结字节、检查记录被破坏 | `test_delivery_requires_bound_evidence`；不进行正常写回 |
| 写入意图之后、原子重命名之后但 fsync 之前、fsync 之后数据库未确认 | `test_sigkill_recovery[intent/before_fsync/partial]`；真实子进程 SIGKILL 和真实文件系统变化 |
| 全部文件已达 M、确认丢失 | `test_sigkill_recovery[all]`；已应用 inode 不被再次替换 |
| 后续用户修改、持续占用 | `test_sigkill_recovery[third_state]`；保留新内容，needs_recovery 阻止排队任务启动 |
| 部分写入取消、超时、用户暂存动作 | `test_sigkill_recovery[cancel/timeout/index_change]`；只安全回滚本次内容，保留最新 index |
| 回滚本身再次 SIGKILL | `test_sigkill_recovery[rollback_crash]`；重新核对已回滚路径，不重复覆盖 |
| 恢复保持原生上下文 | 所有 SIGKILL 交付恢复场景都核对最终 checkpoint 中的 Todo 原样保留 |
| 已 applied 后取消、报告发布中断 | `test_sigkill_recovery[applied_cancel/publication]`；保留交付历史，只补 checkpoint 和正式发布 |
| 检查/原子交换之间的外部写入、根与父目录替换 | `test_external_writer_preserved`；实际被替换的用户内容保留于私有槽，不强制回滚 |
| 硬链接、xattr、符号链接与 flock | `test_unsupported_original_metadata_is_rejected`、`test_lock_and_directory_rollback_do_not_remove_user_files` |
| 授权与最低预算 | `test_auto_apply_needs_separate_authorization_and_budget` |
| 重启后同名工作区改指其他目录 | `test_sigkill_recovery[root_reconfigured]`；拒绝继续修改旧目录，保留原交付事实和占用 |
| 保留与 GC | 已发布终态删除暂存/备份；needs_recovery 即使 Task 终态且过期仍保留现场和占用 |
| 兼容回归 | 完整既有分析、Review、候选修复、固定验证器、Task/API、历史迁移和运行角色授权用例 |

自动测试使用真实 Git、PostgreSQL、Docker、Linux 文件系统和 SyntheticModel。故障进程在真实 rename 或持久写入意图之后被 SIGKILL，未以单纯函数失败模拟文件系统窗口。HTTP 守卫阻止自动测试访问真实模型。

## 导出样例

严格集成 runner 的 `delivery-examples/` 导出 Task、候选、版本清单与包、实际检查/日志、attempt、结构化与 Markdown 报告、完整交付计划和逐文件状态。十个主要样例为：

1. `simple`：无并发修改自动交付。
2. `clean_merge`：同文件合并及 M 独立验收。
3. `merged_failure`：M 失败后同图修复并交付。
4. `conflict`：冲突停止，无正常写回。
5. `cleanup_refresh`：容器清理后重新捕获、合并和验证。
6. `recovery-partial`：部分写回 SIGKILL 后继续。
7. `recovery-all`：全量写回、确认丢失、不重复交换。
8. `recovery-third_state`：后续用户编辑保留并进入恢复状态。
9. `recovery-cancel`：取消安全回滚。
10. `recovery-publication`：应用成功后只恢复发布。

另含滚动撤销、刷新上限、fsync 前中断、写意图中断、超时、暂存动作、回滚再中断及 applied 后取消样例。

## 上线条件

阶段三真实模型语义验收未通过，本阶段没有用 SyntheticModel 或恢复测试替代该条件。生产白名单继续关闭。上线前仍需独立验证真实模型的修复效果、因果解释、测试合理性与最终报告。宿主 Linux、本地文件系统、同盘私有目录、固定镜像和固定 Git 可执行文件身份是当前支持边界。

最终运行结果见下方记录；中途主动中断的开发批次不计为验收通过。

## 最终运行记录

- 完整回归：**863 通过，0 失败，0 错误，0 跳过**，794.47 秒。证据：`.data/integration/agent-core-test-54b48ef1d7b1/`。
- 最终交付、恢复、文件系统及发布兼容专项：**57 通过，0 失败，0 错误，0 跳过**，136.29 秒。证据：`.data/integration/agent-core-test-5865347e4536/`。
- 完整回归之后最后增加了“同名工作区路径重配后拒绝恢复旧目录”的局部授权守卫；最终专项覆盖该守卫、全部自动交付/恢复路径及此前发现的最小发布参数兼容问题。
- 两次严格运行都通过 `core_0012` 迁移、运行角色授权、readiness 和 doctor；`metadata.json` 的 `new_unreleased_containers=[]`，专用数据库及卷均已移除。
- 最终交付样例位于最后一次运行的 `delivery-examples/`，包括清单、B→A/P→A/U→M 包和补丁、真实检查、日志、计划、逐文件确认及报告。完整回归同时保留既有阶段样例。
- `pytest.xml`、`summary.json`、`metadata.json`、`integration.log` 保存计数、固定执行镜像、依赖和源码摘要。最终专项元数据对应最终实现源码。
- 此前完整回归发现过一处普通发布参数缺省 `profile` 的兼容错误，已修复并由完整回归及最终专项验证；中断和失败的开发运行不计为通过。

生产白名单未启用，真实模型语义验收仍未通过，不能将上述确定性结果视为生产上线批准。
