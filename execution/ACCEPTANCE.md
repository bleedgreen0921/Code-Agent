# 阶段二验收

2026-10-15，工作包 A（执行环境与资源闭环）、B（工具与权限接入）、C（故障回归与交付文档）已完成。

统一入口：

```bash
python -m scripts.postgres_integration --port 55433
```

最终报告：66 tests，0 failures，0 errors，0 skipped；pytest 执行时间 32.24 秒。报告位于 `.data/integration/agent-core-test-21e2cfa1b7cb/pytest.xml`，同目录 `summary.json` 由入口校验后生成。测试使用锁定依赖、预先构建的执行镜像、真实 PostgreSQL 16、真实 Docker 与合成模型，禁止真实模型 HTTP。

覆盖范围：

- 原有 API、绑定、权限、原生 checkpoint、会话继承、调用限制与提交确认丢失回归。
- 只读搜索读取；哈希修改及 Shell 验证；Docker 不可用时 Worker 继续执行 chat。
- 实际只读挂载、网络关闭、非 root 用户、资源限额与环境变量隔离。
- 路径穿越、越界链接、特殊文件、非 UTF-8、超大文件、创建/哈希/唯一匹配冲突和原子发布失败。
- Shell 非零退出后的模型调整；命令与文件超时；长单行、stderr 洪泛、最终 Hook 包装的输出上限。
- 同 Run 工具串行、重叠工作区独占、不同目录并行、终态遗留资源占用容量。
- 文件/Shell 取消、后台进程清理、Worker SIGKILL、租约失效、实际内存耗尽与容器绝对期限。
- 创建和移除确认丢失、资源释放提交确认丢失、Docker 不可达、daemon 不匹配、未知创建结果保留占用与恢复清理。
- 清理前禁止发布成功、清理期间续租、取消/失租与最终发布竞争。
- 空库初始化、旧核心库及 core_0002 升级、重复迁移、旧业务库无修改拒绝、运行角色权限和既有 checkpoint 兼容。
- 成功轮次仅继承消息，容器和 Shell 状态不跨 Run 继承。

`ruff check agent_service db scripts tests execution` 与 `git diff --check` 均通过。最终验收后不存在带本项目执行资源标签的遗留容器。

部署与故障处理见 [主文档](../README.md)。直接修改工作区、失败不回滚、固定无网络、超时终止及异常任务不自动重放的边界保持不变。创建结果未知且不可见时保留占用，不提供 HTTP 强制解锁。
