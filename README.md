# Agent 执行框架

本机运行的任务 API、异步 Worker 和 PostgreSQL 持久化。执行层使用 LangChain `create_agent`、原生消息状态、PostgreSQL Checkpointer 和摘要中间件。生产工具列表为空；当前不提供文件、命令、RAG、个人长期记忆或代码审查功能。

## 安装与初始化

需要 Python 3.12、普通 PostgreSQL 16 和兼容 OpenAI Chat Completions 的模型端点。LangChain 1.4.2、LangGraph 1.2.12、Checkpoint Postgres 3.1.2 保持锁定。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
export MIGRATION_DATABASE_URL='postgresql://owner:OWNER_PASSWORD@127.0.0.1/agent_core'
export AGENT_DATABASE_URL='postgresql://agent_runtime:RUNTIME_PASSWORD@127.0.0.1/agent_core'
.venv/bin/python -m db.migrate
.venv/bin/python -m db.bootstrap
.venv/bin/python -m db.doctor --json
```

请预先创建空数据库。迁移 owner 需要创建表；bootstrap 还需要创建运行角色的权限。运行角色必须与 owner 分离，无 schema、数据库创建权或 revision 写权限。迁移通过单条 Alembic 链 `core_0001 → core_0002` 创建/升级核心表，版本权威为 `agent.alembic_version`，并调用框架 Checkpointer 的 `setup()`；重复执行安全，API/Worker 不执行 DDL。

迁移接受空库、结构匹配的当前 `agent_core_0001` 核心库及已知 Alembic revision。旧核心库先经只读结构/标识校验，再登记 core_0001 并升级；历史会话与 Run 回填 chat 绑定，历史模型配置保持 null。旧业务 schema、未知 revision、未知非空库、结构不匹配库在任何 DDL 和权限修改前拒绝。checkpoint 表继续由框架维护。

已有核心部署升级顺序：停止 API/Worker → 备份数据库 → 安装新锁文件依赖 → `python -m db.migrate` → `python -m db.bootstrap` → `python -m db.doctor --json` → 重启 API/Worker。不要混跑旧新 Worker。在途任务仍按失租失败处理，不自动重放。

## 启动

```bash
export AGENT_MODEL_URL='http://127.0.0.1:9000/v1'
export AGENT_MODEL='your-model'
# 模型服务需要认证时，设置 AGENT_MODEL_KEY。
.venv/bin/python -m db.config_check --role agent-worker
.venv/bin/python -m scripts.serve --service agent
# 另一个终端，沿用 AGENT_* 配置：
.venv/bin/python -m agent_service.worker
```

正式启动只需 API、Worker、PostgreSQL 和模型端点。API 默认绑定 `127.0.0.1:8002`、关闭 forwarded headers，检查真实客户端 IP、Host 和 Origin 的 loopback 地址。不要通过代理扩大访问范围。API 只需要 `AGENT_DATABASE_URL`；Worker 另需模型名称与 URL。HTTP 客户端和数据库连接池随服务生命周期关闭。

```bash
curl -X POST http://127.0.0.1:8002/v1/runs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: example-1' \
  -d '{"task":"解释这个任务"}'
curl -X POST http://127.0.0.1:8002/v1/conversations \
  -H 'Content-Type: application/json' -d '{"title":"工作会话"}'
curl -X POST http://127.0.0.1:8002/v1/conversations/CONVERSATION_UUID/turns \
  -H 'Content-Type: application/json' -d '{"task":"继续讨论"}'
```

接口见 [API 契约](contracts/README.md)。

## 配置

可选 `AGENT_RUNTIME_CONFIG` 指向 TOML 文件，由标准库 tomllib 读取并通过 Pydantic 校验，仅启动加载。参见 [配置示例](config/runtime.example.toml)。未指定时继续使用空工具的内置 chat；chat 不可覆写。API 和 Worker 使用一致的注册配置，API 无需模型 Key 或模型端点。

```bash
export AGENT_RUNTIME_CONFIG=/absolute/path/to/runtime.toml
curl -X POST http://127.0.0.1:8002/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"task":"讨论项目","profile":"project-chat","workspace_id":"project"}'
```

配置只声明版本、工作区和 profile。生产注册工具仍为空，不可通过 TOML 或 HTTP 指定 Python 模块、工具实现、middleware、状态或输出 Schema。未知工具、保留名称冲突、非法根目录导致启动失败。修改配置后按上述停止/重启流程上线；不支持热更新。

以下配置均使用 `AGENT_` 前缀，数值为正。时间单位为秒，token 数为框架估算值。

| 环境变量 | 默认值 | 用途 |
| --- | ---: | --- |
| `AGENT_QUEUE_TIMEOUT` | 60 | 提交后的排队期限 |
| `AGENT_RUN_TIMEOUT` | 300 | 首次认领后的总执行期限，包含摘要、模型、工具和等待 |
| `AGENT_MODEL_TIMEOUT_SECONDS` | 120 | 单次 SDK 请求超时 |
| `AGENT_MAX_OUTPUT_TOKENS` | 4096 | 模型最大输出长度 |
| `AGENT_MAX_ACTIVE_RUNS` | 2 | 全局运行/取消中任务上限 |
| `AGENT_MAX_QUEUED_RUNS` | 8 | 全局排队上限 |
| `AGENT_MAX_UNFINISHED_RUNS` | 10 | 全局未完成任务上限 |
| `AGENT_SUMMARY_TRIGGER_TOKENS` | 20000 | 摘要触发阈值 |
| `AGENT_SUMMARY_KEEP_TOKENS` | 6000 | 摘要后近期消息保留量 |
| `AGENT_SUMMARY_TRIM_TOKENS` | 16000 | 摘要输入裁剪上限 |
| `AGENT_LEASE_SECONDS` | 30 | Run 租约与 Worker 在线判定窗口 |
| `AGENT_HEARTBEAT_SECONDS` | 2 | 执行中的续租与取消检查间隔 |
| `AGENT_POLL_SECONDS` | 0.2 | Worker 调度轮询间隔 |

各进程应使用一致配置。摘要保留量必须小于触发值；租约需大于两倍续租间隔。主 Agent 每次运行最多 16 次模型调用、10 次工具调用，超限抛错并发布 `QUOTA_EXCEEDED`；并行工具执行由图配置 `max_concurrency=2` 限制。SDK 自动重试为零，没有工具重试中间件。

模型次数限制只计算主 Agent 循环。摘要调用不计入该限制，使用锁定版本 `SummarizationMiddleware` 自身的重试行为（最多三次尝试），仍受总执行期限约束。近期消息边界和摘要策略由框架管理；没有跨服务硬 token 配额或细粒度调用账本。

## 验证和诊断

```bash
# CI 与本地使用同一个入口，需要 Docker Compose；自动清理专用容器。
.venv/bin/python -m scripts.postgres_integration
# 保留专用测试库用于诊断，凭据写入权限 0600 的本地文件。
.venv/bin/python -m scripts.postgres_integration --keep-db
.venv/bin/python -m db.trace_run RUN_UUID
```

测试入口使用普通 `postgres:16`，生成独立凭据，覆盖继承的生产配置，阻止真实 HTTP，通过合成模型与标准工具验证执行。测试报告必须非空、零失败、零错误、零跳过。新测试直接 `pytest` 也启用 HTTP 禁令；数据库用例必须由隔离入口配置的 `AGENT_TEST_DATABASE=1` 执行。

诊断使用原生 checkpoint 历史、消息、生命周期事件和 Worker 结构化日志。trace 同时提供提交绑定和实际运行摘要；摘要按白名单记录模型、脱敏端点、提示词/工具 Schema 指纹、中间件配置及框架版本，不保存密钥或 DSN。异常退出或失租的在途任务标记失败，不重放模型/工具；checkpoint 留供诊断。发布提交确认丢失时仅查询/重试结果发布，不重新执行 Agent。
