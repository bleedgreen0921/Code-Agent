# Agent 执行框架

本机运行的 Agent 执行服务，包含任务 API、异步 Worker 和 PostgreSQL 持久化。执行层使用 LangChain `create_agent`、原生消息状态、PostgreSQL Checkpointer 和摘要中间件。提供空工具聊天、容器内工作区搜索读取和文件修改/Shell 三种内置 Profile，并支持 Todo、Skills、受控只读委派、人工管理的项目知识、静态授权 MCP 工具，以及独立持久 Task 与后台操作。所有工作区工具在 Docker 中运行，直接修改注册目录。

基础增强阶段 1～6 及三批异常、可靠性与可信验证增强已实施，当前数据库版本为 `core_0008`。最新行为与验收见[第三批说明](docs/第三批可信验证与重复错误控制.md)和[独立验收记录](docs/第三批可信验证与重复错误控制验收.md)。测试使用合成模型，实际模型任务效果需单独评估；原阶段路线见[基础增强路线图](docs/基础架构增强路线图.md)。

## 安装与初始化

需要 Python 3.12、普通 PostgreSQL 16 和兼容 OpenAI Chat Completions 的模型端点。工作区工具另需同一 Linux 主机的 Docker daemon；多个 Worker 必须连接同一数据库与同一 daemon。LangChain 1.4.2、LangGraph 1.2.12、Checkpoint Postgres 3.1.2 保持锁定。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
export MIGRATION_DATABASE_URL='postgresql://owner:OWNER_PASSWORD@127.0.0.1/agent_core'
export AGENT_DATABASE_URL='postgresql://agent_runtime:RUNTIME_PASSWORD@127.0.0.1/agent_core'
.venv/bin/python -m db.migrate
.venv/bin/python -m db.bootstrap
.venv/bin/python -m db.doctor --json
```

请预先创建空数据库。迁移 owner 需要创建表；bootstrap 还需要创建运行角色的权限。运行角色必须与 owner 分离，无 schema、数据库创建权或 revision 写权限。迁移通过单条 Alembic 链 `core_0001 → core_0002 → core_0003 → core_0004 → core_0005 → core_0006 → core_0007 → core_0008` 创建/升级核心表，版本权威为 `agent.alembic_version`，并调用框架 Checkpointer 和 PostgreSQL Store 的 `setup()`；重复执行安全，API/Worker 不执行 DDL。

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

可选 `AGENT_RUNTIME_CONFIG` 指向 TOML 文件，由标准库 tomllib 读取并通过 Pydantic 校验，仅启动加载。参见 [配置示例](config/runtime.example.toml)。未指定时继续使用空工具的内置 chat；三个内置 Profile 均不可覆写。API 和 Worker 使用一致的注册配置，API 无需模型 Key 或模型端点。

代码分析、只读 Review 和基础修复可显式启用独立的[代码任务配置](config/runtime.code.example.toml)，提供 `code-analyze`、`code-review`、`code-fix`。替换配置、依赖准备和 API 示例见[阶段一使用说明](docs/代码能力阶段一使用说明.md)，自动化证据见[阶段一验收](docs/代码能力阶段一验收.md)。

```bash
export AGENT_RUNTIME_CONFIG=/absolute/path/to/runtime.toml
curl -X POST http://127.0.0.1:8002/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"task":"讨论项目","profile":"project-chat","workspace_id":"project"}'
```

配置声明版本、工作区、Profile、技能目录、execution 限制、knowledge 开关及静态 MCP 服务。工具仅能选用启动时的可信注册表，不可通过 TOML 或 HTTP 指定 Python 模块、工具实现、middleware、状态或输出 Schema。未知工具、保留名称冲突、非法根目录导致启动失败。修改配置后按上述停止/重启流程上线；不支持热更新。

以下配置均使用 `AGENT_` 前缀，数值为正。时间单位为秒，token 数为框架估算值。

| 环境变量 | 默认值 | 用途 |
| --- | ---: | --- |
| `AGENT_QUEUE_TIMEOUT` | 60 | 提交后的排队期限 |
| `AGENT_RUN_TIMEOUT` | 300 | 首次认领后的总执行期限，包含摘要、模型、工具和等待 |
| `AGENT_MODEL_TIMEOUT_SECONDS` | 120 | 单次 SDK 请求超时 |
| `AGENT_MAX_OUTPUT_TOKENS` | 4096 | 模型最大输出长度 |
| `AGENT_MAX_ACTIVE_RUNS` | 2 | 全局活动 Run 与未释放执行资源的 Run 上限（去重） |
| `AGENT_MAX_QUEUED_RUNS` | 8 | 全局排队上限 |
| `AGENT_MAX_UNFINISHED_RUNS` | 10 | 全局未完成任务上限 |
| `AGENT_SUMMARY_TRIGGER_TOKENS` | 20000 | 消息历史摘要触发阈值，非模型上下文窗口硬上限 |
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

测试入口使用普通 `postgres:16`，生成独立凭据，覆盖继承的生产配置，阻止真实 HTTP，预先构建执行镜像，通过合成模型、真实 PostgreSQL 和真实 Docker 验证执行。缺少必要环境直接失败。测试报告必须非空、零失败、零错误、零跳过。新测试直接 `pytest` 也启用 HTTP 禁令；数据库用例必须由隔离入口配置的 `AGENT_TEST_DATABASE=1` 执行。

诊断使用原生 checkpoint 历史、消息、生命周期事件和 Worker 结构化日志。trace 同时提供提交绑定和实际运行摘要；摘要按白名单记录模型、脱敏端点、提示词/工具 Schema 指纹、中间件配置及框架版本，不保存密钥或 DSN。异常退出或失租的在途任务标记失败，不重放模型/工具；checkpoint 留供诊断。发布提交确认丢失时仅查询/重试结果发布，不重新执行 Agent。

## 工作区工具与镜像

部署前构建镜像，然后将命令最后输出的 `sha256:...` ID 写入 TOML 的 `[execution].image`。构建使用固定 Python 镜像 digest、Debian snapshot 和现有完整依赖锁文件。Run 中不会构建、拉取镜像或安装依赖。

```bash
.venv/bin/python -m scripts.build_execution_image
export AGENT_RUNTIME_CONFIG=/absolute/path/to/runtime.toml
.venv/bin/python -m db.config_check --role agent-worker
```

| Profile | 工具 | 工作区 |
| --- | --- | --- |
| `chat` | 无 | 不启动 Docker；可不绑定工作区 |
| `workspace_read` | `glob_search`、`grep_search`、`read_file` | 必须绑定；只读挂载 |
| `workspace_write` | 上述工具、`edit_file`、`shell` | 必须绑定；读写挂载 |

表中列出工作区执行工具。两个工作区 Profile 还默认启用 `write_todos`、`delegate_read`；注册技能时启用 `read_skill`，开启项目知识时启用 `knowledge_get`、`knowledge_search`。MCP 与后台工具由自定义 Profile 显式选择。

自定义 Profile 可选择这些工具的子集；包含 edit_file 或 shell 时使用读写挂载，其余子集只读。内置搜索复用锁定版本的 `FilesystemFileSearchMiddleware`，在容器内调用，固定使用 rg；Shell 复用 `ShellToolMiddleware` 和派生 Docker 策略。兼容实现集中在 [execution_adapter.py](agent_service/execution_adapter.py)。

```bash
curl -X POST http://127.0.0.1:8002/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"task":"查找测试入口并读取相关文件","profile":"workspace_read","workspace_id":"project"}'
```

文件路径相对于工作区，`/foo` 也表示工作区下的 foo。read_file 提供 UTF-8 内容、行号、SHA-256、总行数与截断标志，默认最多 200 行；可用 start_line/max_lines 分段读取。edit_file 使用 create=true/new_text 创建且不覆盖，或使用 expected_hash/old_text/new_text 作唯一文本替换；冲突作为可恢复工具错误反馈。文件大小上限默认 10 MiB；读写拒绝符号链接、目录穿越及特殊文件，修改以同目录临时文件原子发布。搜索同时执行原生路径检查，不读取越界链接。Shell 的 cwd 和变量仅在当前 Run 内保留，禁止显式重启；普通非零退出码为工具错误，模型可调整命令。

容器固定关闭网络、根文件系统只读、移除 capabilities、禁止提权，只挂载本次工作区。默认 1 CPU、512 MiB、128 PID、独立 64 MiB /tmp；使用 Worker 非 root UID:GID，root Worker 必须显式配置非 root user。容器进程只接收固定环境白名单，不继承服务密钥。所有工具最终返回内容（含包装文字）最多 16 KiB、200 行，artifact 保留退出码、哈希和截断标志。单个工具默认期限 30 秒且不超过 Run 剩余期限；超时终止环境并使 Run 失败，不重启 Shell。容器 PID 1 同时执行 Run 绝对截止时间。

**直接修改注册工作区；失败、取消及超时保留已经发生的修改，不自动回滚。** 不同 Run 不继承容器、Shell cwd、变量或后台进程；成功会话轮次仅继承消息。异常任务不自动重放，不恢复旧 Agent。

## 执行资源恢复与诊断

认领 Run 和预留 execution_resources 在同一数据库 advisory lock 事务中完成。规范路径、别名及父子重叠目录独占；冲突任务继续排队，并受 60 秒排队期限约束；Worker 可选择后续不冲突任务。不同目录允许并行，默认全局容量 2。终态任务的未释放资源仍占容量和工作区。

启动顺序为预留 → 创建停止态容器 → 持久化 ID → 按 ID 启动。所有退出路径先关闭工具入口，再移除容器、收敛本地 Docker 客户端和 Shell 线程，最后确认资源释放。成功仅在清理确认后发布，清理和发布期间继续续租。取消和失租竞争仍由数据库最终发布检查裁决。

Worker 启动与轮询时清理遗留资源，仅处理本项目数据库记录与容器标签同时匹配的对象，不重放模型或工具。执行 `python -m db.doctor --json` 可查看未释放资源的 phase、state、daemon_id、container_id 和 cleanup_error；timeline 增加 environment_preparing、environment_active、environment_closing、environment_cleanup_failed、environment_released 和 environment_recovered 事件。trace 保留标准 ToolMessage 和有效配置摘要，无容器句柄或密钥。

| 错误码 | 含义与处理 |
| --- | --- |
| `COMMAND_TIMEOUT` | 单个工具超时；环境关闭，任务失败 |
| `EXECUTION_ENVIRONMENT_UNAVAILABLE` | Docker、固定镜像或辅助程序无法准备；检查配置和 daemon |
| `EXECUTION_ENVIRONMENT_LOST` | 容器/Shell/辅助进程异常退出或内存限制触发 |
| `RESOURCE_CLEANUP_FAILED` | 执行结束后未确认清理；不发布 completed，保留占用 |

清理默认等待 10 秒。Docker 不可达时恢复同一 daemon 后 Worker 自动重试；daemon ID 不匹配时必须恢复原 daemon。创建结果未知且容器不可见时，不能以一次 inspect 未找到为由释放资源；诊断保留 creating 与 unknown 原因。没有 HTTP 强制解锁入口。请先恢复基础设施并核对数据库记录和 Docker 状态；不要删除占用记录来绕过并发保护。


## 复杂任务、技能与上下文

`workspace_read` 和 `workspace_write` 默认启用原生 `write_todos`；复杂任务可按需规划，简单任务直接回答。每次调用替换整张清单，最多 20 项、每项 200 字、序列化 8 KiB，状态为 pending / in_progress / completed。非法或超限输入不更新清单；同一响应中的重复 Todo 更新由原生中间件拒绝。Todo 只反映模型声明的进度，不是验证证据，也不能代替最终回答。新增工具共用每轮主模型 16 次、工具 10 次的额度。

在 TOML 注册 `[skills].directories = ["/absolute/skills"]`，目录的直接子目录使用 `<name>/SKILL.md`：

```markdown
---
name: code-review
description: Review code changes and identify correctness issues.
---
Read relevant code, check concrete behavior, and distinguish verified findings from assumptions.
```

格式遵循 [Agent Skills 规范](https://agentskills.io/specification)。技能必须包含合法且与目录一致的 name、非空 description；UTF-8、普通文件、注册目录内链接、唯一名称及有效输出限制在启动时检查。技能正文必须完整容纳在工具输出限制内（默认 16 KiB、200 行），过长时启动失败并提示缩短正文。每个 Profile 的名称和描述目录最多 8 KiB。可选元数据保留，但 `allowed-tools` 不授予权限；不自动读取项目技能、references/assets/scripts，不执行附带代码。

内置工作区 Profile 可读取全部已注册技能；空目录不暴露 `read_skill`。自定义 Profile 通过 `tools = ["write_todos", "read_skill"]` 和 `skills = ["code-review"]` 显式选择能力，使用 read_skill 时白名单必须非空。只使用这些内存工具的 Profile 无需 Docker。默认 chat 仍为空工具，既有 chat 绑定保持兼容。

API 和 Worker 启动时读取不可变技能快照，修改后须同步重启。技能名称、描述、正文哈希和实现版本参与引用它的 Profile 绑定；排队任务遇到新快照按 `CONFIGURATION_CHANGED` 结束，无关技能修改不会改变未引用它的自定义 Profile。初始请求仅包含名称和描述；`read_skill(name)` 返回当前白名单内的完整正文及名称/哈希 artifact。工具只读内存，不挂载技能目录，也不访问项目文件。历史技能消息不会扩大当前权限，摘要后可再次读取同一快照。

每次主模型调用在扩展 Hook 之后补入可信 Profile 提示、工作区模式、工具范围、无网络、截止时间、失败不回滚、技能目录和当前 Run 的 Todo。尚未规划时明确清单为空。这些上下文仅用于本次请求，不写入消息历史或逐轮积累。日志事件 `model_context` 记录 `history_tokens_approx` 与 `request_tokens_approx`，后者包含系统内容和工具 Schema；估算不是硬配额。摘要仍使用原生模板、分段和重试策略，仅附加代码任务的保留提示，默认触发/保留/输入裁剪分别为 20000/6000/16000 token。

会话唯一继承白名单为 messages：只读取同会话上一成功 Run 发布的明确最终 checkpoint，继承原生摘要和历史工具消息。Todo、计数、业务状态、结构化结果和运行资源每轮重新开始。失败、取消和失租轮次不参与继承。上下文与技能能力在阶段三未新增表、迁移、详情字段、端点或流式协议；当前 Task 续执行的消息与结果衔接见下文“持久任务与后台操作”。

现有 `/v1/runs/{run_id}/trace` 和 `python -m db.trace_run RUN_UUID` 可查看原生 checkpoint 的 `channel_values.todos` 与工具消息，按配置中的最终 checkpoint_id 定位成功结果。`runtime_config` 记录上下文策略版本、Todo 限制、可见技能及哈希、摘要提示版本；技能正文只进入工具消息，不复制到配置快照。架构与扩展规则见 [架构说明](docs/architecture.md) 和 [运行说明](docs/runtime.md)，上下文与技能能力的阶段三验证见 [阶段三验收](docs/阶段三实施验收.md)。

## 受控只读委派（阶段四）

`workspace_read`、`workspace_write` 默认提供 `delegate_read(task)`，`chat` 继续为空工具。自定义 Profile 必须显式选择 `delegate_read`，并同时选择至少一个 `glob_search`、`grep_search` 或 `read_file`。`task` 包含目标、必要背景和期望结果，非空且最多 8 KiB UTF-8；简单查询可直接调用父工具。

每个 Run 最多实际启动两次、串行、深度一层。每次子图只输入本次 task，不继承父历史、Todo、技能正文或上次子任务状态。子工具取固定读取白名单与父权限的交集，保留父参数校验器和技能白名单，重新创建并绑定真实工具实例；无 Shell、文件写入、Todo、自定义工具或再次委派。父图只收到子图的最终文本或稳定错误码。

委派必须独占一次工具批次；混合批次和同批多个委派均不会启动子图，普通工具继续执行。非法输入、权限拒绝和批次拒绝不消耗启动额度；已经启动的局部失败消耗一次，重复调用 ID 不重放。额度保存在 Run 的服务端对象中，摘要和 Hook 修改诊断状态均不能重置。

父图保留原生 16 次模型/10 次工具限制，委派计入父工具次数；每个子图使用独立原生 8/6 限次。合计最多 32 次主循环模型、22 次工具调用；原生摘要额外调用照常发生，不提供硬 token 配额。子图和父图共享剩余总期限、容器、工作区占用和清理流程；子工具保留 30 秒及 16 KiB/200 行默认限制，整个委派不占工作区命令锁，也不套用命令计时器。读写父容器中的子图只读性来自受控工具与校验，不构成独立只读容器隔离。

子配额耗尽返回 `SUBAGENT_QUOTA_EXCEEDED`；SDK 模型调用失败返回脱敏的 `SUBAGENT_MODEL_FAILED`；无合法最终文本返回 `SUBAGENT_INVALID_OUTPUT`。用尽两次启动额度返回 `SUBAGENT_LIMIT_REACHED`。取消、总超时、失租、工具超时、容器故障、持久化故障、Hook 契约和未知异常终止整个 Run。子 Agent 不创建 Run 或队列任务，不占额外 Worker 槽位，不后台执行；重启只清理、终结异常任务，不续跑子 checkpoint。

原生子 checkpoint 使用同一 thread_id 下框架生成的独立 namespace。父状态 `delegations` 最多保存两条关联摘要，`reported_usage` 保存原生回调聚合值与完整性标记；总量包含父、子及摘要，只取根收集器，缺失报告不视为零。父最终发布与会话继承只接受根 namespace 的明确最终 checkpoint，下一轮仍仅继承 messages。完整过程可通过 trace 查看：

```bash
# 默认查询所有 namespace；空 namespace 只查询根图。
python -m db.trace_run RUN_UUID
python -m db.trace_run RUN_UUID --namespace ''
python -m db.trace_run RUN_UUID --namespace 'tools:FRAMEWORK_GENERATED_ID'
curl 'http://127.0.0.1:8002/v1/runs/RUN_UUID/trace?namespace='
```

只读委派在阶段四未增加依赖、表或迁移，当时数据库版本为 `core_0003`；当前部署统一升级到 `core_0008`。API 与 Worker 必须同步更新和重启；已有工作区 Profile 的排队绑定变化仍按 `CONFIGURATION_CHANGED` 处理，第二批期限策略进入 chat 绑定，历史绑定仍保留。详见 [阶段四验收](docs/阶段四实施验收.md)。


## 项目知识与 MCP（阶段五）

通过 `[knowledge].enabled = true` 开启人工管理、Agent 只读的项目知识。新增工作区知识 GET/PUT/DELETE 管理 API，使用 ETag 前置条件和事务锁；Agent 通过 knowledge_get/knowledge_search 获取当前条目或元数据列表，不提供文本检索、Embedding 或自动记忆写入。

静态 `[mcp]` 声明 HTTP 服务、本地 tools/list 目录、凭据环境变量引用和 read/write 分类，Profile.tools 显式选择稳定别名；三个内置 Profile 不自动获得 MCP。同步 MCP 工具由 Worker 使用原生 MCPAdapter，每 Run 独立连接和契约核验；写调用独占批次、先记录派发，未知结果终止 Run，不重发请求。知识或 MCP 单独使用均无需 Docker，已有工作区容器继续无网络。

阶段五通过 `core_0004` 增加外部写入审计并初始化原生 Store；当前部署须迁移到 `core_0008` 并重新 bootstrap，API/Worker 不执行 DDL。配置、完整 API、目录导出、凭据配置和外部结果核对见 [项目知识与外部工具](docs/项目知识与外部工具.md)，验收证据见 [阶段五验收](docs/阶段五实施验收.md)。


## 持久任务与后台操作

`POST /v1/tasks` 创建独立 Task，固定目标、Profile、工作区与前置依赖。后台命令和原生 MCP Tasks 立即启动；Run 发布阶段报告后释放执行槽。Worker 重启核对已有资源，后台结果若未被成功 Run 消费，在操作结束并确认清理后创建唯一续 Run。当前 Task 在本地后台执行期间可读取和只读委派，文件可能变化；写入和普通 Shell 被禁止。未知结果失败且不自动重试，外部副作用无法撤回。

阶段六引入 core_0005；当前部署升级 core_0008、重新 bootstrap 并构建新执行镜像。显式选择配置示例的 task-workspace Profile；内置 Profile 不增加后台工具。API、默认限额、状态和故障核对见 [持久任务与后台操作](docs/持久任务与后台操作.md)。


## 第一批：异常契约与诊断

第一批数据库版本为 `core_0006`，只新增 Run、后台操作、同步外部操作的可空 `failure_info`。详情增加只读 `unresolved`，doctor 同时提供全局摘要。历史 `failure_info: null` 表示该记录未提供详情，不回填或推断历史副作用。

非法后台回执不能成功；普通更新保护操作终态、结果和原资源身份。操作未知与 Task 失败在同一事务内保存，内部迟到事实补记不会恢复 Task、消费结果或改写已发布 checkpoint。清理字段与执行结果独立。API 和 Worker 的受支持入口安装同一安全 JSON 日志格式器，保留关联字段及堆栈位置，省略异常消息、源码和参数。

部署顺序为停止旧 API/Worker → owner 执行 `python -m db.migrate` → `python -m db.bootstrap` → `python -m db.doctor --json` → 启动新进程。运行角色无需增加权限。第一批不增加重试、自动重放、恢复接口或工作区回滚。完整字段、状态规则及边界见 [异常契约与诊断](docs/第一批异常契约与诊断.md)，验证证据见 [独立验收记录](docs/第一批异常契约与诊断验收.md)。

第二批数据库版本为 `core_0007`，增加跨 Worker 的观察预算、远程间隔和取消尝试/确认，执行提前为清理与发布预留时间；维护中断不延长租约授权。详见[第二批行为说明](docs/第二批后台核对与清理可靠性.md)与[验收记录](docs/第二批后台核对与清理可靠性验收.md)。

## 第三批可信验证与重复错误控制

core_0008 仅新增可空的 Run verification 报告。可信 Extension 可注册异步验证器，选择 report 或 required，并显式开启默认关闭的重复错误 middleware。验证在明确最终 checkpoint 后、清理前执行，报告与结果在原事务中发布。required 阶段消息不继承、后台结果不提前消费；历史 null 不代表验证通过。无新增模型调用、验证重试或修复循环。详见[行为与注册示例](docs/第三批可信验证与重复错误控制.md)及[验收](docs/第三批可信验证与重复错误控制验收.md)。

Git Review 阶段二支持当前未提交改动和指定历史提交，部署、参数、限制及 API 示例见[阶段二使用说明](docs/代码能力阶段二使用说明.md)。

代码修复演示见 [Python 示例](examples/python-repair/README.md) 和
[阶段三使用说明](docs/代码能力阶段三使用说明.md)。
显式启用固定可信门禁时使用 [独立配置](config/runtime.code.checked.example.toml)，
部署及检查范围见 [阶段四使用说明](docs/代码能力阶段四使用说明.md)。
