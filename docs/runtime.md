# 运行配置与可信扩展

本文描述已完成阶段 1～6 的当前运行配置。`AGENT_RUNTIME_CONFIG` 在进程启动时读取 TOML；API 与 Worker 必须使用相同配置。不配置文件时，内置 `chat` 保持默认提示词、空工具和不要求工作区。配置格式固定为 `version = 1`，支持 `[execution]` 执行限制、`[skills].directories`、`[knowledge].enabled`、静态 `[mcp]` / `[[mcp.servers]]`、`[[workspaces]]` 的 `id/root` 和 `[[profiles]]` 的 `id/system_prompt/tools/requires_workspace/skills`。不接受导入路径或运行代码。根目录必须存在且为绝对目录，符号链接规范化后作为工作区身份；普通文件内容变化不改变身份。

提交绑定包含 profile ID、profile 配置指纹、工作区 ID/根路径身份和总指纹，使用规范 JSON 与 SHA-256。只绑定相关 profile 和工作区，修改无关声明不会影响已排队 Run。profile 指纹还覆盖工具 Schema 与可信扩展版本。工具顺序不影响绑定。会话固定 ID 和创建时根路径身份；新轮次采用当前 profile 版本，提交后 Worker 不允许版本静默变化。模型名称、端点及超时在领取时记录，不参与 HTTP 幂等输入比较。

生产 `Registry.from_env()` 注册工作区工具、Todo、技能读取、只读委派、知识、后台工具声明及静态目录中的 MCP 工具；Profile 选择本轮允许的名称。Worker 按 Run 组装真实工具实例，后台工具受 Task 根 Agent 身份约束，知识与 MCP 分别按工作区和静态服务配置绑定。可信 Python 代码可通过 `Registry(config, tools=[standard_tool], validators={...}, extensions={...})` 注册扩展，再交给 `Resources(registry=registry)`。配置文件仅能选择已注册名称。`Extension` 提供 `middleware_factory`、`state_schema`、`response_format` 和显式 `version`；每个 Run 调用一次工厂，必须返回全新 middleware 实例。工具实现、参数校验、Hook、状态或输出行为变化时，必须提升对应 profile 的 `Extension.version`，即使公开 Schema 不变；不要把这些实现放进 TOML 或 HTTP 请求。

原生工厂可以独立调用：

```python
from agent_service.runtime import build_agent

agent = build_agent(
    model,
    tools=[standard_tool],
    store=native_store,
    checkpointer=checkpointer,
    system_prompt="可信系统提示词",
    middleware=[native_middleware],
    state_schema=BusinessState,  # 扩展原生 AgentState，保留 messages reducer
    response_format=OutputSchema,
)
```

直接可信调用未传策略时，允许显式工具和传入中间件提供的工具；重复名称在调用模型前报错。Worker 仅允许 Profile 声明的中间件工具，使用本轮真实实例绑定权限，显式工具中排除中间件已提供的实例。应用 Worker 从提交绑定解析 profile 策略，不能通过 `Worker(tools=...)` 开放未注册工具。`RunContext` 是冻结的数据类，包含 run_id、conversation_id、profile、workspace_id、规范根路径、工作区身份、绑定指纹、数据库 deadline_at、Agent 父子标识与深度，以及 Task 身份和结果交付相关上下文。运行资源通过 ToolRuntime.store 或可信运行上下文访问；模型 Key、DSN、HTTP/数据库句柄不进入 checkpoint 状态或摘要。

工具调用沿原生异步 `awrap_tool_call` 组合：生命周期外层 → 原生控制及附加 Hook → 最终权限检查 → 显式 ToolErrorMiddleware → 原生执行器。ToolErrorMiddleware 统一放在权限内侧，因此不能恢复权限控制自身的错误。每次框架调用最多进入一次底层 handler；模型 Hook 同样不能隐式重试。Hook 可以改写名称和参数；改目标时需同时使用注册的真实工具对象，可用 `dataclasses.replace(request, tool=registered_tool, tool_call=...)`。运行上下文和原始调用 ID 不能替换。

权限检查覆盖最终允许名称、注册对象身份、绑定指纹和可信参数校验。Pydantic 工具使用同一 tool_call_schema 解析参数；JSON Schema 工具由其共用解析/校验函数检查。校验器签名为 `(parsed_arguments, context)`，返回 False 或抛 `PermissionDenied` 表示拒绝，其它异常终止 Run。原生参数错误继续由框架返回。拒绝生成关联原始调用 ID 的 `ToolMessage(status="error")`，内容含 `PERMISSION_DENIED`，没有工具副作用，模型可以修正，且仍计入工具调用限制。

可恢复错误建议使用 `agent_service.policy.ToolErrorMiddleware((SpecificRecoverableError,))`，它基于原生 ToolErrorMiddleware，只返回稳定安全说明，不透传异常原文。也可直接使用原生 ToolErrorMiddleware 的显式回调；取消、中断、配额和超时信号仍由边界保护并重新抛出。普通 Hook 不允许吞掉底层控制错误或未知执行异常。默认没有工具重试。主模型 16 次、工具 10 次、图并发 2；摘要仍使用锁定框架的原生行为。

生命周期 start/finish 日志不默认记录完整参数；finish 包含 Run ID、原调用 ID、有效工具名、耗时和结果类型。日志失败可降级，权限和执行控制失败必须阻止操作。每个 Run 的图、策略和 middleware 独立，生命周期管理的模型、HTTP 客户端和数据库池复用。

`submitted_binding` 在提交时写入；`runtime_config` 在有效租约下幂等写入并确认后，Worker 才调用 Agent。运行摘要只记录模型名称、去除用户信息/路径/查询的端点、白名单参数、profile/提示词/工具 Schema 指纹、中间件配置与扩展版本、框架版本。引入运行摘要前的历史运行保持 runtime_config 为 null。普通答案使用最终 AI 文本；非 Task Run 的结构化答案使用原生 structured_response 的 JSON 文本，不追加模型调用；Task Run 使用最终 AI 文本。会话与同一 Task 的续 Run 只继承成功 Run 的消息，图内业务状态、计数和 Run 资源重新开始。

核心数据库由 `python -m db.migrate` 使用 Alembic 标准程序化升级，单链 `core_0001 → core_0002 → core_0003 → core_0004 → core_0005 → core_0006 → core_0007 → core_0008`。`core_0002` 增加运行绑定和配置摘要；`core_0003` 增加 execution_resources；`core_0004` 增加同步外部写入审计 external_operations；`core_0005` 增加 Task、依赖、后台操作和事件，并在 Run 中增加 task_id / execution_no。`core_0006` 仅增加三处可空 JSONB failure_info，值必须为对象且不超过 2 KiB；历史记录保持 NULL。阶段三的上下文能力与阶段四的只读委派自身未新增迁移。core_0001.sql 是唯一基线定义，structure.json 是只读校验签名；测试验证已知旧版本升级和空库初始化得到相同结构。迁移全程使用同一 advisory lock，先校验再 stamp/upgrade；revision 表由 agent.alembic_version 承担版本权威，旧 schema_version 仅保留只读来源标识。原生 checkpoint 与 Store 表由迁移入口调用框架 setup() 管理，不由应用迁移重建。运行角色只有 revision 的 SELECT，readiness/doctor 检查版本和权限；启动服务不执行 DDL。


## 上下文、Todo 与技能

内置 workspace_read/workspace_write 按 Run 创建 TodoListMiddleware；chat 保持空工具。自定义 Profile 在 tools 中选择 write_todos/read_skill，并以 skills 声明技能白名单。服务端注册绝对技能目录，只解析直接子目录的 SKILL.md，使用 PyYAML 安全读取 frontmatter 和 UTF-8 正文，在启动时验证格式、链接边界与输出上限。不可变快照绑定名称、描述和正文 SHA-256；只读内存工具不依赖 Docker，也不挂载技能目录。可选元数据不产生授权效果。配置修改后 API/Worker 同步重启。

Todo 的可信参数验证在最终 Hook 改写后执行；使用框架自身组合状态 Schema，保留自定义业务状态和结构化输出。生命周期限制直接 ToolMessage 和 Command.update.messages 的正文，保留 Command 的其它状态及路由、消息调用 ID、状态与 artifact，字典 artifact 补充 truncated 标志。默认所有工具输出为 16 KiB/200 行；Todo 状态不裁剪，技能正文启动时必须能完整返回。

主模型请求组合顺序为原生摘要与限制 → 扩展 Hook/原生 Todo 提示 → 权限与执行中间件 → RunContextMiddleware → 模型调用边界。每次补入可信提示和当前约束、技能目录、Todo；不把这些上下文加入持久消息。history_tokens_approx 仅估算历史，request_tokens_approx 还包含系统与工具 Schema。20000 是摘要触发阈值，不是模型上下文窗口硬上限；6000 保留与 16000 摘要输入裁剪继续遵循原生分段。摘要失败使用既有异常和资源清理路径，不重放工具。

`inherited_state()` 是唯一状态继承入口，白名单固定 messages；只从同会话或同 Task 上一成功 Run 的明确根 namespace 最终 checkpoint 读取。Todo、计数、自定义状态、结构化结果和 Run 资源每轮重建；Task 依赖数据与未消费后台结果由 Worker 重新注入。trace 原生 checkpoint.channel_values.todos 可查看进度；runtime_config 记录上下文策略版本、Todo 限制、可见技能哈希和摘要提示版本，无技能正文。

## 受控只读委派

`delegate_read(task)` 在内置 workspace_read/workspace_write 默认开启，自定义 Profile 显式启用并至少拥有一种工作区读取/搜索工具。公开参数不接受 Profile、模型、工作区、权限、期限或 checkpoint。`RunContext` 的 agent_id、parent_agent_id、agent_depth 标识父子关系；run_id 始终是资源所属的实际父 Run。`build_agent` 的可信参数 model_call_limit/tool_call_limit 默认为 16/10，子图独立设置为 8/6，服务端固定两次启动与一层深度。

配置指纹和 runtime_config 对启用委派的 Profile 记录实现版本、子提示版本、读取白名单和限次；未启用的 Profile 保持原绑定计算。部署需同步更新 API/Worker，旧排队工作区绑定变化按既有规则拒绝；只读委派自身未新增迁移或依赖。自定义 Profile 示例见 [配置示例](../config/runtime.example.toml)。

`delegations` 为最多两条原生父状态记录：delegation_id、original_tool_call_id、status、error_code、namespace、final_checkpoint、model_calls、tool_calls 和 reported_usage。失败子图保留中间原生 checkpoint，可按 metadata 中 parent_run_id/delegation_id/original_tool_call_id/agent_depth/delegation_version 追踪；仅成功子图有最终关联指针。reported_usage.models 为模型报告聚合，complete=false 表示缺失报告或异常中止，missing_reports 记录缺失数；不得将缺失用量解释为零。

HTTP `/v1/runs/{run_id}/trace?namespace=...` 与 CLI `python -m db.trace_run RUN_UUID --namespace ...` 精确过滤 namespace，省略读取全部，空字符串读取根。子 namespace 由框架生成，复制实际 trace 值，不自行构造。子 checkpoint 不用于执行恢复，父最终指针和会话继承都限定根 namespace。Run 重启后不恢复父子执行。


## 项目知识与同步 MCP

项目知识使用限定工作区的原生 PostgreSQL Store 只读视图；人工管理 API 以 ETag 和事务级锁完成创建、更新及删除。MCP 采用静态 HTTP 服务、冻结目录、Profile 白名单与每 Run 独立原生 MCPAdapter；外部写入先确认派发记录，未知结果不重发。core_0004 新增 external_operations，trace/timeline/doctor 提供关联核对。详细契约、配置、部署和故障处理见 [项目知识与外部工具](项目知识与外部工具.md)。


## 持久 Task 与后台操作

`POST /v1/tasks` 使用固定目标、Profile、工作区与前置依赖创建 Task。后台工具通过自定义 Profile 显式选择，配置示例提供 `task-workspace`；内置 Profile 不自动获得后台工具。本地后台命令使用独立 Docker 容器，远程后台工具使用独立启用 Tasks 扩展的连接及原生句柄，不走同步 MCPAdapter 调用路径。

Run 发布阶段报告后释放执行槽，后台操作按独立租约监控；操作结束、清理确认且结果尚未被成功 Run 消费时创建唯一续 Run。当前 Task 在本地后台操作期间只能读取、搜索和只读委派，文件没有一致快照保证；写入和普通 Shell 被禁止。重启只核对已有操作，不恢复失败 Run 或重放外部提交。API、限额、状态及故障处理见 [持久任务与后台操作](持久任务与后台操作.md)。


## 异常契约与安全诊断

公共分类、控制异常判定和模板位于 `agent_service.failure`；Worker、执行边界、操作保存与 sweep 共用。Hook 契约及数据库故障分别产生 HOOK_CONTRACT_ERROR、PERSISTENCE_FAILED。异常组包含控制异常时保留传播；原生错误消息与调用计数仍由框架产生。工具诊断放入标准 ToolMessage.artifact.failure_info，先预留诊断预算再裁剪正文与其它 artifact。

`scripts.serve` 与 `agent_service.worker` 安装共享 JSON 格式器。日志只接受关联字段及既有用量/委派统计白名单；字符串不超过 256 UTF-8 字节，单条 JSON 不超过 16 KiB。堆栈仅含异常类型、文件、函数和行号；不格式化异常消息、原因链、源码或局部变量。超限先裁剪统计再裁剪堆栈并标记 truncated，格式化失败返回固定安全事件。

Run/会话轮次详情与 Task 详情在同一 REPEATABLE READ READ ONLY 快照中读取主体、关系与未决摘要；不取 admission 锁，不触发维护或外部观察。列表不逐条聚合。完整状态守卫、摘要定义与部署约束见 [第一批契约](第一批异常契约与诊断.md)。

第二批 core_0007 增加持久观察预算、远程轮询下限和独立取消尝试/确认；统一有效执行期限、清理/发布预留及维护故障隔离见[后台核对与清理可靠性](第二批后台核对与清理可靠性.md)。

第三批 core_0008 仅新增 runs.verification；可信异步验证在明确 checkpoint 与清理之间执行，required 完成门禁、消息来源和后台结果消费在原事务内收敛。默认关闭的重复错误 middleware 在摘要/模型前停止三次相同可信可修正错误。注册、报告限制、受控文件访问、异常优先级与部署见[第三批说明](第三批可信验证与重复错误控制.md)。
