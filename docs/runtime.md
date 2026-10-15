# 运行配置与可信扩展

`AGENT_RUNTIME_CONFIG` 在进程启动时读取 TOML；API 与 Worker 必须使用相同配置。不配置文件时，内置 `chat` 保持默认提示词、空工具和不要求工作区。配置格式固定为 `version = 1`，支持 `[execution]` 执行限制、`[skills].directories`、`[[workspaces]]` 的 `id/root` 和 `[[profiles]]` 的 `id/system_prompt/tools/requires_workspace/skills`。不接受导入路径或运行代码。根目录必须存在且为绝对目录，符号链接规范化后作为工作区身份；普通文件内容变化不改变身份。

提交绑定包含 profile ID、profile 配置指纹、工作区 ID/根路径身份和总指纹，使用规范 JSON 与 SHA-256。只绑定相关 profile 和工作区，修改无关声明不会影响已排队 Run。profile 指纹还覆盖工具 Schema 与可信扩展版本。工具顺序不影响绑定。会话固定 ID 和创建时根路径身份；新轮次采用当前 profile 版本，提交后 Worker 不允许版本静默变化。模型名称、端点及超时在领取时记录，不参与 HTTP 幂等输入比较。

生产 `Registry.from_env()` 注册标准工作区工具、Todo 与技能读取声明；Profile 选择本轮允许的名称。后续可信 Python 代码可通过 `Registry(config, tools=[standard_tool], validators={...}, extensions={...})` 注册扩展，再交给 `Resources(registry=registry)`。配置文件仅能选择已注册名称。`Extension` 提供 `middleware_factory`、`state_schema`、`response_format` 和显式 `version`；每个 Run 调用一次工厂，必须返回全新 middleware 实例。工具实现、参数校验、Hook、状态或输出行为变化时，必须提升对应 profile 的 `Extension.version`，即使公开 Schema 不变；不要把这些实现放进 TOML 或 HTTP 请求。

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

直接可信调用未传策略时，允许显式工具和传入中间件提供的工具；重复名称在调用模型前报错。Worker 仅允许 Profile 声明的中间件工具，使用本轮真实实例绑定权限，显式工具中排除中间件已提供的实例。应用 Worker 从提交绑定解析 profile 策略，不能通过 `Worker(tools=...)` 开放未注册工具。`RunContext` 是冻结的数据类，包含 run_id、conversation_id、profile、workspace_id、规范根路径、工作区身份、绑定指纹及数据库 deadline_at。资源句柄通过 ToolRuntime.store 等框架资源接口访问；模型 Key、DSN、HTTP/数据库句柄不进入状态或摘要。

工具调用沿原生异步 `awrap_tool_call` 组合：生命周期外层 → 原生控制及附加 Hook → 最终权限检查 → 显式 ToolErrorMiddleware → 原生执行器。ToolErrorMiddleware 统一放在权限内侧，因此不能恢复权限控制自身的错误。每次框架调用最多进入一次底层 handler；模型 Hook 同样不能隐式重试。Hook 可以改写名称和参数；改目标时需同时使用注册的真实工具对象，可用 `dataclasses.replace(request, tool=registered_tool, tool_call=...)`。运行上下文和原始调用 ID 不能替换。

权限检查覆盖最终允许名称、注册对象身份、绑定指纹和可信参数校验。Pydantic 工具使用同一 tool_call_schema 解析参数；JSON Schema 工具由其共用解析/校验函数检查。校验器签名为 `(parsed_arguments, context)`，返回 False 或抛 `PermissionDenied` 表示拒绝，其它异常终止 Run。原生参数错误继续由框架返回。拒绝生成关联原始调用 ID 的 `ToolMessage(status="error")`，内容含 `PERMISSION_DENIED`，没有工具副作用，模型可以修正，且仍计入工具调用限制。

可恢复错误建议使用 `agent_service.policy.ToolErrorMiddleware((SpecificRecoverableError,))`，它基于原生 ToolErrorMiddleware，只返回稳定安全说明，不透传异常原文。也可直接使用原生 ToolErrorMiddleware 的显式回调；取消、中断、配额和超时信号仍由边界保护并重新抛出。普通 Hook 不允许吞掉底层控制错误或未知执行异常。默认没有工具重试。主模型 16 次、工具 10 次、图并发 2；摘要仍使用锁定框架的原生行为。

生命周期 start/finish 日志不默认记录完整参数；finish 包含 Run ID、原调用 ID、有效工具名、耗时和结果类型。日志失败可降级，权限和执行控制失败必须阻止操作。每个 Run 的图、策略和 middleware 独立，生命周期管理的模型、HTTP 客户端和数据库池复用。

`submitted_binding` 在提交时写入；`runtime_config` 在有效租约下幂等写入并确认后，Worker 才调用 Agent。运行摘要只记录模型名称、去除用户信息/路径/查询的端点、白名单参数、profile/提示词/工具 Schema 指纹、中间件配置与扩展版本、框架版本。历史运行的 runtime_config 保持 null。普通答案使用最终 AI 文本；结构化答案使用原生 structured_response 的 JSON 文本，不追加模型调用。会话仍只继承成功 Run 的消息，业务状态、计数和运行资源重新开始。

核心数据库由 `python -m db.migrate` 使用 Alembic 标准程序化升级，单链 `core_0001 → core_0002 → core_0003`。core_0003 增加阶段二的 execution_resources 与资源事件/占用约束；阶段三不新增迁移。core_0001.sql 是唯一基线定义，structure.json 是只读校验签名；测试验证当前基线升级和空库初始化得到相同结构。迁移全程使用同一 advisory lock，先校验再 stamp/upgrade；revision 表由 agent.alembic_version 承担版本权威，旧 schema_version 仅保留只读来源标识。原生 checkpoint 表不纳入应用迁移重建。运行角色只有 revision 的 SELECT，readiness/doctor 检查版本和权限；启动服务不执行 DDL。


阶段三内置 workspace_read/workspace_write 按 Run 创建 TodoListMiddleware；chat 保持空工具。自定义 Profile 在 tools 中选择 write_todos/read_skill，并以 skills 声明技能白名单。服务端注册绝对技能目录，只解析直接子目录的 SKILL.md，使用 PyYAML 安全读取 frontmatter 和 UTF-8 正文，在启动时验证格式、链接边界与输出上限。不可变快照绑定名称、描述和正文 SHA-256；只读内存工具不依赖 Docker，也不挂载技能目录。可选元数据不产生授权效果。配置修改后 API/Worker 同步重启。

Todo 的可信参数验证在最终 Hook 改写后执行；使用框架自身组合状态 Schema，保留自定义业务状态和结构化输出。生命周期限制直接 ToolMessage 和 Command.update.messages 的正文，保留 Command 的其它状态及路由、消息调用 ID、状态与 artifact，字典 artifact 补充 truncated 标志。默认所有工具输出为 16 KiB/200 行；Todo 状态不裁剪，技能正文启动时必须能完整返回。

主模型请求组合顺序为原生摘要与限制 → 扩展 Hook/原生 Todo 提示 → 权限与执行中间件 → RunContextMiddleware → 模型调用边界。每次补入可信提示和当前约束、技能目录、Todo；不把这些上下文加入持久消息。history_tokens_approx 仅估算历史，request_tokens_approx 还包含系统与工具 Schema。20000 是摘要触发阈值，不是模型上下文窗口硬上限；6000 保留与 16000 摘要输入裁剪继续遵循原生分段。摘要失败使用既有异常和资源清理路径，不重放工具。

`inherited_state()` 是唯一状态继承入口，白名单固定 messages；只从同会话上一成功 Run 的明确最终 checkpoint 读取。Todo、计数、自定义状态、结构化结果和资源每轮重建。trace 原生 checkpoint.channel_values.todos 可查看进度；runtime_config 仅新增策略版本、Todo 限制、可见技能哈希和摘要提示版本，无技能正文。

阶段四新增 `delegate_read(task)`，内置 workspace_read/workspace_write 默认开启，自定义 Profile 显式启用并至少拥有一种工作区读取/搜索工具。公开参数不接受 Profile、模型、工作区、权限、期限或 checkpoint。`RunContext` 增加默认兼容的 agent_id、parent_agent_id、agent_depth；run_id 始终是资源所属的实际父 Run。`build_agent` 的可信参数 model_call_limit/tool_call_limit 默认为 16/10，子图独立设置为 8/6，服务端固定两次启动与一层深度。

配置指纹和 runtime_config 对启用委派的 Profile 记录实现版本、子提示版本、读取白名单和限次；未启用的 Profile 保持原绑定计算。部署需同步更新 API/Worker，旧排队工作区绑定变化按既有规则拒绝，不新增迁移或依赖。自定义 Profile 示例见 config/runtime.example.toml。

`delegations` 为最多两条原生父状态记录：delegation_id、original_tool_call_id、status、error_code、namespace、final_checkpoint、model_calls、tool_calls 和 reported_usage。失败子图保留中间原生 checkpoint，可按 metadata 中 parent_run_id/delegation_id/original_tool_call_id/agent_depth/delegation_version 追踪；仅成功子图有最终关联指针。reported_usage.models 为模型报告聚合，complete=false 表示缺失报告或异常中止，missing_reports 记录缺失数；不得将缺失用量解释为零。

HTTP `/v1/runs/{run_id}/trace?namespace=...` 与 CLI `python -m db.trace_run RUN_UUID --namespace ...` 精确过滤 namespace，省略读取全部，空字符串读取根。子 namespace 由框架生成，复制实际 trace 值，不自行构造。子 checkpoint 不用于执行恢复，父最终指针和会话继承都限定根 namespace。Run 重启后不恢复父子执行。


## 阶段五补充

项目知识使用限定工作区的原生 PostgreSQL Store 只读视图；人工管理 API 以 ETag 和事务级锁完成创建、更新及删除。MCP 采用静态 HTTP 服务、冻结目录、Profile 白名单与每 Run 独立原生 MCPAdapter；外部写入先确认派发记录，未知结果不重发。core_0004 新增 external_operations，trace/timeline/doctor 提供关联核对。详细契约、配置、部署和故障处理见 [项目知识与外部工具](项目知识与外部工具.md)。


持久 Task、后台 Docker/MCP Tasks、跨 Run 结果衔接与独立监控租约见 [持久任务与后台操作](持久任务与后台操作.md)。core_0005 扩展现有事务与工作区冲突检查，不引入外部队列或 Workflow 引擎。
