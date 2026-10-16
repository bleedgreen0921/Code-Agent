# 执行与持久化

本文描述已完成阶段 1～6 的当前实现；各阶段当时的版本与测试结果见对应验收记录。当前 Alembic 迁移链为 `core_0001 → core_0002 → core_0003 → core_0004 → core_0005`。

`agent.runs` 保存任务输入、profile/工作区 ID、提交绑定、实际运行摘要、状态、租约、结果及明确的最终 checkpoint 配置；`agent.conversations` 指向最后成功的 Run；`agent.events` 保存生命周期；`agent.workers` 保存心跳。`execution_resources` 保存 Run 执行资源，`external_operations` 保存同步 MCP 外部写入审计；`tasks`、`task_dependencies`、`background_operations`、`task_events` 保存持久 Task、依赖、后台操作与事件。数据库还包含基线/迁移标识、框架维护的 checkpoint 表及原生 Store 表。不使用 pgvector、Embedding 或其它服务 schema。

API 提交与 Worker 认领通过 PostgreSQL 事务 advisory lock 统一串行化容量检查、幂等竞争和会话互斥。不同 Worker 的运行中与取消中 Run 合计不超过全局上限。排队到期、执行期限和租约均使用数据库时间；过期任务在 Worker 调度时终结。没有 Worker 时，过期排队任务仍可在队列概览中被观察到；下次调度或相关状态事务会收敛终态。

Worker 每次认领 Run 生成新 lease_token，同一个 Run 不自动重放模型或工具。续租和最终发布要求 token 一致、状态仍可执行、租约和总期限未过期。最终答案、最终 checkpoint 指针、会话或 Task 成功指针、后台结果消费标记及完成事件同事务提交。取消、失租、超时均阻止迟到成功。发布确认丢失时，Worker 先读终态，必要时仅重试发布；不会重新调用图。持久 Task 可在成功 Run 之后创建新的续 Run 处理后台结果，每个 Task 最多五次 Run。

每个 Run 的 thread_id 为其 UUID。会话下一轮或同一 Task 的续 Run 从上个成功 Run 已发布的 checkpoint_id 精确读取消息，加上本轮 HumanMessage，交由新线程中的原生消息 reducer 与摘要中间件处理。Task 还注入依赖数据与未消费后台结果。主 Agent 限次计数重新开始；不复制旧 Run 的调用计数。失败、取消、失租的中间 checkpoint 不会进入后续历史。原生 checkpoint 保留供 trace 与诊断使用。

`agent_service.runtime.build_agent` 接受标准 BaseTool 列表、checkpointer、可选 BaseStore，以及 system_prompt、middleware、state_schema、response_format 和 permission_policy；默认工具为空、store=None。工具可使用 `langchain.tools.ToolRuntime[RunContext]` 获取冻结的运行身份、profile、工作区根路径/身份、绑定指纹、数据库截止时间及 store。Task 执行还提供 task_id 和结果交付相关的运行上下文。没有 Provider、自定义工具结果、工具意图检查或旧授权协议。直接传入 BaseStore 是标准扩展点；开启项目知识后，Worker 按工作区注入原生 PostgreSQL Store 的只读视图，通过显式知识工具读取，不自动检索或写入。

图使用标准 `create_agent`。普通输出读取最终 AIMessage 文本；非 Task Run 启用结构化输出时读取 structured_response 并编码为 JSON 文本写入 answer，不要求最后消息为 AIMessage，也不额外调用模型；Task Run 使用最终 AI 文本。原生 interrupt/resume 能通过构建后的图和 `Command` 使用；Worker 检测到暂停图时记录 `GRAPH_INTERRUPTED` 失败，HTTP 层不提供恢复或审批 API。失租、异常退出、重启不会认领已有 running Run：租约到期后记录失败，模型/工具操作不自动重放。正常关闭会中止本 Worker 的运行任务并记录 `WORKER_STOPPED`。后台操作的独立监控与 waiting_background Task 的续执行见下文。

资源层为每个进程管理异步数据库池、异步 HTTP 客户端和 SDK 所需的同步客户端；执行只使用异步模型调用。Checkpointer 直接使用框架实现，初始化只在迁移入口进行。运行角色具有核心业务和 checkpoint 表的 DML 权限，没有表 ownership、schema CREATE、数据库 CREATE 或 revision 写权限。

每个 Run 独立构建图、权限策略和 middleware 实例。模型、HTTP 客户端、数据库池按进程复用；跨 Run 仅继承成功消息，业务状态和 Run 执行资源不复制。后台操作独立归属于 Task，其状态通过数据库核对和结果注入衔接。Worker 核对提交绑定后，在有效租约下幂等持久化运行摘要，确认成功才开始调用图。配置摘要确认丢失只重试该数据库操作。详细扩展约束见 [运行扩展说明](runtime.md)。


## 上下文、Todo 与技能

使用标准 create_agent 与框架原生状态组合。注册表保存标准工具声明，Worker 在每轮将显式工具与原生中间件工具统一绑定权限，拒绝重复或未声明名称。Todo 属于原生 todos 状态，技能来自 API/Worker 启动时的不可变内存快照；两者共用现有模型/工具额度、可信参数验证、生命周期与输出限制。

RunContextMiddleware 在每次主模型调用前重建可信运行约束、技能名称/描述目录与当前 Todo，仅修改本次 ModelRequest；摘要只处理原生消息，不保存注入内容。原生 SummarizationMiddleware 继续处理工具调用配对与近期消息。运行配置快照记录上下文/Todo/技能/摘要版本；read_skill 的完整正文作为普通工具消息进入 checkpoint，摘要后可重新读取。

跨轮继承统一经过 messages 白名单和同会话或同 Task 的成功最终 checkpoint 检查。历史 Todo/技能工具消息仅为历史，不恢复清单、不扩展技能白名单；其它图状态和 Run 资源均重新创建。默认 chat 指纹保持兼容，新能力版本仅进入启用这些能力的 Profile 绑定。

## 受控只读委派

采用[官方工具封装子 Agent 模式](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)，沿用 `create_agent`、原生 middleware、Command、checkpoint namespace 和报告用量回调。Worker.prepare 按 Run 创建 DelegationController，负责真实启动 ID 的原子预留与串行执行；该对象不进入 checkpoint，诊断状态不是额度权威。父工具生命周期记录原批次与调用 ID，外层 middleware 提前拒绝混合批次，实际委派入口在 Hook 之后再次检查批次、层级和参数。委派不进入 WorkspaceExecutionMiddleware 的工作区锁；内部读取工具仍进入同一父 scope 的锁、健康检查和命令期限。

子图独立实例化文件与技能读取工具，以固定白名单和父策略交集形成实例绑定策略，并复用所选工具的父参数验证器。子 context 沿用实际 run_id、工作区绑定及截止时间，使用新 agent_id、父标识和深度 1。子图不安装父扩展 middleware，不传父 state_schema、response_format 或 BaseStore；只输入一条 HumanMessage。子提示由可信 Profile 提示与固定只读指令构成，运行约束按子权限重新注入。

子调用以 values/checkpoints 流和同步 durability 等待完成，再核对最终 checkpoint 无 next/interrupt、最终 AI 文本合法且明确指针已持久化。成功仅通过 ToolMessage 返回最终文本，Command 将最多两条关联诊断写入父原生状态。局部失败只有原生调用配额、明确 SDK 模型错误、非法最终文本；其它异常沿 await 链外传，父生命周期防止 Hook 吞掉控制错误。Run 取消后冻结父 scope，等待图和本地执行收敛，再使用既有清理和发布流程。

根用量回调只挂一次，随调用链观察父、子、原生摘要；子回调单独记录自身报告，不再加回根聚合。成功汇总写入父状态，异常结束的日志明确标记不完整。trace 直接读原生 checkpointer，支持可选 namespace 精确过滤；默认全部读取。父图发布和跨 Run 继承明确限定根 namespace，子成功状态无法更新会话或 Task 成功指针。子调用无独立 Run、额外队列/槽位、恢复或进度端点；只读委派自身未引入表或迁移。


## 项目知识与同步 MCP

项目知识使用限定工作区的原生 PostgreSQL Store 只读视图；人工管理 API 以 ETag 和事务级锁完成创建、更新及删除。MCP 采用静态 HTTP 服务、冻结目录、Profile 白名单与每 Run 独立原生 MCPAdapter；外部写入先确认派发记录，未知结果不重发。core_0004 新增 external_operations，trace/timeline/doctor 提供关联核对。详细契约、配置、部署和故障处理见 [项目知识与外部工具](项目知识与外部工具.md)。


## 持久 Task 与后台操作

`core_0005` 扩展现有事务与工作区冲突检查，不引入外部队列或 Workflow 引擎。Task 固定目标、能力绑定和前置依赖；Run 继续表示一次有期限的 Agent 执行。后台 Docker 命令或原生 MCP Tasks 使用独立操作记录和监控租约，Run 发布阶段报告后释放执行槽。操作结束、清理确认且结果尚未被成功 Run 消费时，创建唯一续 Run 处理结果。

Worker 重启先核对已有资源；waiting_background Task 可以继续监控和衔接结果，已执行 Run 的失败 checkpoint 不恢复，外部提交不重放。未知结果会终结 Task 并保留诊断和未释放占用。完整状态、限制和故障核对见 [持久任务与后台操作](持久任务与后台操作.md)。


第一批异常与诊断使用 `core_0006`，保留原有派发/发布事务。统一安全 failure_info 在可信边界产生；操作 unknown 与所属 Task 失败原子保存，普通终态更新及身份换绑被拒绝，内部事实补记不调度 Task。详情摘要独立使用只读快照。见 [第一批异常契约与诊断](第一批异常契约与诊断.md) 和 [独立验收记录](第一批异常契约与诊断验收.md)。
