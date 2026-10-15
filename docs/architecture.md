# 执行与持久化

`agent.runs` 保存任务输入、profile/工作区 ID、提交绑定、实际运行摘要、状态、租约、结果及明确的最终 checkpoint 配置；`agent.conversations` 指向最后成功的 Run；`agent.events` 保存生命周期；`agent.workers` 保存心跳。数据库包含这些核心表、阶段二 execution_resources、基线标识和框架维护的 checkpoint 表；阶段三不新增表或迁移。不使用 pgvector、Embedding 或其它服务 schema。

API 提交与 Worker 认领通过 PostgreSQL 事务 advisory lock 统一串行化容量检查、幂等竞争和会话互斥。不同 Worker 的运行中与取消中 Run 合计不超过全局上限。排队到期、执行期限和租约均使用数据库时间；过期任务在 Worker 调度时终结。没有 Worker 时，过期排队任务仍可在队列概览中被观察到；下次调度或相关状态事务会收敛终态。

Worker 每次认领生成新 lease_token，任务只执行一次。续租和最终发布要求 token 一致、状态仍可执行、租约和总期限未过期。最终答案、最终 checkpoint 指针、会话成功指针和完成事件同事务提交。取消、失租、超时均阻止迟到成功。发布确认丢失时，Worker 先读终态，必要时仅重试发布；不会重新调用图。

每个 Run 的 thread_id 为其 UUID。会话下一轮从上个成功 Run 已发布的 checkpoint_id 精确读取消息，加上本轮 HumanMessage，交由新线程中的原生消息 reducer 与摘要中间件处理。主 Agent 限次计数重新开始；不复制旧 Run 的调用计数。失败、取消、失租的中间 checkpoint 不会进入后续历史。原生 checkpoint 保留供 trace 与诊断使用。

`agent_service.runtime.build_agent` 接受标准 BaseTool 列表、checkpointer、可选 BaseStore，以及 system_prompt、middleware、state_schema、response_format 和 permission_policy；默认工具为空、store=None。工具可使用 `langchain.tools.ToolRuntime[RunContext]` 获取冻结的运行身份、profile、工作区根路径/身份、绑定指纹、数据库截止时间及 store。没有 Provider、自定义工具结果、工具意图检查或旧授权协议。传入 BaseStore 只是标准扩展点，不创建长期记忆表、不自动检索或写入。

图使用标准 `create_agent`。普通输出读取最终 AIMessage 文本；启用结构化输出时读取 structured_response 并编码为 JSON 文本写入 answer，不要求最后消息为 AIMessage，也不额外调用模型。原生 interrupt/resume 能通过构建后的图和 `Command` 使用；Worker 检测到暂停图时记录 `GRAPH_INTERRUPTED` 失败，HTTP 层不增加恢复或审批 API。失租、异常退出、重启不会认领已有 running Run：租约到期后记录失败，模型/工具操作不自动重放。正常关闭会中止本 Worker 的运行任务并记录 `WORKER_STOPPED`。

资源层为每个进程管理异步数据库池、异步 HTTP 客户端和 SDK 所需的同步客户端；执行只使用异步模型调用。Checkpointer 直接使用框架实现，初始化只在迁移入口进行。运行角色具有核心业务和 checkpoint 表的 DML 权限，没有表 ownership、schema CREATE、数据库 CREATE 或 revision 写权限。

每个 Run 独立构建图、权限策略和 middleware 实例。模型、HTTP 客户端、数据库池按进程复用；会话仅继承成功轮次消息，业务状态和运行资源不跨 Run 复制。Worker 核对提交绑定后，在有效租约下幂等持久化运行摘要，确认成功才开始调用图。配置摘要确认丢失只重试该数据库操作。详细扩展约束见 [运行扩展说明](runtime.md)。


阶段三继续使用标准 create_agent 与框架原生状态组合。注册表保存标准工具声明，Worker 在每轮将显式工具与原生中间件工具统一绑定权限，拒绝重复或未声明名称。Todo 属于原生 todos 状态，技能来自 API/Worker 启动时的不可变内存快照；两者共用现有模型/工具额度、可信参数验证、生命周期与输出限制。

RunContextMiddleware 在每次主模型调用前重建可信运行约束、技能名称/描述目录与当前 Todo，仅修改本次 ModelRequest；摘要只处理原生消息，不保存注入内容。原生 SummarizationMiddleware 继续处理工具调用配对与近期消息。运行配置快照记录上下文/Todo/技能/摘要版本；read_skill 的完整正文作为普通工具消息进入 checkpoint，摘要后可重新读取。

跨轮继承统一经过 messages 白名单和同会话成功最终 checkpoint 检查。历史 Todo/技能工具消息仅为历史，不恢复清单、不扩展技能白名单；其它状态和资源均重新创建。默认 chat 指纹保持兼容，新能力版本仅进入启用这些能力的 Profile 绑定。
