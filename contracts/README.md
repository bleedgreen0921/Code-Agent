# 核心 API

所有业务接口限本机访问。请求模型禁止未知字段。独立 Run 接受 `{"task":"...","profile":"chat","workspace_id":null}`，后两项可省略。会话创建接受可选 `title`、`profile`、`workspace_id`；轮次只接受 `{"task":"..."}`，绑定从会话读取。旧 `mode`、`tool_authorizations` 返回 422。状态为 `queued/running/cancelling/completed/failed/cancelled`。

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| POST | `/v1/runs` | 提交独立任务，202；幂等重放返回 200 |
| GET | `/v1/runs` | 列表，支持 status、created_after、created_before、limit、cursor |
| GET | `/v1/runs/summary` | 排队期限、运行租约及取消中任务计数 |
| GET | `/v1/runs/{run_id}` | 任务状态与结果 |
| POST | `/v1/runs/{run_id}/cancel` | 排队任务直接取消；执行中返回 cancelling/202 |
| GET | `/v1/runs/{run_id}/trace` | 提交绑定 submitted_binding、运行摘要 runtime_config、原生 checkpoint、消息、metadata、pending_writes，limit 默认 50、最大 200 |
| GET | `/v1/runs/{run_id}/timeline` | 按序列号排序的任务生命周期事件 |
| POST | `/v1/conversations` | 创建会话，201，可选 title、profile、workspace_id |
| GET | `/v1/conversations` | 会话列表，支持 limit、cursor |
| GET | `/v1/conversations/{id}` | 会话与轮次 |
| POST | `/v1/conversations/{id}/turns` | 提交轮次，202；幂等重放返回 200 |
| GET | `/v1/conversations/{id}/turns/{turn_id}` | 轮次状态与结果 |
| POST | `/v1/conversations/{id}/turns/{turn_id}/cancel` | 取消轮次 |
| GET | `/v1/workers` | Worker 心跳和在线状态 |
| GET | `/health` | 进程存活 |
| GET | `/health/ready` | 数据库基线、checkpoint 表和运行角色检查 |
| GET | `/health/accepting` | 在线 Worker 与剩余队列容量 |

提交可带 `Idempotency-Key`（1–200 字符，不能全为空白）。独立任务与各会话分别作为作用域；相同 Key、相同规范化输入返回原任务；省略 profile 与显式 chat 等价。task、profile 或 workspace_id 不同返回 409 `IDEMPOTENCY_CONFLICT`。重放不重新校验当前服务端配置、不重新执行；即使配置已修改或删除，仍返回原 Run。同会话只允许一个未完成任务，否则 409 `CONVERSATION_BUSY`。容量超限返回 429 `QUEUE_CAPACITY_EXCEEDED`。列表 cursor 是上一页给出的 UUID，按 UUID 降序分页。

未知 profile、workspace_id 或缺少 profile 要求的工作区返回 422 `INVALID_REQUEST`。会话固定 profile 和工作区 ID；每轮使用提交时的当前 profile 版本。根路径身份变化后，续轮返回 409 `CONFIGURATION_CHANGED`，需新建会话。

完成后 `result` 只含 `{"answer":"模型最终文本"}`；结构化输出也保持 answer 为字符串，内容为原生 structured_response 的 JSON 编码。其它状态 result 为 null。任务响应含 run_id/id、task、status、profile、workspace_id、conversation_id、turn_no、创建/开始/结束时间和 error_code。轮次的 turn_id 与 run_id 相同。没有 claims、citations、acceptance、token_usage。文档、证据、个人记忆接口均为 404。

终态不可取消或覆写；重复取消返回当前状态。取消与成功发布竞争时，由数据库事务先后顺序决定结果；已经进入 cancelling 的任务不可发布成功。执行超时为 `RUN_TIMEOUT`，排队超时为 `QUEUE_TIMEOUT`，失租为 `LEASE_EXPIRED`。Worker 执行前发现提交绑定变化或配置被删除，标记 failed / `CONFIGURATION_CHANGED`，不调用模型。框架限次为 `QUOTA_EXCEEDED`。模型认证、超时、限流和其它 SDK 错误分别为 `MODEL_AUTH_FAILED`、`MODEL_TIMEOUT`、`MODEL_RATE_LIMITED`、`MODEL_CALL_FAILED`，其它执行错误为 `EXECUTION_FAILED`。

错误响应统一为 `{"error":{"code":"...","message":"..."},"request_id":"..."}`，并提供 `X-Request-Id`。HTTP 层不提供 interrupt 审批或 resume 接口。

内置 workspace_read/workspace_write 必须指定已注册 workspace_id。HTTP 不接受宿主路径、镜像、Docker 参数或环境变量。工作区直接修改，取消和失败不回滚。所有工作区工具在无网络 Docker 容器内运行；空工具 chat 不依赖 Docker。

工具超时为 COMMAND_TIMEOUT；环境准备失败为 EXECUTION_ENVIRONMENT_UNAVAILABLE；执行容器或进程异常退出为 EXECUTION_ENVIRONMENT_LOST；正常执行后清理未确认时为 RESOURCE_CLEANUP_FAILED。清理确认前不发布 completed，未释放资源持续占用工作区和执行容量。timeline 包含 environment_preparing、environment_active、environment_closing、environment_cleanup_failed、environment_released、environment_recovered；这些事件不是新的 Run 状态。


## 阶段五补充

项目知识使用限定工作区的原生 PostgreSQL Store 只读视图；人工管理 API 以 ETag 和事务级锁完成创建、更新及删除。MCP 采用静态 HTTP 服务、冻结目录、Profile 白名单与每 Run 独立原生 MCPAdapter；外部写入先确认派发记录，未知结果不重发。core_0004 新增 external_operations，trace/timeline/doctor 提供关联核对。详细契约、配置、部署和故障处理见 [项目知识与外部工具](../docs/项目知识与外部工具.md)。
