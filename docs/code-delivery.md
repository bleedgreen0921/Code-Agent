# 阶段四：最终验证、合并、自动写回与恢复

`candidate_only` 仍只导出候选，既有请求不会自动升级。自动模式需要请求中的 `delivery_mode="auto_apply"` 和服务端 `code_repair.auto_apply_workspaces` 同时授权，默认白名单为空。模型工具只能修改隔离候选。

## 配置和部署

停止旧 API/Worker，备份数据库及受管存储，再依次执行：

```bash
python -m db.migrate
python -m db.bootstrap
python -m db.doctor --json
python -m db.config_check --role agent-worker
```

迁移头为 `core_0012`，保留历史迁移与结构签名。禁止新旧 Worker 混跑。沿用固定执行镜像及可信开发检查、独立验收注册。启动时固定宿主 `/usr/bin/git` 的版本和可执行文件摘要，普通文本使用受控 `git merge-file --diff3`，不使用项目 merge driver、hooks 或过滤器。

为测试工作区创建私有目录：

```bash
install -d -m 0700 /same-filesystem/service-private/delivery
```

在 `[code_repair.auto_apply_workspaces]` 下配置 `project = "/same-filesystem/service-private/delivery"`。目录必须由服务账号持有、位于所有注册工作区之外，和工作区处于同一本机 Linux 文件系统。启动原子探针验证 `renameat2` 的 exchange/no-replace 支持。嵌套挂载、跨文件系统、特殊文件、硬链接、ACL/扩展属性、无法保真的权限不允许降级复制。完整示例见 `config/runtime.code.repair.example.toml`。

阶段三真实模型语义验收仍未通过；本阶段的确定性测试不能替代上线所需的真实模型验收，生产白名单应继续保持关闭。

## 请求与查询

```json
{
  "stage": "repair",
  "delivery_mode": "auto_apply",
  "task": "修复指定问题并自动交付到原工作区",
  "workspace_id": "project",
  "target": {"kind": "workspace"},
  "check_suite_id": "project-development",
  "acceptance_suite_id": "project-acceptance"
}
```

不配置开发套件时可以省略 `check_suite_id`；独立验收仍然必需。正常流程不增加人工确认或 apply 调用。

- `GET /v1/code/tasks/{id}`：Task、代码 phase、候选和实际交付状态。
- `GET /v1/code/tasks/{id}/delivery`：B/P/A/U/M 关联、观察、合并、计划、实际逐文件状态、占用和恢复诊断。
- `GET /v1/code/tasks/{id}/delivery/events?limit=100&cursor=...`：持久事件分页。
- 既有 artifacts 接口：候选 B→A 包、相对滚动 P 的候选差异、最终 U→M 的清单/补丁/包、候选证据、实际检查结果和正式 `DeliveryReportV1`。

没有公共 apply 指定补丁或任意 checkpoint resume 接口。

## 版本、检查和预算

B 永久指向修复起点。A 是每次提交的冻结候选，U 是准备交付时可信捕获的原目录，M 是交付版本。P 初始为 B；合并检查失败并完成候选安装核对后才更新为该次 U，下一次使用 `merge(P,A_next,U_next)`。用户改动的区间、文件存在状态和模式来源在后续回流中保留；无法安全定位的变化保守阻止自动交付。

A 通过开发检查与独立验收后才进入合并。U 等于 P 时可使用同一冻结 A 的 gate 记录；产生 M 时，即使摘要碰巧等于 A，也新执行所有选定开发检查和独立验收。M 失败后在原可写目录继续同一 Agent 图，安装前后分别持久记录，不把混合目录当成有效版本。

检查成功后保存唯一计划，通过 LangGraph 原生中断交给宿主组件，释放全部容器，再预检和写回。清理后 U 再变时内部恢复同一图，登记新的受控容器代次并重新验证。已经 applied 的恢复路径只处理交付结果及发布，禁止模型和新检查。

总上限沿用 900 秒并受全局 Run/Task 时限约束；32 次模型、64 次工具、12 次检查、4 轮 gate，自动模式另以 Run 绑定的数据库计数约束所有内部恢复。初次 U 不计刷新，随后跨轮最多三次。模型和检查截止时间预留最后 120 秒；首次待交付计划固定最长 60 秒写回截止时间，刷新或重启不延长。配置须能容纳基线检查、A 检查、M 检查与交付预留。观察、合并、导出、暂存和备份计入同一受管存储预算。

## 写回、取消与恢复

工作区数据库占用在容器释放后仍有效；重叠目录任务不能启动。可信写回和恢复同时需要数据库租约与同一私有 `flock`，旧进程仍持锁时不能接管。这不是对外部编辑器的强制锁。

文件事务使用目录 FD、无跟随访问、父目录和根身份核对。新增使用原子 no-replace；替换原子交换并保留实际被替换的 inode；删除移入私有隔离区。每次执行前记录意图，执行后 fsync 并记录实际状态。原目录 Git index 不执行 add、commit、reset、checkout 或历史恢复。

| 交付状态 | 含义 |
| --- | --- |
| pending | 准备或交接，尚无正向文件变化 |
| applying | 已持久记录写入意图，等待实际操作或确认 |
| applied | 全部文件和最终上下文核对完成；历史事实 |
| rolled_back | 安全撤销本次影响，或确认没有正向写入 |
| needs_recovery | 无法安全继续/撤销，保留版本、备份和工作区占用 |

独立 `impact` 为 `none/partial/all/unknown`。Task 失败或取消不能推断原文件没有变化。applied 后的正常用户编辑不改变历史交付事实。

重启 Worker 仅接管原交付日志：证据、原生 checkpoint、容器释放、身份、截止时间有效且现场可解释为部分 U / 部分 M 时继续；已达 M 的路径只确认，不重复交换。第三种内容或类型不调用模型猜测合并。无法继续时只回滚仍属于本次写入的文件，不覆盖用户后续修改。回滚中断也可再次核对。取消和超时阻止后续正向写入；已持久 applied 不追溯撤销。

可信宿主管理 CLI：

```bash
python -m scripts.delivery inspect TASK_UUID
python -m scripts.delivery recover TASK_UUID
python -m scripts.delivery rollback TASK_UUID
```

CLI 必须使用原服务配置与账号。同名工作区若改指其他路径，恢复会拒绝继续修改旧目录，需要先恢复原交付配置。`inspect` 只读核对计划、现场和备份；`recover` 在原条件有效时继续；`rollback` 请求安全回滚。没有 force、accept-current、强制解锁或保留现状确认。如果日志无法解释现场，先自行保存新修改并人工恢复相关路径，再重新核对；占用在此之前保留。

交付成功但正式报告发布中断时保留 applied，仅恢复最终 checkpoint、结果校验和正式发布。正式完成事务核对 A 的报告证据、M 的检查、全部容器释放、applied 与文件日志。报告明确 A 的源码行号和 M 的真实交付内容，不能互相冒用。

正常终态内容保留七天。pending/applying/needs_recovery，以及 applied 但尚未发布的产物不因 Task 终态或保留期到期而删除。
