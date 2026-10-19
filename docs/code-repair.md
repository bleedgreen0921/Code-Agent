# 阶段三：隔离修复与反馈循环

本阶段只输出候选，不写回用户工作区。`code_repair.enabled` 默认 false；请求必须显式声明 `delivery_mode=candidate_only`。仅支持任务启动时的当前工作区，包括 staged、unstaged 和非忽略 untracked 内容。历史 analysis / Review 任务最多八个，同工作区，作为可选上下文；不能继承其源码位置、复现状态或检查通过结论。

## 部署和可信注册

停旧 API / Worker，备份数据库，运行 `python -m db.migrate` 升级到 `core_0011`，再运行 `python -m db.bootstrap` 授权运行角色。历史迁移不变，新旧 Worker 不可混用。重新执行 `python -m scripts.build_execution_image`，将输出的不可变镜像 ID 同时配置给执行环境和验收定义。使用相同配置启动 API / Worker，并运行 `python -m db.doctor --json`、`python -m db.config_check --role agent-worker`。

配置示例：[runtime.code.repair.example.toml](../config/runtime.code.repair.example.toml)。所有启用代码阶段共享同一个既存、可写、与工作区不重叠的受管存储根目录。允许仅启用 repair。全局 `AGENT_RUN_TIMEOUT` 和 Task 时限仍会缩短修复的 900 秒默认上限；需要完整预算时配置相应全局时限。

项目开发检查和独立验收通过可信 Python 启动注册，HTTP / 模型不能注册或覆盖。沿用 `Registry.from_env(project_checks=...)` 和 `Resources(registry=...)` 的服务启动扩展点，增加 `acceptance_suites`：

```python
from agent_service.profiles import Registry
from agent_service.repair_registry import AcceptanceSuiteDefinition, acceptance_assets

suite = AcceptanceSuiteDefinition(
    id="sample-contract", version="1", workspaces=("project",),
    image="sha256:<实际执行镜像 ID>",
    asset_directory="sample", asset_digest=acceptance_assets("sample"),
    required_cases=("test_sample.TestAnswer.test_value",),
    application_module="sample", import_path=".",
)
registry = Registry.from_env(acceptance_suites=(suite,))
# 将同一 registry 注入 API 和 Worker 的 Resources；不接受来自项目配置的定义。
```

示例 `sample` 套件仅验收 `sample.answer()` 返回 1，不能用于其他项目需求的通用验收。新增项目验收时，在 **服务仓库** `execution/acceptance/<suite>/` 编写独立 unittest 用例，预构建入镜像，并注册用例方法的完整、非空 ID 集合。验收代码通过 `from acceptance_support import invoke` 调用固定模块的函数，参数和返回值使用 JSON；候选应用在独立子进程加载。当前适配器固定为 `python-acceptance-rpc-v1`，固定空应用环境，适用于 JSON 函数入口。其他语言、HTTP 应用或有状态协议需要先实现新的可信适配器，不能把项目测试命令登记为独立验收。

验收 runner 不导入候选测试、conftest、pytest 配置或插件。模型可修改开发测试，但无法改变镜像内的资产、冻结用例、导入路径和判定规则。资产身份、协议或必需用例集合损坏会终止执行；零用例、跳过、预期失败或不完整结果不能通过。注册验收只证明覆盖条件，不能自动证明模型的原因解释。

## API

```json
{
  "stage": "repair",
  "delivery_mode": "candidate_only",
  "task": "修复问题并补充回归测试",
  "workspace_id": "project",
  "target": {"kind": "workspace"},
  "context_task_ids": [],
  "check_suite_id": null,
  "acceptance_suite_id": "sample-contract"
}
```

提交到 `POST /v1/code/tasks`，所有新字段参与幂等摘要。无授权验收套件时拒绝创建。普通 Run、Task 和 Conversation 入口不能使用 `code-repair-isolated`。

复用 Task/Run 查询、取消、时间线、检查和产物接口，另有：

- `GET /v1/code/tasks/{id}/candidate`：基线、候选、当前检查点、冻结版本、预算、产物身份。
- `GET /v1/code/tasks/{id}/repair-attempts`：按游标分页的逐轮提交、检查和反馈。
- `GET /v1/code/tasks/{id}`：`delivery_mode=candidate_only`、`delivery_status=not_applied`，正式报告或失败诊断引用。

## 候选和版本

B 是任务启动时捕获的不可变工作区快照，保留 index 和原始 Git 对象上下文。W 是逐字节独立复制的可写源码。W 的 `.git` 由另一份独立 Git 副本只读覆盖挂载。原目录、B、版本库和产物根目录不挂载给模型。模型容器继续断网、非 root、只读根文件系统、资源限制和绝对截止时间看门狗。

修复使用独立 `RepairExecutionScope`，组合现有快照和子资源服务，不放开分析 / Review 的只读挂载。同任务的 W 和执行容器持续存在。源码、工具、Shell、冻结和检查启动在同一候选锁下串行执行。Shell 是受控前台命令，每次独立启动；普通非零退出保留开发反馈并保存发生的修改。Shell 超时、候选 OOM、失联属于控制故障；后台进程阻止检查点和验收，模型可用 Shell 清理后继续。

每次成功编辑及每个已结束 Shell 返回前，复制内容、核对一致性、生成并登记不可变检查点。登记确认丢失只重试不可变登记，不重复执行修改。当前实现每个检查点保存完整源码副本，按总量收费，未使用硬链接或增量去重。冻结检查读取独立只读版本 Rn，不读取 W。后续修改不会沿用旧版本的通过结果。

纳入规则由服务端冻结：B 中已有文件纳入；新增普通文件纳入；固定缓存目录（如 `__pycache__`、`.pytest_cache`、`.venv`、`node_modules`）和字节码排除，B 中已存在的文件优先保留。候选 `.gitignore` 无法隐藏变化。符号链接和特殊文件阻止封存。不复制原工作区的忽略文件。文件大小、文件数量和总接纳量是受管限额，不是宿主机磁盘硬配额。

## 图内验收

可写修复开始前执行一次 B 的独立验收，占用同一检查预算。基线通过仍允许修复，但报告明确没有观察到基线验收失败。

模型通过 ToolStrategy 单独提交 `RepairDraftV1`。`RepairGateMiddleware.aafter_model` 校验目标处理状态、已实际交付的源码引用和当前文件摘要，登记 `(Run, AI 消息)` 唯一 attempt，冻结 Rn，运行全部选定开发检查及独立验收。通过后关闭候选控制入口，结束同一次图调用。失败时保存检查身份、版本、真实日志、未完成检查和剩余预算，清除 `structured_response`，跳回同一图的 model 节点；消息、计数器、截止时间及 W 都不重建。

普通结束文本要求纠正；无效报告在原模型预算内纠正。报告与普通工具混合提交时拒绝整个批次并补齐工具错误回执。报告仍声明未完成目标、引用旧文件摘要或未实际交付的行号，均不能成功。每轮由唯一提交身份防止重复启动检查。图外只核对已通过记录、源码和产物字节，不重新验收。

默认预算：32 次模型、64 次普通工具、12 次注册检查（包含基线）、4 轮 gate、每次检查 60 秒、每轮 120 秒、运行 900 秒、清理发布预留 60 秒、候选及版本总接纳量 4 GiB。文件限额沿用阶段一。普通开发检查为至少一轮 gate 保留检查配额，创建时验证基线与一轮完整验收可放入预算。

## 产物、发布和恢复

基线另有 `repair_baseline` ZIP，包含 B 的源码和独立 Git 上下文，可直接下载。每个版本输出：`repair_manifest`、`repair_patch`、`repair_package`（ZIP，真实文件字节及权限模式），以及内容摘要。补丁计算 B→Rn，用户原有修改属于 B，不被当成 Agent 的改动。二进制、权限模式和无末尾换行等不能由文本补丁完整表达的变化在清单中标明。版本重建以清单和文件包为准。

服务端生成 `RepairReportV1` 及 Markdown，包含草稿、B/Rn 身份、各轮检查、验收资产、预算与 `not_applied`。发布事务重新验证源码、重建包、文本补丁、检查定义和结果、源码证据及最终 graph checkpoint，且要求有效租约和资源全部释放。发布确认丢失只核对已有结果，不再次验收。`completed` 仅表示候选成功发布。

取消优先。失败保留 `repair_diagnostic`、最后一致版本、补丁、包和真实执行记录；Run answer 保持为空。未安全捕获的尾部修改不能冒充最后检查点。Worker 重启清理登记容器、将中断检查 / gate 标为无法确认、保留版本并生成诊断，不自动重放 Agent 或任何修改、检查。

终态内容默认七天。活动依赖、未释放资源和持有产物读取事务阻止 GC。W、Git 副本、全部冻结版本及包都归属 Task；失去登记的过期准备目录也受限量 GC 管理。保留期结束后下载返回 410。

阶段四再实现原目录刷新、三方合并、合并后验证和可信写回。本阶段不创建交付日志或自动应用补丁。

## 验证

```bash
python -m scripts.postgres_integration
# 开发时可指定专项；同样拒绝真实 HTTP 和任何跳过。
python -m scripts.postgres_integration --tests tests/test_code_repair.py
```

集成目录导出 `repair-examples`，覆盖独立修复、历史 Review、连续两次失败后通过、开发测试削弱、旧结果失效、预算耗尽、取消恢复和已有用户改动。

真实模型人工验收需另行运行，不属于 SyntheticModel 自动结果。使用新建示例项目、独立验收和显式 `candidate_only` 请求，检查模型定位是否准确、回归测试是否有意义、限制是否诚实，同时核对原工作区无变化。本地没有可用真实模型连接时应保留此项待办，不能以合成模型通过代替人工验收。
