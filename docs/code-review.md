# 固定范围 Review

Review 复用代码任务队列、受管快照、独立检查容器和原子报告发布。默认关闭；不编辑、修复或写回项目。请求省略 stage 时仍为阶段一 analysis，旧报告继续可读。

## 配置与部署

使用 `config/runtime.code.review.example.toml`；API、Worker 和 doctor 使用同一配置和 `scripts.code_analysis_example` 的可信 Python 检查注册。Review 和 analysis 的检查授权分别配置，启用两者时必须共用同一受管存储目录。

停止旧 API/Worker，备份数据库，然后运行：

```bash
python -m db.migrate
python -m db.bootstrap
python -m scripts.build_execution_image
# 将上一步不可变镜像 ID 写入配置
export AGENT_RUNTIME_CONFIG=/absolute/runtime.toml
export AGENT_RUN_TIMEOUT=600
python -m scripts.code_analysis_example --role doctor
```

迁移为 `core_0010`，保留旧结构签名；新增 range 和 reproduction 表与运行时授权。禁止新旧 Worker 混跑 Review。全部部署验收及真实模型人工语义验收通过后，将 `code_review.enabled` 设为 true，再以相同配置启动 API/Worker。配置默认关闭不意味着 Review 可以使用普通文本报告绕过校验。

## 创建与查询

```json
{"stage":"review","task":"审查改动引入的行为问题","workspace_id":"project","target":{"kind":"workspace","mode":"all"}}
```

发送到 `POST /v1/code/tasks`，可用 `Idempotency-Key`。`context_task_ids` 最多 8 个，只允许同工作区的 analysis 或 Review 任务；历史报告不继承当前证据或检查状态，截断保留标记。`check_suite_id` 只能选择注册且授权的套件。请求不接受命令、宿主路径、镜像或 Git 算法参数。

| target | 比较 base | 执行副本和 head |
|---|---|---|
| workspace/all（默认） | 捕获的 HEAD，无提交为空树 | staged + unstaged + 非忽略 untracked |
| workspace/staged | 捕获的 HEAD，无提交为空树 | index 原始 blob；不读取未使用的工作区内容 |
| workspace/unstaged | 捕获的 index | 工作区副本；差异仅包含 index 路径，untracked 仅作执行上下文 |
| commit/ref | 指定提交第一父提交；根提交为空树 | 指定提交原始 tree/blob |
| range/base/head/comparison=direct | 指定 base 提交 | 指定 head 提交，不要求祖先关系 |
| range/base/head/comparison=merge_base | 唯一最佳共同祖先 | 指定 head；无共同祖先或多个最佳结果失败 |

单提交示例：`{"kind":"commit","ref":"HEAD~2"}`。范围示例：`{"kind":"range","base":"main","head":"feature","comparison":"merge_base"}`。所有引用在准备阶段固定；封存后移动分支不改变范围。报告记录请求引用、解析提交、实际比较 base、两侧 tree、内容摘要和时间。unstaged 额外记录只用于差异的 `diff_head_tree`，与包含 untracked 上下文的实际 `head_tree` 区分。

沿用任务详情、取消、时间线、checks、artifacts 接口，新增：

- `GET /v1/code/tasks/{id}/changes?offset=0&limit=30`：固定差异清单，跟随 `next_offset`。
- `GET /v1/code/tasks/{id}/reproductions?limit=30&cursor=...`：复现记录。
- `GET /v1/code/tasks/{id}/artifacts/{artifact_id}`：JSON、Markdown、完整固定差异、脚本、执行结果和日志。

任务详情提供 resolved_range、finding_count、coverage 和执行说明；正式报告过期后仍保留任务元数据和产物墓碑。跨任务产物访问被拒绝。GC 保留活动依赖和未释放资源引用。

## 工具与证据

新 Profile 为 `code-review-isolated`，仅专用代码任务入口可用，不连接旧固定验证器。保留 head 搜索、普通读取、Git log、注册检查、Skill 和规划，移除自由 Git diff/read。

`review_diff` 只读固定清单或指定 change_id 补丁，分页 offset 是输出记录位置。每条补丁记录包含 old_line/new_line；问题位置不能使用 output_line。重命名按删除加新增处理。二进制、过长行、不可读取的 base 类型、单文件补丁或总差异预算超限都记录限制，不伪造完整覆盖。

`review_read_file` 固定 side=base/head，支持已删除文件。完整源码行先按消息预算裁剪，再在最外层工具中间件核对实际送达内容，持久化送达回执。被中间件更改、截断或仅保存在产物中的内容不计入已读。补齐普通分页可以消除未读限制；硬截断、未补齐页面和必要上下文不足不能消除。

`read_review_execution` 分页读取当前 Run 的 result/log。复现摘要里的 details 可能截断，完整观察可从 result 分页或结果产物读取；摘要明确标记。注册检查的问题引用必须有具体失败用例和同一次执行的诊断日志；整个套件失败不能冒充问题已复现。

## 临时复现

`run_review_reproduction` 只接受 hypothesis 和最多 16 KiB 的 Python：

```python
def reproduce():
    from sample import answer
    actual = answer()
    return {"observed": actual == 2, "details": f"actual={actual}"}
```

只执行 head。源码只读，脚本和临时文件位于 `/tmp`。独立容器断网、非 root、只读根文件系统，并沿用资源限制；不挂载原目录、凭据、产物根或 Docker socket。正常返回 observed=true/false 对应 observed/not_observed；未处理异常、缺依赖、非法返回、超时、OOM、协议或结果不完整均为 inconclusive。stdout/stderr 进入日志，不作为可信结果协议。

脚本、假设、摘要、runner 版本、资源和调用 ID 在执行前登记。检查与复现共享 12 次预算，其中复现最多 4 次，每次最多 60 秒。确认丢失只核对原执行或重试不可变结果写入，不重新启动。清理确认后才返回结果；取消、失租、环境丢失或清理失败阻止完成和正式发布。重启的维护流程清理登记资源，并将中断复现标记 inconclusive。

复现是模型编写的观察脚本产生的证据，不自动证明问题由改动引入。base 永不执行；引入依据只能来自静态前后对照，报告明确没有运行前后对照。

## 报告与验收

模型必须返回 `ReviewReportV1`：summary、findings、changes 逐项声明、unconfirmed、missing_context。正式问题包含 P0—P3 优先级、真实变化位置、触发条件、原因和影响、差异及两侧源码证据、引入依据、复现状态和建议。已删除问题定位 base；真实权限或空文件变化允许文件级位置。不变代码只能作为辅助上下文。

服务端独立计算覆盖，不接受模型 complete 字段。覆盖包括清单与补丁/hunk 实际送达、逐项声明、有效源码上下文、主动排除和必要上下文缺失。最终区分：发现问题；发现问题且覆盖不足；范围内未发现有证据支持的问题；覆盖不足不能作完整无问题结论。空范围仅显示“该比较范围没有变化”。未运行检查不使静态覆盖自动不完整，但明确显示未执行验证。

发布前重新读取固定范围、封存摘要、源码/差异送达回执及检查/复现持久化记录，再计算覆盖和确定性 Markdown；清理完成后与 Run 完成状态同事务发布。预算耗尽前模型可提交可校验的部分报告；没有有效结构化报告则失败，保留诊断，不重建 Run 重置预算。

自动验收命令：`python -m scripts.postgres_integration`。使用真实 PostgreSQL、Git、Docker 和 SyntheticModel，禁止真实模型 HTTP、跳过用例和未登记执行容器。Review 样例导出可通过 `AGENT_REVIEW_EXAMPLE_DIR` 指定目录；默认随测试临时目录保存。

真实模型人工验收应另外核对：问题是否实际成立、触发条件是否明确、是否由本次改动引入、建议是否可执行。确定性脚本观察和结构校验不能替代这一步。未完成该项之前保持默认关闭。

大量变更的正式报告仅内嵌前 32 项覆盖明细并标记 `details_truncated`；完整逐项覆盖记录保存在 `review_coverage` 产物中，发布时重新计算并核对其摘要。报告摘要的截断不改变服务端计算的覆盖结论，也不会把未送达模型的内容计为已读。

问题的 `observed` 也可由本 Run 的具体失败注册用例支持，但必须同时引用 `check_case_refs`，且该用例失败事件与同次执行诊断日志都已送达。没有具体诊断的套件失败不能使用这一状态。临时复现和注册检查的来源在报告中分别列出。
