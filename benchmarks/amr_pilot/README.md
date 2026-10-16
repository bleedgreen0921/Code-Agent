# AMR 代码修复首轮预试

AMR-Benchmark 可以提供小规模代码修复任务素材。本目录从原仓库提取三个函数，增加确定性行为契约和验收器，用于测试 Agent 的读取、修改、验证闭环。这里评估的是代码任务执行能力，不是无线信号调制分类准确率，也不运行 AMR 模型训练。

## 源码筛选结论

已只读扫描 `/home/qjj/AMR/AMR-Benchmark` 的 209 个 Python 文件，文件内容去重后为 180 份；未发现现成测试套件、依赖锁文件。README 声明的环境为 Python 3.6.10、TensorFlow-gpu 1.14.0、Keras-gpu 2.2.4。完整训练还需要外部 AMR 数据集，部分路径固定为原作者的机器路径，因此本轮选取只依赖 NumPy 的函数。

重复实现不能当作独立任务：`calculate_confusion_matrix` 有 58 处定义，按 AST 去重只有 2 种；`l2_normalize` 有 17 处定义，按 AST 去重为 3 种。本轮每类行为只选一份。

原仓库已有本地修改和缓存，本轮没有导入或执行原目录的模块。三个选中文件的函数体原样提取，只补独立运行需要的 import 与来源注释；来源、行号和 SHA-256 见 [provenance.json](provenance.json)。完整复制前后文件哈希及 Git 状态保存在本机 `.data/amr-pilot/`，结束时已核对一致，见 [原仓库完整性记录](../../.data/amr-pilot/source-integrity.json)。

## 三个候选任务

| ID | 函数来源 | 明确要求 | 原实现表现 |
|---|---|---|---|
| normalize_zero | RML2018/DAE/main.py:24，l2_normalize | 正常向量保持 L2 归一化；零向量返回零；支持两个 axis；不修改输入 | 正常测试通过，零向量产生无效除法 |
| confusion_missing | RML201610a/CNN2/mltools.py:91，calculate_confusion_matrix | 缺失类别行置零；空输入返回零矩阵与标量计数；保留正常统计 | 正常测试通过，缺失类别归一化产生无效除法 |
| split_variable | RML201610a/CNN2/dataset2016.py:5，load_data | 支持各组不同样本数，按组 60%/20%/其余划分；无重叠、无遗漏、标签一致、可复现 | 每组 1000 条通过，小规模变长组索引越界 |

这些是基于真实代码定义的边界修复/行为扩展任务，新增要求见 [cases.json](cases.json)，不能将它们描述成已经过上游人工确认的缺陷。原始函数留在 [fixtures](fixtures) 中，不修改这些固定副本；每次模型运行另外创建工作副本。

## 验收规则

[evaluator.py](evaluator.py) 使用小数组和临时生成的 pickle，无需原始数据集。每题分别检查正常行为与目标边界：归一化检查零值/混合输入及两个 axis；混淆矩阵检查空输入、缺失类别及固定随机样本；划分检查多个组大小、索引覆盖、标签和复现性。

校准先运行原始副本，再运行参考实现。只有“原始正常测试通过、边界失败、参考实现全部通过”的任务才进入预试。本轮三题均满足。参考实现是用来校准验收器的可行实现，模型补丁无需与它相同。验收范围由这些契约与有限测试决定。

候选代码只在无网络、只读根文件系统、非 root、限制 CPU/内存/PID 的 Docker 容器内执行。容器只挂载当前候选目录和不可写的验收脚本，不挂载原 AMR 仓库、模型权重或宿主服务凭据。验收脚本和参考实现不通过模型工具提供；这是一轮普通功能修复预试，没有验证对抗性评分绕过。

## 真实模型结果（2026-10-16）

使用本机 `qwen3.5-0.8b`，BF16 GGUF、8192 上下文、temperature=0、关闭 thinking。每题从独立副本运行 3 次，每次最多 8 次主模型调用、8 次工具调用、1024 输出 token/请求、90 秒。模型工具为仅访问当前 candidate.py 的 read_file、整文件替换 edit_file，以及 Docker run_checks。最终评分在 Agent 执行结束后独立重跑验收，不采用模型的完成声明。

| 任务 | 验收通过 | 主要现象 |
|---|---:|---|
| normalize_zero | 0/3 | 重复改写，零向量问题仍存在，耗尽工具额度 |
| confusion_missing | 0/3 | 缺失类别问题未解决；部分修改将标量计数误改为矩阵 |
| split_variable | 0/3 | 工具参数 JSON 解析失败，服务返回 500；固定输出预算下长工具参数被截断 |
| 合计 | 0/9 | 本轮没有验收容器基础设施错误 |

记录见 [首次汇总](../../.data/amr-pilot/run-2b58f7e5ca82/summary.json) 与 [任务校准](../../.data/amr-pilot/calibration.json)，各任务目录保留修改后的源码、diff、用量和验收结果；异常请求的用量可能缺失，记录仅代表模型已报告值。首次批次异常退出的轨迹保存不足；随后仅增加 InMemorySaver 诊断记录，复跑 normalize_zero 一次仍失败，见 [诊断汇总](../../.data/amr-pilot/run-436974f95fe4/summary.json)。该额外一次不混入 9 次汇总，原批次记录没有覆盖。

结果属于当前模型、工具、提示词和预算的组合：整文件替换会放大长代码的输出预算需求，工具反馈与修改还可能位于同一并行批次。不能据此把所有失败归因于模型，也不能从三个任务推断通用代码能力。后续可冻结这三题，分别比较更强模型、补丁式编辑或更大的输出预算，每次只改变一个因素；保留失败和服务解析错误。

本轮复用当前项目的 build_agent、权限/生命周期中间件、运行上下文和原生工具循环，但使用三种预试专用可信工具；不覆盖正式工作区工具、API、Worker、PostgreSQL、Task 调度或完整仓库级任务。它可以作为首轮预试基线，正式效果基准仍需扩充多个仓库和独立任务。

## 复现

主机运行器使用已安装当前项目 requirements.lock 的 Python 环境；NumPy 仅安装在验收镜像，不增加项目生产依赖。本机此次使用 `/tmp/agent-phase6-clean-venv/bin/python`。

```bash
mkdir -p .data/amr-pilot/build
python -m pip download --only-binary=:all: --no-deps numpy==2.5.3 --dest .data/amr-pilot/build
docker build --network none -f benchmarks/amr_pilot/Dockerfile \
  -t local/amr-pilot:numpy-2.5.3 .data/amr-pilot/build
docker image inspect --format '{{.Id}}' local/amr-pilot:numpy-2.5.3 > .data/amr-pilot/image-id.txt
python -m benchmarks.amr_pilot.run calibrate
python -m benchmarks.amr_pilot.run pilot --repeats 3
```

模型服务默认 `http://127.0.0.1:1234/v1`，可用 `--base-url` 和 `--model` 修改；此运行器用于无需真实 Key 的本机端点。单题复现使用 `--case normalize_zero --repeats 1`。每批结果写入新的 `.data/amr-pilot/run-*`，不会覆盖既有批次。模型服务启停见 [本机模型验证记录](../../.data/local-model-smoke/README.md)。

fixtures 和 provenance 为固定输入，若要改变任务，应创建新版本并重新校准；后续批次汇总记录验收器、任务描述和运行器哈希。校准与预试不属于默认 pytest/CI，避免合成测试入口访问真实模型。
