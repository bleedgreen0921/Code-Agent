# Python 修复演示项目

此示例故意保留切片边界缺陷，初始测试失败是预期行为。只依赖 Python 标准库，离线执行。

`first_n(items, n)` 返回前 n 项列表，保留顺序，不修改输入；n=0 返回空列表，
n 超过输入长度时返回全部元素，n<0 抛出 ValueError。

- `sample.py`：候选实现，初始使用错误的 `n + 1` 切片终点。
- `tests/test_sample.py`：开发测试，覆盖零长度、正常长度、空输入及超过长度。
- 允许修改：`sample.py` 和与问题有关的开发测试。保护所有用户已有修改。
- 容器工作目录：`/workspace`。
- 目标检查：`python -B -m unittest tests.test_sample -v`。
- 相关回归：`python -B -m unittest discover -s tests -v`。
- 镜像：部署准备期构建的不可变 workspace-v2 镜像；无需运行期安装依赖。

准备一个独立副本作为注册工作区，禁止直接注册框架仓库中的本目录：

```bash
cp -a examples/python-repair /absolute/path/to/project
cd /absolute/path/to/project
git init -b main
git add .
git -c user.name='Repair Demo' -c user.email='demo@example.invalid' commit -m 'Initial repair demo'
```

普通修复使用 `config/runtime.code.example.toml` 中的 code-fix。
交付应包含修改原因、涉及文件、真实命令及退出状态、用户已有改动保留情况、遗留事项。
新增回归应先在修改前实现上实际失败，再验证修正后目标测试及相关回归通过。
开发检查不生成正式 verification 报告；正式门禁需显式选择 code-fix-checked。
