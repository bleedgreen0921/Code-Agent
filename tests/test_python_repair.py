"""Phase three: actual failing feedback and red/green regression in native budgets."""
import hashlib
import os
import shutil

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.app import app
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_code_profiles import ROOT, load_example, completed, submit
from tests.test_git_tools import git
from tests.test_profiles_policy import call

TARGET = 'python -B -m unittest tests.test_sample -v'
REGRESSION = 'python -B -m unittest discover -s tests -v'
BUG = 'return items[:n + 1]'
PARTIAL = 'return [] if n == 0 else items[:n + 1]'
SLICE = 'return items[:n]'
FINAL = 'if n < 0:\n        raise ValueError("n must be nonnegative")\n    return items[:n]'
NEGATIVE = '''import unittest
from sample import first_n

class NegativeTests(unittest.TestCase):
    def test_negative(self):
        with self.assertRaises(ValueError):
            first_n([1, 2, 3], -1)
'''


def prepare(tmp_path, monkeypatch, *, regression=False):
    registry, workspace = load_example(tmp_path, monkeypatch, image=os.environ['AGENT_TEST_EXECUTION_IMAGE'])
    shutil.copytree(ROOT / 'examples/python-repair', workspace, dirs_exist_ok=True)
    if regression:
        source = workspace / 'sample.py'
        source.write_text(source.read_text().replace(BUG, SLICE))
    (workspace / 'notes.txt').write_text('initial\n')
    git(workspace, 'init', '-q', '-b', 'main')
    git(workspace, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'add', '.')
    git(workspace, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial')
    (workspace / 'notes.txt').write_text('staged user content\n')
    git(workspace, 'add', 'notes.txt')
    (workspace / 'notes.txt').write_text('unstaged user content\n')
    (workspace / 'user-untracked.txt').write_text('user data\n')
    return registry, workspace


@pytest.mark.parametrize('regression', [False, True], ids=['partial-fix-feedback', 'new-red-green-regression'])
def test_project_repair_loop(database, tmp_path, monkeypatch, regression):
    registry, workspace = prepare(tmp_path, monkeypatch, regression=regression)
    source = (workspace / 'sample.py').read_text()
    before = {str(p.relative_to(workspace)): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    history = []
    checks = (f'{TARGET}\n{REGRESSION}\ngit diff --cached --stat\ngit diff --stat')
    answer = ('## 修复摘要\n完成局部修复。\n## 修改内容\nsample.py 切片边界' +
              ('与负数拒绝；新增 tests/test_negative.py。' if regression else '；真实失败后再次修正。') +
              '\n## 检查结果\n' +
              ('新增回归先退出 1，1 项失败；修改后目标 4 项及全部 5 项通过，退出 0。' if regression else
               '目标初始退出 1，2 项失败；第一次修改后退出 1，1 项失败；第二次修改后目标及回归各 4 项通过，退出 0。') +
              '\n未启用正式验证。\n## 遗留问题\n' +
              ('无。' if regression else '负数契约尚未覆盖和修复，留待独立回归任务。') +
              '用户已有暂存、未暂存和未跟踪内容均保留。')

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, 'resources', r, raising=False)

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            step = len(tools)
            common = [call('read_skill', {'name':'code-fix'}, 'skill'),
                      call('git_status', {}, 'before-status'),
                      call('git_diff', {'source':'staged'}, 'before-diff'),
                      call('read_file', {'path':'sample.py'}, 'source')]
            if step < 4:
                return common[step]
            digest = tools[3].artifact['sha256']
            if regression:
                if step == 4:
                    return call('edit_file', {'path':'tests/test_negative.py', 'create':True, 'new_text':NEGATIVE}, 'new-test')
                if step == 5:
                    return call('shell', {'command':'python -B -m unittest tests.test_negative -v'}, 'red')
                if step == 6:
                    assert tools[-1].artifact['exit_code'] == 1 and 'FAILED (failures=1)' in tools[-1].text
                    history.append('red')
                    return call('edit_file', {'path':'sample.py', 'expected_hash':digest, 'old_text':SLICE, 'new_text':FINAL}, 'fix')
                if step == 7:
                    return call('shell', {'command':checks}, 'green')
                if step == 8:
                    assert 'Ran 4 tests' in tools[-1].text and 'Ran 5 tests' in tools[-1].text
                    assert tools[-1].artifact['exit_code'] == 0 and 'FAILED' not in tools[-1].text
                    return call('read_file', {'path':'tests/test_negative.py'}, 'confirm-test')
            else:
                if step == 4:
                    return call('shell', {'command':TARGET}, 'initial-red')
                if step == 5:
                    assert tools[-1].artifact['exit_code'] == 1 and 'FAILED (failures=2)' in tools[-1].text
                    return call('edit_file', {'path':'sample.py', 'expected_hash':digest, 'old_text':BUG, 'new_text':PARTIAL}, 'partial')
                if step == 6:
                    return call('shell', {'command':TARGET}, 'still-red')
                if step == 7:
                    assert tools[-1].artifact['exit_code'] == 1 and 'FAILED (failures=1)' in tools[-1].text
                    history.append('red')
                    return call('edit_file', {'path':'sample.py', 'expected_hash':hashlib.sha256(source.replace(BUG, PARTIAL).encode()).hexdigest(), 'old_text':PARTIAL, 'new_text':SLICE}, 'fix')
                if step == 8:
                    return call('shell', {'command':checks}, 'green')
            if step == 9:
                if not regression:
                    assert tools[-1].artifact['exit_code'] == 0 and tools[-1].text.count('Ran 4 tests') == 2
                    assert 'FAILED' not in tools[-1].text
                return call('git_status', {}, 'after-status')
            assert step == 10
            return AIMessage(answer)

        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            run_id = await submit(client, 'code-fix')
            tools = await completed(client, r, worker, run_id, answer)
        assert len(tools) == 10 and len(model.history) == 11
        assert history == ['red']
        assert (workspace / 'sample.py').read_text() == source.replace(SLICE if regression else BUG, FINAL if regression else SLICE)
        for name, data in before.items():
            if name != 'sample.py':
                assert (workspace / name).read_bytes() == data, name
        assert 'notes.txt' in tools[-1].text and 'user-untracked.txt' in tools[-1].text
    database(check)


def test_development_check_feedback_distinguishes_missing_and_unexecuted(database, tmp_path, monkeypatch):
    registry, workspace = prepare(tmp_path, monkeypatch)
    answer = ('## 修复摘要\n未修改。\n## 修改内容\n无。\n## 检查结果\n'
              'missing-check-command 退出 127，命令缺失，检查未正常执行；目标测试未执行。'
              '\n## 遗留问题\n需执行目标测试；用户已有改动保留。')
    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, 'resources', r, raising=False)
        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call('shell', {'command':'missing-check-command'}, 'missing')
            assert tools[-1].artifact['exit_code'] == 127
            assert 'command not found' in tools[-1].text and 'FAILED' not in tools[-1].text
            return AIMessage(answer)
        worker = Worker(r, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            await completed(client, r, worker, await submit(client, 'code-fix'), answer)
    database(check)
