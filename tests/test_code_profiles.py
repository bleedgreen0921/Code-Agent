"""Delivered config/skills, synthetic decisions, real API/PostgreSQL/Docker tools."""

import hashlib
import os
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.app import app
from agent_service.execution import DockerClient
from agent_service.profiles import DEFAULT_BINDING, Extension, Registry
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_core import execute
from tests.test_profiles_policy import call

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("code-analyze", "code-review", "code-fix")
READ_TOOLS = {"glob_search", "grep_search", "read_file", "read_skill", "write_todos"}
COMMAND = "python -B -m unittest discover -s tests -v"
BUGGY = "def first_n(items, n):\n    return items[:n + 1]\n"
FIXED = "def first_n(items, n):\n    return items[:n]\n"
UNITTEST = """import unittest
from sample import first_n


class FirstNTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(first_n([1, 2, 3], 0), [])

    def test_normal(self):
        self.assertEqual(first_n([1, 2, 3], 2), [1, 2])

    def test_empty(self):
        self.assertEqual(first_n([], 2), [])
"""


def load_example(tmp_path, monkeypatch, *, image=None):
    """Only substitute the three deployment placeholders in the shipped file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = (ROOT / "config/runtime.code.example.toml").read_text()
    replacements = {
        "/ABSOLUTE/PATH/TO/project": str(workspace),
        "/ABSOLUTE/PATH/TO/agent/project_skills": str(ROOT / "project_skills"),
        "sha256:REPLACE_WITH_64_HEX_DIGITS": image or "sha256:" + "0" * 64,
    }
    for old, new in replacements.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    config = tmp_path / "runtime.toml"
    config.write_text(text)
    monkeypatch.setenv("AGENT_RUNTIME_CONFIG", str(config))
    return Registry.from_env(), workspace


def files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


async def inspect_mount(r, run_id, workspace, writable):
    resource = await r.store.execution_resource(run_id)
    client = DockerClient()
    obj = await client.inspect(resource["container_id"])
    assert len(obj["Mounts"]) == 1
    mount = obj["Mounts"][0]
    assert mount["Source"] == str(workspace)
    assert mount["Destination"] == "/workspace" and mount["RW"] is writable
    if not writable:
        code, _, stderr = await client.call(
            "exec",
            obj["Id"],
            "python",
            "-B",
            "-c",
            "open('/workspace/sample.py', 'w').write('forbidden')",
            check=False,
        )
        assert code != 0 and b"Read-only file system" in stderr


async def submit(client, profile):
    missing_workspace = await client.post(
        "/v1/runs", json={"task": "检查指定代码", "profile": profile}
    )
    assert missing_workspace.status_code == 422, missing_workspace.text
    response = await client.post(
        "/v1/runs",
        json={
            "task": "检查 sample.py 的 first_n；代码修复时运行既有单元测试。",
            "profile": profile,
            "workspace_id": "project",
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


async def completed(client, r, worker, run_id, answer):
    row = await execute(r, worker)
    assert str(row["id"]) == run_id
    assert row["status"] == "completed", row
    response = await client.get("/v1/runs/" + run_id)
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == {"answer": answer}
    assert body["verification"] is None and row["verification"] is None
    state = await worker.agent.aget_state(row["final_checkpoint"])
    assert state.values["messages"][-1].text == answer
    resource = await r.store.execution_resource(row["id"])
    assert resource["state"] == "released"
    assert await DockerClient().inspect(resource["container_id"]) is None
    return [m for m in state.values["messages"] if isinstance(m, ToolMessage)]


def test_delivered_config_skills_and_default_compatibility(tmp_path, monkeypatch):
    registry, _ = load_example(tmp_path, monkeypatch)
    assert set(registry.profiles) == {
        *NAMES,
        "chat",
        "workspace_read",
        "workspace_write",
    }
    assert set(registry.skills) == set(NAMES)
    assert not registry.extensions
    assert not registry.knowledge.enabled and not registry.mcp_tools
    for name in NAMES:
        profile = registry.profiles[name]
        assert set(profile.tools) == READ_TOOLS | (
            {"edit_file", "shell", "git_status", "git_diff"} if name == "code-fix" else
            {"git_status", "git_log", "git_diff", "git_read_file"} if name == "code-review" else set()
        )
        assert profile.skills == (name,) and profile.requires_workspace
        with pytest.raises(ValueError, match="requires a workspace"):
            registry.bind(name)
        binding = registry.bind(name, "project")
        assert registry.resolve(binding) == profile
        assert "verification" not in binding and "no_progress" not in binding
        extension = registry.extensions.get(name, Extension())
        assert extension.verification_mode is None and not extension.no_progress_enabled
        skill = registry.skills[name]
        assert len(skill.body.encode()) <= 6 * 1024
        assert len(skill.body.splitlines()) <= 80
        assert skill.sha256 == hashlib.sha256(skill.body.encode()).hexdigest()
    for name in ("workspace_read", "workspace_write"):
        assert set(registry.profiles[name].skills) == set(NAMES)
    monkeypatch.setenv(
        "AGENT_RUNTIME_CONFIG", str(ROOT / "config/runtime.example.toml")
    )
    original = Registry.from_env()
    assert not original.skills and not set(NAMES) & original.profiles.keys()
    monkeypatch.delenv("AGENT_RUNTIME_CONFIG")
    default = Registry.from_env()
    assert default.bind() == DEFAULT_BINDING == registry.bind()
    assert set(default.profiles) == {"chat", "workspace_read", "workspace_write"}
    for name in default.profiles:
        assert original.profiles[name] == default.profiles[name]


@pytest.mark.parametrize("profile", NAMES[:2])
@pytest.mark.parametrize("buggy", [True, False], ids=["boundary-bug", "no-findings"])
def test_readonly_code_api_tools_and_skill_isolation(
    database, tmp_path, monkeypatch, profile, buggy
):
    registry, workspace = load_example(
        tmp_path, monkeypatch, image=os.environ["AGENT_TEST_EXECUTION_IMAGE"]
    )
    (workspace / "sample.py").write_text(BUGGY if buggy else FIXED)
    before = files(workspace)
    finding = (
        "sample.py:2：n=0 且输入非空时返回一项，违反返回前 n 项的约定。"
        if buggy
        else "未发现可确认问题"
    )
    answer = (
        "## 结论\n" + finding + "\n## 关键代码与调用关系\nsample.py:1 定义 first_n。"
        "\n## 执行和异常路径\nsample.py:2 直接返回切片。"
        "\n## 证据与未确认事项\n仅静态读取 sample.py；未执行测试，未接入可信最终验证。"
        if profile == "code-analyze"
        else "## 审查结论\n已审查指定函数。\n## 问题清单\n"
        + finding
        + "\n## 审查范围与未确认事项\n仅 sample.py；未执行测试，未接入可信最终验证。"
    )

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, "resources", r, raising=False)
        responses = [
            call("read_skill", {"name": profile}, "skill"),
            *[
                call("read_skill", {"name": other}, other)
                for other in NAMES
                if other != profile
            ],
            call("glob_search", {"pattern": "*.py"}, "glob"),
            call(
                "grep_search", {"pattern": "first_n", "output_mode": "content"}, "grep"
            ),
            call("read_file", {"path": "sample.py"}, "read"),
            call(
                "edit_file",
                {"path": "forbidden", "create": True, "new_text": "bad"},
                "edit",
            ),
            call("shell", {"command": "touch forbidden-shell"}, "shell"),
            AIMessage(answer),
        ]

        async def reply(messages):
            if len(responses) == 9:
                await inspect_mount(r, run_id, workspace, False)
            return responses.pop(0)

        worker = Worker(r, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            run_id = await submit(client, profile)
            tools = await completed(client, r, worker, run_id, answer)
        assert len(tools) == 8
        assert tools[0].text == registry.skills[profile].body
        assert tools[0].artifact == {
            "name": profile,
            "sha256": registry.skills[profile].sha256,
        }
        assert all(
            m.status == "error" and "SKILL_NOT_AVAILABLE" in m.text for m in tools[1:3]
        )
        assert "/sample.py" in tools[3].text
        assert "/sample.py:1:def first_n" in tools[4].text
        assert "2:     return items[:n" in tools[5].text
        assert (
            tools[5].artifact["sha256"]
            == hashlib.sha256(before["sample.py"]).hexdigest()
        )
        assert all(m.status == "error" for m in tools[6:])
        assert files(workspace) == before

    database(check)


def test_code_fix_api_real_failing_tests_hash_edit_and_passing_tests(
    database, tmp_path, monkeypatch
):
    registry, workspace = load_example(
        tmp_path, monkeypatch, image=os.environ["AGENT_TEST_EXECUTION_IMAGE"]
    )
    (workspace / "sample.py").write_text(BUGGY)
    (workspace / "tests").mkdir()
    (workspace / "tests/test_sample.py").write_text(UNITTEST)
    (workspace / "notes.txt").write_text("用户已有内容，不能修改。\n")
    before = files(workspace)
    answer = (
        "## 修复摘要\n修正返回前 n 项的边界错误。\n## 修改内容\n"
        "sample.py:2 将切片终点改为 n。\n## 检查结果\n"
        f"`{COMMAND}`：修改前退出 1，两项失败；修改后退出 0，三项通过。"
        "未接入可信最终验证。\n## 遗留问题\n无。"
    )

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, "resources", r, raising=False)

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            step = len(tools)
            if step == 0:
                await inspect_mount(r, run_id, workspace, True)
                return call("read_skill", {"name": "code-fix"}, "skill")
            if step in (1, 2):
                return call("read_skill", {"name": NAMES[step - 1]}, f"denied-{step}")
            if step == 3:
                return call("read_file", {"path": "sample.py"}, "source")
            if step == 4:
                return call("read_file", {"path": "tests/test_sample.py"}, "tests")
            if step == 5:
                return call("shell", {"command": COMMAND}, "before")
            if step == 6:
                assert tools[-1].status == "error"
                assert tools[-1].artifact["exit_code"] == 1
                assert "FAILED (failures=2)" in tools[-1].text
                assert "Ran 3 tests" in tools[-1].text
                return call(
                    "edit_file",
                    {
                        "path": "sample.py",
                        "old_text": "return items[:n + 1]",
                        "new_text": "return items[:n]",
                        "expected_hash": tools[3].artifact["sha256"],
                    },
                    "edit",
                )
            if step == 7:
                assert tools[-1].status == "success"
                return call("shell", {"command": COMMAND}, "after")
            assert step == 8
            assert tools[-1].status == "success"
            assert tools[-1].artifact["exit_code"] == 0
            assert "Ran 3 tests" in tools[-1].text and "OK" in tools[-1].text
            return AIMessage(answer)

        worker = Worker(r, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            run_id = await submit(client, "code-fix")
            tools = await completed(client, r, worker, run_id, answer)
        assert tools[0].text == registry.skills["code-fix"].body
        assert tools[0].artifact["sha256"] == registry.skills["code-fix"].sha256
        assert all(
            m.status == "error" and "SKILL_NOT_AVAILABLE" in m.text for m in tools[1:3]
        )
        assert (
            tools[3].artifact["sha256"]
            == hashlib.sha256(before["sample.py"]).hexdigest()
        )
        assert (
            tools[4].artifact["sha256"]
            == hashlib.sha256(before["tests/test_sample.py"]).hexdigest()
        )
        assert files(workspace) == {**before, "sample.py": FIXED.encode()}

    database(check)
