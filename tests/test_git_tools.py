"""Deterministic Git fixtures; all tool reads execute in real isolated containers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.app import app
from agent_service.execution import DockerClient
from agent_service.execution_config import WORKSPACE_TOOLS, WRITE_TOOLS
from agent_service.profiles import Profile
from agent_service.worker import Worker
from agent_service.workspace_tools import GitDiffInput, GitLogInput, GitReadFileInput, GitStatusInput
from tests.models import SyntheticModel
from tests.test_code_profiles import completed, inspect_mount, load_example
from tests.test_core import execute
from tests.test_execution import finish_scope, install, scope_for
from tests.test_profiles_policy import call

GIT_TOOLS = {"git_status", "git_log", "git_diff", "git_read_file"}


def git(root, *args, data=None):
    return subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, check=True).stdout.decode().strip()


def init(root, *, commit=True):
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Git Test")
    if commit:
        (root / "sample.py").write_text("root\n")
        git(root, "add", ".")
        git(root, "commit", "-qm", "root title\n\nprivate body omitted")
        return git(root, "rev-parse", "HEAD")


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


async def read(scope, name, **args):
    result = await scope.file_call(name, args)
    assert "error" not in result, result
    assert len(result["content"].encode()) <= scope.settings.output_bytes
    assert len(result["content"].splitlines()) <= scope.settings.output_lines
    if name.startswith("git_"):
        assert len(json.dumps({k: v for k, v in result.items() if k != "content"}, ensure_ascii=False).encode()) <= scope.settings.output_bytes
    return result


async def error(scope, name, code, **args):
    result = await scope.file_call(name, args)
    assert result.get("code") == code, result
    assert result.get("error")
    return result


@pytest.mark.parametrize("schema,args", [
    (GitStatusInput, {"offset": -1}), (GitStatusInput, {"limit": 101}),
    (GitStatusInput, {"limit": "2"}), (GitStatusInput, {"command": "status"}),
    (GitLogInput, {"limit": 51}), (GitLogInput, {"offset": -1}),
    (GitDiffInput, {"source": "HEAD"}), (GitDiffInput, {"source": "commit"}),
    (GitDiffInput, {"source": "staged", "commit": "HEAD"}),
    (GitDiffInput, {"source": "staged", "view": "patch"}),
    (GitDiffInput, {"source": "staged", "start_line": 0}),
    (GitDiffInput, {"source": "staged", "max_lines": 201}),
    (GitReadFileInput, {"source": "index", "path": "x", "commit": "HEAD"}),
    (GitReadFileInput, {"source": "commit", "path": "x"}),
    *[(GitReadFileInput, {"source": "index", "path": path}) for path in
      ("../x", "/workspace/x", ":(glob)*", "a/../b", "x\x00y", "a//b")],
])
def test_git_strict_schemas(schema, args):
    with pytest.raises(ValueError):
        schema.model_validate(args)


@pytest.mark.parametrize("writable", [False, True])
def test_git_live_versions_literal_names_and_no_writes(database, tmp_path, writable):
    init(tmp_path)
    names = ["space file", "中文", "tab\tfile", "newline\nfile", "-option", "literal[abc]*"]
    for name in names:
        (tmp_path / name).write_text("staged\n")
    (tmp_path / "sample.py").write_text("staged\n")
    git(tmp_path, "add", ".")
    (tmp_path / "sample.py").write_text("root\n")  # Changes cancel in HEAD->worktree.
    (tmp_path / "untracked").mkdir()
    (tmp_path / "untracked/file").write_text("user work\n")
    before = snapshot(tmp_path)

    async def check(r):
        scope = await scope_for(r, tmp_path, "workspace_write" if writable else "workspace_read")
        try:
            state = await read(scope, "git_status")
            entries = {e["path"]: e for e in state["records"]}
            assert entries["sample.py"]["staged"] and entries["sample.py"]["unstaged"]
            assert entries["untracked/file"]["untracked"]
            assert all(name in entries for name in names)
            # Real mount mode, including a writable repository with read-only Git tools.
            obj = await DockerClient().inspect(scope.container_id)
            assert obj["Mounts"][0]["RW"] is writable
            staged = await read(scope, "git_diff", source="staged")
            unstaged = await read(scope, "git_diff", source="unstaged")
            assert "sample.py" in {e["path"] for e in staged["records"]}
            assert [e["path"] for e in unstaged["records"]] == ["sample.py"]
            for source, plus in (("staged", "+staged"), ("unstaged", "+root")):
                patch = await read(scope, "git_diff", source=source, path="sample.py", view="patch")
                assert plus in patch["content"] and "NOT source lines" in patch["content"]
            index = await read(scope, "git_read_file", source="index", path="sample.py")
            worktree = await read(scope, "read_file", path="sample.py")
            assert index["records"][0]["text"] == "staged"
            assert "1: root" in worktree["content"]
            for name in names:
                blob = await read(scope, "git_read_file", source="index", path=name)
                assert blob["records"][0]["text"] == "staged"
                patch = await read(scope, "git_diff", source="staged", view="patch", path=name)
                assert "+staged" in patch["content"]
            assert snapshot(tmp_path) == before
            assert not list(tmp_path.rglob("*.lock"))
        finally:
            await finish_scope(scope)
    database(check)


def test_git_unborn_clean_and_conflicted_index(database, tmp_path):
    init(tmp_path, commit=False)
    (tmp_path / "sample.py").write_text("index\n")
    git(tmp_path, "add", ".")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            state = await read(scope, "git_status")
            assert state["HEAD"] is None
            diff = await read(scope, "git_diff", source="staged", view="patch", path="sample.py")
            assert "+index" in diff["content"]
            await error(scope, "git_log", "GIT_REFERENCE_INVALID")
            git(tmp_path, "commit", "-qm", "first")
            assert not (await read(scope, "git_status"))["records"]
            assert not (await read(scope, "git_diff", source="staged"))["records"]
            blob = git(tmp_path, "rev-parse", "HEAD:sample.py")
            git(tmp_path, "update-index", "--force-remove", "sample.py")
            git(tmp_path, "update-index", "--index-info", data=(
                f"100644 {blob} 1\tsample.py\n100644 {blob} 2\tsample.py\n100644 {blob} 3\tsample.py\n").encode())
            state = await read(scope, "git_status")
            assert state["records"][0]["conflict"]
            await error(scope, "git_read_file", "GIT_INDEX_CONFLICT", source="index", path="sample.py")
            await error(scope, "git_diff", "GIT_INDEX_CONFLICT", source="staged", view="patch", path="sample.py")
        finally:
            await finish_scope(scope)
    database(check)


def test_git_history_root_merge_deletion_and_version_isolation(database, tmp_path):
    root = init(tmp_path)
    git(tmp_path, "branch", "literal+name;safe", root)
    git(tmp_path, "checkout", "-qb", "side")
    (tmp_path / "side").write_text("side\n")
    git(tmp_path, "add", "."); git(tmp_path, "commit", "-qm", "side")
    git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "sample.py").write_text("second\n")
    git(tmp_path, "commit", "-qam", "second")
    first = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "merge", "--no-ff", "-qm", "merge", "side")
    merged = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "rm", "-q", "sample.py"); git(tmp_path, "commit", "-qm", "deleted")
    deleted = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "sample.py").write_text("different worktree\n")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            literal = await read(scope, "git_log", revision="literal+name;safe")
            assert literal["revision"] == root
            history = await read(scope, "git_log", limit=2)
            fixed = history["revision"]
            assert fixed == deleted and history["next_offset"] == 2
            assert "private body" not in history["content"]
            git(tmp_path, "commit", "--allow-empty", "-qm", "HEAD moved")
            rest = await read(scope, "git_log", revision=fixed, offset=2)
            assert len({e["commit"] for e in history["records"] + rest["records"]}) == 5
            diff = await read(scope, "git_diff", source="commit", commit=root)
            assert diff["parent_count"] == 0 and diff["records"][0]["status"] == "A"
            merge = await read(scope, "git_diff", source="commit", commit=merged)
            assert merge["parent_count"] == 2 and merge["base"] == first
            assert [e["path"] for e in merge["records"]] == ["side"]
            removed = await read(scope, "git_diff", source="commit", commit=deleted, view="patch", path="sample.py")
            assert "-second" in removed["content"]
            old = await read(scope, "git_read_file", source="commit", commit=merged, path="sample.py")
            assert old["records"][0]["text"] == "second"
            (tmp_path / "sample.py").unlink()
            assert (await read(scope, "git_read_file", source="commit", commit=root, path="sample.py"))["records"][0]["text"] == "root"
            await error(scope, "git_read_file", "GIT_FILE_NOT_FOUND", source="commit", commit=deleted, path="sample.py")
        finally:
            await finish_scope(scope)
    database(check)


def test_git_pagination_bytes_records_and_patch_lines(database, tmp_path):
    root = init(tmp_path)
    for n in range(65):
        (tmp_path / (f"file-{n:03}-" + "长" * 20)).write_text("new\n")
    (tmp_path / "sample.py").write_text("".join(f"line {n}\n" for n in range(450)))
    git(tmp_path, "add", ".")
    for n in range(60):
        git(tmp_path, "commit", "--allow-empty", "-qm", f"history {n}")
    # Stage another large list and patch after committing fixture history.
    for p in tmp_path.glob("file-*"):
        p.write_text("newer\n")
    (tmp_path / "sample.py").write_text("".join(f"changed {n}\n" for n in range(450)))
    git(tmp_path, "add", ".")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            scope.settings = scope.settings.model_copy(update={"output_bytes": 1600, "output_lines": 8})
            async def all_pages(name, key, **args):
                result, cursor, seen = [], 0 if key == "offset" else 1, set()
                while True:
                    assert cursor not in seen
                    seen.add(cursor)
                    batch = await read(scope, name, **args, **{key: cursor})
                    result.extend(batch["records"])
                    next_position = batch["next_offset" if key == "offset" else "next_line"]
                    if next_position is None:
                        assert not batch["truncated"]
                        break
                    assert batch["truncated"] and str(next_position) in batch["content"]
                    assert next_position == cursor + len(batch["records"])
                    cursor = next_position
                return result
            state = await all_pages("git_status", "offset", limit=100)
            assert len(state) == 66 and len({e["path"] for e in state}) == 66
            changes = await all_pages("git_diff", "start_line", source="staged")
            assert len(changes) == 66
            patch = await all_pages("git_diff", "start_line", source="staged", view="patch", path="sample.py")
            assert len(patch) > 850 and [e["position"] for e in patch] == list(range(1, len(patch) + 1))
            blob = await all_pages("git_read_file", "start_line", source="index", path="sample.py")
            assert [e["line"] for e in blob] == list(range(1, 451))
            log = await all_pages("git_log", "offset", revision=git(tmp_path, "rev-parse", "HEAD"), limit=50)
            assert len(log) == 61 and log[-1]["commit"] == root
            _, output, _ = await scope.client.call("exec", scope.container_id, "python", "-I", "-c",
                "from pathlib import Path; print(sum(p.read_text().strip() == 'git' for p in Path('/proc').glob('[0-9]*/comm')))")
            assert output.strip() == b"0"  # Early page termination reaps Git children.
        finally:
            await finish_scope(scope)
    database(check)


def test_git_external_programs_never_execute(database, tmp_path):
    oid = init(tmp_path)
    marker = "/workspace/external-executed"
    command = "touch " + marker
    for key in ("diff.external", "diff.evil.command", "diff.evil.textconv", "core.fsmonitor"):
        git(tmp_path, "config", key, command)
    (tmp_path / ".gitattributes").write_text("*.py diff=evil filter=evil\n")
    (tmp_path / "sample.py").write_text("staged\n")
    git(tmp_path, "-c", "core.fsmonitor=false", "add", ".")
    (tmp_path / "sample.py").write_text("unstaged\n")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            before = snapshot(tmp_path)
            for name, args in (("git_status", {}), ("git_diff", {"source": "unstaged", "path": "sample.py", "view": "patch"}), ("git_diff", {"source": "staged", "path": "sample.py", "view": "patch"}), ("git_diff", {"source": "commit", "commit": oid, "path": "sample.py", "view": "patch"})):
                await read(scope, name, **args)
            assert snapshot(tmp_path) == before
            for key in ("filter.evil.clean", "filter.evil.process"):
                git(tmp_path, "config", key, command)
            before = snapshot(tmp_path)
            await error(scope, "git_status", "GIT_UNSUPPORTED")
            await error(scope, "git_diff", "GIT_UNSUPPORTED", source="unstaged")
            await read(scope, "git_diff", source="staged", view="patch", path="sample.py")
            await read(scope, "git_read_file", source="index", path="sample.py")
            await read(scope, "git_read_file", source="commit", commit=oid, path="sample.py")
            assert snapshot(tmp_path) == before
            assert not (tmp_path / "external-executed").exists()
        finally:
            await finish_scope(scope)
    database(check)


def test_git_unsupported_invalid_and_missing_objects(database, tmp_path):
    oid = init(tmp_path)
    (tmp_path / "binary").write_bytes(b"a\x00b")
    (tmp_path / "nonutf").write_bytes(b"\xff\xfe")
    (tmp_path / "long").write_text("x" * 100000)
    (tmp_path / "big").write_text("x" * 300000)
    (tmp_path / "link").symlink_to("sample.py")
    git(tmp_path, "add", ".")
    blob = git(tmp_path, "rev-parse", "HEAD:sample.py")
    git(tmp_path, "tag", "main", oid)  # Ambiguous short name.
    git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{oid},module")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            listing = await read(scope, "git_diff", source="staged")
            assert next(v for v in listing["records"] if v["path"] == "module")["new_mode"] == "160000"
            await error(scope, "git_read_file", "GIT_UNSUPPORTED", source="index", path="module/child")
            for revision in ("--help", "HEAD..HEAD", "HEAD^!", "missing", blob, "main"):
                await error(scope, "git_log", "GIT_REFERENCE_INVALID", revision=revision)
            for path in ("../x", ":(glob)*", "/tmp/x", "x\x00y"):
                await error(scope, "git_read_file", "GIT_ARGUMENTS_INVALID", source="index", path=path)
            for path in ("binary", "nonutf", "link", "module"):
                await error(scope, "git_read_file", "GIT_UNSUPPORTED", source="index", path=path)
            for path in ("binary", "long", "link", "module"):
                await error(scope, "git_diff", "GIT_UNSUPPORTED", source="staged", view="patch", path=path)
            await error(scope, "git_read_file", "GIT_FILE_NOT_FOUND", source="index", path="absent")
            bounded = await read(scope, "git_read_file", source="index", path="long")
            assert bounded["truncated"] and bounded["next_line"] == 1
            assert "not read" in bounded["content"]
            scope.settings = scope.settings.model_copy(update={"file_bytes": 1024})
            await error(scope, "git_read_file", "GIT_UNSUPPORTED", source="index", path="big")
            git(tmp_path, "config", "extensions.partialClone", "origin")
            await error(scope, "git_log", "GIT_UNSUPPORTED")
            git(tmp_path, "config", "--unset", "extensions.partialClone")
            object_path = tmp_path / ".git/objects" / blob[:2] / blob[2:]
            object_path.unlink()
            await error(scope, "git_read_file", "GIT_READ_FAILED", source="commit", commit=oid, path="sample.py")
        finally:
            await finish_scope(scope)
    database(check)


@pytest.mark.parametrize("layout", ["absent", "gitfile", "bare", "alternates"])
def test_git_repository_boundary(database, tmp_path, layout):
    if layout == "gitfile":
        (tmp_path / ".git").write_text("gitdir: /elsewhere\n")
    elif layout == "bare":
        git(tmp_path, "init", "-q", "--bare")
    elif layout == "alternates":
        init(tmp_path)
        (tmp_path / ".git/objects/info/alternates").write_text("/elsewhere\n")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            await error(scope, "git_status", "GIT_UNSUPPORTED" if layout == "alternates" else "GIT_REPOSITORY_UNAVAILABLE")
        finally:
            await finish_scope(scope)
    database(check)


def test_git_only_profile_workspace_snapshot_and_child_isolation(database, tmp_path):
    from agent_service.delegation import READ_TOOLS
    from agent_service.snapshot import runtime_snapshot
    assert GIT_TOOLS <= WORKSPACE_TOOLS and not GIT_TOOLS & WRITE_TOOLS
    assert not GIT_TOOLS & READ_TOOLS
    init(tmp_path)

    async def check(r):
        install(r, tmp_path, profiles=(Profile(id="git-only", tools=tuple(sorted(GIT_TOOLS))),))
        with pytest.raises(ValueError, match="requires a workspace"):
            r.registry.bind("git-only")
        model = SyntheticModel(responses=[call("git_status", {}), AIMessage("done")])
        binding = r.registry.bind("git-only", "source")
        snap = runtime_snapshot(model, r.settings, r.registry, binding, ())
        assert snap["execution"]["tool_implementation"] == "workspace-v2"
        submitted, _ = await r.store.submit("inspect", profile="git-only", workspace_id="source")
        worker = Worker(r, model=model)
        row = await execute(r, worker)
        assert row["id"] == submitted["id"] and row["status"] == "completed", row
        state = await worker.agent.aget_state(row["final_checkpoint"])
        assert state.values["messages"][-2].artifact["source"] == "status"
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"
    database(check)


@pytest.mark.parametrize("mode", ["current", "commit", "fix"])
def test_git_code_profiles_api_roundtrip(database, tmp_path, monkeypatch, mode):
    registry, workspace = load_example(tmp_path, monkeypatch, image=os.environ["AGENT_TEST_EXECUTION_IMAGE"])
    oid = init(workspace)
    (workspace / "sample.py").write_text("staged\n")
    git(workspace, "add", ".")
    (workspace / "sample.py").write_text("worktree\n")
    (workspace / "untracked").write_text("user notes\n")
    before = snapshot(workspace)
    profile = "code-fix" if mode == "fix" else "code-review"
    answer = ("## 修复摘要\n已修改工作区文本。\n## 修改内容\nsample.py:1。\n## 检查结果\n已核对修改前后状态与差异；未运行测试；无可信最终验证。\n## 遗留问题\n无。" if mode == "fix" else
              "## 审查结论\n未发现可确认问题。\n## 问题清单\n无。\n## 审查范围与未确认事项\n" + (f"提交模式：{oid}，原生根提交比较，仅 sample.py。" if mode == "commit" else "当前改动模式：HEAD 到 index、index 到工作区及未跟踪文件分别读取。") + "静态审查，未运行检查，无可信最终验证。")

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, "resources", r, raising=False)
        if mode == "current":
            actions = [("git_status", {}), ("git_diff", {"source": "staged"}), ("git_diff", {"source": "unstaged"}),
                       ("git_diff", {"source": "staged", "view": "patch", "path": "sample.py"}),
                       ("git_diff", {"source": "unstaged", "view": "patch", "path": "sample.py"}),
                       ("git_read_file", {"source": "index", "path": "sample.py"}),
                       ("read_file", {"path": "sample.py"}), ("read_file", {"path": "untracked"})]
        elif mode == "commit":
            actions = [("git_log", {}), ("git_diff", {"source": "commit", "commit": oid}),
                       ("git_diff", {"source": "commit", "commit": oid, "view": "patch", "path": "sample.py"}),
                       ("git_read_file", {"source": "commit", "commit": oid, "path": "sample.py"})]
        else:
            actions = [("git_status", {}), ("git_diff", {"source": "unstaged"}), ("read_file", {"path": "sample.py"}),
                       ("edit_file", {"path": "sample.py", "old_text": "worktree", "new_text": "fixed", "expected_hash": hashlib.sha256(b"worktree\n").hexdigest()}),
                       ("git_status", {}), ("git_diff", {"source": "unstaged", "view": "patch", "path": "sample.py"}),
                       ("git_log", {})]  # fix is not permitted to read history.
        responses = [call("read_skill", {"name": profile}, "skill"),
                     *[call(name, args, f"step-{n}") for n, (name, args) in enumerate(actions)], AIMessage(answer)]
        initial = True
        async def reply(messages):
            nonlocal initial
            if initial:
                await inspect_mount(r, run_id, workspace, mode == "fix")
                initial = False
            return responses.pop(0)
        worker = Worker(r, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            response = await client.post("/v1/runs", json={"task": "审查当前未提交改动" if mode == "current" else "从历史选定提交审查" if mode == "commit" else "修改后核对 Git 状态", "profile": profile, "workspace_id": "project"})
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            tools = await completed(client, r, worker, run_id, answer)
        if mode == "current":
            assert tools[6].artifact["source"] == "index" and "staged" in tools[6].text
            assert "worktree" in tools[7].text
        elif mode == "commit":
            assert tools[-1].artifact["commit"] == oid and "root" in tools[-1].text
            assert "worktree" not in tools[-1].text
        else:
            assert tools[-1].status == "error"
            assert "+fixed" in tools[-2].text
            assert (workspace / "sample.py").read_text() == "fixed\n"
        after = snapshot(workspace)
        if mode == "fix":
            assert {k: v for k, v in after.items() if k != "sample.py"} == {k: v for k, v in before.items() if k != "sample.py"}
        else:
            assert after == before
    database(check)


@pytest.mark.parametrize("ending", ["timeout", "cancel", "lease", "environment"])
def test_git_lifecycle_controls_and_private_logs(database, tmp_path, monkeypatch, caplog, ending):
    import asyncio
    from dataclasses import replace
    from agent_service.config import Settings
    from agent_service.execution import ExecutionScope
    from agent_service.execution_config import ExecutionSettings

    init(tmp_path)
    started = asyncio.Event()
    original = ExecutionScope.file_call
    scope_seen = []

    async def delayed(self, name, args):
        if name != "git_diff":
            return await original(self, name, args)
        scope_seen.append(self)
        task = asyncio.create_task(self.client.call(
            "exec", "-i", self.container_id, "/usr/local/bin/python", "-I", "-c",
            "import time; time.sleep(20)", check=False, timeout=min(self.settings.command_timeout, self.remaining())))
        try:
            while not self.client.processes:
                await asyncio.sleep(0.01)
            started.set()
            await task
            await self.healthy()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def check(r):
        install(r, tmp_path,
                profiles=(Profile(id="git-lifecycle", tools=tuple(sorted(GIT_TOOLS))),),
                execution=ExecutionSettings(image=os.environ["AGENT_TEST_EXECUTION_IMAGE"], command_timeout=1.0 if ending == "timeout" else 30.0))
        monkeypatch.setattr(ExecutionScope, "file_call", delayed)
        caplog.set_level("INFO", logger="agent_service.policy")
        submitted, _ = await r.store.submit("inspect", profile="git-lifecycle", workspace_id="source")
        model = SyntheticModel(responses=[call("git_status", {}, "status"), call("git_diff", {"source": "staged", "path": "SOURCE_SECRET"}, "diff"), AIMessage("must not publish")])
        worker = Worker(r, model=model)
        claim = await r.store.claim(worker.id)
        running = asyncio.create_task(worker.execute(claim))
        await asyncio.wait_for(started.wait(), 10)
        if ending == "cancel":
            await r.store.cancel(submitted["id"])
        elif ending == "lease":
            async with r.pool.connection() as conn:
                await conn.execute("UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (submitted["id"],))
        elif ending == "environment":
            await DockerClient().call("rm", "--force", scope_seen[0].container_id)
        await asyncio.wait_for(running, 10)
        row = await r.store.get(submitted["id"])
        if ending == "cancel":
            assert row["status"] == "cancelled"
        else:
            assert row["error_code"] == {"timeout": "COMMAND_TIMEOUT", "lease": "LEASE_EXPIRED", "environment": "EXECUTION_ENVIRONMENT_LOST"}[ending], row
        assert row["answer"] is None and len(model.history) == 2
        resource = await r.store.execution_resource(row["id"])
        assert resource["state"] == "released"
        assert await DockerClient().inspect(resource["container_id"]) is None
        assert not scope_seen[0].client.processes
        records = [v for v in caplog.records if getattr(v, "tool_name", "") == "git_diff"]
        assert records and all(str(v.run_id) == str(submitted["id"]) and v.call_id == "diff" for v in records)
        assert "SOURCE_SECRET" not in str([vars(v) for v in records])
    database(check, replace(Settings(), heartbeat_seconds=0.03, lease_seconds=2.0))


def test_git_live_source_change_is_reread_and_visible(database, tmp_path):
    init(tmp_path)
    (tmp_path / "sample.py").write_text("first observation\n")
    metadata = snapshot(tmp_path / ".git")
    # Deterministic fixture mutation between observations, inside the real container.
    probe = '''import sys, json
from pathlib import Path
sys.path.insert(0, '/opt/agent')
from git_tools import Git, operation
original = Git.observation
calls = 0
def observe(self, path=None):
    global calls
    calls += 1
    if calls == 2:
        Path('/workspace/sample.py').write_text('changed during observation\\n')
    return original(self, path)
Git.observation = observe
print(json.dumps(operation(Path('/workspace'), 'git_diff', {'source':'unstaged','view':'patch','path':'sample.py'}, 10485760)))
'''

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            _, output, _ = await scope.client.call("exec", scope.container_id, "python", "-I", "-c", probe)
            result = json.loads(output)
            assert result["source_changed"]
            assert "source changed, reread" in result["content"]
            assert "+changed during observation" in result["content"]
            assert "+first observation" not in result["content"]
            assert snapshot(tmp_path / ".git") == metadata
        finally:
            await finish_scope(scope)
    database(check)
