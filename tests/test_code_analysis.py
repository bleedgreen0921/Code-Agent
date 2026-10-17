"""Isolated analysis acceptance: real PostgreSQL/Docker, synthetic Agent only."""

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.app import app
from agent_service.code_registry import (
    PROFILE,
    CodeAnalysisSettings,
    ProjectCheckDefinition,
)
from agent_service.execution_config import ExecutionSettings
from agent_service.profiles import Registry, RuntimeFile, Workspace
from agent_service.worker import Worker
from contracts.code import CodeTaskCreate
from contracts.errors import ApiError
from tests.models import SyntheticModel
from tests.test_core import execute


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE
    )


def fixture(
    tmp_path, *, adapter="python-unittest-v1", body=None, checks=True, **budget
):
    source = tmp_path / "project"
    source.mkdir()
    storage = tmp_path / "managed"
    storage.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.email", "test@example.invalid")
    git(source, "config", "user.name", "Synthetic")
    (source / "sample.py").write_text("def answer():\n    return 1\n")
    (source / "tests").mkdir()
    (source / "tests/test_sample.py").write_text(
        body
        or "import unittest\nfrom sample import answer\nclass TestAnswer(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(answer(), 1)\n"
    )
    (source / ".gitignore").write_text("ignored/\n__pycache__/\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "original")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    definition = ProjectCheckDefinition(
        "unit",
        "1",
        adapter,
        ("project",),
        description="Check sample answer",
        timeout=budget.pop("check_timeout", 60),
    )
    settings = CodeAnalysisSettings(
        enabled=True,
        storage_root=str(storage),
        check_suites={"python": ("unit",)} if checks else {},
        workspace_suites={"project": ("python",)} if checks else {},
        default_suites={"project": "python"} if checks else {},
        **budget,
    )
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=ExecutionSettings(image=os.environ["AGENT_TEST_EXECUTION_IMAGE"]),
            code_analysis=settings,
        ),
        project_checks=(definition,) if checks else (),
    )
    return registry, source, storage, commit


def call(name, args, identity):
    return AIMessage(
        "",
        tool_calls=[{"name": name, "args": args, "id": identity, "type": "tool_call"}],
    )


def make_report(evidence):
    return {
        "version": 1,
        "scope": {
            "question": "Analyze answer",
            "covered": ["sample.answer"],
            "missing": [],
        },
        "implementation_locations": [
            {
                "symbol": "answer",
                "responsibility": "Return one",
                "source": {
                    "evidence_id": evidence,
                    "path": "sample.py",
                    "start_line": 1,
                    "end_line": 2,
                },
            }
        ],
        "call_relationships": [],
        "behavior_conclusions": [
            {
                "conclusion": "Returns one",
                "conditions": "When called",
                "classification": "fact",
                "basis": "source",
                "evidence": [
                    {
                        "evidence_id": evidence,
                        "path": "sample.py",
                        "start_line": 1,
                        "end_line": 2,
                    }
                ],
                "check_run_ids": [],
            }
        ],
        "execution_paths": ["answer returns a constant"],
        "exception_paths": ["No explicit exception handler"],
        "unconfirmed": [],
    }


def files(root):
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mode & 0o777,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "adapter,target,expected",
    [
        ("python-unittest-v1", "workspace", "passed"),
        ("python-pytest-v1", "commit", "passed"),
        ("python-unittest-v1", "workspace", "failed"),
        ("python-pytest-v1", "workspace", "inconclusive"),
        ("python-unittest-v1", "workspace", "timeout"),
        ("python-unittest-v1", "workspace", "none"),
    ],
)
def test_complete_code_task(database, tmp_path, monkeypatch, adapter, target, expected):
    body = None
    if expected == "failed":
        body = 'import unittest\nclass Fail(unittest.TestCase):\n    def test_failure(self):\n        self.fail("real failure")\n'
    elif expected == "inconclusive":
        body = "# zero tests\n"
    elif expected == "timeout":
        body = 'import unittest,time\nclass Slow(unittest.TestCase):\n    def test_slow(self):\n        print("before timeout", flush=True)\n        time.sleep(20)\n'
    registry, source, storage, commit = fixture(
        tmp_path,
        adapter=adapter,
        body=body,
        checks=expected != "none",
        check_timeout=2 if expected == "timeout" else 60,
    )
    if target == "commit":
        (source / "sample.py").write_text("def answer():\n    return 99\n")
        git(source, "add", "sample.py")
        git(source, "commit", "-qm", "new HEAD")
    else:
        (source / "staged.txt").write_text("staged\n")
        git(source, "add", "staged.txt")
        (source / "staged.txt").write_text("unstaged\n")
        (source / "untracked.txt").write_text("local data\n")
        (source / "ignored").mkdir()
        (source / "ignored/secret").write_text("excluded\n")
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, "resources", r, raising=False)
        invalid_sent = False

        async def reply(messages):
            nonlocal invalid_sent
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            source_reads = [m for m in tools if m.name == "read_file"]
            checks = [m for m in tools if m.name == "run_project_check"]
            if not source_reads:
                return call("read_file", {"path": "sample.py"}, "source")
            assert "2:     return 1" in source_reads[-1].text
            if not checks and expected != "none":
                return call("run_project_check", {"check_id": "unit"}, "check")
            if checks:
                result = checks[-1].artifact["result"]
                assert result["status"] == (
                    "inconclusive" if expected == "timeout" else expected
                )
                if expected == "timeout":
                    assert "timeout" in result["limitations"]
            report = make_report(source_reads[-1].artifact["evidence_id"])
            if not invalid_sent:
                invalid_sent = True
                report["implementation_locations"][0]["source"]["end_line"] = 900
                return call("AnalysisReportV1", report, "bad-report")
            assert any(
                "actually read" in m.text
                for m in tools
                if m.tool_call_id == "bad-report"
            )
            if checks:
                report["behavior_conclusions"].append(
                    {
                        "conclusion": "Check evidence is recorded",
                        "conditions": "Registered check invocation",
                        "classification": "fact",
                        "basis": "execution",
                        "evidence": [],
                        "check_run_ids": [checks[-1].artifact["id"]],
                    }
                )
            return call("AnalysisReportV1", report, "report")

        worker = Worker(r, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            payload = {
                "task": "Analyze answer",
                "workspace_id": "project",
                "target": {"kind": "commit", "ref": commit}
                if target == "commit"
                else {"kind": "workspace"},
            }
            response = await client.post(
                "/v1/code/tasks", json=payload, headers={"Idempotency-Key": "analysis"}
            )
            assert response.status_code == 202, response.text
            task_id = response.json()["task_id"]
            replay = await client.post(
                "/v1/code/tasks", json=payload, headers={"Idempotency-Key": "analysis"}
            )
            assert replay.status_code == 200
            run = await execute(r, worker)
            row = await r.store.get(run["id"])
            assert row["status"] == "completed", row
            detail = await client.get("/v1/code/tasks/" + task_id)
            data = detail.json()
            assert data["phase"] == "completed", data
            assert data["snapshot"]["commit_sha"] == commit
            assert not await r.store.code_resources()
            assert (await r.store.execution_resource(run["id"]))["state"] == "released"
            report = await client.get(
                f"/v1/code/tasks/{task_id}/artifacts/{data['report_artifact_id']}"
            )
            assert report.status_code == 200, report.text
            authority = report.json()["authority"]
            assert authority["snapshot_id"] == data["snapshot"]["id"]
            md = await client.get(
                f"/v1/code/tasks/{task_id}/artifacts/{data['markdown_artifact_id']}"
            )
            assert (
                "未配置检查集；未执行检查" in md.text
                if expected == "none"
                else "**"
                + ("inconclusive" if expected == "timeout" else expected)
                + "**"
                in md.text
            )
            artifacts = (await client.get(f"/v1/code/tasks/{task_id}/artifacts")).json()
            assert artifacts["items"] and all(
                "storage_key" not in a for a in artifacts["items"]
            )
            records = (await client.get(f"/v1/code/tasks/{task_id}/checks")).json()[
                "items"
            ]
            assert len(records) == (0 if expected == "none" else 1)
            resource = await r.store.execution_resource(run["id"])
            snap = Path(resource["execution_root"])
            if target == "workspace":
                assert (snap / "staged.txt").read_text() == "unstaged\n"
                assert git(snap, "show", ":staged.txt") == b"staged\n"
                assert (snap / "untracked.txt").read_text() == "local data\n"
                assert not (snap / "ignored").exists()
            assert files(source) == before

    database(check)


def test_code_admission_and_entry_guards(database, tmp_path, monkeypatch):
    registry, source, _, _ = fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        monkeypatch.setattr(app.state, "resources", r, raising=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            for endpoint, payload in [
                (
                    "/v1/runs",
                    {"task": "bad", "profile": PROFILE, "workspace_id": "project"},
                ),
                (
                    "/v1/tasks",
                    {"task": "bad", "profile": PROFILE, "workspace_id": "project"},
                ),
                ("/v1/conversations", {"profile": PROFILE, "workspace_id": "project"}),
            ]:
                response = await client.post(endpoint, json=payload)
                assert response.status_code == 422, response.text
                assert "CODE_TASK_REQUIRED" in response.text
            base = {"task": "analysis", "workspace_id": "project"}
            for update in (
                {"stage": "review"},
                {"check_suite_id": "unknown"},
                {"target": {"kind": "commit", "ref": "--evil"}},
                {"target": {"kind": "workspace", "root": "/tmp"}},
                {"profile": "workspace_write"},
            ):
                response = await client.post("/v1/code/tasks", json={**base, **update})
                assert response.status_code == 422, response.text
            first = await client.post(
                "/v1/code/tasks", json=base, headers={"Idempotency-Key": "same"}
            )
            assert first.status_code == 202
            response = await client.post(
                "/v1/code/tasks",
                json={**base, "target": {"kind": "commit", "ref": "HEAD"}},
                headers={"Idempotency-Key": "same"},
            )
            assert response.status_code == 409
            old, _ = await r.store.create_task("legacy")
            response = await client.post(
                "/v1/code/tasks", json={**base, "context_task_ids": [old["task_id"]]}
            )
            assert response.status_code == 422
            response = await client.get(
                f"/v1/code/tasks/{first.json()['task_id']}/artifacts/{uuid4()}"
            )
            assert response.status_code == 404

    database(check)


def test_source_report_rejects_forged_version_and_check_ids(tmp_path):
    from types import SimpleNamespace

    from agent_service.code_reports import EvidenceLedger

    row = {"id": uuid4(), "task_id": uuid4(), "_snapshot_id": uuid4()}
    store = SimpleNamespace(
        registry=SimpleNamespace(code=SimpleNamespace(settings=CodeAnalysisSettings()))
    )
    ledger = EvidenceLedger(store, row, {"commit_sha": "a" * 40})
    eid = str(uuid4())
    report = make_report(eid)
    with pytest.raises(ValueError, match="belong"):
        ledger.validate(report)
    ledger.sources[eid] = {
        "snapshot_id": str(row["_snapshot_id"]),
        "run_id": str(row["id"]),
        "path": "sample.py",
        "start_line": 1,
        "end_line": 2,
        "source": "index",
        "commit": None,
    }
    with pytest.raises(ValueError, match="version mismatch"):
        ledger.validate(report)
    for ref in (
        report["implementation_locations"][0]["source"],
        report["behavior_conclusions"][0]["evidence"][0],
    ):
        ref["source"] = "index"
    assert ledger.validate(report)
    report["behavior_conclusions"][0]["basis"] = "execution"
    with pytest.raises(ValueError, match="real check"):
        ledger.validate(report)
    report["behavior_conclusions"][0]["check_run_ids"] = [str(uuid4())]
    with pytest.raises(ValueError, match="Unknown check"):
        ledger.validate(report)


@pytest.mark.parametrize(
    "mutation,status,limitation",
    [
        ({}, "passed", None),
        (
            {"selected": 0, "started": [], "finished": [], "events": []},
            "inconclusive",
            "zero_tests",
        ),
        (
            {"events": [{"case": "one", "outcome": "skipped"}]},
            "inconclusive",
            "skipped_cases",
        ),
        ({"complete": False}, "inconclusive", "incomplete_protocol"),
        (
            {"events": [{"case": "one", "outcome": "failed"}], "exit_code": 1},
            "failed",
            None,
        ),
    ],
)
def test_check_protocol(mutation, status, limitation):
    from agent_service.project_checks import classify_result

    raw = {
        "protocol": 1,
        "adapter": "python-unittest-v1",
        "complete": True,
        "selected": 1,
        "started": ["one"],
        "finished": ["one"],
        "events": [{"case": "one", "outcome": "passed"}],
        "collection_errors": [],
        "exit_code": 0,
        **mutation,
    }
    result = classify_result(
        json.dumps(raw).encode(),
        exit_code=raw["exit_code"],
        adapter="python-unittest-v1",
    )
    assert result["status"] == status
    if limitation:
        assert limitation in result["limitations"]
    assert (
        classify_result(b"OK: all tests passed", exit_code=0)["status"]
        == "inconclusive"
    )


@pytest.mark.parametrize(
    "fault,code",
    [
        ("symlink", "SNAPSHOT_UNSUPPORTED"),
        ("shallow", "SNAPSHOT_UNSUPPORTED"),
        ("oversize", "SNAPSHOT_LIMIT"),
        ("conflict", "SNAPSHOT_UNSUPPORTED"),
    ],
)
def test_snapshot_rejects_unsupported_and_preserves_source(
    database, tmp_path, fault, code
):
    registry, source, _, _ = fixture(
        tmp_path, file_bytes=128 if fault == "oversize" else 10 * 1024**2
    )
    if fault == "symlink":
        (source / "link").symlink_to("/etc/passwd")
    elif fault == "shallow":
        (source / ".git/shallow").write_bytes(git(source, "rev-parse", "HEAD"))
    elif fault == "oversize":
        (source / "big").write_bytes(b"x" * 129)
    else:
        oid = git(source, "hash-object", "sample.py").decode().strip()
        subprocess.run(
            ["git", "-C", str(source), "update-index", "--index-info"],
            input=f"0 {'0' * 40}\tsample.py\n100644 {oid} 1\tsample.py\n100644 {oid} 2\tsample.py\n".encode(),
            check=True,
        )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        model = SyntheticModel()
        row = await execute(r, Worker(r, model=model))
        assert row["status"] == "failed" and row["error_code"] == code, row
        assert not model.history
        assert not await r.store.code_resources()
        assert not (await r.store.task(task["id"]))["unresolved"][
            "unreleased_resources"
        ]
        assert files(source) == before

    database(check)


def test_check_source_mount_readonly_and_tmp_writable(database, tmp_path):
    test = """import unittest
from pathlib import Path
class Isolation(unittest.TestCase):
    def test_permissions(self):
        Path('/tmp/allowed').write_text('temporary')
        with self.assertRaises(OSError):
            Path('/workspace/sample.py').write_text('changed')
        self.assertFalse(Path('/var/run/docker.sock').exists())
        self.assertFalse(Path('/snapshot').exists())
        self.assertFalse(Path('/artifacts').exists())
        self.assertNotIn('AGENT_DATABASE_URL', __import__('os').environ)
"""
    registry, source, _, _ = fixture(tmp_path, body=test)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "sample.py"}, "source")
            if len(tools) == 1:
                return call("run_project_check", {"check_id": "unit"}, "check")
            assert tools[-1].artifact["result"]["status"] == "passed"
            return call(
                "AnalysisReportV1",
                make_report(tools[0].artifact["evidence_id"]),
                "report",
            )

        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        run = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert run["status"] == "completed", run
        assert files(source) == before

    database(check)


@pytest.mark.parametrize("mode", ["cancel", "lease", "worker"])
def test_cancel_or_abandon_during_check_is_recoverable(database, tmp_path, mode):
    from agent_service.code_execution import recover_code_resources
    from agent_service.config import Settings
    from agent_service.execution import recover_resources

    registry, source, _, _ = fixture(
        tmp_path,
        body="import unittest,time\nclass Slow(unittest.TestCase):\n    def test_wait(self):\n        time.sleep(60)\n",
    )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry

        async def reply(messages):
            return call("run_project_check", {"check_id": "unit"}, "check")

        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        worker = Worker(r, model=SyntheticModel(reply=reply))
        row = await r.store.claim(worker.id)
        running = asyncio.create_task(worker.execute(row))
        try:
            for attempt in range(200):
                resources = await r.store.code_resources(run_id=row["id"])
                if any(
                    e["purpose"] == "check" and e["state"] == "active"
                    for e in resources
                ):
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("Check container did not start")
            if mode == "cancel":
                await r.store.cancel_task(task["id"])
            elif mode == "lease":
                async with r.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                        (row["id"],),
                    )
            else:
                running.cancel()
            try:
                await asyncio.wait_for(running, 10)
            except asyncio.CancelledError:
                pass
            await r.store.maintain()
            await recover_code_resources(r.store, registry.execution)
            await recover_resources(r.store, registry.execution)
            await r.store.maintain()
            assert not await r.store.code_resources()
            assert (await r.store.execution_resource(row["id"]))["state"] == "released"
            detail = await r.store.code_task(task["id"])
            assert detail["status"] in ("cancelled", "failed"), detail
            assert detail["report_artifact_id"] is None
            assert files(source) == before
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    database(check, Settings(heartbeat_seconds=0.05, lease_seconds=2))


def test_unreleased_child_blocks_success_capacity_and_is_visible(
    database, tmp_path, monkeypatch
):
    from agent_service.code_execution import recover_code_resources
    from agent_service.execution import DockerClient

    registry, source, _, _ = fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        async with r.store.locked() as conn:
            resource = await r.store.reserve_code_resource(conn, row, "check", source)
        # Even if the main record was released, child occupancy remains authoritative.
        await r.store.resource_update(row["id"], row["lease_token"], state="released")
        assert not await r.store.finish(
            row["id"], row["lease_token"], answer="invalid", checkpoint={"x": 1}
        )
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST_FAILURE")
        summary = await r.store.unresolved(task_id=task["id"])
        assert summary["unreleased_resources"] == 1
        assert summary["records"][0]["kind"] == "code_execution_resource"
        # An uncertain Docker create cannot be declared absent or retried.
        await r.store.code_resource_update(
            resource["id"],
            row["lease_token"],
            phase="creating",
            daemon_id=await DockerClient().daemon(),
        )
        await recover_code_resources(r.store, registry.execution)
        current = await r.store.code_resource(resource["id"])
        assert current["state"] == "cleanup_failed"
        assert "unknown" in current["cleanup_error"]

    database(check)


def test_artifact_gc_active_reference_and_tombstone(database, tmp_path):
    registry, source, _, _ = fixture(tmp_path, checks=False, retention_seconds=1)

    async def check(r):
        r.registry = r.store.registry = registry

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "sample.py"}, "source")
            return call(
                "AnalysisReportV1",
                make_report(tools[0].artifact["evidence_id"]),
                "report",
            )

        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        row = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert row["status"] == "completed", row
        detail = await r.store.code_task(task["id"])
        artifact = detail["report_artifact_id"]
        dependent, _ = await r.store.create_code_task(
            CodeTaskCreate(
                task="followup", workspace_id="project", context_task_ids=[task["id"]]
            )
        )
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET finished_at=clock_timestamp()-interval '2 second' WHERE id=%s",
                (task["id"],),
            )
        await r.store.gc_code_content()
        assert (await r.store.code_artifact(task["id"], artifact))[1]
        with pytest.raises(ApiError) as missing:
            await r.store.code_artifact(dependent["id"], artifact)
        assert missing.value.status == 404
        await r.store.cancel_task(dependent["id"])
        await r.store.gc_code_content()
        with pytest.raises(ApiError) as expired:
            await r.store.code_artifact(task["id"], artifact)
        assert expired.value.status == 410
        assert (await r.store.code_task(task["id"]))["snapshot"]["state"] == "deleted"

    database(check)


def test_sigkill_worker_during_real_check_then_recover(database, tmp_path):
    """Hard process death leaves only journaled resources; a new Worker reaps them."""
    import sys

    from agent_service.code_execution import recover_code_resources
    from agent_service.execution import recover_resources

    registry, source, storage, _ = fixture(
        tmp_path,
        body="import unittest,time\nclass Slow(unittest.TestCase):\n    def test_wait(self):\n        time.sleep(60)\n",
    )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        child_code = """
import asyncio,json,sys
from agent_service.code_registry import CodeAnalysisSettings,ProjectCheckDefinition
from agent_service.execution_config import ExecutionSettings
from agent_service.profiles import Registry,RuntimeFile,Workspace
from agent_service.resources import Resources
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_code_analysis import call
values=json.loads(sys.argv[1])
registry=Registry(RuntimeFile(version=1,workspaces=[Workspace(id='project',root=values['source'])],
    execution=ExecutionSettings(image=values['image']),
    code_analysis=CodeAnalysisSettings(enabled=True,storage_root=values['storage'],check_suites={'python':('unit',)},
        workspace_suites={'project':('python',)},default_suites={'project':'python'})),
    project_checks=(ProjectCheckDefinition('unit','1','python-unittest-v1',('project',),description='Check sample answer'),))
async def main():
    async with Resources(registry=registry) as r:
        worker=Worker(r,model=SyntheticModel(reply=lambda messages:call('run_project_check',{'check_id':'unit'},'check')))
        row=await r.store.claim(worker.id)
        await worker.execute(row)
asyncio.run(main())
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            child_code,
            json.dumps(
                {
                    "source": str(source),
                    "storage": str(storage),
                    "image": registry.execution.image,
                }
            ),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            for attempt in range(400):
                resources = await r.store.code_resources()
                if any(
                    e["purpose"] == "check" and e["state"] == "active"
                    for e in resources
                ):
                    break
                if process.returncode is not None:
                    raise AssertionError((await process.stderr.read()).decode())
                await asyncio.sleep(0.025)
            else:
                raise AssertionError("Child check did not start")
            process.kill()
            await process.wait()
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE task_id=%s",
                    (task["id"],),
                )
            await r.store.maintain()
            await recover_code_resources(r.store, registry.execution)
            await recover_resources(r.store, registry.execution)
            assert not await r.store.code_resources()
            row = (await r.store.runs(task_id=task["id"]))["items"][0]
            assert row["status"] == "failed" and row["error_code"] == "LEASE_EXPIRED"
            assert (await r.store.execution_resource(row["id"]))["state"] == "released"
            assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
            checks = (await r.store.code_page(task["id"], "checks"))["items"]
            assert checks[0]["status"] == "inconclusive"
            assert files(source) == before
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    database(check)


def test_snapshot_raw_copy_deletion_mode_binary_unborn_and_moving_ref(tmp_path):
    from execution.code_snapshot import prepare

    limits = CodeAnalysisSettings().model_dump()
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "new.py").write_text("print(1)\n")
    git(root, "add", "new.py")
    unborn = prepare(root, tmp_path / "unborn", {"kind": "workspace"}, limits)
    assert unborn["commit_sha"] is None
    assert git(tmp_path / "unborn", "status", "--porcelain").startswith(b"A ")
    git(root, "commit", "-qm", "first")
    first = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "tag", "moving")
    (root / "new.py").write_text("print(2)\n")
    (root / "binary").write_bytes(b"\x00\xff")
    (root / "run").write_text("#!/bin/sh\n")
    (root / "run").chmod(0o755)
    git(root, "add", ".")
    git(root, "commit", "-qm", "second")
    (root / "new.py").unlink()
    before = files(root)
    prepare(root, tmp_path / "dirty", {"kind": "workspace"}, limits)
    assert not (tmp_path / "dirty/new.py").exists()
    assert (tmp_path / "dirty/binary").read_bytes() == b"\x00\xff"
    assert (tmp_path / "dirty/run").stat().st_mode & 0o111
    assert files(root) == before
    original = prepare(
        root, tmp_path / "fixed", {"kind": "commit", "ref": first}, limits
    )
    assert original["commit_sha"] == first
    assert (tmp_path / "fixed/new.py").read_text() == "print(1)\n"
    assert git(tmp_path / "fixed", "status", "--porcelain") == b""


def test_snapshot_unstable_retries_do_not_reparse_target(tmp_path, monkeypatch):
    from execution import code_snapshot as helper

    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "file").write_text("one")
    git(root, "add", ".")
    git(root, "commit", "-qm", "one")
    first = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "tag", "moving")
    (root / "file").write_text("two")
    git(root, "add", ".")
    git(root, "commit", "-qm", "two")
    real_metadata = helper.metadata
    observations = 0

    def moving(git_instance, limits):
        nonlocal observations
        value = real_metadata(git_instance, limits)
        observations += 1
        if observations == 1:
            git(root, "tag", "-f", "moving", "HEAD")
        return value

    monkeypatch.setattr(helper, "metadata", moving)
    result = helper.prepare(
        root,
        tmp_path / "out",
        {"kind": "commit", "ref": "moving"},
        CodeAnalysisSettings().model_dump(),
    )
    assert result["commit_sha"] == first and result["attempts"] == 2
    assert (tmp_path / "out/file").read_text() == "one"

    def unstable(git_instance, limits):
        nonlocal observations
        value = real_metadata(git_instance, limits)
        observations += 1
        (root / "file").write_text(str(observations))
        return value

    monkeypatch.setattr(helper, "metadata", unstable)
    with pytest.raises(helper.SnapshotError) as exc:
        helper.prepare(
            root,
            tmp_path / "unstable",
            {"kind": "workspace"},
            CodeAnalysisSettings().model_dump(),
        )
    assert exc.value.code == "SNAPSHOT_UNSTABLE"


def test_check_protocol_failure_with_incomplete_coverage_and_duplicate_pass():
    from agent_service.project_checks import classify_result

    raw = {
        "protocol": 1,
        "adapter": "python-unittest-v1",
        "complete": True,
        "selected": 2,
        "started": ["fail", "collection"],
        "finished": ["fail", "collection"],
        "events": [
            {"case": "fail", "outcome": "failed"},
            {"case": "collection", "outcome": "collection_error"},
        ],
        "collection_errors": ["missing dependency"],
        "exit_code": 1,
    }
    result = classify_result(json.dumps(raw), exit_code=1)
    assert result["status"] == "failed" and not result["complete"]
    assert "collection_errors" in result["limitations"]
    raw.update(
        selected=1,
        started=["one"],
        finished=["one"],
        collection_errors=[],
        exit_code=0,
        events=[
            {"case": "one", "outcome": "passed"},
            {"case": "one", "outcome": "passed"},
        ],
    )
    assert classify_result(json.dumps(raw), exit_code=0)["status"] == "inconclusive"
