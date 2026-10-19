"""Review observation lifecycle: real Docker, no replay after uncertain dispatch."""

import asyncio
import json
import sys
import psycopg
import pytest
from langchain_core.messages import ToolMessage
from agent_service.code_execution import recover_code_resources
from agent_service.config import Settings
from agent_service.execution import DockerClient, recover_resources
from agent_service.execution_errors import EnvironmentUnavailable
from agent_service.worker import Worker
from contracts.review import CodeReviewCreate
from tests.models import SyntheticModel
from tests.test_code_analysis import call, files
from tests.test_code_review import review_fixture
from tests.test_core import execute

SCRIPT = 'from sample import answer\ndef reproduce():\n    return {"observed":answer()==1,"details":str(answer())}\n'
SLOW = 'import time\ndef reproduce():\n    time.sleep(60)\n    return {"observed":False,"details":"done"}\n'
EMPTY = {
    "version": 1,
    "summary": "Empty range",
    "findings": [],
    "changes": [],
    "unconfirmed": [],
    "missing_context": [],
}


def test_result_ack_loss_reuses_original_reproduction(database, tmp_path, monkeypatch):
    registry, source, _, _ = review_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        original = r.store.finish_reproduction
        confirmations = 0

        async def lost(*args):
            nonlocal confirmations
            result = await original(*args)
            confirmations += 1
            if confirmations == 1:
                raise psycopg.OperationalError("injected acknowledgement loss")
            return result

        monkeypatch.setattr(r.store, "finish_reproduction", lost)
        step = 0
        ids = []

        async def reply(messages):
            nonlocal step
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            step += 1
            if step in (1, 2):
                if tools:
                    ids.append(json.loads(tools[-1].text)["records"][0]["id"])
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "Returns one", "script": SCRIPT},
                    "same-call",
                )
            ids.append(json.loads(tools[-1].text)["records"][0]["id"])
            return call("ReviewReportV1", EMPTY, "report")

        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        assert confirmations == 2 and len(set(ids)) == 1
        records = (await r.store.code_page(task["id"], "reproductions"))["items"]
        assert len(records) == 1 and records[0]["status"] == "observed"
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


@pytest.mark.parametrize("phase", ["reservation", "create", "outage"])
def test_uncertain_dispatch_never_replays(database, tmp_path, monkeypatch, phase):
    registry, source, _, _ = review_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        original_reserve = r.store.reserve_reproduction
        original_call = DockerClient.call
        disconnected = False
        reservations = 0

        async def reserve(*args):
            nonlocal reservations
            result = await original_reserve(*args)
            reservations += 1
            if phase == "reservation":
                raise psycopg.OperationalError("lost reservation confirmation")
            return result

        async def docker(self, *args, **kwargs):
            nonlocal disconnected
            if disconnected:
                raise EnvironmentUnavailable("Docker disconnected")
            result = await original_call(self, *args, **kwargs)
            if phase == "create" and args and args[0] == "create":
                resources = await r.store.code_resources()
                if any(e["purpose"] == "check" for e in resources):
                    raise EnvironmentUnavailable("create confirmation lost")
            if phase == "outage" and "/opt/agent/review_runner.py" in args:
                disconnected = True
            return result

        monkeypatch.setattr(r.store, "reserve_reproduction", reserve)
        monkeypatch.setattr(DockerClient, "call", docker)
        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        result = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    reply=lambda messages: call(
                        "run_review_reproduction",
                        {"hypothesis": "One", "script": SCRIPT},
                        "once",
                    )
                ),
            ),
        )
        assert result["status"] == "failed", result
        assert reservations == 1
        assert (await r.store.code_task(task["id"]))["report_artifact_id"] is None
        disconnected = False
        monkeypatch.setattr(DockerClient, "call", original_call)
        await recover_code_resources(r.store, registry.execution)
        await recover_resources(r.store, registry.execution)
        records = (await r.store.code_page(task["id"], "reproductions"))["items"]
        assert len(records) == 1 and records[0]["status"] == "inconclusive"
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


@pytest.mark.parametrize("mode", ["cancel", "lease", "worker"])
def test_interrupt_reproduction_reconciles(database, tmp_path, mode):
    registry, source, _, _ = review_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        worker = Worker(
            r,
            model=SyntheticModel(
                reply=lambda messages: call(
                    "run_review_reproduction",
                    {"hypothesis": "Wait", "script": SLOW},
                    "once",
                )
            ),
        )
        row = await r.store.claim(worker.id)
        running = asyncio.create_task(worker.execute(row))
        try:
            for _ in range(300):
                if any(
                    e["purpose"] == "check" and e["state"] == "active"
                    for e in await r.store.code_resources()
                ):
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("Observation container did not start")
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
            detail = await r.store.code_task(task["id"])
            assert (
                detail["status"] in ("cancelled", "failed")
                and detail["report_artifact_id"] is None
            )
            records = (await r.store.code_page(task["id"], "reproductions"))["items"]
            assert len(records) == 1 and records[0]["status"] == "inconclusive"
            assert files(source) == before
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    database(check, Settings(heartbeat_seconds=0.05, lease_seconds=2))


def test_sigkill_observation_and_restart(database, tmp_path):
    registry, source, storage, _ = review_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        child = """
import asyncio,json,sys
from agent_service.resources import Resources
from agent_service.profiles import Registry,RuntimeFile,Workspace
from agent_service.execution_config import ExecutionSettings
from agent_service.code_registry import CodeReviewSettings,ProjectCheckDefinition
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_code_analysis import call
v=json.loads(sys.argv[1])
registry=Registry(RuntimeFile(version=1,workspaces=[Workspace(id='project',root=v['source'])],execution=ExecutionSettings(image=v['image']),
code_review=CodeReviewSettings(enabled=True,storage_root=v['storage'],check_suites={'python':('unit',)},workspace_suites={'project':('python',)},default_suites={'project':'python'})),
project_checks=(ProjectCheckDefinition('unit','1','python-unittest-v1',('project',),description='Check sample answer'),))
async def main():
 async with Resources(registry=registry) as r:
  worker=Worker(r,model=SyntheticModel(reply=lambda messages:call('run_review_reproduction',{'hypothesis':'Wait','script':v['script']},'once')))
  row=await r.store.claim(worker.id)
  await worker.execute(row)
asyncio.run(main())
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            child,
            json.dumps(
                {
                    "source": str(source),
                    "storage": str(storage),
                    "image": registry.execution.image,
                    "script": SLOW,
                }
            ),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            for _ in range(400):
                if any(
                    e["purpose"] == "check" and e["state"] == "active"
                    for e in await r.store.code_resources()
                ):
                    break
                if process.returncode is not None:
                    raise AssertionError((await process.stderr.read()).decode())
                await asyncio.sleep(0.025)
            else:
                raise AssertionError("Child reproduction did not start")
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
            records = (await r.store.code_page(task["id"], "reproductions"))["items"]
            assert len(records) == 1 and records[0]["status"] == "inconclusive"
            assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
            assert files(source) == before
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    database(check)


@pytest.mark.parametrize(
    "scenario",
    [
        "shared_budget",
        "reproduction_limit",
        "cross_identity",
        "invalid_report_budget",
        "log_truncation",
        "oom",
        "missing_dependency",
        "readonly",
    ],
)
def test_review_budgets_and_execution_boundaries(database, tmp_path, scenario):
    settings = {}
    if scenario == "shared_budget":
        settings = {"check_calls": 1}
    if scenario == "reproduction_limit":
        settings = {"reproduction_calls": 1}
    if scenario == "invalid_report_budget":
        settings = {"model_calls": 3}
    if scenario == "log_truncation":
        settings = {"log_bytes": 1024}
    if scenario == "oom":
        settings = {"check_seconds": 5}
    registry, source, _, _ = review_fixture(tmp_path, **settings)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        step = 0

        async def reply(messages):
            nonlocal step
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            step += 1
            if scenario == "invalid_report_budget":
                value = {
                    **EMPTY,
                    "changes": [
                        {
                            "change_id": "invented",
                            "status": "reviewed",
                            "reason": "Fake",
                            "evidence": [],
                            "missing_context": [],
                        }
                    ],
                }
                return call("ReviewReportV1", value, "bad" + str(step))
            if step == 1 and scenario in ("shared_budget", "cross_identity"):
                return call("run_project_check", {"check_id": "unit"}, "shared")
            if scenario in ("shared_budget", "cross_identity") and step == 2:
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "One", "script": SCRIPT},
                    "shared" if scenario == "cross_identity" else "repro",
                )
            if scenario == "reproduction_limit" and step <= 2:
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "One", "script": SCRIPT},
                    "repro" + str(step),
                )
            if (
                scenario in ("log_truncation", "oom", "missing_dependency", "readonly")
                and step == 1
            ):
                scripts = {
                    "log_truncation": 'print("x"*4000)\n' + SCRIPT,
                    "oom": 'def reproduce():\n    blocks=[bytearray(64*1024*1024) for _ in range(32)]\n    return {"observed":True,"details":"allocated"}\n',
                    "missing_dependency": "import intentionally_absent_review_dependency\n"
                    + SCRIPT,
                    "readonly": """from pathlib import Path
import os
import socket
def reproduce():
    try:
        Path('/workspace/sample.py').write_text('forbidden')
        return {'observed':False,'details':'Source writable'}
    except OSError: pass
    Path('/tmp/scratch').write_text('allowed')
    assert os.getuid()!=0
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/snapshot').exists()
    assert not Path('/original').exists()
    return {'observed':True,'details':'Source read-only, /tmp writable, non-root, no host mounts'}
""",
                }
                return call(
                    "run_review_reproduction",
                    {"hypothesis": scenario, "script": scripts[scenario]},
                    "repro",
                )
            if scenario in ("shared_budget", "reproduction_limit"):
                assert "budget" in tools[-1].text.lower()
            elif scenario in ("oom", "missing_dependency"):
                assert (
                    json.loads(tools[-1].text)["records"][0]["result"]["status"]
                    == "inconclusive"
                )
            elif scenario in ("readonly", "log_truncation"):
                result = json.loads(tools[-1].text)["records"][0]["result"]
                assert result["status"] == "observed"
                if scenario == "log_truncation":
                    assert result["log_truncated"]
            return call("ReviewReportV1", EMPTY, "report")

        model = SyntheticModel(reply=reply)
        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=model))
        assert result["status"] == (
            "failed"
            if scenario in ("cross_identity", "invalid_report_budget")
            else "completed"
        ), result
        if scenario == "cross_identity":
            assert result["error_code"] == "CHECK_RESULT_UNKNOWN"
        if scenario == "invalid_report_budget":
            assert len(model.history) == 3 and result["error_code"] == "QUOTA_EXCEEDED"
        assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


@pytest.mark.parametrize(
    "phase", ["snapshot", "publication", "source_corruption", "report_corruption"]
)
def test_review_preparation_publication_and_integrity(
    database, tmp_path, monkeypatch, phase
):
    from pathlib import Path

    registry, source, _, _ = review_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        original_call = DockerClient.call
        original_stage = r.store.stage_code_report

        async def docker(self, *args, **kwargs):
            result = await original_call(self, *args, **kwargs)
            if phase == "snapshot" and "/opt/agent/code_snapshot.py" in args:
                await r.store.cancel_task(task["id"])
            return result

        async def stage(*args, **kwargs):
            result = await original_stage(*args, **kwargs)
            if phase == "publication":
                await r.store.cancel_task(task["id"])
            if phase == "report_corruption":
                async with r.pool.connection() as conn:
                    record = await (
                        await conn.execute(
                            "SELECT * FROM agent.code_artifacts WHERE id=%s",
                            (result["report"],),
                        )
                    ).fetchone()
                    data = json.loads(
                        r.store.artifact_storage.read(
                            record["storage_key"],
                            record["sha256"],
                            record["size_bytes"],
                        )
                    )
                    data["authority"]["coverage"]["complete"] = not data["authority"][
                        "coverage"
                    ]["complete"]
                    raw = json.dumps(data).encode()
                    r.store.artifact_storage.path(record["storage_key"]).write_bytes(
                        raw
                    )
                    import hashlib

                    await conn.execute(
                        "UPDATE agent.code_artifacts SET sha256=%s,size_bytes=%s WHERE id=%s",
                        (hashlib.sha256(raw).hexdigest(), len(raw), record["id"]),
                    )
            return result

        monkeypatch.setattr(DockerClient, "call", docker)
        monkeypatch.setattr(r.store, "stage_code_report", stage)

        async def reply(messages):
            if phase == "source_corruption":
                metadata = await r.store.task(task["id"])
                resource = await r.store.execution_resource(metadata["current_run_id"])
                (Path(resource["execution_root"]) / "sample.py").write_text("corrupt")
            return call("ReviewReportV1", EMPTY, "report")

        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] in ("cancelled", "failed"), result
        if phase == "source_corruption":
            assert result["error_code"] == "SNAPSHOT_INVALID"
        if phase == "report_corruption":
            assert result["error_code"] == "REPORT_INVALID"
        assert (await r.store.code_task(task["id"]))["report_artifact_id"] is None
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


def test_review_context_does_not_inherit_execution_and_gc_respects_dependency(
    database, tmp_path
):
    from agent_service.profiles import Registry, RuntimeFile, Workspace
    from contracts.errors import ApiError

    original, source, _, _ = review_fixture(tmp_path, retention_seconds=1)
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=original.execution,
            code_review=original.review.settings,
        ),
        project_checks=tuple(original.review.checks.values()),
    )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        step = 0

        async def parent(messages):
            nonlocal step
            step += 1
            if step == 1:
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "Returns one", "script": SCRIPT},
                    "repro",
                )
            return call("ReviewReportV1", EMPTY, "report")

        first, _ = await r.store.create_code_task(
            CodeReviewCreate(task="prior review", workspace_id="project")
        )
        first_run = await execute(r, Worker(r, model=SyntheticModel(reply=parent)))
        assert first_run["status"] == "completed"

        async def child(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("review_diff", {}, "list")
            return call("ReviewReportV1", EMPTY, "report")

        second, _ = await r.store.create_code_task(
            CodeReviewCreate(
                task="current review",
                workspace_id="project",
                context_task_ids=[first["id"]],
            )
        )
        second_run = await execute(r, Worker(r, model=SyntheticModel(reply=child)))
        assert second_run["status"] == "completed"
        detail = await r.store.code_task(second["id"])
        assert (
            detail["coverage"]["complete"]
            and not detail["reproductions"]
            and detail["execution_notice"] == "未做执行验证"
        )
        script = (await r.store.code_page(first["id"], "reproductions"))["items"][0][
            "script_artifact_id"
        ]
        # A script cannot be fetched through a different task's artifact route.
        with pytest.raises(ApiError) as error:
            await r.store.code_artifact(second["id"], script)
        assert error.value.status == 404
        pending, _ = await r.store.create_code_task(
            CodeReviewCreate(
                task="keep prior artifacts",
                workspace_id="project",
                context_task_ids=[first["id"]],
            )
        )
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET finished_at=clock_timestamp()-interval '2 seconds' WHERE id=%s",
                (first["id"],),
            )
        await r.store.gc_code_content()
        assert (await r.store.code_artifact(first["id"], script))[1] == SCRIPT.encode()
        await r.store.cancel_task(pending["id"])
        await r.store.gc_code_content()
        with pytest.raises(ApiError) as error:
            await r.store.code_artifact(first["id"], script)
        assert error.value.status == 410
        assert (await r.store.code_task(first["id"]))["status"] == "completed"
        assert files(source) == before

    database(check)
