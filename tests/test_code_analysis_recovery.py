"""Confirmation loss, storage corruption and Docker outage across code-task boundaries."""

import json
from pathlib import Path

import psycopg
import pytest
from langchain_core.messages import ToolMessage

from agent_service.code_execution import recover_code_resources
from agent_service.execution import DockerClient, recover_resources
from agent_service.execution_errors import EnvironmentUnavailable
from agent_service.worker import Worker
from contracts.code import CodeTaskCreate
from tests.models import SyntheticModel
from tests.test_code_analysis import call, files, fixture, make_report
from tests.test_core import execute


def test_check_acknowledgement_loss_reconciles_without_reexecution(
    database, tmp_path, monkeypatch
):
    registry, source, _, _ = fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        original = r.store.finish_code_check
        confirmations = 0

        async def lost_ack(*args):
            nonlocal confirmations
            result = await original(*args)
            confirmations += 1
            if confirmations == 1:
                raise psycopg.OperationalError("injected lost commit acknowledgement")
            return result

        monkeypatch.setattr(r.store, "finish_code_check", lost_ack)
        step = 0
        execution_ids = []

        async def reply(messages):
            nonlocal step
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            step += 1
            if step == 1:
                return call("read_file", {"path": "sample.py"}, "read")
            if step in (2, 3):
                if step == 3:
                    execution_ids.append(tools[-1].artifact["id"])
                return call("run_project_check", {"check_id": "unit"}, "same-identity")
            execution_ids.append(tools[-1].artifact["id"])
            return call(
                "AnalysisReportV1",
                make_report(tools[0].artifact["evidence_id"]),
                "report",
            )

        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        assert len(set(execution_ids)) == 1 and confirmations == 2
        assert len((await r.store.code_page(task["id"], "checks"))["items"]) == 1
        async with r.pool.connection() as conn:
            resources = await (
                await conn.execute(
                    "SELECT * FROM agent.code_execution_resources WHERE run_id=%s AND purpose='check'",
                    (result["id"],),
                )
            ).fetchall()
        assert len(resources) == 1 and resources[0]["state"] == "released"
        assert files(source) == before

    database(check)


def test_check_reservation_acknowledgement_loss_never_restarts(
    database, tmp_path, monkeypatch
):
    registry, _, _, _ = fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        original = r.store.reserve_code_check
        calls = 0

        async def lost_ack(*args):
            nonlocal calls
            calls += 1
            await original(*args)
            raise psycopg.OperationalError("injected reservation acknowledgement loss")

        monkeypatch.setattr(r.store, "reserve_code_check", lost_ack)
        model = SyntheticModel(
            reply=lambda messages: call(
                "run_project_check", {"check_id": "unit"}, "once"
            )
        )
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=model))
        assert (
            result["status"] == "failed"
            and result["error_code"] == "PERSISTENCE_FAILED"
        ), result
        assert calls == 1
        await recover_code_resources(r.store, registry.execution)
        checks = (await r.store.code_page(task["id"], "checks"))["items"]
        assert len(checks) == 1 and checks[0]["status"] == "inconclusive"
        assert not await r.store.code_resources()
        assert (await r.store.code_task(task["id"]))["report_artifact_id"] is None

    database(check)


def test_docker_outage_retains_journal_and_blocks_formal_report(
    database, tmp_path, monkeypatch
):
    registry, _, _, _ = fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        original = DockerClient.call
        disconnected = False

        async def outage(self, *args, **kwargs):
            nonlocal disconnected
            if disconnected:
                raise EnvironmentUnavailable("injected Docker outage")
            result = await original(self, *args, **kwargs)
            if "/opt/agent/project_runner.py" in args:
                disconnected = True
            return result

        monkeypatch.setattr(DockerClient, "call", outage)
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        result = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    reply=lambda messages: call(
                        "run_project_check", {"check_id": "unit"}, "check"
                    )
                ),
            ),
        )
        assert result["status"] == "failed", result
        detail = await r.store.code_task(task["id"])
        assert detail["report_artifact_id"] is None
        assert detail["unresolved"]["unreleased_resources"] >= 1
        assert await r.store.code_resources()
        disconnected = False
        monkeypatch.setattr(DockerClient, "call", original)
        await recover_code_resources(r.store, registry.execution)
        await recover_resources(r.store, registry.execution)
        assert not await r.store.code_resources()
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(check)


@pytest.mark.parametrize("corrupt", ["source", "artifact"])
def test_corruption_stops_publication(database, tmp_path, corrupt):
    registry, source, _, _ = fixture(tmp_path, checks=False)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "sample.py"}, "read")
            resource = await r.store.execution_resource(
                (await r.store.task(task["id"]))["current_run_id"]
            )
            if corrupt == "source":
                (Path(resource["execution_root"]) / "sample.py").write_text("corrupt")
                return call("read_file", {"path": "sample.py"}, "reread")
            identity = tools[0].artifact["evidence_id"]
            async with r.pool.connection() as conn:
                row = await (
                    await conn.execute(
                        "SELECT * FROM agent.code_artifacts WHERE id=%s", (identity,)
                    )
                ).fetchone()
            r.store.artifact_storage.path(row["storage_key"]).write_bytes(b"corrupt")
            return call("AnalysisReportV1", make_report(identity), "report")

        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "failed", result
        assert (await r.store.code_task(task["id"]))["report_artifact_id"] is None
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


def test_sealed_root_binding_loss_cannot_fallback(database, tmp_path, monkeypatch):
    registry, source, _, _ = fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        original = r.store.seal_code_snapshot

        async def unbind(*args, **kwargs):
            await original(*args, **kwargs)
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.execution_resources SET execution_root=NULL,snapshot_id=NULL WHERE run_id=%s",
                    (args[0]["id"],),
                )

        monkeypatch.setattr(r.store, "seal_code_snapshot", unbind)
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        model = SyntheticModel()
        result = await execute(r, Worker(r, model=model))
        assert (
            result["status"] == "failed" and result["error_code"] == "SNAPSHOT_INVALID"
        ), result
        assert not model.history and not await r.store.code_resources()
        assert files(source) == before

    database(check)


def test_artifact_commit_confirmation_loss_is_idempotent(
    database, tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager
    from uuid import uuid4

    registry, _, _, _ = fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        row = await r.store.claim(uuid4())
        original = r.store.locked
        confirmations = 0

        @asynccontextmanager
        async def lost_ack():
            nonlocal confirmations
            async with original() as conn:
                yield conn
            confirmations += 1
            if confirmations == 1:
                raise psycopg.OperationalError(
                    "injected metadata commit acknowledgement loss"
                )

        with monkeypatch.context() as patch:
            patch.setattr(r.store, "locked", lost_ack)
            artifact = await r.store.put_code_artifact(
                row, "check_log", b"one immutable log"
            )
        assert confirmations == 2
        result = await r.store.code_artifact(task["id"], artifact["id"])
        assert result[1] == b"one immutable log"
        assert len((await r.store.code_page(task["id"], "artifacts"))["items"]) == 1
        await r.store.resource_update(row["id"], row["lease_token"], state="released")
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST_FINISHED")

    database(check)


def test_invalid_protocol_boolean_version_and_deep_json_are_inconclusive():
    from agent_service.project_checks import classify_result

    raw = {
        "protocol": True,
        "adapter": "python-unittest-v1",
        "complete": True,
        "selected": 1,
        "started": ["one"],
        "finished": ["one"],
        "events": [{"case": "one", "outcome": "passed"}],
        "collection_errors": [],
        "exit_code": 0,
    }
    assert classify_result(json.dumps(raw), exit_code=0)["status"] == "inconclusive"
    assert (
        classify_result("[" * 2000 + "0" + "]" * 2000, exit_code=0)["status"]
        == "inconclusive"
    )


def test_gc_advances_past_already_deleted_batches(database, tmp_path):
    from uuid import uuid4

    registry, _, _, _ = fixture(tmp_path, retention_seconds=1)

    async def check(r):
        r.registry = r.store.registry = registry
        identities = []
        for index in range(21):
            task, _ = await r.store.create_code_task(
                CodeTaskCreate(task="diagnostic task", workspace_id="project")
            )
            row = await r.store.claim(uuid4())
            artifact = await r.store.put_code_artifact(
                row, "check_log", str(index).encode()
            )
            identities.append(artifact["id"])
            await r.store.resource_update(
                row["id"], row["lease_token"], state="released"
            )
            await r.store.finish(
                row["id"], row["lease_token"], error_code="TEST_FINISHED"
            )
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET finished_at=clock_timestamp()-interval '2 second'"
            )
        await r.store.gc_code_content()
        await r.store.gc_code_content()
        async with r.pool.connection() as conn:
            count = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM agent.code_artifacts WHERE id=ANY(%s::uuid[]) AND state='deleted'",
                    (identities,),
                )
            ).fetchone()
        assert count["n"] == 21

    database(check)


def test_gc_reclaims_aged_unregistered_files(database, tmp_path):
    import os
    import time
    from uuid import uuid4

    registry, _, _, _ = fixture(tmp_path, retention_seconds=1)

    async def check(r):
        r.registry = r.store.registry = registry
        identity = uuid4()
        key, _ = r.store.artifact_storage.write(identity, b"orphaned diagnostic", 1024)
        path = r.store.artifact_storage.path(key)
        await r.store.gc_code_content()
        assert path.exists()
        os.utime(path, (time.time() - 2, time.time() - 2))
        await r.store.gc_code_content()
        assert not path.exists()

    database(check)


def test_check_oom_does_not_kill_analysis_container(database, tmp_path):
    body = """import unittest
class Memory(unittest.TestCase):
    def test_memory(self):
        blocks=[bytearray(64*1024*1024) for _ in range(32)]
"""
    registry, source, _, _ = fixture(tmp_path, body=body, check_timeout=10)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "sample.py"}, "source")
            if len(tools) == 1:
                return call("run_project_check", {"check_id": "unit"}, "check")
            assert tools[-1].artifact["result"]["status"] == "inconclusive"
            assert tools[-1].artifact["result"]["exit_code"] in (137, 124)
            report = make_report(tools[0].artifact["evidence_id"])
            report["unconfirmed"] = [
                "Memory-limited check could not complete; test coverage is inconclusive."
            ]
            return call("AnalysisReportV1", report, "report")

        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


def test_invalid_report_retries_consume_original_model_budget(database, tmp_path):
    registry, _, _, _ = fixture(tmp_path, checks=False, model_calls=3)

    async def check(r):
        r.registry = r.store.registry = registry

        def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "sample.py"}, "source")
            report = make_report(tools[0].artifact["evidence_id"])
            report["implementation_locations"][0]["source"]["end_line"] = 999
            return call("AnalysisReportV1", report, "invalid-" + str(len(tools)))

        model = SyntheticModel(reply=reply)
        task, _ = await r.store.create_code_task(
            CodeTaskCreate(task="analyze", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=model))
        assert (
            result["status"] == "failed" and result["error_code"] == "QUOTA_EXCEEDED"
        ), result
        assert len(model.history) == 3
        assert (await r.store.code_task(task["id"]))["report_artifact_id"] is None
        assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
        assert not await r.store.code_resources()

    database(check)
