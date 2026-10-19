"""Candidate-only repair acceptance: real Git/PostgreSQL/Docker, synthetic model."""

import asyncio
import io
import json
import os
import zipfile
from pathlib import Path

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from agent_service.app import app
from agent_service.profiles import Registry, RuntimeFile, Workspace
from agent_service.repair_registry import (
    REPAIR_PROFILE,
    AcceptanceSuiteDefinition,
    CodeRepairSettings,
    acceptance_assets,
)
from agent_service.worker import Worker
from contracts.errors import ApiError
from contracts.repair import CodeRepairCreate
from tests.models import SyntheticModel
from tests.test_code_analysis import call, files, fixture, git
from tests.test_core import execute


def repair_fixture(
    tmp_path, *, development=False, execution_overrides=None, **settings
):
    old, source, storage, commit = fixture(tmp_path)
    old.execution = old.execution.model_copy(update=execution_overrides or {})
    suite = AcceptanceSuiteDefinition(
        "independent",
        "1",
        ("project",),
        old.execution.image,
        "sample",
        acceptance_assets("sample"),
        ("test_sample.TestAnswer.test_value",),
        "sample",
    )
    defaults = dict(
        enabled=True,
        storage_root=str(storage),
        acceptance_suites={"project": ("independent",)},
        default_acceptance_suites={"project": "independent"},
    )
    if development:
        defaults.update(
            check_suites={"python": ("unit",)},
            workspace_suites={"project": ("python",)},
            default_suites={"project": "python"},
        )
    defaults.update(settings)
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=old.execution,
            code_repair=CodeRepairSettings(**defaults),
        ),
        project_checks=tuple(old.code.checks.values()) if development else (),
        acceptance_suites=(suite,),
    )
    return registry, source, storage, commit


def draft(refs=()):
    return dict(
        version=1,
        summary="Restore the answer contract",
        cause="Wrong return constant",
        changes=["Return one from answer"],
        baseline_confirmation="Consult actual baseline check; historical status is not inherited.",
        goals=[
            dict(
                goal="answer returns one",
                status="resolved",
                explanation="Current candidate implements the requested value",
            )
        ],
        unfinished=[],
        sources=list(refs),
        check_run_ids=[],
        test_changes=[],
        limitations=["Acceptance covers the registered answer contract only."],
    )


def feedback(messages):
    return [
        json.loads(m.content.split(": ", 1)[1])
        for m in messages
        if isinstance(m, HumanMessage)
        and m.content.startswith("Repair gate feedback (trusted controller): ")
    ]


async def export_repair(resources, task_id, destination):
    destination.mkdir(parents=True, exist_ok=True)
    metadata = await resources.store.code_metadata(task_id)
    owners = [(task_id, destination)]
    from uuid import UUID

    owners += [
        (UUID(identity), destination / "contexts" / identity)
        for identity in metadata["request"]["context_task_ids"]
    ]
    for owner_id, directory in owners:
        directory.mkdir(parents=True, exist_ok=True)
        detail = await resources.store.code_task(owner_id)
        (directory / "task.json").write_text(
            json.dumps(detail, default=str, ensure_ascii=False, indent=2)
        )
        if owner_id == task_id:
            (directory / "candidate.json").write_text(
                json.dumps(
                    await resources.store.repair_candidate(owner_id),
                    default=str,
                    indent=2,
                )
            )
        for category in ("checks", "artifacts", "repair_attempts"):
            if category == "repair_attempts" and owner_id != task_id:
                continue
            records, cursor = [], None
            while True:
                page = await resources.store.code_page(owner_id, category, 100, cursor)
                records += page["items"]
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            (directory / (category + ".json")).write_text(
                json.dumps(records, default=str, ensure_ascii=False, indent=2)
            )
            if category == "artifacts":
                for artifact in records:
                    _, data = await resources.store.code_artifact(
                        owner_id, artifact["id"]
                    )
                    kind = artifact["kind"]
                    suffix = (
                        ".zip"
                        if kind
                        in (
                            "repair_package",
                            "repair_baseline",
                            "delivery_package",
                            "integration_package",
                        )
                        else ".md"
                        if kind.endswith("markdown")
                        else ".diff"
                        if kind
                        in ("repair_patch", "delivery_patch", "integration_patch")
                        else ".log"
                        if kind.endswith("log")
                        else ".json"
                    )
                    (
                        directory / (kind + "-" + str(artifact["id"]) + suffix)
                    ).write_bytes(data)


def test_repair_contract_and_default_off():
    assert not Registry().repair.settings.enabled
    with pytest.raises(ValidationError):
        CodeRepairCreate(task="fix", workspace_id="project")
    for target in (
        {"kind": "commit", "ref": "HEAD"},
        {"kind": "workspace", "mode": "staged"},
        {"kind": "range", "base": "x", "head": "y"},
    ):
        with pytest.raises(ValidationError):
            CodeRepairCreate(
                task="fix",
                workspace_id="project",
                delivery_mode="candidate_only",
                target=target,
            )


def test_repair_entry_authority_idempotency(database, tmp_path):
    registry, source, storage, _ = repair_fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            body = dict(
                stage="repair",
                task="fix",
                workspace_id="project",
                delivery_mode="candidate_only",
            )
            unauthorized = await client.post(
                "/v1/code/tasks", json={**body, "acceptance_suite_id": "unregistered"}
            )
            assert unauthorized.status_code == 422
            missing = await client.post(
                "/v1/code/tasks",
                json={k: v for k, v in body.items() if k != "delivery_mode"},
            )
            assert missing.status_code == 422
            for url, value in (
                ("/v1/runs", dict(task="fix")),
                ("/v1/tasks", dict(task="fix")),
                ("/v1/conversations", {}),
            ):
                response = await client.post(
                    url,
                    json={
                        **value,
                        "profile": REPAIR_PROFILE,
                        "workspace_id": "project",
                    },
                )
                assert response.status_code == 422
            first = await client.post(
                "/v1/code/tasks", json=body, headers={"Idempotency-Key": "repair"}
            )
            assert first.status_code == 202, first.text
            second = await client.post(
                "/v1/code/tasks", json=body, headers={"Idempotency-Key": "repair"}
            )
            assert second.status_code == 200
            conflict = await client.post(
                "/v1/code/tasks",
                json={**body, "acceptance_suite_id": "independent"},
                headers={"Idempotency-Key": "repair"},
            )
            assert conflict.status_code == 409
            detail = (await client.get("/v1/code/tasks/" + first.json()["id"])).json()
            assert (
                detail["delivery_status"] == "not_applied"
                and detail["delivery_mode"] == "candidate_only"
            )
        disabled = Registry(
            RuntimeFile(
                version=1,
                workspaces=[Workspace(id="project", root=str(source))],
                execution=registry.execution,
                code_repair=CodeRepairSettings(enabled=True, storage_root=str(storage)),
            )
        )
        r.store.registry = disabled
        with pytest.raises(ApiError, match="acceptance"):
            await r.store.create_code_task(
                CodeRepairCreate(
                    task="no authority",
                    workspace_id="project",
                    delivery_mode="candidate_only",
                )
            )

    database(check)


@pytest.mark.parametrize(
    "scenario",
    [
        "standalone",
        "two-failures",
        "user-changes",
        "test-bypass",
        "stale-check",
        "budget",
        "baseline-passed",
    ],
)
def test_repair_same_graph_versions_and_export(
    database, tmp_path, scenario, monkeypatch
):
    registry, source, storage, _ = repair_fixture(
        tmp_path,
        development=scenario in ("test-bypass", "stale-check"),
        gate_rounds=1 if scenario in ("test-bypass", "budget") else 4,
    )
    if scenario != "baseline-passed":
        (source / "sample.py").write_text("def answer():\n    return 2\n")
    if scenario == "user-changes":
        (source / "sample.py").write_text("def answer():\n    return 3\n")
        git(source, "add", "sample.py")
        (source / "sample.py").write_text("def answer():\n    return 4\n")
        (source / "user.txt").write_text("user untracked change\n")
        (source / "user.txt").chmod(0o755)
    before = files(source)
    before_modes = {
        p.relative_to(source).as_posix(): p.stat().st_mode
        for p in source.rglob("*")
        if p.is_file()
    }

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Make answer return one",
                workspace_id="project",
                delivery_mode="candidate_only",
            )
        )
        state = dict(step=0)
        captured = {}
        original_prepare = __import__(
            "agent_service.repair_execution", fromlist=["RepairExecutionScope"]
        ).RepairExecutionScope.prepare

        async def inspect_scope(scope):
            await original_prepare(scope)
            captured["scope"] = scope
            obj = await scope.client.inspect(scope.container_id)
            mounts = {m["Destination"]: m for m in obj["Mounts"]}
            assert mounts["/workspace"]["RW"] and not mounts["/workspace/.git"]["RW"]
            assert str(source) not in {m["Source"] for m in obj["Mounts"]}
            assert scope.manifest["content_digest"] == scope.current["content_digest"]

        monkeypatch.setattr(
            "agent_service.repair_execution.RepairExecutionScope.prepare", inspect_scope
        )

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            gates = feedback(messages)
            n = len(gates)
            if scenario in ("test-bypass", "budget"):
                if state["step"] == 0:
                    state["step"] += 1
                    return call(
                        "shell",
                        {
                            "command": "printf 'import unittest\nclass TestSafe(unittest.TestCase):\n    def test_value(self):\n        self.assertTrue(True)\n' > tests/test_sample.py; printf 'raise RuntimeError(\"candidate conftest must not load in acceptance\")\n' > conftest.py; printf '[pytest]\naddopts = --collect-only\n' > pytest.ini"
                        },
                        "weaken",
                    )
                return call("RepairDraftV1", draft(), "submit-failure")
            if scenario == "stale-check" and state["step"] < 3:
                step = state["step"]
                state["step"] += 1
                return [
                    call(
                        "shell",
                        {
                            "command": "printf 'def answer():\n    return 1\n' > sample.py"
                        },
                        "good",
                    ),
                    call("run_project_check", {"check_id": "unit"}, "old-pass"),
                    call(
                        "shell",
                        {
                            "command": "printf 'def answer():\n    return 8\n' > sample.py"
                        },
                        "break-after-check",
                    ),
                ][step]
            if scenario == "two-failures" and n < 2:
                return call("RepairDraftV1", draft(), "attempt-" + str(n))
            if scenario == "stale-check" and n == 0:
                return call("RepairDraftV1", draft(), "must-recheck")
            if state.get("fixed") is None and scenario != "baseline-passed":
                state["fixed"] = True
                return call(
                    "shell",
                    {
                        "command": "printf 'def answer():\n    return 1\n' > sample.py; printf '\\000\\377binary' > added.bin; printf 'new file\n' > new.py; chmod 755 new.py"
                    },
                    "fix",
                )
            if not state.get("read"):
                state["read"] = True
                return call("read_file", {"path": "sample.py"}, "final-read")
            meta = next(m.artifact for m in reversed(tools) if m.name == "read_file")
            ref = {
                k: meta[k]
                for k in ("evidence_id", "path", "sha256", "start_line", "end_line")
            }
            return call("RepairDraftV1", draft([ref]), "final-report")

        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        result = await execute(r, worker)
        success = scenario not in ("test-bypass", "budget")
        assert result["status"] == ("completed" if success else "failed"), result
        assert files(source) == before
        assert {
            p.relative_to(source).as_posix(): p.stat().st_mode
            for p in source.rglob("*")
            if p.is_file()
        } == before_modes
        assert not await r.store.code_resources()
        detail = await r.store.code_task(task["id"])
        assert detail["delivery_status"] == "not_applied"
        candidate = detail["candidate"]
        revisions = detail["revisions"]
        assert candidate and revisions
        assert (
            detail["report_artifact_id"] is not None
            if success
            else detail["diagnostic_artifact_id"] is not None
        )
        attempts = (await r.store.code_page(task["id"], "repair_attempts", 100))[
            "items"
        ]
        if scenario == "two-failures":
            assert (
                len(attempts) == 3
                and sum(a["status"] == "failed" for a in attempts) == 2
            )
            assert (
                candidate["usage"]["gate_rounds"] == 3
                and candidate["usage"]["model_calls"] >= 5
            )
            assert len({a["run_id"] for a in attempts}) == 1
            assert len({a["candidate_id"] for a in attempts}) == 1
        checks = (await r.store.code_page(task["id"], "checks", 100))["items"]
        assert any(c["purpose"] == "baseline" for c in checks)
        if scenario == "test-bypass":
            assert all(
                c["status"] != "passed" for c in checks if c["purpose"] == "acceptance"
            )
            assert any(
                c["purpose"] == "development" and c["status"] == "passed"
                for c in checks
            )
        if scenario == "stale-check":
            old = next(c for c in checks if c["tool_call_id"] == "tool:old-pass")
            assert old["status"] == "passed"
            assert any(
                c["purpose"] == "acceptance"
                and c["status"] == "failed"
                and c["revision_id"] != old["revision_id"]
                for c in checks
            )
        if success:
            assert candidate["status"] == "published"
            _, report = await r.store.code_artifact(
                task["id"], detail["report_artifact_id"]
            )
            report = json.loads(report)
            if scenario == "baseline-passed":
                assert (
                    "No baseline acceptance failure"
                    in report["authority"]["baseline_notice"]
                )
            final = next(
                rev for rev in revisions if rev["id"] == candidate["frozen_revision_id"]
            )
            _, manifest_bytes = await r.store.code_artifact(
                task["id"], final["manifest_artifact_id"]
            )
            manifest = json.loads(manifest_bytes)
            if scenario == "user-changes":
                assert {c["path"] for c in manifest["changes"]} == {
                    "sample.py",
                    "added.bin",
                    "new.py",
                }
                assert manifest["manifest"]["user.txt"]["mode"] == 0o755
            _, package = await r.store.code_artifact(
                task["id"], final["package_artifact_id"]
            )
            if scenario == "standalone":
                app.state.resources = r
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
                ) as client:
                    download = await client.get(
                        f"/v1/code/tasks/{task['id']}/artifacts/{final['package_artifact_id']}"
                    )
                    assert (
                        download.status_code == 200
                        and download.headers["content-type"] == "application/zip"
                    )
                    assert download.content == package
            with zipfile.ZipFile(io.BytesIO(package)) as archive:
                if scenario != "baseline-passed":
                    assert archive.read("added.bin") == b"\x00\xffbinary"
            final_state = await worker.agent.aget_state(result["final_checkpoint"])
            assert final_state.next == ()
        else:
            assert result["answer"] is None
            assert any(a["status"] == "failed" for a in attempts)
        output = (
            Path(os.environ.get("AGENT_REPAIR_EXAMPLE_DIR", str(tmp_path / "exports")))
            / scenario
        )
        await export_repair(r, task["id"], output)

    database(check)


@pytest.mark.parametrize(
    "boundary",
    [
        "plain",
        "mixed",
        "stale-source",
        "unfinished",
        "invalid",
        "special-file",
        "background",
    ],
)
def test_repair_gate_correction_boundaries(database, tmp_path, boundary):
    registry, source, _, _ = repair_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        state = {"step": 0}

        async def reply(messages):
            step = state["step"]
            state["step"] += 1
            if boundary == "plain" and step == 0:
                return AIMessage("Done without structured report")
            if boundary == "mixed" and step == 0:
                return AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "RepairDraftV1",
                            "args": draft(),
                            "id": "mixed-report",
                            "type": "tool_call",
                        },
                        {
                            "name": "shell",
                            "args": {"command": "touch forbidden-mixed"},
                            "id": "must-not-run",
                            "type": "tool_call",
                        },
                    ],
                )
            if boundary == "unfinished" and step == 0:
                return call(
                    "RepairDraftV1",
                    {**draft(), "unfinished": ["Actual target is unfinished"]},
                    "unfinished",
                )
            if boundary == "invalid" and step == 0:
                return call(
                    "RepairDraftV1", {"summary": "Incomplete JSON report"}, "invalid"
                )
            if boundary == "special-file":
                if step == 0:
                    return call(
                        "shell", {"command": "ln -s sample.py unsupported"}, "symlink"
                    )
                if step == 1:
                    last = next(
                        m for m in reversed(messages) if isinstance(m, ToolMessage)
                    )
                    assert last.status == "error"
                    return call("shell", {"command": "rm unsupported"}, "remove-link")
            if boundary == "background":
                if step == 0:
                    return call(
                        "shell",
                        {"command": "sleep 45 >/tmp/bg.log 2>&1 & echo $! > bg.pid"},
                        "background",
                    )
                if step == 1:
                    last = next(
                        m for m in reversed(messages) if isinstance(m, ToolMessage)
                    )
                    assert last.status == "error" and "processes" in last.content
                    return call(
                        "shell",
                        {"command": 'kill "$(cat bg.pid)"; rm bg.pid'},
                        "stop-background",
                    )
            if boundary == "stale-source":
                if step == 0:
                    return call("read_file", {"path": "sample.py"}, "old-read")
                if step == 1:
                    meta = next(
                        m.artifact
                        for m in reversed(messages)
                        if isinstance(m, ToolMessage) and m.name == "read_file"
                    )
                    state["old_ref"] = {
                        k: meta[k]
                        for k in (
                            "evidence_id",
                            "path",
                            "sha256",
                            "start_line",
                            "end_line",
                        )
                    }
                    return call(
                        "shell",
                        {
                            "command": "printf 'def answer():\n    return 1 # newer\n' > sample.py"
                        },
                        "newer-code",
                    )
                if step == 2:
                    return call("RepairDraftV1", draft([state["old_ref"]]), "stale-ref")
                if step == 3:
                    assert any(
                        "stale" in (f.get("reason") or "") for f in feedback(messages)
                    )
                    return call("read_file", {"path": "sample.py"}, "fresh-read")
                meta = next(
                    m.artifact
                    for m in reversed(messages)
                    if isinstance(m, ToolMessage) and m.name == "read_file"
                )
                return call(
                    "RepairDraftV1",
                    draft(
                        [
                            {
                                k: meta[k]
                                for k in (
                                    "evidence_id",
                                    "path",
                                    "sha256",
                                    "start_line",
                                    "end_line",
                                )
                            }
                        ]
                    ),
                    "fresh-report",
                )
            return call("RepairDraftV1", draft(), "corrected")

        worker = Worker(r, model=SyntheticModel(reply=reply))
        result = await execute(r, worker)
        assert result["status"] == "completed", result
        candidate = await r.store.repair_candidate(task["id"])
        assert candidate["candidate"]["status"] == "published"
        assert files(source) == before
        final = next(
            v
            for v in candidate["revisions"]
            if v["id"] == candidate["candidate"]["frozen_revision_id"]
        )
        _, data = await r.store.code_artifact(task["id"], final["manifest_artifact_id"])
        assert "forbidden-mixed" not in json.loads(data)["manifest"]
        if boundary == "mixed":
            final_state = await worker.agent.aget_state(result["final_checkpoint"])
            messages = final_state.values["messages"]
            assert any(
                isinstance(m, ToolMessage)
                and m.tool_call_id == "must-not-run"
                and m.status == "error"
                for m in messages
            )

    database(check)


@pytest.mark.parametrize("ack", ["revision", "check", "gate", "publication"])
def test_repair_lost_confirmation_never_replays_mutation_or_acceptance(
    database, tmp_path, monkeypatch, ack
):
    registry, source, _, _ = repair_fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        method = {
            "revision": "save_repair_revision",
            "check": "finish_code_check",
            "gate": "finish_repair_attempt",
            "publication": "finish",
        }[ack]
        original = getattr(r.store, method)
        calls = 0

        async def lose(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original(*args, **kwargs)
            # Lose a write checkpoint acknowledgement rather than the initial baseline.
            target = 2 if ack == "revision" else 1
            if calls == target:
                raise psycopg.OperationalError("Synthetic commit acknowledgement loss")
            return result

        monkeypatch.setattr(r.store, method, lose)
        model = SyntheticModel(
            responses=[
                call("shell", {"command": "printf x >> once.txt"}, "append-once"),
                call("RepairDraftV1", draft(), "report"),
            ]
        )
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        result = await execute(r, Worker(r, model=model))
        assert result["status"] == "completed", result
        assert len(model.history) == 2
        detail = await r.store.repair_candidate(task["id"])
        final = next(
            v
            for v in detail["revisions"]
            if v["id"] == detail["candidate"]["frozen_revision_id"]
        )
        _, package = await r.store.code_artifact(
            task["id"], final["package_artifact_id"]
        )
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            assert archive.read("once.txt") == b"x"
        checks = (await r.store.code_page(task["id"], "checks", 100))["items"]
        assert len(checks) == 2
        assert not await r.store.code_resources()

    database(check)


@pytest.mark.parametrize(
    "mode",
    [
        "cancel",
        "lease",
        "worker",
        "shell-timeout",
        "model-budget",
        "tool-budget",
        "storage-budget",
    ],
)
def test_repair_failures_retain_last_consistent_checkpoint(database, tmp_path, mode):
    from agent_service.code_execution import recover_code_resources
    from agent_service.config import Settings
    from agent_service.execution import recover_resources

    registry, source, storage, _ = repair_fixture(
        tmp_path,
        model_calls=2 if mode == "model-budget" else 32,
        tool_calls=1 if mode == "tool-budget" else 64,
        storage_bytes=25000 if mode == "storage-budget" else 4 * 1024**3,
        execution_overrides={"command_timeout": 1} if mode == "shell-timeout" else {},
    )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        entered = asyncio.Event()

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call(
                    "shell",
                    {"command": "printf 'retained\n' > retained.txt"},
                    "checkpoint",
                )
            entered.set()
            if mode == "shell-timeout":
                return call("shell", {"command": "sleep 30"}, "timeout")
            if mode == "tool-budget":
                return call("shell", {"command": "touch not-admitted"}, "excess-tool")
            if mode == "model-budget":
                return AIMessage("No report yet")
            if mode == "storage-budget":
                return call(
                    "shell",
                    {"command": "head -c 1000000 /dev/zero > huge.bin"},
                    "storage",
                )
            await asyncio.Event().wait()

        worker = Worker(r, model=SyntheticModel(reply=reply))
        row = await r.store.claim(worker.id)
        running = asyncio.create_task(worker.execute(row))
        try:
            if mode in ("cancel", "lease", "worker"):
                await asyncio.wait_for(entered.wait(), 15)
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
                await asyncio.wait_for(running, 20)
            except asyncio.CancelledError:
                if mode != "worker":
                    raise
            await r.store.maintain()
            await recover_code_resources(r.store, registry.execution)
            await recover_resources(r.store, registry.execution)
            await r.store.maintain()
            result = await r.store.get(row["id"])
            assert result["status"] in ("failed", "cancelled"), result
            assert result["answer"] is None
            detail = await r.store.repair_candidate(task["id"])
            assert detail["diagnostic_artifact_id"]
            assert detail["revisions"]
            if entered.is_set():
                last = detail["revisions"][-1]
                _, data = await r.store.code_artifact(
                    task["id"], last["manifest_artifact_id"]
                )
                assert "retained.txt" in json.loads(data)["manifest"]
            assert files(source) == before
            assert not await r.store.code_resources()
            output = Path(
                os.environ.get("AGENT_REPAIR_EXAMPLE_DIR", str(tmp_path / "exports"))
            ) / ("recovery-" + mode)
            output.mkdir(parents=True, exist_ok=True)
            _, data = await r.store.code_artifact(
                task["id"], detail["diagnostic_artifact_id"]
            )
            (output / "diagnostic.json").write_bytes(data)
            await export_repair(r, task["id"], output)
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    database(check, Settings(heartbeat_seconds=0.05, lease_seconds=2))


def test_historical_review_is_context_not_current_reproduction(database, tmp_path):
    from agent_service.code_registry import CodeReviewSettings
    from contracts.review import CodeReviewCreate

    registry, source, storage, _ = repair_fixture(tmp_path)
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=registry.execution,
            code_repair=registry.repair.settings,
            code_review=CodeReviewSettings(enabled=True, storage_root=str(storage)),
        ),
        acceptance_suites=tuple(registry.repair.acceptance.values()),
    )
    (source / "sample.py").write_text("def answer():\n    return 2\n")

    async def check(r):
        r.registry = r.store.registry = registry
        review, _ = await r.store.create_code_task(
            CodeReviewCreate(
                task="Historical review of return two", workspace_id="project"
            )
        )

        def prior_reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("review_diff", {}, "list")
            changes = json.loads(tools[-1].content)["records"]
            return call(
                "ReviewReportV1",
                dict(
                    version=1,
                    summary="Historical range contained return two; not reproduced.",
                    findings=[],
                    changes=[
                        dict(
                            change_id=c["change_id"],
                            status="unreviewed",
                            reason="Context only",
                            evidence=[],
                            missing_context=[],
                        )
                        for c in changes
                    ],
                    unconfirmed=["No reproduction was performed"],
                    missing_context=[],
                ),
                "report",
            )

        historical = await execute(
            r, Worker(r, model=SyntheticModel(reply=prior_reply))
        )
        assert historical["status"] == "completed", historical
        (source / "sample.py").write_text("def answer():\n    return 3\n")
        before = files(source)
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Return one",
                workspace_id="project",
                delivery_mode="candidate_only",
                context_task_ids=[review["id"]],
            )
        )
        state = {"step": 0}

        def repair_reply(messages):
            step = state["step"]
            state["step"] += 1
            if step == 0:
                return call(
                    "read_repair_context", {"task_id": str(review["id"])}, "context"
                )
            if step == 1:
                tool = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
                assert "Historical" in tool.content
                return call("read_file", {"path": "sample.py"}, "current")
            if step == 2:
                tool = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
                assert "return 3" in tool.content
                return call(
                    "shell",
                    {"command": "printf 'def answer():\n    return 1\n' > sample.py"},
                    "fix",
                )
            return call(
                "RepairDraftV1",
                {
                    **draft(),
                    "baseline_confirmation": "Current baseline returned three; historical return-two review and non-reproduction were not inherited.",
                },
                "report",
            )

        result = await execute(r, Worker(r, model=SyntheticModel(reply=repair_reply)))
        assert result["status"] == "completed", result
        assert files(source) == before
        detail = await r.store.code_task(task["id"])
        _, report = await r.store.code_artifact(
            task["id"], detail["report_artifact_id"]
        )
        output = (
            Path(os.environ.get("AGENT_REPAIR_EXAMPLE_DIR", str(tmp_path / "exports")))
            / "historical-review"
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_bytes(report)
        await export_repair(r, task["id"], output)

    database(check)


def test_hash_edits_checkpoint_create_delete_and_ignore_policy(database, tmp_path):
    registry, source, _, _ = repair_fixture(tmp_path)
    (source / "remove.txt").write_text("baseline delete\n")
    (source / "user-mode.txt").write_text("user permissions\n")
    (source / "user-mode.txt").chmod(0o640)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        state = {"step": 0}

        def reply(messages):
            step = state["step"]
            state["step"] += 1
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if step == 0:
                return call("read_file", {"path": "sample.py"}, "read")
            if step == 1:
                state["hash"] = tools[-1].artifact["sha256"]
                return call(
                    "edit_file",
                    {
                        "path": "sample.py",
                        "expected_hash": "0" * 64,
                        "old_text": "return 1",
                        "new_text": "return 0",
                    },
                    "wrong-hash",
                )
            if step == 2:
                assert tools[-1].status == "error"
                return call(
                    "edit_file",
                    {
                        "path": "sample.py",
                        "expected_hash": state["hash"],
                        "old_text": "return 1",
                        "new_text": "return 1 # confirmed",
                    },
                    "edit",
                )
            if step == 3:
                assert tools[-1].artifact["revision_id"]
                return call(
                    "edit_file",
                    {
                        "path": "created.py",
                        "create": True,
                        "new_text": "created = True\n",
                    },
                    "create",
                )
            if step == 4:
                return call(
                    "shell",
                    {
                        "command": "rm remove.txt; printf '*\n' > .gitignore; printf 'must export\n' > hidden.py; mkdir -p __pycache__; printf cache > __pycache__/ignore.pyc; chmod 600 user-mode.txt"
                    },
                    "delete-ignore-mode",
                )
            return call("RepairDraftV1", draft(), "report")

        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        detail = await r.store.repair_candidate(task["id"])
        revisions = detail["revisions"]
        assert not any(v["tool_call_id"] == "tool:wrong-hash" for v in revisions)
        assert {"tool:edit", "tool:create", "tool:delete-ignore-mode"} <= {
            v["tool_call_id"] for v in revisions
        }
        final = next(
            v for v in revisions if v["id"] == detail["candidate"]["frozen_revision_id"]
        )
        _, data = await r.store.code_artifact(task["id"], final["manifest_artifact_id"])
        doc = json.loads(data)
        changes = {c["path"]: c for c in doc["changes"]}
        assert changes["remove.txt"]["after"] is None
        assert changes["hidden.py"]["after"]
        assert changes["user-mode.txt"]["before"]["mode"] == 0o640
        assert changes["user-mode.txt"]["after"]["mode"] == 0o600
        assert not changes["user-mode.txt"]["text_diff_complete"]
        assert "__pycache__/ignore.pyc" not in doc["manifest"]
        assert files(source) == before

    database(check)


@pytest.mark.parametrize(
    "failure",
    ["cleanup", "asset-protocol", "check-timeout", "check-oom", "candidate-oom"],
)
def test_repair_control_faults_and_independent_check_feedback(
    database, tmp_path, monkeypatch, failure
):
    from agent_service.execution import DockerClient
    from agent_service.repair_execution import RepairExecutionScope

    registry, source, _, _ = repair_fixture(
        tmp_path, gate_rounds=1 if failure in ("check-timeout", "check-oom") else 4
    )
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        original_call = DockerClient.call
        checks = 0

        async def inject(client, *args, **kwargs):
            nonlocal checks
            if "/opt/agent/acceptance_runner.py" in args:
                checks += 1
                if checks == 2 and failure in (
                    "asset-protocol",
                    "check-timeout",
                    "check-oom",
                ):
                    if failure == "asset-protocol":
                        return (
                            0,
                            b'{"control_error":"assets changed"}',
                            b"asset failure",
                        )
                    return (
                        (124 if failure == "check-timeout" else 137),
                        b"",
                        b"Independent check interrupted",
                    )
            return await original_call(client, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "call", inject)
        if failure == "cleanup":
            original_close = RepairExecutionScope.close

            async def failed_close(scope, deadline=None):
                await original_close(scope, deadline)
                return False

            monkeypatch.setattr(RepairExecutionScope, "close", failed_close)
        responses = [call("RepairDraftV1", draft(), "report")]
        if failure == "candidate-oom":
            responses = [
                call("shell", {"command": "python -c 'x=bytearray(2*1024**3)'"}, "oom")
            ]
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        result = await execute(r, Worker(r, model=SyntheticModel(responses=responses)))
        assert result["status"] == "failed", result
        detail = await r.store.code_task(task["id"])
        assert detail["report_artifact_id"] is None and detail["diagnostic_artifact_id"]
        expected = {
            "cleanup": "RESOURCE_CLEANUP_FAILED",
            "asset-protocol": "ACCEPTANCE_INVALID",
            "check-timeout": "REPAIR_BUDGET_EXHAUSTED",
            "check-oom": "REPAIR_BUDGET_EXHAUSTED",
            "candidate-oom": "EXECUTION_ENVIRONMENT_LOST",
        }
        assert result["error_code"] == expected[failure], result
        records = (await r.store.code_page(task["id"], "checks", 100))["items"]
        if failure in ("check-timeout", "check-oom"):
            check = next(c for c in records if c["purpose"] == "acceptance")
            assert check["status"] == "inconclusive"
            assert ("timeout" if failure == "check-timeout" else "oom") in check[
                "result"
            ]["limitations"]
            attempt = (await r.store.code_page(task["id"], "repair_attempts"))["items"][
                0
            ]
            assert attempt["feedback"]["checks"][0]["diagnostic"]
        assert files(source) == before
        assert not await r.store.code_resources()

    database(check)


def test_repair_sigkill_restart_preserves_checkpoint_without_replay(database, tmp_path):
    import sys

    from agent_service.code_execution import recover_code_resources
    from agent_service.execution import recover_resources

    registry, source, storage, _ = repair_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        child = """
import asyncio,json,sys
from agent_service.resources import Resources
from agent_service.profiles import Registry,RuntimeFile
from agent_service.repair_registry import AcceptanceSuiteDefinition
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_code_analysis import call
v=json.loads(sys.argv[1])
s=v['suite']
for name in ('workspaces','required_cases','environment'): s[name]=tuple(s[name])
registry=Registry(RuntimeFile.model_validate_json(json.dumps(v['config'])),acceptance_suites=(AcceptanceSuiteDefinition(**s),))
async def reply(messages):
 from langchain_core.messages import ToolMessage
 if not any(isinstance(m,ToolMessage) for m in messages):
  return call('shell',{'command':"printf 'survives crash' > survives.txt"},'write-once')
 await asyncio.Event().wait()
async def main():
 async with Resources(registry=registry) as r:
  worker=Worker(r,model=SyntheticModel(reply=reply))
  row=await r.store.claim(worker.id)
  await worker.execute(row)
asyncio.run(main())
"""
        config = RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=registry.execution,
            code_repair=registry.repair.settings,
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            child,
            json.dumps(
                {
                    "config": config.model_dump(mode="json", exclude_defaults=True),
                    "suite": next(iter(registry.repair.acceptance.values())).manifest(),
                }
            ),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            for _ in range(400):
                detail = await r.store.repair_candidate(task["id"])
                if any(
                    v["tool_call_id"] == "tool:write-once" for v in detail["revisions"]
                ):
                    break
                if process.returncode is not None:
                    raise AssertionError((await process.stderr.read()).decode())
                await asyncio.sleep(0.025)
            else:
                raise AssertionError("Child repair did not checkpoint")
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
            detail = await r.store.repair_candidate(task["id"])
            assert detail["diagnostic_artifact_id"]
            assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
            assert not await r.store.code_resources()
            current = next(
                v
                for v in detail["revisions"]
                if v["id"] == detail["candidate"]["current_revision_id"]
            )
            _, data = await r.store.code_artifact(
                task["id"], current["package_artifact_id"]
            )
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                assert archive.read("survives.txt") == b"survives crash"
            assert files(source) == before
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    database(check)


def test_repair_gc_owns_writable_revisions_packages_and_abandoned_directories(
    database, tmp_path
):
    from uuid import uuid4

    registry, source, storage, _ = repair_fixture(tmp_path, retention_seconds=1)

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        result = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[call("RepairDraftV1", draft(), "report")]
                ),
            ),
        )
        assert result["status"] == "completed", result
        detail = await r.store.repair_candidate(task["id"])
        candidate_dir = storage / "candidates" / detail["candidate"]["id"].hex
        revision_dirs = [
            storage / "revisions" / v["id"].hex for v in detail["revisions"]
        ]
        assert candidate_dir.is_dir() and all(p.is_dir() for p in revision_dirs)
        abandoned = storage / "revisions" / uuid4().hex
        abandoned.mkdir()
        os.utime(abandoned, (1, 1))
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET finished_at=clock_timestamp()-interval '2 seconds' WHERE id=%s",
                (task["id"],),
            )
        await r.store.gc_code_content()
        assert (
            not candidate_dir.exists()
            and all(not p.exists() for p in revision_dirs)
            and not abandoned.exists()
        )
        artifact = detail["revisions"][-1]["package_artifact_id"]
        with pytest.raises(ApiError) as error:
            await r.store.code_artifact(task["id"], artifact)
        assert error.value.status == 410

    database(check)


@pytest.mark.parametrize(
    "fault", ["writable-fallback", "final-checkpoint", "package-bytes", "check-digest"]
)
def test_repair_rejects_mismatched_mount_or_publication(
    database, tmp_path, monkeypatch, fault
):
    registry, source, _, _ = repair_fixture(tmp_path)
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        if fault == "writable-fallback":
            original = r.store.create_repair_candidate

            async def unbound(row, *args):
                saved = await original(row, *args)
                async with r.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE agent.execution_resources SET execution_root=workspace_root WHERE run_id=%s",
                        (row["id"],),
                    )
                return saved

            monkeypatch.setattr(r.store, "create_repair_candidate", unbound)
        else:
            original = r.store.stage_code_report

            async def corrupt(row, *args):
                ids = await original(row, *args)
                async with r.pool.connection() as conn:
                    if fault == "final-checkpoint":
                        await conn.execute(
                            "UPDATE agent.code_repair_candidates SET final_checkpoint='{}' WHERE run_id=%s",
                            (row["id"],),
                        )
                    elif fault == "check-digest":
                        await conn.execute(
                            "UPDATE agent.code_check_runs SET result_digest='invalid' WHERE run_id=%s AND purpose='acceptance'",
                            (row["id"],),
                        )
                    else:
                        artifact = await (
                            await conn.execute(
                                "SELECT a.* FROM agent.code_artifacts a JOIN agent.code_repair_revisions v ON v.package_artifact_id=a.id JOIN agent.code_repair_candidates c ON c.frozen_revision_id=v.id WHERE c.run_id=%s",
                                (row["id"],),
                            )
                        ).fetchone()
                        path = r.store.artifact_storage.path(artifact["storage_key"])
                        data = bytearray(path.read_bytes())
                        data[0] ^= 1
                        path.write_bytes(data)
                return ids

            monkeypatch.setattr(r.store, "stage_code_report", corrupt)
        model = SyntheticModel(responses=[call("RepairDraftV1", draft(), "report")])
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="Repair", workspace_id="project", delivery_mode="candidate_only"
            )
        )
        result = await execute(r, Worker(r, model=model))
        assert result["status"] == "failed", result
        detail = await r.store.code_task(task["id"])
        assert detail["report_artifact_id"] is None
        assert detail["diagnostic_artifact_id"]
        assert detail["candidate"]["status"] == "failed"
        assert files(source) == before
        assert not await r.store.code_resources()

    database(check)


@pytest.mark.parametrize("limit", ["checks", "stage-time", "global-time"])
def test_repair_admission_reserves_baseline_and_one_gate(database, tmp_path, limit):
    from agent_service.config import Settings

    registry, _, _, _ = repair_fixture(
        tmp_path,
        **(
            {"check_calls": 1}
            if limit == "checks"
            else {"run_seconds": 121}
            if limit == "stage-time"
            else {}
        ),
    )

    async def check(r):
        r.registry = r.store.registry = registry
        with pytest.raises(ApiError) as error:
            await r.store.create_code_task(
                CodeRepairCreate(
                    task="Repair",
                    workspace_id="project",
                    delivery_mode="candidate_only",
                )
            )
        assert error.value.status == 422 and "budget" in error.value.message.lower()
        assert not (await r.store.list_tasks())["items"]

    database(check, Settings(run_timeout=121) if limit == "global-time" else Settings())
