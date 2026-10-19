"""Automatic delivery: real Git/PostgreSQL/Docker and real atomic renames."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from agent_service.worker import Worker
from contracts.repair import CodeRepairCreate
from tests.models import SyntheticModel
from tests.test_code_analysis import call, git
from tests.test_code_repair import draft, export_repair, repair_fixture
from tests.test_core import execute


def auto_fixture(tmp_path, **settings):
    private = tmp_path / "delivery-private"
    private.mkdir(mode=0o700)
    registry, source, storage, commit = repair_fixture(
        tmp_path, auto_apply_workspaces={"project": str(private)}, **settings
    )
    return registry, source, storage, private


@pytest.mark.parametrize(
    "scenario",
    [
        "simple",
        "no_changes",
        "clean_merge",
        "conflict",
        "merge_equal",
        "merged_failure",
        "rolling_revert",
        "cleanup_refresh",
        "refresh_limit",
        "inode_refresh",
        "ignored_files",
    ],
)
def test_auto_delivery(database, tmp_path, monkeypatch, scenario):
    registry, source, storage, private = auto_fixture(tmp_path)
    baseline = (
        "def answer():\n    return 2\n\ndef user_factor():\n    return 0\n\n"
        + "".join("# context %d\n" % n for n in range(12))
        + "# tail\n"
    )
    fixed = baseline.replace("return 2", "return 1")
    (source / "sample.py").write_text(fixed if scenario == "no_changes" else baseline)
    git(source, "add", "sample.py")
    (source / "user.txt").write_text("unstaged untracked\n")
    index = (source / ".git/index").read_bytes()

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="fix and deliver",
                workspace_id="project",
                delivery_mode="auto_apply",
            )
        )
        calls = 0

        async def reply(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                if scenario in ("clean_merge", "merged_failure", "rolling_revert"):
                    user = baseline.replace(
                        "# tail",
                        "# user change"
                        if scenario == "clean_merge"
                        else "# user change\nassert user_factor() == 1",
                    )
                    (source / "sample.py").write_text(user)
                if scenario == "conflict":
                    (source / "sample.py").write_text(
                        baseline.replace("return 2", "return 3")
                    )
                if scenario == "merge_equal":
                    (source / "sample.py").write_text(fixed)
                return call(
                    "shell",
                    {
                        "command": "sed -i 's/return 2/return 1/' sample.py"
                        + (
                            "; printf '\ngenerated.bin\n' >> .gitignore; printf payload > generated.bin"
                            if scenario == "ignored_files"
                            else ""
                        )
                    },
                    "edit",
                )
            if scenario in ("merged_failure", "rolling_revert") and calls == 3:
                # The merged user fault must remain protected. Repair by handling
                # it earlier, outside the protected text range.
                if scenario == "rolling_revert":
                    (source / "sample.py").write_text(baseline)
                return call(
                    "shell",
                    {"command": "sed -i 's/return 0/return 1/' sample.py"},
                    "fix-merged",
                )
            return call("RepairDraftV1", draft(), "report-" + str(calls))

        if scenario in ("cleanup_refresh", "refresh_limit", "inode_refresh"):
            original = r.store.bind_delivery_handoff
            once = False

            async def after_cleanup(row, checkpoint):
                nonlocal once
                await original(row, checkpoint)
                if not once or scenario == "refresh_limit":
                    once = (once or 0) + 1
                    if scenario == "inode_refresh":
                        replacement = source / "replacement"
                        replacement.write_text(baseline)
                        replacement.replace(source / "sample.py")
                    else:
                        (source / "sample.py").write_text(
                            baseline.replace(
                                "# tail", "# user after cleanup " + str(once)
                            )
                        )

            monkeypatch.setattr(r.store, "bind_delivery_handoff", after_cleanup)
        model = SyntheticModel(reply=reply)
        result = await execute(r, Worker(r, model=model))
        detail = await r.store.delivery_detail(task["id"])
        delivery = detail["delivery"]
        if scenario in ("conflict", "refresh_limit"):
            assert result["status"] == "failed", result
            assert delivery["status"] in ("pending", "rolled_back"), detail
            assert (
                "return 2" in (source / "sample.py").read_text()
                or scenario == "conflict"
            )
            if scenario == "conflict":
                assert result["error_code"] == "DELIVERY_CONFLICT"
            else:
                assert len(detail["integrations"]) >= 1
                assert delivery["refreshes"] == 3
                assert result["error_code"] == "DELIVERY_REFRESH_EXHAUSTED"
        else:
            assert result["status"] == "completed", (result, detail)
            assert delivery["status"] == "applied" and delivery["impact"] == (
                "none" if scenario in ("no_changes", "merge_equal") else "all"
            )
            assert delivery["phase"] == "published"
            assert (
                (source / "sample.py")
                .read_text()
                .startswith("def answer():\n    return 1")
            )
            if scenario in ("clean_merge", "cleanup_refresh", "merged_failure"):
                assert "# user" in (source / "sample.py").read_text()
            if scenario in ("clean_merge", "cleanup_refresh", "merge_equal"):
                merged = [i for i in detail["integrations"] if i["evidence"]["merged"]]
                assert merged and all(i["check_ids"] for i in merged)
                assert (
                    merged[-1]["candidate_revision_id"]
                    != merged[-1]["final_revision_id"]
                )
            if scenario in ("merged_failure", "rolling_revert"):
                assert any(i["status"] == "installed" for i in detail["integrations"])
                assert detail["integrations"][-1]["base_observation_id"] is not None
                assert len(model.history) == 4
                assert "return 1" in (source / "sample.py").read_text()
            if scenario == "rolling_revert":
                assert "assert user_factor()" not in (source / "sample.py").read_text()
            if scenario == "ignored_files":
                assert (source / "generated.bin").read_bytes() == b"payload"
            if scenario in ("cleanup_refresh", "inode_refresh"):
                assert delivery["refreshes"] == 1
                assert len(model.history) == 2
                assert len(delivery["handoffs"]) == 2
            report_meta = await r.store.code_metadata(task["id"])
            _, report = await r.store.code_artifact(
                task["id"], report_meta["report_artifact_id"]
            )
            report = json.loads(report)
            assert report["delivery_mode"] == "auto_apply"
            assert (
                report["delivery"]["final_digest"] == delivery["plan"]["final_digest"]
            )
        assert (source / ".git/index").read_bytes() == index
        assert (source / "user.txt").read_text() == "unstaged untracked\n"
        assert (
            len(await r.store.task_runs(task["id"])) == 1
            if hasattr(r.store, "task_runs")
            else True
        )
        export = os.environ.get("AGENT_DELIVERY_EXAMPLE_DIR")
        if export:
            directory = Path(export) / scenario
            await export_repair(r, task["id"], directory)
            (directory / "delivery.json").write_text(
                json.dumps(detail, default=str, indent=2)
            )

    database(check)


def _crash_worker(configuration, phase):
    """A real subprocess dies after a real rename, leaving PostgreSQL unconfirmed."""
    import asyncio
    import signal

    from agent_service.delivery_fs import FileTransaction
    from agent_service.execution_config import ExecutionSettings
    from agent_service.profiles import Registry, RuntimeFile, Workspace
    from agent_service.repair_registry import (
        AcceptanceSuiteDefinition,
        CodeRepairSettings,
    )
    from agent_service.resources import Resources

    runtime, acceptance = configuration
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(**x) for x in runtime["workspaces"]],
            execution=ExecutionSettings(**runtime["execution"]),
            code_repair=CodeRepairSettings(**runtime["repair"]),
        ),
        acceptance_suites=tuple(AcceptanceSuiteDefinition(**x) for x in acceptance),
    )
    if phase == "publication":

        async def crash_publish(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGKILL)

        Worker.publish = crash_publish
    elif phase == "intent":
        from agent_service.delivery_apply import DeliveryController

        original_journal = DeliveryController.journal

        async def crash_intent(self, phase, detail, **kwargs):
            result = await original_journal(self, phase, detail, **kwargs)
            if phase == "write_intent":
                os.kill(os.getpid(), signal.SIGKILL)
            return result

        DeliveryController.journal = crash_intent
    elif phase == "before_fsync":
        from agent_service import delivery_fs

        original_rename = delivery_fs.rename

        def crash_rename(*args):
            original_rename(*args)
            os.kill(os.getpid(), signal.SIGKILL)

        delivery_fs.rename = crash_rename
    else:
        original = FileTransaction.mutate

        def crash(self, operation, rollback=False):
            result = original(self, operation, rollback)
            if (
                phase == "partial"
                or (phase == "all" and operation == self.plan["files"][-1])
                or (phase.startswith("rollback:") and rollback)
            ):
                os.kill(os.getpid(), signal.SIGKILL)
            return result

        FileTransaction.mutate = crash

    async def run():
        async with Resources() as r:
            r.registry = r.store.registry = registry
            model = SyntheticModel(
                responses=[
                    call(
                        "write_todos",
                        {
                            "todos": [
                                {
                                    "content": "deliver the verified candidate",
                                    "status": "completed",
                                }
                            ]
                        },
                        "todos",
                    ),
                    call(
                        "shell",
                        {
                            "command": "printf 'agent a\n' > a.txt; printf 'agent new\n' > new.txt; sed -i 's/return 2/return 1/' sample.py"
                        },
                        "fix",
                    ),
                    call("RepairDraftV1", draft(), "done"),
                ]
            )
            if phase.startswith("rollback:"):
                from uuid import UUID

                from agent_service.delivery_recovery import recover_one

                await recover_one(r, UUID(phase.split(":", 1)[1]))
                raise AssertionError("Rollback fault was not reached")
            worker = Worker(r, model=model)
            row = await r.store.claim(worker.id)
            assert row
            await worker.execute(row)

    asyncio.run(run())


@pytest.mark.parametrize(
    "scenario",
    [
        "partial",
        "all",
        "publication",
        "third_state",
        "cancel",
        "timeout",
        "index_change",
        "intent",
        "before_fsync",
        "applied_cancel",
        "rollback_crash",
        "root_reconfigured",
    ],
)
def test_sigkill_recovery(database, tmp_path, scenario):
    import asyncio
    import multiprocessing

    from agent_service.delivery_recovery import recover_one

    registry, source, storage, private = auto_fixture(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    (source / "a.txt").write_text("original a\n")
    index = (source / ".git/index").read_bytes()
    configuration = (
        {
            "workspaces": [{"id": "project", "root": str(source)}],
            "execution": registry.execution.model_dump(),
            "repair": registry.repair.settings.model_dump(),
        },
        [s.manifest() for s in registry.repair.acceptance.values()],
    )

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="deliver", workspace_id="project", delivery_mode="auto_apply"
            )
        )
        process = multiprocessing.get_context("spawn").Process(
            target=_crash_worker,
            args=(
                configuration,
                ("publication" if scenario == "applied_cancel" else scenario)
                if scenario
                in ("all", "publication", "intent", "before_fsync", "applied_cancel")
                else "partial",
            ),
        )
        process.start()
        await asyncio.to_thread(process.join, 45)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
            pytest.fail("Fault worker did not reach the injection point")
        assert process.exitcode == -9, process.exitcode
        detail = await r.store.delivery_detail(task["id"])
        delivery = detail["delivery"]
        assert delivery["status"] in ("applying", "applied")
        run = await r.store.get(delivery["run_id"])
        assert run["status"] == "delivering"
        assert (source / "a.txt").read_text() == (
            "original a\n" if scenario == "intent" else "agent a\n"
        )
        if scenario == "third_state":
            (source / "a.txt").write_text("later user change\n")
        if scenario in ("cancel", "applied_cancel", "rollback_crash"):
            await r.store.cancel_task(task["id"])
        if scenario == "index_change":
            (source / "later.txt").write_text("later staging")
            git(source, "add", "later.txt")
        latest_index = (source / ".git/index").read_bytes()
        async with r.store.locked() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (run["id"],),
            )
            await conn.execute(
                "UPDATE agent.code_deliveries SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (delivery["id"],),
            )
            if scenario == "timeout":
                await conn.execute(
                    "UPDATE agent.code_deliveries SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (delivery["id"],),
                )
        if scenario == "root_reconfigured":
            from agent_service.code_artifacts import CodeFailure
            from agent_service.execution_config import ExecutionSettings
            from agent_service.profiles import Registry, RuntimeFile, Workspace
            from agent_service.repair_registry import (
                AcceptanceSuiteDefinition,
                CodeRepairSettings,
            )

            remapped = tmp_path / "remapped"
            remapped.mkdir()
            git(remapped, "init", "-q")
            (remapped / "user.txt").write_text("new registered root")
            runtime, acceptance = configuration
            changed = Registry(
                RuntimeFile(
                    version=1,
                    workspaces=[Workspace(id="project", root=str(remapped))],
                    execution=ExecutionSettings(**runtime["execution"]),
                    code_repair=CodeRepairSettings(**runtime["repair"]),
                ),
                acceptance_suites=tuple(
                    AcceptanceSuiteDefinition(**x) for x in acceptance
                ),
            )
            r.registry = r.store.registry = changed
            with pytest.raises(CodeFailure, match="Registered workspace root changed"):
                await recover_one(r, run["id"])
            assert (source / "a.txt").read_text() == "agent a\n"
            assert (source / "sample.py").read_text().endswith("return 2\n")
            assert not (source / "new.txt").exists()
            assert (remapped / "user.txt").read_text() == "new registered root"
            assert (source / ".git/index").read_bytes() == index
            r.registry = r.store.registry = registry
            return
        # Capture applied inode identities: completion-ack loss must never apply a
        # file a second time, including the already-applied prefix of a partial run.
        inode = (source / "a.txt").stat().st_ino
        if scenario == "rollback_crash":
            rollback_process = multiprocessing.get_context("spawn").Process(
                target=_crash_worker, args=(configuration, "rollback:" + str(run["id"]))
            )
            rollback_process.start()
            await asyncio.to_thread(rollback_process.join, 45)
            if rollback_process.is_alive():
                rollback_process.kill()
                await asyncio.to_thread(rollback_process.join, 5)
                pytest.fail("Rollback fault process did not exit")
            assert rollback_process.exitcode == -9
            assert (source / "a.txt").read_text() == "original a\n"
            async with r.store.locked() as conn:
                await conn.execute(
                    "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (run["id"],),
                )
                await conn.execute(
                    "UPDATE agent.code_deliveries SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (delivery["id"],),
                )
        outcome = await recover_one(r, run["id"])
        if scenario == "third_state":
            assert outcome["status"] == "needs_recovery", outcome
            assert (source / "a.txt").read_text() == "later user change\n"
            observed = await r.store.delivery_detail(task["id"])
            conflict = next(f for f in observed["files"] if f["path"] == "a.txt")
            assert conflict["phase"] == "rollback_blocked"
            assert (
                conflict["actual"]["target"]["sha256"]
                == hashlib.sha256(b"later user change\n").hexdigest()
            )
            assert not (source / "new.txt").exists()
            await r.store.create_code_task(
                CodeRepairCreate(
                    task="blocked by recovery",
                    workspace_id="project",
                    delivery_mode="auto_apply",
                )
            )
            assert await r.store.claim(Worker(r, model=SyntheticModel()).id) is None
            async with r.store.locked() as conn:
                await conn.execute(
                    "UPDATE agent.tasks SET status='failed',finished_at=clock_timestamp()-interval '8 days' WHERE id=%s",
                    (task["id"],),
                )
            await r.store.gc_code_content()
            assert (private / delivery["plan"]["id"]).exists()
        elif scenario in ("cancel", "timeout", "index_change", "rollback_crash"):
            assert outcome["status"] == "rolled_back", outcome
            assert (source / "a.txt").read_text() == "original a\n"
            assert (source / "sample.py").read_text().endswith("return 2\n")
        else:
            assert outcome["status"] == "applied" and outcome["phase"] == "published", (
                outcome
            )
            if scenario != "intent":
                assert (source / "a.txt").stat().st_ino == inode
            assert (source / "new.txt").read_text() == "agent new\n"
            assert (source / "sample.py").read_text().endswith("return 1\n")
            completed = await r.store.get(run["id"])
            assert completed["status"] == (
                "cancelled" if scenario == "applied_cancel" else "completed"
            ), completed
            saved = await r.checkpointer.aget_tuple(outcome["final_checkpoint"])
            assert saved.checkpoint["channel_values"]["todos"] == [
                {"content": "deliver the verified candidate", "status": "completed"}
            ]
            metadata = await r.store.code_metadata(task["id"])
            _, data = await r.store.code_artifact(
                task["id"], metadata["report_artifact_id"]
            )
            report = json.loads(data)
            assert report["delivery"]["recovery"]["event_count"] >= 1
            assert report["delivery"]["cancel_requested"] == (
                scenario == "applied_cancel"
            )
        assert (source / ".git/index").read_bytes() == latest_index
        if scenario != "index_change":
            assert latest_index == index
        if os.environ.get("AGENT_DELIVERY_EXAMPLE_DIR"):
            directory = Path(os.environ["AGENT_DELIVERY_EXAMPLE_DIR"]) / (
                "recovery-" + scenario
            )
            await export_repair(r, task["id"], directory)
            (directory / "delivery.json").write_text(
                json.dumps(
                    await r.store.delivery_detail(task["id"]), default=str, indent=2
                )
            )

    database(check)


@pytest.mark.parametrize(
    "tamper", ["native_checkpoint", "frozen_bytes", "check_result"]
)
def test_delivery_requires_bound_evidence(database, tmp_path, monkeypatch, tamper):
    from psycopg.types.json import Jsonb

    registry, source, storage, private = auto_fixture(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="fix", workspace_id="project", delivery_mode="auto_apply"
            )
        )
        original = r.store.bind_delivery_handoff

        async def damage(row, checkpoint):
            await original(row, checkpoint)
            delivery = await r.store.delivery_for_run(row["id"])
            if tamper == "native_checkpoint":
                bad = {
                    **checkpoint,
                    "configurable": {
                        **checkpoint["configurable"],
                        "thread_id": "another-run",
                    },
                }
                async with r.store.locked() as conn:
                    await conn.execute(
                        "UPDATE agent.code_deliveries SET checkpoint=%s WHERE id=%s",
                        (Jsonb(bad), delivery["id"]),
                    )
            elif tamper == "check_result":
                async with r.store.locked() as conn:
                    await conn.execute(
                        "UPDATE agent.code_check_runs SET status='failed' WHERE id=%s",
                        (delivery["plan"]["final_check_ids"][0],),
                    )
            else:
                async with r.pool.connection() as conn:
                    revision = await (
                        await conn.execute(
                            "SELECT storage_key FROM agent.code_repair_revisions WHERE id=%s",
                            (delivery["plan"]["final_revision_id"],),
                        )
                    ).fetchone()
                (
                    r.store.artifact_storage.path(revision["storage_key"])
                    / "source/sample.py"
                ).write_text("damaged frozen bytes")

        monkeypatch.setattr(r.store, "bind_delivery_handoff", damage)
        model = SyntheticModel(
            responses=[
                call(
                    "shell",
                    {"command": "sed -i 's/return 2/return 1/' sample.py"},
                    "fix",
                ),
                call("RepairDraftV1", draft(), "submit"),
            ]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["status"] != "completed"
        assert (source / "sample.py").read_text().endswith("return 2\n")
        delivery = await r.store.delivery_for_run(result["id"])
        assert delivery["status"] == "needs_recovery", delivery
        assert len(model.history) == 2

    database(check)


def test_binary_add_delete_mode_and_resolved_retention(database, tmp_path):
    registry, source, storage, private = auto_fixture(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    (source / "delete.bin").write_bytes(b"\0old")
    (source / "mode").write_bytes(b"unchanged")
    index = (source / ".git/index").read_bytes()

    async def check(r):
        r.registry = r.store.registry = registry
        task, _ = await r.store.create_code_task(
            CodeRepairCreate(
                task="fix", workspace_id="project", delivery_mode="auto_apply"
            )
        )
        model = SyntheticModel(
            responses=[
                call(
                    "shell",
                    {
                        "command": "sed -i 's/return 2/return 1/' sample.py; printf '\\000\\377new' > binary; rm delete.bin; chmod 755 mode; mkdir -p newdir/sub; printf nested > newdir/sub/new"
                    },
                    "fix",
                ),
                call("RepairDraftV1", draft(), "submit"),
            ]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["status"] == "completed", result
        assert (source / "binary").read_bytes() == b"\0\xffnew"
        assert not (source / "delete.bin").exists()
        assert (source / "mode").stat().st_mode & 0o777 == 0o755
        assert (source / "newdir/sub/new").read_bytes() == b"nested"
        assert (source / ".git/index").read_bytes() == index
        delivery = await r.store.delivery_for_run(result["id"])
        staging = private / delivery["plan"]["id"]
        assert staging.exists()
        async with r.store.locked() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET finished_at=clock_timestamp()-interval '8 days' WHERE id=%s",
                (task["id"],),
            )
        await r.store.gc_code_content()
        assert not staging.exists()
        assert (source / "sample.py").read_text().endswith("return 1\n")
        assert (source / ".git/index").read_bytes() == index

    database(check)


def test_auto_apply_needs_separate_authorization_and_budget(database, tmp_path):
    from contracts.errors import ApiError

    registry, source, storage, _ = repair_fixture(tmp_path)

    async def check(r):
        r.registry = r.store.registry = registry
        with pytest.raises(ApiError, match="not authorized"):
            await r.store.create_code_task(
                CodeRepairCreate(
                    task="fix", workspace_id="project", delivery_mode="auto_apply"
                )
            )

    database(check)
    with pytest.raises(ValueError, match="Budget"):
        registry.repair.settings = registry.repair.settings.model_copy(
            update={"check_calls": 2}
        )
        registry.repair.validate_budget(
            (),
            acceptance=next(iter(registry.repair.acceptance.values())),
            auto_apply=True,
        )
