"""Third batch: frozen evidence, native loop limits and real publication boundaries."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import psycopg
import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from agent_service.app import app
from agent_service.execution import ExecutionScope
from agent_service.execution_errors import EnvironmentLost
from agent_service.failure import exception_code, failure_info
from agent_service.no_progress import NoProgress, argument_fingerprint
from agent_service.profiles import (
    Extension,
    PermissionPolicy,
    Profile,
    Registry,
    RuntimeFile,
    Workspace,
)
from agent_service.runtime import RunContext, build_agent, run_config
from agent_service.verification import (
    VerificationCheck,
    VerificationResult,
    VerificationReport,
)
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_background import setup, task, poll
from tests.test_core import execute
from tests.test_execution import install as install_workspace
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install


def registry(extension=None, *, tools=(), validators=None):
    return Registry(
        RuntimeFile(
            version=1,
            profiles=[Profile(id="verified", tools=tuple(t.name for t in tools))],
        ),
        tools=tools,
        validators=validators,
        extensions={"verified": extension} if extension else None,
    )


async def passed(context, answer, checkpoint):
    assert context.candidate_answer == answer and context.checkpoint == checkpoint
    assert not hasattr(context, "task_store") and not hasattr(
        context, "execution_authority"
    )
    assert context.remaining_execution() > 0
    return VerificationResult(status="passed", summary="synthetic")


@pytest.mark.parametrize(
    "values",
    [
        {"verification_mode": "auto"},
        {"verification_mode": True},
        {"verification_mode": "required"},
        {"verifier": "module.function"},
        {"verifier": lambda a, b, c: None},
        {"verification_timeout": 0},
        {"verification_timeout": True},
        {"verification_timeout": "10"},
        {"verification_timeout": float("nan")},
        {"verification_timeout": float("inf")},
        {"no_progress_enabled": 1},
        {"no_progress_enabled": "false"},
    ],
)
def test_extension_rejects_invalid_registration(values):
    with pytest.raises(ValueError):
        Extension(**values)


def test_registration_async_signature_and_legacy_binding():
    async def invalid(a):
        return None

    with pytest.raises(ValueError):
        Extension(verifier=invalid)
    default = registry().bind("verified")
    assert default == registry(Extension()).bind("verified")
    assert default == registry(Extension(verifier=passed, verification_timeout=1)).bind(
        "verified"
    )
    assert "verification" not in default and "no_progress" not in default
    report = registry(Extension(verification_mode="report")).bind("verified")
    assert report != default
    assert report != registry(
        Extension(verification_mode="report", verifier=passed)
    ).bind("verified")
    assert report != registry(
        Extension(verification_mode="report", verification_timeout=2)
    ).bind("verified")
    assert default != registry(Extension(no_progress_enabled=True)).bind("verified")
    assert default != registry(Extension(version="2")).bind("verified")


@pytest.mark.parametrize(
    "values",
    [
        {"status": "success"},
        {"status": True},
        {"status": "passed", "run_id": "forged"},
        {"status": "passed", "checkpoint": {}},
        {"status": "passed", "summary": "中" * 342},
        {
            "status": "passed",
            "checks": tuple(VerificationCheck(status="passed") for _ in range(17)),
        },
        {"status": "passed", "summary": 10},
        {"status": "passed", "reason": "unbounded text"},
        {"status": "passed", "checks": [{"status": "passed"}]},
    ],
)
def test_result_contract(values):
    with pytest.raises(ValidationError):
        VerificationResult(**values)
    with pytest.raises(ValidationError):
        VerificationCheck(status="passed", explanation="中" * 171)


@pytest.mark.parametrize("mode", [None, "report", "required"])
@pytest.mark.parametrize("status", ["passed", "failed", "inconclusive", "not_checked"])
def test_mode_matrix_empty_coverage_and_exact_checkpoint(database, mode, status):
    async def check(r):
        invocations = []

        async def verifier(ctx, answer, cp):
            invocations.append((ctx, answer, cp))
            saved = await r.checkpointer.aget_tuple({"configurable": cp.model_dump()})
            assert (
                saved
                and saved.checkpoint["channel_values"]["messages"][-1].text == answer
            )
            with pytest.raises((AttributeError, TypeError)):
                ctx.profile = "other"
            return VerificationResult(
                status=status, checks=(VerificationCheck(status=status),)
            )

        install(r, registry(Extension(verification_mode=mode, verifier=verifier)))
        await r.store.submit("test", profile="verified")
        model = SyntheticModel()
        run = await execute(r, Worker(r, model=model))
        assert len(model.history) == 1 and len(invocations) == int(mode is not None)
        expected = "completed" if mode != "required" or status == "passed" else "failed"
        assert run["status"] == expected, run
        if mode:
            report = VerificationReport.model_validate_json(
                json.dumps(run["verification"])
            )
            assert (
                report.status == status and report.final_candidate and not report.files
            )
            assert (
                report.candidate_sha256
                == hashlib.sha256(b"synthetic answer").hexdigest()
            )
            assert report.checkpoint.checkpoint_id == invocations[0][2].checkpoint_id
        else:
            assert run["verification"] is None
        if expected == "failed":
            assert run["answer"] is None and run["final_checkpoint"] is None
            assert run["error_code"] == (
                "VERIFICATION_FAILED"
                if status == "failed"
                else "VERIFICATION_INCONCLUSIVE"
            )

    database(check)


def test_report_without_verifier_and_current_task_query(database):
    async def check(r):
        install(r, registry(Extension(verification_mode="report")))
        t, _ = await r.store.create_task("test", profile="verified")
        assert (await r.store.task(t["id"]))["verification"] is None
        run = await execute(r)
        assert run["status"] == "completed"
        assert run["verification"]["reason"] == "verifier_not_registered"
        assert (await r.store.task(t["id"]))["verification"] == run["verification"]
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://127.0.0.1",
        ) as client:
            for path in (
                f"/v1/runs/{run['id']}",
                f"/v1/runs/{run['id']}/trace",
                f"/v1/tasks/{t['id']}",
            ):
                response = await client.get(path)
                assert response.status_code == 200, response.text
                assert response.json()["verification"] == run["verification"]

    database(check)


@pytest.mark.parametrize("mode", ["report", "required"])
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("timeout", None),
        ("own_timeout", "VERIFICATION_EXECUTION_FAILED"),
        ("exception", "VERIFICATION_EXECUTION_FAILED"),
        ("return_dict", "VERIFICATION_CONTRACT_ERROR"),
        ("oversize", "VERIFICATION_CONTRACT_ERROR"),
        ("serialization", "VERIFICATION_CONTRACT_ERROR"),
        ("forged", "VERIFICATION_CONTRACT_ERROR"),
        ("access", "VERIFICATION_CONTRACT_ERROR"),
        ("caught_access", "VERIFICATION_CONTRACT_ERROR"),
        ("persistence", "PERSISTENCE_FAILED"),
        ("environment", "EXECUTION_ENVIRONMENT_LOST"),
        ("group", "PERSISTENCE_FAILED"),
    ],
)
def test_verifier_exception_contract(database, mode, kind, expected):
    async def check(r):
        async def verifier(ctx, answer, cp):
            if kind == "timeout":
                await asyncio.sleep(1)
            if kind == "own_timeout":
                raise TimeoutError("private timeout")
            if kind == "exception":
                raise ValueError("private details")
            if kind == "return_dict":
                return {"status": "passed"}
            if kind == "oversize":
                return VerificationResult.model_construct(
                    status="passed", summary="x" * 1025
                )
            if kind == "serialization":
                return VerificationResult.model_construct(
                    status="passed", summary=object()
                )
            if kind == "forged":
                return VerificationResult.model_copy(
                    VerificationResult(status="passed"),
                    update={"source_run_id": "forged"},
                )
            if kind in ("access", "caught_access"):
                if kind == "access":
                    await ctx.read_file("test")
                else:
                    try:
                        await ctx.read_file("test")
                    except Exception:
                        pass
                    return VerificationResult(status="passed")
            if kind == "persistence":
                raise psycopg.OperationalError("private details")
            if kind == "environment":
                raise EnvironmentLost("private details")
            if kind == "group":
                raise ExceptionGroup(
                    "private details",
                    [ValueError("private"), psycopg.OperationalError("private")],
                )

        install(
            r,
            registry(
                Extension(
                    verification_mode=mode, verifier=verifier, verification_timeout=0.05
                )
            ),
        )
        await r.store.submit("test", profile="verified")
        model = SyntheticModel()
        run = await execute(r, Worker(r, model=model))
        code = expected or ("VERIFICATION_INCONCLUSIVE" if mode == "required" else None)
        assert run["error_code"] == code, run
        assert run["status"] == ("failed" if code else "completed")
        assert len(model.history) == 1
        assert "private" not in json.dumps(run["verification"])
        if kind == "timeout":
            assert run["verification"]["reason"] == "timeout"

    database(check)


@pytest.mark.parametrize(
    "kind",
    [
        "stable",
        "partial",
        "changed",
        "deleted",
        "repeated",
        "denied",
        "path",
        "recheck",
        "budget",
        "path_budget",
    ],
)
def test_controlled_file_coverage_and_cleanup_order(
    database, tmp_path, monkeypatch, kind
):
    path = tmp_path / "a.txt"
    path.write_text("first\nsecond\nthird\n")

    async def check(r):
        original_read = ExecutionScope.file_call
        original_close = ExecutionScope.close
        events = []
        verifier_calls = 0

        async def read(scope, name, args):
            events.append("read")
            if kind == "recheck" and len(events) > 2:
                raise OSError("private details")
            return await original_read(scope, name, args)

        async def close(scope, *args):
            events.append("cleanup")
            return await original_close(scope, *args)

        async def verifier(ctx, answer, cp):
            nonlocal verifier_calls
            verifier_calls += 1
            events.append("verify")
            if kind == "path":
                await ctx.read_file("../outside")
            elif kind == "path_budget":
                await ctx.read_file("x" * 513)
            elif kind == "budget":
                for _ in range(65):
                    await ctx.read_file("a.txt")
            else:
                value = await ctx.read_file(
                    "./a.txt",
                    start_line=2 if kind == "partial" else 1,
                    max_lines=1 if kind == "partial" else 200,
                )
                assert value.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
                if kind == "changed":
                    path.write_text("changed")
                if kind == "deleted":
                    path.unlink()
                if kind == "repeated":
                    path.write_text("different")
                    await ctx.read_file("a.txt")
            return VerificationResult(
                status="passed",
                checks=(
                    VerificationCheck(
                        status="passed", explanation="Observed candidate"
                    ),
                ),
            )

        install_workspace(
            r,
            tmp_path,
            extensions={
                "workspace_read": Extension(
                    verification_mode="required", verifier=verifier
                )
            },
        )
        if kind == "denied":
            # Trusted Profile validator applies equally at the verifier read boundary.
            reg = Registry(
                RuntimeFile(
                    version=1,
                    execution=r.registry.execution,
                    workspaces=[Workspace(id="source", root=str(tmp_path))],
                ),
                validators={"read_file": lambda args, ctx: False},
                extensions={
                    "workspace_read": Extension(
                        verification_mode="required", verifier=verifier
                    )
                },
            )
            install(r, reg)
        monkeypatch.setattr(ExecutionScope, "file_call", read)
        monkeypatch.setattr(ExecutionScope, "close", close)
        await r.store.submit("test", profile="workspace_read", workspace_id="source")
        run = await execute(r)
        assert (
            verifier_calls == 1 and events[0] == "verify" and events[-1] == "cleanup"
        ), events
        report = run["verification"]
        assert report is not None, run
        assert (await r.store.execution_resource(run["id"]))["state"] == "released"
        if kind in ("stable", "partial"):
            assert run["status"] == "completed", run
            observation = report["files"][0]
            assert observation["path"] == "a.txt" and observation["consistent"]
            assert observation["start_line"] == (2 if kind == "partial" else 1)
            assert observation["end_line"] == (2 if kind == "partial" else 3)
            assert observation["truncated"] == (kind == "partial")
        else:
            assert run["status"] == "failed", run
            assert run["error_code"] == (
                "VERIFICATION_CONTRACT_ERROR"
                if kind in ("denied", "path")
                else "VERIFICATION_INCONCLUSIVE"
            )
            if kind in ("changed", "deleted", "repeated"):
                assert report["reason"] == "workspace_changed"
        assert (
            "first" not in json.dumps(report)
            and len(json.dumps(report).encode()) <= 32768
        )

    database(check)


@pytest.mark.parametrize("kind", ["cleanup", "cancel", "lease", "deadline"])
def test_passed_report_never_overrides_control(database, tmp_path, monkeypatch, kind):
    async def check(r):
        async def verifier(ctx, answer, cp):
            if kind == "deadline":
                await asyncio.sleep(3)
            return VerificationResult(status="passed")

        install_workspace(
            r,
            tmp_path,
            extensions={
                "workspace_read": Extension(
                    verification_mode="required", verifier=verifier
                )
            },
        )
        original = ExecutionScope.close

        async def close(scope, *args):
            result = await original(scope, *args)
            if kind == "cleanup":
                return False
            if kind == "cancel":
                await r.store.cancel(scope.row["id"])
            if kind == "lease":
                async with r.store.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                        (scope.row["id"],),
                    )
            return result

        monkeypatch.setattr(ExecutionScope, "close", close)
        await r.store.submit("test", profile="workspace_read", workspace_id="source")
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        if kind == "deadline":
            # Reserve cleanup/publication, leaving only a short verification window.
            row["deadline_at"] = datetime.now(timezone.utc) + timedelta(
                seconds=r.registry.execution.cleanup_timeout
                + r.registry.reliability.publication_reserve
                + 0.7
            )
        await worker.execute(row)
        run = await r.store.get(row["id"])
        assert run["status"] in ("failed", "cancelled"), run
        assert run["answer"] is None and run["final_checkpoint"] is None
        if kind in ("cleanup", "cancel"):
            assert run["verification"]["status"] == "passed"
        if kind == "deadline":
            assert run["error_code"] == "RUN_TIMEOUT"

    database(check)


def test_publication_ack_loss_and_mismatch_no_replay(database, monkeypatch):
    async def check(r):
        calls = 0

        async def verifier(*args):
            nonlocal calls
            calls += 1
            return VerificationResult(status="passed")

        install(r, registry(Extension(verification_mode="required", verifier=verifier)))
        await r.store.submit("test", profile="verified")
        original = r.store.finish
        publishes = []

        async def finish(*args, **kwargs):
            publishes.append(kwargs)
            assert await original(*args, **kwargs)
            raise psycopg.OperationalError("commit acknowledgement lost")

        monkeypatch.setattr(r.store, "finish", finish)
        model = SyntheticModel()
        worker = Worker(r, model=model)
        run = await execute(r, worker)
        assert (
            run["status"] == "completed"
            and calls == 1
            and len(model.history) == 1
            and len(publishes) == 1
        )

        async def transport_lost(*args, **kwargs):
            raise psycopg.OperationalError("commit acknowledgement lost")

        monkeypatch.setattr(r.store, "finish", transport_lost)
        frozen = {k: v for k, v in publishes[0].items() if k != "delivered"}
        for field, value in (
            ("answer", "different"),
            ("verification", None),
            ("checkpoint", {}),
        ):
            row = {"id": run["id"], "lease_token": run["lease_token"]}
            assert not await worker.publish(row, **{**frozen, field: value})
        assert calls == 1 and len(model.history) == 1

    database(check)


@pytest.mark.parametrize(
    "field",
    [
        "source_run_id",
        "candidate_sha256",
        "checkpoint",
        "binding_fingerprint",
        "mode",
        "extension_version",
    ],
)
def test_database_rejects_report_binding_mismatch(database, field):
    async def check(r):
        install(r, registry(Extension(verification_mode="required", verifier=passed)))
        await r.store.submit("test", profile="verified")
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        outcome = await worker.generate_and_verify(row, None)
        report = json.loads(json.dumps(outcome.verification))
        report[field] = (
            {
                "thread_id": str(row["id"]),
                "checkpoint_ns": "",
                "checkpoint_id": "foreign",
            }
            if field == "checkpoint"
            else ("report" if field == "mode" else "0" * 64)
        )
        with pytest.raises(ValueError):
            await r.store.finish(
                row["id"],
                row["lease_token"],
                answer=outcome.answer,
                checkpoint=outcome.checkpoint,
                verification=report,
            )
        with pytest.raises(ValueError):
            await r.store.finish(
                row["id"],
                row["lease_token"],
                answer=outcome.answer,
                checkpoint=outcome.checkpoint,
            )
        assert (await r.store.get(row["id"]))["status"] == "running"
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST")

    database(check)


def test_required_stage_race_inheritance_and_trusted_delivery(
    database, tmp_path, monkeypatch
):
    async def check(r):
        verifier_calls = []

        async def verifier(ctx, answer, cp):
            verifier_calls.append(ctx.run_id)
            return VerificationResult(status="passed")

        setup(
            r,
            tmp_path,
            extensions={
                "task_work": Extension(verification_mode="required", verifier=verifier)
            },
        )
        t = await task(r)
        first_model = SyntheticModel(
            responses=[
                call(
                    "start_background_shell",
                    {"command": "sleep 0.3; echo trusted-result"},
                ),
                AIMessage("unverified-stage-message"),
            ]
        )
        first_worker = Worker(r, model=first_model)
        original = r.store.finish

        async def finish(*args, **kwargs):
            if kwargs.get("verification", {}).get("reason") == "stage_report":
                # Complete the operation after snapshot but before publication.
                await poll(r, t["id"])
            return await original(*args, **kwargs)

        monkeypatch.setattr(r.store, "finish", finish)
        first = await execute(r, first_worker)
        assert (
            first["status"] == "completed"
            and first["verification"]["reason"] == "stage_report"
        ), first
        assert not verifier_calls
        current = await r.store.task(t["id"])
        assert (
            current["status"] != "completed" and current["last_success_run_id"] is None
        )
        await r.store.maintain()
        current = await r.store.task(t["id"])
        assert (
            current["verification"] is None
        )  # current queued Run must not borrow old evidence
        successor = await r.store.get(current["current_run_id"])
        assert successor["previous_run_id"] is None
        assert (await r.store.operations(t["id"]))[0]["consumed_run_id"] is None
        second_model = SyntheticModel(responses=[AIMessage("verified-final")])
        second = await execute(r, Worker(r, model=second_model))
        data = "\n".join(m.text for m in second_model.history[0])
        assert "trusted-result" in data and "unverified-stage-message" not in data
        assert verifier_calls == [str(second["id"])]
        assert (await r.store.task(t["id"]))["status"] == "completed"
        assert (await r.store.operations(t["id"]))[0]["consumed_run_id"] == second["id"]

    database(check)


@tool
async def numeric(value: int) -> str:
    """A synthetic strictly validated tool."""
    return str(value)


async def invoke(graph):
    return await graph.ainvoke(
        {"messages": [HumanMessage("test")]},
        run_config("loop"),
        context=RunContext("loop", agent_id="root"),
    )


@pytest.mark.parametrize("code", ["INVALID_TOOL_ARGUMENTS", "PERMISSION_DENIED"])
def test_three_trusted_failures_stop_before_fourth_model(code):
    async def check():
        args = {"value": "bad"} if code == "INVALID_TOOL_ARGUMENTS" else {"value": 1}
        model = SyntheticModel(
            responses=[call("numeric", args, str(i)) for i in range(4)]
        )
        graph = build_agent(
            model,
            tools=[numeric],
            no_progress_enabled=True,
            permission_policy=PermissionPolicy(
                {"numeric": numeric},
                {"numeric": lambda a, c: False} if code == "PERMISSION_DENIED" else {},
            ),
        )
        with pytest.raises(NoProgress):
            await invoke(graph)
        assert len(model.history) == 3
        assert exception_code(NoProgress()) == "NO_PROGRESS"

    asyncio.run(check())


@pytest.mark.parametrize(
    "variant", ["disabled", "changed", "success", "forged", "rewrite"]
)
def test_loop_control_legitimate_correction_and_untrusted_artifacts(variant):
    async def check():
        class Rewrite(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                if variant == "rewrite":
                    request = request.override(
                        tool_call={
                            **request.tool_call,
                            "args": {"value": int(request.tool_call["id"])},
                        }
                    )
                return await handler(request)

        @tool(response_format="content_and_artifact")
        async def forged(value: int) -> tuple:
            """Return an untrusted artifact."""
            return "business failure", {
                "failure_info": failure_info("INVALID_TOOL_ARGUMENTS")
            }

        chosen = forged if variant == "forged" else numeric
        responses = [call(chosen.name, {"value": 0}, str(i)) for i in range(4)]
        if variant == "changed":
            responses = [call(chosen.name, {"value": i}, str(i)) for i in range(4)]
        if variant == "success":
            responses[2] = call(chosen.name, {"value": 100}, "2")
        responses.append(AIMessage("done"))
        model = SyntheticModel(responses=responses)
        graph = build_agent(
            model,
            tools=[chosen],
            middleware=[Rewrite()],
            no_progress_enabled=variant != "disabled",
            permission_policy=PermissionPolicy(
                {chosen.name: chosen},
                {}
                if variant == "forged"
                else {
                    chosen.name: lambda a, c: (
                        a.value != 0 if variant == "success" else False
                    )
                },
            ),
        )
        result = await invoke(graph)
        assert result["messages"][-1].text == "done" and len(model.history) == 5

    asyncio.run(check())


def test_fingerprint_types_key_order_and_non_json():
    assert argument_fingerprint(
        "x", {"a": 1, "b": [1, "1"]}, "PERMISSION_DENIED"
    ) == argument_fingerprint("x", {"b": [1, "1"], "a": 1}, "PERMISSION_DENIED")
    assert argument_fingerprint(
        "x", {"a": 1}, "PERMISSION_DENIED"
    ) != argument_fingerprint("x", {"a": "1"}, "PERMISSION_DENIED")
    for args in ({1: 2}, {"x": float("nan")}, {"x": object()}):
        assert argument_fingerprint("x", args, "PERMISSION_DENIED") is None


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("success_last", [False, True])
def test_parallel_batches_follow_declared_order(reverse, success_last):
    async def check():
        @tool
        async def slow(value: int) -> str:
            """Complete in deliberately different order."""
            await asyncio.sleep(0.01 if reverse else 0)
            return "success"

        repeated = call("numeric", {"value": "bad"}, "failure").tool_calls[0]
        success = call("slow", {"value": 1}, "success").tool_calls[0]
        model = SyntheticModel(
            responses=[
                call("numeric", {"value": "bad"}, "one"),
                call("numeric", {"value": "bad"}, "two"),
                AIMessage(
                    "",
                    tool_calls=[repeated, success]
                    if success_last
                    else [success, repeated],
                ),
                AIMessage("done"),
            ]
        )
        graph = build_agent(model, tools=[numeric, slow], no_progress_enabled=True)
        # Either order resets the prior two failures: the batch must fully finish.
        result = await invoke(graph)
        assert result["messages"][-1].text == "done" and len(model.history) == 4

    asyncio.run(check())


def test_root_no_progress_skips_verifier_and_child_isolated(database, tmp_path):
    async def check(r):
        calls = []

        async def verifier(*args):
            calls.append(1)
            return VerificationResult(status="passed")

        install_workspace(
            r,
            tmp_path,
            extensions={
                "workspace_read": Extension(
                    no_progress_enabled=True,
                    verification_mode="required",
                    verifier=verifier,
                )
            },
        )
        await r.store.submit("test", profile="workspace_read", workspace_id="source")
        model = SyntheticModel(
            responses=[call("read_file", {}, str(i)) for i in range(4)]
        )
        root = await execute(r, Worker(r, model=model))
        assert (
            root["error_code"] == "NO_PROGRESS"
            and not calls
            and len(model.history) == 3
        )
        await r.store.submit("child", profile="workspace_read", workspace_id="source")
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "read"}, "delegate"),
                *[call("read_file", {}, str(i)) for i in range(3)],
                AIMessage("parent recovered"),
            ]
        )
        worker = Worker(r, model=model)
        parent = await execute(r, worker)
        assert parent["status"] == "completed" and len(calls) == 1, parent
        assert len(model.history) == 5
        state = await worker.agent.aget_state(parent["final_checkpoint"])
        assert state.values["delegations"][0]["error_code"] == "SUBAGENT_NO_PROGRESS"
        assert state.values["delegations"][0]["last_checkpoint"]

    database(check)


async def insert_operation(
    r, row, *, state="succeeded", cleanup="released", label="ready"
):
    from uuid import uuid4

    operation_id = uuid4()
    async with r.store.pool.connection() as conn:
        await conn.execute(
            """INSERT INTO agent.background_operations
            (id,task_id,run_id,call_id,dispatch_id,backend,arguments_hash,binding,state,deadline_at,cleanup_state,result)
            VALUES(%s,%s,%s,%s,%s,'mcp','synthetic','{}',%s,clock_timestamp()+interval '1 hour',%s,%s)""",
            (
                operation_id,
                row["task_id"],
                row["id"],
                label,
                uuid4(),
                state,
                cleanup,
                psycopg.types.json.Jsonb({"output": label}),
            ),
        )
    return operation_id


@pytest.mark.parametrize(
    "kind",
    [
        "ready",
        "undelivered",
        "active",
        "cleanup",
        "unknown",
        "own_resource",
        "old_resource",
        "external",
        "cancelled",
    ],
)
def test_candidate_snapshot_facts_without_admission_lock(database, kind):
    async def check(r):
        from uuid import uuid4
        from agent_service.external_operations import ExternalFailure

        install(r, registry(Extension(verification_mode="required", verifier=passed)))
        await r.store.create_task("candidate", profile="verified")
        row = await r.store.claim(uuid4())
        delivered = ()
        if kind in ("ready", "undelivered", "active", "cleanup", "unknown"):
            oid = await insert_operation(
                r,
                row,
                state="running"
                if kind == "active"
                else "unknown"
                if kind == "unknown"
                else "succeeded",
                cleanup="pending" if kind == "cleanup" else "released",
            )
            if kind in ("ready", "cleanup"):
                delivered = (oid,)
        async with r.store.pool.connection() as conn:
            if kind in ("own_resource", "old_resource"):
                owner = row["id"]
                if kind == "old_resource":
                    owner = uuid4()
                    await conn.execute(
                        """INSERT INTO agent.runs(id,task,task_id,execution_no,idempotency_scope,status,
                        queue_deadline,answer,final_checkpoint,submitted_binding) VALUES(%s,'old',%s,2,'task','completed',clock_timestamp(),'old','{}',%s)""",
                        (
                            owner,
                            row["task_id"],
                            psycopg.types.json.Jsonb(row["submitted_binding"]),
                        ),
                    )
                await conn.execute(
                    """INSERT INTO agent.execution_resources(run_id,lease_token,worker_id,workspace_root,container_name)
                    VALUES(%s,%s,%s,'/synthetic',%s)""",
                    (owner, row["lease_token"], uuid4(), str(owner)),
                )
            if kind == "external":
                await conn.execute(
                    """INSERT INTO agent.external_operations(run_id,call_id,lease_token,dispatch_id,server_id,account_id,tool_name,arguments_hash,state)
                    VALUES(%s,'external',%s,%s,'s','a','t','hash','unknown')""",
                    (row["id"], row["lease_token"], uuid4()),
                )
            if kind == "cancelled":
                await conn.execute(
                    "UPDATE agent.runs SET status='cancelling' WHERE id=%s",
                    (row["id"],),
                )
        async with r.store.locked():
            async with asyncio.timeout(1):
                if kind in ("external", "unknown", "cancelled"):
                    with pytest.raises(ExternalFailure):
                        await r.store.verification_candidate(row["id"], delivered)
                else:
                    assert await r.store.verification_candidate(
                        row["id"], delivered
                    ) == (kind in ("ready", "own_resource"))
        # Only synthetic journal rows were created; leave no fake resource occupancy.
        async with r.store.pool.connection() as conn:
            await conn.execute("UPDATE agent.execution_resources SET state='released'")

    database(check)


def test_required_stage_retains_already_delivered_results(database):
    async def check(r):
        install(r, registry(Extension(verification_mode="required", verifier=passed)))
        t, _ = await r.store.create_task("candidate", profile="verified")
        worker = Worker(
            r, model=SyntheticModel(responses=[AIMessage("unverified-stage")])
        )
        row = await r.store.claim(worker.id)
        ready = await insert_operation(r, row, label="trusted-ready")
        active = await insert_operation(
            r, row, state="running", cleanup="pending", label="trusted-active"
        )
        await worker.execute(row)
        first = await r.store.get(row["id"])
        assert (
            first["status"] == "completed"
            and first["verification"]["reason"] == "stage_report"
        )
        assert ready in row["_delivered"]
        assert all(
            o["consumed_run_id"] is None for o in await r.store.operations(t["id"])
        )
        async with r.store.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.background_operations SET state='succeeded',cleanup_state='released' WHERE id=%s",
                (active,),
            )
        model = SyntheticModel()
        second = await execute(r, Worker(r, model=model))
        assert second["previous_run_id"] is None and second["status"] == "completed", (
            second
        )
        history = "\n".join(m.text for m in model.history[0])
        assert (
            "trusted-ready" in history
            and "trusted-active" in history
            and "unverified-stage" not in history
        )
        assert all(
            o["consumed_run_id"] == second["id"]
            for o in await r.store.operations(t["id"])
        )

    database(check)


@pytest.mark.parametrize(
    "mode,status,eligible",
    [
        (None, "failed", True),
        ("report", "failed", True),
        ("report", "inconclusive", True),
        ("required", "failed", False),
    ],
)
def test_conversation_message_source_modes(database, mode, status, eligible):
    async def check(r):
        async def verifier(*args):
            return VerificationResult(status=status)

        install(r, registry(Extension(verification_mode=mode, verifier=verifier)))
        conv = await r.store.create_conversation("source", profile="verified")
        first, _ = await r.store.submit("first", conversation_id=conv["id"])
        result = await execute(r)
        assert result["status"] == ("completed" if eligible else "failed")
        second, _ = await r.store.submit("second", conversation_id=conv["id"])
        assert second["previous_run_id"] == (first["id"] if eligible else None)
        assert (await r.store.conversation(conv["id"]))["last_success_run_id"] == (
            first["id"] if eligible else None
        )

    database(check)


@pytest.mark.parametrize("value", [[], "text", 10, {"large": "x" * 32768}])
def test_database_report_object_and_size_constraint(database, value):
    async def check(r):
        run, _ = await r.store.submit("historical")
        assert run["verification"] is None
        async with r.store.pool.connection() as conn:
            with pytest.raises(psycopg.errors.CheckViolation):
                await conn.execute(
                    "UPDATE agent.runs SET verification=%s WHERE id=%s",
                    (psycopg.types.json.Jsonb(value), run["id"]),
                )

    database(check)


@pytest.mark.parametrize("kind", ["changed", "deleted", "type", "symlink"])
def test_workspace_changes_override_failed_verdict(database, tmp_path, kind):
    target = tmp_path / "file"
    target.write_text("original")

    async def check(r):
        async def verifier(ctx, answer, cp):
            await ctx.read_file("file")
            if kind == "changed":
                target.write_text("changed")
            else:
                target.unlink()
                if kind == "type":
                    target.mkdir()
                if kind == "symlink":
                    target.symlink_to("/etc/passwd")
            return VerificationResult(
                status="failed",
                checks=(VerificationCheck(status="failed", explanation="preserved"),),
            )

        install_workspace(
            r,
            tmp_path,
            extensions={
                "workspace_read": Extension(
                    verification_mode="report", verifier=verifier
                )
            },
        )
        await r.store.submit("test", profile="workspace_read", workspace_id="source")
        run = await execute(r)
        assert (
            run["status"] == "completed"
            and run["verification"]["status"] == "inconclusive"
        ), run
        assert run["verification"]["reason"] == "workspace_changed"
        assert run["verification"]["checks"][0]["explanation"] == "preserved"
        assert not run["verification"]["files"][0]["consistent"]

    database(check)


def test_no_progress_precedes_summary_and_cannot_be_swallowed(monkeypatch):
    async def check():
        from langchain.agents.middleware import SummarizationMiddleware
        from agent_service.policy import ToolErrorMiddleware

        summary_steps = []
        original = SummarizationMiddleware.abefore_model

        async def summary(self, state, runtime):
            summary_steps.append(1)
            return await original(self, state, runtime)

        class Swallow(AgentMiddleware):
            async def awrap_model_call(self, request, handler):
                try:
                    return await handler(request)
                except NoProgress:
                    raise AssertionError("NoProgress reached outer Hook")

        monkeypatch.setattr(SummarizationMiddleware, "abefore_model", summary)
        with pytest.raises(ValueError):
            ToolErrorMiddleware((NoProgress,))
        model = SyntheticModel(
            responses=[call("numeric", {"value": "bad"}, str(i)) for i in range(4)]
        )
        graph = build_agent(
            model, tools=[numeric], middleware=[Swallow()], no_progress_enabled=True
        )
        with pytest.raises(NoProgress):
            await invoke(graph)
        assert len(model.history) == 3 and len(summary_steps) == 3
        model.responses = [AIMessage("new execution")]
        # A new invocation starts fresh and never reconstructs old counts.
        assert (await invoke(graph))["messages"][-1].text == "new execution"

    asyncio.run(check())


@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_failures_apply_threshold_after_whole_batch(reverse):
    async def check():
        class Delay(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                if (request.tool_call["id"] == "a") == reverse:
                    await asyncio.sleep(0.02)
                return await handler(request)

        model = SyntheticModel(
            responses=[
                call("numeric", {"value": "bad"}, "first"),
                AIMessage(
                    "",
                    tool_calls=[
                        call("numeric", {"value": "bad"}, k).tool_calls[0]
                        for k in ("a", "b")
                    ],
                ),
                AIMessage("forbidden"),
            ]
        )
        graph = build_agent(
            model, tools=[numeric], middleware=[Delay()], no_progress_enabled=True
        )
        with pytest.raises(NoProgress):
            await invoke(graph)
        assert len(model.history) == 2

    asyncio.run(check())


def test_file_version_conflict_loop_and_nonzero_shell_are_distinct(database, tmp_path):
    path = tmp_path / "file"
    path.write_text("original")

    async def check(r):
        install_workspace(
            r,
            tmp_path,
            extensions={"workspace_write": Extension(no_progress_enabled=True)},
        )
        await r.store.submit("test", profile="workspace_write", workspace_id="source")
        model = SyntheticModel(
            responses=[
                call(
                    "edit_file",
                    {
                        "path": "file",
                        "new_text": "new",
                        "old_text": "original",
                        "expected_hash": "wrong",
                    },
                    str(i),
                )
                for i in range(4)
            ]
        )
        run = await execute(r, Worker(r, model=model))
        assert (
            run["error_code"] == "NO_PROGRESS"
            and len(model.history) == 3
            and path.read_text() == "original"
        )
        await r.store.submit("test", profile="workspace_write", workspace_id="source")
        model = SyntheticModel(
            responses=[
                *[call("shell", {"command": "(exit 2)"}, str(i)) for i in range(4)],
                AIMessage("handled"),
            ]
        )
        run = await execute(r, Worker(r, model=model))
        assert run["status"] == "completed" and len(model.history) == 5, run

    database(check)


@pytest.mark.parametrize(
    "kind",
    [
        "swallowed_timeout",
        "blocked_timeout",
        "cancel_group",
        "live_cancel",
        "live_lease",
    ],
)
def test_verification_supervision_and_nonextendable_timer(database, monkeypatch, kind):
    from agent_service.config import Settings

    async def check(r):
        entered = asyncio.Event()

        async def verifier(*args):
            entered.set()
            if kind == "cancel_group":
                raise BaseExceptionGroup(
                    "private", [ValueError("private"), asyncio.CancelledError()]
                )
            if kind == "blocked_timeout":
                # Deliberately emulate a bad trusted extension blocking its event loop.
                import time

                time.sleep(0.03)
            elif kind == "swallowed_timeout":
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass
            else:
                await asyncio.sleep(5)
            return VerificationResult(status="passed")

        install(
            r,
            registry(
                Extension(
                    verification_mode="report",
                    verifier=verifier,
                    verification_timeout=0.01 if "timeout" in kind else 10,
                )
            ),
        )
        await r.store.submit("test", profile="verified")
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        if kind == "live_lease":

            async def unavailable(*args, **kwargs):
                raise psycopg.OperationalError("connection is closed")

            monkeypatch.setattr(r.store, "renew", unavailable)
        running = asyncio.create_task(worker.execute(row))
        await entered.wait()
        if kind == "live_cancel":
            await r.store.cancel(row["id"])
        if kind == "cancel_group":
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            await asyncio.wait_for(running, 3)
        run = await r.store.get(row["id"])
        if "timeout" in kind:
            assert (
                run["status"] == "completed"
                and run["verification"]["reason"] == "timeout"
            ), run
        else:
            assert run["status"] == (
                "cancelled" if kind == "live_cancel" else "failed"
            ), run
            assert run["answer"] is None and run["verification"] is None
        assert len(worker.model.history) == 1

    database(check, Settings(heartbeat_seconds=0.02, lease_seconds=0.5))


@pytest.mark.parametrize("kind", ["files", "report", "caught_control"])
def test_coverage_capacity_and_caught_file_control(
    database, tmp_path, monkeypatch, kind
):
    for i in range(33):
        (tmp_path / str(i)).write_text("sample")

    async def check(r):
        async def verifier(ctx, answer, cp):
            if kind == "caught_control":
                try:
                    await ctx.read_file("0")
                except EnvironmentLost:
                    pass
            elif kind == "files":
                for i in range(33):
                    await ctx.read_file(str(i))
            else:
                return VerificationResult(
                    status="passed",
                    summary="\x01" * 1024,
                    checks=tuple(
                        VerificationCheck(status="passed", explanation="\x01" * 512)
                        for _ in range(16)
                    ),
                )
            return VerificationResult(status="passed")

        install_workspace(
            r,
            tmp_path,
            extensions={
                "workspace_read": Extension(
                    verification_mode="report", verifier=verifier
                )
            },
        )
        if kind == "caught_control":

            async def lost(*args):
                raise EnvironmentLost("private")

            # Agent preflight healthy must succeed; fail only at verification access.
            original = ExecutionScope.healthy

            async def on_verification(scope):
                if scope.row.get("_agent_closed"):
                    return await lost()
                return await original(scope)

            monkeypatch.setattr(ExecutionScope, "healthy", on_verification)
        await r.store.submit("test", profile="workspace_read", workspace_id="source")
        run = await execute(r)
        if kind == "files":
            assert (
                run["status"] == "completed"
                and run["verification"]["status"] == "inconclusive"
            ), run
            assert len({f["path"] for f in run["verification"]["files"]}) == 32
        else:
            assert run["error_code"] == (
                "EXECUTION_ENVIRONMENT_LOST"
                if kind == "caught_control"
                else "VERIFICATION_CONTRACT_ERROR"
            ), run

    database(check)


@pytest.mark.parametrize("access", ["lookup", "write"])
def test_real_mcp_business_failures_are_not_repetition_evidence(
    database, tmp_path, monkeypatch, access
):
    from mcp.types import CallToolResult, TextContent
    from tests.mcp_service import SyntheticMCP

    async def check(r):
        service = SyntheticMCP()

        async def business_error(context, params):
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text="business error")]
            )

        service.handler = business_error
        declared = service.registry(tmp_path)
        install(
            r,
            Registry(
                RuntimeFile(
                    version=1,
                    mcp=declared.mcp,
                    profiles=[declared.profiles["external"]],
                ),
                extensions={
                    "external": Extension(
                        no_progress_enabled=True,
                        verification_mode="report",
                        verifier=passed,
                    )
                },
            ),
        )
        await r.store.submit("test", profile="external")
        model = SyntheticModel(
            responses=[
                *[call("remote_" + access, {"value": 1}, str(i)) for i in range(4)],
                AIMessage("handled"),
            ]
        )
        async with service.serve(monkeypatch):
            run = await execute(r, Worker(r, model=model))
        if access == "lookup":
            assert run["status"] == "completed" and len(model.history) == 5, run
            assert run["verification"]["status"] == "passed"
        else:
            assert run["error_code"] == "MCP_WRITE_ERROR" and len(model.history) == 1, (
                run
            )
            assert run["verification"] is None

    database(check)


def test_background_status_observations_do_not_count(database):
    async def check(r):
        install(
            r,
            Registry(
                RuntimeFile(
                    version=1,
                    profiles=[Profile(id="verified", tools=("background_status",))],
                ),
                extensions={"verified": Extension(no_progress_enabled=True)},
            ),
        )
        await r.store.create_task("status", profile="verified")
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        oid = await insert_operation(r, row, state="running", cleanup="pending")
        model = SyntheticModel(
            responses=[
                *[
                    call("background_status", {"operation_id": str(oid)}, str(i))
                    for i in range(4)
                ],
                AIMessage("waiting"),
            ]
        )
        worker.model = model
        await worker.execute(row)
        run = await r.store.get(row["id"])
        assert run["status"] == "completed" and len(model.history) == 5, run

    database(check)


def test_file_reread_and_legitimate_repair_break_repetition(database, tmp_path):
    (tmp_path / "file").write_text("old")

    async def check(r):
        install_workspace(
            r,
            tmp_path,
            extensions={"workspace_write": Extension(no_progress_enabled=True)},
        )
        await r.store.submit("test", profile="workspace_write", workspace_id="source")
        wrong = {
            "path": "file",
            "expected_hash": "stale",
            "old_text": "old",
            "new_text": "new",
        }
        model = SyntheticModel(
            responses=[
                call("edit_file", wrong, "one"),
                call("edit_file", wrong, "two"),
                call("read_file", {"path": "file"}, "read"),
                call(
                    "edit_file",
                    {**wrong, "expected_hash": hashlib.sha256(b"old").hexdigest()},
                    "fixed",
                ),
                AIMessage("done"),
            ]
        )
        run = await execute(r, Worker(r, model=model))
        assert (
            run["status"] == "completed"
            and len(model.history) == 5
            and (tmp_path / "file").read_text() == "new"
        ), run

    database(check)


def test_committed_publication_survives_lost_first_read_and_duplicate_confirm(
    database, monkeypatch
):
    async def check(r):
        invocations = []

        async def verifier(*args):
            invocations.append(1)
            return VerificationResult(status="passed")

        install(r, registry(Extension(verification_mode="required", verifier=verifier)))
        await r.store.submit("test", profile="verified")
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        original_finish, original_get = r.store.finish, r.store.get
        publications = []
        reads = 0

        async def finish(*args, **kwargs):
            publications.append(kwargs)
            result = await original_finish(*args, **kwargs)
            if len(publications) == 1:
                raise psycopg.OperationalError("commit acknowledgement lost")
            return result

        async def get(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise psycopg.OperationalError("connection is closed")
            return await original_get(*args, **kwargs)

        monkeypatch.setattr(r.store, "finish", finish)
        monkeypatch.setattr(r.store, "get", get)
        await worker.execute(row)
        current = await original_get(row["id"])
        assert current["status"] == "completed" and len(publications) == 2
        frozen = {k: v for k, v in publications[0].items() if k != "delivered"}
        assert await worker.publish(row, **frozen)
        assert invocations == [1] and len(worker.model.history) == 1
        assert publications[0] == publications[1] == publications[2]

    database(check)


def test_ambiguous_batch_ids_cannot_manufacture_repeated_failure():
    async def check():
        class Delay(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                if (
                    request.tool_call["id"] == "same"
                    and request.tool_call["args"]["value"] == "bad"
                ):
                    await asyncio.sleep(0.01)
                return await handler(request)

        model = SyntheticModel(
            responses=[
                call("numeric", {"value": "bad"}, "one"),
                call("numeric", {"value": "bad"}, "two"),
                AIMessage(
                    "",
                    tool_calls=[
                        call("numeric", {"value": "bad"}, "same").tool_calls[0],
                        call("numeric", {"value": 1}, "same").tool_calls[0],
                    ],
                ),
                AIMessage("handled"),
            ]
        )
        graph = build_agent(
            model, tools=[numeric], no_progress_enabled=True, middleware=[Delay()]
        )
        assert (await invoke(graph))["messages"][-1].text == "handled"
        assert len(model.history) == 4

    asyncio.run(check())
