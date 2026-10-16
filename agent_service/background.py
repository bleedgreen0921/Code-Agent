"""Task-owned background tools and restart-safe resource observation."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_service.failure import failure_info, is_control
from agent_service.background_store import public_operation
from agent_service.execution import DockerClient, LABEL, FIXED_ENV, check_environment
from agent_service.external_operations import ExternalFailure
from agent_service.mcp_config import digest, contract
from agent_service.output import limit_output
from agent_service.policy import active_call
from agent_service.tasks import OP_TERMINAL

log = logging.getLogger(__name__)
BACKGROUND_TOOLS = {
    "start_background_shell",
    "background_status",
    "background_result",
    "cancel_background",
}


class BackgroundReceipt(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    exit_code: int
    timed_out: bool
    cancelled: bool
    truncated: bool
    total_bytes: int = Field(ge=0)
    output: str

    @model_validator(mode="after")
    def consistent(self):
        if self.timed_out and self.cancelled:
            raise ValueError("Contradictory receipt flags")
        return self


def docker_identity(inspected, op):
    """Validate only fields needed for observing and destroying the original resource."""
    try:
        identity = inspected["Id"]
        labels = inspected["Config"]["Labels"]
        state = inspected["State"]
        valid = (
            isinstance(identity, str)
            and len(identity) == 64
            and (not op["container_id"] or identity == op["container_id"])
            and labels.get(LABEL) == op["container_name"]
            and labels.get("io.agent-framework.operation") == str(op["id"])
            and type(state["Running"]) is bool
            and isinstance(state["Status"], str)
        )
    except (KeyError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ExternalFailure("BACKGROUND_RESOURCE_IDENTITY_CHANGED")
    return identity


class ShellInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=32768)
    cwd: str = Field(default=".", min_length=1, max_length=1024)


class OperationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: UUID


def reject_background(code, message=None):
    call = active_call.get()
    if call:
        call.diagnostic = failure_info(code, effect="not_dispatched")
    raise ToolException(message or code)


def authorized(row, *, start=None):
    call = active_call.get()
    if (
        not row
        or not row.get("task_id")
        or call is None
        or getattr(call.context, "agent_depth", 0)
        or getattr(call.context, "task_id", None) != str(row["task_id"])
    ):
        reject_background("BACKGROUND_REQUIRES_TASK_ROOT")
    if start and (
        len(call.batch) != 1
        or call.batch[0]["name"] != start
        or call.batch[0]["id"] != call.call_id
    ):
        reject_background("BACKGROUND_START_BATCH_REJECTED")
    return call


class BackgroundBatchMiddleware(AgentMiddleware):
    def __init__(self, policy):
        self.starts = {"start_background_shell"} | {
            n
            for n, t in policy.tools.items()
            if (t.metadata or {}).get("background_start")
        }

    async def awrap_tool_call(self, request, handler):
        call = active_call.get()
        if request.tool_call["name"] in self.starts and len(call.batch) != 1:
            return ToolMessage(
                content='{"code":"BACKGROUND_START_BATCH_REJECTED"}',
                tool_call_id=call.call_id,
                status="error",
                artifact={
                    "failure_info": failure_info("BACKGROUND_START_BATCH_REJECTED")
                },
            )
        return await handler(request)


def background_tools(db=None, row=None, registry=None):
    async def start_background_shell(command: str, cwd: str = "."):
        call = authorized(row, start="start_background_shell")
        if Path(cwd).is_absolute() or ".." in Path(cwd).parts or "\x00" in cwd:
            reject_background(
                "INVALID_TOOL_ARGUMENTS", "Invalid workspace-relative directory"
            )
        root = registry.workspaces[row["workspace_id"]]
        try:
            target = (Path(root) / cwd).resolve(strict=True)
        except (OSError, ValueError):
            reject_background("INVALID_TOOL_ARGUMENTS", "Invalid workspace directory")
        if not target.is_relative_to(root) or not target.is_dir():
            reject_background("INVALID_TOOL_ARGUMENTS", "Invalid workspace directory")
        # Preflight does not dispatch anything. Reserved operations are never replayed.
        await check_environment(registry.execution)
        _, image_raw, _ = await DockerClient().call(
            "image", "inspect", registry.execution.image
        )
        if (
            json.loads(image_raw)[0]
            .get("Config", {})
            .get("Labels", {})
            .get("io.agent-framework.background")
            != "background-v1"
        ):
            raise ExternalFailure("BACKGROUND_IMAGE_UNSUPPORTED")
        op = await db.reserve_operation(
            row,
            call.call_id,
            "docker",
            digest({"command": command, "cwd": cwd}),
            {
                "execution": registry.execution.model_dump(),
                "workspace_identity": row["submitted_binding"]["workspace_identity"],
            },
            workspace_root=root,
        )
        try:
            await start_docker(db, registry, op, command, cwd)
        finally:
            # Reconciliation only observes this dispatch; it never repeats create/start.
            await db.release_monitor(op, 0)
        return json.dumps(public_operation(op, result=False), default=str)

    async def owned_operation(operation_id, *, cancel=False):
        from contracts.errors import ApiError

        try:
            if cancel:
                return await db.cancel_operation(row["task_id"], operation_id)
            return await db.operation(row["task_id"], operation_id, raw=True)
        except ApiError:
            reject_background("BACKGROUND_OPERATION_NOT_FOUND")

    async def background_status(operation_id: UUID):
        authorized(row)
        return json.dumps(
            public_operation(await owned_operation(operation_id), result=False),
            default=str,
        )

    async def background_result(operation_id: UUID):
        authorized(row)
        op = await owned_operation(operation_id)
        if (
            op["state"] not in ("succeeded", "failed", "cancelled")
            or op["cleanup_state"] != "released"
        ):
            reject_background("BACKGROUND_RESULT_NOT_READY")
        # Mark only if the full bounded JSON fits the final tool output boundary.
        payload = {key: op[key] for key in ("id", "state", "error_code")}
        result = dict(op["result"] or {})
        output = result.pop("output", result.pop("content", ""))
        text = (
            json.dumps({**payload, "result": result}, default=str, ensure_ascii=False)
            + "\nUntrusted result output:\n"
            + output
        )
        text, truncated = limit_output(
            text, registry.execution.output_bytes, registry.execution.output_lines
        )
        if not truncated:
            row.setdefault("_delivery_candidates", {})[active_call.get().call_id] = (
                op["id"],
                text,
            )
        return text

    async def cancel_background(operation_id: UUID):
        authorized(row)
        return json.dumps(await owned_operation(operation_id, cancel=True), default=str)

    declarations = [
        (
            "start_background_shell",
            "Immediately start an authorized Docker command for this Task. Must be the only call in its batch. During execution workspace reads may change; writes and ordinary Shell are forbidden.",
            ShellInput,
            start_background_shell,
        ),
        (
            "background_status",
            "Read background operation metadata without waiting.",
            OperationInput,
            background_status,
        ),
        (
            "background_result",
            "Read a persisted bounded result after execution and resource cleanup finish.",
            OperationInput,
            background_result,
        ),
        (
            "cancel_background",
            "Request cancellation of one operation owned by this Task. Cancellation does not undo side effects.",
            OperationInput,
            cancel_background,
        ),
    ]
    return [
        StructuredTool.from_function(
            name=n,
            description=d,
            args_schema=s,
            coroutine=f,
            handle_tool_error=True,
            metadata={"background_start": n == "start_background_shell"},
        )
        for n, d, s, f in declarations
    ]


async def start_docker(db, registry, op, command, cwd):
    cfg = registry.execution
    client = DockerClient()
    daemon = await client.daemon()
    root = op["workspace_root"]
    if str(Path(root).resolve(strict=True)) != root or "," in root:
        raise ExternalFailure("BACKGROUND_WORKSPACE_CHANGED")
    await db.operation_update(op, daemon_id=daemon, phase="creating")
    args = [
        "create",
        "--pull=never",
        "--name",
        op["container_name"],
        "--label",
        f"{LABEL}={op['container_name']}",
        "--label",
        f"io.agent-framework.operation={op['id']}",
        "--restart=no",
        "--network",
        "none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        cfg.effective_user(),
        "--cpus",
        str(cfg.cpus),
        "--memory",
        str(cfg.memory_bytes),
        "--memory-swap",
        str(cfg.memory_bytes),
        "--pids-limit",
        str(cfg.pids),
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,size={cfg.tmp_bytes},mode=1777",
        "--mount",
        f"type=bind,source={root},target=/workspace",
        "--workdir",
        "/workspace",
        "--log-driver",
        "local",
        "--log-opt",
        "max-size=64k",
        "--log-opt",
        "max-file=1",
        "--log-opt",
        "compress=false",
    ]
    for k, v in FIXED_ENV.items():
        args += ["--env", f"{k}={v}"]
    args += [
        "--entrypoint",
        "/usr/local/bin/python",
        cfg.image,
        "-I",
        "/opt/agent/background.py",
        json.dumps(
            {"command": command, "cwd": cwd, "deadline": op["deadline_at"].timestamp()}
        ),
    ]
    _, out, _ = await client.call(*args, timeout=cfg.cleanup_timeout)
    identity = out.decode().strip()
    if len(identity) != 64:
        raise ExternalFailure("BACKGROUND_CREATE_UNCONFIRMED")
    await db.operation_update(op, container_id=identity, phase="created")
    await db.operation_update(op, phase="starting")
    await client.call("start", identity, timeout=cfg.cleanup_timeout)
    # API cancellation may race: preserve it rather than overwrite it with running.
    latest = await db.operation(op["task_id"], op["id"], raw=True)
    await db.operation_update(
        op,
        state="cancelling" if latest["cancel_requested_at"] else "running",
        phase="started",
    )


async def monitor_docker(db, registry, op):
    cfg = registry.execution
    if (
        op["binding"]["execution"] != cfg.model_dump()
        or registry.workspaces.get((await db.task(op["task_id"]))["workspace_id"])
        != op["workspace_root"]
    ):
        raise ExternalFailure("BACKGROUND_CONFIGURATION_CHANGED")
    client = DockerClient()
    daemon = await client.daemon()
    if op["daemon_id"] and daemon != op["daemon_id"]:
        raise ExternalFailure("BACKGROUND_DAEMON_CHANGED")
    inspected = await client.inspect(op["container_id"] or op["container_name"])
    if inspected is None:
        if op["phase"] == "reserved":
            save = (
                db.operation_reconcile
                if op["state"] == "unknown"
                else db.operation_update
            )
            await save(
                op,
                state="failed",
                result={"error": "Dispatch did not begin"},
                error_code="BACKGROUND_NOT_STARTED",
                cleanup_state="released",
            )
        elif op["state"] in OP_TERMINAL and op["container_id"]:
            await db.operation_cleanup(op, cleanup_state="released", cleanup_error=None)
        else:
            await db.operation_update(
                op, state="unknown", error_code="BACKGROUND_RESULT_UNKNOWN"
            )
        return
    docker_identity(inspected, op)
    if not op["container_id"]:
        await db.operation_update(op, container_id=inspected["Id"])
    expired = datetime.now(timezone.utc) >= op["deadline_at"]
    cancel = (
        op["cancel_requested_at"] is not None or expired or op["state"] == "unknown"
    )
    stopped_for_timeout = inspected["State"]["Running"] and expired
    if inspected["State"]["Running"] and not cancel:
        await db.operation_update(op, state="running", phase="started")
        return
    confirmed_stop = False
    if inspected["State"]["Running"]:
        # Stop the entire container, including detached process groups.
        await client.call(
            "stop", "--time", "1", inspected["Id"], timeout=cfg.cleanup_timeout
        )
        inspected = await client.inspect(inspected["Id"])
        if not inspected or inspected["State"]["Running"]:
            raise ExternalFailure("BACKGROUND_STOP_UNCONFIRMED")
        docker_identity(inspected, op)
        confirmed_stop = True
    if op["state"] not in OP_TERMINAL or op["state"] == "unknown":
        if inspected["State"].get("Status") == "created":
            state, result, code = (
                "failed",
                {"error": "Container was created but never started"},
                "BACKGROUND_NOT_STARTED",
            )
        else:
            _, raw, _ = await client.call("logs", inspected["Id"], maximum=65536)
            try:
                result = BackgroundReceipt.model_validate_json(raw).model_dump()
                # Leave room for delivery metadata, so background_result can return
                # the complete persisted result within the ordinary tool boundary.
                result["output"], clipped = limit_output(
                    result["output"],
                    min(15360, cfg.output_bytes - 384),
                    max(1, cfg.output_lines - 3),
                )
                result["truncated"] = result["truncated"] or clipped
                state = (
                    "cancelled"
                    if result.get("cancelled") and not stopped_for_timeout
                    else "failed"
                    if stopped_for_timeout
                    or result["timed_out"]
                    or result["exit_code"] != 0
                    else "succeeded"
                )
                code = (
                    "BACKGROUND_TIMEOUT"
                    if stopped_for_timeout or result["timed_out"]
                    else "BACKGROUND_COMMAND_FAILED"
                    if result["exit_code"] != 0 and state == "failed"
                    else None
                )
            except (ValueError, TypeError):
                # Known cancellation can destroy the receipt, but cannot imply command success.
                state, result, code = (
                    (
                        "cancelled",
                        {"cancelled": True, "effect_unknown": True},
                        "BACKGROUND_RECEIPT_INVALID",
                    )
                    if confirmed_stop
                    and op["cancel_requested_at"] is not None
                    and not expired
                    else ("unknown", None, "BACKGROUND_RECEIPT_INVALID")
                )
        if op["state"] != "unknown" or state != "unknown":
            save = (
                db.operation_reconcile
                if op["state"] == "unknown"
                else db.operation_update
            )
            await save(
                op,
                state=state,
                result=result,
                error_code=code,
                failure_info=failure_info(code) if code else None,
            )
    # Result commit is confirmed before resource removal; remove acknowledgements can be reconciled.
    await client.call("rm", "--force", inspected["Id"], timeout=cfg.cleanup_timeout)
    if await client.inspect(inspected["Id"]) is not None:
        raise ExternalFailure("BACKGROUND_CLEANUP_UNCONFIRMED")
    await db.operation_cleanup(op, cleanup_state="released", cleanup_error=None)


@asynccontextmanager
async def task_connection(client):
    """Keep connection shutdown failures inside the credential-safe error boundary."""
    try:
        async with client:
            yield client
    except asyncio.CancelledError:
        raise
    except (ToolException, ExternalFailure):
        raise
    except Exception as exc:
        if is_control(exc):
            raise
        raise ExternalFailure("MCP_CONNECTION_FAILED") from None


def mcp_background_tool(db, row, registry, alias, entry):
    native = entry.declaration()

    async def invoke(**arguments):
        call = authorized(row, start=alias)
        client = await task_client(registry, entry)
        async with task_connection(client):
            await verify_task_client(client, entry)
            op = await db.reserve_operation(
                row,
                call.call_id,
                "mcp",
                digest(arguments),
                remote_identity(client, entry),
            )
            try:
                from fastmcp_tasks import call_tool_task
                from mcp.types import CallToolResult

                client.last_explicit_result = None
                await db.operation_update(op, phase="submitting")
                try:
                    handle = await call_tool_task(
                        client,
                        entry.selection.name,
                        arguments,
                        timeout=min(
                            registry.mcp.call_timeout,
                            max(0.1, db.settings.reconciliation_seconds - 5),
                        ),
                    )
                except Exception:
                    raw = client.last_explicit_result
                    if isinstance(raw, CallToolResult):
                        result = bounded_remote_result(
                            client, raw.model_dump(mode="json", by_alias=True)
                        )
                        await db.operation_update(
                            op,
                            state="failed",
                            error_code="MCP_TASK_PROTOCOL_ERROR",
                            result=result,
                            cleanup_state="released",
                        )
                    else:
                        await db.operation_update(
                            op, state="unknown", error_code="MCP_RESULT_UNKNOWN"
                        )
                        raise ExternalFailure("MCP_RESULT_UNKNOWN") from None
                else:
                    receipt = client.redact(
                        handle.create_result.model_dump(mode="json", by_alias=True)
                    )
                    from agent_service.output import bounded_artifact
                    from fastmcp_tasks.client_models import ClientCreateTaskResult

                    try:
                        receipt_id = ClientCreateTaskResult.model_validate(
                            receipt
                        ).task_id
                    except ValidationError:
                        receipt_id = None
                    if (
                        receipt_id != handle.task_id
                        or not handle.task_id
                        or bounded_artifact(receipt, 8000) != receipt
                        or len(handle.task_id) > 1024
                    ):
                        await db.operation_update(
                            op, state="unknown", error_code="MCP_TASK_RECEIPT_INVALID"
                        )
                        raise ExternalFailure("MCP_TASK_RECEIPT_INVALID")
                    await db.operation_update(
                        op,
                        remote_handle=handle.task_id,
                        receipt=receipt,
                        state="running",
                        phase="submitted",
                    )
            finally:
                await db.release_monitor(op, 0)
        return json.dumps(public_operation(op, result=False), default=str)

    return StructuredTool(
        name=alias,
        description=native.description,
        args_schema=native.args_schema,
        coroutine=invoke,
        handle_tool_error=True,
        metadata={
            "background_start": True,
            "external_access": entry.selection.access,
            "execution": "background",
        },
    )


async def task_client(registry, entry):
    from agent_service.mcp_runtime import make_client

    return make_client(
        entry.server,
        registry.mcp,
        (registry.execution.output_bytes, registry.execution.output_lines),
        background=True,
    )


async def verify_task_client(client, entry):
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    caps = client.server_capabilities
    if (
        client.protocol_version not in MODERN_PROTOCOL_VERSIONS
        or not caps
        or "io.modelcontextprotocol/tasks" not in (caps.extensions or {})
    ):
        raise ExternalFailure("MCP_TASKS_UNSUPPORTED")
    tools = await client.list_tools(cache_mode="bypass")
    if {t.name: contract(t) for t in tools}.get(
        entry.selection.name
    ) != entry.definition:
        raise ExternalFailure("MCP_TASK_CONTRACT_CHANGED")


def remote_identity(client, entry):
    # Include the actual credential identity as a one-way digest. Never persist secrets.
    return {
        "manifest": entry.manifest(),
        "alias": entry.selection.alias,
        "credential_hash": digest(client.redactions),
        "protocol": client.protocol_version,
    }


def bounded_remote_result(client, value):
    value = client.redact(value)
    # Keep result delivery within a single tool response, including metadata.
    text, truncated = limit_output(
        json.dumps(value, ensure_ascii=False),
        min(12000, client.output_limits[0] - 384),
        min(180, max(1, client.output_limits[1] - 3)),
    )
    return {"content": text, "truncated": truncated}


async def monitor_mcp(db, registry, op):
    from fastmcp_tasks import ToolTask
    from fastmcp_tasks.client_models import ClientCreateTaskResult

    entry = registry.mcp_tools.get(op["binding"].get("alias"))
    if (
        not entry
        or entry.selection.execution != "background"
        or entry.manifest() != op["binding"].get("manifest")
    ):
        raise ExternalFailure("BACKGROUND_CONFIGURATION_CHANGED")
    client = await task_client(registry, entry)
    async with client:
        await verify_task_client(client, entry)
        if remote_identity(client, entry) != op["binding"]:
            raise ExternalFailure("MCP_TASK_IDENTITY_CHANGED")
        if not op["remote_handle"] or not op["receipt"]:
            await db.operation_update(
                op, state="unknown", error_code="MCP_RESULT_UNKNOWN"
            )
            return None
        handle = ToolTask(
            client,
            entry.selection.name,
            ClientCreateTaskResult.model_validate(op["receipt"]),
        )
        if handle.task_id != op["remote_handle"]:
            raise ExternalFailure("MCP_TASK_IDENTITY_CHANGED")
        try:
            current = await handle.status()
        except ValidationError:
            raise ExternalFailure("MCP_TASK_PROTOCOL_ERROR") from None
        if current.task_id != op["remote_handle"]:
            raise ExternalFailure("MCP_TASK_IDENTITY_CHANGED")
        status = current.status
        delay = max(
            db.settings.background_poll_seconds, (current.poll_interval_ms or 0) / 1000
        )
        expired = datetime.now(timezone.utc) >= op["deadline_at"]
        if status in ("completed", "failed", "cancelled"):
            if status == "completed":
                from mcp.types import CallToolResult
                from agent_service.schema import local_schema

                from jsonschema import ValidationError as OutputSchemaError

                try:
                    parsed = CallToolResult.model_validate(current.result)
                    if (
                        entry.definition.get("outputSchema") is not None
                        and not parsed.is_error
                    ):
                        local_schema(entry.definition["outputSchema"]).validate(
                            parsed.structured_content
                        )
                except (ValidationError, OutputSchemaError):
                    raise ExternalFailure("MCP_TASK_PROTOCOL_ERROR") from None
            elif status == "failed" and not isinstance(current.error, dict):
                raise ExternalFailure("MCP_TASK_PROTOCOL_ERROR")
            result = bounded_remote_result(
                client,
                current.result
                if status == "completed"
                else current.error
                if status == "failed"
                else {"cancelled": True},
            )
            state = {
                "completed": "succeeded",
                "failed": "failed",
                "cancelled": "cancelled",
            }[status]
            if status == "completed" and parsed.is_error:
                state = "failed"
            code = "MCP_TASK_EXECUTION_FAILED" if state == "failed" else None
            if op["state"] != "unknown" and op["error_code"] and status != "completed":
                code = op["error_code"]
                state = "failed"
            # Completion is authoritative even if cancellation was requested.
            save = (
                db.operation_reconcile
                if op["state"] == "unknown"
                else db.operation_update
            )
            await save(
                op,
                state=state,
                result=result,
                error_code=code,
                failure_info=failure_info(code) if code else None,
                cleanup_state="released",
                cleanup_error=None,
            )
            return delay
        cancel = (
            op["cancel_requested_at"]
            or expired
            or status == "input_required"
            or op["state"] == "unknown"
        )
        if cancel:
            now = datetime.now(timezone.utc)
            if not op["cancel_sent_at"]:
                # Persist intent before the single cooperative cancellation request.
                changes = {
                    "cancel_requested_at": op["cancel_requested_at"] or now,
                    "cancel_sent_at": now,
                }
                if op["state"] != "unknown":
                    changes.update(
                        state="cancelling",
                        error_code="MCP_INTERACTION_UNSUPPORTED"
                        if status == "input_required"
                        else "BACKGROUND_TIMEOUT"
                        if expired
                        else op["error_code"],
                    )
                # Historic unknown facts may have no diagnosis. Cancellation intent
                # must neither backfill nor replace their protected execution fields.
                await db.operation_update(op, **changes)
                await handle.cancel()
            elif (
                now - op["cancel_sent_at"]
            ).total_seconds() >= db.settings.reconciliation_seconds:
                await db.operation_update(
                    op, state="unknown", error_code="MCP_CANCEL_UNCONFIRMED"
                )
        return delay


async def monitor_operation(db, registry, op):
    delay = db.settings.background_poll_seconds
    try:
        op.update(await db.monitor_record(op))
        # Leave time in the lease for persistence and disconnect cleanup.
        async with asyncio.timeout(max(0.1, db.settings.reconciliation_seconds - 2)):
            if op["backend"] == "docker":
                await monitor_docker(db, registry, op)
            else:
                delay = await monitor_mcp(db, registry, op) or delay
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        code = (
            exc.code
            if isinstance(exc, ExternalFailure)
            else "BACKGROUND_RECONCILIATION_FAILED"
        )
        try:
            if op["state"] in OP_TERMINAL:
                await db.operation_cleanup(
                    op, cleanup_state="failed", cleanup_error="RESOURCE_CLEANUP_FAILED"
                )
            else:
                await db.operation_update(
                    op,
                    state="unknown",
                    error_code=code,
                    failure_info=failure_info(code, effect="unknown"),
                    cleanup_state="failed",
                    cleanup_error="RESOURCE_CLEANUP_FAILED",
                )
        except Exception:
            log.exception(
                "background_diagnostic_unconfirmed",
                extra={
                    "operation_id": op["id"],
                    "task_id": op["task_id"],
                    "run_id": op["run_id"],
                    "error_code": code,
                },
            )
    finally:
        try:
            await db.release_monitor(op, delay)
        except Exception:
            log.exception(
                "background_monitor_release_unconfirmed",
                extra={
                    "operation_id": op["id"],
                    "task_id": op["task_id"],
                    "run_id": op["run_id"],
                },
            )
