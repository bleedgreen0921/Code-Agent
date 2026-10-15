"""Lease-bound async worker. A claimed Run is never replayed after failure."""

import asyncio
import json
import logging
import signal
from contextlib import suppress
from uuid import uuid4

import openai
import psycopg
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import HumanMessage

from agent_service.resources import Resources
from agent_service.execution import ExecutionScope, recover_resources, check_environment
from agent_service.execution_config import WORKSPACE_TOOLS
from agent_service.execution_errors import ExecutionFailure, EnvironmentUnavailable
from agent_service.execution_adapter import WorkspaceExecutionMiddleware
from agent_service.workspace_tools import assemble
from agent_service.profiles import ConfigurationChanged, Extension, tool_schema
from agent_service.context import inherited_state
from agent_service.skills import read_skill_tool
from langchain.agents.middleware import TodoListMiddleware
from agent_service.snapshot import runtime_snapshot
from agent_service.runtime import (
    RunContext,
    build_agent,
    model_from_resources,
    run_config,
)

from agent_service.delegation import (
    DelegationController,
    DelegationDiagnostics,
    ReportedUsage,
    delegate_tool,
    final_text,
    InvalidFinalText,
)

log = logging.getLogger(__name__)


class IncompleteGraphError(RuntimeError):
    pass


class Worker:
    def __init__(self, resources, *, model=None, tools=(), store=None):
        self.resources = resources
        self.db = resources.store
        self.settings = resources.settings
        self.id = uuid4()
        self.model = model if model is not None else model_from_resources(resources)
        self.native_store = store
        if tools and any(resources.registry.tools.get(t.name) is not t for t in tools):
            raise ValueError("Worker tools must be registered in the trusted registry")
        # Kept for native checkpoint diagnostics; execution always uses a local graph.
        self.agent = build_agent(
            self.model, checkpointer=resources.checkpointer, settings=self.settings
        )
        self.tasks = {}
        self.stopping = asyncio.Event()

    async def prepare(self, row, scope=None):
        registry = self.resources.registry
        binding = row["submitted_binding"]
        profile = registry.resolve(binding)
        actual_tools = {name: registry.tools[name] for name in profile.tools}
        native_middleware = []
        execution_middleware = []
        if set(profile.tools) & WORKSPACE_TOOLS:
            if scope is None:
                raise RuntimeError(
                    "Workspace execution requires a Run-owned resource scope"
                )
            row["execution_tools"] = profile.tools
            await scope.prepare()
            instances, native_middleware = assemble(scope, profile.tools)
            actual_tools.update({tool.name: tool for tool in instances})
            execution_middleware = [
                WorkspaceExecutionMiddleware(
                    scope, set(profile.tools) & WORKSPACE_TOOLS
                )
            ]
        extension = registry.extensions.get(profile.id, Extension())
        if "write_todos" in profile.tools:
            native_middleware.append(TodoListMiddleware())
        if "read_skill" in profile.tools:
            actual_tools["read_skill"] = read_skill_tool(
                {n: registry.skills[n] for n in profile.skills}
            )
        middleware = (*extension.middleware_factory(), *native_middleware)
        generated = [t for m in middleware for t in getattr(m, "tools", ())]
        names = [t.name for t in generated]
        if "delegate_read" in names:
            raise ValueError("delegate_read must be assembled by the Run owner")
        if len(names) != len(set(names)) or set(names) - set(profile.tools):
            raise ValueError("Duplicate or undeclared middleware tools")
        if any(
            tool_schema(t) != tool_schema(registry.tools[t.name]) for t in generated
        ):
            raise ValueError("Middleware tool schema differs from its declaration")
        actual_tools.update({t.name: t for t in generated})
        policy = registry.policy(binding, actual_tools=actual_tools)
        usage = row["_reported_usage"] = ReportedUsage()
        if "delegate_read" in profile.tools:
            controller = DelegationController(
                model=self.model,
                checkpointer=self.resources.checkpointer,
                settings=self.settings,
                scope=scope,
                policy=policy,
                skills=[registry.skills[n] for n in sorted(profile.skills)],
                system_prompt=profile.system_prompt,
            )
            actual_tools["delegate_read"] = delegate_tool(controller)
            policy = registry.policy(binding, actual_tools=actual_tools)
            middleware = (
                *middleware,
                DelegationDiagnostics(controller, usage, extension.state_schema),
            )
        snapshot = runtime_snapshot(
            self.model, self.settings, registry, binding, middleware
        )
        # Retry only the DB operation, before invoking the graph even once.
        for attempt in range(3):
            try:
                confirmed = await self.db.record_runtime_config(
                    row["id"], row["lease_token"], snapshot
                )
                if not confirmed:
                    raise RuntimeError(
                        "Runtime snapshot not confirmed under a valid lease"
                    )
                break
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise
        graph_tools = [
            t
            for t in policy.tools.values()
            if t.name not in {t.name for t in generated}
        ]
        names = [t.name for t in graph_tools] + [
            t.name for m in middleware for t in getattr(m, "tools", ())
        ]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate runtime tool names")
        graph = build_agent(
            self.model,
            checkpointer=self.resources.checkpointer,
            tools=graph_tools,
            store=self.native_store,
            settings=self.settings,
            system_prompt=profile.system_prompt,
            middleware=middleware,
            state_schema=extension.state_schema,
            response_format=extension.response_format,
            permission_policy=policy,
            execution_middleware=execution_middleware,
            output_limits=(
                registry.execution.output_bytes,
                registry.execution.output_lines,
            ),
            skills=[registry.skills[n] for n in sorted(profile.skills)],
        )
        self.agent = graph
        context = RunContext(
            run_id=str(row["id"]),
            agent_id=str(row["id"]),
            conversation_id=str(row["conversation_id"])
            if row["conversation_id"]
            else None,
            profile=profile.id,
            workspace_id=binding["workspace_id"],
            workspace_root=registry.workspaces.get(binding["workspace_id"]),
            workspace_identity=binding["workspace_identity"],
            binding_fingerprint=binding["fingerprint"],
            deadline_at=row["deadline_at"],
        )
        return graph, context

    async def generate(self, row, scope=None):
        graph, context = await self.prepare(row, scope)
        messages = []
        if row["previous_run_id"]:
            prior = await self.db.get(row["previous_run_id"])
            if (
                prior["status"] != "completed"
                or not prior["final_checkpoint"]
                or not prior["final_checkpoint"]
                .get("configurable", {})
                .get("checkpoint_id")
                or prior["conversation_id"] != row["conversation_id"]
                or row["conversation_id"] is None
                or prior["final_checkpoint"].get("configurable", {}).get("thread_id")
                != str(prior["id"])
                or prior["final_checkpoint"]
                .get("configurable", {})
                .get("checkpoint_ns", "")
                != ""
            ):
                raise RuntimeError("Successful predecessor checkpoint required")
            # Read the exact published checkpoint, never the latest intermediate state.
            snapshot = await graph.aget_state(prior["final_checkpoint"])
            if snapshot.next or not snapshot.values.get("messages"):
                raise RuntimeError("Predecessor final checkpoint unavailable")
            messages = inherited_state(snapshot.values)["messages"]
        config = run_config(row["id"])
        config["callbacks"] = [row["_reported_usage"]]
        state = await graph.ainvoke(
            {"messages": [*messages, HumanMessage(content=row["task"])]},
            config,
            context=context,
            durability="sync",
        )
        if scope:
            await scope.healthy()
        final = await graph.aget_state(
            {**config, "configurable": {**config["configurable"], "checkpoint_ns": ""}}
        )
        if final.next or state.get("__interrupt__"):
            raise IncompleteGraphError(
                "Graph interrupted; no HTTP resume endpoint is enabled"
            )
        if "structured_response" in state:
            from fastapi.encoders import jsonable_encoder

            answer = json.dumps(
                jsonable_encoder(state["structured_response"]),
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            try:
                answer = final_text(state)
            except InvalidFinalText as exc:
                raise IncompleteGraphError(str(exc)) from exc
        if not final.config.get("configurable", {}).get("checkpoint_id"):
            raise RuntimeError("Final checkpoint was not persisted")
        return answer, final.config

    async def publish(self, row, **result):
        """Only retry publication. An uncertain commit never causes model/tool replay."""
        for attempt in range(3):
            try:
                return await self.db.finish(row["id"], row["lease_token"], **result)
            except (psycopg.Error, OSError, TimeoutError):
                # A commit acknowledgement may be lost after the transaction committed.
                try:
                    current = await self.db.get(row["id"])
                    if current["status"] in ("completed", "failed", "cancelled"):
                        return current["status"] == "completed"
                except (psycopg.Error, OSError, TimeoutError):
                    pass
                if attempt == 2:
                    log.exception(
                        "result_publication_unconfirmed",
                        extra={"run_id": str(row["id"])},
                    )
                    return False
                await asyncio.sleep(0.05 * (attempt + 1))

    async def execute(self, row):
        scope = (
            ExecutionScope(self.db, row, self.resources.registry.execution)
            if await self.db.execution_resource(row["id"])
            else None
        )
        work = asyncio.create_task(self.generate(row, scope))
        lost = False
        cancelled = False
        shutdown = False
        abort = asyncio.Event()
        result = {}

        async def renew():
            nonlocal lost, cancelled
            while True:
                await asyncio.sleep(self.settings.heartbeat_seconds)
                try:
                    status = await self.db.renew(row["id"], row["lease_token"])
                    if status is None:
                        lost = True
                        abort.set()
                        await self.db.maintain()
                    elif status == "cancelling":
                        cancelled = True
                        abort.set()
                except (psycopg.Error, OSError, TimeoutError):
                    # Publication still checks lease validity in the database.
                    log.exception("lease_renewal_unconfirmed")

        heartbeat = asyncio.create_task(renew())
        interrupt = asyncio.create_task(abort.wait())
        try:
            done, _ = await asyncio.wait(
                {work, interrupt},
                timeout=float(row["remaining"]),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if lost or cancelled:
                pass
            elif work in done:
                answer, checkpoint = await work
                result = {"answer": answer, "checkpoint": checkpoint}
            else:
                result = {"error_code": "RUN_TIMEOUT"}
        except asyncio.CancelledError:
            shutdown = True
            result = {"error_code": "WORKER_STOPPED"}
        except Exception as exc:
            result = {"error_code": failure_code(exc)}
            log.exception(
                "run_execution_failed", extra={"run_id": str(row["id"]), **result}
            )
        finally:
            # Freeze admissions before cancelling graph work; executor threads cannot respawn.
            if scope:
                with scope.sync_lock:
                    scope.closing = True
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            if usage := row.get("_reported_usage"):
                log.info(
                    "run_reported_usage",
                    extra={
                        "run_id": str(row["id"]),
                        "reported_usage": usage.report(
                            aborted=not bool(result.get("answer"))
                        ),
                    },
                )
            try:
                clean = await scope.close() if scope else True
                if not clean and not result.get("error_code"):
                    result = {"error_code": "RESOURCE_CLEANUP_FAILED"}
                if not lost:
                    await self.publish(
                        row, **({"cancelled": True} if cancelled else result)
                    )
            finally:
                heartbeat.cancel()
                interrupt.cancel()
                await asyncio.gather(heartbeat, interrupt, return_exceptions=True)
        if shutdown:
            raise asyncio.CancelledError

    async def tick(self):
        await self.db.heartbeat(self.id)
        await self.db.maintain()
        await recover_resources(self.db, self.resources.registry.execution)
        for run_id, task in list(self.tasks.items()):
            if task.done():
                await task
                del self.tasks[run_id]
        while len(self.tasks) < self.settings.max_active_runs:
            row = await self.db.claim(self.id)
            if row is None:
                break
            self.tasks[row["id"]] = asyncio.create_task(self.execute(row))

    async def run(self):
        if self.resources.registry.execution.image:
            try:
                await check_environment(self.resources.registry.execution)
            except EnvironmentUnavailable:
                # A broken workspace environment must not prevent empty-tool chat runs.
                # Workspace runs perform their own fatal preflight before any model call.
                log.exception("execution_environment_preflight_failed")
        try:
            while not self.stopping.is_set():
                await self.tick()
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self.stopping.wait(), self.settings.poll_seconds
                    )
        finally:
            for task in self.tasks.values():
                task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            await self.db.heartbeat(self.id, stopped=True)


def failure_code(exc):
    if isinstance(exc, ExecutionFailure):
        return exc.code
    if isinstance(exc, ConfigurationChanged):
        return "CONFIGURATION_CHANGED"
    if isinstance(exc, (ModelCallLimitExceededError, ToolCallLimitExceededError)):
        return "QUOTA_EXCEEDED"
    if isinstance(exc, openai.APITimeoutError):
        return "MODEL_TIMEOUT"
    if isinstance(exc, TimeoutError):
        return "RUN_TIMEOUT"
    if isinstance(exc, openai.AuthenticationError):
        return "MODEL_AUTH_FAILED"
    if isinstance(exc, openai.RateLimitError):
        return "MODEL_RATE_LIMITED"
    if isinstance(exc, openai.APIError):
        return "MODEL_CALL_FAILED"
    if isinstance(exc, IncompleteGraphError):
        return "GRAPH_INTERRUPTED"
    return "EXECUTION_FAILED"


async def main():
    async with Resources() as resources:
        worker = Worker(resources)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, worker.stopping.set)
        await worker.run()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "run_id",
            "status",
            "error_code",
            "history_tokens_approx",
            "request_tokens_approx",
            "reported_usage",
            "delegation",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    asyncio.run(main())
