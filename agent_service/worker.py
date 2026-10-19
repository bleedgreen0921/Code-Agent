"""Lease-bound async worker. A claimed Run is never replayed after failure."""

import asyncio
import json
import logging
import signal
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import psycopg
from langchain_core.messages import HumanMessage

from agent_service.reliability import remaining_execution, transient_database, backoff
from agent_service.failure import failure_info, control_leaf, exception_code
from agent_service.resources import Resources
from agent_service.execution import ExecutionScope, recover_resources, check_environment
from agent_service.execution_config import WORKSPACE_TOOLS
from agent_service.execution_errors import EnvironmentUnavailable
from agent_service.execution_adapter import WorkspaceExecutionMiddleware
from agent_service.workspace_tools import assemble
from agent_service.profiles import ConfigurationChanged, Extension, tool_schema
from agent_service.context import inherited_state
from agent_service.skills import read_skill_tool
from langchain.agents.middleware import TodoListMiddleware
from agent_service.snapshot import runtime_snapshot
from agent_service.verification import message_source_eligible
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


class MaintenanceBudgetExpired(TimeoutError):
    pass


class IncompleteGraphError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    answer: str
    checkpoint: dict
    verification: dict | None = None
    error_code: str | None = None
    code_artifacts: dict | None = None

    def publication(self):
        return {
            "answer": self.answer,
            "checkpoint": self.checkpoint,
            "verification": self.verification,
            "error_code": self.error_code,
            **({"code_artifacts": self.code_artifacts} if self.code_artifacts else {}),
        }


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
        self.monitors = {}
        self.stopping = asyncio.Event()

    async def prepare(self, row, scope=None):
        registry = self.resources.registry
        binding = row["submitted_binding"]
        profile = registry.resolve(binding)
        actual_tools = {name: registry.tools[name] for name in profile.tools}
        graph_store = self.native_store
        if registry.knowledge.enabled and binding["workspace_id"]:
            from agent_service.knowledge import (
                ReadOnlyStore,
                knowledge_tools,
                namespace,
            )

            graph_store = ReadOnlyStore(
                self.resources.knowledge_store,
                namespace(binding["workspace_id"], binding["workspace_identity"]),
            )
            actual_tools.update(
                {
                    t.name: t
                    for t in knowledge_tools(graph_store)
                    if t.name in profile.tools
                }
            )
        if mcp_scope := row.get("_mcp_scope"):
            actual_tools.update(await mcp_scope.prepare(profile))
        elif set(profile.tools) & registry.mcp_tools.keys():
            raise RuntimeError("MCP requires a Run-owned connection scope")
        if row.get("task_id"):
            from agent_service.background import background_tools

            actual_tools.update(
                {
                    t.name: t
                    for t in background_tools(self.db, row, registry)
                    if t.name in profile.tools
                }
            )
        native_middleware = []
        execution_middleware = []
        if set(profile.tools) & (WORKSPACE_TOOLS | {"start_background_shell"}):
            if scope is None:
                raise RuntimeError(
                    "Workspace execution requires a Run-owned resource scope"
                )
            row["execution_tools"] = profile.tools
            await scope.prepare()
            if getattr(scope, "evidence", None):
                from agent_service.project_checks import project_check_tool

                actual_tools["run_project_check"] = project_check_tool(scope)
                if scope.row["_code"]["stage"] == "review":
                    from agent_service.review_tools import review_tools

                    actual_tools.update(
                        {tool.name: tool for tool in review_tools(scope)}
                    )
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
                no_progress_enabled=extension.no_progress_enabled,
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
        if row.get("_code_scope"):
            snapshot["code_analysis"] = registry.code_for(row).manifest
            snapshot["middleware"].update(
                model_call_limit=registry.code_for(row).settings.model_calls,
                tool_call_limit=registry.code_for(row).settings.tool_calls,
            )
            snapshot["execution"].update(
                snapshot_id=str(row["_snapshot_id"]),
                mount_identity=row["_code_scope"].manifest["content_digest"],
                git_view_digest=row["_code_scope"].manifest["git_digest"],
                read_only_workspace=True,
            )
            snapshot["run_timeout_seconds"] = min(
                self.settings.run_timeout, registry.code_for(row).settings.run_seconds
            )
            snapshot["deadline_at"] = row["deadline_at"].isoformat()
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
        code_scope = row.get("_code_scope")
        response_format = extension.response_format
        if code_scope:
            from langchain.agents.structured_output import ToolStrategy

            response_format = ToolStrategy(
                code_scope.evidence.response_schema(), handle_errors=True
            )
        graph = build_agent(
            self.model,
            checkpointer=self.resources.checkpointer,
            tools=graph_tools,
            store=graph_store,
            settings=self.settings,
            system_prompt=profile.system_prompt,
            middleware=middleware,
            state_schema=extension.state_schema,
            response_format=response_format,
            **(
                {
                    "model_call_limit": registry.code_for(row).settings.model_calls,
                    "tool_call_limit": registry.code_for(row).settings.tool_calls,
                }
                if code_scope
                else {}
            ),
            permission_policy=policy,
            execution_middleware=execution_middleware,
            output_limits=(
                registry.execution.output_bytes,
                registry.execution.output_lines,
            ),
            skills=[registry.skills[n] for n in sorted(profile.skills)],
            no_progress_enabled=extension.no_progress_enabled,
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
            workspace_root=str(code_scope.snapshot_root)
            if code_scope
            else registry.workspaces.get(binding["workspace_id"]),
            workspace_identity=binding["workspace_identity"],
            binding_fingerprint=binding["fingerprint"],
            deadline_at=row["deadline_at"],
            effective_deadline_at=row.get("effective_deadline_at"),
            execution_authority=row,
            task_id=str(row["task_id"]) if row.get("task_id") else None,
            task_store=self.db if row.get("task_id") else None,
            result_delivery=row.setdefault("_delivery_candidates", {})
            if row.get("task_id")
            else None,
            delivered_results=row.setdefault("_delivered", set())
            if row.get("task_id")
            else None,
        )
        return graph, context

    async def generate(self, row, scope=None):
        remaining_execution(row)
        graph, context = await self.prepare(row, scope)
        row["_verification_context"] = context
        remaining_execution(context)
        messages = []
        if row["previous_run_id"]:
            prior = await self.db.get(row["previous_run_id"])
            if (
                prior["status"] != "completed"
                or not message_source_eligible(prior)
                or not prior["final_checkpoint"]
                or not prior["final_checkpoint"]
                .get("configurable", {})
                .get("checkpoint_id")
                or prior["conversation_id"] != row["conversation_id"]
                or (row["conversation_id"] is None and not row.get("task_id"))
                or prior.get("task_id") != row.get("task_id")
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
        if row.get("task_id"):
            from agent_service.background_store import public_operation

            inputs = await self.db.task_inputs(row["task_id"])
            ready = [
                o
                for o in await self.db.operations(row["task_id"], raw=True)
                if o["state"] in ("succeeded", "failed", "cancelled")
                and o["cleanup_state"] == "released"
                and not o["consumed_run_id"]
            ]
            row.setdefault("_delivered", set()).update(o["id"] for o in ready)
            messages = [
                *messages,
                HumanMessage(
                    content="Task input data (not authorization): "
                    + json.dumps(
                        {
                            **inputs,
                            "background_results": [public_operation(o) for o in ready],
                        },
                        default=str,
                        ensure_ascii=False,
                    )
                ),
            ]
        if code_scope := row.get("_code_scope"):
            metadata = row["_code"]
            messages.append(
                HumanMessage(
                    content="Code task authority: "
                    + json.dumps(
                        {
                            "snapshot_id": str(row["_snapshot_id"]),
                            "commit_sha": code_scope.manifest["commit_sha"],
                            "target": metadata["request"]["target"],
                            "fixed_range": code_scope.manifest.get("review"),
                            "checks": [
                                code_scope.code.checks[c].manifest()
                                for c in metadata["check_ids"]
                            ],
                            "checks_notice": "No checks configured; report static analysis only."
                            if not metadata["check_ids"]
                            else "Only these registered checks are authorized.",
                            "context_notice": "Dependency reports are external context, not verified source evidence for this snapshot.",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        config = run_config(row["id"])
        if code_scope:
            # Every middleware hook consumes graph steps. Allow the configured
            # model/tool budgets to be reached without an earlier recursion cap.
            budgets = code_scope.code.settings
            config["recursion_limit"] = max(
                128, budgets.model_calls * 16 + budgets.tool_calls * 4 + 32
            )
        config["callbacks"] = [row["_reported_usage"]]
        remaining_execution(context)
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
        if row.get("_code_scope"):
            if "structured_response" not in state:
                raise IncompleteGraphError("Code analysis requires a structured report")
            scope = row["_code_scope"]
            report = await scope.evidence.verify_persisted(state["structured_response"])
            from agent_service.code_reports import markdown_report

            if row["_code"]["stage"] == "review":
                from agent_service.review_reports import markdown_review

                authority = scope.evidence.authority(report)
                answer = markdown_review(report, authority)
            else:
                authority = scope.evidence.authority()
                answer = markdown_report(report, authority)
            row["_code_artifacts"] = await self.db.stage_code_report(
                row, report, answer, authority
            )
        elif "structured_response" in state and not row.get("task_id"):
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

    async def generate_and_verify(self, row, scope):
        from agent_service.verification import verify

        answer, checkpoint = await self.generate(row, scope)
        row["_agent_closed"] = True
        row["_candidate_answer"], row["_candidate_checkpoint"] = answer, checkpoint
        context = row.get("_verification_context")
        report, error = await verify(
            row, self.resources.registry, self.db, context, scope, answer, checkpoint
        )
        row["_verification"] = report
        return ExecutionResult(
            answer, checkpoint, report, error, row.get("_code_artifacts")
        )

    async def publish(self, row, **result):
        """Only retry publication. An uncertain commit never causes model/tool replay."""
        if result.get("error_code") and "failure_info" not in result:
            result["failure_info"] = failure_info(result["error_code"])
        from agent_service.code_artifacts import CodeFailure

        for attempt in range(3):
            try:
                confirmed = await self.db.finish(
                    row["id"],
                    row["lease_token"],
                    delivered=row.get("_delivered", ()),
                    **result,
                )
                if not confirmed:
                    current = await self.db.get(row["id"])
                    if current["status"] in ("completed", "failed", "cancelled"):
                        from agent_service.verification import publication_matches

                        return current["lease_token"] == row[
                            "lease_token"
                        ] and publication_matches(current, result)
                return confirmed
            except CodeFailure as exc:
                # A rejected formal report rolls back completion. Publish only the
                # failure on the same Run; never invoke the graph again.
                result = {
                    "error_code": exc.code,
                    "failure_info": failure_info(exc.code),
                }
                if attempt == 2:
                    return False
            except (psycopg.Error, OSError, TimeoutError):
                # A commit acknowledgement may be lost after the transaction committed.
                try:
                    current = await self.db.get(row["id"])
                    if current["status"] in ("completed", "failed", "cancelled"):
                        from agent_service.verification import publication_matches

                        return current["lease_token"] == row[
                            "lease_token"
                        ] and publication_matches(current, result)
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
        from agent_service.code_execution import CodeExecutionScope

        is_code = "code_analysis" in row["submitted_binding"]
        scope = (
            CodeExecutionScope(
                self.db, row, self.resources.registry.execution, self.resources.registry
            )
            if is_code
            else ExecutionScope(self.db, row, self.resources.registry.execution)
            if row.get("_execution_reserved", False)
            else None
        )
        from agent_service.mcp_runtime import MCPScope

        mcp_scope = None
        try:
            profile = self.resources.registry.resolve(row["submitted_binding"])
            if set(profile.tools) & self.resources.registry.mcp_tools.keys():
                mcp_scope = row["_mcp_scope"] = MCPScope(
                    self.db, row, self.resources.registry
                )
        except ConfigurationChanged:
            pass  # prepare publishes the established configuration error.
        registry = self.resources.registry
        cleanup_budget = (registry.execution.cleanup_timeout if scope else 0) + (
            registry.mcp.cleanup_timeout if mcp_scope else 0
        )
        publication_reserve = registry.reliability.publication_reserve
        if is_code:
            cleanup_budget = max(
                cleanup_budget,
                self.resources.registry.code_for(row).settings.reserve_seconds
                - publication_reserve,
            )
        row["effective_deadline_at"] = row["deadline_at"] - timedelta(
            seconds=cleanup_budget + publication_reserve
        )
        row["_lease_until"] = row["lease_until"]
        work = asyncio.create_task(self.generate_and_verify(row, scope))
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
                    confirmation = await self.db.renew(
                        row["id"], row["lease_token"], with_lease=True
                    )
                    status = confirmation["status"] if confirmation else None
                    if confirmation:
                        row["_lease_until"] = confirmation["lease_until"]
                    if status is None:
                        lost = True
                        abort.set()
                    elif status == "cancelling":
                        cancelled = True
                        abort.set()
                except (psycopg.Error, OSError, TimeoutError):
                    # Publication still checks lease validity in the database.
                    log.exception(
                        "lease_renewal_unconfirmed",
                        extra={"run_id": row["id"], "task_id": row.get("task_id")},
                    )

        async def supervise_lease():
            nonlocal lost
            while True:
                remaining = (
                    row["_lease_until"] - datetime.now(timezone.utc)
                ).total_seconds()
                if remaining <= 0:
                    lost = True
                    abort.set()
                    return
                await asyncio.sleep(min(remaining, self.settings.heartbeat_seconds))

        heartbeat = asyncio.create_task(renew())
        lease_guard = asyncio.create_task(supervise_lease())
        interrupt = asyncio.create_task(abort.wait())
        try:
            done, _ = await asyncio.wait(
                {work, interrupt},
                timeout=max(
                    0,
                    min(
                        float(row["remaining"]),
                        (
                            row["effective_deadline_at"] - datetime.now(timezone.utc)
                        ).total_seconds(),
                    ),
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if lost or cancelled:
                pass
            elif work in done:
                outcome = await work
                result = outcome.publication()
            else:
                result = {"error_code": "RUN_TIMEOUT"}
        except asyncio.CancelledError:
            shutdown = True
            result = {"error_code": "WORKER_STOPPED"}
        except Exception as exc:
            result = {"error_code": failure_code(exc)}
            log.exception(
                "run_execution_failed",
                extra={
                    "run_id": str(row["id"]),
                    "task_id": row.get("task_id"),
                    **result,
                },
            )
        finally:
            row["_execution_closed"] = True
            # Freeze admissions before cancelling graph work; executor threads cannot respawn.
            if mcp_scope:
                mcp_scope.closing = True
            if scope:
                with scope.sync_lock:
                    scope.closing = True
            if not work.done():
                work.cancel()
            loop = asyncio.get_running_loop()
            cleanup_deadline = min(
                loop.time() + cleanup_budget,
                loop.time()
                + max(
                    0,
                    (row["deadline_at"] - datetime.now(timezone.utc)).total_seconds()
                    - publication_reserve,
                ),
            )
            # asyncio.wait bounds a graph that is slow to honour cancellation.
            await asyncio.wait({work}, timeout=max(0, cleanup_deadline - loop.time()))
            graph_stopped = work.done()
            if graph_stopped:
                with suppress(BaseException):
                    work.result()
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

            async def close_resource(resource):
                if resource is None:
                    return True
                try:
                    async with asyncio.timeout_at(cleanup_deadline):
                        return await resource.close(cleanup_deadline)
                except Exception:
                    log.exception(
                        "run_cleanup_unconfirmed", extra={"run_id": row["id"]}
                    )
                    return False

            try:
                mcp_clean = await close_resource(mcp_scope)
                clean = await close_resource(scope)
                clean = clean and mcp_clean and graph_stopped
                if not clean and not result.get("error_code"):
                    result["error_code"] = "RESOURCE_CLEANUP_FAILED"
                if lost:
                    # Publish no result, but allow the existing DB expiry rules to
                    # close the Run when maintenance is available in this process.
                    try:
                        async with asyncio.timeout(publication_reserve):
                            await self.db.maintain()
                    except Exception:
                        log.exception(
                            "lease_renewal_unconfirmed", extra={"run_id": row["id"]}
                        )
                if not lost:
                    seconds = max(
                        0,
                        min(
                            publication_reserve,
                            (
                                row["deadline_at"] - datetime.now(timezone.utc)
                            ).total_seconds(),
                        ),
                    )
                    try:
                        async with asyncio.timeout(seconds):
                            if row.get("_verification"):
                                result.update(
                                    verification=row["_verification"],
                                    candidate_answer=row["_candidate_answer"],
                                    candidate_checkpoint=row["_candidate_checkpoint"],
                                )
                            await self.publish(
                                row,
                                **{
                                    **result,
                                    **({"cancelled": True} if cancelled else {}),
                                },
                            )
                    except TimeoutError:
                        log.error(
                            "result_publication_unconfirmed",
                            extra={"run_id": row["id"]},
                        )
            finally:
                heartbeat.cancel()
                lease_guard.cancel()
                interrupt.cancel()
                await asyncio.gather(
                    heartbeat, lease_guard, interrupt, return_exceptions=True
                )
        if shutdown:
            raise asyncio.CancelledError

    async def maintenance_phase(self, limit):
        async def phase():
            await self.db.heartbeat(self.id)
            await self.db.maintain()

        pending = asyncio.create_task(phase())
        stop = asyncio.create_task(self.stopping.wait())
        try:
            done, _ = await asyncio.wait(
                {pending, stop}, timeout=limit, return_when=asyncio.FIRST_COMPLETED
            )
            if stop in done:
                return False
            if pending not in done:
                raise MaintenanceBudgetExpired()
            await pending
            return True
        finally:
            pending.cancel()
            stop.cancel()
            await asyncio.wait({pending, stop}, timeout=min(0.2, max(0, limit)))

            # Consume late failures without waiting beyond the maintenance budget.
            def consume(task):
                if not task.cancelled():
                    task.exception()

            pending.add_done_callback(consume)

    async def maintain(self):
        """Only maintenance DB transport failures own this finite recovery window."""
        policy = self.resources.registry.reliability
        loop = asyncio.get_running_loop()
        failures = 0
        first = None
        while not self.stopping.is_set():
            try:
                limit = (
                    policy.maintenance_window
                    if first is None
                    else max(0, first + policy.maintenance_window - loop.time())
                )
                return await self.maintenance_phase(limit)
            except Exception as exc:
                if not transient_database(exc) and not isinstance(
                    exc, MaintenanceBudgetExpired
                ):
                    raise
                if isinstance(exc, MaintenanceBudgetExpired):
                    first = (
                        first
                        if first is not None
                        else loop.time() - policy.maintenance_window
                    )
                failures += 1
                first = first if first is not None else loop.time()
                remaining = first + policy.maintenance_window - loop.time()
                if failures >= policy.maintenance_max_failures or remaining <= 0:
                    self.stopping.set()
                    log.error(
                        "worker_maintenance_exhausted",
                        extra={"recovery_attempts": failures},
                    )
                    raise RuntimeError(
                        "Worker maintenance recovery exhausted"
                    ) from None
                delay = min(
                    remaining,
                    backoff(
                        failures,
                        policy.maintenance_backoff_base,
                        policy.maintenance_backoff_cap,
                    ),
                )
                log.info(
                    "worker_maintenance_retry",
                    extra={
                        "recovery_attempts": failures,
                        "recovery_phase": "maintenance",
                    },
                )
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.stopping.wait(), delay)
        return False

    async def tick(self):
        if not await self.maintain():
            return
        from agent_service.code_execution import recover_code_resources

        await recover_code_resources(self.db, self.resources.registry.execution)
        now = asyncio.get_running_loop().time()
        if now >= getattr(self, "_code_gc_at", 0):
            await self.db.gc_code_content()
            self._code_gc_at = now + self.settings.reconciliation_seconds
        await recover_resources(self.db, self.resources.registry.execution)
        from agent_service.background import monitor_operation

        for oid, task in list(self.monitors.items()):
            if task.done():
                await task
                del self.monitors[oid]
        for op in await self.db.claim_operations():
            self.monitors[op["id"]] = asyncio.create_task(
                monitor_operation(self.db, self.resources.registry, op)
            )
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
        if (
            self.resources.registry.code.settings.enabled
            or self.resources.registry.review.settings.enabled
        ):
            from agent_service.code_execution import preflight
            from agent_service.execution import DockerClient

            await preflight(DockerClient(), self.resources.registry.execution)
        if self.resources.registry.execution.image:
            try:
                await check_environment(
                    self.resources.registry.execution,
                    checks=self.resources.registry.check_definitions(),
                )
            except EnvironmentUnavailable:
                if self.resources.registry.check_definitions():
                    raise
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
            for task in self.monitors.values():
                task.cancel()
            await asyncio.gather(*self.monitors.values(), return_exceptions=True)
            try:
                async with asyncio.timeout(
                    min(2, self.resources.registry.reliability.publication_reserve)
                ):
                    await self.db.heartbeat(self.id, stopped=True)
            except Exception:
                log.exception(
                    "lease_renewal_unconfirmed", extra={"recovery_phase": "shutdown"}
                )


def failure_code(exc):
    if isinstance(control_leaf(exc), IncompleteGraphError):
        return "GRAPH_INTERRUPTED"
    return exception_code(exc)


async def main():
    async with Resources() as resources:
        worker = Worker(resources)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, worker.stopping.set)
        await worker.run()


if __name__ == "__main__":
    from agent_service.logging import install_logging

    install_logging()
    asyncio.run(main())
