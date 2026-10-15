"""Lease-bound async worker. A claimed Run is never replayed after failure."""

import asyncio
import json
import logging
import signal
import time
from contextlib import suppress
from uuid import uuid4

import openai
import psycopg
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage

from agent_service.resources import Resources
from agent_service.profiles import ConfigurationChanged, Extension
from agent_service.snapshot import runtime_snapshot
from agent_service.runtime import (
    RunContext,
    build_agent,
    model_from_resources,
    run_config,
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

    async def prepare(self, row):
        registry = self.resources.registry
        binding = row["submitted_binding"]
        profile = registry.resolve(binding)
        policy = registry.policy(binding)
        extension = registry.extensions.get(profile.id, Extension())
        middleware = tuple(extension.middleware_factory())
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
        graph = build_agent(
            self.model,
            checkpointer=self.resources.checkpointer,
            tools=list(policy.tools.values()),
            store=self.native_store,
            settings=self.settings,
            system_prompt=profile.system_prompt,
            middleware=middleware,
            state_schema=extension.state_schema,
            response_format=extension.response_format,
            permission_policy=policy,
        )
        self.agent = graph
        context = RunContext(
            run_id=str(row["id"]),
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

    async def generate(self, row):
        graph, context = await self.prepare(row)
        messages = []
        if row["previous_run_id"]:
            prior = await self.db.get(row["previous_run_id"])
            if prior["status"] != "completed" or not prior["final_checkpoint"]:
                raise RuntimeError("Successful predecessor checkpoint required")
            # Read the exact published checkpoint, never the latest intermediate state.
            snapshot = await graph.aget_state(prior["final_checkpoint"])
            if snapshot.next or not snapshot.values.get("messages"):
                raise RuntimeError("Predecessor final checkpoint unavailable")
            messages = list(snapshot.values["messages"])
        config = run_config(row["id"])
        state = await graph.ainvoke(
            {"messages": [*messages, HumanMessage(content=row["task"])]},
            config,
            context=context,
        )
        final = await graph.aget_state(config)
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
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or last.tool_calls:
                raise IncompleteGraphError(
                    "Graph did not produce a final assistant message"
                )
            answer = last.text
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
        work = asyncio.create_task(self.generate(row))
        deadline = time.monotonic() + float(row["remaining"])
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                done, _ = await asyncio.wait(
                    {work}, timeout=min(self.settings.heartbeat_seconds, remaining)
                )
                if work in done:
                    answer, checkpoint = await work
                    await self.publish(row, answer=answer, checkpoint=checkpoint)
                    return
                status = await self.db.renew(row["id"], row["lease_token"])
                if status is None:
                    # Converge expired tasks before returning; never publish after losing ownership.
                    await self.db.maintain()
                    return
                if status == "cancelling":
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
                    await self.publish(row, cancelled=True)
                    return
        except asyncio.CancelledError:
            await self.publish(row, error_code="WORKER_STOPPED")
            raise
        except Exception as exc:
            code = failure_code(exc)
            log.exception(
                "run_execution_failed",
                extra={"run_id": str(row["id"]), "error_code": code},
            )
            await self.publish(row, error_code=code)
        finally:
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)

    async def tick(self):
        await self.db.heartbeat(self.id)
        await self.db.maintain()
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
        for key in ("run_id", "status", "error_code"):
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
