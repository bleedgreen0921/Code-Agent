"""Delivery-only restart handling. No model calls, edits, or new test executions."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from agent_service.code_artifacts import CodeFailure
from agent_service.delivery_apply import DeliveryController
from agent_service.delivery_evidence import (
    delivery_summary,
    validate_delivery,
    validate_handoff,
)
from agent_service.delivery_fs import DeliveryConflict, FileTransaction, private_lock
from agent_service.delivery_pipeline import (
    DeliveryHandoffMiddleware,
    PersistentDeliveryBudget,
)
from agent_service.repair_gate import RepairGateMiddleware
from agent_service.repair_reports import markdown_repair
from agent_service.runtime import RunContext, build_agent, run_config
from contracts.repair import RepairDraftV1

log = logging.getLogger(__name__)


class RecoveryModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "forbidden-delivery-recovery"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise CodeFailure(
            "DELIVERY_CONTROL_INVALID", "Recovery must never call a model"
        )

    async def _agenerate(self, *args, **kwargs):
        raise CodeFailure(
            "DELIVERY_CONTROL_INVALID", "Recovery must never call a model"
        )


def recovery_graph(resources, row, delivery):
    scope = SimpleNamespace(
        row=row,
        store=resources.store,
        delivery_recovery=True,
        delivery=SimpleNamespace(
            record=delivery, plan=delivery["plan"], handoffs=delivery["handoffs"]
        ),
    )
    profile = resources.registry.profiles[row["profile"]]
    graph = build_agent(
        RecoveryModel(),
        checkpointer=resources.checkpointer,
        settings=resources.settings,
        tools=[resources.registry.tools[n] for n in profile.tools],
        response_format=ToolStrategy(RepairDraftV1, handle_errors=True),
        state_schema=TodoListMiddleware.state_schema,
        gate_middleware=[
            RepairGateMiddleware(scope),
            PersistentDeliveryBudget(scope),
            DeliveryHandoffMiddleware(scope),
        ],
    )
    return graph


async def recover_one(resources, run_id, *, rollback=False):
    store = resources.store
    record = await store.delivery_for_run(run_id)
    if not record:
        raise ValueError("Delivery not found")
    if record["status"] == "applied" and record["phase"] == "published":
        return record
    if rollback:
        async with store.locked() as conn:
            await conn.execute(
                "UPDATE agent.code_deliveries SET rollback_requested=true WHERE run_id=%s AND status<>'applied'",
                (run_id,),
            )
    run = await store.get(run_id)
    if (
        resources.registry.workspaces.get(run["workspace_id"])
        != record["workspace_root"]
    ):
        raise CodeFailure(
            "CONFIGURATION_CHANGED",
            "Registered workspace root changed; original delivery configuration is required for recovery",
        )
    private = resources.registry.repair.settings.auto_apply_workspaces.get(
        run["workspace_id"]
    )
    if not private:
        raise ValueError("Workspace is not authorized for delivery recovery")
    with private_lock(Path(private), Path(record["workspace_root"])):
        row = await store.claim_delivery_recovery(run_id, uuid4())
        if row is None:
            return dict(status=record["status"], busy=True)
        if rollback:
            async with store.locked() as conn:
                await conn.execute(
                    "UPDATE agent.code_deliveries SET rollback_requested=true WHERE run_id=%s AND status<>'applied'",
                    (run_id,),
                )
        delivery = await store.delivery_for_run(run_id)
        graph = (
            recovery_graph(resources, row, delivery) if delivery["checkpoint"] else None
        )
        if graph:
            saved = await graph.aget_state(delivery["checkpoint"])
            if not saved.tasks or not any(t.interrupts for t in saved.tasks):
                # A final checkpoint may already exist even if its DB association
                # was lost. Inspect the same Run's latest native checkpoint.
                latest = await graph.aget_state(run_config(run_id))
                if latest.next or not latest.values.get("structured_response"):
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID",
                        "Original native handoff is unavailable",
                    )

        async def verify_original(d):
            if d["status"] != "applied":
                await validate_handoff(graph, d)
            return await validate_delivery(store, d)

        controller = DeliveryController(
            store,
            row,
            private,
            resources.registry.repair.settings.model_dump(),
            verify_original,
        )
        # flock is already held across lease takeover, file reconciliation and
        # publication; do not open a second lock description in the controller.
        outcome = await controller.run(
            rollback=rollback, recovery=True, already_locked=True
        )
        if outcome["status"] == "applied":
            proof = await validate_delivery(store, outcome, require_applied=True)
            latest = await graph.aget_state(run_config(run_id))
            if latest.next:
                pending = [i for t in latest.tasks for i in t.interrupts]
                if (
                    not pending
                    or pending[-1].value.get("plan_id") != outcome["plan"]["id"]
                ):
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID", "Applied handoff checkpoint differs"
                    )
                payload = pending[-1]
                resume = dict(
                    delivery_id=str(outcome["id"]),
                    plan_id=outcome["plan"]["id"],
                    action="applied",
                )
                row["_internal_delivery_resume"] = resume
                await graph.ainvoke(
                    Command(resume={payload.id: resume}),
                    run_config(run_id),
                    context=RunContext(
                        run_id=str(run_id), execution_authority={"_agent_closed": True}
                    ),
                    durability="sync",
                )
                latest = await graph.aget_state(run_config(run_id))
            raw = latest.values.get("structured_response")
            raw = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
            if raw != proof["report"]:
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID",
                    "Final checkpoint draft differs from frozen A report",
                )
            if latest.next or not latest.config.get("configurable", {}).get(
                "checkpoint_id"
            ):
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID", "Final native checkpoint not confirmed"
                )
            authority = {
                **proof["authority"],
                "delivery_mode": "auto_apply",
                "delivery_status": "applied",
                "delivery": await delivery_summary(store, outcome),
            }
            answer = markdown_repair(proof["report"], authority)
            # Artifact staging needs the original code stage, not an Agent scope.
            row["_code"] = await store.code_metadata(row["task_id"])
            artifacts = await store.stage_code_report(
                row, proof["report"], answer, authority
            )
            publication = dict(
                answer=answer, checkpoint=latest.config, code_artifacts=artifacts
            )
            await store.finish_delivery_recovery(row, outcome, publication)
        elif outcome["status"] == "rolled_back":
            await store.finish_delivery_recovery(row, outcome)
        else:
            async with store.locked() as conn:
                await conn.execute(
                    "UPDATE agent.code_deliveries SET lease_until=clock_timestamp() WHERE id=%s",
                    (delivery["id"],),
                )
        return await store.delivery_for_run(run_id)


async def recover_deliveries(resources):
    for delivery in await resources.store.recoverable_deliveries():
        try:
            async with asyncio.timeout(60):
                await recover_one(resources, delivery["run_id"])
        except (BlockingIOError, TimeoutError):
            pass  # Owner still holds flock, or the bounded recovery batch yielded.
        except Exception:
            log.exception(
                "delivery_recovery_unconfirmed",
                extra={"run_id": str(delivery["run_id"])},
            )


async def inspect_delivery(resources, task_id):
    detail = await resources.store.delivery_detail(task_id)
    delivery = detail["delivery"]
    if delivery and delivery["plan"]:
        run = await resources.store.get(delivery["run_id"])
        private = resources.registry.repair.settings.auto_apply_workspaces.get(
            run["workspace_id"]
        )
        if private:
            tx = None
            try:
                tx = FileTransaction(
                    Path(resources.registry.workspaces[run["workspace_id"]]),
                    Path(private),
                    delivery["plan"],
                    resources.registry.repair.settings.file_bytes,
                )
                detail["filesystem"] = [
                    dict(path=f["path"], actual=tx.inspect(f))
                    for f in delivery["plan"]["files"]
                ]
            except (DeliveryConflict, OSError) as exc:
                detail["filesystem_error"] = str(exc)
            finally:
                if tx:
                    tx.close()
    return detail
