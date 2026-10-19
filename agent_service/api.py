from datetime import datetime
from uuid import UUID

from contracts.v1 import (
    ConversationCreate,
    RunCreate,
    RunStatus,
    TurnCreate,
    TaskCreate,
    TaskStatus,
)
from fastapi import APIRouter, Header, Query, Request, Response
from agent_service.trace import encode_trace, trace_config


router = APIRouter(prefix="/v1")


def db(request):
    return request.app.state.resources.store


def accepted(row):
    return {
        "run_id": str(row["id"]),
        "status": row["status"],
        "created_at": row["created_at"],
    }


@router.post("/runs")
async def submit(
    body: RunCreate,
    request: Request,
    response: Response,
    key: str | None = Header(None, alias="Idempotency-Key"),
):
    row, replay = await db(request).submit(
        body.task, key, profile=body.profile, workspace_id=body.workspace_id
    )
    response.status_code = 200 if replay else 202
    return accepted(row)


@router.get("/runs")
async def runs(
    request: Request,
    status: RunStatus | None = None,
    task_id: UUID | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).runs(
        status=status,
        task_id=task_id,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        cursor=cursor,
    )


@router.get("/runs/summary")
async def summary(request: Request):
    return await db(request).summary()


@router.get("/runs/{run_id}")
async def run(run_id: UUID, request: Request):
    return await db(request).run_detail(run_id)


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: UUID, request: Request, response: Response):
    row = await db(request).cancel(run_id)
    response.status_code = 202 if row["status"] == "cancelling" else 200
    return accepted(row)


@router.get("/runs/{run_id}/timeline")
async def timeline(run_id: UUID, request: Request):
    return await db(request).timeline(run_id)


@router.get("/runs/{run_id}/trace")
async def trace(
    run_id: UUID,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    namespace: str | None = None,
):
    row = await db(request).get(run_id)
    checkpoints = []
    async for cp in request.app.state.resources.checkpointer.alist(
        trace_config(run_id, namespace), limit=limit
    ):
        checkpoints.append(
            {
                "config": cp.config,
                "parent_config": cp.parent_config,
                "metadata": cp.metadata,
                "checkpoint": cp.checkpoint,
                "pending_writes": cp.pending_writes,
            }
        )
    return encode_trace(
        {
            "run_id": str(run_id),
            "task_id": row["task_id"],
            "execution_no": row["execution_no"],
            "status": row["status"],
            "submitted_binding": row["submitted_binding"],
            "runtime_config": row["runtime_config"],
            "final_checkpoint": row["final_checkpoint"],
            "verification": row["verification"],
            "external_operations": await db(request).external_operations(run_id),
            "checkpoints": checkpoints,
        }
    )


@router.get("/workers")
async def workers(request: Request):
    return await db(request).workers()


@router.post("/conversations", status_code=201)
async def create_conversation(
    request: Request, body: ConversationCreate = ConversationCreate()
):
    row = await db(request).create_conversation(
        body.title, body.profile, body.workspace_id
    )
    return {**row, "conversation_id": str(row["id"])}


@router.get("/conversations")
async def conversations(
    request: Request, limit: int = Query(30, ge=1, le=100), cursor: UUID | None = None
):
    return await db(request).conversations(limit, cursor)


@router.get("/conversations/{conversation_id}")
async def conversation(conversation_id: UUID, request: Request):
    return await db(request).conversation(conversation_id)


@router.post("/conversations/{conversation_id}/turns")
async def submit_turn(
    conversation_id: UUID,
    body: TurnCreate,
    request: Request,
    response: Response,
    key: str | None = Header(None, alias="Idempotency-Key"),
):
    row, replay = await db(request).submit(body.task, key, conversation_id)
    response.status_code = 200 if replay else 202
    return {**accepted(row), "turn_id": str(row["id"]), "turn_no": row["turn_no"]}


@router.get("/conversations/{conversation_id}/turns/{turn_id}")
async def turn(conversation_id: UUID, turn_id: UUID, request: Request):
    return await db(request).run_detail(turn_id, conversation_id)


@router.post("/conversations/{conversation_id}/turns/{turn_id}/cancel")
async def cancel_turn(
    conversation_id: UUID, turn_id: UUID, request: Request, response: Response
):
    row = await db(request).cancel(turn_id, conversation_id)
    response.status_code = 202 if row["status"] == "cancelling" else 200
    return accepted(row)


@router.post("/tasks")
async def create_task(
    body: TaskCreate,
    request: Request,
    response: Response,
    key: str | None = Header(None, alias="Idempotency-Key"),
):
    row, replay = await db(request).create_task(
        body.task,
        key,
        profile=body.profile,
        workspace_id=body.workspace_id,
        depends_on=body.depends_on,
    )
    response.status_code = 200 if replay else 202
    return row


@router.get("/tasks")
async def tasks(
    request: Request,
    status: TaskStatus | None = None,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).list_tasks(limit, cursor, status)


@router.get("/tasks/{task_id}")
async def task(task_id: UUID, request: Request):
    return await db(request).task(task_id)


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: UUID, request: Request, response: Response):
    row = await db(request).cancel_task(task_id)
    response.status_code = 202 if row["status"] == "cancelling" else 200
    return row


@router.get("/tasks/{task_id}/timeline")
async def task_timeline(task_id: UUID, request: Request):
    return await db(request).task_timeline(task_id)


@router.get("/tasks/{task_id}/operations")
async def task_operations(task_id: UUID, request: Request):
    await db(request).task(task_id)
    return {"items": await db(request).operations(task_id)}


@router.get("/tasks/{task_id}/operations/{operation_id}")
async def operation(task_id: UUID, operation_id: UUID, request: Request):
    return await db(request).operation(task_id, operation_id)


@router.post("/tasks/{task_id}/operations/{operation_id}/cancel")
async def cancel_operation(
    task_id: UUID, operation_id: UUID, request: Request, response: Response
):
    row = await db(request).cancel_operation(task_id, operation_id)
    response.status_code = 202 if row["state"] == "cancelling" else 200
    return row


from contracts.code import CodeTaskCreate
from contracts.review import CodeReviewCreate
from contracts.repair import CodeRepairCreate
from typing import Annotated
from pydantic import Discriminator, Tag


def code_stage(value):
    return value.get("stage", "analysis") if isinstance(value, dict) else value.stage


@router.post("/code/tasks")
async def create_code_task(
    body: Annotated[
        Annotated[CodeTaskCreate, Tag("analysis")]
        | Annotated[CodeReviewCreate, Tag("review")]
        | Annotated[CodeRepairCreate, Tag("repair")],
        Discriminator(code_stage),
    ],
    request: Request,
    response: Response,
    key: str | None = Header(None, alias="Idempotency-Key"),
):
    row, replay = await db(request).create_code_task(body, key)
    response.status_code = 200 if replay else 202
    return row


@router.get("/code/tasks/{task_id}")
async def code_task(task_id: UUID, request: Request):
    return await db(request).code_task(task_id)


@router.get("/code/tasks/{task_id}/checks")
async def code_checks(
    task_id: UUID,
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).code_page(task_id, "checks", limit, cursor)


@router.get("/code/tasks/{task_id}/artifacts")
async def code_artifacts(
    task_id: UUID,
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).code_page(task_id, "artifacts", limit, cursor)


@router.get("/code/tasks/{task_id}/artifacts/{artifact_id}")
async def code_artifact(task_id: UUID, artifact_id: UUID, request: Request):
    metadata, data = await db(request).code_artifact(task_id, artifact_id)
    media = (
        "text/markdown; charset=utf-8"
        if metadata["kind"]
        in ("analysis_markdown", "review_markdown", "repair_markdown")
        else "application/zip"
        if metadata["kind"]
        in (
            "repair_package",
            "repair_baseline",
            "delivery_package",
            "integration_package",
        )
        else "application/json"
        if metadata["kind"]
        in (
            "repair_report",
            "repair_manifest",
            "repair_diagnostic",
            "delivery_manifest",
            "integration_manifest",
            "delivery_candidate_proof",
            "analysis_report",
            "review_report",
            "source_evidence",
            "review_evidence",
            "review_changes",
            "review_coverage",
            "check_result",
            "reproduction_result",
        )
        else "text/plain; charset=utf-8"
    )
    return Response(
        data,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact_id}"',
            "ETag": '"' + metadata["sha256"] + '"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/code/tasks/{task_id}/changes")
async def review_changes(
    task_id: UUID,
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await db(request).review_changes(task_id, limit, offset)


@router.get("/code/tasks/{task_id}/reproductions")
async def review_reproductions(
    task_id: UUID,
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).code_page(task_id, "reproductions", limit, cursor)


@router.get("/code/tasks/{task_id}/candidate")
async def repair_candidate(task_id: UUID, request: Request):
    return await db(request).repair_candidate(task_id)


@router.get("/code/tasks/{task_id}/repair-attempts")
async def repair_attempts(
    task_id: UUID,
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).code_page(task_id, "repair_attempts", limit, cursor)


@router.get("/code/tasks/{task_id}/delivery")
async def code_delivery(task_id: UUID, request: Request):
    return await db(request).delivery_detail(task_id)


@router.get("/code/tasks/{task_id}/delivery/events")
async def code_delivery_events(
    task_id: UUID,
    request: Request,
    limit: int = Query(default=100, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).delivery_events(task_id, limit, cursor)
