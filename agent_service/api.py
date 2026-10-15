from datetime import datetime
from uuid import UUID

from contracts.v1 import ConversationCreate, RunCreate, RunStatus, TurnCreate
from fastapi import APIRouter, Header, Query, Request, Response
from agent_service.trace import encode_trace, trace_config

from agent_service.storage import public_run

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
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    limit: int = Query(30, ge=1, le=100),
    cursor: UUID | None = None,
):
    return await db(request).runs(
        status=status,
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
    return public_run(await db(request).get(run_id))


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
            "status": row["status"],
            "submitted_binding": row["submitted_binding"],
            "runtime_config": row["runtime_config"],
            "final_checkpoint": row["final_checkpoint"],
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
    return public_run(await db(request).get(turn_id, conversation_id))


@router.post("/conversations/{conversation_id}/turns/{turn_id}/cancel")
async def cancel_turn(
    conversation_id: UUID, turn_id: UUID, request: Request, response: Response
):
    row = await db(request).cancel(turn_id, conversation_id)
    response.status_code = 202 if row["status"] == "cancelling" else 200
    return accepted(row)
