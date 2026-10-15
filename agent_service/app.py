from contextlib import asynccontextmanager

import psycopg
from contracts.errors import install_errors
from contracts.local import LoopbackOnly
from fastapi import FastAPI, Request, Response

from agent_service.api import router
from agent_service.resources import Resources


@asynccontextmanager
async def lifespan(app):
    async with Resources() as resources:
        app.state.resources = resources
        yield


app = FastAPI(title="Agent execution framework", version="2.0.0", lifespan=lifespan)
install_errors(app)
app.add_middleware(LoopbackOnly)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent"}


@app.get("/health/ready")
async def ready(request: Request, response: Response):
    try:
        await request.app.state.resources.store.ready()
        return {"status": "ok", "service": "agent"}
    except (psycopg.Error, OSError, TimeoutError, RuntimeError):
        response.status_code = 503
        return {"status": "unavailable", "service": "agent"}


@app.get("/health/accepting")
async def accepting(request: Request, response: Response):
    try:
        store = request.app.state.resources.store
        queue, workers = await store.summary(), await store.workers()
        cfg = store.settings
        available = (
            workers["online_count"] > 0
            and queue["queued_within_deadline"] < cfg.max_queued_runs
            and (
                queue["queued_within_deadline"]
                + queue["queued_past_deadline"]
                + queue["running_lease_valid"]
                + queue["running_lease_expired"]
                + queue["cancelling"]
                < cfg.max_unfinished_runs
            )
        )
        response.status_code = 200 if available else 503
        return {"service": "agent", "accepting": available}
    except (psycopg.Error, OSError, TimeoutError):
        response.status_code = 503
        return {"service": "agent", "accepting": False}
