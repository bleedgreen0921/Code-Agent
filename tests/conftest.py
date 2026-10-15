import asyncio
import os

import psycopg
import pytest
from agent_service.config import Settings
from agent_service.resources import Resources
from scripts.integration_pytest import no_real_http


@pytest.fixture(autouse=True)
def http_guard():
    with no_real_http() as attempts:
        yield
    assert not attempts, "Tests attempted real HTTP"


@pytest.fixture
def database():
    if os.environ.get("AGENT_TEST_DATABASE") != "1":
        pytest.fail(
            "Use python -m scripts.postgres_integration for an isolated test database"
        )
    with psycopg.connect(os.environ["MIGRATION_DATABASE_URL"], autocommit=True) as conn:
        conn.execute(
            "TRUNCATE agent.conversations,agent.runs,agent.events,agent.workers,agent.checkpoints,agent.checkpoint_blobs,agent.checkpoint_writes CASCADE"
        )

    def run(fn, settings=None):
        async def scoped():
            async with Resources(settings or Settings()) as resources:
                return await fn(resources)

        return asyncio.run(scoped())

    return run
