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
            "TRUNCATE agent.tasks,agent.task_dependencies,agent.background_operations,agent.task_events,agent.conversations,agent.runs,agent.events,agent.workers,agent.checkpoints,agent.checkpoint_blobs,agent.checkpoint_writes CASCADE"
        )

    def run(fn, settings=None):
        async def scoped():
            async with Resources(settings or Settings()) as resources:
                try:
                    return await fn(resources)
                finally:
                    # Test-owned background containers must be removed even on assertion
                    # failure, before the next test truncates their ownership journal.
                    from agent_service.execution import DockerClient, LABEL

                    client = DockerClient()
                    for resource in await resources.store.code_resources():
                        item = await client.inspect(
                            resource["container_id"] or resource["container_name"]
                        )
                        if item:
                            assert (
                                item["Config"]["Labels"].get(LABEL)
                                == resource["container_name"]
                            )
                            assert item["Config"]["Labels"].get(
                                "io.agent-framework.lease"
                            ) == str(resource["lease_token"])
                            await client.call("rm", "--force", item["Id"])
                    for op in await resources.store.operations(raw=True):
                        if op["backend"] != "docker":
                            continue
                        item = await client.inspect(
                            op["container_id"] or op["container_name"]
                        )
                        if item:
                            assert (
                                item["Config"]["Labels"].get(LABEL)
                                == op["container_name"]
                            )
                            assert item["Config"]["Labels"].get(
                                "io.agent-framework.operation"
                            ) == str(op["id"])
                            await client.call("rm", "--force", item["Id"])

        return asyncio.run(scoped())

    return run
