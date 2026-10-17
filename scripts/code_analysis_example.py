"""Example trusted startup registration shared by API, Worker and doctor.

Copy this module into deployment-owned Python code to register project checks.
Neither TOML nor HTTP can import Python or supply commands.
"""

import argparse
import asyncio
import json
import signal
from contextlib import asynccontextmanager

from agent_service.code_registry import ProjectCheckDefinition
from agent_service.profiles import Registry
from agent_service.resources import Resources

PROJECT_CHECKS = (
    ProjectCheckDefinition(
        id="project-unittest",
        version="1",
        adapter_id="python-unittest-v1",
        workspaces=("project",),
        description="Discover project unittest tests",
        paths=("tests",),
    ),
    ProjectCheckDefinition(
        id="project-pytest",
        version="1",
        adapter_id="python-pytest-v1",
        workspaces=("project",),
        description="Run project pytest tests",
        paths=("tests",),
    ),
)


def registry():
    return Registry.from_env(project_checks=PROJECT_CHECKS)


@asynccontextmanager
async def api_lifespan(app):
    async with Resources(registry=registry()) as resources:
        app.state.resources = resources
        yield


async def worker_main():
    from agent_service.worker import Worker

    async with Resources(registry=registry()) as resources:
        worker = Worker(resources)
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, worker.stopping.set)
        await worker.run()


async def doctor():
    from agent_service.code_execution import preflight
    from agent_service.execution import DockerClient, check_environment

    async with Resources(registry=registry()) as resources:
        await check_environment(resources.registry.execution)
        await preflight(DockerClient(), resources.registry.execution)
        print(
            json.dumps(
                {
                    "ready": await resources.store.ready(),
                    "code_execution_resources": await resources.store.code_resources(),
                    "unresolved": await resources.store.unresolved(),
                },
                default=str,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("api", "worker", "doctor"))
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    from agent_service.logging import install_logging

    install_logging()
    if args.role == "api":
        import uvicorn

        from agent_service.app import app

        if not 1 <= args.port <= 65535:
            parser.error("Invalid port")
        app.router.lifespan_context = api_lifespan
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            proxy_headers=False,
            log_config=None,
            access_log=False,
        )
    elif args.role == "worker":
        asyncio.run(worker_main())
    else:
        asyncio.run(doctor())


if __name__ == "__main__":
    main()
