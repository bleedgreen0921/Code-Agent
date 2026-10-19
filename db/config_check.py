import argparse
import asyncio
import os

from agent_service.config import Settings
from agent_service.profiles import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("agent-api", "agent-worker"), required=True)
    args = parser.parse_args()
    Settings.from_env()
    registry = Registry.from_env()
    if args.role == "agent-worker" and registry.execution.image:
        from agent_service.execution import check_environment

        asyncio.run(
            check_environment(registry.execution, checks=registry.check_definitions())
        )
        if registry.code.settings.enabled or registry.review.settings.enabled:
            from agent_service.code_execution import preflight
            from agent_service.execution import DockerClient

            asyncio.run(preflight(DockerClient(), registry.execution))
    names = ["AGENT_DATABASE_URL"]
    if args.role == "agent-worker":
        names += ["AGENT_MODEL_URL", "AGENT_MODEL"]
    for name in names:
        if not os.environ.get(name):
            raise ValueError(f"{name} required")
    print("Configuration valid")


if __name__ == "__main__":
    main()
