import argparse
import asyncio
import json

from agent_service.resources import Resources


async def inspect():
    async with Resources() as resources:
        return {
            "ready": await resources.store.ready(),
            "workers": await resources.store.workers(),
            "execution_resources": await resources.store.resource_diagnostics(),
            "background_operations": await resources.store.operations(),
            "external_operations": await resources.store.external_operations(),
            "unresolved": await resources.store.unresolved(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.parse_args()
    print(json.dumps(asyncio.run(inspect()), default=str))


if __name__ == "__main__":
    main()
