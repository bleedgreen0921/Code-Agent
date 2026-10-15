"""Export or validate a local standard tools/list snapshot (administrator only)."""

import argparse
import asyncio
import json
from pathlib import Path

from agent_service.mcp_config import contract, read_catalog
from agent_service.profiles import Registry


async def export(server, registry):
    from agent_service.mcp_runtime import make_client
    from langchain.mcp import MCPAdapter

    async with asyncio.timeout(registry.mcp.init_timeout):
        async with MCPAdapter(
            make_client(server, registry.mcp, (16384, 200))
        ) as adapter:
            tools = await adapter.client.list_tools(cache_mode="bypass")
            return {"tools": [contract(t) for t in tools]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("path", type=Path)
    out = sub.add_parser("export")
    out.add_argument("server_id")
    out.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        catalog = read_catalog(str(args.path.resolve()))
        from agent_service.schema import local_schema

        for tool in catalog.values():
            local_schema(tool["inputSchema"])
            if tool.get("outputSchema") is not None:
                local_schema(tool["outputSchema"])
        print(json.dumps({"valid": True, "tools": sorted(catalog)}))
    else:
        # Server configuration must already contain an initial {"tools": []}
        # catalog with no selected tools; exporting never enables tools itself.
        registry = Registry.from_env()
        server = next((s for s in registry.mcp.servers if s.id == args.server_id), None)
        if server is None:
            parser.error("Unknown configured server")
        payload = json.dumps(
            asyncio.run(export(server, registry)), ensure_ascii=False, indent=2
        )
        if len(payload.encode()) > 1024 * 1024:
            raise ValueError("Catalog exceeds 1 MiB")
        with args.output.open("x", encoding="utf-8") as file:
            file.write(payload + "\n")
        print(json.dumps({"exported": str(args.output)}))


if __name__ == "__main__":
    main()
