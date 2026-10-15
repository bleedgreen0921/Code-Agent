import argparse
import asyncio
import json
from uuid import UUID

from agent_service.resources import Resources
from agent_service.trace import encode_trace, trace_config


async def inspect(run_id, namespace=None):
    async with Resources() as resources:
        row = await resources.store.get(run_id)
        history = [
            {
                "config": cp.config,
                "metadata": cp.metadata,
                "checkpoint": cp.checkpoint,
                "pending_writes": cp.pending_writes,
            }
            async for cp in resources.checkpointer.alist(
                trace_config(run_id, namespace)
            )
        ]
        return {
            "run_id": str(run_id),
            "status": row["status"],
            "submitted_binding": row["submitted_binding"],
            "runtime_config": row["runtime_config"],
            "final_checkpoint": row["final_checkpoint"],
            "external_operations": await resources.store.external_operations(run_id),
            "history": history,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=UUID)
    parser.add_argument(
        "--namespace", default=None, help="Exact namespace; empty string selects root"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            encode_trace(asyncio.run(inspect(args.run_id, args.namespace))),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
