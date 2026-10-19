"""Trusted host delivery management; no force, accept-current or unlock action."""

import argparse
import asyncio
import json
from uuid import UUID

from agent_service.delivery_recovery import inspect_delivery, recover_one
from agent_service.resources import Resources


async def run(action, task_id):
    async with Resources() as resources:
        if action == "inspect":
            return await inspect_delivery(resources, task_id)
        detail = await resources.store.delivery_detail(task_id)
        if not detail["delivery"]:
            raise ValueError("No automatic delivery for this task")
        try:
            await recover_one(
                resources, detail["delivery"]["run_id"], rollback=action == "rollback"
            )
        except BlockingIOError:
            detail = await inspect_delivery(resources, task_id)
            detail["busy"] = (
                "A live process still holds the workspace flock; no takeover performed."
            )
            return detail
        return await inspect_delivery(resources, task_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "recover", "rollback"))
    parser.add_argument("task_id", type=UUID)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.action, args.task_id)),
            default=str,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
