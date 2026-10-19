"""Journalled host delivery and bounded recovery; no graph/model/check execution."""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

from agent_service.code_artifacts import CodeFailure
from agent_service.delivery_fs import (
    DeliveryConflict,
    FileTransaction,
    private_lock,
)
from agent_service.delivery_merge import current_context
from agent_service.profiles import fingerprint


class DeliveryController:
    def __init__(self, store, row, private, limits, verify_evidence):
        self.store, self.row, self.private, self.limits = (
            store,
            row,
            Path(private),
            limits,
        )
        self.verify_evidence = verify_evidence
        self.token = row["lease_token"]
        self.transaction = None

    async def io(self, operation, *args):
        # Never release flock while a filesystem mutation still runs in a thread.
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def control(self, forward=False):
        return await self.store.delivery_control(
            self.row["id"], self.token, forward=forward
        )

    async def journal(self, phase, detail, **kwargs):
        return await self.store.delivery_journal(
            self.row["id"], self.token, phase, detail, **kwargs
        )

    async def context(self, manifest, plan):
        found, git = await self.io(
            current_context,
            Path(self.row["workspace_root"]),
            self.limits,
            plan["identity"],
            self.private,
            manifest,
        )
        if found != manifest or git != plan["git_state"]:
            raise DeliveryConflict(
                "Original source inventory or Git state differs from verified context"
            )

    async def run(self, rollback=False, recovery=False, already_locked=False):
        root = Path(self.row["workspace_root"])
        with nullcontext() if already_locked else private_lock(self.private, root):
            delivery = await self.store.acquire_delivery_lease(
                self.row["id"], self.token, recovery
            )
            if delivery["status"] == "applied":
                return delivery  # Immutable historical fact: never re-check later user edits.
            if delivery["status"] == "rolled_back":
                return delivery
            plan = delivery["plan"]
            if not plan or (
                delivery["status"] == "pending" and not delivery["checkpoint"]
            ):
                return await self.journal(
                    "rolled_back",
                    {"reason": "No file plan or forward operation"},
                    status="rolled_back",
                    impact="none",
                )
            transaction = None
            try:
                disk = await self.io(
                    lambda: json.loads(
                        (self.private / plan["id"] / "plan.json").read_bytes()
                    )
                )
                if fingerprint(plan) != delivery["plan_digest"] or disk != plan:
                    raise DeliveryConflict("Persistent plan identity differs")
                await self.verify_evidence(delivery)
                transaction = self.transaction = FileTransaction(
                    root, self.private, plan, self.limits["file_bytes"]
                )
                transaction.deadline_timestamp = (
                    delivery["deadline_at"].timestamp()
                    if delivery["deadline_at"]
                    else None
                )
                if (
                    rollback
                    or delivery["rollback_requested"]
                    or delivery["cancel_requested"]
                ):
                    return await self.rollback(plan)
                await self.control(forward=True)
                if delivery["status"] == "pending":
                    try:
                        await self.context(plan["before_manifest"], plan)
                        for operation in plan["files"]:
                            actual = await self.io(transaction.inspect, operation)
                            if transaction.classify(operation, actual) != "before":
                                raise DeliveryConflict(
                                    "Preflight file identity differs: "
                                    + operation["path"]
                                )
                    except (DeliveryConflict, OSError) as exc:
                        if recovery:
                            return await self.journal(
                                "rolled_back",
                                {"reason": str(exc), "forward_operations": False},
                                status="rolled_back",
                                impact="none",
                            )
                        # No mutations yet: only the live graph may capture/merge/
                        # revalidate, never the delivery-only recovery worker.
                        return dict(delivery, refresh_required=True, reason=str(exc))
                progress = dict(plan["before_manifest"])
                for operation in plan["files"]:
                    actual = await self.io(transaction.inspect, operation)
                    phase = transaction.classify(operation, actual)
                    if phase == "after":
                        if operation["after"] is None:
                            progress.pop(operation["path"], None)
                        else:
                            progress[operation["path"]] = operation["after"]
                    elif phase != "before":
                        raise DeliveryConflict(
                            "Recovery found a third file state: " + operation["path"]
                        )
                await self.context(progress, plan)
                await self.journal(
                    "applying", {"plan_id": plan["id"]}, status="applying"
                )
                for number, directory in enumerate(plan["directories"]):
                    await self.control(forward=True)
                    await self.journal(
                        "directory_intent",
                        directory,
                        status="applying",
                        impact="unknown",
                    )
                    await self.io(transaction.create_directory, directory)
                    await self.journal(
                        "directory_confirmed", directory, impact="partial"
                    )
                for number, operation in enumerate(plan["files"]):
                    await self.control(forward=True)
                    await self.context(progress, plan)
                    actual = await self.io(transaction.inspect, operation)
                    if transaction.classify(operation, actual) == "after":
                        await self.journal("confirmed", actual, ordinal=number)
                        continue
                    await self.journal(
                        "write_intent",
                        actual,
                        ordinal=number,
                        status="applying",
                        impact="unknown",
                    )
                    await self.control(forward=True)
                    actual = await self.io(transaction.mutate, operation)
                    # Fault injection tests kill here, after the real rename/fsync
                    # and before PostgreSQL acknowledges it.
                    await self.journal(
                        "confirmed", actual, ordinal=number, impact="partial"
                    )
                    if operation["after"] is None:
                        progress.pop(operation["path"], None)
                    else:
                        progress[operation["path"]] = operation["after"]
                await self.control(forward=True)
                await self.context(plan["after_manifest"], plan)
                for number, operation in enumerate(plan["files"]):
                    actual = await self.io(transaction.inspect, operation)
                    if transaction.classify(operation, actual) != "after":
                        raise DeliveryConflict(
                            "Final file identity differs: " + operation["path"]
                        )
                    await self.journal("confirmed", actual, ordinal=number)
                return await self.journal(
                    "applied",
                    {"plan_id": plan["id"], "final_digest": plan["final_digest"]},
                    status="applied",
                    impact="all" if plan["files"] or plan["directories"] else "none",
                )
            except asyncio.CancelledError:
                # Leave the durable intent and occupancy. Recovery obtains flock,
                # reads cancellation/deadlines, then safely rolls back or continues.
                raise
            except (DeliveryConflict, OSError, CodeFailure, ValueError) as exc:
                if transaction is None:
                    return await self.journal(
                        "needs_recovery",
                        {"reason": str(exc)},
                        status="needs_recovery",
                        impact="unknown",
                    )
                try:
                    await self.control()
                    return await self.rollback(plan, str(exc))
                except (DeliveryConflict, OSError, CodeFailure, ValueError) as failed:
                    return await self.journal(
                        "needs_recovery",
                        {"reason": str(exc), "rollback": str(failed)},
                        status="needs_recovery",
                        impact="unknown",
                    )
            finally:
                if transaction:
                    transaction.close()

    async def rollback(self, plan, reason="Rollback requested"):
        blocked = []
        impact = False
        for number in reversed(range(len(plan["files"]))):
            operation = plan["files"][number]
            await self.control()
            try:
                actual = await self.io(self.transaction.inspect, operation)
                phase = self.transaction.classify(operation, actual)
                if phase in ("before", "rolled_back"):
                    await self.journal("rollback_confirmed", actual, ordinal=number)
                    continue
                if phase != "after":
                    await self.journal("rollback_blocked", actual, ordinal=number)
                    blocked.append(operation["path"])
                    continue
                impact = True
                await self.journal("rollback_intent", actual, ordinal=number)
                actual = await self.io(self.transaction.mutate, operation, True)
                await self.journal("rollback_confirmed", actual, ordinal=number)
            except (DeliveryConflict, OSError) as exc:
                try:
                    observed = await self.io(self.transaction.inspect, operation)
                except (DeliveryConflict, OSError) as unreadable:
                    observed = {"unreadable": str(unreadable)}
                await self.journal(
                    "rollback_blocked",
                    {"reason": str(exc), "observed": observed},
                    ordinal=number,
                )
                blocked.append(dict(path=operation["path"], reason=str(exc)))
        for directory in reversed(plan["directories"]):
            await self.control()
            try:
                await self.journal("directory_rollback_intent", directory)
                await self.io(self.transaction.create_directory, directory, True)
                await self.journal("directory_rollback_confirmed", directory)
            except (DeliveryConflict, OSError) as exc:
                blocked.append(dict(path=directory["path"], reason=str(exc)))
        if blocked:
            return await self.journal(
                "needs_recovery",
                dict(reason=reason, blocked=blocked),
                status="needs_recovery",
                impact="unknown",
            )
        return await self.journal(
            "rolled_back",
            dict(reason=reason, writes_reverted=impact),
            status="rolled_back",
            impact="none",
        )
