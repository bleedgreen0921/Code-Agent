"""Snapshot preparation and independent check containers share the Run resource journal."""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

import psycopg
from langchain_core.tools import ToolException

from agent_service.code_artifacts import CodeFailure
from agent_service.code_registry import VERSION, asset_digest
from agent_service.code_reports import EvidenceLedger
from agent_service.execution import ExecutionScope, recover_resources
from agent_service.execution_errors import CleanupFailed, EnvironmentUnavailable
from agent_service.profiles import fingerprint


class ChildStore:
    """Adapt the existing identity/cleanup protocol to a specific child journal row."""

    def __init__(self, store, identity):
        self.store, self.identity = store, identity

    async def execution_resource(self, run_id):
        return await self.store.code_resource(self.identity)

    async def resource_update(self, run_id, token, **changes):
        return await self.store.code_resource_update(self.identity, token, **changes)

    async def orphan_resources(self):
        return [
            r
            for r in await self.store.code_resources(orphans=True)
            if r["id"] == self.identity
        ]


async def preflight(client, settings):
    _, raw, _ = await client.call("image", "inspect", settings.image)
    labels = json.loads(raw)[0].get("Config", {}).get("Labels", {}) or {}
    if (
        labels.get("io.agent-framework.code-analysis") != VERSION
        or labels.get("io.agent-framework.code-assets") != asset_digest()
    ):
        raise EnvironmentUnavailable(
            "Code analysis image runner/assets version mismatch"
        )


def verify_snapshot(root, manifest, limits):
    # Host reads only service-managed files. No project modules or Git commands run here.
    from execution.code_snapshot import digest, scan

    found = scan(root, limits)
    if found != manifest["manifest"]:
        raise CodeFailure(
            "SNAPSHOT_INVALID", "Sealed snapshot is missing or its digest changed"
        )
    if (
        digest({k: v for k, v in found.items() if not k.startswith(".git/")})
        != manifest["content_digest"]
        or digest({k: v for k, v in found.items() if k.startswith(".git/")})
        != manifest["git_digest"]
    ):
        raise CodeFailure(
            "SNAPSHOT_INVALID", "Sealed snapshot manifest is inconsistent"
        )


class CodeExecutionScope(ExecutionScope):
    force_readonly = True
    fixed_env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    def __init__(self, store, row, settings, registry):
        super().__init__(store, row, settings)
        self.registry = registry
        self.children = []
        self.manifest = None
        self.snapshot_root = None
        self.evidence = None
        self.check_lock = asyncio.Lock()

    async def child(self, purpose, root, *, resource=None, deadline=None):
        self.check_open()
        if resource is None:
            async with self.store.locked() as conn:
                await self.store.code_live(conn, self.row)
                resource = await self.store.reserve_code_resource(
                    conn, self.row, purpose, root
                )
        child_row = {
            **self.row,
            "execution_tools": (),
            "effective_deadline_at": deadline or self.row["effective_deadline_at"],
        }
        scope = ExecutionScope(
            ChildStore(self.store, resource["id"]), child_row, self.settings
        )
        scope.force_readonly = True
        scope.fixed_env = self.fixed_env
        self.children.append(scope)
        return scope

    async def prepare(self):
        if not self.row.get("task_id"):
            raise CodeFailure(
                "CODE_TASK_REQUIRED", "Code profile requires a dedicated task"
            )
        await preflight(self.client, self.settings)
        metadata = self.row["_code"] = await self.store.code_metadata(
            self.row["task_id"]
        )
        if (
            not metadata
            or metadata["policy_fingerprint"]
            != self.row["submitted_binding"]["fingerprint"]
        ):
            raise CodeFailure(
                "CONFIGURATION_CHANGED", "Code task policy binding changed"
            )
        limits = self.registry.code.settings.model_dump()
        sid, directory = await self.store.reserve_code_snapshot(
            self.row, metadata["request"]["target"]
        )
        self.row["_snapshot_id"] = sid
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        directory.mkdir(mode=0o700)
        # API/Worker and execution user share a host; only this designated output
        # directory is writable by the trusted preparer, never the storage root.
        uid, gid = map(int, self.settings.effective_user().split(":"))
        if os.getuid() == 0:
            os.chown(directory, uid, gid)
        elif uid != os.getuid():
            raise EnvironmentUnavailable(
                "Snapshot preparation requires matching host execution UID"
            )
        original = self.registry.workspaces[self.row["workspace_id"]]
        end = min(
            self.row["effective_deadline_at"],
            datetime.now(timezone.utc) + timedelta(seconds=limits["snapshot_seconds"]),
        )
        preparer = await self.child("snapshot", original, deadline=end)
        preparer.extra_mounts = (f"type=bind,source={directory},target=/snapshot",)
        try:
            async with asyncio.timeout(
                max(0, (end - datetime.now(timezone.utc)).total_seconds())
            ):
                await preparer.prepare()
                code, out, _ = await preparer.client.call(
                    "exec",
                    "-i",
                    preparer.container_id,
                    "/usr/local/bin/python",
                    "-I",
                    "/opt/agent/code_snapshot.py",
                    stdin=json.dumps(
                        {"target": metadata["request"]["target"], "limits": limits}
                    ).encode(),
                    timeout=max(
                        0.01, (end - datetime.now(timezone.utc)).total_seconds()
                    ),
                )
                response = json.loads(out)
                if code or not response.get("ok"):
                    raise CodeFailure(
                        response.get("code", "SNAPSHOT_INVALID"),
                        response.get("error", "Snapshot preparation failed"),
                    )
        except TimeoutError as exc:
            raise CodeFailure(
                "SNAPSHOT_UNSTABLE", "Snapshot preparation deadline expired"
            ) from exc
        finally:
            clean = await preparer.close()
            if not clean:
                raise CleanupFailed("Snapshot preparer cleanup unconfirmed")
        raw = (directory / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        self.snapshot_root = directory / "source"
        await asyncio.to_thread(
            verify_snapshot, self.snapshot_root, self.manifest, limits
        )
        await self.store.seal_code_snapshot(
            self.row, self.manifest, self.snapshot_root, hashlib.sha256(raw).hexdigest()
        )
        self.evidence = EvidenceLedger(self.store, self.row, self.manifest)
        self.row["_code_scope"] = self
        mounted = await self.store.execution_resource(self.row["id"])
        if (
            not mounted
            or mounted.get("execution_root") != str(self.snapshot_root)
            or mounted.get("snapshot_id") != sid
        ):
            raise CodeFailure(
                "SNAPSHOT_INVALID",
                "Sealed execution root was not bound; original workspace fallback is forbidden",
            )
        await super().prepare()

    async def file_call(self, name, args):
        self.check_open()
        if name in ("edit_file", "shell"):
            raise CodeFailure("PERMISSION_DENIED", "Analysis is read only")
        # Validate relevant working file digest after read in the ledger; Git metadata
        # is fixed and the full snapshot is verified before every execution mount.
        await asyncio.to_thread(
            verify_snapshot,
            self.snapshot_root,
            self.manifest,
            self.registry.code.settings.model_dump(),
        )
        result = await super().file_call(name, args)
        if name.startswith("git_") and "content" in result:
            result["content"] = (
                result["content"]
                .replace(
                    "No atomic snapshot; submodule interiors excluded",
                    "Sealed code snapshot; submodules unsupported",
                )
                .replace('"observation":"live"', '"observation":"sealed"')
            )
        return result

    async def run(self, check_id, call_id):
        if check_id not in self.row["_code"]["check_ids"]:
            raise ToolException("Unknown or unauthorized project check ID")
        async with self.check_lock:
            self.check_open()
            if self.remaining() < 1:
                raise ToolException("Insufficient remaining analysis time for a check")
            check = self.registry.code.checks[check_id]
            # Only a sealed verified root can be mounted for project execution.
            await asyncio.to_thread(
                verify_snapshot,
                self.snapshot_root,
                self.manifest,
                self.registry.code.settings.model_dump(),
            )
            record, created = await self.store.reserve_code_check(
                self.row, call_id, check, self.snapshot_root
            )
            if record is None:
                raise ToolException("Registered check execution budget exhausted")
            if not created:
                if record["result"] is None:
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN",
                        "Original check execution requires reconciliation; it will not be replayed",
                    )
                return self.remember(record)
            resource = await self.store.code_resource(record["resource_id"])
            if (
                not resource
                or resource.get("execution_root") != str(self.snapshot_root)
                or resource.get("snapshot_id") != self.row["_snapshot_id"]
            ):
                raise CodeFailure(
                    "SNAPSHOT_INVALID",
                    "Check resource does not refer to the sealed snapshot",
                )
            seconds = min(
                check.timeout,
                self.registry.code.settings.check_seconds,
                self.remaining(),
            )
            deadline = min(
                self.row["effective_deadline_at"],
                datetime.now(timezone.utc) + timedelta(seconds=seconds),
            )
            scope = await self.child(
                "check", self.snapshot_root, resource=resource, deadline=deadline
            )
            adapter = self.registry.code.adapters[check.adapter_id]
            raw, logs, exit_code, timed_out, oom = b"", b"", None, False, False
            try:
                await scope.prepare()
                try:
                    exit_code, raw, logs = await scope.client.call(
                        "exec",
                        "-i",
                        scope.container_id,
                        "/usr/local/bin/python",
                        "-I",
                        "-B",
                        "/opt/agent/project_runner.py",
                        stdin=json.dumps(adapter.request(check)).encode(),
                        timeout=max(
                            0.01,
                            (deadline - datetime.now(timezone.utc)).total_seconds(),
                        ),
                        maximum=max(
                            self.registry.code.settings.result_bytes,
                            self.registry.code.settings.log_bytes,
                        )
                        + 1,
                        check=False,
                        timeout_result=True,
                    )
                    timed_out = exit_code == 124
                except TimeoutError:
                    timed_out = True
                info = await scope.client.inspect(scope.container_id)
                if info is None:
                    raise EnvironmentUnavailable(
                        "Check container disappeared before result reconciliation"
                    )
                oom = bool(info["State"].get("OOMKilled"))
                if not timed_out and not info["State"]["Running"] and not oom:
                    timed_out = datetime.now(timezone.utc) >= deadline
                    if not timed_out:
                        raise EnvironmentUnavailable(
                            "Check execution container stopped unexpectedly"
                        )
            finally:
                if not await scope.close():
                    raise CleanupFailed("Check container cleanup unconfirmed")
            settings = self.registry.code.settings
            result = adapter.parse(
                raw[: settings.result_bytes],
                exit_code=exit_code,
                expected_cases=check.expected_cases,
                adapter=check.adapter_id,
                timed_out=timed_out,
                oom=oom,
                truncated=len(raw) > settings.result_bytes,
            )
            result["log_truncated"] = len(logs) > settings.log_bytes
            log_artifact = await self.store.put_code_artifact(
                self.row,
                "check_log",
                logs[: settings.log_bytes],
                metadata={
                    "check_run_id": str(record["id"]),
                    "truncated": result["log_truncated"],
                },
                maximum=settings.log_bytes,
            )
            result_artifact = await self.store.put_code_artifact(
                self.row,
                "check_result",
                json.dumps(
                    {
                        "summary": result,
                        "protocol": raw[: settings.result_bytes].decode(
                            "utf-8", "replace"
                        ),
                    },
                    ensure_ascii=False,
                ).encode(),
                maximum=settings.result_bytes * 6 + 65536,
            )
            # If commit confirmation is lost, retry only this immutable result write.
            for attempt in range(3):
                try:
                    completed = await self.store.finish_code_check(
                        record["id"], result, log_artifact["id"], result_artifact["id"]
                    )
                    break
                except (psycopg.Error, OSError, TimeoutError):
                    if attempt == 2:
                        raise
            return self.remember(completed)

    def remember(self, record):
        value = {
            "id": str(record["id"]),
            "check_id": record["check_id"],
            "snapshot_id": str(record["snapshot_id"]),
            "result": record["result"],
            "result_digest": record["result_digest"],
            "log_artifact_id": str(record["log_artifact_id"])
            if record["log_artifact_id"]
            else None,
        }
        self.evidence.checks[str(record["id"])] = value
        return value

    async def close(self, deadline=None):
        child_clean = True
        for child in reversed(self.children):
            child_clean = await child.close(deadline) and child_clean
        # Include resources whose reservation committed but acknowledgement was lost.
        for resource in await self.store.code_resources(run_id=self.row["id"]):
            from agent_service.execution import cleanup_resource

            adapter = ChildStore(self.store, resource["id"])
            try:
                async with asyncio.timeout(self.settings.cleanup_timeout):
                    await cleanup_resource(adapter, resource, self.client)
                    await adapter.resource_update(
                        resource["run_id"],
                        resource["lease_token"],
                        state="released",
                        cleanup_error=None,
                    )
            except Exception:
                child_clean = False
        clean = await super().close(deadline)
        return clean and child_clean


async def recover_code_resources(store, settings):
    for resource in await store.code_resources(orphans=True):
        await recover_resources(ChildStore(store, resource["id"]), settings)
    async with store.locked() as conn:
        # Interrupted checks are diagnostics, never pass and never restart.
        interrupted = {
            "status": "inconclusive",
            "selected": 0,
            "executed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "limitations": ["interrupted_execution"],
        }
        from psycopg.types.json import Jsonb

        await conn.execute(
            """UPDATE agent.code_check_runs c SET status='inconclusive',result=%s,
            result_digest=%s,finished_at=clock_timestamp() FROM agent.runs r
            WHERE c.run_id=r.id AND c.status='running' AND
            (r.status IN ('failed','cancelled','completed') OR r.lease_until<=clock_timestamp())""",
            (Jsonb(interrupted), fingerprint(interrupted)),
        )
