"""Independent writable repair scope composed with existing snapshot/child services."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import psycopg
from langchain_core.tools import ToolException

from agent_service.code_artifacts import CodeFailure
from agent_service.code_execution import ChildStore, CodeExecutionScope
from agent_service.execution import ExecutionScope, cleanup_resource
from agent_service.execution_errors import (
    CleanupFailed,
    CommandTimeout,
    EnvironmentLost,
)
from agent_service.profiles import fingerprint
from agent_service.project_checks import classify_result
from agent_service.repair_versions import (
    baseline_archive,
    copy_manifest,
    freeze,
    source_manifest,
    verify_revision,
)


class RepairExecutionScope(ExecutionScope):
    """Never mounts the original or baseline as a writable model workspace."""

    fixed_env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    prepare_snapshot = CodeExecutionScope.prepare_snapshot
    child = CodeExecutionScope.child

    def __init__(self, store, row, settings, registry):
        super().__init__(store, row, settings)
        self.registry, self.code = registry, registry.repair
        self.children = []
        self.manifest = self.evidence = self.snapshot_root = None
        self.candidate = self.current = self.document = None
        self.write_closed = False
        self.rounds = 0
        self.checks = {}
        self.tool_results = {}
        self.check_lock = self.lock

    async def managed_io(self, operation, *args):
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        self.native_tasks.add(task)

        def settled(done):
            self.native_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(settled)
        return await asyncio.shield(task)

    async def prepare(self):
        from agent_service.repair_reports import RepairLedger

        await self.prepare_snapshot()
        self.baseline_root = self.snapshot_root
        self.baseline = source_manifest(self.manifest["manifest"])
        self.contexts = await self.store.repair_context_manifest(
            self.row, self.manifest["content_digest"]
        )
        self.acceptance = self.code.select_acceptance(
            self.row["workspace_id"],
            self.row["_code"]["request"].get("acceptance_suite_id"),
        )
        identity = uuid4()
        directory = self.store.artifact_storage.path("candidates/" + identity.hex)
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        directory.mkdir(mode=0o700)
        limits = self.code.settings.model_dump()
        baseline_package = await self.managed_io(
            baseline_archive,
            self.baseline_root,
            self.manifest["manifest"],
            limits["snapshot_bytes"],
        )
        self.baseline_artifact = await self.store.put_code_artifact(
            self.row,
            "repair_baseline",
            baseline_package,
            metadata={
                "content_digest": self.manifest["content_digest"],
                "git_digest": self.manifest["git_digest"],
            },
            maximum=self.code.settings.storage_bytes,
        )
        # Baseline, independent Git copy and writable content are charged on admission.
        admitted = sum(i["size"] for i in self.manifest["manifest"].values()) * 2 + len(
            baseline_package
        )
        if admitted > self.code.settings.storage_bytes:
            raise CodeFailure(
                "ARTIFACT_LIMIT",
                "Baseline and writable candidate exceed storage budget",
            )
        self.candidate = await self.store.create_repair_candidate(
            self.row, identity, self.acceptance.manifest(), admitted
        )
        self.candidate_root = directory / "source"
        await self.managed_io(
            copy_manifest,
            self.baseline_root,
            self.candidate_root,
            self.baseline,
            limits["file_bytes"],
        )
        git_manifest = {
            p.removeprefix(".git/"): info
            for p, info in self.manifest["manifest"].items()
            if p.startswith(".git/")
        }
        await self.managed_io(
            copy_manifest,
            self.baseline_root / ".git",
            directory / "git",
            git_manifest,
            limits["snapshot_bytes"],
        )
        (self.candidate_root / ".git").mkdir()
        self.extra_mounts = (
            f"type=bind,source={directory / 'git'},target=/workspace/.git,readonly",
        )
        self.snapshot_root = self.candidate_root
        self.evidence = RepairLedger(self.store, self.row, self.manifest, self)
        self.row["_code_scope"] = self
        await self.checkpoint("baseline")
        # Initial acceptance is controlled, version-bound and counts against the same budget.
        await self.run_check(
            self.acceptance, "baseline:acceptance", self.current, purpose="baseline"
        )
        mounted = await self.store.execution_resource(self.row["id"])
        if (
            not mounted
            or mounted["execution_root"] != str(self.candidate_root)
            or mounted["candidate_id"] != self.candidate["id"]
            or mounted["snapshot_id"] != self.row["_snapshot_id"]
        ):
            raise CodeFailure(
                "SNAPSHOT_INVALID",
                "Writable candidate mount was not bound; fallback is forbidden",
            )
        await ExecutionScope.prepare(self)
        await self.store.code_phase(self.row, "repair")

    async def quiescent(self):
        if not self.container_id:
            return
        await self.healthy()
        _, raw, _ = await self.client.call(
            "exec",
            "-i",
            self.container_id,
            "/usr/local/bin/python",
            "-I",
            "-B",
            "/opt/agent/repair_helper.py",
            timeout=min(self.settings.command_timeout, self.remaining()),
        )
        if json.loads(raw)["running_pids"]:
            raise ToolException(
                "Candidate has live processes. Stop all background processes before checkpoint or acceptance."
            )

    async def checkpoint(self, call_id):
        if self.write_closed:
            raise CodeFailure("PERMISSION_DENIED", "Candidate writes have closed")
        await self.quiescent()
        identity = uuid4()
        key = "revisions/" + identity.hex
        directory = self.store.artifact_storage.path(key)
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        directory.mkdir(mode=0o700)
        try:
            document, patch, package = await self.managed_io(
                freeze,
                self.candidate_root,
                directory / "source",
                self.baseline_root,
                self.baseline,
                self.code.settings.model_dump(),
            )
            await self.quiescent()
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        encoded = json.dumps(document, ensure_ascii=False, sort_keys=True).encode()
        size = (
            sum(v["size"] for v in document["manifest"].values())
            + len(encoded)
            + len(patch)
            + len(package)
        )
        # Check before writing artifacts; save_repair_revision repeats this check transactionally.
        if self.candidate["admitted_bytes"] + size > self.code.settings.storage_bytes:
            raise CodeFailure(
                "ARTIFACT_LIMIT", "Candidate storage admission budget exhausted"
            )
        ids = {}
        for kind, data in (
            ("manifest", encoded),
            ("patch", patch),
            ("package", package),
        ):
            item = await self.store.put_code_artifact(
                self.row,
                "repair_" + kind,
                data,
                metadata={
                    "candidate_id": str(self.candidate["id"]),
                    "revision_id": str(identity),
                    "content_digest": document["content_digest"],
                },
                maximum=self.code.settings.storage_bytes,
            )
            ids[kind + "_artifact_id"] = item["id"]
        revision = dict(
            id=identity,
            task_id=self.row["task_id"],
            run_id=self.row["id"],
            candidate_id=self.candidate["id"],
            tool_call_id=call_id,
            storage_key=key,
            content_digest=document["content_digest"],
            size_bytes=size,
            **ids,
        )
        # Immutable registration may be retried; the write command is never replayed.
        for attempt in range(3):
            try:
                self.current = await self.store.save_repair_revision(self.row, revision)
                break
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise
        self.candidate["admitted_bytes"] += size
        self.candidate["current_revision_id"] = identity
        self.document = document
        return self.current

    async def file_call(self, name, args):
        if self.write_closed:
            raise CodeFailure("PERMISSION_DENIED", "Candidate control entry is closed")
        result = await ExecutionScope.file_call(self, name, args)
        if name == "read_file" and "error" not in result:
            path = args["path"].removeprefix("/workspace/").lstrip("/")
            path = "/".join(p for p in path.split("/") if p != ".")
            self.evidence.manifest = {
                **self.manifest,
                "manifest": {path: {"sha256": result["sha256"]}},
            }
        return result

    async def shell(self, command=None, restart=None):
        self.check_open()
        if self.write_closed:
            raise CodeFailure("PERMISSION_DENIED", "Candidate writes have closed")
        if restart or not command or "\x00" in command:
            raise ToolException(
                "Provide a foreground command. Shell restart is disabled."
            )
        try:
            code, out, err = await self.client.call(
                "exec",
                "-i",
                self.container_id,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                command,
                timeout=min(self.remaining(), self.settings.command_timeout),
                maximum=self.settings.output_bytes,
                check=False,
            )
        except TimeoutError as exc:
            self.failure = CommandTimeout("Candidate Shell deadline expired")
            raise self.failure from exc
        if code == 137:
            raise EnvironmentLost("Candidate process was killed")
        await self.healthy()
        from agent_service.output import limit_output

        content, truncated = limit_output(
            (out + err).decode("utf-8", "replace"),
            self.settings.output_bytes,
            self.settings.output_lines,
        )
        return content, {
            "exit_code": code,
            "truncated": truncated,
            "development_feedback": True,
        }

    async def run(self, check_id, call_id):
        if check_id not in self.row["_code"]["check_ids"]:
            raise ToolException("Unknown or unauthorized project check ID")
        await self.quiescent()
        return await self.run_check(
            self.code.checks[check_id], "tool:" + call_id, self.current
        )

    async def run_check(
        self, definition, call_id, revision, purpose="development", round_deadline=None
    ):
        if self.remaining() < 1:
            raise CodeFailure(
                "REPAIR_BUDGET_EXHAUSTED", "Insufficient time to start a check"
            )
        settings = self.code.settings
        root = self.store.artifact_storage.path(revision["storage_key"]) / "source"
        _, data = await self.store.code_artifact(
            self.row["task_id"], revision["manifest_artifact_id"]
        )
        document = json.loads(data)
        await self.managed_io(
            verify_revision, root, document, self.baseline, settings.model_dump()
        )
        acceptance = purpose in ("acceptance", "baseline")
        record, created = await self.store.reserve_repair_check(
            self.row,
            self.candidate["id"],
            revision,
            call_id,
            definition.manifest(),
            purpose,
            definition.asset_digest if acceptance else None,
        )
        if not created:
            if record["result"] is None:
                raise CodeFailure(
                    "CHECK_RESULT_UNKNOWN", "Interrupted check will not be replayed"
                )
            return self.remember(record)
        resource = await self.store.code_resource(record["resource_id"])
        if (
            resource["revision_id"] != revision["id"]
            or resource["candidate_id"] != self.candidate["id"]
            or resource["execution_root"] != str(root)
        ):
            raise CodeFailure("SNAPSHOT_INVALID", "Check mount identity differs")
        deadline = min(
            self.row["effective_deadline_at"],
            datetime.now(timezone.utc)
            + timedelta(seconds=min(definition.timeout, settings.check_seconds)),
            round_deadline or self.row["effective_deadline_at"],
        )
        if deadline <= datetime.now(timezone.utc):
            raise CodeFailure(
                "REPAIR_BUDGET_EXHAUSTED", "Candidate check round deadline expired"
            )
        scope = await self.child("check", root, resource=resource, deadline=deadline)
        raw, logs, code, timed_out, oom = b"", b"", None, False, False
        try:
            await scope.prepare()
            request = (
                definition.manifest()
                if acceptance
                else self.code.adapters[definition.adapter_id].request(definition)
            )
            code, raw, logs = await scope.client.call(
                "exec",
                "-i",
                scope.container_id,
                "/usr/local/bin/python",
                "-I",
                "-B",
                "/opt/agent/acceptance_runner.py"
                if acceptance
                else "/opt/agent/project_runner.py",
                stdin=json.dumps(request).encode(),
                timeout=max(
                    0.01, (deadline - datetime.now(timezone.utc)).total_seconds()
                ),
                maximum=max(settings.result_bytes, settings.log_bytes) + 1,
                check=False,
                timeout_result=True,
            )
            timed_out = code == 124
            info = await scope.client.inspect(scope.container_id)
            if not info:
                raise EnvironmentLost("Check container disappeared")
            oom = bool(info["State"].get("OOMKilled")) or code == 137
            if not info["State"]["Running"] and not oom:
                if datetime.now(timezone.utc) >= deadline:
                    timed_out = True
                else:
                    raise EnvironmentLost("Check container unexpectedly stopped")
        finally:
            if not await scope.close():
                raise CleanupFailed("Candidate check cleanup unconfirmed")
        protocol = raw
        control_error = False
        if acceptance and not timed_out and not oom:
            try:
                envelope = json.loads(raw)
                if (
                    set(envelope) != {"asset_digest", "suite_id", "version", "result"}
                    or envelope["asset_digest"] != definition.asset_digest
                    or envelope["suite_id"] != definition.id
                    or envelope["version"] != definition.version
                ):
                    raise ValueError("Acceptance identity/protocol differs")
                protocol = json.dumps(envelope["result"]).encode()
            except (ValueError, TypeError, KeyError):
                control_error = True
        result = classify_result(
            protocol[: settings.result_bytes],
            exit_code=code,
            adapter=definition.runner if acceptance else definition.adapter_id,
            expected_cases=definition.required_cases
            if acceptance
            else definition.expected_cases,
            timed_out=timed_out,
            oom=oom,
            truncated=len(raw) > settings.result_bytes,
        )
        if acceptance and not timed_out and not oom:
            # Assertion/import errors are feedback; damaged inventory or protocol is a control fault.
            if any(
                x in result["limitations"]
                for x in (
                    "invalid_protocol",
                    "incomplete_protocol",
                    "missing_required_cases",
                    "incomplete_execution",
                    "missing_outcomes",
                    "multiple_outcomes",
                    "unselected_events",
                    "exit_protocol_mismatch",
                    "result_limit",
                )
            ):
                control_error = True
            if result["expected_failures"] or result["skipped"]:
                result.update(status="inconclusive", complete=False)
            if result["selected"] != len(definition.required_cases) or result[
                "executed"
            ] != len(definition.required_cases):
                control_error = True
        result["log_truncated"] = len(logs) > settings.log_bytes
        log = await self.store.put_code_artifact(
            self.row,
            "check_log",
            logs[: settings.log_bytes],
            metadata={
                "check_run_id": str(record["id"]),
                "revision_id": str(revision["id"]),
            },
            maximum=settings.log_bytes,
        )
        saved = await self.store.put_code_artifact(
            self.row,
            "check_result",
            json.dumps(
                {
                    "summary": result,
                    "protocol": protocol[: settings.result_bytes].decode(
                        "utf-8", "replace"
                    ),
                }
            ).encode(),
            maximum=settings.result_bytes * 6 + 65536,
        )
        for attempt in range(3):
            try:
                record = await self.store.finish_code_check(
                    record["id"], result, log["id"], saved["id"]
                )
                break
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise
        value = self.remember(record)
        if control_error:
            raise CodeFailure(
                "ACCEPTANCE_INVALID",
                "Independent acceptance asset or protocol is damaged",
            )
        if self.container_id:
            await self.healthy()
        return value

    def remember(self, record):
        value = {
            k: str(record[k]) if record[k] is not None else None
            for k in (
                "id",
                "revision_id",
                "candidate_id",
                "snapshot_id",
                "log_artifact_id",
                "result_artifact_id",
            )
        }
        value.update(
            check_id=record["check_id"],
            purpose=record["purpose"],
            result=record["result"],
            result_digest=record["result_digest"],
            definition_digest=fingerprint(record["definition"]),
            acceptance_digest=record["acceptance_digest"],
        )
        self.checks[value["id"]] = value
        self.evidence.checks[value["id"]] = value
        return value

    async def candidate_diff(self, view="files", offset=0, limit=50):
        if view == "files":
            lines = self.document["changes"]
        else:
            _, data = await self.store.code_artifact(
                self.row["task_id"], self.current["patch_artifact_id"]
            )
            lines = data.decode("utf-8", "replace").splitlines()
        return self.page(
            lines,
            offset,
            limit,
            revision_id=str(self.current["id"]),
            content_digest=self.current["content_digest"],
        )

    def page(self, records, offset, limit, **meta):
        kept = []
        for record in records[offset : offset + limit]:
            candidate = json.dumps(
                dict(
                    records=[*kept, record], next_offset=offset + len(kept) + 1, **meta
                ),
                ensure_ascii=False,
            )
            if len(candidate.encode()) > self.settings.output_bytes - 256:
                break
            kept.append(record)
        result = dict(
            records=kept,
            next_offset=offset + len(kept)
            if offset + len(kept) < len(records)
            else None,
            **meta,
        )
        return json.dumps(result, ensure_ascii=False), {
            **meta,
            "next_offset": result["next_offset"],
        }

    async def read_repair_execution(
        self, execution_id, view="result", offset=0, limit=50
    ):
        if execution_id not in self.checks:
            raise ToolException("Completed check in this repair Run required")
        record = self.checks[execution_id]
        artifact = (
            record["log_artifact_id"] if view == "log" else record["result_artifact_id"]
        )
        _, data = await self.store.code_artifact(self.row["task_id"], UUID(artifact))
        if view == "log":
            records = data.decode("utf-8", "replace").splitlines()
        else:
            value = json.loads(data)
            try:
                events = json.loads(value["protocol"]).get("events", [])
            except ValueError:
                events = []
            records = [record, *events]
        return self.page(
            records,
            offset,
            limit,
            execution_id=execution_id,
            revision_id=record["revision_id"],
        )

    async def read_repair_context(self, task_id, offset=0, limit=50):
        if task_id not in self.row["_code"]["request"]["context_task_ids"]:
            raise ToolException("Context task was not authorized in the request")
        metadata = await self.store.code_metadata(UUID(task_id))
        if (
            not metadata
            or metadata["stage"] not in ("analysis", "review")
            or not metadata["report_artifact_id"]
        ):
            raise ToolException("Formal historical report unavailable")
        artifact, data = await self.store.code_artifact(
            UUID(task_id), metadata["report_artifact_id"]
        )
        if artifact["visibility"] != "formal":
            raise ToolException("Only formal context reports may be read")
        text = data.decode("utf-8")
        records = [text[i : i + 1024] for i in range(0, len(text), 1024)]
        return self.page(
            records,
            offset,
            limit,
            context_task_id=task_id,
            notice="Historical evidence is not inherited; reconfirm against current baseline.",
            baseline_digest=self.manifest["content_digest"],
        )

    async def close(self, deadline=None):
        child_clean = True
        for child in reversed(self.children):
            child_clean = await child.close(deadline) and child_clean
        for resource in await self.store.code_resources(run_id=self.row["id"]):
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
        return await ExecutionScope.close(self, deadline) and child_clean
