"""Rolling P/A/U integration and the live graph's private delivery handoff."""

import json
import os
import stat
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from agent_service.code_artifacts import CodeFailure
from agent_service.delivery_fs import DeliveryConflict, durable_json
from agent_service.delivery_merge import (
    check_protection,
    current_context,
    merge_versions,
    observe,
    protection,
    update_protection,
)
from agent_service.profiles import fingerprint
from agent_service.repair_versions import (
    build_exports,
    freeze,
)
from execution.code_snapshot import digest


class DeliveryPipeline:
    def __init__(self, scope, record):
        self.scope, self.record = scope, record
        self.root = Path(record["workspace_root"])
        self.private = Path(
            scope.code.settings.auto_apply_workspaces[scope.row["workspace_id"]]
        )
        self.observation = self.integrated = self.integration = self.final = None
        self.protected = {}
        self.algorithm = scope.code.manifest["merge_algorithm"]
        self.refreshes = 0
        self.initial_identity = record["root_identity"]
        self.persistent_usage = {}
        self.plan = None
        self.handoffs = []
        self.force_refresh = False

    async def capture(self):
        scope = self.scope
        if self.observation:
            files, git = await scope.managed_io(
                current_context,
                self.root,
                scope.code.settings.model_dump(),
                self.initial_identity,
            )
            if (
                files == self.observation["document"]["source_manifest"]
                and git == self.observation["document"]["git_state"]
                and not self.force_refresh
            ):
                return self.observation
            if self.refreshes >= 3:
                raise CodeFailure(
                    "DELIVERY_REFRESH_EXHAUSTED",
                    "Original directory changed after three refreshes",
                )
        identity = uuid4()
        key = "observations/" + identity.hex
        directory = scope.store.artifact_storage.path(key)
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        directory.mkdir(mode=0o700)
        # Admit worst-case capture before any managed data grows.
        await scope.store.reserve_delivery_bytes(
            scope.row, scope.code.settings.snapshot_bytes
        )
        document = await scope.managed_io(
            observe,
            self.root,
            directory / "source",
            scope.code.settings.model_dump(),
            self.initial_identity,
        )
        await scope.managed_io(durable_json, directory / "manifest.json", document)
        size = sum(v["size"] for v in document["manifest"].values())
        await scope.store.settle_delivery_bytes(
            scope.row, scope.code.settings.snapshot_bytes, size
        )
        await scope.store.save_observation(
            scope.row, identity, key, document, size, self.observation is not None
        )
        if self.observation:
            self.refreshes += 1
        self.force_refresh = False
        self.observation = dict(id=identity, storage_key=key, document=document)
        return self.observation

    async def revision(self, source, call_id):
        scope = self.scope
        identity = uuid4()
        key = "revisions/" + identity.hex
        directory = scope.store.artifact_storage.path(key)
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        directory.mkdir(mode=0o700)
        document, patch, package = await scope.managed_io(
            freeze,
            source,
            directory / "source",
            scope.baseline_root,
            scope.baseline,
            scope.code.settings.model_dump(),
        )
        ids = {}
        encoded = json.dumps(document, sort_keys=True).encode()
        size = (
            sum(v["size"] for v in document["manifest"].values())
            + len(encoded)
            + len(patch)
            + len(package)
        )
        for kind, data in [
            ("manifest", encoded),
            ("patch", patch),
            ("package", package),
        ]:
            artifact = await scope.store.put_code_artifact(
                scope.row,
                "repair_" + kind,
                data,
                metadata={
                    "revision_id": str(identity),
                    "content_digest": document["content_digest"],
                },
                maximum=scope.code.settings.storage_bytes,
            )
            ids[kind + "_artifact_id"] = artifact["id"]
        record = await scope.store.save_repair_revision(
            scope.row,
            dict(
                id=identity,
                task_id=scope.row["task_id"],
                run_id=scope.row["id"],
                candidate_id=scope.candidate["id"],
                tool_call_id=call_id,
                storage_key=key,
                content_digest=document["content_digest"],
                size_bytes=size,
                **ids,
            ),
            update_current=False,
        )
        scope.candidate["admitted_bytes"] += size
        return record, document

    async def validate(self, agent, checks):
        scope = self.scope
        observation = await self.capture()
        u = observation["document"]["source_manifest"]
        p = (
            self.integrated["document"]["source_manifest"]
            if self.integrated
            else scope.baseline
        )
        base_root = (
            scope.store.artifact_storage.path(self.integrated["storage_key"]) / "source"
            if self.integrated
            else scope.baseline_root
        )
        agent_root = scope.store.artifact_storage.path(agent["storage_key"]) / "source"
        _, encoded = await scope.store.code_artifact(
            scope.row["task_id"], agent["manifest_artifact_id"]
        )
        a = json.loads(encoded)["manifest"]
        check_protection(
            self.protected,
            agent_root,
            a,
            u,
            p,
            scope.code.settings.file_bytes,
            scope.store.artifact_storage.path(observation["storage_key"]) / "source",
        )
        identity = uuid4()
        self.integration = dict(
            id=identity,
            delivery_id=self.record["id"],
            base_observation_id=self.integrated["id"] if self.integrated else None,
            observation_id=observation["id"],
            candidate_revision_id=agent["id"],
            status="preparing",
            evidence={
                "base_digest": digest(p),
                "agent_digest": digest(a),
                "user_digest": digest(u),
                "algorithm": self.algorithm,
                "image": scope.settings.image,
                "merged": p != u,
            },
        )
        delta = {
            key: str(agent[key])
            for key in (
                "manifest_artifact_id",
                "patch_artifact_id",
                "package_artifact_id",
            )
        }
        if p != scope.baseline:
            delta_doc, delta_patch, delta_package = await scope.managed_io(
                build_exports,
                agent_root,
                base_root,
                p,
                a,
                scope.code.settings.model_dump(),
            )
            parts = [
                ("manifest", json.dumps(delta_doc, sort_keys=True).encode()),
                ("patch", delta_patch),
                ("package", delta_package),
            ]
            await scope.store.reserve_delivery_bytes(
                scope.row, sum(len(data) for _, data in parts)
            )
            for kind, data in parts:
                item = await scope.store.put_code_artifact(
                    scope.row,
                    "integration_" + kind,
                    data,
                    maximum=scope.code.settings.storage_bytes,
                )
                delta[kind + "_artifact_id"] = str(item["id"])
        self.integration["evidence"]["candidate_delta"] = delta
        await scope.store.save_integration(scope.row, self.integration)
        if p == u:
            self.final, self.final_document = agent, json.loads(encoded)
            self.final_checks = checks
            await scope.store.finish_integration(
                scope.row,
                identity,
                "passed",
                agent["id"],
                [c["id"] for c in checks],
                {"reused_same_frozen_version": True},
            )
            return True, checks, None
        # A separate immutable revision and new check executions are mandatory even
        # when M happens to have the same content digest as A.
        merged_key = "integrations/" + identity.hex
        merged_parent = scope.store.artifact_storage.path(merged_key)
        merged_parent.parent.mkdir(mode=0o700, exist_ok=True)
        merged_parent.mkdir(mode=0o700)
        await scope.store.reserve_delivery_bytes(
            scope.row, scope.code.settings.snapshot_bytes
        )
        merged_root = merged_parent / "source"
        manifest, conflicts = await scope.managed_io(
            merge_versions,
            base_root,
            p,
            agent_root,
            a,
            scope.store.artifact_storage.path(observation["storage_key"]) / "source",
            u,
            merged_root,
            scope.code.settings.file_bytes,
            self.algorithm,
            {
                **scope.code.settings.model_dump(),
                "deadline": scope.row["effective_deadline_at"].timestamp(),
            },
        )
        await scope.managed_io(
            durable_json,
            merged_parent / "merge.json",
            {
                **self.integration["evidence"],
                "manifest": manifest,
                "conflicts": conflicts,
            },
        )
        if conflicts:
            await scope.store.finish_integration(
                scope.row,
                identity,
                "conflict",
                evidence={"conflicts": conflicts, "storage_key": merged_key},
            )
            raise CodeFailure(
                "DELIVERY_CONFLICT",
                "Original and candidate changes conflict; versions retained",
            )
        await scope.store.settle_delivery_bytes(
            scope.row,
            scope.code.settings.snapshot_bytes,
            sum(v["size"] for v in manifest.values()),
        )
        final, document = await self.revision(merged_root, "merge:" + identity.hex)
        self.final, self.final_document = final, document
        await scope.store.finish_integration(
            scope.row, identity, "checking", final["id"]
        )
        merged_checks = []
        end = min(
            scope.row["effective_deadline_at"],
            datetime.now(UTC) + timedelta(seconds=scope.code.settings.round_seconds),
        )
        for check_id in scope.row["_code"]["check_ids"]:
            merged_checks.append(
                await scope.run_check(
                    scope.code.checks[check_id],
                    "gate:merge:" + identity.hex + ":dev:" + check_id,
                    final,
                    round_deadline=end,
                )
            )
        merged_checks.append(
            await scope.run_check(
                scope.acceptance,
                "gate:merge:" + identity.hex + ":acceptance",
                final,
                purpose="acceptance",
                round_deadline=end,
            )
        )
        async with scope.store.locked() as conn:
            await scope.store.code_live(conn, scope.row)
            await conn.execute(
                "UPDATE agent.code_check_runs SET integration_id=%s WHERE id=ANY(%s::uuid[])",
                (identity, [c["id"] for c in merged_checks]),
            )
        passed = all(
            c["result"]["status"] == "passed" and c["result"]["complete"]
            for c in merged_checks
        )
        self.final_checks = merged_checks
        await scope.store.finish_integration(
            scope.row,
            identity,
            "passed" if passed else "failed",
            final["id"],
            [c["id"] for c in merged_checks],
            {
                "storage_key": merged_key,
                "final_digest": document["content_digest"],
                "integrated_user_changes": [
                    n for n in set(a) | set(manifest) if a.get(n) != manifest.get(n)
                ],
            },
        )
        if passed:
            return True, merged_checks, None
        # Install under the same candidate lock. A mixed install is never accepted:
        # durable 'installing' precedes changes; 'installed' follows a fresh freeze.
        await scope.store.finish_integration(
            scope.row,
            identity,
            "installing",
            final["id"],
            [c["id"] for c in merged_checks],
        )
        await scope.quiescent()
        await scope.store.reopen_repair(scope.row)
        scope.write_closed = False
        scope.candidate["status"] = "writable"
        try:
            await scope.managed_io(self.install, merged_root, manifest)
        except (ValueError, OSError) as exc:
            raise CodeFailure(
                "SNAPSHOT_INVALID",
                "Merged candidate installation could not be verified",
            ) from exc
        await scope.checkpoint("integration-install:" + identity.hex)
        fresh = protection(
            agent_root, a, merged_root, manifest, scope.code.settings.file_bytes
        )
        self.protected = update_protection(
            self.protected,
            fresh,
            merged_root,
            manifest,
            scope.store.artifact_storage.path(observation["storage_key"]) / "source",
            u,
            p,
            scope.code.settings.file_bytes,
        )
        self.integrated = observation  # Only after verified candidate installation.
        await scope.store.update_delivery(
            scope.row,
            integrated_observation_id=observation["id"],
            protected=self.protected,
        )
        await scope.store.finish_integration(
            scope.row,
            identity,
            "installed",
            final["id"],
            [c["id"] for c in merged_checks],
        )
        return (
            False,
            merged_checks,
            "Merged version failed checks; user changes are installed in the same candidate. Continue repair and reread source evidence.",
        )

    def install(self, source, manifest):
        from agent_service.delivery_fs import identity, source_parts
        from agent_service.repair_versions import read_regular, scan_candidate

        scope = self.scope
        root_fd = os.open(scope.candidate_root, os.O_DIRECTORY | os.O_NOFOLLOW)
        original = identity(os.fstat(root_fd))

        @contextmanager
        def parent(name, create=False):
            parts = source_parts(name)
            fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    if create:
                        try:
                            os.mkdir(part, 0o755, dir_fd=fd)
                            os.fsync(fd)
                        except FileExistsError:
                            pass
                    child = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                yield fd, parts[-1]
                os.fsync(fd)
            finally:
                os.close(fd)

        try:
            for name in sorted(
                set(scope.document["manifest"]) - set(manifest),
                key=lambda n: len(n.split("/")),
                reverse=True,
            ):
                with parent(name) as (fd, leaf):
                    info = os.stat(leaf, dir_fd=fd, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise DeliveryConflict(
                            "Unsupported candidate inode during installation"
                        )
                    os.unlink(leaf, dir_fd=fd)
            for name, expected in manifest.items():
                data, mode = read_regular(source, name, scope.code.settings.file_bytes)
                with parent(name, True) as (fd, leaf):
                    handle = os.open(
                        leaf,
                        os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                        0o600,
                        dir_fd=fd,
                    )
                    with os.fdopen(handle, "wb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                            raise DeliveryConflict(
                                "Unsupported candidate target during installation"
                            )
                        stream.truncate(0)
                        stream.write(data)
                        stream.flush()
                        os.fchmod(stream.fileno(), mode)
                        os.fsync(stream.fileno())
            os.fsync(root_fd)
            if (
                identity(scope.candidate_root.stat()) != original
                or scan_candidate(
                    scope.candidate_root,
                    scope.baseline,
                    scope.code.settings.model_dump(),
                )
                != manifest
            ):
                raise DeliveryConflict("Candidate installation did not verify")
        finally:
            os.close(root_fd)

    async def stage(self, draft):
        from agent_service.delivery_fs import stage_plan

        scope = self.scope
        await scope.evidence.verify_persisted(draft)
        scope.usage.update(
            check_calls=len(scope.checks),
            gate_rounds=scope.rounds,
            admitted_bytes=scope.candidate["admitted_bytes"],
            refreshes=self.refreshes,
        )
        await scope.store.record_repair_usage(scope.row, scope.usage)
        user_root = (
            scope.store.artifact_storage.path(self.observation["storage_key"])
            / "source"
        )
        final_root = (
            scope.store.artifact_storage.path(self.final["storage_key"]) / "source"
        )
        actual_document, actual_patch, actual_package = await scope.managed_io(
            build_exports,
            final_root,
            user_root,
            self.observation["document"]["source_manifest"],
            self.final_document["manifest"],
            scope.code.settings.model_dump(),
        )
        exports = {}
        export_size = (
            len(actual_patch)
            + len(actual_package)
            + len(json.dumps(actual_document).encode())
        )
        await scope.store.reserve_delivery_bytes(scope.row, export_size)
        for kind, data in [
            ("manifest", json.dumps(actual_document, sort_keys=True).encode()),
            ("patch", actual_patch),
            ("package", actual_package),
        ]:
            artifact = await scope.store.put_code_artifact(
                scope.row,
                "delivery_" + kind,
                data,
                maximum=scope.code.settings.storage_bytes,
            )
            exports[kind + "_artifact_id"] = str(artifact["id"])
        proof = {
            "version": 1,
            "delivery_mode": "candidate_only",
            "delivery_status": "not_applied",
            "report": draft.model_dump(mode="json")
            if hasattr(draft, "model_dump")
            else draft,
            "authority": scope.evidence.authority(),
        }
        proof_artifact = await scope.store.put_code_artifact(
            scope.row,
            "delivery_candidate_proof",
            json.dumps(proof, sort_keys=True).encode(),
            maximum=scope.code.settings.report_bytes + 131072,
        )
        # Staging and backups count against the same Run's storage ceiling.
        await scope.store.reserve_delivery_bytes(
            scope.row,
            sum(x["size"] for x in self.final_document["manifest"].values())
            + sum(
                x["size"]
                for x in self.observation["document"]["source_manifest"].values()
            ),
        )
        plan_id = uuid4().hex
        await scope.store.delivery_preparation(
            scope.row, "staging_intent", {"plan_id": plan_id}, plan_id
        )
        plan = await scope.managed_io(
            stage_plan,
            self.root,
            self.private,
            plan_id,
            self.observation["document"],
            scope.store.artifact_storage.path(self.final["storage_key"]) / "source",
            self.final_document["manifest"],
            scope.code.settings.file_bytes,
        )
        plan.update(
            delivery_id=str(self.record["id"]),
            run_id=str(scope.row["id"]),
            candidate_revision_id=str(scope.current["id"]),
            final_revision_id=str(self.final["id"]),
            integration_id=str(self.integration["id"]),
            observation_id=str(self.observation["id"]),
            final_check_ids=[c["id"] for c in self.final_checks],
            candidate_proof_id=str(proof_artifact["id"]),
            candidate_report_digest=fingerprint(proof["report"]),
            acceptance=scope.acceptance.manifest(),
            policy_digest=fingerprint(scope.code.manifest),
            image=scope.settings.image,
            baseline_digest=digest(scope.baseline),
            exports=exports,
        )
        from agent_service.delivery_evidence import delivery_summary
        from agent_service.repair_reports import markdown_repair

        forecast = {
            **self.record,
            "plan": plan,
            "plan_digest": fingerprint(plan),
            "status": "applied",
            "impact": "all" if plan["files"] or plan["directories"] else "none",
            "applied_at": datetime.now(UTC),
        }
        summary = await delivery_summary(scope.store, forecast)
        authority = {
            **proof["authority"],
            "delivery_mode": "auto_apply",
            "delivery_status": "applied",
            "delivery": summary,
        }
        predicted = {
            "version": 1,
            "delivery_mode": "auto_apply",
            "delivery_status": "applied",
            "report": proof["report"],
            "authority": authority,
            "delivery": summary,
        }
        if (
            len(json.dumps(predicted, ensure_ascii=False, sort_keys=True).encode())
            + 32768
            > scope.code.settings.report_bytes + 131072
            or len(markdown_repair(proof["report"], authority).encode()) + 512
            > scope.code.settings.report_bytes * 3
        ):
            raise CodeFailure(
                "ARTIFACT_LIMIT",
                "Final delivery report cannot fit the configured publication budget",
            )
        await scope.managed_io(
            durable_json, self.private / plan["id"] / "plan.json", plan
        )
        self.plan = plan
        self.record = await scope.store.stage_delivery_plan(
            scope.row, plan, self.integration["id"]
        )
        return plan

    async def rebuild(self):
        from agent_service.execution import ExecutionScope

        scope = self.scope
        await scope.store.reopen_repair(scope.row)
        await scope.store.update_delivery(
            scope.row, checkpoint=None, phase="integrating"
        )
        self.force_refresh = True
        await scope.store.rebuild_repair_resource(scope.row)
        scope.closing = False
        scope.container_id = None
        scope.session = None
        scope.children = []
        await ExecutionScope.prepare(scope)


class PersistentDeliveryBudget(AgentMiddleware):
    def __init__(self, scope):
        self.scope = scope

    async def awrap_model_call(self, request, handler):
        self.scope.delivery.persistent_usage = (
            await self.scope.store.consume_delivery_usage(self.scope.row, "model_calls")
        )
        return await handler(request)

    async def awrap_tool_call(self, request, handler):
        self.scope.delivery.persistent_usage = (
            await self.scope.store.consume_delivery_usage(self.scope.row, "tool_calls")
        )
        return await handler(request)


class DeliveryHandoffMiddleware(AgentMiddleware):
    def __init__(self, scope):
        self.scope = scope

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_agent(self, state, runtime):
        scope, delivery = self.scope, self.scope.delivery
        index = 0
        while True:
            if index == len(delivery.handoffs):
                if getattr(scope, "delivery_recovery", False):
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID", "Recovery cannot prepare a new plan"
                    )
                plan = await delivery.stage(state["structured_response"])
                delivery.handoffs.append(
                    {
                        "payload": {
                            "kind": "delivery-handoff-v1",
                            "delivery_id": str(delivery.record["id"]),
                            "plan_id": plan["id"],
                        },
                        "result": None,
                    }
                )
                await scope.store.update_delivery(scope.row, handoffs=delivery.handoffs)
            entry = delivery.handoffs[index]
            result = interrupt(entry["payload"])
            if entry["result"] is not None:
                if result != entry["result"]:
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID", "Historical resume differs"
                    )
                if result["action"] == "applied":
                    scope.row["_delivery_applied"] = True
                    return None
                index += 1
                continue
            expected = scope.row.pop("_internal_delivery_resume", None)
            if (
                result != expected
                or result["plan_id"] != entry["payload"]["plan_id"]
                or result["delivery_id"] != entry["payload"]["delivery_id"]
            ):
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID",
                    "Only matching service-constructed delivery resumes are accepted",
                )
            if result["action"] == "applied":
                scope.row["_delivery_applied"] = True
                # No model/tool or new plan after the applied handoff.
                return None
            if result["action"] != "refresh" or getattr(
                scope, "delivery_recovery", False
            ):
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID",
                    "Recovery cannot refresh or call the model",
                )
            await delivery.rebuild()
            async with scope.lock:
                passed, checks, reason = await delivery.validate(
                    scope.current, [scope.checks[i] for i in scope.passed["check_ids"]]
                )
            entry["result"] = result
            await scope.store.update_delivery(scope.row, handoffs=delivery.handoffs)
            if passed:
                async with scope.store.locked() as conn:
                    await scope.store.code_live(conn, scope.row)
                    await conn.execute(
                        "UPDATE agent.code_repair_candidates SET status='passed',frozen_revision_id=%s,report_digest=%s WHERE run_id=%s",
                        (
                            scope.current["id"],
                            scope.passed["draft_digest"],
                            scope.row["id"],
                        ),
                    )
                delivery.plan = None
                index += 1
                continue
            scope.passed = None
            scope.write_closed = False
            delivery.plan = None
            delivery.handoffs = []
            await scope.store.update_delivery(scope.row, handoffs=[])
            return {
                "structured_response": None,
                "jump_to": "model",
                "messages": [
                    HumanMessage(
                        content="Repair gate feedback (trusted controller): "
                        + json.dumps(
                            {
                                "reason": reason,
                                "checks": checks,
                                "refreshes": delivery.refreshes,
                            }
                        )
                    )
                ],
            }
