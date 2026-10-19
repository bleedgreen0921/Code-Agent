"""Revalidate immutable delivery evidence without running tests or project code."""

import asyncio
import json
from uuid import UUID

from agent_service.code_artifacts import CodeFailure
from agent_service.delivery_fs import content
from agent_service.profiles import fingerprint
from agent_service.repair_versions import (
    build_exports,
    source_manifest,
    verify_package,
    verify_revision,
)


async def validate_delivery(store, delivery, conn=None, require_applied=False):
    if conn is None:
        async with store.pool.connection() as connection:
            return await validate_delivery(store, delivery, connection, require_applied)
    try:
        plan = delivery["plan"]
        if (
            not plan
            or fingerprint(plan) != delivery["plan_digest"]
            or delivery["protocol"] != 1
            or plan["delivery_id"] != str(delivery["id"])
            or plan["run_id"] != str(delivery["run_id"])
            or plan["identity"] != delivery["root_identity"]
        ):
            raise ValueError("Plan identity/protocol differs")
        run = await (
            await conn.execute(
                "SELECT * FROM agent.runs WHERE id=%s", (delivery["run_id"],)
            )
        ).fetchone()
        if (
            store.registry.workspaces.get(run["workspace_id"])
            != delivery["workspace_root"]
        ):
            raise ValueError(
                "Registered workspace root differs from the authorized original directory"
            )
        metadata = await (
            await conn.execute(
                "SELECT * FROM agent.code_tasks WHERE task_id=%s",
                (delivery["task_id"],),
            )
        ).fetchone()
        candidate = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_candidates WHERE run_id=%s",
                (run["id"],),
            )
        ).fetchone()
        artifact, encoded = await store.code_artifact(
            delivery["task_id"], UUID(plan["candidate_proof_id"])
        )
        if artifact["kind"] != "delivery_candidate_proof":
            raise ValueError("Candidate proof kind differs")
        proof = json.loads(encoded)
        from agent_service.repair_reports import validate_published

        await validate_published(store, conn, run, candidate, proof)
        integration = await (
            await conn.execute(
                "SELECT * FROM agent.code_integrations WHERE id=%s AND delivery_id=%s",
                (plan["integration_id"], delivery["id"]),
            )
        ).fetchone()
        observation = await (
            await conn.execute(
                "SELECT * FROM agent.code_workspace_observations WHERE id=%s AND delivery_id=%s",
                (plan["observation_id"], delivery["id"]),
            )
        ).fetchone()
        if (
            not integration
            or integration["status"] != "passed"
            or not observation
            or str(integration["candidate_revision_id"])
            != plan["candidate_revision_id"]
            or str(integration["final_revision_id"]) != plan["final_revision_id"]
            or str(integration["observation_id"]) != plan["observation_id"]
            or integration["check_ids"] != plan["final_check_ids"]
            or observation["document"]["source_manifest"] != plan["before_manifest"]
            or observation["document"]["git_state"] != plan["git_state"]
            or observation["document"]["identity"] != plan["identity"]
            or plan["policy_digest"] != fingerprint(store.registry.repair.manifest)
            or plan["image"] != store.registry.execution.image
            or metadata["request"]["delivery_mode"] != "auto_apply"
        ):
            raise ValueError(
                "Integration, original observation or policy association differs"
            )
        if (
            plan["candidate_revision_id"] != proof["authority"]["revision_id"]
            or plan["candidate_report_digest"] != fingerprint(proof["report"])
            or plan["acceptance"] != candidate["acceptance"]
            or integration["evidence"]["agent_digest"]
            != proof["authority"]["content_digest"]
            or integration["evidence"]["user_digest"] != plan["observation_digest"]
            or plan["observation_digest"] != observation["document"]["content_digest"]
        ):
            raise ValueError("B/P/A/U/M proof association differs")
        if (
            integration["evidence"]["algorithm"]
            != store.registry.repair.manifest["merge_algorithm"]
        ):
            raise ValueError("Fixed merge algorithm identity differs")
        revision = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_revisions WHERE id=%s AND candidate_id=%s AND run_id=%s",
                (plan["final_revision_id"], candidate["id"], run["id"]),
            )
        ).fetchone()
        _, encoded = await store.code_artifact(
            delivery["task_id"], revision["manifest_artifact_id"]
        )
        document = json.loads(encoded)
        baseline = await (
            await conn.execute(
                "SELECT * FROM agent.code_snapshots WHERE id=%s",
                (candidate["baseline_id"],),
            )
        ).fetchone()
        baseline_root = store.artifact_storage.path(baseline["storage_key"]) / "source"
        baseline_doc = json.loads((baseline_root.parent / "manifest.json").read_bytes())
        b = source_manifest(baseline_doc["manifest"])
        from execution.code_snapshot import digest

        if plan["baseline_digest"] != digest(b):
            raise ValueError("Permanent B identity differs")
        limits = store.registry.repair.settings.model_dump()
        root = store.artifact_storage.path(revision["storage_key"]) / "source"
        await asyncio.to_thread(verify_revision, root, document, b, limits)
        if (
            document["manifest"] != plan["after_manifest"]
            or document["content_digest"] != plan["final_digest"]
        ):
            raise ValueError("Final immutable revision differs from staged delivery")
        _, package = await store.code_artifact(
            delivery["task_id"], revision["package_artifact_id"]
        )
        await asyncio.to_thread(
            verify_package, baseline_root, b, document, package, limits["file_bytes"]
        )
        checks = await (
            await conn.execute(
                "SELECT * FROM agent.code_check_runs WHERE id=ANY(%s::uuid[]) AND run_id=%s",
                (plan["final_check_ids"], run["id"]),
            )
        ).fetchall()
        required = set(metadata["check_ids"])
        if (
            len(checks) != len(required) + 1
            or {c["check_id"] for c in checks if c["purpose"] == "development"}
            != required
            or sum(c["purpose"] == "acceptance" for c in checks) != 1
            or any(
                str(c["revision_id"]) != plan["final_revision_id"]
                or c["status"] != "passed"
                or not c["result"]["complete"]
                or c["result_digest"] != fingerprint(c["result"])
                for c in checks
            )
        ):
            raise ValueError("M does not have its own complete required checks")
        for check in checks:
            expected = (
                candidate["acceptance"]
                if check["purpose"] == "acceptance"
                else json.loads(
                    json.dumps(
                        store.registry.repair.checks[check["check_id"]].manifest()
                    )
                )
            )
            if (
                check["purpose"] == "acceptance"
                and check["acceptance_digest"]
                != candidate["acceptance"]["asset_digest"]
            ):
                raise ValueError("Final acceptance asset digest differs")
            if check["definition"] != expected:
                raise ValueError("Required check authority differs")
            if (
                integration["evidence"]["merged"]
                and str(check["integration_id"]) != plan["integration_id"]
            ):
                raise ValueError("Merged version reused a candidate check")
        if integration["base_observation_id"]:
            previous = await (
                await conn.execute(
                    "SELECT * FROM agent.code_workspace_observations WHERE id=%s AND delivery_id=%s",
                    (integration["base_observation_id"], delivery["id"]),
                )
            ).fetchone()
            if not previous:
                raise ValueError("Rolling P observation differs")
            p_root = store.artifact_storage.path(previous["storage_key"]) / "source"
            p_manifest = previous["document"]["source_manifest"]
        else:
            p_root, p_manifest = baseline_root, b
        if digest(p_manifest) != integration["evidence"]["base_digest"]:
            raise ValueError("Rolling P source identity differs")
        a_revision = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_revisions WHERE id=%s AND candidate_id=%s AND run_id=%s",
                (plan["candidate_revision_id"], candidate["id"], run["id"]),
            )
        ).fetchone()
        a_root = store.artifact_storage.path(a_revision["storage_key"]) / "source"
        _, a_encoded = await store.code_artifact(
            delivery["task_id"], a_revision["manifest_artifact_id"]
        )
        a_manifest = json.loads(a_encoded)["manifest"]
        p_document, p_patch, p_package = await asyncio.to_thread(
            build_exports, a_root, p_root, p_manifest, a_manifest, limits
        )
        for kind, expected_data in [
            ("manifest", json.dumps(p_document, sort_keys=True).encode()),
            ("patch", p_patch),
            ("package", p_package),
        ]:
            item, data = await store.code_artifact(
                delivery["task_id"],
                UUID(integration["evidence"]["candidate_delta"][kind + "_artifact_id"]),
            )
            if (
                item["kind"] not in ("repair_" + kind, "integration_" + kind)
                or data != expected_data
            ):
                raise ValueError("P -> A immutable export differs")
        user_root = store.artifact_storage.path(observation["storage_key"]) / "source"
        from agent_service.code_execution import verify_snapshot

        await asyncio.to_thread(
            verify_snapshot, user_root, observation["document"], limits
        )
        actual_document, actual_patch, actual_package = await asyncio.to_thread(
            build_exports,
            root,
            user_root,
            plan["before_manifest"],
            plan["after_manifest"],
            limits,
        )
        for kind, expected in [
            ("manifest", json.dumps(actual_document, sort_keys=True).encode()),
            ("patch", actual_patch),
            ("package", actual_package),
        ]:
            artifact, data = await store.code_artifact(
                delivery["task_id"], UUID(plan["exports"][kind + "_artifact_id"])
            )
            if artifact["kind"] != "delivery_" + kind or data != expected:
                raise ValueError("Actual U -> M export differs")
        files = await (
            await conn.execute(
                "SELECT * FROM agent.code_delivery_files WHERE delivery_id=%s AND plan_id=%s ORDER BY ordinal",
                (delivery["id"], plan["id"]),
            )
        ).fetchall()
        if [f["operation"] for f in files] != plan["files"]:
            raise ValueError("Per-file plan differs")
        expected_paths = {
            p
            for p in set(plan["before_manifest"]) | set(plan["after_manifest"])
            if plan["before_manifest"].get(p) != plan["after_manifest"].get(p)
        }
        if {f["path"] for f in plan["files"]} != expected_paths or len(
            plan["files"]
        ) != len(expected_paths):
            raise ValueError("Write set is not exactly U -> M")
        for f in plan["files"]:
            from agent_service.delivery_fs import content

            if (
                content(f["before"]) != plan["before_manifest"].get(f["path"])
                or f["after"] != plan["after_manifest"].get(f["path"])
                or content(f["staged"]) != f["after"]
            ):
                raise ValueError("Staged file association differs")
        if require_applied:
            if (
                delivery["status"] != "applied"
                or delivery["impact"]
                != ("all" if plan["files"] or plan["directories"] else "none")
                or not delivery["checkpoint"]
            ):
                raise ValueError("Delivery is not durably applied")
            if any(f["phase"] != "confirmed" for f in files):
                raise ValueError("File journal is incomplete")
            if await store.unreleased_delivery_resources(conn, run["id"]):
                raise ValueError("Execution cleanup is unconfirmed")
        return proof
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise CodeFailure(
            "DELIVERY_EVIDENCE_INVALID", "Delivery evidence invalid: " + str(exc)
        ) from exc


async def delivery_summary(store, delivery, conn=None):
    if conn is None:
        async with store.pool.connection() as connection:
            return await delivery_summary(store, delivery, connection)
    observations = await (
        await conn.execute(
            "SELECT id,document,created_at FROM agent.code_workspace_observations WHERE delivery_id=%s ORDER BY created_at,id",
            (delivery["id"],),
        )
    ).fetchall()
    integrations = await (
        await conn.execute(
            "SELECT * FROM agent.code_integrations WHERE delivery_id=%s ORDER BY created_at,id",
            (delivery["id"],),
        )
    ).fetchall()
    recovery_rows = await (
        await conn.execute(
            "SELECT phase,created_at FROM agent.code_delivery_events WHERE delivery_id=%s AND phase IN ('recovery_claimed','needs_recovery','rolled_back') ORDER BY created_at,id",
            (delivery["id"],),
        )
    ).fetchall()
    plan = delivery["plan"]
    return dict(
        id=str(delivery["id"]),
        status=delivery["status"],
        impact=delivery["impact"],
        cancel_requested=delivery["cancel_requested"],
        rollback_requested=delivery["rollback_requested"],
        recovery={
            "event_count": len(recovery_rows),
            "truncated": len(recovery_rows) > 64,
            "events": [
                {"phase": r["phase"], "at": r["created_at"].isoformat()}
                for r in recovery_rows[-64:]
            ],
        },
        plan_id=plan["id"],
        plan_digest=delivery["plan_digest"],
        baseline_digest=plan["baseline_digest"],
        candidate_revision_id=plan["candidate_revision_id"],
        final_revision_id=plan["final_revision_id"],
        final_digest=plan["final_digest"],
        observation_id=plan["observation_id"],
        observation_digest=plan["observation_digest"],
        integration_id=plan["integration_id"],
        final_check_ids=plan["final_check_ids"],
        exports=plan["exports"],
        applied_at=delivery["applied_at"].isoformat()
        if delivery["applied_at"]
        else None,
        changes=[
            {
                "path": f["path"],
                "before": content(f["before"]),
                "after": f["after"],
            }
            for f in plan["files"]
        ],
        observations=[
            {
                "id": str(o["id"]),
                "content_digest": o["document"]["content_digest"],
                "git_digest": o["document"]["git_digest"],
                "captured_at": o["created_at"].isoformat(),
            }
            for o in observations
        ],
        integrations=[
            {
                "id": str(i["id"]),
                "base_observation_id": str(i["base_observation_id"])
                if i["base_observation_id"]
                else None,
                "observation_id": str(i["observation_id"]),
                "candidate_revision_id": str(i["candidate_revision_id"]),
                "final_revision_id": str(i["final_revision_id"])
                if i["final_revision_id"]
                else None,
                "status": i["status"],
                "checks": i["check_ids"],
                "candidate_delta": i["evidence"]["candidate_delta"],
                "inputs": {
                    k: i["evidence"][k]
                    for k in ("base_digest", "agent_digest", "user_digest", "algorithm")
                },
            }
            for i in integrations
        ],
        evidence_notice="Source references and causal explanation refer to candidate A. Final M checks and U->M writes are recorded separately; model semantics require independent real-model acceptance.",
    )


async def validate_handoff(graph, delivery):
    """Bind a native interruption to this exact plan and candidate submission."""
    checkpoint = delivery["checkpoint"]
    config = (checkpoint or {}).get("configurable", {})
    if (
        config.get("thread_id") != str(delivery["run_id"])
        or config.get("checkpoint_ns", "") != ""
        or not config.get("checkpoint_id")
    ):
        raise CodeFailure(
            "DELIVERY_CONTROL_INVALID", "Native handoff thread/checkpoint differs"
        )
    saved = await graph.aget_state(checkpoint)
    interruptions = [i for task in saved.tasks for i in task.interrupts]
    expected = {
        "kind": "delivery-handoff-v1",
        "delivery_id": str(delivery["id"]),
        "plan_id": delivery["plan"]["id"],
    }
    if not saved.next or not interruptions or interruptions[-1].value != expected:
        raise CodeFailure(
            "DELIVERY_CONTROL_INVALID",
            "Native interruption does not identify this delivery plan",
        )
    draft = saved.values.get("structured_response")
    draft = draft.model_dump(mode="json") if hasattr(draft, "model_dump") else draft
    if fingerprint(draft) != delivery["plan"]["candidate_report_digest"]:
        raise CodeFailure(
            "DELIVERY_CONTROL_INVALID", "Native handoff submission differs"
        )
