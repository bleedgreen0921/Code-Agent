"""Repair evidence validation and server-generated candidate reports."""

import asyncio
import hashlib
import json
from uuid import UUID

from langchain_core.messages import ToolMessage

from agent_service.code_artifacts import CodeFailure
from agent_service.code_reports import EvidenceLedger
from agent_service.profiles import fingerprint
from agent_service.repair_versions import (
    baseline_archive,
    build_exports,
    source_manifest,
    verify_package,
    verify_revision,
)
from contracts.repair import DeliveryReportV1, RepairDraftV1, RepairReportV1


class RepairLedger(EvidenceLedger):
    def __init__(self, store, row, manifest, scope):
        super().__init__(store, row, manifest)
        self.scope = scope
        self.pending = {}

    async def read_result(self, name, args, result, settings):
        content, meta = await super().read_result(name, args, result, settings)
        content = (
            json.dumps(meta, ensure_ascii=False, sort_keys=True)
            + "\n"
            + content.split("\n", 1)[1]
        )
        from agent_service.policy import active_call

        call = active_call.get()
        self.sources.pop(meta["evidence_id"], None)
        self.pending[call.call_id] = (meta, content)
        return content, meta

    async def confirm_delivery(self, call_id, result):
        item = self.pending.pop(call_id, None)
        if item is None:
            return
        meta, expected = item
        delivered = (
            isinstance(result, ToolMessage)
            and result.status == "success"
            and result.content == expected
        )
        receipt = dict(
            evidence_id=meta["evidence_id"],
            delivered=delivered,
            content_sha256=hashlib.sha256(expected.encode()).hexdigest(),
        )
        await self.store.put_code_artifact(
            self.row,
            "repair_delivery",
            json.dumps(receipt, sort_keys=True).encode(),
            maximum=2048,
        )
        if delivered:
            self.sources[meta["evidence_id"]] = meta

    def response_schema(self):
        return RepairDraftV1

    def validate(self, value, document=None):
        report = (
            value
            if isinstance(value, RepairDraftV1)
            else RepairDraftV1.model_validate(value)
        )
        raw = report.model_dump(mode="json")
        if (
            len(json.dumps(raw, ensure_ascii=False).encode())
            > self.scope.code.settings.report_bytes
        ):
            raise ValueError("Repair draft exceeds report limit")
        if report.unfinished or any(goal.status != "resolved" for goal in report.goals):
            raise ValueError(
                "Target issues remain unfinished or unconfirmed; candidate cannot pass"
            )
        document = document or self.scope.document
        for ref in report.sources:
            source = self.sources.get(str(ref.evidence_id))
            current = document["manifest"].get(ref.path)
            if (
                not source
                or source["run_id"] != str(self.row["id"])
                or source["snapshot_id"] != str(self.row["_snapshot_id"])
                or ref.path != source["path"]
                or ref.sha256 != source["sha256"]
                or not source["start_line"]
                <= ref.start_line
                <= ref.end_line
                <= source["end_line"]
                or not current
                or current["sha256"] != ref.sha256
            ):
                raise ValueError(
                    "Source reference is unknown, stale or outside delivered lines; reread final candidate source"
                )
        for identity in report.check_run_ids:
            if str(identity) not in self.checks:
                raise ValueError("Unknown repair check reference")
        return raw

    async def verify_persisted(self, report):
        raw = self.validate(report)
        if (
            not self.scope.write_closed
            or not self.scope.passed
            or self.scope.passed["draft_digest"] != fingerprint(raw)
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Repair report has not passed the in-graph gate"
            )
        return raw

    def authority(self):
        scope = self.scope
        return dict(
            task_id=str(self.row["task_id"]),
            run_id=str(self.row["id"]),
            snapshot_id=str(self.row["_snapshot_id"]),
            baseline_digest=scope.manifest["content_digest"],
            baseline_artifact_id=str(scope.baseline_artifact["id"]),
            baseline_resolved_at=scope.manifest["resolved_at"],
            candidate_id=str(scope.candidate["id"]),
            revision_id=str(scope.current["id"]),
            content_digest=scope.current["content_digest"],
            acceptance=scope.acceptance.manifest(),
            policy_digest=fingerprint(scope.code.manifest),
            attempt_id=str(scope.passed["id"]),
            checks=list(scope.checks.values()),
            artifacts={
                k: str(scope.current[k])
                for k in (
                    "manifest_artifact_id",
                    "patch_artifact_id",
                    "package_artifact_id",
                )
            },
            usage=scope.usage,
            end_reason="candidate_passed",
            delivery_mode="candidate_only",
            delivery_status="not_applied",
            checks_configured=bool(self.row["_code"]["check_ids"]),
            evidence_notice="Registered acceptance proves only its covered conditions; model causal explanations are not automatically proven.",
            baseline_notice="No baseline acceptance failure was observed."
            if any(
                c["purpose"] == "baseline" and c["result"]["status"] == "passed"
                for c in scope.checks.values()
            )
            else "See actual baseline acceptance records; historical reproduction claims are not inherited.",
        )


def markdown_repair(report, authority):
    lines = [
        "# 隔离修复候选报告",
        "",
        report["summary"],
        "",
        "交付状态：auto_apply / applied"
        if authority["delivery_mode"] == "auto_apply"
        else "交付状态：candidate_only / not_applied（尚未写回原目录）",
        "基线：" + authority["baseline_digest"],
        "冻结候选：" + authority["content_digest"],
        "",
        "## 原因与修改",
        report["cause"],
        *["- " + s for s in report["changes"]],
        "",
        "## 当前基线确认",
        report["baseline_confirmation"],
        authority["baseline_notice"],
        "",
        "## 目标处理",
        *[
            g["goal"] + "：" + g["status"] + "；" + g["explanation"]
            for g in report["goals"]
        ],
        "",
        "## 开发测试修改",
        *report["test_changes"],
        "",
        "## 检查记录",
    ]
    if not authority["checks_configured"]:
        lines.append("未配置开发检查套件；独立验收仍为必需。")
    lines += [
        json.dumps(c, ensure_ascii=False, sort_keys=True) for c in authority["checks"]
    ]
    lines += [
        "",
        "## 产物",
        json.dumps(authority["artifacts"], sort_keys=True),
        "",
        "## 限制",
        *report["limitations"],
        authority["evidence_notice"],
        "",
        "预算：" + json.dumps(authority["usage"], sort_keys=True),
        "",
    ]
    if authority["delivery_mode"] == "auto_apply":
        delivery = authority["delivery"]
        lines += [
            "最终交付版本 M：" + delivery["final_digest"],
            "源码引用对应候选 A；M 的检查和 U→M 写回清单单独记录。",
            "交付记录：" + json.dumps(delivery, ensure_ascii=False, sort_keys=True),
        ]
    return "\n".join(lines)


async def validate_published(store, conn, run, candidate, document):
    """Reconcile persisted identities/bytes, never rerun acceptance during publication."""
    try:
        validated = (
            DeliveryReportV1
            if document.get("delivery_mode") == "auto_apply"
            else RepairReportV1
        ).model_validate(document)
        draft = validated.report.model_dump(mode="json")
        authority = document["authority"]
        attempt = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_attempts WHERE id=%s AND run_id=%s AND status='passed'",
                (authority["attempt_id"], run["id"]),
            )
        ).fetchone()
        revision = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_revisions WHERE id=%s AND candidate_id=%s AND run_id=%s",
                (candidate["frozen_revision_id"], candidate["id"], run["id"]),
            )
        ).fetchone()
        if (
            not attempt
            or not revision
            or candidate["current_revision_id"] != revision["id"]
            or attempt["revision_id"] != revision["id"]
            or attempt["draft_digest"] != fingerprint(draft)
            or candidate["report_digest"] != fingerprint(draft)
            or attempt["draft"] != draft
            or candidate["acceptance"]
            != json.loads(json.dumps(authority["acceptance"]))
            or candidate["policy_digest"] != authority["policy_digest"]
            or candidate["policy_digest"] != fingerprint(store.registry.repair.manifest)
            or authority["revision_id"] != str(revision["id"])
            or authority["content_digest"] != revision["content_digest"]
            or authority["candidate_id"] != str(candidate["id"])
            or authority["run_id"] != str(run["id"])
            or authority["task_id"] != str(run["task_id"])
            or authority["snapshot_id"] != str(candidate["baseline_id"])
            or (authority["delivery_status"], authority["delivery_mode"])
            != (validated.delivery_status, validated.delivery_mode)
            or draft["unfinished"]
            or any(g["status"] != "resolved" for g in draft["goals"])
        ):
            raise ValueError("Candidate report identity differs")
        baseline = await (
            await conn.execute(
                "SELECT * FROM agent.code_snapshots WHERE id=%s AND run_id=%s AND state='sealed'",
                (candidate["baseline_id"], run["id"]),
            )
        ).fetchone()
        root = store.artifact_storage.path(baseline["storage_key"])
        raw = (root / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != baseline["manifest_digest"]
            or authority["baseline_digest"] != baseline["content_digest"]
        ):
            raise ValueError("Baseline identity differs")
        from agent_service.code_execution import verify_snapshot

        limits = store.registry.repair.settings.model_dump()
        await asyncio.to_thread(verify_snapshot, root / "source", manifest, limits)

        async def artifact(identity, kind=None):
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_artifacts WHERE id=%s AND run_id=%s AND state='available'",
                    (identity, run["id"]),
                )
            ).fetchone()
            if (
                not row
                or (kind and row["kind"] != kind)
                or row["snapshot_id"] != candidate["baseline_id"]
            ):
                raise ValueError("Candidate artifact ownership/kind differs")
            return row, store.artifact_storage.read(
                row["storage_key"], row["sha256"], row["size_bytes"]
            )

        baseline_record, baseline_package = await artifact(
            authority["baseline_artifact_id"], "repair_baseline"
        )
        if baseline_record["metadata"] != {
            "content_digest": manifest["content_digest"],
            "git_digest": manifest["git_digest"],
        }:
            raise ValueError("Baseline package identity differs")
        expected_baseline = await asyncio.to_thread(
            baseline_archive,
            root / "source",
            manifest["manifest"],
            limits["snapshot_bytes"],
        )
        if baseline_package != expected_baseline:
            raise ValueError("Baseline package bytes differ")
        _, data = await artifact(revision["manifest_artifact_id"], "repair_manifest")
        version = json.loads(data)
        if version["content_digest"] != revision["content_digest"] or authority[
            "artifacts"
        ] != {
            k: str(revision[k])
            for k in (
                "manifest_artifact_id",
                "patch_artifact_id",
                "package_artifact_id",
            )
        }:
            raise ValueError("Candidate manifest or artifact identities differ")
        await asyncio.to_thread(
            verify_revision,
            store.artifact_storage.path(revision["storage_key"]) / "source",
            version,
            source_manifest(manifest["manifest"]),
            limits,
        )
        _, package = await artifact(revision["package_artifact_id"], "repair_package")
        await asyncio.to_thread(
            verify_package,
            root / "source",
            source_manifest(manifest["manifest"]),
            version,
            package,
            limits["file_bytes"],
        )
        _, patch = await artifact(revision["patch_artifact_id"], "repair_patch")
        rebuilt, expected_patch, expected_package = await asyncio.to_thread(
            build_exports,
            store.artifact_storage.path(revision["storage_key"]) / "source",
            root / "source",
            source_manifest(manifest["manifest"]),
            version["manifest"],
            limits,
        )
        if rebuilt != version or patch != expected_patch or package != expected_package:
            raise ValueError(
                "Manifest, text patch or file package differs from frozen content"
            )
        checks = await (
            await conn.execute(
                "SELECT * FROM agent.code_check_runs WHERE run_id=%s ORDER BY created_at,id",
                (run["id"],),
            )
        ).fetchall()
        expected = {}
        for check in checks:
            if (
                check["status"] == "running"
                or check["result_digest"] != fingerprint(check["result"])
                or check["candidate_id"] != candidate["id"]
            ):
                raise ValueError("Check association or result differs")
            _, saved = await artifact(check["result_artifact_id"], "check_result")
            if json.loads(saved)["summary"] != check["result"]:
                raise ValueError("Check result artifact differs")
            await artifact(check["log_artifact_id"], "check_log")
            value = {
                k: str(check[k]) if check[k] is not None else None
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
                check_id=check["check_id"],
                purpose=check["purpose"],
                result=check["result"],
                result_digest=check["result_digest"],
                definition_digest=fingerprint(check["definition"]),
                acceptance_digest=check["acceptance_digest"],
            )
            expected[value["id"]] = value
        if (
            expected != {c["id"]: c for c in authority["checks"]}
            or candidate["usage"] != authority["usage"]
        ):
            raise ValueError("Reported check records or usage differ")
        gate = [c for c in checks if str(c["id"]) in attempt["check_ids"]]
        metadata = await (
            await conn.execute(
                "SELECT * FROM agent.code_tasks WHERE task_id=%s", (run["task_id"],)
            )
        ).fetchone()
        required = set(metadata["check_ids"])
        if (
            len(gate) != len(required) + 1
            or {c["check_id"] for c in gate if c["purpose"] == "development"}
            != required
            or sum(c["purpose"] == "acceptance" for c in gate) != 1
            or any(
                c["revision_id"] != revision["id"]
                or c["status"] != "passed"
                or not c["result"]["complete"]
                for c in gate
            )
        ):
            raise ValueError("Final required candidate checks did not pass")
        for check in gate:
            if check["purpose"] == "acceptance" and (
                check["acceptance_digest"] != candidate["acceptance"]["asset_digest"]
                or check["definition"] != candidate["acceptance"]
            ):
                raise ValueError("Acceptance authority differs")
            if check["purpose"] == "development" and check["definition"] != json.loads(
                json.dumps(store.registry.repair.checks[check["check_id"]].manifest())
            ):
                raise ValueError("Development check authority differs")
        delivered = {}
        receipts = await (
            await conn.execute(
                "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND kind='repair_delivery' AND state='available'",
                (run["id"],),
            )
        ).fetchall()
        for receipt in receipts:
            value = json.loads(
                store.artifact_storage.read(
                    receipt["storage_key"], receipt["sha256"], receipt["size_bytes"]
                )
            )
            if value["delivered"]:
                delivered[value["evidence_id"]] = value["content_sha256"]
        for ref in draft["sources"]:
            source, data = await artifact(UUID(ref["evidence_id"]), "source_evidence")
            source_document = json.loads(data)
            meta = source_document["metadata"]
            expected_delivery = hashlib.sha256(
                (
                    json.dumps(meta, ensure_ascii=False)
                    + "\n"
                    + source_document["content"]
                ).encode()
            ).hexdigest()
            if (
                delivered.get(ref["evidence_id"]) != expected_delivery
                or source["metadata"] != meta
                or ref["path"] != meta["path"]
                or ref["sha256"] != meta["sha256"]
                or version["manifest"].get(ref["path"], {}).get("sha256")
                != ref["sha256"]
                or not meta["start_line"]
                <= ref["start_line"]
                <= ref["end_line"]
                <= meta["end_line"]
            ):
                raise ValueError("Final source evidence differs")
        if any(i not in expected for i in draft["check_run_ids"]):
            raise ValueError("Unknown final check references")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise CodeFailure(
            "REPORT_INVALID",
            "Persisted repair candidate validation failed: " + str(exc),
        ) from exc
