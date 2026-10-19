"""Delivered fixed-version evidence, deterministic coverage, and Review validation."""

import json
import hashlib
from uuid import uuid4
from pydantic import model_validator
from contracts.review import ReviewReportV1
from agent_service.code_artifacts import CodeFailure
from agent_service.profiles import fingerprint


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def review_skill():
    from pathlib import Path
    from agent_service.skills import Skill

    body = (
        (Path(__file__).resolve().parents[1] / "review_skills/isolated-review/SKILL.md")
        .read_text()
        .split("---", 2)[-1]
        .strip()
    )
    return Skill(
        "isolated-review",
        "Review a fixed sealed range with verifiable source, diff and execution references.",
        body,
        hashlib.sha256(body.encode()).hexdigest(),
        "{}",
    )


class ReviewLedger:
    def __init__(self, store, row, manifest, document):
        self.store, self.row, self.manifest, self.document = (
            store,
            row,
            manifest,
            document,
        )
        self.events, self.sources, self.checks, self.reproductions = {}, {}, {}, {}
        self.pending = {}
        self.range = document["range"]
        self.changes = {c["change_id"]: c for c in document["changes"]}
        self.settings = store.registry.review.settings
        self.output = store.registry.execution

    async def deliver(self, kind, records, *, offset=0, total=None, **metadata):
        identity = str(uuid4())
        meta = {
            "evidence_id": identity,
            "run_id": str(self.row["id"]),
            "snapshot_id": str(self.row["_snapshot_id"]),
            "range_id": self.range["range_id"],
            "kind": kind,
            **metadata,
            "offset": offset,
            "count": 0,
        }
        retained = []
        total = len(records) + offset if total is None else total
        for record in records:
            candidate = {
                "metadata": {**meta, "count": len(retained) + 1},
                "records": [*retained, record],
                "next_offset": total,
                "limitations": ["message_record_limit"],
            }
            if len(encoded(candidate).encode()) > self.output.output_bytes - 256:
                break
            retained.append(record)
        meta["count"] = len(retained)
        next_offset = offset + len(retained) if offset + len(retained) < total else None
        payload = {
            "metadata": meta,
            "records": retained,
            "next_offset": next_offset,
            "limitations": ["message_record_limit"] if records and not retained else [],
        }
        # Single JSON line avoids another line-count crop in tool middleware.
        content = encoded(payload)
        if len(content.encode()) > self.output.output_bytes:
            raise ValueError("Review evidence metadata exceeds message budget")
        await self.store.put_code_artifact(
            self.row,
            "review_evidence",
            content.encode(),
            metadata=meta,
            artifact_id=__import__("uuid").UUID(identity),
            maximum=self.output.output_bytes,
        )
        from agent_service.policy import active_call

        call = active_call.get()
        if call is None:
            raise CodeFailure(
                "REPORT_INVALID", "Evidence delivery requires trusted tool identity"
            )
        self.pending[call.call_id] = (identity, payload, content)
        return content, {**meta, "next_offset": next_offset}

    async def confirm_delivery(self, call_id, result):
        candidate = self.pending.pop(call_id, None)
        if not candidate:
            return
        identity, payload, expected = candidate
        from langchain_core.messages import ToolMessage

        delivered = (
            isinstance(result, ToolMessage)
            and result.status == "success"
            and result.content == expected
        )
        receipt = {
            "evidence_id": identity,
            "delivered": delivered,
            "content_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        }
        await self.store.put_code_artifact(
            self.row,
            "review_delivery",
            encoded(receipt).encode(),
            metadata={"range_id": self.range["range_id"]},
            maximum=1024,
        )
        if delivered:
            self.events[identity] = payload
            if payload["metadata"]["kind"] == "source":
                self.sources[identity] = payload["metadata"]

    async def diff(self, change_id=None, offset=0, limit=50):
        if change_id is None:
            records = [
                {k: v for k, v in c.items() if k not in ("records", "hunks")}
                for c in self.document["changes"]
            ]
            return await self.deliver(
                "list",
                records[offset : offset + limit],
                offset=offset,
                total=len(records),
            )
        if change_id not in self.changes:
            from langchain_core.tools import ToolException

            raise ToolException("Unknown change in fixed range")
        change = self.changes[change_id]
        return await self.deliver(
            "patch",
            change["records"][offset : offset + limit],
            offset=offset,
            total=len(change["records"]),
            change_id=change_id,
            patch_limitations=change["limitations"],
        )

    async def source(self, args, data):
        records = data.get("records", [])
        return await self.deliver(
            "source",
            records,
            offset=args.get("start_line", 1) - 1,
            total=data["total_lines"],
            side=args["side"],
            path=args["path"],
            tree=data["tree"],
            blob=data["blob"],
            sha256=data["sha256"],
        )

    async def read_result(self, name, args, result, settings):
        # Ordinary reads are always head and re-read through the fixed blob helper,
        # ensuring one Review-specific version/evidence format for all source reads.
        path = args["path"].removeprefix("/workspace/").lstrip("/")
        path = "/".join(p for p in path.split("/") if p != ".")
        scope = self.row["_code_scope"]
        fixed_args = {**args, "path": path, "side": "head"}
        data = await scope.file_call(
            "review_read_file", {**fixed_args, "range": self.range}
        )
        if "error" in data:
            from langchain_core.tools import ToolException

            raise ToolException(data["error"])
        return await self.source(fixed_args, data)

    async def execution(self, record):
        return await self.deliver("execution", [record], execution_id=record["id"])

    def validate_ref(self, ref):
        event = self.events.get(str(ref.evidence_id))
        if not event:
            raise ValueError("Unknown source evidence for this Run")
        meta = event["metadata"]
        expected_tree = self.range["base_tree" if ref.side == "base" else "head_tree"]
        if (
            meta["kind"] != "source"
            or meta["range_id"] != self.range["range_id"]
            or meta["run_id"] != str(self.row["id"])
            or meta["snapshot_id"] != str(self.row["_snapshot_id"])
            or meta.get("side") != ref.side
            or meta.get("path") != ref.path
            or meta.get("tree") != expected_tree
        ):
            raise ValueError(
                "Source evidence version, side, path or ownership mismatch"
            )
        lines = {r["line"] for r in event["records"]}
        if (
            ref.end_line < ref.start_line
            or ref.end_line - ref.start_line + 1 > len(lines)
            or not all(n in lines for n in range(ref.start_line, ref.end_line + 1))
        ):
            raise ValueError(
                "Source reference must use actually delivered complete lines"
            )

    def validate(self, value):
        report = (
            value
            if isinstance(value, ReviewReportV1)
            else ReviewReportV1.model_validate(value)
        )
        raw = report.model_dump(mode="json")
        if len(encoded(raw).encode()) > self.settings.report_bytes:
            raise ValueError("Review report exceeds configured byte limit")
        declarations = set()
        for declaration in report.changes:
            if (
                declaration.change_id not in self.changes
                or declaration.change_id in declarations
            ):
                raise ValueError("Unknown or duplicate change declaration")
            declarations.add(declaration.change_id)
            for ref in declaration.evidence:
                self.validate_ref(ref)
                change = self.changes[declaration.change_id]
                if ref.path != change[ref.side + "_path"]:
                    raise ValueError("Declaration evidence must describe its change")
        for finding in report.findings:
            loc = finding.location
            change = self.changes.get(loc.change_id)
            if not change or loc.path != change[loc.side + "_path"]:
                raise ValueError(
                    "Finding location is outside the fixed range or wrong side"
                )
            changed_lines = {
                line for h in change["hunks"] for line in h[loc.side + "_lines"]
            }
            if loc.line is None:
                if not change["file_location"]:
                    raise ValueError(
                        "File-level location requires metadata or empty-file change"
                    )
            elif loc.line not in changed_lines:
                raise ValueError(
                    "Finding primary line must be an actual changed source line"
                )
            diff_events = [self.events.get(str(i)) for i in finding.diff_evidence_ids]
            if not any(
                e
                and e["metadata"]["range_id"] == self.range["range_id"]
                and (
                    (
                        e["metadata"]["kind"] == "patch"
                        and e["metadata"].get("change_id") == loc.change_id
                        and (
                            loc.line is None
                            or any(
                                r[("old_line" if loc.side == "base" else "new_line")]
                                == loc.line
                                for r in e["records"]
                            )
                        )
                    )
                    or (
                        loc.line is None
                        and e["metadata"]["kind"] == "list"
                        and any(r["change_id"] == loc.change_id for r in e["records"])
                    )
                )
                for e in diff_events
            ):
                raise ValueError(
                    "Finding requires delivered diff evidence at its location"
                )
            if any(
                e is None or e["metadata"]["kind"] not in ("patch", "list")
                for e in diff_events
            ):
                raise ValueError("Invalid diff evidence reference")
            for ref in finding.evidence:
                self.validate_ref(ref)
            if loc.line is not None and not any(
                ref.side == loc.side
                and ref.path == loc.path
                and ref.start_line <= loc.line <= ref.end_line
                for ref in finding.evidence
            ):
                raise ValueError("Finding line requires delivered source evidence")
            # Static before/after context on each changed nonempty text side.
            for side in ("base", "head"):
                if (
                    change[side + "_path"]
                    and any(
                        h["old_count" if side == "base" else "new_count"]
                        for h in change["hunks"]
                    )
                    and not any(
                        ref.side == side and ref.path == change[side + "_path"]
                        for ref in finding.evidence
                    )
                ):
                    raise ValueError(
                        "Introduction basis requires source evidence on both available sides"
                    )
            ids = list(map(str, finding.reproduction_ids))
            if finding.reproduction_status == "not_run":
                if ids:
                    raise ValueError("not_run cannot reference reproductions")
            elif not ids and not (
                finding.reproduction_status == "observed" and finding.check_case_refs
            ):
                raise ValueError("Reproduction claim requires an execution record")
            for identity in ids:
                execution = self.reproductions.get(identity)
                delivered = any(
                    e["metadata"]["kind"] == "execution"
                    and e["metadata"].get("execution_id") == identity
                    and e["records"]
                    for e in self.events.values()
                )
                if (
                    not execution
                    or execution["result"]["status"] != finding.reproduction_status
                    or execution["content_digest"] != self.range["content_digest"]
                    or not delivered
                ):
                    raise ValueError(
                        "Reproduction status must match a delivered head execution"
                    )
            if finding.check_case_refs:
                for case in finding.check_case_refs:
                    if not any(
                        e["metadata"]["kind"] == "execution_detail"
                        and any(
                            log["metadata"]["kind"] == "execution_log"
                            and log["metadata"].get("execution_id")
                            == e["metadata"].get("execution_id")
                            and log["records"]
                            for log in self.events.values()
                        )
                        and any(
                            r.get("case") == case
                            and r.get("outcome") in ("failed", "error")
                            for r in e["records"]
                        )
                        for e in self.events.values()
                    ):
                        raise ValueError(
                            "Specific failing case and delivered diagnostic are required"
                        )
        return raw

    def coverage(self, report):
        declarations = {d["change_id"]: d for d in report["changes"]}
        listed = {
            r["change_id"]
            for e in self.events.values()
            if e["metadata"]["kind"] == "list"
            for r in e["records"]
        }
        missing, covered = [], []
        for identity, change in self.changes.items():
            reasons = list(change["limitations"])
            declaration = declarations.get(identity)
            if identity not in listed:
                reasons.append("change_list_unread")
            positions = {
                r["output_line"]
                for e in self.events.values()
                if e["metadata"]["kind"] == "patch"
                and e["metadata"].get("change_id") == identity
                for r in e["records"]
            }
            if not set(range(1, len(change["records"]) + 1)) <= positions:
                reasons.append("patch_or_hunk_unread")
            if not declaration or declaration["status"] != "reviewed":
                reasons.append(
                    declaration["status"] if declaration else "no_declaration"
                )
            elif declaration["missing_context"]:
                reasons.append("necessary_context_missing")
            elif change["hunks"] and not declaration["evidence"]:
                reasons.append("source_context_unread")
            if reasons:
                missing.append(
                    {
                        "change_id": identity,
                        "reasons": sorted(set(reasons)),
                        "reason": declaration["reason"]
                        if declaration
                        else "No declaration",
                    }
                )
            else:
                covered.append(identity)
        limits = list(self.document["limitations"])
        if not self.document["list_complete"]:
            limits.append("change_list_incomplete")
        if report["missing_context"]:
            limits.append("necessary_context_missing")
        if not self.changes and not any(
            e["metadata"]["kind"] == "list" for e in self.events.values()
        ):
            limits.append("change_list_unread")
        return {
            "complete": not missing and not limits,
            "total_changes": len(self.changes),
            "reviewed_changes": len(covered),
            "covered": covered,
            "missing": missing,
            "limitations": limits,
        }

    def authority(self, report):
        full_coverage = self.coverage(report)
        coverage = {
            **full_coverage,
            "covered": full_coverage["covered"][:32],
            "missing": [
                {**item, "reason": item["reason"][:256]}
                for item in full_coverage["missing"][:32]
            ],
            "missing_count": len(full_coverage["missing"]),
            "details_truncated": len(full_coverage["covered"]) > 32
            or len(full_coverage["missing"]) > 32,
            "coverage_digest": fingerprint(full_coverage),
        }
        conclusion = (
            "该比较范围没有变化"
            if not self.changes
            else "发现问题"
            if report["findings"]
            else "在本次范围内未发现有证据支持的问题"
            if coverage["complete"]
            else "审查覆盖不完整，不能给出完整的无问题结论"
        )
        return {
            "task_id": str(self.row["task_id"]),
            "run_id": str(self.row["id"]),
            "snapshot_id": str(self.row["_snapshot_id"]),
            "commit_sha": self.manifest["commit_sha"],
            "content_digest": self.range["content_digest"],
            "git_digest": self.manifest["git_digest"],
            "range": self.range,
            "coverage": coverage,
            "conclusion": conclusion,
            "finding_count": len(report["findings"]),
            "checks": list(self.checks.values()),
            "reproductions": list(self.reproductions.values()),
            "execution_notice": "未做执行验证"
            if not self.checks and not self.reproductions
            else "执行仅针对 head；套件通过不证明没有问题，套件失败不自动复现单个问题。",
            "introduction_notice": "引入依据来自静态前后对照；未执行 base，没有运行前后对照。",
            "reproduction_notice": "复现是模型编写的观察脚本产生的执行证据，不自动证明问题由本次改动引入。",
            "diff_notice": self.document["notice"],
        }

    def response_schema(self):
        ledger = self

        class ValidatedReport(ReviewReportV1):
            @model_validator(mode="after")
            def valid_evidence(self):
                ledger.validate(self)
                return self

        ValidatedReport.__name__ = ValidatedReport.__qualname__ = "ReviewReportV1"
        return ValidatedReport

    async def verify_persisted(self, report):
        async with self.store.pool.connection() as conn:
            fresh = await load_ledger(self.store, conn, self.row)
        raw = fresh.validate(report)
        if fresh.authority(raw) != self.authority(raw):
            raise CodeFailure("REPORT_INVALID", "Persisted Review evidence differs")
        return raw


async def load_ledger(store, conn, row):
    saved = await (
        await conn.execute(
            "SELECT * FROM agent.code_review_ranges WHERE run_id=%s", (row["id"],)
        )
    ).fetchone()
    if not saved:
        raise CodeFailure("REPORT_INVALID", "Review range missing")
    artifact = await (
        await conn.execute(
            "SELECT * FROM agent.code_artifacts WHERE id=%s AND state='available'",
            (saved["changes_artifact_id"],),
        )
    ).fetchone()
    if not artifact or artifact["run_id"] != row["id"]:
        raise CodeFailure("REPORT_INVALID", "Fixed diff artifact unavailable")
    document = json.loads(
        store.artifact_storage.read(
            artifact["storage_key"], artifact["sha256"], artifact["size_bytes"]
        )
    )
    if document["range"] != saved["metadata"]:
        raise CodeFailure("REPORT_INVALID", "Fixed diff range metadata differs")
    snap = await (
        await conn.execute(
            "SELECT * FROM agent.code_snapshots WHERE id=%s AND state='sealed'",
            (saved["snapshot_id"],),
        )
    ).fetchone()
    if not snap or snap["content_digest"] != document["range"]["content_digest"]:
        raise CodeFailure("REPORT_INVALID", "Fixed head snapshot differs")
    import asyncio
    from agent_service.code_execution import verify_snapshot

    directory = store.artifact_storage.path(snap["storage_key"])
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != snap["manifest_digest"]:
        raise CodeFailure("REPORT_INVALID", "Sealed manifest digest mismatch")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("review") != saved["metadata"]
        or manifest["content_digest"] != snap["content_digest"]
        or manifest["git_digest"] != snap["git_digest"]
    ):
        raise CodeFailure("REPORT_INVALID", "Sealed range or head digest mismatch")
    await asyncio.to_thread(
        verify_snapshot,
        directory / "source",
        manifest,
        store.registry.review.settings.model_dump(),
    )
    ledger = ReviewLedger(store, {**row, "_snapshot_id": snap["id"]}, snap, document)
    events = await (
        await conn.execute(
            "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND kind='review_evidence' AND state='available' ORDER BY created_at,id",
            (row["id"],),
        )
    ).fetchall()
    receipts = await (
        await conn.execute(
            "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND kind='review_delivery' AND state='available'",
            (row["id"],),
        )
    ).fetchall()
    delivered = {}
    for receipt in receipts:
        value = json.loads(
            store.artifact_storage.read(
                receipt["storage_key"], receipt["sha256"], receipt["size_bytes"]
            )
        )
        if (
            value["delivered"]
            and receipt["snapshot_id"] == snap["id"]
            and receipt["metadata"]["range_id"] == saved["metadata"]["range_id"]
        ):
            delivered[value["evidence_id"]] = value["content_sha256"]
    for event in events:
        if delivered.get(str(event["id"])) != event["sha256"]:
            continue
        value = json.loads(
            store.artifact_storage.read(
                event["storage_key"], event["sha256"], event["size_bytes"]
            )
        )
        if (
            value["metadata"] != event["metadata"]
            or event["snapshot_id"] != snap["id"]
            or value["metadata"]["range_id"] != saved["metadata"]["range_id"]
            or value["metadata"]["count"] != len(value["records"])
        ):
            raise CodeFailure("REPORT_INVALID", "Stored delivered evidence differs")
        ledger.events[str(event["id"])] = value
    for table, collection in [
        ("code_check_runs", ledger.checks),
        ("code_review_reproductions", ledger.reproductions),
    ]:
        from psycopg import sql

        records = await (
            await conn.execute(
                sql.SQL(
                    "SELECT * FROM agent.{} WHERE run_id=%s ORDER BY created_at,id"
                ).format(sql.Identifier(table)),
                (row["id"],),
            )
        ).fetchall()
        for record in records:
            if (
                record["status"] == "running"
                or record["snapshot_id"] != snap["id"]
                or record["result_digest"] != fingerprint(record["result"])
            ):
                raise CodeFailure(
                    "REPORT_INVALID", "Execution unfinished or result differs"
                )
            if table == "code_review_reproductions":
                script = await (
                    await conn.execute(
                        "SELECT * FROM agent.code_artifacts WHERE id=%s AND state='available'",
                        (record["script_artifact_id"],),
                    )
                ).fetchone()
                if (
                    record["content_digest"] != snap["content_digest"]
                    or record["runner_version"] != "python-observation-v1"
                    or not script
                    or script["run_id"] != row["id"]
                    or script["snapshot_id"] != snap["id"]
                    or script["sha256"] != record["script_digest"]
                    or script["metadata"].get("hypothesis") != record["hypothesis"]
                    or script["metadata"].get("content_digest")
                    != snap["content_digest"]
                ):
                    raise CodeFailure(
                        "REPORT_INVALID", "Reproduction script or head binding differs"
                    )
                store.artifact_storage.read(
                    script["storage_key"], script["sha256"], script["size_bytes"]
                )
            collection[str(record["id"])] = execution_record(record)
            if table == "code_check_runs":
                collection[str(record["id"])]["content_digest"] = snap["content_digest"]
    return ledger


def execution_record(record):
    result = {
        k: str(record[k]) if record[k] is not None else None
        for k in ("id", "snapshot_id", "log_artifact_id")
    }
    result.update(result=record["result"], result_digest=record["result_digest"])
    if "check_id" in record:
        result["check_id"] = record["check_id"]
    else:
        from agent_service.output import limit_output

        details, truncated = limit_output(record["result"].get("details", ""), 1024, 20)
        result["result"] = {
            **record["result"],
            "details": details,
            "details_truncated": truncated,
        }
        result["result_artifact_id"] = (
            str(record["result_artifact_id"]) if record["result_artifact_id"] else None
        )
        result.update(
            {
                k: record[k]
                for k in (
                    "hypothesis",
                    "script_digest",
                    "runner_version",
                    "content_digest",
                )
            }
        )
        result["script_artifact_id"] = str(record["script_artifact_id"])
    return result


def markdown_review(report, authority):
    lines = [
        "# 代码 Review 报告",
        "",
        authority["conclusion"],
        "",
        report["summary"],
        "",
        "## 固定范围",
        encoded(authority["range"]),
        "",
        authority["diff_notice"],
        "",
        authority["introduction_notice"],
        "",
        "## 问题",
    ]
    for finding in report["findings"]:
        loc = finding["location"]
        lines += [
            f"### [{finding['priority']}] {finding['title']}",
            f"{loc['side']}:{loc['path']}:{loc['line'] or '文件级'}；change_id={loc['change_id']}",
            f"触发：{finding['trigger']}",
            f"原因：{finding['cause']}",
            f"影响：{finding['impact']}",
            f"引入依据：{finding['introduction_basis']}",
            f"复现：{finding['reproduction_status']}",
            f"建议：{finding['recommendation']}",
            "证据：" + encoded(finding["evidence"]),
            "差异证据：" + encoded(finding["diff_evidence_ids"]),
            "复现记录：" + encoded(finding["reproduction_ids"]),
            "具体检查用例：" + encoded(finding["check_case_refs"]),
            "限制：" + encoded(finding["limitations"]),
            "",
        ]
    lines += [
        "",
        "## 差异覆盖",
        "完整" if authority["coverage"]["complete"] else "覆盖不完整",
        encoded(authority["coverage"]),
    ]
    for change in report["changes"]:
        lines.append(f"- {change['change_id']}：{change['status']}；{change['reason']}")
    lines += ["", "## 执行记录", authority["execution_notice"]]
    if not authority["checks"]:
        lines.append("未运行注册检查。")
    if not authority["reproductions"]:
        lines.append("未运行临时复现。")
    lines += [
        encoded(record) for record in authority["checks"] + authority["reproductions"]
    ]
    lines += ["", authority["reproduction_notice"], "", "## 未确认事项"]
    lines += ["- " + item for item in report["unconfirmed"] + report["missing_context"]]
    return "\n".join(lines) + "\n"
