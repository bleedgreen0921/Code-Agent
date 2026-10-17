"""Run-owned source evidence and structural report validation (not semantic proof)."""

import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import model_validator

from agent_service.code_artifacts import CodeFailure
from agent_service.profiles import fingerprint
from contracts.code import AnalysisReportV1


def analysis_skill():
    from agent_service.skills import Skill

    body = (
        (
            Path(__file__).resolve().parents[1]
            / "analysis_skills/isolated-analysis/SKILL.md"
        )
        .read_text()
        .split("---", 2)[-1]
        .strip()
    )
    return Skill(
        "isolated-analysis",
        "Analyze a sealed Git snapshot with registered checks and verifiable references.",
        body,
        hashlib.sha256(body.encode()).hexdigest(),
        "{}",
    )


class EvidenceLedger:
    def __init__(self, store, row, manifest):
        self.store, self.row, self.manifest = store, row, manifest
        self.sources = {}
        self.checks = {}

    async def read_result(self, name, args, result, settings):
        # Reserve headers before recording actual delivered complete lines. A cut
        # line never becomes evidence; searches do not create source evidence.
        path = args["path"]
        if path.startswith("/workspace/"):
            path = path[len("/workspace/") :]
        path = path.lstrip("/")
        parts = [part for part in path.split("/") if part != "."]
        path = "/".join(parts)
        start = args.get("start_line", 1)
        if name == "read_file":
            info = self.manifest["manifest"].get(path)
            if not info or result.get("sha256") != info["sha256"]:
                raise CodeFailure(
                    "SNAPSHOT_INVALID", "Read file does not match sealed snapshot"
                )
            lines = result["content"].splitlines()
            source, commit, blob = "workspace", self.manifest["commit_sha"], None
        else:
            if result.get("source_changed"):
                raise CodeFailure("SNAPSHOT_INVALID", "Sealed Git view changed")
            lines = [f"{r['line']}: {r['text']}" for r in result.get("records", [])]
            source, commit, blob = (
                result["source"],
                result.get("commit"),
                result.get("blob"),
            )
        retained = []
        budget = settings.output_bytes - 1800
        for line in lines[: settings.output_lines - 12]:
            if len(("\n".join([*retained, line])).encode()) > budget:
                break
            retained.append(line)
        end = start + len(retained) - 1
        evidence_id = uuid4()
        meta = {
            "evidence_id": str(evidence_id),
            "run_id": str(self.row["id"]),
            "snapshot_id": str(self.row["_snapshot_id"]),
            "path": path,
            "source": source,
            "commit": commit,
            "blob": blob,
            "sha256": result["sha256"],
            "start_line": start,
            "end_line": end,
            "truncated": len(retained) < len(lines) or result.get("truncated", False),
        }
        data = json.dumps(
            {"metadata": meta, "content": "\n".join(retained)},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        await self.store.put_code_artifact(
            self.row,
            "source_evidence",
            data,
            metadata=meta,
            artifact_id=evidence_id,
            maximum=32768,
        )
        self.sources[str(evidence_id)] = meta
        content = json.dumps(meta, ensure_ascii=False) + "\n" + "\n".join(retained)
        return content, meta

    def validate(self, value):
        report = (
            value
            if isinstance(value, AnalysisReportV1)
            else AnalysisReportV1.model_validate(value)
        )
        raw = report.model_dump(mode="json")
        if (
            len(json.dumps(raw, ensure_ascii=False).encode())
            > self.store.registry.code.settings.report_bytes
        ):
            raise ValueError("Report exceeds 64 KiB or configured report limit")
        refs = [v.source for v in report.implementation_locations]
        refs += [
            ref for relation in report.call_relationships for ref in relation.evidence
        ]
        refs += [
            ref
            for conclusion in report.behavior_conclusions
            for ref in conclusion.evidence
        ]
        for ref in refs:
            source = self.sources.get(str(ref.evidence_id))
            if (
                not source
                or source["snapshot_id"] != str(self.row["_snapshot_id"])
                or source["run_id"] != str(self.row["id"])
            ):
                raise ValueError(
                    "Source evidence does not belong to this Run and snapshot"
                )
            if (
                ref.path != source["path"]
                or not source["start_line"]
                <= ref.start_line
                <= ref.end_line
                <= source["end_line"]
            ):
                raise ValueError(
                    "Reference must use an actually read path and complete line range"
                )
            if ref.source != source["source"] or ref.commit != (
                source["commit"] if source["source"] == "commit" else None
            ):
                raise ValueError(
                    "Source version mismatch; historical/index evidence must be identified explicitly"
                )
        for conclusion in report.behavior_conclusions:
            if conclusion.basis == "execution" and not conclusion.check_run_ids:
                raise ValueError("Execution conclusions require real check references")
            if (
                conclusion.basis == "source"
                and conclusion.classification == "fact"
                and not conclusion.evidence
            ):
                raise ValueError("Source facts require read evidence")
            if conclusion.basis == "context" and conclusion.classification == "fact":
                raise ValueError(
                    "Historical context is not verified current-snapshot fact"
                )
            for check_id in conclusion.check_run_ids:
                record = self.checks.get(str(check_id))
                if not record or record["snapshot_id"] != str(self.row["_snapshot_id"]):
                    raise ValueError("Unknown check execution or snapshot mismatch")
        return raw

    def response_schema(self):
        ledger = self

        class ValidatedReport(AnalysisReportV1):
            @model_validator(mode="after")
            def valid_evidence(self):
                ledger.validate(self)
                return self

        ValidatedReport.__name__ = "AnalysisReportV1"
        ValidatedReport.__qualname__ = "AnalysisReportV1"
        return ValidatedReport

    async def verify_persisted(self, report):
        result = self.validate(report)
        for identity, meta in self.sources.items():
            artifact, data = await self.store.code_artifact(
                self.row["task_id"], UUID(identity)
            )
            if (
                artifact["run_id"] != self.row["id"]
                or json.loads(data)["metadata"] != meta
            ):
                raise CodeFailure("REPORT_INVALID", "Source evidence storage mismatch")
        async with self.store.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM agent.code_check_runs WHERE run_id=%s",
                    (self.row["id"],),
                )
            ).fetchall()
        for row in rows:
            current = self.checks.get(str(row["id"]))
            if (
                not current
                or row["result_digest"] != fingerprint(current["result"])
                or row["result"] != current["result"]
            ):
                raise CodeFailure("REPORT_INVALID", "Check evidence storage mismatch")
        return result

    def authority(self):
        return {
            "task_id": str(self.row["task_id"]),
            "run_id": str(self.row["id"]),
            "snapshot_id": str(self.row["_snapshot_id"]),
            "commit_sha": self.manifest["commit_sha"],
            "content_digest": self.manifest["content_digest"],
            "git_digest": self.manifest["git_digest"],
            "checks": list(self.checks.values()),
            "checks_configured": bool(self.row["_code"]["check_ids"]),
            "evidence_notice": "References and execution records are verified; model inferences are not automatically proven.",
        }


def markdown_report(report, authority):
    lines = [
        "# 代码分析报告",
        "",
        report["scope"]["question"],
        "",
        f"快照：{authority['snapshot_id']}；提交：{authority['commit_sha'] or '无提交'}",
        "",
        "## 分析范围",
    ]
    lines += ["- " + v for v in report["scope"]["covered"]]
    lines += ["- 未覆盖：" + v for v in report["scope"]["missing"]]
    lines += ["", "## 实现位置与调用关系"]
    for location in report["implementation_locations"]:
        ref = location["source"]
        lines.append(
            f"- {ref['path']}:{ref['start_line']}–{ref['end_line']} `{location['symbol']}`：{location['responsibility']}（{ref['source']}；证据 {ref['evidence_id']}）"
        )
    for relation in report["call_relationships"]:
        lines.append(
            f"- {relation['caller']} → {relation['callee']}：{relation['explanation']}"
        )
    lines += ["", "## 行为结论"]
    for conclusion in report["behavior_conclusions"]:
        lines.append(
            f"- [{conclusion['classification']}] {conclusion['conclusion']}；条件：{conclusion['conditions']}"
        )
    for heading, key in [
        ("执行路径", "execution_paths"),
        ("异常路径", "exception_paths"),
        ("未确认事项", "unconfirmed"),
    ]:
        lines += ["", "## " + heading, *["- " + v for v in report[key]]]
    lines += ["", "## 服务端检查记录"]
    if not authority["checks_configured"]:
        lines.append("未配置检查集；未执行检查。")
    elif not authority["checks"]:
        lines.append("已配置检查集；未执行检查。")
    for check in authority["checks"]:
        result = check["result"]
        lines.append(
            f"- {check['check_id']}：**{result['status']}**；执行 {result['executed']}；失败 {result['failed']}；错误 {result['errors']}；跳过 {result['skipped']}；限制：{', '.join(result['limitations']) or '无'}；检查记录 {check['id']}"
        )
    lines += ["", authority["evidence_notice"], ""]
    return "\n".join(lines)
