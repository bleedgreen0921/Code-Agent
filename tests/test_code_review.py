"""Real Git, PostgreSQL and Docker Review acceptance; model calls are synthetic."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import httpx
import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from agent_service.app import app
from agent_service.code_registry import CodeReviewSettings, REVIEW_PROFILE
from agent_service.profiles import Registry, RuntimeFile, Workspace
from agent_service.review_execution import classify_observation
from agent_service.review_reports import ReviewLedger
from agent_service.worker import Worker
from contracts.review import CodeReviewCreate
from execution.code_snapshot import SnapshotError
from execution.code_review import prepare_review, read_blob
from tests.models import SyntheticModel
from tests.test_code_analysis import fixture, files, git, call
from tests.test_core import execute


def review_fixture(tmp_path, **settings):
    old, source, storage, commit = fixture(tmp_path)
    config = old.code.settings.model_dump(
        exclude={"profile", "model_calls", "tool_calls", "run_seconds"}
    )
    config.update(settings)
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="project", root=str(source))],
            execution=old.execution,
            code_analysis=old.code.settings,
            code_review=CodeReviewSettings(**config),
        ),
        project_checks=tuple(old.code.checks.values()),
    )
    return registry, source, storage, commit


def local_repo(tmp_path):
    # No Docker image required for trusted raw-Git preparation unit tests.
    source = tmp_path / "repo"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.email", "review@example.invalid")
    git(source, "config", "user.name", "Review")
    (source / "sample.py").write_text("def answer():\n    return 1\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "base")
    return source


def prepare_local(source, output, target, **limits):
    output.parent.mkdir(parents=True, exist_ok=True)
    return prepare_review(
        source, output, target, CodeReviewSettings(**limits).model_dump()
    )


@pytest.mark.parametrize(
    "mode,expected,changed",
    [("all", 3, True), ("staged", 2, True), ("unstaged", 3, True)],
)
def test_workspace_modes_are_raw_and_original_immutable(
    tmp_path, mode, expected, changed
):
    source = local_repo(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    git(source, "add", "sample.py")
    (source / "sample.py").write_text("def answer():\n    return 3\n")
    (source / "context.txt").write_text("untracked\n")
    before = files(source)
    output = tmp_path / "managed/source"
    manifest = prepare_local(source, output, {"kind": "workspace", "mode": mode})
    assert (output / "sample.py").read_text().endswith(f"return {expected}\n")
    assert (output / "context.txt").exists() == (mode != "staged")
    doc = json.loads((output.parent / "changes.json").read_text())
    assert any(c["head_path"] == "context.txt" for c in doc["changes"]) == (
        mode == "all"
    )
    assert (
        read_blob(output, manifest["review"], "head", "sample.py", 10000)["lines"][-1]
        == f"    return {expected}"
    )
    assert files(source) == before


def test_net_change_and_unborn_index(tmp_path):
    source = local_repo(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    git(source, "add", ".")
    (source / "sample.py").write_text("def answer():\n    return 1\n")
    for mode in ["all", "staged", "unstaged"]:
        output = tmp_path / mode / "source"
        prepare_local(source, output, {"kind": "workspace", "mode": mode})
        changes = json.loads((output.parent / "changes.json").read_text())["changes"]
        assert bool(changes) == (mode != "all")
    empty = tmp_path / "unborn"
    empty.mkdir()
    git(empty, "init", "-q")
    (empty / "new.txt").write_text("new\n")
    git(empty, "add", ".")
    for mode in ["all", "staged", "unstaged"]:
        output = tmp_path / ("unborn-" + mode) / "source"
        prepare_local(empty, output, {"kind": "workspace", "mode": mode})
        assert bool(
            json.loads((output.parent / "changes.json").read_text())["changes"]
        ) == (mode != "unstaged")


def test_historical_root_merge_and_invalid_current_index(tmp_path):
    source = local_repo(tmp_path)
    root = git(source, "rev-parse", "HEAD").decode().strip()
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    git(source, "commit", "-qam", "C")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    (source / "sample.py").write_text("def answer():\n    return 3\n")
    git(source, "commit", "-qam", "D")
    # Conflict entries in the unrelated current index must not affect historical C.
    (source / "link").symlink_to("sample.py")
    git(source, "add", "link")
    before = files(source)
    for ref, value, parent_count in [(root, 1, 0), (commit, 2, 1)]:
        output = tmp_path / ref / "source"
        m = prepare_local(source, output, {"kind": "commit", "ref": ref})
        assert m["commit_sha"] == ref
        assert m["review"]["parent_count"] == parent_count
        assert (output / "sample.py").read_text().endswith(f"return {value}\n")
        assert not (output / "link").exists()
    assert files(source) == before
    git(source, "reset", "--hard", "-q", "HEAD")
    git(source, "checkout", "-qb", "side", root)
    (source / "side.txt").write_text("side\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "side")
    git(source, "checkout", "-q", "master")
    git(source, "merge", "--no-ff", "-qm", "merge", "side")
    merged = git(source, "rev-parse", "HEAD").decode().strip()
    m = prepare_local(
        source, tmp_path / "merge/source", {"kind": "commit", "ref": merged}
    )
    assert m["review"]["parent_count"] == 2
    assert (
        m["review"]["selected_parent"]
        == git(source, "rev-parse", "HEAD^1").decode().strip()
    )


def test_ranges_and_merge_base_failures(tmp_path):
    import subprocess

    source = local_repo(tmp_path)
    root = git(source, "rev-parse", "HEAD").decode().strip()
    (source / "a").write_text("a")
    git(source, "add", ".")
    git(source, "commit", "-qm", "left")
    left = git(source, "rev-parse", "HEAD").decode().strip()
    git(source, "checkout", "-qb", "right", root)
    (source / "b").write_text("b")
    git(source, "add", ".")
    git(source, "commit", "-qm", "right")
    right = git(source, "rev-parse", "HEAD").decode().strip()
    before = files(source)
    for comparison, expected in [("direct", left), ("merge_base", root)]:
        m = prepare_local(
            source,
            tmp_path / comparison / "source",
            {"kind": "range", "base": left, "head": right, "comparison": comparison},
        )
        assert m["review"]["comparison_base_commit"] == expected
        assert m["review"]["resolved_base_commit"] == left
    m = prepare_local(
        source,
        tmp_path / "equal/source",
        {"kind": "range", "base": right, "head": right},
    )
    assert json.loads((tmp_path / "equal/changes.json").read_text())["changes"] == []
    tree = git(source, "rev-parse", "HEAD^{tree}").decode().strip()

    def commit_tree(*parents):
        args = ["git", "-C", str(source), "commit-tree", tree]
        for parent in parents:
            args += ["-p", parent]
        return subprocess.check_output(args, input=b"test\n").decode().strip()

    orphan = commit_tree()
    a, b = commit_tree(left, right), commit_tree(right, left)
    for base, head in [(orphan, right), (a, b)]:
        with pytest.raises(SnapshotError, match="exactly one"):
            prepare_local(
                source,
                tmp_path / head / "source",
                {
                    "kind": "range",
                    "base": base,
                    "head": head,
                    "comparison": "merge_base",
                },
            )
    # commit-tree only adds immutable objects; refs, index and source stay unchanged.
    assert {
        k: v for k, v in files(source).items() if not k.startswith(".git/objects/")
    } == {k: v for k, v in before.items() if not k.startswith(".git/objects/")}


def test_diff_metadata_binary_long_lines_and_limits(tmp_path):
    source = local_repo(tmp_path)
    (source / "empty").touch()
    (source / "old").write_text("renamed\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "more")
    (source / "old").rename(source / "new")
    (source / "empty").unlink()
    (source / "sample.py").chmod(0o755)
    (source / "binary").write_bytes(b"\0\xff")
    (source / "long").write_text("a" * 9000)
    output = tmp_path / "managed/source"
    prepare_local(source, output, {"kind": "workspace", "mode": "all"})
    changes = json.loads((output.parent / "changes.json").read_text())["changes"]
    paths = {c["head_path"] or c["base_path"]: c for c in changes}
    assert paths["old"]["status"] == "D" and paths["new"]["status"] == "A"
    assert paths["empty"]["file_location"]
    assert paths["sample.py"]["file_location"]
    assert "binary_content" in paths["binary"]["limitations"]
    assert "long_line" in paths["long"]["limitations"]
    prepare_local(
        source,
        tmp_path / "limit/source",
        {"kind": "workspace", "mode": "all"},
        patch_bytes=10,
    )
    assert all(
        c["limitations"]
        for c in json.loads((tmp_path / "limit/changes.json").read_text())["changes"]
    )


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "workspace", "mode": "bad"},
        {"kind": "range", "base": "-evil", "head": "HEAD"},
        {"kind": "range", "base": "HEAD", "head": "HEAD", "comparison": "auto"},
        {"kind": "commit", "ref": "HEAD..main"},
    ],
)
def test_request_rejects_ambiguous_targets(target):
    with pytest.raises(ValidationError):
        CodeReviewCreate(task="review", workspace_id="project", target=target)


@pytest.mark.parametrize(
    "value,options,status",
    [
        ({"observed": True, "details": "yes"}, {}, "observed"),
        ({"observed": False, "details": "no"}, {}, "not_observed"),
        ({"observed": True, "details": "yes"}, {"timed_out": True}, "inconclusive"),
        ({"observed": True, "details": "yes"}, {"oom": True}, "inconclusive"),
        ({"observed": True, "details": "yes"}, {"truncated": True}, "inconclusive"),
    ],
)
def test_observation_classification(value, options, status):
    raw = json.dumps(
        {
            "protocol": 1,
            "runner_version": "python-observation-v1",
            "script_digest": "abc",
            "status": "observed" if value["observed"] else "not_observed",
            "complete": True,
            "details": value["details"],
            "limitations": [],
        }
    ).encode()
    assert (
        classify_observation(raw, exit_code=0, script_digest="abc", **options)["status"]
        == status
    )
    assert (
        classify_observation(b'{"observed":true}', exit_code=0, script_digest="abc")[
            "status"
        ]
        == "inconclusive"
    )


def report(changes, reads, *, partial=False, finding=None):
    refs = [
        {
            "evidence_id": r.artifact["evidence_id"],
            "side": r.artifact["side"],
            "path": r.artifact["path"],
            "start_line": 1,
            "end_line": 2,
        }
        for r in reads
    ]
    return {
        "version": 1,
        "summary": "Review the fixed change",
        "findings": [finding] if finding else [],
        "changes": [
            {
                "change_id": c["change_id"],
                "status": "partial" if partial else "reviewed",
                "reason": "Compared behavior",
                "evidence": [
                    r for r in refs if r["path"] in (c["base_path"], c["head_path"])
                ],
                "missing_context": [],
            }
            for c in changes
        ],
        "unconfirmed": [],
        "missing_context": [],
    }


@pytest.mark.parametrize(
    "scenario",
    [
        "historical",
        "partial_finding",
        "static",
        "clean",
        "partial",
        "none",
        "observed",
        "not_observed",
        "exception",
        "invalid",
        "timeout",
        "stdout",
    ],
)
def test_review_end_to_end(database, tmp_path, monkeypatch, scenario):
    registry, source, storage, base = review_fixture(
        tmp_path, check_seconds=1 if scenario == "timeout" else 60
    )
    (source / "sample.py").write_text(
        "def answer():\n    return 1 + 0\n"
        if scenario in ("static", "clean")
        else "def answer():\n    return 2\n"
    )
    if scenario == "partial_finding":
        (source / "z_unread.txt").write_text("Necessary review remains\n")
        git(source, "add", "z_unread.txt")
    git(source, "commit", "-qam", "C")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    target = {"kind": "commit", "ref": commit}
    if scenario == "historical":
        (source / "sample.py").write_text("def answer():\n    return 99\n")
        git(source, "commit", "-qam", "D")
    if scenario == "none":
        target = {"kind": "range", "base": commit, "head": commit}
    before = files(source)
    ran_execution = scenario in (
        "historical",
        "partial_finding",
        "observed",
        "not_observed",
        "exception",
        "invalid",
        "timeout",
        "stdout",
    )
    status = (
        "observed"
        if scenario in ("historical", "partial_finding", "observed", "stdout")
        else "not_observed"
        if scenario == "not_observed"
        else "inconclusive"
    )
    invalid_sent = False

    async def check(resources):
        nonlocal invalid_sent
        resources.registry = resources.store.registry = registry
        monkeypatch.setattr(app.state, "resources", resources, raising=False)

        async def reply(messages):
            nonlocal invalid_sent
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            lists = [
                m
                for m in tools
                if m.name == "review_diff"
                and m.artifact
                and m.artifact.get("kind") == "list"
            ]
            if not lists:
                return call("review_diff", {}, "list")
            changes = json.loads(lists[-1].text)["records"]
            if changes and scenario != "partial":
                patches = [
                    m
                    for m in tools
                    if m.name == "review_diff"
                    and m.artifact
                    and m.artifact.get("kind") == "patch"
                ]
                if not patches:
                    return call(
                        "review_diff", {"change_id": changes[0]["change_id"]}, "patch"
                    )
                reads = [m for m in tools if m.name == "review_read_file"]
                if len(reads) < 2:
                    return call(
                        "review_read_file",
                        {"side": "base" if not reads else "head", "path": "sample.py"},
                        "read-" + str(len(reads)),
                    )
                assert "return 1" in reads[0].text
                assert (
                    "return 1 + 0" if scenario in ("static", "clean") else "return 2"
                ) in reads[1].text
            else:
                reads, patches = [], []
            checks = [m for m in tools if m.name == "run_project_check"]
            if scenario == "clean" and not checks:
                return call("run_project_check", {"check_id": "unit"}, "check")
            if scenario == "clean":
                assert checks[-1].artifact["result"]["status"] == "passed"
            runs = [m for m in tools if m.name == "run_review_reproduction"]
            if ran_execution and not runs:
                scripts = {
                    "exception": 'def reproduce():\n    raise RuntimeError("not an observation")\n',
                    "invalid": 'def reproduce():\n    return {"observed": "true", "details":"bad"}\n',
                    "timeout": 'import time\ndef reproduce():\n    time.sleep(10)\n    return {"observed":True,"details":"too late"}\n',
                    "stdout": 'import os\nprint(\'{"status":"not_observed"}\')\nos.write(1,b\'{"status":"not_observed"}\\n\')\ndef reproduce():\n    return {"observed":True,"details":"actual return"}\n',
                }
                script = scripts.get(
                    scenario,
                    'from sample import answer\ndef reproduce():\n    return {"observed": answer() == '
                    + ("99" if scenario == "not_observed" else "2")
                    + ', "details":str(answer())}\n',
                )
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "answer returns two", "script": script},
                    "repro",
                )
            if runs:
                execution = json.loads(runs[-1].text)["records"][0]
                assert execution["result"]["status"] == status
            value = report(changes, reads, partial=scenario == "partial")
            if scenario == "partial_finding":
                value["changes"][1].update(
                    status="unreviewed",
                    reason="Budget reserved for supported finding",
                    evidence=[],
                )
            if scenario in ("historical", "partial_finding"):
                refs = value["changes"][0]["evidence"]
                finding = {
                    "title": "Returns unexpected two",
                    "priority": "P2",
                    "location": {
                        "change_id": changes[0]["change_id"],
                        "side": "head",
                        "path": "sample.py",
                        "line": 2,
                    },
                    "trigger": "Call answer()",
                    "cause": "Constant changed from one to two",
                    "impact": "Callers expecting one fail",
                    "introduction_basis": "Static comparison of base and head return values",
                    "limitations": [],
                    "diff_evidence_ids": [patches[0].artifact["evidence_id"]],
                    "evidence": refs,
                    "reproduction_status": "observed",
                    "reproduction_ids": [execution["id"]],
                    "check_case_refs": [],
                    "recommendation": "Verify the expected contract",
                }
                if not invalid_sent:
                    invalid_sent = True
                    finding["location"]["line"] = 1
                    return call(
                        "ReviewReportV1", {**value, "findings": [finding]}, "bad-report"
                    )
                assert any(
                    "actual changed" in m.text
                    for m in tools
                    if m.tool_call_id == "bad-report"
                )
                value["findings"] = [finding]
            return call("ReviewReportV1", value, "report")

        worker = Worker(resources, model=SyntheticModel(reply=reply))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            payload = {
                "stage": "review",
                "task": "Review fixed change",
                "workspace_id": "project",
                "target": target,
            }
            response = await client.post(
                "/v1/code/tasks", json=payload, headers={"Idempotency-Key": "review"}
            )
            assert response.status_code == 202, response.text
            task_id = response.json()["task_id"]
            assert (
                await client.post(
                    "/v1/code/tasks",
                    json=payload,
                    headers={"Idempotency-Key": "review"},
                )
            ).status_code == 200
            assert (
                await client.post(
                    "/v1/code/tasks",
                    json={**payload, "task": "different"},
                    headers={"Idempotency-Key": "review"},
                )
            ).status_code == 409
            assert (
                await client.post(
                    "/v1/tasks",
                    json={
                        "task": "bypass",
                        "profile": REVIEW_PROFILE,
                        "workspace_id": "project",
                    },
                )
            ).status_code == 422
            await execute(resources, worker)
            detail = (await client.get("/v1/code/tasks/" + task_id)).json()
            assert detail["status"] == "completed", detail
            assert detail["stage"] == "review"
            _, structured = await resources.store.code_artifact(
                __import__("uuid").UUID(task_id),
                __import__("uuid").UUID(detail["report_artifact_id"]),
            )
            document = json.loads(structured)
            authority = document["authority"]
            assert authority["coverage"]["complete"] == (
                scenario not in ("partial", "partial_finding")
            )
            assert len(authority["reproductions"]) == int(ran_execution)
            assert authority["finding_count"] == int(
                scenario in ("historical", "partial_finding")
            )
            if not ran_execution and scenario != "clean":
                assert authority["execution_notice"] == "未做执行验证"
            assert (
                await client.get("/v1/code/tasks/" + task_id + "/changes")
            ).status_code == 200
            assert len(
                (
                    await client.get("/v1/code/tasks/" + task_id + "/reproductions")
                ).json()["items"]
            ) == int(ran_execution)
            artifacts = (
                await client.get("/v1/code/tasks/" + task_id + "/artifacts?limit=100")
            ).json()["items"]
            out = (
                Path(
                    os.environ.get(
                        "AGENT_REVIEW_EXAMPLE_DIR", str(tmp_path / "exports")
                    )
                )
                / scenario
            )
            out.mkdir(parents=True, exist_ok=True)
            for artifact in artifacts:
                _, data = await resources.store.code_artifact(
                    __import__("uuid").UUID(task_id),
                    __import__("uuid").UUID(artifact["id"]),
                )
                suffix = (
                    ".json"
                    if artifact["kind"]
                    in ("review_changes", "review_report", "reproduction_result")
                    else ".md"
                    if artifact["kind"] == "review_markdown"
                    else ".py"
                    if artifact["kind"] == "reproduction_script"
                    else ".txt"
                )
                (out / (artifact["kind"] + "-" + artifact["id"] + suffix)).write_bytes(
                    data
                )
            assert not await resources.store.code_resources()

    database(check)
    assert files(source) == before


def test_staged_ignores_unsupported_unstaged_content(tmp_path):
    source = local_repo(tmp_path)
    (source / "sample.py").unlink()
    (source / "sample.py").symlink_to("/not-used")
    (source / "untracked-link").symlink_to("/not-used")
    before = {p.name: p.readlink() for p in source.iterdir() if p.is_symlink()}
    manifest = prepare_local(
        source, tmp_path / "managed/source", {"kind": "workspace", "mode": "staged"}
    )
    assert manifest["review"]["head_tree"] == manifest["review"]["base_tree"]
    assert (tmp_path / "managed/source/sample.py").read_text().endswith("return 1\n")
    assert {p.name: p.readlink() for p in source.iterdir() if p.is_symlink()} == before


def test_pagination_coverage_and_forged_evidence(tmp_path):
    from copy import deepcopy
    from agent_service.policy import Call, active_call

    source = local_repo(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    manifest = prepare_local(
        source, tmp_path / "managed/source", {"kind": "workspace", "mode": "all"}
    )
    document = json.loads((tmp_path / "managed/changes.json").read_text())

    class Store:
        registry = SimpleNamespace(
            review=SimpleNamespace(settings=CodeReviewSettings()),
            execution=SimpleNamespace(output_bytes=2000, output_lines=4),
        )

        async def put_code_artifact(self, *args, **kwargs):
            return {"id": kwargs.get("artifact_id", uuid4())}

    ledger = ReviewLedger(
        Store(),
        {"id": uuid4(), "task_id": uuid4(), "_snapshot_id": uuid4()},
        manifest,
        document,
    )

    async def delivered(coro, identity):
        token = active_call.set(Call(None, None, identity, "review_diff"))
        try:
            content, artifact = await coro
            await ledger.confirm_delivery(
                identity,
                ToolMessage(
                    content,
                    artifact=artifact,
                    name="review_diff",
                    tool_call_id=identity,
                ),
            )
            return json.loads(content), artifact
        finally:
            active_call.reset(token)

    async def check():
        await delivered(ledger.diff(limit=1), "list")
        change = document["changes"][0]
        identity = change["change_id"]
        value = {
            "version": 1,
            "summary": "Static",
            "findings": [],
            "changes": [
                {
                    "change_id": identity,
                    "status": "reviewed",
                    "reason": "Compared",
                    "evidence": [],
                    "missing_context": [],
                }
            ],
            "unconfirmed": [],
            "missing_context": [],
        }
        assert not ledger.coverage(ledger.validate(value))["complete"]
        page, _ = await delivered(ledger.diff(identity, limit=1), "patch0")
        assert page["next_offset"] == 1
        while page["next_offset"] is not None:
            offset = page["next_offset"]
            page, _ = await delivered(
                ledger.diff(identity, offset=offset, limit=2), "patch" + str(offset)
            )
        assert (
            "patch_or_hunk_unread"
            not in ledger.coverage(value)["missing"][0]["reasons"]
        )
        refs = []
        for side in ["base", "head"]:
            data = read_blob(
                tmp_path / "managed/source",
                manifest["review"],
                side,
                "sample.py",
                10000,
            )
            data["total_lines"] = len(data["lines"])
            data["records"] = [
                {"line": i, "text": t} for i, t in enumerate(data.pop("lines"), 1)
            ]
            _, artifact = await delivered(
                ledger.source({"side": side, "path": "sample.py"}, data),
                "source" + side,
            )
            refs.append(
                {
                    "evidence_id": artifact["evidence_id"],
                    "side": side,
                    "path": "sample.py",
                    "start_line": 1,
                    "end_line": 2,
                }
            )
        value["changes"][0]["evidence"] = refs
        assert ledger.coverage(ledger.validate(value))["complete"]
        finding = {
            "title": "Changed return contract",
            "priority": "P2",
            "location": {
                "change_id": identity,
                "side": "head",
                "path": "sample.py",
                "line": 2,
            },
            "trigger": "Call answer",
            "cause": "Return constant changed",
            "impact": "Caller receives two",
            "introduction_basis": "Base returns one, head returns two",
            "limitations": [],
            "evidence": refs,
            "diff_evidence_ids": [
                key
                for key, event in ledger.events.items()
                if event["metadata"]["kind"] == "patch"
            ],
            "reproduction_status": "not_run",
            "reproduction_ids": [],
            "check_case_refs": [],
            "recommendation": "Check the contract",
        }
        valid_finding = {**value, "findings": [finding]}
        ledger.validate(valid_finding)
        for field, changed, message in [
            ("location", {**finding["location"], "line": 1}, "actual changed"),
            ("location", {**finding["location"], "line": None}, "File-level"),
            ("location", {**finding["location"], "path": "not-in-range.py"}, "outside"),
            ("diff_evidence_ids", [str(uuid4())], "diff evidence"),
            ("reproduction_status", "observed", "execution record"),
            ("check_case_refs", ["whole-suite-failed"], "Specific failing case"),
        ]:
            forged_finding = deepcopy(valid_finding)
            forged_finding["findings"][0][field] = changed
            with pytest.raises(ValueError, match=message):
                ledger.validate(forged_finding)

        value["missing_context"] = ["Required caller context missing"]
        assert not ledger.coverage(ledger.validate(value))["complete"]
        value["missing_context"] = []
        forged = deepcopy(value)
        forged["changes"][0]["evidence"][0]["end_line"] = 3
        with pytest.raises(ValueError, match="actually delivered"):
            ledger.validate(forged)
        forged = deepcopy(value)
        forged["changes"][0]["evidence"][0]["side"] = "head"
        with pytest.raises(ValueError, match="mismatch"):
            ledger.validate(forged)
        forged = deepcopy(value)
        forged["changes"][0]["evidence"][0]["evidence_id"] = str(uuid4())
        with pytest.raises(ValueError, match="Unknown source"):
            ledger.validate(forged)
        # Content modified by outer middleware never counts as delivered evidence.
        ledger.events.clear()
        token = active_call.set(Call(None, None, "mutated", "review_diff"))
        try:
            _, artifact = await ledger.diff()
            await ledger.confirm_delivery(
                "mutated",
                ToolMessage("modified", artifact=artifact, tool_call_id="mutated"),
            )
        finally:
            active_call.reset(token)
        assert not ledger.events
        assert not ledger.coverage(value)["complete"]

    asyncio.run(check())


@pytest.mark.parametrize(
    "scenario",
    [
        "all",
        "staged",
        "unstaged",
        "root",
        "merge",
        "range-direct",
        "range-merge_base",
        "missing-tests",
    ],
)
def test_review_target_copies_checks_and_exports(database, tmp_path, scenario):
    registry, source, _, root = review_fixture(tmp_path)
    expected = 2
    target = (
        {"kind": "workspace", "mode": scenario}
        if scenario in ("all", "staged", "unstaged")
        else {"kind": "commit", "ref": root}
    )
    if scenario in ("all", "staged", "unstaged"):
        (source / "sample.py").write_text("def answer():\n    return 2\n")
        git(source, "add", "sample.py")
        (source / "sample.py").write_text("def answer():\n    return 3\n")
        (source / "context.txt").write_text("Untracked execution context\n")
        expected = 2 if scenario == "staged" else 3
    elif scenario == "missing-tests":
        git(source, "rm", "-qr", "tests")
        git(source, "commit", "-qm", "remove tests")
        target = {
            "kind": "commit",
            "ref": git(source, "rev-parse", "HEAD").decode().strip(),
        }
    elif scenario != "root":
        (source / "sample.py").write_text("def answer():\n    return 2\n")
        git(source, "commit", "-qam", "left")
        left = git(source, "rev-parse", "HEAD").decode().strip()
        git(source, "checkout", "-qb", "side", root)
        (source / "context.txt").write_text("Side context\n")
        git(source, "add", ".")
        git(source, "commit", "-qm", "right")
        if scenario == "merge":
            git(source, "checkout", "-q", "master")
            git(source, "merge", "--no-ff", "-qm", "merge", "side")
            target = {
                "kind": "commit",
                "ref": git(source, "rev-parse", "HEAD").decode().strip(),
            }
        else:
            target = {
                "kind": "range",
                "base": left,
                "head": "side",
                "comparison": scenario.removeprefix("range-"),
            }
            expected = 1
    else:
        expected = 1
    if scenario == "missing-tests":
        expected = 1
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        state = {
            "changes": [],
            "refs": [],
            "list_offset": 0,
            "listed": False,
            "patch": {},
            "read": set(),
            "ran": False,
            "checked": False,
        }

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if tools:
                last = tools[-1]
                if last.name == "review_diff":
                    value = json.loads(last.text)
                    if last.artifact["kind"] == "list":
                        state["changes"] += value["records"]
                        state["list_offset"] = value["next_offset"]
                        state["listed"] = value["next_offset"] is None
                    else:
                        state["patch"][last.artifact["change_id"]] = value[
                            "next_offset"
                        ]
                elif last.name in ("review_read_file", "read_file"):
                    meta = last.artifact
                    records = json.loads(last.text)["records"]
                    if records:
                        state["refs"].append(
                            {
                                "evidence_id": meta["evidence_id"],
                                "side": meta["side"],
                                "path": meta["path"],
                                "start_line": records[0]["line"],
                                "end_line": records[-1]["line"],
                            }
                        )
                    state["read"].add((meta["side"], meta["path"]))
                elif last.name == "run_review_reproduction":
                    assert (
                        json.loads(last.text)["records"][0]["result"]["status"]
                        == "observed"
                    )
                    state["ran"] = True
                elif last.name == "run_project_check":
                    assert last.artifact["result"]["status"] == (
                        "inconclusive"
                        if scenario == "missing-tests"
                        else "passed"
                        if expected == 1
                        else "failed"
                    )
                    state["checked"] = True
            if not state["listed"]:
                return call(
                    "review_diff",
                    {"offset": state["list_offset"], "limit": 1},
                    "list" + str(len(tools)),
                )
            if scenario in ("all", "staged", "unstaged"):
                assert any(
                    c["head_path"] == "context.txt" for c in state["changes"]
                ) == (scenario == "all")
            for change in state["changes"]:
                identity = change["change_id"]
                if (
                    identity not in state["patch"]
                    or state["patch"][identity] is not None
                ):
                    return call(
                        "review_diff",
                        {
                            "change_id": identity,
                            "offset": state["patch"].get(identity, 0),
                            "limit": 3,
                        },
                        "patch" + str(len(tools)),
                    )
                for side in ("base", "head"):
                    path = change[side + "_path"]
                    if path and (side, path) not in state["read"]:
                        return call(
                            "review_read_file",
                            {"side": side, "path": path},
                            "read" + str(len(tools)),
                        )
            if not state["checked"]:
                return call("run_project_check", {"check_id": "unit"}, "check")
            if not state["ran"]:
                script = (
                    'from sample import answer\ndef reproduce():\n    return {"observed":answer()=='
                    + str(expected)
                    + ',"details":str(answer())}\n'
                )
                if scenario == "unstaged":
                    script = "from pathlib import Path\n" + script.replace(
                        "answer()==3",
                        'answer()==3 and Path("context.txt").read_text().startswith("Untracked")',
                    )
                return call(
                    "run_review_reproduction",
                    {"hypothesis": "Reads correct head version", "script": script},
                    "repro",
                )
            return call(
                "ReviewReportV1",
                {
                    "version": 1,
                    "summary": "Compared all changes",
                    "findings": [],
                    "changes": [
                        {
                            "change_id": c["change_id"],
                            "status": "reviewed",
                            "reason": "Read all patches and source context",
                            "evidence": [
                                ref
                                for ref in state["refs"]
                                if ref["path"] in (c["base_path"], c["head_path"])
                            ],
                            "missing_context": [],
                        }
                        for c in state["changes"]
                    ],
                    "unconfirmed": [],
                    "missing_context": [],
                },
                "report",
            )

        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project", target=target)
        )
        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        detail = await r.store.code_task(task["id"])
        assert detail["coverage"]["complete"]
        out = (
            Path(os.environ.get("AGENT_REVIEW_EXAMPLE_DIR", str(tmp_path / "exports")))
            / scenario
        )
        out.mkdir(parents=True, exist_ok=True)
        for artifact in (await r.store.code_page(task["id"], "artifacts", 100))[
            "items"
        ]:
            if artifact["kind"] in (
                "review_report",
                "review_markdown",
                "review_changes",
                "reproduction_script",
                "reproduction_result",
            ):
                _, data = await r.store.code_artifact(task["id"], artifact["id"])
                (out / (artifact["kind"] + ("-" + str(artifact["id"])))).write_bytes(
                    data
                )
        assert not await r.store.code_resources()
        assert files(source) == before

    database(check)


@pytest.mark.parametrize("verification", ["reproduction", "check"])
def test_execution_evidence_pages_and_specific_failure_diagnostic(
    database, tmp_path, verification
):
    registry, source, _, _ = review_fixture(tmp_path)
    (source / "sample.py").write_text("def answer():\n    return 2\n")
    before = files(source)

    async def check(r):
        r.registry = r.store.registry = registry
        state = {}

        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if tools:
                last = tools[-1]
                if last.name == "review_diff":
                    if last.artifact["kind"] == "list":
                        state["change"] = json.loads(last.text)["records"][0]
                    else:
                        state["patch"] = last.artifact["evidence_id"]
                elif last.name in ("read_file", "review_read_file"):
                    meta = last.artifact
                    state[meta["side"]] = {
                        "evidence_id": meta["evidence_id"],
                        "side": meta["side"],
                        "path": "sample.py",
                        "start_line": 1,
                        "end_line": 2,
                    }
                elif last.name == "run_project_check":
                    state["check"] = last.artifact["id"]
                    assert last.artifact["result"]["status"] == "failed"
                elif last.name == "read_review_execution":
                    payload = json.loads(last.text)
                    records = payload["records"]
                    meta = last.artifact
                    if meta["kind"] == "execution_log":
                        state["diagnostic"] = True
                        assert any("AssertionError" in rec["text"] for rec in records)
                    elif meta["execution_id"] == state.get("check"):
                        state["case_page"] = payload["next_offset"]
                        for record in records:
                            if record.get("outcome") == "failed":
                                state["case"] = record["case"]
                    else:
                        state["observation_page"] = payload["next_offset"]
                elif last.name == "run_review_reproduction":
                    state["repro"] = json.loads(last.text)["records"][0]["id"]
            if "change" not in state:
                return call("review_diff", {}, "list")
            if "patch" not in state:
                return call(
                    "review_diff", {"change_id": state["change"]["change_id"]}, "patch"
                )
            if "base" not in state:
                return call(
                    "review_read_file", {"side": "base", "path": "sample.py"}, "base"
                )
            if "head" not in state:
                return call("read_file", {"path": "sample.py"}, "head")
            if "check" not in state:
                return call("run_project_check", {"check_id": "unit"}, "check")
            if "case_page" not in state or state["case_page"] is not None:
                return call(
                    "read_review_execution",
                    {
                        "execution_id": state["check"],
                        "limit": 1,
                        "offset": state.get("case_page", 0),
                    },
                    "case" + str(len(tools)),
                )
            if "diagnostic" not in state:
                return call(
                    "read_review_execution",
                    {"execution_id": state["check"], "view": "log"},
                    "log",
                )
            if verification == "reproduction" and "repro" not in state:
                return call(
                    "run_review_reproduction",
                    {
                        "hypothesis": "Returns two",
                        "script": 'from sample import answer\ndef reproduce():\n    return {"observed":answer()==2,"details":"2"}\n',
                    },
                    "repro",
                )
            if verification == "reproduction" and (
                "observation_page" not in state or state["observation_page"] is not None
            ):
                return call(
                    "read_review_execution",
                    {
                        "execution_id": state["repro"],
                        "limit": 1,
                        "offset": state.get("observation_page", 0),
                    },
                    "observation" + str(len(tools)),
                )
            refs = [state["base"], state["head"]]
            finding = {
                "title": "Return contract broken",
                "priority": "P2",
                "location": {
                    "change_id": state["change"]["change_id"],
                    "side": "head",
                    "path": "sample.py",
                    "line": 2,
                },
                "trigger": "Call answer()",
                "cause": "Constant changed",
                "impact": "Expected return one fails",
                "introduction_basis": "Static one-to-two comparison",
                "limitations": [],
                "diff_evidence_ids": [state["patch"]],
                "evidence": refs,
                "reproduction_status": "observed",
                "reproduction_ids": [state["repro"]]
                if verification == "reproduction"
                else [],
                "check_case_refs": [state["case"]],
                "recommendation": "Restore contract",
            }
            return call(
                "ReviewReportV1",
                {
                    "version": 1,
                    "summary": "Specific failed case and head observation",
                    "findings": [finding],
                    "changes": [
                        {
                            "change_id": state["change"]["change_id"],
                            "status": "reviewed",
                            "reason": "Read fixed patch and source",
                            "evidence": refs,
                            "missing_context": [],
                        }
                    ],
                    "unconfirmed": [],
                    "missing_context": [],
                },
                "report",
            )

        task, _ = await r.store.create_code_task(
            CodeReviewCreate(task="review", workspace_id="project")
        )
        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        assert (await r.store.code_task(task["id"]))["finding_count"] == 1
        assert not await r.store.code_resources() and files(source) == before

    database(check)
