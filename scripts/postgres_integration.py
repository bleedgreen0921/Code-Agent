"""CI/local entry: isolated ordinary PostgreSQL, synthetic tests, strict reports."""

import argparse
import hashlib
import importlib.metadata
import platform
import json
import os
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def validate_report(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = {
        "tests": len(cases),
        **{
            name: sum(c.find(tag) is not None for c in cases)
            for name, tag in [
                ("failures", "failure"),
                ("errors", "error"),
                ("skipped", "skipped"),
            ]
        },
    }
    if not cases or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise RuntimeError(f"Test report rejected: {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    project = "agent-core-test-" + uuid4().hex[:12]
    output = ROOT / ".data" / "integration" / project
    output.mkdir(parents=True)
    env = dict(os.environ)
    # Test children use only generated credentials and unreachable synthetic endpoints.
    for name in list(env):
        if name.startswith(
            (
                "AGENT_",
                "RAG_",
                "OPENAI_",
                "MIGRATION_",
                "PYTEST_",
                "POSTGRES_",
                "LANGSMITH_",
                "LANGCHAIN_",
            )
        ):
            env.pop(name)
    password = secrets.token_hex(24)
    runtime_password = secrets.token_hex(24)
    env.update(
        POSTGRES_INTEGRATION_PASSWORD=password,
        POSTGRES_INTEGRATION_PORT=str(args.port),
        MIGRATION_DATABASE_URL=f"postgresql://postgres:{password}@127.0.0.1:{args.port}/agent_integration",
        AGENT_DATABASE_URL=f"postgresql://agent_runtime:{runtime_password}@127.0.0.1:{args.port}/agent_integration",
        AGENT_TEST_DATABASE="1",
        AGENT_MODEL="synthetic",
        AGENT_MODEL_URL="http://127.0.0.1:9",
        PYTHONUNBUFFERED="1",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    env['AGENT_REVIEW_EXAMPLE_DIR'] = str(output / 'review-examples')
    compose = [
        "docker",
        "compose",
        "-f",
        str(ROOT / "compose.integration.yaml"),
        "-p",
        project,
    ]

    log_path = output / "integration.log"
    print(f"Integration evidence: {output}", flush=True)
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "python": platform.python_version(),
        "dependency_lock_sha256": hashlib.sha256(
            (ROOT / "requirements.lock").read_bytes()
        ).hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "langchain",
                "langgraph",
                "langgraph-checkpoint-postgres",
                "mcp",
                "fastmcp-slim",
                "fastmcp-tasks",
                "psycopg",
                "pydantic",
            )
        },
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("agent_service", "contracts", "db", "execution", "scripts", "tests")
            for p in sorted((ROOT / directory).rglob("*"))
            if p.is_file()
            and (p.suffix in (".py", ".sql", ".json") or p.name == "Dockerfile")
        },
        "checks": [],
    }

    def command(argv, timeout=120):
        with log_path.open("a") as log:
            subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                check=True,
                timeout=timeout,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def containers():
        return set(
            subprocess.check_output(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    "label=io.agent-framework.resource",
                ],
                text=True,
            ).split()
        )

    before_containers = containers()

    try:
        # All tests use a prebuilt immutable image. Runs never pull/build/install.
        from scripts.build_execution_image import build

        with log_path.open("a") as log:
            env["AGENT_TEST_EXECUTION_IMAGE"] = build(output=log)
        metadata["execution_image"] = env["AGENT_TEST_EXECUTION_IMAGE"]
        command(compose + ["up", "-d", "--wait"], timeout=300)
        metadata["postgres_image"] = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Image}}", project + "-postgres-1"],
            text=True,
        ).strip()
        command([sys.executable, "-m", "db.migrate"])
        metadata["checks"].append("core_0010 migration passed")
        command([sys.executable, "-m", "db.bootstrap"])
        metadata["checks"].append("runtime bootstrap passed")
        command([sys.executable, "-m", "db.doctor", "--json"])
        metadata["checks"].append("readiness and read-only doctor passed")
        report = output / "pytest.xml"
        command(
            [
                sys.executable,
                "-m",
                "scripts.integration_pytest",
                "-q",
                "--junitxml",
                str(report),
            ],
            timeout=900,
        )
        counts = validate_report(report)
        (output / "summary.json").write_text(json.dumps(counts, indent=2))
        metadata["tests"] = counts
        print(json.dumps(counts), flush=True)
    finally:
        if args.keep_db:
            credentials = output / "environment.json"
            fd = os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as file:
                json.dump(
                    {
                        k: v
                        for k, v in env.items()
                        if k
                        in (
                            "MIGRATION_DATABASE_URL",
                            "AGENT_DATABASE_URL",
                            "AGENT_TEST_DATABASE",
                            "POSTGRES_INTEGRATION_PASSWORD",
                            "POSTGRES_INTEGRATION_PORT",
                        )
                    },
                    file,
                )
            print(f"Test database retained; credentials: {credentials}")
        else:
            command(compose + ["down", "-v", "--remove-orphans"])
        metadata["database_retained"] = args.keep_db
        metadata["new_unreleased_containers"] = sorted(containers() - before_containers)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if metadata["new_unreleased_containers"]:
            raise RuntimeError("Integration resources remain; inspect metadata.json")


if __name__ == "__main__":
    main()
