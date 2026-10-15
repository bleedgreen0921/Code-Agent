"""CI/local entry: isolated ordinary PostgreSQL, synthetic tests, strict reports."""

import argparse
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
    compose = [
        "docker",
        "compose",
        "-f",
        str(ROOT / "compose.integration.yaml"),
        "-p",
        project,
    ]

    def command(argv, timeout=120):
        subprocess.run(argv, cwd=ROOT, env=env, check=True, timeout=timeout)

    try:
        command(compose + ["up", "-d", "--wait"], timeout=300)
        command([sys.executable, "-m", "db.migrate"])
        command([sys.executable, "-m", "db.bootstrap"])
        command([sys.executable, "-m", "db.doctor", "--json"])
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
            timeout=300,
        )
        counts = validate_report(report)
        (output / "summary.json").write_text(json.dumps(counts, indent=2))
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


if __name__ == "__main__":
    main()
