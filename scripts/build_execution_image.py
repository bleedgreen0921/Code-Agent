"""Build at deployment/test preparation time, print immutable local image ID."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "agent-execution:workspace-v2"


def build(*, output=None):
    subprocess.run(
        ["docker", "build", "-f", "execution/Dockerfile", "-t", TAG, "."],
        cwd=ROOT,
        check=True,
        stdout=output,
        stderr=subprocess.STDOUT if output is not None else None,
    )
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", TAG], text=True
    ).strip()


if __name__ == "__main__":
    print(build())
