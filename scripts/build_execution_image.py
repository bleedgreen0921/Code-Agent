"""Build at deployment/test preparation time, print immutable local image ID."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "agent-execution:workspace-v1"


def build():
    subprocess.run(
        ["docker", "build", "-f", "execution/Dockerfile", "-t", TAG, "."],
        cwd=ROOT,
        check=True,
    )
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", TAG], text=True
    ).strip()


if __name__ == "__main__":
    print(build())
