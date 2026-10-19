"""Trusted tests communicate with application code through a child process only."""

import json
import subprocess
import sys

CONFIG = None


def invoke(function, *args, **kwargs):
    request = {**CONFIG, "function": function, "args": args, "kwargs": kwargs}
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "/opt/agent/acceptance_app.py"],
        input=json.dumps(request).encode(),
        stdout=subprocess.PIPE,
        stderr=None,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
        cwd="/tmp",
        timeout=CONFIG["timeout"],
        check=False,
    )
    if proc.returncode or len(proc.stdout) > 1024**2:
        raise RuntimeError("Candidate application failed: exit " + str(proc.returncode))
    return json.loads(proc.stdout)["value"]
