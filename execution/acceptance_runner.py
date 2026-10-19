"""Independent unittest collection; candidate Python never enters this process."""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/agent")
import acceptance_support
from project_runner import run_unittest


def run(request):
    root = Path("/opt/agent/acceptance") / request["asset_directory"]
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    checksum = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if checksum != request["asset_digest"]:
        raise ValueError("Acceptance assets changed")
    acceptance_support.CONFIG = {
        k: request[k] for k in ("application_module", "import_path", "timeout")
    }
    os.chdir(root)
    # Only independent asset modules may be collected; never project conftest/plugins.
    sys.path.insert(0, str(root))
    result = dict(
        protocol=1,
        adapter=request["runner"],
        complete=False,
        selected=0,
        started=[],
        finished=[],
        events=[],
        collection_errors=[],
    )
    code = run_unittest(
        {"cases": request["required_cases"], "paths": [], "pattern": "test*.py"}, result
    )
    result.update(complete=True, exit_code=code)
    return {
        "asset_digest": checksum,
        "suite_id": request["id"],
        "version": request["version"],
        "result": result,
    }, code


def main():
    request = json.load(sys.stdin)
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        result, code = run(request)
    except BaseException as exc:
        result, code = {"control_error": type(exc).__name__ + ": " + str(exc)[:1000]}, 2
    with os.fdopen(protocol, "w") as stream:
        json.dump(result, stream)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
