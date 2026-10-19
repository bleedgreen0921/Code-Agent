"""Image-only observation runner. Project stdout never enters result protocol."""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

VERSION = "python-observation-v1"


def run(request):
    result = {
        "protocol": 1,
        "runner_version": VERSION,
        "status": "inconclusive",
        "details": "",
        "complete": False,
        "limitations": [],
    }
    try:
        script = request["script"]
        if type(script) is not str or len(script.encode()) > 16384:
            raise ValueError("Script exceeds 16 KiB")
        path = Path("/tmp/reproduction.py")
        path.write_text(script)
        result["script_digest"] = hashlib.sha256(script.encode()).hexdigest()
        os.chdir("/workspace")
        sys.path.insert(0, "/workspace")
        namespace = {"__name__": "review_reproduction", "__file__": str(path)}
        exec(compile(script, str(path), "exec"), namespace)
        observed = namespace["reproduce"]()
        if (
            type(observed) is not dict
            or set(observed) != {"observed", "details"}
            or type(observed["observed"]) is not bool
            or type(observed["details"]) is not str
            or len(observed["details"].encode()) > 32768
        ):
            raise ValueError(
                "reproduce() must return observed: bool and details: bounded str"
            )
        result.update(
            status="observed" if observed["observed"] else "not_observed",
            details=observed["details"],
            complete=True,
        )
    except BaseException as exc:
        result["details"] = type(exc).__name__ + ": " + str(exc)[:2000]
        result["limitations"].append("exception_or_invalid_return")
        traceback.print_exc(file=sys.stderr)
    return result


def main():
    request = json.load(sys.stdin)
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    result = run(request)
    with os.fdopen(protocol, "w") as stream:
        json.dump(result, stream, ensure_ascii=True)


if __name__ == "__main__":
    main()
