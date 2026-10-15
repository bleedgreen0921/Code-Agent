"""Container PID 1: bounded output collection and an absolute execution deadline."""

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def main():
    args = json.loads(sys.argv[1])
    root = Path("/workspace")
    cwd = (root / args["cwd"]).resolve(strict=True)
    if not cwd.is_relative_to(root) or not cwd.is_dir():
        raise ValueError("Invalid workspace directory")
    deadline = args["deadline"]
    output = bytearray()
    total = 0
    lines = 0
    timed_out = False
    cancelled = False

    def stop(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, stop)
    proc = subprocess.Popen(
        ["/bin/bash", "--noprofile", "--norc", "-c", args["command"]],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    os.set_blocking(proc.stdout.fileno(), False)
    selector.register(proc.stdout, selectors.EVENT_READ)
    while selector.get_map():
        if time.time() >= deadline or cancelled:
            timed_out = not cancelled
            break
        for key, _ in selector.select(timeout=min(0.1, max(0, deadline - time.time()))):
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                selector.unregister(key.fileobj)
                continue
            total += len(data)
            lines += data.count(b"\n")
            output.extend(data[: max(0, 16384 - len(output))])
    if proc.poll() is None or timed_out or cancelled:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    code = proc.wait()
    # Kill detached descendants by exiting PID 1 after publishing the receipt.
    text = bytes(output).decode("utf-8", "replace")
    parts = text.splitlines(keepends=True)
    text = "".join(parts[:200])
    raw = text.encode("utf-8")[:16384]
    text = raw.decode("utf-8", "ignore")
    print(
        json.dumps(
            {
                "exit_code": code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "truncated": total > len(raw) or lines > 200,
                "total_bytes": total,
                "output": text,
            }
        ),
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
