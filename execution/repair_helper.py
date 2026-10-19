"""Process observation in the candidate PID namespace; never executes project code."""

import json
import os
from pathlib import Path

running = []
for path in Path("/proc").iterdir():
    if not path.name.isdecimal() or int(path.name) in (1, os.getpid()):
        continue
    try:
        status = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if status[0] != "Z":
            running.append(int(path.name))
    except FileNotFoundError:
        pass
print(json.dumps({"running_pids": sorted(running)}))
