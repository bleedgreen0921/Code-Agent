"""Untrusted application side of the fixed acceptance RPC adapter."""

import importlib
import json
import os
import sys
from pathlib import Path

request = json.load(sys.stdin)
protocol = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
sys.path.insert(0, str(Path("/workspace") / request["import_path"]))
module = importlib.import_module(request["application_module"])
result = getattr(module, request["function"])(*request["args"], **request["kwargs"])
with os.fdopen(protocol, "w") as stream:
    json.dump({"value": result}, stream)
