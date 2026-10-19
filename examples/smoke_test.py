"""Only standard-library imports; safe to run on the offline competition host."""
import json
import os
import platform
import sys

print(json.dumps({
    "bridge_test": "PASS",
    "cwd": os.getcwd(),
    "hostname": platform.node(),
    "python": sys.version,
    "executable": sys.executable,
}, ensure_ascii=False, indent=2))
