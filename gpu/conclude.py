"""Apply the comparison tolerance and summarize the paired, untapped runs."""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
cpu = json.loads((out/"cpu-timing.json").read_text())
gpu = json.loads((out/"gpu-timing.json").read_text())
validation = json.loads((out/"gpu-validation.json").read_text())
assert cpu["node"] == gpu["node"]
cpu_wall = float((out/"cpu-timing-wall-seconds.txt").read_text())
gpu_wall = float((out/"gpu-timing-wall-seconds.txt").read_text())
with np.load(out/"cpu-timing-outputs.npz") as a, np.load(out/"gpu-timing-outputs.npz") as b:
    same_output = {name: bool(np.allclose(a[name], b[name], rtol=1e-4, atol=1e-8))
                   for name in ("xhi","ps_native","ps_full")}
checks = [
    *validation["production_reference"].values(),
    *validation["same_node_cpu"].values(),
    *validation["ion_history"].values(), *validation["thermal"].values(),
]
result = {
    "node": cpu["node"],
    "cpu_wall_seconds": cpu_wall,
    "gpu_wall_seconds": gpu_wall,
    "cpu_coeval_seconds": cpu["coeval_seconds"],
    "gpu_coeval_seconds": gpu["coeval_seconds"],
    "paired_wall_ratio": cpu_wall / gpu_wall,
    "paired_outputs_pass": same_output,
    "validation_pass": all(item["failed"] == 0 for item in checks),
}
(out/"paired_result.json").write_text(json.dumps(result, indent=2) + "\n")
print("PAIRED_RESULT", json.dumps(result), flush=True)
if not result["validation_pass"] or not all(same_output.values()):
    raise SystemExit(2)

