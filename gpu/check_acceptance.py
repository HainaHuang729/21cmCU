from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
gpu = json.loads((out/"gpu-validation.json").read_text())
cpu = json.loads((out/"cpu-reference.json").read_text())
checked = [*gpu["ion_history"].values(), *gpu["thermal"].values(), *gpu["production_reference"].values(),
           *gpu["same_node_cpu"].values(), *cpu["production_reference"].values()]
assert (gpu["rtol"], gpu["atol"]) == (1e-4,1e-8)
assert gpu["thermal_steps"] == 123
runtime = json.loads((out/"gpu-validation-runtime.json").read_text())
assert runtime["resident_filter_inputs"] == gpu["filter_steps"] == 121
assert runtime["host_filter_uploads"] == 0
assert runtime["initial_gpu_heat_steps"] == 2
assert runtime["ic_quadratic_gpu_calls"] == 3
assert runtime["no_light_materializations"] == 0 and any(row["no_light"] for row in runtime["records"])
assert runtime["brightness_seed_uploads"] == 0
assert runtime["ionization"]["poisson_gpu_used"]
assert runtime["gpu_brightness_steps"]==32 and runtime["ionization"]["phase_counts"][5]==45
assert gpu["ion_steps"] > 100 and runtime["ionization"]["resident_density_steps"] > 0
passed = all(item["failed"] == 0 for item in checked)
print("UNCHANGED_TOLERANCE_ACCEPTANCE", passed, flush=True)
if not passed:
    raise SystemExit(2)
