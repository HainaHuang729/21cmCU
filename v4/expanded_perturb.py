"""Fixed-seed grid and redshift holdouts for the standalone perturbation adapter."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from tests import produce_integration_test_data as official

from benchmark_perturb import measure
from perturb import perturb_field_gpu


def run(output, fine_grid=None):
    rows = []
    report = dict(
        version=p21.__version__,
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        threads=2,
        atol=1e-8,
        rtol=1e-4,
        cases=rows,
    )
    for fine, low, box, smooth in (
        (150, 50, 100, False),
        (150, 50, 100, True),
        (192, 64, 200, False),
        (192, 64, 200, True),
        (384, 96, 375, False),
        (384, 96, 375, True),
    ):
        if fine_grid is not None and fine != fine_grid:
            continue
        for seed in (424242, 1850877074):
            for case in ("linear", "no2lpt", "highres"):
                _, options = official.OPTIONS_PT[case]
                with patch.object(official, "SEED", seed):
                    inputs = official.get_all_options_struct(10, **options)[
                        "inputs"
                    ].evolve_input_structs(
                        DIM=fine,
                        HII_DIM=low,
                        BOX_LEN=box,
                        KEEP_3D_VELOCITIES=True,
                        SMOOTH_EVOLVED_DENSITY_FIELD=smooth,
                    )
                assert inputs.random_seed == seed
                ic = p21.compute_initial_conditions(inputs=inputs)
                for z in (6.0, 15.0):
                    kw = dict(redshift=z, initial_conditions=ic)
                    cpu, tc = measure(p21.perturb_field, **kw)
                    gpu, tg = measure(perturb_field_gpu, **kw)
                    fields = {}
                    for key in ("density", "velocity_x", "velocity_y", "velocity_z"):
                        a, b = (x.get(key).astype(np.float64) for x in (cpu, gpu))
                        assert (
                            a.shape == b.shape == (low,) * 3
                            and np.isfinite(a).all()
                            and np.isfinite(b).all()
                        )
                        delta = abs(a - b)
                        fields[key] = dict(
                            max_abs=float(delta.max()),
                            failed=int((delta > 1e-8 + 1e-4 * abs(a)).sum()),
                        )
                        failed = delta > 1e-8 + 1e-4 * abs(a)
                        if failed.any():
                            idx = np.unravel_index(np.where(failed, delta, -1).argmax(), a.shape)
                            fields[key]["worst_failed"] = dict(
                                xyz=list(map(int, idx)), cpu=float(a[idx]), gpu=float(b[idx]),
                                delta=float(delta[idx]),
                            )
                        if key != "density":
                            fields[key]["scaled_failed"] = int(
                                (delta * 1e16 > 1e-8 + 1e-4 * abs(a) * 1e16).sum()
                            )
                    rows.append(
                        dict(
                            fine=fine,
                            low=low,
                            box=box,
                            smooth=smooth,
                            seed=seed,
                            case=case,
                            z=z,
                            cpu=tc,
                            gpu=tg,
                            fields=fields,
                        )
                    )
                    output.write_text(json.dumps(report, indent=2) + "\n")
                del ic, cpu, gpu
                cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
