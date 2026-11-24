"""Run inside a v4.3.2 checkout with its upstream tests on PYTHONPATH."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np
import py21cmfast as p21
from tests import produce_integration_test_data as official
from perturb import perturb_field_gpu


def measure(function, **kwargs):
    times = []
    for _ in range(4):
        start = perf_counter()
        field = function(**kwargs)
        times.append(perf_counter() - start)
    return field, {
        "first": times[0],
        "warm_median": median(times[1:]),
        "samples": times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-fft", action="store_true")
    args = parser.parse_args()
    report = {
        "reference_fft": args.reference_fft,
        "version": p21.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "atol": 1e-8,
        "rtol": 1e-4,
        "cases": {},
    }
    for name in ("linear", "no2lpt", "highres"):
        z, options = official.OPTIONS_PT[name]
        inputs = official.get_all_options_struct(z, **options)[
            "inputs"
        ].evolve_input_structs(KEEP_3D_VELOCITIES=True)
        ic = p21.compute_initial_conditions(inputs=inputs)
        kwargs = dict(redshift=z, initial_conditions=ic)
        cpu, tc = measure(p21.perturb_field, **kwargs)
        gpu, tg = measure(perturb_field_gpu, reference_fft=args.reference_fft, **kwargs)
        fields = {}
        for key in ("density", "velocity_x", "velocity_y", "velocity_z"):
            a, b = (x.get(key).astype(np.float64) for x in (cpu, gpu))
            assert (
                a.shape == b.shape == (50,) * 3
                and np.isfinite(a).all()
                and np.isfinite(b).all()
            )
            delta = abs(a - b)
            fields[key] = dict(
                max_abs=float(delta.max()),
                failed=int((delta > 1e-8 + 1e-4 * abs(a)).sum()),
                peak_reference=float(abs(a).max()),
            )
            if key.startswith("velocity"):
                # Upstream scales velocities by 1e16 in its spectral regression.
                fields[key]["failed_in_upstream_scaled_units"] = int(
                    (delta * 1e16 > 1e-8 + 1e-4 * abs(a) * 1e16).sum()
                )
        report["cases"][name] = dict(cpu=tc, gpu=tg, fields=fields)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
