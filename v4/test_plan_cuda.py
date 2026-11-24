"""Compare generated forward and inverse kernels against the same CPU FFTW inputs."""

import argparse
import ctypes as C
import json
from pathlib import Path
import traceback

import cupy as cp
import numpy as np
import py21cmfast as p21
from fft_plan_cuda import transform
from trace_fft import native_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sizes", nargs="+", type=int)
    args = parser.parse_args()
    lib = C.CDLL(p21.c_21cmfast.__file__)
    results = {}
    for n in args.sizes:
        rng = np.random.default_rng(424242)
        grid = rng.standard_normal((n,) * 3, dtype=np.float32)
        spectrum = native_stage(lib, grid, "forward", 0)
        row = results[n] = {}
        for op in ("forward", "inverse"):
            try:
                ref = (
                    spectrum
                    if op == "forward"
                    else native_stage(lib, spectrum, "inverse", 0)
                )
                got = transform(
                    cp.asarray(grid if op == "forward" else spectrum),
                    n,
                    op == "inverse",
                ).get()
                row[op] = dict(
                    max_abs=float(abs(got - ref).max()),
                    different=int((got != ref).sum()),
                )
            except Exception:
                row[op] = dict(error=traceback.format_exc()[-3500:])
        args.output.write_text(json.dumps(results, indent=2))
        print(n, flush=True)
    if any(
        stage.get("different") != 0
        for row in results.values()
        for stage in row.values()
    ):
        raise SystemExit(1)
