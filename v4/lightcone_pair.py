"""Compare native v4 lightcones with the CUDA evolution adapter."""

# ruff: noqa: E402 -- include scientific-library import time in the benchmark.

from contextlib import ExitStack
from time import perf_counter

START = perf_counter()

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field
from py21cmfast.lightconers import RectilinearLightconer

FIELDS = (
    "brightness_temp",
    "neutral_fraction",
    "xray_ionised_fraction",
    "kinetic_temp_neutral",
    "spin_temperature",
    "cumulative_recombinations",
)


def benchmark_case(fine):
    """Construct the native benchmark inputs and line-of-sight coordinates."""
    from tests import produce_integration_test_data as official

    low, box = {192: (64, 200), 384: (96, 375)}[fine]
    with patch.object(official, "SEED", 424242):
        inputs = official.get_all_options_struct(
            6,
            lc=True,
            DIM=fine,
            HII_DIM=low,
            BOX_LEN=box,
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )["inputs"]
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6,
        max_redshift=15,
        quantities=FIELDS,
        resolution=inputs.simulation_options.cell_size,
    )
    return inputs, lightconer


def run(directory, fine, mode, *, audit_evolution=False):
    inputs, lightconer = benchmark_case(fine)
    with ExitStack() as stack:
        timings = None
        if mode == "gpu":
            from gpu_lightcone import _backend

            timings = stack.enter_context(
                _backend(inputs, lightconer, directory, audit=audit_evolution)
            )
        return _run(directory, fine, mode, inputs, lightconer, timings)


def _run(directory, fine, mode, inputs, lightconer, timings):
    ic = p21.compute_initial_conditions(inputs=inputs, write=False, regenerate=True)
    hash_start = perf_counter()
    digest = hashlib.sha256()
    for key in sorted(ic.arrays):
        if ic.arrays[key].state.computed_in_mem:
            digest.update(key.encode())
            digest.update(memoryview(np.ascontiguousarray(ic.get(key))).cast("B"))
    hash_seconds = perf_counter() - hash_start
    cpu_func = single_field.perturb_field
    calls = []

    def perturb(**kwargs):
        start = perf_counter()
        result = cpu_func(**kwargs)
        calls.append(perf_counter() - start)
        return result

    brightness_calls = []
    with ExitStack() as hooks:
        if timings is None:
            hooks.enter_context(patch.object(single_field, "perturb_field", perturb))
        else:
            calls, brightness_calls = timings.perturb, timings.brightness
        cone = p21.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            initial_conditions=ic,
            write=False,
            regenerate=True,
        )
    path = directory / (mode + ".h5")
    cone.save(path, clobber=True)
    with h5py.File(path, "r+") as saved:
        saved.attrs["benchmark_ic_sha256"] = digest.hexdigest()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    elapsed = perf_counter() - START
    assert calls and len(calls) == len(inputs.node_redshifts)
    stats = dict(
        fine=fine,
        seed=inputs.random_seed,
        nodes=list(inputs.node_redshifts),
        ic_hash=digest.hexdigest(),
        calls=len(calls),
        gpu_brightness_calls=len(brightness_calls),
        gpu_brightness_seconds=sum(brightness_calls),
        perturb_seconds=sum(calls),
        elapsed_seconds=elapsed,
        hash_seconds=hash_seconds,
        file_bytes=path.stat().st_size,
    )
    (directory / (mode + ".json")).write_text(json.dumps(stats, indent=2))


def compare(directory):
    result = {
        m: json.loads((directory / (m + ".json")).read_text()) for m in ("cpu", "gpu")
    }
    assert result["cpu"]["ic_hash"] == result["gpu"]["ic_hash"]
    stats = result["fields"] = {}
    with h5py.File(directory / "cpu.h5") as a, h5py.File(directory / "gpu.h5") as b:
        np.testing.assert_array_equal(
            a["lightcone_distances"][:], b["lightcone_distances"][:]
        )
        for field in FIELDS:
            x, y = (f["lightcones"][field][:].astype(float) for f in (a, b))
            assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            d = abs(x - y)
            bad = d > 1e-8 + 1e-4 * abs(x)
            row = dict(
                shape=x.shape,
                max_abs=float(d.max()),
                failed=int(bad.sum()),
                failed_slices=int(bad.any(axis=(0, 1)).sum()),
            )
            if bad.any():
                idx = np.unravel_index(np.where(bad, d, -1).argmax(), d.shape)
                row.update(
                    xyz=list(map(int, idx)), cpu=float(x[idx]), gpu=float(y[idx])
                )
            stats[field] = row
    (directory / "comparison.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    if sys.argv[3] == "compare":
        compare(directory)
    else:
        run(directory, int(sys.argv[2]), sys.argv[3])
