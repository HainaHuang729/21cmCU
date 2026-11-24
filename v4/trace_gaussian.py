"""Identify the largest failed voxel and replay Gaussian smoothing on its spectrum."""

import ctypes as C
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state
from tests import produce_integration_test_data as official
from trace_fft import native_stage
import perturb


def compare(a, b):
    diff = abs(a.astype(np.result_type(a.dtype, np.float64)) - b)
    return dict(max_abs=float(diff.max()), different=int(np.count_nonzero(diff)))


def previous_weights(k2, radius):
    return cp.exp((-0.5 * 0.643**2) * (k2 * radius * radius).astype(cp.float64))


@patch.object(perturb, "gaussian_weights", previous_weights)
def run():
    total = 0
    largest = -1
    for seed in (424242, 1850877074):
        for case in ("linear", "no2lpt", "highres"):
            with patch.object(official, "SEED", seed):
                inputs = official.get_all_options_struct(
                    10, **official.OPTIONS_PT[case][1]
                )["inputs"].evolve_input_structs(
                    DIM=150,
                    HII_DIM=50,
                    BOX_LEN=100,
                    KEEP_3D_VELOCITIES=True,
                    SMOOTH_EVOLVED_DENSITY_FIELD=True,
                )
            ic = p21.compute_initial_conditions(inputs=inputs)
            for z in (6.0, 15.0):
                kw = dict(redshift=z, initial_conditions=ic)
                cpu = p21.perturb_field(**kw).get("density").astype(float)
                gpu = perturb.perturb_field_gpu(**kw).get("density").astype(float)
                diff = abs(cpu - gpu)
                tol = 1e-8 + 1e-4 * abs(cpu)
                failed = diff > tol
                total += int(failed.sum())
                idx = np.unravel_index(np.where(failed, diff, -1).argmax(), diff.shape)
                if failed[idx] and diff[idx] > largest:
                    largest = diff[idx]
                    target = dict(
                        seed=seed,
                        case=case,
                        z=z,
                        xyz=list(map(int, idx)),
                        cpu=cpu[idx],
                        gpu=gpu[idx],
                        diff=largest,
                        tolerance=tol[idx],
                    )
                    winning = kw
    idx = tuple(target["xyz"])
    inputs = winning["initial_conditions"].inputs
    captured = {}
    original = perturb.transform

    def capture(a, n, inverse=False, threads=2):
        result = original(a, n, inverse, threads=threads)
        if n == 50:
            if not inverse:
                captured["grid"] = a.get()
                captured["spectrum"] = result.get()
            elif "filtered" not in captured:
                captured["filtered"] = a.get()
        return result

    with patch.object(perturb, "transform", capture):
        perturb.perturb_field_gpu(**winning)
    spec = captured["spectrum"]
    lib = C.CDLL(p21.c_21cmfast.__file__)
    sim = inputs.simulation_options
    r = np.float32(sim.DENSITY_SMOOTH_RADIUS * sim.BOX_LEN / 50)
    with c_state(inputs, broadcast_inputs=True):

        def native(x):
            a = x.copy()
            lib.filter_box(
                C.c_void_p(a.ctypes.data),
                (C.c_int * 3)(50, 50, 50),
                2,
                C.c_float(r),
                C.c_float(0),
                C.c_float(0),
            )
            return a

        ref = native(spec)
        unit = native(np.ones_like(spec)).real
        k2 = sum(k * k for k in perturb.modes(50, sim.BOX_LEN))
        stats = {}
        for name, kr2 in (("old", k2 * r * r), ("squared_radius", k2 * (r * r))):
            w = cp.exp((-0.5 * 0.643**2) * kr2.astype(cp.float64))
            product = (cp.asarray(spec) * w).astype(cp.complex64).get()
            stats[name] = dict(
                weights_float=compare(unit, w.astype(cp.float32).get()),
                product=compare(ref, product),
                same_weight_product=compare(
                    (spec * w.get()).astype(np.complex64), product
                ),
                actual_previous_product=compare(captured["filtered"], product),
            )
        inv = native_stage(lib, ref, "inverse", r)
        ig = original(cp.asarray(ref), 50, True).get()
        return dict(
            failed=total,
            target=target,
            radius=float(r),
            forward=compare(native_stage(lib, captured["grid"], "forward", r), spec),
            filters=stats,
            actual_previous=compare(ref, captured["filtered"]),
            fixed_spectrum_inverse=compare(inv, ig),
            replay_density=float(ig[idx] / np.float32(50**3)),
        )


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
