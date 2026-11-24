"""Isolate 2LPT displacement grouping and scale rounding against native CIC."""

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

from perturb import _deposit


def compare(inputs, ic, z):
    sim = inputs.simulation_options
    n = sim.DIM
    lib = C.CDLL(p21.c_21cmfast.__file__)
    lib.dicke.argtypes = [C.c_double]
    lib.dicke.restype = C.c_double
    storage = np.zeros((n, n, 2 * (n // 2 + 1)), np.float32)
    ic.push_to_backend()
    with c_state(inputs, broadcast_inputs=True):
        growth, initial = lib.dicke(z), lib.dicke(sim.INITIAL_REDSHIFT)
        lib.make_density_grid(
            C.c_float(z),
            C.c_void_p(storage.ctypes.data),
            C.c_void_p(int(p21.c_21cmfast.ffi.cast("uintptr_t", ic._cstruct))),
        )
    ref = cp.asarray(storage[:, :, :n].copy())
    density = cp.asarray(ic.get("hires_density"))
    first, second = [
        cp.stack([cp.asarray(ic.get("hires_v" + axis + suffix)) for axis in "xyz"])
        for suffix in ("", "_2LPT")
    ]
    new = "p = a == 0 ? p - correction : xyz[a] + (displacement - correction);"
    assert _deposit.code.count(new) == 1
    old = cp.RawKernel(
        _deposit.code.replace(new, "p -= correction;"),
        "deposit",
        options=("--fmad=false",),
    )
    results = {}
    for grouping, kernel in (("old", old), ("reference", _deposit)):
        for scale in ("old", "reference"):
            first_factor = growth - initial
            second_factor = (-3 / 7) * (growth * growth - initial * initial)

            def convert(x):
                return x * n / sim.BOX_LEN if scale == "old" else x / sim.BOX_LEN * n

            mass = cp.zeros(ref.shape, cp.float64)
            kernel(
                ((n**3 + 255) // 256,),
                (256,),
                (
                    density,
                    first,
                    second,
                    mass,
                    np.int32(n),
                    np.int32(n),
                    initial,
                    convert(first_factor),
                    convert(second_factor),
                    np.bool_(True),
                ),
            )
            delta = cp.abs(mass.astype(cp.float32) - ref)
            results[f"group_{grouping}_scale_{scale}"] = dict(
                different=int(cp.count_nonzero(delta).get()),
                max_abs=float(delta.max().get()),
            )
    return results


if __name__ == "__main__":
    rows = json.loads(Path(sys.argv[1]).read_text())["cases"]
    result = []
    for row in rows:
        if row["case"] != "highres" or row["smooth"]:
            continue
        with patch.object(official, "SEED", row["seed"]):
            inputs = official.get_all_options_struct(
                10, **official.OPTIONS_PT["highres"][1]
            )["inputs"].evolve_input_structs(
                DIM=row["fine"],
                HII_DIM=row["low"],
                BOX_LEN=row["box"],
                KEEP_3D_VELOCITIES=True,
                SMOOTH_EVOLVED_DENSITY_FIELD=False,
            )
        ic = p21.compute_initial_conditions(inputs=inputs)
        result.append(
            dict(
                fine=row["fine"],
                seed=row["seed"],
                z=row["z"],
                variants=compare(inputs, ic, row["z"]),
            )
        )
        Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
