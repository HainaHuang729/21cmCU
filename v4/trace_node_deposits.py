"""Compare native and CUDA float32 CIC grids at each noninteger lightcone node."""

import ctypes as C
import json
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from tests import produce_integration_test_data as official
from py21cmfast.drivers._global_initialization import c_state
from cic_ordered import deposit_ordered
from perturb import _deposit


def run(output, atomic_only=False):
    with patch.object(official, "SEED", 424242):
        inputs = official.get_all_options_struct(
            6,
            lc=True,
            DIM=384,
            HII_DIM=96,
            BOX_LEN=375,
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )["inputs"]
    sim = inputs.simulation_options
    n = sim.DIM
    ic = p21.compute_initial_conditions(inputs=inputs, write=False, regenerate=True)
    ic.push_to_backend()
    address = C.c_void_p(int(p21.c_21cmfast.ffi.cast("uintptr_t", ic._cstruct)))
    lib = C.CDLL(p21.c_21cmfast.__file__)
    lib.dicke.argtypes = [C.c_double]
    lib.dicke.restype = C.c_double
    density = cp.asarray(ic.get("hires_density"))
    vel, second = [
        cp.stack([cp.asarray(ic.get("hires_v" + axis + suffix)) for axis in "xyz"])
        for suffix in ("", "_2LPT")
    ]
    storage = np.zeros((n, n, n + 2), np.float32)
    mass = cp.empty((n, n, n), cp.float64)
    rows = []
    with c_state(inputs, broadcast_inputs=True):
        init = lib.dicke(sim.INITIAL_REDSHIFT)
        for z in inputs.node_redshifts:
            growth = lib.dicke(float(np.float32(z)))

            def cpu():
                lib.make_density_grid(
                    C.c_float(z), C.c_void_p(storage.ctypes.data), address
                )
                return cp.asarray(storage[:, :, :n].copy())

            def gpu():
                mass.fill(0)
                first_scale = (growth - init) / sim.BOX_LEN * n
                second_scale = (
                    (-3 / 7) * (growth * growth - init * init) / sim.BOX_LEN * n
                )
                if atomic_only:
                    _deposit(
                        ((n**3 + 255) // 256,),
                        (256,),
                        (
                            density,
                            vel,
                            second,
                            mass,
                            np.int32(n),
                            np.int32(n),
                            init,
                            first_scale,
                            second_scale,
                            np.bool_(True),
                        ),
                    )
                else:
                    deposit_ordered(
                        density,
                        vel,
                        second,
                        mass,
                        n,
                        n,
                        init,
                        first_scale,
                        second_scale,
                        True,
                    )
                return mass.astype(cp.float32)

            ref, got = cpu(), gpu()
            xyz = cp.argwhere(ref != got).get()
            row = dict(z=z, different=len(xyz))
            if len(xyz):
                repeat_cpu, repeat_gpu = cpu(), gpu()
                row["sites"] = []
                for point in xyz:
                    i = tuple(point.tolist())
                    row["sites"].append(
                        dict(
                            xyz=list(i),
                            cpu=float(ref[i]),
                            gpu=float(got[i]),
                            repeat_cpu=float(repeat_cpu[i]),
                            repeat_gpu=float(repeat_gpu[i]),
                            gpu_mass64=float(mass[i]),
                        )
                    )
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atomic-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.atomic_only)
