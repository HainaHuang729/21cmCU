"""Check the shared CUDA recombination update against the v4.3.2 C spline."""

import ctypes as C
import json
import sys
from pathlib import Path
from unittest.mock import patch
from py21cmfast.lightconers import RectilinearLightconer

import cupy as cp
import numpy as np
import py21cmfast as p21
from scipy.interpolate import CubicSpline
from py21cmfast.drivers._global_initialization import c_state
from tests import produce_integration_test_data as official

root = Path(__file__).resolve().parents[1] / "lightcone"
code = (root / "glibc_math.cuh").read_text() + "\n" + (root / "ion.cu").read_text()
with patch.object(official, "SEED", 424242):
    inputs = official.get_all_options_struct(
        6,
        lc=True,
        DIM=192,
        HII_DIM=64,
        BOX_LEN=200,
        KEEP_3D_VELOCITIES=True,
        PERTURB_ON_HIGH_RES=True,
        SMOOTH_EVOLVED_DENSITY_FIELD=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
    )["inputs"]
lib = C.CDLL(p21.c_21cmfast.__file__)
rate = lib.splined_recombination_rate
rate.argtypes = [C.c_double] * 2
rate.restype = C.c_double
raw = lib.recombination_rate
raw.argtypes = [C.c_double] * 3 + [C.c_int]
raw.restype = C.c_double
lib.dtdz.argtypes = [C.c_float]
lib.dtdz.restype = C.c_double
z = float(np.float32(7.18900992))
dz = float(np.float32(0.2))
step = float(np.float32(0.2))
knots = -10 + (np.arange(250, dtype="f") * np.float32(0.1)).astype("d")
rng = np.random.default_rng(424242)
# Probe both realistic densities and neighboring float32 values at row transitions.
edges = (((np.arange(1, 120) - 0.5) * step + 1) / (1 + z)) ** 3 - 1
edge = edges.astype("f")
rho = np.concatenate(
    [
        rng.uniform(-0.95, 8, 8192).astype("f"),
        edge,
        np.nextafter(edge, np.float32(np.inf)),
        np.nextafter(edge, np.float32(-np.inf)),
    ]
)
n = len(rho)
gamma = np.exp(rng.uniform(-12, 16, n)).astype("f")
gamma[::23] = 0
xh = rng.uniform(0, 1, n).astype("f")
old = rng.uniform(0, 0.02, n).astype("f")
values = np.empty((300, 250))
with c_state(inputs, broadcast_inputs=True, recomb=True):
    dt = abs(lib.dtdz(z)) / 1e15
    for i in range(300):
        rz = float(np.float32(i * step))
        values[i, :-1] = [rate(rz, np.exp(k)) for k in knots[:-1]]
        values[i, -1] = raw(rz, float(np.float32(np.exp(knots[-1]))), 1.0, 1)
    expected = np.array(
        [
            float(o)
            + rate((1 + float(d)) ** (1 / 3) * (1 + z) - 1, float(g))
            * dt
            * dz
            * (1 - float(h))
            for d, g, h, o in zip(rho, gamma, xh, old)
        ],
        dtype="f",
    )
deriv = CubicSpline(knots, values, axis=1, bc_type="natural")(knots, 1)
tables = [cp.asarray(a, order="C") for a in (knots, values, deriv)]


def compare_update(version, fields, z, dt, dz, reference):
    scalar = np.float64 if version == "v432" else np.float32
    options = ("--fmad=false",) + (("-DRECOMB_V432",) if version == "v432" else ())
    kernel = cp.RawModule(code=code, options=options).get_function("recombine")
    args = [cp.asarray(a, dtype="f", order="C") for a in fields]
    out = cp.empty_like(args[0])
    n = out.size
    flag = cp.zeros(1, dtype="i")
    kernel(
        ((n + 255) // 256,),
        (256,),
        (
            *args,
            out,
            *tables,
            np.int32(n),
            scalar(z),
            scalar(dt),
            scalar(dz),
            np.float64(step),
            np.float64(knots[-1] - 1e-7),
            flag,
        ),
    )
    reference = reference.astype("d")
    diff = abs(out.get().astype("d") - reference)
    return dict(
        sample_count=n,
        failures=int(np.sum(~(diff <= 1e-8 + 1e-4 * abs(reference)))),
        max_abs=float(diff.max()),
        nonfinite=int(flag.get()[0]),
    )


result = {
    v: compare_update(v, (rho, gamma, xh, old), z, dt, dz, expected)
    for v in ("v3", "v432")
}
lightconer = RectilinearLightconer.between_redshifts(
    min_redshift=6,
    max_redshift=15,
    quantities=("brightness_temp",),
    resolution=inputs.simulation_options.cell_size,
)
prev = None
prev_z = None
for _, redshift, box, _ in p21.generate_lightcone(
    inputs=inputs, lightconer=lightconer, regenerate=True, write=False
):
    current = np.asarray(box.cumulative_recombinations)
    if prev is not None and redshift <= 7.4 and np.any(current > prev):
        z = float(np.float32(redshift))
        dz = prev_z - z
        dt = abs(lib.dtdz(z)) / 1e15
        fields = [box.density, box.ionisation_rate_G12, box.neutral_fraction, prev]
        result["official_node_v3"] = compare_update("v3", fields, z, dt, dz, current)
        result["official_node"] = dict(
            seed=inputs.random_seed,
            redshift=z,
            previous_redshift=prev_z,
            changed_cells=int(np.sum(current > prev)),
            **compare_update("v432", fields, z, dt, dz, current),
        )
        break
    prev = current.copy()
    prev_z = float(np.float32(redshift))
print(json.dumps(result, indent=2))
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
assert result["v432"]["failures"] == result["v432"]["nonfinite"] == 0

assert result["official_node"]["failures"] == result["official_node"]["nonfinite"] == 0
