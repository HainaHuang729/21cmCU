import ctypes as C, itertools, json, sys
from pathlib import Path
import numpy as np
import py21cmfast as p21
from tests import produce_integration_test_data as official
from py21cmfast.drivers._global_initialization import c_state

d = Path(sys.argv[1])
r = {m: json.loads((d / f"recomb-inputs-{m}.json").read_text()) for m in ("cpu", "gpu")}
i = official.get_all_options_struct(
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
lib = C.CDLL(p21.c_21cmfast.__file__)
rate = lib.splined_recombination_rate
rate.argtypes = [C.c_double] * 2
rate.restype = C.c_double
dt = lib.dtdz
dt.argtypes = [C.c_float]
dt.restype = C.c_double
z, zp = map(lambda row: float(np.float32(row["z"])), r["cpu"][-2:][::-1])
out = {}
with c_state(i, broadcast_inputs=True, recomb=True):
    for rho, gamma, xh in itertools.product(("cpu", "gpu"), repeat=3):
        eff = (1 + r[rho][-1]["density"]) ** (1 / 3) * (1 + z) - 1
        inc = (
            rate(eff, r[gamma][-1]["ionisation_rate_G12"])
            * abs(dt(z))
            / 1e15
            * (zp - z)
            * (1 - r[xh][-1]["neutral_fraction"])
        )
        out["/".join([rho, gamma, xh])] = float(np.float32(inc))
for m in ("cpu", "gpu"):
    assert (
        abs(
            out["/".join([m] * 3)]
            - (
                r[m][-1]["cumulative_recombinations"]
                - r[m][-2]["cumulative_recombinations"]
            )
        )
        < 1e-8
    )
(d / "single-step-swaps.json").write_text(
    json.dumps(
        dict(
            order=["density", "G12", "xHI"],
            swaps=out,
            nodes={m: rows[-2:] for m, rows in r.items()},
        ),
        indent=2,
    )
)
