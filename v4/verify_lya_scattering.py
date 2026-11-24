"""Compare CUDA and native 4.3.2 scattering windows on identical Fourier modes."""

import ctypes as C
import json
import os
import numpy as np
import cupy as cp
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state
from evolution_backend import SRC
from lyman_scattering import build_window

native = C.CDLL(p21.c_21cmfast.__file__).filter_box
native.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_int),
    C.c_int,
    C.c_float,
    C.c_float,
    C.c_float,
]
native.restype = None
kernel = build_window(SRC, reuse=True)
out = []
rng = np.random.default_rng(129887)
for side in (8, 32):
    inputs = p21.InputParameters(random_seed=129887).evolve_input_structs(
        DIM=32,
        HII_DIM=side,
        N_THREADS=1,
        BOX_LEN=24,
    )
    shape = (side, side, side // 2 + 1)
    modes = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")
    with c_state(inputs, broadcast_inputs=True):
        for inner, outer in ((0.2, 0.5), (3.0, 9.0), (12.0, 24.0)):
            for rstar in (0.0, 0.001, 0.1, 2.0, 100.0):
                want = modes.copy()
                want *= np.float32(1 / side**3)
                dimensions = (C.c_int * 3)(side, side, side)
                native(want.ctypes.data, dimensions, 5, inner, outer, rstar)
                got = cp.array(modes)
                grid, block = ((got.size + 127) // 128,), (128,)
                shell = (
                    np.int32(side),
                    np.float64(2 * np.pi / 24),
                    *[np.float64(np.float32(v)) for v in (inner, outer, rstar)],
                )
                kernel(grid, block, (got, *shell))
                actual = got.get()
                np.testing.assert_allclose(actual, want, rtol=1e-5, atol=1e-11)
                other_modes = np.ascontiguousarray(modes[::-1])
                other = cp.asarray(other_modes)
                kernel(grid, block, (other, *shell))
                other_reference = other_modes.copy()
                other_reference *= np.float32(1 / side**3)
                native(other_reference.ctypes.data, dimensions, 5, inner, outer, rstar)
                np.testing.assert_array_equal(other.get(), other_reference)
                out.append(
                    dict(
                        side=side,
                        inner=inner,
                        outer=outer,
                        rstar=rstar,
                        n=want.size,
                        different=int(np.count_nonzero(want != actual)),
                        max_abs=float(np.abs(want - actual).max()),
                    )
                )
assert kernel.builds == kernel.reuses == len(out)
print(
    json.dumps(
        dict(
            job=os.getenv("SLURM_JOB_ID"),
            cases=out,
            kernel_builds=kernel.builds,
            kernel_reuses=kernel.reuses,
            passed=True,
        )
    )
)
