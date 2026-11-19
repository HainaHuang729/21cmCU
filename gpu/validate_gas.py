"""Compare the ionized gas temperature kernel with the native temperature formula."""
import ctypes as C
import json
from pathlib import Path
import sys

import cupy as cp
import numpy as np
import py21cmfast.c_21cmfast as native


destination = Path(sys.argv[1])
lib = C.CDLL(native.__file__)
reference = lib.ComputeFullyIoinizedTemperature
reference.argtypes = [C.c_float] * 3
reference.restype = C.c_float
tre, floor = native.lib.global_params.T_RE, native.lib.global_params.MIN_DENSITY_LOW_LIMIT
kernel = cp.RawKernel(Path(__file__).with_name("ion.cu").read_text(),
                      "ion_temperature", options=("--fmad=false",))
results = []
for z in (5., 5.367247581481934, 10.307236671447754, 15.):
    rng = np.random.default_rng(80000)
    n = 4096
    density = (np.exp(rng.normal(size=n)) - 1.).astype(np.float32)
    density[::16] = np.float32(-1. + floor)
    zre = rng.uniform(z, max(16., z), size=n).astype(np.float32)
    zre[::8] = np.float32(z)
    zre[1::8] = np.float32(-1.)
    xhi = np.zeros(n,dtype=np.float32)
    xhi[2::8] = 0.5
    tk = rng.uniform(10., 3e4, size=n).astype(np.float32)
    previous = rng.uniform(10., 3e4, size=n).astype(np.float32)
    expected = previous.copy()
    for i in np.flatnonzero((zre > 0.) & (xhi < 1e-30)):
        expected[i] = max(reference(zre[i], z, density[i]), tk[i])
    actual = cp.asarray(previous)
    kernel(((n+255)//256,), (256,),
           (*[cp.asarray(x) for x in (density,xhi,zre,tk)],actual,
            np.int32(n),np.float32(z),np.float32(tre),np.float32(floor)))
    error = np.abs(actual.get().astype(np.float64)-expected)
    results.append(dict(z=z, count=n, failed=int((error > 1e-8+1e-4*abs(expected)).sum()),
                        max_abs=float(error.max()), rtol=1e-4, atol=1e-8))
destination.write_text(json.dumps(results,indent=2)+"\n")
print(results, flush=True)
assert all(row["failed"]==0 for row in results)
