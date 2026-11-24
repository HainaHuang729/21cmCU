from pathlib import Path

import cupy as cp
import numpy as np

_module = cp.RawModule(
    code=Path(__file__).with_suffix(".cu").read_text(), options=("--fmad=false",)
)


def absmax(a):
    return cp.maximum(abs(a.min()), abs(a.max()))


def deposit_ordered(rho, vel, second, mass, n, m, growth, first, correction, two):
    counts = cp.zeros(mass.shape, cp.int32)
    flags = cp.empty(mass.shape, cp.bool_)
    params = np.int32(n), np.int32(m), growth, first, correction, np.bool_(two)
    if n == 4 * m and m >= 2:
        _module.get_function("scatter_tile4")(
            ((m**3 + 63) // 64,), (64,), (rho, vel, second, mass, counts, *params)
        )
    else:
        _module.get_function("scatter")(
            ((n**3 + 255) // 256,), (256,), (rho, vel, second, mass, counts, *params)
        )
    bound = 1.0 + abs(growth) * absmax(rho).astype(cp.float64)
    _module.get_function("uncertain")(
        ((mass.size + 255) // 256,),
        (256,),
        (mass, counts, bound, flags, np.int64(mass.size)),
    )
    sites = cp.flatnonzero(flags)
    if sites.size:
        shift = abs(first) * absmax(vel).astype(cp.float64)
        if two:
            shift += abs(correction) * absmax(second).astype(cp.float64)
        if m == 1:
            # Each of the eight corners aliases the sole output cell.
            kernel = _module.get_function("replay")
            blocks, threads = (sites.size + 31) // 32, 32
        else:
            kernel = _module.get_function("replay_block")
            blocks, threads = sites.size, 256
        kernel(
            (blocks,),
            (threads,),
            (rho, vel, second, mass, sites, np.int32(sites.size), shift, *params),
        )
    return sites.size
