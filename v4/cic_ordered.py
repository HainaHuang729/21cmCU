from pathlib import Path

import cupy as cp
import numpy as np

_module = cp.RawModule(
    code=Path(__file__).with_suffix(".cu").read_text(), options=("--fmad=false",)
)


def absmax(a):
    return cp.maximum(abs(a.min()), abs(a.max()))


def deposit_ordered(
    rho, vel, second, mass, n_shape, m_shape, growth, first_scales, second_scales, two
):
    nx, ny, nz = (int(d) for d in n_shape)
    mx, my, mz = (int(d) for d in m_shape)
    counts = cp.zeros(mass.shape, cp.int32)
    flags = cp.empty(mass.shape, cp.bool_)
    if nx == ny == nz and mx == my == mz and nx == 4 * mx and mx >= 2:
        params = (
            np.int32(nx),
            np.int32(mx),
            growth,
            first_scales[0],
            second_scales[0],
            np.bool_(two),
        )
        _module.get_function("scatter_tile4")(
            ((mx**3 + 63) // 64,), (64,), (rho, vel, second, mass, counts, *params)
        )
    else:
        ratio = np.float64(mx / nx)
        params = (
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(mx),
            np.int32(my),
            np.int32(mz),
            ratio,
            np.float64(growth),
            np.float64(first_scales[0]),
            np.float64(first_scales[1]),
            np.float64(first_scales[2]),
            np.float64(second_scales[0]),
            np.float64(second_scales[1]),
            np.float64(second_scales[2]),
            np.bool_(two),
        )
        total_n = nx * ny * nz
        _module.get_function("scatter")(
            ((total_n + 255) // 256,),
            (256,),
            (rho, vel, second, mass, counts, *params),
        )
    bound = 1.0 + abs(growth) * absmax(rho).astype(cp.float64)
    _module.get_function("uncertain")(
        ((mass.size + 255) // 256,),
        (256,),
        (mass, counts, bound, flags, np.int64(mass.size)),
    )
    sites = cp.flatnonzero(flags)
    if sites.size:
        shifts = cp.stack(
            [
                abs(first_scales[a]) * absmax(vel[a]).astype(cp.float64)
                + (
                    abs(second_scales[a]) * absmax(second[a]).astype(cp.float64)
                    if two
                    else 0.0
                )
                for a in range(3)
            ]
        )
        ratio = np.float64(mx / nx)
        replay_params = (
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(mx),
            np.int32(my),
            np.int32(mz),
            ratio,
            np.float64(growth),
            np.float64(first_scales[0]),
            np.float64(first_scales[1]),
            np.float64(first_scales[2]),
            np.float64(second_scales[0]),
            np.float64(second_scales[1]),
            np.float64(second_scales[2]),
            np.bool_(two),
        )
        if 1 in (mx, my, mz):
            # Each of the eight corners can alias the single cell on a 1-wide axis.
            kernel = _module.get_function("replay")
            blocks, threads = (sites.size + 31) // 32, 32
        else:
            kernel = _module.get_function("replay_block")
            blocks, threads = sites.size, 256
        kernel(
            (blocks,),
            (threads,),
            (
                rho,
                vel,
                second,
                mass,
                sites,
                np.int32(sites.size),
                shifts,
                *replay_params,
            ),
        )
    return sites.size
