"""Shared host launcher for IONISE_ENTIRE_SPHERE=True GPU bubble painting."""

import math

import cupy as cp
import numpy as np


def paint_ion_spheres(
    module,
    *,
    shape,
    radius,
    box_len,
    xh,
    triggered,
    eligible,
    gamma_cand,
    gamma12,
    mfp,
    track_painter=False,
    write_xh=True,
):
    """Paint all triggered spheres on device and commit first-crossing Gamma12/MFP."""
    nx, ny, nz = shape
    n = nx * ny * nz
    centers = cp.flatnonzero(triggered).astype(cp.int32)
    n_centers = int(centers.size)
    if n_centers == 0 or radius < 0.0:
        return None
    # Mirror bubble_helper_progs.c:update_in_sphere float parameter + double ceil/pow casts.
    r_norm = float(np.float32(float(radius) / float(box_len)))
    r_in = int(math.ceil(r_norm / math.sqrt(3.0) * nx)) - 1
    r_out = int(math.ceil(r_norm * nx))
    rsq = np.float32((r_norm * float(nx)) ** 2)
    use_rec = eligible is not None
    need_painter = use_rec or track_painter
    min_painter = cp.full(n, 0x7FFFFFFF, dtype=cp.int32) if need_painter else np.uint64(0)
    eligible_arg = eligible if use_rec else np.uint64(0)
    xh_arg = xh if write_xh else np.uint64(0)

    rx_cnt = min(r_out, (nx - 1) // 2) - (-min(r_out, nx // 2)) + 1
    ry_cnt = min(r_out, (ny - 1) // 2) - (-min(r_out, ny // 2)) + 1
    rz_cnt = min(r_out, (nz - 1) // 2) - (-min(r_out, nz // 2)) + 1
    box_vol = rx_cnt * ry_cnt * rz_cnt

    if n_centers * box_vol <= max(64 * n, 16_000_000):
        module.get_function("sphere_paint_sparse")(
            (n_centers,),
            (256,),
            (
                np.int32(n_centers),
                centers,
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.int32(r_in),
                np.int32(r_out),
                rsq,
                xh_arg,
                min_painter,
            ),
        )
    else:
        module.get_function("sphere_paint_pull")(
            ((n + 255) // 256,),
            (256,),
            (
                np.int32(n),
                np.int32(n_centers),
                centers,
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.int32(r_in),
                rsq,
                xh_arg,
                min_painter,
            ),
        )
    if use_rec:
        module.get_function("sphere_commit_gamma")(
            ((n_centers + 255) // 256,),
            (256,),
            (
                np.int32(n_centers),
                centers,
                eligible_arg,
                gamma_cand,
                min_painter,
                np.float32(radius),
                gamma12,
                mfp if mfp is not None else np.uint64(0),
            ),
        )
    return min_painter if need_painter else None
