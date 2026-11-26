"""General 3D CUDA FFT dispatch preserving the cubic even FFTW-plan hot path."""

import cupy as cp
import numpy as np

from fft_plan_cuda import UnsupportedPlan, transform


def transform_3d(data, shape, inverse=False, threads=2, wisdom=False, strict=False):
    """Run forward r2c or unnormalized inverse c2r 3D transform matching FFTW conventions."""
    nx, ny, nz = (int(d) for d in shape)
    if nx == ny == nz and nx % 2 == 0 and not wisdom:
        try:
            return transform(data, nx, inverse, threads)
        except UnsupportedPlan:
            if strict:
                raise
    if inverse:
        arr = cp.asarray(data, dtype=cp.complex64, order="C")
        return cp.ascontiguousarray(
            cp.fft.irfftn(arr, s=(nx, ny, nz), norm="forward").astype(
                cp.float32, copy=False
            )
        )
    arr = cp.asarray(data, dtype=cp.float32, order="C")
    return cp.ascontiguousarray(
        cp.fft.rfftn(arr).astype(cp.complex64, copy=False)
    )
