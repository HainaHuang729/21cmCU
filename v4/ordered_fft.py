"""GPU execution of the single-precision, non-FMA SSE2 FFTW plans verified on a8."""

from pathlib import Path
import cupy as cp
import numpy as np


@cp.memoize(for_each_device=True)
def module(size=50):
    source = Path(__file__).with_name("fftw_gpu")
    if size == 64:
        return cp.RawModule(
            code="\n".join(
                (source / f).read_text()
                for f in ("lanes.cuh", "direct64.cuh", "schedule64.cu")
            ),
            options=("--std=c++11", "--fmad=false"),
            name_expressions=[
                f"line64<{direction},{kind}>"
                for direction in ("true", "false")
                for kind in ("true", "false")
            ],
        )
    names = [
        f"{kind}<{n},{direction}>"
        for kind in ("axis_fft", "real_fft")
        for n in (50, 150)
        for direction in ("false", "true")
    ]
    return cp.RawModule(
        code="\n".join(
            (source / f).read_text()
            for f in ("lanes.cuh", "codelets.cuh", "schedule.cu")
        ),
        options=("--std=c++11", "--fmad=false"),
        name_expressions=names,
    )


@cp.memoize(for_each_device=True)
def twiddles(n, radix, start):
    # Each adjacent pair holds the real and imaginary parts of exp(+2 pi i j k / n).
    phase = np.arange(start, n // radix)[:, None] * np.arange(1, radix)[None, :]
    values = np.exp(2j * np.pi * phase / n).astype(np.complex64)
    values.real[(4 * phase % (2 * n)) == n] = 0
    values.imag[(2 * phase % n) == 0] = 0
    return cp.asarray(values)


def transform(a, n, inverse=False):
    if n not in (50, 64, 150):
        raise ValueError("No verified FFTW schedule for this length")
    expected = (n, n, n // 2 + 1) if inverse else (n, n, n)
    dtype = cp.complex64 if inverse else cp.float32
    if a.shape != expected or a.dtype != dtype:
        raise ValueError(f"Expected {expected} array of {dtype} for this plan")
    flag = "true" if inverse else "false"
    for axis in (0, 1, 2) if inverse else (2, 1, 0):
        src = cp.ascontiguousarray(cp.moveaxis(a, axis, -1))
        real = axis == 2
        length = (n if inverse else n // 2 + 1) if real else n
        dtype = cp.float32 if real and inverse else cp.complex64
        dst = cp.empty((*src.shape[:-1], length), dtype=dtype)
        lines = src.size // src.shape[-1]
        if n == 64:
            kind = "true" if real else "false"
            kernel = module(64).get_function(f"line64<{flag},{kind}>")
            kernel(((lines + 63) // 64,), (64,), (src, dst, np.int32(lines)))
            a = cp.moveaxis(dst, -1, axis)
            continue
        if real:
            w = twiddles(n, 2 if n == 50 else 10, 1)
            second = twiddles(25, 5, 0)
        else:
            w = twiddles(n, 5 if n == 50 else 10, 0)
            second = twiddles(10, 5, 0)
        kernel = module().get_function(
            f"{'real_fft' if real else 'axis_fft'}<{n},{flag}>"
        )
        kernel(((lines + 63) // 64,), (64,), (src, dst, np.int32(lines), w, second))
        a = cp.moveaxis(dst, -1, axis)
    return a
