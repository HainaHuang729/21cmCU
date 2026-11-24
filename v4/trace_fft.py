"""Compare FFTW, cuFFT and CUDA codelets using identical mass and Fourier arrays."""

import ctypes as C
import json
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state
from tests import produce_integration_test_data as official

import perturb
from ordered_fft import transform


def native_stage(lib, x, action, radius):
    n = x.shape[0]
    shape = (n, n, 2 * (n // 2 + 1))
    lib.fftwf_malloc.restype = C.c_void_p
    ptr = C.c_void_p(lib.fftwf_malloc(C.c_size_t(4 * int(np.prod(shape)))))
    if not ptr:
        raise MemoryError("FFTW buffer allocation failed")
    try:
        a = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_float)), shape)
        spectrum = a.view(np.complex64)
        if action == "forward":
            a[:, :, :n] = x
            lib.dft_r2c_cube(False, n, n, 2, ptr)
            return spectrum.copy()
        spectrum[:] = x
        if action == "window":
            lib.filter_box(
                ptr,
                (C.c_int * 3)(n, n, n),
                0,
                C.c_float(radius),
                C.c_float(0),
                C.c_float(0),
            )
            return spectrum.copy()
        lib.dft_c2r_cube(False, n, n, 2, ptr)
        return a[:, :, :n].copy()
    finally:
        lib.fftwf_free(ptr)


def compare():
    z, options = official.OPTIONS_PT["highres"]
    inputs = official.get_all_options_struct(z, **options)[
        "inputs"
    ].evolve_input_structs(KEEP_3D_VELOCITIES=True)
    ic = p21.compute_initial_conditions(inputs=inputs)
    kw = dict(redshift=z, initial_conditions=ic)
    ref = p21.perturb_field(**kw).get("density")
    captured = []
    reference = perturb.reference_density

    def capture(lib, inputs, grid):
        captured.append(grid.copy())
        return reference(lib, inputs, grid)

    with patch.object(perturb, "reference_density", capture):
        native = perturb.perturb_field_gpu(**kw, reference_fft=True).get("density")
        np.testing.assert_array_equal(native, ref)
    fixed = perturb.perturb_field_gpu(**kw).get("density")
    (mass,) = captured
    n = mass.shape[0]
    low = ref.shape[0]
    length = inputs.simulation_options.BOX_LEN
    factor = np.float32(0.620350491 * length / low)
    k2 = sum(k * k for k in perturb.modes(n, length))
    kr = (cp.sqrt(k2) * factor).astype(cp.float64)
    window = cp.where(
        kr < 1e-4,
        1 - kr**2 / 10,
        3 * (cp.sin(kr) - kr * cp.cos(kr)) / cp.where(kr == 0, 1, kr) ** 3,
    ).astype(cp.float32)
    idx = cp.floor(cp.arange(low) * (n / low) + 0.5).astype(cp.int32)
    old = cp.fft.irfftn(cp.fft.rfftn(mass.astype(cp.float64)) * window, s=(n,) * 3)
    old = cp.maximum(
        old[cp.ix_(idx, idx, idx)].astype(cp.float32) - 1, np.float32(-1 + 1e-7)
    ).get()
    diff = abs(old.astype(float) - ref)
    failed = diff > 1e-8 + 1e-4 * abs(ref.astype(float))
    target = np.unravel_index(np.argmax(np.where(failed, diff, -1)), diff.shape)
    site = tuple(int(idx[t]) for t in target)
    stats = {}
    lib = C.CDLL(p21.c_21cmfast.__file__)
    with c_state(inputs, broadcast_inputs=True):
        for size, a in ((n, mass), (low, mass[cp.ix_(idx, idx, idx)].copy())):
            f = native_stage(lib, a.get(), "forward", factor)
            spectrum = native_stage(lib, f, "window", factor) if size == n else f
            inv = native_stage(lib, spectrum, "inverse", factor)
            sf = cp.asarray(spectrum)
            row = {}
            for kind, ff, ii in (
                (
                    "cufft_fp32",
                    cp.fft.rfftn(a),
                    cp.fft.irfftn(sf, s=(size,) * 3, norm="forward"),
                ),
                (
                    "cufft_fp64",
                    cp.fft.rfftn(a.astype(cp.float64)),
                    cp.fft.irfftn(
                        sf.astype(cp.complex128), s=(size,) * 3, norm="forward"
                    ),
                ),
                ("ordered_gpu", transform(a, size), transform(sf, size, True)),
            ):
                fv, iv = ff.get(), ii.get()
                row[kind] = dict(
                    forward_max=float(abs(fv - f).max()),
                    inverse_max=float(abs(iv - inv).max()),
                    forward_different=int((fv != f).sum()),
                    inverse_different=int((iv != inv).sum()),
                )
                if size == n:
                    row[kind].update(
                        site_cpu_raw=float(inv[site]), site_gpu_raw=float(iv[site])
                    )
            stats[size] = row
    return dict(
        target=[int(i) for i in target],
        fine_site=site,
        original_failures=int(failed.sum()),
        target_cpu=float(ref[target]),
        target_previous=float(old[target]),
        target_fixed=float(fixed[target]),
        transform_comparisons=stats,
    )


if __name__ == "__main__":
    import sys

    Path(sys.argv[1]).write_text(json.dumps(compare(), indent=2) + "\n")
