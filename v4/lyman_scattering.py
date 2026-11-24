"""CUDA shell window using the installed 4.3.2 scattering prescription."""

import ctypes as C
import re
from pathlib import Path

import numpy as np
import cupy as cp

from evolution_backend import NATIVE, module


def build_window(source, *, reuse=False):
    native = (source / "filtering.c").read_text()
    struct_start = native.index("struct multiple_scattering_params {")
    struct_end = native.index("};", struct_start) + 2
    start = native.index("double asymptotic_2F3(")
    stop = native.index("void filter_box(", start)
    formulas = re.sub(
        r"^(double|void) (\w+\()",
        r"__device__ \1 \2",
        native[start:stop],
        flags=re.MULTILINE,
    )
    # The native release build reassociates this alternating recurrence. Close
    # to kR=30 the cancellation amplifies even one-bit changes in its terms.
    recurrence = (
        "term *= -1. / (1. + beta / (alpha + 2. * n)) / (1. + beta / (alpha + 1 + 2. * n)) * kR *\n"
        "                    kR / (2. * n) / (2. * n + 3.);"
    )
    if recurrence not in formulas:
        raise ValueError("Unexpected 4.3.2 Ly-alpha scattering recurrence")
    formulas = formulas.replace(
        recurrence,
        """double alpha_n = alpha + 2. * n;
            double denominator = (1. + beta / (alpha_n + 1.)) * (1. + beta / alpha_n);
            term *= (-0.5 / denominator) * (kR * kR) / (n * (2. * n + 3.));""",
    )
    # GSL defines reciprocal gamma at its poles as zero.
    preamble = """
#define DBL_MAX 1.7976931348623157e308
#define M_PI 3.14159265358979323846
#define pow(a, b) pow((double)(a), (double)(b))
__device__ double reciprocal_gamma(double x) {
    return x <= 0. && x == trunc(x) ? 0. : 1. / tgamma(x);
}
"""
    formulas = formulas.replace("gsl_sf_gammainv(", "reciprocal_gamma(")
    kernel = Path(__file__).with_suffix(".cu").read_text()
    device_code = preamble + native[struct_start:struct_end] + "\n" + formulas + kernel
    windows = module([], extra=device_code)
    window = windows.get_function("scattering_window")
    apply_cached = windows.get_function("reuse_scattering_window") if reuse else None
    initialize = C.CDLL(str(NATIVE)).initialize_alphas_and_betas_for_multiple_scattering
    initialize.argtypes = [C.c_double] * 3 + [np.ctypeslib.ndpointer(dtype=np.float64)]
    initialize.restype = None

    last_shell = None
    weights = None
    buffer_size = None

    def launch(grid, block, args):
        nonlocal last_shell, weights, buffer_size
        modes, side, *shell_args = args
        shell = (side, *shell_args)
        if reuse:
            if weights is not None and last_shell == shell:
                apply_cached(grid, block, (modes, side, weights))
                launch.reuses += 1
                return
            if buffer_size != modes.size:
                buffer_size = modes.size
                weights = None
                try:
                    weights = cp.empty(buffer_size, dtype=cp.float64)
                except cp.cuda.memory.OutOfMemoryError:
                    pass
        # These four shell scalars enter the numerically sensitive 2F3 series.
        inner, outer, r_star = args[-3:]
        parameters = np.empty(4, dtype=np.float64)
        initialize(inner, outer, r_star, parameters)
        window(
            grid,
            block,
            args[:-1] + (*parameters, weights if weights is not None else np.uintp(0)),
        )
        last_shell = shell
        launch.builds += 1

    launch.builds = launch.reuses = 0
    return launch
