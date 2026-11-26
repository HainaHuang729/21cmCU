"""Compute the turnover masses entering the native MCG source grids."""

import ctypes as C
import re
from functools import lru_cache

import cupy as cp
import numpy as np

from evolution_backend import ROOT, SHARED, SRC, module, view


def source(native):
    start = native.index("void get_log10_turnovers(")
    pos = native.index("    double log10_mturn_mcg_avg", start)
    hook = """
    if (astro_options_global->USE_MCGS ||
        astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK) {
        bool history = consts->redshift < simulation_options_global->Z_HEAT_MAX;
        bool mcg = astro_options_global->USE_MCGS;
        bool photo = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK;
        bool streaming = matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS;
        double p[]={consts->redshift, consts->mturn_acg_homogeneous,
            astro_params_global->M_TURN_STELLAR_FEEDBACK, consts->vcb_const,
            cosmo_tables_global->V_CB_AVG * sqrt(3*M_PI/8),
            astro_params_global->A_LW, astro_params_global->BETA_LW,
            astro_params_global->A_VCB, astro_params_global->BETA_VCB};
        void *a[]={(mcg && history) ? previous_spin_temp->J_21_LW : NULL,
            (mcg && streaming) ? ini_boxes->lowres_vcb : NULL,
            (photo && history) ? previous_ionize_box->ionisation_rate_G12 : NULL,
            (photo && history) ? previous_ionize_box->z_reion : NULL,
            photo ? log10_mturn_acg_grid : NULL,
            mcg ? log10_mturn_mcg_grid : NULL, averages};
        if (cuda_mcg_turnovers(HII_TOT_NUM_PIXELS,p,a)) Throw(ValueError);
        return;
    }
"""
    callback = "int (*cuda_mcg_turnovers)(int, double *, void **);\n"
    return native[:start] + callback + native[start:pos] + hook + native[pos:]


@lru_cache(maxsize=1)
def _kernel():
    # Retain the installed native fit coefficients, redshift guard and float
    # arguments for the Sobacchi & Mesinger photoheating feedback.
    chem = (SRC / "thermochem.c").read_text()
    constants = re.findall(r"^#define (?:REION_SM13_\w+|HALO_BIAS) .*$", chem, re.M)
    start = chem.index("double reionization_feedback(")
    end = chem.index("\n}", start) + 2
    native = "\n".join(constants) + "\n__device__ " + chem[start:end]
    return module(
        [SHARED / "glibc_math.cuh"],
        native + "\n" + (ROOT / "halo_mcg_turnover.cu").read_text(),
    ).get_function("log_turnover")


def device_turnovers(n, p, a, *, registry=None):
    """Return both unrounded logarithms for the mean and the native float grids."""
    params = np.ctypeslib.as_array(p, (9,))

    def load(addr):
        if not addr:
            return np.uint64(0)
        if registry is not None:
            cached = registry.lookup_ptr(int(addr), n, "f")
            if cached is not None:
                return cached
        return cp.asarray(view(addr, n, "f"))

    fields = [load(a[j]) for j in range(4)]
    photo = bool(a[4])
    mcg_enabled = bool(a[5])
    # Native accumulates log10 of the float threshold *before* narrowing
    # it into the output grid. Preserve that distinction for Nion history.
    acg = cp.empty(n, dtype="d") if photo else np.uint64(0)
    mcg = cp.empty(n, dtype="d") if mcg_enabled else np.uint64(0)
    acg_grid = cp.empty(n, dtype="f") if photo else np.uint64(0)
    mcg_grid = cp.empty(n, dtype="f") if mcg_enabled else np.uint64(0)
    _kernel()(
        ((n + 255) // 256,),
        (256,),
        (
            np.int32(n),
            cp.asarray(params),
            *fields,
            acg,
            mcg,
            acg_grid,
            mcg_grid,
        ),
    )
    return acg, mcg, acg_grid, mcg_grid


def bind(adapter, report, errors, *, registry=None):
    cb = C.CFUNCTYPE(C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_void_p))

    @cb
    def run(n, p, a):
        try:
            acg, mcg, acg_grid, mcg_grid = device_turnovers(n, p, a, registry=registry)
            averages = view(a[6], 2, "d")
            if a[4]:
                acg_grid.get(out=view(a[4], n, "f"))
                averages[0] = acg.mean().item()
            else:
                averages[0] = np.log10(p[1])
            if a[5]:
                mcg_grid.get(out=view(a[5], n, "f"))
                averages[1] = mcg.mean().item()
            else:
                averages[1] = 0.0
            report["mcg_turnover_grids"] = report.get("mcg_turnover_grids", 0) + 1
            report["mcg_turnover_cells"] = report.get("mcg_turnover_cells", 0) + n
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_mcg_turnovers").value = C.cast(
        run, C.c_void_p
    ).value
    return run
