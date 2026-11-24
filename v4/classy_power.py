"""Evaluate native CLASS density and relative-velocity transfers on CUDA."""

from pathlib import Path

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline


class ClassPower:
    @staticmethod
    def _coefficients(k, transfer):
        # GSL's gsl_interp_cspline uses natural boundary conditions in linear k.
        if len(k) < 3 or k[0] != 0 or not np.all(np.diff(k) > 0):
            raise ValueError("Invalid CLASS density transfer wavenumbers")
        if transfer[0] != 0 or not np.all(np.isfinite(transfer)):
            raise ValueError("Invalid CLASS transfer values")
        rows = len(k)
        coefficients = np.zeros((5, rows))
        coefficients[:2] = k, transfer
        coefficients[2:, :-1] = CubicSpline(k, transfer, bc_type="natural").c[2::-1]
        return coefficients

    def __init__(self, k, transfer, eh_at_kmax, velocity=None, *, suppress_streaming):
        rows = len(k)
        coefficients = self._coefficients(k, transfer)
        coefficients[-1, -1] = transfer[-1] / k[-1] / k[-1] / eh_at_kmax
        self.args = (
            cp.asarray(coefficients),
            np.int32(rows),
            np.int32(suppress_streaming),
        )
        self.velocity = (
            cp.asarray(self._coefficients(k, velocity))
            if velocity is not None
            else None
        )
        root = Path(__file__).resolve().parent
        parts = [
            root.parent / "lightcone/table_qag.cuh",
            root / "classy_power.cu",
            root / "sigma_moments.cu",
        ]
        module = cp.RawModule(
            code="\n".join(p.read_text() for p in parts),
            options=("--fmad=false", "-DUSE_CLASS_POWER"),
        )
        self.moments = module.get_function("moments")
        self.kernel = module.get_function("class_power_grid")

    def power(self, k, cosmology, norm, index, *, velocity=False):
        k = cp.ascontiguousarray(k, dtype="d")
        out = cp.empty_like(k)
        params = cp.asarray(np.append(cosmology, (norm, index)))
        if velocity and self.velocity is None:
            raise ValueError(
                "Relative velocity requires a native CLASS velocity transfer"
            )
        table, rows, suppress_streaming = self.args
        self.kernel(
            ((k.size + 255) // 256,),
            (256,),
            (
                np.int32(k.size),
                k,
                params,
                self.velocity if velocity else table,
                rows,
                suppress_streaming,
                np.int32(velocity),
                out,
            ),
        )
        return out
