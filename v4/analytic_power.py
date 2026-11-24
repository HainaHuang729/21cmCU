"""Evaluate the native analytic transfer functions with CUDA mass quadrature."""

import re

import cupy as cp
import numpy as np

from evolution_backend import ROOT, SHARED, SRC, module


class AnalyticPower:
    def __init__(self, spectrum, omega_m, hubble, omega_b):
        self.args = (np.int32(spectrum), *map(np.float32, (omega_m, hubble, omega_b)))
        native = (SRC / "cosmology.c").read_text()
        names = ("BBKS", "Efstathiou", "Peebles", "White")
        functions = [
            "__device__ double analytic_exp(double x) { return exp(x); }",
            "__device__ double square(double x) { return x*x; }",
            "__device__ double cube(double x) { return x*x*x; }",
            "__device__ double fourth(double x) { double y=x*x; return y*y; }",
        ]
        for name in names:
            header = f"double transfer_function_{name}(double k)"
            start = native.index(header)
            body = native[start : native.index("\n}", start) + 2]
            body = body.replace("exp(", "analytic_exp(")
            # This NVRTC build has no device powi symbol.
            body = re.sub(
                r"pow\(([^(),\n]+),\s*([234])\)",
                lambda match: (
                    f"{('square', 'cube', 'fourth')[int(match[2]) - 2]}({match[1]})"
                ),
                body,
            )
            functions.append(
                re.sub(
                    r"cosmo_params_global->(OMm|hlittle|OMb)",
                    lambda match: {"OMm": "om", "hlittle": "h", "OMb": "ob"}[match[1]],
                    body.replace(
                        header,
                        f"__device__ {header[:-1]}, float om, float h, float ob)",
                    ),
                )
            )
        cases = "\n".join(
            f"case {i}: tf = transfer_function_{name}(k, om, h, ob); break;"
            for i, name in enumerate(names, 1)
        )
        source = """
struct AnalyticDensity {
    int spectrum;
    float om, h, ob;
    __device__ double power(double k, const double *p) const {
        if (k == 0.) return 0.;
        double tf;
        switch (spectrum) {
            TRANSFER_CASES
            default: return nan("");
        }
        tf *= k*k;
        double primordial = p[8]*pow(k/.05, p[9]-1.);
        return p[7]*primordial*tf*tf/(k*k*k);
    }
};
extern "C" __global__ void analytic_power_grid(
    int n, const double *k, const double *p, int spectrum,
    float om, float h, float ob, double *out
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = AnalyticDensity{spectrum, om, h, ob}.power(k[i], p);
}
""".replace("TRANSFER_CASES", cases)
        compiled = module(
            [SHARED / "table_qag.cuh"],
            "\n".join(functions) + source + (ROOT / "sigma_moments.cu").read_text(),
            options=("-DUSE_ANALYTIC_POWER",),
        )
        self.moments = compiled.get_function("moments")
        self.kernel = compiled.get_function("analytic_power_grid")

    def power(self, k, cosmology, norm, index):
        k = cp.ascontiguousarray(k, dtype="d")
        out = cp.empty_like(k)
        params = cp.asarray(np.append(cosmology, (norm, index)))
        self.kernel(
            ((k.size + 255) // 256,),
            (256,),
            (np.int32(k.size), k, params, *self.args, out),
        )
        return out
