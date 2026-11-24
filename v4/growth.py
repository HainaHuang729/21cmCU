"""Expansion and growth coefficients for thermal evolution and perturbations."""

import ctypes
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import py21cmfast
from py21cmfast.drivers._global_initialization import c_state

FIELDS = ("OMm", "OMl", "OMr", "OMtot", "wl", "hlittle")


def device_source():
    src = (Path(py21cmfast.__file__).parent / "src/cosmology.c").read_text()
    pieces = [
        "__device__ inline double expansion_pow(double x,double y){return pow(x,y);}\n__device__ inline double expansion_sqrt(double x){return sqrt(x);}"
    ]
    for name in ("dicke", "dtdz", "hubble", "ddicke_dz"):
        a = src.index("double " + name + "(")
        b = src.index("\n}", a) + 2
        func = src[a:b]
        func = re.sub(r"(?:LOG_ERROR|LOG_WARNING)\([^;]+;", "", func)
        func = func.replace("Throw(ValueError);", 'return nan("");')
        func = re.sub(r"\bHo\b", "(((float)c[5])*3.2407e-18)", func)
        for i, field in enumerate(FIELDS):
            func = func.replace("cosmo_params_global->" + field, f"((float)c[{i}])")
        func = func.replace(") {", ",const double *c) {", 1)
        if name == "ddicke_dz":
            func = func.replace("dicke(z + dz)", "dicke(z + dz,c)").replace(
                "dicke(z)", "dicke(z,c)"
            )
        if name == "dicke":
            # Mirror the official x86-64 build's reassociation. A single ulp in
            # D matters when the upstream derivative subtracts at dz=1e-10.
            begin = func.index("        omegaM_z =")
            end = func.index("    } else if", begin)
            func = (
                func[:begin]
                + """
        double x=1.+z,x2=x*x;
        float m=c[0];
        double matter=((double)m*x)*x2;
        double om=matter/(((x2*x2)*(float)c[2]+matter)+(float)c[1]);
        double f0=2.5*m/(((double)((209.f-m)*m)*(1./140.)+1./70.)+pow((double)m,4./7.));
        double den=((209.-om)*(om*(1./140.))+1./70.)+pow(om,4./7.);
        return (2.5*om)/((den*f0)*x);
"""
                + func[end:]
            )
        # NVRTC supplies real-power overloads (the host C library takes doubles).
        func = re.sub(r", (-?\d+)\)", r", \1.0)", func)
        pieces.append(
            "__device__ "
            + func.replace("pow(", "expansion_pow(").replace("sqrt(", "expansion_sqrt(")
        )
    return "\n".join(pieces)


@lru_cache(maxsize=1)
def _native_growth_derivative():
    derivative = ctypes.CDLL(py21cmfast.c_21cmfast.__file__).ddickedt
    derivative.argtypes = [ctypes.c_double]
    derivative.restype = ctypes.c_double
    return derivative


def perturb_coefficients(inputs, z):
    z = float(np.float32(z))
    # The native finite difference at dz=1e-10 amplifies tiny libm rounding
    # differences. Read all three node scalars from the same upstream state.
    with c_state(inputs, broadcast_inputs=True):
        dicke = py21cmfast.c_21cmfast.lib.dicke
        growth = dicke(z)
        initial_growth = dicke(
            float(np.float32(inputs.simulation_options.INITIAL_REDSHIFT))
        )
        velocity_rate = _native_growth_derivative()(z) / growth
    coefficients = np.array([growth, initial_growth, velocity_rate])
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite perturbation growth coefficient")
    return coefficients
