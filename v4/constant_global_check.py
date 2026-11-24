"""Native / CUDA 4.3.2 unconditional mass integrals, with one shared sigma(M) table."""

import ctypes as C
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)

import evolution_backend as ev

PROBE_SOURCE = """
void probe_table(int gpu, int sfr, float zlo, float zhi) {
    ScalingConstants sc;
    set_scaling_constants(zlo,&sc,false);
    if (gpu) gpu_global_table(zpp_interp_points_SFR,zlo,zhi,&sc,sfr);
    else if (sfr) initialize_sfrd_unconditional_tables(zpp_interp_points_SFR,zlo,zhi,&sc);
    else initialize_nion_unconditional_tables(zpp_interp_points_SFR,zlo,zhi,&sc);
}
void probe_scalar(int gpu, int stage, double z, double zlim, double *result, double *bounds) {
    ScalingConstants sc;
    set_scaling_constants(z,&sc,false);
    double m=log(minimum_source_mass(z,false)), upper=log(M_MAX_INTEGRAL);
    bounds[0]=exp(m); bounds[1]=minimum_source_mass(z,true);
    bool sfr = stage==29;
    double z2=sfr ? z : zlim;
    if (!gpu) {
        result[0]=nion_unconditional_acg(z,m,upper,sc.mturn_acg_homogeneous,&sc);
        result[1]=sfr ? sfrd_unconditional_acg(z2,m,upper,sc.mturn_acg_homogeneous,&sc)
                      : nion_unconditional_acg(z2,m,upper,sc.mturn_acg_homogeneous,&sc);
    } else {
        double redshifts[]={z,z2};
        gpu_global_pair(stage,2,redshifts,sfr,m,upper,sc.mturn_acg_homogeneous,&sc,result);
    }
}
"""


def compare(ref, actual):
    assert actual.shape == ref.shape
    assert np.isfinite(ref).all() and np.isfinite(actual).all()
    delta = abs(actual - ref)
    return dict(
        count=len(ref),
        range=[float(ref.min()), float(ref.max())],
        max_abs=float(delta.max()),
        max_rel=float((delta / np.maximum(abs(ref), 1e-100)).max()),
        different_float32=int(np.count_nonzero(ref.astype("f") != actual.astype("f"))),
    )


def verify(output):
    _, cuda, nsc = ev.thermal_source(False)
    engine = ev.Evolution(cuda, nsc, False)
    report = {
        "job": os.getenv("SLURM_JOB_ID"),
        "cuda_device": ev.cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "sigma": "same native 4.3.2 mass/slope tables",
        "cases": {},
    }
    cb = ev.CB(engine.dispatch)
    with TemporaryDirectory() as tmp:
        source = ev.radiation_setup_source() + PROBE_SOURCE
        ad = ev.compile_adapter(
            Path(tmp),
            {
                "probe.c": source,
                "dispatch.c": "int (*gpu_dispatch)(int,int,double*,void**);",
            },
            ["-DGPU_TABLE_AUDIT=0"],
        )
        C.c_void_p.in_dll(ad, "gpu_dispatch").value = C.cast(cb, C.c_void_p).value
        ad.probe_table.argtypes = [C.c_int, C.c_int, C.c_float, C.c_float]
        ptr = np.ctypeslib.ndpointer(dtype="d", flags="C_CONTIGUOUS")
        ad.probe_scalar.argtypes = [C.c_int, C.c_int, C.c_double, C.c_double, ptr, ptr]
        for hmf, source_model, threshold in (
            ("ST", "CONST-ION-EFF", "virial"),
            ("PS", "CONST-ION-EFF", "virial"),
            ("ST", "CONST-ION-EFF", "mass"),
            ("ST", "E-INTEGRAL", "mass"),
        ):
            title = f"{source_model}/{hmf}/{threshold}"
            case = {}
            report["cases"][title] = case
            inputs = p21.InputParameters(
                random_seed=81033, node_redshifts=(35.0, 7.0)
            ).evolve_input_structs(
                DIM=32,
                HII_DIM=16,
                BOX_LEN=24.0,
                N_THREADS=1,
                HMF=hmf,
                SOURCE_MODEL=source_model,
                USE_TS_FLUCT=True,
                USE_METALLICITY=False,
                USE_EXP_FILTER=False,
                USE_UPPER_STELLAR_TURNOVER=False,
                M_MIN_in_Mass=threshold == "mass",
                ION_Tvir_MIN=np.log10(57000.0),
                X_RAY_Tvir_MIN=np.log10(8500.0),
                M_TURN_STELLAR_FEEDBACK=8.42,
                HII_EFF_FACTOR=27.5,
                F_STAR10_ACG=-1.44,
            )
            manager.init(inputs, broadcast_inputs=True, ps=True, sigma=True)
            for sfr in (0, 1):
                f = "SFRD_z_table_acg" if sfr else "Nion_z_table_acg"
                vals = []
                axes = []
                times = []
                tab = ev.Table.from_address(engine.symbols[f])
                for gpu in (0, 1):
                    t = perf_counter()
                    ad.probe_table(gpu, sfr, 7.6666, 30.011)
                    times.append(perf_counter() - t)
                    if engine.error:
                        raise engine.error
                    vals.append(
                        ev.view(C.cast(tab.y, C.c_void_p).value, tab.n, "d").copy()
                    )
                    axes.append((tab.low, tab.width))
                case[f] = compare(*vals) | dict(
                    exact_axis=axes[0] == axes[1], seconds=times
                )
            for stage in (22, 29):
                vals = [[], []]
                for z in (7.123456, 11.89123, 22.31234):
                    for gpu in (0, 1):
                        result = np.empty(2)
                        bounds = np.empty(2)
                        ad.probe_scalar(gpu, stage, z, 34.987, result, bounds)
                        if engine.error:
                            raise engine.error
                        vals[gpu].extend(result)
                    case.setdefault("mass_limits_by_redshift", {})[z] = list(bounds)
                case[f"stage_{stage}"] = compare(*(np.asarray(a) for a in vals))
            print(title, case, flush=True)
            C.CDLL(str(ev.NATIVE)).free_unconditional_tables()
            manager.free()
    report["inputs"] = {
        "random_seed": 81033,
        "grid": [32, 16],
        "box_mpc": 24.0,
        "N_THREADS": 1,
        "HII_EFF_FACTOR": 27.5,
        "log10_F_STAR10_ACG": -1.44,
        "ION_Tvir_MIN_kelvin": 57000.0,
        "X_RAY_Tvir_MIN_kelvin": 8500.0,
        "log10_M_TURN_STELLAR_FEEDBACK": 8.42,
        "history_redshifts": [7.6666, 30.011],
        "history_nodes": 400,
        "ionization_limit_redshift": 34.987,
    }
    report["passed"] = all(
        rec.get("max_rel", 0) < 1e-10
        and rec.get("different_float32", 0) == 0
        and rec.get("exact_axis", True)
        for v in report["cases"].values()
        for rec in v.values()
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit(
            "An unconditional source integral differs from the native reference"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
