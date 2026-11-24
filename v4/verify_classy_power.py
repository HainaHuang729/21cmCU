"""Replay the native CLASS spectrum, mass moments, seeded initial fields and one 2LPT step."""

import argparse
import ctypes
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)

from evolution_backend import evolution_backend, native_tables
from perturb import perturb_field_gpu
from verify_power_setup import snapshot


def difference(expected, actual, scale=1.0):
    a = np.asarray(expected, dtype="d") * scale
    b = np.asarray(actual, dtype="d") * scale
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite data or shape mismatch")
    delta = abs(a - b)
    nz = a != 0
    return dict(
        values=a.size,
        unequal=int((a != b).sum()),
        max_abs=float(delta.max()),
        max_relative_nonzero=float((delta[nz] / abs(a[nz])).max(initial=0.0)),
        failed=int((delta > 1.0e-8 + 1.0e-4 * abs(a)).sum()),
        unit_scale=scale,
    )


def one(name, cosmo, seed, threads, length):
    inputs = p21.InputParameters(
        random_seed=seed,
        cosmo_params=p21.CosmoParams(**cosmo),
        node_redshifts=[10.0, 9.0],
    ).evolve_input_structs(
        DIM=16,
        Z_HEAT_MAX=10.0,
        SOURCE_MODEL="CHMF-SAMPLER",
        HII_DIM=8,
        BOX_LEN=length,
        R_BUBBLE_MAX=min(15.0, length),
        N_THREADS=threads,
        K_MAX_FOR_CLASS=20.0,
        POWER_SPECTRUM="CLASS",
        V_CB_MODEL="AVG-AUTO",
        PERTURB_ON_HIGH_RES=False,
        USE_MCGS=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        KEEP_3D_VELOCITIES=True,
    )
    # The same cached-property object supplies the official CLASS table to both paths.
    t = inputs.cosmo_tables
    k = np.asarray(t.transfer_density.x_values)
    high = k[-1]
    probes = np.unique(
        np.r_[
            k,
            (k[:-1] + k[1:]) * 0.5,
            np.nextafter(high, np.inf),
            [
                0.0,
                1e-8,
                1.01 * high,
                1.5 * high,
                3 * high,
                10 * high,
                100.0,
                300.0,
                1000.0,
                5000.0,
            ],
        ]
    )
    lib = ctypes.CDLL(p21.c_21cmfast.__file__)
    lib.power_in_k.argtypes = [ctypes.c_double]
    lib.power_in_k.restype = ctypes.c_double
    kw = dict(inputs=inputs, regenerate=True, write=False)
    syms = native_tables()
    z = 9.375
    manager.free()
    try:
        native_tables_out = snapshot(inputs, syms)
        expected_power = np.array([lib.power_in_k(v) for v in probes])
        ic = p21.compute_initial_conditions(**kw)
        evolved = p21.perturb_field(redshift=z, initial_conditions=ic, **kw)
    finally:
        manager.free()
    with TemporaryDirectory() as tmp:
        stages = Path(tmp) / "stages.json"
        with evolution_backend(stages, thermal=False) as engine:
            try:
                gpu_tables_out = snapshot(inputs, syms)
                params = gpu_tables_out["eh_constants"]
                # POWER_INDEX lives in the native float-valued CosmoParams struct.
                observed_power = engine.class_power.power(
                    cp.asarray(probes),
                    params,
                    t.ps_norm,
                    np.float32(inputs.cosmo_params.POWER_INDEX),
                ).get()
                got = p21.compute_initial_conditions(**kw)
                moved = perturb_field_gpu(
                    redshift=z,
                    initial_conditions=got,
                    inputs=inputs,
                    resident=engine,
                    strict_fft=True,
                )
            finally:
                manager.free()
        counted = json.loads(stages.read_text())
    sigma = {
        key: difference(exp, gpu_tables_out[key])
        for key, exp in native_tables_out.items()
    }
    fields = {key: difference(ic.get(key), got.get(key)) for key in ic.arrays}
    moved_fields = {
        key: difference(
            evolved.get(key), moved.get(key), 1.0e16 if key.startswith("vel") else 1.0
        )
        for key in evolved.arrays
    }
    inside = probes <= high
    pk = {
        "inside": difference(expected_power[inside], observed_power[inside]),
        "tail": difference(expected_power[~inside], observed_power[~inside]),
    }
    ok = (
        all(row["max_relative_nonzero"] < 1.0e-11 for row in pk.values())
        and all(
            row["unequal"] == 0 for key, row in sigma.items() if key != "eh_constants"
        )
        and all(
            row["failed"] == 0 for row in (*fields.values(), *moved_fields.values())
        )
        and counted["gpu_seed_and_modes"] > 0
        and counted["sigma_moments"] > 0
    )
    return dict(
        case=name,
        passed=ok,
        dim=16,
        hii_dim=8,
        box_len_Mpc=length,
        seed=seed,
        threads=threads,
        redshift=z,
        power_spectrum="CLASS",
        v_cb_model="AVG-AUTO",
        native_vcb_avg_km_s=t.V_CB_AVG,
        class_knots=len(k),
        requested_k_max_for_class_inverse_Mpc=inputs.simulation_options.K_MAX_FOR_CLASS,
        last_k_inverse_Mpc=high,
        norm="sigma8" if t.USE_SIGMA_8 else "A_s",
        ps_norm=t.ps_norm,
        cosmology=cosmo,
        peak_ic_wave_inverse_Mpc=np.sqrt(3) * 8 * 2 * np.pi / length,
        power=pk,
        moments=sigma,
        initial_fields=fields,
        perturbed_fields=moved_fields,
        stages=counted,
    )


def main(output):
    cases = []
    for cfg in (
        ("sigma8_one_thread", {}, 424242, 1, 20.0),
        (
            "primordial_two_threads",
            {"A_s": 2.02e-9},
            7654321,
            2,
            2.0,
        ),
    ):
        cases.append(one(*cfg))
        output.write_text(
            json.dumps({"cases": cases}, indent=2, allow_nan=False) + "\n"
        )
    result = dict(
        passed=all(c["passed"] for c in cases),
        cases=cases,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        py21cmfast=p21.__version__,
        classy=version("classy"),
        reference="Native CPU 4.3.2 with the same cached CLASS transfer",
        scope="IC, sigma and one 2LPT node; no lightcone",
        host="Native cached CLASS density transfer and streaming mean; natural cubic coefficients, EH tail match, and native MtoR integration radii",
        device="Transfer interpolation and extrapolation, streaming suppression, QAG normalization and mass/derivative tables, Gaussian modes, density, 2LPT displacements and perturbation",
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit("CLASS numerical comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
