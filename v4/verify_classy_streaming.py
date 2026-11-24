"""Compare native CLASS P_vcb and the CUDA FLUCTS magnitude, including DC and each axis."""

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

from evolution_backend import evolution_backend, native_tables, view
from verify_classy_power import difference


def one(threads, n, low, seed, *, axes=False):
    inputs = p21.InputParameters(
        random_seed=seed, node_redshifts=[10.0, 9.0]
    ).evolve_input_structs(
        DIM=n,
        HII_DIM=low,
        BOX_LEN=20.0,
        N_THREADS=threads,
        Z_HEAT_MAX=10.0,
        K_MAX_FOR_CLASS=20.0,
        SOURCE_MODEL="CHMF-SAMPLER",
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        USE_MCGS=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
    )
    table = inputs.cosmo_tables
    k = np.asarray(table.transfer_density.x_values)
    probes = np.unique(
        np.r_[
            k,
            (k[1:] + k[:-1]) / 2,
            np.nextafter(k[-1], np.inf),
            1.5 * k[-1],
            3 * k[-1],
            300.0,
            1.0e-8,
        ]
    )
    lib = ctypes.CDLL(p21.c_21cmfast.__file__)
    lib.power_in_vcb.argtypes = [ctypes.c_double]
    lib.power_in_vcb.restype = ctypes.c_double
    patterns = {"gaussian": None}
    if axes:
        const = np.ones((n, n, n), dtype="f")
        patterns["dc"] = const
        for i, label in enumerate("xyz"):
            phase = np.arange(n, dtype="d") * (2 * np.pi / n)
            coord = 1.0 + 0.4 * np.cos(phase) + 0.1 * np.sin(phase)
            shape = [1, 1, 1]
            shape[i] = n
            patterns[label] = const * coord.astype("f").reshape(shape)
    symbols = native_tables()
    kw = dict(inputs=inputs, regenerate=True, write=False)
    manager.free()
    try:
        manager.init(inputs, broadcast_inputs=True, ps=True)
        native_p = np.array([lib.power_in_vcb(x) for x in probes])
        expected = {
            key: p21.compute_initial_conditions(initial_density=grid, **kw)
            for key, grid in patterns.items()
        }
    finally:
        manager.free()
    with TemporaryDirectory() as tmp:
        counts = Path(tmp) / "stages.json"
        with evolution_backend(counts, thermal=False) as engine:
            try:
                manager.init(inputs, broadcast_inputs=True, ps=True)
                actual_p = engine.class_power.power(
                    cp.asarray(probes),
                    view(symbols["cosmo_consts"], 8, "d"),
                    table.ps_norm,
                    np.float32(inputs.cosmo_params.POWER_INDEX),
                    velocity=True,
                ).get()
                compared = {}
                for name, grid in patterns.items():
                    got = p21.compute_initial_conditions(initial_density=grid, **kw)
                    native = expected[name]
                    ref = native.get("lowres_vcb")
                    vel = got.get("lowres_vcb")
                    fields = {
                        key: difference(native.get(key), got.get(key))
                        for key in native.arrays
                    }
                    compared[name] = dict(
                        shape=ref.shape,
                        min=float(ref.min()),
                        max=float(ref.max()),
                        mean_km_s=float(ref.mean(dtype="d")),
                        field=fields.pop("lowres_vcb"),
                        density_and_displacements=fields,
                        stored_field_nonidentical=int(
                            (engine.ic_fields["lowres_vcb"].get() != vel).sum()
                        ),
                    )
            finally:
                manager.free()
        stages = json.loads(counts.read_text())
    bands = {"interpolation": probes <= k[-1], "log_tail": probes > k[-1]}
    power = {
        name: difference(native_p[mask], actual_p[mask]) for name, mask in bands.items()
    }
    ok = (
        native_p[0] == actual_p[0] == 0.0
        and all(m["max_relative_nonzero"] < 1.0e-11 for m in power.values())
        and stages["gpu_seed_and_modes"] == len(patterns)
        and all(
            row["field"]["failed"] == row["stored_field_nonidentical"] == 0
            and all(
                field["failed"] == 0
                for field in row["density_and_displacements"].values()
            )
            and (row["max"] == 0.0 if key == "dc" else row["max"] > row["min"] >= 0.0)
            for key, row in compared.items()
        )
    )
    return dict(
        passed=ok,
        seed=seed,
        dim=n,
        hii_dim=low,
        threads=threads,
        box_len_Mpc=20.0,
        resampling_ratio=n / low,
        top_hat_filter=n != low,
        mean_streaming_speed_km_s=table.V_CB_AVG,
        knots=len(k),
        last_k_inverse_Mpc=k[-1],
        k_max_for_class_inverse_Mpc=20.0,
        speed_unit="km/s",
        power_at_zero=[float(native_p[0]), float(actual_p[0])],
        velocity_power=power,
        fields=compared,
        stages=stages,
    )


def main(path):
    cases = [
        one(1, 16, 8, 424242, axes=True),
        one(2, 16, 16, 7654321),
        one(1, 24, 16, 512324),
    ]
    data = dict(
        passed=all(c["passed"] for c in cases),
        cases=cases,
        version=p21.__version__,
        classy=version("classy"),
        job=os.environ.get("SLURM_JOB_ID"),
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        convention="Each Fourier velocity component uses the same complex density mode with i*k_i/k times sqrt(P_vcb/P_m), times native c in km/s. At unequal grid sizes the high-resolution top hat and inverse FFT precede nearest-cell resampling. The squared components accumulate into float x, y, z; sqrt and division by box volume give lowres_vcb.",
        scope="Seeded or supplied initial fields and the velocity power spectrum; no lightcone",
    )
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if not data["passed"]:
        raise SystemExit("FLUCTS verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
