"""Replay Eulerian emissivity normalization using the native CONST lightcone cache."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)

from evolution_backend import evolution_backend
from verify_constant_lightcone import NODES, specification


def fields(box):
    return {
        k: box.get(k).copy()
        for k, v in box.arrays.items()
        if v.state.computed_in_mem or v.state.on_disk
    }


def compare(native, actual):
    assert actual.shape == native.shape
    delta = abs(native.astype("d") - actual.astype("d"))
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite halo emissivity")
    return {
        "count": native.size,
        "different": int(np.count_nonzero(delta)),
        "max_abs": float(delta.max()),
        "failed_strict": int(
            np.count_nonzero(delta > 1e-8 + 1e-4 * abs(native.astype("d")))
        ),
    }


def replay(directory, output):
    result = {
        "job": os.getenv("SLURM_JOB_ID"),
        "seed": 812445,
        "redshift": 8.7,
        "cases": {},
    }
    for name, model, thermal, edge, audit in (
        ("constant-thermal", "CONST-ION-EFF", True, False, False),
        ("constant-cold", "CONST-ION-EFF", False, False, False),
        ("e-integral", "E-INTEGRAL", True, False, False),
        ("constant-density-floor", "CONST-ION-EFF", True, True, False),
        ("e-integral-audit", "E-INTEGRAL", True, False, True),
    ):
        base = p21.InputParameters(
            random_seed=result["seed"], node_redshifts=NODES
        ).evolve_input_structs(**specification(thermal))
        inputs = base.evolve_input_structs(
            SOURCE_MODEL=model, N_THREADS=2 if model == "E-INTEGRAL" else 1
        )
        cache = p21.OutputCache(
            directory / ("thermal" if thermal else "nonthermal") / "cpu-cache"
        )
        box = dict(inputs=base, redshift=result["redshift"])
        density = cache.load(p21.PerturbedField.new(**box)).get("density").copy()
        if edge:
            density.ravel()[:4] = [
                -1.0,
                np.nextafter(np.float32(-1.0), np.float32(0.0)),
                np.float32(-1.0) + np.float32(2**-23),
                -0.95,
            ]
        pf = p21.PerturbedField.new(inputs=inputs, redshift=result["redshift"])
        pf.set("density", density)
        kwargs = dict(
            inputs=inputs,
            redshift=result["redshift"],
            perturbed_field=pf,
            regenerate=True,
            write=False,
        )
        if model == "CONST-ION-EFF" and not edge:
            native = fields(cache.load(p21.EmissivityFields.new(**box)))
            origin = "saved native EmissivityFields"
        else:
            native = fields(p21.compute_emissivity_fields(**kwargs))
            origin = "native compute_emissivity_fields on saved density"
        manager.free()
        with TemporaryDirectory() as scratch:
            counts = Path(scratch) / "stages.json"
            with evolution_backend(counts, audit=audit):
                actual = fields(p21.compute_emissivity_fields(**kwargs))
            counters = json.loads(counts.read_text())
        result["cases"][name] = {
            "source": model,
            "baseline": origin,
            "density_min": float(density.min()),
            "threads": inputs.simulation_options.N_THREADS,
            "stages": {
                k: counters[k]
                for k in (
                    "emission",
                    "emissivity_mean_normalization",
                    "emissivity_global_source_integrals",
                )
            },
            "fields": {k: compare(v, actual[k]) for k, v in native.items()},
        }
    result["passed"] = all(
        c["stages"]["emissivity_mean_normalization"] == (name != "e-integral-audit")
        and all(m["failed_strict"] == 0 for m in c["fields"].values())
        for name, c in result["cases"].items()
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit("Eulerian emissivity normalization replay failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_directory",
        type=Path,
        help="root containing the thermal and nonthermal cpu-cache directories",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay(args.cache_directory, args.output)
