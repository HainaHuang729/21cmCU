"""Replay native and CUDA EH normalization and mass-variance setup on identical inputs."""

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

from evolution_backend import Table, lagrangian_backend, native_tables, view


def snapshot(inputs, symbols):
    manager.init(inputs, broadcast_inputs=True, ps=True, sigma=True)
    arrays = {"eh_constants": view(symbols["cosmo_consts"], 8, "d").copy()}
    for name in ("Sigma_InterpTable", "dSigmasqdm_InterpTable"):
        table = Table.from_address(symbols[name])
        if not table.allocated or table.n < 2:
            raise ValueError(f"Missing mass-variance table {name}")
        arrays[name] = np.ctypeslib.as_array(table.y, (table.n,)).copy()
        arrays[name + "_axis"] = np.array([table.low, table.width])
    return arrays


def main(output):
    symbols = native_tables()
    data = {"slurm_job": os.environ.get("SLURM_JOB_ID"), "cases": {}}
    for case, cosmology in (
        ("default", {}),
        ("sigma8", {"SIGMA_8": 0.76, "POWER_INDEX": 0.99}),
        ("primordial_amplitude", {"A_s": 2.02e-9, "POWER_INDEX": 0.94}),
    ):
        inputs = p21.InputParameters(
            random_seed=73625, cosmo_params=p21.CosmoParams(**cosmology)
        ).evolve_input_structs(DIM=32, HII_DIM=16, BOX_LEN=24.0, N_THREADS=1)
        manager.free()
        try:
            cpu = snapshot(inputs, symbols)
        finally:
            manager.free()
        with TemporaryDirectory() as directory:
            counters = Path(directory) / "stages.json"
            with lagrangian_backend(counters, inputs=inputs):
                try:
                    gpu = snapshot(inputs, symbols)
                finally:
                    manager.free()
            stages = json.loads(counters.read_text())
        fields = {}
        for name, ref in cpu.items():
            actual = gpu[name]
            assert ref.shape == actual.shape
            delta = np.abs(actual.astype(float) - ref)
            fields[name] = {
                "values": ref.size,
                "unequal": int((actual != ref).sum()),
                "max_abs": float(delta.max()),
                "failed_atol1e-8_rtol1e-4": int(
                    (~(delta <= 1e-8 + 1e-4 * abs(ref.astype(float)))).sum()
                ),
            }
        data["cases"][case] = {
            "cosmology": cosmology,
            "fields": fields,
            "counters": stages,
        }
    data["passed"] = all(
        case["counters"].get("gpu_eh_transfer") == 1
        and case["counters"].get("gpu_sigma_mass_table") == 1
        and all(
            row["failed_atol1e-8_rtol1e-4"] == 0
            and (name == "eh_constants" or row["unequal"] == 0)
            for name, row in case["fields"].items()
        )
        for case in data["cases"].values()
    )
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if not data["passed"]:
        raise SystemExit("Matter power setup differed from the native CPU reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
