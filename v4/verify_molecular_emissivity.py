"""Replay one molecular emissivity node with the previous native fields from a saved run."""

import argparse
import json
import os
from time import perf_counter
from unittest.mock import patch
from pathlib import Path

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers.single_field import compute_emissivity_fields
from functools import partial
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from verify_chmf_lightcone import read_native_field
from py21cmfast.wrapper import inputs as input_module

from halo_grid import halo_grid_backend


@contextmanager
def source_backend(inputs, counts):
    if inputs.matter_options.lagrangian_source_grid:
        with halo_grid_backend(counts):
            yield
    else:
        from evolution_backend import evolution_backend

        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "stages.json"
            with evolution_backend(report, molecular=True):
                yield
            counts.update(json.loads(report.read_text()))


def replay(directory, redshift):
    read = partial(read_native_field, directory)

    inputs = p21.LightCone.from_file(directory / "cpu.h5").inputs
    if not inputs.astro_options.USE_MCGS:
        raise ValueError("The CPU reference must enable molecular cooling")
    reference = read("EmissivityFields", redshift)
    previous = [z for z in inputs.node_redshifts if z > redshift]
    args = {}
    if previous:
        args.update(
            previous_spin_temp=read("TsBox", min(previous)),
            previous_ionize_box=read("IonizedBox", min(previous)),
        )
    if inputs.matter_options.has_discrete_halos:
        args["halo_catalog"] = read("HaloCatalog", redshift)
    if not inputs.matter_options.lagrangian_source_grid:
        args["perturbed_field"] = read("PerturbedField", redshift)
    counters = {}
    class_calls = []
    solve = input_module.run_classy

    def observe(**kwargs):
        class_calls.append(str(kwargs.get("output")))
        return solve(**kwargs)

    start = perf_counter()
    with (
        patch.object(input_module, "run_classy", observe),
        source_backend(inputs, counters),
    ):
        result = compute_emissivity_fields(
            redshift=redshift,
            inputs=inputs,
            initial_conditions=read("InitialConditions"),
            write=False,
            regenerate=True,
            **args,
        )
    elapsed = perf_counter() - start
    fields = {}
    for name, array in reference.arrays.items():
        if not array.state.on_disk:
            continue
        expected = np.asarray(reference.get(name), dtype="d")
        got = np.asarray(result.get(name), dtype="d")
        if (
            expected.shape != got.shape
            or not np.isfinite(expected).all()
            or not np.isfinite(got).all()
        ):
            raise ValueError(f"Invalid {name} grid")
        diff = np.abs(got - expected)
        peak = float(np.abs(expected).max())
        fields[name] = dict(
            n=diff.size,
            max_abs=float(diff.max()),
            peak=peak,
            different=int(np.count_nonzero(diff)),
            failed=int((diff > 1e-5 * np.abs(expected) + 1e-10 * peak).sum()),
        )
    passed = (
        fields["sfrd_mcg"]["peak"] > 0
        and (
            counters.get("subgrid_molecular_advection_cells", 0)
            or counters.get("eulerian_mcg_emissivity_cells", 0)
        )
        > 0
        and (
            inputs.matter_options.SOURCE_MODEL != "CHMF-SAMPLER"
            or counters.get("sampled_mcg_halos", 0) > 0
        )
        and all(field["failed"] == 0 for field in fields.values())
    )
    return dict(
        source=inputs.matter_options.SOURCE_MODEL,
        redshift=redshift,
        cpu_job=json.loads((directory / "cpu.json").read_text())["job"],
        gpu_job=os.getenv("SLURM_JOB_ID"),
        fields=fields,
        counters=counters,
        elapsed_seconds=elapsed,
        class_solver_calls=class_calls,
        tolerance="1e-5 * abs(cpu) + 1e-10 * max(abs(cpu)) per field",
        passed=passed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("redshift", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    row = replay(args.directory, args.redshift)
    args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if row["passed"] else 1)
