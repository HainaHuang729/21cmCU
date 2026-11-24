"""Replay a molecular Eulerian ionization step using saved native current and previous fields."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers.single_field import compute_ionization_field

from evolution_backend import evolution_backend
from verify_chmf_lightcone import read_native_field


def replay(directory, z):
    inputs = p21.LightCone.from_file(directory / "cpu.h5").inputs
    if (
        inputs.matter_options.SOURCE_MODEL != "E-INTEGRAL"
        or not inputs.astro_options.USE_MCGS
    ):
        raise ValueError("Expected an Eulerian molecular lightcone")
    read = partial(read_native_field, directory)
    previous = min(node for node in inputs.node_redshifts if node > z)
    reference = read("IonizedBox", z)
    with TemporaryDirectory() as tmp:
        report = Path(tmp) / "stages.json"
        with evolution_backend(report, molecular=True):
            result = compute_ionization_field(
                inputs=inputs,
                initial_conditions=read("InitialConditions"),
                perturbed_field=read("PerturbedField", z),
                previous_perturbed_field=read("PerturbedField", previous),
                previous_ionized_box=read("IonizedBox", previous),
                spin_temp=read("TsBox", z),
                emissivity_fields=read("EmissivityFields", z),
                regenerate=True,
                write=False,
            )
        counts = json.loads(report.read_text())
    values = []
    for name, arr in reference.arrays.items():
        if arr.state.on_disk:
            values.append((name, reference.get(name), result.get(name)))
    for name in (
        "log10_mturn_ave_acg",
        "log10_mturn_ave_mcg",
        "nion_unconditional_acg",
        "nion_unconditional_mcg",
    ):
        values.append((name, getattr(reference, name), getattr(result, name)))
    fields = {}
    for name, expected, got in values:
        a, b = np.asarray(expected, "d"), np.asarray(got, "d")
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid {name} field")
        diff = np.abs(a - b)
        fields[name] = dict(
            n=a.size,
            shape=a.shape,
            max_abs=float(diff.max()),
            different=int(np.count_nonzero(diff)),
            failed=int(np.count_nonzero(diff > 1e-8 + 1e-4 * np.abs(a))),
            peak=float(np.max(np.abs(a))),
        )
    passed = (
        all(f["failed"] == 0 for f in fields.values())
        and counts.get("ion_mcg_radius_previous_correction", 0) >= 2
    )
    return dict(
        passed=passed,
        gpu_job=os.getenv("SLURM_JOB_ID"),
        cpu_job=json.loads((directory / "cpu.json").read_text())["job"],
        z=z,
        previous=previous,
        fields=fields,
        counters=counts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("z", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = replay(args.directory, args.z)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if record["passed"] else 1)
