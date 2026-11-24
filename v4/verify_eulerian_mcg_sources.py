"""Replay one Eulerian molecular emissivity node on archived native IC, density and feedback."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import py21cmfast as p21
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)

import halo_mcg_tables
from evolution_backend import evolution_backend


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite field or changed shape")
    delta = abs(a.astype("d") - b)
    nz = a != 0
    return {
        "values": a.size,
        "unequal": int((a != b).sum()),
        "max_abs": float(delta.max()),
        "max_rel": float((delta[nz] / abs(a[nz])).max(initial=0.0)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run(directory, metal, audit=True):
    def load(name, z=None):
        files = list(directory.rglob(f"{name}.h5"))
        if z is not None:
            files = [p for p in files if f"{z:.4f}" in p.parts]
        assert len(files) == 1, (name, z, files)
        return read_output_struct(files[0])

    z = 11.5
    spin, ion = load("TsBox", 12.0), load("IonizedBox", 12.0)
    expected = load("EmissivityFields", z)
    inputs = expected.inputs
    kw = dict(
        redshift=z,
        initial_conditions=load("InitialConditions"),
        perturbed_field=load("PerturbedField", z),
        previous_spin_temp=spin,
        previous_ionize_box=ion,
        regenerate=True,
        write=False,
    )
    if metal:
        inputs = inputs.evolve_input_structs(USE_METALLICITY=True)
        # Same archived feedback/IC/density, with only the source law varied.
        for key in (
            "initial_conditions",
            "perturbed_field",
            "previous_spin_temp",
            "previous_ionize_box",
        ):
            object.__setattr__(kw[key], "inputs", inputs)
        expected = p21.compute_emissivity_fields(inputs=inputs, **kw)
    manager.free()
    checked = {}
    original = halo_mcg_tables.device_values

    def monitor(resources, p, nx, ny, kind, mini):
        q = ("Nion", "SFRD", "Xray")[kind]
        name = f"{q}_conditional_table2D_{('acg', 'mcg')[mini]}"
        t = halo_mcg_tables.Table2D.from_address(resources[0][name])
        reference = np.stack(
            [np.ctypeslib.as_array(t.values[i], (t.ny,)) for i in range(t.nx)]
        )
        result = original(resources, p, nx, ny, kind, mini)
        checked[name] = compare(reference, result.get())
        return result

    with TemporaryDirectory() as tmp:
        stages = Path(tmp) / "stages.json"
        with (
            (
                patch.object(halo_mcg_tables, "device_values", monitor)
                if audit
                else nullcontext()
            ),
            evolution_backend(stages, thermal=True, molecular=True, audit=audit),
        ):
            got = p21.compute_emissivity_fields(inputs=inputs, **kw)
            manager.free()
        counts = json.loads(stages.read_text())
    fields = {key: compare(expected.get(key), got.get(key)) for key in expected.arrays}
    turns = {
        key: compare(getattr(expected, key), getattr(got, key))
        for key in ("log10_mturn_acg_ave", "log10_mturn_mcg_ave")
    }
    ok = (
        all(row["max_rel"] < 1.0e-5 for row in fields.values())
        and all(row["unequal"] == 0 for row in checked.values())
        and all(row["max_abs"] < 1.0e-12 for row in turns.values())
        and len(checked) == (4 + 2 * metal if audit else 0)
        and fields["sfrd_mcg"]["max"] > 0
        and fields["sfrd_acg"]["max"] > 0
        and counts.get("eulerian_mcg_emissivity_cells")
        == inputs.simulation_options.HII_DIM**3
        and counts.get("eulerian_mcg_global_integrals") == 4 + 2 * metal
    )
    return dict(
        passed=ok,
        metallicity=metal,
        table_audit=audit,
        fields=fields,
        tables=checked,
        turnovers=turns,
        stage_counts=counts,
        source=inputs.matter_options.SOURCE_MODEL,
        redshift=z,
        seed=inputs.random_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--no-table-audit", action="store_true")
    args = parser.parse_args()
    result = run(args.cache, args.metal, not args.no_table_audit)
    result["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit("Eulerian emissivity check failed")
