"""Replay individual constant-efficiency ionization steps using cached native input fields."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import py21cmfast as p21

from compare_lightcone_ps import field_summary
from gpu_lightcone import _backend
from verify_constant_lightcone import specification


def main(mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report = {"job": os.getenv("SLURM_JOB_ID"), "cases": {}}
    for name, nodes, recombination in (
        ("early_neutral", (45.0,), "none"),
        ("cell_recombinations", (9.0, 8.7), "inhomogeneous"),
    ):
        options = specification(False)
        options.update(
            HMF="PS", RECOMB_MODEL=recombination, CELL_RECOMB=True, Z_HEAT_MAX=nodes[0]
        )
        inputs = p21.InputParameters(
            random_seed=531091, node_redshifts=nodes
        ).evolve_input_structs(**options)
        case = directory / name
        cache = p21.OutputCache(case / "cpu-cache")
        compute = dict(write=True, regenerate=False, cache=cache)
        if mode == "cpu":
            ic = p21.compute_initial_conditions(inputs=inputs, **compute)
        else:
            ic = cache.load(p21.InitialConditions.new(inputs=inputs))
        counts = {}
        for step, z in enumerate(nodes):
            if mode == "cpu":
                pf = p21.perturb_field(
                    initial_conditions=ic, redshift=z, inputs=inputs, **compute
                )
            else:
                pf = cache.load(p21.PerturbedField.new(inputs=inputs, redshift=z))
            previous = {}
            if step:
                previous = {
                    "previous_ionized_box": cache.load(
                        p21.IonizedBox.new(inputs=inputs, redshift=nodes[step - 1])
                    ),
                    "previous_perturbed_field": cache.load(
                        p21.PerturbedField.new(inputs=inputs, redshift=nodes[step - 1])
                    ),
                }
            kwargs = dict(
                perturbed_field=pf, initial_conditions=ic, inputs=inputs, **previous
            )
            if mode == "cpu":
                p21.compute_ionization_field(**kwargs, **compute)
                continue
            native = cache.load(p21.IonizedBox.new(inputs=inputs, redshift=z))
            geometry = p21.lightconers.RectilinearLightconer.between_redshifts(
                min_redshift=z,
                max_redshift=z + 0.1,
                resolution=inputs.simulation_options.cell_size,
                cosmo=inputs.cosmo_params.cosmo,
            )
            routing = case / f"routing-{step}"
            with _backend(inputs, geometry, routing):
                gpu = p21.compute_ionization_field(
                    **kwargs, write=False, regenerate=True
                )
            assert native.inputs == gpu.inputs
            fields = {}
            for field, arr in native.arrays.items():
                if arr.state.computed_in_mem or arr.state.on_disk:
                    fields[field] = field_summary(native.get(field), gpu.get(field))
            xh = native.get("neutral_fraction")
            stage_counts = json.loads((routing / "gpu-stages.json").read_text())
            counts[z] = {
                "fields": fields,
                "neutral_fraction_range": [float(xh.min()), float(xh.max())],
                "fully_ionized": int(np.sum(xh == 0)),
                "partial": int(np.sum((xh > 0) & (xh < 1))),
                "nion_unconditional_acg": field_summary(
                    np.array(native.nion_unconditional_acg),
                    np.array(gpu.nion_unconditional_acg),
                ),
                "stages": {k: v for k, v in stage_counts.items() if v},
            }
        if mode == "gpu":
            report["cases"][name] = counts
    if mode == "gpu":
        rows = [r for case in report["cases"].values() for r in case.values()]
        report["passed"] = (
            all(
                m["failed"] == 0
                for r in rows
                for m in [r["nion_unconditional_acg"], *r["fields"].values()]
            )
            and report["cases"]["early_neutral"][45.0]["stages"].get(
                "fully_neutral_box"
            )
            == 1
            and all(
                r["fully_ionized"] and r["partial"] and r["stages"].get("recombination")
                for r in report["cases"]["cell_recombinations"].values()
            )
        )
        (directory / "ionization.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if not report["passed"]:
            raise SystemExit("Constant-efficiency ionization replay failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    main(args.mode, args.directory)
