"""Replay the inputs to the first divergent recombination update at a lightcone site."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import py21cmfast as p21
from tests import produce_integration_test_data as official
from py21cmfast.drivers import single_field
from py21cmfast.lightconers import RectilinearLightconer
from lightcone_pair import FIELDS


def run(directory, mode):
    parent = json.loads((directory / "comparison.json").read_text())
    x, y, idx = parent["fields"]["cumulative_recombinations"]["xyz"]
    with patch.object(official, "SEED", parent["cpu"]["seed"]):
        inputs = official.get_all_options_struct(
            6,
            lc=True,
            DIM=384,
            HII_DIM=96,
            BOX_LEN=375,
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )["inputs"]
    cell = inputs.simulation_options.cell_size
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=6, max_redshift=15, quantities=FIELDS, resolution=cell
    )
    distance = lc.get_lc_distances_in_pixels(cell)[idx]

    def gpu(**kwargs):
        from perturb import perturb_field_gpu

        return perturb_field_gpu(
            redshift=kwargs["redshift"], initial_conditions=kwargs["initial_conditions"]
        )

    rows = []
    with patch.object(
        single_field,
        "perturb_field",
        gpu if mode == "gpu" else single_field.perturb_field,
    ):
        for _, z, coeval, _ in p21.generate_lightcone(
            inputs=inputs, lightconer=lc, write=False, regenerate=True
        ):
            row = {"z": z}
            for field in (
                "density",
                "neutral_fraction",
                "ionisation_rate_G12",
                "z_reion",
                "cumulative_recombinations",
                "xray_ionised_fraction",
            ):
                row[field] = float(
                    lc.coeval_subselect(distance, getattr(coeval, field), cell)[x, y]
                )
            if "mean_free_path" in coeval.ionized_box.arrays:
                row["mean_free_path"] = float(
                    lc.coeval_subselect(
                        distance, coeval.ionized_box.get("mean_free_path"), cell
                    )[x, y]
                )
            rows.append(row)
            if z <= 7.19:
                break
    (directory / f"recomb-inputs-{mode}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
