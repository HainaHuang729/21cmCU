"""Follow the worst failed voxel of each failing lightcone field across coeval nodes."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import py21cmfast as p21
from astropy.units import Mpc
import lightcone_pair


def run(directory, fine):
    comparison = json.loads((directory / "comparison.json").read_text())
    targets = {k: v["xyz"] for k, v in comparison["fields"].items() if v["failed"]}
    output = {}
    for mode in ("cpu", "gpu"):
        histories = {key: [] for key in targets}

        def replay(**kwargs):
            lc = kwargs["lightconer"]
            cell = kwargs["inputs"].simulation_options.cell_size
            pixels = lc.get_lc_distances_in_pixels(cell)
            for _, z, coeval, cone in p21.generate_lightcone(**kwargs):
                for key, (x, y, slice_index) in targets.items():
                    d = pixels[slice_index]
                    record = dict(
                        z=z,
                        node_mpc=float(
                            coeval.cosmo_params.cosmo.comoving_distance(z).to_value(Mpc)
                        ),
                        slice_mpc=float(lc.lc_distances[slice_index].to_value(Mpc)),
                    )
                    for field in ("density", key):
                        record[field] = float(
                            lc.coeval_subselect(d, getattr(coeval, field), cell)[x, y]
                        )
                    histories[key].append(record)
            return cone

        # Benchmark outputs are immutable; store replay products in an isolated directory.
        replay_dir = directory / "site-replay"
        replay_dir.mkdir(exist_ok=True)
        with patch.object(p21, "run_lightcone", replay):
            lightcone_pair.run(replay_dir, fine, mode)
        output[mode] = histories
    (directory / "site-history.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]))
