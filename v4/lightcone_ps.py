"""Measure the production brightness power spectrum on each emitted v4 coeval cube."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from py21cmfast.drivers.lightcone import generate_lightcone

import lightcone_pair


def capture(directory, fine, mode, config):
    target = np.asarray(json.loads(config.read_text())["ps_k"])
    assert target.shape == (32,)
    rows = []
    if mode == "gpu":
        import cupy as xp
    else:
        xp = np

    def run_with_spectra(**kwargs):
        sim = kwargs["inputs"].simulation_options
        n, box = sim.HII_DIM, sim.BOX_LEN

        def shells(module):
            k = module.fft.fftfreq(n, d=box / n) * 2 * module.pi
            return module.sqrt(
                k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
            )

        edges = np.linspace(0, shells(np).max() * 1.0000001, 61)
        centers = (edges[:-1] + edges[1:]) / 2
        assert centers[0] < target[0] < target[-1] < centers[-1]
        bins = xp.clip(xp.digitize(shells(xp).ravel(), xp.asarray(edges)) - 1, 0, 59)
        count = xp.maximum(xp.bincount(bins, minlength=60), 1)
        factor = xp.asarray(centers**3 / (2 * np.pi**2))
        for _, z, coeval, cone in generate_lightcone(**kwargs):
            if coeval is None:
                continue
            field = xp.asarray(coeval.brightness_temp, dtype=xp.float64)
            assert field.shape == (n, n, n)
            fk = xp.fft.fftn(field - field.mean()) / n**3
            power = xp.abs(fk) ** 2 * box**3
            spectrum = (
                xp.bincount(bins, weights=power.ravel(), minlength=60) / count * factor
            )
            ps = xp.interp(xp.asarray(target), xp.asarray(centers), spectrum)
            rows.append(dict(z=float(z), ps=ps.tolist()))
        return cone

    directory.mkdir(parents=True, exist_ok=True)
    with patch.object(lightcone_pair.p21, "run_lightcone", run_with_spectra):
        lightcone_pair.run(directory, fine, mode)
    stats = json.loads((directory / f"{mode}.json").read_text())
    assert stats["nodes"] == [row["z"] for row in rows]
    (directory / f"{mode}-ps.json").write_text(
        json.dumps(
            dict(k=target.tolist(), rows=rows, ic_hash=stats["ic_hash"]), indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("fine", type=int)
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    capture(args.directory, args.fine, args.mode, args.config)
