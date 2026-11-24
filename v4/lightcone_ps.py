"""Measure the production brightness power spectrum on each emitted v4 coeval cube."""

import argparse
import fcntl
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
from py21cmfast.io.h5 import read_inputs
from py21cmfast.drivers.lightcone import generate_lightcone

import lightcone_pair


def k_grid(config):
    target = np.asarray(json.loads(config.read_text())["ps_k"], dtype="d")
    if target.shape != (32,) or not np.isfinite(target).all():
        raise ValueError(f"{config} needs 32 finite ps_k values")
    return target


def capture(directory, fine, mode, config, *, audit_evolution=False):
    """Run fresh physics and observe every coeval, including for programmatic audits."""
    target = k_grid(config)
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

        radii = shells(xp)
        edges = xp.linspace(0, radii.max() * 1.0000001, 61)
        centers = (edges[:-1] + edges[1:]) / 2
        assert centers[0] < target[0] < target[-1] < centers[-1]
        bins = xp.clip(xp.digitize(radii.ravel(), edges) - 1, 0, 59)
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
        lightcone_pair.run(directory, fine, mode, audit_evolution=audit_evolution)
    stats = json.loads((directory / f"{mode}.json").read_text())
    assert stats["nodes"] == [row["z"] for row in rows]
    (directory / f"{mode}-ps.json").write_text(
        json.dumps(
            dict(k=target.tolist(), rows=rows, ic_hash=stats["ic_hash"]), indent=2
        )
    )


CPU_PRODUCTS = ("cpu.h5", "cpu.json", "cpu-ps.json")


def validate_reference(directory, fine, target):
    """Check identity and completeness using metadata and the small saved spectra."""
    h5_path, stats_path, ps_path = (directory / name for name in CPU_PRODUCTS)
    stats = json.loads(stats_path.read_text())
    spectra = json.loads(ps_path.read_text())
    inputs, lightconer = lightcone_pair.benchmark_case(fine)
    nodes = list(inputs.node_redshifts)
    digest = stats["ic_hash"]
    if (
        not isinstance(digest, str)
        or not re.fullmatch("[0-9a-f]{64}", digest)
        or digest != spectra["ic_hash"]
    ):
        raise ValueError("Mismatched initial-condition SHA-256 in the CPU records")
    if stats["nodes"] != nodes:
        raise ValueError("Saved CPU redshift nodes differ from the current benchmark")
    settings = {
        "fine": fine,
        "seed": inputs.random_seed,
        "calls": len(nodes),
        "file_bytes": h5_path.stat().st_size,
    }
    for key, expected in settings.items():
        if stats[key] != expected:
            raise ValueError(f"Saved CPU {key}={stats[key]} (expected {expected})")
    # Older native recordings precede the GPU brightness timing counters.
    if (
        stats.get("gpu_brightness_calls", 0) != 0
        or stats.get("gpu_brightness_seconds", 0) != 0
    ):
        raise ValueError("The CPU reference reports accelerated brightness calls")
    for field in ("elapsed_seconds", "perturb_seconds", "hash_seconds"):
        if not np.isfinite(stats[field]) or stats[field] < 0:
            raise ValueError(f"Invalid saved CPU timing for {field}")
    if not np.array_equal(spectra["k"], target):
        raise ValueError("Saved CPU wavenumbers differ from the requested ps_k")
    rows = spectra["rows"]
    if [r["z"] for r in rows] != nodes:
        raise ValueError("Saved CPU spectrum redshifts differ from the requested nodes")
    power = np.asarray([r["ps"] for r in rows], dtype="d")
    if power.shape != (len(nodes), 32) or not np.isfinite(power).all():
        raise ValueError("Incomplete or nonfinite native CPU power spectra")

    p21 = lightcone_pair.p21
    with h5py.File(h5_path, "r") as saved:
        if (
            not saved.attrs.get("lightcone")
            or saved.attrs.get("__version__") != p21.__version__
            or saved["InputParameters"].attrs.get("21cmFAST-version") != p21.__version__
        ):
            raise ValueError("Native LightCone file type or 21cmFAST version differs")
        # Archives made before the HDF5 digest attribute still have the IC hash in both JSONs.
        if saved.attrs.get("benchmark_ic_sha256", digest) != digest:
            raise ValueError("Initial-condition hash in the LightCone differs")
        if read_inputs(saved) != inputs:
            raise ValueError("Saved native InputParameters differ from the benchmark")
        if not np.array_equal(
            saved["lightcone_distances"][:], lightconer.lc_distances.to_value("Mpc")
        ):
            raise ValueError("Saved native comoving distances differ")
        validated = lightconer.validate_options(
            inputs, include_dvdr_in_tau21=True, apply_rsds=False
        )
        groups = (
            (
                "lightcones",
                set(validated.quantities),
                lightconer.get_shape(inputs.simulation_options),
            ),
            ("global_quantities", set(p21.LightCone.get_fields(inputs)), (len(nodes),)),
        )
        for group_name, expected, shape in groups:
            group = saved[group_name]
            if (
                not isinstance(group, h5py.Group)
                or set(group) != expected
                or any(
                    not isinstance(value, h5py.Dataset)
                    or value.shape != shape
                    or value.dtype.kind != "f"
                    for value in group.values()
                )
            ):
                raise ValueError(
                    f"Saved {group_name} fields or shapes differ from the native benchmark"
                )


def cpu_cli(directory, fine, config, rerun=False):
    target = k_grid(config)
    directory.mkdir(parents=True, exist_ok=True)
    # Serialize native reference creation. Multiple queued acceptances may share the same directory.
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not rerun and any(
            os.path.lexists(directory / name) for name in CPU_PRODUCTS
        ):
            try:
                validate_reference(directory, fine, target)
            except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                raise ValueError(
                    f"{directory}: invalid CPU reference ({exc}). "
                    "Use a matching reference or --rerun-cpu to replace it."
                ) from exc
            print(f"Reusing native CPU reference in {directory}")
            return
        # Complete and validate the new run before replacing a previously usable reference.
        with TemporaryDirectory(dir=directory, prefix=".cpu-reference-") as temp:
            stage = Path(temp)
            capture(stage, fine, "cpu", config)
            validate_reference(stage, fine, target)
            for name in CPU_PRODUCTS:
                (stage / name).replace(directory / name)
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("fine", type=int)
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--rerun-cpu",
        action="store_true",
        help="Recompute and replace the native CPU reference for retiming or repair",
    )
    args = parser.parse_args()
    if args.mode == "cpu":
        cpu_cli(args.directory, args.fine, args.config, args.rerun_cpu)
    elif args.rerun_cpu:
        parser.error("--rerun-cpu requires cpu mode")
    else:
        capture(args.directory, args.fine, "gpu", args.config)
