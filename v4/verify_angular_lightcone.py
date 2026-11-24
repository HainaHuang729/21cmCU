"""Check CUDA angular sampling and a small native 4.3.2 full-sky lightcone."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import py21cmfast as p21
from astropy import units
from py21cmfast.lightconers import AngularLightconer, Lightconer
from scipy.spatial.transform import Rotation
from scipy.special import lpmv

from compare_lightcone_ps import field_summary
from verify_constant_lightcone import cache_files, describe_cache, specification


def sky():
    mu, weight = np.polynomial.legendre.leggauss(8)
    longitude = np.arange(16) * (2 * np.pi / 16)
    return (
        np.repeat(np.arcsin(mu), 16),
        np.tile(longitude, 8),
        np.repeat(weight * 2 * np.pi / 16, 16),
    )


def angular_power(values):
    lat, lon, weight = sky()
    rows = []
    for ell in range(1, 6):
        power = 0.0
        for m in range(ell + 1):
            norm = np.sqrt(
                (2 * ell + 1)
                / (4 * np.pi)
                * math.factorial(ell - m)
                / math.factorial(ell + m)
            )
            conjugate = norm * lpmv(m, ell, np.sin(lat)) * np.exp(-1j * m * lon)
            alm = (weight * conjugate) @ values.astype(np.float64)
            power += abs(alm) ** 2 * (2 if m else 1)
        rows.append(power / (2 * ell + 1))
    return np.asarray(rows)


def compare_arrays(ref, got):
    a, b = np.asarray(ref), np.asarray(got)
    return dict(**field_summary(a, b), different=int(np.count_nonzero(a != b)))


def slice_checks():
    from angular_slices import make_angular_slices

    inputs = p21.InputParameters(random_seed=61)
    cell = inputs.simulation_options.cell_size
    cosmo = inputs.cosmo_params.cosmo
    z1, z2 = 8.0, 9.0
    lo, hi = cosmo.comoving_distance([z1, z2])
    distances = units.Quantity(
        [lo * (1 - 0.5e-6), lo, lo * 0.25 + hi * 0.75, hi - cell / 4, hi]
    )
    lat, lon, _ = sky()
    rng = np.random.default_rng(720)

    def box(z):
        values = {
            k: rng.normal(size=(8, 7, 9)).astype(dtype)
            for k, dtype in [
                ("density", "f"),
                ("velocity_x", "f"),
                ("velocity_y", "f"),
                ("velocity_z", "f"),
                ("z_reion", "f"),
                ("wide", "d"),
            ]
        }
        return SimpleNamespace(
            redshift=z,
            simulation_options=inputs.simulation_options,
            cosmo_params=inputs.cosmo_params,
            **values,
        )

    first, second = box(z1), box(z2)
    rows = []
    for order in (0, 1, 3, 5):
        for tile in (False, True, None) if order < 2 else (None,):
            geometry = AngularLightconer(
                lc_distances=distances,
                latitude=lat,
                longitude=lon,
                origin=np.array([19.2, -16.5, 4.4]) * units.pixel,
                rotation=Rotation.from_euler("xyz", [0.3, 0.7, -0.9]),
                interpolation_order=order,
                tile_first=tile,
                quantities=("density", "z_reion", "wide", "los_velocity"),
                interp_kinds={"z_reion": "mean_max"},
            )
            errors = {}
            for c1, c2 in [(first, second), (second, first), (box(12), box(13))]:
                for (name, i, a), (other, j, b) in zip(
                    Lightconer.make_lightcone_slices(geometry, c1, c2),
                    make_angular_slices(geometry, c1, c2),
                    strict=True,
                ):
                    assert name == other
                    np.testing.assert_array_equal(i, j)
                    if name is None:
                        assert a is b is None
                        continue
                    assert getattr(a, "unit", None) == getattr(b, "unit", None), (
                        name,
                        order,
                        tile,
                        type(a),
                        type(b),
                    )
                    assert a.dtype == b.dtype, (name, a.dtype, b.dtype, order, tile)
                    cur = compare_arrays(a, b)
                    row = errors.setdefault(
                        name, dict(tested=0, failed=0, different=0, max_abs=0.0)
                    )
                    for k in ("tested", "failed", "different"):
                        row[k] += cur[k]
                    row["max_abs"] = max(row["max_abs"], cur["max_abs"])
            rows.append(dict(order=order, tile_first=tile, fields=errors))
    return rows


def run(mode, directory):
    kwargs = specification(True)
    kwargs.update(KEEP_3D_VELOCITIES=True)
    nodes = (9.0, 8.7, 8.4)
    inputs = p21.InputParameters(
        random_seed=812445, node_redshifts=nodes
    ).evolve_input_structs(**kwargs)
    lat, lon, _ = sky()
    cone = AngularLightconer.between_redshifts(
        min_redshift=8.46,
        max_redshift=8.91,
        latitude=lat,
        longitude=lon,
        origin=np.array([-7.25, 2.7, 19.3]) * units.pixel,
        rotation=Rotation.from_euler("zyx", [-0.3, 0.2, 0.4]),
        quantities=(
            "brightness_temp",
            "neutral_fraction",
            "z_reion",
            "kinetic_temperature",
            "density",
        ),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    cpu_file = directory / "cpu.h5"
    if mode != "cpu" or cpu_file.exists():
        ref = p21.LightCone.from_file(cpu_file)
        meta = json.loads((directory / "cpu.json").read_text())
        assert ref.inputs == inputs and meta["requested"] == list(cone.quantities)
        np.testing.assert_array_equal(
            meta["coordinates"], cone.lc_distances.to_value("Mpc")
        )
        np.testing.assert_array_equal(meta["latitude"], lat)
        np.testing.assert_array_equal(meta["longitude"], lon)
        np.testing.assert_array_equal(meta["rotation"], cone.rotation.as_matrix())
        np.testing.assert_array_equal(meta["origin"], cone.origin.to_value(units.pixel))
        if mode == "cpu":
            return
    if mode == "cpu":
        driver = p21.run_lightcone
    else:
        from gpu_lightcone import run_lightcone

        driver = run_lightcone
    replay = mode == "gpu-cached"
    cache_mode = "gpu" if replay else mode
    if replay:
        describe_cache(cache_files(directory / "gpu-cache"), True)
    result = driver(
        inputs=inputs,
        lightconer=cone,
        include_dvdr_in_tau21=True,
        apply_rsds=True,
        n_rsd_subcells=4,
        cache=p21.OutputCache(directory / f"{cache_mode}-cache"),
        regenerate=mode == "gpu",
        write=True,
        **(
            {"diagnostics_dir": directory / f"{mode}-routing", "fallback_to_cpu": False}
            if mode != "cpu"
            else {}
        ),
    )
    assert type(result).__name__ == "AngularLightcone" and result.inputs == inputs
    result.save(directory / f"{mode}.h5", clobber=True)
    record = dict(
        job=os.getenv("SLURM_JOB_ID"),
        version=p21.__version__,
        nodes=nodes,
        seed=inputs.random_seed,
        shape=result.lightcones["brightness_temp"].shape,
        options=kwargs,
        requested=list(cone.quantities),
        coordinates=cone.lc_distances.to_value("Mpc").tolist(),
        latitude=lat.tolist(),
        longitude=lon.tolist(),
        rotation=cone.rotation.as_matrix().tolist(),
        origin=cone.origin.to_value(units.pixel).tolist(),
    )
    if mode != "cpu":
        ref = p21.LightCone.from_file(cpu_file)
        np.testing.assert_array_equal(
            ref.lightcone_distances, result.lightcone_distances
        )
        if replay:
            stages = json.loads(
                (directory / f"{mode}-routing" / "gpu-stages.json").read_text()
            )
            assert all(
                not stages[k]
                for k in (
                    "gpu_seed_and_modes",
                    "heat",
                    "emission",
                    "ion_center",
                    "recombination",
                )
            )
            record["from_saved_coevals"] = True
        record["cpu_job"] = meta["job"]
        for name in ("lightcones", "global_quantities"):
            a, b = getattr(ref, name), getattr(result, name)
            assert set(a) == set(b)
            record[name] = {k: compare_arrays(a[k], b[k]) for k in a}
        record["los_velocity_km_per_second"] = compare_arrays(
            ref.lightcones["los_velocity"].astype("d") * units.Mpc.to(units.km),
            result.lightcones["los_velocity"].astype("d") * units.Mpc.to(units.km),
        )
        record["cuda_slice_counts"] = json.loads(
            (directory / f"{mode}-routing" / "angular-slices.json").read_text()
        )
        record["brightness_angular_power"] = {
            k: compare_arrays(
                angular_power(ref.lightcones[k]), angular_power(result.lightcones[k])
            )
            for k in ("brightness_temp", "brightness_temp_with_rsds")
        }
        record["passed"] = (
            bool(record["cuda_slice_counts"].get("los_velocity"))
            and not record["los_velocity_km_per_second"]["failed"]
        ) and all(
            not r["failed"]
            for k in ("lightcones", "global_quantities", "brightness_angular_power")
            for r in record[k].values()
        )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["slices", "cpu", "gpu", "gpu-cached"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.mode == "slices":
        rows = slice_checks()
        report = dict(
            job=os.environ.get("SLURM_JOB_ID"),
            cases=rows,
            passed=all(not f["failed"] for r in rows for f in r["fields"].values()),
        )
    else:
        report = run(args.mode, args.directory)
    if report:
        (args.directory / f"{args.mode}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if not report.get("passed", True):
            raise SystemExit("Angular comparison failed")
