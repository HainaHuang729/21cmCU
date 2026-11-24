"""Compare lightcone slices and the production estimator on matching coeval nodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def metrics(a, b):
    delta = np.abs(b - a)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite comparison")
    return delta, delta > 1e-8 + 1e-4 * np.abs(a)


def field_summary(cpu, gpu):
    if cpu.shape != gpu.shape:
        raise ValueError(f"CPU/GPU field shape mismatch: {cpu.shape} != {gpu.shape}")
    delta, bad = metrics(cpu.astype("d"), gpu.astype("d"))
    return {
        "tested": int(delta.size),
        "failed": int(bad.sum()),
        "max_abs": float(delta.max()),
    }


def compare_saved_slices(cpu_path, gpu_path):
    """Check saved quantities and equal-time transverse power on each rectilinear slice."""

    with h5py.File(cpu_path) as cpu, h5py.File(gpu_path) as gpu:
        distances = cpu["lightcone_distances"][:]
        np.testing.assert_array_equal(distances, gpu["lightcone_distances"][:])
        np.testing.assert_array_equal(
            cpu["InputParameters/node_redshifts"][:],
            gpu["InputParameters/node_redshifts"][:],
        )
        for key in ("random_seed", "21cmFAST-version"):
            assert (
                cpu["InputParameters"].attrs[key] == gpu["InputParameters"].attrs[key]
            )
        products = {}
        for group in ("lightcones", "global_quantities"):
            assert set(cpu[group]) == set(gpu[group])
            products[group] = {
                key: field_summary(cpu[group][key][:], gpu[group][key][:])
                for key in cpu[group]
            }
        shape = cpu["lightcones/brightness_temp"].shape
        if shape != gpu["lightcones/brightness_temp"].shape or shape[0] != shape[1]:
            raise ValueError(
                "Transverse angular averaging requires matching square slices"
            )
        side = shape[0]
        # The transverse comoving box length comes from the simulation. LOS sampling
        # can differ from the transverse pixel size for other lightconers.
        box = float(cpu["InputParameters/simulation_options"].attrs["BOX_LEN"])
        assert box == gpu["InputParameters/simulation_options"].attrs["BOX_LEN"]
        modes = np.fft.fftfreq(side) * side
        radius = np.hypot(modes[:, None], modes[None, :])
        shells = [
            (radius >= i - 0.5) & (radius < i + 0.5) & (radius <= side / 2)
            for i in range(1, side // 2 + 1)
        ]
        powers = []
        for f in (cpu, gpu):
            fields = f["lightcones/brightness_temp"][:].astype("d")
            fields -= fields.mean(axis=(0, 1), keepdims=True)
            power = (box / (side * side)) ** 2 * abs(
                np.fft.fft2(fields, axes=(0, 1))
            ) ** 2
            powers.append(np.stack([power[mask].mean(axis=0) for mask in shells]))
        products["brightness_transverse_power"] = {
            "unit": "mK^2 Mpc^2",
            "k_unit": "Mpc^-1",
            "k_mean": [float(radius[mask].mean() * 2 * np.pi / box) for mask in shells],
            "modes_per_bin": [int(mask.sum()) for mask in shells],
            "slices": int(distances.size),
            **field_summary(*powers),
        }
    return products


def compare(cone, cpu_ps, gpu_ps):
    from lightcone_pair import FIELDS

    report = json.loads((cone / "comparison.json").read_text())
    samples = [
        json.loads((path / f"{mode}-ps.json").read_text())
        for mode, path in (("cpu", cpu_ps), ("gpu", gpu_ps))
    ]
    a, b = samples
    assert a["k"] == b["k"] and len(a["k"]) == 32
    assert (
        a["ic_hash"]
        == b["ic_hash"]
        == report["cpu"]["ic_hash"]
        == report["gpu"]["ic_hash"]
    )
    assert (
        [r["z"] for r in a["rows"]]
        == [r["z"] for r in b["rows"]]
        == report["cpu"]["nodes"]
        == report["gpu"]["nodes"]
    )
    x, y = [np.asarray([r["ps"] for r in s["rows"]]) for s in samples]
    d, bad = metrics(x, y)
    np.testing.assert_equal(x.shape, (len(report["cpu"]["nodes"]), 32))
    report["ps32"] = dict(
        k=a["k"],
        redshifts=report["cpu"]["nodes"],
        cpu=x.tolist(),
        gpu=y.tolist(),
        tested=int(d.size),
        failed=int(bad.sum()),
        max_abs=float(d.max()),
        max_relative_nonzero=float((d[x != 0] / abs(x[x != 0])).max()),
        failed_per_node=bad.sum(axis=1).tolist(),
    )
    with (
        h5py.File(cone / "cpu.h5") as ref,
        h5py.File(cone / "gpu.h5") as got,
        h5py.File(cpu_ps / "cpu.h5") as ref_ps,
        h5py.File(gpu_ps / "gpu.h5") as got_ps,
    ):
        for file in (got, ref_ps, got_ps):
            np.testing.assert_array_equal(
                ref["lightcone_distances"][:], file["lightcone_distances"][:]
            )
        for group, result_key in (
            ("lightcones", "auxiliary_fields"),
            ("global_quantities", "global_quantities"),
        ):
            names = set(ref[group])
            assert names == set(got[group]), f"Different fields in {group}"
            if group == "lightcones":
                names -= set(FIELDS)
            report[result_key] = {}
            for name in sorted(names):
                x, y = (f[group][name][:].astype(float) for f in (ref, got))
                np.testing.assert_equal(x.shape, y.shape)
                d, bad = metrics(x, y)
                report[result_key][name] = dict(
                    tested=int(d.size), failed=int(bad.sum()), max_abs=float(d.max())
                )
        for field in FIELDS:
            key = f"lightcones/{field}"
            x, y = ref[key][:], got[key][:]
            repeat = {}
            for mode, original, file in (("cpu", x, ref_ps), ("gpu", y, got_ps)):
                delta, failed = metrics(
                    original.astype(float), file[key][:].astype(float)
                )
                repeat[mode] = dict(
                    max_abs=float(delta.max()),
                    different=int(np.count_nonzero(delta)),
                    failed=int(failed.sum()),
                )
            d, bad = metrics(x.astype(float), y.astype(float))
            f = report["fields"][field]
            assert bad.sum() == f["failed"]
            f.update(
                max_abs_per_slice=d.max(axis=(0, 1)).tolist(),
                failed_per_slice=bad.sum(axis=(0, 1)).tolist(),
                ps_run_repeat=repeat,
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("cone", "cpu_ps", "gpu_ps", "output"):
        parser.add_argument(arg, type=Path)
    opts = parser.parse_args()
    report = compare(opts.cone, opts.cpu_ps, opts.gpu_ps)
    opts.output.write_text(json.dumps(report, indent=2) + "\n")
    failures = sum(
        row["failed"]
        for group in ("fields", "auxiliary_fields")
        for row in report[group].values()
    )
    global_failures = sum(row["failed"] for row in report["global_quantities"].values())
    if failures or global_failures or report["ps32"]["failed"]:
        raise SystemExit(
            f"{failures} lightcone cells, {global_failures} global values and "
            f"{report['ps32']['failed']} PS bins exceed tolerance"
        )
