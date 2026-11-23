"""Compare lightcone slices and the production estimator on matching coeval nodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from lightcone_pair import FIELDS


def metrics(a, b):
    delta = np.abs(b - a)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite comparison")
    return delta, delta > 1e-8 + 1e-4 * np.abs(a)


def compare(cone, cpu_ps, gpu_ps):
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
    with h5py.File(cone / "cpu.h5") as ref, h5py.File(
        cone / "gpu.h5"
    ) as got, h5py.File(cpu_ps / "cpu.h5") as ref_ps, h5py.File(
        gpu_ps / "gpu.h5"
    ) as got_ps:
        for file in (got, ref_ps, got_ps):
            np.testing.assert_array_equal(
                ref["lightcone_distances"][:], file["lightcone_distances"][:]
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
    opts.output.write_text(
        json.dumps(compare(opts.cone, opts.cpu_ps, opts.gpu_ps), indent=2) + "\n"
    )
