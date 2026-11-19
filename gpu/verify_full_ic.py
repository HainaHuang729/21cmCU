"""Compare complete files from the matched CPU and GPU IC runs."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np

RTOL = 1e-4
ATOL = 1e-8


def compare(reference, candidate):
    assert reference.shape == candidate.shape and reference.dtype == candidate.dtype
    count = failed = nonzero_ref = nonzero_gpu = 0
    max_abs = sum_square = 0.0
    for begin in range(0, reference.shape[0], 8):
        cpu = reference[begin:begin + 8].astype(np.float64)
        gpu = candidate[begin:begin + 8].astype(np.float64)
        assert np.isfinite(cpu).all() and np.isfinite(gpu).all()
        delta = abs(cpu - gpu)
        count += delta.size
        failed += np.count_nonzero(delta > ATOL + RTOL * abs(cpu))
        nonzero_ref += np.count_nonzero(cpu)
        nonzero_gpu += np.count_nonzero(gpu)
        max_abs = max(max_abs, float(delta.max()))
        sum_square += float(np.sum(delta**2))
    return {
        "count": count, "failed": failed, "max_abs": max_abs,
        "rms": (sum_square / count)**.5,
        "nonzero_cpu": nonzero_ref, "nonzero_gpu": nonzero_gpu,
    }


cpu, gpu = map(Path, sys.argv[1:])
summary = {"rtol": RTOL, "atol": ATOL}
with h5py.File(cpu / "initial-conditions.h5") as ref, h5py.File(gpu / "initial-conditions.h5") as got:
    a, b = ref["InitialConditions"], got["InitialConditions"]
    assert set(a) == set(b) and len(a) == 14
    summary["initial_conditions"] = {name: compare(a[name], b[name]) for name in a}
    for axis in "xyz":
        stats = summary["initial_conditions"][f"hires_v{axis}"]
        assert stats["nonzero_cpu"] == stats["nonzero_gpu"] == 0
        stats = summary["initial_conditions"][f"hires_v{axis}_2LPT"]
        assert stats["nonzero_cpu"] and stats["nonzero_gpu"]
with h5py.File(cpu / "coeval-fields.h5") as ref, h5py.File(gpu / "coeval-fields.h5") as got:
    np.testing.assert_array_equal(ref["redshifts"], got["redshifts"])
    assert len(ref["redshifts"]) == 32
    totals = {}
    for index in range(32):
        a, b = ref[f"z{index:02d}"], got[f"z{index:02d}"]
        assert set(a) == set(b) and len(a) == 13
        for name in a:
            stats = compare(a[name], b[name])
            total = totals.setdefault(name, {"count": 0, "failed": 0, "max_abs": 0.0})
            total["count"] += stats["count"]
            total["failed"] += stats["failed"]
            total["max_abs"] = max(total["max_abs"], stats["max_abs"])
    summary["coeval"] = totals
print(json.dumps(summary, indent=2))
