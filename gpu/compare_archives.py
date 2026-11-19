"""Verify optimized archives against the late-write GPU baseline."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np


def check_attrs(ref, opt):
    assert set(ref.attrs) == set(opt.attrs)
    for key in ref.attrs:
        np.testing.assert_equal(ref.attrs[key], opt.attrs[key], err_msg=f"{ref.name}:{key}")


def compare_h5(expected, actual):
    counts = {"datasets": 0, "values": 0, "failed": 0, "max_abs": 0.0}
    with h5py.File(expected) as ref, h5py.File(actual) as opt:
        check_attrs(ref, opt)
        nodes = []
        ref.visit(nodes.append)
        other = []
        opt.visit(other.append)
        assert nodes == other
        for name in nodes:
            a, b = ref[name], opt[name]
            check_attrs(a, b)
            if not isinstance(a, h5py.Dataset):
                continue
            assert a.shape == b.shape and a.dtype == b.dtype
            counts["datasets"] += 1
            counts["values"] += a.size
            for begin in range(0, a.shape[0], 8):
                left, right = a[begin:begin + 8], b[begin:begin + 8]
                if np.array_equal(left, right):
                    continue
                delta = abs(left.astype(np.float64) - right)
                counts["failed"] += int(np.count_nonzero(delta > 1e-8 + 1e-4 * abs(left)))
                counts["max_abs"] = max(counts["max_abs"], float(delta.max()))
    return counts


ref, opt = map(Path, sys.argv[1:])
names = ("initial-conditions.h5", "coeval-fields.h5")
checks = {name: compare_h5(ref / name, opt / name) for name in names}
assert checks["initial-conditions.h5"]["datasets"] == 14
assert checks["coeval-fields.h5"]["datasets"] == 417  # 416 fields and the redshift axis.
with np.load(ref/"gpu-timing-outputs.npz") as a, np.load(opt/"gpu-timing-outputs.npz") as b:
    assert set(a.files) == set(b.files)
    checks["spectra_and_global_history"] = {
        name: {"max_abs": float(np.max(np.abs(a[name]-b[name]))),
               "passing": bool(np.allclose(a[name], b[name], rtol=1e-4, atol=1e-8))}
        for name in a.files
    }
print(json.dumps(checks, indent=2))
