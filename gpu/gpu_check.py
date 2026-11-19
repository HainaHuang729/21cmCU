"""Read every coeval and IC dataset produced by the integrated launcher."""
import h5py
import numpy as np


def verify(directory):
    names = {"coeval-fields.h5": 417, "initial-conditions.h5": 14}
    result = {}
    for filename, expected in names.items():
        path = directory / filename
        datasets = 0
        with h5py.File(path) as h5:
            def validate(_, value):
                nonlocal datasets
                if isinstance(value, h5py.Dataset):
                    for start in range(0, value.shape[0], 8):
                        assert np.isfinite(value[start:start+8]).all()
                    datasets += 1
            h5.visititems(validate)
        assert datasets == expected, (filename, datasets)
        result[filename] = {"file_bytes": path.stat().st_size, "datasets_read": datasets}
    return result
