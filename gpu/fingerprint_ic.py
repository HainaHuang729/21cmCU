"""Fingerprint native IC data and record allocated bytes for each field."""
import hashlib
import json
from pathlib import Path
import sys

import h5py

with h5py.File(Path(sys.argv[1]) / "initial-conditions.h5") as archive:
    result = {}
    for name, field in archive["InitialConditions"].items():
        checksum = hashlib.sha256()
        for begin in range(0, field.shape[0], 8):
            checksum.update(field[begin:begin + 8].tobytes())
        result[name] = {
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "sha256": checksum.hexdigest(),
            "bytes_allocated": field.id.get_storage_size(),
        }
    assert len(result) == 14
print(json.dumps(result, indent=2))
