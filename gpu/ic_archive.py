"""Archive the initialized IC while its device fields are used for evolution."""
from concurrent.futures import ThreadPoolExecutor
import os
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np
from py21cmfast.outputs import InitialConditions


class BackgroundICArchive:
    def __init__(self, path, coeval_start, *, sparse_zeros=False):
        self.path = path
        self.coeval_start = coeval_start
        self.sparse_zeros = sparse_zeros
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.box = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True)
        self.box = None

    def capture(self, box):
        if not isinstance(box, InitialConditions):
            return
        if self.job is not None:
            raise RuntimeError("Expected a single IC computation")
        self.box = box  # Keep C-backed NumPy arrays alive until the writer joins.
        arrays = {name: getattr(box, name) for name in box._array_state}
        primitives = {name: getattr(box, name) for name in box.primitive_fields}

        # Use the native writer for metadata without marking the evolving box
        # as on disk. Its prepare/purge logic therefore cannot reload this file.
        metadata = SimpleNamespace(
            _name=box._name,
            _inputs=box._inputs,
            _filter_params=box._filter_params,
            _array_state=box._array_state,
            _global_params=box._global_params,
            **{name: getattr(box, name) for name in box._inputs},
        )
        metadata.write_data_to_hdf5_group = lambda group: group.attrs.update(primitives)
        with h5py.File(self.path, "w") as output:
            type(box).write(metadata, fname=output)
            if box._name not in output:
                raise OSError("Could not write native IC metadata")
        self.job = self.executor.submit(self._write, box._name, arrays)

    def _write(self, group_name, arrays):
        begin = perf_counter()
        zero_arrays = {}
        with h5py.File(self.path, "r+") as output:
            group = output[group_name]
            for name, array in arrays.items():
                if (self.sparse_zeros and name in {"hires_vx", "hires_vy", "hires_vz"}
                        and array.dtype == np.float32 and not np.any(array.view(np.uint32))):
                    dataset = group.create_dataset(name, shape=array.shape, dtype=array.dtype)
                    if dataset.id.get_storage_size() != 0:
                        raise RuntimeError(f"Unexpected allocation for empty HDF5 field {name}")
                    zero_arrays[name] = array.nbytes
                else:
                    group.create_dataset(name, data=array)
        data_written = perf_counter()
        with self.path.open("rb") as stream:
            os.fsync(stream.fileno())
        synced = perf_counter()
        return {
            "write_seconds": data_written - begin,
            "unallocated_zero_arrays": zero_arrays,
            "background_fsync_seconds": synced - data_written,
            "background_write_started_in_coeval_seconds": begin - self.coeval_start,
            "background_sync_finished_in_coeval_seconds": synced - self.coeval_start,
        }

    def finish(self):
        if self.job is None:
            raise RuntimeError("IC write hook was never called")
        begin = perf_counter()
        report = self.job.result()
        report["coeval_wait_for_background_ic_seconds"] = perf_counter() - begin
        return report
