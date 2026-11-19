"""Write coeval and complete IC HDF5 files after the simulation finishes."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter
from types import SimpleNamespace

import h5py

from coeval_export import export_evolved
from parallel_lzf import ParallelLZFWriter

_initial_conditions = None


def _write_ic(path, workers):
    start = perf_counter()
    box = _initial_conditions
    arrays = {name: getattr(box, name) for name in box._array_state}
    metadata = SimpleNamespace(
        _name=box._name, _inputs=box._inputs, _filter_params=box._filter_params,
        _array_state=box._array_state, _global_params=box._global_params,
        **{name: getattr(box, name) for name in box._inputs},
    )
    primitives = {name: getattr(box, name) for name in box.primitive_fields}
    metadata.write_data_to_hdf5_group = lambda group: group.attrs.update(primitives)
    with ParallelLZFWriter(workers) as writer, h5py.File(path, "w") as output:
        type(box).write(metadata, fname=output)
        group = output[box._name]
        writer.write(
            (group, name, values, tuple(min(n, b) for n, b in zip(values.shape, (16, 64, 64))))
            for name, values in arrays.items()
        )
    return {
        "write_seconds": perf_counter() - start, "bytes": path.stat().st_size,
        "compression": "lzf", "workers": workers, "encoded_chunks": writer.chunks,
    }


def archive_pair(coeval, config, out, workers):
    global _initial_conditions
    # Fork before any writer opens HDF5 or starts encoding threads.
    # The child only reads host IC arrays and uses no CUDA entry points.
    _initial_conditions = coeval[0].init_struct
    try:
        with ProcessPoolExecutor(1, mp_context=get_context("fork")) as pool:
            ic = pool.submit(_write_ic, out / "initial-conditions.h5", workers)
            evolved = export_evolved(coeval, config, out / "coeval-fields.h5", "lzf", workers=workers)
            return ic.result(), evolved
    finally:
        _initial_conditions = None
