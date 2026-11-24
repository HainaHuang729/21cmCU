"""Bitwise comparison of DexM's ordered CUDA draws with installed add_properties_cat."""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi
from py21cmfast.drivers._global_initialization import c_state

from evolution_backend import NATIVE
from halo_rng import assign_properties


def verify(path):
    library = C.CDLL(str(NATIVE))
    native = library.add_properties_cat
    native.argtypes = [C.c_ulonglong, C.c_float, C.c_void_p]
    native.restype = C.c_int
    library.omp_set_num_threads.argtypes = [C.c_int]
    rows = []
    for threads in (1, 2):
        inputs = p21.InputParameters(random_seed=42).evolve_input_structs(
            N_THREADS=threads
        )
        with c_state(inputs, broadcast_inputs=True):
            library.omp_set_num_threads(threads)
            for seed, count in (
                (0, 0),
                (0, 1),
                (0, 2),
                (37251, 3),
                (92, 911),
                (113, 4097),
                (2**32, 487),
                (2**32 + 3, 144),
                (2**64 - 1, 10003),
                (163179, 700001),
            ):
                # Keep inactive catalogue capacity after the active prefix, as in the API.
                original = np.full((3, count + 7), -123.0, dtype="f")
                got = original.copy()
                catalog = ffi.new("HaloCatalog *")
                catalog.n_halos, catalog.buffer_size = count, count + 7
                for name, array in zip(
                    ("star_rng", "sfr_rng", "xray_rng"), original, strict=True
                ):
                    setattr(catalog, name, ffi.cast("float *", int(array.ctypes.data)))
                status = native(seed, 6.0, int(ffi.cast("uintptr_t", catalog)))
                if status:
                    raise RuntimeError(f"Native reference returned status {status}")
                assign_properties(seed, threads, threads, [a[:count] for a in got])
                different = np.count_nonzero(
                    original.view("u4") != got.view("u4"), axis=1
                )
                if not (
                    np.all(original[:, count:] == -123.0)
                    and np.all(got[:, count:] == -123.0)
                ):
                    raise ValueError("Modified inactive catalogue entries")
                rows.append(
                    dict(
                        seed=seed,
                        threads=threads,
                        count=count,
                        mismatches=different.tolist(),
                        ordered_sha256=hashlib.sha256(
                            original[:, :count].tobytes()
                        ).hexdigest(),
                    )
                )
    path.write_text(
        json.dumps(dict(job=os.environ.get("SLURM_JOB_ID"), cases=rows), indent=2)
        + "\n"
    )
    if any(any(row["mismatches"]) for row in rows):
        raise SystemExit("CUDA draws differ from native")
    print(
        f"{len(rows)} exact cases, {sum(r['count'] * 3 for r in rows)} Gaussian draws"
    )


def verify_ic(path):
    from gpu_lightcone import _backend
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )

    rows = []
    for seed in (2**32, 37251):
        for threads in (1, 2):
            inputs = p21.InputParameters(
                random_seed=seed, node_redshifts=[8.0]
            ).evolve_input_structs(DIM=32, HII_DIM=16, BOX_LEN=24, N_THREADS=threads)
            lightconer = p21.lightconers.RectilinearLightconer.between_redshifts(
                min_redshift=8.2,
                max_redshift=8.4,
                quantities=("brightness_temp",),
                resolution=inputs.simulation_options.cell_size,
                cosmo=inputs.cosmo_params.cosmo,
            )
            manager.free()
            native = p21.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            with (
                TemporaryDirectory() as directory,
                _backend(inputs, lightconer, Path(directory)),
            ):
                gpu = p21.compute_initial_conditions(
                    inputs=inputs, regenerate=True, write=False
                )
            fields = {}
            if set(native.arrays) != set(gpu.arrays):
                raise ValueError("Different requested initial fields")
            for name in native.arrays:
                a, b = native.get(name), gpu.get(name)
                np.testing.assert_equal(a.shape, b.shape)
                fields[name] = dict(
                    n=a.size, different=int((a.view("u4") != b.view("u4")).sum())
                )
            rows.append(dict(seed=seed, threads=threads, fields=fields))
    path.write_text(
        json.dumps(dict(job=os.environ.get("SLURM_JOB_ID"), cases=rows), indent=2)
        + "\n"
    )
    if any(field["different"] for row in rows for field in row["fields"].values()):
        raise SystemExit("CUDA initial fields differ from native")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--initial-conditions",
        action="store_true",
        help="Compare all native initial fields for the shared seed path",
    )
    args = parser.parse_args()
    (verify_ic if args.initial_conditions else verify)(args.output)
