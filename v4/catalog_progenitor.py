"""Batch the exact progenitor word sequence used by the native catalogue."""

import ctypes as C
import os
from threading import Lock
from time import perf_counter

import cupy as cp
import numpy as np

from catalog_libm import inverse_module, source_for_host
from catalog_poisson_source import configured_gsl_source
from catalog_tan_source import has_tangent_source

Double = C.POINTER(C.c_double)
Int = C.POINTER(C.c_int)
Float = C.POINTER(C.c_float)


class RatioTable(C.Structure):
    _fields_ = [
        ("nx", C.c_int),
        ("ny", C.c_int),
        ("axes", C.c_double * 4),
        ("values", C.POINTER(Double)),
    ]


SampleWindow = C.CFUNCTYPE(
    C.c_int,
    C.c_int,
    Double,
    C.c_int,
    C.c_int,
    C.POINTER(C.c_uint),
    C.c_int,
    C.c_int,
    C.c_int,
    C.POINTER(RatioTable),
    C.c_double,
    C.c_int,
    C.c_int,
    Int,
    Float,
    Double,
    Int,
)
PrepareGrid = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, Float, Float, Double)
PrepareProgenitors = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, Float, Double, C.c_int)


def _new_worker_stream():
    return cp.cuda.Stream(non_blocking=True)


def _pinned_output(device_array):
    memory = cp.cuda.alloc_pinned_memory(device_array.nbytes)
    return np.ndarray(device_array.shape, dtype=device_array.dtype, buffer=memory)


class _WindowBuffers:
    def __init__(self, count, capacity, limit):
        self.count = count
        self.capacity = capacity
        self.limit = limit
        self.offsets = cp.empty(count + 1, dtype=np.int32)
        self.masses = cp.empty(capacity, dtype=np.float32)
        self.draws = cp.empty((capacity, 10), dtype=np.float64)
        self.result = cp.empty(4, dtype=np.int32)
        self.scratch = cp.empty(limit, dtype=np.float32)
        self.branches = None
        self.host_offsets = _pinned_output(self.offsets)
        self.host_masses = _pinned_output(self.masses)
        self.host_draws = _pinned_output(self.draws)
        self.host_result = _pinned_output(self.result)

    def branch_scratch(self):
        if self.branches is None:
            self.branches = cp.empty((self.limit, 2), dtype=np.float32)
        return self.branches


def bind(library, report, errors, word_blocks, catalog_tables):
    state = {}
    path = source_for_host(os.environ.get("P21_CUDA_GLIBC_SOURCE"))
    grid = os.environ.get("P21_CUDA_GRID_SAMPLE") == "1"
    progenitor = os.environ.get("P21_CUDA_PROGENITOR") == "1"
    if path is None or not (grid or progenitor):
        return None, state

    device = cp.cuda.runtime.getDevice()
    gsl_source = configured_gsl_source()
    device_tangent = gsl_source is not None and has_tangent_source(path)
    host_tangent = gsl_source is not None and not device_tangent
    module = inverse_module(path, device, gsl_source)
    tangents = library.catalog_word_tangents
    tangents.argtypes = [
        C.c_int,
        C.POINTER(C.c_uint),
        np.ctypeslib.ndpointer(np.float64),
    ]
    tangents.restype = None
    kernel = module.get_function("draw_catalog_prefix")
    prepare_grid_kernel = module.get_function("prepare_catalog_grid")
    prepare_progenitor_kernel = module.get_function("prepare_catalog_progenitors")
    table_lock = Lock()
    view = np.ctypeslib.as_array
    worker_streams = {}
    worker_buffers = {}

    def worker_stream(worker):
        # Each worker's condition preparation and random-word windows stay
        # ordered, while independent workers can execute concurrently.
        with table_lock:
            stream = worker_streams.get(worker)
            if stream is None:
                stream = _new_worker_stream()
                worker_streams[worker] = stream
            return stream

    def output_buffers(worker, count, capacity, limit):
        with table_lock:
            buffers = worker_buffers.get(worker)
            if buffers is not None:
                if (
                    count <= buffers.count
                    and capacity <= buffers.capacity
                    and limit <= buffers.limit
                ):
                    return buffers
                count = max(count, buffers.count)
                capacity = max(capacity, buffers.capacity)
                limit = max(limit, buffers.limit)
            buffers = _WindowBuffers(count, capacity, limit)
            worker_buffers[worker] = buffers
            return buffers

    def prepare_conditions(count, worker, samples, stage, prepare, args):
        try:
            with table_lock:
                table = catalog_tables.get("moments")
                resident = catalog_tables.get("samples")
            if table is None:
                return 1
            moments, origin, width = table
            if moments.shape[0] < 2 or not np.isfinite(width) or width <= 0:
                return 1
            with cp.cuda.Device(device), worker_stream(worker):
                source = None
                if resident is not None:
                    address, values = resident
                    offset, remainder = divmod(
                        C.cast(samples, C.c_void_p).value - address, values.itemsize
                    )
                    if remainder == 0 and 0 <= offset <= values.size - count:
                        source = values[offset : offset + count]
                reused_samples = source is not None
                if source is None:
                    source = cp.asarray(view(samples, shape=(count,)))
                conditions = cp.empty((count, 4), dtype=np.float64)
                prepare(
                    ((count + 127) // 128,),
                    (128,),
                    (
                        np.int32(count),
                        source,
                        *args(
                            moments,
                            np.int32(moments.shape[0]),
                            np.float64(origin),
                            np.float64(width),
                        ),
                        conditions,
                    ),
                )
            with table_lock:
                state[stage, worker] = conditions
                key = f"halo_cuda_{stage}_conditions_prepared"
                report[key] = report.get(key, 0) + count
                if reused_samples:
                    field = "density" if stage == "grid" else "masses"
                    key = f"halo_cuda_{stage}_resident_{field}"
                    report[key] = report.get(key, 0) + count
            return 0
        except Exception as exc:
            with table_lock:
                errors.append(exc)
            return 1

    @PrepareGrid
    def prepare_grid(count, worker, density, overlap, bounds):
        def args(*moments):
            return (
                cp.asarray(view(overlap, shape=(count,))),
                *moments,
                *view(bounds, shape=(6,)),
            )

        return prepare_conditions(
            count, worker, density, "grid", prepare_grid_kernel, args
        )

    @PrepareProgenitors
    def prepare_progenitors(count, worker, masses, bounds, sampling):
        if sampling == 1 and not device_tangent:
            return 1
        if sampling in (2, 3):
            try:
                with cp.cuda.Device(device), table_lock:
                    if "tree" not in state:
                        from evolution_backend import Table, native_tables

                        sig_name = "Sigma_InterpTable"
                        names = (sig_name, "dSigmasqdm_InterpTable", "J_split_table")
                        addresses = native_tables(
                            names if sampling == 2 else (sig_name,)
                        )
                        native_sigma = Table.from_address(addresses[sig_name])
                        sigma = cp.asarray(
                            view(native_sigma.y, shape=(native_sigma.n,)), blocking=True
                        )
                        low, width = native_sigma.low, native_sigma.width
                        params = view(bounds, shape=(16,))
                        if sampling == 2:
                            from halo_mass_function import NumberTable

                            slope = Table.from_address(addresses[names[1]])
                            j = NumberTable.from_address(addresses[names[2]])
                            extra_tables = tuple(
                                cp.asarray(view(t.y, shape=(t.n,)), blocking=True)
                                for t in (slope, j)
                            )
                            options = [
                                native_sigma.n,
                                j.n,
                                low,
                                width,
                                j.low,
                                j.width,
                                params[2] / params[0],
                                params[2] / params[1],
                                params[3],
                                *params[9:13],
                            ]
                        else:
                            extra_tables = (np.uintp(0),) * 2
                            options = [
                                native_sigma.n,
                                low,
                                width,
                                *params[:3],
                                params[13],
                                params[9],
                                params[8],
                                params[14],
                            ]
                        catalog_tables["sigma"] = sigma, low, width
                        state["tree"] = (
                            sigma,
                            *extra_tables,
                            cp.asarray(options, dtype=np.float64, blocking=True),
                        )
                        state["binary"] = sampling == 2
            except Exception as exc:
                with table_lock:
                    errors.append(exc)
                return 1
        with table_lock:
            saved = catalog_tables.get("sigma")
        if saved is None:
            return 1
        sigma, low, width = saved

        def args(*moments):
            return (
                *moments,
                sigma,
                np.int32(sigma.size),
                np.float64(low),
                np.float64(width),
                *view(bounds, shape=(9,)),
                np.int32(sampling),
                np.int32(view(bounds, shape=(16,))[14]),
                np.float64(view(bounds, shape=(16,))[15]),
            )

        return prepare_conditions(
            count, worker, masses, "progenitor", prepare_progenitor_kernel, args
        )

    @SampleWindow
    def fill(
        count,
        conditions,
        condition_offset,
        worker,
        native_words,
        word_offset,
        word_count,
        from_catalog,
        table,
        minimum,
        limit,
        capacity,
        offsets,
        masses,
        draws,
        result,
    ):
        try:
            started = perf_counter()
            stream = worker_stream(worker)
            with cp.cuda.Device(device), stream:
                buffers = output_buffers(worker, count, capacity, limit)
                if table:
                    with table_lock:
                        if "ratios" not in state:
                            t = table.contents
                            rows = np.stack(
                                [view(t.values[i], shape=(t.ny,)) for i in range(t.nx)]
                            )
                            state.update(
                                ratios=cp.asarray(rows, blocking=True),
                                axes=cp.asarray(
                                    [*t.axes, t.axes[1]],
                                    dtype=np.float64,
                                    blocking=True,
                                ),
                                shape=(np.int32(t.nx), np.int32(t.ny)),
                            )
                        inverse = state["ratios"], state["axes"], *state["shape"]
                    tree = (np.uintp(0),) * 5
                else:
                    inverse = np.uintp(0), np.uintp(0), np.int32(0), np.int32(0)
                    with table_lock:
                        binary = state["binary"]
                        tree_state = state["tree"]
                    branches = buffers.branch_scratch() if binary else np.uintp(0)
                    tree = (*tree_state, branches)
                if conditions:
                    parameters = view(conditions, shape=(count, 4))
                    device_conditions = cp.asarray(parameters)
                else:
                    stage = "progenitor" if from_catalog else "grid"
                    with table_lock:
                        device_conditions = state[stage, worker][
                            condition_offset : condition_offset + count
                        ]
                with table_lock:
                    block = word_blocks[worker]
                    saved_tan = state.get(("tangents", worker))
                stop = word_offset + word_count
                device_words = block[word_offset:stop]
                device_tangents = np.uintp(0)
                if (
                    host_tangent
                    and conditions
                    and np.any((parameters[:, 3] == 4) & (parameters[:, 2] > 10))
                ):
                    if saved_tan is None or saved_tan[0] is not block:
                        host_tangents = np.empty(block.size, dtype=np.float64)
                        tangents(block.size, native_words, host_tangents)
                        saved_tan = (block, cp.asarray(host_tangents))
                        with table_lock:
                            state[("tangents", worker)] = saved_tan
                    device_tangents = saved_tan[1][word_offset:stop]
                kernel(
                    (1,),
                    (1,),
                    (
                        np.int32(count),
                        device_conditions,
                        device_words,
                        np.int32(word_count),
                        device_tangents,
                        *inverse,
                        np.float64(minimum),
                        np.int32(from_catalog),
                        buffers.scratch,
                        np.int32(limit),
                        np.int32(capacity),
                        buffers.offsets,
                        buffers.masses,
                        buffers.draws,
                        buffers.result,
                        *tree,
                    ),
                )
                buffers.result.get(out=buffers.host_result)
                done, consumed, high_means, stored = map(int, buffers.host_result)
                if not (
                    0 <= done <= count
                    and 0 <= consumed <= word_count
                    and 0 <= high_means <= done
                ):
                    raise ValueError("Invalid CUDA progenitor word cursor")
                if not 0 <= stored <= capacity or (stored and not done):
                    raise ValueError("Invalid CUDA progenitor batch size")
                native_offsets = view(offsets, shape=(done + 1,))
                if stored:
                    local_offsets = buffers.host_offsets[: done + 1]
                    local_masses = buffers.host_masses[:stored]
                    local_draws = buffers.host_draws[:stored]
                    buffers.offsets[: done + 1].get(out=local_offsets, blocking=False)
                    buffers.masses[:stored].get(out=local_masses, blocking=False)
                    buffers.draws[:stored].get(out=local_draws, blocking=False)
                    stream.synchronize()
                    if local_offsets[-1] != stored:
                        raise ValueError("Inconsistent CUDA progenitor batch size")
                    native_offsets[:] = local_offsets
                    view(masses, shape=(stored,))[:] = local_masses
                    view(draws, shape=(stored, 10))[:] = local_draws
                else:
                    native_offsets.fill(0)
                result[0], result[1] = done, consumed

            name = "halo_cuda_progenitor" if from_catalog else "halo_cuda_grid_sample"
            with table_lock:
                for key, value in (
                    (f"{name}_conditions", done),
                    (f"{name}_masses", stored),
                    (f"{name}_words_consumed", consumed),
                    (f"{name}_high_poisson_conditions", high_means),
                    (
                        f"{name}_device_tangent_conditions",
                        high_means if device_tangent else 0,
                    ),
                    (f"{name}_sampling_seconds", perf_counter() - started),
                ):
                    report[key] = report.get(key, 0) + value
            return 0
        except Exception as exc:
            with table_lock:
                errors.append(exc)
            return 1

    C.c_void_p.in_dll(library, "sample_progenitor_window").value = C.cast(
        fill, C.c_void_p
    ).value
    C.c_int.in_dll(library, "sample_high_poisson_enabled").value = (
        gsl_source is not None
    )
    C.c_int.in_dll(library, "sample_grid_enabled").value = grid
    C.c_int.in_dll(library, "sample_progenitor_enabled").value = progenitor
    if grid and device_tangent:
        C.c_void_p.in_dll(library, "prepare_cuda_grid").value = C.cast(
            prepare_grid, C.c_void_p
        ).value
    if progenitor:
        C.c_void_p.in_dll(library, "prepare_cuda_progenitors").value = C.cast(
            prepare_progenitors, C.c_void_p
        ).value
    return (fill, prepare_grid, prepare_progenitors), state
