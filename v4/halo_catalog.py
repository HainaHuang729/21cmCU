"""4.3.2 CHMF catalogue generation, conditional mass tables, and placement."""

import ctypes as ct
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock, get_ident
from unittest.mock import patch

from native_shim import load_shim

import cupy as cp
import numpy as np
import py21cmfast
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper.outputs import HaloCatalog

from halo_dexm import finder_source, install_finder
import halo_mass_function
import catalog_progenitor
from ic_random import RNG_ORDER, Stream, stream_module

HERE = Path(__file__).resolve().parent
RandomWords = ct.CFUNCTYPE(
    ct.c_int, ct.c_int, ct.c_uint, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint)
)
Place = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_float),
)


def _sampler_source(src):
    # The native loops preserve catalog ordering and own the per-worker cursor.
    original = (src / "Stochasticity.c").read_text()
    opening = original[: original.index("void print_hs_consts(")]
    native_functions = (
        "print_hs_consts",
        "stoc_set_consts_cond",
        "stoc_sample",
        "condense_sparse_halolist",
    )
    prototypes = []
    for name in native_functions:
        signature = re.search(
            r"^(?:void|int) " + name + r"\([^{}]+?\) \{", original, re.M
        )
        prototypes.append(signature[0][:-2] + ";")
    loops = original[
        original.index("int sample_halo_grids(") : original.index(
            "int single_test_sample("
        )
    ]
    sections = loops.split("int sample_halo_progenitors(")
    if len(sections) != 2:
        raise ValueError("Unknown halo sampler layout")
    for sphere, function in enumerate(sections):
        # The two samplers consume position and property draws in opposite orders.
        function = function.replace(
            "bool out_of_buffer = false;",
            "bool out_of_buffer = false; CatalogBatch *batch = calloc(1, sizeof(*batch));\n"
            "        if (!batch) Throw(MemoryAllocError);",
        )
        if sphere:
            begin = """R1 = MtoR(prog_buf[jj]);
                queue_halo(batch, halofield_out, istart+count, !out_of_buffer,
                           pos_desc, R2-R1, rng_arr[threadnum], corr_arr, propbuf_in,
                           prepared ? window->draws[prepared_start + jj] : NULL);
                if (!out_of_buffer) halofield_out->halo_masses[istart+count] = prog_buf[jj];"""
            function = function.replace(
                "#pragma omp for\n        for (ii = 0; ii < nhalo_in; ii++) {",
                """ProgenitorWindow *window = calloc(1, sizeof(*window));
        if (!window) Throw(MemoryAllocError);
        int workers = omp_get_num_threads();
        size_huge width = nhalo_in / workers, remainder = nhalo_in % workers;
        index_huge first = width * threadnum + (threadnum < remainder ? threadnum : remainder);
        index_huge end = first + width + (threadnum < remainder);
        for (ii = first; ii < end; ii++) {""",
            )
            function = function.replace(
                "            stoc_set_consts_cond(&hs_constants_priv, M2);",
                """            if ((demand->enabled ||
                 (prepare_cuda_progenitors && catalog_uses_gpu_tables() &&
                  (matter_options_global->SAMPLE_METHOD == SAMPLE_BINARY_SPLIT ||
                   matter_options_global->SAMPLE_METHOD == SAMPLE_PARTITION))) &&
                window->next == window->count)
                prepare_progenitor_window(window, halofield_in, ii, end, hs_constants,
                                          rng_arr[threadnum]);
            bool prepared = window->next < window->count;
            int prepared_start = prepared ? window->offsets[window->next] : 0;
            if (prepared) {
                n_prog = window->offsets[++window->next] - prepared_start;
                memcpy(prog_buf, window->masses + prepared_start, n_prog * sizeof(float));
            }
            if (!prepared || ii == 0)
                load_catalog_condition(window, ii, M2, &hs_constants_priv);""",
            )
            function = function.replace(
                "            stoc_sample(&hs_constants_priv,",
                "            if (!prepared) stoc_sample(&hs_constants_priv,",
            )
            start = function.index("set_prop_rng(rng_arr[threadnum], true,")
            corr, buf = "corr_arr", "propbuf_out"
        else:
            function = function.replace(
                "#pragma omp for reduction(+ : total_volume_excluded)",
                """ProgenitorWindow *window = calloc(1, sizeof(*window));
        if (!window) Throw(MemoryAllocError);
        int workers = omp_get_num_threads();
        int slabs = lo_dim[0] / workers, extra = lo_dim[0] % workers;
        int end_slab = slabs * (threadnum + 1) +
                       (threadnum < extra ? threadnum + 1 : extra);
        index_huge end_cell = (index_huge)end_slab * lo_dim[1] * lo_dim[2];
        #pragma omp for schedule(static) reduction(+ : total_volume_excluded)""",
            )
            function = function.replace(
                "                    stoc_set_consts_cond(&hs_constants_priv, delta);",
                """                    if (demand->enabled && window->next == window->count)
                        prepare_grid_window(window, dens_field, halo_overlap_box, cell_idx,
                            end_cell,
                            count < arraysize_local ? arraysize_local - count : 0,
                            hs_constants, rng_arr[threadnum]);
                    bool prepared = window->next < window->count;
                    int prepared_start = prepared ? window->offsets[window->next] : 0;
                    if (prepared) {
                        nh_buf = window->offsets[++window->next] - prepared_start;
                        memcpy(hm_buf, window->masses + prepared_start, nh_buf * sizeof(float));
                    }
                    if (!prepared || cell_idx == 0)
                        load_catalog_condition(window, cell_idx, delta, &hs_constants_priv);""",
            )
            function = function.replace(
                "                    stoc_sample(&hs_constants_priv,",
                "                    if (!prepared) stoc_sample(&hs_constants_priv,",
            )
            start = function.index("random_point_in_cell(")
            begin = """queue_halo(batch, halofield_out, istart+count, !out_of_buffer,
                           (double[3]){x,y,z}, simulation_options_global->BOX_LEN / lo_dim[0],
                           rng_arr[threadnum], NULL, NULL,
                           prepared ? window->draws[prepared_start + i] : NULL);
                if (!out_of_buffer) halofield_out->halo_masses[istart+count] = hm_buf[i];"""
            corr, buf = "NULL", "prop_buf"
        stop = function.index("count++;", start)
        function = (
            function[:start]
            + begin
            + f"""
#if LOG_LEVEL >= ULTRA_DEBUG_LEVEL
                if (!out_of_buffer) {{
                    flush_halos(batch, halofield_out, {corr});
                    {buf}[0] = halofield_out->star_rng[istart+count];
                    {buf}[1] = halofield_out->sfr_rng[istart+count];
                    {buf}[2] = halofield_out->xray_rng[istart+count];
                }}
#endif
                """
            + function[stop:]
        )
        function = function.replace(
            "istart_threads[threadnum] = istart;",
            f"""flush_halos(batch,halofield_out,{corr});
        record_catalog_conditions(window, {sphere});
        free(batch);
        free(window);
        istart_threads[threadnum] = istart;""",
        )
        sections[sphere] = function
    return "\n".join(
        [
            opening,
            *prototypes,
            (HERE / "halo_catalog_random.h").read_text(),
            halo_mass_function.source(src, original),
            (HERE / "halo_catalog_batch.h").read_text(),
            (HERE / "halo_progenitor_batch.h").read_text(),
            halo_mass_function.sampling_loop(
                "int sample_halo_progenitors(".join(sections)
            ),
        ]
    )


def _batch_callback(report, errors):
    kernel = cp.RawKernel(
        (HERE / "halo_catalog.cu").read_text(), "place_halos", options=("--fmad=false",)
    )
    device = cp.cuda.runtime.getDevice()
    lock = Lock()
    worker_cache = {}
    names = ("halo_grid_positions", "halo_progenitor_positions")
    properties = ("halo_grid_property_gaussians", "halo_progenitor_property_gaussians")
    for name in (
        *names,
        *properties,
        "halo_property_host_roundings",
        "halo_property_batches",
    ):
        report.setdefault(name, 0)

    def _thread_buffers(count):
        tid = get_ident()
        with lock:
            entry = worker_cache.get(tid)
        if entry is not None and entry[1].shape[0] >= count:
            return entry
        cap = max(count, 65536 if entry is None else max(count, entry[1].shape[0]))
        with cp.cuda.Device(device):
            stream = cp.cuda.Stream(non_blocking=True) if entry is None else entry[0]
            rows_dev = cp.empty((cap, 17), dtype=np.float64)
            packed_dev = cp.empty((cap, 7), dtype=np.float32)
            rows_pin_mem = cp.cuda.alloc_pinned_memory(cap * 17 * 8)
            rows_pin = np.frombuffer(
                rows_pin_mem, dtype=np.float64, count=cap * 17
            ).reshape(cap, 17)
            out_pin_mem = cp.cuda.alloc_pinned_memory(cap * 7 * 4)
            out_pin = np.frombuffer(
                out_pin_mem, dtype=np.float32, count=cap * 7
            ).reshape(cap, 7)
            entry = (
                stream,
                rows_dev,
                packed_dev,
                rows_pin,
                out_pin,
                rows_pin_mem,
                out_pin_mem,
            )
        with lock:
            worker_cache[tid] = entry
        return entry

    @Place
    def place(count, sphere, xy, z, correlations, records, target):
        try:
            corr = (
                np.ctypeslib.as_array(correlations, shape=(3,))
                if sphere
                else (0.0, 0.0, 0.0)
            )
            stream, rows_dev, packed_dev, rows_pin, out_pin, _, _ = _thread_buffers(
                count
            )
            in_view = np.ctypeslib.as_array(records, shape=(count, 17))
            out_view = np.ctypeslib.as_array(target, shape=(count, 7))
            np.copyto(rows_pin[:count], in_view)
            with cp.cuda.Device(device), stream:
                cp.cuda.runtime.memcpyAsync(
                    rows_dev.data.ptr,
                    rows_pin.ctypes.data,
                    count * 17 * 8,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
                kernel(
                    ((count + 127) // 128,),
                    (128,),
                    (
                        rows_dev,
                        packed_dev,
                        np.int32(count),
                        np.int32(sphere),
                        np.float64(xy),
                        np.float64(z),
                        *(np.float64(c) for c in corr),
                    ),
                )
                cp.cuda.runtime.memcpyAsync(
                    out_pin.ctypes.data,
                    packed_dev.data.ptr,
                    count * 7 * 4,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )
                stream.synchronize()
            np.copyto(out_view, out_pin[:count])
            flags = out_view[:, 6].astype("u1")
            ambiguous = int(
                np.count_nonzero(flags & 1)
                + np.count_nonzero(flags & 2)
                + np.count_nonzero(flags & 4)
            )
            with lock:
                report["halo_property_host_roundings"] += ambiguous
                report["halo_property_batches"] += 1
                report[properties[sphere]] += 3 * count
                report[names[sphere]] += count
            return 0
        except Exception as exc:
            with lock:
                errors.append(exc)
            return 1

    return place


def _random_callback(report, errors):
    module = stream_module()
    streams = {}
    worker_streams = {}
    word_blocks = {}
    device = cp.cuda.runtime.getDevice()
    lock = Lock()

    @RandomWords
    def supply(worker, seed, first, count, words):
        try:
            kind = RNG_ORDER[worker % 5]
            with lock:
                cuda_stream = worker_streams.get(worker)
                if cuda_stream is None:
                    with cp.cuda.Device(device):
                        cuda_stream = cp.cuda.Stream(non_blocking=True)
                    worker_streams[worker] = cuda_stream
                rng_stream = None if first else streams.get(worker)
            with cp.cuda.Device(device), cuda_stream:
                if first or rng_stream is None:
                    rng_stream = Stream(module, kind, seed)
                out = np.ctypeslib.as_array(words, shape=(count,))
                block = rng_stream.integers(count // 624)
                block.get(out=out)
            with lock:
                streams[worker] = rng_stream
                word_blocks[worker] = block
                name = f"halo_cuda_{kind}_words"
                report[name] = report.get(name, 0) + out.size
            return 0
        except Exception as exc:
            with lock:
                errors.append(exc)
            return 1

    return supply, streams, word_blocks


@contextmanager
def halo_catalog_backend(report):
    """Accelerate CHMF placement and the supported ordered progenitor draw path."""
    src = Path(py21cmfast.__file__).parent / "src"
    native = Path(py21cmfast.c_21cmfast.__file__)
    errors = []
    place = _batch_callback(report, errors)
    random_words, streams, word_blocks = _random_callback(report, errors)
    with TemporaryDirectory(prefix="v432-halo-catalog-") as temp:
        build = Path(temp)
        library = load_shim(
            build,
            "sampler.so",
            {"finder.c": finder_source(src), "sampler.c": _sampler_source(src)},
            src,
            native,
        )
        finder_state, finder_callback = install_finder(library, report, errors)
        catalog_tables = {}
        _table_callbacks = halo_mass_function.bind(
            library, report, errors, catalog_tables
        )
        _progenitor_callback, progenitor_state = catalog_progenitor.bind(
            library, report, errors, word_blocks, catalog_tables
        )
        ct.c_void_p.in_dll(library, "place_on_gpu").value = ct.cast(
            place, ct.c_void_p
        ).value
        ct.c_void_p.in_dll(library, "catalog_random_words").value = ct.cast(
            random_words, ct.c_void_p
        ).value
        compute = ffi.cast(
            ffi.typeof(lib.ComputeHaloCatalog),
            ct.cast(library.ComputeHaloCatalog, ct.c_void_p).value,
        )
        condition_counts = {
            event: (ct.c_ulonglong * 2).in_dll(library, f"catalog_condition_{event}")
            for event in ("builds", "reuses")
        }

        def compute_catalog(*args):
            try:
                status = compute(*args)
            finally:
                for event, counts in condition_counts.items():
                    for index, stage in enumerate(("grid", "progenitor")):
                        if counts[index]:
                            name = f"halo_{stage}_condition_{event}"
                            report[name] = report.get(name, 0) + counts[index]
                            counts[index] = 0
                finder_state.clear()
                streams.clear()
                word_blocks.clear()
                catalog_tables.clear()
                progenitor_state.clear()
            if errors:
                raise errors.pop()
            return status

        with patch.object(
            HaloCatalog, "_c_compute_function", staticmethod(compute_catalog)
        ):
            try:
                yield
            finally:
                streams.clear()
                word_blocks.clear()
