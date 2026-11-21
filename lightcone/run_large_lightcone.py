"""Generate matched standard rectilinear lightcones for strict slice comparison."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
import py21cmfast
import py21cmfast.c_21cmfast as native

FIELDS = ("brightness_temp","xH_box","x_e_box","Tk_box","Ts_box","dNrec_box")
config = json.loads((Path(os.environ["PS_GPU_ROOT"])/"point_config.json").read_text())
mode, destination, scratch = sys.argv[1:]
destination = Path(destination)
destination.mkdir()
dim = int(os.environ.get("PS_DIM", "384"))
hii_dim = int(os.environ.get("PS_HII_DIM", "96"))
box_len = float(os.environ.get("PS_BOX_LEN", "375.0"))
max_z = float(os.environ.get("PS_MAX_Z", "14.0"))
win_env = os.environ.get("PS_WINDOW_MAX_Z", "14.0").strip().lower()
window_max_z = None if win_env in ("", "none", "off", "0") else float(win_env)
if os.environ.get("PS_SEED"):
    config["seed"] = int(os.environ["PS_SEED"])
if os.environ.get("PS_POINT_INDEX"):
    config["index"] = int(os.environ["PS_POINT_INDEX"])
if os.environ.get("PS_ASTRO_JSON"):
    config["astro"].update(json.loads(os.environ["PS_ASTRO_JSON"]))
if os.environ.get("PS_COSMO_JSON"):
    config["cosmo"].update(json.loads(os.environ["PS_COSMO_JSON"]))
config["user"].update(DIM=dim, HII_DIM=hii_dim, BOX_LEN=box_len)
if mode not in ("cpu","gpu","gpu-last","gpu-guard","gpu-all","gpu-density"):
    raise ValueError(mode)
gpu = mode!="cpu"
from py21cmfast.outputs import InitialConditions
ic_start=perf_counter()
path=os.environ["PS_CONTROL_IC"]
if gpu:
    from coupled import CoupledRuntime
    runtime=CoupledRuntime()
    runtime.install(native,cached_ic=True,same_gsl=True)
    fields=["hires_density","lowres_density"] + [
      "lowres_v"+axis+suffix for suffix in ("","_2LPT") for axis in "xyz"
    ]
    init_box=InitialConditions.from_file(path,arrays_to_load=fields)
    def keep_loaded_ic(flag_options,force=False):
        if init_box.user_params.USE_RELATIVE_VELOCITIES or flag_options.USE_HALO_FIELD:
            raise ValueError("Unsupported IC features for the GPU control")
    init_box.prepare_for_spin_temp=keep_loaded_ic
    runtime.perturbation.load_cpu_ic(init_box)
else:
    init_box=InitialConditions.from_file(path)
ic_seconds=perf_counter()-ic_start
assert (int(init_box.random_seed), init_box.user_params.DIM, init_box.user_params.HII_DIM) == (config["seed"], dim, hii_dim) and np.isclose(init_box.user_params.BOX_LEN, box_len)
native_scales=0
native_seconds=0.
if gpu:
    import cupy as cp
    import ion_runtime
    library=C.CDLL(native.__file__)
    library.SetNativeFirstHeatHigh.argtypes=[C.c_int]
    library.SetNativeFirstHeatHigh(1)
    ion=runtime.ionization
if mode in ("gpu-last","gpu-guard","gpu-all","gpu-density"):
    # Use the native reference on the final scale.
    actual_filter = ion.filter
    actual_lookup = ion.lookup
    actual_apply = ion.apply
    set_mode = library.SetGPUConditionalMode
    set_mode.argtypes = [C.c_int]
    modes = runtime.tables.gpu_modes
    args_ptr = C.POINTER(ion_runtime.IonArgs)
    long_ptr = C.POINTER(C.c_long)
    float_ptr = C.POINTER(C.c_float)
    reset = library.IonReferenceReset
    bounds = library.IonReferenceBounds
    bounds.argtypes = [args_ptr]; bounds.restype = C.c_int
    mean = library.IonReferenceMean
    mean.argtypes = [args_ptr,C.POINTER(C.c_double)]; mean.restype = C.c_int
    cells = library.IonReferenceCells
    cells.argtypes = [args_ptr,long_ptr,C.c_long,float_ptr,float_ptr,C.c_int]; cells.restype = C.c_int
    pending = False
    native_seconds = 0.
    batch_density_seconds = 0.
    delta_upload_seconds = 0.
    xerec_replay_seconds = 0.
    native_scales = 0
    checked_scales = 0
    guard_scales = []
    prepared_node = None
    # A screening band on the physical photon-budget decision, independent of output tolerances.
    band = float(os.environ.get("PS_GUARD_EPS_MULT", "32"))*np.finfo(np.float32).eps
    rec_replay_scales = 0
    screened_noop_scales = 0
    active_density_replays = 0
    screened_density = (mode == "gpu-density") and (os.environ.get("PS_ION_SCREENED_DENSITY", "1") == "1")

    if screened_density:
        os.environ.setdefault("PS_ION_SLIM_EXPORT", "1")
        ion.skip_density_fft = False
        single_density_mt = library.IonReferenceSingleDensityMT
        single_density_mt.argtypes = [args_ptr, float_ptr]; single_density_mt.restype = C.c_int
        xerec_box_opt = library.IonReferenceXeRecBoxOpt
        xerec_box_opt.argtypes = [args_ptr, float_ptr, float_ptr, C.c_int]; xerec_box_opt.restype = C.c_int
        ion_vox = hii_dim ** 3
        pinned_delta_mem = cp.cuda.alloc_pinned_memory(ion_vox * 4)
        pinned_xe_mem = cp.cuda.alloc_pinned_memory(ion_vox * 4)
        pinned_rec_mem = cp.cuda.alloc_pinned_memory(ion_vox * 4)
        pinned_delta = np.frombuffer(pinned_delta_mem, dtype=np.float32, count=ion_vox)
        pinned_xe = np.frombuffer(pinned_xe_mem, dtype=np.float32, count=ion_vox)
        pinned_rec = np.frombuffer(pinned_rec_mem, dtype=np.float32, count=ion_vox)
        scale_replayed_density = False
        scale_is_noop = False
        amp_mult = float(os.environ.get("PS_XEREC_AMP_MULT", "0"))
        screen_ratio = float(os.environ.get("PS_ION_SCREEN_RATIO", "0.99"))

        density_addr_to_iz = {}
        ion_prefetch_hits = 0
        ion_prefetch_misses = 0
        ion_prefetch_wait_seconds = 0.0

        def after_heat_filter_cb(dens_addr, side, dk):
            iz = density_addr_to_iz.get(dens_addr)
            if iz is not None:
                runtime.submit_heat_prefetch_for_index(iz + 1)

        def set_lc_schedule(scrollz, perturb_list, _user_params):
            density_addr_to_iz.clear()
            for iz, pf in enumerate(perturb_list):
                if hasattr(pf, "density"):
                    density_addr_to_iz[pf.density.ctypes.data] = iz
            if runtime.prefetch_enabled:
                runtime._lc_schedule = (scrollz, perturb_list)
                runtime._before_heat_filter_cb = None
                runtime._after_heat_filter_cb = after_heat_filter_cb

        def clear_lc_schedule():
            runtime.join_and_clear_prefetch()
            density_addr_to_iz.clear()

        if os.environ.get("PS_ASYNC_FFTW_PREFETCH", "1") == "1":
            py21cmfast._ps_set_lightcone_schedule = set_lc_schedule
            py21cmfast._ps_clear_lightcone_schedule = clear_lc_schedule

        def native_scale_filter(a):
            global prepared_node, native_seconds, batch_density_seconds, delta_upload_seconds
            global scale_replayed_density, scale_is_noop
            start = perf_counter()
            scale_replayed_density = False
            scale_is_noop = False
            if prepared_node != a.z:
                reset()
                prepared_node = a.z
            if a.last:
                runtime.wait_heat_prefetch()
                t0 = perf_counter()
                if single_density_mt(C.byref(a), pinned_delta.ctypes.data_as(float_ptr)):
                    raise RuntimeError("Native single density FFTW failed")
                batch_density_seconds += perf_counter() - t0
                tu = perf_counter()
                cp.cuda.runtime.memcpyAsync(
                    ion.delta.data.ptr,
                    pinned_delta.ctypes.data,
                    pinned_delta.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice,
                    cp.cuda.get_current_stream().ptr,
                )
                delta_upload_seconds += perf_counter() - tu
                scale_replayed_density = True
            else:
                actual_filter(a)
            native_seconds += perf_counter() - start

        def _upload_pinned_xerec(skip_rec):
            stream_ptr = cp.cuda.get_current_stream().ptr
            cp.cuda.runtime.memcpyAsync(
                ion.xe.data.ptr, pinned_xe.ctypes.data, pinned_xe.nbytes,
                cp.cuda.runtime.memcpyHostToDevice, stream_ptr,
            )
            if skip_rec:
                ion.rec.fill(0)
            else:
                cp.cuda.runtime.memcpyAsync(
                    ion.rec.data.ptr, pinned_rec.ctypes.data, pinned_rec.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice, stream_ptr,
                )

        def native_scale_lookup(a):
            global native_seconds, native_scales, checked_scales, rec_replay_scales, xerec_replay_seconds
            global batch_density_seconds, delta_upload_seconds
            global screened_noop_scales, active_density_replays, scale_replayed_density, scale_is_noop
            actual_lookup(a)
            start = perf_counter()
            if a.last:
                runtime.wait_heat_prefetch()
                tx = perf_counter()
                skip_rec = 1 if ion.rec_is_zero else 0
                if xerec_box_opt(C.byref(a), pinned_xe.ctypes.data_as(float_ptr), pinned_rec.ctypes.data_as(float_ptr), skip_rec):
                    raise RuntimeError("Native xe/rec FFTW replay failed")
                _upload_pinned_xerec(skip_rec)
                xerec_replay_seconds += perf_counter() - tx
                rec_replay_scales += 1
                native_scales += 1
                native_seconds += perf_counter() - start
                return 0
            if not scale_replayed_density:
                if not ion.screen_scale(a, screen_ratio):
                    scale_is_noop = True
                    screened_noop_scales += 1
                    native_scales += 1
                    native_seconds += perf_counter() - start
                    return 0
                runtime.wait_heat_prefetch()
                t0 = perf_counter()
                if single_density_mt(C.byref(a), pinned_delta.ctypes.data_as(float_ptr)):
                    raise RuntimeError("Native single density FFTW failed")
                batch_density_seconds += perf_counter() - t0
                tu = perf_counter()
                cp.cuda.runtime.memcpyAsync(
                    ion.delta.data.ptr,
                    pinned_delta.ctypes.data,
                    pinned_delta.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice,
                    cp.cuda.get_current_stream().ptr,
                )
                delta_upload_seconds += perf_counter() - tu
                scale_replayed_density = True
                active_density_replays += 1
                native_seconds += perf_counter() - start
                return 2
            if amp_mult > 0.0:
                distance = ion.minimum_margin_amp(a, ion.xe_max, ion.rec_max)
                need_xerec = (distance <= amp_mult)
            else:
                distance = ion.minimum_margin(a)
                need_xerec = (distance <= band)
            checked_scales += 1
            if need_xerec:
                guard_scales.append([a.z, a.radius, float(distance)])
                runtime.wait_heat_prefetch()
                tx = perf_counter()
                skip_rec = 1 if ion.rec_is_zero else 0
                if xerec_box_opt(C.byref(a), pinned_xe.ctypes.data_as(float_ptr), pinned_rec.ctypes.data_as(float_ptr), skip_rec):
                    raise RuntimeError("Native xe/rec FFTW replay failed")
                _upload_pinned_xerec(skip_rec)
                xerec_replay_seconds += perf_counter() - tx
                rec_replay_scales += 1
            native_scales += 1
            native_seconds += perf_counter() - start
            return 0

        def native_scale_apply(a):
            if not scale_is_noop:
                actual_apply(a)

    elif mode == "gpu-density":
        ion.skip_density_fft = True
        int_ptr = C.POINTER(C.c_int)
        sched_count = library.IonReferenceScheduleCount
        sched_count.argtypes = []; sched_count.restype = C.c_int
        batch_density = library.IonReferenceBatchDensity
        batch_density.argtypes = [args_ptr, float_ptr, C.c_int]; batch_density.restype = C.c_int
        batch_prefetch = library.IonReferenceBatchDensityPrefetch
        batch_prefetch.argtypes = [
            C.c_int, C.c_double, C.c_int, float_ptr,
            float_ptr, C.c_int,
            int_ptr, float_ptr, int_ptr, float_ptr, float_ptr,
        ]
        batch_prefetch.restype = C.c_int
        batch_commit = library.IonReferenceCommitPrefetch
        batch_commit.argtypes = [C.c_int, float_ptr, int_ptr, float_ptr, float_ptr]
        batch_commit.restype = C.c_int
        batch_info = library.IonReferenceBatchScaleInfo
        batch_info.argtypes = [C.c_int, C.c_float, C.c_int, float_ptr, float_ptr]; batch_info.restype = C.c_int
        xerec_box = library.IonReferenceXeRecBox
        xerec_box.argtypes = [args_ptr, float_ptr, float_ptr]; xerec_box.restype = C.c_int
        ion_vox = hii_dim ** 3
        pinned_xe_mem = cp.cuda.alloc_pinned_memory(ion_vox * 4)
        pinned_rec_mem = cp.cuda.alloc_pinned_memory(ion_vox * 4)
        pinned_xe = np.frombuffer(pinned_xe_mem, dtype=np.float32, count=ion_vox)
        pinned_rec = np.frombuffer(pinned_rec_mem, dtype=np.float32, count=ion_vox)
        pinned_batch_mem = None
        pinned_batch = None
        scale_idx = 0

        ion_pool = ThreadPoolExecutor(max_workers=1) if os.environ.get("PS_ASYNC_FFTW_PREFETCH", "1") == "1" else None
        density_addr_to_iz = {}
        ion_active_started = False
        ion_future = None
        ion_pf_meta = None
        ion_prefetch_hits = 0
        ion_prefetch_misses = 0
        ion_prefetch_wait_seconds = 0.0
        nt_bg = max(1, int(os.environ.get("OMP_NUM_THREADS", "8")) - 1)

        def after_heat_filter_cb(dens_addr, side, dk):
            global ion_future, ion_pf_meta
            iz = density_addr_to_iz.get(dens_addr)
            if iz is None:
                return
            if ion_active_started and ion_pool is not None and pinned_batch is not None:
                out_n = C.c_int(0)
                out_r = np.empty(128, dtype=np.float32)
                out_f = np.empty(128, dtype=np.int32)
                out_lo = np.empty(128, dtype=np.float32)
                out_hi = np.empty(128, dtype=np.float32)
                ion_pf_meta = (dens_addr, int(side), float(dk), iz, out_n, out_r, out_f, out_lo, out_hi)
                ion_future = ion_pool.submit(
                    batch_prefetch,
                    int(side), float(dk), nt_bg, C.cast(C.c_void_p(dens_addr), float_ptr),
                    pinned_batch.ctypes.data_as(float_ptr), pinned_batch.shape[0],
                    C.byref(out_n), out_r.ctypes.data_as(float_ptr), out_f.ctypes.data_as(int_ptr),
                    out_lo.ctypes.data_as(float_ptr), out_hi.ctypes.data_as(float_ptr),
                )
            else:
                runtime.submit_heat_prefetch_for_index(iz + 1)

        def before_heat_filter_cb():
            global ion_future, ion_pf_meta
            if ion_future is not None:
                try:
                    ion_future.result()
                finally:
                    ion_future = None
                    ion_pf_meta = None

        def set_lc_schedule(scrollz, perturb_list, _user_params):
            density_addr_to_iz.clear()
            for iz, pf in enumerate(perturb_list):
                if hasattr(pf, "density"):
                    density_addr_to_iz[pf.density.ctypes.data] = iz
            if runtime.prefetch_enabled:
                runtime._lc_schedule = (scrollz, perturb_list)
                runtime._before_heat_filter_cb = before_heat_filter_cb
                runtime._after_heat_filter_cb = after_heat_filter_cb

        def clear_lc_schedule():
            global ion_future, ion_pf_meta
            if ion_future is not None:
                try:
                    ion_future.result()
                finally:
                    ion_future = None
                    ion_pf_meta = None
            if ion_pool is not None:
                ion_pool.shutdown(wait=True)
            runtime.join_and_clear_prefetch()
            density_addr_to_iz.clear()

        if os.environ.get("PS_ASYNC_FFTW_PREFETCH", "1") == "1":
            py21cmfast._ps_set_lightcone_schedule = set_lc_schedule
            py21cmfast._ps_clear_lightcone_schedule = clear_lc_schedule

        def native_scale_filter(a):
            global prepared_node, native_seconds, batch_density_seconds, delta_upload_seconds
            global pinned_batch_mem, pinned_batch, scale_idx
            global ion_active_started, ion_future, ion_pf_meta, ion_prefetch_hits, ion_prefetch_misses, ion_prefetch_wait_seconds
            start = perf_counter()
            if prepared_node != a.z:
                t0 = perf_counter()
                dens_ptr_val = C.cast(a.density, C.c_void_p).value
                committed = False
                if ion_future is not None:
                    tw = perf_counter()
                    rc = ion_future.result()
                    ion_prefetch_wait_seconds += perf_counter() - tw
                    meta = ion_pf_meta
                    ion_future = None
                    ion_pf_meta = None
                    if rc == 0 and meta is not None and meta[0] == dens_ptr_val and meta[1] == int(a.side) and meta[2] == float(a.dk):
                        _, _, _, _, out_n, out_r, out_f, out_lo, out_hi = meta
                        if batch_commit(
                            out_n.value,
                            out_r.ctypes.data_as(float_ptr),
                            out_f.ctypes.data_as(int_ptr),
                            out_lo.ctypes.data_as(float_ptr),
                            out_hi.ctypes.data_as(float_ptr),
                        ) == 0:
                            committed = True
                            ion_prefetch_hits += 1
                    if not committed:
                        ion_prefetch_misses += 1
                if not committed:
                    runtime.wait_heat_prefetch()
                    reset()
                    n_sched = sched_count()
                    if n_sched <= 0:
                        raise RuntimeError(f"Invalid ion schedule count: {n_sched}")
                    if pinned_batch is None or pinned_batch.shape[0] < n_sched:
                        pinned_batch_mem = cp.cuda.alloc_pinned_memory(n_sched * ion_vox * 4)
                        pinned_batch = np.frombuffer(pinned_batch_mem, dtype=np.float32, count=n_sched * ion_vox).reshape(n_sched, ion_vox)
                    if batch_density(C.byref(a), pinned_batch.ctypes.data_as(float_ptr), n_sched):
                        raise RuntimeError("Native batch density FFTW failed")
                prepared_node = a.z
                scale_idx = 0
                ion_active_started = True
                iz_cur = density_addr_to_iz.get(dens_ptr_val)
                if iz_cur is not None:
                    runtime.submit_heat_prefetch_for_index(iz_cur + 1)
                batch_density_seconds += perf_counter() - t0
            lo_v, hi_v = C.c_float(), C.c_float()
            if batch_info(scale_idx, a.radius, a.filtered, C.byref(lo_v), C.byref(hi_v)):
                raise RuntimeError(f"Batch scale mismatch z={a.z} idx={scale_idx} R={a.radius} filtered={a.filtered}")
            a.lo = min(0.0, float(lo_v.value))
            a.hi = max(0.0, float(hi_v.value))
            tu = perf_counter()
            src_buf = pinned_batch[scale_idx]
            cp.cuda.runtime.memcpyAsync(
                ion.delta.data.ptr,
                src_buf.ctypes.data,
                src_buf.nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                cp.cuda.get_current_stream().ptr,
            )
            delta_upload_seconds += perf_counter() - tu
            scale_idx += 1
            if not a.last:
                ion.filter_xe_rec(a)
            native_seconds += perf_counter() - start

        def native_scale_lookup(a):
            global native_seconds, native_scales, checked_scales, rec_replay_scales, xerec_replay_seconds
            actual_lookup(a)
            start = perf_counter()
            full = bool(a.last)
            if not full:
                distance = ion.minimum_margin(a)
                checked_scales += 1
                if distance <= band:
                    guard_scales.append([a.z, a.radius, distance])
                    full = True
            if full:
                runtime.wait_heat_prefetch()
                tx = perf_counter()
                if xerec_box(C.byref(a), pinned_xe.ctypes.data_as(float_ptr), pinned_rec.ctypes.data_as(float_ptr)):
                    raise RuntimeError("Native xe/rec FFTW replay failed")
                ion.xe.set(pinned_xe)
                ion.rec.set(pinned_rec)
                xerec_replay_seconds += perf_counter() - tx
                rec_replay_scales += 1
            native_seconds += perf_counter() - start
            native_scales += 1

        def native_scale_apply(a):
            actual_apply(a)
    else:
        indices = np.arange(hii_dim ** 3, dtype=np.int64)
        fields = np.empty((3, indices.size), dtype=np.float32)
        collapse = np.empty(indices.size, dtype=np.float32)
        batch_density_seconds = 0.
        delta_upload_seconds = 0.
        xerec_replay_seconds = 0.
        ion_prefetch_hits = 0
        ion_prefetch_misses = 0
        ion_prefetch_wait_seconds = 0.

        def reference_filter(a):
            global prepared_node, native_seconds,pending
            start=perf_counter()
            pending=True
            if prepared_node!=a.z:
                reset()
                prepared_node=a.z
            if bounds(C.byref(a)):
                raise RuntimeError("Native ion filter failed")
            runtime.tables.gpu_modes=modes & ~1
            set_mode(modes & ~1)
            native_seconds+=perf_counter()-start

        def native_scale_filter(a):
            global pending, native_seconds
            actual_filter(a)
            pending = False
            if a.last or mode in ("gpu-all","gpu-density"):
                reference_filter(a)

        def native_scale_lookup(a):
            global native_seconds,native_scales,checked_scales,rec_replay_scales
            if not pending:
                actual_lookup(a)
                if mode=="gpu-guard":
                    distance=ion.minimum_margin(a)
                    checked_scales+=1
                    if distance<=band:
                        guard_scales.append([a.z,a.radius,distance])
                        reference_filter(a)
                        return 2
                return
            start = perf_counter()
            total = C.c_double()
            if mean(C.byref(a),C.byref(total)):
                raise RuntimeError("Native collapse calculation failed")
            full = (mode!="gpu-density" or bool(a.last))
            def read_ref(flag):
                status=cells(C.byref(a),indices.ctypes.data_as(long_ptr),indices.size,
                    fields.ctypes.data_as(float_ptr),collapse.ctypes.data_as(float_ptr),flag)
                if status: raise RuntimeError("Native scale field extraction failed")
            read_ref(1 if full else 2)
            ion.delta=cp.asarray(fields[0])
            ion.fc=cp.asarray(collapse)
            ion.collapse_sum=cp.asarray(total.value,dtype=cp.float64)
            if not full:
                distance=ion.minimum_margin(a)
                checked_scales+=1
                if distance<=band:
                    guard_scales.append([a.z,a.radius,distance])
                    read_ref(1)
                    full=True
            if full:
                ion.xe,ion.rec=cp.asarray(fields[1:])
                rec_replay_scales+=1
            native_seconds += perf_counter()-start
            native_scales += 1

        def native_scale_apply(a):
            try:
                actual_apply(a)
            finally:
                if pending:
                    runtime.tables.gpu_modes = modes
                    set_mode(modes)

    ion.filter= native_scale_filter
    ion.lookup= native_scale_lookup
    ion.apply= native_scale_apply


begin = perf_counter()
lc = py21cmfast.run_lightcone(
    redshift=5., max_redshift=max_z, user_params=config["user"],
    cosmo_params=config["cosmo"], astro_params=config["astro"], flag_options=config["flags"],
    init_box=init_box, random_seed=config["seed"], direc=scratch, regenerate=True, write=False, cleanup=True,
    lightcone_quantities=FIELDS, global_quantities=FIELDS,
    Z_HEAT_MAX=35., EVOLVE_DENSITY_LINEARLY=False,
)
qag_profile=runtime.tables.finish_qag_profile() if gpu else None
physics = perf_counter() - begin
z = np.asarray(lc.lightcone_redshifts)
distance = lc.lightcone_distances
distance = distance.to_value("Mpc") if hasattr(distance, "to_value") else np.asarray(distance)
if window_max_z is None:
    window_sel = slice(None)
else:
    window = z <= window_max_z
    nz_idx = np.flatnonzero(window)
    if nz_idx.size == 0:
        window_sel = slice(0, 0)
    elif nz_idx[-1] - nz_idx[0] + 1 == nz_idx.size:
        window_sel = slice(int(nz_idx[0]), int(nz_idx[-1]) + 1)
    else:
        window_sel = window
    z = z[window]
    distance = distance[window]
depth = len(z)
t_write_start = perf_counter()
with h5py.File(destination/"lightcone.h5","w") as out:
    out.attrs.update(point=config["index"], seed=config["seed"], dim=config["user"]["DIM"], hii_dim=config["user"]["HII_DIM"], box_len=config["user"]["BOX_LEN"])
    out.create_dataset("redshift",data=z)
    out.create_dataset("distance_Mpc",data=distance)
    out.create_dataset("node_redshift",data=np.asarray(lc.node_redshifts))
    assert int(lc.random_seed)==config["seed"]
    if gpu:
        all_finite = library.PSArrayAllFinite
        all_finite.argtypes = [C.POINTER(C.c_float), C.c_ulonglong, C.c_int]
        all_finite.restype = C.c_int
        nt_fin = min(8, max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    if gpu and os.environ.get("PS_PARALLEL_LZF") == "1":
        from parallel_lzf import ParallelLZFWriter
        def iter_fields():
            for field in FIELDS:
                arr = np.asarray(lc.lightcones[field],dtype=np.float32)[...,window_sel]
                assert arr.shape == (config["user"]["HII_DIM"],)*2+(depth,)
                if arr.flags.c_contiguous:
                    assert all_finite(arr.ctypes.data_as(C.POINTER(C.c_float)), np.uint64(arr.size), nt_fin) == 1, field
                else:
                    for k in range(0,depth,16):
                        assert np.isfinite(arr[:,:,k:k+16]).all(),field
                yield (out,field,arr,(min(64,arr.shape[0]),min(64,arr.shape[1]),min(16,depth)))
        with ParallelLZFWriter(workers=min(8, os.cpu_count() or 8)) as writer:
            writer.write(iter_fields())
    else:
        for field in FIELDS:
            arr = np.asarray(lc.lightcones[field],dtype=np.float32)[...,window_sel]
            assert arr.shape == (config["user"]["HII_DIM"],)*2+(depth,)
            if gpu and arr.flags.c_contiguous:
                assert all_finite(arr.ctypes.data_as(C.POINTER(C.c_float)), np.uint64(arr.size), nt_fin) == 1, field
            else:
                for k in range(0,depth,16):
                    assert np.isfinite(arr[:,:,k:k+16]).all(),field
            out.create_dataset(field,data=arr,compression="lzf",shuffle=True,
                               chunks=(min(64,arr.shape[0]),min(64,arr.shape[1]),min(16,depth)))
    out.flush()
write_lzf_seconds = perf_counter() - t_write_start
t_fsync_start = perf_counter()
with (destination/"lightcone.h5").open("rb") as f:
    os.fsync(f.fileno())
fsync_seconds = perf_counter() - t_fsync_start
report={"mode":mode, "dim":dim, "hii_dim":hii_dim, "box_len":config["user"]["BOX_LEN"], "seed":config["seed"], "point":config["index"], "max_redshift":max_z, "ic_preparation_seconds":ic_seconds, "physics_seconds":physics, "slice_count":depth,
        "seconds_to_sync":perf_counter()-begin,"fields":FIELDS,"bytes":(destination/"lightcone.h5").stat().st_size}
if gpu:
    report["ic_read_and_upload_seconds"]=ic_seconds
    report["evicted_snapshots"]=runtime.report()["evicted_snapshots"]
    report["native_first_heat_rows"]=runtime.tables.native_first_heat_rows
    report["qag_profile"]=qag_profile
    report["gpu_adaptive_heat_rows"]=runtime.tables.gpu_adaptive_heat_rows
    report["gpu_adaptive_panels"]=runtime.tables.gpu_adaptive_panels
    report["full_cpu_thermal_replays"]=runtime.cpu_replays
    report["native_scale_seconds"]=native_seconds
    report["batch_density_seconds"]=batch_density_seconds
    report["delta_upload_seconds"]=delta_upload_seconds
    report["xerec_replay_seconds"]=xerec_replay_seconds
    report["write_lzf_seconds"]=write_lzf_seconds
    report["fsync_seconds"]=fsync_seconds
    report["native_scales"]=native_scales
    report["screened_noop_scales"]=screened_noop_scales
    report["active_density_replays"]=active_density_replays
    report["native_rec_filter_scales"]=rec_replay_scales
    report["guard_checked_scales"]=checked_scales
    report["guard_replay_scales"]=guard_scales
    report["guard_relative_band"]=float(band)
    rep = runtime.report()
    fn_prof = library.GetPSNativeProfile
    fn_prof.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_ulonglong)]
    fn_prof.restype = None
    secs = (C.c_double * 5)()
    calls = (C.c_ulonglong * 5)()
    fn_prof(secs, calls)
    names = ("ic", "perturbation", "thermal", "ionization", "brightness")
    report["native_profile"] = {n: {"seconds": float(secs[i]), "calls": int(calls[i])} for i, n in enumerate(names)}
    report["runtime_summary"] = {
        "gpu_perturbation_calls": rep["gpu_perturbation_calls"],
        "heat_steps": len(rep["records"]),
        "heat_gpu_seconds": sum(r["seconds"] for r in rep["records"]),
        "heat_prefetch_hits": rep.get("heat_prefetch_hits", 0),
        "heat_prefetch_misses": rep.get("heat_prefetch_misses", 0),
        "heat_prefetch_wait_seconds": rep.get("heat_prefetch_wait_seconds", 0.0),
        "ion_prefetch_hits": ion_prefetch_hits if mode == "gpu-density" else 0,
        "ion_prefetch_misses": ion_prefetch_misses if mode == "gpu-density" else 0,
        "ion_prefetch_wait_seconds": ion_prefetch_wait_seconds if mode == "gpu-density" else 0.0,
        "heat_filter_seconds": sum(rep["filter_step_seconds"]),
        "gpu_conditional_tables": rep["gpu_conditional_tables"],
        "gpu_xray": rep["gpu_xray"],
        "gpu_sigma": rep["gpu_sigma"],
        "gpu_mhr": rep["gpu_mhr"],
        "gpu_global_collapse": rep["gpu_global_collapse"],
        "gpu_brightness_seconds": rep["gpu_brightness_seconds"],
        "ion_phase_counts": rep["ionization"]["phase_counts"],
        "ion_phase_seconds": rep["ionization"]["phase_seconds"],
    }
(destination/"timing.json").write_text(json.dumps(report,indent=2)+"\n")
print("LIGHTCONE_DONE",json.dumps(report),flush=True)
