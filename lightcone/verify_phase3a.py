"""Phase 3A verification: bitwise FFTW unit tests, collapse ULP check, failure dump, and fixed thermal trace."""
import ctypes as C
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import types

import h5py
import numpy as np
import py21cmfast
import py21cmfast.c_21cmfast as native
import py21cmfast.wrapper as W

FIELDS = ("brightness_temp", "xH_box", "x_e_box", "Tk_box", "Ts_box", "dNrec_box")
Z_TARGET = 6.6094489097595215
SLICE123_XY = (54, 56)


class StopEarly(Exception):
    pass


def ulp_diff_scalar(a, b):
    return int(ulp_diff_array(np.asarray([a], dtype=np.float32), np.asarray([b], dtype=np.float32))[0][0])


def ulp_diff_array(a, b):
    ua = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.int64)
    ub = np.asarray(b, dtype=np.float32).view(np.uint32).astype(np.int64)
    ma = np.where((ua & 0x80000000) == 0, ua, -(ua & 0x7FFFFFFF))
    mb = np.where((ub & 0x80000000) == 0, ub, -(ub & 0x7FFFFFFF))
    diff = ma - mb
    adiff = np.abs(diff)
    idx = int(np.argmax(adiff))
    return diff, adiff, idx


def load_config():
    cfg = json.loads((Path(os.environ["PS_GPU_ROOT"]) / "point_config.json").read_text())
    cfg["user"].update(DIM=384, HII_DIM=96, BOX_LEN=375.0)
    return cfg


def load_ic(gpu):
    from py21cmfast.outputs import InitialConditions
    path = os.environ["PS_CONTROL_IC"]
    if gpu:
        fields = ["hires_density", "lowres_density"] + [
            "lowres_v" + axis + suffix for suffix in ("", "_2LPT") for axis in "xyz"
        ]
        init_box = InitialConditions.from_file(path, arrays_to_load=fields)

        def keep_loaded_ic(flag_options, force=False):
            if init_box.user_params.USE_RELATIVE_VELOCITIES or flag_options.USE_HALO_FIELD:
                raise ValueError("Unsupported IC features")

        init_box.prepare_for_spin_temp = keep_loaded_ic
        return init_box
    return InitialConditions.from_file(path)


def run_part_a(out_json):
    import cupy as cp
    from coupled import CoupledRuntime
    import ion_runtime

    cfg = load_config()
    runtime = CoupledRuntime()
    runtime.install(native, cached_ic=True, same_gsl=True)
    init_box = load_ic(True)
    runtime.perturbation.load_cpu_ic(init_box)

    lib = C.CDLL(native.__file__)
    lib.SetNativeFirstHeatHigh.argtypes = [C.c_int]
    lib.SetNativeFirstHeatHigh(1)
    lib.SetPSKeepFFTWPlans.argtypes = [C.c_int]
    lib.SetPSKeepFFTWPlans.restype = None

    args_ptr = C.POINTER(ion_runtime.IonArgs)
    long_ptr = C.POINTER(C.c_long)
    float_ptr = C.POINTER(C.c_float)

    reset = lib.IonReferenceReset
    bounds = lib.IonReferenceBounds
    bounds.argtypes = [args_ptr]; bounds.restype = C.c_int
    mean = lib.IonReferenceMean
    mean.argtypes = [args_ptr, C.POINTER(C.c_double)]; mean.restype = C.c_int
    cells = lib.IonReferenceCells
    cells.argtypes = [args_ptr, long_ptr, C.c_long, float_ptr, float_ptr, C.c_int]; cells.restype = C.c_int
    sched_count = lib.IonReferenceScheduleCount
    sched_count.argtypes = []; sched_count.restype = C.c_int
    batch_density = lib.IonReferenceBatchDensity
    batch_density.argtypes = [args_ptr, float_ptr, C.c_int]; batch_density.restype = C.c_int
    batch_info = lib.IonReferenceBatchScaleInfo
    batch_info.argtypes = [C.c_int, C.c_float, C.c_int, float_ptr, float_ptr]; batch_info.restype = C.c_int
    xerec_box = lib.IonReferenceXeRecBox
    xerec_box.argtypes = [args_ptr, float_ptr, float_ptr]; xerec_box.restype = C.c_int
    test_perturb = lib.TestMaterializePerturbDensity
    test_perturb.argtypes = [C.c_int, C.c_int, float_ptr]; test_perturb.restype = C.c_int

    n_vox = 96 ** 3
    indices = np.arange(n_vox, dtype=np.int64)
    fields_buf = np.empty((3, n_vox), dtype=np.float32)
    collapse_buf = np.empty(n_vox, dtype=np.float32)

    ion = runtime.ionization
    orig_filter = ion.filter
    orig_lookup = ion.lookup
    state = {"phase": None, "z": 100.0}

    probe_scales = {0, 7, 15, 24, 31}
    uncached_deltas = []
    uncached_lo_hi = []
    uncached_xerec = {}
    saved_args = []
    collapse_records = []
    results = {}

    scale_counter = {"idx": 0, "uncached_bounds_s": 0.0, "uncached_mean_s": 0.0, "uncached_cells_s": 0.0}

    def w_filter(a):
        if state["phase"] != "ion" or abs(a.z - Z_TARGET) >= 1e-4:
            orig_filter(a)
            return
        if scale_counter["idx"] == 0:
            lib.SetPSKeepFFTWPlans(0)
            reset()
        t0 = perf_counter()
        assert bounds(C.byref(a)) == 0
        scale_counter["uncached_bounds_s"] += perf_counter() - t0
        bound_lo, bound_hi = float(a.lo), float(a.hi)
        orig_filter(a)
        a.lo, a.hi = bound_lo, bound_hi
        saved_args.append(ion_runtime.IonArgs.from_buffer_copy(C.string_at(C.addressof(a), C.sizeof(ion_runtime.IonArgs))))
        uncached_lo_hi.append((min(0.0, bound_lo), max(0.0, bound_hi)))

    def w_lookup(a):
        if state["phase"] != "ion" or abs(a.z - Z_TARGET) >= 1e-4:
            orig_lookup(a)
            return
        s = scale_counter["idx"]
        # Copy the GPU-computed conditional table rows into host buffers so IonReferenceMean evaluates the exact same table.
        key = tuple(C.cast(ptr, C.c_void_p).value for ptr in (a.low, a.high))
        low_gpu, high_gpu = runtime.tables.device_rows[key]
        low_np = np.ascontiguousarray(low_gpu.get(), dtype=np.float32)
        high_np = np.ascontiguousarray(high_gpu.get(), dtype=np.float32)
        C.memmove(key[0], low_np.ctypes.data, int(a.nl) * 4)
        C.memmove(key[1], high_np.ctypes.data, int(a.nh) * 4)
        t0 = perf_counter()
        total = C.c_double()
        assert mean(C.byref(a), C.byref(total)) == 0
        assert np.isfinite(total.value)
        scale_counter["uncached_mean_s"] += perf_counter() - t0

        flag = 1 if s in probe_scales else 2
        t0 = perf_counter()
        assert cells(
            C.byref(a),
            indices.ctypes.data_as(long_ptr),
            n_vox,
            fields_buf.ctypes.data_as(float_ptr),
            collapse_buf.ctypes.data_as(float_ptr),
            flag,
        ) == 0
        scale_counter["uncached_cells_s"] += perf_counter() - t0

        ref_delta = fields_buf[0].copy()
        ref_fc = collapse_buf.copy()
        ref_total = float(total.value)
        uncached_deltas.append(ref_delta)
        if s in probe_scales:
            uncached_xerec[s] = (fields_buf[1].copy(), fields_buf[2].copy())

        ion.delta = cp.asarray(ref_delta)
        orig_lookup(a)
        gpu_fc = ion.fc.get()
        gpu_total = float(ion.collapse_sum.get())

        _, adiff_u, worst_i = ulp_diff_array(gpu_fc, ref_fc)
        abs_diff = np.abs(gpu_fc.astype(np.float64) - ref_fc.astype(np.float64))
        rel_diff = abs_diff / np.maximum(np.abs(ref_fc.astype(np.float64)), 1e-30)
        collapse_records.append(
            {
                "scale": s,
                "R": float(a.radius),
                "max_ulp": int(adiff_u.ravel()[worst_i]),
                "nonzero_ulp_voxels": int(np.count_nonzero(adiff_u)),
                "max_abs": float(np.max(abs_diff)),
                "max_rel": float(np.max(rel_diff)),
                "sum_rel_diff": abs(gpu_total - ref_total) / max(abs(ref_total), 1e-30),
                "worst_cpu": float(ref_fc.ravel()[worst_i]),
                "worst_gpu": float(gpu_fc.ravel()[worst_i]),
            }
        )

        if a.last:
            # --- Run A1, A2, A3 while user_params_ufunc and Z_TARGET arrays are live ---
            # A2: IonReferenceBatchDensity vs uncached
            lib.SetPSKeepFFTWPlans(1)
            n_sched = sched_count()
            assert n_sched == len(uncached_deltas), (n_sched, len(uncached_deltas))
            batch_buf = np.empty((n_sched, n_vox), dtype=np.float32)

            reset()
            t0 = perf_counter()
            assert batch_density(C.byref(a), batch_buf.ctypes.data_as(float_ptr), n_sched) == 0
            a2_cold_plan_s = perf_counter() - t0

            reset()
            t0 = perf_counter()
            assert batch_density(C.byref(a), batch_buf.ctypes.data_as(float_ptr), n_sched) == 0
            a2_warm_s = perf_counter() - t0

            a2_max_abs = 0.0
            a2_mismatch_scales = 0
            a2_info_mismatches = 0
            for j in range(n_sched):
                lo_v, hi_v = C.c_float(), C.c_float()
                if batch_info(j, saved_args[j].radius, saved_args[j].filtered, C.byref(lo_v), C.byref(hi_v)) != 0:
                    a2_info_mismatches += 1
                else:
                    if (min(0.0, float(lo_v.value)), max(0.0, float(hi_v.value))) != uncached_lo_hi[j]:
                        a2_info_mismatches += 1
                if not np.array_equal(batch_buf[j], uncached_deltas[j]):
                    a2_mismatch_scales += 1
                    dmax = float(np.max(np.abs(batch_buf[j].astype(np.float64) - uncached_deltas[j].astype(np.float64))))
                    a2_max_abs = max(a2_max_abs, dmax)

            # A3: IonReferenceXeRecBox vs uncached IonReferenceCells(flag=1) on probe scales
            xe_out = np.empty(n_vox, dtype=np.float32)
            rec_out = np.empty(n_vox, dtype=np.float32)
            assert xerec_box(C.byref(saved_args[0]), xe_out.ctypes.data_as(float_ptr), rec_out.ctypes.data_as(float_ptr)) == 0
            a3_xe_max_abs = 0.0
            a3_rec_max_abs = 0.0
            t0 = perf_counter()
            for j in sorted(probe_scales):
                assert xerec_box(C.byref(saved_args[j]), xe_out.ctypes.data_as(float_ptr), rec_out.ctypes.data_as(float_ptr)) == 0
                ref_xe, ref_rec = uncached_xerec[j]
                if not np.array_equal(xe_out, ref_xe):
                    a3_xe_max_abs = max(a3_xe_max_abs, float(np.max(np.abs(xe_out.astype(np.float64) - ref_xe.astype(np.float64)))))
                if not np.array_equal(rec_out, ref_rec):
                    a3_rec_max_abs = max(a3_rec_max_abs, float(np.max(np.abs(rec_out.astype(np.float64) - ref_rec.astype(np.float64)))))
            a3_s = perf_counter() - t0

            # A1: MaterializePerturbDensity cached vs uncached
            src_dens = np.ctypeslib.as_array(a.density, shape=(n_vox,)).copy()
            d_unc = src_dens.copy()
            d_cac = src_dens.copy()
            lib.SetPSKeepFFTWPlans(0)
            unc_times = []
            for _ in range(3):
                tmp = src_dens.copy()
                t0 = perf_counter()
                assert test_perturb(0, 8, tmp.ctypes.data_as(float_ptr)) == 0
                unc_times.append(perf_counter() - t0)
                d_unc = tmp

            lib.SetPSKeepFFTWPlans(1)
            # warmup plan creation
            warm = src_dens.copy()
            assert test_perturb(1, 8, warm.ctypes.data_as(float_ptr)) == 0
            cac_times = []
            for _ in range(10):
                tmp = src_dens.copy()
                t0 = perf_counter()
                assert test_perturb(1, 8, tmp.ctypes.data_as(float_ptr)) == 0
                cac_times.append(perf_counter() - t0)
                d_cac = tmp
            a1_max_abs = 0.0 if np.array_equal(d_cac, d_unc) else float(np.max(np.abs(d_cac.astype(np.float64) - d_unc.astype(np.float64))))
            lib.SetPSKeepFFTWPlans(0)

            unc_total_density_s = scale_counter["uncached_bounds_s"] + scale_counter["uncached_cells_s"]
            results.update(
                {
                    "z_target": Z_TARGET,
                    "A1_perturb_materialize": {
                        "max_abs": a1_max_abs,
                        "bitwise_equal": bool(a1_max_abs == 0.0),
                        "uncached_mean_ms": 1000.0 * float(np.mean(unc_times)),
                        "cached_mean_ms": 1000.0 * float(np.mean(cac_times)),
                        "speedup": float(np.mean(unc_times) / np.mean(cac_times)),
                    },
                    "A2_batch_density": {
                        "scales": n_sched,
                        "max_abs": a2_max_abs,
                        "bitwise_equal": bool(a2_max_abs == 0.0 and a2_mismatch_scales == 0),
                        "mismatch_scales": a2_mismatch_scales,
                        "info_mismatches": a2_info_mismatches,
                        "batch_cold_plan_ms": 1000.0 * a2_cold_plan_s,
                        "batch_warm_ms": 1000.0 * a2_warm_s,
                        "uncached_bounds_plus_cells_ms": 1000.0 * unc_total_density_s,
                        "uncached_bounds_ms": 1000.0 * scale_counter["uncached_bounds_s"],
                        "uncached_cells_ms": 1000.0 * scale_counter["uncached_cells_s"],
                        "uncached_mean_ms": 1000.0 * scale_counter["uncached_mean_s"],
                        "speedup_vs_bounds_plus_cells": float(unc_total_density_s / a2_warm_s),
                    },
                    "A3_xerec_box": {
                        "scales_checked": len(probe_scales),
                        "xe_max_abs": a3_xe_max_abs,
                        "rec_max_abs": a3_rec_max_abs,
                        "bitwise_equal": bool(a3_xe_max_abs == 0.0 and a3_rec_max_abs == 0.0),
                        "five_scales_ms": 1000.0 * a3_s,
                    },
                    "A4_gpu_collapse": {
                        "glibc_math_fma": bool(ion.glibc_math_fma),
                        "scales": len(collapse_records),
                        "max_ulp": max(r["max_ulp"] for r in collapse_records),
                        "total_nonzero_ulp_voxels": sum(r["nonzero_ulp_voxels"] for r in collapse_records),
                        "max_abs": max(r["max_abs"] for r in collapse_records),
                        "max_rel": max(r["max_rel"] for r in collapse_records),
                        "max_sum_rel_diff": max(r["sum_rel_diff"] for r in collapse_records),
                        "worst_scale": max(collapse_records, key=lambda r: (r["max_ulp"], r["max_abs"])),
                    },
                }
            )
        scale_counter["idx"] = s + 1

    ion.filter = w_filter
    ion.lookup = w_lookup

    orig_st = W.spin_temperature
    orig_ion = W.ionize_box

    def wrap_st(**kwargs):
        state["phase"] = "st"
        state["z"] = float(kwargs["redshift"])
        return orig_st(**kwargs)

    def wrap_ion(**kwargs):
        state["phase"] = "ion"
        state["z"] = float(kwargs["redshift"])
        res = orig_ion(**kwargs)
        if state["z"] <= Z_TARGET + 1e-4:
            raise StopEarly()
        return res

    W.spin_temperature = wrap_st
    W.ionize_box = wrap_ion
    try:
        py21cmfast.run_lightcone(
            redshift=5.0,
            max_redshift=14.0,
            user_params=cfg["user"],
            cosmo_params=cfg["cosmo"],
            astro_params=cfg["astro"],
            flag_options=cfg["flags"],
            init_box=init_box,
            random_seed=cfg["seed"],
            direc=os.environ["PARTA_SCRATCH"],
            regenerate=True,
            write=False,
            cleanup=True,
            lightcone_quantities=FIELDS,
            global_quantities=FIELDS,
            Z_HEAT_MAX=35.0,
            EVOLVE_DENSITY_LINEARLY=False,
        )
    except StopEarly:
        pass
    finally:
        W.spin_temperature = orig_st
        W.ionize_box = orig_ion

    Path(out_json).write_text(json.dumps(results, indent=2) + "\n")
    print("PART_A_PHASE3A_DONE", json.dumps(results), flush=True)

    assert results["A1_perturb_materialize"]["bitwise_equal"]
    assert results["A2_batch_density"]["bitwise_equal"]
    assert results["A2_batch_density"]["info_mismatches"] == 0
    assert results["A3_xerec_box"]["bitwise_equal"]
    assert results["A4_gpu_collapse"]["max_ulp"] == 0


def check_repeat(path_a, path_b, out_json):
    eq = {}
    max_diff = {}
    with h5py.File(path_a, "r") as ha, h5py.File(path_b, "r") as hb:
        for f in FIELDS:
            same = True
            md = 0.0
            n = ha[f].shape[2]
            for k in range(0, n, 16):
                aa = ha[f][:, :, k:k + 16]
                bb = hb[f][:, :, k:k + 16]
                assert np.isfinite(aa).all() and np.isfinite(bb).all(), (f, k)
                if not np.array_equal(aa, bb):
                    same = False
                    md = max(md, float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64)))))
            eq[f] = same
            max_diff[f] = md
    res = {"repeat_equal": eq, "max_diff": max_diff, "all_equal": all(eq.values())}
    Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    print("REPEAT_CHECK", json.dumps(res), flush=True)


def dump_failures(cpu_h5, gpu_h5, out_json):
    atol, rtol = 1e-8, 1e-4
    failures = []
    with h5py.File(cpu_h5, "r") as cpu, h5py.File(gpu_h5, "r") as gpu:
        zs = cpu["redshift"][:]
        nodes = np.sort(cpu["node_redshift"][:])
        for start in range(0, len(zs), 16):
            ref = {f: cpu[f][:, :, start:start + 16] for f in FIELDS}
            tst = {f: gpu[f][:, :, start:start + 16] for f in FIELDS}
            for f in FIELDS:
                c64 = ref[f].astype(np.float64)
                g64 = tst[f].astype(np.float64)
                bad = np.abs(g64 - c64) > atol + rtol * np.abs(c64)
                if np.any(bad):
                    for ii, jj, kk in zip(*np.nonzero(bad)):
                        i, j, dk = int(ii), int(jj), int(kk)
                        sl = start + dk
                        right = int(np.searchsorted(nodes, zs[sl]))
                        failures.append(
                            {
                                "field": f,
                                "pixel": [i, j],
                                "slice": sl,
                                "z": float(zs[sl]),
                                "node_bounds": nodes[max(0, right - 1):right + 1].tolist(),
                                "cpu": float(ref[f][i, j, dk]),
                                "gpu": float(tst[f][i, j, dk]),
                                "abs_diff": abs(float(g64[i, j, dk] - c64[i, j, dk])),
                                "ulp_diff": ulp_diff_scalar(tst[f][i, j, dk], ref[f][i, j, dk]),
                                "all_fields_cpu": {fld: float(ref[fld][i, j, dk]) for fld in FIELDS},
                                "all_fields_gpu": {fld: float(tst[fld][i, j, dk]) for fld in FIELDS},
                            }
                        )
    out = {"count": len(failures), "failures": failures}
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print("FAILING_CELLS", json.dumps({"count": len(failures), "items": [(x["field"], x["pixel"], x["slice"], x["ulp_diff"]) for x in failures[:20]]}), flush=True)


def diag_thermal(cpu_h5, out_json):
    from coupled import CoupledRuntime

    cfg = load_config()
    with h5py.File(cpu_h5, "r") as cpu:
        cpu_all_123 = {f: float(cpu[f][54, 56, 123]) for f in FIELDS}
        cpu_all_124 = {f: float(cpu[f][54, 56, 124]) for f in FIELDS}
        zs_h5 = cpu["redshift"][:]
        z_123 = float(zs_h5[123])
        z_124 = float(zs_h5[124])

    def capture_pass(use_gpu):
        init_box = load_ic(use_gpu)
        if use_gpu:
            runtime = CoupledRuntime()
            runtime.install(native, cached_ic=True, same_gsl=True)
            runtime.perturbation.load_cpu_ic(init_box)
            lib = C.CDLL(native.__file__)
            lib.SetNativeFirstHeatHigh.argtypes = [C.c_int]
            lib.SetNativeFirstHeatHigh(1)
        steps = []
        orig_st = W.spin_temperature
        orig_ion = W.ionize_box
        orig_bt = W.brightness_temperature

        def wrap_st(**kwargs):
            st2 = orig_st(**kwargs)
            z = float(kwargs["redshift"])
            steps.append(
                (
                    z,
                    st2.x_e_box.copy(),
                    st2.Tk_box.copy(),
                    st2.Ts_box.copy(),
                )
            )
            return st2

        astro_obj = py21cmfast.AstroParams(cfg["astro"])
        flag_obj = py21cmfast.FlagOptions(cfg["flags"])

        def dummy_ion(**kwargs):
            z = float(kwargs["redshift"])
            if z < 5.88:
                raise StopEarly()
            pf_obj = kwargs["perturbed_field"]
            return types.SimpleNamespace(
                redshift=z,
                _inputs=(),
                _get_box_structures=lambda: ("xH_box", "dNrec_box"),
                user_params=pf_obj.user_params,
                cosmo_params=pf_obj.cosmo_params,
                astro_params=astro_obj,
                flag_options=flag_obj,
                random_seed=pf_obj.random_seed,
                xH_box=np.zeros((96, 96, 96), dtype=np.float32),
                dNrec_box=np.zeros((96, 96, 96), dtype=np.float32),
                log10_Mturnover_ave=0.0,
                log10_Mturnover_MINI_ave=0.0,
                filename="dummy_ion",
            )

        def dummy_bt(**kwargs):
            ib_obj = kwargs["ionized_box"]
            z = float(kwargs["redshift"]) if "redshift" in kwargs else float(ib_obj.redshift)
            return types.SimpleNamespace(
                redshift=z,
                _inputs=(),
                _get_box_structures=lambda: ("brightness_temp",),
                user_params=ib_obj.user_params,
                cosmo_params=ib_obj.cosmo_params,
                astro_params=ib_obj.astro_params,
                flag_options=ib_obj.flag_options,
                random_seed=ib_obj.random_seed,
                brightness_temp=np.zeros((96, 96, 96), dtype=np.float32),
                filename="dummy_bt",
            )

        W.spin_temperature = wrap_st
        W.ionize_box = dummy_ion
        W.brightness_temperature = dummy_bt
        try:
            py21cmfast.run_lightcone(
                redshift=5.0,
                max_redshift=14.0,
                user_params=cfg["user"],
                cosmo_params=cfg["cosmo"],
                astro_params=cfg["astro"],
                flag_options=cfg["flags"],
                init_box=init_box,
                random_seed=cfg["seed"],
                direc=os.environ["PARTC_SCRATCH"] + ("/gpu" if use_gpu else "/cpu"),
                regenerate=True,
                write=False,
                cleanup=True,
                lightcone_quantities=FIELDS,
                global_quantities=FIELDS,
                Z_HEAT_MAX=35.0,
                EVOLVE_DENSITY_LINEARLY=False,
            )
        except StopEarly:
            pass
        finally:
            W.spin_temperature = orig_st
            W.ionize_box = orig_ion
            W.brightness_temperature = orig_bt
        return steps

    os.makedirs(os.environ["PARTC_SCRATCH"] + "/cpu", exist_ok=True)
    os.makedirs(os.environ["PARTC_SCRATCH"] + "/gpu", exist_ok=True)
    # CRITICAL: run pure unhooked CPU pass FIRST before CoupledRuntime installs any global C hooks!
    cpu_steps = capture_pass(False)
    gpu_steps = capture_pass(True)
    assert len(gpu_steps) == len(cpu_steps)

    sx, sy = SLICE123_XY
    iz123 = 123 % 96  # 27
    iz124 = 124 % 96  # 28
    first_box_drift = None
    first_iz27_drift = None
    first_iz28_drift = None
    summary_steps = []

    def pixel_entry(g_arr, c_arr, ix, iy, iz):
        gv = g_arr[ix, iy, iz]
        cv = c_arr[ix, iy, iz]
        return {"cpu": float(cv), "gpu": float(gv), "diff": float(gv) - float(cv), "ulp": ulp_diff_scalar(gv, cv)}

    for i, ((zc, cxe, ctk, cts), (zg, gxe, gtk, gts)) in enumerate(zip(cpu_steps, gpu_steps)):
        assert abs(zg - zc) < 1e-6, (i, zg, zc)
        box_stats = {}
        any_box = False
        for name, ga, ca in (("xe", gxe, cxe), ("Tk", gtk, ctk), ("Ts", gts, cts)):
            d = np.abs(ga.astype(np.float64) - ca.astype(np.float64))
            idx = int(np.argmax(d))
            coord = list(map(int, np.unravel_index(idx, d.shape)))
            mval = float(d.ravel()[idx])
            if mval != 0.0:
                any_box = True
            box_stats[name] = {
                "max_abs": mval,
                "nonzero": int(np.count_nonzero(d)),
                "argmax": coord,
                "cpu": float(ca[coord[0], coord[1], coord[2]]),
                "gpu": float(ga[coord[0], coord[1], coord[2]]),
                "ulp": ulp_diff_scalar(ga[coord[0], coord[1], coord[2]], ca[coord[0], coord[1], coord[2]]),
            }
        entry = {
            "step": i,
            "z": zc,
            "box_max": box_stats,
            "iz27": {
                "xe": pixel_entry(gxe, cxe, sx, sy, iz123),
                "Tk": pixel_entry(gtk, ctk, sx, sy, iz123),
                "Ts": pixel_entry(gts, cts, sx, sy, iz123),
            },
            "iz28_54_56": {
                "xe": pixel_entry(gxe, cxe, sx, sy, iz124),
                "Tk": pixel_entry(gtk, ctk, sx, sy, iz124),
                "Ts": pixel_entry(gts, cts, sx, sy, iz124),
            },
            "iz28_53_56": {
                "xe": pixel_entry(gxe, cxe, 53, 56, iz124),
                "Tk": pixel_entry(gtk, ctk, 53, 56, iz124),
                "Ts": pixel_entry(gts, cts, 53, 56, iz124),
            },
        }
        summary_steps.append(entry)
        if first_box_drift is None and any_box:
            first_box_drift = entry
        if first_iz27_drift is None and (
            entry["iz27"]["Tk"]["diff"] != 0.0 or entry["iz27"]["Ts"]["diff"] != 0.0 or entry["iz27"]["xe"]["diff"] != 0.0
        ):
            first_iz27_drift = entry
        if first_iz28_drift is None and (
            entry["iz28_53_56"]["Tk"]["diff"] != 0.0
            or entry["iz28_53_56"]["Ts"]["diff"] != 0.0
            or entry["iz28_53_56"]["xe"]["diff"] != 0.0
        ):
            first_iz28_drift = entry

    res = {
        "total_steps": len(summary_steps),
        "first_z": summary_steps[0]["z"],
        "last_z": summary_steps[-1]["z"],
        "z_123": z_123,
        "z_124": z_124,
        "cpu_h5_slice123": cpu_all_123,
        "cpu_h5_slice124": cpu_all_124,
        "first_box_drift": first_box_drift,
        "first_iz27_drift": first_iz27_drift,
        "first_iz28_drift": first_iz28_drift,
        "all_steps": summary_steps,
    }
    Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    print(
        "PART_C_THERMAL_DONE",
        json.dumps(
            {
                "steps": len(summary_steps),
                "first_z": res["first_z"],
                "first_box_drift": first_box_drift,
                "first_iz27_drift": first_iz27_drift,
                "first_iz28_drift": first_iz28_drift,
            }
        ),
        flush=True,
    )


def main():
    cmd = sys.argv[1]
    if cmd == "partA":
        run_part_a(sys.argv[2])
    elif cmd == "repeat":
        check_repeat(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "dump_failures":
        dump_failures(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "diag_thermal":
        diag_thermal(sys.argv[2], sys.argv[3])
    else:
        raise ValueError(cmd)


if __name__ == "__main__":
    main()
