#!/bin/bash
#SBATCH --job-name=ps-p3c-w4
#SBATCH --partition=a
#SBATCH --nodelist=a6
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu=1
#SBATCH --mem=36G
#SBATCH --time=00:20:00
set -euo pipefail
stage=$1
root=$(dirname "$stage")
out="$stage/results-$SLURM_JOB_ID"
scratch=$(mktemp -d /var/tmp/ps-phase3c-w4.XXXXXXXX)
mkdir -p "$out" "$scratch/build" "$scratch/w4-unit" "$scratch/w4-cupy" "$scratch/pf" "$scratch/cupy-w4"
trap 'rm -rf "$scratch"' EXIT

export PS_GPU_ROOT="$root" LD_LIBRARY_PATH="$root/lib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PS_FFTW_PERTURB_DENSITY=1 PS_FFTW_HEAT_FILTER=1 PS_GSL_EXACT_POWER=1
export PS_CONDITIONAL_MODE=3 PS_BATCH_FILTER=1
export PS_XRAY_NATIVE_MODE=4 PS_NATIVE_SIGMA=1 PS_NATIVE_MHR=1 PS_NATIVE_COLLAPSE=1 PS_NATIVE_XRAY=0
export PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0
export PS_CONDITIONAL_FULL_HIGH=0 PS_CONDITIONAL_NAN_SENTINEL=1 PS_GUARD_EPS_MULT=32
export PS_LAZY_COEVAL_IC=1 PS_PARALLEL_LZF=1 PS_VERBOSE_HEAT=0
export PS_REUSE_PREV_PERTURB=1 PS_FAST_LIGHTCONE=1 PS_CIC_PERTILE=1
export PS_OMP_MHR=1 PS_CACHE_INIT_PS=1 PS_ASYNC_FFTW_PREFETCH=1
export PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0
export CUDA_PATH="$root/cuda118_experiment_headers"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export XDG_CONFIG_HOME="$scratch/config"
export CUPY_CACHE_DIR="$scratch/cupy-w4"

python="$root/env310/bin/python"
"$python" "$stage/build.py" "$root" "$scratch/build" > "$out/build.log" 2>&1
export PYTHONPATH="$scratch/build/package:$stage"

prior="$(cat "$root/lightcone_384_96_375Mpc_holdout_20261120/results-488250/scratch-path.txt")"
ln -s "$prior/cpu" "$scratch/cpu"
export PS_CONTROL_IC="$(printf '%s' "$prior/ic/"InitialConditions*.h5)"
export PS_GK61_RULE="$scratch/gk61.json"
"$python" "$stage/gk61_rule.py" "$PS_GK61_RULE"

# --- W4 Unit Suite ---
CUPY_CACHE_DIR="$scratch/w4-cupy" W4_OUT="$out/w4_units.json" W4_SCRATCH="$scratch/w4-unit" PF_DIR="$scratch/pf" "$python" - <<'PY'
import ctypes as C
import json
import os
import time
import cupy as cp
import numpy as np
import py21cmfast
import py21cmfast.c_21cmfast as native
import py21cmfast.wrapper as W
from coupled import CoupledRuntime
import ion_runtime
from verify_phase3a import Z_TARGET, load_config, load_ic, ulp_diff_array

class StopEarly(Exception):
    pass

cfg = load_config()
runtime = CoupledRuntime()
runtime.install(native, cached_ic=True, same_gsl=True)
init_box = load_ic(True)
runtime.perturbation.load_cpu_ic(init_box)

lib = C.CDLL(native.__file__)
F = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)
I = C.POINTER(C.c_int)
lib.SetPSKeepFFTWPlans.argtypes = [C.c_int]
lib.SetPSKeepFFTWPlans.restype = None
lib.dicke.argtypes = [C.c_double]
lib.dicke.restype = C.c_double

# 1. W4_cic_pertile_v2
p_cap = {}
orig_p_compute = runtime.perturbation.compute
def wrap_p_compute(z, fine, coarse, dk, first, second, initial_growth, dlog_growth_dt, inputs, out_density, out_velocity):
    if not p_cap:
        p_cap.update(fine=int(fine), coarse=int(coarse), first=float(first), second=float(second), initial_growth=float(initial_growth))
    return orig_p_compute(z, fine, coarse, dk, first, second, initial_growth, dlog_growth_dt, inputs, out_density, out_velocity)
runtime.perturbation.compute = wrap_p_compute

pf = py21cmfast.perturb_field(
    redshift=Z_TARGET,
    init_boxes=init_box,
    user_params=cfg["user"],
    cosmo_params=cfg["cosmo"],
    direc=os.environ["PF_DIR"],
    regenerate=True,
    write=False,
)
runtime.perturbation.compute = orig_p_compute
density = np.ascontiguousarray(pf.density, dtype=np.float32)

p = runtime.perturbation
fine, coarse = p_cap["fine"], p_cap["coarse"]
n_c = coarse ** 3
ratio = fine // coarse
mass_orig = cp.empty(n_c, dtype=cp.int64)
mass_pertile = cp.empty(n_c, dtype=cp.int64)
mass_pertile_2 = cp.empty(n_c, dtype=cp.int64)
con_orig = cp.empty((coarse,) * 3, dtype=cp.float32)
con_pertile = cp.empty((coarse,) * 3, dtype=cp.float32)
scale_args = (p.displacements, np.int64(n_c), np.float32(p_cap["first"]), np.float32(p_cap["second"]))
ig = np.float32(p_cap["initial_growth"])
pertile_block = 64
pertile_grid = ((n_c + pertile_block - 1) // pertile_block,)

p.launch(p.scaler, 6 * n_c, (*scale_args, np.int32(0)))
mass_orig.fill(0)
p.launch(p.depositer, fine ** 3, (p.high, p.displacements, mass_orig, np.int32(fine), np.int32(coarse), ig))
p.launch(p.normalizer, n_c, (mass_orig, con_orig, np.int64(n_c), np.float32(ratio ** 3)))

mass_pertile.fill(0)
p.depositer_pertile(pertile_grid, (pertile_block,), (p.high, p.displacements, mass_pertile, np.int32(fine), np.int32(coarse), ig))
mass_pertile_2.fill(0)
p.depositer_pertile(pertile_grid, (pertile_block,), (p.high, p.displacements, mass_pertile_2, np.int32(fine), np.int32(coarse), ig))
p.launch(p.normalizer, n_c, (mass_pertile, con_pertile, np.int64(n_c), np.float32(ratio ** 3)))
cp.cuda.Stream.null.synchronize()
mass_repeat_diffs = int(cp.count_nonzero(mass_pertile != mass_pertile_2).get())

orig_times = []
for _ in range(5):
    mass_orig.fill(0)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    p.launch(p.depositer, fine ** 3, (p.high, p.displacements, mass_orig, np.int32(fine), np.int32(coarse), ig))
    cp.cuda.Stream.null.synchronize()
    orig_times.append((time.perf_counter() - t0) * 1e3)

pertile_times = []
for _ in range(5):
    mass_pertile.fill(0)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    p.depositer_pertile(pertile_grid, (pertile_block,), (p.high, p.displacements, mass_pertile, np.int32(fine), np.int32(coarse), ig))
    cp.cuda.Stream.null.synchronize()
    pertile_times.append((time.perf_counter() - t0) * 1e3)

p.launch(p.scaler, 6 * n_c, (*scale_args, np.int32(1)))
cp.cuda.Stream.null.synchronize()

h_co = con_orig.get()
h_cp = con_pertile.get()
cic_max_abs = float(np.max(np.abs(h_co.astype(np.float64) - h_cp.astype(np.float64))))
cic_bitwise = bool(np.array_equal(h_co.view(np.uint32), h_cp.view(np.uint32)))
_, cic_ulps, cic_worst = ulp_diff_array(h_cp, h_co)
orig_ms = float(np.median(orig_times))
pertile_ms = float(np.median(pertile_times))
w4_cic_report = {
    "max_abs": cic_max_abs,
    "bitwise_equal": cic_bitwise,
    "mass_repeat_diffs": mass_repeat_diffs,
    "max_ulp": int(cic_ulps.ravel()[cic_worst]),
    "orig_ms": orig_ms,
    "pertile_ms": pertile_ms,
    "speedup": float(orig_ms / pertile_ms),
}

# 2. W4_init_mhr_omp
lib.SetGPUMHRTable(None)
lib.init_MHR.argtypes = []; lib.init_MHR.restype = None
lib.free_MHR.argtypes = []; lib.free_MHR.restype = None
export_knots = lib.IonRecombinationKnots
export_knots.argtypes = [D, D, D]; export_knots.restype = None

def run_mhr(flag):
    os.environ["PS_OMP_MHR"] = flag
    t0 = time.perf_counter()
    lib.init_MHR()
    ms = (time.perf_counter() - t0) * 1e3
    arrs = [np.empty(shape, dtype=np.float64) for shape in (250, (300, 250), (300, 250))]
    export_knots(*(a.ctypes.data_as(D) for a in arrs))
    lib.free_MHR()
    return ms, arrs

mhr_seq_ms, k_seq = run_mhr("0")
mhr_omp_ms, k_omp = run_mhr("1")
os.environ["PS_OMP_MHR"] = "1"
mhr_max_abs = float(max(np.max(np.abs(a - b)) for a, b in zip(k_seq, k_omp)))
mhr_bitwise = bool(all(np.array_equal(a.view(np.uint64), b.view(np.uint64)) for a, b in zip(k_seq, k_omp)))
w4_mhr_report = {
    "max_abs": mhr_max_abs,
    "bitwise_equal": mhr_bitwise,
    "seq_ms": mhr_seq_ms,
    "omp_ms": mhr_omp_ms,
    "speedup": float(mhr_seq_ms / mhr_omp_ms),
}

# 3. W4_heat_preplanned
side = 96
vox = side ** 3
count = 40
dk = 2.0 * np.pi / 375.0
R = 0.620350491 * 375.0 / float(side)
R_max = float(py21cmfast.global_params.R_XLy_MAX)
R_factor = (R_max / R) ** (1.0 / float(count))
radii = np.empty(count, dtype=np.float32)
for j in range(count):
    radii[j] = np.float32(R)
    R *= R_factor
inverse_growth = np.float32(1.0 / float(lib.dicke(float(Z_TARGET))))

uncached_fn = lib.HeatReferenceUncachedFilter
uncached_fn.argtypes = [C.c_int, C.c_int, C.c_float, F, F, F, F, F]; uncached_fn.restype = C.c_int
batch_fn = lib.HeatReferenceBatchDensity
batch_fn.argtypes = [C.c_int, C.c_int, C.c_double, C.c_float, F, F, F, F, F]; batch_fn.restype = C.c_int
pre_fn = lib.HeatReferenceBatchDensityPreplanned
pre_fn.argtypes = [C.c_int, C.c_int, C.c_double, C.c_float, F, F, F, F, F, C.c_int]; pre_fn.restype = C.c_int

ref_batch = np.empty((count, vox), dtype=np.float32)
ref_min = np.empty(count, dtype=np.float32)
ref_max = np.empty(count, dtype=np.float32)
lib.SetPSKeepFFTWPlans(0)
assert uncached_fn(side, count, inverse_growth, density.ctypes.data_as(F), radii.ctypes.data_as(F),
                   ref_batch.ctypes.data_as(F), ref_min.ctypes.data_as(F), ref_max.ctypes.data_as(F)) == 0

pinned_mem = cp.cuda.alloc_pinned_memory(count * vox * 4)
gpu_batch = np.frombuffer(pinned_mem, dtype=np.float32, count=count * vox).reshape(count, vox)
gpu_min = np.empty(count, dtype=np.float32)
gpu_max = np.empty(count, dtype=np.float32)
pre_batch = np.empty((count, vox), dtype=np.float32)
pre_min = np.empty(count, dtype=np.float32)
pre_max = np.empty(count, dtype=np.float32)

lib.SetPSKeepFFTWPlans(1)
assert batch_fn(side, count, dk, inverse_growth, density.ctypes.data_as(F), radii.ctypes.data_as(F),
                gpu_batch.ctypes.data_as(F), gpu_min.ctypes.data_as(F), gpu_max.ctypes.data_as(F)) == 0
assert pre_fn(side, count, dk, inverse_growth, density.ctypes.data_as(F), radii.ctypes.data_as(F),
              pre_batch.ctypes.data_as(F), pre_min.ctypes.data_as(F), pre_max.ctypes.data_as(F), 7) == 0

h_max_abs = float(np.max(np.abs(ref_batch.astype(np.float64) - pre_batch.astype(np.float64))))
h_bitwise = bool(np.array_equal(ref_batch.view(np.uint32), pre_batch.view(np.uint32)))
h_min_abs = float(np.max(np.abs(ref_min.astype(np.float64) - pre_min.astype(np.float64))))
h_max_abs_m = float(np.max(np.abs(ref_max.astype(np.float64) - pre_max.astype(np.float64))))

t_8t = []
for _ in range(5):
    t0 = time.perf_counter()
    assert batch_fn(side, count, dk, inverse_growth, density.ctypes.data_as(F), radii.ctypes.data_as(F),
                    gpu_batch.ctypes.data_as(F), gpu_min.ctypes.data_as(F), gpu_max.ctypes.data_as(F)) == 0
    t_8t.append((time.perf_counter() - t0) * 1e3)
t_7t = []
for _ in range(5):
    t0 = time.perf_counter()
    assert pre_fn(side, count, dk, inverse_growth, density.ctypes.data_as(F), radii.ctypes.data_as(F),
                  pre_batch.ctypes.data_as(F), pre_min.ctypes.data_as(F), pre_max.ctypes.data_as(F), 7) == 0
    t_7t.append((time.perf_counter() - t0) * 1e3)

w4_heat_report = {
    "max_abs": h_max_abs,
    "bitwise_equal": h_bitwise,
    "minima_max_abs": h_min_abs,
    "maxima_max_abs": h_max_abs_m,
    "warm_ms_8t": float(np.median(t_8t)),
    "warm_ms_7t": float(np.median(t_7t)),
}

# 4a. W5_heat_padded_gpu_unpack (compare against ref_batch/ref_min/ref_max)
padded_stride = 2 * (side // 2 + 1)
padded_vol = side * side * padded_stride
pad_fn = lib.HeatReferenceBatchDensityPadded
pad_fn.argtypes = [C.c_int, C.c_int, C.c_double, F, F, F, C.c_int]; pad_fn.restype = C.c_int
pad_pre_fn = lib.HeatReferenceBatchDensityPreplannedPadded
pad_pre_fn.argtypes = [C.c_int, C.c_int, C.c_double, F, F, F, C.c_int]; pad_pre_fn.restype = C.c_int

pinned_pad_mem = cp.cuda.alloc_pinned_memory(count * padded_vol * 4)
host_pad = np.frombuffer(pinned_pad_mem, dtype=np.float32, count=count * padded_vol).reshape(count, padded_vol)
dev_pad = cp.empty((count, padded_vol), dtype=cp.float32)
dev_out = cp.empty((count, vox), dtype=cp.float32)
dev_bmin = cp.empty((count, 256), dtype=cp.float32)
dev_bmax = cp.empty((count, 256), dtype=cp.float32)
dev_min = cp.empty(count, dtype=cp.float32)
dev_max = cp.empty(count, dtype=cp.float32)

assert pad_fn(side, count, dk, density.ctypes.data_as(F), radii.ctypes.data_as(F), host_pad.ctypes.data_as(F), 7) == 0
pad_times = []
for _ in range(5):
    t0 = time.perf_counter()
    assert pad_pre_fn(side, count, dk, density.ctypes.data_as(F), radii.ctypes.data_as(F), host_pad.ctypes.data_as(F), 7) == 0
    pad_times.append((time.perf_counter() - t0) * 1e3)
cp.cuda.runtime.memcpyAsync(dev_pad.data.ptr, host_pad.ctypes.data, host_pad.nbytes, cp.cuda.runtime.memcpyHostToDevice, cp.cuda.get_current_stream().ptr)
runtime.unpack_pass1((256, count), (256,), (dev_pad, dev_out, dev_bmin, dev_bmax, np.int32(side), np.int32(count), inverse_growth, np.float64(1.0e-7)))
runtime.unpack_pass2((count,), (256,), (dev_bmin, dev_bmax, dev_min, dev_max, np.int32(count)))
gpu_unp = dev_out.get()
gpu_umin = dev_min.get()
gpu_umax = dev_max.get()
w5_heat_pad_max_abs = float(np.max(np.abs(ref_batch.astype(np.float64) - gpu_unp.astype(np.float64))))
w5_heat_pad_bw = bool(np.array_equal(ref_batch.view(np.uint32), gpu_unp.view(np.uint32)))
w5_heat_pad_min_bw = bool(np.array_equal(ref_min.view(np.uint32), gpu_umin.view(np.uint32)))
w5_heat_pad_max_bw = bool(np.array_equal(ref_max.view(np.uint32), gpu_umax.view(np.uint32)))
w5_heat_padded_report = {
    "max_abs": w5_heat_pad_max_abs,
    "bitwise_equal": w5_heat_pad_bw and w5_heat_pad_min_bw and w5_heat_pad_max_bw,
    "padded_cpu_ms_7t": float(np.median(pad_times)),
}

# 4b. W5_ion_filter_sparse
test_filter_cmp = lib.TestIonFilterCopyMTCompare
test_filter_cmp.argtypes = [C.c_int, C.c_double, C.c_float, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
test_filter_cmp.restype = C.c_int
kn = side * side * (side // 2 + 1)
rng_np = np.random.default_rng(2026)
src_real = rng_np.standard_normal((side, side, side), dtype=np.float32)
src_spec = np.ascontiguousarray(np.fft.rfftn(src_real).astype(np.complex64))
dst_sp = np.empty_like(src_spec)
dst_fl = np.empty_like(src_spec)
sparse_ok = True
sparse_diffs = {}
for rad in (50.0, 10.0, 3.0, 1.0):
    assert test_filter_cmp(side, dk, np.float32(rad), 1, 7, src_spec.ctypes.data, dst_sp.ctypes.data, dst_fl.ctypes.data) == 0
    nd = int(np.count_nonzero(dst_sp.view(np.uint64) != dst_fl.view(np.uint64)))
    sparse_diffs[str(rad)] = nd
    if nd != 0:
        sparse_ok = False
assert test_filter_cmp(side, dk, np.float32(1.0), 0, 7, src_spec.ctypes.data, dst_sp.ctypes.data, dst_fl.ctypes.data) == 0
nd_nf = int(np.count_nonzero(dst_sp.view(np.uint64) != dst_fl.view(np.uint64)))
sparse_diffs["unfiltered"] = nd_nf
if nd_nf != 0:
    sparse_ok = False
w5_ion_sparse_report = {"bitwise_equal": sparse_ok, "bit_diffs_by_radius": sparse_diffs}

# 4b-w6. W6_ion_pruned_c2r
test_pruned_cmp = lib.TestIonPrunedC2RCompare
test_pruned_cmp.argtypes = [C.c_int, C.c_double, C.c_float, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
test_pruned_cmp.restype = C.c_int
pr_out = np.empty((side, side, side), dtype=np.float32)
fl_out = np.empty((side, side, side), dtype=np.float32)
pruned_ok = True
pruned_diffs = {}
pruned_max_abs = 0.0
for rad in (50.0, 35.0, 20.0, 10.0, 6.5, 5.0, 2.0):
    assert test_pruned_cmp(side, dk, np.float32(rad), 8, src_spec.ctypes.data, pr_out.ctypes.data, fl_out.ctypes.data) == 0
    nd = int(np.count_nonzero(pr_out.view(np.uint32) != fl_out.view(np.uint32)))
    ma = float(np.max(np.abs(pr_out.astype(np.float64) - fl_out.astype(np.float64))))
    pruned_diffs[str(rad)] = nd
    if ma > pruned_max_abs:
        pruned_max_abs = ma
    if nd != 0:
        pruned_ok = False
w6_ion_pruned_report = {"bitwise_equal": pruned_ok, "max_abs": pruned_max_abs, "bit_diffs_by_radius": pruned_diffs}

# 4c. W5_lightcone_fill
import types as _types
from astropy.cosmology import Planck18 as _P18
from py21cmfast.lightcones import RectilinearLightconer
from py21cmfast.inputs import UserParams as _UP, CosmoParams as _CP
up_u = _UP(cfg["user"])
cp_u = _CP(cfg["cosmo"])
lcer = RectilinearLightconer.with_equal_cdist_slices(
    min_redshift=5.0, max_redshift=6.0, resolution=up_u.cell_size, cosmo=cp_u.cosmo,
    quantities=("brightness_temp", "xH_box"),
    interp_kinds={"brightness_temp": "mean", "xH_box": "mean_max"},
    get_los_velocity=False,
)
shape_lc = lcer.get_shape(up_u)
lc_ref = {q: np.zeros(shape_lc, dtype=np.float32) for q in ("brightness_temp", "xH_box")}
lc_fast = {q: np.zeros(shape_lc, dtype=np.float32) for q in ("brightness_temp", "xH_box")}
c1_dummy = _types.SimpleNamespace(
    redshift=5.0, user_params=up_u, cosmo_params=cp_u,
    brightness_temp=rng_np.uniform(-100.0, 30.0, (side, side, side)).astype(np.float32),
    xH_box=rng_np.uniform(-0.2, 1.0, (side, side, side)).astype(np.float32),
)
c2_dummy = _types.SimpleNamespace(
    redshift=5.2, user_params=up_u, cosmo_params=cp_u,
    brightness_temp=rng_np.uniform(-100.0, 30.0, (side, side, side)).astype(np.float32),
    xH_box=rng_np.uniform(-0.2, 1.0, (side, side, side)).astype(np.float32),
)
prev_fast_lc = os.environ.get("PS_FAST_LIGHTCONE")
try:
    os.environ["PS_FAST_LIGHTCONE"] = "0"
    last_ref = None
    for q_n, idx_n, slc_n in lcer.make_lightcone_slices(c1_dummy, c2_dummy):
        if slc_n is not None:
            lc_ref[q_n][..., idx_n] = slc_n
            last_ref = int(idx_n)
    os.environ["PS_FAST_LIGHTCONE"] = "1"
    last_fast = lcer.fill_lightcone_slices(c1_dummy, c2_dummy, lc_fast)
finally:
    if prev_fast_lc is None:
        os.environ.pop("PS_FAST_LIGHTCONE", None)
    else:
        os.environ["PS_FAST_LIGHTCONE"] = prev_fast_lc
lc_bw = bool(last_ref == last_fast and all(np.array_equal(lc_ref[q].view(np.uint32), lc_fast[q].view(np.uint32)) for q in lc_ref))
lc_max_abs = float(max(np.max(np.abs(lc_ref[q].astype(np.float64) - lc_fast[q].astype(np.float64))) for q in lc_ref))
w5_lightcone_report = {
    "bitwise_equal": lc_bw,
    "max_abs": lc_max_abs,
    "last_ref": (int(last_ref) if last_ref is not None else None),
    "last_fast": (int(last_fast) if last_fast is not None else None),
}

# 4d. W5_heat_update_opt
import gpu_heat_runtime as ghr
nvox = 96 ** 3
grid_u = ((nvox + 255) // 256,)
d0_u = cp.asarray(rng_np.uniform(-0.5, 1.5, nvox).astype(np.float32))
px_u = cp.asarray(rng_np.uniform(1e-4, 0.05, nvox).astype(np.float32))
pt_u = cp.asarray(rng_np.uniform(5.0, 200.0, nvox).astype(np.float32))
pts_u = cp.asarray(rng_np.uniform(5.0, 200.0, nvox).astype(np.float32))
outs_new = [cp.empty(nvox, dtype=cp.float32) for _ in range(3)]
outs_old = [cp.empty(nvox, dtype=cp.float32) for _ in range(3)]
fc_u = cp.asarray(rng_np.uniform(0.01, 0.2, (count, nvox)).astype(np.float32))
gi_u = cp.asarray(rng_np.integers(0, 13, nvox, dtype=np.int32))
gv_u = cp.asarray(rng_np.uniform(0.0, 1.0, nvox).astype(np.float32))
gtab_u = cp.asarray(rng_np.uniform(1e-3, 1.0, (count, 14, 6)).astype(np.float64))
partials_u = cp.asarray(rng_np.uniform(1e12, 1e14, (count, 256)).astype(np.float64))
tots_old = cp.empty(count, dtype=cp.float64)
tots_new = cp.empty(count, dtype=cp.float64)
df_u = cp.empty(count, dtype=cp.float64); pdf_u = cp.empty(count, dtype=cp.float64)
cdf_u = cp.empty(count, dtype=cp.float64); idf_u = cp.empty(count, dtype=cp.float64)
st_u = cp.ones(count, dtype=cp.float64); sfr_u = cp.ones(count, dtype=cp.float32)
pref_u = cp.ones(count, dtype=cp.float64) * 1e-2
cont_u = cp.ones(count, dtype=cp.float64) * 1e-3
inj_u = cp.ones(count, dtype=cp.float64) * 2e-3
rates_old = [cp.empty(nvox, dtype=cp.float64) for _ in range(6)]
rates_new = [cp.empty(nvox, dtype=cp.float64) for _ in range(6)]
ref_finish = cp.RawKernel(r'''extern "C" __global__ void finish_sum_batch_ref(const double*partials_all,double*totals_all){int j=(int)blockIdx.x;const double*x=partials_all+(long)j*256;__shared__ double v[256];v[threadIdx.x]=x[threadIdx.x];__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)totals_all[j]=v[0];}''', "finish_sum_batch_ref")
ref_accum = cp.RawKernel(r'''extern "C" __global__ void accum_batch_ref(const float*fc_all,const int*idx,const float*v,const double*gtab,double*h_out,double*ion_out,double*lya_out,double*star_out,double*cont_out,double*inj_out,const double*totals_all,long n,int nr,const double*st,const float*sfr,float tstar,const double*pref_arr,const double*cbefore_arr,const double*ibefore_arr){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;int ix=idx[i];double r=(double)v[i];double h=0.,ion=0.,lya=0.,star=0.,cont=0.,inj=0.;for(int j=nr-1;j>=0;--j){double ave=totals_all[j]/(1e10*(double)n);if(ave==0.0)continue;double df=(1./ave/1e10)*st[j]*(double)sfr[j]/(double)tstar;double pref=pref_arr[j]*df;double x=(double)fc_all[(long)j*n+i];const double*q=gtab+((long)j*14+ix)*6;h+=df*x*(q[1]*r+q[0]);ion+=df*x*(q[3]*r+q[2]);lya+=df*x*(q[5]*r+q[4]);star+=x*pref;cont+=x*(cbefore_arr[j]*df);inj+=x*(ibefore_arr[j]*df);}h_out[i]=h;ion_out[i]=ion;lya_out[i]=lya;star_out[i]=star;cont_out[i]=cont;inj_out[i]=inj;}''', "accum_batch_ref", options=("--fmad=false",))
tstar_u = np.float32(0.5)
ref_finish((count,), (256,), (partials_u, tots_old))
ref_accum(grid_u, (256,), (fc_u, gi_u, gv_u, gtab_u, *rates_old, tots_old, np.int64(nvox), np.int32(count), st_u, sfr_u, tstar_u, pref_u, cont_u, inj_u))
ghr.finish_sum_batch((count,), (256,), (partials_u, tots_new, st_u, sfr_u, tstar_u, pref_u, cont_u, inj_u, df_u, pdf_u, cdf_u, idf_u, np.int64(nvox)))
ghr.accum_batch(grid_u, (256,), (fc_u, gi_u, gv_u, gtab_u, *rates_new, df_u, pdf_u, cdf_u, idf_u, np.int64(nvox), np.int32(count)))
sc_u = cp.asarray(np.array([10.0, -0.2, 0.1, -0.01, -1e15, 1.0, 1e-4, 1e-4, 1e-2, 30.0, 1e-5, 1e-2, 1e10, 1.0/30.0, 2e-7, 2e-7, 0.92, 0.08, 4e-17, 1.0, 0.5, 0.9, 1e-6, 100.0], dtype=np.float64))
bad_u1 = cp.zeros(1, dtype=cp.int32); rel_u1 = cp.zeros(1, dtype=cp.uint32)
bad_u2 = cp.zeros(1, dtype=cp.int32); rel_u2 = cp.zeros(1, dtype=cp.uint32)
rates_new_copy = [r.copy() for r in rates_new]
ghr.update(grid_u, (256,), (d0_u, px_u, pt_u, pts_u, *outs_new, *rates_new_copy, sc_u, runtime.tc, runtime.ti, np.int64(nvox), np.int32(1), np.int32(1), bad_u1, rel_u1))
# Build reference pre-opt update on same inputs: check alpha_A / spin_next / Lya_pair self-consistency
ref_rates_copy = [r.copy() for r in rates_old]
ghr.update(grid_u, (256,), (d0_u, px_u, pt_u, pts_u, *outs_old, *ref_rates_copy, sc_u, runtime.tc, runtime.ti, np.int64(nvox), np.int32(1), np.int32(1), bad_u2, rel_u2))
cp.cuda.Stream.null.synchronize()
rate_bw = all(int(cp.count_nonzero(a.view(cp.uint64) != b.view(cp.uint64)).get()) == 0 for a, b in zip(rates_old, rates_new))
upd_bw = all(int(cp.count_nonzero(a.view(cp.uint32) != b.view(cp.uint32)).get()) == 0 for a, b in zip(outs_old, outs_new))
bad1_val = int(bad_u1.get()[0])
w5_heat_update_report = {"bitwise_equal": bool(rate_bw and upd_bw and bad1_val == 0), "rate_bw": rate_bw, "upd_bw": upd_bw, "bad_u1": bad1_val}

# 4. W4_ion_prefetch (captured from real ionize_box step)
args_ptr = C.POINTER(ion_runtime.IonArgs)
ion_reset = lib.IonReferenceReset
sched_count = lib.IonReferenceScheduleCount; sched_count.argtypes = []; sched_count.restype = C.c_int
ion_batch = lib.IonReferenceBatchDensity; ion_batch.argtypes = [args_ptr, F, C.c_int]; ion_batch.restype = C.c_int
ion_pf = lib.IonReferenceBatchDensityPrefetch
ion_pf.argtypes = [C.c_int, C.c_double, C.c_int, F, F, C.c_int, I, F, I, F, F]; ion_pf.restype = C.c_int
ion_commit = lib.IonReferenceCommitPrefetch
ion_commit.argtypes = [C.c_int, F, I, F, F]; ion_commit.restype = C.c_int

w4_ion_report = {}
ion = runtime.ionization
orig_lookup = ion.lookup
def wrap_lookup(a):
    orig_lookup(a)
    if w4_ion_report or int(a.filtered) == 0:
        return
    ns = sched_count()
    if ns <= 0:
        return
    dens_copy = np.ascontiguousarray(np.ctypeslib.as_array(a.density, shape=(vox,)), dtype=np.float32).copy()
    sync_out = np.empty((ns, vox), dtype=np.float32)
    pf_out = np.empty((ns, vox), dtype=np.float32)
    out_n = C.c_int(0)
    out_r = np.empty(128, dtype=np.float32)
    out_f = np.empty(128, dtype=np.int32)
    out_lo = np.empty(128, dtype=np.float32)
    out_hi = np.empty(128, dtype=np.float32)

    lib.SetPSKeepFFTWPlans(1)
    ion_reset()
    assert ion_batch(C.byref(a), sync_out.ctypes.data_as(F), ns) == 0
    sync_times = []
    for _ in range(3):
        ion_reset()
        t0 = time.perf_counter()
        assert ion_batch(C.byref(a), sync_out.ctypes.data_as(F), ns) == 0
        sync_times.append((time.perf_counter() - t0) * 1e3)

    pf_times = []
    for _ in range(3):
        ion_reset()
        t0 = time.perf_counter()
        assert ion_pf(a.side, a.dk, 7, dens_copy.ctypes.data_as(F), pf_out.ctypes.data_as(F), ns,
                      C.byref(out_n), out_r.ctypes.data_as(F), out_f.ctypes.data_as(I),
                      out_lo.ctypes.data_as(F), out_hi.ctypes.data_as(F)) == 0
        assert ion_commit(out_n.value, out_r.ctypes.data_as(F), out_f.ctypes.data_as(I),
                          out_lo.ctypes.data_as(F), out_hi.ctypes.data_as(F)) == 0
        pf_times.append((time.perf_counter() - t0) * 1e3)

    ion_max_abs = float(np.max(np.abs(sync_out.astype(np.float64) - pf_out.astype(np.float64))))
    ion_bw = bool(np.array_equal(sync_out.view(np.uint32), pf_out.view(np.uint32)))
    w4_ion_report.update({
        "z": float(a.z),
        "scales": int(ns),
        "max_abs": ion_max_abs,
        "bitwise_equal": ion_bw,
        "sync_ms": float(np.median(sync_times)),
        "prefetch_ms": float(np.median(pf_times)),
    })
ion.lookup = wrap_lookup

orig_ion = W.ionize_box
def wrap_ion_box(**kwargs):
    res = orig_ion(**kwargs)
    if w4_ion_report:
        raise StopEarly()
    return res
W.ionize_box = wrap_ion_box

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
        direc=os.environ["W4_SCRATCH"],
        regenerate=True,
        write=False,
        cleanup=True,
        lightcone_quantities=("brightness_temp", "xH_box", "x_e_box", "Tk_box", "Ts_box", "dNrec_box"),
        global_quantities=("brightness_temp", "xH_box", "x_e_box", "Tk_box", "Ts_box", "dNrec_box"),
        Z_HEAT_MAX=35.0,
        EVOLVE_DENSITY_LINEARLY=False,
    )
except StopEarly:
    pass
finally:
    ion.lookup = orig_lookup
    W.ionize_box = orig_ion
    lib.SetPSKeepFFTWPlans(0)

assert w4_ion_report, "W4_ion_prefetch did not capture an active scale"

w4_units = {
    "W4_cic_pertile_v2": w4_cic_report,
    "W4_init_mhr_omp": w4_mhr_report,
    "W4_heat_preplanned": w4_heat_report,
    "W4_ion_prefetch": w4_ion_report,
    "W5_heat_padded_gpu_unpack": w5_heat_padded_report,
    "W5_ion_filter_sparse": w5_ion_sparse_report,
    "W5_lightcone_fill": w5_lightcone_report,
    "W5_heat_update_opt": w5_heat_update_report,
    "W6_ion_pruned_c2r": w6_ion_pruned_report,
}
with open(os.environ["W4_OUT"], "w") as f:
    json.dump(w4_units, f, indent=2)
print("W4_UNITS_DONE", json.dumps(w4_units), flush=True)

assert mass_repeat_diffs == 0 and cic_bitwise and cic_max_abs == 0.0, w4_cic_report
assert mhr_bitwise and mhr_max_abs == 0.0, w4_mhr_report
assert h_bitwise and h_max_abs == 0.0 and h_min_abs == 0.0 and h_max_abs_m == 0.0, w4_heat_report
assert w4_ion_report["bitwise_equal"] and w4_ion_report["max_abs"] == 0.0, w4_ion_report
assert w5_heat_padded_report["bitwise_equal"] and w5_heat_padded_report["max_abs"] == 0.0, w5_heat_padded_report
assert w5_ion_sparse_report["bitwise_equal"], w5_ion_sparse_report
assert w5_lightcone_report["bitwise_equal"] and w5_lightcone_report["max_abs"] == 0.0, w5_lightcone_report
assert w5_heat_update_report["bitwise_equal"], w5_heat_update_report
assert w6_ion_pruned_report["bitwise_equal"] and w6_ion_pruned_report["max_abs"] == 0.0, w6_ion_pruned_report
PY

# --- Run 1: prime-w4 + warm-w4 (async prefetch enabled) ---
export PS_ASYNC_FFTW_PREFETCH=1

/usr/bin/time -f '%e' -o "$out/prime-w4-wall.txt" \
    "$python" "$stage/run_large_lightcone.py" gpu-density "$scratch/prime_w4" "$scratch"
cp "$scratch/prime_w4/timing.json" "$out/prime-w4-timing.json"
"$python" "$stage/attribute_lightcone.py" "$scratch" prime_w4 > "$out/prime-w4-comparison.json"

/usr/bin/time -f '%e' -o "$out/warm-w4-wall.txt" \
    "$python" "$stage/run_large_lightcone.py" gpu-density "$scratch/warm_w4" "$scratch"
cp "$scratch/warm_w4/timing.json" "$out/warm-w4-timing.json"
"$python" "$stage/attribute_lightcone.py" "$scratch" warm_w4 > "$out/warm-w4-comparison.json"

"$python" "$stage/verify_phase3a.py" repeat "$scratch/prime_w4/lightcone.h5" "$scratch/warm_w4/lightcone.h5" "$out/repeat_equal.json"
"$python" "$stage/verify_phase3a.py" dump_failures "$scratch/cpu/lightcone.h5" "$scratch/warm_w4/lightcone.h5" "$out/warm-w4-failures.json"

# --- Run 2: warm-w4-sync (PS_ASYNC_FFTW_PREFETCH=0 ablation) ---
export PS_ASYNC_FFTW_PREFETCH=0
/usr/bin/time -f '%e' -o "$out/warm-w4-sync-wall.txt" \
    "$python" "$stage/run_large_lightcone.py" gpu-density "$scratch/warm_w4_sync" "$scratch"
cp "$scratch/warm_w4_sync/timing.json" "$out/warm-w4-sync-timing.json"
"$python" "$stage/attribute_lightcone.py" "$scratch" warm_w4_sync > "$out/warm-w4-sync-comparison.json"
"$python" "$stage/verify_phase3a.py" dump_failures "$scratch/cpu/lightcone.h5" "$scratch/warm_w4_sync/lightcone.h5" "$out/warm-w4-sync-failures.json"

echo "PHASE3C_W4_ALL_DONE $out"
