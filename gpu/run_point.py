from time import perf_counter as _clock
_worker_entry=_clock()
"""Run the fixed point in one isolated process."""
import argparse
import ctypes as C
from contextlib import ExitStack
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("cpu-reference", "gpu-validation", "cpu-timing", "gpu-timing", "cpu-trajectory"))
parser.add_argument("out", type=Path)
parser.add_argument("scratch", type=Path)
group=parser.add_mutually_exclusive_group()
group.add_argument("--cpu-ic", type=Path)
group.add_argument("--gsl-modes", action="store_true")
args = parser.parse_args()
root = Path(os.environ["PS_GPU_ROOT"])
out = args.out
scratch = args.scratch
config = json.loads((root / "point_config.json").read_text())
gpu = args.mode.startswith("gpu")

from py21cmfast import run_coeval
import py21cmfast.c_21cmfast as native

sha = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
if args.mode == "cpu-timing":
    assert sha == config["native_sha256"]
trajectory_lib = C.CDLL(native.__file__)
trajectory_lib.SetTrajectoryPath.argtypes = [C.c_char_p]
trajectory_lib.SetTrajectoryPath.restype = None
if os.environ.get("PS_PRODUCTION_PS_ONLY") != "1":
    trajectory_lib.SetTrajectoryPath(os.fsencode(out / "trajectory.tsv"))
_import_done=_clock()
print("POINT", args.mode, config["index"], config["seed"], sha, flush=True)

def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def difference(actual, expected, rtol=1e-4, atol=1e-8):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    finite = np.isfinite(actual) & np.isfinite(expected)
    if not finite.any():
        return {"n": int(actual.size), "nonfinite": int(actual.size),
                "failed": int(actual.size), "max_abs": None, "max_rel": None}
    delta = np.abs(actual - expected)
    valid_delta = np.where(finite, delta, -1)
    worst = np.unravel_index(valid_delta.argmax(), actual.shape)
    squares = np.square(np.where(finite, delta, 0))
    return {
        "n": int(actual.size),
        "nonfinite": int((~finite).sum()),
        "failed": int((~(finite & (delta <= atol + rtol*np.abs(expected)))).sum()),
        "max_abs": float(valid_delta[worst]),
        "max_abs_index": [int(i) for i in worst],
        "max_rel": float(np.max(np.where(finite, delta / np.maximum(np.abs(expected), 1e-25), 0))),
        "sum_sq": float(squares.sum()),
        "rms": float(np.sqrt(squares.sum() / finite.sum())),
        "mean_ref": float(expected[finite].mean()),
        "mean_test": float(actual[finite].mean()),
    }

fields = ("xe", "Tk", "Ts")
FieldPtr = C.POINTER(C.c_float)
Observe = C.CFUNCTYPE(C.c_int, C.c_double, C.c_double, C.c_long,
                     FieldPtr, FieldPtr, FieldPtr)
records = []
reference_steps = None
if args.mode == "gpu-validation":
    reference_steps = json.loads((out / "cpu_heat_steps.json").read_text())

def observe(z, previous_z, count, *pointers):
    step = len(records)
    try:
        # The first native step uses infinity as a sentinel for absent history.
        previous_z = previous_z if np.isfinite(previous_z) else None
        record = {"z": z, "previous_z": previous_z}
        if args.mode == "cpu-reference":
            path = scratch / f"heat-{step:03d}.npy"
            reference = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(3, count))
            for i, pointer in enumerate(pointers):
                reference[i] = np.ctypeslib.as_array(pointer, shape=(count,))
            del reference
        else:
            expected_step = reference_steps[step]
            assert (z, previous_z) == (expected_step["z"], expected_step["previous_z"])
            reference = np.load(scratch / f"heat-{step:03d}.npy", mmap_mode="r")
            record["fields"] = {
                name: difference(np.ctypeslib.as_array(pointer, shape=(count,)), reference[i])
                for i, (name, pointer) in enumerate(zip(fields, pointers))
            }
        records.append(record)
        if step % 20 == 0:
            print("HEAT_VALIDATION", args.mode, step, z, flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1

ion_records = []
IonObserver = C.CFUNCTYPE(C.c_int, C.c_float, C.c_float, C.c_ulonglong, FieldPtr, FieldPtr)

def observe_ion(z, previous_z, count, neutral, recombinations):
    try:
        step = len(ion_records)
        fields_ion = (np.ctypeslib.as_array(neutral,shape=(count,)),
                      np.ctypeslib.as_array(recombinations,shape=(count,)))
        path = scratch / f"ion-{step:03d}.npy"
        row = {"z":z,"previous_z":previous_z}
        if args.mode == "cpu-reference":
            ref = np.lib.format.open_memmap(path,mode="w+",dtype="float32",shape=(2,count))
            for i,f in enumerate(fields_ion):
                ref[i] = f
        else:
            assert z == expected_ion[step]["z"] and previous_z == expected_ion[step]["previous_z"]
            ref = np.load(path,mmap_mode="r")
            row["fields"] = {key:difference(f,r) for key,f,r in zip(("xhi","dNrec"),fields_ion,ref)}
        ion_records.append(row)
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if args.mode in ("cpu-reference", "gpu-validation"):
    native_lib = C.CDLL(native.__file__)
    observer = Observe(observe)
    native_lib.SetHeatObserver.argtypes = [C.c_void_p]
    native_lib.SetHeatObserver.restype = None
    native_lib.SetHeatObserver(C.cast(observer, C.c_void_p))
    if args.mode == "gpu-validation":
        expected_ion = json.loads((out/"cpu_ion_steps.json").read_text())
    ion_callback = IonObserver(observe_ion)
    native_lib.SetIonObserver.argtypes = [C.c_void_p]
    native_lib.SetIonObserver.restype = None
    native_lib.SetIonObserver(C.cast(ion_callback,C.c_void_p))

side = config["user"]["HII_DIM"]
box = config["box"]
coordinates = np.fft.fftfreq(side, d=box/side) * 2 * np.pi
kx, ky, kz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
kmag = np.sqrt(kx**2 + ky**2 + kz**2)
edges = np.linspace(0, kmag.max() * 1.0000001, 61)
native_k = 0.5*(edges[1:] + edges[:-1])
target_k = np.asarray(config["ps_k"])
assert native_k[0] < target_k[0] < target_k[-1] < native_k[-1]
bins = np.clip(np.digitize(kmag.ravel(), edges)-1, 0, 59)
counts = np.maximum(np.bincount(bins, minlength=60), 1)
factor = native_k**3 / (2*np.pi**2)
del kx, ky, kz, kmag

def cpu_ps(field):
    f = np.asarray(field, np.float64)
    fk = np.fft.fftn(f-f.mean()) / side**3
    power = np.abs(fk)**2 * box**3
    return np.bincount(bins, weights=power.ravel(), minlength=60) / counts * factor

def gpu_ps(field):
    f = cp.asarray(field, dtype=cp.float64)
    fk = cp.fft.fftn(f-f.mean()) / side**3
    power = cp.abs(fk)**2 * box**3
    spectrum = cp.bincount(device_bins, weights=power.ravel(), minlength=60) / device_counts
    return spectrum * device_factor

generated_ic=None
full_ic_metrics=None
if os.environ.get("PS_GENERATE_FULL_IC") == "1":
    from py21cmfast import initial_conditions
    begin=_clock()
    generated_ic=initial_conditions(
        user_params=config["user"],cosmo_params=config["cosmo"],random_seed=config["seed"],
        direc=str(scratch),write=False,regenerate=True,
    )
    ic_seconds=_clock()-begin
    begin=_clock()
    complete_ic_path=out/"initial-conditions.h5"
    generated_ic.write(fname=complete_ic_path)
    full_ic_metrics={"compute_seconds":ic_seconds,"write_seconds":_clock()-begin,
                     "bytes":complete_ic_path.stat().st_size}

if gpu:
    import cupy as cp
    from coupled import CoupledRuntime

    coord=cp.fft.fftfreq(side,d=box/side)*2*cp.pi
    shell=cp.sqrt(coord[:,None,None]**2 + coord[None,:,None]**2 + coord[None,None,:]**2)
    device_edges=cp.asarray(edges)
    device_bins=cp.clip(cp.digitize(shell.ravel(),device_edges)-1,0,59)
    device_counts=cp.maximum(cp.bincount(device_bins,minlength=60),1)
    device_native=cp.asarray(native_k)
    device_target=cp.asarray(target_k)
    device_factor=cp.asarray(factor)
    del shell
    if args.mode == "gpu-validation":
        field = np.random.default_rng(487255).normal(size=(side,)*3).astype(np.float32)
        preflight = difference(gpu_ps(field).get(), cpu_ps(field))
        write_json(out / "ps_preflight.json", preflight)
        assert preflight["failed"] == 0
        del field
    runtime = CoupledRuntime()
    runtime.install(native, cached_ic=args.cpu_ic is not None or generated_ic is not None, same_gsl=args.gsl_modes)

def summarize_heat():
    result = {}
    for name in fields:
        steps = [(row["z"], row["fields"][name]) for row in records]
        worst_z, worst = max(steps, key=lambda entry: entry[1]["max_abs"] or 0.)
        tested = sum(item["n"] for _, item in steps)
        nonfinite = sum(item["nonfinite"] for _, item in steps)
        absolute_fails = sum(item["failed"] for _, item in steps)
        result[name] = {
            "cells": tested,
            "nonfinite": nonfinite,
            "failed": absolute_fails,
            "first_failed_z": next((z for z,item in steps if item["failed"]), None),
            "max_abs": worst["max_abs"],
            "max_abs_z": worst_z,
            "max_abs_index": worst.get("max_abs_index"),
            "max_rel": max(item["max_rel"] or 0. for _,item in steps),
            "rms": float(np.sqrt(sum(item.get("sum_sq",0.) for _,item in steps)
                                 / max(tested - nonfinite, 1))),
        }
    return result

_setup_done=_clock()
init_box=generated_ic
if generated_ic is not None:
    runtime.perturbation.load_cpu_ic(generated_ic)
ic_read_seconds=None
ic_reads=[]
if args.cpu_ic is not None:
    from py21cmfast.outputs import InitialConditions
    original_ic_read=InitialConditions.read
    def timed_ic_read(self,*args,**kwargs):
        t=_clock()
        try:
            return original_ic_read(self,*args,**kwargs)
        finally:
            keys=kwargs.get("keys")
            ic_reads.append({"keys":list(keys) if keys is not None else None,"seconds":_clock()-t})
    InitialConditions.read=timed_ic_read
    if not args.cpu_ic.is_file():
        raise FileNotFoundError(args.cpu_ic)
    begin=time.perf_counter()
    ic_fields=["hires_density","lowres_density"]
    ic_fields.extend("lowres_v"+axis for axis in "xyz")
    ic_fields.extend("lowres_v"+axis+"_2LPT" for axis in "xyz")
    init_box=InitialConditions.from_file(str(args.cpu_ic),arrays_to_load=ic_fields)
    ic_read_seconds=time.perf_counter()-begin
    if os.environ.get("PS_KEEP_CPU_IC") == "1":
        def prepare_without_disk_reload(flag_options, force=False):
            if init_box.user_params.USE_RELATIVE_VELOCITIES or flag_options.USE_HALO_FIELD:
                raise RuntimeError("IC reuse trial supports the fixed production PS model")
        init_box.prepare_for_spin_temp=prepare_without_disk_reload

    assert int(init_box.random_seed)==config["seed"]
    assert init_box.hires_density.size==config["user"]["DIM"]**3
    if gpu and os.environ.get("PS_NATIVE_PERTURB_TRACE") != "1":
        runtime.perturbation.load_cpu_ic(init_box)

_inputs_ready=_clock()
start = time.perf_counter()
background_ic = None
with ExitStack() as resources:
    work = resources.enter_context(tempfile.TemporaryDirectory(prefix="coeval-", dir=scratch))
    kwargs = {}
    if os.environ.get("PS_OVERLAP_IC_WRITE") == "1":
        from ic_archive import BackgroundICArchive
        background_ic = resources.enter_context(BackgroundICArchive(
            out / "initial-conditions.h5", start, sparse_zeros=os.environ.get("PS_SPARSE_IC_ZEROS") == "1",
        ))
        kwargs["hooks"] = {background_ic.capture: {}}

    try:
        coeval = run_coeval(
            redshift=config["z"], user_params=config["user"], cosmo_params=config["cosmo"],
            astro_params=config["astro"], flag_options=config["flags"],
            random_seed=config["seed"], init_box=init_box, direc=work, write=False, regenerate=True, cleanup=True,
            Z_HEAT_MAX=35.0, EVOLVE_DENSITY_LINEARLY=False, **kwargs,
        )
    finally:
        if gpu:
            write_json(out / f"{args.mode}-runtime.json", runtime.report())
        if args.mode == "cpu-reference":
            write_json(out / "cpu_heat_steps.json", records)
        if args.mode == "gpu-validation":
            write_json(out / "thermal_validation.json",
                       {"steps": records, "summary": summarize_heat() if records else {}})
    if args.mode == "cpu-reference":
        write_json(out/"cpu_ion_steps.json",ion_records)
    elif args.mode == "gpu-validation":
        assert len(ion_records)==len(expected_ion)
        write_json(out/"ion_history_validation.json",ion_records)
    coeval_seconds = time.perf_counter() - start
    early_ic_metrics = background_ic.finish() if background_ic else None

if gpu and args.cpu_ic is not None:
    assert runtime.perturbation.host_ic_uploads == (0 if os.environ.get("PS_NATIVE_PERTURB_TRACE") == "1" else 1)
    assert runtime.perturbation.calls==123
np.testing.assert_allclose([c.redshift for c in coeval], config["z"], rtol=0, atol=1e-6)
assert all(int(c.random_seed) == config["seed"] for c in coeval)
start = time.perf_counter()
if gpu:
    xhi=cp.stack([runtime.ionization.neutral_fields[c.xH_box.ctypes.data].mean(dtype=cp.float64)
                  for c in coeval]).get()
    spectra=cp.stack([gpu_ps(runtime.brightness.fields[c.brightness_temp.ctypes.data])
                      for c in coeval])
    target_ps=cp.stack([cp.interp(device_target,device_native,row) for row in spectra]).get()
    native_ps=spectra.get()
    if os.environ.get("PS_ESTIMATOR_PAIR") == "1":
        ps_cpu_same_brightness = np.stack([
            cpu_ps(runtime.brightness.fields[c.brightness_temp.ctypes.data].get())
            for c in coeval
        ])
        ps32_cpu_same_brightness = np.stack([
            np.interp(target_k, native_k, row) for row in ps_cpu_same_brightness
        ])
        write_json(out / "same-brightness-ps.json", {
            "native_60": difference(native_ps, ps_cpu_same_brightness),
            "production_32": difference(target_ps, ps32_cpu_same_brightness),
        })

else:
    xhi=np.array([np.mean(c.xH_box,dtype=np.float64) for c in coeval])
    native_ps=np.array([cpu_ps(c.brightness_temp) for c in coeval])
    target_ps=np.array([np.interp(target_k,native_k,row) for row in native_ps])
np.savez_compressed(out / f"{args.mode}-outputs.npz", z=config["z"], xhi=xhi,
                    native_k=native_k, ps_native=native_ps, ps_k=target_k, ps_full=target_ps)
postprocess_seconds = time.perf_counter() - start

report = {
    "mode": args.mode, "point": config["index"], "seed": config["seed"],
    "native_sha256": sha, "node": os.environ.get("SLURMD_NODENAME"),
    "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if gpu else None,
    "ic_read_seconds":ic_read_seconds,
    "coeval_seconds": coeval_seconds, "postprocess_with_write_seconds": postprocess_seconds,
    "rtol": 1e-4, "atol": 1e-8,
    "worker_import_seconds":_import_done-_worker_entry,
    "worker_setup_seconds":_setup_done-_import_done,
    "worker_input_seconds":_inputs_ready-_setup_done,
    "ic_read_events":ic_reads,
}
if args.mode in ("cpu-reference", "gpu-validation", "gpu-timing", "cpu-trajectory"):
    with np.load(root / "reference.npz") as reference:
        np.testing.assert_array_equal(reference["theta"], config["theta"])
        np.testing.assert_array_equal(reference["effective_ic_seeds"], [config["seed"]])
        np.testing.assert_array_equal(reference["z"], config["z"])
        np.testing.assert_array_equal(reference["ps_k"], target_k)
        np.testing.assert_allclose(reference["native_k"], native_k, rtol=1e-14, atol=0)
        report["production_reference"] = {
            "xhi": difference(xhi, reference["global_xHI"][0]),
            "ps_native": difference(native_ps, reference["ps_native"][0]),
            "ps_full": difference(target_ps, reference["ps_full"][0]),
        }
if args.mode == "gpu-validation":
    with np.load(out / "cpu-reference-outputs.npz") as reference:
        report["same_node_cpu"] = {
            "xhi": difference(xhi, reference["xhi"]),
            "ps_native": difference(native_ps, reference["ps_native"]),
            "ps_full": difference(target_ps, reference["ps_full"]),
        }
    assert len(records) == len(reference_steps)
    report["ion_steps"] = len(ion_records)
    report["ion_history"] = {
        k: {"failed":sum(row["fields"][k]["failed"] for row in ion_records),
            "n":sum(row["fields"][k]["n"] for row in ion_records),
            "max_abs":max(row["fields"][k]["max_abs"] or 0 for row in ion_records),
            "max_rel":max(row["fields"][k]["max_rel"] or 0 for row in ion_records)}
        for k in ("xhi","dNrec")
    }
    report["thermal_steps"] = len(records)
    report["thermal"] = summarize_heat()
    report["filter_steps"] = len(runtime.filter_steps)
    assert report["filter_steps"] > 0
if gpu:
    from native_profile import PHASES
    seconds, calls = (C.c_double*5)(), (C.c_ulonglong*5)()
    profile = runtime.lib.GetPSNativeProfile
    profile.argtypes = [C.POINTER(C.c_double),C.POINTER(C.c_ulonglong)]
    profile.restype = None
    profile(seconds,calls)
    report["native_phases"] = {phase: {"seconds":t,"calls":n}
                               for (_,_,phase),t,n in zip(PHASES,seconds,calls)}
paired = os.environ.get("PS_PAIRED_ARCHIVE") == "1"
if paired:
    assert len(runtime.ic.seconds) == 1
    if report["production_reference"]["ps_full"]["failed"]:
        raise ValueError("Production 32-bin validation failed before the archive could be written")
    from paired_archives import archive_pair
    archive_start = _clock()
    full_ic_metrics, report["coeval_archive"] = archive_pair(
        coeval, config, out, int(os.environ["PS_COEVAL_LZF_WORKERS"]),
    )
    full_ic_metrics.update(source="production GSL and cuFFT",
                           gpu_build_seconds=runtime.ic.seconds[0],
                           compute_seconds=report["native_phases"]["ic"]["seconds"])
    report["paired_archive_seconds"] = _clock() - archive_start
elif os.environ.get("PS_ARCHIVE_GPU_IC") == "1":
    assert len(runtime.ic.seconds) == 1
    complete_ic_path = out / "initial-conditions.h5"
    if early_ic_metrics is None:
        begin = _clock()
        if os.environ.get("PS_SPARSE_IC_ZEROS") == "1":
            from ic_archive import BackgroundICArchive
            with BackgroundICArchive(complete_ic_path, begin, sparse_zeros=True) as archive:
                archive.capture(coeval[0].init_struct)
                written = archive.finish()
            early_ic_metrics = {
                "write_seconds": written["write_seconds"],
                "ic_fsync_before_coeval_archive_seconds": written["background_fsync_seconds"],
                "unallocated_zero_arrays": written["unallocated_zero_arrays"],
            }
        else:
            coeval[0].init_struct.write(fname=complete_ic_path)
            early_ic_metrics = {"write_seconds": _clock() - begin}
    full_ic_metrics = {
        "source": "production GSL and cuFFT",
        "gpu_build_seconds": runtime.ic.seconds[0],
        "compute_seconds": report["native_phases"]["ic"]["seconds"],
        "bytes": complete_ic_path.stat().st_size,
        **early_ic_metrics,
    }
if full_ic_metrics is not None:
    report["complete_ic"]=full_ic_metrics
if os.environ.get("PS_FULL_COEVAL") == "1" and not paired:
    if report["production_reference"]["ps_full"]["failed"]:
        raise ValueError("Production 32-bin validation failed before the archive could be written")
    from coeval_export import export_evolved
    report["coeval_archive"]=export_evolved(coeval,config,out/"coeval-fields.h5",
                                            os.environ.get("PS_COEVAL_COMPRESSION","none"),
                                            workers=int(os.environ.get("PS_COEVAL_LZF_WORKERS","1")))
write_json(out / f"{args.mode}.json", report)
print("POINT_RESULT", json.dumps(report), flush=True)
if gpu and (out.name in {"gpu-fftw-density","xray","native-all"} or os.environ.get("PS_COMPARE_VOXELS") == "1"):
    import h5py
    voxel_start = time.perf_counter()
    checks = {}
    names = ("x_e_box", "Tk_box", "Ts_box", "xH_box", "dNrec_box", "brightness_temp")
    ref_file = root / "ic_backend_cross_20261118" / "cpu-CPU-IC" / "fields.h5"
    with h5py.File(ref_file, "r") as reference_fields:
        np.testing.assert_allclose(reference_fields["redshifts"][:], config["z"], rtol=0., atol=1.e-6)
        for index, box_i in enumerate(coeval):
            group = reference_fields[f"z{index:02d}"]
            checks[str(box_i.redshift)] = {
                name: difference(np.asarray(getattr(box_i, name)), group[name][:]) for name in names
            }
    write_json(out / "voxel-validation.json", {
        "checks": checks,
        "elapsed": time.perf_counter() - voxel_start,
        "summary": {
            name: {"failed": sum(row[name]["failed"] for row in checks.values()),
                   "n": sum(row[name]["n"] for row in checks.values()),
                   "max_abs": max(row[name]["max_abs"] for row in checks.values())}
            for name in names
        }
    })

