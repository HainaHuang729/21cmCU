"""Reproducible benchmark CLI for 21cmCU single-GPU, I/O, CLASS/angular, and multi-GPU suites."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import py21cmcu
from astropy.units import pixel, pixel_scale
from py21cmfast.rsds import apply_rsds, include_dvdr_in_tau21


def _log(msg: str) -> None:
    sys.stderr.write(f"[bench {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _reset_and_gc() -> None:
    gc.collect()
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()


def _max_rss_gb() -> float:
    """Return process high-water mark RSS in GiB (Linux ru_maxrss is KiB)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(ru) / (1024.0 * 1024.0), 2)


def _gpu_counts(n_avail: int) -> list[int]:
    return [c for c in (1, 2, 3, 4) if c <= n_avail]


def _default_cpu_threads() -> int:
    return min(16, max(4, os.cpu_count() or 8))


def collect_metadata() -> dict[str, Any]:
    """Collect host CPU, GPU, CUDA, and package versions."""
    gpu_names: list[str] = []
    gpu_mem_mib: list[int] = []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    gpu_names.append(parts[0])
                    gpu_mem_mib.append(int(parts[1]))
    except OSError:
        pass

    return {
        "hostname": socket.gethostname(),
        "cpu_count": int(os.cpu_count() or 1),
        "bench_cpu_threads": _default_cpu_threads(),
        "cuda_compiled": bool(py21cmcu.cuda_compiled()),
        "cuda_available": bool(py21cmcu.cuda_available()),
        "cuda_device_count": int(py21cmcu.cuda_device_count()),
        "gpu_names": gpu_names,
        "gpu_memory_mib": gpu_mem_mib,
        "py21cmcu_version": py21cmcu.__version__,
        "numpy_version": np.__version__,
    }


def _run_coeval_pair(
    inputs: py21cmcu.InputParameters, *, out_redshifts: list[float] | tuple[float, ...]
) -> dict[str, Any]:
    """Time CPU vs GPU-fast for a run_coeval workload."""
    dim = int(inputs.simulation_options.DIM)
    hii_dim = int(inputs.simulation_options.HII_DIM)
    _log(f"coeval CPU ({dim}/{hii_dim}) starting ...")
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        coevs_cpu = py21cmcu.run_coeval(
            out_redshifts=list(out_redshifts), inputs=inputs, regenerate=True, write=False
        )
    cpu_s = time.perf_counter() - t0
    del coevs_cpu
    _log(f"coeval CPU ({dim}/{hii_dim}) done in {cpu_s:.2f}s; starting GPU ...")

    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        coevs_gpu = py21cmcu.run_coeval(
            out_redshifts=list(out_redshifts), inputs=inputs, regenerate=True, write=False
        )
        st = py21cmcu.cuda_stats()
    gpu_s = time.perf_counter() - t0
    x_hi_final = round(float(np.mean(coevs_gpu[-1].neutral_fraction)), 4)
    del coevs_gpu
    _reset_and_gc()

    return {
        "cpu_s": round(cpu_s, 3),
        "gpu_s": round(gpu_s, 3),
        "speedup": round(cpu_s / max(gpu_s, 1e-6), 2),
        "xH_final": x_hi_final,
        "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 3),
        "d2h_gb": round(st["registry_d2h_bytes"] / 1e9, 3),
        "registry_hits": int(st["registry_hits"]),
    }


def run_core_suite() -> dict[str, Any]:
    """Run CPU vs GPU stage-level (IC, perturb, ionize) and 3-z L-INTEGRAL+TS coeval suite."""
    n_threads = _default_cpu_threads()
    inputs = py21cmcu.InputParameters(
        random_seed=7101,
        node_redshifts=[12.0, 10.0, 8.0],
    ).evolve_input_structs(
        DIM=512,
        HII_DIM=256,
        BOX_LEN=300.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    out: dict[str, Any] = {}
    out["B1_256_512_LINT_TS_3z_fast"] = _run_coeval_pair(inputs, out_redshifts=[8.0])

    # Stage breakdown on 512/256 grid
    # 1. InitialConditions (2LPT)
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        ic_cpu = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
    ic_cpu_s = time.perf_counter() - t0
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        ic_gpu = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
    ic_gpu_s = time.perf_counter() - t0
    out["stage_ic_512_256_2lpt"] = {
        "cpu_s": round(ic_cpu_s, 3),
        "gpu_s": round(ic_gpu_s, 3),
        "speedup": round(ic_cpu_s / max(ic_gpu_s, 1e-6), 2),
    }

    # 2. Perturb field (256^3, 2LPT)
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        pf_cpu = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic_cpu, inputs=inputs, regenerate=True, write=False
        )
    pf_cpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        pf_gpu = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic_gpu, inputs=inputs, regenerate=True, write=False
        )
    pf_gpu_s = time.perf_counter() - t0
    out["stage_perturb_256_2lpt"] = {
        "cpu_s": round(pf_cpu_s, 3),
        "gpu_s": round(pf_gpu_s, 3),
        "speedup": round(pf_cpu_s / max(pf_gpu_s, 1e-6), 2),
    }

    del ic_cpu, pf_cpu, pf_gpu
    _reset_and_gc()

    # 3. Ionize box (256^3, E-INTEGRAL, no TS)
    eor_inputs = py21cmcu.InputParameters(random_seed=7102).evolve_input_structs(
        DIM=512,
        HII_DIM=256,
        BOX_LEN=300.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        ic_eor = py21cmcu.compute_initial_conditions(
            inputs=eor_inputs, regenerate=True, write=False
        )
        pf_eor = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic_eor, regenerate=True, write=False
        )
    _reset_and_gc()

    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        ib_cpu = py21cmcu.compute_ionization_field(
            perturbed_field=pf_eor,
            initial_conditions=ic_eor,
            inputs=eor_inputs,
            regenerate=True,
            write=False,
        )
    ib_cpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        ib_gpu = py21cmcu.compute_ionization_field(
            perturbed_field=pf_eor,
            initial_conditions=ic_eor,
            inputs=eor_inputs,
            regenerate=True,
            write=False,
        )
    ib_gpu_s = time.perf_counter() - t0
    out["stage_ionize_256_eint"] = {
        "cpu_s": round(ib_cpu_s, 3),
        "gpu_s": round(ib_gpu_s, 3),
        "speedup": round(ib_cpu_s / max(ib_gpu_s, 1e-6), 2),
    }
    del ic_eor, pf_eor, ic_gpu, ib_cpu, ib_gpu
    _reset_and_gc()
    return out


def run_highres_suite() -> dict[str, Any]:
    """Run B3 (512/1024 lo-res 2LPT IC+PF) and B6 (768^3 EoR) CPU vs GPU benchmarks."""
    n_threads = _default_cpu_threads()
    out: dict[str, Any] = {}

    inp_b3 = py21cmcu.InputParameters(random_seed=7201).evolve_input_structs(
        DIM=1024,
        HII_DIM=512,
        BOX_LEN=1000.0,
        N_THREADS=n_threads,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=False,
    )
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        ic_c = py21cmcu.compute_initial_conditions(inputs=inp_b3, regenerate=True, write=False)
        pf_c = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic_c, inputs=inp_b3, regenerate=True, write=False
        )
    cpu_b3 = time.perf_counter() - t0
    del ic_c, pf_c
    _reset_and_gc()

    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        ic = py21cmcu.compute_initial_conditions(inputs=inp_b3, regenerate=True, write=False)
        pf = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic, inputs=inp_b3, regenerate=True, write=False
        )
        st = py21cmcu.cuda_stats()
    gpu_b3 = time.perf_counter() - t0
    out["B3_512_1024_2LPT_lo_fast"] = {
        "cpu_s": round(cpu_b3, 3),
        "gpu_s": round(gpu_b3, 3),
        "speedup": round(cpu_b3 / max(gpu_b3, 1e-6), 2),
        "pf_mean": round(float(np.mean(pf.density)), 6),
        "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 3),
    }
    del ic, pf
    _reset_and_gc()

    inp_b6 = py21cmcu.InputParameters(random_seed=7601).evolve_input_structs(
        DIM=768,
        HII_DIM=768,
        BOX_LEN=768.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    out["B6_768_EoR_fast"] = _run_coeval_pair(inp_b6, out_redshifts=[7.0])
    return out


def run_lightcone_suite() -> dict[str, Any]:
    """Run Direction D lightcone + RSD + 1D/2D power spectrum CPU vs GPU benchmark."""
    n_threads = _default_cpu_threads()
    inputs = py21cmcu.InputParameters(
        random_seed=4242,
        node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).evolve_input_structs(
        DIM=256,
        HII_DIM=128,
        BOX_LEN=200.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lightconer = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=6.5,
        max_redshift=11.0,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "neutral_fraction"),
        cosmo=inputs.cosmo_params.cosmo,
    )

    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        lc_cpu = py21cmcu.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )
        t_lc_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = py21cmcu.lightcone_chunk_power_spectra(
            lc_cpu, quantity="brightness_temp_with_rsds", n_bins=16
        )
        t_chunks_cpu = time.perf_counter() - t0
    del lc_cpu

    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        lc = py21cmcu.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )
        t_lc = time.perf_counter() - t0
        st = py21cmcu.cuda_stats()

        t0 = time.perf_counter()
        chunks = py21cmcu.lightcone_chunk_power_spectra(
            lc, quantity="brightness_temp_with_rsds", n_bins=16
        )
        t_chunks = time.perf_counter() - t0

        # Standalone RSD microbenchmark on 128x128x512 volume (CPU vs GPU)
        shape = (128, 128, 512)
        rng = np.random.default_rng(123)
        tb = rng.normal(-20.0, 15.0, size=shape).astype(np.float32)
        vel = rng.normal(0.0, 2.0e-16, size=shape).astype(np.float32)
        tau21 = np.abs(rng.normal(0.02, 0.02, size=shape)).astype(np.float32)
        zs = np.linspace(6.5, 11.0, shape[2])
        _ = include_dvdr_in_tau21(
            tb[:, :, :16], vel[:, :, :16], zs[:16], inputs, periodic=False, tau_21=tau21[:, :, :16]
        )
        _ = apply_rsds(tb[:, :, :16], vel[:, :, :16], zs[:16], inputs, periodic=False, n_rsd_subcells=4)

        t0 = time.perf_counter()
        dv = include_dvdr_in_tau21(tb, vel, zs, inputs, periodic=False, tau_21=tau21)
        t_dv_gpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = apply_rsds(dv, vel, zs, inputs, periodic=False, n_rsd_subcells=4)
        t_rsd_gpu = time.perf_counter() - t0

    with py21cmcu.use_mode("strict", use_cuda=False):
        t0 = time.perf_counter()
        dv_c = include_dvdr_in_tau21(tb, vel, zs, inputs, periodic=False, tau_21=tau21)
        t_dv_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = apply_rsds(dv_c, vel, zs, inputs, periodic=False, n_rsd_subcells=4)
        t_rsd_cpu = time.perf_counter() - t0

    shape_lc = list(lc.lightcones["brightness_temp_with_rsds"].shape)
    del lc
    _reset_and_gc()

    return {
        "LC_256_128_z6.5_11_with_rsd_fast": {
            "cpu_s": round(t_lc_cpu, 3),
            "gpu_s": round(t_lc, 3),
            "speedup": round(t_lc_cpu / max(t_lc, 1e-6), 2),
            "shape": shape_lc,
            "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 3),
            "registry_hits": int(st["registry_hits"]),
            "chunk_ps_cpu_s": round(t_chunks_cpu, 4),
            "chunk_ps_gpu_s": round(t_chunks, 4),
            "n_chunks": len(chunks["chunks"]),
        },
        "GPU_RSD_512slices": {
            "dvdr_cpu_s": round(t_dv_cpu, 4),
            "dvdr_gpu_s": round(t_dv_gpu, 4),
            "dvdr_speedup": round(t_dv_cpu / max(t_dv_gpu, 1e-6), 2),
            "apply_rsds_cpu_s": round(t_rsd_cpu, 4),
            "apply_rsds_gpu_s": round(t_rsd_gpu, 4),
            "apply_rsds_speedup": round(t_rsd_cpu / max(t_rsd_gpu, 1e-6), 2),
        },
    }


def run_class_angular_suite() -> dict[str, Any]:
    """Run Direction H CLASS transfer cache + AngularLightconer CPU vs GPU benchmarks."""
    n_threads = _default_cpu_threads()
    out: dict[str, Any] = {}

    # 1. CLASS transfer table cold vs warm in-memory & disk cache + IC compute CPU vs GPU
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["P21_CLASS_CACHE"] = "1"
        os.environ["P21_CLASS_DISK_CACHE"] = "1"
        py21cmcu.clear_class_cache()
        class_kw = dict(
            DIM=256,
            HII_DIM=128,
            BOX_LEN=192.0,
            N_THREADS=n_threads,
            POWER_SPECTRUM="CLASS",
            V_CB_MODEL="FLUCTS",
            K_MAX_FOR_CLASS=1.0,
            SOURCE_MODEL="E-INTEGRAL",
            USE_TS_FLUCT=True,
            Z_HEAT_MAX=12.0,
            USE_MCGS=True,
            RECOMB_MODEL="inhomogeneous",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
        )
        with py21cmcu.use_mode("fast", use_cuda=True, devices=[0], direc=tmpdir):
            inp1 = py21cmcu.InputParameters(
                random_seed=7301, node_redshifts=[12.0, 10.0]
            ).evolve_input_structs(**class_kw)
            t0 = time.perf_counter()
            _ = inp1.cosmo_tables
            cold_s = time.perf_counter() - t0

            inp2 = py21cmcu.InputParameters(
                random_seed=7302, node_redshifts=[12.0, 10.0]
            ).evolve_input_structs(**class_kw)
            t0 = time.perf_counter()
            _ = inp2.cosmo_tables
            mem_hit_s = time.perf_counter() - t0

            py21cmcu.clear_class_cache()
            inp3 = py21cmcu.InputParameters(
                random_seed=7303, node_redshifts=[12.0, 10.0]
            ).evolve_input_structs(**class_kw)
            t0 = time.perf_counter()
            _ = inp3.cosmo_tables
            disk_hit_s = time.perf_counter() - t0

        # Warm CosmoTables IC compute CPU vs GPU (256/128 CLASS+FLUCTS)
        t0 = time.perf_counter()
        with py21cmcu.use_mode("strict", use_cuda=False):
            _ = py21cmcu.compute_initial_conditions(inputs=inp1, regenerate=True, write=False)
        ic_class_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
            _ = py21cmcu.compute_initial_conditions(inputs=inp1, regenerate=True, write=False)
        ic_class_gpu = time.perf_counter() - t0

    out["class_cache"] = {
        "cold_build_s": round(cold_s, 3),
        "mem_hit_s": round(mem_hit_s, 6),
        "disk_hit_s": round(disk_hit_s, 4),
        "ic_256_128_class_flucts_cpu_s": round(ic_class_cpu, 3),
        "ic_256_128_class_flucts_gpu_s": round(ic_class_gpu, 3),
        "ic_speedup": round(ic_class_cpu / max(ic_class_gpu, 1e-6), 2),
    }

    # 2. AngularLightconer E2E + slice interpolation CPU vs GPU (256/128, z=7-10)
    ang_inputs = py21cmcu.InputParameters(
        random_seed=7401, node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    ).evolve_input_structs(
        DIM=256,
        HII_DIM=128,
        BOX_LEN=192.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=11.0,
        KEEP_3D_VELOCITIES=True,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lc_ang = py21cmcu.AngularLightconer.like_rectilinear(
        ang_inputs.simulation_options,
        match_at_z=7.8,
        max_redshift=9.8,
        interpolation_order=1,
        cosmo=ang_inputs.cosmo_params.cosmo,
    )

    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        alc_cpu = py21cmcu.run_lightcone(
            lightconer=lc_ang,
            inputs=ang_inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            regenerate=True,
            write=False,
        )
    ang_cpu_s = time.perf_counter() - t0
    del alc_cpu

    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        alc_gpu = py21cmcu.run_lightcone(
            lightconer=lc_ang,
            inputs=ang_inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            regenerate=True,
            write=False,
        )
    ang_gpu_s = time.perf_counter() - t0

    # Isolated slice + LOS velocity interpolation microbenchmark
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        c_lo, c_hi = py21cmcu.run_coeval(
            out_redshifts=[7.5, 9.0], inputs=ang_inputs, regenerate=True, write=False
        )
    pixeq = pixel_scale(ang_inputs.simulation_options.cell_size / pixel)
    dc_a = ang_inputs.cosmo_params.cosmo.comoving_distance(c_hi.redshift).to(
        pixel, equivalencies=pixeq
    )
    dc_b = ang_inputs.cosmo_params.cosmo.comoving_distance(c_lo.redshift).to(
        pixel, equivalencies=pixeq
    )
    lcd = 0.5 * (dc_a + dc_b)
    with py21cmcu.use_mode("strict", use_cuda=False):
        t0 = time.perf_counter()
        _ = lc_ang.interpolate_slice(
            lcd,
            c_hi.brightness_temp,
            c_lo.brightness_temp,
            dc_a,
            dc_b,
            ang_inputs.simulation_options.cell_size,
            kind="mean",
        )
        _ = lc_ang.interpolate_los_velocity_slice(lcd, c_hi, c_lo, dc_a, dc_b)
        slice_cpu_s = time.perf_counter() - t0
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        t0 = time.perf_counter()
        _ = lc_ang.interpolate_slice(
            lcd,
            c_hi.brightness_temp,
            c_lo.brightness_temp,
            dc_a,
            dc_b,
            ang_inputs.simulation_options.cell_size,
            kind="mean",
        )
        _ = lc_ang.interpolate_los_velocity_slice(lcd, c_hi, c_lo, dc_a, dc_b)
        slice_gpu_s = time.perf_counter() - t0

    out["AngularLightconer_256_128"] = {
        "e2e_cpu_s": round(ang_cpu_s, 3),
        "e2e_gpu_s": round(ang_gpu_s, 3),
        "e2e_speedup": round(ang_cpu_s / max(ang_gpu_s, 1e-6), 2),
        "slice_plus_vel_cpu_s": round(slice_cpu_s, 4),
        "slice_plus_vel_gpu_s": round(slice_gpu_s, 4),
        "slice_speedup": round(slice_cpu_s / max(slice_gpu_s, 1e-6), 2),
        "shape": list(alc_gpu.lightcones["brightness_temp_with_rsds"].shape),
    }
    del alc_gpu, c_lo, c_hi
    _reset_and_gc()
    return out


def run_io_lazy_suite() -> dict[str, Any]:
    """Run Direction I DeviceLZF HDF5 I/O and CUDA_LAZY_D2H residency benchmarks."""
    n_threads = _default_cpu_threads()
    out: dict[str, Any] = {}

    # 1. Lightcone HDF5 I/O comparison: uncompressed vs CPU LZF vs GPU DeviceLZF (256x256x512 = 134 MB)
    zz, yy, xx = np.meshgrid(
        np.linspace(0.0, 2.0, 512, dtype=np.float32),
        np.linspace(0.0, 1.0, 256, dtype=np.float32),
        np.linspace(0.0, 1.0, 256, dtype=np.float32),
        indexing="ij",
    )
    bt = np.round(-25.0 * np.sin(3.0 * xx + 2.0 * yy) * np.cos(zz) - 10.0, decimals=2).astype(
        np.float32
    )
    bt = np.ascontiguousarray(bt.transpose(2, 1, 0))
    xhi = np.clip(np.round(0.5 + 0.45 * np.tanh(bt / 15.0), decimals=3), 0.0, 1.0).astype(
        np.float32
    )
    fields = {"brightness_temp": bt, "neutral_fraction": xhi}
    raw_mb = sum(a.nbytes for a in fields.values()) / (1024.0 * 1024.0)

    io_results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode_label, comp, use_cuda in (
            ("none", "none", False),
            ("cpu_lzf", "lzf", False),
            ("device_lzf", "device-lzf", True),
        ):
            path = Path(tmpdir) / f"lc_{mode_label}.h5"
            t0 = time.perf_counter()
            with py21cmcu.use_mode("strict", use_cuda=use_cuda):
                meta = py21cmcu.save_lightcone(
                    fields,
                    path,
                    compression=comp,
                    shuffle=True,
                    chunk_z=32,
                )
            write_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            with h5py.File(path, "r") as h5f:
                for k in fields:
                    _ = h5f["lightcones"][k][:]
            read_s = time.perf_counter() - t0

            io_results[mode_label] = {
                "raw_mb": round(raw_mb, 2),
                "compressed_mb": round(meta["compressed_bytes"] / (1024.0 * 1024.0), 2),
                "ratio": round(meta["raw_bytes"] / max(meta["compressed_bytes"], 1), 2),
                "write_s": round(write_s, 4),
                "read_s": round(read_s, 4),
                "write_mb_s": round(raw_mb / max(write_s, 1e-6), 1),
                "read_mb_s": round(raw_mb / max(read_s, 1e-6), 1),
            }
    out["hdf5_io_256x256x512"] = io_results

    # 2. CUDA_LAZY_D2H eager vs lazy on B1 (512/256, 3z, L-INTEGRAL + TS)
    inputs = py21cmcu.InputParameters(
        random_seed=7101, node_redshifts=[12.0, 10.0, 8.0]
    ).evolve_input_structs(
        DIM=512,
        HII_DIM=256,
        BOX_LEN=300.0,
        N_THREADS=n_threads,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    residency: dict[str, Any] = {}
    for lazy in (False, True):
        _reset_and_gc()
        t0 = time.perf_counter()
        with py21cmcu.use_mode("fast", use_cuda=True, devices=[0], lazy_d2h=lazy):
            cv = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)
            wall_run = time.perf_counter() - t0
            st = py21cmcu.cuda_stats()
            _ = float(np.mean(cv[-1].brightness_temp))
            wall_with_read = time.perf_counter() - t0
        key = "lazy_d2h" if lazy else "eager_d2h"
        residency[key] = {
            "run_wall_s": round(wall_run, 3),
            "wall_with_tb_read_s": round(wall_with_read, 3),
            "h2d_mb": round(st["registry_h2d_bytes"] / 1e6, 2),
            "d2h_mb": round(st["registry_d2h_bytes"] / 1e6, 2),
            "registry_hits": int(st["registry_hits"]),
            "registry_misses": int(st["registry_misses"]),
        }
        del cv
        _reset_and_gc()
    out["residency_B1_3z"] = residency
    return out


def run_batch_suite() -> dict[str, Any]:
    """Run Direction E CPU sequential vs single-GPU concurrent batch throughput benchmark."""
    n_threads = _default_cpu_threads()
    n_tasks = 4
    tasks = []
    for i in range(n_tasks):
        inp = py21cmcu.InputParameters(
            random_seed=8100 + i,
            node_redshifts=[12.0, 10.0, 8.0],
        ).evolve_input_structs(
            DIM=192,
            HII_DIM=96,
            BOX_LEN=192.0,
            N_THREADS=n_threads,
            SOURCE_MODEL="L-INTEGRAL",
            USE_TS_FLUCT=True,
            Z_HEAT_MAX=12.0,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
        )
        tasks.append(
            py21cmcu.BatchTask(
                task_id=f"bench_{i}",
                inputs=inp,
                mode="coeval",
                out_redshifts=(8.0,),
                cuda_mode="fast",
                ps_n_bins=16,
            )
        )

    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    with py21cmcu.use_mode("strict", use_cuda=False):
        res_cpu = py21cmcu.run_batch(
            tasks,
            devices=(0,),
            workers_per_gpu=1,
            threads_per_worker=n_threads,
            use_cuda=False,
            in_process=True,
        )
    cpu_wall = time.perf_counter() - t0
    assert all(r.ok for r in res_cpu), [r.error for r in res_cpu if not r.ok]
    out["batch_4x_192_96_cpu"] = {
        "wall_s": round(cpu_wall, 3),
        "throughput_per_hour": round(3600.0 * n_tasks / max(cpu_wall, 1e-6), 1),
    }

    for w in (1, 2):
        gpu_tasks = []
        for t in tasks:
            inp_w = t.inputs.evolve_input_structs(N_THREADS=4)
            gpu_tasks.append(
                py21cmcu.BatchTask(
                    task_id=t.task_id,
                    inputs=inp_w,
                    mode=t.mode,
                    out_redshifts=t.out_redshifts,
                    cuda_mode=t.cuda_mode,
                    ps_n_bins=t.ps_n_bins,
                )
            )
        t0 = time.perf_counter()
        res = py21cmcu.run_batch(
            gpu_tasks,
            devices=(0,),
            workers_per_gpu=w,
            threads_per_worker=4,
            use_cuda=True,
            in_process=False,
        )
        wall = time.perf_counter() - t0
        assert all(r.ok for r in res), [r.error for r in res if not r.ok]
        out[f"batch_4x_192_96_workers_{w}"] = {
            "wall_s": round(wall, 3),
            "throughput_per_hour": round(3600.0 * n_tasks / max(wall, 1e-6), 1),
            "speedup_vs_cpu": round(cpu_wall / max(wall, 1e-6), 2),
        }
    return out


def _bench_ic_box(
    *, dim: int, hii_dim: int, box_len: float, perturb_algo: str, devices: list[int]
) -> dict[str, Any]:
    """Run IC + perturb_field + 1D P(k) for a large grid on the requested GPU device list."""
    _log(f"mgpu {dim}^3 ({perturb_algo}, g={len(devices)}) starting ...")
    inputs = py21cmcu.InputParameters(random_seed=9101).evolve_input_structs(
        DIM=dim,
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        N_THREADS=_default_cpu_threads(),
        PERTURB_ALGORITHM=perturb_algo,
        PERTURB_ON_HIGH_RES=False,
    )
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=devices):
        ic = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
        t_ic = time.perf_counter() - t0
        t1 = time.perf_counter()
        pf = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic, inputs=inputs, regenerate=True, write=False
        )
        t_pf = time.perf_counter() - t1
        dens = np.asarray(pf.density, dtype=np.float32)
        t2 = time.perf_counter()
        ps1 = py21cmcu.power_spectrum_1d(dens, box_len, n_bins=16, use_cuda=True)
        t_ps = time.perf_counter() - t2
        st = py21cmcu.cuda_stats()
        valid = ps1["counts"] > 0
        pk_finite = bool(np.any(valid) and np.all(np.isfinite(ps1["pk"][valid])))
    t_tot = time.perf_counter() - t0
    del ic, pf, dens, ps1
    _reset_and_gc()
    return {
        "status": "ok",
        "dim": dim,
        "hii_dim": hii_dim,
        "voxels_B": round((dim**3) / 1e9, 3),
        "algo": perturb_algo,
        "n_gpus": len(devices),
        "ic_s": round(t_ic, 3),
        "perturb_s": round(t_pf, 3),
        "ps1d_s": round(t_ps, 4),
        "total_s": round(t_tot, 3),
        "mgpu_ic_runs": int(st["mgpu_ic_runs"]),
        "pk_finite": pk_finite,
        "max_rss_gb": _max_rss_gb(),
    }


def run_mgpu_suite() -> dict[str, Any]:
    """Run Direction G multi-GPU scaling across 1024^3, 1536^3, 1792^3, 2048^3, and 16x batch."""
    n_avail = int(py21cmcu.cuda_device_count())
    if n_avail < 2:
        return {"skipped": f"Requires >= 2 GPUs, found {n_avail}"}

    counts = _gpu_counts(n_avail)
    out: dict[str, Any] = {"gpu_counts_tested": counts}

    # 1. 1024^3 (DIM=1024, HII_DIM=256) scaling across 1, 2, ..., n_avail GPUs
    ic_1024: dict[str, Any] = {}
    for algo in ("ZELDOVICH", "2LPT"):
        base_ic_s: float | None = None
        for g in counts:
            key = f"{algo}_g{g}"
            try:
                rec = _bench_ic_box(
                    dim=1024, hii_dim=256, box_len=512.0, perturb_algo=algo, devices=list(range(g))
                )
                if base_ic_s is None:
                    base_ic_s = float(rec["ic_s"])
                rec["ic_speedup_vs_1gpu"] = round(base_ic_s / max(float(rec["ic_s"]), 1e-6), 2)
                ic_1024[key] = rec
            except Exception as exc:  # noqa: BLE001
                ic_1024[key] = {"status": "error", "error": str(exc), "n_gpus": g, "algo": algo}
                _reset_and_gc()
    out["ic_1024"] = ic_1024

    # 2. 1536^3 (DIM=1536, HII_DIM=256) out-of-1-GPU scaling across 2, 3 (, 4) GPUs
    ic_1536: dict[str, Any] = {}
    for algo in ("ZELDOVICH", "2LPT"):
        for g in [c for c in counts if c >= 2]:
            key = f"{algo}_g{g}"
            try:
                ic_1536[key] = _bench_ic_box(
                    dim=1536, hii_dim=256, box_len=768.0, perturb_algo=algo, devices=list(range(g))
                )
            except Exception as exc:  # noqa: BLE001
                ic_1536[key] = {"status": "error", "error": str(exc), "n_gpus": g, "algo": algo}
                _reset_and_gc()
    out["ic_1536"] = ic_1536

    # 3. 1792^3 (DIM=1792, 5.75B voxels) on max available GPUs (>= 3)
    if n_avail >= 3:
        ic_1792: dict[str, Any] = {}
        for algo in ("ZELDOVICH", "2LPT"):
            key = f"{algo}_g{n_avail}"
            try:
                ic_1792[key] = _bench_ic_box(
                    dim=1792,
                    hii_dim=256,
                    box_len=896.0,
                    perturb_algo=algo,
                    devices=list(range(n_avail)),
                )
            except Exception as exc:  # noqa: BLE001
                ic_1792[key] = {
                    "status": "error",
                    "error": str(exc),
                    "n_gpus": n_avail,
                    "algo": algo,
                }
                _reset_and_gc()
        out["ic_1792"] = ic_1792

    # 4. 2048^3 (DIM=2048, 8.59B voxels) on max available GPUs (Zel'dovich always attempted; 2LPT if >=4 GPUs)
    if n_avail >= 3:
        ic_2048: dict[str, Any] = {}
        algos_2048 = ("ZELDOVICH", "2LPT") if n_avail >= 4 else ("ZELDOVICH",)
        for algo in algos_2048:
            key = f"{algo}_g{n_avail}"
            try:
                ic_2048[key] = _bench_ic_box(
                    dim=2048,
                    hii_dim=256,
                    box_len=1024.0,
                    perturb_algo=algo,
                    devices=list(range(n_avail)),
                )
            except Exception as exc:  # noqa: BLE001
                ic_2048[key] = {
                    "status": "error",
                    "error": str(exc),
                    "n_gpus": n_avail,
                    "algo": algo,
                }
                _reset_and_gc()
        out["ic_2048"] = ic_2048

    # 5. Multi-GPU ensemble throughput: 16 x (192/96 L-INTEGRAL+TS) across 1, 2, ..., n_avail GPUs
    n_batch_tasks = 16
    batch_tasks = []
    for i in range(n_batch_tasks):
        inp = py21cmcu.InputParameters(
            random_seed=9200 + i, node_redshifts=[12.0, 10.0, 8.0]
        ).evolve_input_structs(
            DIM=192,
            HII_DIM=96,
            BOX_LEN=192.0,
            N_THREADS=4,
            SOURCE_MODEL="L-INTEGRAL",
            USE_TS_FLUCT=True,
            Z_HEAT_MAX=12.0,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
        )
        batch_tasks.append(
            py21cmcu.BatchTask(
                task_id=f"mgpu_b_{i}",
                inputs=inp,
                mode="coeval",
                out_redshifts=(8.0,),
                cuda_mode="fast",
                ps_n_bins=16,
            )
        )

    batch_mgpu: dict[str, Any] = {}
    base_batch_s: float | None = None
    for g in counts:
        t0 = time.perf_counter()
        res = py21cmcu.run_batch(
            batch_tasks,
            devices=tuple(range(g)),
            workers_per_gpu=2,
            threads_per_worker=4,
            use_cuda=True,
            in_process=False,
        )
        wall = time.perf_counter() - t0
        assert all(r.ok for r in res), [r.error for r in res if not r.ok]
        if base_batch_s is None:
            base_batch_s = wall
        batch_mgpu[f"batch_16x_g{g}"] = {
            "n_gpus": g,
            "workers_per_gpu": 2,
            "wall_s": round(wall, 3),
            "throughput_per_hour": round(3600.0 * n_batch_tasks / max(wall, 1e-6), 1),
            "speedup_vs_1gpu": round(base_batch_s / max(wall, 1e-6), 2),
        }
    out["batch_16x_192_96"] = batch_mgpu
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 21cmCU benchmark suites")
    parser.add_argument(
        "--suite",
        default="lightcone,batch",
        help="Comma-separated suites: core, highres, lightcone, class_angular, io_lazy, batch, mgpu, all",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip suites already present in --json if the output file exists",
    )
    args = parser.parse_args()

    requested = {s.strip().lower() for s in args.suite.split(",") if s.strip()}
    if "all" in requested:
        requested.discard("all")
        requested |= {"core", "highres", "lightcone", "class_angular", "io_lazy", "batch"}

    results: dict[str, Any] = {}
    if args.resume and args.json is not None and args.json.exists():
        results = json.loads(args.json.read_text())
        _log(f"resuming from existing {args.json} with suites {sorted(results.keys())}")
    results["metadata"] = collect_metadata()
    suite_runners = {
        "core": run_core_suite,
        "highres": run_highres_suite,
        "lightcone": run_lightcone_suite,
        "class_angular": run_class_angular_suite,
        "io_lazy": run_io_lazy_suite,
        "batch": run_batch_suite,
        "mgpu": run_mgpu_suite,
    }
    for name in ("core", "highres", "lightcone", "class_angular", "io_lazy", "batch", "mgpu"):
        if name in requested:
            if args.resume and name in results:
                _log(f"skipping already completed suite '{name}'")
                continue
            _log(f"starting suite '{name}' ...")
            t_suite = time.perf_counter()
            results[name] = suite_runners[name]()
            _log(f"finished suite '{name}' in {time.perf_counter() - t_suite:.1f}s")
            results["metadata"]["max_rss_gb"] = _max_rss_gb()
            if args.json is not None:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(results, indent=2) + "\n")

    results["metadata"]["max_rss_gb"] = _max_rss_gb()
    text = json.dumps(results, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
