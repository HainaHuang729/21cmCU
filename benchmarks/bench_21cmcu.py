"""Reproducible benchmark CLI for 21cmCU core, high-res, lightcone/RSD/PS, and batch suites."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import py21cmcu
from py21cmfast.rsds import apply_rsds, include_dvdr_in_tau21


def _reset_and_gc() -> None:
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()


def run_core_suite() -> dict:
    """Run B1 (256/512 L-INTEGRAL+TS) and E-INTEGRAL benchmarks."""
    inputs = py21cmcu.InputParameters(
        random_seed=7101,
        node_redshifts=[12.0, 10.0, 8.0],
    ).evolve_input_structs(
        DIM=512,
        HII_DIM=256,
        BOX_LEN=300.0,
        N_THREADS=8,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True):
        coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)
        st = py21cmcu.cuda_stats()
    wall = time.perf_counter() - t0
    _reset_and_gc()
    return {
        "B1_256_512_LINT_TS_3z_fast": {
            "wall_s": round(wall, 3),
            "xH_final": round(float(np.mean(coevs[-1].neutral_fraction)), 4),
            "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 2),
            "registry_hits": int(st["registry_hits"]),
        }
    }


def run_highres_suite() -> dict:
    """Run B3 (512/1024 lo-res 2LPT IC+PF) and B6 (768^3 EoR) benchmarks."""
    out: dict = {}

    inp_b3 = py21cmcu.InputParameters(random_seed=7201).evolve_input_structs(
        DIM=1024,
        HII_DIM=512,
        BOX_LEN=1000.0,
        N_THREADS=8,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=False,
    )
    _reset_and_gc()
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True):
        ic = py21cmcu.compute_initial_conditions(inputs=inp_b3, regenerate=True, write=False)
        pf = py21cmcu.perturb_field(
            redshift=8.0, initial_conditions=ic, inputs=inp_b3, regenerate=True, write=False
        )
        st = py21cmcu.cuda_stats()
    wall_b3 = time.perf_counter() - t0
    out["B3_512_1024_2LPT_lo_fast"] = {
        "wall_s": round(wall_b3, 3),
        "pf_mean": round(float(np.mean(pf.density)), 6),
        "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 2),
    }
    del ic, pf
    _reset_and_gc()

    inp_b6 = py21cmcu.InputParameters(random_seed=7601).evolve_input_structs(
        DIM=768,
        HII_DIM=768,
        BOX_LEN=768.0,
        N_THREADS=8,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    t0 = time.perf_counter()
    with py21cmcu.use_mode("fast", use_cuda=True):
        cv = py21cmcu.run_coeval(out_redshifts=7.0, inputs=inp_b6, regenerate=True, write=False)[0]
        st6 = py21cmcu.cuda_stats()
    wall_b6 = time.perf_counter() - t0
    out["B6_768_EoR_fast"] = {
        "wall_s": round(wall_b6, 3),
        "xH": round(float(np.mean(cv.neutral_fraction)), 4),
        "h2d_gb": round(st6["registry_h2d_bytes"] / 1e9, 2),
    }
    del cv
    _reset_and_gc()
    return out


def run_lightcone_suite() -> dict:
    """Run Direction D lightcone + RSD + 1D/2D power spectrum benchmark."""
    inputs = py21cmcu.InputParameters(
        random_seed=4242,
        node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).evolve_input_structs(
        DIM=256,
        HII_DIM=128,
        BOX_LEN=200.0,
        N_THREADS=8,
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
    with py21cmcu.use_mode("fast", use_cuda=True):
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

        # Standalone GPU RSD microbenchmark on 128x128x512 volume
        shape = (128, 128, 512)
        rng = np.random.default_rng(123)
        tb = rng.normal(-20.0, 15.0, size=shape).astype(np.float32)
        vel = rng.normal(0.0, 2.0e-16, size=shape).astype(np.float32)
        tau21 = np.abs(rng.normal(0.02, 0.02, size=shape)).astype(np.float32)
        zs = np.linspace(6.5, 11.0, shape[2])
        _ = include_dvdr_in_tau21(tb[:, :, :16], vel[:, :, :16], zs[:16], inputs, periodic=False, tau_21=tau21[:, :, :16])
        _ = apply_rsds(tb[:, :, :16], vel[:, :, :16], zs[:16], inputs, periodic=False, n_rsd_subcells=4)

        t0 = time.perf_counter()
        dv = include_dvdr_in_tau21(tb, vel, zs, inputs, periodic=False, tau_21=tau21)
        t_dv = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = apply_rsds(dv, vel, zs, inputs, periodic=False, n_rsd_subcells=4)
        t_rsd = time.perf_counter() - t0
    _reset_and_gc()

    return {
        "LC_256_128_z6.5_11_with_rsd_fast": {
            "wall_s": round(t_lc, 3),
            "shape": list(lc.lightcones["brightness_temp_with_rsds"].shape),
            "h2d_gb": round(st["registry_h2d_bytes"] / 1e9, 2),
            "registry_hits": int(st["registry_hits"]),
            "chunk_ps_wall_s": round(t_chunks, 4),
            "n_chunks": len(chunks["chunks"]),
        },
        "GPU_RSD_512slices": {
            "dvdr_s": round(t_dv, 4),
            "apply_rsds_s": round(t_rsd, 4),
        },
    }


def run_batch_suite() -> dict:
    """Run Direction E concurrent single-GPU batch throughput benchmark."""
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
            N_THREADS=4,
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

    out = {}
    for w in (1, 2):
        t0 = time.perf_counter()
        res = py21cmcu.run_batch(
            tasks,
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
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 21cmCU benchmark suites")
    parser.add_argument(
        "--suite",
        default="lightcone,batch",
        help="Comma-separated suites: core, highres, lightcone, batch, all",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    requested = {s.strip().lower() for s in args.suite.split(",") if s.strip()}
    if "all" in requested:
        requested = {"core", "highres", "lightcone", "batch"}

    results: dict = {}
    if "core" in requested:
        results["core"] = run_core_suite()
    if "highres" in requested:
        results["highres"] = run_highres_suite()
    if "lightcone" in requested:
        results["lightcone"] = run_lightcone_suite()
    if "batch" in requested:
        results["batch"] = run_batch_suite()

    text = json.dumps(results, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
