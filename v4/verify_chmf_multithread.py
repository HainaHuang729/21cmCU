"""Verify CHMF-SAMPLER across N_THREADS >= 1 (ordered catalogs, parity, and 96^3 speed)."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from unittest.mock import patch

import cupy as cp
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)
from py21cmfast.lightconers import RectilinearLightconer

from compare_lightcone_ps import compare_saved_slices
from gpu_lightcone import run_lightcone
from ic_random import RNG_ORDER
from verify_chmf_lightcone import FIELDS, THERMAL_FIELDS, record_catalogs


def _expected_rng_counters(threads: int) -> list[str]:
    return [f"halo_cuda_{RNG_ORDER[t % 5]}_words" for t in range(threads)]


def _run_case(
    name: str,
    work_dir: Path,
    *,
    dim: int,
    hii_dim: int,
    box_len: float,
    nodes: list[float],
    bounds: tuple[float, float],
    threads: int,
    seed: int,
    cuda_prefix: bool,
):
    case_dir = work_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    inputs = p21.InputParameters(
        random_seed=seed, node_redshifts=nodes
    ).evolve_input_structs(
        SOURCE_MODEL="CHMF-SAMPLER",
        USE_UPPER_STELLAR_TURNOVER=True,
        DIM=dim,
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        N_THREADS=threads,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        HII_FILTER="spherical-tophat",
        USE_EXP_FILTER=True,
        CELL_RECOMB=True,
        USE_METALLICITY=True,
        Z_HEAT_MAX=float(nodes[0]),
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=bounds[0],
        max_redshift=bounds[1],
        quantities=FIELDS + THERMAL_FIELDS,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    env_patch = {
        "P21_CUDA_GRID_SAMPLE": "1" if cuda_prefix else "0",
        "P21_CUDA_PROGENITOR": "1" if cuda_prefix else "0",
    }
    results = {}
    cones = {}

    with patch.dict(os.environ, env_patch):
        for mode in ("cpu", "gpu"):
            manager.free()
            cats = []
            with ExitStack() as stack:
                record_catalogs(stack, cats)
                t0 = perf_counter()
                if mode == "gpu":
                    cone = run_lightcone(
                        inputs=inputs,
                        lightconer=lc,
                        diagnostics_dir=case_dir / "routing",
                        fallback_to_cpu=False,
                        write=False,
                        regenerate=True,
                    )
                else:
                    cone = p21.run_lightcone(
                        inputs=inputs,
                        lightconer=lc,
                        write=False,
                        regenerate=True,
                    )
                elapsed = perf_counter() - t0
            h5_path = case_dir / f"{mode}.h5"
            t_save = perf_counter()
            cone.save(h5_path, clobber=True)
            with h5_path.open("rb") as f:
                os.fsync(f.fileno())
            save_s = perf_counter() - t_save
            cones[mode] = cone
            results[mode] = {
                "elapsed_seconds": elapsed,
                "save_fsync_seconds": save_s,
                "catalogs": cats,
            }

        manager.free()
        t_warm = perf_counter()
        _ = run_lightcone(
            inputs=inputs,
            lightconer=lc,
            diagnostics_dir=case_dir / "routing_warm",
            fallback_to_cpu=False,
            write=False,
            regenerate=True,
        )
        warm_gpu_s = perf_counter() - t_warm

    cmp_res = compare_saved_slices(case_dir / "cpu.h5", case_dir / "gpu.h5")
    gs = json.loads((case_dir / "routing" / "gpu-stages.json").read_text())
    br = json.loads((case_dir / "routing" / "backend-routing.json").read_text())
    counters = {**gs, **br["counters"]}

    required = [
        "ionization_halo_barrier",
        "thermal_update",
        *_expected_rng_counters(threads),
    ]
    if cuda_prefix:
        required.append("halo_cuda_progenitor_conditions")
    missing = [k for k in required if counters.get(k, 0) < 1]

    failed_lc = {
        k: v["failed"] for k, v in cmp_res["lightcones"].items() if v["failed"] != 0
    }
    failed_gq = {
        k: v["failed"]
        for k, v in cmp_res["global_quantities"].items()
        if v["failed"] != 0
    }
    failed_ps = cmp_res["brightness_transverse_power"]["failed"]
    catalogs_equal = results["cpu"]["catalogs"] == results["gpu"]["catalogs"]
    passed = (
        catalogs_equal
        and not missing
        and not failed_lc
        and not failed_gq
        and failed_ps == 0
    )

    return {
        "name": name,
        "dim": dim,
        "hii_dim": hii_dim,
        "box_len_Mpc": box_len,
        "threads": threads,
        "cuda_prefix": cuda_prefix,
        "cpu_seconds": results["cpu"]["elapsed_seconds"],
        "gpu_seconds": results["gpu"]["elapsed_seconds"],
        "warm_gpu_seconds": warm_gpu_s,
        "speedup_warm": results["cpu"]["elapsed_seconds"] / warm_gpu_s,
        "total_halos": sum(r["n"] for r in results["cpu"]["catalogs"]),
        "catalogs_equal": catalogs_equal,
        "missing_counters": missing,
        "failed_lightcones": failed_lc,
        "failed_global_quantities": failed_gq,
        "failed_transverse_ps": failed_ps,
        "tb_max_abs": cmp_res["lightcones"]["brightness_temp"]["max_abs"],
        "sampling_seconds": {
            "grid": counters.get("halo_cuda_grid_sample_sampling_seconds", 0.0),
            "progenitor": counters.get("halo_cuda_progenitor_sampling_seconds", 0.0),
            "property_batches": counters.get("halo_property_batches", 0),
        },
        "passed": passed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--suite",
        choices=("all", "small", "medium"),
        default="all",
    )
    args = p.parse_args()

    cases = []
    if args.suite in ("all", "small"):
        for threads in (1, 2, 4, 8):
            for cuda_prefix in (False, True):
                suffix = "prefix" if cuda_prefix else "hybrid"
                cases.append(
                    _run_case(
                        f"chmf_32_{threads}t_{suffix}",
                        args.work_dir,
                        dim=32,
                        hii_dim=16,
                        box_len=24.0,
                        nodes=[12.0, 11.5, 11.0],
                        bounds=(11.1, 11.9),
                        threads=threads,
                        seed=129887,
                        cuda_prefix=cuda_prefix,
                    )
                )
    if args.suite in ("all", "medium"):
        for threads in (1, 2, 8):
            cases.append(
                _run_case(
                    f"chmf_96_{threads}t_hybrid",
                    args.work_dir,
                    dim=96,
                    hii_dim=32,
                    box_len=84.0,
                    nodes=[11.5, 11.0, 10.5, 10.0, 9.5],
                    bounds=(9.6, 11.3),
                    threads=threads,
                    seed=81033,
                    cuda_prefix=False,
                )
            )
        cases.append(
            _run_case(
                "chmf_96_1t_prefix",
                args.work_dir,
                dim=96,
                hii_dim=32,
                box_len=84.0,
                nodes=[11.5, 11.0, 10.5, 10.0, 9.5],
                bounds=(9.6, 11.3),
                threads=1,
                seed=81033,
                cuda_prefix=True,
            )
        )

    out = {
        "scope": "CHMF-SAMPLER N_THREADS >= 1 catalog/lightcone parity and 96^3 throughput verification.",
        "host": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for c in cases:
        print(
            f"{c['name']}: passed={c['passed']} cats_eq={c['catalogs_equal']} "
            f"cpu={c['cpu_seconds']:.2f}s warm_gpu={c['warm_gpu_seconds']:.2f}s "
            f"speedup={c['speedup_warm']:.2f}x tb_max={c['tb_max_abs']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
