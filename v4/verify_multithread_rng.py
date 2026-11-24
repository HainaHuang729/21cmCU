"""Verify bitwise GSL PRNG, multi-thread ICs, and 8-thread lightcone parity in 21cmFAST 4.3.2."""

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field
from py21cmfast.lightconers import RectilinearLightconer

from compare_lightcone_ps import compare_saved_slices
from ic_random import RNG_ORDER, Stream, gaussian, seeds, stream_module


class _GslRngType(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("max", ctypes.c_ulong),
        ("min", ctypes.c_ulong),
        ("size", ctypes.c_size_t),
        ("set", ctypes.c_void_p),
        ("get", ctypes.c_void_p),
        ("get_double", ctypes.c_void_p),
    ]


class _GslRng(ctypes.Structure):
    _fields_ = [("type", ctypes.POINTER(_GslRngType)), ("state", ctypes.c_void_p)]


def _load_gsl():
    ctypes.CDLL("libgslcblas.so", mode=ctypes.RTLD_GLOBAL)
    gsl = ctypes.CDLL("libgsl.so.25")
    gsl.gsl_rng_alloc.argtypes = [ctypes.POINTER(_GslRngType)]
    gsl.gsl_rng_alloc.restype = ctypes.POINTER(_GslRng)
    gsl.gsl_rng_set.argtypes = [ctypes.POINTER(_GslRng), ctypes.c_ulong]
    gsl.gsl_rng_get.argtypes = [ctypes.POINTER(_GslRng)]
    gsl.gsl_rng_get.restype = ctypes.c_ulong
    gsl.gsl_ran_ugaussian.argtypes = [ctypes.POINTER(_GslRng)]
    gsl.gsl_ran_ugaussian.restype = ctypes.c_double
    gsl.gsl_rng_free.argtypes = [ctypes.POINTER(_GslRng)]
    gsl.gsl_ran_choose.argtypes = [
        ctypes.POINTER(_GslRng),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    gsl.gsl_ran_shuffle.argtypes = [
        ctypes.POINTER(_GslRng),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    return gsl


def check_stream_words_and_gaussians(word_count=100_000, gauss_count=10_000):
    """Compare raw uint32 words and Box-Muller Gaussians for all 5 GSL generators."""
    gsl = _load_gsl()
    mod = stream_module()
    sym_map = {
        "mt": "gsl_rng_mt19937",
        "gfsr": "gsl_rng_gfsr4",
        "cmrg": "gsl_rng_cmrg",
        "mrg": "gsl_rng_mrg",
        "taus2": "gsl_rng_taus2",
    }
    test_seeds = (0, 1, 424242, 1850877074)
    blocks = (word_count + 623) // 624
    results = {}
    for kind in RNG_ORDER:
        t = ctypes.POINTER(_GslRngType).in_dll(gsl, sym_map[kind])
        per_seed = {}
        for seed in test_seeds:
            r = gsl.gsl_rng_alloc(t)
            try:
                gsl.gsl_rng_set(r, seed)
                c_words = np.array(
                    [gsl.gsl_rng_get(r) for _ in range(word_count)], dtype=np.uint32
                )
                gsl.gsl_rng_set(r, seed)
                c_gauss = np.array(
                    [gsl.gsl_ran_ugaussian(r) for _ in range(gauss_count)],
                    dtype=np.float64,
                )
            finally:
                gsl.gsl_rng_free(r)

            stream = Stream(mod, kind, seed)
            t0 = perf_counter()
            g_words = stream.integers(blocks)[:word_count].get()
            cp.cuda.runtime.deviceSynchronize()
            t_words = perf_counter() - t0

            g_stream = Stream(mod, kind, seed)
            g_gauss = gaussian(g_stream, gauss_count).get()

            word_mismatches = int(np.sum(c_words != g_words))
            gauss_max_abs = float(np.max(np.abs(c_gauss - g_gauss)))
            gauss_fp64_ulp_diffs = int(
                np.sum(c_gauss.view(np.uint64) != g_gauss.view(np.uint64))
            )
            assert word_mismatches == 0, (
                f"{kind} seed={seed}: {word_mismatches} word mismatches"
            )
            assert gauss_max_abs <= 1e-14, (
                f"{kind} seed={seed}: Gaussian max_abs={gauss_max_abs} exceeds 1e-14"
            )
            per_seed[str(seed)] = {
                "words_tested": word_count,
                "word_mismatches": word_mismatches,
                "gaussians_tested": gauss_count,
                "gauss_fp64_ulp_diffs": gauss_fp64_ulp_diffs,
                "gauss_max_abs": gauss_max_abs,
                "stream_seconds": t_words,
            }
        results[kind] = per_seed

    # Also verify seeds() helper against gsl_ran_choose + gsl_ran_shuffle for N_THREADS = 1..16
    mt_t = ctypes.POINTER(_GslRngType).in_dll(gsl, "gsl_rng_mt19937")
    num_int = (2**31 - 1) // 16
    many = np.arange(num_int, dtype=np.uint32)
    seed_checks = {}
    for cnt in (1, 2, 4, 5, 8, 16):
        mismatches = 0
        for seed in test_seeds:
            r = gsl.gsl_rng_alloc(mt_t)
            try:
                gsl.gsl_rng_set(r, seed)
                c_seeds = np.zeros(cnt, dtype=np.uint32)
                gsl.gsl_ran_choose(
                    r,
                    c_seeds.ctypes.data_as(ctypes.c_void_p),
                    cnt,
                    many.ctypes.data_as(ctypes.c_void_p),
                    num_int,
                    4,
                )
                gsl.gsl_ran_shuffle(r, c_seeds.ctypes.data_as(ctypes.c_void_p), cnt, 4)
            finally:
                gsl.gsl_rng_free(r)
            py_seeds = seeds(mod, seed, cnt)
            if c_seeds.tolist() != py_seeds:
                mismatches += 1
        assert mismatches == 0, f"seeds() mismatch for cnt={cnt}"
        seed_checks[str(cnt)] = {
            "seeds_tested": len(test_seeds),
            "mismatches": mismatches,
        }
    return {"stream_checks": results, "thread_seed_checks": seed_checks}


def check_ic_multithread(
    thread_counts=(1, 2, 4, 5, 8), dim=64, hii_dim=32, box_len=100.0
):
    """Compare CPU vs GPU ComputeInitialConditions across N_THREADS."""
    from gpu_lightcone import _backend

    reports = {}
    for threads in thread_counts:
        inputs = p21.InputParameters(random_seed=424242).evolve_input_structs(
            SOURCE_MODEL="E-INTEGRAL",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            USE_METALLICITY=False,
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            CELL_RECOMB=False,
            DIM=dim,
            HII_DIM=hii_dim,
            BOX_LEN=box_len,
            N_THREADS=threads,
            POWER_SPECTRUM="EH",
            PERTURB_ALGORITHM="2LPT",
            USE_FFTW_WISDOM=False,
        )
        lightconer = RectilinearLightconer.between_redshifts(
            min_redshift=10.0,
            max_redshift=11.0,
            resolution=inputs.simulation_options.cell_size,
        )
        t0 = perf_counter()
        ic_cpu = p21.compute_initial_conditions(
            inputs=inputs, write=False, regenerate=True
        )
        cpu_s = perf_counter() - t0
        with _backend(inputs, lightconer):
            t1 = perf_counter()
            ic_gpu = p21.compute_initial_conditions(
                inputs=inputs, write=False, regenerate=True
            )
            gpu_s = perf_counter() - t1

        fields = {}
        for key in sorted(ic_cpu.arrays):
            if not ic_cpu.arrays[key].state.computed_in_mem:
                continue
            a = np.ascontiguousarray(ic_cpu.get(key))
            b = np.ascontiguousarray(ic_gpu.get(key))
            diff = int(np.sum(a.view(np.uint32) != b.view(np.uint32)))
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            assert diff == 0, (
                f"N_THREADS={threads} IC {key}: {diff} bit diffs (max_abs={max_abs})"
            )
            fields[key] = {
                "values": int(a.size),
                "unequal_bits": diff,
                "max_abs": max_abs,
            }
        reports[str(threads)] = {
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
            "fields": fields,
        }
    return reports


def check_8thread_lightcone(work_dir: Path, point_path: Path):
    """Compare 8-thread CPU vs GPU on 192^3 -> 64^3 (200 Mpc, z=6..15, E-INTEGRAL, strict_fft=True)."""
    from lightcone_pair import benchmark_case
    from lightcone_ps import k_grid
    from gpu_lightcone import _backend

    lc_dir = work_dir / "lc_8t"
    lc_dir.mkdir(parents=True, exist_ok=True)
    target_k = k_grid(point_path)
    base_inputs, lightconer = benchmark_case(192)
    inputs = base_inputs.evolve_input_structs(N_THREADS=8)

    def run_side(mode):
        rows = []
        xp = cp if mode == "gpu" else np
        n, box = inputs.simulation_options.HII_DIM, inputs.simulation_options.BOX_LEN
        k1 = xp.fft.fftfreq(n, d=box / n) * 2 * xp.pi
        radii = xp.sqrt(
            k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + k1[None, None, :] ** 2
        )
        edges = xp.linspace(0, radii.max() * 1.0000001, 61)
        centers = (edges[:-1] + edges[1:]) / 2
        bins = xp.clip(xp.digitize(radii.ravel(), edges) - 1, 0, 59)
        count = xp.maximum(xp.bincount(bins, minlength=60), 1)
        factor = xp.asarray(centers**3 / (2 * np.pi**2))
        target_xp = xp.asarray(target_k)

        diag_dir = lc_dir / "routing"
        t_total_0 = perf_counter()
        with ExitStack() as stack:
            if mode == "gpu":
                stack.enter_context(_backend(inputs, lightconer, diag_dir))
            t_ic0 = perf_counter()
            ic = p21.compute_initial_conditions(
                inputs=inputs, write=False, regenerate=True
            )
            ic_s = perf_counter() - t_ic0
            digest = hashlib.sha256()
            for key in sorted(ic.arrays):
                if ic.arrays[key].state.computed_in_mem:
                    digest.update(key.encode())
                    digest.update(
                        memoryview(np.ascontiguousarray(ic.get(key))).cast("B")
                    )
            t_evo0 = perf_counter()
            for _, z, coeval, cone in p21.generate_lightcone(
                lightconer=lightconer,
                inputs=inputs,
                initial_conditions=ic,
                write=False,
                regenerate=True,
            ):
                if coeval is None:
                    continue
                field = xp.asarray(coeval.brightness_temp, dtype=xp.float64)
                fk = xp.fft.fftn(field - field.mean()) / n**3
                power = xp.abs(fk) ** 2 * box**3
                spec = (
                    xp.bincount(bins, weights=power.ravel(), minlength=60)
                    / count
                    * factor
                )
                rows.append(
                    {"z": float(z), "ps": xp.interp(target_xp, centers, spec).tolist()}
                )
            evo_s = perf_counter() - t_evo0
            h5_path = lc_dir / f"{mode}.h5"
            t_save0 = perf_counter()
            cone.save(h5_path, clobber=True)
            with h5_path.open("rb") as f:
                os.fsync(f.fileno())
            save_s = perf_counter() - t_save0
        total_s = perf_counter() - t_total_0
        (lc_dir / f"{mode}-ps.json").write_text(
            json.dumps(
                {"k": target_k.tolist(), "rows": rows, "ic_hash": digest.hexdigest()},
                indent=2,
            )
        )
        return {
            "ic_seconds": ic_s,
            "evolution_seconds": evo_s,
            "save_fsync_seconds": save_s,
            "total_seconds": total_s,
            "ic_hash": digest.hexdigest(),
        }

    # Warm GPU kernels/shims once before timing GPU cleanly
    run_side("gpu")
    gpu_stats = run_side("gpu")
    cpu_stats = run_side("cpu")
    assert gpu_stats["ic_hash"] == cpu_stats["ic_hash"], "8-thread IC SHA-256 mismatch!"

    slice_cmp = compare_saved_slices(lc_dir / "cpu.h5", lc_dir / "gpu.h5")
    for grp in ("lightcones", "global_quantities"):
        for k, v in slice_cmp[grp].items():
            assert v["failed"] == 0, f"8-thread {grp}/{k} failed: {v}"
    assert slice_cmp["brightness_transverse_power"]["failed"] == 0, slice_cmp[
        "brightness_transverse_power"
    ]

    ps_cpu = np.asarray(
        [r["ps"] for r in json.loads((lc_dir / "cpu-ps.json").read_text())["rows"]]
    )
    ps_gpu = np.asarray(
        [r["ps"] for r in json.loads((lc_dir / "gpu-ps.json").read_text())["rows"]]
    )
    ps_diff = np.abs(ps_cpu - ps_gpu)
    ps_bad = int(np.sum(ps_diff > 1e-8 + 1e-4 * np.abs(ps_cpu)))
    assert ps_bad == 0, f"8-thread brightness_power failures: {ps_bad}"

    return {
        "grid": "192^3 / 64^3",
        "box_Mpc": 200,
        "threads": 8,
        "nodes": len(inputs.node_redshifts),
        "ic_hash": cpu_stats["ic_hash"],
        "cpu": cpu_stats,
        "gpu": gpu_stats,
        "speedup_total": cpu_stats["total_seconds"] / gpu_stats["total_seconds"],
        "lightcones": slice_cmp["lightcones"],
        "global_quantities_tested": sum(
            v["tested"] for v in slice_cmp["global_quantities"].values()
        ),
        "global_quantities_failed": sum(
            v["failed"] for v in slice_cmp["global_quantities"].values()
        ),
        "brightness_power_bins_tested": int(ps_cpu.size),
        "brightness_power_failed": ps_bad,
        "brightness_power_max_abs": float(ps_diff.max()),
        "brightness_transverse_power": {
            k: slice_cmp["brightness_transverse_power"][k]
            for k in ("slices", "tested", "failed", "max_abs")
        },
        "gpu_stages": json.loads((lc_dir / "routing" / "gpu-stages.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = {
        "scope": "Verify bitwise CMRG/MRG/TAUS2/MT/GFSR4 CUDA PRNG streams, N_THREADS in {1,2,4,5,8} ICs, and 8-thread 192->64 E-INTEGRAL lightcone parity in 21cmFAST 4.3.2.",
        "host": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    out.update(check_stream_words_and_gaussians())
    out["ic_multithread_64"] = check_ic_multithread()
    out["lightcone_192_8t"] = check_8thread_lightcone(args.work_dir, args.point)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("ALL MULTITHREAD RNG + 8-THREAD LIGHTCONE CHECKS PASSED")
    print(
        json.dumps(
            {
                "ic_threads_tested": list(out["ic_multithread_64"].keys()),
                "cpu_s": out["lightcone_192_8t"]["cpu"]["total_seconds"],
                "gpu_s": out["lightcone_192_8t"]["gpu"]["total_seconds"],
                "speedup": out["lightcone_192_8t"]["speedup_total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
