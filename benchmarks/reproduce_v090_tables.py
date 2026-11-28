"""Reproduce v0.9.0 live GPU benchmarks and export Markdown/LaTeX paper tables."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTX_JSON = REPO_ROOT / "benchmarks" / "results_rtx4090_v060.json"
DEFAULT_T4_JSON = REPO_ROOT / "benchmarks" / "results_t4_v090.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "paper" / "tables"


def _fmt_num(val: float, digits: int = 4) -> str:
    return f"{val:.{digits}f}"


def export_tables(rtx_json: Path, t4_json: Path, out_dir: Path) -> None:
    """Export LaTeX and Markdown tables from verified v0.6.0 RTX 4090 and v0.9.0 T4 JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rtx = json.loads(rtx_json.read_text(encoding="utf-8"))
    t4 = json.loads(t4_json.read_text(encoding="utf-8"))

    rtx_rows = [
        (
            r"\texttt{apply\_rsds} (512 slices)",
            f"{rtx['lightcone']['GPU_RSD_512slices']['apply_rsds_cpu_s']:.4f}",
            f"{rtx['lightcone']['GPU_RSD_512slices']['apply_rsds_gpu_s']:.4f}",
            f"{rtx['lightcone']['GPU_RSD_512slices']['apply_rsds_speedup']:.2f}",
        ),
        (
            r"Rectilinear LC ($z=6.5\text{--}11$)",
            f"{rtx['lightcone']['LC_256_128_z6.5_11_with_rsd_fast']['cpu_s']:.3f}",
            f"{rtx['lightcone']['LC_256_128_z6.5_11_with_rsd_fast']['gpu_s']:.3f}",
            f"{rtx['lightcone']['LC_256_128_z6.5_11_with_rsd_fast']['speedup']:.2f}",
        ),
        (
            r"Angular LC ($z=7.8\text{--}9.8$)",
            f"{rtx['class_angular']['AngularLightconer_256_128']['e2e_cpu_s']:.3f}",
            f"{rtx['class_angular']['AngularLightconer_256_128']['e2e_gpu_s']:.3f}",
            f"{rtx['class_angular']['AngularLightconer_256_128']['e2e_speedup']:.2f}",
        ),
        (
            r"\texttt{include\_dvdr\_in\_tau21}",
            f"{rtx['lightcone']['GPU_RSD_512slices']['dvdr_cpu_s']:.3f}",
            f"{rtx['lightcone']['GPU_RSD_512slices']['dvdr_gpu_s']:.4f}",
            f"{rtx['lightcone']['GPU_RSD_512slices']['dvdr_speedup']:.2f}",
        ),
        (
            r"Perturb Field (2LPT, $256^3$)",
            f"{rtx['core']['stage_perturb_256_2lpt']['cpu_s']:.3f}",
            f"{rtx['core']['stage_perturb_256_2lpt']['gpu_s']:.3f}",
            f"{rtx['core']['stage_perturb_256_2lpt']['speedup']:.2f}",
        ),
        (
            r"Coeval 3z (\texttt{L-INT+TS}, 512/256)",
            f"{rtx['core']['B1_256_512_LINT_TS_3z_fast']['cpu_s']:.3f}",
            f"{rtx['core']['B1_256_512_LINT_TS_3z_fast']['gpu_s']:.3f}",
            f"{rtx['core']['B1_256_512_LINT_TS_3z_fast']['speedup']:.2f}",
        ),
        (
            r"Ionize Box ($256^3$, \texttt{E-INT})",
            f"{rtx['core']['stage_ionize_256_eint']['cpu_s']:.3f}",
            f"{rtx['core']['stage_ionize_256_eint']['gpu_s']:.3f}",
            f"{rtx['core']['stage_ionize_256_eint']['speedup']:.2f}",
        ),
        (
            r"\texttt{CLASS+FLUCTS} IC (256/128)",
            f"{rtx['class_angular']['class_cache']['ic_256_128_class_flucts_cpu_s']:.3f}",
            f"{rtx['class_angular']['class_cache']['ic_256_128_class_flucts_gpu_s']:.3f}",
            f"{rtx['class_angular']['class_cache']['ic_speedup']:.2f}",
        ),
        (
            r"EoR Coeval ($768^3$, $z=7.0$)",
            f"{rtx['highres']['B6_768_EoR_fast']['cpu_s']:.3f}",
            f"{rtx['highres']['B6_768_EoR_fast']['gpu_s']:.3f}",
            f"{rtx['highres']['B6_768_EoR_fast']['speedup']:.2f}",
        ),
        (
            r"2LPT IC (512/256)",
            f"{rtx['core']['stage_ic_512_256_2lpt']['cpu_s']:.3f}",
            f"{rtx['core']['stage_ic_512_256_2lpt']['gpu_s']:.3f}",
            f"{rtx['core']['stage_ic_512_256_2lpt']['speedup']:.2f}",
        ),
        (
            r"IC + PF (1024/512, 2LPT lo)",
            f"{rtx['highres']['B3_512_1024_2LPT_lo_fast']['cpu_s']:.3f}",
            f"{rtx['highres']['B3_512_1024_2LPT_lo_fast']['gpu_s']:.3f}",
            f"{rtx['highres']['B3_512_1024_2LPT_lo_fast']['speedup']:.2f}",
        ),
        (
            r"Concurrent Batch ($4\times 192/96$)",
            f"{rtx['batch']['batch_4x_192_96_cpu']['wall_s']:.3f}",
            f"{rtx['batch']['batch_4x_192_96_workers_2']['wall_s']:.3f}",
            f"{rtx['batch']['batch_4x_192_96_workers_2']['speedup_vs_cpu']:.2f}",
        ),
    ]

    tex_rtx = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Workload & CPU 16T [s] & GPU [s] & Speedup \\",
        r"\midrule",
        *[rf"{w} & {c} & {g} & \textbf{{{s}$\times$}} \\" for w, c, g, s in rtx_rows],
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (out_dir / "table_rtx4090_single_gpu.tex").write_text("\n".join(tex_rtx), encoding="utf-8")

    p10_128 = t4["phase10_topology_128"]
    p10_256 = t4["phase10_topology_256"]
    p11 = t4["phase11_multiprobe_256_1M"]
    dr = t4["direction_r_batch_pool_4x2_128_64"]

    t4_rows = [
        (
            "Bispectrum 3D ($128^3$, equil.)",
            _fmt_num(p10_128["bispectrum_3d"]["cpu_s"]),
            _fmt_num(p10_128["bispectrum_3d"]["gpu_s"]),
            f"{p10_128['bispectrum_3d']['speedup']:.2f}",
            r"$\max|\Delta B|/|B|=1.31\times 10^{-6}$",
        ),
        (
            "Bispectrum 3D ($256^3$, equil.)",
            _fmt_num(p10_256["bispectrum_3d"]["cpu_s"]),
            _fmt_num(p10_256["bispectrum_3d"]["gpu_s"]),
            f"{p10_256['bispectrum_3d']['speedup']:.2f}",
            r"$\max|\Delta B|/|B|=8.21\times 10^{-7}$",
        ),
        (
            "MFP BSD ($128^3$, 200k rays)",
            _fmt_num(p10_128["mfp_bsd_200k"]["cpu_s"]),
            _fmt_num(p10_128["mfp_bsd_200k"]["gpu_s"]),
            f"{p10_128['mfp_bsd_200k']['speedup']:.2f}",
            r"$\langle R\rangle$ rel diff $3.66\times 10^{-3}$",
        ),
        (
            "MFP BSD ($256^3$, 200k rays)",
            _fmt_num(p10_256["mfp_bsd_200k"]["cpu_s"]),
            _fmt_num(p10_256["mfp_bsd_200k"]["gpu_s"]),
            f"{p10_256['mfp_bsd_200k']['speedup']:.2f}",
            r"$\langle R\rangle$ rel diff $5.0\times 10^{-4}$",
        ),
        (
            "Minkowski ($128^3$, 15 thresh.)",
            _fmt_num(p10_128["minkowski_3d"]["cpu_s"]),
            _fmt_num(p10_128["minkowski_3d"]["gpu_s"]),
            f"{p10_128['minkowski_3d']['speedup']:.2f}",
            r"Exact Euler $\chi$",
        ),
        (
            "Minkowski ($256^3$, 15 thresh.)",
            _fmt_num(p10_256["minkowski_3d"]["cpu_s"]),
            _fmt_num(p10_256["minkowski_3d"]["gpu_s"]),
            f"{p10_256['minkowski_3d']['speedup']:.2f}",
            r"Exact Euler $\chi$",
        ),
        (
            r"\texttt{cic\_deposit\_3d} ($256^3$, 1M)",
            _fmt_num(p11["cic_256_1M_particles"]["cpu_s"]),
            _fmt_num(p11["cic_256_1M_particles"]["gpu_s"]),
            f"{p11['cic_256_1M_particles']['speedup']:.2f}",
            r"$\max|\Delta\delta|=3.81\times 10^{-6}$",
        ),
        (
            r"\texttt{cross\_power\_21cm\_galaxy} ($256^3$, 1M)",
            _fmt_num(p11["cross_power_256_1M"]["cpu_s"]),
            _fmt_num(p11["cross_power_256_1M"]["gpu_s"]),
            f"{p11['cross_power_256_1M']['speedup']:.2f}",
            r"$\max|\Delta r(k)|=2.91\times 10^{-8}$",
        ),
        (
            r"\texttt{BatchPool} Warm ($4\times 2$, T4)",
            f"{dr['cold_total_s']:.4f} (cold)",
            f"{dr['warm_total_s']:.4f} (warm)",
            f"{dr['speedup']:.3f}",
            rf"Round 4: {dr['warm_round4_s']:.4f}\,s; 1 PID",
        ),
    ]

    tex_t4 = [
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Workload & Baseline [s] & T4 GPU [s] & Speedup & Parity / Notes \\",
        r"\midrule",
        *[rf"{w} & {b} & {g} & \textbf{{{s}$\times$}} & {n} \\" for w, b, g, s, n in t4_rows],
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (out_dir / "table_t4_v090_science.tex").write_text("\n".join(tex_t4), encoding="utf-8")

    md_lines = [
        "# 21cmCU v0.9.0 Reproducible Benchmark Tables",
        "",
        "## Colab Tesla T4 Science & Multi-Probe Suite (`benchmarks/results_t4_v090.json`)",
        "",
        "| Workload | Baseline [s] | T4 GPU [s] | Speedup | Parity / Notes |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            f"| {w} | `{b}` | `{g}` | **`{s}×`** | {n} |"
            for w, b, g, s, n in t4_rows
        ],
        "",
        "## RTX 4090 Single-GPU Core Suite (`benchmarks/results_rtx4090_v060.json`)",
        "",
        "| Workload | CPU 16T [s] | 1× RTX 4090 [s] | Speedup |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {w} | `{c}` | `{g}` | **`{s}×`** |"
            for w, c, g, s in rtx_rows
        ],
        "",
    ]
    (out_dir / "tables_v090.md").write_text("\n".join(md_lines), encoding="utf-8")


def run_live_v090(out_path: Path) -> dict[str, Any]:
    """Run live Phase 10 topology and Phase 11 CIC + cross-power benchmarks on CUDA."""
    import py21cmcu

    if not py21cmcu.cuda_available():
        raise RuntimeError("CUDA device is required for --run-live")

    rng = np.random.default_rng(9501)
    results: dict[str, Any] = {}

    for n in (128, 256):
        white = np.fft.rfftn(rng.normal(0.0, 1.0, size=(n, n, n)))
        g = np.fft.irfftn(white, s=(n, n, n)).astype(np.float32)
        g /= max(float(np.std(g)), 1e-6)
        ng = (g + 0.25 * (g**2 - 1.0)).astype(np.float32)
        ion = (ng > 0.35).astype(np.float32)

        _ = py21cmcu.bispectrum_3d(ng, 128.0, n_bins=8, use_cuda=True)
        t0 = time.perf_counter()
        b_cpu = py21cmcu.bispectrum_3d(ng, 128.0, n_bins=8, use_cuda=False)
        b_cpu_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        b_gpu = py21cmcu.bispectrum_3d(ng, 128.0, n_bins=8, use_cuda=True)
        b_gpu_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        m_cpu = py21cmcu.bubble_size_distribution_mfp(ion, 128.0, n_rays=200_000, use_cuda=False)
        m_cpu_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m_gpu = py21cmcu.bubble_size_distribution_mfp(ion, 128.0, n_rays=200_000, use_cuda=True)
        m_gpu_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_cpu = py21cmcu.minkowski_functionals_3d(ng, 128.0, n_thresholds=15, use_cuda=False)
        k_cpu_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        k_gpu = py21cmcu.minkowski_functionals_3d(ng, 128.0, n_thresholds=15, use_cuda=True)
        k_gpu_s = time.perf_counter() - t0

        valid_b = np.isfinite(b_cpu["B"]) & (np.abs(b_cpu["B"]) > 1e-6)
        rel_b = (
            float(
                np.max(
                    np.abs(b_gpu["B"][valid_b] - b_cpu["B"][valid_b])
                    / np.abs(b_cpu["B"][valid_b])
                )
            )
            if np.any(valid_b)
            else 0.0
        )
        results[f"phase10_topology_{n}"] = {
            "bispectrum_3d": {
                "cpu_s": round(b_cpu_s, 4),
                "gpu_s": round(b_gpu_s, 4),
                "speedup": round(b_cpu_s / max(b_gpu_s, 1e-6), 2),
                "max_rel_diff_B": rel_b,
            },
            "mfp_bsd_200k": {
                "cpu_s": round(m_cpu_s, 4),
                "gpu_s": round(m_gpu_s, 4),
                "speedup": round(m_cpu_s / max(m_gpu_s, 1e-6), 2),
                "mean_r_rel_diff": round(
                    abs(m_gpu["mean_r_mpc"] - m_cpu["mean_r_mpc"])
                    / max(m_cpu["mean_r_mpc"], 1e-6),
                    5,
                ),
            },
            "minkowski_3d": {
                "cpu_s": round(k_cpu_s, 4),
                "gpu_s": round(k_gpu_s, 4),
                "speedup": round(k_cpu_s / max(k_gpu_s, 1e-6), 2),
                "euler_exact": bool(np.array_equal(k_gpu["euler"], k_cpu["euler"])),
            },
        }

    shape = (256, 256, 256)
    box = (256.0, 256.0, 256.0)
    pos = rng.uniform(0.0, 256.0, size=(1_000_000, 3)).astype(np.float32)
    weights = rng.uniform(0.5, 2.0, size=1_000_000).astype(np.float32)
    tb = rng.normal(0.0, 15.0, size=shape).astype(np.float32)

    _ = py21cmcu.cic_deposit_3d(pos[:1024], shape, box, weights=weights[:1024], overdensity=True, use_cuda=True)
    t0 = time.perf_counter()
    g_cpu = py21cmcu.cic_deposit_3d(pos, shape, box, weights=weights, overdensity=True, use_cuda=False)
    cic_cpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_gpu = py21cmcu.cic_deposit_3d(pos, shape, box, weights=weights, overdensity=True, use_cuda=True)
    cic_gpu_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_cpu = py21cmcu.cross_power_21cm_galaxy(tb, pos, box, weights=weights, n_bins=12, use_cuda=False)
    cp_cpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    c_gpu = py21cmcu.cross_power_21cm_galaxy(tb, pos, box, weights=weights, n_bins=12, use_cuda=True)
    cp_gpu_s = time.perf_counter() - t0

    valid_r = np.isfinite(c_cpu["r_k"])
    results["phase11_multiprobe_256_1M"] = {
        "cic_256_1M_particles": {
            "cpu_s": round(cic_cpu_s, 4),
            "gpu_s": round(cic_gpu_s, 4),
            "speedup": round(cic_cpu_s / max(cic_gpu_s, 1e-6), 2),
            "max_abs_diff": float(np.max(np.abs(g_gpu - g_cpu))),
        },
        "cross_power_256_1M": {
            "cpu_s": round(cp_cpu_s, 4),
            "gpu_s": round(cp_gpu_s, 4),
            "speedup": round(cp_cpu_s / max(cp_gpu_s, 1e-6), 2),
            "max_abs_rk_diff": float(np.max(np.abs(c_gpu["r_k"][valid_r] - c_cpu["r_k"][valid_r]))),
        },
    }
    py21cmcu.cleanup_cuda()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    """CLI entrypoint for exporting paper tables or running live v0.9.0 GPU benchmarks."""
    parser = argparse.ArgumentParser(description="Reproduce v0.9.0 tables & GPU benchmarks")
    parser.add_argument("--export-tables", action="store_true", help="Write LaTeX & Markdown tables")
    parser.add_argument("--run-live", action="store_true", help="Execute live GPU benchmarks")
    parser.add_argument("--rtx-json", type=Path, default=DEFAULT_RTX_JSON)
    parser.add_argument("--t4-json", type=Path, default=DEFAULT_T4_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "results_live_v090.json")
    args = parser.parse_args()

    if not args.export_tables and not args.run_live:
        args.export_tables = True

    if args.export_tables:
        export_tables(args.rtx_json, args.t4_json, args.out_dir)
    if args.run_live:
        res = run_live_v090(args.out)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
