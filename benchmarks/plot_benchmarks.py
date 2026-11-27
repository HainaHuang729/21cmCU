"""Generate publication-grade benchmark figures from verbatim JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe Okabe-Ito palette
C_SLATE = "#64748B"
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_PURPLE = "#CC79A7"
C_AMBER = "#E69F00"
C_DARK = "#1E293B"
C_GRID = "#E2E8F0"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "xtick.color": C_DARK,
            "ytick.color": C_DARK,
            "text.color": C_DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def plot_speedup_overview(data: dict, out_path: Path) -> None:
    """Plot CPU (16T) vs 1x RTX 4090 wall-time (left) and speedup factor (right)."""
    core = data["core"]
    hi = data["highres"]
    lc = data["lightcone"]
    ca = data["class_angular"]
    batch = data["batch"]

    rows = [
        (
            "apply_rsds (512 slices)",
            lc["GPU_RSD_512slices"]["apply_rsds_cpu_s"],
            lc["GPU_RSD_512slices"]["apply_rsds_gpu_s"],
            lc["GPU_RSD_512slices"]["apply_rsds_speedup"],
        ),
        (
            "Rectilinear LC + RSD (256/128)",
            lc["LC_256_128_z6.5_11_with_rsd_fast"]["cpu_s"],
            lc["LC_256_128_z6.5_11_with_rsd_fast"]["gpu_s"],
            lc["LC_256_128_z6.5_11_with_rsd_fast"]["speedup"],
        ),
        (
            "Angular LC + RSD (256/128)",
            ca["AngularLightconer_256_128"]["e2e_cpu_s"],
            ca["AngularLightconer_256_128"]["e2e_gpu_s"],
            ca["AngularLightconer_256_128"]["e2e_speedup"],
        ),
        (
            "include_dvdr (512 slices)",
            lc["GPU_RSD_512slices"]["dvdr_cpu_s"],
            lc["GPU_RSD_512slices"]["dvdr_gpu_s"],
            lc["GPU_RSD_512slices"]["dvdr_speedup"],
        ),
        (
            "Perturb Field (256^3, 2LPT)",
            core["stage_perturb_256_2lpt"]["cpu_s"],
            core["stage_perturb_256_2lpt"]["gpu_s"],
            core["stage_perturb_256_2lpt"]["speedup"],
        ),
        (
            "Coeval 3z L-INT+TS (512/256)",
            core["B1_256_512_LINT_TS_3z_fast"]["cpu_s"],
            core["B1_256_512_LINT_TS_3z_fast"]["gpu_s"],
            core["B1_256_512_LINT_TS_3z_fast"]["speedup"],
        ),
        (
            "Ionize Box (256^3, E-INT)",
            core["stage_ionize_256_eint"]["cpu_s"],
            core["stage_ionize_256_eint"]["gpu_s"],
            core["stage_ionize_256_eint"]["speedup"],
        ),
        (
            "CLASS+FLUCTS IC (256/128)",
            ca["class_cache"]["ic_256_128_class_flucts_cpu_s"],
            ca["class_cache"]["ic_256_128_class_flucts_gpu_s"],
            ca["class_cache"]["ic_speedup"],
        ),
        (
            "768^3 EoR Coeval (E-INT)",
            hi["B6_768_EoR_fast"]["cpu_s"],
            hi["B6_768_EoR_fast"]["gpu_s"],
            hi["B6_768_EoR_fast"]["speedup"],
        ),
        (
            "IC 512/256 (2LPT, fast)",
            core["stage_ic_512_256_2lpt"]["cpu_s"],
            core["stage_ic_512_256_2lpt"]["gpu_s"],
            core["stage_ic_512_256_2lpt"]["speedup"],
        ),
        (
            "1024/512 IC + PF (2LPT lo)",
            hi["B3_512_1024_2LPT_lo_fast"]["cpu_s"],
            hi["B3_512_1024_2LPT_lo_fast"]["gpu_s"],
            hi["B3_512_1024_2LPT_lo_fast"]["speedup"],
        ),
        (
            "Batch 4x Coeval (192/96, w=2)",
            batch["batch_4x_192_96_cpu"]["wall_s"],
            batch["batch_4x_192_96_workers_2"]["wall_s"],
            batch["batch_4x_192_96_workers_2"]["speedup_vs_cpu"],
        ),
    ]

    # Plot top-to-bottom from fastest relative speedup down
    labels = [r[0] for r in rows][::-1]
    cpu_t = np.array([r[1] for r in rows][::-1])
    gpu_t = np.array([r[2] for r in rows][::-1])
    speedups = np.array([r[3] for r in rows][::-1])
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )

    h = 0.36
    ax1.barh(y + h / 2, cpu_t, height=h, color=C_SLATE, alpha=0.88, label="CPU (16 OpenMP threads)")
    ax1.barh(y - h / 2, gpu_t, height=h, color=C_BLUE, alpha=0.95, label="GPU (1x RTX 4090, fast)")
    ax1.set_xscale("log")
    ax1.set_xlim(0.01, max(cpu_t) * 5.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_xlabel("Wall-Clock Time [s] (log scale, lower is better)")
    ax1.set_title("(a) Wall-Clock Execution Time", pad=24)
    ax1.grid(axis="x", linestyle="--", alpha=0.55, color=C_GRID)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CBD5E1",
        fontsize=8.8,
    )

    for yi, ct, gt in zip(y, cpu_t, gpu_t, strict=True):
        c_str = f"{ct:.2f}s" if ct < 10 else f"{ct:.1f}s"
        g_str = f"{gt:.3f}s" if gt < 0.2 else (f"{gt:.2f}s" if gt < 10 else f"{gt:.1f}s")
        ax1.text(ct * 1.12, yi + h / 2, c_str, va="center", ha="left", fontsize=8.0, color=C_SLATE)
        ax1.text(gt * 1.12, yi - h / 2, g_str, va="center", ha="left", fontsize=8.0, color=C_BLUE, fontweight="bold")

    ax2.barh(y, speedups, height=0.56, color=C_TEAL, alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_xlim(1.0, max(speedups) * 3.8)
    ax2.axvline(1.0, color=C_DARK, linewidth=0.9, linestyle=":")
    ax2.set_xlabel("Speedup Factor (CPU / GPU, log scale)")
    ax2.set_title("(b) Single-GPU Speedup over 16-Core CPU", pad=24)
    ax2.grid(axis="x", linestyle="--", alpha=0.55, color=C_GRID)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for yi, sp in zip(y, speedups, strict=True):
        lbl = f"{sp:.1f}x" if sp >= 10 else f"{sp:.2f}x"
        ax2.text(sp * 1.12, yi, lbl, va="center", ha="left", fontsize=8.8, fontweight="bold", color=C_DARK)

    meta = data["metadata"]
    fig.suptitle(
        f"21cmCU Single-GPU Performance Overview (Node {meta['hostname']}: "
        f"1x {meta['gpu_names'][0]} vs {meta['bench_cpu_threads']} CPU Threads)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.21, right=0.97, top=0.83, bottom=0.11, wspace=0.07)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_mgpu_scaling(data: dict, out_path: Path) -> None:
    """Plot Direction G+K multi-GPU slab decomposition scaling & batch throughput."""
    mgpu = data["mgpu"]
    ic1024 = mgpu["ic_1024"]
    ic1536 = mgpu["ic_1536"]
    ic1792 = mgpu["ic_1792"]
    ic2048 = mgpu["ic_2048"]
    batch16 = mgpu["batch_16x_192_96"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15.8, 5.2), gridspec_kw={"width_ratios": [1.32, 1.08, 1.0]}
    )

    # Panel (a): Out-of-single-GPU capacity envelope (1024^3 1-GPU up to 2048^3 3-GPU native)
    configs = [
        "1024^3\n(1.07B)\n1 GPU",
        "1536^3\n(3.62B)\n2 GPUs",
        "1536^3\n(3.62B)\n3 GPUs",
        "1792^3\n(5.75B)\n3 GPUs",
        "2048^3\n(8.59B)\n3 GPUs",
    ]
    zeld_s = [
        ic1024["ZELDOVICH_g1"]["ic_s"],
        ic1536["ZELDOVICH_g2"]["ic_s"],
        ic1536["ZELDOVICH_g3"]["ic_s"],
        ic1792["ZELDOVICH_g3"]["ic_s"],
        ic2048["ZELDOVICH_g3"]["ic_s"],
    ]
    lpt2_s = [
        ic1024["2LPT_g1"]["ic_s"],
        ic1536["2LPT_g2"]["ic_s"],
        ic1536["2LPT_g3"]["ic_s"],
        ic1792["2LPT_g3"]["ic_s"],
    ]
    x = np.arange(len(configs))
    w = 0.35
    x_zeld = np.array([*(x[:4] - w / 2), float(x[4])])
    b1 = ax1.bar(x_zeld, zeld_s, width=w, color=C_BLUE, label="Zel'dovich IC (native VRAM)")
    b2 = ax1.bar(x[:4] + w / 2, lpt2_s, width=w, color=C_ORANGE, label="2LPT IC (hybrid pinned stream)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8.4)
    ax1.set_ylabel("IC Wall-Clock Time [s]")
    ax1.set_ylim(0, max(lpt2_s) * 1.46)
    ax1.set_title("(a) Large-Grid Native Multi-GPU IC")
    ax1.grid(axis="y", linestyle="--", alpha=0.55, color=C_GRID)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", frameon=True, edgecolor="#CBD5E1", fontsize=8.2)

    for rect in [*b1, *b2]:
        val = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            val + 2.0,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    nat_2048 = ic2048["ZELDOVICH_g3"]
    ax1.text(
        0.98,
        0.97,
        f"2048^3 (8.59B) Zel'dovich: native in-VRAM\n"
        f"3x RTX 4090 ({nat_2048['ic_s']:.1f}s IC, 3.71x vs v0.5.0)\n"
        f"2LPT needs >=235 GB host RAM",
        transform=ax1.transAxes,
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
        ha="right",
        va="top",
    )

    # Panel (b): PCIe host-staged vs direct scaling at 1024^3 (1,2,3 GPUs) & 1536^3 (2,3 GPUs)
    g1024 = [1, 2, 3]
    t1024_z = [ic1024[f"ZELDOVICH_g{g}"]["ic_s"] for g in g1024]
    t1024_2 = [ic1024[f"2LPT_g{g}"]["ic_s"] for g in g1024]
    g1536 = [2, 3]
    t1536_z = [ic1536[f"ZELDOVICH_g{g}"]["ic_s"] for g in g1536]
    t1536_2 = [ic1536[f"2LPT_g{g}"]["ic_s"] for g in g1536]

    ax2.plot(g1024, t1024_z, "o-", color=C_BLUE, linewidth=2.0, markersize=7, label="1024^3 Zel'dovich")
    ax2.plot(g1024, t1024_2, "s--", color=C_ORANGE, linewidth=2.0, markersize=6.5, label="1024^3 2LPT")
    ax2.plot(g1536, t1536_z, "o-", color=C_TEAL, linewidth=2.3, markersize=7.5, label="1536^3 Zel'dovich")
    ax2.plot(g1536, t1536_2, "s--", color=C_PURPLE, linewidth=2.3, markersize=7, label="1536^3 2LPT")
    ax2.set_yscale("log")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["1 GPU\n(24 GB)", "2 GPUs\n(48 GB)", "3 GPUs\n(72 GB)"], fontsize=9)
    ax2.set_ylim(2.2, 210.0)
    ax2.set_ylabel("IC Wall-Clock Time [s] (log scale)")
    ax2.set_title("(b) Slab Transpose Scaling vs Capacity")
    ax2.grid(True, linestyle="--", alpha=0.55, color=C_GRID)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        loc="upper left",
        frameon=True,
        edgecolor="#CBD5E1",
        fontsize=8.0,
    )

    for g, v in zip(g1536, t1536_z, strict=True):
        ax2.text(g, v * 1.16, f"{v:.1f}s", ha="center", va="bottom", fontsize=8.0, color=C_TEAL, fontweight="bold")
    for g, v in zip(g1536, t1536_2, strict=True):
        ax2.text(g, v * 1.15, f"{v:.1f}s", ha="center", va="bottom", fontsize=8.0, color=C_PURPLE, fontweight="bold")
    ax2.text(1.06, t1024_z[0] * 0.72, f"{t1024_z[0]:.2f}s", fontsize=8.0, color=C_BLUE, fontweight="bold")

    # Panel (c): Multi-GPU run_batch throughput (16 x 192/96 L-INTEGRAL+TS, workers_per_gpu=2)
    bg = [1, 2, 3]
    tp = [batch16[f"batch_16x_g{g}"]["throughput_per_hour"] for g in bg]
    sp = [batch16[f"batch_16x_g{g}"]["speedup_vs_1gpu"] for g in bg]
    bars = ax3.bar(bg, tp, width=0.52, color=[C_BLUE, C_TEAL, C_ORANGE], alpha=0.92)
    ax3.set_xticks(bg)
    ax3.set_xticklabels(["1 GPU\n(2 workers)", "2 GPUs\n(4 workers)", "3 GPUs\n(6 workers)"], fontsize=9)
    ax3.set_ylabel("Throughput [tasks / hour]")
    ax3.set_ylim(0, max(tp) * 1.24)
    ax3.set_title("(c) Batch Ensemble (16 x 192/96 Coeval)")
    ax3.grid(axis="y", linestyle="--", alpha=0.55, color=C_GRID)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    for rect, thr, s in zip(bars, tp, sp, strict=True):
        ax3.text(
            rect.get_x() + rect.get_width() / 2,
            thr + 220,
            f"{thr:,.0f}/h\n({s:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=C_DARK,
        )

    fig.suptitle(
        "Direction G+K Multi-GPU Slab Decomposition & Concurrent Ensemble Scaling (Up to 3x RTX 4090)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.15, wspace=0.26)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residency_and_io(data: dict, out_path: Path) -> None:
    """Plot Direction I CUDA_LAZY_D2H residency savings (left) & DeviceLZF HDF5 I/O (right)."""
    io_lazy = data["io_lazy"]
    res = io_lazy["residency_B1_3z"]
    hdf = io_lazy["hdf5_io_256x256x512"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.28}
    )

    # Panel (a): Eager vs Lazy D2H on B1 (512/256, 3z)
    modes = ["Eager D2H\n(Default)", "Lazy D2H\n(CUDA_LAZY_D2H=1)"]
    d2h_gb = [res["eager_d2h"]["d2h_mb"] / 1000.0, res["lazy_d2h"]["d2h_mb"] / 1000.0]
    walls = [res["eager_d2h"]["run_wall_s"], res["lazy_d2h"]["run_wall_s"]]

    x = np.arange(len(modes))
    bars_d2h = ax1.bar(x - 0.17, d2h_gb, width=0.32, color=C_BLUE, label="D2H Transfer Volume [GB]")
    ax1_r = ax1.twinx()
    bars_t = ax1_r.bar(x + 0.17, walls, width=0.32, color=C_TEAL, label="Coeval Run Time [s]")

    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=9.8)
    ax1.set_ylabel("Host D2H Transfer [GB]", color=C_BLUE)
    ax1_r.set_ylabel("Wall-Clock Time [s]", color=C_TEAL)
    ax1.set_ylim(0, max(d2h_gb) * 1.28)
    ax1_r.set_ylim(0, max(walls) * 1.28)
    ax1.set_title("(a) GPU Residency D2H Reduction (512/256, 3z)")
    ax1.grid(axis="y", linestyle="--", alpha=0.55, color=C_GRID)
    ax1.spines["top"].set_visible(False)
    ax1_r.spines["top"].set_visible(False)

    d2h_save = 100.0 * (1.0 - d2h_gb[1] / d2h_gb[0])
    t_save = 100.0 * (1.0 - walls[1] / walls[0])
    for i, rect in enumerate(bars_d2h):
        lbl = f"{d2h_gb[i]:.2f} GB" if i == 0 else f"{d2h_gb[i]:.2f} GB\n(-{d2h_save:.1f}%)"
        ax1.text(rect.get_x() + rect.get_width() / 2, d2h_gb[i] + 0.15, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BLUE)
    for i, rect in enumerate(bars_t):
        lbl = f"{walls[i]:.2f}s" if i == 0 else f"{walls[i]:.2f}s\n(-{t_save:.1f}%)"
        ax1_r.text(rect.get_x() + rect.get_width() / 2, walls[i] + 0.10, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_TEAL)

    ax1.text(
        0.5,
        0.92,
        f"Both modes: 0.00 GB H2D, {res['lazy_d2h']['registry_hits']} registry hits",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    # Panel (b): HDF5 Lightcone Write/Read throughput & size ratio
    keys = ["none", "cpu_lzf", "device_lzf"]
    k_labels = [
        f"Uncompressed\n({hdf['none']['compressed_mb']:.0f} MB, 1.00x)",
        f"CPU h5py LZF\n({hdf['cpu_lzf']['compressed_mb']:.1f} MB, {hdf['cpu_lzf']['ratio']:.2f}x)",
        f"GPU DeviceLZF\n({hdf['device_lzf']['compressed_mb']:.1f} MB, {hdf['device_lzf']['ratio']:.2f}x)",
    ]
    w_tp = [hdf[k]["write_mb_s"] for k in keys]
    r_tp = [hdf[k]["read_mb_s"] for k in keys]
    x2 = np.arange(len(keys))
    w = 0.34
    bw = ax2.bar(x2 - w / 2, w_tp, width=w, color=C_ORANGE, label="HDF5 Write [MB/s raw]")
    br = ax2.bar(x2 + w / 2, r_tp, width=w, color=C_SLATE, label="HDF5 Read [MB/s raw]")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(k_labels, fontsize=9.0)
    ax2.set_ylabel("Throughput [MB/s raw]")
    ax2.set_ylim(0, max(w_tp) * 1.22)
    ax2.set_title("(b) Lightcone HDF5 I/O (256x256x512, 256 MB raw)")
    ax2.grid(axis="y", linestyle="--", alpha=0.55, color=C_GRID)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=True, edgecolor="#CBD5E1", fontsize=8.8)

    for rect, val in zip([*bw, *br], [*w_tp, *r_tp], strict=True):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            val + 25,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )
    speedup_lzf = hdf["device_lzf"]["write_mb_s"] / hdf["cpu_lzf"]["write_mb_s"]
    ax2.annotate(
        f"{speedup_lzf:.2f}x faster write\nthan CPU LZF",
        xy=(2 - w / 2, w_tp[2] + 95),
        xytext=(1.42, max(w_tp) * 0.62),
        arrowprops={"arrowstyle": "->", "color": C_DARK, "lw": 1.1},
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7ED", "edgecolor": C_ORANGE},
    )

    fig.suptitle(
        "Direction I Residency Transfer Reduction & DeviceLZF HDF5 Compression (1x RTX 4090)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.16, wspace=0.32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 21cmCU benchmark figures")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("benchmarks/results_rtx4090_v050.json"),
        help="Input benchmark JSON file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Output figure directory",
    )
    args = parser.parse_args()

    _apply_style()
    data = json.loads(args.json.read_text())
    plot_speedup_overview(data, args.out_dir / "speedup_overview.png")
    plot_mgpu_scaling(data, args.out_dir / "mgpu_scaling.png")
    plot_residency_and_io(data, args.out_dir / "residency_and_io.png")


if __name__ == "__main__":
    main()
