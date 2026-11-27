"""Generate RTX 4090 showcase visualizations for the 21cmCU GitHub homepage."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

C_SLATE = "#64748B"
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_PURPLE = "#CC79A7"
C_AMBER = "#E69F00"
C_DARK = "#0F172A"


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


def _eor_cmap() -> mcolors.LinearSegmentedColormap:
    """High-contrast 21cm brightness temperature colormap (deep absorption -> black zero -> hot emission)."""
    colors = [
        (0.00, "#031542"),
        (0.28, "#0284c7"),
        (0.50, "#05070e"),
        (0.75, "#ea580c"),
        (1.00, "#fef08a"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("eor_21cm", colors)


def _plot_3d_cube_cutaway(
    ax: plt.Axes,
    cube: np.ndarray,
    box_len_mpc: float,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    title: str,
) -> None:
    """Render 3 orthogonal exterior faces of a 3D EoR cube in isometric view."""
    nx, ny, nz = cube.shape
    ax.set_facecolor("#0b1020")
    # Subsample faces to <= 64x64 for crisp vector-grid speed
    step_x = max(1, nx // 64)
    step_y = max(1, ny // 64)
    step_z = max(1, nz // 64)

    xs = np.linspace(0.0, box_len_mpc, nx)[::step_x]
    ys = np.linspace(0.0, box_len_mpc, ny)[::step_y]
    zs = np.linspace(0.0, box_len_mpc, nz)[::step_z]

    # Top face: z = max
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    Zg = np.full_like(Xg, box_len_mpc)
    fc_top = cmap(norm(cube[::step_x, ::step_y, -1]))
    ax.plot_surface(
        Xg, Yg, Zg, facecolors=fc_top, rstride=1, cstride=1, shade=False, antialiased=False
    )

    # Front X face: x = max
    Yg2, Zg2 = np.meshgrid(ys, zs, indexing="ij")
    Xg2 = np.full_like(Yg2, box_len_mpc)
    fc_x = cmap(norm(cube[-1, ::step_y, ::step_z]))
    ax.plot_surface(
        Xg2, Yg2, Zg2, facecolors=fc_x, rstride=1, cstride=1, shade=False, antialiased=False
    )

    # Front Y face: y = 0
    Xg3, Zg3 = np.meshgrid(xs, zs, indexing="ij")
    Yg3 = np.zeros_like(Xg3)
    fc_y = cmap(norm(cube[::step_x, 0, ::step_z]))
    ax.plot_surface(
        Xg3, Yg3, Zg3, facecolors=fc_y, rstride=1, cstride=1, shade=False, antialiased=False
    )

    ax.view_init(elev=24, azim=-40)
    ax.set_xlim(0.0, box_len_mpc)
    ax.set_ylim(0.0, box_len_mpc)
    ax.set_zlim(0.0, box_len_mpc)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("X [Mpc]", color="#334155", labelpad=5, fontsize=8.8)
    ax.set_ylabel("Y [Mpc]", color="#334155", labelpad=5, fontsize=8.8)
    ax.set_zlabel("LOS [Mpc]", color="#334155", labelpad=5, fontsize=8.8)
    ax.tick_params(colors="#334155", labelsize=7.8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#475569")
        axis._axinfo["grid"]["color"] = (0.3, 0.35, 0.45, 0.35)
    ax.set_title(title, color=C_DARK, fontsize=11.0, pad=10)


def generate_v070_observables_and_calibration(out_path: Path) -> None:
    """Plot verified v0.7.0 observables speedups and 6-seed strict-vs-fast calibration."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=300, constrained_layout=True)

    # Panel A: v0.7.0 RTX 4090 GPU Speedups (verbatim verified numbers)
    labels = [
        "HERA-350 3D Mock\n(256³, 1000 h)",
        "SKA1-Low 3D Mock\n(256³, 1000 h)",
        "1-Point Moments & PDF\n(256³, hist_diff = 0)",
        "Horizon Wedge Filter\n(256³, cuFFT)",
        "Philox Thermal Noise\n(256³, deterministic)",
    ]
    speedups = [71.88, 71.65, 26.43, 23.86, 10.71]
    details = [
        "22.48 ms vs 1.616 s CPU",
        "23.04 ms vs 1.651 s CPU",
        "single-pass GPU",
        "vs SciPy FFT",
        "vs NumPy Philox",
    ]
    colors = [C_BLUE, C_BLUE, C_TEAL, C_TEAL, C_SLATE]
    y = np.arange(len(labels))
    bars = ax1.barh(y, speedups, color=colors, height=0.56, edgecolor="none")
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Speedup over 16-Core CPU (×)")
    ax1.set_xlim(0, 118)
    ax1.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("v0.7.0 GPU Kernels: Mock Obs & Observables (RTX 4090)")

    for bar, sp, det in zip(bars, speedups, details, strict=True):
        ax1.text(
            sp + 1.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{sp:.2f}× ({det})",
            va="center",
            ha="left",
            fontsize=8.7,
            fontweight="bold",
            color=C_DARK,
        )

    ax1.text(
        0.98,
        0.06,
        "Zero-D2H Lightcone (256/128, 12z L-INTEGRAL):\n1.496 s → 1.169 s  (-21.9% time, -8.5% D2H)",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F1F5F9", "edgecolor": "#94A3B8"},
    )

    # Panel B: 6-Seed strict vs fast statistical parity (verbatim from docs/CALIBRATION.md)
    regimes = [
        "E-INTEGRAL\n(96/192, 192 Mpc)",
        "L-INTEGRAL\n(96/192, 192 Mpc)",
        "CHMF-SAMPLER\n+ MCG + CLASS",
    ]
    med_1d = [0.68, 0.88, 0.92]
    med_2d = [0.70, 0.86, 1.11]
    global_notes = [
        "max|ΔxH|=0.0015\nmax|ΔTb|=0.055 mK",
        "max|ΔxH|=0.0002\nmax|ΔTb|=0.234 mK",
        "max|ΔxH|=0.0006\nmax|ΔTb|=0.372 mK",
    ]
    x = np.arange(len(regimes))
    w = 0.34
    ax2.axhspan(0.0, 1.0, color="#DCFCE7", alpha=0.55, label="1σ Cosmic Variance Band (O(1) SEM)")
    ax2.axhline(1.0, color=C_TEAL, linestyle="--", linewidth=1.1)
    b1 = ax2.bar(x - w / 2, med_1d, width=w, color=C_BLUE, label="1D Δ²(k) Med |Δ| / SEM")
    b2 = ax2.bar(x + w / 2, med_2d, width=w, color=C_ORANGE, label="2D P(k⊥, k∥) Med |Δ| / SEM")

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.025,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    for xi, note in zip(x, global_notes, strict=True):
        ax2.text(
            xi,
            1.21,
            note,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
        )

    ax2.set_xticks(x, regimes)
    ax2.set_ylabel("Median |ΔP| / SEM (6-Seed Ensembles)")
    ax2.set_ylim(0, 1.82)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax2.legend(loc="upper left", ncol=1, frameon=True, fontsize=8.3)
    ax2.set_title("strict vs fast Calibration Parity (6 Seeds, RTX 4090)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_showcases(
    out_dir: Path,
    *,
    use_cuda: bool = True,
    dim: int | None = None,
    hii_dim: int | None = None,
    n_threads: int = 8,
) -> dict[str, float | str]:
    """Run live lightcone & interferometer mocks and render visual figures."""
    import py21cmcu

    _apply_style()

    eff_dim = int(dim) if dim is not None else (256 if use_cuda else 160)
    eff_hii = int(hii_dim) if hii_dim is not None else (128 if use_cuda else 80)
    eff_threads = int(n_threads)
    box_len = 256.0

    n_nodes = 22 if use_cuda else 16
    z_nodes = [float(x) for x in (np.geomspace(21.5, 6.8, n_nodes) - 1.0)]
    inputs = py21cmcu.InputParameters(
        random_seed=42,
        node_redshifts=z_nodes,
    ).evolve_input_structs(
        DIM=eff_dim,
        HII_DIM=eff_hii,
        BOX_LEN=box_len,
        N_THREADS=eff_threads,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=20.0,
        PHOTON_CONS_TYPE="no-photoncons",
        USE_UPPER_STELLAR_TURNOVER=False,
    )
    cosmo = inputs.cosmo_params.cosmo
    res_mpc = box_len / float(eff_hii)
    lcr = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=6.3,
        max_redshift=17.5,
        resolution=res_mpc * u.Mpc,
        quantities=("brightness_temp", "neutral_fraction"),
        cosmo=cosmo,
    )

    mode_str = "fast" if use_cuda else "strict"
    with py21cmcu.use_mode(mode_str, use_cuda=use_cuda, lazy_d2h=use_cuda):
        t0 = time.perf_counter()
        lc = py21cmcu.run_lightcone(
            inputs=inputs,
            lightconer=lcr,
            global_quantities=("brightness_temp", "neutral_fraction"),
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            stream_perturbed_fields=True,
            write=False,
        )
        lc_wall_s = time.perf_counter() - t0

        tb_lc = np.asarray(lc.lightcones["brightness_temp"], dtype=np.float32)
        xh_lc = np.asarray(lc.lightcones["neutral_fraction"], dtype=np.float32)
        dist = np.linspace(0.0, float(lc.lightcone_dimensions[2]), tb_lc.shape[2], dtype=np.float64)
        zs = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        node_z = np.asarray(lc.node_redshifts, dtype=np.float64)
        global_tb = np.asarray(lc.global_quantities["brightness_temp"], dtype=np.float64)
        global_xh = np.asarray(lc.global_quantities["neutral_fraction"], dtype=np.float64)

        # Pick EoR midpoint redshift near xHI ~ 0.5 for interferometer mock
        idx_eor = int(np.argmin(np.abs(global_xh - 0.50)))
        z_mid = float(node_z[idx_eor])

        cv_inputs = py21cmcu.InputParameters(
            random_seed=42,
            node_redshifts=[20.5, 15.0, 12.0, 10.0, z_mid] if z_mid < 9.5 else [15.0, z_mid],
        ).evolve_input_structs(
            DIM=eff_dim,
            HII_DIM=eff_hii,
            BOX_LEN=box_len,
            N_THREADS=eff_threads,
            SOURCE_MODEL="L-INTEGRAL",
            USE_TS_FLUCT=True,
            Z_HEAT_MAX=20.0,
            PHOTON_CONS_TYPE="no-photoncons",
            USE_UPPER_STELLAR_TURNOVER=False,
        )
        cv = py21cmcu.run_coeval(inputs=cv_inputs, out_redshifts=[z_mid], write=False)[0]
        tb_box = np.asarray(cv.brightness_temp, dtype=np.float32)
        xh_box_mean = float(np.mean(np.asarray(cv.neutral_fraction, dtype=np.float32)))

        cfg_ska = py21cmcu.get_instrument_preset("SKA1-Low").replace(t_int_hours=1000.0)
        cfg_hera = py21cmcu.get_instrument_preset("HERA-350").replace(t_int_hours=1000.0)
        if use_cuda:
            _ = py21cmcu.mock_observation_3d(tb_box, box_len, z_mid, cfg_ska, seed=2026, use_cuda=True)

        t1 = time.perf_counter()
        mock_ska = py21cmcu.mock_observation_3d(
            tb_box, box_len, z_mid, cfg_ska, seed=2026, use_cuda=use_cuda
        )
        ska_ms = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        mock_hera = py21cmcu.mock_observation_3d(
            tb_box, box_len, z_mid, cfg_hera, seed=2026, use_cuda=use_cuda
        )
        hera_ms = (time.perf_counter() - t2) * 1e3

        edges_2d = np.logspace(np.log10(0.05), np.log10(0.85), 9)
        ps2d_ska = py21cmcu.power_spectrum_2d(
            mock_ska,
            box_len,
            kperp_edges=edges_2d,
            kpara_edges=edges_2d,
            remove_mean=True,
            use_cuda=use_cuda,
        )
        fc_ska = py21cmcu.forecast_power_spectrum_error(
            tb_box.shape, box_len, z_mid, cfg_ska, signal_field=tb_box, n_bins=10, use_cuda=use_cuda
        )
        fc_hera = py21cmcu.forecast_power_spectrum_error(
            tb_box.shape, box_len, z_mid, cfg_hera, signal_field=tb_box, n_bins=10, use_cuda=use_cuda
        )
        slope = float(py21cmcu.horizon_slope(z_mid, cosmo=cosmo))

    # --- Figure 1: Lightcone Showcase + Global History + 3D Cutaway ---
    eor_cmap = _eor_cmap()
    vmin_tb = float(max(np.percentile(tb_lc, 0.5), -140.0))
    vmax_tb = float(min(max(np.percentile(tb_lc, 99.5), 20.0), 45.0))
    norm_tb = mcolors.TwoSlopeNorm(vmin=vmin_tb, vcenter=0.0, vmax=vmax_tb)

    fig = plt.figure(figsize=(14.0, 9.6), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.35], hspace=0.40, wspace=0.22)

    ax_tb = fig.add_subplot(gs[0, :])
    im_tb = ax_tb.imshow(
        tb_lc[:, tb_lc.shape[1] // 2, :],
        origin="lower",
        aspect="auto",
        extent=[dist[0], dist[-1], 0.0, box_len],
        cmap=eor_cmap,
        norm=norm_tb,
    )
    ax_tb.set_ylabel("Transverse Y [Mpc]")
    lc_tag = f"{lc_wall_s:.2f} s on 1× RTX 4090" if use_cuda else f"{int(box_len)} Mpc box"
    ax_tb.set_title(
        f"21cm Brightness Temperature Lightcone δTb(x, z) — L-INTEGRAL + Ts + RSD "
        f"({tb_lc.shape[0]}×{tb_lc.shape[1]}×{tb_lc.shape[2]} voxels, {lc_tag})",
        pad=26,
    )
    cbar_tb = fig.colorbar(im_tb, ax=ax_tb, pad=0.015, aspect=16)
    cbar_tb.set_label("δTb [mK]")

    # Top redshift ticks
    ax_top = ax_tb.twiny()
    ax_top.set_xlim(ax_tb.get_xlim())
    z_ticks = [z for z in (6.5, 7.5, 8.5, 10.0, 12.0, 14.5, 17.0) if zs.min() <= z <= zs.max()]
    if zs[0] <= zs[-1]:
        z_pos = np.interp(z_ticks, zs, dist)
    else:
        z_pos = np.interp(z_ticks, zs[::-1], dist[::-1])
    ax_top.set_xticks(z_pos, [f"{z:.1f}" for z in z_ticks])
    ax_top.set_xlabel("Redshift z", labelpad=5)

    ax_xh = fig.add_subplot(gs[1, :], sharex=ax_tb)
    im_xh = ax_xh.imshow(
        xh_lc[:, xh_lc.shape[1] // 2, :],
        origin="lower",
        aspect="auto",
        extent=[dist[0], dist[-1], 0.0, box_len],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax_xh.set_xlabel("Comoving Lightcone Distance [Mpc]")
    ax_xh.set_ylabel("Transverse Y [Mpc]")
    ax_xh.set_title("Neutral Hydrogen Fraction Lightcone xHI(x, z) — Reionization Bubble Percolation")
    cbar_xh = fig.colorbar(im_xh, ax=ax_xh, pad=0.015, aspect=16)
    cbar_xh.set_label("xHI")

    # Bottom-Left: Global History
    ax_hist = fig.add_subplot(gs[2, 0])
    order = np.argsort(node_z)
    z_ord = node_z[order]
    l1 = ax_hist.plot(
        z_ord, global_tb[order], color=C_ORANGE, lw=2.4, marker="o", ms=3.5, label="⟨δTb⟩ [mK]"
    )
    ax_hist.axhline(0.0, color="#94A3B8", lw=0.9, ls=":")
    ax_hist.set_xlabel("Redshift z")
    ax_hist.set_ylabel("Mean Brightness Temp ⟨δTb⟩ [mK]", color=C_ORANGE)
    ax_hist.tick_params(axis="y", labelcolor=C_ORANGE)
    ax_hist.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax_hist.spines["top"].set_visible(False)

    ax_hist_r = ax_hist.twinx()
    l2 = ax_hist_r.plot(
        z_ord, global_xh[order], color=C_TEAL, lw=2.4, ls="--", marker="s", ms=3.5, label="⟨xHI⟩"
    )
    ax_hist_r.set_ylabel("Neutral Fraction ⟨xHI⟩", color=C_TEAL)
    ax_hist_r.tick_params(axis="y", labelcolor=C_TEAL)
    ax_hist_r.set_ylim(-0.05, 1.08)
    ax_hist_r.spines["top"].set_visible(False)
    lines = l1 + l2
    ax_hist.legend(lines, [ln.get_label() for ln in lines], loc="lower left", fontsize=8.8)
    ax_hist.set_title("Global Cosmic Dawn & Reionization Evolution")

    # Bottom-Right: 3D Cutaway Cube from EoR box at xHI ~ 0.5
    ax_3d = fig.add_subplot(gs[2, 1], projection="3d")
    _plot_3d_cube_cutaway(
        ax_3d,
        tb_box,
        box_len,
        eor_cmap,
        norm_tb,
        f"3D EoR Volume Cutaway (z = {z_mid:.2f}, ⟨xHI⟩ = {xh_box_mean:.2f})",
    )

    fig.savefig(out_dir / "lightcone_showcase.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Interferometer Mocks, 2D Wedge PS & 1D Forecast ---
    fig2 = plt.figure(figsize=(14.0, 8.2), dpi=300, constrained_layout=True)
    gs2 = fig2.add_gridspec(2, 3, height_ratios=[1.0, 1.05])

    sl = tb_box.shape[2] // 2
    tb_fluct = tb_box[:, :, sl] - float(np.mean(tb_box[:, :, sl]))
    lim = float(
        max(
            abs(np.percentile(tb_fluct, 2.0)),
            abs(np.percentile(tb_fluct, 98.0)),
            15.0,
        )
    )
    ska_title = (
        f"SKA1-Low 1000 h GPU Mock ({ska_ms:.1f} ms)"
        if use_cuda
        else "SKA1-Low 1000 h Mock (UV + PSF + Wedge + Noise)"
    )
    hera_title = (
        f"HERA-350 1000 h GPU Mock ({hera_ms:.1f} ms)"
        if use_cuda
        else "HERA-350 1000 h Mock (UV + PSF + Wedge + Noise)"
    )
    panels = [
        (tb_fluct, f"True δTb − ⟨δTb⟩ (z = {z_mid:.2f}, ⟨xHI⟩ = {xh_box_mean:.2f})", lim),
        (mock_ska[:, :, sl], ska_title, 10.0),
        (mock_hera[:, :, sl], hera_title, 25.0),
    ]
    for col, (arr2d, ptitle, plim) in enumerate(panels):
        axp = fig2.add_subplot(gs2[0, col])
        im = axp.imshow(
            arr2d,
            origin="lower",
            extent=[0.0, box_len, 0.0, box_len],
            cmap="RdBu_r",
            vmin=-plim,
            vmax=plim,
        )
        axp.set_title(ptitle, fontsize=10.2)
        axp.set_xlabel("X [Mpc]")
        if col == 0:
            axp.set_ylabel("Y [Mpc]")
        else:
            axp.set_yticklabels([])
        cb = fig2.colorbar(im, ax=axp, shrink=0.86, pad=0.02)
        cb.set_label("δTb [mK]", fontsize=8.5)

    gs_bot = gs2[1, :].subgridspec(1, 2, wspace=0.22)
    ax_2d = fig2.add_subplot(gs_bot[0, 0])
    kp_edges = ps2d_ska["kperp_edges"]
    kz_edges = ps2d_ska["kpara_edges"]
    pk_vals = ps2d_ska["pk"]
    pk2d = np.where(np.isfinite(pk_vals), np.log10(np.maximum(pk_vals, 1e-1)), np.nan)
    pos_vals = pk_vals[np.isfinite(pk_vals) & (pk_vals > 1e-1)]
    vmax_2d = float(np.max(np.log10(pos_vals))) if pos_vals.size else 5.0
    im2d = ax_2d.pcolormesh(
        kp_edges, kz_edges, pk2d.T, cmap="inferno", vmin=-1.0, vmax=vmax_2d, shading="auto"
    )
    kp_line = np.linspace(kp_edges[0], kp_edges[-1], 200)
    kz_wedge = slope * kp_line + 0.05 * cosmo.h
    ax_2d.plot(kp_line, kz_wedge, color="cyan", ls="--", lw=2.0, label="Horizon Wedge + 0.05 h/Mpc")
    ax_2d.set_xscale("log")
    ax_2d.set_yscale("log")
    ax_2d.set_xlim(kp_edges[0], kp_edges[-1])
    ax_2d.set_ylim(kz_edges[0], kz_edges[-1])
    ticks_k = [0.05, 0.1, 0.2, 0.4, 0.8]
    ticks_vis = [t for t in ticks_k if kp_edges[0] <= t <= kp_edges[-1]]
    ax_2d.set_xticks(ticks_vis, [f"{t:g}" for t in ticks_vis])
    ax_2d.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_2d.set_xlabel("k⊥ [Mpc⁻¹]")
    ax_2d.set_ylabel("k∥ [Mpc⁻¹]")
    ax_2d.set_title(f"2D Cylindrical Power Spectrum (SKA1-Low Mock, z = {z_mid:.2f})")
    ax_2d.legend(loc="lower right", fontsize=8.5, facecolor="#0F172A", labelcolor="white")
    cbar_2d = fig2.colorbar(im2d, ax=ax_2d, pad=0.02)
    cbar_2d.set_label("log₁₀ P(k⊥, k∥) [mK² Mpc³]")

    ax_1d = fig2.add_subplot(gs_bot[0, 1])
    k_vals = fc_ska["k"]
    sig_d2 = fc_ska["signal_delta_sq"]
    err_ska = fc_ska["sigma_delta_sq"]
    err_hera = fc_hera["sigma_delta_sq"]
    valid = np.isfinite(k_vals) & np.isfinite(sig_d2) & (sig_d2 > 0)
    kv = k_vals[valid]
    sv = sig_d2[valid]

    ax_1d.loglog(kv, sv, color=C_DARK, lw=2.4, marker="o", ms=4.5, label="True 21cm Δ²(k)")
    v_s = valid & np.isfinite(err_ska) & (err_ska > 0)
    ax_1d.fill_between(
        k_vals[v_s],
        np.maximum(sig_d2[v_s] - err_ska[v_s], 1e-3),
        sig_d2[v_s] + err_ska[v_s],
        color=C_BLUE,
        alpha=0.25,
        label=f"SKA1-Low 1000 h ±1σ (SNR = {fc_ska['total_snr']:.1f})",
    )
    ax_1d.loglog(k_vals[v_s], err_ska[v_s], color=C_BLUE, ls="--", lw=1.6, label="SKA1-Low 1σ Noise")

    v_h = valid & np.isfinite(err_hera) & (err_hera > 0)
    ax_1d.loglog(
        k_vals[v_h],
        err_hera[v_h],
        color=C_ORANGE,
        ls=":",
        lw=2.0,
        label=f"HERA-350 1000 h 1σ Noise (SNR = {fc_hera['total_snr']:.1f})",
    )
    ax_1d.set_xlabel("k [Mpc⁻¹]")
    ax_1d.set_ylabel("Δ²₂₁(k) [mK²]")
    ax_1d.set_title(f"1D Spherical Power Spectrum & Sensitivity Forecast (z = {z_mid:.2f})")
    ax_1d.grid(True, which="both", ls="--", alpha=0.35, color="#CBD5E1")
    ax_1d.spines["top"].set_visible(False)
    ax_1d.spines["right"].set_visible(False)
    ax_1d.legend(loc="best", fontsize=8.4)

    fig2.savefig(out_dir / "interferometer_mock_showcase.png", bbox_inches="tight")
    plt.close(fig2)

    if py21cmcu.cuda_available():
        py21cmcu.cleanup_cuda()
    return {
        "backend": "cuda" if use_cuda else "cpu",
        "grid": f"{eff_dim}/{eff_hii}",
        "lc_shape": str(tb_lc.shape),
        "lc_wall_s": round(lc_wall_s, 3),
        "z_mid": round(z_mid, 3),
        "xh_mid": round(xh_box_mean, 4),
        "ska_ms": round(ska_ms, 2),
        "hera_ms": round(hera_ms, 2),
        "ska_snr_1d": round(float(fc_ska["total_snr"]), 2),
        "hera_snr_1d": round(float(fc_hera["total_snr"]), 2),
    }


def main() -> None:
    """CLI entrypoint for showcase figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--hii-dim", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate_v070_observables_and_calibration(args.out_dir / "v070_observables_and_calibration.png")
    print(f"Wrote {args.out_dir / 'v070_observables_and_calibration.png'}")
    if not args.metrics_only:
        metrics = run_showcases(
            args.out_dir,
            use_cuda=not args.cpu,
            dim=args.dim,
            hii_dim=args.hii_dim,
            n_threads=args.threads,
        )
        print("Showcase Metrics:", metrics)


if __name__ == "__main__":
    main()
