"""Statistical calibration benchmark comparing 21cmCU strict vs fast precision modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import py21cmcu


def _regime_configs() -> list[dict]:
    node_eor = list(
        py21cmcu.get_logspaced_redshifts(
            min_redshift=6.5, max_redshift=15.0, z_step_factor=1.15
        )
    )
    node_mcg = list(
        py21cmcu.get_logspaced_redshifts(
            min_redshift=7.5, max_redshift=15.0, z_step_factor=1.15
        )
    )
    return [
        {
            "name": "E-INTEGRAL",
            "seeds": [6101, 6102, 6103, 6104, 6105, 6106],
            "node_z": node_eor,
            "sample_indices": [0, len(node_eor) // 2, -1],
            "evolve": {
                "DIM": 192,
                "HII_DIM": 96,
                "BOX_LEN": 192.0,
                "N_THREADS": 8,
                "SOURCE_MODEL": "E-INTEGRAL",
                "USE_TS_FLUCT": True,
                "Z_HEAT_MAX": 15.0,
                "USE_EXP_FILTER": False,
                "USE_UPPER_STELLAR_TURNOVER": False,
                "PHOTON_CONS_TYPE": "no-photoncons",
            },
        },
        {
            "name": "L-INTEGRAL",
            "seeds": [6201, 6202, 6203, 6204, 6205, 6206],
            "node_z": node_eor,
            "sample_indices": [0, len(node_eor) // 2, -1],
            "evolve": {
                "DIM": 192,
                "HII_DIM": 96,
                "BOX_LEN": 192.0,
                "N_THREADS": 8,
                "SOURCE_MODEL": "L-INTEGRAL",
                "USE_TS_FLUCT": True,
                "Z_HEAT_MAX": 15.0,
                "USE_UPPER_STELLAR_TURNOVER": False,
                "PHOTON_CONS_TYPE": "no-photoncons",
            },
        },
        {
            "name": "CHMF-SAMPLER+MCG+CLASS",
            "seeds": [6301, 6302, 6303, 6304, 6305, 6306],
            "node_z": node_mcg,
            "sample_indices": [0, len(node_mcg) // 2, -1],
            "evolve": {
                "DIM": 192,
                "HII_DIM": 64,
                "BOX_LEN": 150.0,
                "N_THREADS": 8,
                "SOURCE_MODEL": "CHMF-SAMPLER",
                "POWER_SPECTRUM": "CLASS",
                "USE_MCGS": True,
                "RECOMB_MODEL": "inhomogeneous",
                "V_CB_MODEL": "FLUCTS",
                "USE_TS_FLUCT": True,
                "Z_HEAT_MAX": 15.0,
                "PHOTON_CONS_TYPE": "no-photoncons",
            },
        },
    ]


def _run_regime(cfg: dict, mode: str) -> dict:
    xh_all = []
    tb_all = []
    ts_all = []
    ps_all = []  # (n_seeds, n_z, n_bins)
    ps2_all = []  # (n_seeds, n_z, n_kp, n_kz)
    k_ref = None
    out_redshifts = None
    walls = []

    for seed in cfg["seeds"]:
        inp = py21cmcu.InputParameters(
            random_seed=seed, node_redshifts=cfg["node_z"]
        ).evolve_input_structs(**cfg["evolve"])
        py21cmcu.cleanup_cuda()
        t0 = time.perf_counter()
        with py21cmcu.use_mode(mode, use_cuda=True):
            coevals = py21cmcu.run_coeval(
                inputs=inp,
                regenerate=True,
                write=False,
            )
            sampled = [coevals[i] for i in cfg["sample_indices"]]
            box_len = float(inp.simulation_options.BOX_LEN)
            k_edges = np.geomspace(0.08, 1.0, 13, dtype=np.float64)
            specs = [
                py21cmcu.power_spectrum_1d(
                    c.brightness_temp, box_len, k_edges=k_edges, use_cuda=True
                )
                for c in sampled
            ]
            specs2 = [
                py21cmcu.power_spectrum_2d(
                    c.brightness_temp, box_len, n_kperp=6, n_kpara=6, use_cuda=True
                )
                for c in sampled
            ]
        walls.append(time.perf_counter() - t0)
        py21cmcu.cleanup_cuda()

        if out_redshifts is None:
            out_redshifts = [round(float(c.redshift), 4) for c in sampled]
        xh_all.append([float(np.mean(c.neutral_fraction)) for c in sampled])
        tb_all.append([float(np.mean(c.brightness_temp)) for c in sampled])
        ts_all.append([float(np.mean(c.ts_box.spin_temperature)) for c in sampled])
        ps_all.append([sp["delta_sq"] for sp in specs])
        ps2_all.append([sp["pk"] for sp in specs2])
        if k_ref is None:
            k_ref = specs[0]["k"]

    return {
        "wall_mean_s": float(np.mean(walls)),
        "wall_total_s": float(np.sum(walls)),
        "out_redshifts": out_redshifts,
        "xH": np.asarray(xh_all, dtype=np.float64),
        "Tb": np.asarray(tb_all, dtype=np.float64),
        "Ts": np.asarray(ts_all, dtype=np.float64),
        "delta_sq": np.asarray(ps_all, dtype=np.float64),
        "pk_2d": np.asarray(ps2_all, dtype=np.float64),
        "k": np.asarray(k_ref, dtype=np.float64),
    }


def _summarize_comparison(cfg: dict, r_strict: dict, r_fast: dict) -> dict:
    dxh = np.abs(np.mean(r_fast["xH"], axis=0) - np.mean(r_strict["xH"], axis=0))
    dtb = np.abs(np.mean(r_fast["Tb"], axis=0) - np.mean(r_strict["Tb"], axis=0))
    dts_rel = np.abs(
        np.mean(r_fast["Ts"], axis=0) - np.mean(r_strict["Ts"], axis=0)
    ) / np.maximum(np.mean(r_strict["Ts"], axis=0), 1e-12)

    ps_s_mean = np.mean(r_strict["delta_sq"], axis=0)
    ps_f_mean = np.mean(r_fast["delta_sq"], axis=0)
    ps_s_sem = np.std(r_strict["delta_sq"], axis=0, ddof=1) / np.sqrt(len(cfg["seeds"]))
    valid = np.isfinite(ps_s_mean) & (ps_s_mean > 0.0) & np.isfinite(ps_f_mean)
    rel_diff = np.abs(ps_f_mean[valid] - ps_s_mean[valid]) / ps_s_mean[valid]
    z_score = np.abs(ps_f_mean[valid] - ps_s_mean[valid]) / np.maximum(
        ps_s_sem[valid], 1e-12
    )

    ps2_s_mean = np.mean(r_strict["pk_2d"], axis=0)
    ps2_f_mean = np.mean(r_fast["pk_2d"], axis=0)
    ps2_s_sem = np.std(r_strict["pk_2d"], axis=0, ddof=1) / np.sqrt(len(cfg["seeds"]))
    v2 = np.isfinite(ps2_s_mean) & (ps2_s_mean > 0.0) & np.isfinite(ps2_f_mean)
    rel_diff_2d = np.abs(ps2_f_mean[v2] - ps2_s_mean[v2]) / ps2_s_mean[v2]
    z_score_2d = np.abs(ps2_f_mean[v2] - ps2_s_mean[v2]) / np.maximum(
        ps2_s_sem[v2], 1e-12
    )

    return {
        "regime": cfg["name"],
        "n_seeds": len(cfg["seeds"]),
        "out_redshifts": r_strict["out_redshifts"],
        "strict_wall_mean_s": round(r_strict["wall_mean_s"], 3),
        "fast_wall_mean_s": round(r_fast["wall_mean_s"], 3),
        "speedup": round(r_strict["wall_mean_s"] / max(r_fast["wall_mean_s"], 1e-6), 3),
        "max_dxH": float(np.max(dxh)),
        "max_dTb_mK": float(np.max(dtb)),
        "max_rel_dTs": float(np.max(dts_rel)),
        "ps_rel_rms": float(np.sqrt(np.mean(rel_diff**2))),
        "ps_rel_max": float(np.max(rel_diff)),
        "ps_median_z_score": float(np.median(z_score)),
        "ps2d_rel_rms": float(np.sqrt(np.mean(rel_diff_2d**2))),
        "ps2d_median_z_score": float(np.median(z_score_2d)),
        "strict_xH_mean": [float(v) for v in np.mean(r_strict["xH"], axis=0)],
        "fast_xH_mean": [float(v) for v in np.mean(r_fast["xH"], axis=0)],
        "strict_Tb_mean": [float(v) for v in np.mean(r_strict["Tb"], axis=0)],
        "fast_Tb_mean": [float(v) for v in np.mean(r_fast["Tb"], axis=0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate 21cmCU strict vs fast modes")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    summaries = []
    for cfg in _regime_configs():
        print(f"Running calibration regime: {cfg['name']} ...", flush=True)
        rs = _run_regime(cfg, "strict")
        rf = _run_regime(cfg, "fast")
        s = _summarize_comparison(cfg, rs, rf)
        summaries.append(s)
        print(
            f"  {s['regime']}: strict={s['strict_wall_mean_s']:.3f}s fast={s['fast_wall_mean_s']:.3f}s "
            f"({s['speedup']:.2f}x) | max|dxH|={s['max_dxH']:.5f} max|dTb|={s['max_dTb_mK']:.4f}mK "
            f"ps1d_rms={s['ps_rel_rms']:.4f} (med |dz|/sem={s['ps_median_z_score']:.2f}) "
            f"ps2d_rms={s['ps2d_rel_rms']:.4f}",
            flush=True,
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
