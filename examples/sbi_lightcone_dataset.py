"""Example: Generate an SBI-ready 21cm lightcone summary dataset with GPU acceleration."""

from __future__ import annotations

import argparse
from pathlib import Path

import py21cmcu


def build_sbi_dataset(
    out_h5: Path,
    *,
    n_samples: int = 8,
    dim: int = 192,
    hii_dim: int = 96,
    box_len: float = 192.0,
    devices: tuple[int, ...] = (0,),
    workers_per_gpu: int = 2,
    save_lightcones_dir: Path | None = None,
) -> None:
    """Run a Sobol sweep over astrophysical parameters with wedge + thermal noise."""
    bounds = {
        "F_STAR10_ACG": (-1.8, -0.7),
        "F_ESC10_ACG": (-1.8, -0.5),
        "ALPHA_STAR_ACG": (0.2, 0.8),
        "M_TURN": (8.0, 9.5),
    }
    param_rows = py21cmcu.generate_sobol_parameter_grid(bounds, n_samples=n_samples, seed=42)

    tasks: list[py21cmcu.BatchTask] = []
    for idx, row in enumerate(param_rows):
        inp = py21cmcu.InputParameters(
            random_seed=int(row["random_seed"]),
            node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0],
        ).evolve_input_structs(
            DIM=dim,
            HII_DIM=hii_dim,
            BOX_LEN=box_len,
            N_THREADS=4,
            SOURCE_MODEL="L-INTEGRAL",
            USE_TS_FLUCT=True,
            Z_HEAT_MAX=10.0,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
            F_STAR10_ACG=float(row["F_STAR10_ACG"]),
            F_ESC10_ACG=float(row["F_ESC10_ACG"]),
            ALPHA_STAR_ACG=float(row["ALPHA_STAR_ACG"]),
            M_TURN=float(row["M_TURN"]),
        )
        lc_path = (
            str(save_lightcones_dir / f"lc_{idx:04d}.h5")
            if save_lightcones_dir is not None
            else None
        )
        tasks.append(
            py21cmcu.BatchTask(
                task_id=f"sobol_{idx:04d}",
                inputs=inp,
                mode="lightcone",
                min_redshift=6.5,
                max_redshift=9.5,
                cuda_mode="fast",
                apply_rsds=True,
                ps_n_bins=12,
                compute_ps_2d=True,
                ps_2d_n_kperp=6,
                ps_2d_n_kpara=6,
                wedge_filter=True,
                wedge_buffer=0.05,
                noise_sigma_mk=2.0,
                lightcone_out_path=lc_path,
                lightcone_compression="device-lzf",
            )
        )

    py21cmcu.run_batch(
        tasks,
        devices=devices,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=4,
        output_h5=out_h5,
        checkpoint_h5=out_h5,
        resume=True,
        use_cuda=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SBI 21cm lightcone summary dataset")
    parser.add_argument("--out", type=Path, default=Path("sbi_dataset.h5"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hii-dim", type=int, default=96)
    parser.add_argument("--box-len", type=float, default=192.0)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--save-lightcones-dir", type=Path, default=None)
    ns = parser.parse_args()
    devs = tuple(int(x.strip()) for x in ns.devices.split(",") if x.strip())
    build_sbi_dataset(
        ns.out,
        n_samples=ns.samples,
        dim=ns.dim,
        hii_dim=ns.hii_dim,
        box_len=ns.box_len,
        devices=devs,
        workers_per_gpu=ns.workers_per_gpu,
        save_lightcones_dir=ns.save_lightcones_dir,
    )
