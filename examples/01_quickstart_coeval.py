"""Minimal quickstart: run a single-GPU 21cm coeval box + 1D power spectrum."""

from __future__ import annotations

import argparse
import tempfile
import time

import numpy as np

import py21cmcu as p21


def main() -> None:
    parser = argparse.ArgumentParser(description="21cmCU GPU quickstart coeval example")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hii-dim", type=int, default=64)
    parser.add_argument("--box-len", type=float, default=128.0)
    parser.add_argument("--redshift", type=float, default=8.0)
    parser.add_argument("--mode", choices=["fast", "strict"], default="fast")
    args = parser.parse_args()

    inputs = p21.InputParameters(
        random_seed=42,
        node_redshifts=[
            args.redshift,
            args.redshift + 1.5,
            args.redshift + 3.0,
            args.redshift + 4.5,
        ],
    ).evolve_input_structs(
        DIM=args.dim,
        HII_DIM=args.hii_dim,
        BOX_LEN=args.box_len,
        N_THREADS=4,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=args.redshift + 4.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    p21.reset_cuda_stats()
    with tempfile.TemporaryDirectory() as tmpdir, p21.use_mode(
        args.mode, use_cuda=True, lazy_d2h=True, direc=tmpdir
    ):
        t0 = time.perf_counter()
        coeval = p21.run_coeval(
            inputs=inputs, out_redshifts=[args.redshift], write=False, regenerate=True
        )[0]
        ps = p21.power_spectrum_1d(
            coeval.brightness_temp,
            box_length=args.box_len,
            n_bins=12,
            remove_mean=True,
        )
        elapsed = time.perf_counter() - t0
        stats = p21.cuda_stats()
        p21.cleanup_cuda()

    xh_mean = float(np.mean(coeval.neutral_fraction))
    tb_mean = float(np.mean(coeval.brightness_temp))
    valid = ps["counts"] > 0
    print(f"Coeval z={args.redshift:.1f} ({args.hii_dim}^3, mode={args.mode}): {elapsed:.3f} s")
    print(f"<x_HI> = {xh_mean:.4f}, <delta T_b> = {tb_mean:.2f} mK")
    print(f"1D Delta^2(k) finite bins: {int(np.sum(valid))}/{ps['k'].size}")
    print(f"CUDA registry hits = {stats['registry_hits']}, D2H = {stats['registry_d2h_bytes'] / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
