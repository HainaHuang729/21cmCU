"""Minimal quickstart: multi-GPU slab domain-decomposed IC + Perturb + 1D P(k)."""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

import py21cmcu as p21


def main() -> None:
    parser = argparse.ArgumentParser(description="21cmCU multi-GPU slab IC example")
    parser.add_argument("--dim", type=int, default=512, help="Try 1536, 1792, or 2048 on 2-3 GPUs")
    parser.add_argument("--hii-dim", type=int, default=256)
    parser.add_argument("--box-len", type=float, default=512.0)
    parser.add_argument("--algo", choices=["ZELDOVICH", "2LPT"], default="ZELDOVICH")
    parser.add_argument("--devices", default="0,1", help="Comma-separated GPU IDs, e.g. 0,1,2")
    args = parser.parse_args()

    req_devs = tuple(int(d.strip()) for d in args.devices.split(",") if d.strip())
    n_avail = p21.cuda_device_count()
    active = tuple(d for d in req_devs if d < n_avail) or (0,)
    if len(active) < 2:
        print(f"Notice: only {n_avail} GPU(s) visible; running single-GPU path on {active}.")

    # Set P21_CUDA_FORCE_MGPU=1 only for demonstration if testing slab path at smaller DIM
    if len(active) >= 2 and args.dim < 1280:
        os.environ.setdefault("P21_CUDA_FORCE_MGPU", "1")

    inputs = p21.InputParameters(
        random_seed=77,
        node_redshifts=[8.0],
    ).evolve_input_structs(
        DIM=args.dim,
        HII_DIM=args.hii_dim,
        BOX_LEN=args.box_len,
        N_THREADS=8,
        PERTURB_ON_HIGH_RES=False,
        PERTURB_ALGORITHM=args.algo,
    )

    p21.reset_cuda_stats()
    with tempfile.TemporaryDirectory() as tmpdir, p21.use_mode(
        "fast", use_cuda=True, devices=active, direc=tmpdir
    ):
        t0 = time.perf_counter()
        ic = p21.compute_initial_conditions(inputs=inputs, write=False, regenerate=True)
        t_ic = time.perf_counter() - t0

        t1 = time.perf_counter()
        pf = p21.perturb_field(
            initial_conditions=ic,
            inputs=inputs,
            redshift=8.0,
            write=False,
            regenerate=True,
        )
        t_pf = time.perf_counter() - t1

        ps = p21.power_spectrum_1d(pf.density, box_length=args.box_len, n_bins=16, remove_mean=True)
        stats = p21.cuda_stats()
        p21.cleanup_cuda()

    print(
        f"IC ({args.dim}^3 -> {args.hii_dim}^3, {args.algo}) on GPUs {active}: "
        f"IC={t_ic:.3f} s, Perturb={t_pf:.3f} s"
    )
    print(
        f"Finite delta std={float(np.std(pf.density)):.4f}, "
        f"mgpu_ic_runs={stats['mgpu_ic_runs']}, mgpu_fft_runs={stats['mgpu_fft_runs']}, "
        f"k_bins={int(np.sum(ps['counts'] > 0))}"
    )


if __name__ == "__main__":
    main()
