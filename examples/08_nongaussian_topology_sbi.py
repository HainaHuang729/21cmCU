"""Example: Higher-order 3D Bispectrum, MFP Bubble Size Distribution, Minkowski Functionals & SBI."""

from __future__ import annotations

import tempfile

import numpy as np

import py21cmcu as p21


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, p21.config.use(direc=tmpdir):
        _run_demo()


def _run_demo() -> None:
    use_cuda = p21.cuda_compiled() and p21.cuda_available()
    inputs = p21.InputParameters(
        random_seed=42,
        node_redshifts=[8.0, 9.5, 11.0, 12.5],
    ).evolve_input_structs(
        DIM=96,
        HII_DIM=48,
        BOX_LEN=128.0,
        N_THREADS=4,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    with p21.use_mode("fast", use_cuda=use_cuda, lazy_d2h=True):
        coeval = p21.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
        tb = np.asarray(coeval.brightness_temp, dtype=np.float32)
        xh = np.asarray(coeval.neutral_fraction, dtype=np.float32)

        for cfg in ("equilateral", "squeezed", "folded"):
            bi = p21.bispectrum_3d(tb, 128.0, config=cfg, n_bins=5, use_cuda=use_cuda)
            print(f"Bispectrum ({cfg}): k1={np.round(bi['k1'], 3)}, Q={np.round(bi['Q'], 3)}")

        bsd = p21.bubble_size_distribution_mfp(
            xh, 128.0, threshold=0.5, below_threshold=True, n_rays=50_000, use_cuda=use_cuda
        )
        print(
            f"MFP BSD: x_ion={bsd['ionized_fraction']:.3f}, "
            f"mean_R={bsd['mean_r']:.2f} Mpc, std_R={bsd['std_r']:.2f} Mpc"
        )

        mk = p21.minkowski_functionals_3d(
            tb, 128.0, thresholds=[-2.0, -1.0, 0.0, 1.0, 2.0], standardize=True, use_cuda=use_cuda
        )
        print(f"Minkowski V0={np.round(mk['V0'], 3)}, euler={mk['euler'].tolist()}")

        sim_ng = p21.SBISimulator(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
            mode="coeval",
            out_redshift=8.0,
            summary="nongaussian",
            ps_n_bins=8,
            instrument="SKA1-Low",
            use_cuda=use_cuda,
        )
        ds = p21.OnlineSBIDataset(
            sim_ng,
            {"F_STAR10_ACG": (-1.6, -1.0), "F_ESC10_ACG": (-1.4, -0.7)},
            n_samples=2,
            batch_size=2,
            sampling="sobol",
            seed=7,
        )
        th, feats = next(iter(ds))
        print(f"Non-Gaussian SBI batch: theta={th.shape}, features={feats.shape}")
        if use_cuda:
            p21.cleanup_cuda()


if __name__ == "__main__":
    main()
