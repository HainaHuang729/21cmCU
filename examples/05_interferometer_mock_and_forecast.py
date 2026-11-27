"""GPU-accelerated interferometer mock observation and sensitivity forecasting demo."""

from __future__ import annotations

import numpy as np

import py21cmcu


def main() -> None:
    """Run an EoR coeval box through SKA1-Low & HERA-350 mock pipelines and 1D/2D SNR forecasts."""
    z = 8.0
    inputs = py21cmcu.InputParameters(
        random_seed=42, node_redshifts=[z]
    ).evolve_input_structs(
        DIM=128,
        HII_DIM=64,
        BOX_LEN=192.0,
        N_THREADS=4,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    )

    with py21cmcu.use_mode("fast", use_cuda=py21cmcu.cuda_available()):
        cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[z], write=False)[0]
        tb = np.asarray(cv.brightness_temp, dtype=np.float32)

        for name in ("SKA1-Low", "HERA-350"):
            mock_tb, meta = py21cmcu.mock_observation_3d(
                tb,
                box_length=192.0,
                redshift=z,
                instrument=name,
                seed=2026,
                return_metadata=True,
            )
            stats = py21cmcu.field_moments_and_pdf(mock_tb, n_bins=32)
            fc1d = py21cmcu.forecast_power_spectrum_error(
                tb.shape,
                box_length=192.0,
                redshift=z,
                instrument=name,
                signal_field=tb,
                n_bins=10,
            )
            fc2d = py21cmcu.forecast_power_spectrum_2d_error(
                tb.shape,
                box_length=192.0,
                redshift=z,
                instrument=name,
                signal_field=tb,
                n_kperp=6,
                n_kpara=6,
            )
            print(
                f"{name:8s} | backend={meta['backend']:4s} | T_sys={meta['t_sys_k']:.1f} K | "
                f"PSF={meta['psf_fwhm_arcmin']:.2f}' | mock_std={stats['std']:.2f} mK | "
                f"1D SNR={fc1d['total_snr']:.2f} | 2D SNR={fc2d['total_snr']:.2f}"
            )

    if py21cmcu.cuda_available():
        py21cmcu.cleanup_cuda()


if __name__ == "__main__":
    main()
