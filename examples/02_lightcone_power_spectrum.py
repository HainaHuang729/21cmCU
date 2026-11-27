"""Minimal quickstart: GPU rectilinear lightcone + RSDs + chunk PS + wedge/noise + DeviceLZF."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

import py21cmcu as p21


def main() -> None:
    parser = argparse.ArgumentParser(description="21cmCU GPU lightcone + observables example")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hii-dim", type=int, default=64)
    parser.add_argument("--box-len", type=float, default=128.0)
    parser.add_argument("--z-min", type=float, default=7.0)
    parser.add_argument("--z-max", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("lightcone_quickstart.h5"))
    args = parser.parse_args()

    inputs = p21.InputParameters(
        random_seed=101,
        node_redshifts=[6.5, 7.5, 8.5, 9.5, 11.0],
    ).evolve_input_structs(
        DIM=args.dim,
        HII_DIM=args.hii_dim,
        BOX_LEN=args.box_len,
        N_THREADS=4,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=10.5,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    lightconer = p21.RectilinearLightconer.between_redshifts(
        min_redshift=args.z_min,
        max_redshift=args.z_max,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "neutral_fraction"),
        cosmo=inputs.cosmo_params.cosmo,
    )

    with tempfile.TemporaryDirectory() as tmpdir, p21.use_mode(
        "fast", use_cuda=True, lazy_d2h=True, direc=tmpdir
    ):
        t0 = time.perf_counter()
        lc = p21.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )
        tb_rsd = np.asarray(lc.lightcones["brightness_temp_with_rsds"], dtype=np.float32)
        chunk_ps = p21.lightcone_chunk_power_spectra(
            lc, quantity="brightness_temp_with_rsds", n_chunks=4, n_bins=10
        )

        # Apply mock foreground horizon wedge + thermal noise to the middle lightcone cube
        z_mid = float(np.mean(lc.lightcone_redshifts))
        box_dims = (args.box_len, args.box_len, args.box_len)
        nx, ny, nz = tb_rsd.shape
        sub = np.ascontiguousarray(tb_rsd[:, :, : min(nz, nx)])
        box_sub = (
            args.box_len,
            args.box_len,
            args.box_len * (sub.shape[2] / float(nx)),
        )
        noisy = p21.add_thermal_noise_3d(sub, box_sub, sigma_mk=2.0, seed=7)
        obs = p21.apply_wedge_filter_3d(
            noisy, box_sub, redshift=z_mid, buffer_h_mpc=0.05
        )
        obs_ps = p21.power_spectrum_1d(obs, box_sub, n_bins=10, remove_mean=True)

        p21.save_lightcone(lc, args.out, compression="device-lzf")
        elapsed = time.perf_counter() - t0
        p21.cleanup_cuda()

    print(
        f"Lightcone shape={tb_rsd.shape}, chunks={len(chunk_ps['chunks'])}, "
        f"z_mid={z_mid:.2f}, box_full={box_dims[0]:.0f} Mpc"
    )
    print(f"Observed (noisy+wedge) median P(k): {float(np.nanmedian(obs_ps['pk'])):.3e} mK^2 Mpc^3")
    print(f"Saved DeviceLZF HDF5 -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB) in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
