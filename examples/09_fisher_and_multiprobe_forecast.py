"""Example 09: CRN Fisher forecast + gravityCU halo 21cm x galaxy cross-power & kSZ map."""

from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np

import py21cmcu


def main() -> None:
    use_cuda = bool(py21cmcu.cuda_compiled() and py21cmcu.cuda_available())
    print(f"Running Example 09 (use_cuda={use_cuda})")

    with tempfile.TemporaryDirectory(prefix="21cmcu_ex09_") as tmp:
        tmp_dir = Path(tmp)
        with py21cmcu.config.use(direc=str(tmp_dir / "cache")):
            inputs = py21cmcu.InputParameters(
                random_seed=2026, node_redshifts=[8.0]
            ).evolve_input_structs(
                DIM=64,
                HII_DIM=32,
                BOX_LEN=64.0,
                N_THREADS=2,
                SOURCE_MODEL="E-INTEGRAL",
                USE_TS_FLUCT=False,
                RECOMB_MODEL="none",
                USE_EXP_FILTER=False,
                USE_UPPER_STELLAR_TURNOVER=False,
                KEEP_3D_VELOCITIES=True,
            )

            # 1) CRN Fisher forecast for SKA1-Low vs HERA-350
            ska = py21cmcu.fisher_forecast_21cm(
                inputs,
                param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                fiducial=(-1.3, -1.0),
                step_sizes=(0.05, 0.05),
                out_redshifts=(8.0,),
                instrument="SKA1-Low",
                summary="ps1d",
                ps_n_bins=8,
                model_error_frac=0.1,
                use_cuda=use_cuda,
            )
            ska_prior = ska.with_prior({"F_STAR10_ACG": 0.2})
            print("SKA1-Low Fisher summary:", ska.summary_dict())
            py21cmcu.plot_fisher_contours(
                [ska, ska_prior],
                labels=["SKA1-Low", "SKA1-Low + prior"],
                out_path=tmp_dir / "fisher_contours.png",
            )

            # 2) Fiducial EoR coeval + synthetic gravityCU HDF5 halo catalog -> 21cm x galaxy cross-power
            cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
            dens = np.asarray(cv.density, dtype=np.float64)
            prob = np.maximum(1.0 + 2.0 * dens, 0.0).ravel()
            prob /= np.sum(prob)

            rng = np.random.default_rng(77)
            n_halos = 20_000
            idx = rng.choice(dens.size, size=n_halos, replace=True, p=prob)
            ix, iy, iz = np.unravel_index(idx, dens.shape)
            dx = 64.0 / dens.shape[0]
            pos = (np.column_stack([ix, iy, iz]) + rng.uniform(0.0, 1.0, size=(n_halos, 3))) * dx
            masses = 1e10 * rng.lognormal(mean=0.0, sigma=0.5, size=n_halos)

            halo_path = tmp_dir / "gravitycu_halos_z8.h5"
            with h5py.File(halo_path, "w") as fl:
                fl.attrs["BoxSize"] = 64.0
                grp = fl.create_group("Group")
                grp.create_dataset("GroupPos", data=pos.astype(np.float32))
                grp.create_dataset("GroupMass", data=masses.astype(np.float32))

            halos = py21cmcu.load_gravitycu_halos(halo_path, mass_min=5e9, units="Mpc")
            cross = py21cmcu.cross_power_21cm_galaxy(
                cv.brightness_temp,
                halos["positions"],
                halos["box_length"],
                n_bins=8,
                use_cuda=use_cuda,
            )
            err = py21cmcu.cross_power_21cm_galaxy_error(
                cross["pk_21"],
                cross["pk_gal_shot_sub"],
                cross["pk_cross"],
                cross["counts"],
                shot_noise_gal=cross["shot_noise_pk"],
            )
            print("21cm x galaxy r(k):", np.round(cross["r_k"], 3), "total SNR:", round(err["total_snr"], 2))

            # 3) Patchy kSZ map & angular power spectrum
            ksz = py21cmcu.ksz_momentum_map(cv, cosmo=inputs.cosmo_params.cosmo)
            print("kSZ map rms [uK]:", round(ksz["rms_uK"], 4), "max D_ell [uK^2]:", round(float(np.nanmax(ksz["dl_uK2"])), 4))

    if use_cuda:
        py21cmcu.cleanup_cuda()


if __name__ == "__main__":
    main()
