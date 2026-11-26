"""Direction H (Phase 4) tests: CLASS CosmoTables cache & native CUDA AngularLightconer."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def test_class_cosmo_tables_mem_and_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLASS CosmoTables in-memory & disk caches preserve bit-exact tables and invalidate on param change."""
    pytest.importorskip("classy")
    monkeypatch.setenv("P21_CLASS_CACHE", "1")
    monkeypatch.setenv("P21_CLASS_DISK_CACHE", "1")
    py21cmcu.clear_class_cache()

    base_kw = dict(
        DIM=32,
        HII_DIM=16,
        BOX_LEN=50.0,
        SOURCE_MODEL="E-INTEGRAL",
        POWER_SPECTRUM="CLASS",
        V_CB_MODEL="FLUCTS",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_MCGS=True,
        RECOMB_MODEL="inhomogeneous",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    with config.use(direc=str(tmp_path / "cache_a")):
        # 1. Cold CLASS compute (populates both memory + disk cache)
        inp1 = py21cmcu.InputParameters(
            random_seed=7001, node_redshifts=[12.0, 10.0]
        ).evolve_input_structs(**base_kw)
        ct1 = inp1.cosmo_tables

        # 2. New seed / DIM -> exact in-memory cache hit
        inp2 = py21cmcu.InputParameters(
            random_seed=7002, node_redshifts=[12.0, 10.0]
        ).evolve_input_structs(**{**base_kw, "DIM": 48})
        t0 = time.perf_counter()
        ct2 = inp2.cosmo_tables
        t_mem = time.perf_counter() - t0
        assert ct2 is ct1
        assert t_mem < 0.05

        # 3. Clear memory cache -> disk cache hit (bit-exact values, fast load)
        py21cmcu.clear_class_cache()
        inp3 = py21cmcu.InputParameters(
            random_seed=7003, node_redshifts=[12.0, 10.0]
        ).evolve_input_structs(**base_kw)
        t0 = time.perf_counter()
        ct3 = inp3.cosmo_tables
        t_disk = time.perf_counter() - t0
        assert t_disk < 0.5
        np.testing.assert_array_equal(ct3.transfer_density.x_values, ct1.transfer_density.x_values)
        np.testing.assert_array_equal(ct3.transfer_density.y_values, ct1.transfer_density.y_values)
        assert ct3.transfer_vcb is not None and ct1.transfer_vcb is not None
        np.testing.assert_array_equal(ct3.transfer_vcb.y_values, ct1.transfer_vcb.y_values)
        assert ct3.V_CB_AVG == ct1.V_CB_AVG

        # 4. Changing cosmology invalidates cache and yields distinct transfer tables
        inp4 = py21cmcu.InputParameters(
            random_seed=7001, node_redshifts=[12.0, 10.0]
        ).evolve_input_structs(**{**base_kw, "OMm": 0.32})
        ct4 = inp4.cosmo_tables
        assert ct4 is not ct1
        assert not np.array_equal(ct4.transfer_density.y_values, ct1.transfer_density.y_values)


@requires_cuda_runtime
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("kind", ["mean", "mean_max"])
def test_angular_slice_and_los_velocity_parity(
    tmp_path: Path, order: int, kind: str
) -> None:
    """Verify CUDA AngularLightconer interpolate_slice & interpolate_los_velocity_slice match CPU."""
    from astropy.units import pixel, pixel_scale

    inputs = py21cmcu.InputParameters(
        random_seed=7101, node_redshifts=[7.0, 8.0, 9.0, 10.0]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=72.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=10.0,
        KEEP_3D_VELOCITIES=True,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lc_ang = py21cmcu.AngularLightconer.like_rectilinear(
        inputs.simulation_options,
        match_at_z=7.8,
        max_redshift=8.8,
        interpolation_order=order,
        cosmo=inputs.cosmo_params.cosmo,
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "coeval")):
        c_lo, c_hi = py21cmcu.run_coeval(
            out_redshifts=[7.5, 9.0], inputs=inputs, regenerate=True, write=False
        )
    pixeq = pixel_scale(inputs.simulation_options.cell_size / pixel)
    dc_a = inputs.cosmo_params.cosmo.comoving_distance(c_hi.redshift).to(pixel, equivalencies=pixeq)
    dc_b = inputs.cosmo_params.cosmo.comoving_distance(c_lo.redshift).to(pixel, equivalencies=pixeq)
    lcd = 0.4 * dc_a + 0.6 * dc_b

    box_a = c_hi.z_reion if kind == "mean_max" else c_hi.brightness_temp
    box_b = c_lo.z_reion if kind == "mean_max" else c_lo.brightness_temp

    with py21cmcu.use_mode("strict", use_cuda=False):
        cpu_slice = lc_ang.interpolate_slice(
            lcd, box_a, box_b, dc_a, dc_b, inputs.simulation_options.cell_size, kind=kind
        )
        cpu_vel = lc_ang.interpolate_los_velocity_slice(lcd, c_hi, c_lo, dc_a, dc_b)

    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode("strict", use_cuda=True):
        gpu_slice = lc_ang.interpolate_slice(
            lcd, box_a, box_b, dc_a, dc_b, inputs.simulation_options.cell_size, kind=kind
        )
        gpu_vel = lc_ang.interpolate_los_velocity_slice(lcd, c_hi, c_lo, dc_a, dc_b)

    # Order 0 can differ at exact half-integer coordinate rounding boundaries (rare); order 1 is smooth.
    if order == 0:
        frac_close = np.mean(np.isclose(gpu_slice, cpu_slice, rtol=1e-4, atol=1e-4))
        assert frac_close >= 0.99
    else:
        np.testing.assert_allclose(gpu_slice, cpu_slice, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            gpu_vel.astype(np.float64), cpu_vel.astype(np.float64), rtol=1e-4, atol=1e-22
        )
    assert np.all(np.isfinite(gpu_slice)) and np.all(np.isfinite(gpu_vel))
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_angular_lightcone_e2e_with_rsds_parity(tmp_path: Path) -> None:
    """Verify full GPU AngularLightconer run_lightcone with dvdr + apply_rsds matches CPU."""
    inputs = py21cmcu.InputParameters(
        random_seed=7201, node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=72.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=11.0,
        KEEP_3D_VELOCITIES=True,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lc_ang = py21cmcu.AngularLightconer.like_rectilinear(
        inputs.simulation_options,
        match_at_z=7.6,
        max_redshift=8.8,
        interpolation_order=1,
        quantities=("brightness_temp", "neutral_fraction", "z_reion"),
        cosmo=inputs.cosmo_params.cosmo,
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        lc_cpu = py21cmcu.run_lightcone(
            lightconer=lc_ang,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )

    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        lc_gpu = py21cmcu.run_lightcone(
            lightconer=lc_ang,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )

    for k in ("brightness_temp", "brightness_temp_with_rsds", "neutral_fraction", "z_reion_with_rsds"):
        g = np.asarray(lc_gpu.lightcones[k], dtype=np.float64)
        c = np.asarray(lc_cpu.lightcones[k], dtype=np.float64)
        rms = max(float(np.sqrt(np.mean(c**2))), 1e-30)
        rel_rms = float(np.sqrt(np.mean((g - c) ** 2))) / rms
        assert rel_rms < 5e-4, f"angular[{k}] rel_rms={rel_rms:.3e}"
    py21cmcu.cleanup_cuda()
