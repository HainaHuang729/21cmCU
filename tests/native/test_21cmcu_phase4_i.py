"""Phase 4-I tests for native CUDA DeviceLZF HDF5 I/O and CUDA_LAZY_D2H residency."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import py21cmcu
import pytest
from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers import single_field

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


@pytest.mark.parametrize("compression", ["lzf", "none", "device-lzf"])
def test_save_lightcone_cpu_fallback_roundtrip(tmp_path: Path, compression: str) -> None:
    """Verify ``save_lightcone`` roundtrips on CPU even when ``compression='device-lzf'``."""
    rng = np.random.default_rng(101)
    fields = {
        "brightness_temp": rng.normal(loc=-20.0, scale=10.0, size=(16, 16, 25)).astype(
            np.float32
        ),
        "neutral_fraction": np.clip(rng.uniform(size=(16, 16, 25)), 0.0, 1.0).astype(
            np.float32
        ),
    }
    out_file = tmp_path / f"cpu_{compression}.h5"
    with config.use(USE_CUDA=False):
        meta = py21cmcu.save_lightcone(
            fields,
            out_file,
            compression=compression,
            chunk_z=8,
            attrs={"tag": "cpu-test"},
        )

    expected_used = "none" if compression == "none" else "lzf"
    assert meta["compression_used"] == expected_used
    assert not meta["has_nonfinite"]
    with h5py.File(out_file, "r") as h5f:
        assert h5f.attrs["tag"] == "cpu-test"
        for k, arr in fields.items():
            np.testing.assert_array_equal(h5f["lightcones"][k][:], arr)


@requires_cuda_runtime
@pytest.mark.parametrize("shuffle", [True, False])
def test_device_lzf_bit_exact_roundtrip_tail_and_fallback(
    tmp_path: Path, shuffle: bool
) -> None:
    """Verify CUDA DeviceLZF handles structured data, tail chunks, noise fallback, and NaNs."""
    rng = np.random.default_rng(2026)
    # 1. Structured smooth field + non-divisible Z tail (37 not divisible by chunk_z=16)
    zz, yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 37, dtype=np.float32),
        np.linspace(0.0, 1.0, 32, dtype=np.float32),
        np.linspace(0.0, 1.0, 32, dtype=np.float32),
        indexing="ij",
    )
    smooth = np.round(10.0 * np.sin(xx + yy) + zz, decimals=2).astype(np.float32)
    smooth = np.ascontiguousarray(smooth.transpose(2, 1, 0))

    # 2. Quantized noise (compresses modestly) and raw uniform bytes / NaN check
    noise_u32 = rng.integers(0, 2**31 - 1, size=(32, 32, 37), dtype=np.uint32)
    incompressible = noise_u32.view(np.float32)
    incompressible = np.where(np.isfinite(incompressible), incompressible, 1.25).astype(np.float32)

    with_nan = smooth.copy()
    with_nan[3, 4, 5] = np.nan

    out_path = tmp_path / f"dev_lzf_shuf{int(shuffle)}.h5"
    with py21cmcu.use_mode("strict", use_cuda=True):
        meta = py21cmcu.save_lightcone(
            {"smooth": smooth, "incomp": incompressible, "nan_field": with_nan},
            out_path,
            compression="device-lzf",
            shuffle=shuffle,
            chunk_z=16,
        )

    assert meta["compression_used"] == "device-lzf"
    assert meta["has_nonfinite"] is True
    assert meta["compressed_bytes"] < meta["raw_bytes"]

    with h5py.File(out_path, "r") as h5f:
        np.testing.assert_array_equal(h5f["lightcones/smooth"][:], smooth)
        np.testing.assert_array_equal(h5f["lightcones/incomp"][:], incompressible)
        read_nan = h5f["lightcones/nan_field"][:]
        assert np.isnan(read_nan[3, 4, 5])
        mask = np.ones_like(with_nan, dtype=bool)
        mask[3, 4, 5] = False
        np.testing.assert_array_equal(read_nan[mask], with_nan[mask])


@requires_cuda_runtime
def test_device_lzf_lightcone_object_save(tmp_path: Path) -> None:
    """Verify saving an end-to-end CUDA LightCone preserves all lightcones and globals."""
    inputs = py21cmcu.InputParameters(
        random_seed=8101, node_redshifts=[7.0, 8.0, 9.0]
    ).evolve_input_structs(
        DIM=32,
        HII_DIM=16,
        BOX_LEN=50.0,
        SOURCE_MODEL="L-INTEGRAL",
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lightconer = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=7.2,
        max_redshift=8.8,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "neutral_fraction"),
        cosmo=inputs.cosmo_params.cosmo,
    )
    out_h5 = tmp_path / "e2e_lightcone_dlzf.h5"
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "cache")):
        lc = py21cmcu.run_lightcone(
            inputs=inputs, lightconer=lightconer, regenerate=True, write=False
        )
        meta = py21cmcu.save_lightcone(lc, out_h5, compression="device-lzf")

    assert meta["compression_used"] == "device-lzf"
    assert not meta["has_nonfinite"]
    with h5py.File(out_h5, "r") as h5f:
        np.testing.assert_array_equal(h5f["lightcone_redshifts"][:], lc.lightcone_redshifts)
        for name, arr in lc.lightcones.items():
            np.testing.assert_array_equal(h5f["lightcones"][name][:], arr)
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
@pytest.mark.parametrize("source_model", ["L-INTEGRAL", "E-INTEGRAL"])
def test_lazy_d2h_strict_parity_and_reduced_transfers(
    tmp_path: Path, source_model: str
) -> None:
    """Verify ``CUDA_LAZY_D2H=True`` matches eager ``strict`` outputs with less D2H traffic."""
    inputs = py21cmcu.InputParameters(
        random_seed=8201, node_redshifts=[12.0, 10.0, 8.5]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=75.0,
        SOURCE_MODEL=source_model,
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=11.0,
        USE_EXP_FILTER=(source_model == "L-INTEGRAL"),
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(
        "strict", use_cuda=True, lazy_d2h=False, direc=str(tmp_path / "eager")
    ):
        coeval_eager = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.5], write=False)[0]
        stats_eager = py21cmcu.cuda_stats()

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(
        "strict", use_cuda=True, lazy_d2h=True, direc=str(tmp_path / "lazy")
    ):
        coeval_lazy = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.5], write=False)[0]
        stats_lazy = py21cmcu.cuda_stats()

    assert stats_lazy["registry_d2h_bytes"] < stats_eager["registry_d2h_bytes"]
    np.testing.assert_allclose(
        coeval_lazy.brightness_temp, coeval_eager.brightness_temp, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        coeval_lazy.neutral_fraction, coeval_eager.neutral_fraction, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        coeval_lazy.spin_temperature, coeval_eager.spin_temperature, rtol=1e-5, atol=1e-4
    )
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_lazy_d2h_on_demand_readback_and_eviction_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify lazy RadiationFields/EmissivityFields materialize cleanly on .get() and eviction."""
    inputs = py21cmcu.InputParameters(
        random_seed=8301, node_redshifts=[12.0, 10.5, 9.5]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=75.0,
        SOURCE_MODEL="E-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=11.0,
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode(
        "strict", use_cuda=True, lazy_d2h=False, direc=str(tmp_path / "ref")
    ):
        ic_ref = single_field.compute_initial_conditions(inputs=inputs)
        pf_ref = single_field.perturb_field(
            redshift=9.5, inputs=inputs, initial_conditions=ic_ref
        )
        em_ref = py21cmcu.compute_emissivity_fields(
            redshift=9.5, inputs=inputs, initial_conditions=ic_ref, perturbed_field=pf_ref
        )

    # Tight 1 MB registry forces eviction / direct readback materialization cleanly
    monkeypatch.setenv("P21_CUDA_REGISTRY_MB", "1")
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(
        "strict", use_cuda=True, lazy_d2h=True, direc=str(tmp_path / "lazy_evict")
    ):
        ic_lz = single_field.compute_initial_conditions(inputs=inputs)
        pf_lz = single_field.perturb_field(
            redshift=9.5, inputs=inputs, initial_conditions=ic_lz
        )
        em_lz = py21cmcu.compute_emissivity_fields(
            redshift=9.5, inputs=inputs, initial_conditions=ic_lz, perturbed_field=pf_lz
        )
        nion_lazy = em_lz.get("n_ion").copy()
        sfrd_lazy = em_lz.sfrd_acg.copy()

    np.testing.assert_allclose(nion_lazy, em_ref.n_ion, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sfrd_lazy, em_ref.sfrd_acg, rtol=1e-6, atol=1e-6)
    py21cmcu.cleanup_cuda()
