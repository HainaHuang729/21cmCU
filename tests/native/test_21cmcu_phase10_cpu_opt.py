"""Native CPU optimization tests: bit-exact FFTW plan cache & OpenMP RSDS."""

from __future__ import annotations

import numpy as np
import pytest

import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.rsds import apply_rsds, rsds_shift


def _get_fftw_cache_stats() -> tuple[int, int, int]:
    hits = ffi.new("unsigned long long *")
    misses = ffi.new("unsigned long long *")
    entries = ffi.new("int *")
    lib.p21_dft_get_cache_stats(hits, misses, entries)
    return int(hits[0]), int(misses[0]), int(entries[0])


def _get_fftw_scratch_stats() -> tuple[int, int, int]:
    hits = ffi.new("unsigned long long *")
    misses = ffi.new("unsigned long long *")
    pooled = ffi.new("int *")
    lib.p21_dft_get_scratch_stats(hits, misses, pooled)
    return int(hits[0]), int(misses[0]), int(pooled[0])


def test_fftw_plan_cache_lifecycle():
    """clean_fftw destroys cached plans and resets stats safely."""
    lib.clean_fftw()
    hits, misses, entries = _get_fftw_cache_stats()
    assert (hits, misses, entries) == (0, 0, 0)

    rc = lib.test_dft_cube(16, 2, 4)
    assert rc == 0
    hits, misses, entries = _get_fftw_cache_stats()
    assert hits > 0
    assert misses > 0
    assert entries > 0

    lib.clean_fftw()
    assert _get_fftw_cache_stats() == (0, 0, 0)

    rc = lib.test_dft_cube(16, 2, 2)
    assert rc == 0
    hits2, _, entries2 = _get_fftw_cache_stats()
    assert hits2 > 0
    assert entries2 > 0
    lib.clean_fftw()


def test_fftw_plan_cache_bit_exact(monkeypatch):
    """P21_FFTW_PLAN_CACHE=1 produces bit-identical coeval boxes vs =0."""
    monkeypatch.setitem(p21.config, "USE_CUDA", False)
    inputs = p21.InputParameters(
        random_seed=42,
        node_redshifts=(13.0, 10.0, 8.0),
        cosmo_params=p21.CosmoParams(),
        simulation_options=p21.SimulationOptions(
            HII_DIM=16,
            DIM=32,
            BOX_LEN=32.0,
            N_THREADS=2,
            Z_HEAT_MAX=12.0,
        ),
        matter_options=p21.MatterOptions(),
        astro_options=p21.AstroOptions(USE_TS_FLUCT=True),
        astro_params=p21.AstroParams(),
    )

    monkeypatch.setenv("P21_FFTW_PLAN_CACHE", "0")
    lib.clean_fftw()
    ref = p21.run_coeval(inputs=inputs, out_redshifts=(8.0,), write=False)[0]
    assert _get_fftw_cache_stats() == (0, 0, 0)

    monkeypatch.setenv("P21_FFTW_PLAN_CACHE", "1")
    lib.clean_fftw()
    cached = p21.run_coeval(inputs=inputs, out_redshifts=(8.0,), write=False)[0]
    hits, misses, entries = _get_fftw_cache_stats()
    assert hits > 0
    assert misses > 0
    assert entries > 0

    checked = 0
    for struct_name, ref_box in ref.output_structs.items():
        cached_box = getattr(cached, struct_name)
        for arr_name, arr_obj in ref_box.arrays.items():
            if arr_obj.value is None:
                continue
            a = np.asarray(arr_obj.value)
            b = np.asarray(cached_box.get(arr_name))
            assert np.array_equal(a, b), (
                f"FFTW cache mismatch in {struct_name}.{arr_name}: "
                f"max_abs={np.max(np.abs(a - b))}"
            )
            checked += 1
    assert checked >= 4

    lib.clean_fftw()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("nz", [15, 32])
@pytest.mark.parametrize("n_rsd_subcells", [1, 3, 4])
@pytest.mark.parametrize("disp_dtype", [np.float32, np.float64])
def test_cpu_rsds_shift_bit_exact(
    monkeypatch,
    dtype,
    periodic: bool,
    nz: int,
    n_rsd_subcells: int,
    disp_dtype,
):
    """Native C/OpenMP rsds_shift matches Python/SciPy/cosmotile to 0.0 max_abs."""
    monkeypatch.setitem(p21.config, "USE_CUDA", False)
    rng = np.random.default_rng(12345 + nz + n_rsd_subcells + (1 if periodic else 0))
    n_los = 96
    field = rng.normal(loc=0.5, scale=1.25, size=(nz, n_los)).astype(dtype)
    disp = rng.normal(loc=0.0, scale=2.7, size=(nz, n_los)).astype(disp_dtype)
    if periodic:
        disp[0, :8] = -2.2 * nz
        disp[-1, -8:] = 2.4 * nz

    monkeypatch.setenv("P21_CPU_RSDS_NATIVE", "0")
    out_ref = rsds_shift(
        field,
        disp,
        periodic=periodic,
        n_rsd_subcells=n_rsd_subcells,
    )

    monkeypatch.setenv("P21_CPU_RSDS_NATIVE", "1")
    out_nat = rsds_shift(
        field,
        disp,
        periodic=periodic,
        n_rsd_subcells=n_rsd_subcells,
    )

    assert out_nat.dtype == out_ref.dtype
    assert np.array_equal(out_nat, out_ref), (
        f"rsds_shift mismatch (dtype={dtype}, periodic={periodic}, nz={nz}, "
        f"S={n_rsd_subcells}, disp={disp_dtype}): "
        f"max_abs={np.max(np.abs(out_nat - out_ref))}"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("n_rsd_subcells", [1, 4])
def test_cpu_apply_rsds_3d_bit_exact(
    monkeypatch,
    dtype,
    periodic: bool,
    n_rsd_subcells: int,
):
    """Native C/OpenMP 3D apply_rsds matches Python fallback to 0.0 max_abs."""
    monkeypatch.setitem(p21.config, "USE_CUDA", False)
    inputs = p21.InputParameters(
        random_seed=7,
        cosmo_params=p21.CosmoParams(),
        simulation_options=p21.SimulationOptions(HII_DIM=16, DIM=32, BOX_LEN=32.0),
    )
    rng = np.random.default_rng(999 + n_rsd_subcells + (1 if periodic else 0))
    field = rng.normal(loc=-10.0, scale=25.0, size=(16, 16, 32)).astype(dtype)
    vel = rng.normal(loc=0.0, scale=5e-18, size=(16, 16, 32)).astype(np.float32)

    monkeypatch.setenv("P21_CPU_RSDS_NATIVE", "0")
    out_ref = apply_rsds(
        field,
        vel,
        redshifts=8.0,
        inputs=inputs,
        periodic=periodic,
        n_rsd_subcells=n_rsd_subcells,
    )

    monkeypatch.setenv("P21_CPU_RSDS_NATIVE", "1")
    out_nat = apply_rsds(
        field,
        vel,
        redshifts=8.0,
        inputs=inputs,
        periodic=periodic,
        n_rsd_subcells=n_rsd_subcells,
    )

    assert out_nat.dtype == out_ref.dtype
    assert np.array_equal(out_nat, out_ref), (
        f"apply_rsds mismatch (dtype={dtype}, periodic={periodic}, S={n_rsd_subcells}): "
        f"max_abs={np.max(np.abs(out_nat - out_ref))}"
    )


def test_cpu_scratch_pool_lifecycle(monkeypatch):
    """clean_fftw frees pooled fftwf_malloc scratch arrays and resets stats."""
    monkeypatch.setitem(p21.config, "USE_CUDA", False)
    monkeypatch.setenv("P21_CPU_SCRATCH_POOL", "1")
    lib.clean_fftw()
    assert _get_fftw_scratch_stats() == (0, 0, 0)

    inputs = p21.InputParameters(
        random_seed=77,
        node_redshifts=(13.0, 10.0, 8.0),
        cosmo_params=p21.CosmoParams(),
        simulation_options=p21.SimulationOptions(
            HII_DIM=16,
            DIM=32,
            BOX_LEN=32.0,
            N_THREADS=2,
            Z_HEAT_MAX=12.0,
        ),
        matter_options=p21.MatterOptions(SOURCE_MODEL="E-INTEGRAL"),
        astro_options=p21.AstroOptions(
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
        ),
        astro_params=p21.AstroParams(),
    )
    p21.run_coeval(inputs=inputs, out_redshifts=(8.0,), write=False)
    hits, misses, pooled = _get_fftw_scratch_stats()
    assert hits > 0
    assert misses > 0
    assert pooled > 0

    lib.clean_fftw()
    assert _get_fftw_scratch_stats() == (0, 0, 0)


@pytest.mark.parametrize(
    ("source_model", "use_exp_filter", "use_mcgs"),
    [
        ("E-INTEGRAL", False, False),
        ("L-INTEGRAL", True, True),
    ],
)
def test_cpu_fused_filter_and_scratch_bit_exact(
    monkeypatch,
    source_model: str,
    use_exp_filter: bool,
    use_mcgs: bool,
):
    """Fused k-space filtering + scratch pool match disabled CPU path with max_abs == 0.0."""
    monkeypatch.setitem(p21.config, "USE_CUDA", False)
    inputs = p21.InputParameters(
        random_seed=101,
        node_redshifts=(13.0, 10.0, 8.0),
        cosmo_params=p21.CosmoParams(),
        simulation_options=p21.SimulationOptions(
            HII_DIM=16,
            DIM=32,
            BOX_LEN=32.0,
            N_THREADS=2,
            Z_HEAT_MAX=12.0,
        ),
        matter_options=p21.MatterOptions(SOURCE_MODEL=source_model),
        astro_options=p21.AstroOptions(
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            USE_EXP_FILTER=use_exp_filter,
            USE_MCGS=use_mcgs,
            USE_UPPER_STELLAR_TURNOVER=False,
        ),
        astro_params=p21.AstroParams(),
    )

    monkeypatch.setenv("P21_CPU_FILTER_FUSED", "0")
    monkeypatch.setenv("P21_CPU_SCRATCH_POOL", "0")
    lib.clean_fftw()
    ref = p21.run_coeval(inputs=inputs, out_redshifts=(8.0,), write=False)[0]
    assert _get_fftw_scratch_stats() == (0, 0, 0)

    monkeypatch.setenv("P21_CPU_FILTER_FUSED", "1")
    monkeypatch.setenv("P21_CPU_SCRATCH_POOL", "1")
    lib.clean_fftw()
    opt = p21.run_coeval(inputs=inputs, out_redshifts=(8.0,), write=False)[0]
    hits, misses, _ = _get_fftw_scratch_stats()
    assert hits > 0
    assert misses > 0

    checked = 0
    for struct_name, ref_box in ref.output_structs.items():
        opt_box = getattr(opt, struct_name)
        for arr_name in ref_box.arrays:
            a = ref_box.get(arr_name)
            if a is None:
                continue
            a_arr = np.asarray(a)
            b_arr = np.asarray(opt_box.get(arr_name))
            assert np.array_equal(a_arr, b_arr), (
                f"Fused/scratch mismatch ({source_model}) in {struct_name}.{arr_name}: "
                f"max_abs={np.max(np.abs(a_arr - b_arr))}"
            )
            checked += 1
    assert checked >= 10

    # Also check all 5 filter kernels directly via test_filter
    rng = np.random.default_rng(555)
    box_in = np.ascontiguousarray(rng.normal(1.0, 0.2, size=(16, 16, 16)), dtype=np.float32)
    out_ref = np.empty((16, 16, 16), dtype=np.float64)
    out_opt = np.empty((16, 16, 16), dtype=np.float64)
    for f_flag in (0, 1, 2, 3, 4, 5):
        monkeypatch.setenv("P21_CPU_FILTER_FUSED", "0")
        rc0 = lib.test_filter(
            ffi.cast("float *", ffi.from_buffer(box_in)),
            4.0,
            8.0,
            2.0,
            int(f_flag),
            ffi.cast("double *", ffi.from_buffer(out_ref)),
        )
        monkeypatch.setenv("P21_CPU_FILTER_FUSED", "1")
        rc1 = lib.test_filter(
            ffi.cast("float *", ffi.from_buffer(box_in)),
            4.0,
            8.0,
            2.0,
            int(f_flag),
            ffi.cast("double *", ffi.from_buffer(out_opt)),
        )
        assert rc0 == 0 and rc1 == 0
        assert np.array_equal(out_opt, out_ref), (
            f"test_filter flag={f_flag} mismatch: max_abs={np.max(np.abs(out_opt - out_ref))}"
        )

    lib.clean_fftw()
