"""Phase 12 tests (v0.9.0 Phase 10): 3D Bispectrum, MFP BSD, Minkowski Functionals & SBI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import py21cmcu
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _smoothed_gaussian_box(
    shape: tuple[int, int, int], box_len: float, *, sigma_k: float, std_target: float, seed: int
) -> np.ndarray:
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    white = np.fft.rfftn(rng.normal(0.0, 1.0, size=shape))
    kx = np.fft.fftfreq(nx, d=box_len / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=box_len / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=box_len / nz) * (2.0 * np.pi)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    filt = np.exp(-0.5 * (kmag / sigma_k) ** 2)
    filt[0, 0, 0] = 0.0
    g = np.fft.irfftn(white * filt, s=shape, axes=(0, 1, 2))
    g -= np.mean(g)
    g *= std_target / np.std(g)
    return g.astype(np.float32)


def test_bispectrum_gaussian_and_quadratic_nongaussian() -> None:
    """Verify Q ~= 0 for Gaussian fields and Q ~= 2*lambda for local quadratic fields."""
    shape = (32, 32, 32)
    box_len = 128.0
    lam = 0.25
    n_seeds = 24

    q_gauss = []
    q_eq = []
    q_sq = []
    q_fo = []

    for s in range(n_seeds):
        g = _smoothed_gaussian_box(shape, box_len, sigma_k=0.70, std_target=0.25, seed=9200 + s)
        ng = (g + lam * (g**2 - np.mean(g**2))).astype(np.float32)

        rg = py21cmcu.bispectrum_3d(
            g, box_len, config="equilateral", k_values=[0.45], dk=0.20, use_cuda=False
        )
        re = py21cmcu.bispectrum_3d(
            ng, box_len, config="equilateral", k_values=[0.45], dk=0.20, use_cuda=False
        )
        rs = py21cmcu.bispectrum_3d(
            ng, box_len, config="squeezed", k1_fixed=0.14, k_values=[0.45], dk=0.18, use_cuda=False
        )
        rf = py21cmcu.bispectrum_3d(
            ng, box_len, config="folded", k_values=[0.50], dk=0.20, use_cuda=False
        )
        assert np.all(re["n_triangles"] > 0)
        assert np.all(rs["n_triangles"] > 0)
        assert np.all(rf["n_triangles"] > 0)
        q_gauss.append(float(rg["Q"][0]))
        q_eq.append(float(re["Q"][0]))
        q_sq.append(float(rs["Q"][0]))
        q_fo.append(float(rf["Q"][0]))

    assert abs(float(np.mean(q_gauss))) < 0.08
    q_theory = 2.0 * lam
    assert abs(float(np.mean(q_eq)) - q_theory) / q_theory < 0.15
    assert abs(float(np.mean(q_sq)) - q_theory) / q_theory < 0.15
    assert abs(float(np.mean(q_fo)) - q_theory) / q_theory < 0.15


def test_mfp_bsd_analytic_sphere_and_edge_cases() -> None:
    """Verify MFP mean_R ~= 0.75 * R0 on an isolated sphere and edge cases (empty / full)."""
    n = 64
    box_len = 128.0
    r0 = 24.0
    dx = box_len / n
    c = (np.arange(n) - n / 2.0 + 0.5) * dx
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    r2 = x * x + y * y + z * z

    field = np.where(r2 <= r0 * r0, 0.0, 1.0).astype(np.float32)
    res = py21cmcu.bubble_size_distribution_mfp(
        field,
        box_len,
        threshold=0.5,
        below_threshold=True,
        n_rays=60_000,
        n_bins=24,
        seed=42,
        use_cuda=False,
    )
    expected_xion = (4.0 / 3.0 * np.pi * (r0**3)) / (box_len**3)
    assert abs(res["ionized_fraction"] - expected_xion) < 0.005
    assert abs(res["mean_r"] - 0.75 * r0) / (0.75 * r0) < 0.05
    assert int(res["counts"].sum()) == 60_000
    np.testing.assert_allclose(float(np.sum(res["pdf"] * np.diff(res["r_edges"]))), 1.0, rtol=1e-4)

    all_neutral = np.ones((16, 16, 16), dtype=np.float32)
    empty_res = py21cmcu.bubble_size_distribution_mfp(all_neutral, 32.0, n_rays=2000, use_cuda=False)
    assert empty_res["ionized_fraction"] == 0.0
    assert empty_res["mean_r"] == 0.0
    assert int(empty_res["counts"].sum()) == 0

    all_ionized = np.zeros((16, 16, 16), dtype=np.float32)
    full_res = py21cmcu.bubble_size_distribution_mfp(
        all_ionized, 32.0, r_max=32.0, n_rays=2000, use_cuda=False
    )
    assert full_res["ionized_fraction"] == 1.0
    assert np.isclose(full_res["mean_r"], 32.0, rtol=1e-5)


def test_minkowski_functionals_analytic_shapes_and_gaussian_symmetry() -> None:
    """Verify exact cubical Euler numbers (ball/shell/torus), sphere V0..V2, and GRF parity."""
    n = 32
    c = np.arange(n) - n / 2.0 + 0.5
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    ball = (r <= 8.0).astype(np.float32)
    shell = ((r >= 4.0) & (r <= 10.0)).astype(np.float32)
    tube = (np.sqrt(x * x + y * y) <= 5.0).astype(np.float32)

    assert int(py21cmcu.minkowski_functionals_3d(ball, 32.0, thresholds=[0.5], use_cuda=False)["euler"][0]) == 1
    assert int(py21cmcu.minkowski_functionals_3d(shell, 32.0, thresholds=[0.5], use_cuda=False)["euler"][0]) == 2
    assert int(py21cmcu.minkowski_functionals_3d(tube, 32.0, thresholds=[0.5], use_cuda=False)["euler"][0]) == 0

    # Larger sphere checks accurate Crofton prefactors for V0, V1, V2, and Euler=1
    nb = 96
    rb = 28.0
    cb = np.arange(nb) - nb / 2.0 + 0.5
    xb, yb, zb = np.meshgrid(cb, cb, cb, indexing="ij")
    sphere = ((xb * xb + yb * yb + zb * zb) <= rb * rb).astype(np.float32)
    mk_s = py21cmcu.minkowski_functionals_3d(sphere, float(nb), thresholds=[0.5], use_cuda=False)
    vol = float(nb**3)
    assert int(mk_s["euler"][0]) == 1
    np.testing.assert_allclose(mk_s["V0"][0], (4.0 / 3.0 * np.pi * rb**3) / vol, rtol=0.02)
    np.testing.assert_allclose(mk_s["V1"][0], (4.0 * np.pi * rb**2) / (6.0 * vol), rtol=0.02)
    np.testing.assert_allclose(mk_s["V2"][0], (4.0 * rb) / (3.0 * vol), rtol=0.02)

    # Gaussian random field parity: V1 even, V2 odd, V3 even around nu=0 (resolved ensemble)
    v0_acc, v1_acc, v2_acc, v3_acc = [], [], [], []
    for s in range(8):
        g = _smoothed_gaussian_box((64, 64, 64), 64.0, sigma_k=0.4, std_target=1.0, seed=9300 + s)
        mk_g = py21cmcu.minkowski_functionals_3d(
            g, 64.0, thresholds=[-1.5, 0.0, 1.5], standardize=True, use_cuda=False
        )
        v0_acc.append(mk_g["V0"])
        v1_acc.append(mk_g["V1"])
        v2_acc.append(mk_g["V2"])
        v3_acc.append(mk_g["V3"])
    v0_m = np.mean(v0_acc, axis=0)
    v1_m = np.mean(v1_acc, axis=0)
    v2_m = np.mean(v2_acc, axis=0)
    v3_m = np.mean(v3_acc, axis=0)
    assert abs(v0_m[1] - 0.5) < 0.02
    assert np.isclose(v1_m[0], v1_m[2], rtol=0.05)
    assert np.isclose(v2_m[0], -v2_m[2], rtol=0.06)
    assert np.isclose(v3_m[0], v3_m[2], rtol=0.12)


def test_eor_coeval_topology_and_nongaussian_sbi(tmp_path: Path) -> None:
    """Verify Phase 10 tools and SBISimulator(summary='nongaussian') on a real coeval box."""
    use_cuda = bool(py21cmcu.cuda_compiled() and py21cmcu.cuda_available())
    with py21cmcu.config.use(direc=str(tmp_path / "eor")):
        inputs = py21cmcu.InputParameters(random_seed=9401, node_redshifts=[8.0]).evolve_input_structs(
            DIM=24,
            HII_DIM=12,
            BOX_LEN=48.0,
            N_THREADS=2,
            SOURCE_MODEL="E-INTEGRAL",
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
        )
        cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
        bi = py21cmcu.bispectrum_3d(
            cv.brightness_temp, 48.0, config="equilateral", n_bins=4, use_cuda=use_cuda
        )
        bsd = py21cmcu.bubble_size_distribution_mfp(
            cv.neutral_fraction, 48.0, n_rays=5000, n_bins=12, use_cuda=use_cuda
        )
        mk = py21cmcu.minkowski_functionals_3d(
            cv.brightness_temp, 48.0, n_thresholds=7, standardize=True, use_cuda=use_cuda
        )
        assert bi["B"].shape == (4,)
        assert bsd["counts"].sum() <= 5000
        assert mk["V0"].shape == (7,)

        sim_ng = py21cmcu.SBISimulator(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
            mode="coeval",
            out_redshift=8.0,
            summary="nongaussian",
            ps_n_bins=6,
            wedge_filter=True,
            noise_sigma_mk=1.0,
            use_cuda=use_cuda,
        )
        ds = py21cmcu.OnlineSBIDataset(
            sim_ng,
            {"F_STAR10_ACG": (-1.5, -1.1), "F_ESC10_ACG": (-1.3, -0.8)},
            n_samples=2,
            batch_size=2,
            sampling="sobol",
            seed=11,
        )
        th, feats = next(iter(ds))
        assert th.shape == (2, 2)
        # 5 global/moments + 6 delta^2 + 6 equilateral Q + 3*5 Minkowski functionals = 32
        assert feats.shape == (2, 32)
        assert np.all(np.isfinite(feats))


@requires_cuda_runtime
def test_phase10_gpu_cpu_parity() -> None:
    """Verify exact GPU vs CPU parity for Bispectrum, MFP BSD, and Minkowski Functionals."""
    g = _smoothed_gaussian_box((32, 32, 32), 128.0, sigma_k=0.45, std_target=0.25, seed=9501)
    ng = (g + 0.25 * (g**2 - np.mean(g**2))).astype(np.float32)

    for cfg in ("equilateral", "squeezed", "folded"):
        b_cpu = py21cmcu.bispectrum_3d(ng, 128.0, config=cfg, n_bins=5, use_cuda=False)
        b_gpu = py21cmcu.bispectrum_3d(ng, 128.0, config=cfg, n_bins=5, use_cuda=True)
        np.testing.assert_allclose(b_gpu["n_triangles"], b_cpu["n_triangles"], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(b_gpu["pk1"], b_cpu["pk1"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(b_gpu["B"], b_cpu["B"], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(b_gpu["Q"], b_cpu["Q"], rtol=1e-3, atol=1e-4)

    n = 48
    c = (np.arange(n) - n / 2.0 + 0.5) * (96.0 / n)
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    xh = np.where(x * x + y * y + z * z <= 18.0**2, 0.0, 1.0).astype(np.float32)
    m_cpu = py21cmcu.bubble_size_distribution_mfp(xh, 96.0, n_rays=80_000, seed=42, use_cuda=False)
    m_gpu = py21cmcu.bubble_size_distribution_mfp(xh, 96.0, n_rays=80_000, seed=42, use_cuda=True)
    assert np.isclose(m_gpu["ionized_fraction"], m_cpu["ionized_fraction"], atol=1e-7)
    assert np.isclose(m_gpu["mean_r"], m_cpu["mean_r"], rtol=0.02)
    assert np.isclose(m_gpu["std_r"], m_cpu["std_r"], rtol=0.03)

    mk_cpu = py21cmcu.minkowski_functionals_3d(
        ng, 128.0, n_thresholds=15, standardize=True, use_cuda=False
    )
    mk_gpu = py21cmcu.minkowski_functionals_3d(
        ng, 128.0, n_thresholds=15, standardize=True, use_cuda=True
    )
    np.testing.assert_array_equal(mk_gpu["euler"], mk_cpu["euler"])
    np.testing.assert_allclose(mk_gpu["V0"], mk_cpu["V0"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mk_gpu["V1"], mk_cpu["V1"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mk_gpu["V2"], mk_cpu["V2"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mk_gpu["V3"], mk_cpu["V3"], rtol=1e-10, atol=1e-12)
    py21cmcu.cleanup_cuda()
