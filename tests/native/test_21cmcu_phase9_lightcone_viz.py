"""Phase 9 tests: zero-D2H lightcone optimization, quick_lightcone, parallel redshifts, & 3D viz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import py21cmcu
from py21cmfast.c_21cmfast import ffi, lib

requires_cuda_runtime = pytest.mark.skipif(
    not (py21cmcu.cuda_compiled() and py21cmcu.cuda_available()),
    reason="CUDA runtime not available on this host",
)


def test_quick_coeval_and_streaming_lightcone_parity() -> None:
    """Verify quick_coeval and quick_lightcone streaming PerturbedField exact parity."""
    cv = py21cmcu.quick_coeval(8.0, box_len=96.0, dim=48, hii_dim=24, seed=101, use_cuda=False)
    assert cv.brightness_temp.shape == (24, 24, 24)
    assert np.isclose(cv.redshift, 8.0)

    lc_eager = py21cmcu.quick_lightcone(
        7.5,
        9.0,
        box_len=96.0,
        dim=48,
        hii_dim=24,
        n_nodes=4,
        seed=202,
        use_cuda=False,
        lazy_d2h=False,
        stream_perturbed_fields=False,
    )
    lc_stream = py21cmcu.quick_lightcone(
        7.5,
        9.0,
        box_len=96.0,
        dim=48,
        hii_dim=24,
        n_nodes=4,
        seed=202,
        use_cuda=False,
        lazy_d2h=False,
        stream_perturbed_fields=True,
    )
    assert lc_eager.shape == lc_stream.shape
    np.testing.assert_allclose(
        lc_eager.lightcones["brightness_temp"],
        lc_stream.lightcones["brightness_temp"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        lc_eager.global_quantities["neutral_fraction"],
        lc_stream.global_quantities["neutral_fraction"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_parallel_redshifts_lightcone_parity() -> None:
    """Verify causal-free redshift-parallel lightcone matches serial quick_lightcone."""
    common = dict(
        z_min=7.5,
        z_max=9.5,
        box_len=64.0,
        dim=32,
        hii_dim=16,
        n_nodes=5,
        seed=303,
        use_cuda=False,
        lazy_d2h=False,
        stream_perturbed_fields=True,
        SOURCE_MODEL="L-INTEGRAL",
        USE_MCGS=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lc_ser = py21cmcu.quick_lightcone(**common, parallel_redshifts=False)
    lc_par = py21cmcu.quick_lightcone(**common, parallel_redshifts=True, n_workers=2)

    assert lc_ser.shape == lc_par.shape
    np.testing.assert_allclose(
        lc_ser.lightcones["brightness_temp"],
        lc_par.lightcones["brightness_temp"],
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        lc_ser.global_quantities["brightness_temp"],
        lc_par.global_quantities["brightness_temp"],
        rtol=1e-5,
        atol=1e-4,
    )


@requires_cuda_runtime
def test_gpu_zero_d2h_mean_and_optimized_lightcone() -> None:
    """Verify p21_cuda_field_mean_f32 parity and reduced D2H on optimized GPU lightcones."""
    arr = np.random.default_rng(17).normal(-15.0, 22.0, size=(32, 32, 32)).astype(np.float32)
    out = ffi.new("double *")
    rc = lib.p21_cuda_field_mean_f32(
        ffi.cast("const float *", ffi.from_buffer(arr)),
        int(arr.size),
        out,
    )
    assert rc == 0
    np.testing.assert_allclose(float(out[0]), float(np.mean(arr)), rtol=1e-6, atol=1e-6)

    py21cmcu.reset_cuda_stats()
    lc_eager = py21cmcu.quick_lightcone(
        7.5,
        9.5,
        box_len=96.0,
        dim=48,
        hii_dim=24,
        n_nodes=5,
        seed=404,
        use_cuda=True,
        lazy_d2h=False,
        stream_perturbed_fields=False,
    )
    d2h_eager = py21cmcu.cuda_stats()["registry_d2h_bytes"]

    py21cmcu.reset_cuda_stats()
    lc_opt = py21cmcu.quick_lightcone(
        7.5,
        9.5,
        box_len=96.0,
        dim=48,
        hii_dim=24,
        n_nodes=5,
        seed=404,
        use_cuda=True,
        lazy_d2h=True,
        stream_perturbed_fields=True,
    )
    d2h_opt = py21cmcu.cuda_stats()["registry_d2h_bytes"]

    assert d2h_opt < d2h_eager
    np.testing.assert_allclose(
        lc_eager.lightcones["brightness_temp"],
        lc_opt.lightcones["brightness_temp"],
        rtol=1e-4,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        lc_eager.global_quantities["neutral_fraction"],
        lc_opt.global_quantities["neutral_fraction"],
        rtol=1e-4,
        atol=1e-4,
    )
    py21cmcu.cleanup_cuda()


def test_export_volume_and_lightcone_3d_html(tmp_path: Path) -> None:
    """Verify standalone offline 3D WebGL HTML viewer generation and optional matplotlib plots."""
    lc = py21cmcu.quick_lightcone(
        7.5,
        9.0,
        box_len=64.0,
        dim=32,
        hii_dim=16,
        n_nodes=4,
        seed=505,
        use_cuda=False,
        lazy_d2h=False,
        instrument="SKA1-Low",
        html_3d_path=tmp_path / "lc_3d.html",
    )
    html_file = tmp_path / "lc_3d.html"
    assert html_file.exists() and html_file.stat().st_size > 2000
    text = html_file.read_text(encoding="utf-8")
    assert "webgl2" in text
    assert "DecompressionStream" in text
    assert "PAYLOAD_B64" in text
    assert "brightness_temp_mock" in text
    assert "https://" not in text and "http://" not in text

    vol_html = py21cmcu.export_volume_3d_html(
        lc.lightcones["neutral_fraction"],
        tmp_path / "xhi_3d.html",
        box_length=lc.lightcone_dimensions,
        cmap="viridis",
        max_xy=12,
        max_z=16,
    )
    assert vol_html.exists() and vol_html.stat().st_size > 2000

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    fig_lc = py21cmcu.plot_lightcone_summary(lc, out_path=tmp_path / "lc_summary.png")
    fig_cmp = py21cmcu.plot_mock_comparison(
        lc.lightcones["brightness_temp"][..., :16],
        lc.lightcones["brightness_temp_mock"][..., :16],
        box_length=64.0,
        redshift=8.0,
        out_path=tmp_path / "mock_cmp.png",
    )
    assert (tmp_path / "lc_summary.png").exists()
    assert (tmp_path / "mock_cmp.png").exists()
    import matplotlib.pyplot as plt

    plt.close(fig_lc)
    plt.close(fig_cmp)
