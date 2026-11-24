"""Run the existing small-box 21cmFAST GPU parity checks with pytest."""

from argparse import Namespace

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def cuda_runtime():
    cupy = pytest.importorskip("cupy")
    if not cupy.is_available():
        pytest.skip("No CUDA device available")
    pytest.importorskip("py21cmfast")


def test_gsl_prng_and_independent_initial_conditions():
    from verify_multithread_rng import (
        check_ic_multithread,
        check_stream_words_and_gaussians,
    )

    check_stream_words_and_gaussians(word_count=2048, gauss_count=512)
    results = check_ic_multithread(thread_counts=(1, 2), dim=32, hii_dim=16)
    assert set(results) == {"1", "2"}
    assert all(run["fields"] for run in results.values())


def test_eulerian_thermal_lightcone(tmp_path):
    from verify_constant_lightcone import main

    for mode in ("cpu", "gpu"):
        main(
            mode,
            tmp_path,
            thermal=True,
            require_stage=("thermal_update",),
            source_model="E-INTEGRAL",
        )


@pytest.mark.parametrize("cuda_prefix", [False, True], ids=["hybrid", "prefix"])
def test_multithreaded_chmf_lightcone(tmp_path, cuda_prefix):
    from verify_chmf_multithread import _run_case

    result = _run_case(
        "chmf",
        tmp_path,
        dim=32,
        hii_dim=16,
        box_len=24.0,
        nodes=[12.0, 11.5, 11.0],
        bounds=(11.1, 11.9),
        threads=2,
        seed=129887,
        cuda_prefix=cuda_prefix,
    )
    assert result["passed"], result


def test_molecular_cooling_and_lyman_alpha_scattering(tmp_path):
    from verify_chmf_lightcone import main

    for mode in ("cpu", "gpu"):
        main(
            Namespace(
                directory=tmp_path,
                mode=mode,
                source_model="E-INTEGRAL",
                thermal=True,
                molecular=True,
                lya_scattering=True,
                class_streaming=None,
                cosmology_from=None,
                sample_method=None,
                integration_method=None,
                matter_filter=None,
                seed=129887,
                dim=16,
                hii_dim=8,
                box_mpc=24.0,
                threads=1,
                nodes=[12.0, 11.5, 11.0],
                bounds=[11.1, 11.9],
            ),
        )


def test_angular_lightcone(tmp_path):
    from verify_angular_lightcone import run

    run("cpu", tmp_path)
    result = run("gpu", tmp_path)
    assert result["passed"], result
