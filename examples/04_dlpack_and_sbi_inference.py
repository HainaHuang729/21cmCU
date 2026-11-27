"""Example: Zero-copy CUDA DLPack export, GPU observables/moments, 21CMMC likelihood & SBI."""

from __future__ import annotations

import tempfile

import numpy as np

import py21cmcu as p21


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, p21.config.use(direc=tmpdir):
        _run_demo()


def _run_demo() -> None:
    base_inputs = p21.InputParameters(
        random_seed=42,
        node_redshifts=[8.0, 9.5, 11.0, 12.5],
    ).evolve_input_structs(
        DIM=96,
        HII_DIM=48,
        BOX_LEN=96.0,
        N_THREADS=4,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    sim = p21.SBISimulator(
        base_inputs=base_inputs,
        param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
        mode="coeval",
        out_redshift=8.0,
        summary="stats",
        ps_n_bins=8,
        wedge_filter=True,
        wedge_buffer=0.05,
        noise_sigma_mk=2.0,
    )
    batch_theta = np.array([[-1.3, -1.0], [-1.5, -1.2]], dtype=np.float64)
    summaries = sim(batch_theta)
    print("SBI summary batch shape:", summaries.shape, "first xH/Tb:", summaries[0, :2])

    field_sim = p21.SBISimulator(
        base_inputs=base_inputs,
        param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
        mode="coeval",
        out_redshift=8.0,
        summary="field",
        as_cuda_view=True,
    )
    with field_sim(batch_theta[0]) as gpu_view:
        print(
            f"Exported GPU CudaFieldView: shape={gpu_view.shape}, device={gpu_view.device_id}, "
            f"__cuda_array_interface__ ptr={hex(gpu_view.ptr)}"
        )
        arr = gpu_view.to_numpy()
        mom = p21.field_moments_and_pdf(arr, n_bins=16)
        print(f"Moments: mean={mom['mean']:.2f} mK, skew={mom['skewness']:.3f}, kurt={mom['kurtosis']:.3f}")

    # Optional PyTorch zero-copy DLPack check
    try:
        import torch

        with p21.use_mode("fast", use_cuda=True, lazy_d2h=True):
            cv = p21.run_coeval(inputs=base_inputs, out_redshifts=[8.0], write=False, regenerate=True)[0]
            t = p21.to_torch(cv.brightness_temp)
            print("PyTorch zero-copy tensor:", tuple(t.shape), t.device, t.dtype, float(torch.mean(t)))
    except ImportError:
        print("PyTorch not installed; verified pure-ctypes DLPack / __cuda_array_interface__.")

    mock_sim = p21.Likelihood21cm(
        base_inputs=base_inputs,
        param_names=("F_STAR10_ACG",),
        data_pk=np.ones(8),
        sigma_pk=np.ones(8),
        mode="coeval",
        out_redshift=8.0,
    ).simulate([-1.3])
    like = p21.Likelihood21cm(
        base_inputs=base_inputs,
        param_names=("F_STAR10_ACG",),
        data_pk=mock_sim["delta_sq"],
        sigma_pk=np.maximum(np.abs(mock_sim["delta_sq"]) * 0.1, 1.0),
        mode="coeval",
        out_redshift=8.0,
    )
    print("21CMMC log-likelihood at truth:", like([-1.3]), "offset:", like([-1.7]))
    p21.cleanup_cuda()


if __name__ == "__main__":
    main()
