"""Example: Warm persistent BatchPool and zero-disk OnlineSBIDataset streaming."""

from __future__ import annotations

import tempfile

import py21cmcu as p21


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, p21.config.use(direc=tmpdir):
        _run_demo()


def _run_demo() -> None:
    use_cuda = p21.cuda_compiled() and p21.cuda_available()
    n_gpus = min(2, p21.cuda_device_count()) if use_cuda else 1
    devices = tuple(range(max(1, n_gpus)))

    base_inputs = p21.InputParameters(
        random_seed=700,
        node_redshifts=[8.0, 9.5, 11.0, 12.5],
    ).evolve_input_structs(
        DIM=96,
        HII_DIM=48,
        BOX_LEN=96.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=12.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )

    bounds = {
        "F_STAR10_ACG": (-1.6, -1.0),
        "F_ESC10_ACG": (-1.4, -0.7),
    }

    with p21.BatchPool(
        devices=devices,
        workers_per_gpu=1,
        use_cuda=use_cuda,
        in_process=(not use_cuda),
    ) as pool:
        sim = p21.SBISimulator(
            base_inputs=base_inputs,
            param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
            mode="coeval",
            out_redshift=8.0,
            summary="stats",
            ps_n_bins=8,
            instrument="SKA1-Low",
            cuda_mode="fast",
            use_cuda=use_cuda,
            batch_pool=pool,
        )

        try:
            import torch

            as_torch = True
        except ImportError:
            as_torch = False

        dataset = p21.OnlineSBIDataset(
            sim,
            bounds,
            n_samples=4,
            batch_size=2,
            sampling="sobol",
            seed=42,
            as_torch=as_torch,
        )
        for step, (theta_b, x_b) in enumerate(dataset):
            print(
                f"Batch {step}: theta shape={tuple(theta_b.shape)}, x shape={tuple(x_b.shape)}, "
                f"type={type(x_b).__name__}, warm tasks_executed={pool.tasks_executed}"
            )

        if as_torch and use_cuda:
            import torch

            field_sim = p21.SBISimulator(
                base_inputs=base_inputs,
                param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                mode="coeval",
                out_redshift=8.0,
                summary="field",
                instrument="SKA1-Low",
                use_cuda=True,
                lazy_d2h=True,
                as_torch=True,
            )
            field_ds = p21.OnlineSBIDataset(
                field_sim,
                bounds,
                n_samples=2,
                batch_size=2,
                sampling="uniform",
                seed=99,
                as_torch=True,
            )
            th_t, vol_t = next(iter(field_ds))
            torch.cuda.synchronize()
            print(
                f"Zero-disk field stream: theta={tuple(th_t.shape)} ({th_t.device}), "
                f"vol={tuple(vol_t.shape)} ({vol_t.device})"
            )


if __name__ == "__main__":
    main()
