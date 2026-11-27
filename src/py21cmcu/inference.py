"""High-level likelihood (21CMMC) and SBI simulator wrappers for 21cmCU."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .analysis import _normalize_box_length, lightcone_chunk_power_spectra, power_spectrum_1d
from .batch import BatchTask, run_batch
from .interop import to_cuda_view, to_torch
from .observables import add_thermal_noise_3d, apply_wedge_filter_3d, field_moments_and_pdf

_ALIAS_MAP = {
    "F_STAR10": "F_STAR10_ACG",
    "F_ESC10": "F_ESC10_ACG",
    "F_STAR7_MINI": "F_STAR7_MCG",
    "F_ESC7_MINI": "F_ESC7_MCG",
    "ALPHA_STAR": "ALPHA_STAR_ACG",
    "ALPHA_STAR_MINI": "ALPHA_STAR_MCG",
    "L_X": "LX_OVER_SFR_ACG",
    "L_X_MINI": "LX_OVER_SFR_MCG",
}


def _evolve_from_vector(
    base_inputs: Any,
    param_names: Sequence[str],
    theta: Sequence[float] | np.ndarray,
    *,
    seed_offset: int = 0,
) -> Any:
    vals = np.asarray(theta, dtype=np.float64).ravel()
    if vals.size != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} parameters ({list(param_names)}), got shape {vals.shape}"
        )
    overrides = {_ALIAS_MAP.get(name, name): float(v) for name, v in zip(param_names, vals, strict=True)}
    if "random_seed" in overrides:
        seed = round(overrides.pop("random_seed"))
        cloned = base_inputs.clone(random_seed=seed)
    elif seed_offset:
        cloned = base_inputs.clone(random_seed=int(base_inputs.random_seed) + int(seed_offset))
    else:
        cloned = base_inputs
    return cloned.evolve_input_structs(**overrides) if overrides else cloned


class Likelihood21cm:
    """21CMMC-compatible Gaussian power-spectrum log-likelihood wrapper powered by 21cmCU."""

    def __init__(
        self,
        *,
        base_inputs: Any,
        param_names: Sequence[str],
        data_pk: np.ndarray,
        sigma_pk: np.ndarray | None = None,
        cov_inv: np.ndarray | None = None,
        mode: Literal["coeval", "lightcone"] = "coeval",
        out_redshift: float | None = None,
        min_redshift: float | None = None,
        max_redshift: float | None = None,
        ps_n_bins: int | None = None,
        cuda_mode: Literal["strict", "fast"] = "fast",
        use_cuda: bool = True,
        lazy_d2h: bool = True,
        wedge_filter: bool = False,
        wedge_buffer: float = 0.0,
    ) -> None:
        self.base_inputs = base_inputs
        self.param_names = tuple(param_names)
        self.data_pk = np.asarray(data_pk, dtype=np.float64).ravel()
        self.sigma_pk = (
            np.asarray(sigma_pk, dtype=np.float64).ravel() if sigma_pk is not None else None
        )
        self.cov_inv = np.asarray(cov_inv, dtype=np.float64) if cov_inv is not None else None
        if self.sigma_pk is None and self.cov_inv is None:
            raise ValueError("Provide either sigma_pk or cov_inv")
        self.mode = mode
        self.out_redshift = out_redshift
        self.min_redshift = min_redshift
        self.max_redshift = max_redshift
        self.ps_n_bins = int(ps_n_bins) if ps_n_bins is not None else int(self.data_pk.size)
        self.cuda_mode = cuda_mode
        self.use_cuda = bool(use_cuda)
        self.lazy_d2h = bool(lazy_d2h)
        self.wedge_filter = bool(wedge_filter)
        self.wedge_buffer = float(wedge_buffer)

    def simulate(self, theta: Sequence[float] | np.ndarray) -> dict[str, Any]:
        """Run one forward realization and return 1D power spectrum + global summaries."""
        import py21cmcu

        inputs = _evolve_from_vector(self.base_inputs, self.param_names, theta)
        box_len = _normalize_box_length(
            (
                float(inputs.simulation_options.BOX_LEN),
                float(inputs.simulation_options.BOX_LEN),
                float(inputs.simulation_options.NON_CUBIC_FACTOR * inputs.simulation_options.BOX_LEN),
            )
        )
        with py21cmcu.use_mode(self.cuda_mode, use_cuda=self.use_cuda, lazy_d2h=self.lazy_d2h):
            if self.mode == "coeval":
                z_out = (
                    float(self.out_redshift)
                    if self.out_redshift is not None
                    else float(min(inputs.node_redshifts))
                )
                coeval = py21cmcu.run_coeval(
                    inputs=inputs, out_redshifts=[z_out], write=False, regenerate=True
                )[0]
                tb = np.asarray(coeval.brightness_temp, dtype=np.float32)
                if self.wedge_filter:
                    tb = apply_wedge_filter_3d(
                        tb,
                        box_len,
                        z_out,
                        buffer_h_mpc=self.wedge_buffer,
                        cosmo=inputs.cosmo_params.cosmo,
                        use_cuda=self.use_cuda,
                    )
                ps = power_spectrum_1d(
                    tb, box_len, n_bins=self.ps_n_bins, remove_mean=True, use_cuda=self.use_cuda
                )
                return {
                    "k": ps["k"],
                    "pk": ps["pk"],
                    "delta_sq": ps["delta_sq"],
                    "mean_xH": float(np.mean(coeval.neutral_fraction)),
                    "mean_Tb": float(np.mean(coeval.brightness_temp)),
                }

            z_nodes = sorted(float(z) for z in inputs.node_redshifts)
            min_z = self.min_redshift if self.min_redshift is not None else z_nodes[0] + 0.05
            max_z = self.max_redshift if self.max_redshift is not None else z_nodes[-1] - 0.05
            lc_def = py21cmcu.RectilinearLightconer.between_redshifts(
                min_redshift=float(min_z),
                max_redshift=float(max_z),
                resolution=inputs.simulation_options.cell_size,
                quantities=("brightness_temp", "neutral_fraction"),
                cosmo=inputs.cosmo_params.cosmo,
            )
            lc = py21cmcu.run_lightcone(
                lightconer=lc_def,
                inputs=inputs,
                include_dvdr_in_tau21=True,
                apply_rsds=True,
                regenerate=True,
                write=False,
            )
            chunks = lightcone_chunk_power_spectra(
                lc,
                quantity="brightness_temp_with_rsds",
                n_chunks=1,
                n_bins=self.ps_n_bins,
                remove_mean=True,
                use_cuda=self.use_cuda,
            )["chunks"]
            ps0 = chunks[0]
            return {
                "k": ps0["k"],
                "pk": ps0["pk"],
                "delta_sq": ps0["delta_sq"],
                "mean_xH": float(np.mean(lc.lightcones["neutral_fraction"])),
                "mean_Tb": float(np.mean(lc.lightcones["brightness_temp_with_rsds"])),
            }

    def log_likelihood(self, theta: Sequence[float] | np.ndarray) -> float:
        """Compute Gaussian log-likelihood log p(d | theta) = -0.5 * chi^2."""
        sim = self.simulate(theta)
        model = np.asarray(sim["delta_sq"], dtype=np.float64).ravel()
        valid = np.isfinite(model) & np.isfinite(self.data_pk)
        if not np.any(valid):
            return -np.inf
        diff = model - self.data_pk
        diff = np.where(valid, diff, 0.0)
        if self.cov_inv is not None:
            chi2 = float(diff @ self.cov_inv @ diff)
        else:
            assert self.sigma_pk is not None
            chi2 = float(np.sum((diff[valid] / self.sigma_pk[valid]) ** 2))
        return -0.5 * chi2

    def __call__(self, theta: Sequence[float] | np.ndarray) -> float:
        return self.log_likelihood(theta)


class SBISimulator:
    """sbi-compatible GPU simulator returning summary vectors, arrays, or DLPack tensors."""

    def __init__(
        self,
        *,
        base_inputs: Any,
        param_names: Sequence[str],
        mode: Literal["coeval", "lightcone"] = "coeval",
        out_redshift: float | None = None,
        min_redshift: float | None = None,
        max_redshift: float | None = None,
        summary: Literal["stats", "ps1d", "field"] = "stats",
        ps_n_bins: int = 12,
        wedge_filter: bool = True,
        wedge_buffer: float = 0.05,
        noise_sigma_mk: float | None = 2.0,
        cuda_mode: Literal["strict", "fast"] = "fast",
        use_cuda: bool = True,
        devices: tuple[int, ...] = (0,),
        workers_per_gpu: int = 1,
        as_torch: bool = False,
        as_cuda_view: bool = False,
    ) -> None:
        self.base_inputs = base_inputs
        self.param_names = tuple(param_names)
        self.mode = mode
        self.out_redshift = out_redshift
        self.min_redshift = min_redshift
        self.max_redshift = max_redshift
        self.summary = summary
        self.ps_n_bins = int(ps_n_bins)
        self.wedge_filter = bool(wedge_filter)
        self.wedge_buffer = float(wedge_buffer)
        self.noise_sigma_mk = noise_sigma_mk
        self.cuda_mode = cuda_mode
        self.use_cuda = bool(use_cuda)
        self.devices = tuple(devices)
        self.workers_per_gpu = int(workers_per_gpu)
        self.as_torch = bool(as_torch)
        self.as_cuda_view = bool(as_cuda_view)

    def _simulate_single_stats(self, theta: np.ndarray, idx: int = 0) -> Any:
        import py21cmcu

        inputs = _evolve_from_vector(self.base_inputs, self.param_names, theta, seed_offset=idx)
        z_nodes = sorted(float(z) for z in inputs.node_redshifts)
        z_out = float(self.out_redshift) if self.out_redshift is not None else z_nodes[0]
        box_len = (
            float(inputs.simulation_options.BOX_LEN),
            float(inputs.simulation_options.BOX_LEN),
            float(inputs.simulation_options.NON_CUBIC_FACTOR * inputs.simulation_options.BOX_LEN),
        )

        with py21cmcu.use_mode(self.cuda_mode, use_cuda=self.use_cuda, lazy_d2h=True):
            if self.mode == "coeval":
                cv = py21cmcu.run_coeval(
                    inputs=inputs, out_redshifts=[z_out], write=False, regenerate=True
                )[0]
                arr = np.asarray(cv.brightness_temp, dtype=np.float32)
                mean_xh = float(np.mean(cv.neutral_fraction))
                mean_tb = float(np.mean(arr))
                box_eval = box_len
                z_eval = z_out
            else:
                min_z = self.min_redshift if self.min_redshift is not None else z_nodes[0] + 0.05
                max_z = self.max_redshift if self.max_redshift is not None else z_nodes[-1] - 0.05
                lc_def = py21cmcu.RectilinearLightconer.between_redshifts(
                    min_redshift=float(min_z),
                    max_redshift=float(max_z),
                    resolution=inputs.simulation_options.cell_size,
                    quantities=("brightness_temp", "neutral_fraction"),
                    cosmo=inputs.cosmo_params.cosmo,
                )
                lc = py21cmcu.run_lightcone(
                    lightconer=lc_def,
                    inputs=inputs,
                    include_dvdr_in_tau21=True,
                    apply_rsds=True,
                    regenerate=True,
                    write=False,
                )
                arr = np.asarray(lc.lightcones["brightness_temp_with_rsds"], dtype=np.float32)
                mean_xh = float(np.mean(lc.lightcones["neutral_fraction"]))
                mean_tb = float(np.mean(arr))
                cell = float(inputs.simulation_options.cell_size.value)
                box_eval = (arr.shape[0] * cell, arr.shape[1] * cell, arr.shape[2] * cell)
                z_eval = float(np.mean(lc.lightcone_redshifts))

            if self.noise_sigma_mk is not None:
                arr = add_thermal_noise_3d(
                    arr,
                    box_eval,
                    sigma_mk=self.noise_sigma_mk,
                    redshift=z_eval,
                    seed=int(inputs.random_seed) * 31 + 7,
                    use_cuda=self.use_cuda,
                )
            if self.wedge_filter:
                arr = apply_wedge_filter_3d(
                    arr,
                    box_eval,
                    z_eval,
                    buffer_h_mpc=self.wedge_buffer,
                    cosmo=inputs.cosmo_params.cosmo,
                    use_cuda=self.use_cuda,
                )

            if self.summary == "field":
                if self.as_torch:
                    return to_torch(arr)
                if self.as_cuda_view:
                    return to_cuda_view(arr)
                return arr

            ps = power_spectrum_1d(
                arr, box_eval, n_bins=self.ps_n_bins, remove_mean=True, use_cuda=self.use_cuda
            )
            delta2 = np.nan_to_num(ps["delta_sq"], nan=0.0).astype(np.float32)
            if self.summary == "ps1d":
                vec = delta2
            else:
                mom = field_moments_and_pdf(arr, n_bins=16, use_cuda=self.use_cuda)
                vec = np.concatenate(
                    [
                        np.array(
                            [mean_xh, mean_tb, mom["std"], mom["skewness"], mom["kurtosis"]],
                            dtype=np.float32,
                        ),
                        delta2,
                    ]
                )
            return vec

    def __call__(self, theta: Sequence[float] | np.ndarray) -> Any:
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 1:
            return self._simulate_single_stats(arr, idx=0)
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D parameter array, got shape {arr.shape}")

        if self.summary == "field":
            return [self._simulate_single_stats(arr[i], idx=i) for i in range(arr.shape[0])]

        if self.summary == "ps1d" and len(arr) >= 2:
            tasks = []
            for i in range(arr.shape[0]):
                inp = _evolve_from_vector(self.base_inputs, self.param_names, arr[i], seed_offset=i)
                tasks.append(
                    BatchTask(
                        task_id=f"sbi_{i:04d}",
                        inputs=inp,
                        mode=self.mode,
                        out_redshifts=(float(self.out_redshift),)
                        if self.out_redshift is not None and self.mode == "coeval"
                        else None,
                        min_redshift=self.min_redshift,
                        max_redshift=self.max_redshift,
                        cuda_mode=self.cuda_mode,
                        ps_n_bins=self.ps_n_bins,
                        wedge_filter=self.wedge_filter,
                        wedge_buffer=self.wedge_buffer,
                        noise_sigma_mk=self.noise_sigma_mk,
                    )
                )
            results = run_batch(
                tasks,
                devices=self.devices,
                workers_per_gpu=self.workers_per_gpu,
                use_cuda=self.use_cuda,
            )
            rows = [
                np.nan_to_num(r.ps_1d["spectra"][0]["delta_sq"], nan=0.0).astype(np.float32)
                for r in results
            ]
            return np.stack(rows, axis=0)

        rows = [self._simulate_single_stats(arr[i], idx=i) for i in range(arr.shape[0])]
        return np.stack(rows, axis=0)


__all__ = ["Likelihood21cm", "SBISimulator"]
