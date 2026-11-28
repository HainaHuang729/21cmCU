"""Common-Random-Number (CRN) Fisher Matrix Forecaster for 21cmCU and multi-probe inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._internal import evolve_from_vector as _evolve_from_vector
from ._internal import normalize_box_length as _normalize_box_length
from .analysis import (
    power_spectrum_1d,
    power_spectrum_2d,
)
from .instrument import (
    InterferometerConfig,
    forecast_power_spectrum_2d_error,
    forecast_power_spectrum_error,
)
from .observables import field_moments_and_pdf
from .topology import bispectrum_3d, minkowski_functionals_3d


def _symmetrize_and_invert(fisher: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    sym = 0.5 * (fisher + fisher.T)
    eigvals = np.linalg.eigvalsh(sym)
    min_ev = float(np.min(eigvals))
    max_ev = float(np.max(np.abs(eigvals)))
    cond = (max_ev / min_ev) if min_ev > 0.0 else float("inf")
    if min_ev > 0.0 and cond < 1e12:
        cov = np.linalg.inv(sym)
    else:
        cov = np.linalg.pinv(sym, rcond=1e-12)
    cov = 0.5 * (cov + cov.T)
    return sym, cov, float(cond)


def _normalize_prior_vector(
    prior_sigma: Mapping[str, float] | Sequence[float],
    param_names: tuple[str, ...],
) -> np.ndarray:
    n_p = len(param_names)
    if isinstance(prior_sigma, Mapping):
        out = np.full(n_p, np.inf, dtype=np.float64)
        for i, name in enumerate(param_names):
            if name in prior_sigma:
                out[i] = float(prior_sigma[name])
    else:
        out = np.asarray(prior_sigma, dtype=np.float64).ravel()
        if out.size != n_p:
            raise ValueError(f"Expected {n_p} prior sigmas, got {out.size}")
    if np.any(out <= 0.0):
        raise ValueError("All prior_sigma values must be strictly positive (or inf)")
    return out


@dataclass
class FisherForecastResult:
    """Container for Fisher matrix, marginalized parameter covariance, and forecast diagnostics."""

    param_names: tuple[str, ...]
    fiducial: np.ndarray
    step_sizes: np.ndarray
    fisher_matrix: np.ndarray
    covariance_matrix: np.ndarray
    jacobian: np.ndarray
    fiducial_observable: np.ndarray
    data_sigma: np.ndarray
    valid_mask: np.ndarray
    sigma_marginalized: np.ndarray
    sigma_unmarginalized: np.ndarray
    correlation_matrix: np.ndarray
    fom: float
    condition_number: float
    n_seeds: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_prior(
        self, prior_sigma: Mapping[str, float] | Sequence[float]
    ) -> FisherForecastResult:
        """Return a new FisherForecastResult with independent Gaussian priors added."""
        sig = _normalize_prior_vector(prior_sigma, self.param_names)
        inv_var = np.where(np.isfinite(sig), 1.0 / (sig**2), 0.0)
        new_fisher = self.fisher_matrix.copy()
        new_fisher[np.diag_indices_from(new_fisher)] += inv_var
        return _build_fisher_result(
            param_names=self.param_names,
            fiducial=self.fiducial,
            step_sizes=self.step_sizes,
            fisher_matrix=new_fisher,
            jacobian=self.jacobian,
            fiducial_observable=self.fiducial_observable,
            data_sigma=self.data_sigma,
            valid_mask=self.valid_mask,
            n_seeds=self.n_seeds,
            metadata={**self.metadata, "prior_added": True},
        )

    def combine(self, other: FisherForecastResult) -> FisherForecastResult:
        """Combine two independent Fisher forecasts over matching parameters (F = F1 + F2)."""
        if self.param_names != other.param_names:
            raise ValueError(
                f"Cannot combine Fisher matrices with mismatched param_names: "
                f"{self.param_names} vs {other.param_names}"
            )
        if not np.allclose(self.fiducial, other.fiducial, rtol=1e-6, atol=1e-9):
            raise ValueError("Cannot combine Fisher matrices evaluated at different fiducials")
        return _build_fisher_result(
            param_names=self.param_names,
            fiducial=self.fiducial,
            step_sizes=self.step_sizes,
            fisher_matrix=self.fisher_matrix + other.fisher_matrix,
            jacobian=self.jacobian,
            fiducial_observable=self.fiducial_observable,
            data_sigma=self.data_sigma,
            valid_mask=self.valid_mask,
            n_seeds=self.n_seeds + other.n_seeds,
            metadata={"combined": True, "sources": [self.metadata, other.metadata]},
        )

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of parameter uncertainties and FoM."""
        return {
            "param_names": list(self.param_names),
            "fiducial": [float(v) for v in self.fiducial],
            "sigma_marginalized": {
                k: float(v) for k, v in zip(self.param_names, self.sigma_marginalized, strict=True)
            },
            "sigma_unmarginalized": {
                k: float(v)
                for k, v in zip(self.param_names, self.sigma_unmarginalized, strict=True)
            },
            "fom": float(self.fom),
            "condition_number": float(self.condition_number),
            "n_valid_bins": int(np.sum(self.valid_mask)),
            "n_seeds": int(self.n_seeds),
            "correlation_matrix": self.correlation_matrix.tolist(),
        }


def _build_fisher_result(
    *,
    param_names: tuple[str, ...],
    fiducial: np.ndarray,
    step_sizes: np.ndarray,
    fisher_matrix: np.ndarray,
    jacobian: np.ndarray,
    fiducial_observable: np.ndarray,
    data_sigma: np.ndarray,
    valid_mask: np.ndarray,
    n_seeds: int,
    metadata: dict[str, Any] | None = None,
) -> FisherForecastResult:
    sym, cov, cond = _symmetrize_and_invert(fisher_matrix)
    diag_f = np.diag(sym)
    diag_c = np.diag(cov)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_marg = np.where(diag_c > 0.0, np.sqrt(diag_c), np.inf)
        sig_unmarg = np.where(diag_f > 0.0, 1.0 / np.sqrt(diag_f), np.inf)
        denom = np.outer(sig_marg, sig_marg)
        corr = np.where((denom > 0.0) & np.isfinite(denom), cov / denom, 0.0)
        np.fill_diagonal(corr, 1.0)
    det_f = float(np.linalg.det(sym))
    fom = float(np.sqrt(max(det_f, 0.0)))
    return FisherForecastResult(
        param_names=tuple(param_names),
        fiducial=np.asarray(fiducial, dtype=np.float64).copy(),
        step_sizes=np.asarray(step_sizes, dtype=np.float64).copy(),
        fisher_matrix=sym,
        covariance_matrix=cov,
        jacobian=np.asarray(jacobian, dtype=np.float64).copy(),
        fiducial_observable=np.asarray(fiducial_observable, dtype=np.float64).copy(),
        data_sigma=np.asarray(data_sigma, dtype=np.float64).copy(),
        valid_mask=np.asarray(valid_mask, dtype=bool).copy(),
        sigma_marginalized=sig_marg,
        sigma_unmarginalized=sig_unmarg,
        correlation_matrix=corr,
        fom=fom,
        condition_number=cond,
        n_seeds=int(n_seeds),
        metadata={} if metadata is None else dict(metadata),
    )


def _resolve_vector(
    vals: Mapping[str, float] | Sequence[float] | np.ndarray,
    param_names: tuple[str, ...],
    *,
    label: str,
) -> np.ndarray:
    n_p = len(param_names)
    if isinstance(vals, Mapping):
        missing = [p for p in param_names if p not in vals]
        if missing:
            raise ValueError(f"Missing {label} entries for parameters: {missing}")
        return np.array([float(vals[p]) for p in param_names], dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64).ravel()
    if arr.size != n_p:
        raise ValueError(f"Expected {n_p} {label} entries, got shape {arr.shape}")
    return arr


class FisherForecaster:
    """Common-Random-Number (CRN) low-variance Fisher matrix forecaster for 21cmCU."""

    def __init__(
        self,
        *,
        base_inputs: Any | None,
        param_names: Sequence[str],
        fiducial: Mapping[str, float] | Sequence[float] | np.ndarray,
        step_sizes: Mapping[str, float] | Sequence[float] | np.ndarray,
        out_redshifts: Sequence[float] = (8.0,),
        instrument: str | InterferometerConfig = "SKA1-Low",
        summary: Literal["ps1d", "ps2d", "nongaussian"] = "ps1d",
        observable_fn: Callable[[Any, Any], np.ndarray] | None = None,
        ps_n_bins: int = 12,
        n_kperp: int = 8,
        n_kpara: int = 8,
        n_seeds: int = 1,
        order: Literal["central", "richardson"] = "central",
        model_error_frac: float = 0.0,
        covariance: np.ndarray | None = None,
        data_sigma: np.ndarray | None = None,
        prior_sigma: Mapping[str, float] | Sequence[float] | None = None,
        include_sample_variance: bool = True,
        apply_wedge: bool = True,
        apply_uv_mask: bool = True,
        cuda_mode: Literal["strict", "fast"] = "strict",
        use_cuda: bool = True,
        lazy_d2h: bool = True,
        instrument_kwargs: dict[str, Any] | None = None,
        _model_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> None:
        self.base_inputs = base_inputs
        self.param_names = tuple(str(p) for p in param_names)
        if not self.param_names:
            raise ValueError("param_names must be non-empty")
        self.fiducial = _resolve_vector(fiducial, self.param_names, label="fiducial")
        self.step_sizes = _resolve_vector(step_sizes, self.param_names, label="step_sizes")
        if np.any(self.step_sizes <= 0.0):
            raise ValueError("All step_sizes must be > 0")
        self.out_redshifts = tuple(float(z) for z in out_redshifts)
        self.instrument = instrument
        self.summary = summary
        self.observable_fn = observable_fn
        self.ps_n_bins = int(ps_n_bins)
        self.n_kperp = int(n_kperp)
        self.n_kpara = int(n_kpara)
        if int(n_seeds) < 1:
            raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
        self.n_seeds = int(n_seeds)
        if order not in ("central", "richardson"):
            raise ValueError(f"order must be 'central' or 'richardson', got {order!r}")
        self.order = order
        self.model_error_frac = float(model_error_frac)
        self.covariance = (
            np.asarray(covariance, dtype=np.float64) if covariance is not None else None
        )
        self.data_sigma = (
            np.asarray(data_sigma, dtype=np.float64).ravel() if data_sigma is not None else None
        )
        self.prior_sigma = prior_sigma
        self.include_sample_variance = bool(include_sample_variance)
        self.apply_wedge = bool(apply_wedge)
        self.apply_uv_mask = bool(apply_uv_mask)
        self.cuda_mode = cuda_mode
        self.use_cuda = bool(use_cuda)
        self.lazy_d2h = bool(lazy_d2h)
        self.instrument_kwargs = dict(instrument_kwargs or {})
        self._model_fn = _model_fn

        if self._model_fn is None and self.base_inputs is None:
            raise ValueError("Provide base_inputs (or use FisherForecaster.from_callable())")

    @classmethod
    def from_callable(
        cls,
        model_fn: Callable[[np.ndarray, int], np.ndarray],
        param_names: Sequence[str],
        fiducial: Mapping[str, float] | Sequence[float] | np.ndarray,
        step_sizes: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        covariance: np.ndarray | None = None,
        data_sigma: np.ndarray | None = None,
        prior_sigma: Mapping[str, float] | Sequence[float] | None = None,
        n_seeds: int = 1,
        order: Literal["central", "richardson"] = "central",
        model_error_frac: float = 0.0,
    ) -> FisherForecaster:
        """Build a CRN Fisher forecaster from an arbitrary ``model_fn(theta, seed) -> vector``."""
        if covariance is None and data_sigma is None and model_error_frac <= 0.0:
            raise ValueError(
                "For custom model_fn, provide covariance, data_sigma, or model_error_frac > 0"
            )
        return cls(
            base_inputs=None,
            param_names=param_names,
            fiducial=fiducial,
            step_sizes=step_sizes,
            covariance=covariance,
            data_sigma=data_sigma,
            prior_sigma=prior_sigma,
            n_seeds=n_seeds,
            order=order,
            model_error_frac=model_error_frac,
            _model_fn=model_fn,
        )

    def _evaluate_21cm(
        self, theta: np.ndarray, seed_offset: int, *, need_auto_sigma: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        import py21cmcu

        inputs = _evolve_from_vector(
            self.base_inputs,
            self.param_names,
            theta,
            seed_offset=seed_offset,
        )
        box_len = _normalize_box_length(
            (
                float(inputs.simulation_options.BOX_LEN),
                float(inputs.simulation_options.BOX_LEN),
                float(
                    inputs.simulation_options.NON_CUBIC_FACTOR * inputs.simulation_options.BOX_LEN
                ),
            )
        )
        with py21cmcu.use_mode(self.cuda_mode, use_cuda=self.use_cuda, lazy_d2h=self.lazy_d2h):
            coevals = py21cmcu.run_coeval(
                inputs=inputs,
                out_redshifts=self.out_redshifts,
                write=False,
                regenerate=True,
            )
            if self.observable_fn is not None:
                obs = np.asarray(self.observable_fn(coevals, inputs), dtype=np.float64).ravel()
                return obs, None

            obs_parts: list[np.ndarray] = []
            sig_parts: list[np.ndarray] = []
            cosmo = inputs.cosmo_params.cosmo

            for cv in coevals:
                tb = np.asarray(cv.brightness_temp, dtype=np.float32)
                z = float(cv.redshift)
                if self.summary == "ps1d":
                    ps = power_spectrum_1d(
                        tb,
                        box_len,
                        n_bins=self.ps_n_bins,
                        remove_mean=True,
                        use_cuda=self.use_cuda,
                    )
                    obs_parts.append(np.asarray(ps["delta_sq"], dtype=np.float64))
                    if need_auto_sigma:
                        fc = forecast_power_spectrum_error(
                            tb.shape,
                            box_len,
                            z,
                            self.instrument,
                            signal_field=tb,
                            n_bins=self.ps_n_bins,
                            include_sample_variance=self.include_sample_variance,
                            apply_uv_mask=self.apply_uv_mask,
                            apply_wedge=self.apply_wedge,
                            use_cuda=self.use_cuda,
                            cosmo=cosmo,
                            **self.instrument_kwargs,
                        )
                        sig_parts.append(np.asarray(fc["sigma_delta_sq"], dtype=np.float64))
                elif self.summary == "ps2d":
                    ps2 = power_spectrum_2d(
                        tb,
                        box_len,
                        n_kperp=self.n_kperp,
                        n_kpara=self.n_kpara,
                        remove_mean=True,
                        use_cuda=self.use_cuda,
                    )
                    obs_parts.append(np.asarray(ps2["pk"], dtype=np.float64).ravel())
                    if need_auto_sigma:
                        fc2 = forecast_power_spectrum_2d_error(
                            tb.shape,
                            box_len,
                            z,
                            self.instrument,
                            signal_field=tb,
                            kperp_edges=ps2["kperp_edges"],
                            kpara_edges=ps2["kpara_edges"],
                            include_sample_variance=self.include_sample_variance,
                            apply_uv_mask=self.apply_uv_mask,
                            apply_wedge=self.apply_wedge,
                            use_cuda=self.use_cuda,
                            cosmo=cosmo,
                            **self.instrument_kwargs,
                        )
                        sig_parts.append(np.asarray(fc2["sigma_pk"], dtype=np.float64).ravel())
                elif self.summary == "nongaussian":
                    ps = power_spectrum_1d(
                        tb,
                        box_len,
                        n_bins=self.ps_n_bins,
                        remove_mean=True,
                        use_cuda=self.use_cuda,
                    )
                    mom = field_moments_and_pdf(tb, use_cuda=self.use_cuda)
                    bi = bispectrum_3d(
                        tb,
                        box_len,
                        config="equilateral",
                        n_bins=max(3, min(6, self.ps_n_bins)),
                        use_cuda=self.use_cuda,
                    )
                    mk = minkowski_functionals_3d(
                        tb,
                        box_len,
                        n_thresholds=5,
                        standardize=True,
                        use_cuda=self.use_cuda,
                    )
                    obs_parts.extend(
                        [
                            np.array(
                                [
                                    float(np.mean(cv.neutral_fraction)),
                                    mom["mean"],
                                    mom["variance"],
                                    mom["skewness"],
                                    mom["kurtosis"],
                                ],
                                dtype=np.float64,
                            ),
                            np.asarray(ps["delta_sq"], dtype=np.float64),
                            np.asarray(bi["Q"], dtype=np.float64),
                            np.asarray(mk["V1"], dtype=np.float64),
                            np.asarray(mk["V2"], dtype=np.float64),
                            np.asarray(mk["V3"], dtype=np.float64),
                        ]
                    )
                else:
                    raise ValueError(
                        f"summary must be 'ps1d', 'ps2d', or 'nongaussian', got {self.summary!r}"
                    )

            obs = np.concatenate(obs_parts)
            auto_sig = np.concatenate(sig_parts) if (need_auto_sigma and sig_parts) else None
            return obs, auto_sig

    def _evaluate(
        self, theta: np.ndarray, seed_idx: int, *, need_auto_sigma: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._model_fn is not None:
            return np.asarray(self._model_fn(theta, seed_idx), dtype=np.float64).ravel(), None
        return self._evaluate_21cm(theta, seed_idx, need_auto_sigma=need_auto_sigma)

    def _compute_jacobian_and_fiducial(
        self, step_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        eff_steps = self.step_sizes * float(step_scale)
        n_p = len(self.param_names)
        need_auto = (
            self._model_fn is None
            and self.covariance is None
            and self.data_sigma is None
            and self.summary in ("ps1d", "ps2d")
        )

        fid_samples: list[np.ndarray] = []
        sig_samples: list[np.ndarray] = []
        jac_samples: list[np.ndarray] = []

        for s in range(self.n_seeds):
            obs0, sig0 = self._evaluate(self.fiducial, s, need_auto_sigma=need_auto)
            fid_samples.append(obs0)
            if sig0 is not None:
                sig_samples.append(sig0)

            jac_s = np.zeros((obs0.size, n_p), dtype=np.float64)
            for j in range(n_p):
                h = float(eff_steps[j])
                th_p = self.fiducial.copy()
                th_m = self.fiducial.copy()
                th_p[j] += h
                th_m[j] -= h
                op, _ = self._evaluate(th_p, s, need_auto_sigma=False)
                om, _ = self._evaluate(th_m, s, need_auto_sigma=False)
                d_full = (op - om) / (2.0 * h)
                if self.order == "richardson":
                    th_ph = self.fiducial.copy()
                    th_mh = self.fiducial.copy()
                    th_ph[j] += 0.5 * h
                    th_mh[j] -= 0.5 * h
                    oph, _ = self._evaluate(th_ph, s, need_auto_sigma=False)
                    omh, _ = self._evaluate(th_mh, s, need_auto_sigma=False)
                    d_half = (oph - omh) / h
                    jac_s[:, j] = (4.0 * d_half - d_full) / 3.0
                else:
                    jac_s[:, j] = d_full
            jac_samples.append(jac_s)

        fid_stack = np.stack(fid_samples, axis=0)
        fid_mean = np.mean(fid_stack, axis=0)
        jac_mean = np.mean(np.stack(jac_samples, axis=0), axis=0)

        auto_sig: np.ndarray | None = None
        if sig_samples:
            auto_sig = np.mean(np.stack(sig_samples, axis=0), axis=0)
        elif self.covariance is None and self.data_sigma is None:
            if self.n_seeds >= 2:
                auto_sig = np.std(fid_stack, axis=0, ddof=1)
            elif self.model_error_frac > 0.0:
                auto_sig = np.zeros_like(fid_mean)
            else:
                raise ValueError(
                    f"Automatic noise forecast requires summary='ps1d'/'ps2d'; for "
                    f"summary={self.summary!r}, pass covariance, data_sigma, n_seeds>=2, "
                    "or model_error_frac>0."
                )

        return fid_mean, jac_mean, auto_sig, eff_steps

    def run(self, *, step_scale: float = 1.0) -> FisherForecastResult:
        """Evaluate CRN finite differences around the fiducial and return FisherForecastResult."""
        fid_obs, jac, auto_sig, eff_steps = self._compute_jacobian_and_fiducial(
            step_scale=step_scale
        )
        n_obs = fid_obs.size

        if self.covariance is not None:
            cov_data = np.asarray(self.covariance, dtype=np.float64)
            if cov_data.shape != (n_obs, n_obs):
                raise ValueError(
                    f"covariance shape {cov_data.shape} does not match observable length {n_obs}"
                )
            data_sig = np.sqrt(np.maximum(np.diag(cov_data), 0.0))
        else:
            base_sig = self.data_sigma if self.data_sigma is not None else auto_sig
            assert base_sig is not None
            if base_sig.size != n_obs:
                raise ValueError(
                    f"data_sigma length {base_sig.size} does not match observable length {n_obs}"
                )
            data_sig = np.array(base_sig, dtype=np.float64, copy=True)
            cov_data = None

        if self.model_error_frac > 0.0:
            mod_var = (self.model_error_frac * np.maximum(np.abs(fid_obs), 1e-12)) ** 2
            data_sig = np.sqrt(data_sig**2 + mod_var)
            if cov_data is not None:
                cov_data = cov_data.copy()
                cov_data[np.diag_indices_from(cov_data)] += mod_var

        valid_mask = np.isfinite(data_sig) & (data_sig > 0.0) & np.all(np.isfinite(jac), axis=1)
        if not np.any(valid_mask):
            raise RuntimeError("No valid finite bins with positive error remained for Fisher matrix")

        j_val = jac[valid_mask]
        if cov_data is not None:
            c_val = cov_data[np.ix_(valid_mask, valid_mask)]
            c_val = 0.5 * (c_val + c_val.T)
            c_inv_j = np.linalg.solve(c_val, j_val)
            fisher = j_val.T @ c_inv_j
        else:
            w = 1.0 / (data_sig[valid_mask] ** 2)
            fisher = j_val.T @ (w[:, None] * j_val)

        res = _build_fisher_result(
            param_names=self.param_names,
            fiducial=self.fiducial,
            step_sizes=eff_steps,
            fisher_matrix=fisher,
            jacobian=jac,
            fiducial_observable=fid_obs,
            data_sigma=data_sig,
            valid_mask=valid_mask,
            n_seeds=self.n_seeds,
            metadata={
                "summary": self.summary,
                "order": self.order,
                "out_redshifts": list(self.out_redshifts),
                "instrument": (
                    self.instrument.name
                    if isinstance(self.instrument, InterferometerConfig)
                    else str(self.instrument)
                ),
            },
        )
        if self.prior_sigma is not None:
            res = res.with_prior(self.prior_sigma)
        return res

    def check_step_convergence(
        self, step_scales: Sequence[float] = (0.5, 1.0, 2.0)
    ) -> dict[str, Any]:
        """Run Fisher forecasts across scaled step sizes to verify derivative convergence."""
        scales = [float(s) for s in step_scales]
        if not scales:
            raise ValueError("step_scales must be non-empty")
        results = [self.run(step_scale=s) for s in scales]
        ref = results[len(results) // 2]
        jac_ref_norm = float(np.linalg.norm(ref.jacobian[ref.valid_mask]))

        jac_diffs: list[float] = []
        sig_diffs: list[float] = []
        for r in results:
            mask = r.valid_mask & ref.valid_mask
            j_diff = float(np.linalg.norm(r.jacobian[mask] - ref.jacobian[mask])) / max(
                jac_ref_norm, 1e-30
            )
            s_diff = float(
                np.max(
                    np.abs(r.sigma_marginalized - ref.sigma_marginalized)
                    / np.maximum(np.abs(ref.sigma_marginalized), 1e-30)
                )
            )
            jac_diffs.append(j_diff)
            sig_diffs.append(s_diff)

        return {
            "scales": scales,
            "results": results,
            "sigma_marginalized": np.stack([r.sigma_marginalized for r in results], axis=0),
            "max_jacobian_rel_diff": float(max(jac_diffs)),
            "max_sigma_rel_diff": float(max(sig_diffs)),
        }


def fisher_forecast_21cm(
    base_inputs: Any,
    param_names: Sequence[str],
    fiducial: Mapping[str, float] | Sequence[float] | np.ndarray,
    step_sizes: Mapping[str, float] | Sequence[float] | np.ndarray,
    **kwargs: Any,
) -> FisherForecastResult:
    """Convenience wrapper executing CRN Fisher matrix forecast via ``FisherForecaster``."""
    return FisherForecaster(
        base_inputs=base_inputs,
        param_names=param_names,
        fiducial=fiducial,
        step_sizes=step_sizes,
        **kwargs,
    ).run()


__all__ = [
    "FisherForecastResult",
    "FisherForecaster",
    "fisher_forecast_21cm",
]
