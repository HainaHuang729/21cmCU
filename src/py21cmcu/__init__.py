"""The 21cmCU GPU-accelerated cosmological simulation package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import py21cmfast as _p21
from py21cmfast import *  # noqa: F403
from py21cmfast import config

from ._runtime import (
    CudaMode,
    cleanup_cuda,
    clear_class_cache,
    cuda_api_count,
    cuda_available,
    cuda_compiled,
    cuda_device_count,
    cuda_stats,
    get_cuda_devices,
    get_cuda_mode,
    reset_cuda_stats,
    set_cuda_devices,
    set_cuda_mode,
    use_mode,
)
from ._version import __version__

if TYPE_CHECKING:
    from ._results import (
        BispectrumResult,
        CrossPowerResult,
        KSZMapResult,
        MFPResult,
        MinkowskiResult,
        PowerSpectrum1DResult,
        PowerSpectrum2DResult,
        ResultMapping,
    )
    from .analysis import (
        lightcone_chunk_power_spectra,
        lightcone_chunk_power_spectra_2d,
        power_spectrum_1d,
        power_spectrum_2d,
    )
    from .batch import BatchPool, BatchResult, BatchTask, run_batch, save_batch_h5
    from .doctor import doctor
    from .fisher import FisherForecastResult, FisherForecaster, fisher_forecast_21cm
    from .inference import Likelihood21cm, OnlineSBIDataset, SBISimulator
    from .instrument import (
        INSTRUMENT_PRESETS,
        InterferometerConfig,
        baseline_density_uv,
        forecast_power_spectrum_2d_error,
        forecast_power_spectrum_error,
        get_instrument_preset,
        mock_observation_3d,
        thermal_noise_power_spectrum,
    )
    from .interop import CudaFieldView, lightcone_to_cuda_views, to_cuda_view, to_torch
    from .io import save_lightcone
    from .multiprobe import (
        cic_deposit_3d,
        cross_power_21cm_galaxy,
        cross_power_21cm_galaxy_error,
        ksz_momentum_map,
        load_gravitycu_halos,
    )
    from .observables import (
        add_thermal_noise_3d,
        apply_wedge_filter_3d,
        field_moments_and_pdf,
        generate_sobol_parameter_grid,
        horizon_slope,
        horizon_wedge_mask,
    )
    from .shortcuts import quick_coeval, quick_lightcone
    from .topology import (
        bispectrum_3d,
        bubble_size_distribution_mfp,
        minkowski_functionals_3d,
    )
    from .viz import (
        export_lightcone_3d_html,
        export_volume_3d_html,
        plot_coeval_slices,
        plot_fisher_contours,
        plot_lightcone_summary,
        plot_mock_comparison,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # results
    "BispectrumResult": ("._results", "BispectrumResult"),
    "CrossPowerResult": ("._results", "CrossPowerResult"),
    "KSZMapResult": ("._results", "KSZMapResult"),
    "MFPResult": ("._results", "MFPResult"),
    "MinkowskiResult": ("._results", "MinkowskiResult"),
    "PowerSpectrum1DResult": ("._results", "PowerSpectrum1DResult"),
    "PowerSpectrum2DResult": ("._results", "PowerSpectrum2DResult"),
    "ResultMapping": ("._results", "ResultMapping"),
    # analysis
    "lightcone_chunk_power_spectra": (".analysis", "lightcone_chunk_power_spectra"),
    "lightcone_chunk_power_spectra_2d": (".analysis", "lightcone_chunk_power_spectra_2d"),
    "power_spectrum_1d": (".analysis", "power_spectrum_1d"),
    "power_spectrum_2d": (".analysis", "power_spectrum_2d"),
    # batch
    "BatchPool": (".batch", "BatchPool"),
    "BatchResult": (".batch", "BatchResult"),
    "BatchTask": (".batch", "BatchTask"),
    "run_batch": (".batch", "run_batch"),
    "save_batch_h5": (".batch", "save_batch_h5"),
    # doctor
    "doctor": (".doctor", "doctor"),
    # fisher
    "FisherForecastResult": (".fisher", "FisherForecastResult"),
    "FisherForecaster": (".fisher", "FisherForecaster"),
    "fisher_forecast_21cm": (".fisher", "fisher_forecast_21cm"),
    # inference
    "Likelihood21cm": (".inference", "Likelihood21cm"),
    "OnlineSBIDataset": (".inference", "OnlineSBIDataset"),
    "SBISimulator": (".inference", "SBISimulator"),
    # instrument
    "INSTRUMENT_PRESETS": (".instrument", "INSTRUMENT_PRESETS"),
    "InterferometerConfig": (".instrument", "InterferometerConfig"),
    "baseline_density_uv": (".instrument", "baseline_density_uv"),
    "forecast_power_spectrum_2d_error": (".instrument", "forecast_power_spectrum_2d_error"),
    "forecast_power_spectrum_error": (".instrument", "forecast_power_spectrum_error"),
    "get_instrument_preset": (".instrument", "get_instrument_preset"),
    "mock_observation_3d": (".instrument", "mock_observation_3d"),
    "thermal_noise_power_spectrum": (".instrument", "thermal_noise_power_spectrum"),
    # interop
    "CudaFieldView": (".interop", "CudaFieldView"),
    "lightcone_to_cuda_views": (".interop", "lightcone_to_cuda_views"),
    "to_cuda_view": (".interop", "to_cuda_view"),
    "to_torch": (".interop", "to_torch"),
    # io
    "save_lightcone": (".io", "save_lightcone"),
    # multiprobe
    "cic_deposit_3d": (".multiprobe", "cic_deposit_3d"),
    "cross_power_21cm_galaxy": (".multiprobe", "cross_power_21cm_galaxy"),
    "cross_power_21cm_galaxy_error": (".multiprobe", "cross_power_21cm_galaxy_error"),
    "ksz_momentum_map": (".multiprobe", "ksz_momentum_map"),
    "load_gravitycu_halos": (".multiprobe", "load_gravitycu_halos"),
    # observables
    "add_thermal_noise_3d": (".observables", "add_thermal_noise_3d"),
    "apply_wedge_filter_3d": (".observables", "apply_wedge_filter_3d"),
    "field_moments_and_pdf": (".observables", "field_moments_and_pdf"),
    "generate_sobol_parameter_grid": (".observables", "generate_sobol_parameter_grid"),
    "horizon_slope": (".observables", "horizon_slope"),
    "horizon_wedge_mask": (".observables", "horizon_wedge_mask"),
    # shortcuts
    "quick_coeval": (".shortcuts", "quick_coeval"),
    "quick_lightcone": (".shortcuts", "quick_lightcone"),
    # topology
    "bispectrum_3d": (".topology", "bispectrum_3d"),
    "bubble_size_distribution_mfp": (".topology", "bubble_size_distribution_mfp"),
    "minkowski_functionals_3d": (".topology", "minkowski_functionals_3d"),
    # viz
    "export_lightcone_3d_html": (".viz", "export_lightcone_3d_html"),
    "export_volume_3d_html": (".viz", "export_volume_3d_html"),
    "plot_coeval_slices": (".viz", "plot_coeval_slices"),
    "plot_fisher_contours": (".viz", "plot_fisher_contours"),
    "plot_lightcone_summary": (".viz", "plot_lightcone_summary"),
    "plot_mock_comparison": (".viz", "plot_mock_comparison"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public submodule symbols on first attribute access (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted public attributes including lazy exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    *_p21.__all__,
    "__version__",
    "BatchPool",
    "BatchResult",
    "BatchTask",
    "BispectrumResult",
    "CrossPowerResult",
    "CudaFieldView",
    "CudaMode",
    "FisherForecastResult",
    "FisherForecaster",
    "INSTRUMENT_PRESETS",
    "InterferometerConfig",
    "KSZMapResult",
    "Likelihood21cm",
    "MFPResult",
    "MinkowskiResult",
    "OnlineSBIDataset",
    "PowerSpectrum1DResult",
    "PowerSpectrum2DResult",
    "ResultMapping",
    "SBISimulator",
    "add_thermal_noise_3d",
    "apply_wedge_filter_3d",
    "baseline_density_uv",
    "bispectrum_3d",
    "bubble_size_distribution_mfp",
    "cic_deposit_3d",
    "cleanup_cuda",
    "clear_class_cache",
    "config",
    "cross_power_21cm_galaxy",
    "cross_power_21cm_galaxy_error",
    "cuda_api_count",
    "cuda_available",
    "cuda_compiled",
    "cuda_device_count",
    "cuda_stats",
    "doctor",
    "export_lightcone_3d_html",
    "export_volume_3d_html",
    "field_moments_and_pdf",
    "fisher_forecast_21cm",
    "forecast_power_spectrum_2d_error",
    "forecast_power_spectrum_error",
    "generate_sobol_parameter_grid",
    "get_cuda_devices",
    "get_cuda_mode",
    "get_instrument_preset",
    "horizon_slope",
    "horizon_wedge_mask",
    "ksz_momentum_map",
    "lightcone_chunk_power_spectra",
    "lightcone_chunk_power_spectra_2d",
    "lightcone_to_cuda_views",
    "load_gravitycu_halos",
    "minkowski_functionals_3d",
    "mock_observation_3d",
    "plot_coeval_slices",
    "plot_fisher_contours",
    "plot_lightcone_summary",
    "plot_mock_comparison",
    "power_spectrum_1d",
    "power_spectrum_2d",
    "quick_coeval",
    "quick_lightcone",
    "reset_cuda_stats",
    "run_batch",
    "save_batch_h5",
    "save_lightcone",
    "set_cuda_devices",
    "set_cuda_mode",
    "thermal_noise_power_spectrum",
    "to_cuda_view",
    "to_torch",
    "use_mode",
]
