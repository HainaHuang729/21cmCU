"""The 21cmCU GPU-accelerated cosmological simulation package."""

from __future__ import annotations

import contextlib
from typing import Any, Literal

import py21cmfast as _p21
from py21cmfast import *  # noqa: F403
from py21cmfast import config
from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib
from py21cmfast.lightconers import clear_lightconer_class_cache as _clear_lc_class_cache
from py21cmfast.wrapper.inputs import (
    clear_cosmo_tables_cache as _clear_cosmo_tables_cache,
)

from .analysis import (
    lightcone_chunk_power_spectra,
    power_spectrum_1d,
    power_spectrum_2d,
)
from .batch import BatchResult, BatchTask, run_batch, save_batch_h5
from .io import save_lightcone

__version__ = "0.3.0"

CudaMode = Literal["strict", "fast"]


def cuda_compiled() -> bool:
    """Return True if the C extension was compiled with native CUDA support."""
    return bool(_lib.p21_cuda_is_compiled())


def cuda_available() -> bool:
    """Return True if a CUDA-capable GPU device is available at runtime."""
    return bool(_lib.p21_cuda_is_available())


def cuda_stats() -> dict[str, int]:
    """Return a snapshot of the native CUDA backend runtime counters."""
    stats = _ffi.new("P21CudaStats *")
    _lib.p21_cuda_get_stats(stats)
    return {
        name: int(getattr(stats, name))
        for name, _ in _ffi.typeof("P21CudaStats").fields
    }


def reset_cuda_stats() -> None:
    """Reset the native CUDA backend runtime counters."""
    _lib.p21_cuda_reset_stats()


def cleanup_cuda() -> None:
    """Release cached CUDA device memory, cuFFT plans, and registry buffers."""
    _lib.p21_cuda_cleanup()


def clear_class_cache() -> None:
    """Clear in-memory CLASS transfer-table and lightconer caches."""
    _clear_cosmo_tables_cache()
    _clear_lc_class_cache()


def get_cuda_mode() -> CudaMode:
    """Return the active CUDA execution precision mode (``strict`` or ``fast``)."""
    return config["CUDA_MODE"]


def set_cuda_mode(mode: CudaMode) -> None:
    """Set the global CUDA execution precision mode (``strict`` or ``fast``)."""
    config["CUDA_MODE"] = mode


@contextlib.contextmanager
def use_mode(
    mode: CudaMode,
    *,
    use_cuda: bool = True,
    lazy_d2h: bool | None = None,
    **extra: Any,
):
    """Temporarily run under the requested CUDA precision mode and config overrides."""
    if lazy_d2h is not None:
        extra["CUDA_LAZY_D2H"] = bool(lazy_d2h)
    with config.use(USE_CUDA=use_cuda, CUDA_MODE=mode, **extra) as cfg:
        yield cfg


__all__ = [
    *_p21.__all__,
    "BatchResult",
    "BatchTask",
    "CudaMode",
    "cleanup_cuda",
    "clear_class_cache",
    "cuda_available",
    "cuda_compiled",
    "cuda_stats",
    "get_cuda_mode",
    "lightcone_chunk_power_spectra",
    "power_spectrum_1d",
    "power_spectrum_2d",
    "reset_cuda_stats",
    "run_batch",
    "save_batch_h5",
    "save_lightcone",
    "set_cuda_mode",
    "use_mode",
]
