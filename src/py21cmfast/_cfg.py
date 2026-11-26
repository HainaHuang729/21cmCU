"""Open and read the configuration file."""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path
from typing import ClassVar

import deprecation
import yaml as pyyaml

from . import yaml
from ._data import _DATA_PATH
from .c_21cmfast import ffi, lib
from .wrapper.structs import StructInstanceWrapper


class ConfigurationError(Exception):
    """An error with the config file."""


def _parse_env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_cuda_mode(value: object) -> str:
    """Normalize and validate CUDA precision mode (``strict`` or ``fast``)."""
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("strict", "fast"):
            return norm
    raise ConfigurationError(
        f"Invalid CUDA_MODE '{value}'. Expected 'strict' or 'fast'."
    )


_DEFAULT_CUDA_MODE = os.environ.get("P21_CUDA_MODE", "strict").strip().lower()


# TODO: force config to be a singleton
class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults: ClassVar = {
        "direc": "~/21cmFAST-cache",
        "ignore_R_BUBBLE_MAX_error": False,
        "external_table_path": _DATA_PATH,
        "HALO_CATALOG_MEM_FACTOR": 1.2,
        "EXTRA_EMISSIVITY_FIELDS": False,
        "safe_read": True,
        "CACHE_ARRAYS_ON_ACCESS": True,
        "USE_CUDA": _parse_env_bool("P21_USE_CUDA", default=False),
        "CUDA_LAZY_D2H": _parse_env_bool("P21_CUDA_LAZY_D2H", default=False),
        "CUDA_MODE": _DEFAULT_CUDA_MODE,
        "CUDA_FAST_HALO_SAMPLER": (
            _parse_env_bool("P21_CUDA_FAST_HALO_SAMPLER", default=False)
            if "P21_CUDA_FAST_HALO_SAMPLER" in os.environ
            else _DEFAULT_CUDA_MODE == "fast"
        ),
    }
    _defaults["wisdoms_path"] = Path(_defaults["direc"]) / "wisdoms"

    _deprecated_keys: ClassVar = {
        "EXTRA_HALOBOX_FIELDS": "EXTRA_EMISSIVITY_FIELDS",
    }

    # The C config_settings struct is process-global, but string values are stored in
    # char buffers allocated by cffi. Keep the most recent buffer for each key alive
    # here, so the struct never points at memory freed with a collected instance.
    _c_string_buffers: ClassVar[dict] = {}

    def __init__(self, *args, file_name: str | Path | None = None, **kwargs):
        explicit_fast_halo = "CUDA_FAST_HALO_SAMPLER" in kwargs
        if args:
            translated_arg0 = self._translate_deprecated(dict(args[0]))
            explicit_fast_halo = (
                explicit_fast_halo or "CUDA_FAST_HALO_SAMPLER" in translated_arg0
            )
            args = (translated_arg0, *args[1:])
        kwargs = self._translate_deprecated(kwargs)

        super().__init__(*args, **kwargs)
        self.file_name = file_name
        # keep the config settings from the C library here
        self._c_config_settings = StructInstanceWrapper(lib.config_settings, ffi)

        for k, v in self._defaults.items():
            if k not in self:
                self[k] = v

        for k in self.keys():
            if k not in self._defaults:
                raise ConfigurationError(
                    f"You passed the key '{k}' to config, which is not known to 21cmFAST."
                )

        self["CUDA_MODE"] = _normalize_cuda_mode(self["CUDA_MODE"])
        is_fast = self["CUDA_MODE"] == "fast"
        if (
            is_fast
            and not explicit_fast_halo
            and "P21_CUDA_FAST_HALO_SAMPLER" not in os.environ
        ):
            self["CUDA_FAST_HALO_SAMPLER"] = True

        self["direc"] = Path(self["direc"]).expanduser().absolute()
        for k in ("wisdoms_path", "external_table_path"):
            self[k] = Path(self[k])

        # since the subclass __setitem__ is not called in the super().__init__ call, we re-do the setting here
        # NOTE: This seems messy but I don't know a better way to do it
        for k in self._c_config_settings:
            self._pass_to_backend(k, self[k])
        self._sync_cuda_backend()

    def _sync_cuda_mode(self, mode: str) -> None:
        """Push CUDA_MODE and derived fast-IC flag to the C/CUDA runtime."""
        is_fast = int(mode == "fast")
        lib.p21_cuda_set_mode(is_fast)
        lib.p21_cuda_set_fast_ic(is_fast)

    def _sync_cuda_backend(self) -> None:
        """Push all CUDA runtime flags to the C/CUDA backend."""
        lib.p21_cuda_set_enabled(int(bool(self["USE_CUDA"])))
        self._sync_cuda_mode(self["CUDA_MODE"])
        lib.p21_cuda_set_fast_halo_sampler(int(bool(self["CUDA_FAST_HALO_SAMPLER"])))
        lib.p21_cuda_set_lazy_d2h(int(bool(self["CUDA_LAZY_D2H"])))

    def __setitem__(self, key, value):
        """Set an item in the config. Also updating the backend if it exists there."""
        key = self._resolve_deprecated_key(key)
        if key == "CUDA_MODE":
            value = _normalize_cuda_mode(value)
        super().__setitem__(key, value)
        if key in self._c_config_settings:
            self._pass_to_backend(key, value)
        elif key == "USE_CUDA":
            if not bool(value):
                lib.p21_cuda_registry_sync_all_lazy()
            lib.p21_cuda_set_enabled(int(bool(value)))
        elif key == "CUDA_MODE":
            self._sync_cuda_mode(value)
            self["CUDA_FAST_HALO_SAMPLER"] = value == "fast"
        elif key == "CUDA_FAST_HALO_SAMPLER":
            lib.p21_cuda_set_fast_halo_sampler(int(bool(value)))
        elif key == "CUDA_LAZY_D2H":
            lib.p21_cuda_set_lazy_d2h(int(bool(value)))

    def _pass_to_backend(self, key, value):
        """Set the value in the backend."""
        # we should possibly do a typemap for the ffi
        if isinstance(value, Path | str):
            buffer = ffi.new("char[]", str(value).encode())
            Config._c_string_buffers[key] = buffer
            setattr(self._c_config_settings, key, buffer)
        else:
            setattr(self._c_config_settings, key, value)

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        kwargs = self._translate_deprecated(kwargs)
        backup = self.copy()
        if "CUDA_MODE" in kwargs:
            self["CUDA_MODE"] = kwargs["CUDA_MODE"]
        for k, v in kwargs.items():
            if k == "CUDA_MODE":
                continue
            self[k] = Path(v).expanduser().absolute() if k == "direc" else v
        try:
            yield self
        finally:
            if "CUDA_MODE" in kwargs:
                self["CUDA_MODE"] = backup["CUDA_MODE"]
            if "CUDA_MODE" in kwargs or "CUDA_FAST_HALO_SAMPLER" in kwargs:
                self["CUDA_FAST_HALO_SAMPLER"] = backup["CUDA_FAST_HALO_SAMPLER"]
            for k in kwargs:
                if k in ("CUDA_MODE", "CUDA_FAST_HALO_SAMPLER"):
                    continue
                self[k] = backup[k]

    def write(self, fname: str | Path | None = None):
        """Write current configuration to file to make it permanent."""
        fname = fname or self.file_name
        if fname is None:
            raise ValueError(
                "No file name given to write the config to, and the config has no file_name."
            )

        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)

        with fname.open("w") as fl:
            yaml.dump(self._as_dict(), fl)

    def _as_dict(self):
        """Return a plain dict defining the instance."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.items()}

    @classmethod
    def load(cls, file_name: str | Path):
        """Create a Config object from a config file."""
        file_name = Path(file_name).expanduser().absolute()

        if file_name.exists():
            with file_name.open() as fl:
                # Config files only hold plain values, so never construct arbitrary
                # Python objects from tags in them.
                cfg = pyyaml.safe_load(fl) or {}
            return cls(cfg, file_name=file_name)
        else:
            return cls(file_name=file_name)

    @classmethod
    def _resolve_deprecated_key(cls, key):
        """Translate a deprecated config key to its current name, warning once."""
        new_key = cls._deprecated_keys.get(key)
        if new_key is None:
            return key
        warnings.warn(
            deprecation.DeprecatedWarning(
                key,
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=f"'{key}' has been renamed to '{new_key}'. Please use '{new_key}' instead.",
            ),
            stacklevel=3,
        )
        return new_key

    @classmethod
    def _translate_deprecated(cls, d: dict) -> dict:
        """Translate any deprecated keys in a dict of config options."""
        return {cls._resolve_deprecated_key(k): v for k, v in d.items()}


# On import, load the default config
config = Config()
"""A singleton Config object used to manage global configuration options."""
