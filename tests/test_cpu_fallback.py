"""Check CPU routing and strict mode without CUDA initialization."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def adapter(monkeypatch):
    modules = {
        "py21cmfast": {"run_lightcone": None, "generate_lightcone": None},
        "py21cmfast.drivers": {},
        "py21cmfast.drivers.coeval": {"run_coeval": None, "generate_coeval": None},
        "py21cmfast.drivers.lightcone": {},
        "py21cmfast.drivers.single_field": {},
        "py21cmfast.drivers._global_initialization": {
            "_GlobalInitManagerSingleton": SimpleNamespace(_lock=RLock()),
        },
        "py21cmfast.lightconers": {
            "AngularLightconer": object,
            "RectilinearLightconer": object,
        },
        "py21cmfast.wrapper": {},
        "py21cmfast.wrapper.outputs": {"OutputStruct": object},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)

    path = Path(__file__).resolve().parents[1] / "v4" / "gpu_lightcone.py"
    spec = importlib.util.spec_from_file_location("_gpu_lightcone_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def odd_grid(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "version", lambda _: "4.3.2")
    return SimpleNamespace(
        simulation_options=SimpleNamespace(NON_CUBIC_FACTOR=1, DIM=31, HII_DIM=16),
    )


@pytest.mark.parametrize(
    ("entry", "generator", "lightcone"),
    [
        ("run_lightcone", False, True),
        ("generate_lightcone", True, True),
        ("run_coeval", False, False),
        ("generate_coeval", True, False),
    ],
)
def test_native_dispatch(
    adapter,
    odd_grid,
    monkeypatch,
    caplog,
    tmp_path,
    entry,
    generator,
    lightcone,
):
    monkeypatch.setattr(adapter, "RectilinearLightconer", SimpleNamespace)
    native = adapter.p21 if lightcone else adapter.coeval_driver
    result = object()
    call = Mock(return_value=iter((result,)) if generator else result)
    monkeypatch.setattr(native, entry, call)
    options = (
        {"lightconer": SimpleNamespace(line_of_sight_axis=2)}
        if lightcone
        else {"out_redshifts": (8.0,)}
    )

    output = getattr(adapter, entry)(
        inputs=odd_grid, diagnostics_dir=tmp_path, write=False, **options
    )

    if generator:
        assert tuple(output) == (result,)
    else:
        assert output is result
    call.assert_called_once_with(inputs=odd_grid, write=False, **options)
    assert "CPU driver" in caplog.text
    assert "even, cubic" in caplog.text
    routing = json.loads((tmp_path / "backend-routing.json").read_text())
    assert routing["backend"] == "cpu"
    assert "even, cubic" in routing["reason"]


@pytest.mark.parametrize(
    ("fallback_to_cpu", "photon_calibration"),
    [(False, None), (True, object())],
)
def test_strict_mode_and_explicit_calibration(
    adapter,
    odd_grid,
    monkeypatch,
    fallback_to_cpu,
    photon_calibration,
):
    call = Mock()
    monkeypatch.setattr(adapter.coeval_driver, "run_coeval", call)
    with pytest.raises(ValueError, match="even, cubic"):
        adapter.run_coeval(
            inputs=odd_grid,
            fallback_to_cpu=fallback_to_cpu,
            photon_calibration=photon_calibration,
        )
    call.assert_not_called()


def test_driver_error_is_propagated_without_retry(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_backend", Mock(return_value=nullcontext()))
    call = Mock(side_effect=ValueError("native driver failed"))
    monkeypatch.setattr(adapter.coeval_driver, "run_coeval", call)
    with pytest.raises(ValueError, match="native driver failed"):
        adapter.run_coeval(inputs=object())
    call.assert_called_once()
