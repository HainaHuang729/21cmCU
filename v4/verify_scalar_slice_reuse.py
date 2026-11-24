"""Compare the production coefficient and slice adapters with 4.3.2 on identical inputs."""

import ctypes as C
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cffi
import cupy as cp
import numpy as np
from astropy import units as u
from py21cmfast.drivers._global_initialization import c_state
from py21cmfast.lightconers import RectilinearLightconer
from tests import produce_integration_test_data as official

from device_slices import DeviceSlices
from evolution_backend import evolution_backend, NATIVE, SRC
import lightcone_device
from lightcone_device import make_lightcone_slices


def scalar_checks(adapter, inputs):
    ffi = cffi.FFI()
    for name, header in (
        ("ScalingConstants", "scaling_relations.h"),
        ("IntegralProperties", "EmissivityFields.h"),
    ):
        ffi.cdef(
            re.search(
                r"typedef struct " + name + r" \{.*?\} " + name + ";",
                (SRC / header).read_text(),
                re.S,
            )[0]
        )
    ffi.cdef(
        """
    void set_scaling_constants(double, ScalingConstants *, _Bool);
    double sigma_z0(double);
    double gpu_sigma_z0(double);
    int get_uhmf_averages(double, double, double, double, ScalingConstants *, IntegralProperties *);
    """
    )
    cpu, gpu = ffi.dlopen(str(NATIVE)), ffi.dlopen(str(adapter))
    errors = {}

    def compare(key, a, b):
        np.testing.assert_allclose(b, a, atol=0, rtol=1e-5)
        assert np.isfinite(a) and np.isfinite(b) and a > 0
        errors.setdefault(key, []).append(abs(b / a - 1))

    with c_state(inputs, broadcast_inputs=True, ps=True, sigma=True):
        for m in [1.1e7, 8.74e9, 3.25e12, 1.07e15]:
            compare("sigma_z0", cpu.sigma_z0(m), gpu.gpu_sigma_z0(m))
        for z in (7.5, 12.0, 20.0):
            sc = ffi.new("ScalingConstants *")
            cpu.set_scaling_constants(z, sc, True)
            args = (1e7, 1e16, sc.mturn_acg_homogeneous, 0.0, sc)
            ref, got = (ffi.new("IntegralProperties *") for _ in range(2))
            assert (
                cpu.get_uhmf_averages(*args, ref)
                == gpu.get_uhmf_averages(*args, got)
                == 0
            )
            for name in ("n_ion", "sfrd_acg", "xray_emissivity"):
                compare(name, getattr(ref, name), getattr(got, name))
    return {k: {"values": len(v), "max_relative": max(v)} for k, v in errors.items()}


def slice_checks(inputs):
    cell = inputs.simulation_options.cell_size
    cosmo = inputs.cosmo_params.cosmo
    za, zb = 7.49, 7.52
    near, far = cosmo.comoving_distance([za, zb])
    distances = u.Quantity(
        [
            near * (1 - 2e-6),
            near * (1 - 0.5e-6),
            near,
            near * 0.6 + far * 0.4,
            far - cell / 4,
            far,
            far + cell,
        ]
    )
    cone = RectilinearLightconer(
        lc_distances=distances,
        quantities=("a", "b", "los_velocity"),
        interp_kinds={"b": "mean_max"},
        index_offset=-5,
    )
    rng = np.random.default_rng(424242)

    def box(z):
        return SimpleNamespace(
            simulation_options=inputs.simulation_options,
            cosmo_params=inputs.cosmo_params,
            redshift=z,
            **{
                k: rng.normal(size=(3, 5, 4)).astype(t)
                for k, t in (("a", "f"), ("b", "f"), ("velocity_z", "d"))
            },
        )

    a, b = box(za), box(zb)
    count = 0
    for first, second in ((a, b), (b, a), (box(12.0), box(13.0))):
        for (k, i, ref), (key, idx, got) in zip(
            cone.make_lightcone_slices(first, second),
            make_lightcone_slices(cone, first, second),
            strict=True,
        ):
            assert k == key
            np.testing.assert_array_equal(i, idx)
            if k is None:
                assert ref is got is None
                continue
            assert np.isfinite(ref).all()
            assert ref.dtype == got.dtype and ref.unit == got.unit
            np.testing.assert_array_equal(ref.value, got.value)
            count += ref.size
    velocity = RectilinearLightconer(
        lc_distances=distances, quantities=("velocity_z", "los_velocity")
    )
    with patch.object(
        lightcone_device,
        "interpolate_slices",
        wraps=lightcone_device.interpolate_slices,
    ) as gather:
        pair = list(make_lightcone_slices(velocity, a, b))
    assert gather.call_count == 1
    for (name, i, ref), (key, j, value) in zip(
        velocity.make_lightcone_slices(a, b), pair, strict=True
    ):
        assert (name, i) == (key, j)
        np.testing.assert_array_equal(ref.value, value.value)
    for i in range(0, len(pair), 2):
        assert not np.shares_memory(pair[i][2].value, pair[i + 1][2].value)
    # v3 passes float32 distances and may provide flattened resident grids.
    v3 = DeviceSlices(None)
    fields = [rng.normal(size=(4, 4, 4)).astype("f") for _ in range(2)]
    k = np.array([2, 0, 1], dtype="i")
    wa, wb = (
        np.array([0.25, 0.75, 1.0], dtype="f"),
        np.array([0.75, 0.25, 0.0], dtype="f"),
    )
    out = np.zeros((4, 4, 3), dtype="f")
    v3.fill(["probe"], [(*fields, out, True)], np.arange(3), k, wa, wb, 1)
    left, right = [f[..., k] for f in fields]
    np.testing.assert_array_equal(
        out, np.where(left * right < 0, np.maximum(left, right), wa * left + wb * right)
    )
    return {
        "v4_values_exact": count,
        "v3_values_exact": out.size,
        "dual_velocity_values_exact": sum(v.size for _, _, v in pair),
        "dual_velocity_cuda_gathers_per_node_pair": gather.call_count,
    }


def temperature_check(engine):
    lib = C.CDLL(str(NATIVE))
    native = lib.ComputeFullyIonizedTemperature
    native.argtypes, native.restype = [C.c_float] * 4, C.c_float
    z, tre = 7.2, 18000.0
    rng = np.random.default_rng(424242)
    n = 2048
    zr = rng.uniform(z, z + 3, n).astype("f")
    zr[::5], zr[1::7] = -1, z
    neutral = np.where(np.arange(n) % 3, 0.0, 0.3).astype("f")
    density = rng.uniform(-1, 3, n).astype("f")
    density[::17] = -1
    tk = rng.uniform(20, 24000, n).astype("f")
    out = tk.copy()
    expected = out.copy()
    for i in np.flatnonzero((zr > 0) & (neutral < 1e-30)):
        expected[i] = max(native(zr[i], z, density[i], tre), tk[i])
        out[i] = np.nan
    ptrs = (C.c_void_p * 5)(*[x.ctypes.data for x in (zr, neutral, density, tk, out)])
    assert engine.dispatch(16, n, (C.c_double * 2)(z, tre), ptrs) == 0, engine.error
    np.testing.assert_allclose(out, expected, atol=1e-8, rtol=1e-4)
    return {
        "values": n,
        "active": int(np.sum((zr > 0) & (neutral < 1e-30))),
        "max_abs_K": float(abs(out - expected).max()),
    }


def main(output):
    inputs = official.get_all_options_struct(
        6,
        lc=True,
        DIM=192,
        HII_DIM=64,
        BOX_LEN=200,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
    )["inputs"]
    report = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "slices": slice_checks(inputs),
    }
    with (
        TemporaryDirectory() as tmp,
        evolution_backend(Path(tmp) / "stages.json") as engine,
    ):
        (adapter,) = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.endswith("/evolution.so")
        }
        report["scalars"] = scalar_checks(adapter, inputs)
        report["ion_temperature"] = temperature_check(engine)
        assert engine.counts[28] >= 4 and engine.counts[29] == 3
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
