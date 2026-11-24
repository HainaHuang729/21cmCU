"""Compare native 4.3.2 and strict CUDA IC and perturbation stages on small cubes."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)

from evolution_backend import evolution_backend
from perturb import perturb_field_gpu


def compare(native, gpu, scale):
    if native.shape != gpu.shape or native.dtype != gpu.dtype:
        raise ValueError("Array shape or dtype changed")
    a, b = np.asarray(native, dtype="d") * scale, np.asarray(gpu, dtype="d") * scale
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite array")
    delta = abs(a - b)
    return dict(
        tested=int(a.size),
        nonidentical=int(np.count_nonzero(native.view("u4") != gpu.view("u4"))),
        failed=int(np.count_nonzero(delta > 1e-8 + 1e-4 * abs(a))),
        max_abs=float(delta.max() / scale),
    )


def run(output):
    rows = []
    for threads in (1, 2):
        for algorithm in ("LINEAR", "ZELDOVICH"):
            for high in (False, True):
                for smooth in (False, True):
                    fine, low, length = (48, 24, 47.7) if smooth else (32, 16, 24.0)
                    seed = 424242 if high else 1850877075
                    all_velocities = smooth != high
                    inputs = p21.InputParameters(
                        random_seed=seed, node_redshifts=[10.0]
                    ).evolve_input_structs(
                        DIM=fine,
                        HII_DIM=low,
                        BOX_LEN=length,
                        N_THREADS=threads,
                        SOURCE_MODEL="E-INTEGRAL",
                        USE_EXP_FILTER=False,
                        USE_METALLICITY=False,
                        USE_UPPER_STELLAR_TURNOVER=False,
                        USE_TS_FLUCT=False,
                        PERTURB_ALGORITHM=algorithm,
                        PERTURB_ON_HIGH_RES=high,
                        SMOOTH_EVOLVED_DENSITY_FIELD=smooth,
                        KEEP_3D_VELOCITIES=all_velocities,
                    )
                    kw = dict(inputs=inputs, regenerate=True, write=False)
                    ic = p21.compute_initial_conditions(**kw)
                    snapshots = [
                        (z, p21.perturb_field(redshift=z, initial_conditions=ic, **kw))
                        for z in (6.5, 14.3)
                    ]
                    manager.free()
                    with ExitStack() as stack:
                        temp = Path(stack.enter_context(TemporaryDirectory()))
                        engine = stack.enter_context(
                            evolution_backend(temp / "stages.json", thermal=False)
                        )
                        stack.callback(manager.free)
                        generated = p21.compute_initial_conditions(**kw)
                        if set(ic.arrays) != set(generated.arrays):
                            raise ValueError("InitialConditions field names changed")
                        initial = {
                            key: compare(ic.get(key), generated.get(key), 1.0)
                            for key in ic.arrays
                        }
                        row = dict(
                            threads=threads,
                            seed=seed,
                            algorithm=algorithm,
                            high=high,
                            smooth=smooth,
                            keep_3d_velocities=all_velocities,
                            dim=fine,
                            hii_dim=low,
                            box_len=length,
                            initial=initial,
                            nodes=[],
                        )
                        for z, expected in snapshots:
                            actual = perturb_field_gpu(
                                redshift=z,
                                initial_conditions=generated,
                                inputs=inputs,
                                resident=engine,
                                strict_fft=True,
                            )
                            if set(expected.arrays) != set(actual.arrays):
                                raise ValueError("PerturbedField field names changed")
                            fields = {
                                key: compare(
                                    expected.get(key),
                                    actual.get(key),
                                    1e16 if key.startswith("velocity") else 1.0,
                                )
                                for key in expected.arrays
                            }
                            row["nodes"].append(dict(z=z, fields=fields))
                        rows.append(row)
    result = dict(
        version=p21.__version__,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        tolerance=dict(atol=1e-8, rtol=1e-4, velocity_scale=1e16),
        rows=rows,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    if any(value["nonidentical"] for row in rows for value in row["initial"].values()):
        raise AssertionError("Native and CUDA initial conditions differ")
    if any(
        value["failed"]
        for row in rows
        for node in row["nodes"]
        for value in node["fields"].values()
    ):
        raise AssertionError("Native and CUDA perturbations differ beyond tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
