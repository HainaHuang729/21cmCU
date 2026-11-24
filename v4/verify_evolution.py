"""Compare all allocated fields at each v4 driver boundary on paired lightcones."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from py21cmfast.drivers import single_field

from lightcone_ps import capture

STAGES = (
    "compute_emissivity_fields",
    "compute_radiation_fields",
    "compute_spin_temperature",
    "compute_ionization_field",
)


def run(directory, fine, mode, config):
    directory.mkdir(parents=True, exist_ok=True)
    gpu = mode == "gpu"
    stats = {}
    with ExitStack() as stack:
        reference = stack.enter_context(
            h5py.File(directory / "reference-fields.h5", "r" if gpu else "w")
        )
        for name in STAGES:
            native = getattr(single_field, name)

            def observed(*args, _native=native, _stage=name, **kwargs):
                obj = _native(*args, **kwargs)
                key = f"{_stage}/{obj.redshift:.12g}"
                for field, arr in obj.arrays.items():
                    if not arr.state.computed_in_mem:
                        continue
                    path = f"{key}/{field}"
                    data = np.asarray(obj.get(field))
                    if not gpu:
                        reference.create_dataset(path, data=data, compression="lzf")
                    else:
                        expected = reference[path][:]
                        if data.shape != expected.shape:
                            raise ValueError(path)
                        delta = abs(data.astype("d") - expected.astype("d"))
                        bad = ~(delta <= 1e-8 + 1e-4 * abs(expected.astype("d")))
                        scale = float(np.max(abs(expected)))
                        stats[path] = dict(
                            max_abs=float(delta.max()),
                            failed=int(bad.sum()),
                            cells=data.size,
                            different=int(np.count_nonzero(delta)),
                            reference_peak=scale,
                            max_abs_over_peak=float(delta.max() / scale)
                            if scale
                            else None,
                        )
                        if bad.any():
                            idx = np.unravel_index(
                                np.where(bad, delta, -1).argmax(), data.shape
                            )
                            stats[path].update(
                                index=[int(x) for x in idx],
                                cpu=float(expected[idx]),
                                gpu=float(data[idx]),
                            )
                return obj

            stack.enter_context(patch.object(single_field, name, observed))
        capture(directory, fine, mode, config, audit_evolution=gpu)
        if gpu:
            names = []
            reference.visititems(
                lambda k, v: names.append(k) if isinstance(v, h5py.Dataset) else None
            )
            assert stats and set(names) == set(stats)
            (directory / "intermediates.json").write_text(json.dumps(stats))
            failures = sum(row["failed"] for row in stats.values())
            for filename, count in (
                ("thermal-terms.json", 12),
                ("source-tables.json", 2),
            ):
                results = json.loads((directory / filename).read_text())
                if len(results) != count:
                    raise ValueError(f"Incomplete {filename}")
                failures += sum(row["failed"] for row in results.values())
            calls = json.loads((directory / "gpu-stages.json").read_text())
            # This audit runs the thermal E-INTEGRAL fixture. The audit leaves
            # the native mean-fix call observable, and resident spectra bypass
            # the standalone native FFT and filtering shims.
            required = """
                emission heat ion_center recombination shell_filter shell_accumulate source_table
                gpu_seed_and_modes ion_filter_prepare ion_filter_resident_radius ion_filter_extrema
                ion_photon_lookup radiation_units radiation_setup_grid ionized_temperature
                xray_frequency_integrals global_source_integrals sigma_moments mhr_table lyman_shells
                ion_global_source_integrals fully_neutral_box global_neutral_fraction eh_constants
                initial_heat thermal_constants scalar_sigma emissivity_global_source_integrals
            """.split()
            missing = [name for name in required if calls.get(name, 0) < 1]
            if missing:
                raise ValueError(f"Missing thermal audit stages: {', '.join(missing)}")
            if failures:
                raise SystemExit(f"{failures} intermediate values exceed tolerance")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("fine", type=int)
    p.add_argument("mode", choices=["cpu", "gpu"])
    p.add_argument("config", type=Path)
    q = p.parse_args()
    run(q.directory, q.fine, q.mode, q.config)
