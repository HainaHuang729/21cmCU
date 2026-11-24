"""Compare ordered NUMBER-LIMITED CHMF catalogues and the archived mass-limited control."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import py21cmfast as p21


FIELDS = ("halo_masses", "halo_coords", "star_rng", "sfr_rng", "xray_rng")
SEED, Z_GRID, Z_PROGENITORS = 424242, 10.0, 10.22


def record(catalog):
    count = int(catalog.n_halos)
    hashes = {}
    combined = hashlib.sha256()
    for name in FIELDS:
        array = np.ascontiguousarray(catalog.get(name)[:count])
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name}")
        hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
        combined.update(name.encode())
        combined.update(array.tobytes())
    return dict(count=count, fields_sha256=hashes, ordered_sha256=combined.hexdigest())


def main(mode, directory, threads, mass_report, mass_hash_report):
    directory.mkdir(parents=True, exist_ok=True)
    native_record = directory / "cpu.json"
    inputs = p21.InputParameters(
        random_seed=SEED, node_redshifts=[Z_PROGENITORS, Z_GRID]
    ).evolve_input_structs(
        SOURCE_MODEL="CHMF-SAMPLER",
        SAMPLE_METHOD="NUMBER-LIMITED",
        DIM=32,
        HII_DIM=16,
        BOX_LEN=24.0,
        N_THREADS=threads,
    )
    spec = dict(
        method=inputs.matter_options.SAMPLE_METHOD,
        seed=SEED,
        nodes=[Z_GRID, Z_PROGENITORS],
        fine=32,
        coarse=16,
        box_mpc=24.0,
        threads=threads,
        version=p21.__version__,
    )
    cache = p21.OutputCache(directory / "native-cache")
    reference = None
    if mode == "gpu" or native_record.exists():
        reference = json.loads(native_record.read_text())
        if reference["case"] != spec:
            raise ValueError("Saved native case differs")
        if mode == "cpu":
            print(f"Reusing {native_record}")
            return
    if mode == "cpu":
        context = nullcontext()
        ic = p21.compute_initial_conditions(
            inputs=inputs, cache=cache, write=True, regenerate=False
        )
    else:
        from halo_catalog import halo_catalog_backend

        counters = {}
        context = halo_catalog_backend(counters)
        ic = cache.load(p21.InitialConditions.new(inputs=inputs))
    kwargs = dict(
        inputs=inputs,
        initial_conditions=ic,
        cache=cache,
        write=mode == "cpu",
        regenerate=mode == "gpu",
    )
    with context:
        grid = p21.determine_halo_catalog(redshift=Z_GRID, **kwargs)
        progenitors = p21.determine_halo_catalog(
            redshift=Z_PROGENITORS, descendant_halos=grid, **kwargs
        )
    actual = dict(grid=record(grid), progenitors=record(progenitors))
    if not all(r["count"] for r in actual.values()):
        raise ValueError("Expected nonempty grid and progenitor lists")
    output = dict(case=spec, job=os.getenv("SLURM_JOB_ID"), catalogs=actual)
    if mode == "gpu":
        m = json.loads(mass_report.read_text())["default_model"]
        if (m["seed"], m["dimensions"], m["box_mpc"], m["nodes"]) != (
            SEED,
            [32, 16],
            24,
            [Z_GRID, Z_PROGENITORS],
        ):
            raise ValueError("Different MASS-LIMITED control parameters")
        control = next(c for c in m["paired"] if c["threads"] == threads)["samples"]
        mass_counts = {s["label"]: s["cpu_count"] for s in control}
        output.update(
            native_cpu_job=reference["job"],
            counters=counters,
            mass_limited_counts=mass_counts,
            cpu_gpu_equal=reference["catalogs"] == actual,
        )
        if threads == 2:
            # This earlier two-thread MASS-LIMITED reference has raw ordered native hashes.
            cases = json.loads(mass_hash_report.read_text())["cases"]
            same = next(
                r
                for r in cases
                if r["seed"] == SEED
                and r["redshift"] == Z_GRID
                and r["dimensions"] == [32, 16]
                and r["box_len_Mpc"] == 24
            )
            output["initial_grid_matches_native_mass_limited_sha256"] = (
                actual["grid"]["ordered_sha256"] == same["catalogs"][0]["native_sha256"]
            )
        output["passed"] = (
            output["cpu_gpu_equal"]
            and counters.get("halo_cuda_mt_words", 0) > 0
            and (counters.get("halo_cuda_gfsr_words", 0) > 0) == (threads == 2)
            and mass_counts["grid"] == actual["grid"]["count"]
            and mass_counts["progenitors"] != actual["progenitors"]["count"]
            and output.get("initial_grid_matches_native_mass_limited_sha256", True)
            and counters["halo_grid_property_gaussians"] == 3 * actual["grid"]["count"]
            and counters["halo_progenitor_property_gaussians"]
            == 3 * actual["progenitors"]["count"]
        )
    (directory / f"{mode}.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        mode,
        threads,
        actual["grid"]["count"],
        actual["progenitors"]["count"],
        output.get("passed"),
    )
    if not output.get("passed", True):
        raise SystemExit("CHMF sampling comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("threads", type=int, choices=(1, 2))
    reports = Path(__file__).resolve().parents[1] / "reports"
    parser.add_argument(
        "--mass-report", type=Path, default=reports / "v432-chmf-property-rng.json"
    )
    parser.add_argument(
        "--mass-hash-report",
        type=Path,
        default=reports / "v432-halo-mass-function.json",
    )
    args = parser.parse_args()
    main(
        args.mode, args.directory, args.threads, args.mass_report, args.mass_hash_report
    )
