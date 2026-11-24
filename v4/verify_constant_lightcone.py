"""Save a small native Eulerian trajectory once, then compare the CUDA route."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import py21cmfast as p21

from compare_lightcone_ps import compare_saved_slices, metrics


NODES = (9.0, 8.7, 8.4)
FIELDS = (
    "density",
    "velocity_z",
    "neutral_fraction",
    "brightness_temp",
    "ionisation_rate_G12",
    "z_reion",
    "kinetic_temperature",
)
THERMAL_FIELDS = (
    "cumulative_recombinations",
    "mean_free_path",
    "spin_temperature",
    "kinetic_temp_neutral",
    "xray_ionised_fraction",
    "tau_21",
)


def specification(thermal, source_model="CONST-ION-EFF"):
    return dict(
        SOURCE_MODEL=source_model,
        HMF="ST",
        FILTER="spherical-tophat",
        HII_FILTER="spherical-tophat",
        USE_INTERPOLATION_TABLES="hmf-interpolation",
        USE_TS_FLUCT=thermal,
        RECOMB_MODEL="inhomogeneous" if thermal else "none",
        USE_X_RAY_HEATING=True,
        USE_LYA_HEATING=True,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        USE_MCGS=False,
        USE_METALLICITY=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        PHOTON_CONS_TYPE="no-photoncons",
        IONISE_ENTIRE_SPHERE=False,
        Z_HEAT_MAX=NODES[0],
        R_BUBBLE_MAX=12.0,
        HII_EFF_FACTOR=30.0,
        DIM=32,
        HII_DIM=16,
        BOX_LEN=24.0,
        N_THREADS=1,
    )


def cache_files(directory):
    files = {}
    for path in sorted(directory.rglob("*.h5")):
        with h5py.File(path) as handle:
            group = handle[path.stem]
            key = (path.stem, float(group.attrs.get("redshift", -1.0)))
            if key in files:
                raise ValueError(f"Duplicate cache entry for {key} in {directory}")
            files[key] = path
    return files


def describe_cache(files, thermal):
    expected = {"PerturbedField", "EmissivityFields", "IonizedBox", "BrightnessTemp"}
    if thermal:
        expected |= {"TsBox", "RadiationFields"}
    for kind in expected:
        np.testing.assert_allclose(
            sorted(z for name, z in files if name == kind),
            sorted(NODES),
            rtol=0,
            atol=1e-6,
        )
    if ("InitialConditions", -1.0) not in files:
        raise ValueError("No initial conditions in trajectory cache")
    nodes = []
    for (kind, z), path in sorted(files.items(), reverse=True):
        if kind != "IonizedBox":
            continue
        with h5py.File(path) as handle:
            fields = handle[kind]["OutputFields"]
            neutral = fields["neutral_fraction"][:]
            entry = dict(
                z=z,
                neutral_mean=float(neutral.astype("d").mean()),
                partial=int(np.count_nonzero((neutral > 0) & (neutral < 1))),
                ionized=int((neutral == 0).sum()),
            )
            if thermal:
                entry["max_recombinations"] = float(
                    fields["cumulative_recombinations"][:].max()
                )
            nodes.append(entry)
    if not any(n["partial"] for n in nodes) or not any(n["ionized"] for n in nodes):
        raise ValueError(
            f"Native integration case did not exercise both barrier outcomes: {nodes}"
        )
    if np.ptp([n["neutral_mean"] for n in nodes]) <= 1e-6:
        raise ValueError("No measurable evolution of mean neutral fraction")
    if thermal and max(n["max_recombinations"] for n in nodes) <= 0:
        raise ValueError("No evolving recombinations")
    return nodes


def compare_cache(native, accelerated):
    if native.keys() != accelerated.keys():
        raise ValueError("Native and accelerated cached stages or redshifts differ")
    rows = []
    for (kind, z), cpu in native.items():
        with h5py.File(cpu) as a, h5py.File(accelerated[kind, z]) as b:
            x, y = (h[kind]["OutputFields"] for h in (a, b))
            if set(x) != set(y) or set(x.attrs) != set(y.attrs):
                raise ValueError(f"Different {kind} fields at {z}")
            data = list(x)
            scalars = [
                k
                for k in x.attrs
                if np.issubdtype(np.asarray(x.attrs[k]).dtype, np.number)
            ]
            for key in data + scalars:
                left, right = (
                    np.asarray(f[key][:] if key in data else f.attrs[key], dtype="d")
                    for f in (x, y)
                )
                np.testing.assert_equal(left.shape, right.shape)
                delta, bad = metrics(left, right)
                rows.append(
                    dict(
                        kind=kind,
                        z=z,
                        field=key,
                        tested=int(delta.size),
                        failed=int(bad.sum()),
                        max_abs=float(delta.max()),
                    )
                )
    return rows


def numerical_settings(inputs, lightconer, output_fields):
    return dict(
        interpolation={
            k: lightconer.interp_kinds.get(k, "mean") for k in output_fields
        },
        reduction={
            "cpu_threads": inputs.simulation_options.N_THREADS,
            "mean_dtype": "upstream mean for each field, saved globally as float64",
            "transverse_power_dtype": "float64",
            "transverse_power_bins": inputs.simulation_options.HII_DIM // 2,
        },
        source_parameters={
            k: getattr(inputs.astro_options, k)
            for k in (
                "M_MIN_in_Mass",
                "INTEGRATION_METHOD_ACGS",
                "USE_CMB_HEATING",
                "USE_ADIABATIC_FLUCTUATIONS",
                "LYA_MULTIPLE_SCATTERING",
            )
        }
        | {
            k: getattr(inputs.astro_params, k)
            for k in (
                "HII_EFF_FACTOR",
                "R_BUBBLE_MIN",
                "R_BUBBLE_MAX",
                "DELTA_R_HII_FACTOR",
                "ION_Tvir_MIN",
                "X_RAY_Tvir_MIN",
                "M_TURN_STELLAR_FEEDBACK",
            )
        },
    )


def main(
    mode,
    directory,
    thermal,
    require_stage,
    restore_job=None,
    source_model="CONST-ION-EFF",
    homogeneous=False,
    integration_method=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    options = specification(thermal, source_model)
    if source_model == "E-INTEGRAL":
        options.update(
            INTEGRATION_METHOD_ACGS="GAUSS-LEGENDRE", F_ESC10=-0.7, F_ESC10_ACG=-0.7
        )
    if homogeneous:
        options.update(RECOMB_MODEL="homogeneous", CELL_RECOMB=True)
    if integration_method:
        options["INTEGRATION_METHOD_ACGS"] = integration_method
    inputs = p21.InputParameters(
        random_seed=812445, node_redshifts=NODES
    ).evolve_input_structs(**options)
    fields = FIELDS + (THERMAL_FIELDS if thermal else ())
    if homogeneous:
        # Native homogeneous history is one scalar per epoch, stored in the cache.
        fields = tuple(q for q in fields if q != "cumulative_recombinations")
    lightconer = p21.lightconers.RectilinearLightconer.between_redshifts(
        min_redshift=8.44,
        max_redshift=8.94,
        quantities=fields,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    output_fields = set(fields) | {"los_velocity"}
    cpu_path = directory / "cpu.h5"
    cpu_json = directory / "cpu.json"
    reference_files = None
    if mode in ("gpu", "compare") or cpu_path.exists() or cpu_json.exists():
        reference = p21.LightCone.from_file(cpu_path)
        meta = json.loads(cpu_json.read_text())
        if (
            reference.inputs != inputs
            or set(reference.lightcones) != output_fields
            or meta["options"] != options
            or meta["version"] != p21.__version__
        ):
            raise ValueError(
                "Existing CPU reference does not match requested specification"
            )
        np.testing.assert_array_equal(
            reference.lightcone_distances.to_value("Mpc"),
            lightconer.lc_distances.to_value("Mpc"),
        )
        reference_files = cache_files(directory / "cpu-cache")
        if describe_cache(reference_files, thermal) != meta["ionization"]:
            raise ValueError("Incomplete or inconsistent saved trajectory")
        if mode == "cpu":
            print(f"Reusing {cpu_path}")
            return
    if mode == "compare":
        record = json.loads((directory / "gpu.json").read_text())
        accelerated = p21.LightCone.from_file(directory / "gpu.h5")
        if accelerated.inputs != inputs or set(accelerated.lightcones) != output_fields:
            raise ValueError(
                "Saved CUDA cone does not match the requested inputs or fields"
            )
        evaluate_gpu(directory, record, reference_files, meta["job"], require_stage)
        save_result(directory, "gpu", record)
        return
    run = p21.run_lightcone
    kwargs = {}
    if mode == "gpu":
        from gpu_lightcone import run_lightcone

        run = run_lightcone
        kwargs.update(diagnostics_dir=directory / "routing", fallback_to_cpu=False)
    cache_dir = directory / f"{mode}-cache"
    if restore_job:
        if mode != "cpu":
            raise ValueError("Cached reconstruction requires CPU mode")
        describe_cache(cache_files(cache_dir), thermal)
    start = perf_counter()
    cone = run(
        inputs=inputs,
        lightconer=lightconer,
        write=True,
        regenerate=not restore_job,
        apply_rsds=False,
        include_dvdr_in_tau21=True,
        cache=p21.OutputCache(cache_dir),
        **kwargs,
    )
    seconds = perf_counter() - start
    if cone.inputs != inputs or set(cone.lightcones) != output_fields:
        raise ValueError("Wrong run inputs or output fields")
    if not all(
        np.isfinite(a).all()
        for a in (*cone.lightcones.values(), *cone.global_quantities.values())
    ):
        raise ValueError("Nonfinite native or accelerated fields")
    out = directory / f"{mode}.h5"
    cone.save(out, clobber=True)
    files = cache_files(cache_dir)
    record = dict(
        mode=mode,
        options=options,
        version=p21.__version__,
        seed=inputs.random_seed,
        node_redshifts=list(NODES),
        fields=list(fields),
        slices=cone.n_slices,
        requested_bounds=[8.44, 8.94],
        backend_seconds_including_cache_writes=seconds,
        job=restore_job or os.environ.get("SLURM_JOB_ID"),
        reconstructed_from_complete_cache=bool(restore_job),
        ionization=describe_cache(files, thermal),
        comparison={"atol": 1.0e-8, "rtol": 1.0e-4},
        include_dvdr_in_tau21=True,
        apply_rsds=False,
        global_quantities=sorted(cone.global_quantities),
        **numerical_settings(inputs, lightconer, output_fields),
        cache={
            str(path.relative_to(directory)): [kind, z]
            for (kind, z), path in files.items()
        },
    )
    if mode == "gpu":
        evaluate_gpu(directory, record, reference_files, meta["job"], require_stage)
    save_result(directory, mode, record)


def evaluate_gpu(directory, record, reference_files, reference_job, require_stage):
    products = compare_saved_slices(directory / "cpu.h5", directory / "gpu.h5")
    rows = compare_cache(reference_files, cache_files(directory / "gpu-cache"))
    routing_files = sorted((directory / "routing").glob("*.json"))
    record["routing"] = {f.name: json.loads(f.read_text()) for f in routing_files}
    counters = dict(record["routing"].get("gpu-stages.json", {}))
    counters.update(
        record["routing"].get("backend-routing.json", {}).get("counters", {})
    )
    record["missing_stages"] = [
        name for name in require_stage if not counters.get(name, 0)
    ]
    record.update(products=products, cached_fields=rows, reference_job=reference_job)
    record["passed"] = (
        bool(counters)
        and not record["missing_stages"]
        and not any(row["failed"] for row in rows)
        and not products["brightness_transverse_power"]["failed"]
        and not any(
            row["failed"]
            for group in ("lightcones", "global_quantities")
            for row in products[group].values()
        )
    )


def save_result(directory, mode, record):
    (directory / f"{mode}.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    if not record.get("passed", True):
        raise SystemExit(f"Failed comparison; see {directory / 'gpu.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu", "compare"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--thermal", action="store_true")
    parser.add_argument(
        "--source-model",
        choices=["CONST-ION-EFF", "E-INTEGRAL"],
        default="CONST-ION-EFF",
    )
    parser.add_argument(
        "--require-stage",
        action="append",
        default=[],
        help="Require a positive named CUDA routing counter (repeatable)",
    )
    parser.add_argument(
        "--restore-cpu-job",
        metavar="JOB_ID",
        help="Assemble lightcone from a complete native stage cache generated by this job",
    )
    parser.add_argument(
        "--homogeneous",
        action="store_true",
        help="Use native scalar recombination history",
    )
    parser.add_argument("--integration-method", choices=["GAUSS-LEGENDRE", "GSL-QAG"])
    args = parser.parse_args()
    main(
        args.mode,
        args.directory,
        args.thermal,
        args.require_stage,
        args.restore_cpu_job,
        args.source_model,
        args.homogeneous,
        args.integration_method,
    )
