"""Compare small native and accelerated 4.3.2 Lagrangian-source lightcones.

Run with mode cpu once, then mode gpu with the same directory and case options.
Mode compare checks the saved HDF5 outputs and catalog digests without evolving either box.
"""

import argparse
import hashlib
import json
import os
import platform
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import h5py
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field

from compare_lightcone_ps import compare_saved_slices


FIELDS = ("brightness_temp", "neutral_fraction", "density", "velocity_z", "n_ion")
THERMAL_FIELDS = (
    "spin_temperature",
    "kinetic_temp_neutral",
    "xray_ionised_fraction",
    "cumulative_recombinations",
    "xray_emissivity",
    "sfrd_acg",
)
CATALOG_FIELDS = ("halo_coords", "halo_masses", "star_rng", "sfr_rng", "xray_rng")
THERMAL_STAGES = (
    "thermal_initial_conditions",
    "radiation_shell_filter",
    "radiation_shell_accumulation",
    "radiation_xray_optical_depth_and_frequency",
    "radiation_lyman_spectral_factors",
    "thermal_update",
    "ionization_mhr_table",
    "ionization_cumulative_recombinations",
)
RTOL, ATOL = 1e-4, 1e-8


def read_native_field(directory, name, redshift=None):
    from py21cmfast.io.h5 import read_output_struct

    matches = [
        f
        for f in (directory / "cpu-cache").rglob(f"{name}.h5")
        if redshift is None or f"{redshift:.4f}" in f.parts
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one cached {name} at {redshift}, found {matches}")
    return read_output_struct(matches[0], struct=name)


def saved_class_signature(path, streaming):
    """Fingerprint only persisted native cosmology data, without invoking CLASS."""
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("__version__") != "4.3.2":
            raise ValueError(f"{path}: expected a 4.3.2 lightcone")
        group = h5.get("InputParameters/cosmo_tables")
        if group is None:
            raise ValueError(f"{path}: no archived CLASS tables")
        scalars = {}
        for key in ("USE_SIGMA_8", "ps_norm", "V_CB_AVG"):
            if key not in group.attrs:
                raise ValueError(f"{path}: missing CLASS scalar {key}")
            val = group.attrs[key]
            if np.ndim(val) != 0 or not np.isfinite(val):
                raise ValueError(f"{path}: invalid CLASS scalar {key}")
            scalars[key] = val.item()
        transfers = {}
        for name, table_name in (
            ("density", "transfer_density"),
            ("relative_velocity", "transfer_vcb"),
        ):
            if table_name not in group:
                if name == "density" or streaming == "FLUCTS":
                    raise ValueError(f"{path}: missing CLASS transfer {table_name}")
                continue
            table = group[table_name].attrs
            points = {}
            for key in ("x_values", "y_values"):
                data = np.ascontiguousarray(table[key])
                if (
                    data.ndim != 1
                    or data.size != table["size"]
                    or not np.isfinite(data).all()
                ):
                    raise ValueError(f"{path}: invalid {table_name}/{key}")
                points[key] = hashlib.sha256(data.view(np.uint8)).hexdigest()
            transfers[name] = points
    return {"transfers": transfers, "scalars": scalars}


def reuse_class_tables(inputs, path):
    """Match every parameter read by the native 4.3.2 cosmo_tables getter."""
    from py21cmfast.io.h5 import read_inputs

    signature = saved_class_signature(path, inputs.matter_options.V_CB_MODEL)
    saved = read_inputs(path)

    def dependencies(params):
        return (
            params.cosmo_params,
            params.simulation_options.K_MAX_FOR_CLASS,
            params.matter_options.POWER_SPECTRUM,
            params.matter_options.V_CB_MODEL,
            params.astro_options.USE_MCGS,
        )

    if (
        p21.__version__ != "4.3.2"
        or inputs.matter_options.POWER_SPECTRUM != "CLASS"
        or dependencies(saved) != dependencies(inputs)
    ):
        raise ValueError(
            "Saved CLASS tables use different cosmology, transfer or streaming parameters"
        )
    # Use the native HDF5 loader's cached_property slot; never trigger a new solver here.
    tables = object.__getattribute__(saved, "cosmo_tables")
    object.__setattr__(inputs, "cosmo_tables", tables)
    return signature


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@contextmanager
def native_partition_reference():
    from ctypes import cast, c_void_p
    from tempfile import TemporaryDirectory

    from py21cmfast.c_21cmfast import ffi, lib

    from native_shim import load_shim, repair_partition_rejection

    src = Path(p21.__file__).parent / "src"
    native = Path(p21.c_21cmfast.__file__)
    with TemporaryDirectory(prefix="v432-partition-reference-") as temp:
        library = load_shim(
            Path(temp),
            "partition-reference.so",
            {
                "Stochasticity.c": repair_partition_rejection(
                    (src / "Stochasticity.c").read_text()
                ),
                "HaloCatalog.c": (src / "HaloCatalog.c").read_text(),
            },
            src,
            native,
        )
        compute = ffi.cast(
            ffi.typeof(lib.ComputeHaloCatalog),
            cast(library.ComputeHaloCatalog, c_void_p).value,
        )
        with patch.object(
            p21.HaloCatalog, "_c_compute_function", staticmethod(compute)
        ):
            yield


def record_catalogs(stack, rows):
    native = single_field.determine_halo_catalog

    def observe(**kwargs):
        output = native(**kwargs)
        count = int(output.n_halos)
        row = {"z": output.redshift, "n": count}
        for name in CATALOG_FIELDS:
            values = np.ascontiguousarray(output.get(name)[:count])
            row[name] = (
                hashlib.sha256(values.view(np.uint8)).hexdigest() if count else None
            )
        rows.append(row)
        return output

    stack.enter_context(patch.object(single_field, "determine_halo_catalog", observe))


def requested_fields(case):
    fields = FIELDS + (THERMAL_FIELDS if case.thermal else ())
    return fields + (("J_21_LW", "sfrd_mcg") if case.molecular else ())


def check_metadata(cone, inputs, case):
    expected = requested_fields(case)
    if cone.inputs != inputs or not set(expected).issubset(cone.lightcones):
        raise ValueError("LightCone inputs or fields do not match the requested case")
    if cone.n_slices < 1 or cone.shape != (case.hii_dim, case.hii_dim, cone.n_slices):
        raise ValueError("Unexpected lightcone shape")
    if not all(np.isfinite(x).all() for x in cone.lightcones.values()):
        raise ValueError("Non-finite lightcone field")


def check_run_record(record, cone, specs, mode):
    if record.get("mode") != mode or record.get("version") != p21.__version__:
        raise ValueError(f"Saved {mode} record is from a different mode or version")
    # Older records predate the explicit source_model and thermal CLI switches.
    recorded = {
        "source_model": "CHMF-SAMPLER",
        "thermal": "spin_temperature" in cone.lightcones,
        **record,
    }
    if any(recorded.get(key) != value for key, value in specs.items()):
        raise ValueError(f"Saved {mode} record is from a different case")


def evaluate_gpu(directory, case, native, reference, cone, record):
    products = compare_saved_slices(directory / "cpu.h5", directory / "gpu.h5")
    coordinates = native.lightcone_distances.to_value("Mpc")
    equal_coordinates = np.array_equal(
        coordinates, cone.lightcone_distances.to_value("Mpc")
    )
    record.update(
        reference_job=reference["job"],
        reference_elapsed_seconds=reference["elapsed_seconds"],
        coordinates_equal=equal_coordinates,
        inputs_equal=native.inputs == cone.inputs,
        catalogs_equal=record["catalogs"] == reference["catalogs"],
        fields=products["lightcones"],
        global_quantities=products["global_quantities"],
        brightness_transverse_power=products["brightness_transverse_power"],
        rtol=RTOL,
        atol=ATOL,
    )
    if case.class_streaming:
        cpu = saved_class_signature(directory / "cpu.h5", case.class_streaming)
        gpu = saved_class_signature(directory / "gpu.h5", case.class_streaming)
        # Old reports contain only the transfer hashes, or no fingerprint at all.
        # The two archived HDF5 files are authoritative in compare-only mode.
        record["class_tables"] = gpu["transfers"]
        record["class_normalization"] = gpu["scalars"]
        record["class_tables_equal"] = gpu == cpu
        donor = reference.get("donor_class_signature")
        if case.cosmology_from:
            donor = saved_class_signature(case.cosmology_from, case.class_streaming)
        if donor is not None:
            record["donor_class_tables_equal"] = cpu == donor
    if case.molecular:
        # Absolute tolerances in physical units can exceed the entire SFRD field.
        # Scale the absolute term by each native field's measured peak.
        rows = {}
        for name in ("sfrd_acg", "sfrd_mcg", "xray_emissivity", "J_21_LW"):
            expected = np.asarray(native.lightcones[name], dtype="d")
            got = np.asarray(cone.lightcones[name], dtype="d")
            peak = float(np.max(np.abs(expected)))
            delta = np.abs(got - expected)
            rows[name] = dict(
                peak=peak,
                different=int(np.count_nonzero(delta)),
                max_abs=float(delta.max()),
                tested=expected.size,
                failed=int((delta > 1e-5 * np.abs(expected) + 1e-10 * peak).sum()),
            )
        record["scaled_molecular_fields"] = rows
    if case.class_streaming == "FLUCTS":
        from py21cmfast.io.h5 import read_output_struct

        velocities = []
        for backend in ("cpu", "gpu"):
            (path,) = (directory / f"{backend}-cache").rglob("InitialConditions.h5")
            ic = read_output_struct(path, struct="InitialConditions")
            velocities.append(np.asarray(ic.get("lowres_vcb"), dtype="d"))
        expected, actual = velocities
        if actual.shape != expected.shape or not np.isfinite(velocities).all():
            raise ValueError("Invalid relative-velocity IC grid")
        record["streaming_velocity"] = dict(
            cells=expected.size,
            min_km_s=float(expected.min()),
            max_km_s=float(expected.max()),
            rms_km_s=float(np.sqrt(np.mean(expected**2))),
            different=int(np.count_nonzero(actual != expected)),
        )
        if np.ptp(expected) == 0 or record["streaming_velocity"]["different"]:
            raise ValueError(
                "The spatial relative-velocity grid must vary and match the native IC"
            )
    if case.thermal and len(case.nodes) > 1:
        counters = record["routing"]["counters"]
        required = THERMAL_STAGES
        if case.lya_scattering:
            required += ("radiation_lya_scattering_filter",)
        if case.molecular:
            # The MCG optical depth uses both photon histories and the actual shell turnover.
            required = tuple(
                s for s in required if s != "radiation_xray_optical_depth_and_frequency"
            ) + (
                "mcg_xray_optical_depth_and_frequency",
                "radiation_mcg_global_nion",
                "radiation_ionization_interpolation",
                "mcg_turnover_grids",
                "mcg_lw_shells",
                "mcg_spin_lw",
                "mcg_lw_normalizations",
            )
            if case.source_model == "E-INTEGRAL":
                required += (
                    "eulerian_mcg_emissivity_cells",
                    "eulerian_mcg_global_integrals",
                    "eulerian_mcg_ion_turnover_grids",
                    "eulerian_mcg_ion_filter_radius",
                    "eulerian_mcg_ion_ion_center",
                    "eulerian_mcg_ion_global_nion",
                    "ion_mcg_radius_history",
                    "ion_mcg_radius_previous_correction",
                )
            else:
                required += (
                    "subgrid_molecular_advection_cells",
                    "ionization_halo_barrier",
                    "ionization_global_nion",
                )
                required += tuple(
                    f"subgrid_{quantity}_{pop}_2d_tables"
                    for quantity in ("nion", "sfrd", "xray")
                    for pop in ("acg", "mcg")
                )
            if case.source_model == "CHMF-SAMPLER":
                required += ("sampled_mcg_halos",)
        # An archived run should retain the routing contract of its own code version.
        if case.mode == "gpu":
            record["required_thermal_stages"] = list(required)
        if "required_thermal_stages" in record:
            record["missing_thermal_stages"] = [
                stage
                for stage in record["required_thermal_stages"]
                if counters.get(stage, 0) < 1
            ]
        elif "missing_thermal_stages" not in record:
            raise ValueError("The saved run has no routing validation record")
    record["passed"] = (
        all(
            r["failed"] == 0 for r in record.get("scaled_molecular_fields", {}).values()
        )
        and record.get("class_tables_equal", True)
        and record.get("donor_class_tables_equal", True)
        and not record.get("missing_thermal_stages")
        and record["brightness_transverse_power"]["failed"] == 0
        and all(
            record[key]
            for key in ("inputs_equal", "coordinates_equal", "catalogs_equal")
        )
        and all(
            row["failed"] == 0
            for group in ("fields", "global_quantities")
            for row in record[group].values()
        )
    )


def main(case):
    directory = case.directory
    directory.mkdir(parents=True, exist_ok=True)
    specs = {
        "source_model": case.source_model,
        "thermal": case.thermal,
        "seed": case.seed,
        "dim": case.dim,
        "hii_dim": case.hii_dim,
        "box_mpc": case.box_mpc,
        "nthreads": case.threads,
        "nodes": case.nodes,
        "bounds": case.bounds,
    }
    options = (
        {
            "USE_TS_FLUCT": True,
            "RECOMB_MODEL": "inhomogeneous",
            "Z_HEAT_MAX": max(case.nodes),
        }
        if case.thermal
        else {}
    )
    if case.lya_scattering:
        if not case.thermal:
            raise ValueError("Ly-alpha scattering needs a thermal history")
        options["LYA_MULTIPLE_SCATTERING"] = True
        specs["lya_scattering"] = True
    if case.molecular:
        if not case.thermal:
            raise ValueError("Molecular cooling needs a thermal history")
        options.update(
            USE_MCGS=True,
            USE_REIONIZATION_PHOTOHEATING_FEEDBACK=True,
            V_CB_MODEL="AVG-DEBUG",
            M_TURN_STELLAR_FEEDBACK=6.0,
        )
        specs["molecular"] = True
    if case.class_streaming:
        if (case.class_streaming == "NONE") == case.molecular:
            raise ValueError(
                "Use nonzero CLASS streaming with --molecular and NONE without it"
            )
        options.update(POWER_SPECTRUM="CLASS", V_CB_MODEL=case.class_streaming)
        specs["class_streaming"] = True
        if case.class_streaming != "AVG-AUTO":
            specs["streaming_model"] = case.class_streaming
    if case.source_model == "E-INTEGRAL":
        # Two native filter radii for the 24 Mpc / 8-cell Eulerian fixture.
        options.update(
            USE_METALLICITY=False,
            USE_EXP_FILTER=False,
            CELL_RECOMB=False,
            INTEGRATION_METHOD_ACGS="GAUSS-LEGENDRE",
            INTEGRATION_METHOD_MCGS="GAUSS-LEGENDRE",
            R_BUBBLE_MAX=4.0,
            DELTA_R_HII_FACTOR=2.0,
            N_STEP_TS=12,
        )
    if case.sample_method is not None:
        if case.source_model != "CHMF-SAMPLER":
            raise ValueError("A progenitor sampling method requires CHMF-SAMPLER")
        options["SAMPLE_METHOD"] = case.sample_method
        specs["sample_method"] = case.sample_method
        if case.sample_method == "PARTITION":
            specs["native_partition_reference_null_fix"] = True
    if case.integration_method is not None:
        options["INTEGRATION_METHOD_ACGS"] = case.integration_method
        specs["integration_method"] = case.integration_method
    if case.matter_filter is not None:
        options["FILTER"] = case.matter_filter
        specs["matter_filter"] = case.matter_filter
    inputs = p21.InputParameters(random_seed=case.seed, node_redshifts=case.nodes)
    inputs = inputs.evolve_input_structs(
        SOURCE_MODEL=case.source_model,
        USE_UPPER_STELLAR_TURNOVER=case.source_model in ("CHMF-SAMPLER", "DEXM-ESF"),
        DIM=case.dim,
        HII_DIM=case.hii_dim,
        BOX_LEN=case.box_mpc,
        N_THREADS=case.threads,
        **options,
    )
    lightconer = p21.lightconers.RectilinearLightconer.between_redshifts(
        min_redshift=case.bounds[0],
        max_redshift=case.bounds[1],
        quantities=requested_fields(case),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    cpu_h5, cpu_json = directory / "cpu.h5", directory / "cpu.json"
    native, reference = None, None
    reused_cosmology = False
    if case.cosmology_from and case.mode == "cpu" and not cpu_h5.exists():
        donor_signature = reuse_class_tables(inputs, case.cosmology_from)
        reused_cosmology = True
    if case.mode != "cpu" or cpu_h5.exists() or cpu_json.exists():
        native = p21.LightCone.from_file(cpu_h5)
        reference = json.loads(cpu_json.read_text())
        check_metadata(native, inputs, case)
        np.testing.assert_array_equal(
            native.lightcone_distances.to_value("Mpc"),
            lightconer.lc_distances.to_value("Mpc"),
        )
        check_run_record(reference, native, specs, "cpu")
        if case.class_streaming and case.mode == "gpu":
            saved_class_signature(cpu_h5, case.class_streaming)
            inputs = native.inputs
            reused_cosmology = True
        if case.mode == "cpu":
            print(f"Reusing {cpu_h5}")
            return

    if case.mode == "compare":
        cone = p21.LightCone.from_file(directory / "gpu.h5")
        record = json.loads((directory / "gpu.json").read_text())
        check_metadata(cone, inputs, case)
        check_run_record(record, cone, specs, "gpu")
        evaluate_gpu(directory, case, native, reference, cone, record)
        print(
            json.dumps(
                {
                    "cpu_job": reference["job"],
                    "gpu_job": record["job"],
                    "fields": record["fields"],
                    "global_quantities": record["global_quantities"],
                    "scaled_molecular_fields": record.get("scaled_molecular_fields"),
                    "catalogs_equal": record["catalogs_equal"],
                    "class_tables_equal": record.get("class_tables_equal"),
                    "donor_class_tables_equal": record.get("donor_class_tables_equal"),
                    "power": record["brightness_transverse_power"],
                    "passed": record["passed"],
                },
                indent=2,
                allow_nan=False,
            )
        )
        if not record["passed"]:
            raise SystemExit("Saved native and accelerated results differ")
        return

    rows = []
    run = p21.run_lightcone
    options = {}
    if case.mode == "gpu":
        from gpu_lightcone import run_lightcone

        run = run_lightcone
        options.update(diagnostics_dir=directory / "routing", fallback_to_cpu=False)

    if case.molecular:
        options["cache"] = p21.OutputCache(directory / f"{case.mode}-cache")
    class_calls = []
    with ExitStack() as stack:
        record_catalogs(stack, rows)
        if case.mode == "cpu" and case.sample_method == "PARTITION":
            stack.enter_context(native_partition_reference())
        if case.class_streaming:
            from py21cmfast.wrapper import inputs as native_inputs

            solver = native_inputs.run_classy

            def measure_class(**kwargs):
                before = perf_counter()
                value = solver(**kwargs)
                class_calls.append(
                    {
                        "seconds": perf_counter() - before,
                        "requested": str(kwargs.get("output")),
                    }
                )
                return value

            stack.enter_context(
                patch.object(native_inputs, "run_classy", measure_class)
            )
        start = perf_counter()
        cone = run(
            inputs=inputs,
            lightconer=lightconer,
            write=case.molecular,
            regenerate=True,
            **options,
        )
        elapsed = perf_counter() - start
    check_metadata(cone, inputs, case)
    if case.molecular and any(
        not np.any(cone.lightcones[q] > 0) for q in ("J_21_LW", "sfrd_mcg")
    ):
        raise ValueError(
            "The molecular fixture needs nonzero star formation and LW radiation"
        )
    if inputs.matter_options.has_discrete_halos:
        if sorted(r["z"] for r in rows) != sorted(case.nodes) or not sum(
            r["n"] for r in rows
        ):
            raise ValueError(
                "Expected a nonempty catalog trajectory covering every native node"
            )
    elif rows:
        raise ValueError("Unexpected sampled catalog in an integral-source run")
    cone.save(directory / f"{case.mode}.h5", clobber=True)
    record = {
        **specs,
        "mode": case.mode,
        "host": platform.node(),
        "job": os.getenv("SLURM_JOB_ID"),
        "version": p21.__version__,
        "elapsed_seconds": elapsed,
        "catalogs": rows,
        "fields_requested": requested_fields(case),
        "slice_count": cone.n_slices,
        "slice_bounds": np.asarray(cone.lightcone_redshifts)[[0, -1]].tolist(),
    }
    if case.class_streaming:
        record["class_solver_calls_during_run"] = class_calls
        record["native_cosmo_tables_reused"] = reused_cosmology
        if case.mode == "cpu":
            signature = saved_class_signature(cpu_h5, case.class_streaming)
            record["class_tables"] = signature["transfers"]
            record["class_normalization"] = signature["scalars"]
            if case.cosmology_from:
                record["cosmology_from"] = str(case.cosmology_from)
                record["donor_class_signature"] = donor_signature
                if signature != donor_signature:
                    raise ValueError(
                        "Persisted CPU cosmology differs from its CLASS donor"
                    )
    if case.mode == "gpu":
        record["routing"] = json.loads(
            (directory / "routing" / "backend-routing.json").read_text()
        )
        evaluate_gpu(directory, case, native, reference, cone, record)
    save_json(directory / f"{case.mode}.json", record)
    if not record.get("passed", True):
        raise SystemExit(
            f"Native and accelerated results differ; inspect {directory / 'gpu.json'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-streaming",
        nargs="?",
        const="AVG-AUTO",
        choices=("NONE", "AVG-AUTO", "FLUCTS"),
        help="Use CLASS with density alone, its mean streaming velocity, or a spatial velocity field",
    )
    parser.add_argument(
        "--cosmology-from",
        type=Path,
        help="Seed a new native CPU reference with matching archived CLASS tables",
    )
    parser.add_argument("mode", choices=("cpu", "gpu", "compare"))
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--source-model",
        choices=("CHMF-SAMPLER", "DEXM-ESF", "L-INTEGRAL", "E-INTEGRAL"),
        default="CHMF-SAMPLER",
    )
    parser.add_argument(
        "--lya-scattering",
        action="store_true",
        help="Enable Ly-alpha multiple scattering",
    )
    parser.add_argument(
        "--thermal",
        action="store_true",
        help="Run a short Lagrangian-source history with spin fluctuations, inhomogeneous recombinations and Z_HEAT_MAX at the highest requested node",
    )
    parser.add_argument(
        "--molecular",
        action="store_true",
        help="Use molecular cooling and save the native coeval history for single-node replays",
    )
    parser.add_argument(
        "--sample-method",
        choices=("MASS-LIMITED", "NUMBER-LIMITED", "BINARY-SPLIT", "PARTITION"),
    )
    parser.add_argument("--integration-method", choices=("GAUSS-LEGENDRE", "GSL-QAG"))
    parser.add_argument("--matter-filter", choices=("spherical-tophat", "gaussian"))
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--hii-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12451)
    parser.add_argument("--box-mpc", type=float, default=42.0)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--nodes", type=float, nargs="+", default=[9.3, 9.1, 8.9])
    parser.add_argument("--bounds", type=float, nargs=2, default=[8.92, 9.15])
    main(parser.parse_args())
