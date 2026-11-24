"""Compare sampled MCG halo emissivities to the 4.3.2 native driver at a single node."""

import argparse
import json
import os
from pathlib import Path

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.wrapper.outputs import HaloCatalog, IonizedBox, TsBox

from halo_grid import halo_grid_backend


def inputs_and_history(mcg, photo, metal, suppress, mixed=False, *, auto=False):
    side, low = (16, 8) if auto else (24, 12)
    inputs = p21.InputParameters(
        random_seed=424242, node_redshifts=[10.0, 9.0]
    ).evolve_input_structs(
        DIM=side,
        HII_DIM=low,
        BOX_LEN=20.0,
        N_THREADS=1,
        SOURCE_MODEL="CHMF-SAMPLER",
        PERTURB_ON_HIGH_RES=False,
        V_CB_MODEL="AVG-AUTO" if auto else ("AVG-DEBUG" if mcg else "NONE"),
        POWER_SPECTRUM="CLASS" if auto else "EH",
        **({"V_CB_AVG_DEBUG": 55.0, "K_MAX_FOR_CLASS": 20.0} if auto else {}),
        USE_TS_FLUCT=True,
        USE_MCGS=mcg,
        RECOMB_MODEL="inhomogeneous",
        USE_METALLICITY=metal,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=photo,
        Z_HEAT_MAX=10.0,
        M_TURN_STELLAR_FEEDBACK=9.5 if suppress else 5.0,
        SAMPLER_MIN_MASS=5e7 if mixed else 1000.0,
    )
    x, y, z = np.indices((low,) * 3)
    spin = TsBox.new(inputs=inputs, redshift=10.0)
    ion = IonizedBox.new(inputs=inputs, redshift=10.0)
    if mcg:
        spin.set("J_21_LW", np.asarray((1 + x + y + z) / 300.0, dtype="f"))
    if photo or mcg:
        ion.set("ionisation_rate_G12", np.where(x > low // 2, 0.3, 0.02).astype("f"))
        ion.set("z_reion", np.where(x > low // 2, 12.0, -1.0).astype("f"))
    return inputs, spin if mcg else None, ion if (photo or mcg) else None


def catalog(inputs):
    rng = np.random.default_rng(1776)
    n = 521
    cat = HaloCatalog.new(inputs=inputs, redshift=9.0, buffer_size=n, n_halos=n)
    masses = (10.0 ** rng.uniform(7.0, 10.0, n)).astype("f")
    masses[0] = 0.0
    coords = rng.uniform(0, 20.0, (n, 3)).astype("f")
    coords[1:9] = [20.0, 0.125, 19.875]
    cat.set("halo_masses", masses)
    cat.set("halo_coords", coords)
    for k in ("star_rng", "sfr_rng", "xray_rng"):
        cat.set(k, rng.normal(size=n).astype("f"))
    return cat


def compare(fields, expected):
    result = {}
    for key in expected.arrays:
        if not expected.arrays[key].state.computed_in_mem:
            continue
        a, b = expected.get(key), fields.get(key)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(
                f"Shape or finiteness changed in {key}: native {np.count_nonzero(~np.isfinite(a))}, device {np.count_nonzero(~np.isfinite(b))}"
            )
        difference = abs(a.astype("d") - b.astype("d"))
        nonzero = a != 0
        relative = difference[nonzero] / abs(a[nonzero].astype("d"))
        normal = np.abs(a.astype("d")) >= np.finfo("f").tiny
        # Compare float32 encodings so a flushed subnormal cannot pass the
        # ordinary absolute-error tolerance or disappear through host arithmetic.
        native_bits, gpu_bits = a.view(np.uint32), b.view(np.uint32)
        magnitude_bits = native_bits & np.uint32(0x7FFFFFFF)
        subnormal = (magnitude_bits > 0) & (magnitude_bits < 0x00800000)
        if np.any(difference):
            idx = np.unravel_index(difference.argmax(), a.shape)
            print(
                key,
                "largest discrepancy",
                repr(float(a[idx])),
                repr(float(b[idx])),
                flush=True,
            )
        result[key] = dict(
            cells=a.size,
            nonzero=int(nonzero.sum()),
            nonidentical=int(np.count_nonzero(difference)),
            native_subnormals=int(subnormal.sum()),
            subnormal_bit_mismatches=int(
                np.count_nonzero(subnormal & (native_bits != gpu_bits))
            ),
            peak=float(np.max(abs(a))),
            max_abs=float(difference.max()),
            max_rel=float(relative.max(initial=0.0)),
            max_normal_rel=float(
                (difference[normal] / np.abs(a[normal].astype("d"))).max(initial=0.0)
            ),
            failed=int(np.count_nonzero(difference > 1e-8 + 1e-4 * abs(a.astype("d")))),
        )
    return result


def run(path):
    cases = []
    for label, mcg, photo, metal, suppress, mixed in (
        ("no-mcg-control", False, False, True, False, False),
        ("constant-xray", True, False, False, False, False),
        ("local-turnovers", True, True, True, False, False),
        ("above-atomic-threshold", True, False, True, True, False),
        ("with-unresolved-sources", True, True, True, False, True),
    ):
        with p21.config.use(EXTRA_EMISSIVITY_FIELDS=True):
            inputs, spin, ion = inputs_and_history(mcg, photo, metal, suppress, mixed)
            kw = dict(
                inputs=inputs,
                redshift=9.0,
                initial_conditions=p21.compute_initial_conditions(
                    inputs=inputs, write=False, regenerate=True
                ),
                halo_catalog=catalog(inputs),
                previous_spin_temp=spin,
                previous_ionize_box=ion,
                write=False,
                regenerate=True,
            )
            expected = p21.compute_emissivity_fields(**kw)
            report = {}
            with halo_grid_backend(report):
                fields = p21.compute_emissivity_fields(**kw)
            result = compare(fields, expected)
        print(
            label,
            {
                k: (v["nonidentical"], v["max_abs"], v["max_rel"], v["peak"])
                for k, v in result.items()
            },
            flush=True,
        )
        assert all(row["failed"] == 0 for row in result.values())
        assert all(row["subnormal_bit_mismatches"] == 0 for row in result.values())
        assert all(row["max_normal_rel"] <= 1.0e-4 for row in result.values())
        assert report["halo_grid_halos"] == 521
        if mcg:
            assert report["sampled_mcg_halos"] == 521
            for field in ("sfrd_mcg", "stellar_mass_density_mcg"):
                assert bool(result[field]["nonzero"]) != suppress
        cases.append(
            dict(
                label=label,
                mcg=mcg,
                photoheating=photo,
                metallicity=metal,
                sampler_min_mass=inputs.simulation_options.SAMPLER_MIN_MASS,
                fields=result,
                counters=report,
            )
        )
        print(
            label,
            [(k, v["nonzero"], v["nonidentical"]) for k, v in result.items()],
            flush=True,
        )
    assert (
        sum(
            row["native_subnormals"]
            for case in cases
            for row in case["fields"].values()
        )
        > 0
    )
    path.write_text(
        json.dumps(
            dict(
                job=os.environ.get("SLURM_JOB_ID"),
                version=p21.__version__,
                device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                dim=24,
                hii_dim=12,
                box_mpc=20.0,
                seed=424242,
                threads=1,
                redshift=9.0,
                catalog="521 matched halo rows, one masked, log-uniform masses from 1e7 to 1e10 Msun, shared positions at the periodic boundary and three identical Gaussian scatter arrays",
                scope="compute_emissivity_fields on the same native IC and catalogue. At a 1000 Msun sampler threshold, the tested mass interval contains only resolved sources. The 5e7 Msun threshold also exercises native conditional MCG integrals and advection below the threshold.",
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )


def verify_auto(path):
    import importlib.metadata
    from unittest.mock import patch
    from py21cmfast.wrapper import inputs as native_inputs
    import halo_mcg_turnover

    history = {}
    rms = native_inputs.compute_rms
    decouple = native_inputs.find_redshift_kinematic_decoupling

    def observe_rms(*args, **kwargs):
        result = rms(*args, **kwargs)
        if kwargs.get("kind") == "v_cb":
            history["rms_km_s"] = float(result.to_value("km/s")[0])
        return result

    def observe_decoupling(*args, **kwargs):
        result = decouple(*args, **kwargs)
        history["decoupling_redshift"] = float(result)
        return result

    with p21.config.use(EXTRA_EMISSIVITY_FIELDS=True):
        inputs, spin, ion = inputs_and_history(True, True, True, False, auto=True)
        with (
            patch.object(native_inputs, "compute_rms", observe_rms),
            patch.object(
                native_inputs, "find_redshift_kinematic_decoupling", observe_decoupling
            ),
        ):
            history["cosmo_table_mean_km_s"] = float(inputs.cosmo_tables.V_CB_AVG)
        np.testing.assert_allclose(
            history["cosmo_table_mean_km_s"],
            np.sqrt(8.0 / (3.0 * np.pi)) * history["rms_km_s"],
            atol=1.0e-12,
            rtol=0,
        )
        assert history["cosmo_table_mean_km_s"] != 27.0
        kw = dict(
            inputs=inputs,
            redshift=9.0,
            initial_conditions=p21.compute_initial_conditions(
                inputs=inputs, write=False, regenerate=True
            ),
            halo_catalog=catalog(inputs),
            previous_spin_temp=spin,
            previous_ionize_box=ion,
            write=False,
            regenerate=True,
        )
        expected = p21.compute_emissivity_fields(**kw)
        report = {}
        kernel = halo_mcg_turnover._kernel()

        def observe_turnover(grid, block, args):
            params = args[1].get()
            history["gpu_turnover_vcb_km_s"] = float(params[3])
            history["gpu_turnover_sigma_vcb_km_s"] = float(params[4])
            return kernel(grid, block, args)

        with (
            patch.object(halo_mcg_turnover, "_kernel", return_value=observe_turnover),
            halo_grid_backend(report),
        ):
            gpu = p21.compute_emissivity_fields(**kw)
        fields = compare(gpu, expected)
        history["log10_mean_turnover_mcg_cpu"] = float(expected.log10_mturn_mcg_ave)
        history["log10_mean_turnover_mcg_gpu"] = float(gpu.log10_mturn_mcg_ave)
    assert report["mcg_turnover_cells"] == 8**3 and report["sampled_mcg_halos"] == 521
    assert (
        fields["sfrd_mcg"]["nonzero"] > 0
        and fields["stellar_mass_density_mcg"]["nonzero"] > 0
    )
    assert all(
        v["failed"] == 0 and v["max_normal_rel"] <= 1.0e-4 for v in fields.values()
    )
    assert history["gpu_turnover_vcb_km_s"] == history["cosmo_table_mean_km_s"]
    np.testing.assert_allclose(
        history["gpu_turnover_sigma_vcb_km_s"],
        history["gpu_turnover_vcb_km_s"] * np.sqrt(3.0 * np.pi / 8),
        rtol=1.0e-14,
    )
    history["mean_turnover_abs_error"] = abs(
        history["log10_mean_turnover_mcg_gpu"] - history["log10_mean_turnover_mcg_cpu"]
    )
    print(history, flush=True)
    assert history["mean_turnover_abs_error"] < 1.0e-8 + 1.0e-4 * abs(
        history["log10_mean_turnover_mcg_cpu"]
    )
    data = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        version=p21.__version__,
        classy_version=importlib.metadata.version("classy"),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        scope="One CHMF-SAMPLER emissivity stage on a shared native CLASS-generated IC. Native CLASS and its derived velocity mean run on the host. No complete lightcone.",
        native_model="AVG-AUTO",
        power_spectrum="CLASS",
        debug_velocity_km_s=inputs.astro_params.V_CB_AVG_DEBUG,
        k_max_for_class_inverse_mpc=inputs.simulation_options.K_MAX_FOR_CLASS,
        source="CHMF-SAMPLER",
        seed=424242,
        redshift=9.0,
        dim=16,
        hii_dim=8,
        box_mpc=20.0,
        threads=1,
        halo_rows=521,
        use_mcg=True,
        metallicity=True,
        photoheating=True,
        sampler_min_mass=inputs.simulation_options.SAMPLER_MIN_MASS,
        history=history,
        counters=report,
        fields=fields,
    )
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--avg-auto", action="store_true")
    args = parser.parse_args()
    (verify_auto if args.avg_auto else run)(args.output)
