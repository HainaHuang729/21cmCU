"""Run paired large-box CPU vs GPU verification for newly ported 21cmFAST 4.3.2 features."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.lightconers import RectilinearLightconer

from compare_lightcone_ps import compare_saved_slices
from gpu_lightcone import run_lightcone
from verify_chmf_lightcone import (
    CATALOG_FIELDS,
    FIELDS,
    THERMAL_FIELDS,
    record_catalogs,
    saved_class_signature,
)


def _inspect_failures(case_dir: Path, cones: dict, products: dict, molecular: bool):
    import h5py
    from py21cmfast.io.h5 import read_output_struct

    diag = {"failing_lightcone_fields": {}}
    redshifts = np.asarray(cones["cpu"].lightcone_redshifts, dtype=np.float64)
    with (
        h5py.File(case_dir / "cpu.h5", "r") as cpu_h5,
        h5py.File(case_dir / "gpu.h5", "r") as gpu_h5,
    ):
        lc_cpu = cpu_h5["lightcones"]
        lc_gpu = gpu_h5["lightcones"]
        available_fields = sorted(lc_cpu.keys())

        for key, row in products["lightcones"].items():
            if row["failed"] == 0:
                continue
            a = np.asarray(lc_cpu[key][:], dtype=np.float64)
            b = np.asarray(lc_gpu[key][:], dtype=np.float64)
            tol = 1e-8 + 1e-4 * np.abs(a)
            delta = np.abs(b - a)
            bad = delta > tol
            bad_idx = np.argwhere(bad)
            ratios = delta[bad] / tol[bad]
            order = np.argsort(ratios)[::-1][:15]
            top_sites = []
            sibling_names = [
                f
                for f in (
                    "spin_temperature",
                    "tau_21",
                    "neutral_fraction",
                    "density",
                    "los_velocity",
                    "velocity_z",
                    "kinetic_temp_neutral",
                    "xray_ionised_fraction",
                    "J_21_LW",
                    "sfrd_mcg",
                    "sfrd_acg",
                )
                if f in available_fields and f != key
            ]
            for idx_pos in order:
                ix, iy, iz = (int(v) for v in bad_idx[idx_pos])
                z_val = float(redshifts[iz])
                t_cmb = 2.7255 * (1.0 + z_val)
                site = {
                    "xyz": [ix, iy, iz],
                    "z": z_val,
                    "T_CMB": t_cmb,
                    "cpu": float(a[ix, iy, iz]),
                    "gpu": float(b[ix, iy, iz]),
                    "abs_diff": float(delta[ix, iy, iz]),
                    "tol": float(tol[ix, iy, iz]),
                    "ratio": float(delta[ix, iy, iz] / tol[ix, iy, iz]),
                    "siblings": {},
                }
                for sib in sibling_names:
                    sc = float(lc_cpu[sib][ix, iy, iz])
                    sg = float(lc_gpu[sib][ix, iy, iz])
                    site["siblings"][sib] = {
                        "cpu": sc,
                        "gpu": sg,
                        "diff": sg - sc,
                    }
                top_sites.append(site)

            summary = {
                "failed": int(bad.sum()),
                "max_abs": float(delta.max()),
                "max_ratio": float(ratios.max()),
                "cpu_abs_at_failed": [
                    float(np.abs(a[bad]).min()),
                    float(np.median(np.abs(a[bad]))),
                    float(np.abs(a[bad]).max()),
                ],
                "top_sites": top_sites,
            }
            if "spin_temperature" in available_fields and key == "brightness_temp":
                ts_bad = np.asarray(lc_cpu["spin_temperature"][:], dtype=np.float64)[
                    bad
                ]
                z_bad = redshifts[bad_idx[:, 2]]
                tcmb_bad = 2.7255 * (1.0 + z_bad)
                summary["ts_minus_tcmb_abs_at_failed"] = [
                    float(np.abs(ts_bad - tcmb_bad).min()),
                    float(np.median(np.abs(ts_bad - tcmb_bad))),
                    float(np.abs(ts_bad - tcmb_bad).max()),
                ]
            diag["failing_lightcone_fields"][key] = summary

    if (
        molecular
        and (case_dir / "cpu-cache").exists()
        and (case_dir / "gpu-cache").exists()
    ):
        cpu_files = {
            p.relative_to(case_dir / "cpu-cache"): p
            for p in sorted((case_dir / "cpu-cache").rglob("*.h5"))
        }
        gpu_files = {
            p.relative_to(case_dir / "gpu-cache"): p
            for p in sorted((case_dir / "gpu-cache").rglob("*.h5"))
        }
        coeval_diffs = []
        for rel in sorted(set(cpu_files) & set(gpu_files)):
            struct = rel.stem
            sc = read_output_struct(cpu_files[rel], struct=struct)
            sg = read_output_struct(gpu_files[rel], struct=struct)
            z_attr = getattr(sc, "redshift", None)
            field_diffs = {}
            for fname, arr_obj in getattr(sc, "arrays", {}).items():
                if not (arr_obj.state.computed_in_mem or arr_obj.state.on_disk):
                    continue
                sg_obj = sg.arrays.get(fname)
                if sg_obj is None or not (
                    sg_obj.state.computed_in_mem or sg_obj.state.on_disk
                ):
                    continue
                ac = np.asarray(sc.get(fname), dtype=np.float64)
                ag = np.asarray(sg.get(fname), dtype=np.float64)
                d = np.abs(ag - ac)
                n_diff = int(np.count_nonzero(d))
                if n_diff:
                    peak = float(np.max(np.abs(ac))) if ac.size else 0.0
                    tol_strict = 1e-8 + 1e-4 * np.abs(ac)
                    tol_scaled = 1e-5 * np.abs(ac) + 1e-10 * peak
                    field_diffs[fname] = {
                        "shape": list(ac.shape),
                        "different": n_diff,
                        "max_abs": float(d.max()),
                        "peak": peak,
                        "failed_strict": int((d > tol_strict).sum()),
                        "failed_scaled": int((d > tol_scaled).sum()),
                    }
            if field_diffs:
                coeval_diffs.append(
                    {
                        "relpath": str(rel),
                        "struct": struct,
                        "redshift": float(z_attr) if z_attr is not None else None,
                        "fields": field_diffs,
                    }
                )
        diag["coeval_diffs"] = coeval_diffs
    return diag


def _check_all_passed(products, scaled_molecular=None):
    failures = []
    for grp in ("lightcones", "global_quantities"):
        for k, row in products[grp].items():
            if row["failed"] != 0:
                failures.append(
                    f"{grp}/{k} failed={row['failed']} max_abs={row['max_abs']}"
                )
    if products["brightness_transverse_power"]["failed"] != 0:
        failures.append(
            f"brightness_transverse_power failed={products['brightness_transverse_power']['failed']}"
        )
    if scaled_molecular:
        for k, row in scaled_molecular.items():
            if row["failed"] != 0:
                failures.append(f"scaled_molecular/{k} failed={row['failed']}")
    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def _run_pair(
    name: str,
    work_dir: Path,
    inputs,
    lightconer,
    *,
    class_streaming=None,
    molecular=False,
    track_catalogs=False,
    required_counters=(),
):
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )
    from py21cmfast.wrapper import inputs as native_inputs

    case_dir = work_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    cones = {}

    for mode in ("cpu", "gpu"):
        manager.free()
        active_inputs = inputs
        if mode == "gpu" and class_streaming:
            saved_class_signature(case_dir / "cpu.h5", class_streaming)
            active_inputs = cones["cpu"].inputs

        cat_rows = []
        class_calls = []
        run_fn = run_lightcone if mode == "gpu" else p21.run_lightcone
        extra_kwargs = {}
        if mode == "gpu":
            extra_kwargs.update(
                diagnostics_dir=case_dir / "routing", fallback_to_cpu=False
            )
        if molecular:
            extra_kwargs["cache"] = p21.OutputCache(case_dir / f"{mode}-cache")

        with ExitStack() as stack:
            if track_catalogs:
                record_catalogs(stack, cat_rows)
            if class_streaming:
                orig_run_classy = native_inputs.run_classy

                def timed_classy(**kwargs):
                    t0 = perf_counter()
                    val = orig_run_classy(**kwargs)
                    class_calls.append(
                        {
                            "seconds": perf_counter() - t0,
                            "output": str(kwargs.get("output")),
                        }
                    )
                    return val

                stack.enter_context(
                    patch.object(native_inputs, "run_classy", timed_classy)
                )

            # Run once cleanly (note for GPU: run warm pass too if non-CLASS-heavy so we record both first & warm)
            t0 = perf_counter()
            cone = run_fn(
                inputs=active_inputs,
                lightconer=lightconer,
                write=molecular,
                regenerate=True,
                **extra_kwargs,
            )
            elapsed = perf_counter() - t0

        h5_path = case_dir / f"{mode}.h5"
        t_save0 = perf_counter()
        cone.save(h5_path, clobber=True)
        with h5_path.open("rb") as f:
            os.fsync(f.fileno())
        save_s = perf_counter() - t_save0

        cones[mode] = cone
        class_s = sum(c["seconds"] for c in class_calls)
        results[mode] = {
            "elapsed_seconds": elapsed,
            "save_fsync_seconds": save_s,
            "class_solver_seconds": class_s,
            "elapsed_excl_class_seconds": elapsed - class_s,
            "class_solver_calls": len(class_calls),
            "slice_count": int(cone.n_slices),
            "catalogs": cat_rows,
        }

    # Warm GPU timing repeat cleanly reusing cached cosmo_tables/shims/CHMF modules
    warm_gpu_s = None
    if not molecular:
        manager.free()
        t_warm0 = perf_counter()
        _ = run_lightcone(
            inputs=cones["cpu"].inputs if class_streaming else inputs,
            lightconer=lightconer,
            diagnostics_dir=case_dir / "routing_warm",
            fallback_to_cpu=False,
            write=False,
            regenerate=True,
        )
        warm_gpu_s = perf_counter() - t_warm0

    products = compare_saved_slices(case_dir / "cpu.h5", case_dir / "gpu.h5")

    scaled_molecular = None
    vcb_check = None
    if molecular:
        for q in ("J_21_LW", "sfrd_mcg"):
            assert np.any(cones["cpu"].lightcones[q] > 0), (
                f"{name}: zero {q} in CPU run"
            )
        scaled_molecular = {}
        for q in ("sfrd_acg", "sfrd_mcg", "xray_emissivity", "J_21_LW"):
            exp = np.asarray(cones["cpu"].lightcones[q], dtype=np.float64)
            got = np.asarray(cones["gpu"].lightcones[q], dtype=np.float64)
            peak = float(np.max(np.abs(exp)))
            delta = np.abs(got - exp)
            scaled_molecular[q] = {
                "peak": peak,
                "max_abs": float(delta.max()),
                "tested": int(exp.size),
                "failed": int((delta > 1e-5 * np.abs(exp) + 1e-10 * peak).sum()),
            }
    if class_streaming == "FLUCTS":
        (cpu_ic_h5,) = (case_dir / "cpu-cache").rglob("InitialConditions.h5")
        (gpu_ic_h5,) = (case_dir / "gpu-cache").rglob("InitialConditions.h5")
        vc = np.asarray(
            read_output_struct(cpu_ic_h5, struct="InitialConditions").get("lowres_vcb"),
            dtype=np.float64,
        )
        vg = np.asarray(
            read_output_struct(gpu_ic_h5, struct="InitialConditions").get("lowres_vcb"),
            dtype=np.float64,
        )
        vcb_check = {
            "cells": int(vc.size),
            "min_km_s": float(vc.min()),
            "max_km_s": float(vc.max()),
            "different": int(np.count_nonzero(vc != vg)),
        }

    catalogs_equal = None
    if track_catalogs:
        catalogs_equal = results["cpu"]["catalogs"] == results["gpu"]["catalogs"]

    gs_path = case_dir / "routing" / "gpu-stages.json"
    br_path = case_dir / "routing" / "backend-routing.json"
    gpu_stages = json.loads(gs_path.read_text()) if gs_path.exists() else {}
    backend_routing = json.loads(br_path.read_text()) if br_path.exists() else None
    counters = {
        **gpu_stages,
        **(backend_routing["counters"] if backend_routing else {}),
    }
    missing_counters = [c for c in required_counters if counters.get(c, 0) < 1]

    ok, msg = _check_all_passed(products, scaled_molecular)
    failure_reasons = []
    if vcb_check is not None and vcb_check["different"] != 0:
        failure_reasons.append(f"lowres_vcb mismatch: {vcb_check}")
    if track_catalogs:
        if not catalogs_equal:
            failure_reasons.append("ordered halo catalogs differ")
        if sum(r["n"] for r in results["cpu"]["catalogs"]) <= 0:
            failure_reasons.append("empty catalogs")
    if missing_counters:
        failure_reasons.append(
            f"missing required GPU dispatch counters {missing_counters}"
        )
    if not ok:
        failure_reasons.append(msg)

    passed = len(failure_reasons) == 0
    diagnostics = None
    if not passed:
        diagnostics = _inspect_failures(case_dir, cones, products, molecular)
        print(f"DIAGNOSTICS {name}: {'; '.join(failure_reasons)}", flush=True)
        for fld, fdiag in diagnostics.get("failing_lightcone_fields", {}).items():
            print(
                f"  field={fld} failed={fdiag['failed']} max_abs={fdiag['max_abs']:.6e} "
                f"max_ratio={fdiag['max_ratio']:.3f} cpu_abs_at_failed={fdiag['cpu_abs_at_failed']}",
                flush=True,
            )
            if "ts_minus_tcmb_abs_at_failed" in fdiag:
                print(
                    f"  |Ts - T_CMB| at failed={fdiag['ts_minus_tcmb_abs_at_failed']}",
                    flush=True,
                )
            for s in fdiag["top_sites"][:8]:
                ts = s["siblings"].get("spin_temperature", {})
                tau = s["siblings"].get("tau_21", {})
                tk = s["siblings"].get("kinetic_temp_neutral", {})
                print(
                    f"    xyz={s['xyz']} z={s['z']:.4f} T_CMB={s['T_CMB']:.4f} "
                    f"Tb_cpu={s['cpu']:.6e} Tb_gpu={s['gpu']:.6e} dTb={s['abs_diff']:.3e} tol={s['tol']:.3e} "
                    f"Ts_cpu={ts.get('cpu')} dTs={ts.get('diff')} tau_cpu={tau.get('cpu')} dtau={tau.get('diff')} "
                    f"dTk={tk.get('diff')}",
                    flush=True,
                )
        for cdiff in diagnostics.get("coeval_diffs", []):
            print(
                f"  coeval {cdiff['relpath']} z={cdiff['redshift']}: {cdiff['fields']}",
                flush=True,
            )

    return {
        "name": name,
        "dim": int(inputs.simulation_options.DIM),
        "hii_dim": int(inputs.simulation_options.HII_DIM),
        "box_len_Mpc": float(inputs.simulation_options.BOX_LEN),
        "threads": int(inputs.simulation_options.N_THREADS),
        "nodes": [float(z) for z in inputs.node_redshifts],
        "cpu": results["cpu"],
        "gpu": results["gpu"],
        "warm_gpu_seconds": warm_gpu_s,
        "speedup_excl_class": (
            results["cpu"]["elapsed_excl_class_seconds"]
            / (
                warm_gpu_s
                if warm_gpu_s is not None
                else results["gpu"]["elapsed_seconds"]
            )
        ),
        "inputs_equal": bool(cones["cpu"].inputs == cones["gpu"].inputs),
        "catalogs_equal": catalogs_equal,
        "vcb_ic_check": vcb_check,
        "lightcones": products["lightcones"],
        "global_quantities": products["global_quantities"],
        "brightness_transverse_power": {
            k: products["brightness_transverse_power"][k]
            for k in ("slices", "tested", "failed", "max_abs")
        },
        "scaled_molecular_fields": scaled_molecular,
        "missing_required_counters": missing_counters,
        "dispatch_counters": counters,
        "passed": passed,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        **({"diagnostics": diagnostics} if diagnostics is not None else {}),
    }


def case1_molecular_class_flucts_lya(work_dir: Path):
    nodes = [20.0, 18.5, 17.1, 15.8, 14.6, 13.5, 12.5, 11.6, 10.8, 10.0]
    q_list = FIELDS + THERMAL_FIELDS + ("J_21_LW", "sfrd_mcg")
    inputs = p21.InputParameters(
        random_seed=129887, node_redshifts=nodes
    ).evolve_input_structs(
        SOURCE_MODEL="E-INTEGRAL",
        HMF="ST",
        DIM=192,
        HII_DIM=64,
        BOX_LEN=200.0,
        N_THREADS=2,
        PERTURB_ON_HIGH_RES=False,
        KEEP_3D_VELOCITIES=False,
        SMOOTH_EVOLVED_DENSITY_FIELD=True,
        POWER_SPECTRUM="CLASS",
        V_CB_MODEL="FLUCTS",
        USE_MCGS=True,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=True,
        USE_TS_FLUCT=True,
        USE_X_RAY_HEATING=True,
        USE_LYA_HEATING=True,
        LYA_MULTIPLE_SCATTERING=True,
        PHOTON_CONS_TYPE="no-photoncons",
        RECOMB_MODEL="inhomogeneous",
        USE_METALLICITY=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        INTEGRATION_METHOD_ACGS="GAUSS-LEGENDRE",
        INTEGRATION_METHOD_MCGS="GAUSS-LEGENDRE",
        M_TURN_STELLAR_FEEDBACK=6.0,
        R_BUBBLE_MAX=50.0,
        Z_HEAT_MAX=20.0,
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=10.2,
        max_redshift=18.2,
        quantities=q_list,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    return _run_pair(
        "case1_e_integral_class_flucts_mcgs_lya_192",
        work_dir,
        inputs,
        lc,
        class_streaming="FLUCTS",
        molecular=True,
        required_counters=(
            "gpu_seed_and_modes",
            "mcg_xray_optical_depth_and_frequency",
            "radiation_lya_scattering_filter",
            "eulerian_mcg_emissivity_cells",
            "eulerian_mcg_ion_ion_center",
        ),
    )


def case2_bbks_gaussian_192(work_dir: Path):
    from tests import produce_integration_test_data as official

    with patch.object(official, "SEED", 424242):
        inputs = official.get_all_options_struct(
            6,
            lc=True,
            DIM=192,
            HII_DIM=64,
            BOX_LEN=200.0,
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )["inputs"].evolve_input_structs(
            POWER_SPECTRUM="BBKS",
            FILTER="gaussian",
            N_THREADS=2,
        )
    q_list = (
        "brightness_temp",
        "neutral_fraction",
        "xray_ionised_fraction",
        "kinetic_temp_neutral",
        "spin_temperature",
        "cumulative_recombinations",
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=6.0,
        max_redshift=15.0,
        quantities=q_list,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    return _run_pair(
        "case2_e_integral_bbks_gaussian_192",
        work_dir,
        inputs,
        lc,
        required_counters=("gpu_seed_and_modes", "sigma_moments", "ion_center", "heat"),
    )


def _case3_chmf_thermal_96(work_dir: Path, *, threads: int, name: str):
    nodes = [11.5, 11.0, 10.5, 10.0, 9.5]
    q_list = FIELDS + THERMAL_FIELDS
    inputs = p21.InputParameters(
        random_seed=81033, node_redshifts=nodes
    ).evolve_input_structs(
        SOURCE_MODEL="CHMF-SAMPLER",
        USE_UPPER_STELLAR_TURNOVER=True,
        DIM=96,
        HII_DIM=32,
        BOX_LEN=84.0,
        N_THREADS=threads,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        HII_FILTER="spherical-tophat",
        USE_EXP_FILTER=True,
        CELL_RECOMB=True,
        USE_METALLICITY=True,
        Z_HEAT_MAX=11.5,
    )
    lc = RectilinearLightconer.between_redshifts(
        min_redshift=9.6,
        max_redshift=11.3,
        quantities=q_list,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    required = (
        "halo_cuda_mt_words",
        "halo_cuda_progenitor_conditions",
        "ionization_halo_barrier",
        "thermal_update",
    )
    if threads > 1:
        required += ("halo_cuda_gfsr_words",)
    return _run_pair(
        name,
        work_dir,
        inputs,
        lc,
        track_catalogs=True,
        required_counters=required,
    )


def case3_chmf_thermal_96(work_dir: Path):
    return _case3_chmf_thermal_96(work_dir, threads=2, name="case3_chmf_thermal_96")


def case3_chmf_thermal_96_1t(work_dir: Path):
    return _case3_chmf_thermal_96(work_dir, threads=1, name="case3_chmf_thermal_96_1t")


def main():
    case_map = {
        "case1": case1_molecular_class_flucts_lya,
        "case2": case2_bbks_gaussian_192,
        "case3": case3_chmf_thermal_96,
        "case3_1t": case3_chmf_thermal_96_1t,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default="case1,case2,case3,case3_1t",
        help="Comma-separated subset of case1,case2,case3,case3_1t",
    )
    args = parser.parse_args()

    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in selected if c not in case_map]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}; valid: {list(case_map)}")

    out = {
        "scope": "Paired large-box 21cmFAST 4.3.2 CUDA vs CPU acceptance for molecular cooling + CLASS FLUCTS + Lya multiple scattering, BBKS + Gaussian filter, and 96^3 thermal CHMF-SAMPLER.",
        "host": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "tolerance": "abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)",
        "cases": [],
    }
    for key in selected:
        res = case_map[key](args.work_dir)
        out["cases"].append(res)
        status = "PASSED" if res["passed"] else "FAILED"
        print(
            f"{status} {res['name']}: cpu={res['cpu']['elapsed_seconds']:.2f}s "
            f"gpu={res['gpu']['elapsed_seconds']:.2f}s warm_gpu={res['warm_gpu_seconds']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if any(not c["passed"] for c in out["cases"]):
        failed_names = [c["name"] for c in out["cases"] if not c["passed"]]
        raise SystemExit(f"Verification failed for cases: {failed_names}")


if __name__ == "__main__":
    main()
