"""Unified parity and option-coverage suite for the 21cmFAST 4.3.2 GPU adapter."""

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import traceback
from unittest.mock import patch

from astropy import units
import numpy as np
import py21cmfast as p21
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_lightcone as gl  # noqa: E402

ATOL_EXACT = 1e-8
ATOL_GENERAL = 1e-5
RTOL = 1e-4


@contextmanager
def native_partition_reference():
    """Patch HaloCatalog CPU reference with repaired ST PARTITION rejection storage."""
    from ctypes import c_void_p, cast
    from tempfile import TemporaryDirectory
    from native_shim import load_shim, repair_partition_rejection
    from py21cmfast.c_21cmfast import ffi, lib

    src = Path(p21.__file__).parent / "src"
    native = Path(p21.c_21cmfast.__file__)
    with TemporaryDirectory(prefix="v432-partition-ref-") as temp:
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


def make_inputs(**overrides):
    """Build 4.3.2 InputParameters matching the small-box verification baseline."""
    cosmo_kw = overrides.pop("cosmo", {})
    sim_kw = dict(HII_DIM=16, BOX_LEN=48.0, HIRES_TO_LOWRES_FACTOR=2, N_THREADS=2)
    sim_kw.update(overrides.pop("sim", {}))
    matter_kw = dict(
        HMF="ST",
        V_CB_MODEL="NONE",
        POWER_SPECTRUM="EH",
        PERTURB_ON_HIGH_RES=False,
        USE_INTERPOLATION_TABLES="hmf-interpolation",
        MINIMIZE_MEMORY=False,
        KEEP_3D_VELOCITIES=False,
        SAMPLE_METHOD="MASS-LIMITED",
        FILTER="spherical-tophat",
        HALO_FILTER="spherical-tophat",
        SMOOTH_EVOLVED_DENSITY_FIELD=False,
        DEXM_OPTIMIZE=False,
        PERTURB_ALGORITHM="2LPT",
        USE_FFTW_WISDOM=False,
        SOURCE_MODEL="E-INTEGRAL",
    )
    matter_kw.update(overrides.pop("matter", {}))
    astro_kw = dict(
        USE_MCGS=False,
        USE_X_RAY_HEATING=True,
        USE_CMB_HEATING=True,
        USE_LYA_HEATING=True,
        USE_TS_FLUCT=True,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
        LYA_MULTIPLE_SCATTERING=False,
        USE_ADIABATIC_FLUCTUATIONS=True,
        PHOTON_CONS_TYPE="no-photoncons",
        USE_METALLICITY=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        HII_FILTER="spherical-tophat",
        HEAT_FILTER="spherical-tophat",
        IONISE_ENTIRE_SPHERE=False,
        RECOMB_MODEL="inhomogeneous",
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        INTEGRATION_METHOD_ACGS="GAUSS-LEGENDRE",
        INTEGRATION_METHOD_MCGS="GAUSS-LEGENDRE",
    )
    astro_kw.update(overrides.pop("astro", {}))
    astro_params_kw = dict(R_BUBBLE_MAX=15.0)
    astro_params_kw.update(overrides.pop("astro_params", {}))
    seed = overrides.pop("random_seed", 42)
    needs_evol = astro_kw["USE_TS_FLUCT"] or astro_kw["RECOMB_MODEL"] != "none"
    z_max = 36.0 if needs_evol else 12.0
    node_z = overrides.pop(
        "node_redshifts",
        p21.get_logspaced_redshifts(
            min_redshift=8.0, max_redshift=z_max, z_step_factor=1.22
        ),
    )
    return p21.InputParameters(
        random_seed=seed,
        node_redshifts=node_z,
        cosmo_params=p21.CosmoParams(**cosmo_kw),
        simulation_options=p21.SimulationOptions(**sim_kw),
        matter_options=p21.MatterOptions(**matter_kw),
        astro_options=p21.AstroOptions(**astro_kw),
        astro_params=p21.AstroParams(**astro_params_kw),
    )


def is_general_fft(inputs):
    """Return True when 3D transforms route through cuFFT rather than exact FFTW plans."""
    sim = inputs.simulation_options
    return bool(
        sim.NON_CUBIC_FACTOR != 1
        or sim.DIM % 2
        or sim.HII_DIM % 2
        or inputs.matter_options.USE_FFTW_WISDOM
    )


def compare_arrays(a, b, label, *, atol, rtol, allow_mcg_cutoff=False):
    """Compare CPU and GPU arrays with exact or tiered float32 tolerances."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise AssertionError(
            f"{label}: shape mismatch CPU {a_arr.shape} vs GPU {b_arr.shape}"
        )
    if a_arr.dtype.kind == "b" or np.issubdtype(a_arr.dtype, np.integer):
        if not np.array_equal(a_arr, b_arr):
            raise AssertionError(f"{label}: exact mismatch")
        return 0.0, 0.0, 0
    if not (np.isfinite(a_arr).all() and np.isfinite(b_arr).all()):
        raise AssertionError(f"{label}: non-finite values")
    af = a_arr.astype(np.float64)
    bf = b_arr.astype(np.float64)
    diff = np.abs(af - bf)
    bad = diff > (atol + rtol * np.abs(af))
    n_bad = int(np.count_nonzero(bad))
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(af), 1e-30)
    mask = diff > atol
    max_rel = float((diff[mask] / denom[mask]).max()) if np.any(mask) else 0.0
    if n_bad:
        if allow_mcg_cutoff and label.endswith(
            "ionized_box.nion_conditional_filtered_mcg"
        ):
            pct = 100.0 * n_bad / max(1, af.size)
            if n_bad * 1000 <= af.size:
                return max_abs, max_rel, n_bad
            raise AssertionError(
                f"{label}: {n_bad}/{af.size} ({pct:.3f}%) > 0.1% cutoff allowance"
            )
        raise AssertionError(
            f"{label}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e} "
            f"({n_bad}/{af.size}) exceeds atol={atol}, rtol={rtol}"
        )
    return max_abs, max_rel, 0


def extract_coeval_arrays(items):
    """Collect computed Coeval output arrays across evaluated redshifts."""
    out = {}
    for idx, item in enumerate(items):
        coeval = item[0] if isinstance(item, tuple) else item
        for sname, struct in coeval.output_structs.items():
            if struct is None:
                continue
            for aname, arr in struct.arrays.items():
                if arr.state.computed_in_mem:
                    out[f"z{idx}.{sname}.{aname}"] = np.array(struct.get(aname))
    return out


def extract_lc_arrays(lc):
    """Collect lightcone volume slices and global evolution series."""
    out = {f"lc.{k}": np.asarray(v) for k, v in lc.lightcones.items()}
    out.update({f"global.{k}": np.asarray(v) for k, v in lc.global_quantities.items()})
    return out


def check_diag(diag_dir):
    """Verify non-zero GPU stage counters and zero CPU backend fallback."""
    diag = Path(diag_dir)
    routing_path = diag / "backend-routing.json"
    if routing_path.exists():
        routing = json.loads(routing_path.read_text())
        if routing.get("backend") == "cpu":
            raise AssertionError(f"Unexpected CPU fallback: {routing}")
    files = {}
    total_counts = 0
    for path in sorted(diag.glob("*.json")):
        data = json.loads(path.read_text())
        files[path.name] = data
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    total_counts += int(val > 0)
                elif isinstance(val, dict):
                    for nested in val.values():
                        if isinstance(nested, (int, float)) and not isinstance(
                            nested, bool
                        ):
                            total_counts += int(nested > 0)
    if total_counts <= 0:
        raise AssertionError(f"Expected non-zero GPU stage counters in {files}")
    return files


def _build_lightconer(inputs, *, los_axis=-1, angular=False, min_z=8.0, max_z=12.0):
    if angular:
        mu, _ = np.polynomial.legendre.leggauss(4)
        lon = np.arange(8) * (2 * np.pi / 8)
        return p21.AngularLightconer.between_redshifts(
            min_redshift=min_z,
            max_redshift=max_z,
            latitude=np.repeat(np.arcsin(mu), 8),
            longitude=np.tile(lon, 4),
            origin=np.array([-2.25, 1.5, 4.1]) * units.pixel,
            rotation=Rotation.from_euler("zyx", [-0.2, 0.15, 0.3]),
            quantities=(
                "brightness_temp",
                "neutral_fraction",
                "z_reion",
                "kinetic_temperature",
                "density",
            ),
            resolution=inputs.simulation_options.cell_size,
            cosmo=inputs.cosmo_params.cosmo,
        )
    return p21.RectilinearLightconer.between_redshifts(
        min_redshift=min_z,
        max_redshift=max_z,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        line_of_sight_axis=los_axis,
    )


def run_parity_case(
    suite,
    name,
    kind,
    inputs,
    *,
    extra_emissivity=False,
    los_axis=-1,
    angular=False,
    expect_error=False,
    cpu_partition_shim=False,
    omit_out_redshifts=False,
    wisdom_tmp=False,
    lc_min_z=8.0,
    lc_max_z=12.0,
):
    """Run one CPU-vs-GPU parity test under strict or general-FFT tolerances."""
    t0 = perf_counter()
    general = is_general_fft(inputs)
    atol = ATOL_GENERAL if general else ATOL_EXACT
    rtol = RTOL
    allow_mcg = general and inputs.astro_options.USE_MCGS
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        diag_dir = tmp_path / "diag"
        cpu_cache = p21.OutputCache(tmp_path / "cpu_cache")
        gpu_cache = p21.OutputCache(tmp_path / "gpu_cache")
        wdir = tmp_path / "wisdoms"
        wdir.mkdir(exist_ok=True)
        cfg_kw = {"EXTRA_EMISSIVITY_FIELDS": extra_emissivity}
        if wisdom_tmp:
            cfg_kw["wisdoms_path"] = str(wdir)
        try:
            with p21.config.use(**cfg_kw):
                if kind in ("run_coeval", "generate_coeval"):
                    out_kw = (
                        {}
                        if omit_out_redshifts
                        else {
                            "out_redshifts": tuple(
                                float(z) for z in inputs.node_redshifts[-2:]
                            )
                        }
                    )
                    cpu_fn = (
                        p21.run_coeval
                        if kind == "run_coeval"
                        else lambda **kw: list(p21.generate_coeval(**kw))
                    )
                    gpu_fn = (
                        gl.run_coeval
                        if kind == "run_coeval"
                        else lambda **kw: list(gl.generate_coeval(**kw))
                    )
                    cpu_err = gpu_err = None
                    cpu_res = gpu_res = None
                    try:
                        with ExitStack() as cstack:
                            if cpu_partition_shim:
                                cstack.enter_context(native_partition_reference())
                            cpu_res = cpu_fn(
                                inputs=inputs,
                                cache=cpu_cache,
                                write=True,
                                regenerate=True,
                                **out_kw,
                            )
                    except Exception as exc:
                        cpu_err = f"{type(exc).__name__}: {exc}"
                    try:
                        gpu_res = gpu_fn(
                            inputs=inputs,
                            cache=gpu_cache,
                            write=True,
                            regenerate=True,
                            fallback_to_cpu=False,
                            diagnostics_dir=diag_dir,
                            **out_kw,
                        )
                    except Exception as exc:
                        gpu_err = f"{type(exc).__name__}: {exc}"
                    if expect_error:
                        if cpu_err and gpu_err:
                            return {
                                "suite": suite,
                                "name": name,
                                "status": "PASS",
                                "note": f"matched error CPU={cpu_err[:72]} | GPU={gpu_err[:72]}",
                                "elapsed": round(perf_counter() - t0, 2),
                            }
                        raise AssertionError(
                            f"Error parity mismatch: cpu_err={cpu_err}, gpu_err={gpu_err}"
                        )
                    if cpu_err or gpu_err:
                        raise AssertionError(f"cpu_err={cpu_err}; gpu_err={gpu_err}")
                    cpu_map = extract_coeval_arrays(cpu_res)
                    gpu_map = extract_coeval_arrays(gpu_res)
                else:
                    lc_obj = _build_lightconer(
                        inputs,
                        los_axis=los_axis,
                        angular=angular,
                        min_z=lc_min_z,
                        max_z=lc_max_z,
                    )
                    cpu_fn = (
                        p21.run_lightcone
                        if kind == "run_lightcone"
                        else lambda **kw: list(p21.generate_lightcone(**kw))[-1][-1]
                    )
                    gpu_fn = (
                        gl.run_lightcone
                        if kind == "run_lightcone"
                        else lambda **kw: list(gl.generate_lightcone(**kw))[-1][-1]
                    )
                    cpu_res = cpu_fn(
                        inputs=inputs,
                        lightconer=lc_obj,
                        cache=cpu_cache,
                        write=True,
                        regenerate=True,
                    )
                    gpu_res = gpu_fn(
                        inputs=inputs,
                        lightconer=lc_obj,
                        cache=gpu_cache,
                        write=True,
                        regenerate=True,
                        fallback_to_cpu=False,
                        diagnostics_dir=diag_dir,
                    )
                    cpu_map = extract_lc_arrays(cpu_res)
                    gpu_map = extract_lc_arrays(gpu_res)
            if set(cpu_map) != set(gpu_map):
                raise AssertionError(f"Key mismatch: {set(cpu_map) ^ set(gpu_map)}")
            worst_abs = worst_rel = 0.0
            mcg_flips = {}
            for key in sorted(cpu_map):
                ma, mr, flips = compare_arrays(
                    cpu_map[key],
                    gpu_map[key],
                    key,
                    atol=atol,
                    rtol=rtol,
                    allow_mcg_cutoff=allow_mcg,
                )
                worst_abs = max(worst_abs, ma)
                worst_rel = max(worst_rel, mr)
                if flips:
                    mcg_flips[key] = flips
            diag_data = check_diag(diag_dir)
            out = {
                "suite": suite,
                "name": name,
                "status": "PASS",
                "atol": atol,
                "rtol": rtol,
                "n_fields": len(cpu_map),
                "max_abs": worst_abs,
                "max_rel": worst_rel,
                "diag_files": sorted(diag_data),
                "elapsed": round(perf_counter() - t0, 2),
            }
            if mcg_flips:
                out["mcg_cutoff_flips"] = mcg_flips
            return out
        except Exception as exc:
            return {
                "suite": suite,
                "name": name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=4),
                "elapsed": round(perf_counter() - t0, 2),
            }


def run_mcg_only_e_gpu(suite, name):
    """Verify GPU-only decoupled USE_MCGS=True without photoheating on E-INTEGRAL."""
    t0 = perf_counter()
    inp = make_inputs(
        sim={"Z_HEAT_MAX": 15.0},
        matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
        astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False},
        astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
        node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            diag = Path(tmp) / "diag"
            res = gl.run_coeval(
                inputs=inp,
                cache=p21.OutputCache(Path(tmp) / "gpu_cache"),
                write=True,
                regenerate=True,
                fallback_to_cpu=False,
                diagnostics_dir=diag,
            )
            assert len(res) == 5
            for coeval in res:
                nf = coeval.ionized_box.get("neutral_fraction")
                assert np.isfinite(nf).all() and np.all((nf >= 0.0) & (nf <= 1.0))
            diag_data = check_diag(diag)
            return {
                "suite": suite,
                "name": name,
                "status": "PASS",
                "note": "GPU-only validation (upstream 4.3.2 C overwrites single 1D ACG table)",
                "diag_files": sorted(diag_data),
                "elapsed": round(perf_counter() - t0, 2),
            }
    except Exception as exc:
        return {
            "suite": suite,
            "name": name,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=4),
            "elapsed": round(perf_counter() - t0, 2),
        }


def build_suite_a():
    """Public coeval/lightcone APIs, LOS axes, curvature, memory, sources, HMF, tables."""
    e_therm = make_inputs()
    cases = [
        ("run_coeval_e_integral", "run_coeval", e_therm, {}),
        (
            "run_coeval_omit_out_redshifts",
            "run_coeval",
            make_inputs(astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none"}),
            {"omit_out_redshifts": True},
        ),
        ("generate_coeval_e_integral", "generate_coeval", e_therm, {}),
        ("los_axis_0_run", "run_lightcone", e_therm, {"los_axis": 0}),
        ("los_axis_1_gen", "generate_lightcone", e_therm, {"los_axis": 1}),
        ("los_axis_2_run", "run_lightcone", e_therm, {"los_axis": 2}),
        (
            "curv_minmem_e_integral",
            "run_lightcone",
            make_inputs(cosmo={"OMtot": 0.996}, matter={"MINIMIZE_MEMORY": True}),
            {},
        ),
        (
            "curv_minmem_l_integral",
            "run_lightcone",
            make_inputs(
                cosmo={"OMtot": 0.996},
                matter={"SOURCE_MODEL": "L-INTEGRAL", "MINIMIZE_MEMORY": True},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True},
            ),
            {},
        ),
        ("extra_e_integral_thermal", "run_coeval", e_therm, {"extra_emissivity": True}),
        (
            "extra_e_integral_nonthermal",
            "run_coeval",
            make_inputs(astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none"}),
            {"extra_emissivity": True},
        ),
        (
            "extra_e_integral_mcg",
            "run_coeval",
            make_inputs(
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
            ),
            {"extra_emissivity": True},
        ),
        (
            "extra_const_ion_thermal",
            "run_coeval",
            make_inputs(matter={"SOURCE_MODEL": "CONST-ION-EFF"}),
            {"extra_emissivity": True},
        ),
        (
            "extra_const_ion_nonthermal",
            "run_coeval",
            make_inputs(
                matter={"SOURCE_MODEL": "CONST-ION-EFF"},
                astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none"},
            ),
            {"extra_emissivity": True},
        ),
        (
            "extra_l_integral_mcg",
            "run_coeval",
            make_inputs(
                matter={
                    "SOURCE_MODEL": "L-INTEGRAL",
                    "POWER_SPECTRUM": "EH",
                    "V_CB_MODEL": "AVG-DEBUG",
                },
                astro={
                    "USE_MCGS": True,
                    "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True,
                    "USE_EXP_FILTER": True,
                    "CELL_RECOMB": True,
                    "USE_METALLICITY": True,
                },
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
            ),
            {"extra_emissivity": True},
        ),
        (
            "extra_chmf_mcg",
            "run_coeval",
            make_inputs(
                sim={"HII_DIM": 8, "BOX_LEN": 48.0, "HIRES_TO_LOWRES_FACTOR": 2, "N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={
                    "SOURCE_MODEL": "CHMF-SAMPLER",
                    "SAMPLE_METHOD": "BINARY-SPLIT",
                    "POWER_SPECTRUM": "EH",
                    "V_CB_MODEL": "AVG-DEBUG",
                },
                astro={
                    "USE_MCGS": True,
                    "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True,
                    "USE_EXP_FILTER": True,
                    "CELL_RECOMB": True,
                    "USE_METALLICITY": True,
                    "USE_UPPER_STELLAR_TURNOVER": True,
                },
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {"extra_emissivity": True},
        ),
    ]
    for hmf in ("PS", "ST", "WATSON", "WATSON-Z", "DELOS", "REED07", "YUNG24"):
        inp = (
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"HMF": hmf},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            )
            if hmf == "YUNG24"
            else make_inputs(matter={"HMF": hmf})
        )
        cases.append((f"hmf_e_integral_{hmf}", "run_coeval", inp, {}))
    cases.extend(
        [
            (
                "adiabatic_false_e",
                "run_coeval",
                make_inputs(astro={"USE_ADIABATIC_FLUCTUATIONS": False}),
                {},
            ),
            (
                "halo_filter_sharpk_dexm",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                    matter={"SOURCE_MODEL": "DEXM-ESF", "HALO_FILTER": "sharp-k"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                    node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
                ),
                {},
            ),
            (
                "halo_filter_gaussian_dexm",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                    matter={"SOURCE_MODEL": "DEXM-ESF", "HALO_FILTER": "gaussian"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                    node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
                ),
                {},
            ),
            (
                "dexm_optimize_true",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0, "DEXM_OPTIMIZE_MINMASS": 1e11},
                    matter={"SOURCE_MODEL": "DEXM-ESF", "DEXM_OPTIMIZE": True},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                    node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
                ),
                {},
            ),
            (
                "chmf_mass_limited",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                    matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "MASS-LIMITED"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                    node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
                ),
                {},
            ),
            (
                "chmf_number_limited",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                    matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "NUMBER-LIMITED"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                    node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
                ),
                {},
            ),
            (
                "chmf_ps_binary_split",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1},
                    matter={"SOURCE_MODEL": "CHMF-SAMPLER", "HMF": "PS", "SAMPLE_METHOD": "BINARY-SPLIT"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                ),
                {},
            ),
            (
                "chmf_ps_partition",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1},
                    matter={"SOURCE_MODEL": "CHMF-SAMPLER", "HMF": "PS", "SAMPLE_METHOD": "PARTITION"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                ),
                {},
            ),
            (
                "chmf_st_partition",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1},
                    matter={"SOURCE_MODEL": "CHMF-SAMPLER", "HMF": "ST", "SAMPLE_METHOD": "PARTITION"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                ),
                {"cpu_partition_shim": True},
            ),
            (
                "gamma_e_nonthermal",
                "run_coeval",
                make_inputs(astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "INTEGRATION_METHOD_ACGS": "GAMMA-APPROX"}),
                {},
            ),
            (
                "gamma_l_nonthermal",
                "run_coeval",
                make_inputs(
                    sim={"N_THREADS": 1},
                    matter={"SOURCE_MODEL": "L-INTEGRAL"},
                    astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "INTEGRATION_METHOD_ACGS": "GAMMA-APPROX"},
                ),
                {},
            ),
            (
                "gamma_e_thermal",
                "run_coeval",
                make_inputs(astro={"INTEGRATION_METHOD_ACGS": "GAMMA-APPROX"}),
                {},
            ),
            (
                "gamma_e_mcg_mcgs_only",
                "run_coeval",
                make_inputs(
                    matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                    astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "INTEGRATION_METHOD_ACGS": "GSL-QAG", "INTEGRATION_METHOD_MCGS": "GAMMA-APPROX"},
                    astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                ),
                {},
            ),
            (
                "interp_sigma_e_thermal",
                "run_coeval",
                make_inputs(matter={"USE_INTERPOLATION_TABLES": "sigma-interpolation"}, astro={"INTEGRATION_METHOD_ACGS": "GSL-QAG"}),
                {},
            ),
            (
                "interp_sigma_e_mcg",
                "run_coeval",
                make_inputs(
                    matter={"USE_INTERPOLATION_TABLES": "sigma-interpolation", "POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                    astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "INTEGRATION_METHOD_ACGS": "GSL-QAG", "INTEGRATION_METHOD_MCGS": "GSL-QAG"},
                    astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                ),
                {},
            ),
            (
                "interp_sigma_l_thermal",
                "run_coeval",
                make_inputs(
                    matter={"SOURCE_MODEL": "L-INTEGRAL", "USE_INTERPOLATION_TABLES": "sigma-interpolation"},
                    astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "INTEGRATION_METHOD_ACGS": "GSL-QAG"},
                ),
                {},
            ),
            (
                "interp_none_e_nonthermal",
                "run_coeval",
                make_inputs(
                    sim={"HII_DIM": 4, "BOX_LEN": 24.0, "HIRES_TO_LOWRES_FACTOR": 2, "N_THREADS": 2},
                    matter={"USE_INTERPOLATION_TABLES": "no-interpolation"},
                    astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "INTEGRATION_METHOD_ACGS": "GSL-QAG"},
                    astro_params={"R_BUBBLE_MAX": 12.0},
                    node_redshifts=[8.0],
                ),
                {},
            ),
            (
                "interp_none_l_nonthermal",
                "run_coeval",
                make_inputs(
                    sim={"HII_DIM": 4, "BOX_LEN": 24.0, "HIRES_TO_LOWRES_FACTOR": 2, "N_THREADS": 1},
                    matter={"SOURCE_MODEL": "L-INTEGRAL", "USE_INTERPOLATION_TABLES": "no-interpolation"},
                    astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "INTEGRATION_METHOD_ACGS": "GSL-QAG"},
                    astro_params={"R_BUBBLE_MAX": 12.0},
                    node_redshifts=[8.0],
                ),
                {},
            ),
        ]
    )
    return [("A", name, kind, inp, kw) for name, kind, inp, kw in cases]


def build_suite_b():
    """Decoupled feedback, heating, recombination, filters, spheres, turnover, V_CB."""
    cases = [
        ("photo_only_e_thermal", "run_coeval", make_inputs(astro={"USE_MCGS": False, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True}), {}),
        ("photo_only_e_nonthermal", "run_coeval", make_inputs(astro={"USE_TS_FLUCT": False, "USE_MCGS": False, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True}), {}),
        ("photo_only_l", "run_coeval", make_inputs(matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_MCGS": False, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        (
            "photo_only_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"USE_MCGS": False, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "mcg_only_l", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "L-INTEGRAL", "POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False, "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "mcg_only_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT", "POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": False, "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        ("sphere_e_nonthermal", "run_coeval", make_inputs(sim={"N_THREADS": 1}, astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "IONISE_ENTIRE_SPHERE": True}), {}),
        ("sphere_e_inhomo", "run_coeval", make_inputs(sim={"N_THREADS": 1}, astro={"RECOMB_MODEL": "inhomogeneous", "IONISE_ENTIRE_SPHERE": True}, astro_params={"R_BUBBLE_MIN": 6.0, "R_BUBBLE_MAX": 15.0}), {}),
        ("sphere_l_inhomo", "run_coeval", make_inputs(sim={"N_THREADS": 1}, matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "IONISE_ENTIRE_SPHERE": True}), {}),
        (
            "sphere_chmf_inhomo", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True, "IONISE_ENTIRE_SPHERE": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        ("no_xray_e", "run_coeval", make_inputs(astro={"USE_X_RAY_HEATING": False, "USE_LYA_HEATING": True}), {}),
        ("no_lya_e", "run_coeval", make_inputs(astro={"USE_X_RAY_HEATING": True, "USE_LYA_HEATING": False}), {}),
        ("no_xray_no_lya_no_cmb_e", "run_coeval", make_inputs(astro={"USE_X_RAY_HEATING": False, "USE_LYA_HEATING": False, "USE_CMB_HEATING": False}), {}),
        (
            "no_xray_no_lya_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True, "USE_X_RAY_HEATING": False, "USE_LYA_HEATING": False, "USE_CMB_HEATING": False},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "no_xray_no_lya_e_mcg", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "USE_X_RAY_HEATING": False, "USE_LYA_HEATING": False},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        ("metal_true_e", "run_coeval", make_inputs(astro={"USE_METALLICITY": True}), {}),
        ("metal_false_l", "run_coeval", make_inputs(matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": False}), {}),
        (
            "metal_false_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": False, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        ("homog_recomb_l", "run_coeval", make_inputs(matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"RECOMB_MODEL": "homogeneous", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        (
            "homog_recomb_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"RECOMB_MODEL": "homogeneous", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "homog_recomb_dexm", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "DEXM-ESF"},
                astro={"RECOMB_MODEL": "homogeneous", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "homog_recomb_e_mcg", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "RECOMB_MODEL": "homogeneous", "CELL_RECOMB": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        ("cell_recomb_false_e", "run_coeval", make_inputs(astro={"RECOMB_MODEL": "inhomogeneous", "CELL_RECOMB": False}), {}),
        ("cell_recomb_false_l_sharpk", "run_coeval", make_inputs(matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"RECOMB_MODEL": "inhomogeneous", "USE_EXP_FILTER": False, "CELL_RECOMB": False, "HII_FILTER": "sharp-k", "USE_METALLICITY": True}), {}),
        (
            "cell_recomb_false_chmf_gaussian", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"RECOMB_MODEL": "inhomogeneous", "USE_EXP_FILTER": False, "CELL_RECOMB": False, "HII_FILTER": "gaussian", "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "dexm_mcg", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "DEXM-ESF", "POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "upper_turn_false_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": False},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "upper_turn_false_dexm", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "DEXM-ESF"},
                astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": False},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "vcb_none_eh_mcg", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "NONE"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
        (
            "vcb_avg_auto_eh_mcg", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-AUTO"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0],
            ),
            {},
        ),
    ]
    return [("B", name, kind, inp, kw) for name, kind, inp, kw in cases]


def build_suite_c():
    """Auto-calibrated PHOTON_CONS_TYPE (z, alpha, f) across models and lightcones."""
    pc_z = list(
        p21.get_logspaced_redshifts(min_redshift=6.5, max_redshift=36.0, z_step_factor=1.25)
    )
    cases = [
        ("auto_pc_z_e", "run_coeval", make_inputs(astro={"PHOTON_CONS_TYPE": "z-photoncons"}, node_redshifts=pc_z), {}),
        ("auto_pc_z_e_box100", "run_coeval", make_inputs(sim={"BOX_LEN": 100.0}, astro={"PHOTON_CONS_TYPE": "z-photoncons"}, astro_params={"R_BUBBLE_MAX": 50.0}, node_redshifts=pc_z), {}),
        ("auto_pc_z_const_ion_eff", "run_coeval", make_inputs(sim={"BOX_LEN": 100.0}, matter={"SOURCE_MODEL": "CONST-ION-EFF"}, astro={"USE_TS_FLUCT": False, "RECOMB_MODEL": "none", "PHOTON_CONS_TYPE": "z-photoncons"}, astro_params={"R_BUBBLE_MAX": 50.0}, node_redshifts=pc_z), {"expect_error": True}),
        ("auto_pc_alpha_e", "run_coeval", make_inputs(astro={"PHOTON_CONS_TYPE": "alpha-photoncons"}, node_redshifts=pc_z), {}),
        (
            "auto_pc_alpha_chmf", "run_coeval",
            make_inputs(
                sim={"N_THREADS": 1, "Z_HEAT_MAX": 15.0},
                matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"},
                astro={"PHOTON_CONS_TYPE": "alpha-photoncons", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True},
                node_redshifts=[18.0, 15.0, 12.0, 9.0, 6.5],
            ),
            {"expect_error": True},
        ),
        ("auto_pc_f_e", "run_coeval", make_inputs(astro={"PHOTON_CONS_TYPE": "f-photoncons"}, node_redshifts=pc_z), {}),
        ("auto_pc_f_l", "run_coeval", make_inputs(matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"PHOTON_CONS_TYPE": "f-photoncons", "USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}, node_redshifts=pc_z), {"expect_error": True}),
        (
            "auto_pc_f_e_mcg", "run_coeval",
            make_inputs(
                sim={"Z_HEAT_MAX": 15.0},
                matter={"POWER_SPECTRUM": "EH", "V_CB_MODEL": "AVG-DEBUG"},
                astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True, "PHOTON_CONS_TYPE": "f-photoncons"},
                astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0},
                node_redshifts=[18.0, 15.0, 12.0, 9.0, 6.5],
            ),
            {},
        ),
        ("auto_pc_f_e_lightcone", "run_lightcone", make_inputs(astro={"PHOTON_CONS_TYPE": "f-photoncons"}, node_redshifts=pc_z), {"lc_min_z": 6.5, "lc_max_z": 12.0}),
    ]
    return [("C", name, kind, inp, kw) for name, kind, inp, kw in cases]


def build_suite_d():
    """Non-cubic, odd dimensions, odd+non-cubic, and USE_FFTW_WISDOM=True grids."""
    ncf15 = {"DIM": 24, "HII_DIM": 8, "HIRES_TO_LOWRES_FACTOR": None, "NON_CUBIC_FACTOR": 1.5, "BOX_LEN": 24.0}
    ncf20 = {"DIM": 16, "HII_DIM": 8, "HIRES_TO_LOWRES_FACTOR": None, "NON_CUBIC_FACTOR": 2.0, "BOX_LEN": 24.0}
    odd = {"DIM": 15, "HII_DIM": 5, "HIRES_TO_LOWRES_FACTOR": None, "NON_CUBIC_FACTOR": 1.0, "BOX_LEN": 15.0}
    odd_nc = {"DIM": 15, "HII_DIM": 5, "HIRES_TO_LOWRES_FACTOR": None, "NON_CUBIC_FACTOR": 1.4, "BOX_LEN": 15.0}
    wis = {"DIM": 16, "HII_DIM": 8, "HIRES_TO_LOWRES_FACTOR": None, "NON_CUBIC_FACTOR": 1.0, "BOX_LEN": 24.0}
    cases = [
        ("ncf15_e_integral", "run_coeval", make_inputs(sim=ncf15), {}),
        ("ncf15_e_mcg", "run_coeval", make_inputs(sim={**ncf15, "Z_HEAT_MAX": 15.0}, matter={"V_CB_MODEL": "AVG-DEBUG"}, astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True}, astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("ncf15_e_sphere", "run_coeval", make_inputs(sim=ncf15, astro={"IONISE_ENTIRE_SPHERE": True}), {}),
        ("ncf15_l_integral", "run_coeval", make_inputs(sim=ncf15, matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        ("ncf15_chmf", "run_coeval", make_inputs(sim={**ncf15, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("ncf15_dexm", "run_coeval", make_inputs(sim={**ncf15, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"SOURCE_MODEL": "DEXM-ESF"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("ncf15_e_rect_lc", "run_lightcone", make_inputs(sim=ncf15), {}),
        ("ncf15_e_ang_lc", "run_lightcone", make_inputs(sim=ncf15, matter={"KEEP_3D_VELOCITIES": True}), {"angular": True, "lc_max_z": 10.5}),
        ("ncf20_e_integral", "run_coeval", make_inputs(sim=ncf20), {}),
        ("ncf20_e_mcg", "run_coeval", make_inputs(sim={**ncf20, "Z_HEAT_MAX": 15.0}, matter={"V_CB_MODEL": "AVG-DEBUG"}, astro={"USE_MCGS": True, "USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True}, astro_params={"M_TURN_STELLAR_FEEDBACK": 6.0}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("ncf20_l_integral", "run_coeval", make_inputs(sim=ncf20, matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        ("ncf20_chmf", "run_coeval", make_inputs(sim={**ncf20, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("ncf20_dexm", "run_coeval", make_inputs(sim={**ncf20, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"SOURCE_MODEL": "DEXM-ESF"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("odd_e_integral", "run_coeval", make_inputs(sim=odd), {}),
        ("odd_l_integral", "run_coeval", make_inputs(sim=odd, matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        ("odd_dexm", "run_coeval", make_inputs(sim={**odd, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"SOURCE_MODEL": "DEXM-ESF"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {}),
        ("odd_ncf14_e_integral", "run_coeval", make_inputs(sim=odd_nc), {}),
        ("odd_ncf14_l_integral", "run_coeval", make_inputs(sim=odd_nc, matter={"SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {}),
        ("wisdom_e_integral", "run_coeval", make_inputs(sim=wis, matter={"USE_FFTW_WISDOM": True}), {"wisdom_tmp": True}),
        ("wisdom_l_integral", "run_coeval", make_inputs(sim=wis, matter={"USE_FFTW_WISDOM": True, "SOURCE_MODEL": "L-INTEGRAL"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True}), {"wisdom_tmp": True}),
        ("wisdom_chmf", "run_coeval", make_inputs(sim={**wis, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"USE_FFTW_WISDOM": True, "SOURCE_MODEL": "CHMF-SAMPLER", "SAMPLE_METHOD": "BINARY-SPLIT"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {"wisdom_tmp": True}),
        ("wisdom_dexm", "run_coeval", make_inputs(sim={**wis, "N_THREADS": 1, "Z_HEAT_MAX": 15.0}, matter={"USE_FFTW_WISDOM": True, "SOURCE_MODEL": "DEXM-ESF"}, astro={"USE_EXP_FILTER": True, "CELL_RECOMB": True, "USE_METALLICITY": True, "USE_UPPER_STELLAR_TURNOVER": True}, node_redshifts=[18.0, 15.0, 12.0, 10.0, 8.0]), {"wisdom_tmp": True}),
    ]
    return [("D", name, kind, inp, kw) for name, kind, inp, kw in cases]


def run_suite(selected, output=None):
    builders = {"A": build_suite_a, "B": build_suite_b, "C": build_suite_c, "D": build_suite_d}
    keys = ["A", "B", "C", "D"] if selected == "all" else [selected]
    results = []
    for key in keys:
        for suite, name, kind, inp, kw in builders[key]():
            res = run_parity_case(suite, name, kind, inp, **kw)
            results.append(res)
            print(json.dumps(res), flush=True)
        if key == "B":
            gpu_mcg = run_mcg_only_e_gpu("B", "mcg_only_e_gpu_safe")
            results.append(gpu_mcg)
            print(json.dumps(gpu_mcg), flush=True)
    failed = [r for r in results if r["status"] != "PASS"]
    payload = {
        "suite": selected,
        "tolerances": {
            "exact_fftw": {"atol": ATOL_EXACT, "rtol": RTOL},
            "general_cufft": {"atol": ATOL_GENERAL, "rtol": RTOL, "mcg_cutoff_max_fraction": 1e-3},
        },
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "results": results,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSUMMARY ({selected}): {payload['passed']}/{payload['total']} PASSED, {payload['failed']} FAILED")
    if failed:
        raise SystemExit(1)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "A", "B", "C", "D"), default="all")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_suite(args.suite, args.output)


if __name__ == "__main__":
    main()
