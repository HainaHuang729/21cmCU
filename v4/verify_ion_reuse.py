"""Replay native v4 center-method inputs on CUDA at each radius of one node."""

import ctypes as C
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.lightconers import RectilinearLightconer
from py21cmfast.wrapper.outputs import IonizedBox, TsBox
from tests import produce_integration_test_data as official

root = Path(__file__).resolve().parent
shared = root.parent / "lightcone"
module = cp.RawModule(
    code="\n".join(
        p.read_text(encoding="utf-8")
        for p in [
            shared / "glibc_math.cuh",
            shared / "ion.cu",
            root / "ion_fraction.cu",
        ]
    ),
    options=("--fmad=false",),
)
assign = module.get_function("ion_fraction_center")
thermal = module.get_function("ion_temperature")
source_dir = Path(p21.__file__).parent / "src"
source = (source_dir / "IonisationBox.c").read_text(encoding="utf-8")

# Capture values at the physical decision, after upstream normalization, and
# compare to the result *after* the unmodified upstream loop has executed.
a = source.index("void find_ionised_regions(")
b = source.index("void set_ionized_temperatures(", a)
part = source[a:b]
part = part.replace(
    "    double mean_fix_term_acg",
    """
    size_t n = HII_TOT_NUM_PIXELS;
    double *saved = capture_enabled ? calloc(5*n, sizeof(double)) : NULL;
    if (capture_enabled && !saved) abort();
    double mean_fix_term_acg""",
)
part = part.replace(
    "                    // check if fully ionized!",
    """
                    if (saved) {
                        saved[index_r] = curr_nion_acg;
                        saved[n+index_r] = curr_nion_mcg;
                        saved[2*n+index_r] = xHII_from_xrays;
                        saved[3*n+index_r] = rec;
                        saved[4*n+index_r] = box->neutral_fraction[index_r];
                    }
                    // check if fully ionized!""",
)
i = part.rindex("}")
part = (
    part[:i]
    + """
    if (saved) {
        capture(n, rspec.R_index, rspec.R, saved, box->neutral_fraction);
        free(saved);
    }
"""
    + part[i:]
)
source = (
    source[:a]
    + """
static int capture_enabled;
typedef void (*observer)(int,int,double,double*,float*);
static observer capture;
void observe_scale(observer cb) {capture=cb; capture_enabled=(cb!=NULL);}
"""
    + part
    + source[b:]
)
native = Path(p21.c_21cmfast.__file__)
fptr = C.POINTER(C.c_float)
dptr = C.POINTER(C.c_double)
signature = C.CFUNCTYPE(None, C.c_int, C.c_int, C.c_double, dptr, fptr)
rows = []
errors = []


def compare(actual, expected):
    diff = abs(actual.astype("d") - expected)
    return dict(
        max_abs=float(diff.max()),
        failures=int(
            np.count_nonzero(~(diff <= 1e-8 + 1e-4 * abs(expected.astype("d"))))
        ),
    )


def check(n, idx, radius, saved, output):
    try:
        fields = np.ctypeslib.as_array(saved, shape=(5, n))
        expected = np.ctypeslib.as_array(output, shape=(n,))
        xhi = cp.asarray(fields[-1], dtype="f")
        assign(
            ((n + 255) // 256,),
            (256,),
            (
                *[cp.asarray(x) for x in fields[:4]],
                xhi,
                np.int32(n),
                np.int32(idx == 0),
            ),
        )
        rows.append(
            dict(
                index=idx,
                radius=radius,
                cells=n,
                changed=int(np.sum(fields[-1] != expected)),
                **compare(xhi.get(), expected),
            )
        )
    except BaseException as exc:
        errors.append(repr(exc))


callback = signature(check)
with patch.object(official, "SEED", 424242):
    inputs = official.get_all_options_struct(
        6,
        lc=True,
        DIM=192,
        HII_DIM=64,
        BOX_LEN=200,
        KEEP_3D_VELOCITIES=True,
        PERTURB_ON_HIGH_RES=True,
        SMOOTH_EVOLVED_DENSITY_FIELD=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
    )["inputs"]
assert not inputs.astro_options.IONISE_ENTIRE_SPHERE
with tempfile.TemporaryDirectory() as tmp:
    cfile = Path(tmp) / "observer.c"
    cfile.write_text(source)
    # Observe the rates immediately before the native spin solver. Attach one
    # capture row to each cell explicitly so OpenMP scheduling is immaterial.
    ts_source = (source_dir / "SpinTemperatureBox.c").read_text(encoding="utf-8")
    ts_source = ts_source.replace(
        "struct Ts_cell get_Ts_fast(",
        "static double *thermal_rows;\nvoid observe_thermal(double *p) { thermal_rows=p; }\n"
        "static struct Ts_cell observed_Ts(double *row, ",
    )
    ts_source = ts_source.replace(
        "ts_cell = get_Ts_fast(redshift,",
        "ts_cell = observed_Ts(thermal_rows ? thermal_rows + 11*box_ct : NULL, redshift,",
    )
    marker = "    if (J_alpha_tot > 1.0e-20)"
    assert ts_source.count(marker) == 1
    ts_source = ts_source.replace(
        marker,
        """
    if (row) {
        row[0]=rad->prev_xe; row[1]=rad->prev_Tk; row[2]=dxe_dzp;
        row[3]=dxheat_dzp+dcomp_dzp+dspec_dzp+dadia_dzp+dCMBheat_dzp+eps_Lya_cont+eps_Lya_inj;
        row[4]=dzp; row[5]=consts->Trad; row[6]=consts->Trad_inv; row[7]=J_alpha_tot;
        row[8]=xa_tilde_fast_arg; row[9]=xCMB; row[10]=xc_fast;
    }
"""
        + marker,
    )
    tfile = Path(tmp) / "thermal.c"
    tfile.write_text(ts_source)
    so = Path(tmp) / "observer.so"
    subprocess.run(
        [
            "cc",
            "-O3",
            "-std=gnu99",
            "-fPIC",
            "-shared",
            "-fopenmp",
            "-Wl,-Bsymbolic-functions",
            "-I" + str(source_dir),
            *os.environ.get("CPPFLAGS", "").split(),
            str(cfile),
            str(tfile),
            str(native),
            "-lm",
            "-o",
            str(so),
        ],
        check=True,
    )
    instrument = C.CDLL(str(so))
    instrument.observe_scale.argtypes = [signature]
    addr = C.cast(instrument.ComputeIonizedBox, C.c_void_p).value
    native_step = ffi.cast(ffi.typeof(lib.ComputeIonizedBox), addr)

    def observed_step(*args):
        return native_step(*args)

    heat_call = ffi.cast(
        ffi.typeof(lib.ComputeTsBox), C.cast(instrument.ComputeTsBox, C.c_void_p).value
    )

    def observed_heat(*args):
        return heat_call(*args)

    instrument.observe_thermal.argtypes = [dptr]
    captured = np.full((inputs.simulation_options.HII_DIM**3, 11), np.nan)
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6,
        max_redshift=15,
        quantities=("brightness_temp",),
        resolution=inputs.simulation_options.cell_size,
    )
    with (
        patch.object(IonizedBox, "_c_compute_function", staticmethod(observed_step)),
        patch.object(TsBox, "_c_compute_function", staticmethod(observed_heat)),
    ):
        for _, z, box, _ in p21.generate_lightcone(
            inputs=inputs, lightconer=lightconer, regenerate=True, write=False
        ):
            if rows or errors:
                break
            if z <= 7.52:
                instrument.observe_scale(callback)
                instrument.observe_thermal(captured.ctypes.data_as(dptr))
    assert rows and not errors, errors
    expected = np.asarray(box.kinetic_temperature).ravel()
    h = np.asarray(box.neutral_fraction).ravel()
    active = (np.asarray(box.z_reion).ravel() > 0) & (h < 1e-30)
    assert active.any()
    n = h.size
    # This kernel writes fully ionized cells only. Poison those before replay.
    initial = np.where(active, np.float32(np.nan), expected)
    out = cp.asarray(initial)
    flag = cp.zeros(1, dtype="i")
    arrays = [
        cp.asarray(np.ravel(x), dtype="f")
        for x in (box.density, h, box.z_reion, box.kinetic_temp_neutral)
    ]
    thermal(
        ((n + 255) // 256,),
        (256,),
        (
            *arrays,
            out,
            np.int32(n),
            np.float32(z),
            np.float32(inputs.astro_params.T_RE),
            np.float32(9e-8),
            flag,
        ),
    )
    assert not flag.get()[0]
    result = dict(
        seed=inputs.random_seed,
        z=z,
        atol=1e-8,
        rtol=1e-4,
        scales=rows,
        fully_ionized_temperature=dict(
            cells=int(active.sum()), **compare(out.get()[active], expected[active])
        ),
    )
    assert np.isfinite(captured).all()
    heat_module = cp.RawModule(
        code="\n".join(
            p.read_text(encoding="utf-8")
            for p in [
                shared / "glibc_math.cuh",
                shared / "update_stabilized.cu",
                root / "thermal_step.cu",
            ]
        ),
        options=("--fmad=false",),
    )
    outputs = [cp.empty(n, dtype="f") for _ in range(3)]
    heat_module.get_function("thermal_step")(
        ((n + 255) // 256,), (256,), (cp.asarray(captured), np.int32(n), *outputs)
    )
    result["thermal_step"] = {
        field: compare(arr.get(), np.ravel(getattr(box, field)))
        for field, arr in zip(
            ("xray_ionised_fraction", "kinetic_temp_neutral", "spin_temperature"),
            outputs,
        )
    }
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    assert all(r["failures"] == 0 for r in rows + [result["fully_ionized_temperature"]])

assert all(r["failures"] == 0 for r in result["thermal_step"].values())
