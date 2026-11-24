"""Paired photon-conservation calibration samples for 21cmFAST 4.3.2."""

from dataclasses import dataclass
import ctypes
from unittest.mock import patch

import numpy as np
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.drivers import coeval, lightcone
from py21cmfast.drivers._global_initialization import c_state
from py21cmfast.wrapper import photoncons
from py21cmfast.wrapper.exceptions import _process_exitcode


def calibration_inputs(inputs):
    """The upstream simplified calibration physics, keeping the Eulerian source model."""
    return inputs.evolve_input_structs(
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_MCGS=False,
        PHOTON_CONS_TYPE="no-photoncons",
        R_BUBBLE_MAX=15
        if inputs.astro_options.RECOMB_MODEL != "none"
        else inputs.astro_params.R_BUBBLE_MAX,
    ).clone(node_redshifts=None)


@dataclass(frozen=True, eq=False)
class PhotonCalibration:
    """Mean neutral fractions paired with their actual simulated redshifts in ascending order."""

    inputs: object
    redshifts: np.ndarray
    neutral_fractions: np.ndarray

    def __post_init__(self):
        for name in ("redshifts", "neutral_fractions"):
            data = np.asarray(getattr(self, name), dtype="d")
            # A bytes owner prevents callers from re-enabling array writes.
            frozen = np.frombuffer(data.tobytes(), dtype="d").reshape(data.shape)
            object.__setattr__(self, name, frozen)
        z, x = self.redshifts, self.neutral_fractions
        # The upstream data-export wrapper provides buffers of 2000 doubles.
        if (
            z.ndim != 1
            or z.shape != x.shape
            or not 2 <= z.size <= 2000
            or not np.isfinite(z).all()
            or not np.isfinite(x).all()
            or np.any(np.diff(z) <= 0)
            or np.any((x < 0) | (x > 1))
        ):
            raise ValueError(
                "Provide 2–2000 matched, finite redshifts and neutral fractions in ascending redshift order"
            )
        if z[0] < 0 or x[-1] < 0.995 or x.min() >= 0.995:
            raise ValueError(
                "Calibration must cover the native 0.995 neutral-fraction onset"
            )
        self.validate(self.inputs)

    def validate(self, inputs):
        if self.inputs != inputs:
            raise ValueError(
                "Photon calibration belongs to different simulation inputs"
            )
        source, mode = (
            inputs.matter_options.SOURCE_MODEL,
            inputs.astro_options.PHOTON_CONS_TYPE,
        )
        allowed = {
            "E-INTEGRAL": ("z-photoncons", "f-photoncons", "alpha-photoncons"),
            "CONST-ION-EFF": ("z-photoncons",),
        }
        if mode not in allowed.get(source, ()):
            raise ValueError(f"Unsupported photon correction {mode} for {source}")

    @classmethod
    def from_boxes(cls, inputs, boxes):
        """Construct a history from existing uncorrected native calibration boxes."""
        expected = calibration_inputs(inputs)
        samples = []
        for box in boxes:
            if box.inputs.clone(node_redshifts=None) != expected:
                raise ValueError(
                    "IonizedBox inputs do not match the native calibration configuration"
                )
            samples.append((box.redshift, np.mean(box.get("neutral_fraction"))))
        if not samples:
            raise ValueError("Calibration requires ionization boxes")
        return cls(inputs, *zip(*sorted(samples), strict=True))


def private_source(source):
    # Bind the three externally visible data symbols to this private shim.
    prefix = (
        "#define photon_cons_allocated gpu_pc_allocated\n"
        "#define fesc_photoncons_yint gpu_pc_yint\n"
        "#define fesc_photoncons_slope gpu_pc_slope\n"
    )
    # The alpha fitter replaces the analytic spline 31 times. Release each
    # old analytic history, retaining the calibration and delta-z splines.
    setup = source.index("int InitialisePhotonCons()")
    source = (
        source[:setup]
        + """
static double *z_arr, *Q_arr, *Q_z, *z_value;
static void release_analytic(void) {
    free(z_arr); z_arr=NULL; free(Q_arr); Q_arr=NULL;
    free(Q_z); Q_z=NULL; free(z_value); z_value=NULL;
    free(z_Q); z_Q=NULL;
    free(Q_value); Q_value=NULL;
    gsl_spline_free(Q_at_z_spline); Q_at_z_spline=NULL;
    gsl_interp_accel_free(Q_at_z_spline_acc); Q_at_z_spline_acc=NULL;
    gsl_spline_free(z_at_Q_spline); z_at_Q_spline=NULL;
    gsl_interp_accel_free(z_at_Q_spline_acc); z_at_Q_spline_acc=NULL;
}
"""
        + source[setup:]
    )
    # A failed analytic solve unwinds through C longjmp. Track scratch
    # pointers in the private image so the scope can release them as well.
    source = source.replace("double *z_arr, *Q_arr;", "")
    for name in ("Q_z", "z_value"):
        source = source.replace(f"double *{name} = calloc", f"{name} = calloc")
    for name in ("z_arr", "Q_arr", "Q_z", "z_value"):
        source = source.replace(
            f"        free({name});", f"        free({name}); {name}=NULL;"
        )
    source = source.replace(
        "N_analytic = nbin;",
        "if (nbin < 3 || nbin > Nmax) Throw(PhotonConsError);\n        N_analytic = nbin;",
    )
    source = source.replace(
        "z_arr = calloc(Nmax", "release_analytic();\n        z_arr = calloc(Nmax"
    )
    source = source.replace(
        "z_arr[cnt] = zi;",
        "if (cnt >= Nmax) Throw(PhotonConsError);\n                z_arr[cnt] = zi;",
    )
    # If the forward smoothing search runs out of samples, native 4.3.2
    # unconditionally replaces its tentative smoothed value with deltaz[i+1]
    # after the loop. Break at that same condition before the speculative read.
    source = source.replace(
        "val2 = deltaz[i + 1 + counter];",
        "if (i + 2 + counter > N_NFsamples + N_extrapolated) break;\n"
        "                val2 = deltaz[i + 1 + counter];",
    )
    source = source.replace(
        "deltaz[i] += deltaz_smoothed[",
        "if (i - (int)floor((float)smoothing_int / 2.) + j > N_NFsamples + N_extrapolated) Throw(PhotonConsError);\n"
        "                    deltaz[i] += deltaz_smoothed[",
    )
    # The upstream destructor assumes that every allocation succeeded. The
    # scope also exits on exceptions or early generator close.
    start = source.index("void FreePhotonConsMemory()")
    end = source.index("// quick functions", start)
    source = (
        source[:start]
        + """
void FreePhotonConsMemory(void) {
    release_analytic();
    free(nf_vals); nf_vals=NULL; free(z_vals); z_vals=NULL;
    free(deltaz); deltaz=NULL; free(deltaz_smoothed); deltaz_smoothed=NULL;
    free(NeutralFractions); NeutralFractions=NULL;
    gsl_spline_free(NFHistory_spline); NFHistory_spline=NULL;
    gsl_interp_accel_free(NFHistory_spline_acc); NFHistory_spline_acc=NULL;
    gsl_spline_free(z_NFHistory_spline); z_NFHistory_spline=NULL;
    gsl_interp_accel_free(z_NFHistory_spline_acc); z_NFHistory_spline_acc=NULL;
    gsl_spline_free(deltaz_spline_for_photoncons); deltaz_spline_for_photoncons=NULL;
    gsl_interp_accel_free(deltaz_spline_for_photoncons_acc); deltaz_spline_for_photoncons_acc=NULL;
    photon_cons_allocated=false;
    nfhistory_spline_allocated=false;
}
int gpu_build_photon_shift(void) {
    int status;
    Try {
        determine_deltaz_for_photoncons();
        photon_cons_allocated=true;
    } Catch(status) {return status;}
    return 0;
}
"""
        + source[end:]
    )
    return prefix + source


def install(stack, adapter, calibration):
    """Initialize isolated C splines and route upstream driver reads to this private copy."""

    class Library:
        @property
        def photon_cons_allocated(self):
            return ctypes.c_bool.in_dll(adapter, "gpu_pc_allocated").value

        def InitialisePhotonCons(self):
            # The upstream alpha scan ignores this C return code. Propagate
            # failed trial histories before the next spline evaluation.
            status = adapter.InitialisePhotonCons()
            _process_exitcode(status, lib.InitialisePhotonCons, ())
            return status

        def __getattr__(self, name):
            if name in (
                "ObtainPhotonConsData",
                "set_alphacons_params",
                "FreePhotonConsMemory",
                "PhotonCons_Calibration",
            ):
                return ffi.cast(
                    ffi.typeof(getattr(lib, name)),
                    ctypes.cast(getattr(adapter, name), ctypes.c_void_p).value,
                )
            return getattr(lib, name)

    private = Library()
    state = photoncons._PhotonConservationState()
    stack.enter_context(patch.object(photoncons, "_photoncons_state", state))
    for module in (photoncons, coeval, lightcone):
        stack.enter_context(patch.object(module, "lib", private))
    stack.callback(adapter.FreePhotonConsMemory)
    inputs = calibration.inputs
    with c_state(inputs, sigma=True):
        private.InitialisePhotonCons()
        z, x = calibration.redshifts, calibration.neutral_fractions
        status = private.PhotonCons_Calibration(
            ffi.cast("double *", z.ctypes.data),
            ffi.cast("double *", x.ctypes.data),
            len(z),
        )
        _process_exitcode(status, lib.PhotonCons_Calibration, ())
        _process_exitcode(
            adapter.gpu_build_photon_shift(), adapter.gpu_build_photon_shift, ()
        )
        history = photoncons._get_photon_nonconservation_data()
        if not all(np.isfinite(values).all() for values in history.values()):
            raise ValueError(
                "The supplied history does not span the native photon calibration spline"
            )
        mode = inputs.astro_options.PHOTON_CONS_TYPE
        if mode == "f-photoncons":
            result = photoncons.photoncons_fesc(inputs)
        elif mode == "alpha-photoncons":
            result = photoncons.photoncons_alpha(inputs)
        else:
            result = history
    state.calibration_inputs = inputs

    def reuse(*, inputs, **kwargs):
        calibration.validate(inputs)
        return result

    stack.enter_context(patch.object(coeval, "setup_photon_cons", reuse))
    return result
