"""Build a private library with a hook at the normal perturb_field entry point."""
from pathlib import Path
import shutil
import sys

_LC_DIR = str(Path(__file__).resolve().parent.parent / "lightcone")
if _LC_DIR not in sys.path:
    sys.path.insert(0, _LC_DIR)

from cffi import FFI
from env_detect import detect_native_deps

root, work = (Path(arg).resolve() for arg in sys.argv[1:])
package = work / "package"
py21 = package / "py21cmfast"
shutil.copytree(root / "resident_filter_20261118/fix/package/py21cmfast", py21)
for extension in py21.glob("c_21cmfast*.so"):
    extension.unlink()
src = py21 / "src"
output_source=py21/"outputs.py"
outputs=output_source.read_text()
before='        # Expose all the fields of the structs to the surface of the Coeval object'
assert outputs.count(before)==1
outputs=outputs.replace(before,
    '        import os\n        lazy_initial_fields = os.environ.get("PS_LAZY_COEVAL_IC") == "1"\n'+before,1)
after='            if box is None:\n                continue\n            for field in box._get_box_structures():'
assert outputs.count(after)==1
outputs=outputs.replace(after,
    '            if box is None or (lazy_initial_fields and box is initial_conditions):\n                continue\n            for field in box._get_box_structures():',1)
before='    @classmethod\n    def get_fields(cls, spin_temp: bool = True)'
assert outputs.count(before)==1
outputs=outputs.replace(before,"""    def __getattr__(self, name):
        initial = self.__dict__.get("init_struct")
        if initial is not None and name in initial.__dict__.get("_array_state", {}):
            return getattr(initial, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

"""+before,1)
compile(outputs,str(output_source),"exec")
output_source.write_text(outputs)

path = src / "PerturbField.c"
source = path.read_text()

def splice(before, after):
    global source
    assert source.count(before) == 1, before
    source = source.replace(before, after, 1)

splice("int ComputePerturbField(", """/* Keep the CPU consumers on the original FFTW rounding path. */
static int MaterializePerturbDensity(struct UserParams *user_params, float *density) {
    const int side = user_params->HII_DIM;
    fftwf_complex *box = fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    if (!box) return 1;
    #pragma omp parallel for num_threads(user_params->N_THREADS)
    for (int i=0; i<side; ++i)
        for (int j=0; j<side; ++j)
            for (int k=0; k<HII_D_PARA; ++k)
                ((float *)box)[HII_R_FFT_INDEX(i,j,k)] = density[HII_R_INDEX(i,j,k)];
    int status = dft_r2c_cube(user_params->USE_FFTW_WISDOM,side,HII_D_PARA,user_params->N_THREADS,box);
    if (!status) status = dft_c2r_cube(user_params->USE_FFTW_WISDOM,side,HII_D_PARA,user_params->N_THREADS,box);
    if (!status) {
        #pragma omp parallel for num_threads(user_params->N_THREADS)
        for (int i=0; i<side; ++i)
            for (int j=0; j<side; ++j)
                for (int k=0; k<HII_D_PARA; ++k) {
                    float value = ((float *)box)[HII_R_FFT_INDEX(i,j,k)] / (float)HII_TOT_NUM_PIXELS;
                    density[HII_R_INDEX(i,j,k)] = value < -1.0 ? -1+FRACT_FLOAT_ERR : value;
                }
    }
    fftwf_free(box);
    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();
    return status;
}
typedef int (*PerturbGPUHook)(float, int, int, double,
    float, float, float, float, const float **, float *, float *);
static PerturbGPUHook perturb_gpu_hook = NULL;
typedef int (*PerturbTraceObserver)(int,const float *,const float *);
static PerturbTraceObserver perturb_trace_observer;
void SetPerturbTraceObserver(void *p) { perturb_trace_observer = (PerturbTraceObserver)p; }

void SetPerturbGPUHook(void *hook) { perturb_gpu_hook = (PerturbGPUHook)hook; }

int ComputePerturbField(""")
anchor = "    mass_factor = pow(f_pixel_factor, 3);"
splice(anchor, anchor + """
    if (perturb_gpu_hook && !global_params.EVOLVE_DENSITY_LINEARLY &&
        !global_params.SMOOTH_EVOLVED_DENSITY_FIELD &&
        !user_params->PERTURB_ON_HIGH_RES && user_params->USE_2LPT &&
        user_params->NON_CUBIC_FACTOR == 1.0 && !user_params->KEEP_3D_VELOCITIES &&
        user_params->DIM % user_params->HII_DIM == 0) {
        const float *fields[] = {boxes->hires_density, boxes->lowres_vx, boxes->lowres_vy,
            boxes->lowres_vz, boxes->lowres_vx_2LPT, boxes->lowres_vy_2LPT, boxes->lowres_vz_2LPT};
        const float first = (growth_factor - init_growth_factor) / user_params->BOX_LEN;
        const float second = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;
        const int result = perturb_gpu_hook(redshift, user_params->DIM, user_params->HII_DIM,
            DELTA_K, first, second, init_growth_factor, dDdt/growth_factor,
            fields, perturbed_field->density, perturbed_field->velocity_z);
        if (result == 3) {
            if (MaterializePerturbDensity(user_params, perturbed_field->density)) Throw(ValueError);
            goto perturb_success;
        }
        if (result == 0) goto perturb_success;
        if (result != 2) Throw(ValueError);
    }
""")
splice("    } // End of Try{}", "    perturb_success: ;\n    if (perturb_trace_observer && perturb_trace_observer(user_params->HII_DIM, perturbed_field->density, perturbed_field->velocity_z)) Throw(ValueError);\n    } // End of Try{}")
path.write_text(source)

ip = src / "IonisationBox.c"
ion = ip.read_text()
ion = ion.replace("int ComputeIonizedBox(", """typedef int (*IonObserver)(float, float, unsigned long long, const float *, const float *);
static IonObserver ion_observer = NULL;
void SetIonObserver(void *hook) { ion_observer = (IonObserver)hook; }

int ComputeIonizedBox(""", 1)
anchor = '    free(overdense_int_boundexceeded_threaded);'
assert ion.count(anchor) == 1
ion = ion.replace(anchor, """    if (ion_observer && ion_observer(redshift, prev_redshift, HII_TOT_NUM_PIXELS,
        box->xH_box, box->dNrec_box)) Throw(ValueError);
""" + anchor)
from ion_patch import patch_ion
ion = patch_ion(ion)
from trajectory_patch import patch_trajectory
ion = patch_trajectory(ion)
ip.write_text(ion)
from heat_seed_patch import patch_heat
sp=src / "SpinTemperatureBox.c"
from xray_patch import patch_xray
from conditional_patch import patch_condition, patch_heat_table_consumer
spin=patch_heat_table_consumer(patch_xray(patch_heat(sp.read_text())))
anchor="/* The GPU filter cache belongs"
assert spin.count(anchor)==1
spin=spin.replace(anchor,
    "typedef int (*GPUPreviousHeatMeans)(float,const float *,double *,double *);\n"
    "static GPUPreviousHeatMeans previous_heat_means;\n"
    "void SetGPUPreviousHeatMeans(void *ptr) {previous_heat_means=(GPUPreviousHeatMeans)ptr;}\n"
    +anchor,1)
anchor="        #pragma omp parallel shared(previous_spin_temp) private(ct)"
assert spin.count(anchor)==1
spin=spin.replace(anchor,
    "        if (!previous_heat_means || previous_heat_means(prev_redshift,previous_spin_temp->x_e_box,&x_e_ave,&Tk_ave)) {\n"
    +anchor,1)
anchor="        x_e_ave /= (float)HII_TOT_NUM_PIXELS;"
assert spin.count(anchor)==1
spin=spin.replace(anchor,"        }\n"+anchor,1)
sp.write_text(spin)
from brightness_patch import patch_brightness
bp=src / "BrightnessTemperatureBox.c"
bp.write_text(patch_brightness(bp.read_text()))

from ic_seed_patch import patch_seed
ic_path=src / "GenerateICs.c"
ic_path.write_text(patch_seed(ic_path.read_text()))
ps=src/"ps.c"
from sigma_patch import patch_sigma
from hmf_patch import patch_hmf
ps.write_text(patch_hmf(patch_sigma(patch_condition(ps.read_text()))))
from mhr_patch import patch_mhr
rec=src/"recombinations.c"
rec.write_text(patch_mhr(rec.read_text()))
from heat_table_cache_patch import patch_heat_table_cache
heat_helpers=src/"heating_helper_progs.c"
heat_helpers.write_text(patch_heat_table_cache(heat_helpers.read_text()))
from native_profile import patch_profile
patch_profile(src)
deps = detect_native_deps(root=root)
missing = []
if not deps["gsl_found"]:
    missing.append("GSL (gsl/gsl_integration.h)")
if not deps["fftw_found"]:
    missing.append("FFTW3f (fftw3.h)")
if missing:
    raise RuntimeError(
        "Missing native build dependencies: "
        + ", ".join(missing)
        + ". Install them via `conda env create -f environment.yml` or pass "
        + "PS_GSL_PREFIX / PS_FFTW_PREFIX."
    )
ffi = FFI()
ffi.set_source(
    "py21cmfast.c_21cmfast",
    '#define LOG_LEVEL 1\n#include "GenerateICs.c"',
    include_dirs=[str(src), *deps["include_dirs"]],
    library_dirs=deps["library_dirs"],
    libraries=["m","gsl","gslcblas","fftw3f_omp","fftw3f"],
    extra_compile_args=["-fopenmp","-O3","-fno-fast-math","-fno-finite-math-only"],
    extra_link_args=["-fopenmp", *[f"-Wl,-rpath,{d}" for d in deps["library_dirs"]]],
)
for name in ("21cmFAST.h","Globals.h"):
    ffi.cdef((src/name).read_text())
ffi.cdef("void SetPerturbGPUHook(void *);")
ffi.cdef("void SetGPUFilterHook(void *); void SetGPUHeatHook(void *); void SetHeatObserver(void *); int ResolveCollapseDensity(const float *, float, const long *, long, float *);")
ffi.compile(tmpdir=str(package),verbose=True)
