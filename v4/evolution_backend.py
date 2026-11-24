"""v4.3.2 native-driver hooks for the Eulerian, atomic-cooling benchmark."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import ctypes as C
import json
import logging
from importlib.metadata import version
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

from native_shim import load_shim

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper.outputs import (
    InitialConditions,
    EmissivityFields,
    IonizedBox,
    RadiationFields,
    RadiationFieldsSetup,
    TsBox,
)
from mhr import build as build_mhr

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "lightcone"
SRC = Path(p21.__file__).parent / "src"
NATIVE = Path(p21.c_21cmfast.__file__)
active_emissivity_history = ContextVar("active_emissivity_history", default=None)
CB = C.CFUNCTYPE(
    C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_void_p)
)
HEADER = """
extern int (*gpu_dispatch)(int,int,double*,void**);
#define GPU_CALL(ID, SCALARS, GRIDS) do { if(gpu_dispatch(ID,HII_TOT_NUM_PIXELS,SCALARS,GRIDS)) Throw(ValueError); } while(0)
"""


def insert(source, function, marker, addition):
    start = source.index(function)
    at = source.index(marker, start)
    return source[:at] + addition + source[at:]


THERMAL_TERMS = (
    "dxe_dzp",
    "dxheat_dzp",
    "dcomp_dzp",
    "dspec_dzp",
    "dadia_dzp",
    "dCMBheat_dzp",
    "eps_Lya_cont",
    "eps_Lya_inj",
    "J_alpha_tot",
    "xa_tilde_fast_arg",
    "xCMB",
    "xc_fast",
)


def thermal_source(audit):
    src = (SRC / "SpinTemperatureBox.c").read_text()
    capture = (
        "if (terms) {"
        + " ".join(f"terms[{j}]={name};" for j, name in enumerate(THERMAL_TERMS))
        + "}\n"
    )
    src = src.replace(
        "    if (J_alpha_tot > 1.0e-20)", capture + "    if (J_alpha_tot > 1.0e-20)"
    )
    src = src.replace(
        "struct local_rad_terms *rad) {",
        "struct local_rad_terms *rad, double *terms) {",
    )
    cell = src[src.index("struct Ts_cell get_Ts_fast(") : src.index("int ComputeTsBox")]
    cell = re.sub(r"#if LOG_LEVEL.*?#endif", "", cell, flags=re.S)
    cell = re.sub(
        r"if \(debug_printed == 0 && omp_get_thread_num\(\) == 0\)\s*LOG_SUPER_DEBUG\(.*?\);",
        "",
        cell,
        flags=re.S,
    )
    cell = cell.replace("debug_printed = 0;", "")
    # Use the same table layout and numerical primitives as the v3 CUDA update.
    cell = re.sub(
        r"(taugp\(zp, rad->delta, rad->prev_xe\)), ([23])\)", r"\1, \2, tc, ti)", cell
    )
    cell = cell.replace(
        "taugp(zp, rad->delta, rad->prev_xe)",
        "(1.342881e-7 / consts->hubble_zp * nh * pow(1.+zp,3.) * (1.+rad->delta)*(1.-rad->prev_xe))",
    )
    keys = list(
        dict.fromkeys(
            re.findall(
                r"physconst\.\w+|astro_(?:params|options)_global->\w+|\b(?:No|N_b0|H_FRAC|HE_FRAC)\b",
                cell,
            )
        )
    )
    for i, key in enumerate(keys):
        cell = re.sub(r"(?<!\w)" + re.escape(key) + r"(?!\w)", f"params[{i}]", cell)
    cell = cell.replace("* nh *", f"* params[{keys.index('No')}] *").replace(
        "M_PI", "3.14159265358979323846"
    )
    cell = cell.replace(
        "struct Ts_cell get_Ts_fast", "__device__ struct Ts_cell get_Ts_fast"
    ).replace(
        "struct local_rad_terms *rad, double *terms)",
        "struct local_rad_terms *rad, double *terms, const double *params, const double *tc, const double *ti)",
    )

    def struct(name):
        return re.search(r"struct " + name + r" \{.*?\};", src, re.S)[0]

    head = "\n".join(
        struct(n)
        for n in ("Ts_cell", "local_rad_terms", "spintemp_from_sfr_prefactors")
    )
    chemistry = (SRC / "thermochem.c").read_text()
    start = chemistry.index("double alpha_A(double T) {")
    end = chemistry.index("\n}", start) + 2
    # v4 evaluates exp directly. v3 uses pow with a rounded value of Euler's number.
    alpha = "__device__ " + chemistry[start:end].replace("alpha_A(", "v432_alpha_A(")
    alpha = re.sub(r"pow\(logT, (\d+)\)", r"pow(logT, \1.0)", alpha)
    cell = cell.replace("alpha_A(", "v432_alpha_A(")
    cuda = (
        alpha
        + "\n"
        + head
        + "\n"
        + cell
        + """
extern "C" __global__ void evolution_heat(int n,const double *params,const double *tc,const double *ti,
const float *den,const float *oldx,const float *oldt,const float *olds,
const double *ion,const double *heat,const double *xlya,const double *cont,const double *inj,
float *xe,float *tk,float *ts,double *terms,const double *lw,float *j21) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 local_rad_terms r={};r.delta=den[i]<=-1.f ? -1.+FRACT_FLOAT_ERR : (double)den[i];
 r.prev_xe=oldx[i];r.prev_Tk=oldt[i];r.prev_Ts=olds[i];r.xray_ionization_rate=ion[i];
 r.xray_heating_rate=heat[i];r.xray_lya_flux=xlya[i];r.lya_flux_continuum=cont[i];
 if(lw) r.lyw_flux=lw[i];
 r.lya_flux_continuum_injected=cont[i];r.lya_flux_injected=inj[i];
 spintemp_from_sfr_prefactors *pref=(spintemp_from_sfr_prefactors *)(params+KEYS);
 Ts_cell v=get_Ts_fast(params[KEYS+12],params[KEYS+13],pref,&r,terms ? terms+12*i : 0,params,tc,ti);
 xe[i]=v.x_e;tk[i]=v.Tk;ts[i]=v.Ts;
 if(j21) j21[i]=v.J_21_LW;
}
""".replace("KEYS", str(len(keys)))
    )
    scalars = (
        keys
        + [
            "zp_consts." + k
            for k in re.findall(
                r"double (\w+);", struct("spintemp_from_sfr_prefactors")
            )
        ]
        + ["redshift", "dzp"]
    )
    hook = """
        double parameters[]={SCALARS};
        void *arrays[]={perturbed_field->density,previous_spin_temp->xray_ionised_fraction,
          previous_spin_temp->kinetic_temp_neutral,previous_spin_temp->spin_temperature,
          radiation_fields->xray_ionization_rate,radiation_fields->xray_heating_rate,
          radiation_fields->xray_lya_flux,astro_options_global->USE_LYA_HEATING ?
          radiation_fields->lya_flux_continuum : radiation_fields->lya_flux_continuum_injected,
          radiation_fields->lya_flux_injected,this_spin_temp->xray_ionised_fraction,
          this_spin_temp->kinetic_temp_neutral,this_spin_temp->spin_temperature,terms,
          astro_options_global->USE_MCGS ? radiation_fields->lyw_flux : NULL,
          astro_options_global->USE_MCGS ? this_spin_temp->J_21_LW : NULL};
        GPU_CALL(1,parameters,arrays);
        return 0;
""".replace("SCALARS", ",".join(scalars))
    loop = "#pragma omp parallel private(box_ct)"
    src = src.replace(
        "&zp_consts, &local_rad);",
        "&zp_consts, &local_rad, terms ? terms+12*box_ct : NULL);",
    )
    if audit:
        src = insert(
            src,
            "int ComputeTsBox(",
            loop,
            "double *terms=malloc(12*sizeof(double)*HII_TOT_NUM_PIXELS); if(!terms) Throw(ValueError);\n",
        )
        src = insert(
            src,
            "int ComputeTsBox(",
            "    }  // End of try",
            hook.replace("return 0;", "free(terms);"),
        )
    else:
        src = insert(
            src,
            "int ComputeTsBox(",
            loop,
            "double *terms=NULL;\n"
            + hook.replace("return 0;", "goto gpu_thermal_done;"),
        )
        src = insert(
            src, "int ComputeTsBox(", "    }  // End of try", "gpu_thermal_done: ;\n"
        )
    src = insert(
        src,
        "void init_first_Ts(",
        "    index_huge box_ct;",
        """
        double p[]={z,astro_options_global->USE_ADIABATIC_FLUCTUATIONS,physconst.T_cmb,
                    physconst.T_21,physconst.A10,No,N_b0};
        void *a[]={dens,box->xray_ionised_fraction,box->kinetic_temp_neutral,box->spin_temperature};
        GPU_CALL(26,p,a); return;
""",
    )
    from growth import device_source, FIELDS

    begin = src.index("void set_zp_consts(")
    end = src.index("\n}", begin) + 2
    body = src[begin:end]
    body = body[: body.index("    LOG_DEBUG")] + "\n}"
    body = body.replace(
        "void set_zp_consts(double zp, struct spintemp_from_sfr_prefactors *consts)",
        'extern "C" __global__ void thermal_constants(double zp,const double *c,const double *p,struct spintemp_from_sfr_prefactors *consts)',
    )
    for f in ["hubble", "dicke", "ddicke_dz", "dtdz"]:
        body = body.replace(f + "(zp)", f + "(zp,c)")
    constants = list(
        dict.fromkeys(
            re.findall(
                r"physconst\.\w+|\b(?:Ho|No|N_b0)\b|cosmo_params_global->hlittle", body
            )
        )
    )
    for i, k in enumerate(constants):
        body = re.sub(r"(?<!\w)" + re.escape(k) + r"(?!\w)", f"p[{i}]", body)
    body = body.replace("M_PI", "3.14159265358979323846").replace(
        "pow(1 + zp, 3)", "pow(1 + zp, 3.)"
    )
    cuda += "\n" + device_source() + "\n" + body
    pars = (
        ["zp", str(len(constants))]
        + ["cosmo_params_global->" + f for f in FIELDS]
        + constants
    )
    src = (
        src[:begin]
        + src[begin : src.index("{", begin) + 1]
        + "double p[]={"
        + ",".join(pars)
        + "}; void *a[]={consts}; GPU_CALL(27,p,a); }"
        + src[end:]
    )
    return HEADER + src, cuda, len(scalars)


class Table(C.Structure):
    _fields_ = [
        ("n", C.c_int),
        ("low", C.c_double),
        ("width", C.c_double),
        ("y", C.POINTER(C.c_float)),
        ("allocated", C.c_bool),
    ]


def _table_2d_fields(value_type):
    return [
        ("nx", C.c_int),
        ("ny", C.c_int),
        ("xmin", C.c_double),
        ("ymin", C.c_double),
        ("xstep", C.c_double),
        ("ystep", C.c_double),
        ("values", C.POINTER(C.POINTER(value_type))),
        ("saved_ll", C.c_double),
        ("saved_ul", C.c_double),
        ("allocated", C.c_bool),
    ]


class Table2DFloat(C.Structure):
    _fields_ = _table_2d_fields(C.c_float)


class Table2DDouble(C.Structure):
    _fields_ = _table_2d_fields(C.c_double)


def native_tables(extra_names=()):
    # Resolve local table symbols in the matching unstripped benchmark library.
    symbols = {}
    for line in subprocess.check_output(
        ["nm", "--defined-only", NATIVE], text=True
    ).splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    base = (
        C.cast(C.CDLL(str(NATIVE)).PyInit_c_21cmfast, C.c_void_p).value
        - symbols["PyInit_c_21cmfast"]
    )
    return {
        name: base + symbols[name]
        for name in (
            "SFRD_conditional_table1D_acg",
            "Nion_conditional_table1D_acg",
            "RR_table",
            "lnGamma_values",
            "Sigma_InterpTable",
            "dSigmasqdm_InterpTable",
            "cosmo_consts",
            "Nion_z_table_acg",
            "Nion_z_table_mcg",
            "SFRD_z_table_acg",
            "x_int_Energy",
            "x_int_XHII",
            "x_int_fheat",
            "x_int_n_Lya",
            "x_int_nion_HI",
            "x_int_nion_HeI",
            "x_int_nion_HeII",
            "xi_GL",
            "wi_GL",
            *extra_names,
        )
    }


def view(ptr, n, dtype):
    ct = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    if not ptr:
        raise ValueError("Unallocated native field")
    return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(ct)), shape=(n,))


EULERIAN_EMISSIVITY = """source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)
        && !astro_options_global->USE_MCGS && !astro_options_global->USE_METALLICITY
        && !config_settings.EXTRA_EMISSIVITY_FIELDS
        && !astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK
        && uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)"""


def emission_sources():
    src = (SRC / "map_mass.c").read_text()
    hook = """
    if (EULERIAN_MODEL) {
        double p[]={consts->lx_over_sfr_acg,0.};
        void *a[]={dens_pointer,emissivity_fields->n_ion,emissivity_fields->sfrd_acg,emissivity_fields->xray_emissivity};
        GPU_CALL(0,p,a); return;
    }
""".replace("EULERIAN_MODEL", EULERIAN_EMISSIVITY)
    src = insert(
        src, "void move_integral_emissivities(", "    double growth_factor", hook
    )
    return HEADER + src


def normalize_emissivity_source(src):
    # Native debug logging reads the pre-normalization field after the mapping
    # call. For that build and explicit table audits, preserve the two calls.
    hook = """
#if !GPU_TABLE_AUDIT && LOG_LEVEL < DEBUG_LEVEL
    if ((EULERIAN_MODEL) && ev_consts->fix_mean) {
        IntegralProperties target;
        double mturn_acg=pow(10.,emissivity_fields->log10_mturn_acg_ave);
        double mturn_mcg=pow(10.,emissivity_fields->log10_mturn_mcg_ave);
        get_uhmf_averages(M_min,M_max,mturn_acg,mturn_mcg,ev_consts,&target);
        double p[]={ev_consts->lx_over_sfr_acg,1.,FRACT_FLOAT_ERR,
                    target.n_ion,target.sfrd_acg,target.xray_emissivity};
        void *a[]={dens_pointer,emissivity_fields->n_ion,emissivity_fields->sfrd_acg,
                   emissivity_fields->xray_emissivity};
        GPU_CALL(0,p,a);
        free_conditional_tables();
        return 0;
    }
#endif
""".replace("EULERIAN_MODEL", EULERIAN_EMISSIVITY)
    return insert(
        src, "int add_integral_contribution(", "    move_integral_emissivities(", hook
    )


def ion_source():
    src = (SRC / "IonisationBox.c").read_text()
    # Nonthermal runs still need the RECFAST temperature and residual ionization.
    # The native calls take a float redshift, including when the driver supplies a double.
    src = src.replace(
        "consts->TK_nofluct = T_RECFAST(redshift, 0);",
        """
        double recfast[2], p[]={(float)redshift}; void *a[]={recfast};
        GPU_CALL(30,p,a); consts->TK_nofluct=recfast[1];
    """,
    )
    hook = """
    if (!consts->lagrangian_source_grids && !astro_options_global->USE_MCGS &&
        !astro_options_global->IONISE_ENTIRE_SPHERE) {
        bool ts=astro_options_global->USE_TS_FLUCT;
        bool rec=uses_recombination(astro_options_global->RECOMB_MODEL);
        double p[]={box_dim[2],rspec.R_index,rspec.R,consts->redshift,
            consts->photoncons_adjustment_factor,mean_fix_term_acg,nion_limit_acg,
            consts->mass_dep_zeta,consts->gamma_prefactor,consts->T_re,ts,rec,
            consts->filter_recombinations,ts?0.:consts->TK_nofluct,ts?0.:consts->adia_TK_term,
            astro_options_global->RECOMB_MODEL == RECOMB_MODEL_HOMOGENEOUS};
        void *a[]={perturbed_field->density,fg_struct->deltax_filtered,box->nion_conditional_filtered_acg,
            ts?(void*)fg_struct->xe_filtered:NULL,rec?(consts->filter_recombinations?
            (void*)fg_struct->N_rec_filtered:(void*)previous_ionize_box->cumulative_recombinations):NULL,
            previous_ionize_box->z_reion,ts?spin_temp->kinetic_temp_neutral:NULL,
            box->neutral_fraction,box->z_reion,
            (rec && !matter_options_global->MINIMIZE_MEMORY)?box->mean_free_path:NULL,
            rec?box->ionisation_rate_G12:NULL,matter_options_global->MINIMIZE_MEMORY?NULL:box->kinetic_temperature};
        GPU_CALL(2,p,a); return;
    }
"""
    src = insert(src, "void find_ionised_regions(", "#pragma omp parallel", hook)
    # Homogeneous MHR uses the stored redshift and all-volume means, including
    # neutral cells. Keep the upstream OpenMP sums (in particular float32 Gamma).
    src = insert(
        src,
        "// Homogeneous recombination rate",
        "        double dNrec_global",
        """
        double p[]={consts->stored_redshift,consts->fabs_dtdz,consts->dz};
        void *a[]={NULL,&global_photionization_rate,&global_xHI,
            previous_ionize_box->cumulative_recombinations,box->cumulative_recombinations};
        if (gpu_dispatch(32,1,p,a)) Throw(ValueError);
        return;
""",
    )
    hook = """
        double p[]={consts->stored_redshift,consts->fabs_dtdz,consts->dz};
        void *a[]={perturbed_field->density,box->ionisation_rate_G12,box->neutral_fraction,
            previous_ionize_box->cumulative_recombinations,box->cumulative_recombinations};
        GPU_CALL(3,p,a); return;
"""
    src = insert(
        src, "// Inhomogeneous recombination rate", "        bool finite_error", hook
    )
    # Unfiltered grids are private to ComputeIonizedBox. Their only reader is
    # copy_filter_transform, so this branch can leave Fourier coefficients on device.
    resident = """(source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)
        && !astro_options_global->USE_MCGS && !astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK
        && simulation_options_global->NON_CUBIC_FACTOR == 1. && !matter_options_global->USE_FFTW_WISDOM
        && astro_options_global->HII_FILTER >= 0 && astro_options_global->HII_FILTER <= 2)"""
    src = insert(
        src,
        "void prepare_box_for_filtering(",
        "    int i, j, k;",
        """
        double p[]={simulation_options_global->HII_DIM,const_factor,limit_min,limit_max,
          simulation_options_global->N_THREADS,RESIDENT}; void *a[]={input_box,output_c_box};
        GPU_CALL(11,p,a); return;
""".replace("RESIDENT", resident),
    )
    src = insert(
        src,
        "void copy_filter_transform(",
        "    memcpy(",
        """
        if (RESIDENT) {
            double p[]={box_dim[0],rspec.R_index,(float)rspec.R,consts->hii_filter,
                        simulation_options_global->N_THREADS,simulation_options_global->BOX_LEN};
            void *a[]={fg_struct->deltax_unfiltered,fg_struct->deltax_filtered,
                       astro_options_global->USE_TS_FLUCT ? fg_struct->xe_unfiltered : NULL,
                       astro_options_global->USE_TS_FLUCT ? fg_struct->xe_filtered : NULL,
                       consts->filter_recombinations ? fg_struct->N_rec_unfiltered : NULL,
                       consts->filter_recombinations ? fg_struct->N_rec_filtered : NULL};
            GPU_CALL(31,p,a); return;
        }
""".replace("RESIDENT", resident),
    )
    src = insert(
        src,
        "void clip_and_get_extrema(",
        "    double min_buf",
        """
        double p[]={simulation_options_global->HII_DIM,lower_limit,upper_limit};
        void *a[]={grid,grid_min,grid_max}; GPU_CALL(12,p,a); return;
""",
    )
    src = insert(
        src,
        "void calculate_conditional_nion(",
        "    double nion_total_acg",
        """
        if (!consts->lagrangian_source_grids && !astro_options_global->USE_MCGS &&
            !astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK) {
            double p[]={simulation_options_global->HII_DIM, -1.+FRACT_FLOAT_ERR,consts->filter_recombinations};
            void *a[]={fg_struct->deltax_filtered,astro_options_global->USE_TS_FLUCT?fg_struct->xe_filtered:NULL,
                consts->filter_recombinations?fg_struct->N_rec_filtered:NULL,
                box->nion_conditional_filtered_acg,&rspec->nion_conditional_ave_acg};
            GPU_CALL(13,p,a); rspec->nion_conditional_ave_mcg=0; return;
        }
""",
    )
    src = insert(
        src,
        "void set_ionized_temperatures(",
        "#pragma omp parallel",
        """
        if(!matter_options_global->MINIMIZE_MEMORY){
            bool ts=astro_options_global->USE_TS_FLUCT;
            double p[]={consts->stored_redshift,consts->T_re,ts?0.:consts->TK_nofluct,ts?0.:consts->adia_TK_term};
            void *a[]={box->z_reion,box->neutral_fraction,perturbed_field->density,
                ts?spin_temp->kinetic_temp_neutral:NULL,box->kinetic_temperature};
            GPU_CALL(16,p,a); return;
        }
""",
    )
    src = insert(
        src,
        "void set_unconditional_nion(",
        "    double nion_curr_acg",
        """
        if (!astro_options_global->USE_MCGS) {
            double result[2], z[]={c->redshift, simulation_options_global->Z_HEAT_MAX};
            gpu_global_pair(22,2,z,false,c->lnMmin,c->lnMmax_gl,mturn_acg,&c->scale_consts,result);
            curr_box->nion_unconditional_acg=result[0]; *nion_limit_acg=result[1];
            curr_box->nion_unconditional_mcg=0; *nion_limit_mcg=0;return;
        }
""",
    )
    src = insert(
        src,
        "double set_fully_neutral_box(",
        "    double global_xH",
        """
        {
            bool ts=astro_options_global->USE_TS_FLUCT;
            double m,recfast[2],zp[]={(float)consts->redshift}; void *history[]={recfast};
            if (!ts) GPU_CALL(30,zp,history);
            double p[]={ts,ts?0.:1.-recfast[0],ts?0.:consts->TK_nofluct,ts?0.:consts->adia_TK_term};
            void *a[]={ts?spin_temp->xray_ionised_fraction:NULL,
                ts?spin_temp->kinetic_temp_neutral:perturbed_field->density,
                box->neutral_fraction,matter_options_global->MINIMIZE_MEMORY?NULL:box->kinetic_temperature,&m};
            GPU_CALL(23,p,a);return m;
        }
""",
    )
    begin = src.index(
        "#pragma omp parallel shared(box)",
        src.index("            float global_photoionization_rate = 0;"),
    )
    stop = src.index("            if (isfinite(global_xH)", begin)
    src = (
        src[:begin]
        + """if(astro_options_global->RECOMB_MODEL != RECOMB_MODEL_HOMOGENEOUS){
        void *a[]={box->neutral_fraction,&global_xH};GPU_CALL(24,NULL,a);
    }else{
        GPU_CALL(24,NULL,NULL);
"""
        + src[begin:stop]
        + "}\n"
        + src[stop:]
    )
    return HEADER + src


def radiation_source():
    src = (SRC / "RadiationFields.c").read_text()
    src = insert(
        src,
        "void one_shell_filter(",
        "    int i, j, k;",
        """
    if ((filter_type==FILTER_SPHERICAL_SHELL_STRAIGHT_LINE ||
         filter_type==FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) &&
        HII_D_PARA==simulation_options_global->HII_DIM) {
        double p[]={simulation_options_global->HII_DIM,simulation_options_global->BOX_LEN,
                    (float)R_inner,(float)R_outer,simulation_options_global->N_THREADS,
                    filter_type==FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING,(float)R_star,
                    astro_options_global->USE_LYA_HEATING && astro_options_global->USE_X_RAY_HEATING};
        void *a[]={input_box,output_box,u_avg,f_avg}; GPU_CALL(p[5] ? 34 : 4,p,a);return;
    }
""",
    )
    src = insert(
        src,
        "void accumulate_radiation_shell(",
        "#pragma omp parallel",
        """
    if (astro_options_global->USE_LYA_HEATING && astro_options_global->USE_X_RAY_HEATING) {
        bool mini=astro_options_global->USE_MCGS;
        bool separate_lw=mini && astro_options_global->LYA_MULTIPLE_SCATTERING;
        double p[]={z_edge_factor,xray_R_factor,R_ct,astro_params_global->N_STEP_TS,
            rad_setup->lya_flux_continuum_prefactor_acg[R_ct],rad_setup->lya_flux_injected_prefactor_acg[R_ct],
            lya_flux_continuum_prefactor_mcg,lya_flux_injected_prefactor_mcg,
            mini ? rad_setup->lyw_flux_prefactor_acg[R_ct] : 0., mini ? rad_setup->lyw_flux_prefactor_mcg[R_ct] : 0.};
        void *a[]={rad_setup->filtered_sfrd_acg_for_lya,rad_setup->filtered_xray_emissivity,
            rad_setup->m_xHII_low_box,rad_setup->inverse_val_box,
            rad_setup->freq_int_heat_tbl,rad_setup->freq_int_heat_tbl_diff,
            rad_setup->freq_int_ion_tbl,rad_setup->freq_int_ion_tbl_diff,
            rad_setup->freq_int_lya_tbl,rad_setup->freq_int_lya_tbl_diff,
            radiation_fields->xray_heating_rate,radiation_fields->xray_ionization_rate,
            radiation_fields->xray_lya_flux,radiation_fields->lya_flux_continuum,radiation_fields->lya_flux_injected,
            mini ? rad_setup->filtered_sfrd_mcg_for_lya : NULL,mini ? radiation_fields->lyw_flux : NULL,
            separate_lw ? rad_setup->filtered_sfrd_acg_for_lw : NULL,
            separate_lw ? rad_setup->filtered_sfrd_mcg_for_lw : NULL};
        GPU_CALL(5,p,a);return;
    }
""",
    )
    src = insert(
        src,
        "void multiply_radiation_fields_by_constants(",
        "    index_huge box_ct;",
        """
        if (astro_options_global->USE_X_RAY_HEATING && astro_options_global->USE_LYA_HEATING) {
            double p[]={xray_prefactor*volunit_inv*2./3./physconst.k_B,xray_prefactor*volunit_inv,
                  xray_prefactor*volunit_inv*Nb_zp,lya_star_prefactor*volunit_inv,
                  lya_star_prefactor*volunit_inv*physconst.h_p*physconst.nu_Ly_alpha/
                  (physconst.nu_ion_HI-physconst.nu_LW_thresh)*1e21};
            void *a[]={radiation_fields->xray_heating_rate,radiation_fields->xray_ionization_rate,
               radiation_fields->xray_lya_flux,radiation_fields->lya_flux_continuum,
               radiation_fields->lya_flux_injected,previous_spin_temp->xray_ionised_fraction,perturbed_field->density,
               astro_options_global->USE_MCGS ? radiation_fields->lyw_flux : NULL};
            GPU_CALL(14,p,a); return;
        }
""",
    )
    return HEADER + src


def use_gpu_scalar_sigma(source):
    return "double gpu_sigma_z0(double);\n" + source.replace(
        "sigma_z0(", "gpu_sigma_z0("
    )


def shared_source_scaling(source):
    return (
        '#include "scaling_relations.h"\n'
        "void (*shared_scaling_constants)(double, ScalingConstants *, bool);\n"
        + source.replace("set_scaling_constants(", "shared_scaling_constants(")
    )


def bind_source_scaling(target, implementation):
    C.c_void_p.in_dll(target, "shared_scaling_constants").value = C.cast(
        implementation, C.c_void_p
    ).value


def radiation_setup_source():
    src = (SRC / "RadiationFieldsSetup.c").read_text()
    src += """
void tabulate_xhii_inverse_widths(float *widths, int threads) {
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < x_int_NXHII - 1; i++) {
        widths[i] = 1. / (x_int_XHII[i + 1] - x_int_XHII[i]);
    }
}
"""
    src = insert(
        src,
        "void calculate_spectral_factors(",
        "    double nuprime;",
        """
        {
            bool mini=astro_options_global->USE_MCGS;
            double p[]={zp,astro_params_global->N_STEP_TS,astro_params_global->POP2_ION,
                        physconst.nu_Ly_alpha,astro_options_global->USE_LYA_HEATING,mini,
                        astro_params_global->POP3_ION,astro_params_global->F_H2_SHIELD,
                        physconst.nu_LW_thresh/physconst.nu_Ly_alpha};
            void *a[]={rad_setup->zpp_avg,rad_setup->lya_flux_continuum_prefactor_acg,
                rad_setup->lya_flux_injected_prefactor_acg,rad_setup->lya_flux_continuum_injected_prefactor_acg,
                mini?rad_setup->lyw_flux_prefactor_acg:NULL,mini?rad_setup->lya_flux_continuum_prefactor_mcg:NULL,
                mini?rad_setup->lya_flux_injected_prefactor_mcg:NULL,mini?rad_setup->lya_flux_continuum_injected_prefactor_mcg:NULL,
                mini?rad_setup->lyw_flux_prefactor_mcg:NULL};
            GPU_CALL(21,p,a); return;
        }
""",
    )
    src = insert(
        src,
        "void fill_freqint_tables(",
        "#pragma omp parallel",
        """
        {
            double p[]={zp,rad_setup->x_e_ave_zp,physconst.c_cms*N_b0,
                cosmo_params_global->OMl/cosmo_params_global->OMm,
                // Native dtdz adds in float precision before calling sqrt.
                2.*sqrt(1 + cosmo_params_global->OMm/cosmo_params_global->OMl)/(3.*Ho),num_R,x_int_NXHII,x_int_NENERGY,
                physconst.nu_ion_HI/physconst.eV_to_Hz,physconst.nu_ion_HeI/physconst.eV_to_Hz,
                physconst.nu_ion_HeII/physconst.eV_to_Hz,physconst.h_p*physconst.eV_to_Hz,H_FRAC,HE_FRAC,
                astro_params_global->NU_X_THRESH,astro_params_global->NU_X_MAX,astro_params_global->X_RAY_SPEC_INDEX,
                physconst.eV_to_Hz,physconst.c_cms/(4.*M_PI)/physconst.nu_Ly_alpha/hubble(zp)};
            void *a[]={rad_setup->zpp_avg,rad_setup->freq_int_heat_tbl,rad_setup->freq_int_ion_tbl,rad_setup->freq_int_lya_tbl,
              rad_setup->freq_int_heat_tbl_diff,rad_setup->freq_int_ion_tbl_diff,rad_setup->freq_int_lya_tbl_diff};
            if (astro_options_global->USE_MCGS) {
                double enabled = pow(10.,log10(sc->mturn_acg_homogeneous)) <=
                                 sc->atomic_cooling_threshold*1.001;
                void *m[]={a[0],a[1],a[2],a[3],a[4],a[5],a[6],rad_setup->ave_log10_MturnLW,&enabled};
                GPU_CALL(33,p,m);
            } else {
                GPU_CALL(17,p,a);
            }
            return;
        }
""",
    )
    begin = src.index("        double x_e_ave_p")
    end = src.index("        LOG_DEBUG", begin)
    src = (
        src[:begin]
        + """
        double p[]={x_int_NXHII,simulation_options_global->N_THREADS};
        void *a[]={previous_spin_temp->xray_ionised_fraction,x_int_XHII,rad_setup->inverse_diff,
            rad_setup->m_xHII_low_box,rad_setup->inverse_val_box,&rad_setup->x_e_ave_zp};
        GPU_CALL(15,p,a);
"""
        + src[end:]
    )
    begin = src.index("#pragma omp parallel private(box_ct)", begin)
    end = src.index("    }  // End of try", begin)
    src = src[:begin] + src[end:]
    marker = "void setup_z_edges("
    src = src.replace(
        marker, (ROOT / "global_integrals.h").read_text() + "\n" + marker, 1
    )
    call = src.index(
        "        initialize_nion_unconditional_tables(zpp_interp_points_SFR,"
    )
    end = src.index(";", call) + 1
    original = src[call:end]
    atomic = original.replace(
        "initialize_nion_unconditional_tables", "gpu_global_table"
    ).replace("&sc);", "&sc, 0);")
    src = (
        src[:call]
        + "        if (astro_options_global->USE_MCGS || !uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES)) {\n"
        + original
        + "\n        } else {\n"
        + atomic
        + "\n        }"
        + src[end:]
    )
    return HEADER + src


def module(files, extra="", options=()):
    return cp.RawModule(
        code="\n".join(p.read_text(encoding="utf8") for p in files) + "\n" + extra,
        options=("--fmad=false",) + options,
    )


class InitialFieldsEngine:
    """Seeded initial conditions and perturbation arrays shared by both drivers."""

    def __init__(self):
        self.ic_fields = {}
        self.ic_vel_stacked = {}
        self.density_pair = None
        self.density_reads = 0
        self.device_registry = None
        self._pending_outputs = None
        self._record_pending = False
        self.error = None
        self.symbols = native_tables()
        self.grid = module(
            [SHARED / "glibc_math.cuh", SHARED / "ion.cu", ROOT / "evolution_grid.cu"],
            options=("-DRECOMB_V432",),
        )

        self.allocate = C.CDLL(str(NATIVE)).allocate_RGTable1D_f
        self.allocate.argtypes = [C.c_int, C.POINTER(Table)]
        self.class_power = None
        self.density_power = None
        self.spectrum = 0
        self.gaussian_mass_derivative = 0.0
        self.scalar_sigmas = {}
        self.moments = module(
            [SHARED / "table_qag.cuh", ROOT / "sigma_moments.cu"]
        ).get_function("moments")
        self.rule = cp.asarray(json.loads((ROOT / "gk61.json").read_text()))

    def initialize_transfer(self, p, a):
        cosmology = (SRC / "cosmology.c").read_text()
        struct = cosmology[
            cosmology.index("struct CosmoConstants {") : cosmology.index(
                "static struct CosmoConstants"
            )
        ]
        start = cosmology.index("void TFset_parameters() {")
        body = cosmology[start : cosmology.index("\n}", start) + 2]
        body = (
            body.replace(
                "void TFset_parameters() {",
                """
extern "C" __global__ void eh_constants(const double *p,CosmoConstants *out){
    CosmoConstants &cosmo_consts=*out;
    float m=p[0],h=p[1],b=p[2],nu=p[3];
    cosmo_consts.omhh=m*h*h;
    cosmo_consts.theta_cmb=p[4]/2.7;
    cosmo_consts.f_nu=fmax((double)(nu/m),1.e-10);
    cosmo_consts.f_baryon=fmax((double)(b/m),1.e-10);
    cosmo_consts.sigma_norm=p[5]?1.:19.739208802178716;
""",
            )
            .replace("cosmo_params_global->OMb", "b")
            .replace("cosmo_params_global->hlittle", "h")
        )
        body = re.sub(r"LOG_DEBUG\([^;]+;", "", body)
        body = re.sub(r"pow\((\w+), (-?\d+)\)", r"pow(\1, \2.0)", body)
        out = cp.empty(8, dtype="d")
        module([], struct + body).get_function("eh_constants")(
            (1,), (1,), (cp.asarray(np.ctypeslib.as_array(p, shape=(6,))), out)
        )
        out.get(out=view(a[0], out.size, out.dtype))
        self.class_power = self.density_power = None
        self.spectrum = int(p[9])
        self.gaussian_mass_derivative = p[10]
        self.scalar_sigmas.clear()
        if p[6]:
            from classy_power import ClassPower

            n = int(p[6])
            k = view(a[1], n, "d")
            tail = C.CDLL(str(NATIVE)).transfer_function_EH
            tail.argtypes, tail.restype = [C.c_double], C.c_double
            if p[7] and p[7] != n:
                raise ValueError(
                    "The native density and velocity transfers must share a k axis"
                )
            velocity = view(a[3], n, "d") if p[7] else None
            self.class_power = ClassPower(
                k,
                view(a[2], n, "d"),
                tail(k[-1]),
                velocity,
                suppress_streaming=bool(p[8]),
            )
            self.density_power = self.class_power
        elif self.spectrum:
            from analytic_power import AnalyticPower

            self.density_power = AnalyticPower(self.spectrum, *p[:3])

    def fill_power_moments(self, stage, p, a):
        nbin = int(p[0])
        c = view(self.symbols["cosmo_consts"], 8, "d")
        if stage == 28:
            # All power-spectrum parameters and the exact MtoR result
            # are part of the cache key. Repeated radii recur at every node.
            key = (c.tobytes(), p[1], p[2], p[3], p[6])
            if key in self.scalar_sigmas:
                view(a[0], 1, "d")[0] = self.scalar_sigmas[key]
                return
        args = cp.asarray(np.concatenate((c, np.array([p[1], p[2]]))))
        size = max(1, nbin)
        out = cp.empty(size, dtype="d")
        slope = cp.empty_like(out)
        moments = self.moments
        transfer_args = ()
        radii = np.uint64(0)
        if self.density_power:
            moments = self.density_power.moments
            transfer_args = self.density_power.args
        if nbin and (self.density_power or self.gaussian_mass_derivative):
            radii = cp.asarray(view(a[0], nbin, "d"))
        moments(
            (size,),
            (32,),
            (
                args,
                np.float64(p[3]),
                np.float64(p[4]),
                np.float64(p[5]),
                np.float64(p[6]),
                self.rule,
                out,
                slope,
                np.float64(self.gaussian_mass_derivative),
                radii,
                *transfer_args,
            ),
        )
        finite = cp.isfinite(out)
        if nbin:
            finite &= cp.isfinite(slope)
        if not bool(finite.all()):
            raise ValueError("Nonfinite CUDA mass variance or derivative")
        if stage == 28:
            value = out.item()
            view(a[0], 1, "d")[0] = self.scalar_sigmas[key] = value
        elif not nbin:
            c[7] = float((p[1] / out[0]) ** 2)
        else:
            for key, value in (
                ("Sigma_InterpTable", out),
                ("dSigmasqdm_InterpTable", slope),
            ):
                t = Table.from_address(self.symbols[key])
                if not t.allocated:
                    self.allocate(size, C.byref(t))
                if t.n != size:
                    raise ValueError(
                        f"Mass-variance table {key} has {t.n} rows, expected {size}"
                    )
                t.low, t.width = p[4], p[5]
                value.astype("f").get(out=np.ctypeslib.as_array(t.y, shape=(size,)))

    def fill_initial_fields(self, p, a):
        from ic_random import sample, initial_fields

        side = int(p[0])
        if int(p[3]) != self.spectrum or side != int(p[6]) or side % 2:
            raise ValueError(
                "CUDA IC requires even cubic grids and an initialized matter spectrum"
            )
        supplied = cp.asarray(view(a[1], side**3, "f")).reshape(side, side, side)
        external = bool(cp.any(supplied))
        if external:
            from fft_plan_cuda import transform

            weighted = supplied * np.float32(p[2]) / np.float32(side**3)
            # The CUDA Hessian kernel reads packed FFTW-order complex modes.
            modes = cp.ascontiguousarray(transform(weighted, side, False, int(p[7])))
        else:
            modes = sample(
                side,
                p[1],
                p[2],
                C.c_ulonglong.from_address(a[0]).value,
                int(p[7]),
                view(self.symbols["cosmo_consts"], 8, "d"),
                p[4],
                p[5],
                self.density_power,
            )
        prefix = "hires_" if p[10] else "lowres_"
        names = ["hires_density", "lowres_density"] + [
            prefix + "v" + axis + suffix for suffix in ("", "_2LPT") for axis in "xyz"
        ]
        streaming = None
        if p[12]:
            if not self.class_power or self.class_power.velocity is None:
                raise ValueError("No CLASS relative-velocity transfer for FLUCTS")
            streaming = (
                self.class_power,
                view(self.symbols["cosmo_consts"], 8, "d"),
                p[4],
                p[5],
                p[12],
            )
        for i, field in initial_fields(
            modes,
            int(p[8]),
            p[1],
            p[2],
            p[9],
            bool(p[10]),
            bool(p[11]),
            int(p[7]),
            self.grid.get_function("spectral_filter"),
            streaming,
        ):
            device = cp.ascontiguousarray(field if i or not external else supplied)
            if i < len(names):
                self.ic_fields[names[i]] = device
            if i == 11:
                self.ic_fields["lowres_vcb"] = device
            if i or not external:
                target = view(a[i + 1], device.size, device.dtype).reshape(device.shape)
                device.get(out=target)
            elif self._record_pending and self._pending_outputs is not None:
                self._pending_outputs[int(a[i + 1])] = device
            if (
                (i or not external)
                and self._record_pending
                and self._pending_outputs is not None
            ):
                self._pending_outputs[int(a[i + 1])] = device
        self.ic_vel_stacked.clear()


class Evolution(InitialFieldsEngine):
    def __init__(self, heat_cuda, nsc, audit, *, scattering_reuse=False):
        super().__init__()
        self.audit = audit
        self.scattering_reuse = scattering_reuse
        self.table_size = int(
            re.search(r"#define NDELTA (\d+)", (SRC / "interp_tables.c").read_text())[1]
        )
        self.counts = [0] * 35
        self.ion_spectra = {}
        self.ion_filtered = {}
        self.ion_state = {}
        self.ion_state_meta = {}
        self._ion_last_nion = None
        self._rad_setup_resident = {}
        self._xray_sec_cache = None
        self.ion_spectrum_reads = 0
        self.ion_residency_enabled = (
            logging.getLogger("py21cmfast.wrapper.outputs").getEffectiveLevel()
            > logging.DEBUG
        )
        self.recfast_spline = None
        self.table_directory = Path(p21.config["external_table_path"])
        self.normalized_emissivity = 0
        self.table_errors = {}
        self.term_errors = {}
        mass_module = module(
            [
                SHARED / "table_qag.cuh",
                ROOT / "hmf_device.cuh",
                ROOT / "source_integrals.cu",
                ROOT / "constant_global.cu",
            ]
        )
        self.mass_integral = mass_module.get_function("source_table")
        self.global_rows = mass_module.get_function("global_rows")
        self.constant_globals = mass_module.get_function("constant_global_integrals")
        self.global_integral = mass_module.get_function("global_source_table")
        self.rec_tables = None
        if heat_cuda is not None:
            self._initialize_thermal_kernels(heat_cuda, nsc)
        self.callback = CB(self.dispatch)

    def clear_ion_caches(self):
        self.ion_spectra.clear()
        self.ion_filtered.clear()
        self.ion_state.clear()
        self.ion_state_meta.clear()
        self._ion_last_nion = None

    def flush_ion_state(self, *, flush_host: bool = True):
        if flush_host and self.error is None:
            for ptr, arr_dev in self.ion_state.items():
                size, dtype = self.ion_state_meta[ptr]
                arr_dev.get(out=view(ptr, size, dtype).reshape(arr_dev.shape))
                if self._record_pending and self._pending_outputs is not None:
                    self._pending_outputs[ptr] = arr_dev
            if self._ion_last_nion is not None:
                ptr, size, arr_dev = self._ion_last_nion
                arr_dev.get(out=view(ptr, size, "f").reshape(arr_dev.shape))
                if self._record_pending and self._pending_outputs is not None:
                    self._pending_outputs[ptr] = arr_dev
        self.ion_state.clear()
        self.ion_state_meta.clear()
        self.ion_filtered.clear()
        self._ion_last_nion = None

    def _initialize_thermal_kernels(self, heat_cuda, nsc):
        self.xhii_inverse_widths = {}
        xr = module(
            [SHARED / "xray.cu", SHARED / "xray_quadrature.cu", ROOT / "tau.cu"],
            options=("-DXRAY_V432",),
        )
        self.xray = xr.get_function("xray_frequency_adaptive_coop")
        self.tau = xr.get_function("tau_cutoffs")
        heat_module = module(
            [SHARED / "glibc_math.cuh", SHARED / "update_stabilized.cu"],
            heat_cuda + (ROOT / "initial_heat.cu").read_text(),
        )
        self.heat = heat_module.get_function("evolution_heat")
        self.thermal_constants = heat_module.get_function("thermal_constants")
        self.initial_heat = heat_module.get_function("initial_heat")
        self.nsc = nsc
        path = self.table_directory / "Lyman_alpha_heating_table.dat"
        count = 101 * 101 * 51  # nT * nT * ngp in 4.3.2 heating_helper_progs.c
        cols = np.loadtxt(path, max_rows=count, ndmin=2)
        if cols.shape != (count, 2):
            raise ValueError(
                f"Expected {count} continuum/injected heating pairs in {path}"
            )
        self.lya = [cp.asarray(col.copy()) for col in cols.T]

    def initialize_secondaries(self):
        # Fill the same native ABI arrays used by the radiation kernels. Parsing
        # the fixed-layout files here also propagates missing/truncated custom
        # data as Python exceptions before writing any process-global tables.
        source = (SRC / "elec_interp.c").read_text()
        nodes = np.array(
            re.findall(r"x_int_XHII\[\d+\] = ([0-9.e+-]+);", source), dtype="f"
        )
        if nodes.size != 14:
            raise ValueError("Unknown 4.3.2 secondary-electron ionization grid")
        tables = []
        for node in nodes:
            name = (
                f"log_xi_{np.log10(float(node)):.1f}.dat"
                if node < 0.3
                else f"xi_{node:.3f}.dat"
            )
            path = self.table_directory / "x_int_tables" / name
            values = np.loadtxt(path, dtype="f", skiprows=3, max_rows=258, ndmin=2)
            if values.shape != (258, 9):
                raise ValueError(
                    f"Expected 258 rows of nine secondary-electron columns in {path}"
                )
            tables.append(values)
        grids = np.stack(tables)
        view(self.symbols["x_int_XHII"], nodes.size, "f")[:] = nodes
        # Native 4.3.2 overwrites the shared energy axis for each ionization row.
        view(self.symbols["x_int_Energy"], 258, "f")[:] = grids[-1, :, 0]
        for name, column in zip(
            (
                "x_int_fheat",
                "x_int_n_Lya",
                "x_int_nion_HI",
                "x_int_nion_HeI",
                "x_int_nion_HeII",
            ),
            (2, 4, 5, 6, 7),
            strict=True,
        ):
            view(self.symbols[name], nodes.size * 258, "f")[:] = grids[
                :, :, column
            ].ravel()

    def recfast(self, redshift):
        if self.recfast_spline is None:
            path = self.table_directory / "recfast_LCDM.dat"
            data = cp.asarray(
                np.loadtxt(path, dtype="f", max_rows=501, ndmin=2)[::-1].copy()
            ).astype("d")
            zgrid = cp.ascontiguousarray(data[:, 0])
            values = cp.ascontiguousarray(data[:, [1, 3]].T)
            if values.shape != (2, 501):
                raise ValueError("Unexpected RECFAST table dimensions")
            # GSL uses natural cubic splines; share the GPU tridiagonal solve with MHR.
            deriv = cp.empty_like(values)
            kernel = module(
                [SHARED / "table_qag.cuh"],
                (SHARED / "mhr.cu").read_text().replace("256", "512"),
            ).get_function("natural_derivatives")
            kernel((1,), (32,), (zgrid, values, np.int32(2), np.int32(501), deriv))
            self.recfast_spline = zgrid, values, deriv
        zgrid, values, deriv = self.recfast_spline
        z = np.float64(redshift)
        if not zgrid[0].item() <= z <= zgrid[-1].item():
            raise ValueError(f"Redshift {z} outside the native RECFAST table")
        idx = cp.clip(cp.searchsorted(zgrid, cp.asarray(z)) - 1, 0, 499)
        h = zgrid[idx + 1] - zgrid[idx]
        t = (z - zgrid[idx]) / h
        u = 1.0 - t
        return u * u * (
            (1.0 + 2.0 * t) * values[:, idx] + t * h * deriv[:, idx]
        ) + t * t * ((3.0 - 2.0 * t) * values[:, idx + 1] - u * h * deriv[:, idx + 1])

    def dispatch(self, stage, n, p, a, *, molecular=False):
        try:
            self.counts[stage] += 1

            def get(i, dtype="f", size=n):
                ptr = int(a[i]) if a[i] else 0
                if ptr:
                    arr_dev = self.ion_state.get(ptr)
                    if (
                        arr_dev is not None
                        and arr_dev.size == size
                        and arr_dev.dtype == cp.dtype(dtype)
                    ):
                        return arr_dev.reshape(-1)
                    arr_dev = self.ion_filtered.get(ptr)
                    if (
                        arr_dev is not None
                        and arr_dev.size == size
                        and arr_dev.dtype == cp.dtype(dtype)
                    ):
                        return arr_dev.reshape(-1)
                    arr_dev = self._rad_setup_resident.get(
                        (ptr, size, str(cp.dtype(dtype)))
                    )
                    if arr_dev is not None:
                        return arr_dev.reshape(-1)
                    if self.density_pair is not None and dtype == "f":
                        host, device = self.density_pair
                        if ptr == int(host.ctypes.data) and size == host.size:
                            self.density_reads += 1
                            return device.reshape(-1)
                    if self.device_registry is not None:
                        cached = self.device_registry.lookup_ptr(ptr, size, dtype)
                        if cached is not None:
                            return cached
                return cp.asarray(view(a[i], size, dtype))

            def put(i, x):
                x.get(out=view(a[i], x.size, x.dtype).reshape(x.shape))

            def put_record(i, x):
                put(i, x)
                if self._record_pending and self._pending_outputs is not None:
                    self._pending_outputs[int(a[i])] = x

            def ion_buf(i, dtype="f", size=n):
                ptr = int(a[i])
                buf = self.ion_state.get(ptr)
                if buf is None:
                    buf = get(i, dtype=dtype, size=size)
                    self.ion_state[ptr] = buf
                    self.ion_state_meta[ptr] = (size, dtype)
                return buf

            def radiation_grid(grids, i, dtype, metric):
                field = grids.get(a[i])
                if field is None:
                    return get(i, dtype)
                history.counts[metric] += 1
                history.counts[metric + "_bytes"] += field.nbytes
                return field.reshape(-1)

            def launch(kernel, args, size=n):
                kernel(((size + 255) // 256,), (256,), args)

            if stage == 0:
                args = []
                thermal = bool(a[2])
                names = ["Nion_conditional_table1D_acg"]
                if thermal:
                    names.append("SFRD_conditional_table1D_acg")
                for name in names:
                    t = Table.from_address(self.symbols[name])
                    if not t.allocated or not (1 < t.n < 1000000) or t.width <= 0:
                        raise ValueError("Invalid v4 emissivity lookup table")
                    args.extend(
                        [
                            cp.asarray(np.ctypeslib.as_array(t.y, shape=(t.n,))),
                            np.float64(t.low),
                            np.float64(t.width),
                            np.int32(t.n),
                        ]
                    )
                outs = [cp.empty(n, dtype="f") for _ in range(3 if thermal else 1)]
                if not thermal:
                    args.extend(
                        (np.uint64(0), np.float64(0), np.float64(0), np.int32(0))
                    )
                outputs = outs if thermal else (outs[0], np.uint64(0), np.uint64(0))
                normalize = bool(p[1])
                count = len(outs)
                blocks = min((n + 255) // 256, 1024)
                partials = cp.empty((count, blocks), dtype="d") if normalize else None
                self.grid.get_function("emission")(
                    (blocks,),
                    (256,),
                    (
                        np.int32(n),
                        get(0),
                        *args,
                        np.float64(p[0]),
                        *outputs,
                        np.float64(p[2] if normalize else 0.0),
                        partials if normalize else np.uint64(0),
                    ),
                )
                if normalize:
                    sums = cp.empty(count, dtype="d")
                    finish = self.grid.get_function("sum_double_pass2")
                    for j in range(count):
                        finish(
                            (1,),
                            (256,),
                            (partials[j], sums[j : j + 1], np.int32(blocks)),
                        )
                    launch(
                        self.grid.get_function("normalize_emission"),
                        (
                            np.int32(n),
                            *outputs,
                            sums,
                            *(np.float64(p[j]) for j in range(3, 6)),
                        ),
                    )
                    self.normalized_emissivity += 1
                for j, x in enumerate(outs, 1):
                    put_record(j, x)
            elif stage == 1:
                outs = [cp.empty(n, dtype="f") for _ in range(3)]
                inputs = [
                    get(i, "f" if i < 4 else "d") if a[i] else cp.zeros(n, dtype="d")
                    for i in range(9)
                ]
                terms = cp.empty((n, len(THERMAL_TERMS)), dtype="d") if a[12] else None
                lw = get(13, "d") if a[13] else np.uint64(0)
                j21 = cp.empty(n, dtype="f") if a[14] else None
                launch(
                    self.heat,
                    (
                        np.int32(n),
                        cp.asarray(np.ctypeslib.as_array(p, shape=(self.nsc,))),
                        *self.lya,
                        *inputs,
                        *outs,
                        terms if terms is not None else np.uint64(0),
                        lw,
                        j21 if j21 is not None else np.uint64(0),
                    ),
                )
                if terms is not None:
                    expected = view(a[12], terms.size, "d").reshape(terms.shape)
                    actual = terms.get()
                    for j, name in enumerate(THERMAL_TERMS):
                        diff = abs(actual[:, j] - expected[:, j])
                        row = self.term_errors.setdefault(
                            name, dict(cells=0, failed=0, max_abs=0.0)
                        )
                        row["cells"] += n
                        row["failed"] += int(
                            np.sum(~(diff <= 1e-8 + 1e-4 * abs(expected[:, j])))
                        )
                        row["max_abs"] = max(row["max_abs"], float(diff.max()))
                if j21 is not None:
                    put_record(14, j21)
                for j, x in enumerate(outs, 9):
                    put_record(j, x)
            elif stage == 2:
                side = int(p[0])
                padded = n // side * (2 * (side // 2 + 1))
                residency = self.ion_residency_enabled
                contiguous = int(
                    residency
                    and int(a[1]) in self.ion_filtered
                    and (not p[10] or int(a[3]) in self.ion_filtered)
                    and (not (p[11] and p[12]) or int(a[4]) in self.ion_filtered)
                )
                field_sizes = {} if contiguous else {1: padded, 3: padded}
                if not contiguous:
                    if p[12]:
                        field_sizes[4] = padded
                    elif p[15]:
                        field_sizes[4] = 1
                elif p[15]:
                    field_sizes[4] = 1
                dummy = cp.empty(1, dtype="f")
                args = []
                for i in range(12):
                    if not a[i]:
                        args.append(np.uint64(0) if i in (9, 11) else dummy)
                    elif residency and 7 <= i < 12:
                        args.append(ion_buf(i))
                    else:
                        args.append(get(i, size=field_sizes.get(i, n)))
                launch(
                    self.grid.get_function("ion_center"),
                    (
                        np.int32(n),
                        cp.asarray(
                            np.ctypeslib.as_array(p, shape=(18 if molecular else 16,))
                        ),
                        *args,
                        get(12) if molecular else np.uint64(0),
                        np.int32(contiguous),
                    ),
                )
                if not residency:
                    for i in range(7, 12):
                        if a[i]:
                            put_record(i, args[i])
            elif stage in (3, 32):
                if self.rec_tables is None:
                    raise RuntimeError("GPU recombination table was not initialized")
                out = cp.empty(n, dtype="f")
                flag = cp.zeros(1, dtype="i")
                launch(
                    self.grid.get_function("recombine"),
                    (
                        get(0) if a[0] else np.uint64(0),
                        get(1),
                        get(2),
                        get(3),
                        out,
                        *self.rec_tables,
                        np.int32(n),
                        *[np.float64(p[i]) for i in range(3)],
                        np.float64(np.float32(0.2)),
                        self.max_log,
                        flag,
                    ),
                )
                if flag.get()[0]:
                    raise ValueError("Nonfinite recombination")
                put_record(4, out)
            elif stage in (4, 34):
                from fft_plan_cuda import transform

                side = int(p[0])
                threads = int(p[4])
                history = active_emissivity_history.get()
                resident = None if history is None else history.shell_field(a[0], n)
                x = (get(0) if resident is None else resident).reshape((side,) * 3)
                if p[2] > 0 and side > 1:
                    f = None
                    if resident is not None and stage == 4:
                        f = history.take_shell_spectrum(a[0])
                    if f is None:
                        f = cp.ascontiguousarray(transform(x, side, False, threads))
                        if (
                            stage == 34
                            and resident is not None
                            and history.share_scattering_spectra
                        ):
                            f = history.retain_shell_spectrum(a[0], f)
                    window = self.grid.get_function("shell_window")
                    args = (
                        f,
                        np.int32(side),
                        np.float64(2 * np.pi / p[1]),
                        np.float64(p[2]),
                        np.float64(p[3]),
                    )
                    if stage == 34:
                        if not hasattr(self, "scattering_window"):
                            from lyman_scattering import build_window

                            self.scattering_window = build_window(
                                SRC, reuse=self.scattering_reuse
                            )
                        window = self.scattering_window
                        args += (np.float64(p[6]),)
                    launch(window, args, size=f.size)
                    result = transform(f, side, True, threads)
                else:
                    result = x
                result = cp.maximum(result, 0)
                retained = (
                    resident is not None
                    and p[7]
                    and history.retain_radiation_grid(
                        history.filtered_grids, a[1], result
                    )
                )
                # The innermost shell leaves the native scratch arrays readable.
                if retained and p[2] > 0:
                    history.counts["filter_downloads_avoided"] += 1
                    history.counts["filter_downloads_avoided_bytes"] += result.nbytes
                else:
                    put(1, result)
                view(a[2], 1, "d")[0] = x.mean(dtype="d").item()
                view(a[3], 1, "d")[0] = result.mean(dtype="d").item()
            elif stage == 5:
                history = active_emissivity_history.get()
                filtered = {} if history is None else history.filtered_grids
                accumulated = {} if history is None else history.accumulated_grids

                def filter_grid(i):
                    return radiation_grid(filtered, i, "f", "filtered_input_reuses")

                def accumulator(i):
                    return radiation_grid(
                        accumulated, i, "d", "accumulator_input_reuses"
                    )

                args = [filter_grid(0), filter_grid(1), get(2, "i"), get(3)]
                args += [get(i, "d", 14 * int(p[3])) for i in range(4, 10)]
                outputs = [(i, accumulator(i)) for i in range(10, 15)]
                mcg_sfr = filter_grid(15) if a[15] else np.uint64(0)
                lw = accumulator(16) if a[16] else None
                launch(
                    self.grid.get_function("accumulate"),
                    (
                        np.int32(n),
                        cp.asarray(np.ctypeslib.as_array(p, shape=(10,))),
                        *args,
                        *(value for _, value in outputs),
                        mcg_sfr,
                        lw if lw is not None else np.uint64(0),
                        filter_grid(17) if a[17] else np.uint64(0),
                        filter_grid(18) if a[18] else np.uint64(0),
                    ),
                )
                if lw is not None:
                    outputs.append((16, lw))
                for i, field in outputs:
                    if history is not None and history.retain_radiation_grid(
                        accumulated, a[i], field
                    ):
                        history.counts["accumulator_downloads_avoided"] += 1
                        history.counts["accumulator_downloads_avoided_bytes"] += (
                            field.nbytes
                        )
                    else:
                        put(i, field)
            elif stage == 6:
                from halo_subgrid import conditional_table

                conditional_table(
                    p,
                    self.mass_integral,
                    self.symbols,
                    self.rule,
                    self.allocate,
                    self.table_size,
                    self.table_errors if self.audit else None,
                )
            elif stage == 21:
                nr = int(p[1])
                if not hasattr(self, "lyman_shells"):
                    self.lyman_shells = module([ROOT / "lyman.cu"]).get_function(
                        "lyman_shells"
                    )
                    path = self.table_directory / "stellar_spectra.dat"
                    stars = np.loadtxt(path, dtype="f", max_rows=22, ndmin=2)
                    if stars.shape != (22, 5):
                        raise ValueError(
                            f"Expected 22 stellar spectral bands in {path}"
                        )
                    self.stars = cp.asarray(stars)
                    helper = (
                        (SRC / "heating_helper_progs.c")
                        .read_text()
                        .split("double frecycle(int n) {")[1]
                        .split("default:")[0]
                    )
                    self.recycle = cp.asarray(
                        [float(x) for x in re.findall(r"return ([.0-9]+);", helper)],
                        dtype="d",
                    )
                populations = 2 if p[5] else 1
                outputs = cp.empty((populations, 4, nr), dtype="d")
                self.lyman_shells(
                    (populations,),
                    (32,),
                    (
                        np.float64(p[0]),
                        np.int32(nr),
                        np.float64(p[2]),
                        np.float64(p[6]),
                        np.float64(p[3]),
                        np.float64(p[7]),
                        np.float64(p[8]),
                        self.stars,
                        self.recycle,
                        get(0, "d", nr),
                        outputs,
                    ),
                )
                for pop in range(populations):
                    fields = [0, 1] if p[4] else [2]
                    if p[5]:
                        fields.append(3)
                    for field in fields:
                        put(1 + 4 * pop + field, outputs[pop, field])
            elif stage == 20:
                self.rec_tables = build_mhr(
                    module([SHARED / "table_qag.cuh", SHARED / "mhr.cu"]),
                    self.rule,
                    p[0],
                    p[1],
                    p[2],
                )
                self.max_log = np.float64(self.rec_tables[0][-1].item() - 1e-7)
            elif stage == 27:
                self._rad_setup_resident.clear()
                pars = cp.asarray(np.ctypeslib.as_array(p, shape=(8 + int(p[1]),)))
                out = cp.empty(12, dtype="d")
                self.thermal_constants(
                    (1,), (1,), (np.float64(p[0]), pars[2:8], pars[8:], out)
                )
                if not bool(cp.isfinite(out).all()):
                    raise ValueError("Invalid CUDA thermal coefficients")
                put(0, out)
            elif stage == 26:
                rec = self.recfast(p[0])
                fields = [cp.empty(n, dtype="f") for _ in range(3)]
                launch(
                    self.initial_heat,
                    (
                        np.int32(n),
                        cp.asarray(np.ctypeslib.as_array(p, shape=(7,))),
                        rec,
                        get(0),
                        *fields,
                    ),
                )
                for i, field in enumerate(fields):
                    put_record(i + 1, field)
            elif stage == 30:
                put(0, self.recfast(p[0]))
            elif stage == 25:
                self.initialize_transfer(p, a)
            elif stage in (19, 28):
                self.fill_power_moments(stage, p, a)
            elif stage == 24:
                if a:
                    view(a[1], 1, "d")[0] = get(0).mean(dtype="d").item()
                else:
                    self.flush_ion_state()
            elif stage == 23:
                if p[0]:
                    neutral = (1.0 - get(0).astype("d")).astype("f")
                    mean = neutral.mean(dtype="d").item()
                    if a[3]:
                        put_record(3, get(1))
                else:
                    mean = p[1]
                    neutral = cp.full(n, mean, dtype="f")
                    if a[3]:
                        put_record(
                            3,
                            (p[2] * (1.0 + p[3] * get(1).astype("d"))).astype("f"),
                        )
                put_record(2, neutral)
                view(a[4], 1, "d")[0] = mean
            elif stage in (18, 22, 29):
                size = int(p[0])
                mass_kernel = int(p[4] if stage == 18 else p[1])
                if stage == 18:
                    t = Table.from_address(
                        self.symbols["SFRD_z_table_acg" if p[3] else "Nion_z_table_acg"]
                    )
                    if not t.allocated:
                        allocate = C.CDLL(str(NATIVE)).allocate_RGTable1D
                        allocate.argtypes = [C.c_int, C.c_void_p]
                        allocate(size, C.byref(t))
                    if size != t.n:
                        raise ValueError("Global table layout mismatch")
                    t.low, t.width = p[1], p[2]
                if mass_kernel:
                    rows = get(0, "d", 12 * size)
                elif stage == 18:
                    rows = cp.empty((size, 23), dtype="d")
                    self.global_rows(
                        ((size + 63) // 64,),
                        (64,),
                        (
                            np.int32(size),
                            np.float64(p[1]),
                            np.float64(p[2]),
                            np.int32(p[3]),
                            get(0, "d", 12),
                            get(1, "d", 23),
                            rows,
                        ),
                    )
                else:
                    rows = get(0, "d", 23 * size)
                sigma, slope = [
                    Table.from_address(self.symbols[k])
                    for k in ("Sigma_InterpTable", "dSigmasqdm_InterpTable")
                ]
                vals = [
                    cp.asarray(np.ctypeslib.as_array(table.y, shape=(table.n,)))
                    for table in (sigma, slope)
                ]
                out = cp.empty(size, dtype="d")
                kernel = self.constant_globals if mass_kernel else self.global_integral
                kernel(
                    (size,),
                    (32,),
                    (
                        rows,
                        *vals,
                        np.float64(sigma.low),
                        np.float64(sigma.width),
                        np.int32(sigma.n),
                        self.rule,
                        out,
                    ),
                )
                if not bool(cp.isfinite(out).all()):
                    raise ValueError("Global mass integral failed")
                if stage == 18:
                    out.get(out=view(C.cast(t.y, C.c_void_p).value, size, "d"))
                else:
                    put(1, out)
            elif stage in (17, 33):
                nr, ni, ne = map(int, (p[5], p[6], p[7]))
                pars = cp.asarray(np.ctypeslib.as_array(p, shape=(19,)))
                if self._xray_sec_cache is None:
                    species = [
                        "x_int_fheat",
                        "x_int_n_Lya",
                        "x_int_nion_HI",
                        "x_int_nion_HeI",
                        "x_int_nion_HeII",
                    ]
                    sec = cp.stack(
                        [
                            cp.asarray(view(self.symbols[key], ni * ne, "f"))
                            for key in species
                        ]
                    )
                    energy = cp.asarray(view(self.symbols["x_int_Energy"], ne, "f"))
                    xhii = cp.asarray(view(self.symbols["x_int_XHII"], ni, "f"))
                    self._xray_sec_cache = (sec, energy, xhii)
                sec, energy, xhii = self._xray_sec_cache
                out = cp.empty((3, ni, nr), dtype="d")
                intervals = cp.empty(out.shape, dtype="B")
                t = Table.from_address(self.symbols["Nion_z_table_acg"])
                history = cp.asarray(view(C.cast(t.y, C.c_void_p).value, t.n, "d"))
                molecular = np.uint64(0)
                if stage == 33 and view(a[8], 1, "d")[0]:
                    # Interpolate each redshift row at the turnover mass for this shell.
                    # The optical-depth kernel then interpolates the two adjacent rows in z.
                    table = Table2DDouble.from_address(self.symbols["Nion_z_table_mcg"])
                    if (
                        not table.allocated
                        or table.nx != t.n
                        or table.ny < 2
                        or table.xmin != t.low
                        or table.xstep != t.width
                        or table.ystep <= 0
                    ):
                        raise ValueError(
                            "Inconsistent native two-population photon histories"
                        )
                    grid = cp.asarray(
                        np.stack(
                            [
                                np.ctypeslib.as_array(
                                    table.values[j], shape=(table.ny,)
                                )
                                for j in range(t.n)
                            ]
                        )
                    )
                    turn = get(7, "d", nr)
                    left = cp.floor((turn - table.ymin) / table.ystep).astype("i")
                    if bool(cp.any((left < 0) | (left + 1 >= table.ny))):
                        raise ValueError(
                            "Molecular turnover outside the native photon history"
                        )
                    wy = (turn - (table.ymin + table.ystep * left)) / table.ystep
                    molecular = cp.ascontiguousarray(
                        ((1.0 - wy) * grid[:, left] + wy * grid[:, left + 1]).T
                    )
                cutoffs = cp.empty(nr, dtype="d")
                self.tau(
                    ((nr + 31) // 32,),
                    (32,),
                    (
                        pars,
                        get(0, "d", nr),
                        history,
                        molecular,
                        np.int32(nr),
                        np.int32(t.n),
                        np.float64(t.low),
                        np.float64(t.width),
                        cutoffs,
                    ),
                )
                if not bool(cp.isfinite(cutoffs).all()):
                    raise ValueError("CUDA optical depth root solve failed")
                self.xray(
                    (out.size,),
                    (16,),
                    (
                        pars,
                        cutoffs,
                        energy,
                        xhii,
                        sec,
                        np.int32(nr),
                        np.int32(ni),
                        np.int32(ne),
                        out,
                        intervals,
                    ),
                )
                if not bool(cp.all(intervals)) or not bool(cp.isfinite(out).all()):
                    raise ValueError("CUDA X-ray quadrature did not converge")
                out[2] *= p[18]
                tbl_size = ni * nr
                for i, grid in enumerate(out):
                    diff_grid = cp.ascontiguousarray(cp.diff(grid, axis=0))
                    padded_diff = cp.zeros((ni, nr), dtype="d")
                    padded_diff[:-1] = diff_grid
                    put(1 + i, grid)
                    put(4 + i, diff_grid)
                    self._rad_setup_resident[(int(a[1 + i]), tbl_size, "float64")] = (
                        grid
                    )
                    self._rad_setup_resident[(int(a[4 + i]), tbl_size, "float64")] = (
                        padded_diff
                    )
            elif stage == 15:
                x = get(0)
                bins, threads = int(p[0]), int(p[1])
                grid = get(1, size=bins)
                # The native vectorized float reciprocal depends on the OpenMP
                # partition. The ionization grid is fixed in 4.3.2.
                if threads not in self.xhii_inverse_widths:
                    host = np.empty(bins - 1, dtype=np.float32)
                    self.tabulate_xhii_inverse_widths(host, threads)
                    self.xhii_inverse_widths[threads] = host, cp.asarray(host)
                host, inverse = self.xhii_inverse_widths[threads]
                view(a[2], bins - 1, "f")[:] = host
                top = (grid[-1].astype("d") * 0.999).astype("f")
                bottom = (grid[0].astype("d") * 1.001).astype("f")
                fixed = cp.where(x < grid[0], bottom, cp.minimum(top, x))
                indices = (cp.searchsorted(grid, fixed, side="right") - 1).astype("i")
                frac = cp.ascontiguousarray(
                    (fixed - grid[indices]) * inverse[indices], dtype="f"
                )
                put(3, indices)
                put(4, frac)
                self._rad_setup_resident[(int(a[3]), n, "int32")] = indices
                self._rad_setup_resident[(int(a[4]), n, "float32")] = frac
                view(a[5], 1, "d")[0] = float(x.mean(dtype="d"))
            elif stage == 16:
                density = get(2)
                neutral_tk = (
                    get(3)
                    if a[3]
                    else (p[2] * (1.0 + p[3] * density.astype("d"))).astype("f")
                )
                residency = self.ion_residency_enabled and int(a[1]) in self.ion_state
                out_tk = ion_buf(4) if residency else get(4)
                data = [get(0), get(1), density, neutral_tk, out_tk]
                flag = cp.zeros(1, dtype=cp.int32)
                launch(
                    self.grid.get_function("ion_temperature"),
                    (
                        data[2],
                        data[1],
                        data[0],
                        data[3],
                        data[4],
                        np.int32(n),
                        np.float32(p[0]),
                        np.float32(p[1]),
                        np.float32(9e-8),
                        flag,
                    ),
                )
                if flag.item():
                    raise ValueError("Nonfinite ionized gas temperature")
                if not residency:
                    put_record(4, data[4])
            elif stage == 11:
                from fft_plan_cuda import transform

                side = int(p[0])
                data = (
                    cp.clip(get(0).astype("d") * p[1], p[2], p[3])
                    .astype("f")
                    .reshape((side,) * 3)
                )
                spectrum = cp.ascontiguousarray(transform(data, side, False, int(p[4])))
                spectrum *= np.float32(1.0 / n)
                if p[5]:
                    self.ion_spectra[int(a[1])] = spectrum
                else:
                    put(1, spectrum.view("f"))
            elif stage == 31:
                from fft_plan_cuda import transform

                side, index, radius, kind, threads, box_len = p[:6]
                side, threads = int(side), int(threads)
                width = 2 * (side // 2 + 1)
                residency = self.ion_residency_enabled
                for i in range(0, 12 if molecular else 6, 2):
                    if not a[i]:
                        continue
                    spectrum = self.ion_spectra[int(a[i])]
                    self.ion_spectrum_reads += 1
                    if index > 0:
                        spectrum = spectrum.copy()
                        launch(
                            self.grid.get_function("spectral_filter"),
                            (
                                spectrum,
                                np.int32(side),
                                np.float64(2 * np.pi / box_len),
                                np.float32(radius),
                                np.int32(kind),
                            ),
                            size=spectrum.size,
                        )
                    real = cp.ascontiguousarray(
                        transform(spectrum, side, True, threads)
                    )
                    if residency:
                        self.ion_filtered[int(a[i + 1])] = real
                    else:
                        padded = view(a[i + 1], side * side * width, "f").reshape(
                            side, side, width
                        )
                        padded[..., :side] = real.get()
            elif stage in (12, 13):
                side = int(p[0])
                ptr0 = int(a[0])

                def padded(idx):
                    return view(a[idx], n // side * (2 * (side // 2 + 1)), "f").reshape(
                        side, side, -1
                    )[:, :, :side]

                resident0 = self.ion_filtered.get(ptr0)
                if resident0 is not None:
                    density = resident0.reshape(-1)
                else:
                    native_density = padded(0)
                    density = cp.asarray(native_density)
                if stage == 12:
                    lo, hi = cp.stack([density.min(), density.max()]).get().tolist()
                    view(a[1], 1, "d")[0] = lo
                    view(a[2], 1, "d")[0] = hi
                    clipped = cp.clip(density, np.float32(p[1]), np.float32(p[2]))
                    if resident0 is not None:
                        self.ion_filtered[ptr0] = clipped
                    else:
                        native_density[:] = clipped.get()
                else:
                    density = cp.maximum(density, np.float32(p[1]))
                    if resident0 is not None:
                        self.ion_filtered[ptr0] = density
                    else:
                        native_density[:] = density.get()
                    if a[1]:
                        ptr1 = int(a[1])
                        res1 = self.ion_filtered.get(ptr1)
                        if res1 is not None:
                            self.ion_filtered[ptr1] = cp.clip(
                                res1.reshape(-1), 0, np.float32(0.999)
                            )
                        else:
                            electron = padded(1)
                            electron[:] = cp.clip(
                                cp.asarray(electron), 0, np.float32(0.999)
                            ).get()
                    if p[2]:
                        ptr2 = int(a[2])
                        res2 = self.ion_filtered.get(ptr2)
                        if res2 is not None:
                            self.ion_filtered[ptr2] = cp.maximum(res2.reshape(-1), 0)
                        else:
                            recomb = padded(2)
                            recomb[:] = cp.maximum(cp.asarray(recomb), 0).get()
                    t = Table.from_address(self.symbols["Nion_conditional_table1D_acg"])
                    vals = cp.asarray(np.ctypeslib.as_array(t.y, shape=(t.n,)))
                    pos = density.astype("d") - t.low
                    ix = cp.floor(pos / t.width).astype("i")
                    out_of_bounds = cp.any((ix < 0) | (ix >= t.n - 1))
                    frac = (density - (t.low + ix.astype("d") * t.width)) / t.width
                    safe_ix = cp.clip(ix, 0, max(0, t.n - 2))
                    photons = cp.exp(
                        vals[safe_ix] * (1.0 - frac) + vals[safe_ix + 1] * frac
                    )
                    mean_val, oob_flag = (
                        cp.stack([photons.mean(), out_of_bounds.astype("d")])
                        .get()
                        .tolist()
                    )
                    if oob_flag:
                        raise ValueError(
                            "Conditional ionization table index outside grid"
                        )
                    photons_f = cp.ascontiguousarray(photons.astype("f"))
                    if resident0 is not None and self.ion_residency_enabled:
                        ptr3 = int(a[3])
                        self.ion_filtered[ptr3] = photons_f
                        self._ion_last_nion = (ptr3, n, photons_f)
                    else:
                        put(3, photons_f)
                    view(a[4], 1, "d")[0] = mean_val
            elif stage == 14:
                history = active_emissivity_history.get()
                accumulated = {} if history is None else history.accumulated_grids
                factors = (
                    p[0] / (1.0 + get(5).astype("d")),
                    p[1],
                    p[2] * (1.0 + get(6).astype("d")),
                    p[3],
                    p[3],
                )
                for i, factor in enumerate(factors):
                    field = radiation_grid(
                        accumulated, i, "d", "accumulator_input_reuses"
                    )
                    put_record(i, field * factor)
                if a[7]:
                    field = radiation_grid(
                        accumulated, 7, "d", "accumulator_input_reuses"
                    )
                    put_record(7, field * p[4])
                accumulated.clear()
            elif stage == 10:
                self.fill_initial_fields(p, a)
            elif stage == 9:
                side, depth, kind = (int(p[i]) for i in range(3))
                if side != depth or kind not in (0, 1, 2):
                    raise ValueError(f"Unsupported spectral filter {kind} or shape")
                host = view(a[0], 2 * n, "f").view("complex64")
                freq = cp.asarray(host)
                launch(
                    self.grid.get_function("spectral_filter"),
                    (
                        freq,
                        np.int32(side),
                        np.float64(2 * np.pi / p[3]),
                        np.float32(p[4]),
                        np.int32(kind),
                    ),
                )
                freq.get(out=host)
            elif stage in (7, 8):
                from fft_plan_cuda import transform

                side, depth, threads, wisdom = (int(p[i]) for i in range(4))
                if depth != side or wisdom:
                    raise ValueError("Unsupported CUDA FFT geometry or wisdom")
                width = 2 * (side // 2 + 1)
                padded = view(a[0], side * side * width, "f").reshape(side, side, width)
                inverse = stage == 8
                data = padded.view("complex64") if inverse else padded[..., :side]
                result = transform(cp.asarray(data), side, inverse, threads).get()
                if inverse:
                    padded[..., :side] = result
                else:
                    padded.view("complex64")[:] = result
            return 0
        except BaseException as exc:
            self.error = exc
            return 1


def initial_conditions_source():
    original_ic = (SRC / "InitialConditions.c").read_text()
    begin = original_ic.index(
        "        fftwf_complex *HIRES_box =",
        original_ic.index("int ComputeInitialConditions("),
    )
    end = original_ic.index('        LOG_SUPER_DEBUG("Cleaned Up.', begin)
    return (
        HEADER
        + original_ic[:begin].replace("seed_rng_threads(r, random_seed);", "")
        + """

            double par[]={hi_dim[0],box_len[0],VOLUME,matter_options_global->POWER_SPECTRUM,
                cosmo_tables_global->ps_norm,cosmo_params_global->POWER_INDEX,hi_dim[2],
                simulation_options_global->N_THREADS,lo_dim[0],
                physconst.l_factor*simulation_options_global->BOX_LEN/(lo_dim[0]+0.0),
                matter_options_global->PERTURB_ON_HIGH_RES,
                matter_options_global->PERTURB_ALGORITHM == PERTURB_ALGORITHM_2LPT,
                matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS ? physconst.c_kms : 0.};
            void *args[]={&random_seed,boxes->hires_density,boxes->lowres_density,
                vel_pointers[0],vel_pointers[1],vel_pointers[2],
                vel_pointers_2LPT[0],vel_pointers_2LPT[1],vel_pointers_2LPT[2],
                    hires_v_2LPT[0],hires_v_2LPT[1],hires_v_2LPT[2],boxes->lowres_vcb};
            GPU_CALL(10,par,args);
    """
        + original_ic[end:]
    )


def compile_adapter(tmp, sources, options=()):
    return load_shim(tmp, "evolution.so", sources, SRC, NATIVE, options)


def install_entry(stack, engine, adapter, cls, name, cleanup=None):
    native = ffi.cast(
        ffi.typeof(getattr(lib, name)),
        C.cast(getattr(adapter, name), C.c_void_p).value,
    )

    def invoke(*args):
        try:
            status = native(*args)
            if engine.error:
                raise engine.error
            return status
        finally:
            if cleanup is not None:
                cleanup()

    stack.enter_context(patch.object(cls, "_c_compute_function", staticmethod(invoke)))


def install_power_setup(stack, adapter, engine):
    """Install the transfer and mass-variance initialization for this C-state scope."""
    from py21cmfast.drivers import _global_initialization as initialization

    previous = initialization.lib
    adapter.gpu_init_ps.argtypes = [C.c_void_p]
    adapter.gpu_sigma.argtypes = [C.c_float, C.c_float, C.c_int]
    ntab = int(
        re.search(
            r"#define N_MASS_INTERP (\d+)", (SRC / "interp_tables.c").read_text()
        )[1]
    )

    class PowerLibrary:
        def __getattr__(self, name):
            return getattr(previous, name)

        def init_ps(self):
            if adapter.gpu_init_ps(engine.symbols["cosmo_consts"]):
                raise engine.error or ValueError(
                    "CUDA power setup could not initialize the native matter spectrum"
                )

        def initialize_sigma_tables(self, lo, hi):
            if adapter.gpu_sigma(lo, hi, ntab):
                raise engine.error

    stack.enter_context(patch.object(initialization, "lib", PowerLibrary()))


@contextmanager
def evolution_backend(
    report, audit=False, *, thermal=True, photon_calibration=None, molecular=False
):
    """Use the Eulerian adapter, optionally omitting spin and radiation stages."""
    if version("21cmfast") != "4.3.2":
        raise ValueError("The native evolution adapter requires 21cmFAST 4.3.2")
    heat_src, cuda, nsc = thermal_source(audit) if thermal else (None, None, 0)
    engine = Evolution(cuda, nsc, audit, scattering_reuse=molecular)
    engine.eulerian_mcg_stages = {}
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        tmp = Path(tmp)
        sources = {
            "heat.c": heat_src,
            "rad_setup.c": radiation_setup_source() if thermal else None,
            "ic.c": initial_conditions_source(),
            "filter.c": HEADER
            + """
#include "filtering.h"
#include "InputParameters.h"
#include "exceptions.h"
void filter_box(fftwf_complex *grid,int dim[3],int kind,float radius,float param,float star) {
    double p[]={dim[0],dim[2],kind,simulation_options_global->BOX_LEN,radius};
    void *a[]={grid};
    if(gpu_dispatch(9,dim[0]*dim[1]*(dim[2]/2+1),p,a)) Throw(ValueError);
}
""",
            "fft.c": HEADER
            + '#include "dft.h"\n#include "exceptions.h"\n'
            + """
int dft_r2c_cube(bool w,int n,int los,int threads,fftwf_complex *grid) {
    double p[]={n,los,threads,w}; void *a[]={grid};
    if(gpu_dispatch(7,n,p,a)) Throw(ValueError);
    return 0;
}
int dft_c2r_cube(bool w,int n,int los,int threads,fftwf_complex *grid) {
    double p[]={n,los,threads,w}; void *a[]={grid};
    if(gpu_dispatch(8,n,p,a)) Throw(ValueError);
    return 0;
}""",
            "ion.c": ion_source(),
            "rad.c": radiation_source() if thermal else None,
            "map.c": emission_sources(),
            "emission.c": insert(
                (SRC / "EmissivityFields.c").read_text(),
                "int get_uhmf_averages(",
                "    // Compute n_ion",
                """
                if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)
                    && uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES)
                    && !astro_options_global->USE_MCGS && !astro_options_global->USE_METALLICITY
                    && !config_settings.EXTRA_EMISSIVITY_FIELDS) {
                    bool thermal=astro_options_global->USE_TS_FLUCT;
                    double result[2]={0}, z[]={consts->redshift,consts->redshift};
                    gpu_global_pair(29,thermal ? 2 : 1,z,thermal,lnMmin,lnMmax,M_turn_acg,consts,result);
                    *averages_out=(IntegralProperties){.n_ion=result[0], .sfrd_acg=result[1],
                        .xray_emissivity=consts->lx_over_sfr_acg*result[1]};
                    return 0;
                }
                """,
            ),
            "scaling.c": (SRC / "scaling_relations.c").read_text(),
            "dispatch.c": "int (*gpu_dispatch)(int,int,double*,void**);",
        }
        sources["emission.c"] = normalize_emissivity_source(sources["emission.c"])
        sources["moments.c"] = HEADER + (ROOT / "power_setup.c").read_text()
        for name, marker in (
            ("ion.c", "struct IonBoxConstants {"),
            ("emission.c", "void set_halo_properties("),
        ):
            src = (
                sources[name]
                .replace("initialize_nion_conditional_tables(", "gpu_init_nion(")
                .replace("initialize_sfrd_conditional_tables(", "gpu_init_sfr(")
            )
            if name == "emission.c":
                src = HEADER + src
            sources[name] = src.replace(
                marker,
                (ROOT / "global_integrals.h").read_text()
                + "\n"
                + (ROOT / "source_integrals.h").read_text()
                + "\n"
                + marker,
                1,
            )
        for name in ("ion.c", "scaling.c"):
            sources[name] = use_gpu_scalar_sigma(sources[name])
        if not thermal:
            for name in ("heat.c", "rad_setup.c", "rad.c"):
                del sources[name]
        if photon_calibration is not None:
            from photon_conservation import private_source

            sources["photon.c"] = private_source((SRC / "photoncons.c").read_text())
        if molecular:
            from eulerian_mcg_sources import source as molecular_emission_source
            from halo_mcg_global import source as global_photon_source

            sources["rad_setup.c"] = global_photon_source(sources["rad_setup.c"])
            sources["emission.c"] = molecular_emission_source(
                sources["emission.c"], audit=audit
            )
            import eulerian_mcg_history
            import eulerian_mcg_ion

            sources["ion.c"] = eulerian_mcg_history.source(
                eulerian_mcg_ion.source(sources["ion.c"])
            )
        adapter = compile_adapter(tmp, sources, [f"-DGPU_TABLE_AUDIT={int(audit)}"])
        if molecular:
            from eulerian_mcg_sources import bind as bind_molecular_emission
            from halo_mcg_global import bind as bind_global_photons

            _molecular_emission = bind_molecular_emission(
                adapter,
                engine.eulerian_mcg_stages,
                lambda exc: setattr(engine, "error", exc),
            )

            _eulerian_ion = eulerian_mcg_ion.bind(adapter, engine)
            _photon_history = eulerian_mcg_history.bind(
                adapter,
                engine.eulerian_mcg_stages,
                lambda exc: setattr(engine, "error", exc),
                engine=engine,
            )

            _global_photons = bind_global_photons(
                adapter,
                engine.eulerian_mcg_stages,
                lambda exc: setattr(engine, "error", exc),
            )
        if thermal:
            engine.tabulate_xhii_inverse_widths = adapter.tabulate_xhii_inverse_widths
            engine.tabulate_xhii_inverse_widths.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                C.c_int,
            ]
            engine.tabulate_xhii_inverse_widths.restype = None
        C.c_void_p.in_dll(adapter, "gpu_dispatch").value = C.cast(
            engine.callback, C.c_void_p
        ).value
        from py21cmfast.drivers import _global_initialization as initialization

        class InitLibrary:
            def __getattr__(self, name):
                return getattr(lib, name)

            def init_heat(self):
                if audit:
                    return lib.init_heat()
                if not thermal:
                    # Nonthermal ionization requests heat state solely for RECFAST.
                    # Stage 30 builds its spline on first use, independent of the
                    # secondary-electron tables consumed by X-ray heating.
                    return 0
                engine.initialize_secondaries()
                return 0

            def destruct_heat(self):
                if audit:
                    lib.destruct_heat()

            def init_MHR(self):
                if adapter.gpu_mhr():
                    raise engine.error

            def free_MHR(self):
                engine.rec_tables = None

        stack.enter_context(patch.object(initialization, "lib", InitLibrary()))
        install_power_setup(stack, adapter, engine)
        if photon_calibration is not None:
            from photon_conservation import install

            # Calibrating creates native-visible power/sigma state during
            # context entry, before the public driver's own cleanup is armed.
            if not initialization._GlobalInitManagerSingleton._scopes:
                stack.callback(initialization._GlobalInitManagerSingleton.free)
            engine.photon_data = install(stack, adapter, photon_calibration)
        entries = [
            (InitialConditions, "ComputeInitialConditions"),
            (EmissivityFields, "ComputeEmissivityFields"),
            (IonizedBox, "ComputeIonizedBox"),
        ]
        if thermal:
            entries.extend(
                [
                    (TsBox, "ComputeTsBox"),
                    (RadiationFields, "UpdateRadiationFields"),
                    (RadiationFieldsSetup, "SetupRadiationFields"),
                ]
            )

        def finalize_ionized():
            try:
                engine.flush_ion_state()
            finally:
                engine.clear_ion_caches()

        for cls, name in entries:
            install_entry(
                stack,
                engine,
                adapter,
                cls,
                name,
                cleanup=finalize_ionized if cls is IonizedBox else None,
            )
        try:
            yield engine
        finally:
            registry_stats = (
                None
                if engine.device_registry is None
                else engine.device_registry.stats()
            )
            report.with_name("resident-inputs.json").write_text(
                json.dumps(
                    {
                        "ic_fields": list(engine.ic_fields),
                        "density_reads_from_device": engine.density_reads,
                        "ionization_spectrum_reads_from_device": engine.ion_spectrum_reads,
                        "device_registry": registry_stats,
                    }
                )
            )
            report.with_name("thermal-terms.json").write_text(
                json.dumps(engine.term_errors, indent=2)
            )
            report.with_name("source-tables.json").write_text(
                json.dumps(engine.table_errors, indent=2)
            )
            report.write_text(
                json.dumps(
                    dict(
                        zip(
                            (
                                "emission",
                                "heat",
                                "ion_center",
                                "recombination",
                                "shell_filter",
                                "shell_accumulate",
                                "source_table",
                                "native_forward_fft_on_gpu",
                                "native_inverse_fft_on_gpu",
                                "native_spectral_filter_on_gpu",
                                "gpu_seed_and_modes",
                                "ion_filter_prepare",
                                "ion_filter_extrema",
                                "ion_photon_lookup",
                                "radiation_units",
                                "radiation_setup_grid",
                                "ionized_temperature",
                                "xray_frequency_integrals",
                                "global_source_integrals",
                                "sigma_moments",
                                "mhr_table",
                                "lyman_shells",
                                "ion_global_source_integrals",
                                "fully_neutral_box",
                                "global_neutral_fraction",
                                "eh_constants",
                                "initial_heat",
                                "thermal_constants",
                                "scalar_sigma",
                                "emissivity_global_source_integrals",
                                "ionization_recfast",
                                "ion_filter_resident_radius",
                                "homogeneous_recombination",
                                "mcg_xray_optical_depth_and_frequency",
                                "radiation_lya_scattering_filter",
                            ),
                            engine.counts,
                        )
                    )
                    | {"emissivity_mean_normalization": engine.normalized_emissivity}
                    | engine.eulerian_mcg_stages,
                    indent=2,
                )
            )
            if molecular:
                from halo_radiation import _STAGES

                counters = {
                    label: engine.counts[number] for number, label in _STAGES.items()
                }
                counters.update(
                    ionization_mhr_table=engine.counts[20],
                    ionization_cumulative_recombinations=engine.counts[3],
                    mcg_lw_shells=engine.counts[5],
                    mcg_spin_lw=engine.counts[1],
                    mcg_lw_normalizations=engine.counts[14],
                )
                counters.update(engine.eulerian_mcg_stages)
                if hasattr(engine, "scattering_window"):
                    counters["radiation_lya_scattering_window_reuses"] = (
                        engine.scattering_window.reuses
                    )
                report.with_name("backend-routing.json").write_text(
                    json.dumps(
                        dict(source_model="E-INTEGRAL", counters=counters), indent=2
                    )
                )


@contextmanager
def lagrangian_backend(report, source="CHMF-SAMPLER", *, inputs=None):
    """CUDA hooks for 4.3.2 Lagrangian integral and sampled-halo source models."""
    from halo_catalog import halo_catalog_backend
    from halo_grid import halo_grid_backend
    from halo_ionization import install_ionization, ionization_source

    stage_counts = {}
    engine = InitialFieldsEngine()
    engine.counters = stage_counts

    def prepare_initial_fields(stage, n, p, a):
        try:
            if stage == 10:
                engine.fill_initial_fields(p, a)
                key = "gpu_seed_and_modes"
            elif stage == 25:
                engine.initialize_transfer(p, a)
                if engine.class_power:
                    key = "gpu_class_transfer"
                elif engine.density_power:
                    key = "gpu_analytic_transfer"
                else:
                    key = "gpu_eh_transfer"
            elif stage == 19:
                engine.fill_power_moments(stage, p, a)
                key = "gpu_sigma_mass_table" if p[0] else "gpu_power_normalization"
            elif stage == 28:
                before = len(engine.scalar_sigmas)
                engine.fill_power_moments(stage, p, a)
                key = "gpu_source_scalar_sigma"
                if len(engine.scalar_sigmas) == before:
                    reused = "gpu_source_scalar_sigma_reuses"
                    stage_counts[reused] = stage_counts.get(reused, 0) + 1
            else:
                raise ValueError(f"Unexpected Lagrangian initialization stage {stage}")
            stage_counts[key] = stage_counts.get(key, 0) + 1
            return 0
        except BaseException as exc:
            engine.error = exc
            return 1

    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        callback = CB(prepare_initial_fields)
        adapter = compile_adapter(
            Path(tmp),
            {
                "ic.c": initial_conditions_source(),
                "dispatch.c": "int (*gpu_dispatch)(int,int,double*,void**);",
                "ion.c": use_gpu_scalar_sigma(ionization_source()),
                "scaling.c": use_gpu_scalar_sigma(
                    (SRC / "scaling_relations.c").read_text()
                ),
                "moments.c": HEADER + (ROOT / "power_setup.c").read_text(),
            },
        )
        C.c_void_p.in_dll(adapter, "gpu_dispatch").value = C.cast(
            callback, C.c_void_p
        ).value
        install_entry(
            stack, engine, adapter, InitialConditions, "ComputeInitialConditions"
        )
        install_power_setup(stack, adapter, engine)
        if source != "L-INTEGRAL":
            stack.enter_context(halo_catalog_backend(stage_counts))
        scaling = adapter.set_scaling_constants
        stack.enter_context(
            halo_grid_backend(stage_counts, scaling=scaling, engine=engine)
        )
        if inputs is not None and inputs.astro_options.USE_TS_FLUCT:
            from halo_radiation import halo_radiation_backend

            stack.enter_context(
                halo_radiation_backend(
                    inputs, stage_counts, scaling=scaling, parent_engine=engine
                )
            )
        install_ionization(
            stack,
            adapter,
            stage_counts,
            use_mhr=inputs is not None
            and inputs.astro_options.RECOMB_MODEL == "inhomogeneous",
            engine=engine,
        )

        try:
            yield engine
        finally:
            report.write_text(json.dumps(stage_counts, indent=2))
            native_stages = [
                "source scaling constants and redshift-dependent prefactors",
                "lightcone distances and shell selection",
            ]
            if source == "CHMF-SAMPLER":
                if not stage_counts.get("halo_inverse_cuda_rows"):
                    native_stages.append("GSL inverse-CDF roots")
                if stage_counts.get("halo_cuda_progenitor_conditions"):
                    native_stages.append("progenitor window-boundary fallback")
                    if not stage_counts.get("halo_cuda_progenitor_conditions_prepared"):
                        native_stages.append("progenitor condition setup")
                else:
                    native_stages.append("mass draws and merger decisions")
                if stage_counts.get("halo_cuda_grid_sample_conditions"):
                    native_stages.append("grid window-boundary fallback")
                    if not stage_counts.get("halo_cuda_grid_conditions_prepared"):
                        native_stages.append("initial-grid condition preparation")
                    if not stage_counts.get(
                        "halo_cuda_grid_sample_high_poisson_conditions"
                    ):
                        native_stages.append("grid Poisson means above ten")
                else:
                    native_stages.append("grid Poisson draws")
                if any(
                    stage_counts.get(f"{name}_high_poisson_conditions", 0)
                    > stage_counts.get(f"{name}_device_tangent_conditions", 0)
                    for name in ("halo_cuda_grid_sample", "halo_cuda_progenitor")
                ):
                    native_stages.append("scalar tangent for high-mean Poisson")
                native_stages.append(
                    "GSL uniform draws and polar rejection for CHMF properties; scalar libm check at ambiguous float rounding"
                )
            if inputs is not None:
                if inputs.matter_options.POWER_SPECTRUM == "CLASS":
                    native_stages.append(
                        "CLASS matter transfer and cosmological mean streaming speed"
                    )
                method = inputs.astro_options.INTEGRATION_METHOD_ACGS
                methods_on_device = ("GAUSS-LEGENDRE", "GSL-QAG", "GAMMA-APPROX")
                if inputs.astro_options.USE_MCGS:
                    mini_method = inputs.astro_options.INTEGRATION_METHOD_MCGS
                    if (
                        method not in methods_on_device
                        or mini_method not in methods_on_device
                    ):
                        native_stages.append(
                            f"two-dimensional conditional tables with ACG {method}, MCG {mini_method}"
                        )
                elif method not in methods_on_device:
                    native_stages.append(f"{method} conditional photon and SFRD tables")
                    if inputs.astro_options.USE_TS_FLUCT:
                        native_stages.append(f"{method} conditional X-ray tables")
            report.with_name("backend-routing.json").write_text(
                json.dumps(
                    {
                        "source_model": source,
                        "native_stages": native_stages,
                        "counters": stage_counts,
                    },
                    indent=2,
                )
            )


chmf_backend = lagrangian_backend
