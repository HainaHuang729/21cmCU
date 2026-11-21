"""Hook the original seed stream after it has produced Hermitian Fourier modes."""
def patch_ic(source):
    def splice(anchor, change):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError(anchor)
        source = source.replace(anchor, change, 1)

    anchor = "int ComputeInitialConditions("
    splice(anchor, """typedef int (*ICGPUHook)(const void *, int, int, double, double, float, float **);
static ICGPUHook ic_gpu_hook = NULL;
void SetICGPUHook(void *hook) { ic_gpu_hook = (ICGPUHook)hook; }

""" + anchor)
    anchor = "    adj_complex_conj(HIRES_box,user_params,cosmo_params);"
    splice(anchor, anchor + """
    int gpu_dense_ic = 0;
    if (ic_gpu_hook) {
        if (user_params->PERTURB_ON_HIGH_RES || user_params->NON_CUBIC_FACTOR != 1.0
            || !user_params->USE_2LPT || user_params->DIM % user_params->HII_DIM) Throw(ValueError);
        float *outputs[] = {boxes->hires_density, boxes->lowres_density,
            boxes->lowres_vx, boxes->lowres_vy, boxes->lowres_vz,
            boxes->lowres_vx_2LPT, boxes->lowres_vy_2LPT, boxes->lowres_vz_2LPT,
            boxes->hires_vx_2LPT, boxes->hires_vy_2LPT, boxes->hires_vz_2LPT};
        if (ic_gpu_hook(HIRES_box, user_params->DIM, user_params->HII_DIM, DELTA_K, VOLUME,
            L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0), outputs)) Throw(ValueError);
        gpu_dense_ic = 1;
    }
""")
    anchor = "    // FFT back to real space\n    int stat ="
    splice(anchor, "    if (gpu_dense_ic) goto ic_relative_velocity;\n" + anchor)
    anchor = "    // ******* Relative Velocity part ******* //"
    splice(anchor, "    ic_relative_velocity: ;\n" + anchor)
    anchor = '    LOG_DEBUG("Completed Relative velocities.");'
    splice(anchor, anchor + "\n    if (gpu_dense_ic) goto ic_finish;")
    anchor = "    fftwf_cleanup_threads();"
    splice(anchor, "    ic_finish: ;\n" + anchor)
    splice("int ComputeInitialConditions(", """static inline unsigned int ic_float_bits(float value) {
    unsigned int bits;
    memcpy(&bits,&value,sizeof(bits));
    return bits;
}
int ComputeInitialConditions(""")
    seed_start=source.index("    int num_int = INT_MAX/16;")
    seed_end=source.index("    gsl_ran_shuffle(",seed_start)
    old=source[seed_start:seed_end]
    splice(old,"""    const int num_int = INT_MAX/16;
    const bool cache_gsl_power = getenv("PS_GSL_EXACT_POWER") != NULL &&
        user_params->NON_CUBIC_FACTOR == 1.f && user_params->POWER_SPECTRUM == 6;
    unsigned int *many_ints = NULL;
    if (cache_gsl_power) {
        // The input population is exactly the integer interval [0,num_int).
        // Preserve the original uniform draws and selection order.
        int selected=0;
        for (int i=0; i<num_int && selected<user_params->N_THREADS; ++i) {
            if ((num_int-i)*gsl_rng_uniform(rseed)<user_params->N_THREADS-selected)
                seeds[selected++]=i;
        }
    } else {
        many_ints=malloc((size_t)num_int*sizeof(*many_ints));
        if (!many_ints) Throw(MemoryAllocError);
        for (int i=0;i<num_int;++i) many_ints[i]=i;
        gsl_ran_choose(rseed,seeds,user_params->N_THREADS,many_ints,num_int,sizeof(*seeds));
    }
""")
    anchor="    init_ps();\n\n#pragma omp parallel shared(HIRES_box,r)"
    splice(anchor,"""    init_ps();
    unsigned int *power_cache=NULL;
    unsigned int min_bits=0,max_bits=0;
    if (cache_gsl_power) {
        const float minimum=DELTA_K, maximum=MIDDLE*DELTA_K;
        min_bits=ic_float_bits((float)sqrt(minimum*minimum))-2;
        max_bits=ic_float_bits((float)sqrt(maximum*maximum+maximum*maximum+maximum*maximum))+2;
        if (max_bits>=min_bits && (size_t)(max_bits-min_bits)<(128U<<20))
            power_cache=calloc((size_t)max_bits-min_bits+1,sizeof(*power_cache));
    }

#pragma omp parallel shared(HIRES_box,r)""")
    splice("                    p = power_in_k(k_mag);\n\n                    // ok, now we can draw","""                    const unsigned int kbits=ic_float_bits(k_mag);
                    if (power_cache && kbits>=min_bits && kbits<=max_bits) {
                        unsigned int *slot=power_cache+(kbits-min_bits);
                        const unsigned int cached=__atomic_load_n(slot,__ATOMIC_RELAXED);
                        if (cached) memcpy(&p,&cached,sizeof(p));
                        else {
                            p=power_in_k(k_mag);
                            __atomic_store_n(slot,ic_float_bits(p),__ATOMIC_RELAXED);
                        }
                    } else p=power_in_k(k_mag);

                    // ok, now we can draw""")
    splice('    LOG_DEBUG("Drawn random fields.");','    free(power_cache);\n    LOG_DEBUG("Drawn random fields.");')
    return source
