"""Add fixed-configuration GPU dispatch and preserve the CPU reference branch."""
from pathlib import Path

def patch_ion(source):
    def insert(anchor, replacement):
        nonlocal source
        assert source.count(anchor)==1, anchor
        source=source.replace(anchor,replacement,1)

    insert("int ComputeIonizedBox(",Path(__file__).with_name("ion_hook.h").read_text()+"\nint ComputeIonizedBox(")
    anchor="        if(flag_options->USE_TS_FLUCT) {\n#pragma omp parallel shared(box,spin_temp)"
    insert(anchor,"""        if (ion_hook && flag_options->USE_TS_FLUCT &&
            flag_options->INHOMO_RECO && user_params->NON_CUBIC_FACTOR==1.0) {
            IonGPU dark = {0};
            dark.phase=5; dark.side=user_params->HII_DIM; dark.z=redshift;
            dark.xe=spin_temp->x_e_box; dark.tk=spin_temp->Tk_box;
            dark.xhi=box->xH_box; dark.temperature=box->temp_kinetic_all_gas;
            dark.nrec=box->dNrec_box;
            if (ion_hook(&dark)) Throw(ValueError);
        } else if(flag_options->USE_TS_FLUCT) {
#pragma omp parallel shared(box,spin_temp)""")
    anchor="        // Take the ionisation fraction from the X-ray ionisations from Ts.c (only if the calculate spin temperature flag is set)"
    insert("    // For recombinations", """    const bool gpu_ion = ion_hook && user_params->NON_CUBIC_FACTOR==1.0 &&
            flag_options->USE_TS_FLUCT && flag_options->INHOMO_RECO &&
            flag_options->USE_MASS_DEPENDENT_ZETA && user_params->USE_INTERPOLATION_TABLES &&
            !flag_options->USE_MINI_HALOS && !flag_options->USE_HALO_FIELD &&
            !flag_options->PHOTON_CONS && global_params.FIND_BUBBLE_ALGORITHM==2 &&
            global_params.HII_FILTER==1;
    // For recombinations""")
    insert(anchor, """        IonGPU gpu = {0};
        if (gpu_ion) {
            gpu.side=user_params->HII_DIM; gpu.threads=user_params->N_THREADS;
            gpu.z=redshift; gpu.prev_z=prev_redshift; gpu.dk=DELTA_K;
            gpu.cut=0.99*Deltac; gpu.crit=global_params.CRIT_DENS_TRANSITION;
            gpu.npoisson=global_params.N_POISSON; gpu.no_rng=user_params->NO_RNG;
            gpu.pixel=pixel_mass; gpu.mmin=M_MIN; gpu.eff=ION_EFF_FACTOR; gpu.tre=global_params.T_RE;
            gpu.density_limit=global_params.MIN_DENSITY_LOW_LIMIT;
            gpu.dt=fabs_dtdz; gpu.dz=ZSTEP; gpu.dz_table=RR_DEL_Z;
            gpu.max_log=RR_lnGamma_min + RR_DEL_lnGamma * (RR_lnGamma_NPTS - 1) - FRACT_FLOAT_ERR;
            gpu.density=perturbed_field->density; gpu.xe=spin_temp->x_e_box; gpu.tk=spin_temp->Tk_box;
            gpu.old_rec=previous_ionize_box->dNrec_box; gpu.old_z=previous_ionize_box->z_re_box;
            gpu.xhi=box->xH_box; gpu.nrec=box->dNrec_box; gpu.zre=box->z_re_box;
            gpu.mfp=box->MFP_box; gpu.gamma12=box->Gamma12_box;
            gpu.temperature=box->temp_kinetic_all_gas; gpu.fcoll=box->Fcoll;
            if (ion_hook(&gpu)) Throw(ValueError);
        } else {
"""+anchor)
    insert('        LOG_SUPER_DEBUG("deltax unfiltered calculated");','        }\n        LOG_SUPER_DEBUG("deltax unfiltered calculated");')
    anchor="            // Copy all relevant quantities from memory into new arrays to be smoothed and FFT'd."
    insert(anchor, """            if (gpu_ion) {
                gpu.phase=1; gpu.radius=R; gpu.last=LAST_FILTER_STEP;
                gpu.filtered=!LAST_FILTER_STEP || ((R - cell_length_factor *
                    (user_params->BOX_LEN/(double)user_params->HII_DIM)) > FRACT_FLOAT_ERR);
                if (ion_hook(&gpu)) Throw(ValueError);
            } else {
"""+anchor)
    anchor="            // Check if this is the last filtering scale.  If so, we don't need deltax_unfiltered anymore."
    insert(anchor,"            }\n"+anchor)
    anchor="                    min_density = max_density = 0.0;"
    insert(anchor,"""                    if (gpu_ion) {
                        min_density=gpu.lo; max_density=gpu.hi;
                    } else {
"""+anchor)
    anchor="                        InterpolationRange(1,R,user_params->BOX_LEN,&min_density, &max_density);"
    insert(anchor,"""                        InterpolationRange(1,R,user_params->BOX_LEN,&min_density, &max_density);""")
    anchor="                    if(user_params->USE_INTERPOLATION_TABLES) {\n                        InterpolationRange(1,R"
    insert(anchor,"                    }\n"+anchor)
    anchor="            // renormalize the collapse fraction so that the mean matches ST,"
    insert(anchor,"""            if (gpu_ion) {
                gpu.phase=2; gpu.low=log10_Nion_spline; gpu.high=Nion_spline;
                gpu.nl=NSFR_low; gpu.nh=NSFR_high;
                gpu.ls=overdense_small_min; gpu.li=overdense_small_bin_width_inv;
                gpu.hs=overdense_large_min; gpu.hi_inv=overdense_large_bin_width_inv;
                gpu.norm=box->mean_f_coll; gpu.fmin=f_coll_min;
                int lookup_status=ion_hook(&gpu);
                if(lookup_status==2 && !gpu_lookup_retried) {
                    gpu_lookup_retried = true;
                    goto reference_table;
                }
                if(lookup_status) Throw(ValueError);
                /* Collapse mean stays on the device for GPU assignment. */
            } else {
"""+anchor)
    anchor="""                sigmaMmax = sigma_z0(massofscaleR);
            }

            if (!flag_options->USE_HALO_FIELD) {"""
    insert(anchor,"""                sigmaMmax = sigma_z0(massofscaleR);
            }
            bool gpu_lookup_retried = false;
            reference_table:
            if (!flag_options->USE_HALO_FIELD) {""")
    anchor="            f_coll /= (double) HII_TOT_NUM_PIXELS;"
    split="            //////////////////////////////  MAIN LOOP THROUGH THE BOX"
    insert(split,"            }\n"+split)
    anchor="#pragma omp parallel shared(deltax_filtered,N_rec_filtered,xe_filtered,box,ST_over_PS,pixel_mass"
    insert(anchor,"""            if (gpu_ion) {
                gpu.phase=3; gpu.norm=box->mean_f_coll; gpu.gamma=Gamma_R_prefactor; gpu.fmin=f_coll_min;
                if (ion_hook(&gpu)) Throw(ValueError);
            } else {
"""+anchor)
    anchor='            LOG_SUPER_DEBUG("z_re_box after R=%f: ", R);'
    insert(anchor,"            }\n"+anchor)
    anchor="#pragma omp parallel shared(box,spin_temp,redshift,deltax_unfiltered_original,TK)"
    insert(anchor,"""        if (gpu_ion) {
            gpu.phase=4;
            if (ion_hook(&gpu)) Throw(ValueError);
        } else {
"""+anchor)
    anchor="        // find the neutral fraction\n        if (LOG_LEVEL >= DEBUG_LEVEL)"
    insert(anchor,"        }\n"+anchor)
    anchor="        // update the N_rec field\n        if (flag_options->INHOMO_RECO)"
    insert(anchor,"        // update the N_rec field\n        if (flag_options->INHOMO_RECO && !gpu_ion)")
    prepare="#pragma omp parallel shared(deltax_unfiltered,perturbed_field,adjustment_factor)"
    insert(prepare,"    if (!gpu_ion) {\n"+prepare)
    duplicate="    memcpy(deltax_unfiltered_original, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);"
    insert(duplicate,duplicate+"\n    }")
    rng="    for(thread_num = 0; thread_num < user_params->N_THREADS; thread_num++){"
    insert(rng,"    if (!gpu_ion) "+rng.lstrip())
    cleanup="    for (i=0; i<user_params->N_THREADS; i++) {\n        gsl_rng_free (r[i]);"
    insert(cleanup,"    if (!gpu_ion) "+cleanup.lstrip())

    sched_anchor = "        while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){"
    insert(sched_anchor, """        if (gpu_ion) {
            float ps_sched_r[128];
            int ps_sched_f[128];
            int ps_sched_n = 0;
            double ps_R = R;
            int ps_last = 0, ps_first = 1;
            while (!ps_last && (M_MIN < RtoM(ps_R)) && ps_sched_n < 128) {
                if ( ((ps_R/(global_params.DELTA_R_HII_FACTOR) - cell_length_factor*(user_params->BOX_LEN)/(float)(user_params->HII_DIM)) <= FRACT_FLOAT_ERR) ||
                     ((ps_R/(global_params.DELTA_R_HII_FACTOR) - global_params.R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
                    ps_last = 1;
                    ps_R = fmax(cell_length_factor*user_params->BOX_LEN/(double)(user_params->HII_DIM), global_params.R_BUBBLE_MIN);
                }
                ps_sched_r[ps_sched_n] = (float)ps_R;
                ps_sched_f[ps_sched_n] = !ps_last || ((ps_R - cell_length_factor * (user_params->BOX_LEN/(double)user_params->HII_DIM)) > FRACT_FLOAT_ERR);
                ps_sched_n++;
                if (ps_first) {
                    ps_R = stored_R;
                    ps_first = 0;
                } else {
                    ps_R /= (global_params.DELTA_R_HII_FACTOR);
                }
            }
            IonReferenceSetSchedule(ps_sched_n, ps_sched_r, ps_sched_f);
        }
""" + sched_anchor)
    quadrature="                    initialiseGL_Nion(NGL_SFR, M_MIN,massofscaleR);"
    insert(quadrature,
        "                    if (!gpu_ion || !conditional_hook || !(conditional_modes & (1 << 0)) ||\n"
        "                        (user_params->FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC))\n"
        "    "+quadrature)
    return source
