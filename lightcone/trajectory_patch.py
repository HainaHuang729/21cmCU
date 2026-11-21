
def patch_trajectory(source):
    start = "int ComputeIonizedBox("
    declaration = """
static FILE *trajectory_file;
static int trajectory_coords[32][3];
static int trajectory_count;
void SetSmallTraceCoordinates(const int *coords, int n) {
    if (n > 32 || n < 0) return;
    memcpy(trajectory_coords, coords, n * sizeof(trajectory_coords[0]));
    trajectory_count = n;
}
float NativeTraceRecStep(float delta, float z, float gamma, float xhi, float prev, float dt, float dz) {
    float density = 1.0 + delta;
    float ze = pow(density, 1.0 / 3.0);
    ze *= (1 + z);
    double increment = splined_recombination_rate(ze - 1., gamma) * dt * dz * (1. - xhi);
    return prev + increment;
}
float NativeTracePartial(float raw, double norm, double floor, float density, float electron,
                        float pixel, float mmin, int draw, double npoisson, float efficiency) {
    double fc = norm * raw;
    float mass = fc * pixel * (1. + density);
    float ave = mass / mmin;
    if (fc <= floor) fc = floor;
    if (fc > 1.) fc = 1.;
    if (ave < npoisson)
        fc = draw * (mass / (float)npoisson) / (pixel * (1. + density));
    if (mass < mmin / 5.) fc = 0.;
    if (fc > 1.) fc = 1.;
    float neutral = 1. - fc * efficiency;
    neutral -= electron;
    return fminf(fmaxf(neutral,0.),1.);
}
void SetTrajectoryPath(const char *path) {
    trajectory_file=fopen(path,"w");
    if(trajectory_file) setvbuf(trajectory_file,NULL,_IOLBF,0);
}
"""
    assert source.count(start)==1
    source=source.replace(start,declaration+start)
    # Sample before the assignment, using the actual arrays and scale normalization.
    anchor="#pragma omp parallel shared(deltax_filtered,N_rec_filtered,xe_filtered,box,ST_over_PS,pixel_mass"
    assert source.count(anchor)==1
    before=r"""
            if(trajectory_file && !gpu_ion) {
                for (int sample=0; sample<trajectory_count; ++sample) {
                    const int a=trajectory_coords[sample][0], b=trajectory_coords[sample][1], c=trajectory_coords[sample][2];
                    const long i=HII_R_INDEX(a,b,c), k=HII_R_FFT_INDEX(a,b,c);
                    const double delta=((float*)deltax_filtered)[k];
                    const double electron=((float*)xe_filtered)[k];
                    const double rec=((float*)N_rec_filtered)[k];
                    const double collapse=box->Fcoll[counter * HII_TOT_NUM_PIXELS + i];
                    fprintf(trajectory_file,"scale %.9g %d %.9g %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                        redshift,sample,R,(int)LAST_FILTER_STEP,delta,electron,rec,collapse,
                        ST_over_PS, (double)ION_EFF_FACTOR, f_coll_min, (double)box->xH_box[i],
                        (double)box->Gamma12_box[i], (double)min_density, (double)max_density, Gamma_R_prefactor);
                }
            }
"""
    source=source.replace(anchor,before+anchor)
    anchor="    if (ion_observer && ion_observer("
    assert source.count(anchor)==1
    end=r"""
    if (trajectory_file) {
        for (int sample=0; sample<trajectory_count; ++sample) {
            const int a=trajectory_coords[sample][0], b=trajectory_coords[sample][1], c=trajectory_coords[sample][2];
            const long i=HII_R_INDEX(a,b,c);
            fprintf(trajectory_file,"step %.9g %d %.9g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                redshift,sample,prev_redshift,(double)box->xH_box[i],(double)box->dNrec_box[i],
                (double)box->Gamma12_box[i],(double)box->z_re_box[i],(double)box->MFP_box[i],
                (double)perturbed_field->density[i],(double)spin_temp->x_e_box[i],
                (double)spin_temp->Tk_box[i],(double)spin_temp->Ts_box[i]);
        }
    }
"""
    return source.replace(anchor,end+anchor)
