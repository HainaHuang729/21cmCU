"""Expose the reference stages and offload the 2LPT quadratic source."""
from ic_patch import patch_ic


def patch_reference(source):
    source = patch_ic(source)

    def edit(anchor, replacement):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError(anchor)
        source = source.replace(anchor, replacement, 1)

    anchor = 'static ICGPUHook ic_gpu_hook = NULL;'
    edit(anchor, anchor + """
typedef int (*ICStageHook)(int, int, int, int, const void *);
typedef int (*ICSourceHook)(int, int, float *, const float *, const float *, const float *);
static ICStageHook ic_stage_hook = NULL;
static ICSourceHook ic_source_hook = NULL;
void SetICStageHook(void *hook) { ic_stage_hook = (ICStageHook)hook; }
void SetICSourceHook(void *hook) { ic_source_hook = (ICSourceHook)hook; }
#define IC_TRACE(stage, axis, layout, buf) \\
    do { if (ic_stage_hook && ic_stage_hook(stage,axis,user_params->DIM,layout,buf)) Throw(ValueError); } while(0)
""")
    edit('        if (ic_gpu_hook(HIRES_box,', '        int ic_status = ic_gpu_hook(HIRES_box,')
    edit('outputs)) Throw(ValueError);\n        gpu_dense_ic = 1;',
         'outputs);\n        if (ic_status != 0 && ic_status != 2) Throw(ValueError);\n        gpu_dense_ic = ic_status == 0;')
    edit('    LOG_DEBUG("FFT\'d hires boxes.");', '    IC_TRACE(0,0,1,HIRES_box);\n    LOG_DEBUG("FFT\'d hires boxes.");')

    dft = 'dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, '
    for label, indent, marker in [
        (1, 8, "\n\n        // Renormalise"),
        (3, 8, "\n\n        // now sample to lower res"),
        (8, 12, "\n\n            // now sample to lower res"),
    ]:
        call = ' '*indent + dft + 'HIRES_box);'
        axis = '0' if label == 1 else 'ii'
        edit(call+marker, ' '*indent+f'IC_TRACE({label},{axis},0,HIRES_box);\n'+call+
             '\n'+' '*indent+f'IC_TRACE({label+1},{axis},1,HIRES_box);'+marker)

    phi = '            '+dft+'phi_1);'
    edit(phi+'\n\n            // Temporarily store',
         '            IC_TRACE(10,phi_component,0,phi_1);\n'+phi+'\n            IC_TRACE(5,phi_component,1,phi_1);\n\n            // Temporarily store')
    edit(phi+'\n\n            // Then we will',
         '            IC_TRACE(10,phi_component+3,0,phi_1);\n'+phi+'\n            IC_TRACE(5,phi_component+3,1,phi_1);\n\n            // Then we will')
    call = '        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_box);'
    edit(call,'        IC_TRACE(6,0,1,HIRES_box);\n'+call+'\n        IC_TRACE(7,0,0,HIRES_box);')

    anchor = '#pragma omp parallel shared(HIRES_box,phi_1,phi_component) private(i,j,k,component_ii,component_jj,component_ij)'
    edit(anchor, """            if (ic_source_hook && user_params->NON_CUBIC_FACTOR == 1.0) {
                const float *diagonal[] = {boxes->hires_vx_2LPT, boxes->hires_vy_2LPT, boxes->hires_vz_2LPT};
                const int a = phi_directions[phi_component][0], b = phi_directions[phi_component][1];
                if (ic_source_hook(user_params->DIM, phi_component, (float *)HIRES_box,
                    diagonal[a], diagonal[b], (float *)phi_1)) Throw(ValueError);
                continue;
            }
"""+anchor)
    return source
