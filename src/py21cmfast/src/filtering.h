#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

#include "InputParameters.h"

void filter_box(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                float R_star);
int p21_cpu_filter_fused_enabled(void);
void p21_filter_boxes_fused(int n_boxes, fftwf_complex **dsts, fftwf_complex *const *srcs,
                            int box_dim[3], int filter_type, float R, float R_param, float R_star,
                            float pre_div);
int test_filter(float *input_box, double R, double R_param, double R_star, int filter_flag,
                double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);
void p21_get_ms_filter_params(double R_inner, double R_outer, double R_star,
                              double out_alpha_beta[4]);

#endif
