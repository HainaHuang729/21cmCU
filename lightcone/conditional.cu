__device__ double conditional_integrand(double logmass, float delta, const double *p,
        const float *sigma, const float *derivative, int nsigma) {
    const double mass=exp(logmass);
    const float lm=(float)logmass;
    const int bin=min(nsigma-2,max(0,(int)floorf((lm-(float)p[0])*(float)p[2])));
    const float base=(float)p[0]+(float)p[1]*(float)bin;
    const float offset=(lm-base)*(float)p[2];
    const float s=sigma[bin]+offset*(sigma[bin+1]-sigma[bin]);
    const float deriv=derivative[bin]+offset*(derivative[bin+1]-derivative[bin]);
    const float s2=(float)p[4];
    float variance=s*s-s2*s2;
    if (variance<0.f) return 0.;
    if (variance==0.f) variance=1.e-6f;
    const float gap=(float)p[5]-delta;
    const float growth=(float)p[3];
    const double conditional=(gap/growth)*pow(10.,(double)deriv)*
         exp(-(double)(gap*gap)/(2.*growth*growth*variance))/(variance*sqrt((double)variance));
    const double star=(p[9]>0. && mass>p[13])||(p[9]<0. && mass<p[13]) ?
        1./p[11] : pow(mass/1.e10,p[9]);
    const double escape=(p[10]>0. && mass>p[14])||(p[10]<0. && mass<p[14]) ?
        1./p[12] : pow(mass/1.e10,p[10]);
    return mass*exp(-p[8]/mass)*star*escape*conditional*0.39894228040143267794;
}

__device__ float native_gl_integrand(float lnM, float delta, const double *p,
        const float *sigma, const float *derivative, int nsigma) {
    const float M = (float)exp((double)lnM);
    const float growthf = (float)p[3];
    const float sigma2 = (float)p[4];
    const float del1 = (float)p[5];
    const float del2 = delta;
    const float MassTurnover = (float)p[8];
    const float Alpha_star = (float)p[9];
    const float Alpha_esc = (float)p[10];
    const float Fstar10 = (float)p[11];
    const float Fesc10 = (float)p[12];
    const float Mlim_Fstar = (float)p[13];
    const float Mlim_Fesc = (float)p[14];

    const float Fstar = ((Alpha_star > 0.f && M > Mlim_Fstar) || (Alpha_star < 0.f && M < Mlim_Fstar)) ?
        (float)(1.0 / (double)Fstar10) : (float)pow((double)M / 1e10, (double)Alpha_star);
    const float Fesc = ((Alpha_esc > 0.f && M > Mlim_Fesc) || (Alpha_esc < 0.f && M < Mlim_Fesc)) ?
        (float)(1.0 / (double)Fesc10) : (float)pow((double)M / 1e10, (double)Alpha_esc);

    const float M1 = (float)log((double)M);
    const float min_mass = (float)p[0];
    const float bin_w = (float)p[1];
    const float inv_w = (float)p[2];
    const int MassBin = min(nsigma - 2, max(0, (int)floor((double)((M1 - min_mass) * inv_w))));
    const float MassBinLow = min_mass + bin_w * (float)MassBin;
    float sigma1 = sigma[MassBin] + (M1 - MassBinLow) * (sigma[MassBin + 1] - sigma[MassBin]) * inv_w;
    const float dsigma_val = derivative[MassBin] + (M1 - MassBinLow) * (derivative[MassBin + 1] - derivative[MassBin]) * inv_w;
    float dsigmadm = (float)(-pow(10.0, (double)dsigma_val));
    sigma1 = sigma1 * sigma1;
    const float s2 = sigma2 * sigma2;
    dsigmadm = (float)((double)dsigmadm / (2.0 * (double)sigma1));

    float dndm = 0.f;
    if (sigma1 > s2) {
        const float diff_d = del1 - del2;
        const float diff_d_sq = diff_d * diff_d;
        const float var = sigma1 - s2;
        dndm = (float)(-((double)(diff_d / growthf)) * (2.0 * (double)sigma1 * (double)dsigmadm) *
            exp(-(double)diff_d_sq / (2.0 * (double)growthf * (double)growthf * (double)var)) /
            pow((double)var, 1.5));
    } else if (sigma1 == s2) {
        const float diff_d = del1 - del2;
        const float diff_d_sq = diff_d * diff_d;
        dndm = (float)(-((double)(diff_d / growthf)) * (2.0 * (double)sigma1 * (double)dsigmadm) *
            exp(-(double)diff_d_sq / (2.0 * (double)growthf * (double)growthf * 1.e-6)) /
            pow(1.e-6, 1.5));
    }
    return (float)((double)M * exp(-(double)(MassTurnover / M)) * (double)Fstar * (double)Fesc *
        (double)dndm / sqrt(2.0 * 3.14159265358979323846));
}

extern "C" __global__ void conditional_table(const double *p,const float *sigma,
    const float *derivative,const float *density,const double *x,const double *w,
    const double *gl_x,const double *gl_w,float *out,int nsigma,int nq,int high,int heat) {
    const int i=blockIdx.x, lane=threadIdx.x;
    const float d=density[i];
    if ((!high && d>p[5]*.9999) || (high && d>p[5])) {
        if (!lane) out[i]=heat ? (high?1.e10f:(10.f*(float)log(10.0))) : (high?1.f:0.f);
        return;
    }
    __shared__ double values[128];
    if (high) {
        // Composite Gauss-Legendre with geometric subdivision near the high-mass endpoint.
        const int divisions=14, order=16;
        const double width=p[7]-p[6];
        double sum=0.;
        for (int j=lane;j<(divisions+1)*order;j+=blockDim.x) {
            const int piece=j/order, k=j%order;
            const double hi=ldexp(1.,-piece),lo=piece==divisions?0.:hi*.5;
            const double u=.5*((hi-lo)*gl_x[k]+hi+lo);
            const double mass=p[7]-width*u;
            sum+=.5*width*(hi-lo)*gl_w[k]*conditional_integrand(mass,d,p,sigma,derivative,nsigma);
        }
        values[lane]=sum;
        __syncthreads();
        for(int offset=blockDim.x/2;offset;offset/=2) {
            if (lane<offset) values[lane]+=values[lane+offset];
            __syncthreads();
        }
        if (lane==0) {
            out[i]=(float)(fmax(values[0],0.)*(heat?1.e10:1.));
        }
    } else if (nq <= 128) {
        if (lane < nq) {
            const float x1 = (float)p[6];
            const float x2 = (float)p[7];
            const double xm = 0.5 * (double)(x2 + x1);
            const double xl = 0.5 * (double)(x2 - x1);
            const float knot = (float)(xm + xl * x[lane]);
            const float weight = (float)((2.0 * xl) / w[lane]);
            values[lane] = (double)(weight * native_gl_integrand(knot, d, p, sigma, derivative, nsigma));
        }
        __syncthreads();
        if (lane == 0) {
            double sum = 0.;
            for (int j = 0; j < nq; ++j) sum += values[j];
            float v = (float)sum;
            if (fabs((double)v) < 1.e-38) v = 1.e-38f;
            v = (float)log10((double)v);
            if (heat) {
                v = (float)((double)v + 10.0);
            } else {
                if (v < -40.f) v = -40.f;
            }
            out[i] = v * (float)log(10.0);
        }
    } else {
        const float x1 = (float)p[6];
        const float x2 = (float)p[7];
        const double xm = 0.5 * (double)(x2 + x1);
        const double xl = 0.5 * (double)(x2 - x1);
        double sum=0.;
        for (int j=lane;j<nq;j+=blockDim.x) {
            const float knot=(float)(xm+xl*x[j]);
            const float weight=(float)((2.0*xl)/w[j]);
            sum+=(double)(weight*native_gl_integrand(knot,d,p,sigma,derivative,nsigma));
        }
        values[lane]=sum;
        __syncthreads();
        for(int offset=blockDim.x/2;offset;offset/=2) {
            if (lane<offset) values[lane]+=values[lane+offset];
            __syncthreads();
        }
        if (lane==0) {
            float v = (float)values[0];
            if (fabs((double)v) < 1.e-38) v = 1.e-38f;
            v = (float)log10((double)v);
            if (heat) {
                v = (float)((double)v + 10.0);
            } else {
                if (v < -40.f) v = -40.f;
            }
            out[i] = v * (float)log(10.0);
        }
    }
}
