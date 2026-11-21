extern "C" __global__ void velocity_gradient(
    const float2 *spectral, float2 *result, int side, double dk
) {
    int i=blockDim.x*blockIdx.x+threadIdx.x, h=side/2+1;
    if(i>=side*side*h)return;
    int z=i%h;
    float kz=(side%2==0 && z==side/2)?0.f:(float)(z*dk);
    float n=(float)(side*side*side);
    result[i]=make_float2((-spectral[i].y*kz)/n,(spectral[i].x*kz)/n);
}

extern "C" __global__ void brightness(
    const float *d,const float *neutral,const float *ts,const float *gradient,
    float *out,int n,float factor,float trad,float hubble,double limit,int rsd
) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n)return;
    float signal=factor*neutral[i]*(1.f+d[i]);
    signal*=1.f-trad/ts[i];
    if(rsd) {
       double dv=fmax(-limit,fmin(limit,(double)gradient[i]));
       signal=(float)(signal/(dv/hubble+1.));
    }
    out[i]=signal;
}
