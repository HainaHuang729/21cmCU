extern "C" __global__ void scale_initial_modes(float2 *modes, int n, double dk,
    double volume, const double *p) {
    const long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    const int h=n/2+1;
    if (i >= (long)n*n*h) return;
    const int x=i/(n*h), y=(i/h)%n, z=i%h;
    const float kx=(float)((x>n/2?x-n:x)*dk);
    const float ky=(float)((y>n/2?y-n:y)*dk);
    const float kz=(float)(z*dk);
    const double k=sqrtf(kx*kx+ky*ky+kz*kz);
    if (i==0) { modes[i]=make_float2(0,0); return; }

    // Eisenstein-Hu transfer function, matching POWER_SPECTRUM=0.
    const double root_alpha=sqrt(p[2]);
    const double q=k*p[0]*p[0]/p[1];
    const double qeff=q/(root_alpha+(1.-root_alpha)/(1.+pow(.43*k*p[3],4.)));
    double transfer=log(2.718281828459045+1.84*p[4]*root_alpha*qeff);
    transfer /= transfer + qeff*qeff*(14.4+325./(1.+60.5*pow(qeff,1.11)));
    const double qnu=3.92*q/sqrt(p[5]/p[6]);
    transfer *= 1. + (1.2*pow(p[5],.64)*pow(p[6],.3+.6*p[5]))/
                    (pow(qnu,-1.6)+pow(qnu,.8));
    const double pk=pow(k,p[8]);
    const float power=(float)(pk*transfer*transfer*p[7]);
    const double amplitude=sqrt(volume*power*0.5);
    const float2 value=modes[i];
    modes[i]=make_float2((float)(value.x*amplitude),(float)(value.y*amplitude));
}

extern "C" __global__ void enforce_real_planes(float2 *modes, int n) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=2*n*n)return;
    const int h=n/2+1, q=index/2, z=(index%2)?n/2:0;
    const int x=q/n, y=q%n;
    const int target=q*h+z;
    if (x>0 && x<n/2) {
        const float2 value=modes[((n-x)*n+(n-y)%n)*h+z];
        modes[target]=make_float2(value.x,-value.y);
    } else if (x==0 || x==n/2) {
        if (y>0 && y<n/2) {
            const float2 value=modes[(x*n+n-y)*h+z];
            modes[target]=make_float2(value.x,-value.y);
        } else if (y==0 || y==n/2) modes[target].y=0.f;
    }
}
