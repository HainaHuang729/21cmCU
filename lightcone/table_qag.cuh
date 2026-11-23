// One warp per integral. Evaluate independent abscissae concurrently and keep
// the embedded rule's accumulation and interval-subdivision order on lane 0.
struct Panel { double lo, hi, area, error; };

template<class F>
__device__ Panel evaluate_panel(double lo, double hi, const F &f, const double *rule) {
    __shared__ double left[31], right[30];
    int lane = threadIdx.x;
    double mid = .5*(lo+hi), half = .5*(hi-lo);
    if (lane < 30) {
        double offset = half*rule[lane];
        left[lane] = f(mid-offset);
        right[lane] = f(mid+offset);
    } else if (lane == 30) {
        left[30] = f(mid);
    }
    __syncwarp();
    Panel result = {lo,hi,0.,0.};
    if (!lane) {
        double kron = left[30]*rule[61], gauss = 0., absolute = fabs(kron);
        for (int parity=1; parity>=0; --parity) {
            for (int i=parity; i<30; i+=2) {
                double pair = left[i]+right[i];
                kron += rule[31+i]*pair;
                absolute += rule[31+i]*(fabs(left[i])+fabs(right[i]));
                if (parity) gauss += rule[62+i/2]*pair;
            }
        }
        double mean = .5*kron, asc = rule[61]*fabs(left[30]-mean);
        for (int i=0; i<30; ++i)
            asc += rule[31+i]*(fabs(left[i]-mean)+fabs(right[i]-mean));
        asc *= half;
        absolute *= half;
        double error = fabs((kron-gauss)*half);
        if (asc && error) error = asc*fmin(1.,pow(200.*error/asc,1.5));
        if (absolute>2.2250738585072014e-308/1.1102230246251565e-14)
            error = fmax(error,1.1102230246251565e-14*absolute);
        result.area = kron*half;
        result.error = error;
    }
    __syncwarp();
    return result;
}

template<class F>
__device__ double table_integral(double lo, double hi, double tolerance,
                                 const F &f, const double *rule) {
    constexpr int limit=1000;
    __shared__ Panel rows[limit];
    __shared__ int selected;
    int count=1;
    Panel first=evaluate_panel(lo,hi,f,rule);
    double area=first.area, error=first.error;
    if (!threadIdx.x) rows[0]=first;
    while (true) {
        if (!threadIdx.x) {
            selected=-1;
            if (error>tolerance*fabs(area) && count<limit) {
                int k=0;
                for(int j=1;j<count;++j) if(rows[j].error>rows[k].error) k=j;
                selected=k;
            }
        }
        __syncwarp();
        int k=selected;
        if (k<0) break;
        Panel prev=rows[k];
        double mid=.5*(prev.lo+prev.hi);
        Panel left=evaluate_panel(prev.lo,mid,f,rule);
        Panel right=evaluate_panel(mid,prev.hi,f,rule);
        if (!threadIdx.x) {
            area+=left.area+right.area-prev.area;
            error+=left.error+right.error-prev.error;
            if (right.error>left.error) { Panel tmp=left; left=right; right=tmp; }
            rows[k]=left;
            rows[count]=right;
        }
        ++count;
    }
    double total=0.;
    if (!threadIdx.x) {
        if (!isfinite(error) || error>tolerance*fabs(area)) total=nan("");
        else for (int j=0;j<count;++j) total+=rows[j].area;
    }
    __syncwarp();
    return total;
}
