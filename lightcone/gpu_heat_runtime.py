"""Isolated resident-grid thermal backend. CPU table setup and other phases remain CPU."""
import ctypes as C, json, os, traceback, time
from pathlib import Path
import numpy as np
import cupy as cp
REPO_ROOT = Path(os.environ["PS_GPU_ROOT"]) if os.environ.get("PS_GPU_ROOT") else Path(__file__).resolve().parent.parent
def _cpu_has_fma():
    try:
        return " fma " in f" {Path('/proc/cpuinfo').read_text()} "
    except OSError:
        return False

GLIBC_MATH_FMA = 1 if _cpu_has_fma() else 0
GLIBC_MATH_HEADER = Path(__file__).with_name("glibc_math.cuh").read_text()
GLIBC_MATH_OPTS = ("--fmad=false", f"-DGLIBC_MATH_FMA={GLIBC_MATH_FMA}")
F=C.POINTER(C.c_float);DD=C.POINTER(C.c_double)
class Args(C.Structure):
 _fields_=[('n',C.c_long),('nr',C.c_int),('init',C.c_int),('use_cmb',C.c_int),('use_lya',C.c_int),('prev_z',C.c_double),('density',C.POINTER(F)),('min_density',F),('max_density',F),('growth',F),('sfr',F),('low',C.POINTER(F)),('high',C.POINTER(F)),('st',DD),('pref',DD),('cont_pref',DD),('inj_pref',DD),('heat',C.POINTER(DD)),('heat_diff',C.POINTER(DD)),('ion',C.POINTER(DD)),('ion_diff',C.POINTER(DD)),('lya',C.POINTER(DD)),('lya_diff',C.POINTER(DD)),('xgrid',F),('prev_x',F),('prev_t',F),('prev_ts',F),('out_x',F),('out_t',F),('out_ts',F),('scalars',C.c_double*24),('electron_sum',C.c_double)]
def arr(p,n):return np.ctypeslib.as_array(p,shape=(n,))
def mat(p,n,m):return np.stack([arr(p[i],m) for i in range(n)])
lookup_source=r'''extern "C" __global__ void lookup(const float*den,const float*lo,const float*hi,float*out,double*vals,int*bad,long n,int nl,int nh,float growth,float low_min,float low_inv,float high_min,float high_inv,float crit,double delta){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;float d=den[i]*growth,f=0,v;int k;if(d<crit){if(d<=-1.)f=0;else{v=(log10f(d+1.)-low_min)*low_inv;k=(int)floorf(v);if(k<0||k+1>nl-1){if(k==nl-1){if(fabs(d-crit)<1e-4){v=(d-high_min)*high_inv;k=(int)floorf(v);if(k<0||k+1>=nh){atomicExch(bad,1);return;}f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);}else f=expf(lo[k]);}else{atomicExch(bad,1);return;}}else f=expf(lo[k]*(1+(float)k-v)+lo[k+1]*(v-(float)k));}}else if(d<.99*delta){v=(d-high_min)*high_inv;k=(int)floorf(v);if(k<0||k+1>=nh){atomicExch(bad,1);return;}f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);}else f=pow(10.,10.);out[i]=(1.+d)*f;vals[i]=(double)f;}'''
# At the low-table upper endpoint, float log rounding can select the
# transition fallback while d remains just below high_min. Use the valid
# low-table endpoint only in this narrowly identified roundoff case.
# This is an explicit numerical policy change, requiring full validation.
needle='if(k<0||k+1>=nh){atomicExch(bad,1);return;}f=hi[k]'
replacement='if(k==-1 && d<high_min && fabs(d-high_min)<1e-6){f=expf(lo[nl-1]);out[i]=(1.+d)*f;vals[i]=(double)f;return;}if(k<0||k+1>=nh){atomicExch(bad,1);return;}f=hi[k]'
assert needle in lookup_source
lookup_source=lookup_source.replace(needle,replacement,1)
for code in (11,12,13):
 lookup_source=lookup_source.replace('atomicExch(bad,1);', 'if(atomicCAS(bad,0,'+str(code)+')==0) printf("LOOKUP_BAD code='+str(code)+' i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\\n",i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);',1)
assert "log10f(d+1.)" in lookup_source
lookup_source = lookup_source.replace("log10f(d+1.)", "glibc_log10f((float)((double)d+1.))")
assert lookup_source.count("expf(") == 3, lookup_source.count("expf(")
lookup_source = lookup_source.replace("expf(", "glibc_expf(")
lookup_source = GLIBC_MATH_HEADER + "\n" + lookup_source
lookup = cp.RawKernel(lookup_source, "lookup", options=GLIBC_MATH_OPTS)
lookup_sum_source = GLIBC_MATH_HEADER + "\n" + r"""
extern "C" __global__ void lookup_sum(
    const float*den,const float*lo,const float*hi,float*out,double*partials,int*bad,
    long n,int nl,int nh,float growth,float low_min,float low_inv,
    float high_min,float high_inv,float crit,double delta){
  double acc=0.;
  for(long i=(long)blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    float d=den[i]*growth,f=0,v;int k;int ok=1;
    if(d<crit){
      if(d<=-1.)f=0;
      else{
        v=(glibc_log10f((float)((double)d+1.))-low_min)*low_inv;
        k=(int)floorf(v);
        if(k<0||k+1>nl-1){
          if(k==nl-1){
            if(fabs(d-crit)<1e-4){
              v=(d-high_min)*high_inv;
              k=(int)floorf(v);
              if(k==-1 && d<high_min && fabs(d-high_min)<1e-6){
                f=glibc_expf(lo[nl-1]);
                out[i]=(1.+d)*f;
                acc+=(double)f;
                continue;
              }
              if(k<0||k+1>=nh){
                if(atomicCAS(bad,0,11)==0) printf("LOOKUP_BAD code=11 i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
                ok=0;
              }else f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);
            }else f=glibc_expf(lo[k]);
          }else{
            if(atomicCAS(bad,0,12)==0) printf("LOOKUP_BAD code=12 i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
            ok=0;
          }
        }else f=glibc_expf(lo[k]*(1+(float)k-v)+lo[k+1]*(v-(float)k));
      }
    }else if(d<.99*delta){
      v=(d-high_min)*high_inv;
      k=(int)floorf(v);
      if(k<0||k+1>=nh){
        if(atomicCAS(bad,0,13)==0) printf("LOOKUP_BAD code=13 i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
        ok=0;
      }else f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);
    }else f=pow(10.,10.);
    if(ok){out[i]=(1.+d)*f;acc+=(double)f;}
  }
  __shared__ double sh[256];
  sh[threadIdx.x]=acc;
  __syncthreads();
  for(int s=128;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
  if(threadIdx.x==0)partials[blockIdx.x]=sh[0];
}
"""
lookup_sum = cp.RawKernel(lookup_sum_source, "lookup_sum", options=GLIBC_MATH_OPTS)
accum_source=r'''extern "C" __global__ void accum(const float*fc,const int*idx,const float*v,const double*tbl,double*h,double*ion,double*lya,double*star,double*cont,double*inj,const double*sum,long n,double st,float sfr,float tstar,double before,double cbefore,double ibefore){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;double ave=(*sum)/(1e10*(double)n);if(ave==0.0)return;double df=(1./ave/1e10)*st*(double)sfr/(double)tstar;double pref=before*df;double x=(double)fc[i],r=(double)v[i];const double*q=tbl+6*idx[i];h[i]+=df*x*(q[1]*r+q[0]);ion[i]+=df*x*(q[3]*r+q[2]);lya[i]+=df*x*(q[5]*r+q[4]);star[i]+=x*pref;cont[i]+=x*(cbefore*df);inj[i]+=x*(ibefore*df);}'''
accum=cp.RawKernel(accum_source,'accum',options=('--fmad=false',))
lookup_sum_batch_source = GLIBC_MATH_HEADER + "\n" + r'''
extern "C" __global__ void lookup_sum_batch(
    const float*den,const unsigned long long*lo_ptrs,const unsigned long long*hi_ptrs,
    float*fc_all,double*partials_all,int*bad,
    long n,int nr,int nl,int nh,
    const float*growths,const float*low_mins,const float*low_invs,
    float high_min,float high_inv,float crit,double delta){
  int j=(int)blockIdx.y;
  if(j>=nr)return;
  const float*den_j=den+(long)j*n;
  const float*lo=(const float*)lo_ptrs[j];
  const float*hi=(const float*)hi_ptrs[j];
  float*out=fc_all+(long)j*n;
  float growth=growths[j],low_min=low_mins[j],low_inv=low_invs[j];
  double acc=0.;
  for(long i=threadIdx.x+(long)blockDim.x*blockIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    float d=den_j[i]*growth,f=0,v;int k;int ok=1;
    if(d<crit){
      if(d<=-1.)f=0;
      else{
        v=(glibc_log10f((float)((double)d+1.))-low_min)*low_inv;
        k=(int)floorf(v);
        if(k<0||k+1>nl-1){
          if(k==nl-1){
            if(fabs(d-crit)<1e-4){
              v=(d-high_min)*high_inv;
              k=(int)floorf(v);
              if(k==-1 && d<high_min && fabs(d-high_min)<1e-6){
                f=glibc_expf(lo[nl-1]);
                out[i]=(1.+d)*f;
                acc+=(double)f;
                continue;
              }
              if(k<0||k+1>=nh){
                if(atomicCAS(bad,0,11)==0) printf("LOOKUP_BAD code=11 j=%d i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",j,i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
                ok=0;
              }else f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);
            }else f=glibc_expf(lo[k]);
          }else{
            if(atomicCAS(bad,0,12)==0) printf("LOOKUP_BAD code=12 j=%d i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",j,i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
            ok=0;
          }
        }else f=glibc_expf(lo[k]*(1+(float)k-v)+lo[k+1]*(v-(float)k));
      }
    }else if(d<.99*delta){
      v=(d-high_min)*high_inv;
      k=(int)floorf(v);
      if(k<0||k+1>=nh){
        if(atomicCAS(bad,0,13)==0) printf("LOOKUP_BAD code=13 j=%d i=%ld d=%.12g v=%.12g k=%d low_min=%.12g low_inv=%.12g high_inv=%.12g\n",j,i,(double)d,(double)v,k,(double)low_min,(double)low_inv,(double)high_inv);
        ok=0;
      }else f=hi[k]*(1.+(float)k-v)+hi[k+1]*(v-(float)k);
    }else f=pow(10.,10.);
    if(ok){out[i]=(1.+d)*f;acc+=(double)f;}
  }
  __shared__ double sh[256];
  sh[threadIdx.x]=acc;
  __syncthreads();
  for(int s=128;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
  if(threadIdx.x==0)partials_all[(long)j*256+blockIdx.x]=sh[0];
}
'''
lookup_sum_batch = cp.RawKernel(lookup_sum_batch_source, "lookup_sum_batch", options=GLIBC_MATH_OPTS)
finish_sum_batch = cp.RawKernel(r'''extern "C" __global__ void finish_sum_batch(const double*partials_all,double*totals_all,const double*st,const float*sfr,float tstar,const double*pref_arr,const double*cbefore_arr,const double*ibefore_arr,double*df_arr,double*pref_df_arr,double*cont_df_arr,double*inj_df_arr,long n){int j=(int)blockIdx.x;const double*x=partials_all+(long)j*256;__shared__ double v[256];v[threadIdx.x]=x[threadIdx.x];__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0){double tot=v[0];totals_all[j]=tot;double ave=tot/(1e10*(double)n);if(ave==0.0){df_arr[j]=0.0;pref_df_arr[j]=0.0;cont_df_arr[j]=0.0;inj_df_arr[j]=0.0;}else{double df=(1./ave/1e10)*st[j]*(double)sfr[j]/(double)tstar;df_arr[j]=df;pref_df_arr[j]=pref_arr[j]*df;cont_df_arr[j]=cbefore_arr[j]*df;inj_df_arr[j]=ibefore_arr[j]*df;}}}''', "finish_sum_batch", options=('--fmad=false',))
accum_batch = cp.RawKernel(r'''extern "C" __global__ void accum_batch(const float* __restrict__ fc_all,const int* __restrict__ idx,const float* __restrict__ v,const double* __restrict__ gtab,double* __restrict__ h_out,double* __restrict__ ion_out,double* __restrict__ lya_out,double* __restrict__ star_out,double* __restrict__ cont_out,double* __restrict__ inj_out,const double* __restrict__ df_arr,const double* __restrict__ pref_df_arr,const double* __restrict__ cont_df_arr,const double* __restrict__ inj_df_arr,long n,int nr){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;int ix=idx[i];double r=(double)v[i];double h=0.,ion=0.,lya=0.,star=0.,cont=0.,inj=0.;for(int j=nr-1;j>=0;--j){double df=df_arr[j];if(df==0.0)continue;double x=(double)fc_all[(long)j*n+i];double dfx=df*x;const double*q=gtab+((long)j*14+ix)*6;h+=dfx*(q[1]*r+q[0]);ion+=dfx*(q[3]*r+q[2]);lya+=dfx*(q[5]*r+q[4]);star+=x*pref_df_arr[j];cont+=x*cont_df_arr[j];inj+=x*inj_df_arr[j];}h_out[i]=h;ion_out[i]=ion;lya_out[i]=lya;star_out[i]=star;cont_out[i]=cont;inj_out[i]=inj;}''', "accum_batch", options=('--fmad=false',))
prep=cp.RawKernel(r'''extern "C" __global__ void prep(const float*xe,const float*xgrid,int*ix,float*iv,long n){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;float x=xe[i];if(x>xgrid[13]*.999)x=xgrid[13]*.999;else if(x<xgrid[0])x=1.001*xgrid[0];int j=13;while(x<xgrid[j])--j;ix[i]=j;float inverse=1./(xgrid[j+1]-xgrid[j]);iv[i]=(x-xgrid[j])*inverse;}''','prep',options=('--fmad=false',))
update_source = GLIBC_MATH_HEADER + "\n" + Path(__file__).with_name("update_stabilized.cu").read_text()
update = cp.RawKernel(update_source, "update", options=GLIBC_MATH_OPTS)
electron_sum=cp.RawKernel(r'''extern "C" __global__ void esum(const float*x,double*out,long n){__shared__ double v[256];double t=0.;for(long i=(long)blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x)t+=(double)x[i];v[threadIdx.x]=t;__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)out[blockIdx.x]=v[0];}''','esum')
finish_sum=cp.RawKernel(r'''extern "C" __global__ void finish(const double*x,double*out,int n){__shared__ double v[256];double t=0.;for(int i=threadIdx.x;i<n;i+=256)t+=x[i];v[threadIdx.x]=t;__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)out[0]=v[0];}''','finish')
class Runtime:
 def __init__(self):
  for k in (lookup,lookup_sum,lookup_sum_batch,finish_sum_batch,accum,accum_batch,prep,update,electron_sum,finish_sum):
   if os.environ.get('PS_VERBOSE_HEAT','0')=='1':
    print('KERNEL_READY',k.name,k.attributes,flush=True)
  table=np.loadtxt(REPO_ROOT/'resident_filter_20261118/fix/package/py21cmfast/_data/Lyman_alpha_heating_table.dat');self.tc=cp.asarray(table[:,0]);self.ti=cp.asarray(table[:,1]);self.states={};self.density=None;self.records=[];self.error=None;self.previous_mean_calls=0;self.previous_mean_host_fallbacks=0;self.upload_bytes=table.nbytes;self.download_bytes=0;self.ws_n=0;self.ws_nr=0
 def _ensure_ws(self,n,nr=40):
  if self.ws_n!=n:
   self.ws_gi=cp.empty(n,dtype=cp.int32);self.ws_gv=cp.empty(n,dtype=cp.float32)
   self.ws_bad=cp.empty(1,dtype=cp.int32);self.ws_relaxed=cp.empty(1,dtype=cp.uint32)
   self.ws_fc=cp.empty(n,dtype=cp.float32);self.ws_rates=cp.empty((6,n),dtype=cp.float64)
   self.ws_partials=cp.empty(256,dtype=cp.float64);self.ws_etotal=cp.empty(1,dtype=cp.float64);self.ws_etotal_t=cp.empty(1,dtype=cp.float64)
   self.ws_gx=cp.empty(14,dtype=cp.float32);self.ws_gsc=cp.empty(24,dtype=cp.float64)
   self._ws_pinned_mem=cp.cuda.alloc_pinned_memory(3*n*4)
   self.ws_pinned_out=np.frombuffer(self._ws_pinned_mem,dtype=np.float32,count=3*n).reshape(3,n)
   self._ws_p_scal_mem=cp.cuda.alloc_pinned_memory(32)
   self.ws_p_bad=np.frombuffer(self._ws_p_scal_mem,dtype=np.int32,count=1,offset=0)
   self.ws_p_relaxed=np.frombuffer(self._ws_p_scal_mem,dtype=np.uint32,count=1,offset=4)
   self.ws_p_etot=np.frombuffer(self._ws_p_scal_mem,dtype=np.float64,count=1,offset=8)
   self.ws_p_etot_t=np.frombuffer(self._ws_p_scal_mem,dtype=np.float64,count=1,offset=16)
   self.ws_n=n;self.ws_nr=0
  if self.ws_nr!=nr:
   self.ws_fc_all=cp.empty((nr,n),dtype=cp.float32)
   self.ws_partials_all=cp.empty((nr,256),dtype=cp.float64)
   self.ws_totals_all=cp.empty(nr,dtype=cp.float64)
   self.ws_df=cp.empty(nr,dtype=cp.float64);self.ws_pref_df=cp.empty(nr,dtype=cp.float64)
   self.ws_cont_df=cp.empty(nr,dtype=cp.float64);self.ws_inj_df=cp.empty(nr,dtype=cp.float64)
   self.ws_gtab=cp.empty((nr,14,6),dtype=cp.float64)
   nr8=((nr+1)//2)*2
   # Contiguous pinned+device metadata blob holding gx, gsc, lo/hi ptrs, and row params
   meta_f64_elems=nr*14*6 + 24 + 2*nr + 3*nr + 7 + (4*nr8)//2
   meta_bytes=(24 + 7 + 5*nr + nr8//2*0 + 5*nr)*8
   o0=0; o1=o0+24*8
   o2=o1+14*4
   o3=o2+nr*8; o4=o3+nr*8
   o5=o4+nr*8; o6=o5+nr*8; o7=o6+nr*8
   o8=o7+nr*8; o9=o8+nr*4; o10=o9+nr*4; o11=o10+nr*4
   total_meta=((o11+nr*4)+63)&~63
   self._ws_meta_bytes=total_meta
   self._ws_meta_mem=cp.cuda.alloc_pinned_memory(total_meta)
   self.ws_meta_dev=cp.empty(total_meta,dtype=cp.uint8)
   mb=np.frombuffer(self._ws_meta_mem,dtype=np.uint8,count=total_meta)
   self.ws_gsc_h=mb[o0:o0+24*8].view(np.float64); self.ws_gx_h=mb[o1:o1+14*4].view(np.float32)
   self.ws_lo_p=mb[o2:o2+nr*8].view(np.uint64); self.ws_hi_p=mb[o3:o3+nr*8].view(np.uint64)
   self.ws_st_h=mb[o4:o4+nr*8].view(np.float64); self.ws_pref_h=mb[o5:o5+nr*8].view(np.float64)
   self.ws_cont_h=mb[o6:o6+nr*8].view(np.float64); self.ws_inj_h=mb[o7:o7+nr*8].view(np.float64)
   self.ws_gr_h=mb[o8:o8+nr*4].view(np.float32); self.ws_lm_h=mb[o9:o9+nr*4].view(np.float32)
   self.ws_li_h=mb[o10:o10+nr*4].view(np.float32); self.ws_sfr_h=mb[o11:o11+nr*4].view(np.float32)
   dm=self.ws_meta_dev
   self.ws_gsc=dm[o0:o0+24*8].view(cp.float64); self.ws_gx=dm[o1:o1+14*4].view(cp.float32)
   self.ws_lo_ptrs=dm[o2:o2+nr*8].view(cp.uint64); self.ws_hi_ptrs=dm[o3:o3+nr*8].view(cp.uint64)
   self.ws_st=dm[o4:o4+nr*8].view(cp.float64); self.ws_pref=dm[o5:o5+nr*8].view(cp.float64)
   self.ws_cont=dm[o6:o6+nr*8].view(cp.float64); self.ws_inj=dm[o7:o7+nr*8].view(cp.float64)
   self.ws_growth=dm[o8:o8+nr*4].view(cp.float32); self.ws_low_min=dm[o9:o9+nr*4].view(cp.float32)
   self.ws_low_inv=dm[o10:o10+nr*4].view(cp.float32); self.ws_sfr=dm[o11:o11+nr*4].view(cp.float32)
   self._ws_freq_mem=cp.cuda.alloc_pinned_memory(nr*14*6*8)
   self.ws_freq_host=np.frombuffer(self._ws_freq_mem,dtype=np.float64,count=nr*14*6).reshape(nr,14,6)
   self.ws_nr=nr
 def upload(self,a):self.upload_bytes+=a.nbytes;return cp.asarray(a)
 def run(self,p):
  try:return self.compute(p.contents)
  except BaseException:
   self.error=traceback.format_exc();print(self.error,flush=True);return 1
 def compute(self,a):
  t=time.perf_counter();n=a.n;nr=a.nr;block=(256,);grid=((n+255)//256,);sc=np.asarray(a.scalars,dtype=np.float64);z=float(sc[0]);
  if a.density and (self.density is None or a.init):
   self.density=self.upload(mat(a.density,nr,n))
  if self.density is None or self.density.shape != (nr,n):
   raise RuntimeError("Heat callback has no density for the current grid")
  gden=self.density
  key=(C.cast(a.prev_x,C.c_void_p).value,float(a.prev_z))
  cache_hit=key in self.states
  if cache_hit:px,pt,pts=self.states[key]
  else:px,pt,pts=[self.upload(arr(p,n).copy()) for p in (a.prev_x,a.prev_t,a.prev_ts)]
  no_light=bool(a.init & 2)
  self._ensure_ws(n,nr)
  self.ws_gsc_h[:]=sc
  self.ws_gx_h[:]=arr(a.xgrid,14)
  tables=[] if no_light else [self.tables.row(a.low[j],a.high[j],1,250,200) for j in range(nr)]
  growth=arr(a.growth,nr);sfr=arr(a.sfr,nr);st=arr(a.st,nr);pref=arr(a.pref,nr);cont=arr(a.cont_pref,nr) if a.use_lya else None;inj=arr(a.inj_pref,nr) if a.use_lya else None
  mins=arr(a.min_density,nr);maxs=arr(a.max_density,nr)
  if not no_light:
   lo_p=self.ws_lo_p;hi_p=self.ws_hi_p
   gr_h=self.ws_gr_h;lm_h=self.ws_lm_h;li_h=self.ws_li_h
   for j in range(nr):
    lo_p[j]=tables[j][0].data.ptr;hi_p[j]=tables[j][1].data.ptr
    gr_h[j]=np.float32(growth[j])
    mn=np.float32(mins[j]*growth[j]);mx=np.float32(maxs[j]*growth[j])
    lmin=np.float32(np.log10(sc[22]) if mn<=-1 else np.log10(1.+float(mn)))
    width=np.float32(1./(float(np.float32(250)-1))*(np.log10(1.+sc[21] if mx>sc[21] else 1.+float(mx))-float(lmin)))
    lm_h[j]=lmin;li_h[j]=np.float32(1./float(width))
   self.ws_st_h[:]=st;self.ws_sfr_h[:]=sfr;self.ws_pref_h[:]=pref
   if cont is not None:self.ws_cont_h[:]=cont
   else:self.ws_cont_h.fill(0)
   if inj is not None:self.ws_inj_h[:]=inj
   else:self.ws_inj_h.fill(0)
  cp.cuda.runtime.memcpyAsync(self.ws_meta_dev.data.ptr,self._ws_meta_mem.ptr,self._ws_meta_bytes,cp.cuda.runtime.memcpyHostToDevice,cp.cuda.get_current_stream().ptr)
  self.upload_bytes+=self._ws_meta_bytes
  gx=self.ws_gx;gsc=self.ws_gsc
  gi,gv,bad=self.ws_gi,self.ws_gv,self.ws_bad;bad.fill(0);prep(grid,block,(px,gx,gi,gv,np.int64(n)))
  freq=self.ws_freq_host
  for cidx,x in enumerate((a.heat,a.heat_diff,a.ion,a.ion_diff,a.lya,a.lya_diff)):
   freq[:,:,cidx]=mat(x,14,nr).T
  cp.cuda.runtime.memcpyAsync(self.ws_gtab.data.ptr,freq.ctypes.data,freq.nbytes,cp.cuda.runtime.memcpyHostToDevice,cp.cuda.get_current_stream().ptr)
  gtab=self.ws_gtab;self.upload_bytes+=freq.nbytes
  rates=[self.ws_rates[r] for r in range(6)]
  ev=getattr(self,'upload_done_event',None)
  if ev is not None:
   cp.cuda.get_current_stream().wait_event(ev)
   self.upload_done_event=None
  if no_light:
   self.ws_rates.fill(0)
  elif getattr(self,"xe_diag",None) is not None:
   self.ws_rates.fill(0);fc=self.ws_fc
   cont_a=cont if cont is not None else np.zeros(nr)
   inj_a=inj if inj is not None else np.zeros(nr)
   for j in reversed(range(nr)):
    mn=np.float32(mins[j]*growth[j]);mx=np.float32(maxs[j]*growth[j]);low_min=np.float32(np.log10(sc[22]) if mn<=-1 else np.log10(1.+float(mn))); width=np.float32(1./(float(np.float32(250)-1))*(np.log10(1.+sc[21] if mx>sc[21] else 1.+float(mx))-float(low_min)));low_inv=np.float32(1./float(width))
    args=[np.float32(v) for v in (growth[j],low_min,low_inv,sc[21],sc[23],sc[21])]
    lookup_sum((256,),block,(gden[j],*tables[j],fc,self.ws_partials,bad,np.int64(n),np.int32(250),np.int32(200),*args,np.float64(1.68)))
    finish_sum((1,),block,(self.ws_partials,self.ws_etotal,np.int32(256)))
    total=self.ws_etotal
    if abs(z-self.xe_diag[1])<1e-6:
     i=self.xe_diag[0]
     density,weighted=map(float,cp.stack([gden[j,i].astype(cp.float64),fc[i].astype(cp.float64)]).get());summed=float(total.get().item())
     jidx=int(gi[i].get())
     jfrac=float(gv[i].get())
     coefficients=freq[j,jidx]
     iz=coefficients[2]+coefficients[3]*jfrac
     mean=summed/(1e10*n)
     df=float(st[j])*float(sfr[j])/(float(sc[20])*mean*1e10) if mean else 0.
     print("GPU_XE_SCALE",z,j,density,float(growth[j]),weighted,mean,df,iz,flush=True)
    accum(grid,block,(fc,gi,gv,gtab[j],*rates,total,np.int64(n),np.float64(st[j]),np.float32(sfr[j]),np.float32(sc[20]),np.float64(pref[j]),np.float64(cont_a[j]),np.float64(inj_a[j])))
  else:
   lookup_sum_batch((256,nr),block,(gden,self.ws_lo_ptrs,self.ws_hi_ptrs,self.ws_fc_all,self.ws_partials_all,bad,np.int64(n),np.int32(nr),np.int32(250),np.int32(200),self.ws_growth,self.ws_low_min,self.ws_low_inv,np.float32(sc[21]),np.float32(sc[23]),np.float32(sc[21]),np.float64(1.68)))
   finish_sum_batch((nr,),block,(self.ws_partials_all,self.ws_totals_all,self.ws_st,self.ws_sfr,np.float32(sc[20]),self.ws_pref,self.ws_cont,self.ws_inj,self.ws_df,self.ws_pref_df,self.ws_cont_df,self.ws_inj_df,np.int64(n)))
   accum_batch(grid,block,(self.ws_fc_all,gi,gv,gtab,*rates,self.ws_df,self.ws_pref_df,self.ws_cont_df,self.ws_inj_df,np.int64(n),np.int32(nr)))
  outputs=[cp.empty(n,dtype=cp.float32) for _ in range(3)]
  relaxed=self.ws_relaxed;relaxed.fill(0)
  capture=getattr(self,"xe_diag",None)
  if capture is not None and abs(z-capture[1])<1e-6:
   i=capture[0]
   frozen=[float(x) for x in cp.stack([gden[0,i].astype(cp.float64),px[i].astype(cp.float64),pt[i].astype(cp.float64),rates[1][i]]).get()]
   d,xe,tk,raw=frozen
   c=self.lib.CompareXeStep
   c.argtypes=[C.c_float]*3+[C.c_double]*8
   c.restype=C.c_float
   expected=c(d,xe,tk,raw,sc[2],sc[4],sc[1],sc[5],sc[19],sc[16],sc[6])
  update(grid,block,(gden[0],px,pt,pts,*outputs,*rates,gsc,self.tc,self.ti,np.int64(n),np.int32(a.use_cmb),np.int32(a.use_lya),bad,relaxed))
  if capture is not None and abs(z-capture[1])<1e-6:
   observed=float(outputs[0][capture[0]].get())
   print("XE_FROZEN_RATE",json.dumps(dict(z=z,density=d,previous_xe=xe,previous_Tk=tk,raw_source=raw,
          prefactor=float(sc[5]),cpu=expected,gpu=observed,absolute_error=abs(expected-observed))),flush=True)
  electron_sum((256,),(256,),(outputs[0],self.ws_partials,np.int64(n)));finish_sum((1,),(256,),(self.ws_partials,self.ws_etotal,np.int32(256)))
  for idx,gpu in enumerate(outputs):
   gpu.get(out=self.ws_pinned_out[idx],blocking=False)
  bad.get(out=self.ws_p_bad,blocking=False)
  self.ws_etotal.get(out=self.ws_p_etot,blocking=False)
  relaxed.get(out=self.ws_p_relaxed,blocking=True)
  err=int(self.ws_p_bad[0])
  if err in (11,12,13):raise RuntimeError('lookup failed z='+str(z)+' code='+str(err))
  if err not in (0,3):raise RuntimeError('GPU invalid index or thermal state: '+str(err))
  copy_nt=min(8,max(1,int(os.environ.get('OMP_NUM_THREADS','8'))))
  nbytes=np.uint64(n*4)
  for idx,host in enumerate((a.out_x,a.out_t,a.out_ts)):
   self._pmemcpy(C.cast(host,C.c_void_p).value,self.ws_pinned_out[idx].ctypes.data,nbytes,copy_nt)
   self.download_bytes+=n*4
  key_out=(C.cast(a.out_x,C.c_void_p).value,z)
  self.states[key_out]=outputs
  a.electron_sum=float(self.ws_p_etot[0])
  self.records.append({'z':z,'seconds':time.perf_counter()-t,'previous_state_cached':cache_hit,'used_memory_bytes':cp.get_default_memory_pool().used_bytes(),'total_pool_bytes':cp.get_default_memory_pool().total_bytes()})
  self.records[-1]['spin_relaxed_cells']=int(self.ws_p_relaxed[0])
  self.records[-1]['no_light']=no_light
  if os.environ.get('PS_VERBOSE_HEAT','0')=='1':
   print('GPU_HEAT_STEP',json.dumps(self.records[-1]),flush=True)
  return 0
 def previous_mean(self,z,previous_x,total_x,total_t):
  key=(C.cast(previous_x,C.c_void_p).value,float(z))
  if key not in self.states:
   self.previous_mean_host_fallbacks+=1
   return 1
  x,t,_=self.states[key]
  self._ensure_ws(x.size)
  electron_sum((256,),(256,),(x,self.ws_partials,np.int64(x.size)))
  finish_sum((1,),(256,),(self.ws_partials,self.ws_etotal,np.int32(256)))
  electron_sum((256,),(256,),(t,self.ws_partials,np.int64(t.size)))
  finish_sum((1,),(256,),(self.ws_partials,self.ws_etotal_t,np.int32(256)))
  self.ws_etotal.get(out=self.ws_p_etot,blocking=False)
  self.ws_etotal_t.get(out=self.ws_p_etot_t,blocking=True)
  total_x[0]=float(self.ws_p_etot[0])
  total_t[0]=float(self.ws_p_etot_t[0])
  self.previous_mean_calls+=1
  return 0
 def install(self,native):
  self.lib=C.CDLL(native.__file__)
  self._pmemcpy=self.lib.PSParallelMemcpy
  self._pmemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_ulonglong,C.c_int]
  self._pmemcpy.restype=None;self.callback=C.CFUNCTYPE(C.c_int,C.POINTER(Args))(self.run);self.lib.SetGPUHeatHook.argtypes=[C.c_void_p];self.lib.SetGPUHeatHook(C.cast(self.callback,C.c_void_p))
  self.mean_callback=C.CFUNCTYPE(C.c_int,C.c_float,F,DD,DD)(self.previous_mean)
  setter=self.lib.SetGPUPreviousHeatMeans
  setter.argtypes=[C.c_void_p]
  setter.restype=None
  setter(C.cast(self.mean_callback,C.c_void_p))
 def report(self):return {'records':self.records,'previous_mean_gpu_calls':self.previous_mean_calls,'previous_mean_cpu_fallbacks':self.previous_mean_host_fallbacks,'error':self.error,'upload_bytes':self.upload_bytes,'download_bytes':self.download_bytes,'note':'heat grid core resident across 40 radii; cached previous thermal states; host snapshots are exported through the native box ABI'}
