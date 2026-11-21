"""Isolated resident-grid thermal backend. CPU table setup and other phases remain CPU."""
import ctypes as C, json, os, traceback, time
from pathlib import Path
import numpy as np
import cupy as cp
REPO_ROOT = Path(os.environ["PS_GPU_ROOT"]) if os.environ.get("PS_GPU_ROOT") else Path(__file__).resolve().parent.parent
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
lookup=cp.RawKernel(lookup_source,'lookup',options=('--fmad=false',))
accum_source=r'''extern "C" __global__ void accum(const float*fc,const int*idx,const float*v,const double*tbl,double*h,double*ion,double*lya,double*star,double*cont,double*inj,const double*sum,long n,double st,float sfr,float tstar,double before,double cbefore,double ibefore){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;double ave=(*sum)/(1e10*(double)n);if(ave==0.0)return;double df=(1./ave/1e10)*st*(double)sfr/(double)tstar;double pref=before*df;double x=(double)fc[i],r=(double)v[i];const double*q=tbl+6*idx[i];h[i]+=df*x*(q[1]*r+q[0]);ion[i]+=df*x*(q[3]*r+q[2]);lya[i]+=df*x*(q[5]*r+q[4]);star[i]+=x*pref;cont[i]+=x*(cbefore*df);inj[i]+=x*(ibefore*df);}'''
accum=cp.RawKernel(accum_source,'accum',options=('--fmad=false',))
prep=cp.RawKernel(r'''extern "C" __global__ void prep(const float*xe,const float*xgrid,int*ix,float*iv,long n){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;float x=xe[i];if(x>xgrid[13]*.999)x=xgrid[13]*.999;else if(x<xgrid[0])x=1.001*xgrid[0];int j=13;while(x<xgrid[j])--j;ix[i]=j;float inverse=1./(xgrid[j+1]-xgrid[j]);iv[i]=(x-xgrid[j])*inverse;}''','prep',options=('--fmad=false',))
update=cp.RawKernel(Path(__file__).with_name('update_stabilized.cu').read_text(),'update',options=('--fmad=false',))
electron_sum=cp.RawKernel(r'''extern "C" __global__ void esum(const float*x,double*out,long n){__shared__ double v[256];double t=0.;for(long i=(long)blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x)t+=(double)x[i];v[threadIdx.x]=t;__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)out[blockIdx.x]=v[0];}''','esum')
finish_sum=cp.RawKernel(r'''extern "C" __global__ void finish(const double*x,double*out,int n){__shared__ double v[256];double t=0.;for(int i=threadIdx.x;i<n;i+=256)t+=x[i];v[threadIdx.x]=t;__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)v[threadIdx.x]+=v[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)out[0]=v[0];}''','finish')
class Runtime:
 def __init__(self):
  for k in (lookup,accum,prep,update,electron_sum,finish_sum):
   print('KERNEL_READY',k.name,k.attributes,flush=True)
  table=np.loadtxt(REPO_ROOT/'resident_filter_20261118/fix/package/py21cmfast/_data/Lyman_alpha_heating_table.dat');self.tc=cp.asarray(table[:,0]);self.ti=cp.asarray(table[:,1]);self.states={};self.density=None;self.records=[];self.error=None;self.previous_mean_calls=0;self.previous_mean_host_fallbacks=0;self.upload_bytes=table.nbytes;self.download_bytes=0
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
  gx=self.upload(arr(a.xgrid,14).copy());gi=cp.empty(n,dtype=cp.int32);gv=cp.empty(n,dtype=cp.float32);bad=cp.zeros(1,dtype=cp.int32);prep(grid,block,(px,gx,gi,gv,np.int64(n)))
  freq=np.stack([mat(x,14,nr) for x in (a.heat,a.heat_diff,a.ion,a.ion_diff,a.lya,a.lya_diff)],axis=-1).transpose(1,0,2).copy();gtab=self.upload(freq)
  tables=[] if no_light else [self.tables.row(a.low[j],a.high[j],1,250,200) for j in range(nr)]
  growth=arr(a.growth,nr);sfr=arr(a.sfr,nr);st=arr(a.st,nr);pref=arr(a.pref,nr);cont=arr(a.cont_pref,nr) if a.use_lya else np.zeros(nr);inj=arr(a.inj_pref,nr) if a.use_lya else np.zeros(nr)
  mins=arr(a.min_density,nr);maxs=arr(a.max_density,nr)
  rates=[cp.zeros(n,dtype=cp.float64) for _ in range(6)];fc=cp.empty(n,dtype=cp.float32);vals=cp.empty(n,dtype=cp.float64)
  for j in reversed(range(0 if no_light else nr)):
   mn=np.float32(mins[j]*growth[j]);mx=np.float32(maxs[j]*growth[j]);low_min=np.float32(np.log10(sc[22]) if mn<=-1 else np.log10(1.+float(mn))); width=np.float32(1./(float(np.float32(250)-1))*(np.log10(1.+sc[21] if mx>sc[21] else 1.+float(mx))-float(low_min)));low_inv=np.float32(1./float(width))
   args=[np.float32(v) for v in (growth[j],low_min,low_inv,sc[21],sc[23],sc[21])]
   lookup(grid,block,(gden[j],*tables[j],fc,vals,bad,np.int64(n),np.int32(250),np.int32(200),*args,np.float64(1.68)))
   total=cp.sum(vals,dtype=cp.float64)
   accum(grid,block,(fc,gi,gv,gtab[j],*rates,total,np.int64(n),np.float64(st[j]),np.float32(sfr[j]),np.float32(sc[20]),np.float64(pref[j]),np.float64(cont[j]),np.float64(inj[j])))
  lookup_error=int(bad.get()[0])
  if lookup_error:raise RuntimeError('lookup failed z='+str(z)+' code='+str(lookup_error))
  outputs=[cp.empty(n,dtype=cp.float32) for _ in range(3)];gsc=self.upload(sc)
  relaxed=cp.zeros(1,dtype=cp.uint32)
  update(grid,block,(gden[0],px,pt,pts,*outputs,*rates,gsc,self.tc,self.ti,np.int64(n),np.int32(a.use_cmb),np.int32(a.use_lya),bad,relaxed))
  cp.cuda.Stream.null.synchronize();err=int(bad.get()[0]);
  if err not in (0,3):raise RuntimeError('GPU invalid index or thermal state: '+str(err))
  for host,gpu in zip((a.out_x,a.out_t,a.out_ts),outputs):arr(host,n)[:]=cp.asnumpy(gpu);self.download_bytes+=n*4
  key_out=(C.cast(a.out_x,C.c_void_p).value,z)
  self.states[key_out]=outputs
  partials=cp.empty(256,dtype=cp.float64);etotal=cp.empty(1,dtype=cp.float64);electron_sum((256,),(256,),(outputs[0],partials,np.int64(n)));finish_sum((1,),(256,),(partials,etotal,np.int32(256)));a.electron_sum=float(etotal.get()[0])
  self.records.append({'z':z,'seconds':time.perf_counter()-t,'previous_state_cached':cache_hit,'used_memory_bytes':cp.get_default_memory_pool().used_bytes(),'total_pool_bytes':cp.get_default_memory_pool().total_bytes()})
  self.records[-1]['spin_relaxed_cells']=int(relaxed.get()[0])
  self.records[-1]['no_light']=no_light
  print('GPU_HEAT_STEP',json.dumps(self.records[-1]),flush=True)
  return 0
 def previous_mean(self,z,previous_x,total_x,total_t):
  key=(C.cast(previous_x,C.c_void_p).value,float(z))
  if key not in self.states:
   self.previous_mean_host_fallbacks+=1
   return 1
  x,t,_=self.states[key]
  values=cp.stack((cp.sum(x,dtype=cp.float64),cp.sum(t,dtype=cp.float64))).get()
  total_x[0],total_t[0]=values
  self.previous_mean_calls+=1
  return 0
 def install(self,native):
  self.lib=C.CDLL(native.__file__);self.callback=C.CFUNCTYPE(C.c_int,C.POINTER(Args))(self.run);self.lib.SetGPUHeatHook.argtypes=[C.c_void_p];self.lib.SetGPUHeatHook(C.cast(self.callback,C.c_void_p))
  self.mean_callback=C.CFUNCTYPE(C.c_int,C.c_float,F,DD,DD)(self.previous_mean)
  setter=self.lib.SetGPUPreviousHeatMeans
  setter.argtypes=[C.c_void_p]
  setter.restype=None
  setter(C.cast(self.mean_callback,C.c_void_p))
 def report(self):return {'records':self.records,'previous_mean_gpu_calls':self.previous_mean_calls,'previous_mean_cpu_fallbacks':self.previous_mean_host_fallbacks,'error':self.error,'upload_bytes':self.upload_bytes,'download_bytes':self.download_bytes,'note':'heat grid core resident across 40 radii; cached previous thermal states; host snapshots are exported through the native box ABI'}
