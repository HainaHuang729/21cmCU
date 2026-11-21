import ctypes as C
import json
import pathlib
import sys
import numpy as np

Fn=C.CFUNCTYPE(C.c_double,C.c_double,C.c_void_p)
class GslFunction(C.Structure):
    _fields_=[("fn",Fn),("params",C.c_void_p)]
C.CDLL("libgslcblas.so",mode=C.RTLD_GLOBAL)
gsl=C.CDLL("libgsl.so")
q=gsl.gsl_integration_qk61
q.restype=None
q.argtypes=[C.POINTER(GslFunction),C.c_double,C.c_double]+[C.POINTER(C.c_double)]*4

def sample(callback):
    outputs=[C.c_double() for _ in range(4)]
    f=GslFunction(Fn(callback),None)
    q(C.byref(f),-1.,1.,*(C.byref(v) for v in outputs))
    return outputs[0].value

observed=set()
sample(lambda x,_: observed.add(abs(x)) or 0.)
points=sorted(observed,reverse=True)
assert len(points)==31 and points[-1]==0.
kronrod=[sample(lambda x,_,p=p:float(x==p)) for p in points]
glx,glw=np.polynomial.legendre.leggauss(30)
assert np.max(np.abs(glx[:15]+points[1:30:2]))<2e-14
assert abs(2*sum(kronrod[:-1])+kronrod[-1]-2)<1e-14
pathlib.Path(sys.argv[1]).write_text(json.dumps(points+kronrod+glw[:15].tolist()))
