"""Create a reusable native InitialConditions HDF5 with the CPU build."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import py21cmfast
import py21cmfast.c_21cmfast as native

root=Path(os.environ["PS_GPU_ROOT"])
dest=Path(sys.argv[1])
cfg=json.loads((root/"point_config.json").read_text())
native_sha=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
assert native_sha==cfg["native_sha256"]
begin=time.perf_counter()
ic=py21cmfast.initial_conditions(user_params=cfg["user"],cosmo_params=cfg["cosmo"],
                                  random_seed=cfg["seed"],regenerate=True,write=False,direc=str(dest),
                                  Z_HEAT_MAX=35.,EVOLVE_DENSITY_LINEARLY=False)
compute=time.perf_counter()-begin
begin=time.perf_counter()
ic.write(direc=str(dest))
save=time.perf_counter()-begin
files=list(dest.glob("InitialConditions*.h5"))
assert len(files)==1, files
h5=files[0]
result={"file":str(h5),"bytes":h5.stat().st_size,"seed":cfg["seed"],"index":cfg["index"],
        "native_sha256":native_sha,"cpu_ic_seconds":compute,"hdf5_write_seconds":save}
(dest/"ic-manifest.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
