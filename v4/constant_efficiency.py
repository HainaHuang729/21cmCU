"""CUDA evaluation of 4.3.2's conditional Press–Schechter source tables."""

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    from evolution_backend import SRC

    # Keep the exact native polynomial and its float interface.
    hmf = (SRC / "hmf.c").read_text()
    start = hmf.index("float erfcc(float x) {")
    stop = hmf.index("\n}", start) + 2
    source = (
        "__device__ "
        + hmf[start:stop]
        + "\n"
        + Path(__file__).with_suffix(".cu").read_text()
    )
    return cp.RawKernel(source, "eps_source_table", options=("--fmad=false",))


def fill(p, values, sfr):
    params = np.ctypeslib.as_array(p, shape=(14,))
    _kernel()(
        ((values.size + 127) // 128,),
        (128,),
        (cp.asarray(params), np.int32(values.size), np.int32(sfr), values),
    )
