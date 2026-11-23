"""Inspect clean ESTIMATE plans for the upstream in-place float32 cubic layout."""
import argparse
import ctypes as C
import json
import re

import py21cmfast.c_21cmfast as native


def inspect(sizes, threads):
    lib = C.CDLL(native.__file__)
    ptr = C.c_void_p
    lib.fftwf_malloc.argtypes = [C.c_size_t]
    lib.fftwf_malloc.restype = lib.fftwf_sprint_plan.restype = ptr
    if not lib.fftwf_init_threads():
        raise RuntimeError("FFTW thread initialization failed")
    rows = []
    for n in sizes:
        buf = ptr(lib.fftwf_malloc(4 * n * n * 2 * (n // 2 + 1)))
        if not buf:
            raise MemoryError(n)
        try:
            for count in threads:
                for direction in ("r2c", "c2r"):
                    lib.fftwf_forget_wisdom()
                    lib.fftwf_plan_with_nthreads(count)
                    plan = getattr(lib, f"fftwf_plan_dft_{direction}_3d")
                    plan.restype = ptr
                    h = ptr(plan(n, n, n, buf, buf, 64))  # FFTW_ESTIMATE
                    if not h:
                        raise RuntimeError((n, count, direction))
                    try:
                        desc = ptr(lib.fftwf_sprint_plan(h))
                        if not desc:
                            raise MemoryError("FFTW plan description")
                        try:
                            tree = C.string_at(desc).decode()
                        finally:
                            lib.fftwf_free(desc)
                    finally:
                        lib.fftwf_destroy_plan(h)
                    rows.append(dict(n=n, threads=count, direction=direction,
                                     codelets=sorted(set(re.findall(r'"([^"]+)"', tree))), tree=tree))
        finally:
            lib.fftwf_free(buf)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="+", type=int)
    parser.add_argument("--threads", nargs="+", type=int, default=[2])
    args = parser.parse_args()
    if min(args.sizes + args.threads) < 1:
        parser.error("sizes and thread counts must be positive")
    print(json.dumps(inspect(args.sizes, args.threads), indent=2))
