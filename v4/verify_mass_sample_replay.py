"""Replay captured 4.3.2 progenitor masses and the exact GSL word segments."""

import argparse
import json
import os
from pathlib import Path

import cupy as cp
import numpy as np

from catalog_libm import link_math_module, source_for_host

HERE = Path(__file__).resolve().parent
REPLAY_KERNEL = r"""
extern "C" __global__ void replay(
    int count, const double *parameters, const unsigned *words, const int *offsets,
    const double *ratios, const double *axes, int nx, int ny,
    float *masses, const int *starts, int capacity, int *status
) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=count) return;
    const double *params=parameters+4*i;
    MassSampleTable table={ratios,nx,ny,axes[0],axes[1],axes[2],axes[3],axes[4]};
    MassSampleWords cursor;
    cursor.words=words+offsets[i];
    cursor.count=offsets[i+1]-offsets[i];
    int total;
    bool ok=draw_limited_catalog_mass(
        table,cursor,params[0],params[1],params[2],params[3]!=0.,
        capacity,masses+starts[i],total);
    status[3*i]=ok;
    status[3*i+1]=cursor.consumed;
    status[3*i+2]=total;
}
"""


def replay(folder, glibc):
    source = (HERE / "catalog_mass_sample.cuh").read_text() + REPLAY_KERNEL
    module = link_math_module(glibc, cp.cuda.runtime.getDevice(), source)
    kernel = module.get_function("replay")
    results = []
    for file in sorted(folder.glob("condition-trace-*.npz")):
        row = np.load(file)
        params, words, offsets, table, axes = (
            row[k] for k in ("parameters", "words", "offsets", "table", "axes")
        )
        counts = row["counts"]
        expected = row["expected"]
        stride = int(max(counts)) + 1
        n = len(counts)
        output = cp.empty((n, stride), "f")
        status = cp.empty((n, 3), "i")
        kernel(
            ((n + 127) // 128,),
            (128,),
            (
                np.int32(n),
                cp.asarray(params),
                cp.asarray(words),
                cp.asarray(offsets),
                cp.asarray(table),
                cp.asarray(axes),
                np.int32(table.shape[0]),
                np.int32(table.shape[1]),
                output,
                cp.arange(n, dtype="i") * stride,
                np.int32(stride),
                status,
            ),
        )
        positions = np.arange(stride)[None, :] < counts[:, None]
        got = output.get()[positions]
        flags = status.get()
        passed = (
            bool(flags[:, 0].all())
            and np.array_equal(flags[:, 1], np.diff(offsets))
            and np.array_equal(flags[:, 2], counts)
            and np.array_equal(got.view("u4"), expected.view("u4"))
        )
        assert passed, file
        results.append(
            dict(
                file=file.name,
                conditions=n,
                words=len(words),
                masses=len(expected),
                rare_conditions=int(np.count_nonzero(params[:, 3])),
                zeroed_masses=int(np.count_nonzero(expected == 0)),
                passed=passed,
            )
        )
    if not results:
        raise ValueError(f"No native condition traces in {folder}")
    return dict(job=os.getenv("SLURM_JOB_ID"), results=results, passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_directory", type=Path)
    parser.add_argument("glibc_math_source", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            replay(args.trace_directory, source_for_host(args.glibc_math_source))
        )
    )
