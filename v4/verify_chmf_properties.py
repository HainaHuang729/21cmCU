"""Check mixed mass-stream consumption, bounded writes and correlated property batches."""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state

from evolution_backend import NATIVE, SRC
from halo_catalog import HERE, _batch_callback
from native_shim import load_shim


# Test the actual native function and the production batch before and after
# an output-buffer overflow. Placeholders for interspersed mass draws also
# check that each subsequent mass decision sees identical GSL words.
HARNESS = """
void set_prop_rng(gsl_rng *, bool, double *, double *, double *);
int replay(int kind, unsigned long seed, int sphere, int active, int total,
           float *coords, float *properties, float *expected, unsigned long *tail_misses) {
    const gsl_rng_type *type = kind ? gsl_rng_gfsr4 : gsl_rng_mt19937;
    gsl_rng *ref=gsl_rng_alloc(type), *rng=gsl_rng_alloc(type);
    gsl_rng_set(ref, seed);
    gsl_rng_set(rng, seed);
    CatalogBatch *batch = calloc(1, sizeof(*batch));
    if (!batch) return 1;
    HaloCatalog out = {.halo_coords=coords, .star_rng=properties,
                       .sfr_rng=properties+total, .xray_rng=properties+2*total};
    double corr[3]={0., 1., .65};
    double pos[]={1.,2.,3.};
    for (int i=0; i<total; ++i) {
        *tail_misses += gsl_rng_get(rng) != gsl_rng_get(ref);
        double parent[]={.25f*(i%7),-1.25f*(i%3),1.75f*(i%5)};
        double draws[3];
        if (sphere) set_prop_rng(ref, true, corr, parent, draws);
        for (int j=0; j<3+sphere; ++j) (void)gsl_rng_uniform(ref);
        if (!sphere && i<active) set_prop_rng(ref, false, NULL, NULL, draws);
        if (i<active)
            for (int j=0;j<3;++j) expected[(size_t)j*total+i]=(float)draws[j];
        queue_halo(batch,&out,i,i<active,pos,.75,rng,sphere ? corr : NULL,parent,NULL);
    }
    flush_halos(batch,&out,sphere ? corr : NULL);
    int failed = batch->failed;
    free(batch);
    for (int i=0; i<2048; ++i) *tail_misses += gsl_rng_get(rng)!=gsl_rng_get(ref);
    gsl_rng_free(ref); gsl_rng_free(rng);
    return failed;
}

int ties(float *actual, float *expected) {
    CatalogBatch *b = calloc(1, sizeof(*b));
    if (!b) return 1;
    b->size = 1;
    float coords[3];
    HaloCatalog out = {.halo_coords=coords, .star_rng=actual, .sfr_rng=actual+1, .xray_rng=actual+2};
    double r2=.25;
    for (int i=0; i<3; ++i) {
        float f=1.f + i * 0x1p-23f;
        double mid=.5*((double)f+nextafterf(f, INFINITY));
        double y=mid/sqrt((-2.*log(r2))/r2);
        b->rows[0][8+i]=y; b->rows[0][11+i]=r2;
        expected[i]=(float)(y*sqrt((-2.*log(r2))/r2));
    }
    flush_halos(b,&out,NULL);
    int failed = b->failed;
    free(b);
    return failed;
}
"""


def check(path):
    inputs = p21.InputParameters(random_seed=424242)
    header = (SRC / "Stochasticity.c").read_text().split("void print_hs_consts(")[0]
    source = header + (HERE / "halo_catalog_batch.h").read_text() + HARNESS
    counters, errors, rows = {}, [], []
    ptr = C.POINTER(C.c_float)
    with TemporaryDirectory() as temp, c_state(inputs, broadcast_inputs=True):
        lib = load_shim(Path(temp), "probe.so", {"probe.c": source}, SRC, NATIVE)
        callback = _batch_callback(counters, errors)
        C.c_void_p.in_dll(lib, "place_on_gpu").value = C.cast(
            callback, C.c_void_p
        ).value
        lib.replay.argtypes = [
            C.c_int,
            C.c_ulong,
            C.c_int,
            C.c_int,
            C.c_int,
            ptr,
            ptr,
            ptr,
            C.POINTER(C.c_ulong),
        ]
        active, total = 4099, 4593
        for kind in (0, 1):
            for seed in (0, 681725):
                for sphere in (0, 1):
                    coords = np.full((total, 3), -900.0, dtype="f")
                    expected = np.full((3, total), -999.0, dtype="f")
                    actual = expected.copy()
                    misses = C.c_ulong(0)
                    status = lib.replay(
                        kind,
                        seed,
                        sphere,
                        active,
                        total,
                        coords.ctypes.data_as(ptr),
                        actual.ctypes.data_as(ptr),
                        expected.ctypes.data_as(ptr),
                        C.byref(misses),
                    )
                    if errors:
                        raise errors[0]
                    np.testing.assert_array_equal(
                        actual.view("u4"), expected.view("u4")
                    )
                    assert (
                        status == 0
                        and misses.value == 0
                        and np.all(coords[active:] == -900.0)
                    )
                    rows.append(
                        dict(
                            engine=("mt19937", "gfsr4")[kind],
                            seed=seed,
                            progenitor=bool(sphere),
                            active=active,
                            discarded=total - active,
                            next_words_tested=total + 2048,
                        )
                    )
        lib.ties.argtypes = [ptr, ptr]
        actual, expected = [np.zeros(3, dtype="f") for _ in range(2)]
        before = counters["halo_property_host_roundings"]
        status = lib.ties(actual.ctypes.data_as(ptr), expected.ctypes.data_as(ptr))
        if errors:
            raise errors[0]
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        assert status == 0 and counters["halo_property_host_roundings"] - before == 3
    path.write_text(
        json.dumps(
            dict(
                job=os.environ.get("SLURM_JOB_ID"),
                rows=rows,
                verified_rounding_ties=3,
                counters=counters,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)
