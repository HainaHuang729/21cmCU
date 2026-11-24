"""Replay controlled CHMF ionization inputs on native and adapted 4.3.2 routines."""

import ctypes as C
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from contextlib import ExitStack

import numpy as np

from evolution_backend import SRC, NATIVE
from halo_ionization import install_ionization, ionization_source

# The reference and CUDA paths receive independent copies of the same buffers.
REPLAY = """
int replay(int side, int kind, int exp_filter, int mode, float *density, float *photons,
           float *old_z, float *xe, float *rec, float *sfr, float *gas_tk,
           float *neutral, float *zre, float *tk, float *per_radius, float *gamma, float *mfp) {
    SimulationOptions *old_sim=simulation_options_global, sim={
        .HII_DIM=side,.NON_CUBIC_FACTOR=1,.N_THREADS=1,.BOX_LEN=42};
    MatterOptions *old_matter=matter_options_global, matter={.SOURCE_MODEL=SOURCE_MODEL_CHMF_SAMPLER};
    AstroOptions *old_astro=astro_options_global, astro={.USE_EXP_FILTER=exp_filter,.USE_TS_FLUCT=(mode & 1)!=0,.USE_MCGS=(mode & 8)!=0,
        .RECOMB_MODEL=(mode & 2)?RECOMB_MODEL_INHOMOGENEOUS:RECOMB_MODEL_NO,.CELL_RECOMB=(mode & 4)==0};
    simulation_options_global=&sim; matter_options_global=&matter; astro_options_global=&astro;
    struct IonBoxConstants c={.redshift=8.9,.stored_redshift=8.9,.hii_filter=kind,
        .photoncons_adjustment_factor=1.,.lagrangian_source_grids=true,.mass_dep_zeta=true,
        .filter_recombinations=(mode & 4)!=0,.gamma_prefactor=.82,.TK_nofluct=4.2,.adia_TK_term=.5,.T_re=19000.,.mfp_meandens=12.};
    struct FilteredGrids *owner;
    allocate_fftw_grids(&owner);
    struct FilteredGrids g=*owner;
    IonizedBox box={.neutral_fraction=neutral,.z_reion=zre,.kinetic_temperature=tk,
                    .ionisation_rate_G12=gamma,.mean_free_path=mfp};
    IonizedBox prev={.z_reion=old_z,.cumulative_recombinations=rec};
    TsBox spin={.kinetic_temp_neutral=gas_tk};
    PerturbedField pf={.density=density};
    int code=0;
    Try {
        prepare_box_for_filtering(density,g.deltax_unfiltered,1.,-.9999999,1e10);
        prepare_box_for_filtering(photons,g.stars_unfiltered,1.,0.,1e10);
        if(mode & 1) prepare_box_for_filtering(xe,g.xe_unfiltered,1.,0.,1.);
        if(mode & 2) prepare_box_for_filtering(sfr,g.sfr_unfiltered,1.,0.,1e10);
        if(mode & 4) prepare_box_for_filtering(rec,g.N_rec_unfiltered,1.,0.,1e10);
        for(int i=2; i>=0; --i){
            struct RadiusSpec r={.R_index=i,.R=2.625*(i+1)};
            copy_filter_transform(&g,&c,r);
            calculate_conditional_nion(&box,&prev,&g,&c,&r);
            find_ionised_regions(&box,&prev,&pf,&spin,r,&c,&g,1e-8,(mode & 8) ? 0.22 : 0.);
            memcpy(per_radius+(2-i)*HII_TOT_NUM_PIXELS,neutral,HII_TOT_NUM_PIXELS*sizeof(float));
        }
        set_ionized_temperatures(&box,&pf,&spin,&c);
    }
    Catch(code) { }
    free_fftw_grids(owner);
    simulation_options_global=old_sim; matter_options_global=old_matter; astro_options_global=old_astro;
    return code;
}
"""


def run(directory, source, gpu, kind, exp_filter, mode, inputs):
    cfile, so = directory / "replay.c", directory / "replay.so"
    cfile.write_text(source + "\n" + REPLAY)
    if not so.exists():
        subprocess.run(
            [
                "cc",
                "-Ofast",
                "-std=gnu99",
                "-shared",
                "-fPIC",
                "-fopenmp",
                "-Wl,-Bsymbolic-functions",
                "-I" + str(SRC),
                *shlex.split(os.environ.get("CPPFLAGS", "")),
                str(cfile),
                str(NATIVE),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
        )
    adapter = C.CDLL(str(so))
    arr = np.ctypeslib.ndpointer(dtype="float32", flags="C_CONTIGUOUS")
    adapter.replay.argtypes = [C.c_int] * 4 + [arr] * 13
    side = inputs[0].shape[0]
    outputs = [
        np.ones_like(inputs[0]),
        np.full_like(inputs[0], -1.0),
        np.zeros_like(inputs[0]),
        np.empty((3, side, side, side), dtype="f"),
        np.zeros_like(inputs[0]),
        np.zeros_like(inputs[0]),
    ]
    counts = {}
    with ExitStack() as stack:
        if gpu:
            install_ionization(stack, adapter, counts)
        status = adapter.replay(side, kind, exp_filter, mode, *inputs, *outputs)
        if status:
            raise ValueError(f"CHMF ionization replay failed with status {status}")
    return outputs, counts


def main(output):
    rng = np.random.default_rng(17841)
    side = 16
    density = rng.uniform(-0.65, 1.15, (side,) * 3).astype("f")
    photons = np.exp(rng.normal(-1.0, 0.55, density.shape)).astype("f")
    photons[:7, :10, :] *= 7
    old_z = rng.choice(np.array([-1.0, 10.4], dtype="f"), density.shape)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / name for name in ("cpu", "gpu")]
        for path in paths:
            path.mkdir()
        extra = (
            rng.uniform(0.001, 0.45, density.shape).astype("f"),
            rng.uniform(0.01, 0.7, density.shape).astype("f"),
            rng.uniform(0.05, 3.0, density.shape).astype("f"),
            rng.uniform(1.0, 140.0, density.shape).astype("f"),
        )
        native_control = {}
        for kind, exp_filter, mode in (
            (0, True, 0),
            (0, False, 0),
            (1, False, 0),
            (2, False, 0),
            (0, True, 1),
            (0, True, 2),
            (0, True, 3),
            (0, False, 7),
            (0, True, 11),
            (0, False, 15),
        ):
            args = kind, exp_filter, mode, (density, photons, old_z, *extra)
            expected, _ = run(
                paths[0], (SRC / "IonisationBox.c").read_text(), False, *args
            )
            key = kind, exp_filter, mode & 7
            if mode & 8:
                changed = int(np.count_nonzero(expected[0] != native_control[key]))
                assert changed, (
                    "Molecular photon floor had no effect on neutral fractions"
                )
            else:
                native_control[key] = expected[0]
                changed = 0
            actual, counts = run(paths[1], ionization_source(), True, *args)
            row = {
                "hii_filter": kind,
                "exp_filter": exp_filter,
                "spin_fluctuations": bool(mode & 1),
                "inhomogeneous_recombinations": bool(mode & 2),
                "cell_recomb": not bool(mode & 4),
                "active_photoionization_rates": int(np.count_nonzero(expected[4])),
                "molecular": bool(mode & 8),
                "changed_from_atomic_control": changed,
                "cuda_calls": counts,
                "fully_ionized": int((expected[0] == 0).sum()),
                "partially_ionized": int(((expected[0] > 0) & (expected[0] < 1)).sum()),
                "fields": {},
            }
            for key, cpu, gpu in zip(
                (
                    "neutral_fraction",
                    "z_reion",
                    "kinetic_temperature",
                    "neutral_fraction_by_radius",
                    "ionisation_rate_G12",
                    "mean_free_path",
                ),
                expected,
                actual,
                strict=True,
            ):
                delta = np.abs(cpu.astype("d") - gpu.astype("d"))
                row["fields"][key] = {
                    "max_abs": float(delta.max()),
                    "failed": int(
                        (~(delta <= 1e-8 + 1e-4 * np.abs(cpu.astype("d")))).sum()
                    ),
                }
            results.append(row)
    output.write_text(json.dumps(results, indent=2) + "\n")
    assert all(
        row["fully_ionized"]
        and row["partially_ionized"]
        and all(v["failed"] == 0 for v in row["fields"].values())
        for row in results
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
