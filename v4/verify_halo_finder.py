"""Compare the DexM catalogue and the overlap grid handed to the native CHMF sampler."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.drivers.single_field import determine_halo_catalog
from py21cmfast.wrapper.outputs import HaloCatalog

from evolution_backend import NATIVE, SRC, view
from halo_dexm import finder_source, install_finder

Capture = C.CFUNCTYPE(None, C.c_ulonglong, C.c_int, *([C.c_void_p] * 6))


MaskCapture = C.CFUNCTYPE(None, C.c_int, C.c_float, C.c_void_p)


def observed(source):
    marker = "            // now lets scroll through the box"
    source = source.replace(
        marker,
        """
            if (matter_options_global->DEXM_OPTIMIZE
                && M > simulation_options_global->DEXM_OPTIMIZE_MINMASS && capture_mask)
                capture_mask(grid_dim, R, forbidden);
"""
        + marker,
        1,
    )
    start = source.index(
        "            stochastic_halofield(random_seed, redshift_desc, redshift, boxes->lowres_density,"
    )
    stop = source.index(";", start) + 1
    return (
        "void (*capture_mask)(int,float,void*);\n"
        "void (*capture_finder)(unsigned long long,int,float*,float*,float*,float*,float*,float*);\n"
        + source[:start]
        + """capture_finder(halos_dexm->n_halos,simulation_options_global->HII_DIM,
                                     halos_dexm->halo_masses,halos_dexm->halo_coords,halo_overlap_box,
                                     halos_dexm->star_rng,halos_dexm->sfr_rng,halos_dexm->xray_rng);
                              halos->n_halos=0;"""
        + source[stop:]
    )


def measure(so, ic, inputs, redshift, use_cuda):
    adapter = C.CDLL(str(so))
    calls, errors, results = {}, [], []
    masks = {}
    if use_cuda:
        finder, callback = install_finder(adapter, calls, errors)
        rebuild = finder.rebuild_forbidden

        def capture_rebuild(side, radius):
            bits = rebuild(side, radius)
            masks[radius] = hashlib.sha256(bits.get().tobytes()).hexdigest()
            return bits

        finder.rebuild_forbidden = capture_rebuild

    @MaskCapture
    def capture_mask(side, radius, data):
        packed = np.packbits(view(data, side**3, "B"), bitorder="little")
        masks[radius] = hashlib.sha256(packed.tobytes()).hexdigest()

    C.c_void_p.in_dll(adapter, "capture_mask").value = C.cast(
        capture_mask, C.c_void_p
    ).value

    @Capture
    def capture(n, side, masses, coords, overlap, star, sfr, xray):
        results.append(
            (
                view(masses, n, "f").copy() if n else np.empty(0, dtype="f"),
                view(coords, n * 3, "f").reshape(n, 3).copy()
                if n
                else np.empty((0, 3), dtype="f"),
                view(overlap, side**3, "f").copy(),
                *(
                    view(ptr, n, "f").copy() if n else np.empty(0, dtype="f")
                    for ptr in (star, sfr, xray)
                ),
            )
        )

    C.c_void_p.in_dll(adapter, "capture_finder").value = C.cast(
        capture, C.c_void_p
    ).value
    fn = ffi.cast(
        ffi.typeof(lib.ComputeHaloCatalog),
        C.cast(adapter.ComputeHaloCatalog, C.c_void_p).value,
    )

    def compute(*args):
        status = fn(*args)
        if errors:
            raise errors.pop()
        return status

    with patch.object(HaloCatalog, "_c_compute_function", staticmethod(compute)):
        determine_halo_catalog(
            inputs=inputs,
            initial_conditions=ic,
            redshift=redshift,
            regenerate=True,
            write=False,
        )
    assert len(results) == 1
    return results[0], calls, masks


def main(path, optimized_only=False):
    rows = []
    with TemporaryDirectory() as directory:
        sources = ((SRC / "HaloCatalog.c").read_text(), finder_source(SRC))
        libs = []
        for mode, src in zip(("native", "gpu"), sources, strict=True):
            tmp = Path(directory)
            cfile, so = tmp / (mode + ".c"), tmp / (mode + ".so")
            cfile.write_text(observed(src))
            subprocess.run(
                [
                    "cc",
                    "-Ofast",
                    "-std=gnu99",
                    "-fPIC",
                    "-shared",
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
            libs.append(so)
        configs = (
            [(32, 16, 24.0, 1, 2.5e12), (48, 16, 42.0, 2, 1e11)]
            if optimized_only
            else [(48, 16, 42.0, 1, None), (32, 16, 24.0, 2, None)]
        )
        for side, coarse, length, threads, minmass in configs:
            z = 6.0
            inputs = p21.InputParameters(
                random_seed=37251, node_redshifts=[z]
            ).evolve_input_structs(
                DIM=side, HII_DIM=coarse, BOX_LEN=length, N_THREADS=threads
            )
            if minmass:
                inputs = inputs.evolve_input_structs(
                    DEXM_OPTIMIZE=True, DEXM_OPTIMIZE_MINMASS=minmass
                )
            ic = p21.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            density = ic.get("hires_density")
            native_density = density.copy()
            axis = np.arange(side)
            x, y, k = np.meshgrid(axis, axis, axis, indexing="ij")
            for case in ("seeded", "periodic_peaks", "zero"):
                if case == "periodic_peaks":
                    density[:] = native_density
                    for center in (
                        (0, 1, side - 1),
                        (3, 1, 2),
                        (side // 2, side // 2, side // 2),
                    ):
                        delta = [
                            np.minimum(abs(ax - pos), side - abs(ax - pos))
                            for ax, pos in zip((x, y, k), center)
                        ]
                        density[:] += (
                            35.0 * np.exp(-sum(d * d for d in delta) / 7.0)
                        ).astype("f")
                    if minmass and minmass > 1e11:
                        # A compact isolated peak crosses only after the transition
                        # to ordinary search; the broad peaks cross at higher mass.
                        center = (side // 2, 1, side // 2)
                        delta = [
                            np.minimum(abs(ax - pos), side - abs(ax - pos))
                            for ax, pos in zip((x, y, k), center)
                        ]
                        density[:] += (
                            50.0 * np.exp(-sum(d * d for d in delta) / 1.8)
                        ).astype("f")
                elif case == "zero":
                    density.fill(0)
                ref, _, native_masks = measure(libs[0], ic, inputs, z, False)
                got, counts, gpu_masks = measure(libs[1], ic, inputs, z, True)
                assert all(a.shape == b.shape for a, b in zip(ref, got, strict=True))
                assert all(native_masks[r] == value for r, value in gpu_masks.items())
                row = {
                    "optimized": bool(minmass),
                    "optimize_minmass_msun": minmass,
                    "matching_forbidden_masks_with_candidates": len(gpu_masks),
                    "case": case,
                    "side": side,
                    "coarse": coarse,
                    "box_mpc": length,
                    "threads": threads,
                    "seed": 37251,
                    "z": z,
                    "resolved_halos": int(ref[0].size),
                    "overlap_nonzero": int(np.count_nonzero(ref[2])),
                    "cuda": counts,
                    "fields": {},
                }
                for name, a, b in zip(
                    (
                        "masses",
                        "coordinates",
                        "overlap",
                        "star_rng",
                        "sfr_rng",
                        "xray_rng",
                    ),
                    ref,
                    got,
                    strict=True,
                ):
                    diff = np.abs(b.astype("d") - a.astype("d"))
                    row["fields"][name] = {
                        "count": int(a.size),
                        "different": int(np.count_nonzero(a != b)),
                        "max_abs": float(diff.max(initial=0.0)),
                        "outside_tolerance": int(
                            (~(diff <= 1e-8 + 1e-4 * np.abs(a))).sum()
                        ),
                        "native_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    }
                rows.append(row)
                path.write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    case,
                    side,
                    row["resolved_halos"],
                    {k: v["different"] for k, v in row["fields"].items()},
                    flush=True,
                )
    assert all(
        all(field["different"] == 0 for field in row["fields"].values()) for row in rows
    )
    assert all(
        r["resolved_halos"] > 0 and r["cuda"]["dexm_candidates"] > r["resolved_halos"]
        for r in rows
        if r["case"] == "periodic_peaks"
    )
    assert all(r["resolved_halos"] == 0 for r in rows if r["case"] == "zero")
    if optimized_only:
        assert any(
            r["cuda"].get("dexm_rebuilt_centres", 0)
            and r["matching_forbidden_masks_with_candidates"] > 1
            for r in rows
        )
        assert any(
            r["cuda"].get("dexm_optimized_accepted", 0) > 0
            and r["cuda"].get("dexm_ordinary_accepted", 0) > 0
            for r in rows
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--optimized" in sys.argv[2:])
