"""Replay the worst failed density voxel per grid with stage-wise FFT substitutions."""

import ctypes as C
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state
from tests import produce_integration_test_data as official
import perturb
from trace_fft import native_stage


def replay(row):
    f, n, box = row["fine"], row["low"], row["box"]
    target = row["fields"]["density"]["worst_failed"]
    idx = tuple(target["xyz"])
    with patch.object(official, "SEED", row["seed"]):
        inputs = official.get_all_options_struct(
            10, **official.OPTIONS_PT[row["case"]][1]
        )["inputs"].evolve_input_structs(
            DIM=f,
            HII_DIM=n,
            BOX_LEN=box,
            KEEP_3D_VELOCITIES=True,
            SMOOTH_EVOLVED_DENSITY_FIELD=row["smooth"],
        )
    ic = p21.compute_initial_conditions(inputs=inputs)
    kw = dict(redshift=row["z"], initial_conditions=ic)
    cpu = p21.perturb_field(**kw).get("density")
    gpu = perturb.perturb_field_gpu(**kw).get("density")
    stages = dict(
        case=row["case"],
        z=row["z"],
        seed=row["seed"],
        smooth=row["smooth"],
        target=target,
        cpu=float(cpu[idx]),
        gpu=float(gpu[idx]),
    )
    lib = C.CDLL(p21.c_21cmfast.__file__)
    native = perturb.reference_density

    class Capture:
        def __getattr__(self, name):
            return getattr(lib, name)

        def smooth_and_clip_density(self, coarse, source):
            shape = (n, n, 2 * (n // 2 + 1))
            grid = np.ctypeslib.as_array(C.cast(coarse, C.POINTER(C.c_float)), shape)[
                :, :, :n
            ].copy()
            stages["native_coarse_before_smoothing"] = float(grid[idx])
            lib.smooth_and_clip_density(coarse, source)

    def probe(_, inputs, grid):
        m = grid.shape[0]
        storage = np.zeros((m, m, 2 * (m // 2 + 1)), dtype=np.float32)
        ic.push_to_backend()
        address = int(p21.c_21cmfast.ffi.cast("uintptr_t", ic._cstruct))
        with c_state(inputs, broadcast_inputs=True):
            lib.make_density_grid(
                C.c_float(row["z"]),
                C.c_void_p(storage.ctypes.data),
                C.c_void_p(address),
            )
        difference = abs(storage[:, :, :m] - grid.get())
        stages["same_ic_deposit_max_abs"] = float(difference.max())
        stages["same_ic_deposit_different"] = int(np.count_nonzero(difference))
        if difference.any():
            site = np.unravel_index(difference.argmax(), difference.shape)
            stages["deposit_worst"] = dict(
                xyz=list(map(int, site)),
                cpu=float(storage[:, :, :m][site]),
                gpu=float(grid[site].get()),
            )
        cpu_mass = cp.asarray(storage[:, :, :m].copy())
        del storage, difference
        original = perturb.transform

        def swap_mass(array, size, inverse=False, **kwargs):
            if size == m and not inverse:
                array = cpu_mass
            return original(array, size, inverse, **kwargs)

        with patch.object(perturb, "transform", swap_mass):
            swapped = perturb.perturb_field_gpu(**kw, strict_fft=True).get("density")
        stages["gpu_tail_on_cpu_deposit"] = dict(
            max_abs=float(abs(swapped - cpu).max()),
            different=int((swapped != cpu).sum()),
            failed=int((abs(swapped - cpu) > 1e-8 + 1e-4 * abs(cpu)).sum()),
            target=float(swapped[idx]),
        )
        del cpu_mass
        full = native(Capture(), inputs, grid)
        stages["native_tail_on_gpu_deposit_max_abs"] = float(abs(full - cpu).max())
        stages["native_tail_on_gpu_deposit_site"] = float(full[idx])
        if inputs.matter_options.PERTURB_ON_HIGH_RES:
            r = np.float32(0.620350491 * box / n)
            fine_site = tuple(int(i * f / n + 0.5) % f for i in idx)
            with c_state(inputs, broadcast_inputs=True):
                nf = native_stage(lib, grid.get(), "forward", r)
                gf = perturb.transform(
                    grid, f, threads=inputs.simulation_options.N_THREADS
                )
                stages["same_mass_forward_max_complex_abs"] = float(
                    cp.max(abs(gf - cp.asarray(nf))).get()
                )
                k2 = sum(k * k for k in perturb.modes(f, box))
                kr = (
                    (cp.sqrt(k2.astype(cp.float64)) * r)
                    .astype(cp.float32)
                    .astype(cp.float64)
                )
                window = cp.where(
                    kr < 1e-4,
                    1 - kr**2 / 10,
                    3
                    * (cp.sin(kr) - kr * cp.cos(kr))
                    / cp.where(kr == 0, 1.0, kr) ** 3,
                )
                nw = native_stage(lib, nf, "window", r)
                stages["same_spectrum_tophat_max_complex_abs"] = float(
                    cp.max(
                        abs(
                            (cp.asarray(nf) * window).astype(cp.complex64)
                            - cp.asarray(nw)
                        )
                    ).get()
                )
                ni = native_stage(lib, nw, "inverse", r)
                gi = perturb.transform(
                    cp.asarray(nw), f, True, threads=inputs.simulation_options.N_THREADS
                ) / np.float32(f**3)
                stages["same_filtered_inverse_site"] = [
                    float(ni[fine_site] / np.float32(f**3)),
                    float(gi[fine_site].get()),
                ]
                stages["fine_site"] = fine_site
                stages["gpu_full_fine_inverse_site"] = float(
                    (
                        perturb.transform(
                            (gf * window).astype(cp.complex64),
                            f,
                            True,
                            threads=inputs.simulation_options.N_THREADS,
                        )[fine_site]
                        / np.float32(f**3)
                    ).get()
                )
        return full

    with patch.object(perturb, "reference_density", probe):
        perturb.perturb_field_gpu(**kw, reference_fft=True)
    return stages


if __name__ == "__main__":
    rows = json.loads(Path(sys.argv[1]).read_text())["cases"]
    assert len(rows) == 24
    row = max(
        rows,
        key=lambda r: r["fields"]["density"].get("worst_failed", {}).get("delta", -1),
    )
    Path(sys.argv[2]).write_text(json.dumps(replay(row), indent=2))
