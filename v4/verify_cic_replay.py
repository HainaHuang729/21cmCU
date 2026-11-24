"""Check the 4.3.2 CUDA CIC scatter and exact repair against serial particle order."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np

import cic_ordered


@dataclass(frozen=True)
class Case:
    name: str
    fine: int
    coarse: int
    second_order: bool
    pattern: str
    seed: int


CASES = (
    Case("axis-rounding", 10, 10, True, "axis", 10),
    Case("half-ulp", 12, 12, False, "boundary", 11),
    Case("two-to-one", 32, 16, True, "signed", 12),
    Case("four-to-one", 32, 8, False, "periodic", 13),
    Case("partial-tile-block", 20, 5, True, "signed", 14),
    Case("order-sensitive-tile", 8, 2, False, "cancellation", 19),
    Case("aliased-tile-corners", 8, 2, True, "boundary", 15),
    Case("fractional-ratio", 18, 8, True, "periodic", 16),
    Case("one-cell-boundary", 4, 1, False, "boundary", 17),
    Case("one-cell-cancellation", 7, 1, True, "signed", 18),
)


def inputs(case):
    rng = np.random.default_rng(case.seed)
    n, m = case.fine, case.coarse
    rho = rng.uniform(-4, 3, (n, n, n)).astype("f")
    first = rng.uniform(-1, 1, (3, m, m, m)).astype("f")
    second = rng.uniform(-1, 1, first.shape).astype("f")
    growth, first_scale, second_scale = 0.73, 1.37, -0.38
    if case.pattern == "axis":
        first.fill(1)
        second.fill(1)
        # At coordinate 1, (1 + 2^-53) - 2^-53 is below 1, while
        # 1 + (2^-53 - 2^-53) is exactly 1. This affects corner assignment.
        first_scale = second_scale = 2.0**-53
    elif case.pattern == "boundary":
        first.fill(0)
        second.fill(0)
        growth = 2.0**-24
        rho.fill(1)
        if n == m:
            # On either side of the half-ULP at 1. Include values sufficiently
            # far away that some sites really stay outside the repair list.
            eps = np.finfo("f").eps
            rho.flat[1::5] = np.nextafter(np.float32(1), np.float32(0))
            rho.flat[2::5] = np.nextafter(np.float32(1), np.float32(2))
            rho.flat[3::5] = 1 + 32 * eps
            rho.flat[4::5] = 1 - 32 * eps
    elif case.pattern == "cancellation":
        # At coarse (0,0,0), native x order is [-B, .75, B] while the
        # wrapped tile visits [B, -B, .75]. Only exact replay gives zero.
        growth = 1.0
        first.fill(0)
        second.fill(0)
        rho.fill(-1)
        rho[0, 0, 0] = -0.75 * 2.0**60
        rho[1, 0, 0] = 0
        rho[7, 0, 0] = 2.0**60
    elif case.pattern == "signed":
        growth = 1.0
        masses = rng.choice([-(2.0**20), -0.125, 0.0, 0.125, 2.0**20], size=rho.shape)
        rho[:] = masses - 1.0
    else:
        first *= n
        second *= 2 * n
        first[:, 0, 0, 0] = [-n, 0, n]
    return (
        rho,
        first,
        second,
        np.float64(growth),
        np.float64(first_scale),
        np.float64(second_scale),
    )


def oracle(case, rho, vel, second, growth, first_scale, second_scale):
    n, m = case.fine, case.coarse
    coords = np.indices((n, n, n), dtype="d").reshape(3, -1)
    ratio = np.float64(m) / n
    nearest = (coords * ratio + 0.5).astype(np.int64) % m
    address = (slice(None), *nearest)
    first_shift = vel[address].astype("d") * first_scale
    position = coords + first_shift
    if case.second_order:
        correction = second[address].astype("d") * second_scale
        position[0] -= correction[0]
        position[1:] = coords[1:] + (first_shift[1:] - correction[1:])
    position *= ratio
    lower = np.floor(position).astype(np.int64)
    frac = position - lower
    particle_mass = 1.0 + rho.ravel().astype("d") * growth
    cells = np.empty((n**3, 8), dtype=np.int64)
    terms = np.empty(cells.shape, dtype="d")
    for corner in range(8):
        bits = np.array([(corner >> axis) & 1 for axis in range(3)])[:, None]
        cells[:, corner] = np.ravel_multi_index((lower + bits) % m, (m, m, m))
        wx, wy, wz = np.where(bits, frac, 1.0 - frac)
        terms[:, corner] = ((wx * wy) * wz) * particle_mass
    mass = np.zeros(m**3, dtype="d")
    # C order interleaves all eight corners of each particle, including zeros
    # and aliased corners. Repeated indices must accumulate without buffering.
    np.add.at(mass, cells.ravel(order="C"), terms.ravel(order="C"))
    counts = np.bincount(cells.ravel(), minlength=mass.size)
    return mass, counts, int(np.count_nonzero(particle_mass < 0))


class Capture:
    """Observe the actual production kernel choices and pre-repair outputs."""

    def __init__(self, module):
        self.module = module
        self.calls = []

    def get_function(self, name):
        kernel = self.module.get_function(name)

        def launch(grid, block, args):
            kernel(grid, block, args)
            self.calls.append(name)
            if name.startswith("scatter"):
                self.scatter = args[3].get().ravel()
                self.counts = args[4].get().ravel()
            elif name == "uncertain":
                self.flags = args[3].get().ravel()

        return launch


def bits(values):
    return np.ascontiguousarray(values).view(f"u{values.dtype.itemsize}")


def verify(case):
    n, m = case.fine, case.coarse
    rho, vel, second, growth, first_scale, second_scale = inputs(case)
    expected, counts, negative = oracle(
        case, rho, vel, second, growth, first_scale, second_scale
    )
    arrays = tuple(cp.asarray(a) for a in (rho, vel, second))
    params = (
        np.int32(n),
        np.int32(m),
        growth,
        first_scale,
        second_scale,
        np.bool_(case.second_order),
    )
    mass = cp.zeros((m, m, m), dtype="d")
    capture = Capture(cic_ordered._module)
    with patch.object(cic_ordered, "_module", capture):
        repaired = cic_ordered.deposit_ordered(*arrays, mass, *params)
    scatter = "scatter_tile4" if n == 4 * m and m >= 2 else "scatter"
    replay = "replay" if m == 1 else "replay_block"
    assert capture.calls == [scatter, "uncertain"] + ([replay] if repaired else [])
    np.testing.assert_array_equal(capture.counts, counts)
    assert counts.sum() == n**3 * 8 and repaired == int(capture.flags.sum())
    if case.pattern == "boundary":
        assert repaired > 0
        if n == m:
            assert repaired < mass.size

    raw_rounding = bits(capture.scatter.astype("f")) != bits(expected.astype("f"))
    assert not np.any(raw_rounding & ~capture.flags)
    if case.pattern == "cancellation":
        assert capture.scatter[0] == 0.75 and expected[0] == 0.0
        assert capture.flags[0]
    final = mass.get().ravel()
    np.testing.assert_array_equal(bits(final.astype("f")), bits(expected.astype("f")))
    np.testing.assert_array_equal(
        bits(final[capture.flags]), bits(expected[capture.flags])
    )

    # Force the actual repair kernel on unflagged cells too. For the larger
    # target grids include faces, corners, interior, and all naturally flagged
    # sites. NaN sentinels detect missing writes and writes outside that set.
    axis = [0, m // 2, m - 1]
    landmarks = np.ravel_multi_index(
        np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1), (m,) * 3
    )
    sites = (
        np.arange(m**3)
        if m <= 8
        else np.unique(
            np.r_[landmarks, np.flatnonzero(capture.flags), np.arange(0, m**3, 79)]
        )
    )
    sites = np.ascontiguousarray(sites[::-1], dtype=np.int64)
    selected = np.zeros(expected.size, dtype=bool)
    selected[sites] = True
    mass.fill(cp.nan)
    shift = abs(first_scale) * float(np.abs(vel).max())
    if case.second_order:
        shift += abs(second_scale) * float(np.abs(second).max())
    blocks, threads = ((sites.size + 31) // 32, 32) if m == 1 else (sites.size, 256)
    capture.module.get_function(replay)(
        (blocks,),
        (threads,),
        (
            *arrays,
            mass,
            cp.asarray(sites),
            np.int32(sites.size),
            cp.asarray(shift, dtype="d"),
            *params,
        ),
    )
    forced = mass.get().ravel()
    np.testing.assert_array_equal(bits(forced[selected]), bits(expected[selected]))
    assert np.isnan(forced[~selected]).all()
    return dict(
        name=case.name,
        fine=n,
        coarse=m,
        second_order=case.second_order,
        seed=case.seed,
        pattern=case.pattern,
        scatter=scatter,
        replay=replay,
        negative_particle_masses=negative,
        counts_tested=int(counts.size),
        corner_contributions=int(counts.sum()),
        rounded_cells_tested=int(expected.size),
        raw_rounding_mismatches=int(raw_rounding.sum()),
        naturally_replayed=int(repaired),
        forced_replay_sites=int(sites.size),
        failures=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    result = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        scope="Synthetic 4.3.2 CIC deposition and repair only. Unbuffered NumPy float64 accumulation orders fine particles in x/y/z order then the eight corners in native order.",
        checks="All scatter counts and final float32 masses; safe raw-scatter cells; float64 bits at every naturally or explicitly replayed cell; untouched cells after forced replay.",
        cases=[verify(case) for case in CASES],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
