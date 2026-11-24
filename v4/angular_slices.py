"""4.3.2 sky sampling from a native pair of coevals on CUDA."""

from itertools import product

import cupy as cp
import numpy as np
from astropy import units
from cosmotile import make_lightcone_slice_interpolator
from cupyx.scipy.ndimage import map_coordinates, spline_filter
from py21cmfast.lightconers import Lightconer

from device_slices import interpolate_values


_rotate_shell = cp.ElementwiseKernel(
    "raw T cart, raw float64 matrix, int64 rays",
    "float64 rotated",
    """
    long ray = i % rays;
    int row = 3 * (i / rays);
    rotated = matrix[row] * cart[ray] + matrix[row + 1] * cart[rays + ray]
              + matrix[row + 2] * cart[2 * rays + ray];
    """,
    "rotate_angular_shell",
    options=("--fmad=false",),
)


class AngularShellGeometry:
    """Keep the native unit rays resident and form shell coordinates on CUDA."""

    def __init__(self, lightconer):
        self.lightconer = lightconer
        self.interpolator = make_lightcone_slice_interpolator(
            latitude=lightconer.latitude,
            longitude=lightconer.longitude,
            distance_to_shell=1.0 * units.pixel,
            interpolation_order=lightconer.interpolation_order,
        )
        self.rays = cp.asarray(
            units.Quantity(self.interpolator.keywords["coordinates"], units.pixel).value
        )
        self.rotation = (
            None
            if lightconer.rotation is None
            else cp.asarray(lightconer.rotation.as_matrix())
        )
        origin = lightconer.origin
        self.origin = (
            None if origin is None else cp.asarray(origin.to_value(units.pixel))
        )
        self.origin_scale = (
            0.0 if origin is None else float(np.max(np.abs(origin.value)))
        )

    def coordinates(self, lcd):
        radius = lcd.to_value(units.pixel)
        if radius <= 0:
            raise ValueError("distance_to_shell must be positive")
        coords = self.rays * radius
        if self.rotation is not None:
            coords = _rotate_shell(
                coords, self.rotation, np.int64(coords.shape[1]), size=coords.size
            ).reshape(coords.shape)
        if self.origin is not None:
            coords += self.origin[:, None]

        if self.rotation is not None and self.lightconer.interpolation_order == 0:
            # BLAS may round a rotation differently on the two devices. At a
            # voxel boundary the rounding must select the same nearest cell.
            guard = 32 * np.finfo(coords.dtype).eps * (radius + self.origin_scale)
            if cp.any(cp.abs(coords - (cp.floor(coords) + 0.5)) <= guard).item():
                native = self.lightconer._get_interpolator(lcd)
                coords = cp.asarray(
                    units.Quantity(native.keywords["coordinates"], units.pixel).value
                )
        return coords


class AngularPair:
    """Reuse the native slice iterator and distance selection with device fields."""

    def __init__(self, lightconer, shells=None, *, registry=None):
        self.geometry = lightconer
        self.shells = AngularShellGeometry(lightconer) if shells is None else shells
        self.registry = registry
        self.quantities = lightconer.quantities
        self.interp_kinds = lightconer.interp_kinds
        self.get_lc_distances_in_pixels = lightconer.get_lc_distances_in_pixels
        self.fields = {}
        self.distance = None

    def field(self, host):
        # Keep the owner alive: the key is valid only for the lifetime of this pair.
        if id(host) not in self.fields:
            dev = (
                cp.asarray(host)
                if self.registry is None
                else self.registry.lookup_host(host)
            )
            self.fields[id(host)] = host, dev
        return self.fields[id(host)][1]

    def shell(self, lcd):
        if self.distance == lcd:
            return
        self.distance = lcd
        self.interpolator = self.shells.interpolator
        raw = self.interpolator.keywords
        self.coords = self.shells.coordinates(lcd)
        self.order = raw["order"]
        # cosmotile 0.2.5 passes prefilter=False to SciPy. Its newer wrapper
        # applies a periodic spline prefilter before sampling higher orders.
        self.prefilter = raw.get("prefilter", True)
        self.boundary = raw.get("mode", "grid-wrap")
        self.weights = (
            None if raw.get("weights") is None else cp.asarray(raw["weights"])
        )
        self.direction = self.coords
        if self.shells.origin is not None:
            self.direction = self.coords - self.shells.origin[:, None]
        squares = self.direction**2
        self.norm = cp.sqrt(squares[0] + squares[1] + squares[2])

    @staticmethod
    def mix(a, b, lcd, dc_a, dc_b, kind):
        if kind not in ("mean", "mean_max"):
            raise ValueError("Unsupported interpolation in distance")
        if a.shape != b.shape:
            raise ValueError("Coeval field shapes differ")
        u, v, den = abs(dc_b - lcd), abs(dc_a - lcd), abs(dc_a - dc_b)
        dtype = (u * np.empty(0, dtype=a.dtype)).dtype.type
        return interpolate_values(
            a,
            b,
            cp.asarray(u.value, dtype=dtype),
            dtype(v.value),
            dtype(den.value),
            np.int32(kind == "mean_max"),
        )

    def sample(self, box):
        if self.order > 1 and self.prefilter:
            box = spline_filter(
                box, order=self.order, mode=self.boundary, output=cp.float64
            )
        # Use a double accumulator like SciPy. Only the sampled points need
        # promoting, even when a few rays intersect a very large float grid.
        values = map_coordinates(
            box,
            self.coords,
            output=cp.float64 if self.order else None,
            order=self.order,
            mode=self.boundary,
            prefilter=False,
        )
        return values.astype(box.dtype, copy=False)

    def average(self, values):
        if self.weights is None:
            return values
        return (self.weights[:, None] * values.reshape(self.weights.size, -1)).sum(
            axis=0
        )

    def interpolate_slice(
        self, lcd, coeval_a, coeval_b, dc_a, dc_b, coeval_res, kind="mean"
    ):
        self.shell(lcd)
        a, b = self.field(coeval_a), self.field(coeval_b)
        if self.geometry.samples_first(a.size, self.interpolator):
            if self.order == 1 and kind == "mean_max":
                # The sign rule acts on individual corners before trilinear sampling.
                lower = cp.floor(self.coords)
                frac = self.coords - lower
                lower = lower.astype(cp.int64)
                out = cp.zeros(frac.shape[1], dtype=cp.float64)
                for corner in product(range(2), repeat=3):
                    ix = tuple(
                        (lower[j] + bit) % a.shape[j] for j, bit in enumerate(corner)
                    )
                    weight = cp.prod(
                        cp.stack(
                            [
                                frac[j] if bit else 1 - frac[j]
                                for j, bit in enumerate(corner)
                            ]
                        ),
                        axis=0,
                    )
                    out += weight * self.mix(a[ix], b[ix], lcd, dc_a, dc_b, kind)
                return self.average(out).get()
            first, second = (self.average(self.sample(x)) for x in (a, b))
            return self.mix(first, second, lcd, dc_a, dc_b, kind).get()
        return self.average(self.sample(self.mix(a, b, lcd, dc_a, dc_b, kind))).get()

    def project(self, components):
        # cosmotile rounds each in-place component product to the sampled dtype
        # before summing the three axes and dividing by the ray length.
        sampled = cp.stack([self.sample(x) for x in components])
        sampled *= -self.direction
        return self.average((sampled[0] + sampled[1] + sampled[2]) / self.norm)

    def interpolate_los_velocity_slice(self, lcd, a, b, dc_a, dc_b):
        self.shell(lcd)
        kind = self.interp_kinds.get("velocity", "mean")
        va, vb = (
            [self.field(getattr(c, "velocity_" + ax)) for ax in "xyz"] for c in (a, b)
        )
        if kind == "mean" and self.geometry.samples_first(
            va[0].size, self.interpolator
        ):
            return self.mix(
                self.project(va), self.project(vb), lcd, dc_a, dc_b, kind
            ).get()
        components = (
            self.mix(x, y, lcd, dc_a, dc_b, kind) for x, y in zip(va, vb, strict=True)
        )
        return self.project(components).get() << units.one


def make_angular_slices(lightconer, first, second, shells=None, *, registry=None):
    yield from Lightconer.make_lightcone_slices(
        AngularPair(lightconer, shells, registry=registry), first, second
    )
