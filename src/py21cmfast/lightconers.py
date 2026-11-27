"""A module for classes that create lightcone slices from Coeval objects."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from functools import cached_property, partial
import threading
from typing import TYPE_CHECKING

import attrs
import deprecation
import numpy as np
from astropy import units
from astropy.cosmology import FLRW, z_at_value
from astropy.units import Mpc, Quantity, pixel, pixel_scale
from classy import Class
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from ._cfg import config
from .c_21cmfast import ffi, lib
from .rsds import estimate_rsd_displacements
from .wrapper.classy_interface import run_classy
from .wrapper.inputs import (
    InputParameters,
    Planck18,  # Not *quite* the same as astropy's Planck18
    SimulationOptions,
)

if TYPE_CHECKING:
    from .drivers.coeval import Coeval

_LIGHTCONERS = {}
_LENGTH = "length"
_LIGHTCONER_CLASSY_CACHE: OrderedDict[tuple, Class] = OrderedDict()
_LIGHTCONER_CLASSY_LOCK = threading.RLock()


def clear_lightconer_class_cache() -> None:
    """Clear the cached CLASS instances used by lightcone RSD boundary estimation."""
    with _LIGHTCONER_CLASSY_LOCK:
        _LIGHTCONER_CLASSY_CACHE.clear()


@attrs.define(kw_only=True, slots=False)
class Lightconer(ABC):
    """A class that creates lightcone slices from Coeval objects.

    Parameters
    ----------
    lc_distances
        The comoving distances to the lightcone slices, in Mpc. Either this or
        the ``lc_redshifts`` must be provided.
    lc_redshifts
        The redshifts of the lightcone slices. Either this or the ``lc_distances``
        must be provided.
    cosmo
        The cosmology to use. Defaults to Planck18.
    quantities
        An iteratable of quantities to include in the lightcone slices. These should
        be attributes of the :class:~`outputs.Coeval` class that are arrays of
        shape ``HII_DIM^3``. A *special* value here is `velocity_los`, which  Defaults to ``("brightness_temp",)``.
    """

    cosmo: FLRW = attrs.field(
        default=Planck18, validator=attrs.validators.instance_of(FLRW)
    )

    _lc_redshifts: np.ndarray = attrs.field(default=None, eq=False)
    lc_distances: Quantity[_LENGTH] = attrs.field(
        eq=attrs.cmp_using(eq=partial(np.allclose, rtol=1e-5, atol=0))
    )

    quantities: tuple[str] = attrs.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attrs.validators.deep_iterable(attrs.validators.instance_of(str)),
    )
    interp_kinds: dict[str, str] = attrs.field()

    @lc_distances.default
    def _lcd_default(self):
        if self._lc_redshifts is None:
            raise ValueError("Either lc_distances or lc_redshifts must be provided")
        return self.cosmo.comoving_distance(self._lc_redshifts)

    @lc_distances.validator
    def _lcd_vld(self, attribute, value):
        if np.any(value < 0):
            raise ValueError("lc_distances must be non-negative")

    @_lc_redshifts.validator
    def _lcz_vld(self, attribute, value):
        if value is None:
            return

        if np.any(value < 0):
            raise ValueError("lc_redshifts must be non-negative")

    @cached_property
    def lc_redshifts(self) -> np.ndarray:
        """The redshifts of the lightcone slices."""
        if self._lc_redshifts is not None:
            return self._lc_redshifts

        d = self.lc_distances
        zmin = z_at_value(self.cosmo.comoving_distance, d.min()).value
        zmax = z_at_value(self.cosmo.comoving_distance, d.max()).value

        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        dgrid = self.cosmo.comoving_distance(zgrid)
        return np.interp(d.value, dgrid.value, zgrid)

    def get_lc_distances_in_pixels(self, resolution: Quantity[_LENGTH]):
        """Get the lightcone distances in pixels, given a resolution."""
        return self.lc_distances.to(pixel, pixel_scale(resolution / pixel))

    @interp_kinds.default
    def _interp_kinds_def(self):
        return {"z_reion": "mean_max"}

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int, int]:
        """Get the shape of the lightcone slices."""
        raise NotImplementedError

    @classmethod
    def between_redshifts(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity[_LENGTH],
        cosmo=Planck18,
        **kw,
    ):
        """Construct a Lightconer with regular comoving dist. slices between two z's."""
        d_at_redshift = cosmo.comoving_distance(min_redshift).to_value(Mpc)
        dmax = cosmo.comoving_distance(max_redshift).to_value(Mpc)
        res = resolution.to_value(Mpc)

        lc_distances = np.arange(d_at_redshift, dmax + res, res)

        return cls(lc_distances=lc_distances * Mpc, cosmo=cosmo, **kw)

    @classmethod
    def with_equal_cdist_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity[_LENGTH],
        cosmo=Planck18,
        **kw,
    ):
        """
        Create a lightconer with equally spaced slices in comoving distance.

        This method is deprecated and will be removed in future versions.
        Instead, use `between_redshifts` to create a lightconer with equally spaced
        slices in comoving distance.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "with_equal_cdist_slices",
                deprecated_in="4.2.0",
                removed_in="5.0.0",
                details=(
                    "with_equal_cdist_slices is deprecated and will be removed in a future version. "
                    "Call between_redshifts instead to silence this warning."
                ),
            ),
            stacklevel=2,
        )
        return cls.between_redshifts(
            min_redshift=min_redshift,
            max_redshift=max_redshift,
            resolution=resolution,
            cosmo=cosmo,
            **kw,
        )

    def make_lightcone_slices(
        self,
        c1: Coeval,
        c2: Coeval,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Make lightcone slices out of two coeval objects.

        Parameters
        ----------
        c1, c2 : Coeval
            The coeval boxes to interpolate.

        Returns
        -------
        quantity
            The field names of the quantities required by the lightcone.
        lcidx
            The indices of the lightcone to which these slices belong.
        scalar_field_slices
            The scalar fields evaluated on the "lightcone" slices that exist within
            the redshift range spanned by ``c1`` and ``c2``.
        """
        if c1.simulation_options != c2.simulation_options:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        cosmo = c1.cosmo_params.cosmo
        pixeleq = pixel_scale(c1.simulation_options.cell_size / pixel)

        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)

        pixlcdist = self.get_lc_distances_in_pixels(c1.simulation_options.cell_size)

        # At the lower redshift, we include some tolerance. This is because the very
        # last slice (lowest redshift) may correspond *exactly* to the lowest coeval
        # box, and due to rounding error in the `z_at_value` call, they might be
        # slightly off.
        lcidx = np.nonzero((pixlcdist >= dcmin * (1 - 1e-6)) & (pixlcdist < dcmax))[0]

        # Return early if no lightcone indices are between the coeval distances.
        if len(lcidx) == 0:
            yield None, lcidx, None

        lc_distances = pixlcdist[lcidx]

        lc_quantities = self.quantities
        coeval_res = c1.simulation_options.cell_size

        for idx, lcd in zip(lcidx, lc_distances, strict=True):
            for q in lc_quantities:
                if q == "los_velocity":
                    continue

                yield (
                    q,
                    idx,
                    self.interpolate_slice(
                        lcd,
                        getattr(c1, q),
                        getattr(c2, q),
                        dc1,
                        dc2,
                        coeval_res,
                        kind=self.interp_kinds.get(q, "mean"),
                    ),
                )

                if "los_velocity" in self.quantities and q == self.quantities[0]:
                    # While doing the first quantity, also add in the los velocity, if desired.
                    # Doing it now means we can keep whatever cached interpolation setup
                    # is used to do construct_lightcone().
                    yield (
                        "los_velocity",
                        idx,
                        self.interpolate_los_velocity_slice(lcd, c1, c2, dc1, dc2),
                    )

    def interpolate_slice(
        self,
        lcd: Quantity[pixel],
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
        coeval_res: Quantity[_LENGTH],
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one lightcone slice of a field from the two coevals that bracket it.

        The coevals are first sub-selected (:meth:`coeval_subselect`), then
        interpolated to the slice's distance (:meth:`redshift_interpolation`), then
        made into a slice (:meth:`construct_lightcone`).

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.
        coeval_a, coeval_b
            The field in the two coeval boxes.
        dc_a, dc_b
            The comoving distances of the two coeval boxes, in pixels.
        coeval_res
            The cell size of the coeval boxes.
        kind
            The kind of interpolation in distance (see :meth:`redshift_interpolation`).
        """
        box_a = self.coeval_subselect(lcd, coeval_a, coeval_res)
        box_b = self.coeval_subselect(lcd, coeval_b, coeval_res)
        box = self.redshift_interpolation(lcd, box_a, box_b, dc_a, dc_b, kind=kind)
        return self.construct_lightcone(lcd, box)

    def interpolate_los_velocity_slice(
        self,
        lcd: Quantity[pixel],
        c_a: Coeval,
        c_b: Coeval,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
    ) -> np.ndarray:
        """Make one slice of the line-of-sight velocity from two bracketing coevals.

        By default, this is the z-component of the velocity, interpolated like
        :meth:`interpolate_slice` does, and made into a slice by
        :meth:`construct_los_velocity_lightcone`.
        """
        # TODO: get the correct component according to self.line_of_sight_axis
        coeval_res = c_a.simulation_options.cell_size
        box_a = self.coeval_subselect(lcd, c_a.velocity_z, coeval_res)
        box_b = self.coeval_subselect(lcd, c_b.velocity_z, coeval_res)
        box = self.redshift_interpolation(
            lcd,
            box_a,
            box_b,
            dc_a,
            dc_b,
            kind=self.interp_kinds.get("velocity_z", "mean"),
        )
        return self.construct_los_velocity_lightcone(lcd, box)

    def coeval_subselect(
        self, lcd: float, coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ) -> np.ndarray:
        """Sub-Select the coeval box required for interpolation at one slice."""
        return coeval

    def redshift_interpolation(
        self,
        dc: float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
        if coeval_a.shape != coeval_b.shape:
            raise ValueError("coeval_a and coeval_b must have the same shape")

        out = (np.abs(dc_b - dc) * coeval_a + np.abs(dc_a - dc) * coeval_b) / np.abs(
            dc_a - dc_b
        )

        if kind == "mean_max":
            flag = coeval_a * coeval_b < 0
            out[flag] = np.maximum(coeval_a, coeval_b)[flag]
        elif kind != "mean":
            raise ValueError("kind must be 'mean' or 'mean_max'")

        return out

    @abstractmethod
    def construct_lightcone(
        self,
        lc_distances: np.ndarray,
        boxes: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the lightcone slices."""

    @abstractmethod
    def construct_los_velocity_lightcone(
        self,
        lc_distances: np.ndarray,
        velocities: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the LoS velocity lightcone slices."""

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Lightconer:
        """Validate 21cmFAST options."""
        if len(inputs.node_redshifts) == 0:
            raise ValueError(
                "You are attempting to run a lightcone with no node_redshifts."
                "Please set node_redshifts on the `inputs` parameter."
            )
        if self.cosmo != inputs.cosmo_params.cosmo:
            raise ValueError(
                "The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters. "
                "Please make sure that your lightconer.cosmo is the same as inputs.cosmo_params.cosmo."
            )
        lcd = self.lc_distances
        d_node_min = self.cosmo.comoving_distance(min(inputs.node_redshifts))
        d_node_max = self.cosmo.comoving_distance(max(inputs.node_redshifts))
        if not ((d_node_min <= lcd.min()) and (lcd.max() <= d_node_max)):
            lcz = self.lc_redshifts
            raise ValueError(
                "The lightcone redshifts are not compatible with the given redshift."
                f"The range of computed redshifts is {min(inputs.node_redshifts)} to {max(inputs.node_redshifts)}, "
                f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}. "
                "Extend the limits of node redshifts to avoid this error."
            )

        if (
            include_dvdr_in_tau21
            and inputs.astro_options.USE_TS_FLUCT
            and "tau_21" not in self.quantities
        ):
            lightconer = attrs.evolve(self, quantities=(*self.quantities, "tau_21"))
        else:
            lightconer = self

        if apply_rsds or include_dvdr_in_tau21:
            lightconer = attrs.evolve(
                lightconer, quantities=(*lightconer.quantities, "los_velocity")
            )

        if not apply_rsds:
            return lightconer

        if classy_output is None:
            ckey = (
                float(inputs.cosmo_params.hlittle),
                float(inputs.cosmo_params.OMm - inputs.cosmo_params.OMb),
                float(inputs.cosmo_params.OMb),
                float(inputs.cosmo_params.POWER_INDEX),
                float(inputs.cosmo_params.SIGMA_8),
            )
            with _LIGHTCONER_CLASSY_LOCK:
                classy_output = _LIGHTCONER_CLASSY_CACHE.get(ckey)
                if classy_output is not None:
                    _LIGHTCONER_CLASSY_CACHE.move_to_end(ckey)
            if classy_output is None:
                classy_output = run_classy(
                    h=inputs.cosmo_params.hlittle,
                    Omega_cdm=inputs.cosmo_params.OMm - inputs.cosmo_params.OMb,
                    Omega_b=inputs.cosmo_params.OMb,
                    n_s=inputs.cosmo_params.POWER_INDEX,
                    sigma8=inputs.cosmo_params.SIGMA_8,
                    output="vTk",
                )
                with _LIGHTCONER_CLASSY_LOCK:
                    _LIGHTCONER_CLASSY_CACHE[ckey] = classy_output
                    _LIGHTCONER_CLASSY_CACHE.move_to_end(ckey)
                    while len(_LIGHTCONER_CLASSY_CACHE) > 16:
                        _LIGHTCONER_CLASSY_CACHE.popitem(last=False)
        lcd_limits_rsd = self.find_required_lightcone_limits(
            classy_output=classy_output, inputs=inputs
        )
        lcd_rsd = (
            np.arange(
                lcd_limits_rsd[0].value,
                lcd_limits_rsd[1].value + inputs.simulation_options.cell_size.value,
                inputs.simulation_options.cell_size.value,
            )
            * self.lc_distances.unit
        )
        # Make a new lightconer which is identical to self, but extends to further boundaries!
        return attrs.evolve(lightconer, lc_redshifts=None, lc_distances=lcd_rsd)

    def find_required_lightcone_limits(
        self, classy_output: Class, inputs: InputParameters
    ) -> list[units.Quantity]:
        """Obtain the redshift limits required for the lightcone to include RSDs.

        This is a *crude* estimation of the maximum/minimum lightcone limits that are
        required in order to simulate all the "mass" that enters the requested
        ligthcone (due to RSD shift). We use the rms of the velocity field from linear
        perturbation theory in order to determine the required lightcone limits.
        If no limit is found, it means that the limits of node_redshifts are not
        sufficient and an error is raised.

        Parameters
        ----------
        classy_output : classy.Class
            An object containing all the information from the CLASS calculation.
        inputs: InputParameters
            The input parameters corresponding to the box.

        Returns
        -------
        lcd_limits_rsd : list
            List that contains the limits of the required lightcone distances.
        """
        # The velocity rms that CLASS returns corresponds to the magnitude of the velocity vector.
        # In case of rectilinear lightcones, since we shift the cells only along one axis (e.g. z-axis),
        # we need the rms of the correponding component of the velocity vector, which is smaller by sqrt(3)
        # due to isotropy
        factor = 1.0 if isinstance(self, AngularLightconer) else 1.0 / np.sqrt(3.0)
        z_node_limits = [min(inputs.node_redshifts), max(inputs.node_redshifts)]
        lcd_limits = [self.lc_distances.min(), self.lc_distances.max()]
        signs = [-1, 1]
        lcd_limits_rsd = []
        for sign, z_node_limit, lcd_limit in zip(
            signs, z_node_limits, lcd_limits, strict=False
        ):
            distances_out = (
                np.arange(
                    lcd_limit.value,
                    self.cosmo.comoving_distance(z_node_limit).value,
                    inputs.simulation_options.cell_size.value * sign,
                )
                * lcd_limit.unit
            )
            if len(distances_out) > 0:
                displacements = estimate_rsd_displacements(
                    classy_output=classy_output,
                    cosmo=self.cosmo,
                    redshifts=z_at_value(self.cosmo.comoving_distance, distances_out),
                    factor=factor,
                )
                if sign < 0:
                    out_indices = distances_out + displacements < lcd_limit
                else:
                    out_indices = distances_out - displacements > lcd_limit
            else:
                out_indices = False
            if np.any(out_indices):
                # If there are displaced coordinates outside the lightcone box, we return the
                # corresponding distance that is closest to the original lightcone boundaries
                lcd_limits_rsd.append(distances_out[out_indices][0])
            else:
                raise ValueError(
                    f"You have set apply_rsds to True with node redshifts between {min(inputs.node_redshifts)} and {max(inputs.node_redshifts)} "
                    f"and lightcone redshifts between {self.lc_redshifts.min()} and {self.lc_redshifts.max()}. "
                    "However, RSDs are expected to contribute the lightcone from higer/lower rdshifts. "
                    "Extend the limits of node redshifts to avoid this error."
                )
        return lcd_limits_rsd

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        return super().__init_subclass__()


@attrs.define(kw_only=True, slots=False)
class RectilinearLightconer(Lightconer):
    """The class rectilinear lightconer."""

    index_offset: int = attrs.field()
    line_of_sight_axis: int = attrs.field(
        default=-1, converter=int, validator=attrs.validators.in_([-1, 0, 1, 2])
    )

    @index_offset.default
    def _index_offset_default(self) -> int:
        # While it probably makes more sense to use zero as the default offset,
        # we use n_lightcone to maintain default backwards compatibility.
        return len(self.lc_distances)

    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ):
        """Sub-select the coeval slice corresponding to this coeval distance."""
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")

    def construct_lightcone(
        self,
        lcd: np.ndarray,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct slices of the lightcone between two coevals."""
        return box

    def construct_los_velocity_lightcone(
        self,
        lcd: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Construct slices of the lightcone between two coevals."""
        return velocity

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int, int]:
        """Get the shape of the lightcone."""
        return (
            simulation_options.HII_DIM,
            simulation_options.HII_DIM,
            len(self.lc_distances),
        )

    def make_lightcone_slices(self, c1: Coeval, c2: Coeval):
        """Batch-interpolate rectilinear lightcone slices on GPU when CUDA is enabled."""
        if not (
            config["USE_CUDA"]
            and lib.p21_cuda_is_compiled()
            and lib.p21_cuda_is_available()
        ):
            yield from super().make_lightcone_slices(c1, c2)
            return

        if c1.simulation_options != c2.simulation_options:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        cosmo = c1.cosmo_params.cosmo
        coeval_res = c1.simulation_options.cell_size
        pixeleq = pixel_scale(coeval_res / pixel)
        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)
        denom = float(np.abs(dc1 - dc2).value)
        if denom <= 0.0:
            yield from super().make_lightcone_slices(c1, c2)
            return

        pixlcdist = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = np.nonzero((pixlcdist >= dcmin * (1 - 1e-6)) & (pixlcdist < dcmax))[0]
        if len(lcidx) == 0:
            yield None, lcidx, None
            return

        lc_distances = pixlcdist[lcidx]
        max_lcpix = pixlcdist.max()
        n_slices = int(len(lcidx))
        w_a = np.ascontiguousarray(
            np.abs(dc2 - lc_distances).value / denom, dtype=np.float32
        )
        w_b = np.ascontiguousarray(
            np.abs(dc1 - lc_distances).value / denom, dtype=np.float32
        )

        # Resolve per-slice z-indices modulo nz from the first array's depth.
        z_indices_raw = [
            -int((max_lcpix - lcd + 1 * pixel).to_value(pixel)) + self.index_offset
            for lcd in lc_distances
        ]

        batched: dict[str, np.ndarray] = {}

        def _try_batch(name: str, attr_name: str, kind: str) -> None:
            if kind not in ("mean", "mean_max"):
                return
            from .drivers.coeval import _get_coeval_array_no_sync

            arr_a = _get_coeval_array_no_sync(c1, attr_name)
            arr_b = _get_coeval_array_no_sync(c2, attr_name)
            if arr_a is None:
                arr_a = np.asarray(getattr(c1, attr_name))
            if arr_b is None:
                arr_b = np.asarray(getattr(c2, attr_name))
            if arr_a.ndim != 3 or arr_a.shape != arr_b.shape:
                return
            nx, ny, nz = arr_a.shape
            a32 = arr_a if (arr_a.dtype == np.float32 and arr_a.flags.c_contiguous) else np.ascontiguousarray(arr_a, dtype=np.float32)
            b32 = arr_b if (arr_b.dtype == np.float32 and arr_b.flags.c_contiguous) else np.ascontiguousarray(arr_b, dtype=np.float32)
            z_idx = np.ascontiguousarray(
                np.mod(z_indices_raw, nz), dtype=np.int32
            )
            out = np.empty((n_slices, nx, ny), dtype=np.float32)
            rc = lib.p21_cuda_rectilinear_slices(
                ffi.cast("const float *", ffi.from_buffer(a32)),
                ffi.cast("const float *", ffi.from_buffer(b32)),
                int(nx),
                int(ny),
                int(nz),
                n_slices,
                ffi.cast("const int *", ffi.from_buffer(z_idx)),
                ffi.cast("const float *", ffi.from_buffer(w_a)),
                ffi.cast("const float *", ffi.from_buffer(w_b)),
                1 if kind == "mean_max" else 0,
                ffi.cast("float *", ffi.from_buffer(out)),
            )
            if rc == 0:
                batched[name] = out

        for q in self.quantities:
            if q == "los_velocity":
                continue
            _try_batch(q, q, self.interp_kinds.get(q, "mean"))
        if "los_velocity" in self.quantities:
            _try_batch(
                "los_velocity",
                "velocity_z",
                self.interp_kinds.get("velocity_z", "mean"),
            )

        for s, (idx, lcd) in enumerate(zip(lcidx, lc_distances, strict=True)):
            for q in self.quantities:
                if q == "los_velocity":
                    continue
                if q in batched:
                    yield (q, idx, batched[q][s])
                else:
                    yield (
                        q,
                        idx,
                        self.interpolate_slice(
                            lcd,
                            getattr(c1, q),
                            getattr(c2, q),
                            dc1,
                            dc2,
                            coeval_res,
                            kind=self.interp_kinds.get(q, "mean"),
                        ),
                    )
                if "los_velocity" in self.quantities and q == self.quantities[0]:
                    if "los_velocity" in batched:
                        yield ("los_velocity", idx, batched["los_velocity"][s])
                    else:
                        yield (
                            "los_velocity",
                            idx,
                            self.interpolate_los_velocity_slice(lcd, c1, c2, dc1, dc2),
                        )


@attrs.define(kw_only=True, slots=False)
class ObliqueRectilinearLightconer(RectilinearLightconer):
    """Rectilinear lightconer slicing along an oblique periodic direction to avoid LOS repetition."""

    direction: tuple[float, float, float] = attrs.field(
        default=(3.0, 5.0, 11.0),
        converter=lambda v: (float(v[0]), float(v[1]), float(v[2])),
    )
    origin: tuple[float, float, float] = attrs.field(
        default=(0.0, 0.0, 0.0),
        converter=lambda v: (float(v[0]), float(v[1]), float(v[2])),
    )
    interpolation_order: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.in_([0, 1])
    )

    @direction.validator
    def _direction_validator(self, attribute, value):
        if float(np.linalg.norm(value)) <= 0.0:
            raise ValueError(f"direction must be non-zero, got {value}")

    def __attrs_post_init__(self) -> None:
        self._oblique_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def orthonormal_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return right-handed orthonormal triad ``(e1, e2, u_los)``."""
        d = np.asarray(self.direction, dtype=np.float64)
        u_los = d / np.linalg.norm(d)
        ref = np.array([1.0, 0.0, 0.0] if abs(u_los[0]) < 0.9 else [0.0, 1.0, 0.0], dtype=np.float64)
        e1 = ref - np.dot(ref, u_los) * u_los
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_los, e1)
        e2 /= np.linalg.norm(e2)
        return e1, e2, u_los

    def _transverse_offsets(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
        key = (int(nx), int(ny))
        cached = self._oblique_cache.get(key)
        if cached is not None:
            return cached
        e1, e2, u_los = self.orthonormal_basis()
        orig = np.asarray(self.origin, dtype=np.float64)
        ii, jj = np.meshgrid(
            np.arange(nx, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            indexing="ij",
        )
        ii_f = ii.ravel()
        jj_f = jj.ravel()
        base = orig[:, None] + e1[:, None] * ii_f[None, :] + e2[:, None] * jj_f[None, :]
        base = np.ascontiguousarray(base, dtype=np.float64)
        unit_xyz = np.ascontiguousarray(
            np.broadcast_to(u_los[:, None], (3, nx * ny)), dtype=np.float64
        )
        self._oblique_cache[key] = (base, unit_xyz)
        return base, unit_xyz

    def _slice_coords(self, s_pix: float, nx: int, ny: int) -> np.ndarray:
        _, _, u_los = self.orthonormal_basis()
        base, _ = self._transverse_offsets(nx, ny)
        return np.ascontiguousarray(base + u_los[:, None] * float(s_pix), dtype=np.float64)

    def _sample_cpu(
        self,
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        coords: np.ndarray,
        w_a: float,
        w_b: float,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
        lcd: Quantity[pixel],
        kind: str,
        out_shape: tuple[int, int],
    ) -> np.ndarray:
        if kind == "mean_max" and self.interpolation_order == 1:
            shape = np.array(arr_a.shape, dtype=np.int64)[:, None]
            floor = np.floor(coords)
            frac = coords - floor
            lower = floor.astype(np.int64) % shape
            upper = (lower + 1) % shape
            out = np.zeros(coords.shape[1], dtype=np.float64)
            for corner in np.ndindex(2, 2, 2):
                idx_c = tuple(upper[ax] if c else lower[ax] for ax, c in enumerate(corner))
                weight = np.prod(
                    [frac[ax] if c else 1.0 - frac[ax] for ax, c in enumerate(corner)],
                    axis=0,
                )
                out += weight * _dimensionless(
                    self.redshift_interpolation(
                        lcd, arr_a[idx_c], arr_b[idx_c], dc_a, dc_b, kind="mean_max"
                    )
                )
            return out.astype(np.float32).reshape(out_shape)

        sa = map_coordinates(arr_a, coords, order=self.interpolation_order, mode="grid-wrap")
        sb = map_coordinates(arr_b, coords, order=self.interpolation_order, mode="grid-wrap")
        if kind == "mean_max":
            return _dimensionless(
                self.redshift_interpolation(lcd, sa, sb, dc_a, dc_b, kind="mean_max")
            ).astype(np.float32).reshape(out_shape)
        return (w_a * sa + w_b * sb).astype(np.float32).reshape(out_shape)

    def _sample_los_velocity_cpu(
        self,
        c1: Coeval,
        c2: Coeval,
        coords: np.ndarray,
        w_a: float,
        w_b: float,
        out_shape: tuple[int, int],
    ) -> np.ndarray:
        _, _, u_los = self.orthonormal_basis()
        va = np.zeros(coords.shape[1], dtype=np.float64)
        vb = np.zeros(coords.shape[1], dtype=np.float64)
        for ax_idx, ax_name in enumerate("xyz"):
            comp_a = np.asarray(getattr(c1, f"velocity_{ax_name}"), dtype=np.float32)
            comp_b = np.asarray(getattr(c2, f"velocity_{ax_name}"), dtype=np.float32)
            va += u_los[ax_idx] * map_coordinates(
                comp_a, coords, order=self.interpolation_order, mode="grid-wrap"
            )
            vb += u_los[ax_idx] * map_coordinates(
                comp_b, coords, order=self.interpolation_order, mode="grid-wrap"
            )
        return (w_a * va + w_b * vb).astype(np.float32).reshape(out_shape)

    def _s_pix_for_distance(self, lcd: Quantity[pixel], coeval_res: Quantity[_LENGTH]) -> float:
        pixlcdist = self.get_lc_distances_in_pixels(coeval_res)
        return float((lcd - pixlcdist.min()).to_value(pixel))

    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ):
        """Sample an oblique planar slice through a 3D coeval array at distance ``lcd``."""
        nx, ny, _ = coeval.shape
        coords = self._slice_coords(self._s_pix_for_distance(lcd, coeval_res), nx, ny)
        return map_coordinates(
            np.asarray(coeval, dtype=np.float32),
            coords,
            order=self.interpolation_order,
            mode="grid-wrap",
        ).astype(np.float32).reshape((nx, ny))

    def interpolate_slice(
        self,
        lcd: Quantity[pixel],
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
        coeval_res: Quantity[_LENGTH],
        kind: str = "mean",
    ) -> np.ndarray:
        """Interpolate one oblique rectilinear slice from two bracketing coeval boxes."""
        nx, ny, _ = coeval_a.shape
        coords = self._slice_coords(self._s_pix_for_distance(lcd, coeval_res), nx, ny)
        denom = float(np.abs(dc_a - dc_b).value)
        if denom <= 0.0:
            return self.coeval_subselect(lcd, coeval_a, coeval_res)
        w_a = float(np.abs(dc_b - lcd).value / denom)
        w_b = float(np.abs(dc_a - lcd).value / denom)
        return self._sample_cpu(
            np.asarray(coeval_a, dtype=np.float32),
            np.asarray(coeval_b, dtype=np.float32),
            coords,
            w_a,
            w_b,
            dc_a,
            dc_b,
            lcd,
            kind,
            (nx, ny),
        )

    def interpolate_los_velocity_slice(
        self,
        lcd: Quantity[pixel],
        c_a: Coeval,
        c_b: Coeval,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
    ) -> np.ndarray:
        """Project 3D velocities onto the oblique LOS unit vector at distance ``lcd``."""
        nx, ny, _ = c_a.velocity_z.shape
        coords = self._slice_coords(
            self._s_pix_for_distance(lcd, c_a.simulation_options.cell_size), nx, ny
        )
        denom = float(np.abs(dc_a - dc_b).value)
        w_a = float(np.abs(dc_b - lcd).value / denom)
        w_b = float(np.abs(dc_a - lcd).value / denom)
        return self._sample_los_velocity_cpu(c_a, c_b, coords, w_a, w_b, (nx, ny))

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Lightconer:
        """Validate 21cmFAST options including 3D velocity retention for oblique LOS."""
        lightconer = super().validate_options(
            inputs=inputs,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            classy_output=classy_output,
        )
        if (
            include_dvdr_in_tau21 or apply_rsds
        ) and not inputs.matter_options.KEEP_3D_VELOCITIES:
            raise ValueError(
                "To account for RSDs or velocity corrections in an ObliqueRectilinearLightconer, "
                "set matter_options.KEEP_3D_VELOCITIES=True (or use quick_lightcone(..., oblique=True))."
            )
        return lightconer

    def make_lightcone_slices(self, c1: Coeval, c2: Coeval):
        """Generate oblique periodic rectilinear slices on GPU (or CPU fallback)."""
        if c1.simulation_options != c2.simulation_options:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        cosmo = c1.cosmo_params.cosmo
        coeval_res = c1.simulation_options.cell_size
        pixeleq = pixel_scale(coeval_res / pixel)
        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)
        denom = float(np.abs(dc1 - dc2).value)
        if denom <= 0.0:
            raise ValueError("c1 and c2 must be at different redshifts")

        pixlcdist = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = np.nonzero((pixlcdist >= dcmin * (1 - 1e-6)) & (pixlcdist < dcmax))[0]
        if len(lcidx) == 0:
            yield None, lcidx, None
            return

        lc_distances = pixlcdist[lcidx]
        min_lcpix = pixlcdist.min()
        w_a = np.abs(dc2 - lc_distances).value / denom
        w_b = np.abs(dc1 - lc_distances).value / denom
        use_gpu = bool(
            config["USE_CUDA"]
            and lib.p21_cuda_is_compiled()
            and lib.p21_cuda_is_available()
        )

        from .drivers.coeval import _get_coeval_array_no_sync

        def _get_f32_pair(attr_name: str) -> tuple[np.ndarray, np.ndarray]:
            arr_a = _get_coeval_array_no_sync(c1, attr_name)
            arr_b = _get_coeval_array_no_sync(c2, attr_name)
            if arr_a is None:
                arr_a = np.asarray(getattr(c1, attr_name))
            if arr_b is None:
                arr_b = np.asarray(getattr(c2, attr_name))
            a32 = (
                arr_a
                if (arr_a.dtype == np.float32 and arr_a.flags.c_contiguous)
                else np.ascontiguousarray(arr_a, dtype=np.float32)
            )
            b32 = (
                arr_b
                if (arr_b.dtype == np.float32 and arr_b.flags.c_contiguous)
                else np.ascontiguousarray(arr_b, dtype=np.float32)
            )
            return a32, b32

        q_arrays = {q: _get_f32_pair(q) for q in self.quantities if q != "los_velocity"}
        has_vel = "los_velocity" in self.quantities
        if has_vel:
            vx_a, vx_b = _get_f32_pair("velocity_x")
            vy_a, vy_b = _get_f32_pair("velocity_y")
            vz_a, vz_b = _get_f32_pair("velocity_z")

        first_q = next(iter(q_arrays.values()))[0]
        nx, ny, nz = first_q.shape
        n_coords = int(nx * ny)
        _, unit_xyz = self._transverse_offsets(nx, ny)
        out_buf = np.empty(n_coords, dtype=np.float32)

        for s, (idx, lcd) in enumerate(zip(lcidx, lc_distances, strict=True)):
            s_pix = float((lcd - min_lcpix).to_value(pixel))
            coords = self._slice_coords(s_pix, nx, ny)
            wa_f = float(w_a[s])
            wb_f = float(w_b[s])

            for q in self.quantities:
                if q == "los_velocity":
                    continue
                kind = self.interp_kinds.get(q, "mean")
                a32, b32 = q_arrays[q]
                slc = None
                if use_gpu and kind in ("mean", "mean_max"):
                    rc = lib.p21_cuda_angular_slice(
                        ffi.cast("const float *", ffi.from_buffer(a32)),
                        ffi.cast("const float *", ffi.from_buffer(b32)),
                        int(nx),
                        int(ny),
                        int(nz),
                        n_coords,
                        ffi.cast("const double *", ffi.from_buffer(coords)),
                        wa_f,
                        wb_f,
                        int(self.interpolation_order),
                        1 if kind == "mean_max" else 0,
                        ffi.cast("float *", ffi.from_buffer(out_buf)),
                    )
                    if rc == 0:
                        slc = out_buf.reshape((nx, ny)).copy()
                if slc is None:
                    slc = self._sample_cpu(
                        a32, b32, coords, wa_f, wb_f, dc1, dc2, lcd, kind, (nx, ny)
                    )
                yield (q, idx, slc)

                if has_vel and q == self.quantities[0]:
                    vslc = None
                    if use_gpu:
                        rc_v = lib.p21_cuda_angular_los_velocity(
                            ffi.cast("const float *", ffi.from_buffer(vx_a)),
                            ffi.cast("const float *", ffi.from_buffer(vy_a)),
                            ffi.cast("const float *", ffi.from_buffer(vz_a)),
                            ffi.cast("const float *", ffi.from_buffer(vx_b)),
                            ffi.cast("const float *", ffi.from_buffer(vy_b)),
                            ffi.cast("const float *", ffi.from_buffer(vz_b)),
                            int(nx),
                            int(ny),
                            int(nz),
                            n_coords,
                            ffi.cast("const double *", ffi.from_buffer(coords)),
                            ffi.cast("const double *", ffi.from_buffer(unit_xyz)),
                            wa_f,
                            wb_f,
                            int(self.interpolation_order),
                            ffi.cast("float *", ffi.from_buffer(out_buf)),
                        )
                        if rc_v == 0:
                            vslc = out_buf.reshape((nx, ny)).copy()
                    if vslc is None:
                        vslc = self._sample_los_velocity_cpu(
                            c1, c2, coords, wa_f, wb_f, (nx, ny)
                        )
                    yield ("los_velocity", idx, vslc)


def _dimensionless(x: np.ndarray | Quantity) -> np.ndarray:
    """Return a dimensionless array or Quantity as a plain array."""
    return Quantity(x, units.dimensionless_unscaled).value


#: Roughly how many cells of a coeval box can be interpolated in redshift in the time
#: it takes to sample the box in one direction (trilinear ``map_coordinates``). Measured
#: on one core for 128^3 and 256^3 boxes and 10^3-10^6 directions (HEALPix and random):
#: the two ways of making an angular slice cost the same at about 9-20 cells per
#: direction. Used by ``AngularLightconer`` to choose between them.
_CELLS_PER_SAMPLE = 16


def _rotation_eq(x, y):
    """Compare two rotations."""
    if x is None and y is None:
        return True

    return np.allclose(x.as_matrix(), y.as_matrix())


@attrs.define(kw_only=True, slots=False)
class AngularLightconer(Lightconer):
    """Angular lightcone slices constructed from rectlinear coevals."""

    latitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    longitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    interpolation_order: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.in_([0, 1, 3, 5])
    )
    origin: Quantity[pixel, (3,), float] = attrs.field(
        eq=attrs.cmp_using(eq=np.allclose)
    )
    rotation: Rotation = attrs.field(
        default=None,
        eq=attrs.cmp_using(eq=_rotation_eq),
        validator=attrs.validators.optional(attrs.validators.instance_of(Rotation)),
    )
    # Whether slices sample (tile) the coevals before interpolating in redshift (True)
    # or interpolate the boxes first (False). None chooses for each slice; see
    # samples_first().
    tile_first: bool | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(bool)),
    )

    def __attrs_post_init__(self) -> None:
        """Post-init."""
        self._cache = {
            "lcd": None,
            "interpolator": None,
        }

    @longitude.validator
    def _longitude_validator(self, attribute, value):
        if value.ndim != 1:
            raise ValueError("longitude must be 1-dimensional")
        if np.any(value < 0) or np.any(value > 2 * np.pi):
            raise ValueError("longitude must be in the range [0, 2pi]")
        if value.shape != self.latitude.shape:
            raise ValueError("longitude and latitude must have the same shape")

    @origin.default
    def _origin_default(self):
        return np.zeros(3) * pixel

    @classmethod
    def like_rectilinear(
        cls,
        simulation_options: SimulationOptions,
        match_at_z: float,
        cosmo: FLRW = Planck18,
        **kw,
    ):
        """Create an angular lightconer with the same pixel size as a rectilinear one.

        This is useful for comparing the two lightconer types.

        Parameters
        ----------
        simulation_options
            The user parameters.
        match_at_z
            The redshift at which the angular lightconer should match the rectilinear
            one.
        cosmo
            The cosmology to use.

        Other Parameters
        ----------------
        All other parameters passed through to the constructor.

        Returns
        -------
        AngularLightconer
            The angular lightconer.
        """
        box_size_radians = (
            simulation_options.BOX_LEN / cosmo.comoving_distance(match_at_z).value
        )

        lon = np.linspace(0, box_size_radians, simulation_options.HII_DIM)
        # This makes the X-values increasing from 0.
        lat = np.linspace(0, box_size_radians, simulation_options.HII_DIM)[::-1]

        LON, LAT = np.meshgrid(lon, lat)
        LON = LON.flatten()
        LAT = LAT.flatten()

        origin_offset = -cosmo.comoving_distance(match_at_z).to(
            pixel, pixel_scale(simulation_options.cell_size / pixel)
        )
        origin = np.array([0, 0, origin_offset.value]) * origin_offset.unit
        rot = Rotation.from_euler("Y", -np.pi / 2)

        return cls.between_redshifts(
            min_redshift=match_at_z,
            resolution=simulation_options.cell_size,
            latitude=LAT,
            longitude=LON,
            origin=origin,
            rotation=rot,
            **kw,
        )

    def interpolate_slice(
        self,
        lcd: Quantity[pixel],
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
        coeval_res: Quantity[_LENGTH],
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one lightcone slice of a field from the two coevals that bracket it.

        If the slice samples the coevals first (see :attr:`tile_first` and
        :meth:`samples_first`), each coeval is sampled at the pixels of the slice, and
        the two sets of samples are interpolated to the slice's distance. Since both
        steps are linear in the field, this gives the same result (to rounding) as
        interpolating the whole box in distance and then sampling it, which is done
        otherwise (as in :meth:`Lightconer.interpolate_slice`).

        The "mean_max" interpolation (used for ``z_reion``) is a rule applied cell by
        cell, so for ``interpolation_order=1`` it is applied at the eight cells
        around each sample point before they are weighted, which again gives the same
        result as sampling the interpolated box.
        """
        interpolator = self._get_interpolator(lcd)
        denom = float(np.abs(dc_a - dc_b).value)
        if (
            config["USE_CUDA"]
            and lib.p21_cuda_is_compiled()
            and lib.p21_cuda_is_available()
            and self.interpolation_order in (0, 1)
            and kind in ("mean", "mean_max")
            and coeval_a.ndim == 3
            and coeval_a.shape == coeval_b.shape
            and denom > 0.0
        ):
            coords = np.ascontiguousarray(
                Quantity(interpolator.keywords["coordinates"], pixel).value,
                dtype=np.float64,
            )
            n_coords = int(coords.shape[1])
            nx, ny, nz = coeval_a.shape
            a32 = np.ascontiguousarray(coeval_a, dtype=np.float32)
            b32 = np.ascontiguousarray(coeval_b, dtype=np.float32)
            out = np.empty(n_coords, dtype=np.float32)
            w_a = float(np.abs(dc_b - lcd).value / denom)
            w_b = float(np.abs(dc_a - lcd).value / denom)
            rc = lib.p21_cuda_angular_slice(
                ffi.cast("const float *", ffi.from_buffer(a32)),
                ffi.cast("const float *", ffi.from_buffer(b32)),
                int(nx),
                int(ny),
                int(nz),
                n_coords,
                ffi.cast("const double *", ffi.from_buffer(coords)),
                w_a,
                w_b,
                int(self.interpolation_order),
                1 if kind == "mean_max" else 0,
                ffi.cast("float *", ffi.from_buffer(out)),
            )
            if rc == 0:
                weights = interpolator.keywords.get("weights")
                if weights is not None:
                    w = np.asarray(weights, dtype=np.float64)
                    out = (w[:, None] * out.reshape(w.size, -1)).sum(axis=0).astype(
                        np.float32
                    )
                return out

        if not self.samples_first(coeval_a.size, interpolator):
            return super().interpolate_slice(
                lcd, coeval_a, coeval_b, dc_a, dc_b, coeval_res, kind=kind
            )

        if kind == "mean_max" and self.interpolation_order == 1:
            return self._trilinear_mean_max(
                interpolator, lcd, coeval_a, coeval_b, dc_a, dc_b
            )

        # Order 0 picks one cell per pixel, so any cell-by-cell rule commutes with it.
        return _dimensionless(
            self.redshift_interpolation(
                lcd,
                interpolator(coeval_a),
                interpolator(coeval_b),
                dc_a,
                dc_b,
                kind=kind,
            )
        )

    def interpolate_los_velocity_slice(
        self,
        lcd: Quantity[pixel],
        c_a: Coeval,
        c_b: Coeval,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
    ) -> np.ndarray:
        """Make one slice of the line-of-sight velocity from two bracketing coevals.

        If the slice samples the coevals first (see :meth:`interpolate_slice`), the
        line-of-sight component of the 3D velocity of each coeval is taken at the
        pixels of the slice, and the two are interpolated to the slice's distance.
        Otherwise the three velocity boxes are interpolated in distance first.
        """
        kind = self.interp_kinds.get("velocity", "mean")
        interpolator = self._get_interpolator(lcd)
        denom = float(np.abs(dc_a - dc_b).value)
        if (
            config["USE_CUDA"]
            and lib.p21_cuda_is_compiled()
            and lib.p21_cuda_is_available()
            and self.interpolation_order in (0, 1)
            and kind == "mean"
            and denom > 0.0
        ):
            coords = np.ascontiguousarray(
                Quantity(interpolator.keywords["coordinates"], pixel).value,
                dtype=np.float64,
            )
            rel_coords = coords
            if getattr(interpolator, "origin", None) is not None:
                orig = np.asarray(
                    Quantity(interpolator.origin, pixel).value, dtype=np.float64
                )
                rel_coords = coords - orig[:, None]
            norms = np.sqrt(np.sum(np.square(rel_coords), axis=0))
            if np.all(norms > 0.0):
                unit_xyz = np.ascontiguousarray(-rel_coords / norms, dtype=np.float64)
                n_coords = int(coords.shape[1])
                vxa = np.ascontiguousarray(c_a.velocity_x, dtype=np.float32)
                vya = np.ascontiguousarray(c_a.velocity_y, dtype=np.float32)
                vza = np.ascontiguousarray(c_a.velocity_z, dtype=np.float32)
                vxb = np.ascontiguousarray(c_b.velocity_x, dtype=np.float32)
                vyb = np.ascontiguousarray(c_b.velocity_y, dtype=np.float32)
                vzb = np.ascontiguousarray(c_b.velocity_z, dtype=np.float32)
                nx, ny, nz = vxa.shape
                out = np.empty(n_coords, dtype=np.float32)
                w_a = float(np.abs(dc_b - lcd).value / denom)
                w_b = float(np.abs(dc_a - lcd).value / denom)
                rc = lib.p21_cuda_angular_los_velocity(
                    ffi.cast("const float *", ffi.from_buffer(vxa)),
                    ffi.cast("const float *", ffi.from_buffer(vya)),
                    ffi.cast("const float *", ffi.from_buffer(vza)),
                    ffi.cast("const float *", ffi.from_buffer(vxb)),
                    ffi.cast("const float *", ffi.from_buffer(vyb)),
                    ffi.cast("const float *", ffi.from_buffer(vzb)),
                    int(nx),
                    int(ny),
                    int(nz),
                    n_coords,
                    ffi.cast("const double *", ffi.from_buffer(coords)),
                    ffi.cast("const double *", ffi.from_buffer(unit_xyz)),
                    w_a,
                    w_b,
                    int(self.interpolation_order),
                    ffi.cast("float *", ffi.from_buffer(out)),
                )
                if rc == 0:
                    weights = interpolator.keywords.get("weights")
                    if weights is not None:
                        w = np.asarray(weights, dtype=np.float64)
                        out = (
                            (w[:, None] * out.reshape(w.size, -1))
                            .sum(axis=0)
                            .astype(np.float32)
                        )
                    return out

        if kind == "mean" and self.samples_first(c_a.velocity_x.size, interpolator):
            # cosmotile needs Quantities here, since the pixel coordinates are.
            los_a, los_b = make_lightcone_slice_vector_field(
                [
                    [
                        getattr(c, f"velocity_{ax}") << units.dimensionless_unscaled
                        for ax in "xyz"
                    ]
                    for c in (c_a, c_b)
                ],
                interpolator,
            )
            return _dimensionless(
                self.redshift_interpolation(lcd, los_a, los_b, dc_a, dc_b, kind=kind)
            )

        coeval_res = c_a.simulation_options.cell_size
        boxes = [
            self.redshift_interpolation(
                lcd,
                self.coeval_subselect(lcd, getattr(c_a, f"velocity_{ax}"), coeval_res),
                self.coeval_subselect(lcd, getattr(c_b, f"velocity_{ax}"), coeval_res),
                dc_a,
                dc_b,
                kind=kind,
            )
            for ax in "xyz"
        ]
        return self.construct_los_velocity_lightcone(lcd, boxes)

    def construct_lightcone(
        self,
        lcd: Quantity[pixel],
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the lightcone slices from bracketing coevals."""
        return self._get_interpolator(lcd)(box)

    def construct_los_velocity_lightcone(
        self,
        lcd: Quantity[pixel],
        velocities: Sequence[np.ndarray],
    ):
        """Construct the LoS velocity lightcone from 3D velocities."""
        return next(
            make_lightcone_slice_vector_field([velocities], self._get_interpolator(lcd))
        )

    def _trilinear_mean_max(
        self,
        interpolator: partial,
        lcd: Quantity[pixel],
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
    ) -> np.ndarray:
        """Sample the "mean_max" interpolation of two coevals, trilinearly.

        This is what sampling ``redshift_interpolation(..., kind="mean_max")`` of the
        whole boxes with ``interpolator`` gives (``map_coordinates`` with ``order=1``
        and ``mode="grid-wrap"``), but the rule is only evaluated at the eight cells
        around each sample point.
        """
        coords = Quantity(interpolator.keywords["coordinates"], pixel).value
        shape = np.array(coeval_a.shape)[:, None]
        floor = np.floor(coords)
        frac = coords - floor
        lower = floor.astype(np.int64) % shape
        upper = (lower + 1) % shape

        out = np.zeros(coords.shape[1])
        for corner in np.ndindex(2, 2, 2):
            index = tuple(upper[ax] if c else lower[ax] for ax, c in enumerate(corner))
            weight = np.prod(
                [frac[ax] if c else 1 - frac[ax] for ax, c in enumerate(corner)],
                axis=0,
            )
            out += weight * _dimensionless(
                self.redshift_interpolation(
                    lcd, coeval_a[index], coeval_b[index], dc_a, dc_b, kind="mean_max"
                )
            )

        # With sub-sampled pixels (cosmotile >= 0.3), average the sub-samples.
        weights = interpolator.keywords.get("weights")
        if weights is not None:
            out = (weights[:, None] * out.reshape(weights.size, -1)).sum(axis=0)
        return out

    def samples_first(self, n_cells: int, interpolator: partial) -> bool:
        """Whether a slice samples the coevals first, or interpolates the boxes first.

        A slice between two coevals can be made in two ways that give the same result
        (to rounding): interpolate the whole boxes in redshift and then sample (tile)
        the interpolated box at the directions of the slice, or sample each box first
        and then interpolate the samples in redshift. The first costs about one
        operation per cell plus one sample per direction, the second two samples per
        direction. Unless ``tile_first`` forces one way, a slice samples first when
        the box has more than ``_CELLS_PER_SAMPLE`` (16) cells per direction. For
        example, a 1500^3 box with a ring of 2^15 directions samples first, while a
        228^3 box with a HEALPix map of Nside 1024 (1.3e7 directions) interpolates
        the box first. Only ``interpolation_order`` 0 and 1 can sample first; higher
        orders pre-filter the whole box.

        Parameters
        ----------
        n_cells
            The number of cells of the coeval box.
        interpolator
            The interpolator onto the slice (its coordinates give the number of
            directions sampled, sub-samples included).

        Returns
        -------
        bool
            ``tile_first`` if it is set, otherwise whether sampling first is cheaper:
            whether the box has more than ``_CELLS_PER_SAMPLE`` cells per direction.
            Always False for ``interpolation_order > 1``.
        """
        if self.interpolation_order > 1:
            return False
        if self.tile_first is not None:
            return self.tile_first
        n_samples = np.shape(interpolator.keywords["coordinates"])[-1]
        return n_samples * _CELLS_PER_SAMPLE < n_cells

    def _get_interpolator(self, lcd: Quantity[pixel]) -> partial:
        """Return the (cached) interpolator onto the slice at distance ``lcd``."""
        if self._cache["lcd"] == lcd:
            return self._cache["interpolator"]
        return self._refresh_cache(lcd)

    def _refresh_cache(self, lcd):
        result = make_lightcone_slice_interpolator(
            latitude=self.latitude,
            longitude=self.longitude,
            distance_to_shell=lcd,
            interpolation_order=self.interpolation_order,
            origin=self.origin,
            rotation=self.rotation,
        )
        self._cache["lcd"] = lcd
        self._cache["interpolator"] = result
        return result

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int]:
        """Get the shape of the lightcone slices."""
        return (len(self.longitude), len(self.lc_redshifts))

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Lightconer:
        """Validate 21cmFAST options."""
        lightconer = super().validate_options(
            inputs=inputs,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            classy_output=classy_output,
        )

        if (
            include_dvdr_in_tau21 or apply_rsds
        ) and not inputs.matter_options.KEEP_3D_VELOCITIES:
            raise ValueError(
                "To account for RSDs or velocity corrections in an angular lightcone, you need to set "
                "matter_options.KEEP_3D_VELOCITIES=True"
            )

        return lightconer
