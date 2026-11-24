"""CUDA interpolation and optical-depth gradient for v4 rectilinear lightcones."""

import cupy as cp
import numpy as np
from astropy import units

from device_slices import interpolate_slices
from py21cmfast.lightconers import RectilinearLightconer


_native_slices = RectilinearLightconer.make_lightcone_slices
_RSD_KERNELS = {}


def _rsd_kernel(precision):
    kernel = _RSD_KERNELS.get(precision)
    if kernel is None:
        kernel = cp.RawKernel(
            r"""
    extern "C" __global__ void shift(const TYPE *field,const double *velocity,TYPE *out,
                                      int nr,int nz,int sub,int periodic) {
        int ray=blockIdx.x*blockDim.x+threadIdx.x;
        if(ray>=nr) return;
        field+=ray*nz; velocity+=ray*nz; int count=nz*sub; out+=ray*count;
        for(int i=0;i<count;i++){
            double x=(i+.5)/sub-.5;
            int left=(int)floor(x);
            if(!periodic) left=max(0,min(nz-2,left));
            double t=x-left;
            int lo=(left+nz)%nz, hi=(left+1+nz)%nz;
            double target=i+(1.-t)*(sub*velocity[lo])+t*(sub*velocity[hi]);
            long a=(long)floor(target), b=a+1;
            double w=b-target;
            TYPE val=field[i/sub]/sub;
            if(periodic){a=(a%count+count)%count; b=(b%count+count)%count;}
            if(a>=0 && a<count) out[a]+=w*val;
            if(b>=0 && b<count) out[b]+=(1.-w)*val;
        }
    }""".replace("TYPE", precision),
            "shift",
            options=("--fmad=false",),
        )
        _RSD_KERNELS[precision] = kernel
    return kernel


def make_lightcone_slices(self, c1, c2, *, registry=None):
    """Batch a v4 node pair through the v3 LOS gather and interpolation kernel."""
    quantities = self.quantities
    if (
        type(self) is not RectilinearLightconer
        or c1.simulation_options != c2.simulation_options
        or c1.cosmo_params != c2.cosmo_params
    ):
        yield from _native_slices(self, c1, c2)
        return

    cell = c1.simulation_options.cell_size
    pixels = self.get_lc_distances_in_pixels(cell).to_value(units.pixel)
    equiv = units.pixel_scale(cell / units.pixel)
    dc1, dc2 = (
        c.cosmo_params.cosmo.comoving_distance(c.redshift).to_value(
            units.pixel, equivalencies=equiv
        )
        for c in (c1, c2)
    )
    # v4 includes a 1e-6 allowance at the near edge. The far edge is open.
    indices = np.flatnonzero(
        (pixels >= min(dc1, dc2) * (1 - 1e-6)) & (pixels < max(dc1, dc2))
    )
    if not indices.size:
        yield None, indices, None
        return

    fields = []
    arrays = {}
    for name in quantities:
        if name == "los_velocity":
            continue
        fields.append((name, name))
        # The native driver emits LOS velocity just after the first requested
        # scalar. Both output names read velocity_z with the same interpolation.
        if name == quantities[0] and "los_velocity" in quantities:
            fields.append(("los_velocity", "velocity_z"))

    for _, attr in fields:
        if attr in arrays:
            continue
        a, b = getattr(c1, attr), getattr(c2, attr)
        kind = self.interp_kinds.get(attr, "mean")
        if (
            a.ndim != 3
            or a.shape != b.shape
            or a.dtype != b.dtype
            or a.dtype not in (np.float32, np.float64)
            or kind not in ("mean", "mean_max")
        ):
            yield from _native_slices(self, c1, c2)
            return
        arrays[attr] = a, b, kind

    distances = pixels[indices]
    offsets = (pixels.max() - distances + 1).astype(np.int64)
    wa, wb = abs(dc2 - distances), abs(dc1 - distances)
    denom = abs(dc1 - dc2)
    scalar = units.Quantity(wa[0], units.pixel)
    indices_by_depth = {}
    weights_by_dtype = {}
    slabs = {}
    read = cp.asarray if registry is None else registry.lookup_host
    for attr, (a, b, kind) in arrays.items():
        depth = a.shape[-1]
        if depth not in indices_by_depth:
            indices_by_depth[depth] = cp.asarray(
                (self.index_offset - offsets) % depth, dtype=cp.int32
            )
        dtype = (scalar * np.empty(0, dtype=a.dtype)).dtype
        if dtype not in weights_by_dtype:
            weights_by_dtype[dtype] = (
                cp.asarray(wa, dtype=dtype),
                cp.asarray(wb, dtype=dtype),
            )
        u, v = weights_by_dtype[dtype]
        slabs[attr] = interpolate_slices(
            read(a),
            read(b),
            indices_by_depth[depth],
            u,
            v,
            denom,
            kind == "mean_max",
        ).get()
    last_use = {attr: i for i, (_, attr) in enumerate(fields)}
    for pos, idx in enumerate(indices):
        for i, (name, attr) in enumerate(fields):
            # Separately yielded fields remain independently writable.
            yield (
                name,
                idx,
                units.Quantity(
                    slabs[attr][..., pos], units.one, copy=i != last_use[attr]
                ),
            )


class LightconeVelocityWorkspace:
    """Share native H(z) and the GPU displacement across finalization fields."""

    def __init__(self):
        self.sources = None

    def prepare(self, los_velocity, redshifts, inputs):
        sources = (los_velocity, redshifts, inputs)
        if self.sources is None or any(
            a is not b for a, b in zip(self.sources, sources)
        ):
            hubble = inputs.cosmo_params.cosmo.H(redshifts)
            self.hubble = cp.asarray(hubble.value)
            self.rate_scale = (1 / units.s / hubble.unit).to_value(units.one)
            self.hubble_unit = hubble.unit
            self._velocity = None
            self._displacement = None
            self.sources = sources
        return self

    @property
    def velocity(self):
        if self._velocity is None:
            self._velocity = cp.asarray(self.sources[0])
        return self._velocity

    @property
    def displacement(self):
        if self._displacement is None:
            inputs = self.sources[2]
            scale = (
                1
                * units.Mpc
                / units.s
                / self.hubble_unit
                / inputs.simulation_options.cell_size
            ).to_value(units.one)
            self._displacement = cp.ascontiguousarray(
                (self.velocity / self.hubble) * scale, dtype="d"
            )
            # The driver applies the optical correction before the RSD fields.
            self._velocity = None
        return self._displacement


def optical_gradient(
    brightness_temp,
    los_velocity,
    redshifts,
    inputs,
    periodic,
    tau_21=None,
    *,
    workspace=None,
):
    state = (workspace or LightconeVelocityWorkspace()).prepare(
        los_velocity, redshifts, inputs
    )
    vel = state.velocity
    dx = inputs.simulation_options.BOX_LEN / inputs.simulation_options.HII_DIM
    if periodic:
        k = cp.fft.rfftfreq(vel.shape[-1], dx) * (2 * cp.pi)
        grad = cp.fft.irfftn(1j * k * cp.fft.rfftn(vel), s=vel.shape)
    else:
        grad = cp.empty_like(vel)
        grad[..., 1:-1] = (vel[..., 2:] - vel[..., :-2]) / (2 * dx)
        grad[..., 0] = (
            (-1.5 / dx) * vel[..., 0]
            + (2.0 / dx) * vel[..., 1]
            - (0.5 / dx) * vel[..., 2]
        )
        grad[..., -1] = (
            (0.5 / dx) * vel[..., -3]
            - (2.0 / dx) * vel[..., -2]
            + (1.5 / dx) * vel[..., -1]
        )
    ratio = (grad / state.hubble) * state.rate_scale
    tb = cp.asarray(brightness_temp)
    if not inputs.astro_options.USE_TS_FLUCT:
        limit = inputs.astro_params.MAX_DVDR
        return (tb / cp.abs(1 + cp.clip(ratio, -limit, limit))).get()
    tau = cp.asarray(tau_21, dtype="d")
    correction = (1 - cp.exp(-tau / cp.abs(1.0 + ratio))) / (1 - cp.exp(-tau))
    return (tb * cp.where(tau < 1e-10, 1.0, correction).astype("f")).get()


def apply_rsds(
    field,
    los_velocity,
    redshifts,
    inputs,
    n_rsd_subcells=4,
    periodic=False,
    *,
    workspace=None,
):
    if field.shape != los_velocity.shape or field.ndim not in (2, 3):
        raise ValueError("Velocity and field must have the same ray grid")
    if not isinstance(n_rsd_subcells, int) or n_rsd_subcells < 1:
        raise ValueError("Expected a positive integer subcell count")
    state = (workspace or LightconeVelocityWorkspace()).prepare(
        los_velocity, redshifts, inputs
    )
    source = cp.ascontiguousarray(cp.asarray(field).reshape(-1, field.shape[-1]))
    rays, depth = source.shape
    if depth < 2 or source.dtype not in (cp.float32, cp.float64):
        raise ValueError(
            "RSD remapping requires floating fields and at least two slices"
        )
    displ = state.displacement.reshape(rays, depth)
    fine = cp.zeros((rays, depth * n_rsd_subcells), dtype=source.dtype)
    precision = "float" if source.dtype == cp.float32 else "double"
    kernel = _rsd_kernel(precision)
    kernel(
        ((rays + 127) // 128,),
        (128,),
        (source, displ, fine, *map(np.int32, (rays, depth, n_rsd_subcells, periodic))),
    )
    return (
        fine.reshape(rays, depth, n_rsd_subcells)
        .sum(axis=-1)
        .reshape(field.shape)
        .get()
    )


class EmissivityHistory:
    """Keep the active pair of source snapshots on device for one radiation call."""

    def __init__(
        self,
        counts,
        *,
        share_scattering_spectra=False,
        keep_radiation_grids=True,
        registry=None,
    ):
        self.counts = counts
        self.share_scattering_spectra = share_scattering_spectra
        self.keep_radiation_grids = keep_radiation_grids
        self.registry = registry
        self.pair = ()
        self.fields = {}
        self.shell_fields = {}
        self.shell_spectra = {}
        self.filtered_grids = {}
        self.accumulated_grids = {}
        self.bytes = 0
        self.limit = cp.cuda.Device().mem_info[0] // 8

    def select(self, earlier, later):
        self.shell_fields.clear()
        self.shell_spectra.clear()
        self.filtered_grids.clear()
        pair = (earlier, later)
        if any(all(box is not current for current in pair) for box in self.pair):
            self.fields = {
                key: value
                for key, value in self.fields.items()
                if any(value[0] is box for box in pair)
            }
            self.bytes = sum(value[2].nbytes for value in self.fields.values())
        self.pair = pair

    def field(self, box, name, host, dtype):
        key = id(box), name, dtype
        cached = self.fields.get(key)
        if cached is not None:
            if cached[1] is host:
                self.counts["reused_uploads"] += 1
                self.counts["reused_bytes"] += cached[2].nbytes
                return cached[2]
            self.bytes -= cached[2].nbytes
            del self.fields[key]
        device = None
        if (
            self.registry is not None
            and isinstance(host, np.ndarray)
            and host.flags.c_contiguous
        ):
            found = self.registry.lookup_ptr(host.ctypes.data, host.size, host.dtype)
            if found is not None:
                device = found.reshape(host.shape)
                if device.dtype != cp.dtype(dtype):
                    device = device.astype(dtype)
                self.counts["reused_uploads"] += 1
                self.counts["reused_bytes"] += device.nbytes
        if device is None:
            device = cp.asarray(host, dtype=dtype)
            self.counts["uploads"] += 1
            self.counts["uploaded_bytes"] += device.nbytes
        if self.retained_bytes + device.nbytes <= self.limit:
            self.fields[key] = box, host, device
            self.bytes += device.nbytes
        return device

    def store_shell_field(self, host, device):
        # The native ABI still receives the real host array. Holding its owner
        # until the next shell prevents the address from being recycled.
        self.shell_fields[host.ctypes.data] = host, device

    def shell_field(self, address, size):
        field = self.shell_fields.get(address)
        if field is None or field[1].size != size or field[1].dtype != cp.float32:
            return None
        device = field[1]
        self.counts["filter_input_reuses"] += 1
        self.counts["filter_input_reused_bytes"] += device.nbytes
        return device

    def retain_shell_spectrum(self, address, spectrum):
        """Give Lyα a writable copy while LW waits for the same forward FFT."""
        if self.retained_bytes + spectrum.nbytes <= self.limit:
            writable = spectrum.copy()
            self.shell_spectra[address] = spectrum
            return writable
        return spectrum

    def take_shell_spectrum(self, address):
        spectrum = self.shell_spectra.pop(address, None)
        if spectrum is not None:
            self.counts["filter_forward_fft_reuses"] += 1
        return spectrum

    @property
    def retained_bytes(self):
        return self.bytes + sum(
            grid.nbytes
            for group in (
                self.shell_spectra,
                self.filtered_grids,
                self.accumulated_grids,
            )
            for grid in group.values()
        )

    def retain_radiation_grid(self, grids, address, device):
        if not self.keep_radiation_grids:
            return False
        if address not in grids:
            if self.retained_bytes + device.nbytes > self.limit:
                return False
            grids[address] = device
        return True


def emissivity_history(
    emissivity_fields_list, interp_fields, redshift, need_c, *, workspace=None
):
    from py21cmfast.drivers.single_field import check_output_consistency
    from py21cmfast.wrapper.outputs import EmissivityFields

    epochs = np.asarray([box.redshift for box in emissivity_fields_list])
    hi = int(np.searchsorted(epochs, redshift))
    if not np.all(np.diff(epochs) > 0) or not 0 < hi < len(epochs):
        raise ValueError("Emissivity history does not bracket redshift")
    earlier, later = emissivity_fields_list[hi - 1 : hi + 1]
    if workspace is not None:
        workspace.select(earlier, later)
    if need_c and earlier.inputs.simulation_options.HII_DIM > 1:
        names = [field for field in interp_fields if field in earlier.arrays]
        if not all(
            box.ensure_arrays_computed(*names) for box in emissivity_fields_list
        ):
            raise ValueError("Missing emissivity field")
        check_output_consistency(
            {f"box-{i}": box for i, box in enumerate(emissivity_fields_list)}
        )
    out = EmissivityFields.new(redshift=redshift, inputs=earlier.inputs)
    if need_c:
        out._init_arrays()
    t = (redshift - earlier.redshift) / (later.redshift - earlier.redshift)
    astro = earlier.inputs.astro_options
    resident_shell = (
        workspace is not None
        and need_c
        and workspace.keep_radiation_grids
        and earlier.inputs.simulation_options.HII_DIM > 1
        and earlier.inputs.simulation_options.NON_CUBIC_FACTOR == 1
        and astro.USE_LYA_HEATING
        and astro.USE_X_RAY_HEATING
    )
    for name in interp_fields:
        a, b = earlier.get(name), later.get(name)
        # NumPy 2 promotes a NumPy float64 scalar but treats a Python
        # float as weak. Preserve the upstream result type on each product.
        left_type = np.result_type(1 - t, a)
        right_type = np.result_type(t, b)
        if workspace is None:
            left, right = (
                cp.asarray(a, dtype=left_type),
                cp.asarray(b, dtype=right_type),
            )
        else:
            left = workspace.field(earlier, name, a, left_type)
            right = workspace.field(later, name, b, right_type)
        data = ((1 - t) * left + t * right).astype(np.asarray(a).dtype)
        if resident_shell:
            # Expose the C-backed numpy buffer as computed without blocking on
            # a per-shell D2H copy; RadiationFields only consumes it via GPU shells.
            host_buf = out.arrays[name]._value
            out.set(name, host_buf)
            workspace.store_shell_field(host_buf, data)
            workspace.counts["shell_downloads_avoided"] += 1
            workspace.counts["shell_downloads_avoided_bytes"] += data.nbytes
        else:
            out.set(name, data.get())
            if workspace is not None and need_c:
                workspace.store_shell_field(out.get(name), data)
    return out


class DriverReductions:
    """Replace field reductions only inside the two upstream driver modules."""

    def __init__(self, registry=None):
        self.registry = registry

    def __getattr__(self, name):
        return getattr(np, name)

    def _to_device(self, x):
        return cp.asarray(x) if self.registry is None else self.registry.lookup_host(x)

    def mean(self, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            return cp.mean(self._to_device(x), *args, **kwargs).item()
        return np.mean(x, *args, **kwargs)

    def any(self, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            return cp.any(self._to_device(x), *args, **kwargs).item()
        return np.any(x, *args, **kwargs)
