"""GPU perturbation adapter for v4.3.2 InitialConditions."""

from contextlib import ExitStack
import ctypes
import cupy as cp
import numpy as np
from fft_plan_cuda import transform, executable, UnsupportedPlan
from py21cmfast import PerturbedField, c_21cmfast
from py21cmfast.drivers._global_initialization import c_state

_deposit = cp.RawKernel(
    r"""
extern "C" __global__ void deposit(const float *rho, const float *vel,
        const float *second, double *mass, int n, int m, double growth0,
        double first_scale, double second_scale, bool two_lpt) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n*n*n) return;
    int xyz[3] = {(int)(t/n/n), (int)(t/n%n), (int)(t%n)};
    double ratio = (double)m/n;
    long long source = 0;
    for (int a=0; a<3; ++a)
        source = source*m + ((int)(xyz[a]*ratio+0.5))%m;
    int base[3]; double fraction[3];
    for (int a=0; a<3; ++a) {
        long long j = source + (long long)a*m*m*m;
        // Match the v4.3.2 benchmark binary (-Ofast): it adds x before
        // the correction, but combines displacements first for y and z.
        double displacement = vel[j]*first_scale;
        double p = xyz[a] + displacement;
        if (two_lpt) {
            double correction = second[j]*second_scale;
            p = a == 0 ? p - correction : xyz[a] + (displacement - correction);
        }
        p *= ratio;
        base[a] = (int)floor(p);
        fraction[a] = p-floor(p);
    }
    for (int corner=0; corner<8; ++corner) {
        double w=1.; long long cell=0;
        for (int a=0; a<3; ++a) {
            int bit=(corner>>a)&1;
            cell=cell*m+(((base[a]+bit)%m)+m)%m;
            w *= bit ? fraction[a] : 1.-fraction[a];
        }
        atomicAdd(mass+cell, w*(1.+rho[t]*growth0));
    }
}""",
    "deposit",
    options=("--fmad=false",),
)


def modes(shape, box_len_xy, box_len_z):
    nx, ny, nz = (int(d) for d in shape)

    def axis(n, length):
        i = cp.arange(n)
        return (cp.where(i <= n // 2, i, i - n) * (2 * np.pi / length)).astype(
            cp.float32
        )

    return (
        axis(nx, box_len_xy)[:, None, None],
        axis(ny, box_len_xy)[None, :, None],
        axis(nz, box_len_z)[: nz // 2 + 1][None, None, :],
    )


def gaussian_weights(k2, radius):
    # The native filter squares the float radius before multiplying by k^2.
    kr2 = k2 * np.float32(radius * radius)
    return cp.exp((-0.5 * 0.643**2) * kr2.astype(cp.float64))


_NATIVE_LIB = None
_PERTURB_GEOM_CACHE = {}


def _get_native_lib():
    global _NATIVE_LIB
    if _NATIVE_LIB is None:
        _NATIVE_LIB = ctypes.CDLL(c_21cmfast.__file__)
    return _NATIVE_LIB


def _geometry_cache(
    n_shape,
    m_shape,
    low_shape,
    box_len_xy,
    box_len_z,
    high,
    smooth,
    smooth_radius,
    n_threads,
    wisdom,
    strict_fft,
):
    n_shape = tuple(int(d) for d in n_shape)
    m_shape = tuple(int(d) for d in m_shape)
    low_shape = tuple(int(d) for d in low_shape)
    key = (
        n_shape,
        m_shape,
        low_shape,
        float(box_len_xy),
        float(box_len_z),
        bool(high),
        bool(smooth),
        float(smooth_radius) if smooth else None,
        int(n_threads),
        bool(wisdom),
        bool(strict_fft),
        int(cp.cuda.runtime.getDevice()),
    )
    cached = _PERTURB_GEOM_CACHE.get(key)
    if cached is not None:
        return cached

    def supported(shape):
        nx, ny, nz = shape
        if wisdom or not (nx == ny == nz) or nx % 2 != 0:
            if strict_fft:
                raise UnsupportedPlan("External wisdom, odd or non-cubic grid enabled")
            return False
        try:
            for inverse_direction in (False, True):
                executable(nx, inverse_direction, n_threads)
        except UnsupportedPlan:
            if strict_fft:
                raise
            return False
        return True

    ordered = {shape for shape in {m_shape, low_shape} if supported(shape)}
    axes = modes(m_shape, box_len_xy, box_len_z)
    k2 = sum(k * k for k in axes)
    denom = cp.where(k2 == 0, 1, k2)
    axes_f64 = tuple(k.astype(cp.float64) for k in axes)
    window = None
    sample = None
    if high:
        radius = np.float32(0.620350491 * box_len_xy / low_shape[0])
        kr = (
            (cp.sqrt(k2.astype(cp.float64)) * radius)
            .astype(cp.float32)
            .astype(cp.float64)
        )
        safe = cp.where(kr == 0, 1.0, kr)
        window = cp.where(
            kr < 1e-4,
            1 - 0.1 * (kr * kr),
            (3 / ((safe * safe) * safe)) * (cp.sin(kr) - kr * cp.cos(kr)),
        )
        ratio = n_shape[0] / low_shape[0]
        idx_xy = cp.floor(cp.arange(low_shape[0]) * ratio + 0.5).astype(cp.int32)
        idx_z = cp.floor(cp.arange(low_shape[2]) * ratio + 0.5).astype(cp.int32)
        sample = cp.ix_(idx_xy, idx_xy, idx_z)
    smooth_weights = None
    if smooth:
        low_k2 = sum(k * k for k in modes(low_shape, box_len_xy, box_len_z))
        radius = np.float32(smooth_radius * box_len_xy / low_shape[0])
        smooth_weights = gaussian_weights(low_k2, radius)
    cached = (ordered, axes_f64, denom, window, sample, smooth_weights)
    _PERTURB_GEOM_CACHE[key] = cached
    return cached


def reference_density(lib, inputs, grid):
    """Apply upstream FFTW operations to GPU-deposited mass, using FFTW alignment."""
    sim, opt = inputs.simulation_options, inputs.matter_options
    if opt.USE_FFTW_WISDOM:
        raise ValueError("Reference FFT mode currently requires USE_FFTW_WISDOM=False")
    ptr = ctypes.c_void_p
    lib.fftwf_malloc.argtypes = [ctypes.c_size_t]
    lib.fftwf_malloc.restype = ptr
    lib.fftwf_free.argtypes = [ptr]
    with ExitStack() as cleanup, c_state(inputs, broadcast_inputs=True):

        def buffer(shape):
            nx, ny, nz = shape
            padded = (nx, ny, 2 * (nz // 2 + 1))
            address = lib.fftwf_malloc(4 * int(np.prod(padded)))
            if not address:
                raise MemoryError("FFTW allocation failed")
            cleanup.callback(lib.fftwf_free, address)
            array = np.ctypeslib.as_array(
                ctypes.cast(address, ctypes.POINTER(ctypes.c_float)), padded
            )
            return ptr(address), array

        n_shape = (sim.DIM, sim.DIM, int(sim.NON_CUBIC_FACTOR * sim.DIM))
        low_shape = (sim.HII_DIM, sim.HII_DIM, int(sim.NON_CUBIC_FACTOR * sim.HII_DIM))
        source, _ = buffer(grid.shape)
        coarse, box = buffer(low_shape)
        if opt.PERTURB_ON_HIGH_RES:
            fine, fine_box = buffer(n_shape)
            fine_box[:, :, : n_shape[2]] = grid.get()
            lib.assign_to_lowres_grid(fine, coarse, source)
        else:
            box[:, :, : low_shape[2]] = grid.get()
        if opt.PERTURB_ALGORITHM != "LINEAR":
            lib.normalise_delta_grid(coarse)
        lib.smooth_and_clip_density(coarse, source)
        return box[:, :, : low_shape[2]].copy()


def perturb_field_gpu(
    *,
    redshift,
    initial_conditions,
    inputs=None,
    reference_fft=False,
    strict_fft=False,
    resident=None,
):
    """Return a CPU-visible v4 PerturbedField, with GPU CIC, FFTs and velocities."""
    ic = initial_conditions
    inputs = ic.inputs if inputs is None else inputs
    sim, opt = inputs.simulation_options, inputs.matter_options
    # SimulationOptions stores BOX_LEN and NON_CUBIC_FACTOR as C floats before
    # computing displacements, mode spacing and filter radii in double precision.
    box_len_xy = float(np.float32(sim.BOX_LEN))
    box_len_z = box_len_xy * float(np.float32(sim.NON_CUBIC_FACTOR))
    n_shape = (sim.DIM, sim.DIM, int(sim.NON_CUBIC_FACTOR * sim.DIM))
    low_shape = (sim.HII_DIM, sim.HII_DIM, int(sim.NON_CUBIC_FACTOR * sim.HII_DIM))
    high = opt.PERTURB_ON_HIGH_RES
    m_shape = n_shape if high else low_shape
    prefix = "hires_" if high else "lowres_"
    lib = _get_native_lib()
    from growth import perturb_coefficients

    growth, growth0, rate = perturb_coefficients(inputs, redshift)
    read = ic.get if resident is None else resident.ic_fields.__getitem__
    if opt.PERTURB_ALGORITHM == "LINEAR":
        grid = (cp.asarray(read(prefix + "density"), dtype=cp.float64) * growth).astype(
            cp.float32
        )
    else:

        def vectors(suffix):
            components = [
                cp.asarray(read(prefix + "v" + axis + suffix)) for axis in "xyz"
            ]
            if resident is not None:
                cache = getattr(resident, "ic_vel_stacked", None)
                if cache is None:
                    cache = resident.ic_vel_stacked = {}
                key = (prefix, suffix, tuple(id(c) for c in components))
                stacked = cache.get(key)
                if stacked is None:
                    stacked = cp.stack(components)
                    cache[key] = stacked
                return stacked
            return cp.stack(components)

        two = opt.PERTURB_ALGORITHM == "2LPT"
        vel = vectors("")
        second = vectors("_2LPT") if two else vel
        mass = cp.zeros(m_shape, dtype=cp.float64)
        from cic_ordered import deposit_ordered

        box_size = (box_len_xy, box_len_xy, box_len_z)
        first_scales = tuple((growth - growth0) / L * d for L, d in zip(box_size, n_shape))
        second_scales = tuple(
            (-3 / 7) * (growth * growth - growth0 * growth0) / L * d
            for L, d in zip(box_size, n_shape)
        )
        deposit_ordered(
            cp.asarray(read("hires_density")),
            vel,
            second,
            mass,
            n_shape,
            m_shape,
            growth0,
            first_scales,
            second_scales,
            two,
        )
        grid = mass.astype(cp.float32)
    native_density = reference_density(lib, inputs, grid) if reference_fft else None

    ordered, axes_f64, denom, window, sample, smooth_weights = _geometry_cache(
        n_shape,
        m_shape,
        low_shape,
        box_len_xy,
        box_len_z,
        high,
        opt.SMOOTH_EVOLVED_DENSITY_FIELD,
        sim.DENSITY_SMOOTH_RADIUS,
        sim.N_THREADS,
        opt.USE_FFTW_WISDOM,
        strict_fft,
    )

    def forward(array, shape):
        if shape in ordered:
            return transform(
                array.astype(cp.float32, copy=False), shape[0], threads=sim.N_THREADS
            )
        return cp.fft.rfftn(array.astype(cp.float64))

    def inverse(spectrum, shape):
        if shape in ordered:
            return transform(
                spectrum.astype(cp.complex64, copy=False),
                shape[0],
                True,
                threads=sim.N_THREADS,
            ) / np.float32(np.prod(shape))
        return cp.fft.irfftn(spectrum, s=shape)

    if high:
        saved = forward(grid, m_shape)
        grid = inverse(saved * window, m_shape)[sample].astype(cp.float32)
    if opt.PERTURB_ALGORITHM != "LINEAR":
        mass_factor = (
            1.0
            if high
            else (int(np.prod(low_shape)) / int(np.prod(n_shape)))
        )
        grid = (grid.astype(cp.float64) * mass_factor).astype(cp.float32) - 1
    density_k = forward(grid, low_shape)
    if opt.SMOOTH_EVOLVED_DENSITY_FIELD:
        density_k *= smooth_weights
    if not high:
        saved = density_k
    out = PerturbedField.new(inputs=inputs, redshift=redshift)
    registry = None if resident is None else getattr(resident, "device_registry", None)
    if reference_fft:
        out.set("density", native_density)
    else:
        density = (
            inverse(density_k, low_shape)
            if low_shape in ordered or opt.SMOOTH_EVOLVED_DENSITY_FIELD
            else grid
        )
        device_density = cp.ascontiguousarray(
            cp.maximum(density, np.float32(-1 + 1e-7)), dtype=cp.float32
        )
        out.set("density", device_density.get())
        if resident is not None:
            resident.density_pair = (out.get("density"), device_density)
        if registry is not None:
            registry.record(out.get("density"), device_density)
    m_total = int(np.prod(m_shape))
    for axis, k_f64 in zip("xyz", axes_f64):
        name = "velocity_" + axis
        if name in out.arrays:
            spec = (saved * (1j * rate * k_f64 / denom / m_total)).astype(cp.complex64)
            if high and n_shape != low_shape:
                spec = (spec * window).astype(cp.complex64)
            v = (
                transform(spec, m_shape[0], True, threads=sim.N_THREADS)
                if m_shape in ordered
                else cp.fft.irfftn(spec, s=m_shape, norm="forward")
            )
            vel_dev = cp.ascontiguousarray(v[sample] if high else v, dtype=cp.float32)
            out.set(name, vel_dev.get())
            if registry is not None:
                registry.record(out.get(name), vel_dev)
    return out
