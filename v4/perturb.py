"""GPU perturbation adapter for cubic v4.3.2 InitialConditions."""
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


def modes(n, length):
    i = cp.arange(n)
    k = (cp.where(i <= n // 2, i, i - n) * (2 * np.pi / length)).astype(cp.float32)
    return k[:, None, None], k[None, :, None], k[None, None, : n // 2 + 1]


def gaussian_weights(k2, radius):
    # The native filter squares the float radius before multiplying by k^2.
    kr2 = k2 * np.float32(radius * radius)
    return cp.exp((-0.5 * 0.643**2) * kr2.astype(cp.float64))


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

        def buffer(n):
            shape = (n, n, 2 * (n // 2 + 1))
            address = lib.fftwf_malloc(4 * int(np.prod(shape)))
            if not address:
                raise MemoryError("FFTW allocation failed")
            cleanup.callback(lib.fftwf_free, address)
            array = np.ctypeslib.as_array(
                ctypes.cast(address, ctypes.POINTER(ctypes.c_float)), shape
            )
            return ptr(address), array

        low = sim.HII_DIM
        source, _ = buffer(grid.shape[0])
        coarse, box = buffer(low)
        if opt.PERTURB_ON_HIGH_RES:
            fine, fine_box = buffer(sim.DIM)
            fine_box[:, :, : sim.DIM] = grid.get()
            lib.assign_to_lowres_grid(fine, coarse, source)
        else:
            box[:, :, :low] = grid.get()
        if opt.PERTURB_ALGORITHM != "LINEAR":
            lib.normalise_delta_grid(coarse)
        lib.smooth_and_clip_density(coarse, source)
        return box[:, :, :low].copy()


def perturb_field_gpu(*, redshift, initial_conditions, reference_fft=False, strict_fft=False):
    """Return a CPU-visible v4 PerturbedField, with GPU CIC, FFTs and velocities."""
    ic = initial_conditions
    inputs = ic.inputs
    sim, opt = inputs.simulation_options, inputs.matter_options
    if sim.NON_CUBIC_FACTOR != 1:
        raise ValueError("This adapter supports cubic boxes")
    n, low, length = sim.DIM, sim.HII_DIM, sim.BOX_LEN
    high = opt.PERTURB_ON_HIGH_RES
    m = n if high else low
    prefix = "hires_" if high else "lowres_"
    lib = ctypes.CDLL(c_21cmfast.__file__)
    for f in (lib.dicke, lib.ddickedt):
        f.argtypes = [ctypes.c_double]
        f.restype = ctypes.c_double
    with c_state(inputs, broadcast_inputs=True):
        native_z = float(np.float32(redshift))  # ComputePerturbedField accepts a C float.
        growth = lib.dicke(native_z)
        growth0 = lib.dicke(sim.INITIAL_REDSHIFT)
        rate = lib.ddickedt(native_z) / growth
    if opt.PERTURB_ALGORITHM == "LINEAR":
        grid = (
            cp.asarray(ic.get(prefix + "density"), dtype=cp.float64) * growth
        ).astype(cp.float32)
    else:

        def vectors(suffix):
            return cp.stack(
                [cp.asarray(ic.get(prefix + "v" + axis + suffix)) for axis in "xyz"]
            )

        two = opt.PERTURB_ALGORITHM == "2LPT"
        vel = vectors("")
        second = vectors("_2LPT") if two else vel
        mass = cp.zeros((m, m, m), dtype=cp.float64)
        from cic_ordered import deposit_ordered
        deposit_ordered(cp.asarray(ic.get("hires_density")), vel, second, mass, n, m,
                        growth0, (growth-growth0)/length*n,
                        (-3/7)*(growth*growth-growth0*growth0)/length*n, two)
        grid = mass.astype(cp.float32)
    native_density = reference_density(lib, inputs, grid) if reference_fft else None

    # Use the native plan for both directions as a pair. Preserve the existing
    # cuFFT backend for layouts the CUDA compiler cannot yet lower.
    def supported(size):
        if opt.USE_FFTW_WISDOM:
            if strict_fft:
                raise UnsupportedPlan("External FFTW wisdom is enabled")
            return False
        try:
            for inverse_direction in (False, True):
                executable(size, inverse_direction, sim.N_THREADS)
        except UnsupportedPlan:
            if strict_fft:
                raise
            return False
        return True

    ordered = {size for size in {m, low} if supported(size)}

    def forward(array, size):
        if size in ordered:
            return transform(array.astype(cp.float32, copy=False), size, threads=sim.N_THREADS)
        return cp.fft.rfftn(array.astype(cp.float64))

    def inverse(spectrum, size):
        if size in ordered:
            return transform(
                spectrum.astype(cp.complex64, copy=False), size, True, threads=sim.N_THREADS
            ) / np.float32(size**3)
        return cp.fft.irfftn(spectrum, s=(size,) * 3)

    axes = modes(m, length)
    k2 = sum(k * k for k in axes)
    if high:
        saved = forward(grid, m)
        radius = np.float32(0.620350491 * length / low)
        kr = (
            (cp.sqrt(k2.astype(cp.float64)) * radius)
            .astype(cp.float32)
            .astype(cp.float64)
        )
        safe = cp.where(kr == 0, 1.0, kr)
        window = cp.where(
            kr < 1e-4, 1 - kr**2 / 10, 3 * (cp.sin(kr) - kr * cp.cos(kr)) / safe**3
        )
        idx = cp.floor(cp.arange(low) * (n / low) + 0.5).astype(cp.int32)
        sample = cp.ix_(idx, idx, idx)
        grid = inverse(saved * window, m)[sample].astype(cp.float32)
    if opt.PERTURB_ALGORITHM != "LINEAR":
        grid = (grid.astype(cp.float64) * (1 if high else (low / n) ** 3)).astype(
            cp.float32
        ) - 1
    density_k = forward(grid, low)
    if opt.SMOOTH_EVOLVED_DENSITY_FIELD:
        low_k2 = sum(k * k for k in modes(low, length))
        radius = np.float32(sim.DENSITY_SMOOTH_RADIUS * length / low)
        density_k *= gaussian_weights(low_k2, radius)
    if not high:
        saved = density_k
    out = PerturbedField.new(inputs=inputs, redshift=redshift)
    if reference_fft:
        out.set("density", native_density)
    else:
        density = (
            inverse(density_k, low)
            if low in ordered or opt.SMOOTH_EVOLVED_DENSITY_FIELD
            else grid
        )
        out.set(
            "density",
            cp.maximum(density, np.float32(-1 + 1e-7)).astype(cp.float32).get(),
        )
    denom = cp.where(k2 == 0, 1, k2)
    for axis, k in zip("xyz", axes):
        name = "velocity_" + axis
        if name in out.arrays:
            spec = (saved * (1j * rate * k.astype(cp.float64) / denom / m**3)).astype(
                cp.complex64
            )
            if high and n != low:
                spec = (spec * window).astype(cp.complex64)
            v = (
                transform(spec, m, True, threads=sim.N_THREADS)
                if m in ordered
                else cp.fft.irfftn(spec, s=(m,) * 3, norm="forward")
            )
            out.set(name, (v[sample] if high else v).get())
    return out
