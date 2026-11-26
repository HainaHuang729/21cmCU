"""Reproduce v4's one- and two-thread GSL realizations on CUDA."""

from pathlib import Path
import cupy as cp
import numpy as np


@cp.memoize(for_each_device=True)
def stream_module():
    return cp.RawModule(code=Path(__file__).with_suffix(".cu").read_text())


RNG_ORDER = ("mt", "gfsr", "cmrg", "mrg", "taus2")
_STATE_SIZES = {"mt": 624, "gfsr": 16384, "cmrg": 6, "mrg": 5, "taus2": 3}
_DENOMS = {
    "mt": 4294967296.0,
    "gfsr": 4294967296.0,
    "cmrg": 2147483647.0,
    "mrg": 2147483647.0,
    "taus2": 4294967296.0,
}


class Stream:
    def __init__(self, module, kind, seed):
        if kind not in _STATE_SIZES:
            raise ValueError(f"Unsupported GSL stream kind: {kind}")
        self.kind = kind
        # GSL first checks its unsigned-long seed for zero, then uses bits 0..31.
        # Note mt19937/gfsr4 default zero to 4357, whereas cmrg/mrg/taus2 default zero to 1 inside CUDA.
        self.seed = np.uint32(
            ((int(seed) or 4357) if kind in ("mt", "gfsr") else int(seed)) & 0xFFFFFFFF
        )
        self.denom = _DENOMS[kind]
        self.kernel = module.get_function(kind + "_stream")
        self.state = cp.empty(_STATE_SIZES[kind], dtype="I")
        self.offset = 0

    def integers(self, blocks):
        size = blocks * 624
        result = cp.empty(size, dtype="I")
        if self.kind in ("mt", "gfsr"):
            args = [
                self.state,
                result,
                np.int32(blocks if self.kind == "mt" else size),
                self.seed,
                np.int32(not self.offset),
            ]
            if self.kind == "gfsr":
                args.append(np.int32(self.offset % 16384))
            self.kernel((1,), (256,), tuple(args))
        else:
            self.kernel(
                (1,),
                (1,),
                (
                    self.state,
                    result,
                    np.int32(size),
                    self.seed,
                    np.int32(not self.offset),
                ),
            )
        self.offset += size
        return result


def seeds(module, seed, count):
    # Scan the same implicit integer population as gsl_ran_choose.
    stream = Stream(module, "mt", seed)
    total = (2**31 - 1) // 16
    chosen = []
    start = 0
    while len(chosen) < count:
        raw = stream.integers(4096)
        idx = cp.arange(start, start + raw.size, dtype="i8")
        scores = raw.astype("d") * ((total - idx) * 2.0**-32)
        pos = 0
        while len(chosen) < count and pos < len(raw):
            hits = cp.flatnonzero(
                scores[pos : min(len(raw), total - start)] < count - len(chosen)
            )
            if hits.size == 0:
                break
            pos += int(hits[0]) + 1
            chosen.append(start + pos - 1)
        start += raw.size
    for i in range(count - 1, 0, -1):
        while True:
            if pos == raw.size:
                raw = stream.integers(1)
                pos = 0
            j = int(raw[pos]) // (0xFFFFFFFF // (i + 1))
            pos += 1
            if j <= i:
                break
        chosen[i], chosen[j] = chosen[j], chosen[i]
    return chosen


def gaussian(stream, count):
    result = cp.empty(count, dtype="d")
    written = 0
    carry = cp.empty(0, dtype="I")
    while written < count:
        blocks = min(8192, max(1, int((count - written) * 2.6 / 624) + 1))
        raw = stream.integers(blocks)
        raw = cp.concatenate((carry, raw[raw != 0]))
        pairs = raw.size // 2
        carry = raw[2 * pairs :]
        u = raw[: 2 * pairs].reshape(-1, 2).astype("d") / stream.denom
        xy = -1.0 + 2.0 * u
        x, y = xy.T
        r = x * x + y * y
        valid = (r <= 1) & (r != 0)
        r = r[valid]
        values = y[valid] * cp.sqrt(-2.0 * cp.log(r) / r)
        take = min(values.size, count - written)
        result[written : written + take] = values[:take]
        written += take
    return result


def sample(
    hi_shape,
    box_len_xy,
    box_len_z,
    volume,
    seed,
    threads,
    cosmology,
    norm,
    index,
    density_power=None,
):
    if threads < 1:
        raise ValueError("GPU IC seed stream requires N_THREADS >= 1")
    nx, ny, nz = (int(d) for d in hi_shape)
    hz = nz // 2 + 1
    module = stream_module()
    boxes = cp.empty((nx, ny, hz), dtype="F")

    def k_axis(dim, length):
        k = cp.arange(dim, dtype="d")
        return cp.where(k <= dim // 2, k, k - dim) * (2 * np.pi / length)

    kx = k_axis(nx, box_len_xy)
    ky = k_axis(ny, box_len_xy)
    kz = k_axis(nz, box_len_z)[:hz]
    sound, alpha, beta, omhh, fnu, _, theta, sigma = cosmology
    for thread, subseed in enumerate(seeds(module, int(seed), threads)):
        first = thread * (nx // threads) + min(thread, nx % threads)
        stop = first + nx // threads + int(thread < nx % threads)
        out = boxes[first:stop]
        g = gaussian(Stream(module, RNG_ORDER[thread % 5], subseed), out.size * 2)
        wave = cp.sqrt(
            kx[first:stop, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )
        if density_power is not None:
            amp = cp.sqrt(
                0.5 * volume * density_power.power(wave, cosmology, norm, index)
            )
        else:
            safe = cp.where(wave == 0, 1.0, wave)
            q = safe * theta**2 / omhh
            qe = q / (
                np.sqrt(alpha) + (1 - np.sqrt(alpha)) / (1 + (0.43 * safe * sound) ** 4)
            )
            tf = cp.log(np.e + 1.84 * beta * np.sqrt(alpha) * qe)
            tf = tf / (tf + qe**2 * (14.4 + 325 / (1 + 60.5 * qe**1.11)))
            qnu = 3.92 * q / np.sqrt(fnu)
            tf *= 1.0 + 1.2 * fnu**0.64 / (qnu**-1.6 + qnu**0.8)
            tf *= safe**2
            amp = cp.sqrt(
                volume
                * 0.5
                * sigma
                * norm
                * (safe / 0.05) ** (index - 1.0)
                * tf**2
                / safe**3
            )
        out.real = g[::2].reshape(out.shape) * amp
        out.imag = g[1::2].reshape(out.shape) * amp
    boxes[0, 0, 0] = 0
    plane = cp.RawModule(
        code=(Path(__file__).resolve().parents[1] / "lightcone/ic_seed.cu").read_text()
    ).get_function("enforce_real_planes_general")
    plane(
        ((2 * nx * ny + 255) // 256,),
        (256,),
        (boxes, np.int32(nx), np.int32(ny), np.int32(nz)),
    )
    return boxes


def initial_fields(
    modes,
    hi_shape,
    lo_shape,
    box_len_xy,
    box_len_z,
    volume,
    radius,
    high,
    second_order,
    threads,
    filter_kernel,
    streaming=None,
    wisdom=False,
):
    from fft_general import transform_3d
    from itertools import combinations

    nx, ny, nz = (int(d) for d in hi_shape)
    lx, ly, lz = (int(d) for d in lo_shape)

    def inverse(f):
        return transform_3d(f, (nx, ny, nz), True, threads, wisdom=wisdom)

    dk_x = 2 * np.pi / box_len_xy
    dk_y = (1.0 / box_len_xy) * (2 * np.pi)
    dk_z = 2 * np.pi / box_len_z

    def sampled(f, target_shape, smooth):
        tx, ty, tz = (int(d) for d in target_shape)
        if smooth and (tx, ty, tz) != (nx, ny, nz):
            f = cp.ascontiguousarray(f).copy()
            filter_kernel(
                ((f.size + 255) // 256,),
                (256,),
                (
                    f,
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nz),
                    np.float64(dk_x),
                    np.float64(dk_z),
                    np.float32(radius),
                    np.int32(0),
                ),
            )
        x = inverse(f)
        if (tx, ty, tz) != (nx, ny, nz):
            ratio = nx / tx
            idx_xy = cp.floor(cp.arange(tx) * ratio + 0.5).astype("i")
            idx_z = cp.floor(cp.arange(tz) * ratio + 0.5).astype("i")
            x = x[cp.ix_(idx_xy, idx_xy, idx_z)]
        return x

    invvol = np.float32(1 / np.float32(volume))
    yield 0, inverse(modes) * invvol
    yield 1, sampled(modes, lo_shape, True) / np.float32(volume)

    def axis_idx(dim):
        k = cp.arange(dim, dtype="d")
        return cp.where(k <= dim // 2, k, k - dim)

    # The native GCC loops in compute_f_gradient/laplacian hoist 1/L for
    # the middle axis, and 2*pi/L for the two outer axes.
    axes = (
        (axis_idx(nx) * dk_x)[:, None, None],
        (axis_idx(ny) * dk_y)[None, :, None],
        (axis_idx(nz)[: nz // 2 + 1] * dk_z)[None, None, :],
    )
    k2 = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    if streaming is not None:
        classy, cosmology, norm, index, speed_kms = streaming
        wave = cp.sqrt(k2)
        matter = classy.power(wave, cosmology, norm, index)
        velocity = classy.power(wave, cosmology, norm, index, velocity=True)
        # Both native spectra vanish at DC. Set its directional weight to zero.
        matter[0, 0, 0] = wave[0, 0, 0] = 1.0
        ratio = cp.sqrt(velocity / matter)
        total = cp.zeros((lx, ly, lz), dtype="f")
        for axis in axes:
            component = cp.empty_like(modes)
            component.real = (-modes.imag * axis) / wave * ratio * speed_kms
            component.imag = (modes.real * axis) / wave * ratio * speed_kms
            real = sampled(component, lo_shape, True).astype("d")
            # The native output accumulator is float after each of x, y and z.
            total[...] = total.astype("d") + real * real
        yield 11, (cp.sqrt(total.astype("d")) / volume).astype("f")
    k2[0, 0, 0] = 1
    target = hi_shape if high else lo_shape

    def deriv(f, i):
        scale = axes[i]
        out = cp.empty_like(f)
        out.real = (-f.imag * scale) / k2
        out.imag = (f.real * scale) / k2
        out[0, 0, 0] = 0
        return sampled(out, target, not high)

    for i in range(3):
        yield 2 + i, deriv(modes, i) / np.float32(volume)
    if second_order:
        kernel = cp.RawModule(
            code=Path(__file__).with_suffix(".cu").read_text(),
            options=("--fmad=false",),
        ).get_function("hessian_modes")

        def hessian(i, j):
            f = cp.empty_like(modes)
            kernel(
                ((modes.size + 255) // 256,),
                (256,),
                (
                    modes,
                    f,
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nz),
                    np.float64(dk_x),
                    np.float64(dk_y),
                    np.float64(dk_z),
                    np.int32(i),
                    np.int32(j),
                ),
            )
            return inverse(f)

        diag = [hessian(i, i) for i in range(3)]
        if not high:
            # The native low-resolution path exposes its three high-resolution
            # diagonal Hessian work arrays through the InitialConditions struct.
            yield from enumerate(diag, 8)
        source = cp.zeros((nx, ny, nz), dtype="f")
        for i, j in combinations(range(3), 2):
            cross = hessian(i, j)
            source += diag[i].astype("d") * diag[j]
            source -= cross.astype("d") * cross
        denom = np.float32(np.float32(volume) * np.float32(volume)) * np.float32(
            nx * ny * nz
        )
        modes2 = transform_3d(
            source / denom, (nx, ny, nz), False, threads, wisdom=wisdom
        )
        for i in range(3):
            yield 5 + i, deriv(modes2, i)

