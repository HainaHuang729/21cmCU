"""Build the CHMF CUDA module with an explicitly supplied glibc math source."""

import math
import platform
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREAMBLE = r"""
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef double double_t;
#define attribute_hidden
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#define WANT_ROUNDING 1
#define WANT_ERRNO 0
#define TOINT_INTRINSICS 0
#define __glibc_unlikely(x) (x)
#define math_narrow_eval(x) (x)
#define math_force_eval(x) ((void)0)
#define math_opt_barrier(x) (x)
#define __builtin_fma(a,b,c) fma((a),(b),(c))
#define check_uflow(x) (x)
#define check_oflow(x) (x)
#define __math_uflow(s) ((s) ? -0.0 : 0.0)
#define __math_oflow(s) ((s) ? -INFINITY : INFINITY)
#define __math_divzero(s) __math_oflow(s)
#define __math_invalid(x) nan("")
#define issignaling_inline(x) false
__device__ static uint64_t asuint64(double x) { return __double_as_longlong(x); }
__device__ static double asdouble(uint64_t x) { return __longlong_as_double(x); }
"""


def definitions(folder):
    config = (folder / "math_config.h").read_text()
    # Declare data in the same initializer order as the corresponding C source.
    for name in ("exp", "pow_log", "log"):
        pattern = (
            rf"extern const struct {name}_data\n(.*?) __{name}_data attribute_hidden;"
        )
        declaration = re.search(pattern, config, re.S).group(0)
        body = (folder / f"e_{name}_data.c").read_text()
        if name == "exp":
            struct = re.sub(
                r"\{.*\}",
                """{
  double invln2N, negln2hiN, negln2loN, shift, poly[4];
  double exp2_shift, exp2_poly[EXP2_POLY_ORDER];
  double invlog10_2N, neglog10_2hiN, neglog10_2loN, exp10_poly[5];
  uint64_t tab[2*(1 << EXP_TABLE_BITS)];
}""",
                declaration,
                flags=re.S,
            )
        elif name == "log":
            poly = "  double poly[LOG_POLY_ORDER - 1]; /* First coefficient is 1.  */"
            declaration = declaration.replace(poly + chr(10), "")
            struct = declaration.replace(
                "  /* See e_log_data.c", poly + chr(10) + "  /* See e_log_data.c"
            )
        else:
            struct = declaration
        if name == "exp":
            yield "#define EXP2_POLY_ORDER 5"
        for line in re.findall(rf"^#define {name.upper()}_\w+ \d+", config, re.M):
            yield line
        yield struct.replace("extern const ", "").split(f" __{name}_data")[0] + ";"
        yield (
            re.sub(
                r"^\.\w+\s*=\s*",
                "",
                body.replace("const struct", "__device__ __constant__ const struct"),
                flags=re.M,
            )
            + "\n#undef N"
        )


def implementations(folder):
    for name in ("exp", "pow", "log"):
        s = (folder / f"e_{name}.c").read_text().split(f"#ifndef __{name}")[0]
        s = s.replace("static inline ", "__device__ static inline ")
        s = s.replace(
            f"double\nSECTION\n__{name}",
            f'extern "C" __device__ double\nreference_{name}',
        )
        for x in (
            "top12",
            "top16",
            "specialcase",
            "log_inline",
            "exp_inline",
            "checkint",
            "zeroinfnan",
        ):
            s = re.sub(rf"\b{x}\b", f"{name}_{x}", s)
        yield s
        for macro in re.findall(r"^#define (\w+)", s, re.M):
            yield f"#undef {macro}"


def device_code(folder):
    code = (
        "#define __FP_FAST_FMA 1\n"
        + PREAMBLE
        + "\n".join([*definitions(folder), *implementations(folder)])
    )
    code = re.sub(r"^#include .*\n", "", code, flags=re.M)
    return re.sub(
        r"(?<!\w)0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?\d+",
        lambda value: repr(float.fromhex(value[0])),
        code,
    )


def source_for_host(path):
    if not path:
        return None
    if platform.machine() != "x86_64" or platform.libc_ver() != ("glibc", "2.39"):
        raise ValueError("The requested CUDA CHMF inverse requires x86-64 glibc 2.39")
    return Path(path).expanduser().resolve(strict=True)


@lru_cache(maxsize=4)
def inverse_module(math_path, device_id, gsl_source=None):
    physics = "\n".join(
        file.read_text()
        for file in (
            HERE.parent / "lightcone" / "table_qag.cuh",
            HERE / "catalog_inverse.cu",
            HERE / "catalog_mass_sample.cuh",
            HERE / "catalog_progenitor.cu",
        )
    )
    physics = physics.replace(
        "// CATALOG_BINARY_SPLIT", (HERE / "catalog_binary_split.cuh").read_text()
    )
    physics = physics.replace(
        "// CATALOG_PARTITION", (HERE / "catalog_partition.cuh").read_text()
    )
    tangent = ""
    if gsl_source is not None:
        from catalog_poisson_source import gsl_poisson_device
        from catalog_tan_source import tan_device_source

        physics = physics.replace(
            "// GSL_DEVICE_DISTRIBUTIONS", gsl_poisson_device(gsl_source)
        )
        tangent = tan_device_source(math_path)
    result = link_math_module(math_path, device_id, physics, tangent)
    verify_math(result.get_function("verify_catalog_math"))
    if tangent:
        verify_tangent(result.get_function("verify_catalog_tangent"))
    return result


def link_math_module(math_path, device_id, physics, tangent=""):
    from cupy.cuda import Device, compiler, function

    arch = Device(device_id).compute_capability
    declarations = """
extern "C" __device__ double reference_exp(double);
extern "C" __device__ double reference_pow(double, double);
extern "C" __device__ double reference_log(double);
"""
    if tangent:
        declarations += """
#define CATALOG_DEVICE_TAN
extern "C" __device__ double reference_catalog_tan(double);
__device__ double catalog_word_tangent(unsigned word) {
    return reference_catalog_tan(3.14159265358979323846 * (word * 2.3283064365386963e-10));
}
extern "C" __global__ void verify_catalog_tangent(int count, const unsigned *words,
                                                   double *values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = catalog_word_tangent(words[i]);
}
"""
    units = (
        ("catalog_math", device_code(math_path) + "\n" + tangent, "true"),
        ("catalog_physics", declarations + physics, "false"),
    )
    sources = [
        (
            name,
            source,
            (
                f"-arch=compute_{arch}",
                f"--fmad={fma}",
                "--relocatable-device-code=true",
                "--std=c++11",
            ),
        )
        for name, source, fma in units
    ]

    def build():
        linker = function.LinkState()
        for name, source, options in sources:
            program = compiler._NVRTCProgram(source, f"{name}.cu", method="ptx")
            ptx, _ = program.compile(options, None)
            linker.add_ptr_data(ptx, f"{name}.ptx")
        return bytes(linker.complete())

    def load(binary):
        result = function.Module()
        result.load(binary)
        return result

    return _cached_module(sources, arch, build, load)


def _cached_module(sources, arch, build, load):
    import fcntl
    import hashlib
    import json
    import os
    from tempfile import TemporaryDirectory

    import cupy as cp
    from cupy.cuda import compiler, driver, runtime

    if os.environ.get("CUPY_CACHE_IN_MEMORY") == "1":
        return load(build())
    key = hashlib.sha256(
        json.dumps(
            (
                arch,
                cp.__version__,
                compiler._get_nvrtc_version(),
                runtime.driverGetVersion(),
                sources,
            )
        ).encode()
    ).hexdigest()
    directory = Path(compiler.get_cache_dir()) / "21cmfast-catalog-v1"
    path = directory / f"{key}.cubin"
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = path.with_suffix(".lock").open("a")
    except OSError:
        return load(build())

    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError:
            return load(build())
        try:
            checksum, binary = path.read_bytes().split(b"\n", 1)
            if hashlib.sha256(binary).hexdigest().encode() == checksum:
                return load(binary)
        except (OSError, ValueError, driver.CUDADriverError):
            pass
        binary = build()
        result = load(binary)
        try:
            with TemporaryDirectory(dir=directory) as temporary:
                candidate = Path(temporary) / path.name
                candidate.write_bytes(
                    hashlib.sha256(binary).hexdigest().encode() + b"\n" + binary
                )
                candidate.replace(path)
        except OSError:
            pass
        return result


def verify_math(kernel):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(424242)
    n = 32768
    points = np.column_stack(
        (
            rng.uniform(-746.0, 33.0, n),
            np.exp(rng.uniform(-21.0, 18.0, n)),
            np.exp(rng.uniform(-28.0, 2.0, n)),
        )
    )
    points[:4096, 2] = 1.0 + rng.uniform(-0.1, 0.1, 4096)
    native = np.column_stack(
        (
            [math.exp(x) for x in points[:, 0]],
            [math.pow(x, -0.81) for x in points[:, 1]],
            [math.log(x) for x in points[:, 2]],
        )
    )
    inputs = cp.asarray(points)
    outputs = cp.empty_like(inputs)
    kernel(((n + 127) // 128,), (128,), (np.int32(n), inputs, outputs))
    failed = np.count_nonzero(outputs.get() != native, axis=0)
    if failed.any():
        raise ValueError(
            "CUDA CHMF math differs from the host libm "
            f"(exp, pow, log mismatches: {failed.tolist()})"
        )


def verify_tangent(kernel):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(424242)
    words = rng.integers(0, 2**32, 32768, dtype=np.uint32)
    words[:5] = [0, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF]
    native = np.array([math.tan(math.pi * (int(word) * 2**-32)) for word in words])
    outputs = cp.empty(words.size, dtype=np.float64)
    kernel(
        (words.size // 128,), (128,), (np.int32(words.size), cp.asarray(words), outputs)
    )
    failed = np.count_nonzero(outputs.get().view(np.uint64) != native.view(np.uint64))
    if failed:
        raise ValueError(
            f"CUDA CHMF tangent differs from scalar host libm ({failed} words)"
        )
