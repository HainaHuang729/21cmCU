"""Translate the integer-gamma and binomial routines from a supplied GSL 2.7 tree."""

import ctypes as C
import os
import re
from pathlib import Path


def configured_gsl_source():
    configured = os.environ.get("P21_CUDA_GSL_SOURCE")
    if not configured:
        return None
    version = C.c_char_p.in_dll(C.CDLL("libgsl.so.25"), "gsl_version").value
    if version != b"2.7":
        raise ValueError("The CUDA high-mean Poisson path requires GSL 2.7")
    return Path(configured).expanduser().resolve(strict=True)


def _function(text, name):
    match = re.search(
        r"(?m)^(?:static )?(?:double|unsigned int)\s+" + name + r"\s*\([^;{}]*\)\s*\{",
        text,
    )
    if match is None:
        raise ValueError(f"GSL source has no definition for {name}")
    end = text.index("\n}", match.end()) + 2
    return text[match.start() : end]


def gsl_poisson_device(source):
    gamma_source = (source / "randist" / "gamma.c").read_text()
    binomial_source = (source / "randist" / "binomial_tpe.c").read_text()
    power_source = (source / "sys" / "pow_int.c").read_text()

    gamma = _function(gamma_source, "gsl_ran_gamma_int")
    gamma = (
        gamma.replace(
            "gsl_ran_gamma_int (const gsl_rng * r, const unsigned int a)",
            "catalog_gamma_int (MassSampleWords &r, unsigned int a, const double *tangents)",
        )
        .replace(
            "prod *= gsl_rng_uniform_pos (r);",
            "double u; do { if (!catalog_uniform(r, u)) return -1.; } while (u == 0.); prod *= u;",
        )
        .replace(
            "gamma_large (r, (double) a)", "catalog_gamma_large(r, (double)a, tangents)"
        )
    )

    large = (
        _function(gamma_source, "gamma_large")
        .replace(
            "gamma_large (const gsl_rng * r, const double a)",
            "catalog_gamma_large(MassSampleWords &r, double a, const double *tangents)",
        )
        .replace(
            "y = tan (M_PI * gsl_rng_uniform (r));",
            "unsigned word; if (!r.take(word)) return -1.; "
            "y = catalog_tangent(word, tangents, r.consumed - 1);",
        )
        .replace("v = gsl_rng_uniform (r);", "if (!catalog_uniform(r, v)) return -1.;")
    )

    binomial = _function(binomial_source, "gsl_ran_binomial")
    binomial = binomial.replace("const gsl_rng * rng", "MassSampleWords &rng")
    binomial = binomial.replace(
        "double u = gsl_rng_uniform (rng);",
        "double u; if (!catalog_uniform(rng, u)) return 0xffffffffu;",
    )
    binomial = binomial.replace(
        "u = gsl_rng_uniform (rng) * p4;",
        "if (!catalog_uniform(rng, u)) return 0xffffffffu; u *= p4;",
    )
    binomial = binomial.replace(
        "v = gsl_rng_uniform (rng);",
        "if (!catalog_uniform(rng, v)) return 0xffffffffu;",
    )
    stirling_start = binomial_source.index("inline static double")
    stirling_end = binomial_source.index("\n}", stirling_start) + 2
    stirling = binomial_source[stirling_start:stirling_end]
    constants = "\n".join(
        re.findall(
            r"^#define (?:SMALL_MEAN|BINV_CUTOFF|FAR_FROM_MEAN) \d+",
            binomial_source,
            re.M,
        )
    )
    if "gsl_rng_uniform" in gamma + large + binomial:
        raise ValueError("Unsupported GSL gamma/binomial random-number call")
    # Preserve the corresponding upstream notices on the generated CUDA source.
    copyright_headers = [
        text[: text.index("*/") + 2]
        for text in (gamma_source, binomial_source, power_source)
    ]
    functions = "\n".join(
        "__device__ " + method
        for method in (
            _function(power_source, "gsl_pow_uint"),
            stirling,
            binomial,
            large,
            gamma,
        )
    )
    functions = re.sub(r"\b(log|exp)\s*\(", r"reference_\1(", functions)
    return "\n".join(
        [
            *copyright_headers,
            "#define CATALOG_HIGH_POISSON 1",
            constants,
            functions,
        ]
    )
