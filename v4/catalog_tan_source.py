"""Translate the glibc 2.39 scalar tangent for angles made from GSL words."""

import re

FILES = ("s_tan.c", "utan.h", "utan.tbl", "dla.h", "mydefs.h")


def has_tangent_source(source):
    return (source / FILES[0]).is_file()


def tan_device_source(source):
    if not has_tangent_source(source):
        return ""
    pieces = {name: (source / name).read_text() for name in FILES}
    tangent = pieces["s_tan.c"]
    end = tangent.index("  /* (---) The case 25 <")
    # GSL gamma proposals only use 0 <= pi * U < pi. Argument reduction
    # outside that interval is never needed by the catalog sampler.
    tangent = tangent[:end] + '  retval = nan("");\nret:\n  return retval;\n}\n'
    tangent = tangent.replace(
        "double\nSECTION\n__tan", 'extern "C" __device__ double\nreference_catalog_tan'
    )
    tangent = tangent.replace("  int __branred (double, double *, double *);", "")
    # The x86 glibc FMA implementation contracts these operations. Explicit
    # FMAs also preserve the contractions that use the shared reduction tail.
    for original, translated in (
        ("t2 = d9.d + x2 * d11.d;", "t2 = fma(x2, d11.d, d9.d);"),
        ("t2 = d7.d + x2 * t2;", "t2 = fma(x2, t2, d7.d);"),
        ("t2 = d5.d + x2 * t2;", "t2 = fma(x2, t2, d5.d);"),
        ("t2 = d3.d + x2 * t2;", "t2 = fma(x2, t2, d3.d);"),
        ("t2 *= x * x2;", ""),
        ("y = x + t2;", "y = fma(x * x2, t2, x);"),
        ("(mfftnhf.d + 256 * w)", "fma(256., w, mfftnhf.d)"),
        ("(mfftnhf.d + 256 * ya)", "fma(256., ya, mfftnhf.d)"),
        ("z + z * z2 * (e0.d + z2 * e1.d)", "fma(z * z2, fma(z2, e1.d, e0.d), z)"),
        ("(x * hpinv.d + toint.d)", "fma(x, hpinv.d, toint.d)"),
        ("(x - xn * mp1.d) - xn * mp2.d", "fma(-xn, mp2.d, fma(-xn, mp1.d, x))"),
        ("da = xn * mp3.d;", ""),
        ("a = t1 - da;", "a = fma(-xn, mp3.d, t1);"),
        ("da = (t1 - a) - da;", "da = fma(-xn, mp3.d, t1 - a);"),
        ("t2 = d9.d + a2 * d11.d;", "t2 = fma(a2, d11.d, d9.d);"),
        ("t2 = d7.d + a2 * t2;", "t2 = fma(a2, t2, d7.d);"),
        ("t2 = d5.d + a2 * t2;", "t2 = fma(a2, t2, d5.d);"),
        ("t2 = d3.d + a2 * t2;", "t2 = fma(a2, t2, d3.d);"),
        ("t2 = da + a * a2 * t2;", "t2 = fma(a * a2, t2, da);"),
    ):
        if original not in tangent:
            raise ValueError(f"Unexpected glibc 2.39 tangent expression: {original}")
        tangent = tangent.replace(original, translated)
    types = pieces["mydefs.h"].replace(
        "typedef int int4;", "typedef unsigned int tan_word;"
    )
    types = types.replace("int4 i[2]", "tan_word i[2]")
    table = "\n".join(
        pieces[name].replace(
            "static const mynumber", "__device__ __constant__ const mynumber"
        )
        for name in ("utan.h", "utan.tbl")
    )
    preamble = """
#define LITTLE_ENDI 1
#define LOW_HALF 0
#define HIGH_HALF 1
#define SET_RESTORE_ROUND_53BIT(x) ((void)0)
#define math_check_force_underflow_nonneg(x) ((void)0)
#define __set_errno(x) ((void)0)
"""
    division = pieces["dla.h"].replace(
        "(((((x)-u)-uu)+(xx))-c*(yy))/(y)", "fma(-c,(yy),(((x)-u)-uu)+(xx))/(y)"
    )
    return re.sub(
        r"^#include .*\n",
        "",
        "\n".join((preamble, types, division, table, tangent)),
        flags=re.M,
    )
