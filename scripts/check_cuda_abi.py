#!/usr/bin/env python3
"""Verify 1:1 parity across cuda_api.h, cuda_stubs.c, cuda/*.cu, and CFFI wrapper headers."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = REPO_ROOT / "src" / "py21cmfast" / "src" / "cuda"
API_HEADER = CUDA_DIR / "cuda_api.h"
COMMON_HEADER = CUDA_DIR / "cuda_common.cuh"
STUBS_FILE = REPO_ROOT / "src" / "py21cmfast" / "src" / "cuda_stubs.c"
CFFI_HEADER = REPO_ROOT / "src" / "py21cmfast" / "src" / "_functionprototypes_wrapper.h"

_DECL_RE = re.compile(
    r'^[ \t]*(?:extern\s+"C"\s+)?'
    r"(?:int|void|double|float|size_t|unsigned\s+long\s+long|const\s+char\s*\*|void\s*\*)"
    r"\s*(p21_cuda_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _collect_symbols() -> tuple[set[str], set[str], set[str], dict[str, set[str]], set[str]]:
    header_syms = set(_DECL_RE.findall(_strip_comments(API_HEADER.read_text(encoding="utf-8"))))
    common_syms = set(_DECL_RE.findall(_strip_comments(COMMON_HEADER.read_text(encoding="utf-8"))))
    stub_syms = set(_DECL_RE.findall(_strip_comments(STUBS_FILE.read_text(encoding="utf-8"))))
    cffi_syms = set(_DECL_RE.findall(_strip_comments(CFFI_HEADER.read_text(encoding="utf-8"))))

    cu_map: dict[str, set[str]] = {}
    for cu_path in sorted(CUDA_DIR.glob("*.cu")):
        syms = set(_DECL_RE.findall(_strip_comments(cu_path.read_text(encoding="utf-8"))))
        cu_map[cu_path.name] = syms
    return header_syms, common_syms, stub_syms, cu_map, cffi_syms


def _print_stub_skeleton(symbol: str) -> int:
    text = _strip_comments(API_HEADER.read_text(encoding="utf-8"))
    pat = re.compile(
        rf"((?:int|void|double|float|size_t|unsigned\s+long\s+long|const\s+char\s*\*|void\s*\*)"
        rf"\s*{re.escape(symbol)}\s*\([^;]*\))\s*;",
        re.DOTALL,
    )
    match = pat.search(text)
    if not match:
        sys.stderr.write(f"Symbol {symbol!r} not found in {API_HEADER}\n")
        return 1
    sig = " ".join(match.group(1).split())
    ret_type = sig.split(symbol, 1)[0].strip()
    args_str = sig[sig.index("(") + 1 : sig.rindex(")")].strip()
    arg_names: list[str] = []
    if args_str and args_str != "void":
        for raw_arg in args_str.split(","):
            tok = raw_arg.strip().split()[-1].lstrip("*")
            if "[" in tok:
                tok = tok.split("[", 1)[0]
            if tok:
                arg_names.append(tok)
    body = [f"{sig} {{"]
    for name in arg_names:
        body.append(f"    (void){name};")
    if ret_type == "void":
        pass
    elif ret_type.endswith("*"):
        body.append("    return NULL;")
    elif ret_type == "int":
        body.append("    return -1;")
    else:
        body.append("    return 0;")
    body.append("}")
    print("\n".join(body))
    return 0


def main() -> int:
    """Validate 1:1 CUDA symbol parity across headers, stubs, and CUDA sources."""
    parser = argparse.ArgumentParser(description="Verify 21cmCU CUDA C/CFFI ABI parity")
    parser.add_argument(
        "--stub-skeleton",
        metavar="SYMBOL",
        default=None,
        help="Print a cuda_stubs.c skeleton for a symbol declared in cuda_api.h",
    )
    args = parser.parse_args()
    if args.stub_skeleton:
        return _print_stub_skeleton(args.stub_skeleton)

    header_syms, common_syms, stub_syms, cu_map, cffi_syms = _collect_symbols()
    cu_syms: set[str] = set()
    dups: dict[str, list[str]] = {}
    for fname, syms in cu_map.items():
        for sym in syms:
            if sym in cu_syms:
                dups.setdefault(sym, []).append(fname)
            cu_syms.add(sym)

    cu_public_syms = cu_syms - common_syms
    errors: list[str] = []
    if dups:
        errors.append(f"Duplicate p21_cuda_* definitions across .cu files: {dups}")
    if header_syms != stub_syms:
        errors.append(
            "cuda_api.h vs cuda_stubs.c mismatch: "
            f"missing_in_stubs={sorted(header_syms - stub_syms)}, "
            f"extra_in_stubs={sorted(stub_syms - header_syms)}"
        )
    if header_syms != cu_public_syms:
        errors.append(
            "cuda_api.h vs cuda/*.cu mismatch: "
            f"missing_in_cu={sorted(header_syms - cu_public_syms)}, "
            f"extra_in_cu={sorted(cu_public_syms - header_syms)}"
        )
    if not cffi_syms.issubset(header_syms):
        errors.append(f"CFFI symbols missing from cuda_api.h: {sorted(cffi_syms - header_syms)}")

    if errors:
        for err in errors:
            sys.stderr.write(f"ERROR: {err}\n")
        return 1

    print(
        f"CUDA ABI OK: cuda_api.h={len(header_syms)}, cuda_stubs.c={len(stub_syms)}, "
        f"cuda/*.cu={len(cu_public_syms)} ({len(cu_map)} files), "
        f"internal_cuh={len(cu_syms & common_syms)}, cffi={len(cffi_syms)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
