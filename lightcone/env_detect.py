"""Auto-detect GSL, FFTW3f, and CUDA toolchains across local, Conda, and system paths."""
from ctypes.util import find_library
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def _add_dir(target_list, seen, candidate):
    """Append an existing directory path to target_list once, preserving search order."""
    if not candidate:
        return
    p = Path(candidate).expanduser()
    try:
        if not p.is_dir():
            return
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p)
    if resolved not in seen:
        seen.add(resolved)
        target_list.append(resolved)


def _add_prefix(prefix, inc_dirs, inc_seen, lib_dirs, lib_seen):
    """Add standard include/lib subdirectories under a software installation prefix."""
    if not prefix:
        return
    p = Path(prefix).expanduser()
    _add_dir(inc_dirs, inc_seen, p / "include")
    _add_dir(lib_dirs, lib_seen, p / "lib")
    _add_dir(lib_dirs, lib_seen, p / "lib64")


def _parse_flag_dirs(cmd, inc_dirs, inc_seen, lib_dirs, lib_seen):
    """Run a pkg-config/gsl-config command and collect any -I/-L directories."""
    if not shutil.which(cmd[0]):
        return
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except OSError:
        return
    if res.returncode != 0:
        return
    tokens = shlex.split(res.stdout)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-I") and len(tok) > 2:
            _add_dir(inc_dirs, inc_seen, tok[2:])
        elif tok == "-I" and i + 1 < len(tokens):
            i += 1
            _add_dir(inc_dirs, inc_seen, tokens[i])
        elif tok.startswith("-L") and len(tok) > 2:
            _add_dir(lib_dirs, lib_seen, tok[2:])
        elif tok == "-L" and i + 1 < len(tokens):
            i += 1
            _add_dir(lib_dirs, lib_seen, tokens[i])
        i += 1


def detect_native_deps(root=None, gsl_prefix=None, fftw_prefix=None):
    """Detect include/library directories and verify GSL and FFTW3f headers.

    Search priority:
      1. Explicit prefixes / PS_GSL_PREFIX / PS_FFTW_PREFIX / compiler env paths
      2. Repo-local <root>/include and <root>/lib (when present)
      3. Active Conda / Python environment prefix
      4. pkg-config (gsl, fftw3f) and gsl-config
      5. Standard Linux/Unix system directories
    """
    inc_dirs, inc_seen = [], set()
    lib_dirs, lib_seen = [], set()

    for prefix in (
        gsl_prefix,
        fftw_prefix,
        os.environ.get("PS_GSL_PREFIX"),
        os.environ.get("PS_FFTW_PREFIX"),
    ):
        _add_prefix(prefix, inc_dirs, inc_seen, lib_dirs, lib_seen)

    for env_key in ("CPATH", "C_INCLUDE_PATH"):
        for entry in os.environ.get(env_key, "").split(os.pathsep):
            _add_dir(inc_dirs, inc_seen, entry)
    for env_key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        for entry in os.environ.get(env_key, "").split(os.pathsep):
            _add_dir(lib_dirs, lib_seen, entry)

    if root is not None:
        r = Path(root).expanduser()
        _add_dir(inc_dirs, inc_seen, r / "include")
        _add_dir(lib_dirs, lib_seen, r / "lib")

    for py_prefix in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        str(Path(sys.executable).resolve().parent.parent),
    ):
        _add_prefix(py_prefix, inc_dirs, inc_seen, lib_dirs, lib_seen)

    for pkg in ("gsl", "fftw3f"):
        _parse_flag_dirs(["pkg-config", "--cflags-only-I", "--libs-only-L", pkg],
                         inc_dirs, inc_seen, lib_dirs, lib_seen)
    _parse_flag_dirs(["gsl-config", "--cflags", "--libs"],
                     inc_dirs, inc_seen, lib_dirs, lib_seen)

    for std_inc in ("/usr/local/include", "/usr/include"):
        _add_dir(inc_dirs, inc_seen, std_inc)
    for std_lib in ("/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"):
        _add_dir(lib_dirs, lib_seen, std_lib)

    gsl_found = any((Path(d) / "gsl" / "gsl_integration.h").is_file() for d in inc_dirs)
    fftw_found = any((Path(d) / "fftw3.h").is_file() for d in inc_dirs)
    return {
        "include_dirs": inc_dirs,
        "library_dirs": lib_dirs,
        "gsl_found": gsl_found,
        "fftw_found": fftw_found,
    }


def find_shared_lib(names, search_dirs):
    """Find a shared library path across search_dirs, falling back to ctypes.util."""
    if isinstance(names, str):
        names = [names]

    for directory in search_dirs:
        base = Path(directory)
        if not base.is_dir():
            continue
        for pattern in names:
            if any(ch in pattern for ch in "*?["):
                for match in sorted(base.glob(pattern)):
                    if match.exists():
                        return str(match)
            else:
                candidate = base / pattern
                if candidate.exists():
                    return str(candidate)

    for pattern in names:
        stem = pattern.split("*")[0]
        if stem.startswith("lib"):
            stem = stem[3:]
        for suffix in (".so.", ".so", ".dylib"):
            if suffix in stem:
                stem = stem.split(suffix, 1)[0]
        if stem:
            found = find_library(stem)
            if found:
                return found

    return names[0]


def detect_cuda_path(explicit=None):
    """Locate a CUDA toolkit root and its header include directory.

    Returns (cuda_path, cuda_include_dir), or (None, None) if unavailable.
    """
    candidates = []
    if explicit:
        candidates.append((explicit, False))
    for env_key in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(env_key):
            candidates.append((os.environ[env_key], False))
    nvcc = shutil.which("nvcc")
    nvcc_prefix = None
    if nvcc:
        resolved_prefix = Path(nvcc).resolve().parent.parent
        if resolved_prefix != Path("/usr"):
            candidates.append((str(resolved_prefix), True))
        else:
            nvcc_prefix = "/usr"
    for py_prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if py_prefix:
            candidates.append((py_prefix, True))
    candidates.append(("/usr/local/cuda", True))
    if nvcc_prefix:
        candidates.append((nvcc_prefix, True))

    seen = set()
    for raw_path, require_runtime_hdr in candidates:
        if not raw_path:
            continue
        p = Path(raw_path).expanduser()
        if not p.is_dir():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)

        hdr_candidates = [
            p / "include",
            p / "targets" / "x86_64-linux" / "include",
        ]
        for inc in hdr_candidates:
            if (inc / "cuda_runtime.h").is_file():
                resolved_inc = str(inc.resolve())
                return key, (None if resolved_inc == "/usr/include" else resolved_inc)
        if not require_runtime_hdr and (p / "include").is_dir():
            resolved_inc = str((p / "include").resolve())
            return key, (None if resolved_inc == "/usr/include" else resolved_inc)

    return None, None
