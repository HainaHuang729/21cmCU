"""Compile the small 4.3.2 C adapters once and load private per-context copies."""

import ctypes
import fcntl
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile


def _digest(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


@lru_cache(maxsize=64)
def _file_identity(path, stat_key):
    return [str(path), _digest(path)]


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_identity(
        path, (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    )


def _build_key(compiler, flags, files, native, directory):
    # Macro-expanded sources include every transitive header, even user-provided
    # and system headers. Relative -I and -include keep the caller's working dir.
    # Normalize the generated files' __FILE__ only. Their private temp path
    # otherwise invalidates every cache lookup in logger/error macros.
    expansion = subprocess.check_output(
        [
            *compiler,
            *flags,
            "-E",
            "-P",
            "-dD",
            f"-fmacro-prefix-map={directory}=.",
            *files,
        ]
    )
    config = {
        "schema": 1,
        "platform": [platform.system(), platform.machine()],
        "cc": compiler,
        "flags": flags,
        "version": subprocess.check_output([*compiler, "--version"]).decode(
            errors="replace"
        ),
        "files": [(p.name, _digest(p)) for p in files],
        "native": _identity(native),
        "library_path": os.environ.get("LIBRARY_PATH"),
    }
    executables = [s for s in compiler if shutil.which(s)]
    for tool in ("cc1", "as", "ld", "collect2"):
        candidate = subprocess.check_output(
            [*compiler, f"-print-prog-name={tool}"], text=True
        ).strip()
        if shutil.which(candidate):
            executables.append(candidate)
    config["toolchain"] = [
        _identity(shutil.which(s)) for s in dict.fromkeys(executables)
    ]
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    key.update(expansion)
    return key.hexdigest()


def repair_partition_rejection(source):
    """Give the native PARTITION calls storage for their unused barrier."""
    start = source.index("bool partition_rejection(")
    end = source.index("\n}", start) + 2
    rejection = source[start:end]
    if rejection.count("growthf, NULL") != 2:
        raise ValueError("Unexpected native PARTITION rejection calls")
    repaired = rejection.replace(
        "double test1, test2, randval;", "double test1, test2, randval, barrier;"
    ).replace("growthf, NULL", "growthf, &barrier")
    return source[:start] + repaired + source[end:]


def load_shim(directory, name, sources, include, native, options=()):
    """Load a distinct ELF image per call so its callback pointers and globals are private."""
    directory = Path(directory).resolve()
    native = Path(native).resolve()
    files = []
    for filename, content in sources.items():
        path = directory / filename
        path.write_text(content, encoding="utf8")
        files.append(path)
    compiler = shlex.split(os.environ.get("CC") or "cc")
    user_flags = shlex.split(os.environ.get("CFLAGS", "")) + shlex.split(
        os.environ.get("CPPFLAGS", "")
    )
    flags = [
        "-Ofast",
        *options,
        "-std=gnu99",
        "-fPIC",
        "-shared",
        "-fopenmp",
        "-Wl,-Bsymbolic-functions",
        "-I" + str(include),
        *user_flags,
    ]
    private = directory / name

    def compile_private():
        subprocess.run(
            [
                *compiler,
                *flags,
                f"-fmacro-prefix-map={directory}=.",
                *files,
                native,
                "-lm",
                "-o",
                private,
            ],
            check=True,
        )

    # Arbitrary user linker inputs/plugins can have their own transitive files.
    # Compile those invocations directly with all the supplied flags intact.
    if any(
        f.startswith(
            (
                "@",
                "-l",
                "-L",
                "-Wl",
                "-Xlinker",
                "-specs",
                "-fplugin",
                "-B",
                "-fprofile",
                "-march=native",
                "-mcpu=native",
                "-mtune=native",
            )
        )
        for f in user_flags
    ):
        compile_private()
        return ctypes.CDLL(str(private))

    key = _build_key(compiler, flags, files, native, directory)
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = Path(
        os.environ.get("P21_GPU_SHIM_CACHE")
        or cache_home / "21cmfast-gpu" / "native-shims"
    )
    artifact = cache / (key + ".so")
    checksum = cache / (key + ".sha256")
    try:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (cache / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (
                artifact.is_file()
                and checksum.is_file()
                and _digest(artifact).encode() == checksum.read_bytes()
            ):
                shutil.copyfile(artifact, private)
            else:
                compile_private()
                with tempfile.TemporaryDirectory(dir=cache) as stage:
                    staged = Path(stage) / name
                    shutil.copyfile(private, staged)
                    staged_hash = Path(stage) / "checksum"
                    staged_hash.write_text(_digest(staged))
                    os.replace(staged, artifact)
                    os.replace(staged_hash, checksum)
    except OSError as exc:
        logging.getLogger(__name__).warning("Native shim cache unavailable: %s", exc)
        compile_private()
    return ctypes.CDLL(str(private))
