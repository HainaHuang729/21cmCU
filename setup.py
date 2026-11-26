#!/usr/bin/env python
"""Setup script for 21cmCU (native C/CUDA backend + py21cmfast compatibility)."""

from pathlib import Path

from setuptools import find_packages, setup

THISDIR = Path(__file__).parent.resolve()

setup(
    name="21cmCU",
    version="0.2.0",
    license="MIT license",
    description="Native CUDA/cuFFT 21cm cosmological simulation engine",
    long_description=(THISDIR / "README.md").read_text(encoding="utf8"),
    long_description_content_type="text/markdown",
    url="https://github.com/HainaHuang729/21cmFAST-GPU",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.12",
    zip_safe=False,
    setup_requires=["cffi>=1.0"],
    entry_points={
        "console_scripts": [
            "21cmcu = py21cmcu.cli:app",
            "21cmfast = py21cmfast.cli:app",
        ]
    },
    cffi_modules=[f"{THISDIR}/build_cffi.py:ffi"],
)
