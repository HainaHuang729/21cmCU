#!/usr/bin/env python
"""Minimal setuptools shim wiring the CFFI native extension to pyproject.toml."""

from pathlib import Path

from setuptools import setup

THISDIR = Path(__file__).parent.resolve()
setup(cffi_modules=[f"{THISDIR}/build_cffi.py:ffi"])
