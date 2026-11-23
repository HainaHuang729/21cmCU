# FFTW codelets for CUDA

The catalog contains generated scalar and SSE2 arithmetic from FFTW 3.3.10. The original source paths and copyright notices are stored with the codelets. The CUDA lane definitions adapt the non-FMA operations to one line per CUDA thread. The FFTW routines are distributed under GPL-2.0-or-later. See [COPYING](COPYING).

To regenerate the catalog from the unpacked [upstream release](https://www.fftw.org/fftw-3.3.10.tar.gz), run `python build_catalog.py /path/to/fftw-3.3.10` in this directory.
