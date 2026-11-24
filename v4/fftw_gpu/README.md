# Running FFTW plan arithmetic on CUDA

The v4.3.2 perturbation adapter obtains the single-precision in-place cubic R2C and C2R plans from the FFTW library linked into the CPU reference. It requests a clean ESTIMATE plan in a child process with the simulation thread count. The plan tree selects the CUDA codelets, their order and the twiddle factors. Each CUDA thread transforms a line. CuPy compiles the three axis kernels with fused multiply-add disabled and caches the executable per device, length, direction and thread count. Inverse output follows FFTW's unnormalized convention.

The compiler handles scalar and SSE2 codelets, mixed-radix butterflies and several prime-length algorithms. A plan with unsupported arithmetic or heterogeneous worker schedules raises `UnsupportedPlan`. The perturbation adapter normally selects double-precision cuFFT for both directions of an unsupported size. Set `strict_fft=True` on `perturb_field_gpu` to report an unsupported plan or external wisdom as an error. `reference_fft=True` selects separate CPU reference operations in the adapter. The standalone compiler has no cuFFT fallback. For a captured plan from the same cubic layout and direction, pass its printed tree to `transform(array, n, inverse, plan_tree=tree)`. The CUDA executable cache also distinguishes trees for the same length.

In the benchmark environment with CuPy, 21cmFAST and its upstream tests on PYTHONPATH, run:

```sh
python v4/inspect_fft_plans.py 96 192 --threads 2
python v4/test_plan_cuda.py verification.json 96 192
```

The second command compares each generated forward transform to the CPU result from the same real input. It then compares both inverse transforms on the CPU spectrum, records maximum errors and unequal-value counts, and exits unsuccessfully on any discrepancy. This checks two-thread FFTW plans. Other builds, planning modes, data layouts, precisions and thread counts can select different arithmetic. The older length-specific kernels remain in ordered_fft.py for diagnostic comparisons.

Regenerate the codelet catalog with `python v4/fftw_gpu/build_catalog.py /path/to/fftw-3.3.10`. The catalog retains the upstream source paths and license notices. The original FFTW copyright and GPL-2.0-or-later apply to the adapted routines. See [COPYING](COPYING) and the [upstream release](https://www.fftw.org/fftw-3.3.10.tar.gz).
