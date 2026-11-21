# 21cmFAST-GPU (`bt` branch)

CUDA acceleration for 21cmFAST v3 (based on commit [aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273)) with the broken-power-law primordial spectrum (`POWER_SPECTRUM=6`, `KP_h_Mpc`, `MS`). Lightcone outputs across all 6 fields match CPU runs within `1e-8 + 1e-4 * |CPU|`. Upstream license: [LICENSE.upstream](LICENSE.upstream).

For standard LCDM (`POWER_SPECTRUM=0` without `KP`/`MS` extensions), use the [main](https://github.com/HainaHuang729/21cmFAST-GPU/tree/main) branch.

## Setup

```bash
git clone -b bt https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
conda env create -f environment.yml
conda activate 21cmfast-gpu
pip install cupy-cuda12x   # or cupy-cuda11x for CUDA 11.x

python lightcone/run_bt_lightcone.py --out runs/bt_out --work runs/bt_work --kp-h-mpc 10.0 --ms 2.5 --seed 725213656658 --dim 384 --hii-dim 96 --box-len 375.0 --max-z 14.0 --ic-threads 16
```

[lightcone/run_bt_lightcone.py](lightcone/run_bt_lightcone.py) locates GSL, `fftw3f`, and CUDA automatically, compiles the CFFI library, converts `KP_h_Mpc` to `KP = KP_h_Mpc * hlittle` via [lightcone/bt_config.py](lightcone/bt_config.py), and matches 16-thread GSL ICs when `--ic-threads 16` is set; run [lightcone/test_bt_config.py](lightcone/test_bt_config.py) to check parameter parsing.

## Benchmarks

Host: AMD Ryzen Threadripper PRO 5975WX (32 cores) + 1x RTX A6000 (GPU runs use 8 CPU threads):

| Grid / Box / Slices | CPU 8T | CPU 16T | CPU 32T | GPU (`phys`) | Speedup (8T / 16T / 32T) | Failed |
|---|---:|---:|---:|---:|:---:|:---:|
| `256 -> 64`, 200 Mpc, 1227 | 107.93 s | 72.99 s | 71.18 s | 13.98 s (9.59 s) | 7.72x / 5.22x / 5.09x | 0 / 6 |
| `384 -> 96`, 300 Mpc, 761 | 266.33 s | - | - | 23.29 s (18.20 s) | 11.44x | 0 / 6 |
| `384 -> 96`, 375 Mpc, 609 | 278.18 s | 163.98 s | 160.13 s | 21.02 s (16.21 s) | 13.23x / 7.80x / 7.62x | 0 / 6 |
| `1024 -> 256`, 250 Mpc, 3922 | 6462.85 s | - | - | 371.37 s (336.33 s) | 17.40x | 0 / 6 |

Implementation notes, unit tests, and 5 additional parameter spot-checks are in [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md) and [reports/wave6-lightcone/summary.json](reports/wave6-lightcone/summary.json).

## Roadmap

- Port the CFFI/CUDA hooks to 21cmFAST v4 (`v4.3+`) (`InputParameters`, `HALO_STOCHASTICITY`, Lyman-alpha multiple scattering, native CLASS transfer functions).
