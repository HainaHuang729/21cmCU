# 21cmFAST-GPU

21cmFAST 的 CUDA / 混合精度保真 GPU 加速实现与严格回归验收归档。基于 [21cmFAST aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273)（含扩展折线原初功率谱 `POWER_SPECTRUM=6`、`KP`、`MS` 支持），通过非侵入式构建期补丁与 C 函数指针钩子运行，上游授权见 [LICENSE.upstream](LICENSE.upstream)。

仓库包含两条已验证管线：
1. **Wave 6 全流程 6 场光锥版本（[lightcone/](lightcone)）** — 最新主线：全部 4 个标准基准（最高 `1024³ -> 256³`，3,922 切片）+ 5 组跨宇宙学/天体物理参数/多种子抽检均实现 **`0 / 6` 格超差**（`abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)`）与冷暖启动 **`repeat_equal = true`（`max_diff = 0.0`）**，生产尺度 **`17.40×`** 加速（`6,462.85 s -> 371.37 s`）。
2. **Coeval + 功率谱归档版本（[gpu/](gpu)）** — 2026-11-19 验证版：完整 `512³ -> 128³` 32 红移 coeval 归档与并行 LZF 落盘（`85.69 s`）。

---

## Wave 6 GPU 光锥 vs 纯 CPU 基准摘要

> 完整模块架构、CUDA 内核分工及数值精度边界分析请见：[docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md)。

### 核心架构区别

| 模块 | 纯 CPU 版本 | Wave 6 GPU 光锥版本（[lightcone/](lightcone)） |
|---|---|---|
| **代码形态** | `resident_filter_20261118/fix/package/py21cmfast/src`（原始 C/Python） | 原代码保持未改；[lightcone/build.py](lightcone/build.py) 编译私有 `c_21cmfast` 扩展并注册 CUDA 回调钩子 |
| **扰动（2LPT + CIC）** | CPU OpenMP + 3D FFTW | CUDA `cic_pertile` + `cic_repair_exact`（[perturb.cu](lightcone/perturb.cu)）；`PS_FFTW_PERTURB_DENSITY=1` 保留 CPU FFTW 物化密度与 $v_z$ |
| **热力学演化（$T_k, x_e, T_s$）** | CPU 多尺度 FFTW 滤波 + GSL 表 + OpenMP ODE | 后台多线程预取 Padded FFTW + CUDA 原地解包；`lookup_sum_batch` / `accum_batch` / `update_stabilized.cu`（`--fmad=false` + [glibc_math.cuh](lightcone/glibc_math.cuh)）GPU 常驻演化 |
| **电离与复合（Ionization）** | 全尺度 3D FFTW + 逐格循环与 MHR 复合 | CUDA 筛选跳跃（`screen_ion_scale_pass1`）+ 守卫回放（`PS_GUARD_EPS_MULT=32`、`PS_XEREC_AMP_MULT=0`）+ 锐截断裁剪 3D FFTW `c2r`（[ion_hook.h](lightcone/ion_hook.h)）+ CUDA 坍缩/电离/复合核（[ion.cu](lightcone/ion.cu)） |
| **21 cm 亮温与光锥插值** | CPU 3D FFTW $\partial_z v_z$ + Astropy Python 切片循环 | `PS_FFTW_BRIGHTNESS=1`（精确 `ComputeBrightnessFastFFTW` 消除 $T_s \approx T_{\rm CMB}$ 过零超差）+ C/OpenMP `PSFillLightconeSlices` 批量插值 + 并行 LZF 压缩 |

### 全部 9 组端到端光锥验收结果（RTX A6000 单卡 + 8 核 CPU，`0 / 6` 超差，`repeat = 0.0`）

#### A. 四大标准基准（[summary.json](reports/wave6-lightcone/summary.json)）

| 基准 | 网格 / 盒子 / 切片 | CPU 墙钟 | **Wave 6 GPU 暖启动 (`phys`)** | 加速比 | 6 场超差 / 复现 | 报告链接 |
|---|---|---:|---:|:---:|:---:|---|
| `384/96` | `384 -> 96`, 375 Mpc, 609 | `278.18 s` | **`21.02 s`** (`16.21 s`) | **`13.23×`** | `0 / 6`, `0.0` | [w4_units.json](reports/wave6-lightcone/results-489606/w4_units.json) |
| `h1_256_64` | `256 -> 64`, 200 Mpc, 1,227 | `107.93 s` | **`13.98 s`** (`9.59 s`) | **`7.72×`** | `0 / 6`, `0.0` | [h1 comparison](reports/wave6-lightcone/results-489605/warm-h1_256_64-comparison.json) |
| `h2_384_96` | `384 -> 96`, 300 Mpc, 761 | `266.33 s` | **`23.29 s`** (`18.20 s`) | **`11.44×`** | `0 / 6`, `0.0` | [h2 comparison](reports/wave6-lightcone/results-489605/warm-h2_384_96-comparison.json) |
| **`1024/256`** | `1024 -> 256`, 250 Mpc, 3,922 | `6,462.85 s` | **`371.37 s`** (`336.33 s`) | **`17.40×`** | `0 / 6`, `0.0` | [warm-1024-timing.json](reports/wave6-lightcone/results-489607/warm-1024-timing.json) |

#### B. 5 组多参数 / 多种子 / 多宇宙学抽检（Job `489621`）

| 样点 | 物理与宇宙学区间 | 网格 / 盒子 / 切片 / Seed | CPU 墙钟 | **Wave 6 GPU 暖启动 (`phys`)** | 加速比 | 报告 |
|---|---|---|---:|---:|:---:|---|
| `p1_early_eor_strong_xray` | 早电离 + 强软 X 射线 | `256->64`, 200 Mpc, 1,227, `111111` | `111.69 s` | **`13.24 s`** (`8.83 s`) | **`8.44×`** | [p1](reports/wave6-lightcone/results-489621/warm-p1_early_eor_strong_xray-comparison.json) |
| `p2_late_eor_weak_hard_xray` | 晚电离 + 弱硬 X 射线 | `256->64`, 250 Mpc, 982, `222222` | `96.61 s` | **`11.82 s`** (`7.94 s`) | **`8.17×`** | [p2](reports/wave6-lightcone/results-489621/warm-p2_late_eor_weak_hard_xray-comparison.json) |
| `p3_cosmo_shift_lcdm_low_kp` | ΛCDM 偏移 + 低 `KP`（`1.33`） | `256->64`, 180 Mpc, 1,353, `333333` | `114.76 s` | **`13.79 s`** (`9.14 s`) | **`8.32×`** | [p3](reports/wave6-lightcone/results-489621/warm-p3_cosmo_shift_lcdm_low_kp-comparison.json) |
| `p4_cosmo_shift_low_s8` | 低 `SIGMA_8`（`0.785`）+ 强产星 | `256->64`, 220 Mpc, 1,113, `444444` | `101.19 s` | **`13.34 s`** (`9.17 s`) | **`7.59×`** | [p4](reports/wave6-lightcone/results-489621/warm-p4_cosmo_shift_low_s8-comparison.json) |
| `p5_384_96_mid_eor_cosmo` | 高分辨率 EoR + 宇宙学混合偏移 | `384->96`, 300 Mpc, 839, `555555` | `266.65 s` | **`21.59 s`** (`16.43 s`) | **`12.35×`** | [p5](reports/wave6-lightcone/results-489621/warm-p5_384_96_mid_eor_cosmo-comparison.json) |

> 9 项底层单元测试（包括 Wave 6 裁剪 3D FFTW `c2r` 7 半径测试）全部 `bitwise_equal = true`（0 比特差异）；Holdout 亮温 GPU 退化回归证据存档于 [489602 failures](reports/wave6-lightcone/results-489602/warm-h1_256_64-failures.json)。

---

## 目录结构

- [lightcone/](lightcone)：Wave 6 完整 GPU 光锥源码、CUDA 核、C 补丁器与验收脚本（60 个文件）
  - [lightcone/build.py](lightcone/build.py)：构建并编译私有 `c_21cmfast` CFFI 扩展模块
  - [lightcone/run_large_lightcone.py](lightcone/run_large_lightcone.py)：CPU / GPU 统一 6 场光锥入口（支持 `PS_ASTRO_JSON`、`PS_COSMO_JSON`、`PS_SEED`、`PS_DIM`/`PS_HII_DIM`/`PS_BOX_LEN`）
  - [lightcone/phase3d_holdout_verify.sbatch](lightcone/phase3d_holdout_verify.sbatch)、[lightcone/phase3e_multiparam_spotcheck.sbatch](lightcone/phase3e_multiparam_spotcheck.sbatch)：完全自包含的现场生成 IC + CPU/GPU 端到端验收脚本（无预缓存光锥依赖）
  - [lightcone/phase3c_w4_verify.sbatch](lightcone/phase3c_w4_verify.sbatch)、[lightcone/phase3d_1024_verify.sbatch](lightcone/phase3d_1024_verify.sbatch)：分别使用集群 `/var/tmp` 预缓存 `384/96` 与 `1024/256` CPU 基准的回归/Slurm 归档脚本
- [gpu/](gpu)：2026-11-19 验证的 Coeval + 功率谱全流程入口（[gpu/launch_ps.py](gpu/launch_ps.py)、[gpu/integrated-lzf.sbatch](gpu/integrated-lzf.sbatch)）
- `resident_filter_20261118/fix/package/py21cmfast`：未修改的基础 C/Python 21cmFAST 源码树
- [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md)：GPU 与 CPU 版本的逐模块架构、算法与精度对照文档
- [reports/](reports)：包含 [reports/wave6-lightcone/](reports/wave6-lightcone/summary.json)（Wave 6 光锥全部验收报告）以及早期 coeval/batch4 报告
- `point_config.json`、`reference.npz`、`LICENSE.upstream`

---

## 快速上手（Wave 6 GPU 光锥）

在 Linux CUDA 节点上准备 Python 3.10+（NumPy、SciPy、Astropy、h5py、cffi、CuPy、PyYAML）、OpenMP C 编译器、GSL 与单精度 FFTW3（`libfftw3f`, `libfftw3f_omp`）。在仓库根目录准备好 `include/`（含 `gsl/`、`fftw3.h`）与 `lib/`（含 `libgsl`、`libfftw3f`、`libfftw3f_omp`），或软链至系统/conda 依赖目录。

### 1. 编译私有 `c_21cmfast` 扩展并生成 GK61 求积规则

```bash
export PS_GPU_ROOT="$PWD"
export LD_LIBRARY_PATH="$PS_GPU_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_PATH=/path/to/cuda
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p build_work
python lightcone/build.py "$PS_GPU_ROOT" "$PWD/build_work"
export PYTHONPATH="$PWD/build_work/package:$PWD/lightcone"

export PS_GK61_RULE="$PWD/build_work/gk61.json"
python lightcone/gk61_rule.py "$PS_GK61_RULE"
```

### 2. 启用 Wave 6 保真加速环境变量运行光锥

先使用纯 CPU 约定生成匹配网格与宇宙学的初始条件 HDF5（或指定已有 `InitialConditions*.h5`），再运行 `gpu-density` 模式：

```bash
export PS_FFTW_PERTURB_DENSITY=1 PS_FFTW_HEAT_FILTER=1 PS_GSL_EXACT_POWER=1
export PS_CONDITIONAL_MODE=3 PS_BATCH_FILTER=1
export PS_XRAY_NATIVE_MODE=4 PS_NATIVE_SIGMA=1 PS_NATIVE_MHR=1 PS_NATIVE_COLLAPSE=1 PS_NATIVE_XRAY=0
export PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0
export PS_CONDITIONAL_FULL_HIGH=0 PS_CONDITIONAL_NAN_SENTINEL=1 PS_GUARD_EPS_MULT=32
export PS_LAZY_COEVAL_IC=1 PS_PARALLEL_LZF=1 PS_VERBOSE_HEAT=0
export PS_REUSE_PREV_PERTURB=1 PS_FAST_LIGHTCONE=1 PS_CIC_PERTILE=1
export PS_OMP_MHR=1 PS_CACHE_INIT_PS=1 PS_ASYNC_FFTW_PREFETCH=1
export PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0

export PS_DIM=384 PS_HII_DIM=96 PS_BOX_LEN=375.0 PS_MAX_Z=14.0 PS_WINDOW_MAX_Z=14.0
export PS_CONTROL_IC=/path/to/InitialConditions.h5
# 可选覆盖宇宙学或天体物理参数：
# export PS_COSMO_JSON='{"OMm": 0.308, "SIGMA_8": 0.815}'
# export PS_ASTRO_JSON='{"F_STAR10": -1.06, "L_X": 39.25}'

python lightcone/run_large_lightcone.py gpu-density out_gpu work_gpu
```

> 注：[phase3d_holdout_verify.sbatch](lightcone/phase3d_holdout_verify.sbatch) 与 [phase3e_multiparam_spotcheck.sbatch](lightcone/phase3e_multiparam_spotcheck.sbatch) 为现场自生成 IC 与 CPU 基线的完整 Slurm 验收模板；在不同集群复用 `.sbatch` 时请按需修改 `#SBATCH` 分区/节点及 Python/依赖路径。

---

## Coeval + 功率谱归档配置（`gpu/`）

- 固定点 80000，seed `1850877074`，盒边长 `250 Mpc`，IC 网格 `512³`，演化网格 `128³`。
- RTX A6000 单卡及 8 个主機执行绪。完整输出包含 14 个 IC 栏位、32 个红移的 416 个演化场、全局历史与两种功率谱。
- 暖机后从启动到共享 NFS 同步落盘，基准 `98.51 s`，双档并行压缩 `85.69 s`，生产 32-bin PS `0/1024` 超差（[coeval-retest.json](reports/coeval-retest.json) · [paired-output.json](reports/paired-output.json) · [parallel-lzf.json](reports/parallel-lzf.json)）。

```sh
python gpu/launch_ps.py runs/point-80000 --cache runs/cache \
  --seed --full-coeval --include-ic --gpu-full-ic \
  --compression lzf --coeval-lzf-workers 4 --paired-archive
```

较早的 2 场矩形光锥并行吞吐测试（单卡四并发 `99.83 s`，约 `144` 条/小时）亦归档于 [lightcone-batch4.json](reports/lightcone-batch4.json) · [lightcone-comparison.json](reports/lightcone-comparison.json)。
