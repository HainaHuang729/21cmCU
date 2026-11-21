# Wave 6 GPU 光锥版本与纯 CPU 参考版本技术对比

本文档详细对照当前主线 **Wave 6 GPU 光锥版本**（[lightcone/](../lightcone)）与 **纯 CPU 参考版本**（[resident_filter_20261118/fix/package/py21cmfast/src](../resident_filter_20261118/fix/package/py21cmfast/src)，基于上游 [21cmFAST aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273) 并包含本研究的 `POWER_SPECTRUM=6` 折线原初功率谱参数 `KP` / `MS` 等扩展）。

---

## 1. 核心设计原则与非侵入式架构

CPU 源码树在仓库中的 `resident_filter_20261118/fix/package/py21cmfast/` 保持完全原封不动。GPU 移植不直接分叉覆盖上游 `.c`/`.py` 主代码，而是通过**构建期补丁 + 运行期 C 函数指针回调钩子（Hook）**实现：

1. **构建期代码注入（[build.py](../lightcone/build.py)）**：
   - 将基础包复制到临时构建目录并卸载旧 `.so` 后，依次运行补丁器改造：
     - Python 层：`outputs.py`（`PS_LAZY_COEVAL_IC`）、`wrapper.py`（`PS_REUSE_PREV_PERTURB`、光锥循环前后异步调度回调 `_ps_set_lightcone_schedule`）、`lightcones.py`（向量化切片选择、`fill_lightcone_slices` 批量填充）。
     - C 物理核心：`PerturbField.c`（`SetPerturbGPUHook`、多线程内存拷贝 `PSParallelMemcpy`、光锥填充核 `PSFillLightconeSlices`、共享 FFTW 计划生命周期 `SetPSKeepFFTWPlans`）、`SpinTemperatureBox.c`（`SetGPUFilterHook`、`SetGPUHeatHook`、`HeatReferenceBatchDensityPreplannedPadded`）、`IonisationBox.c`（[ion_patch.py](../lightcone/ion_patch.py) + [ion_hook.h](../lightcone/ion_hook.h)，注入 `SetIonGPUHook` 及裁剪 3D FFTW `c2r`）、`BrightnessTemperatureBox.c`（[brightness_patch.py](../lightcone/brightness_patch.py)，注入 `SetBrightnessGPUHook` 与 `ComputeBrightnessFastFFTW`）、`GenerateICs.c`（[ic_patch.py](../lightcone/ic_patch.py)、[ic_seed_patch.py](../lightcone/ic_seed_patch.py)）、`ps.c`（[conditional_patch.py](../lightcone/conditional_patch.py)、[sigma_patch.py](../lightcone/sigma_patch.py)、[hmf_patch.py](../lightcone/hmf_patch.py)）、`recombinations.c`（[mhr_patch.py](../lightcone/mhr_patch.py)）、`heating_helper_progs.c`（[heat_table_cache_patch.py](../lightcone/heat_table_cache_patch.py)）。
   - 使用严格的 IEEE 浮点编译选项编译私有 CFFI 扩展 `py21cmfast.c_21cmfast`：
     ```text
     -fopenmp -O3 -mavx2 -ffp-contract=off -fno-fast-math -fno-finite-math-only
     ```
2. **总控入口（[coupled.py](../lightcone/coupled.py) & [run_large_lightcone.py](../lightcone/run_large_lightcone.py)）**：
   - 将上述物理阶段的 GPU/CUDA 后端（CuPy `RawModule`/`RawKernel` 编译 `.cu` 核）注册进 C 扩展的函数指针钩子中，形成 **GPU 驻留常驻计算 + 关键敏感 FFTW 路径受控保留/回放 CPU** 的混合架构。
   - **纯 CPU 回退开关**：只要通过环境变量关闭所有 GPU 钩子或以 `mode=cpu` 运行 [run_large_lightcone.py](../lightcone/run_large_lightcone.py)，同一份编译产物即刻退化为严格纯 CPU 行为，确保同环境可逐格对照。

---

## 2. 逐阶段执行路径对比（GPU 版 vs CPU 版）

下表列出完整光锥生成链条 7 个主要阶段的执行归属、核心实现与精度保留策略：

| 计算阶段 | 纯 CPU 参考实现 | Wave 6 GPU 版实现（[lightcone/](../lightcone)） | 关键差异与保留 CPU 路径原因 |
|---|---|---|---|
| **1. 初始条件（IC）与辅助积分表** | 单/多核 GSL `power_in_k` + 3D FFTW `r2c/c2r` + GSL QAG 计算 $\sigma(M)$、HMF、MHR 复合表 | 默认验收沿用同一份 GSL 缓存 IC（避免 RNG 流差异引入跨样本方差）；提供 [sigma_runtime.py](../lightcone/sigma_runtime.py)（GPU 上仍受 `PS_NATIVE_SIGMA=1` 控制默认用原 C 表）、[conditional_runtime.py](../lightcone/conditional_runtime.py)（`conditional_gk61.cu` GPU 条件坍缩表）、[xray_runtime.py](../lightcone/xray_runtime.py)（GPU X 射线频率积分 + 边界回退 CPU `integrate_over_nu`）、[mhr_patch.py](../lightcone/mhr_patch.py)（OpenMP 并行 `PS_OMP_MHR=1`）、`PS_CACHE_INIT_PS=1`、`PS_GSL_EXACT_POWER=1` | 条件坍缩表中 `CRIT_DENS_TRANSITION` 附近采用混合精度策略与原生高表补齐；`init_MHR` 通过 OpenMP 并行 GSL QAG 保持位级一致；`PS_CACHE_INIT_PS=1` 避免同宇宙学参数下重复 `init_ps()` 计算。 |
| **2. 密度场与速度场演化（2LPT + CIC）** | `PerturbField.c` 在 CPU 上按 2LPT 位移分格投影并对全盒做 3D FFTW 滤波平滑 + 求 $v_z$ | **GPU（[gpu_perturb.py](../lightcone/gpu_perturb.py)、[perturb.cu](../lightcone/perturb.cu)）**：使用 `cic_pertile`（`DIM == 4 * HII_DIM` 时按粗格分块减少原子冲突，否则回退 `cic`）+ `cic_repair_exact` 在 GPU 上完成原子精确定点/浮点和与 2LPT 位移；当启用 `PS_FFTW_PERTURB_DENSITY=1`（默认）时返回状态码 `4`，回传归一化后密度对比度并由 CPU 缓存计划的 `MaterializePerturbDensityAndVel` 执行 3D FFTW `r2c/c2r` 及 $v_z$ 反演 | **为什么保持 FFTW 密度物化**：将 CIC 主计算完全卸载到 GPU 获得高加速，同时把 `Perturb` 输出密度与 $v_z$ 的最后一步 3D FFTW 舍入锁定在 CPU `FFTW_ESTIMATE` 路径上，杜绝下游非线性阈值微差放大。 |
| **3. 热力学与自旋温度演化（$T_k, x_e, T_s$）** | `SpinTemperatureBox.c` 在 CPU 循环各滤波半径 `NR` 并做 3D FFTW 滤波、对数插值表求平均 $f_{\rm coll}$、逐格求解电离/热速率 ODE | **CPU 异步预取 FFTW + GPU 常驻更新（[unguarded_resident.py](../lightcone/unguarded_resident.py)、[gpu_heat_runtime.py](../lightcone/gpu_heat_runtime.py)、[update_stabilized.cu](../lightcone/update_stabilized.cu)）**：<br>① 过滤滤波：`PS_FFTW_HEAT_FILTER=1` + `PS_ASYNC_FFTW_PREFETCH=1` + `PS_HEAT_GPU_UNPACK=1`，用后台线程提前算好下一红移的缓存 `HeatReferenceBatchDensityPreplannedPadded`，上传至 GPU 后由 CUDA 核 `unpack_extrapolate_bounds_pass1/2` 原地解包外推并归约最大最小值；<br>② 坍缩与 X 射线：`lookup_sum_batch`、`finish_sum_batch`、`accum_batch` 在 GPU 常驻显存批量完成 40 尺度求值；<br>③ 状态步进：`update_stabilized.cu` 借助 `--fmad=false` 与 `glibc_math.cuh`（精确匹配 glibc `log10f/expf/fma`）全盒推进，历史状态常驻显存供 `SetGPUPreviousHeatMeans` 秒级归约 | 在 40 个平滑尺度并行下直接用 GPU CUFFT 会使大量极低过密区或线性插值边界出现 `1e-7` 级舍入偏移，触发局部 $T_k/x_e$ 与光深积分差异；采用“后台多线程预规划 FFTW + GPU 解包与 ODE 核常驻”既隐藏 CPU 耗时又实现零超差。 |
| **4. 激波/气泡电离与非均匀复合（Ionization）** | `IonisationBox.c` 从最大过滤半径 `R` 向最小 `R` 步进：对 $\delta, x_e, \mathrm{d}N_{\rm rec}$ 执行 3D FFTW 球对称顶帽滤波、光子数预算比较（游离判据 `ion_eff * f_coll >= 1 - x_e + dNrec`）、最后更新 $\Gamma_{12}, T_{\rm ion}, \mathrm{d}N_{\rm rec}$ | **混合守卫筛选 + 裁剪 FFTW + GPU 物理核（[ion_runtime.py](../lightcone/ion_runtime.py)、[ion.cu](../lightcone/ion.cu)、[ion_hook.h](../lightcone/ion_hook.h)）**：<br>① `gpu-density`（`PS_ION_SCREENED_DENSITY=1`）：CPU 批量/异步预取全尺度精确 FFTW 密度场；当 GPU 初步筛选确认该尺度没有任何格点接近游离判定阈值（`screen_ion_scale_pass1`，`ratio < 0.99`）且 `last=0` 时直接跳过（`screened_noop`）；<br>② 仅在活跃尺度或最后一步回放 GPU/CPU FFTW：非最终步由 `filter_pair`/`bubble_filter_pair` 在 GPU 并行滤波 $x_e$ 与 $\mathrm{d}N_{\rm rec}$，同时由**守卫内核** `min_ion_margin_pass1/2` 计算距游离阈值最小安全裕度，当 `distance <= 32 * eps`（`PS_XEREC_AMP_MULT=0`、`PS_GUARD_EPS_MULT=32`）或达到最后尺度时，触发精确 `IonReferenceXeRecBoxOpt` CPU FFTW 回放；<br>③ **Wave 6 裁剪 3D FFTW `c2r`**：利用 `HII_FILTER=1` 锐 k 空间截断谱外全零特性，仅在有效 $(k_x, k_y, k_z)$ 子体积执行 1D 批处理 FFTW 并融合原地限幅解包（实测与全盒 `c2r` 逐比特 0 差异）。<br>④ 坍缩率 `collapse_with_sum`、电离判定 `assign_ion`、MHR 复合积分 `recombine`、Poisson 随机数 [ion_poisson.cu](../lightcone/ion_poisson.cu) 及最终输出回传全部在 GPU 完成 | 电离判据 `>=` 是硬阶跃函数，极微小的 $x_e$、$\mathrm{d}N_{\rm rec}$ 或 $\delta$ 浮点舍入就会导致个别边界格点提前/延后一步电离并产生 $O(1)$ 级 `xH` / 复合历史分歧。**阈值守卫 + 筛选跳跃 + 锐截断裁剪 FFTW** 兼顾速度（跳过大尺度冗余，大幅缩小有效 FFT 维度）与严格的零边界翻转。 |
| **5. 21 cm 亮温（Brightness Temperature）** | `BrightnessTemperatureBox.c` 在 CPU 上计算速度梯度 $\partial_z v_z$（3D FFTW `r2c/c2r`）并逐格计算 $T_b$（含 RSD 限制） | **默认走位级保真快速 FFTW 路径（`PS_FFTW_BRIGHTNESS=1` -> `ComputeBrightnessFastFFTW`）**，复用了共享的 `ps_perturb_fftw_box` 并用 OpenMP SIMD 归约完成全盒计算；CUDA 内核版本 [brightness.cu](../lightcone/brightness.cu) 仅保留在 `PS_FFTW_BRIGHTNESS=0` 可选开关下 | **为什么放弃 `PS_FFTW_BRIGHTNESS=0`**：在 `results-489602` Holdout 回归测试中开启纯 GPU 亮温核，其余 5 场全为 `0` 超差，但 `h1_256_64` 检出 `17` 格、`h2_384_96` 检出 `4` 格 `brightness_temp` 超差——原因在于 $T_s \approx T_{\rm CMB}$ 零交叉点附近亮温插值相互抵消至 $\sim 10^{-4}\text{ mK}$，cuFFT FP32 $\partial_z v_z$ 与 CPU `FFTW_ESTIMATE` 的 $10^{-7}$ 级微小差异即突破相对容差 `1e-4`。保留 `PS_FFTW_BRIGHTNESS=1` 仅增加 ~`0.08 s` 净耗时即可完全清零超差。 |
| **6. 光锥插值切片构建（Lightcone）** | `wrapper.py` / `lightcones.py` 逐切片用 Astropy `Quantity` 对象切片并由 Python 循环逐个调用 `redshift_interpolation` | **标量索引缓存 + C/OpenMP 批量填充（[build.py](../lightcone/build.py) `PSFillLightconeSlices` + `PS_FAST_LIGHTCONE=1`）**：预抽取纯 `float64` 共动距离与像素权重，按 `float32` 运算顺序在 C/OpenMP `collapse(2)` 循环内批量写入全部 6 场光锥切片 | 单测 `W5_lightcone_fill` 实测与原生 Astropy/NumPy 实现 **逐比特完全相等**（`max_abs = 0.0`），同时消除了数千个光锥切片的 Python 调度瓶颈。 |
| **7. HDF5 输出与压缩落盘** | 单线程 `h5py` + `lzf` + `shuffle` 写盘 | **OpenMP 有限数检查 `PSArrayAllFinite` + 多线程并行 LZF 压缩直写（[parallel_lzf.py](../lightcone/parallel_lzf.py)，`PS_PARALLEL_LZF=1`）** 并调用 `os.fsync` 落盘 | 在 `1024/256`（写 6 场共 3,922 切片约 `3.45 GB` 原始数据）场景下将 `write_lzf` 压缩耗时压到 `11.24 s`、`fsync` `5.00 s`。 |

---

## 3. 数值保真与位级验证机制

1. **防 FMA 融合与 libc 数学库对齐**：全部关键 `.cu` 内核强制 `--fmad=false`，并引入 [glibc_math.cuh](../lightcone/glibc_math.cuh) 严格对齐 x86-64 glibc 的单精度 `log10f`/`expf` 与主机 FMA 状态。
2. **确定性与可复现性（Determinism）**：冷启动（含 CuPy JIT 编译首跑 `prime`）与暖启动（`warm`）输出完全一致。全部 9 组端到端对比均通过 `repeat_equal = true`（6 场最大复跑差 `max_diff = 0.0`）。
3. **单元级 0 比特差异门禁（[results-489606/w4_units.json](../reports/wave6-lightcone/results-489606/w4_units.json)）**：

| 单元测试项 | 验证内容 | 实测最大差异 (`max_abs`) | `bitwise_equal` | 说明 / 加速 |
|---|---|---:|:---:|---|
| `W4_cic_pertile_v2` | CIC 分块原子核 vs 原始原子核 | `0.0` | `true` | `max_ulp = 0`，`3.37×` |
| `W4_init_mhr_omp` | OpenMP 并行 `init_MHR` 复合表 vs 串行 GSL | `0.0` | `true` | `5.19×`（1641.8 ms → 316.3 ms） |
| `W4_heat_preplanned` | 预规划 FFTW 热滤波批次结果与最值 | `0.0` | `true` | 7/8 线程均 `0.0` |
| `W4_ion_prefetch` | 异步预取电离全尺度密度批次 | `0.0` | `true` | 32 尺度无偏差 |
| `W5_heat_padded_gpu_unpack` | GPU Padded FFTW 原地解包外推 | `0.0` | `true` | 消除宿主端解包开销 |
| `W5_ion_filter_sparse` | 稀疏频谱复制锐截断滤波器（4 半径 + 无滤波） | `0` bit diffs | `true` | 50/10/3/1 Mpc 全 0 bit 差 |
| `W5_lightcone_fill` | `PSFillLightconeSlices` 批量插值 vs 原生循环 | `0.0` | `true` | `last_ref == last_fast == 26` |
| `W5_heat_update_opt` | 热更新优化核速率与状态输出 | `0.0` | `true` | `bad_u1 = 0` |
| `W6_ion_pruned_c2r` | 锐截断裁剪 3D FFTW `c2r`（7 个半径 `50`–`2 Mpc`） | `0.0` | `true` | 7 半径全 `0` bit 差 |

---

## 4. 全量端到端性能与精度基准（Wave 6 vs 同配置 8 核纯 CPU）

统一运行环境：单张 `NVIDIA RTX A6000 (48 GB)` + `8` 个主机 CPU 核心（`OMP_NUM_THREADS=8`）。严格回归判据：`abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)`。

### 4.1 四大标准基准（[summary.json](../reports/wave6-lightcone/summary.json)）

| 基准名 | 网格 (`DIM->HII_DIM`) / 盒子 / 切片 | 纯 CPU 墙钟 | **Wave 6 GPU 暖启动 (`phys` / `sync`)** | 加速比 | 6 场超差数 | 冷/暖复现差 | 验收报告 |
|---|---|---:|---:|:---:|:---:|:---:|---|
| `384/96` | `384 -> 96`, 375 Mpc, 609 切片 | `278.18 s` | **`21.02 s`** (`16.21 s` / `18.67 s`) | **`13.23×`** | **`0 / 6`** | `0.0` | [results-489606](../reports/wave6-lightcone/results-489606/w4_units.json) |
| `h1_256_64` | `256 -> 64`, 200 Mpc, 1,227 切片 | `107.93 s` | **`13.98 s`** (`9.59 s` / `12.42 s`) | **`7.72×`** | **`0 / 6`** | `0.0` | [results-489605](../reports/wave6-lightcone/results-489605/warm-h1_256_64-comparison.json) |
| `h2_384_96` | `384 -> 96`, 300 Mpc, 761 切片 | `266.33 s` | **`23.29 s`** (`18.20 s` / `21.30 s`) | **`11.44×`** | **`0 / 6`** | `0.0` | [results-489605](../reports/wave6-lightcone/results-489605/warm-h2_384_96-comparison.json) |
| **`1024/256`** | `1024 -> 256`, 250 Mpc, 3,922 切片 | `6,462.85 s` | **`371.37 s`** (`336.33 s` / `360.42 s`) | **`17.40×`** | **`0 / 6`** | `0.0` | [results-489607](../reports/wave6-lightcone/results-489607/warm-1024-timing.json) |

- **`1024/256` 生产尺度六场最大绝对差（[warm-1024-comparison.json](../reports/wave6-lightcone/results-489607/warm-1024-comparison.json)）**：`brightness_temp = 6.10e-5`、`xH_box = 1.79e-7`、`x_e_box = 4.19e-9`、`Tk_box = 1.37e-4`、`Ts_box = 1.45e-4`、`dNrec_box = 0.0`。
- **`1024/256` 暖启动各阶段剖析（总物理 `336.33 s`）**：
  - `perturbation`: `22.79 s`（92 步）
  - `thermal`: `46.03 s`（含 `heat_gpu = 22.58 s`、`heat_filter = 18.96 s`，另有 `heat_prefetch_wait = 95.23 s`，90 次后台预取 100% 命中）
  - `ionization`: `212.68 s`（`batch_density = 52.88 s`、`xerec_replay = 35.50 s`；全 2,914 个非边界尺度中 `1,261` 次筛选跳过、`1,591` 次密度回放、`741` 次守卫触发 `xe/rec` 精确回放）
  - `brightness`: `40.28 s`（精确 `PS_FFTW_BRIGHTNESS=1`）
  - `write_lzf`: `11.24 s` + `fsync`: `5.00 s`（输出 `3,922` 切片、`3.45 GB` 原始场 LZF 落盘）

### 4.2 五组跨 10D 天体物理 + $\Lambda$CDM 宇宙学 + 多种子端到端抽检（Job `489621`）

| 样点 | 物理 / 宇宙学变化区间 | 网格 / 盒子 / 切片 / Seed | CPU 墙钟 | **GPU 暖启动 (`phys`)** | 加速比 | 超差 / 复现 |
|---|---|---|---:|---:|:---:|:---:|
| `p1_early_eor_strong_xray` | 早电离 + 强软 X 射线（`M_TURN=8.30, L_X=41.60, NU_X_THRESH=200, KP=6.78, MS=1.20`） | `256->64`, 200 Mpc, 1,227, `111111` | `111.69 s` | **`13.24 s`** (`8.83 s`) | **`8.44×`** | `0 / 6`, `0.0` |
| `p2_late_eor_weak_hard_xray` | 晚电离 + 弱硬 X 射线（`F_STAR10=-1.65, M_TURN=10.20, L_X=38.20, NU_X_THRESH=1400, KP=19.662, MS=3.85`） | `256->64`, 250 Mpc, 982, `222222` | `96.61 s` | **`11.82 s`** (`7.94 s`) | **`8.17×`** | `0 / 6`, `0.0` |
| `p3_cosmo_shift_lcdm_low_kp` | $\Lambda$CDM 偏移 + 极低 `KP`（`OMm=0.325, OMb=0.0505, h=0.665, s8=0.840, ns=0.955, KP=1.33`） | `256->64`, 180 Mpc, 1,353, `333333` | `114.76 s` | **`13.79 s`** (`9.14 s`) | **`8.32×`** | `0 / 6`, `0.0` |
| `p4_cosmo_shift_low_s8` | 低 `SIGMA_8` + 高阈产星（`OMm=0.290, OMb=0.0465, h=0.700, s8=0.785, ns=0.980, F_STAR10=-0.15, M_TURN=10.60`） | `256->64`, 220 Mpc, 1,113, `444444` | `101.19 s` | **`13.34 s`** (`9.17 s`) | **`7.59×`** | `0 / 6`, `0.0` |
| `p5_384_96_mid_eor_cosmo` | 高分辨率混合偏移（`OMm=0.315, OMb=0.0492, h=0.672, s8=0.828, ns=0.962, KP=10.50, MS=0.90`） | `384->96`, 300 Mpc, 839, `555555` | `266.65 s` | **`21.59 s`** (`16.43 s`) | **`12.35×`** | `0 / 6`, `0.0` |

---

## 5. 与早期 `gpu/` Coeval 归档版本的关系

- [gpu/](../gpu) 保留了 2026-11-19 完成的 **Coeval + PS 验证主入口**（固定点 `80000`、`512->128` 网格 32 个红移 `coeval` 全 13 场 + 功率谱 + GPU IC 输出，`85.69 s` 端到端并行 LZF 写盘），适合 coeval 功率谱归档复跑。
- [lightcone/](../lightcone) 是最新 **Wave 6 全流程 6 场光锥版本**：在继承并修复 [gpu/](../gpu) 物理核与表回退修复的基础上，新增异步背景 FFTW 预取、Padded 热滤波 GPU 原地解包、筛选密度回放、`PS_XEREC_AMP_MULT=0` 守卫、锐截断裁剪 3D FFTW `c2r`、`ComputeBrightnessFastFFTW`、OpenMP 并行 `init_MHR` / `init_ps()` 缓存以及 `PSFillLightconeSlices` 批量光锥插值，在所有已测分辨率（含 `1024/256`）实现严格 **`0 / 6` 逐格无超差**与高达 **`17.40×`** 加速。
