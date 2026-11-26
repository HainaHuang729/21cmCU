# 21cmFAST 4.3.2 CUDA 移植状态与文件索引

更新于 2026-11-24。以下结论以当前工作树和实际 CPU/GPU 对照为准。GPU 入口接收原生 4.3.2 的参数和光锥对象，复用 v3 CUDA 的 FFT、求积和物理核。

## 移植进度树

状态标记对应具体测试配置。大盒、小盒均指同输入原生 CPU 对照，单步指独立核验，混合指仍有 CPU 计算或回退。

```text
21cmFAST 4.3.2 CUDA 移植
├─ 1. 公共入口与 v3 CUDA 复用
│  ├─ [已接入] 原生 InputParameters、光锥驱动、缓存接口
│  └─ [已接入] FFTW 计划、Gauss–Kronrod 求积、热演化、电离和切片核
├─ 2. 物质谱、质量方差和初始条件
│  ├─ [大盒] EH（含 8 线程）与 BBKS 高斯窗口的初始条件、2LPT 及 192³ 光锥（含摄动场 GPU 常驻缓存）
│  ├─ [大盒] 192³ CLASS FLUCTS + 空间相对流速 `lowres_vcb` 全场零差异
│  ├─ [小盒] EFSTATHIOU、PEEBLES、WHITE；CLASS 五模型；高斯窗口 EH 随机晕与 CLASS 欧拉
│  ├─ [单步] 高斯窗口方差表/初始场、5 种 CUDA GSL PRNG（`mt`/`gfsr`/`cmrg`/`mrg`/`taus2`，1–16 线程）逐位一致
│  └─ [主机] CLASS 求解；GPU 使用原生传递表
├─ 3. 源项与晕目录
│  ├─ [大盒] E-INTEGRAL 原子冷却（2 线程与 8 线程 192³ 全通过）
│  ├─ [中盒] 96³/32³ EH CHMF-SAMPLER 热演化光锥（`1T hybrid` 全通过、`4.81 s` / `4.94×`；`2T/8T` `3.80/5.46 s`，目录及全局量/功率谱全通过，仅电离边界面元余 1–2 个 OpenMP float32 累加序体素）
│  ├─ [小盒] 32³/16³ `CHMF-SAMPLER` `1/2/4/8` 线程 × `hybrid`/`prefix` 共 8 组全通过；CONST-ION-EFF、L-INTEGRAL、DEXM-ESF 与 CLASS 相对流速全通过
│  └─ [混合] 解除 `CHMF-SAMPLER` 两线程上限（支持 `N_THREADS >= 1` 并支持 31-bit `cmrg`/`mrg` 流）；仍保留原生 GSL 随机数与 `prefix` 模式高密度串行回退
├─ 4. 辐射、热演化与电离
│  ├─ [大盒] 原子冷却 192³/384³ 常驻化提速至 `15.58 s`（`4.22×`）/ `37.60 s`（`9.24×`）；BBKS 高斯窗口 192³ 暖态提速至 `11.55 s`（`5.52×`）
│  ├─ [大盒] 192³ 分子冷却+Lyα 多重散射：共动框源项/辐射/Ts/电离箱全通过，总耗时降至 `47.96 s`（不含 CLASS `4.10×` / 总 `5.19×`），光锥 `Tb` 余同源 37 个 $T_s \approx T_{\mathrm{CMB}}$ 近零体素微差
│  ├─ [小盒] 拉格朗日 Lyα 多重散射；欧拉均匀复合
│  ├─ [单步] `DeviceFieldRegistry` GPU 常驻注册表、发射率快照复用与免回传（3,262/3,258 次）、辐射准备常驻、电离滤波半径环路全程常驻 GPU
│  └─ [混合] 壳层几何和部分标量在主机；光子校准拟合走原生 CPU
├─ 5. 光锥生成与并发批处理
│  ├─ [大盒] 矩形 z 轴光锥及速度修正（直连 GPU 切片与亮温，零额外主机上传）
│  ├─ [小盒] 角向光锥、角向速度投影和红移空间畸变
│  └─ [吞吐] `DeviceLZF` / `ParallelLZF` 多线程 HDF5 压缩写出与单卡并发批跑（192³ `286.9/h`，384³ `118.9/h`）
└─ 6. 剩余验收与归档
   ├─ [已扩展] 全五种源模型（含 `CHMF-SAMPLER`）支持 `N_THREADS >= 1`
   ├─ [已扩展] `run_coeval`/`generate_coeval`、自动 `photon_calibration=None`、全套 HMF/热演化/电离/采样开关，以及 `NON_CUBIC_FACTOR != 1`、奇数网格、`USE_FFTW_WISDOM=True`（102/102 小盒覆盖通过）
   ├─ [受限] 大盒仍主要覆盖已归档配置，192³ 单卡 4 并发存在调度竞争、`CHMF prefix` 大密度下串行受限
   ├─ [跳过] v3 / 4.3.2 × CPU / GPU 整条光锥对照计时（已取消重测，保留 2LPT 单节点四方对齐）
   └─ [待归档] 工作树中尚未纳入 Git 的代码和报告
```

## 已完成并验证

| 模块 | 已验证范围 | 证据 |
| --- | --- | --- |
| 欧拉原子冷却大盒 | `192³ / 64³` 与 `384³ / 96³`（2 线程，43 节点）。输入参数、初始条件哈希和切片坐标一致。各通过 8 个光锥场、903 个全局值、1,376 个 32-bin 功率谱与 20,800/25,008 个横向功率谱。经 `DeviceFieldRegistry`（命中 `3,509`/`3,393` 次，避免 `3.90`/`12.75 GB` 传输）、摄动缓存、发射率壳层免下载（`3,262`/`3,258` 次）与电离半径常驻后，CPU / GPU 耗时分别为 **`65.78 / 15.58 秒`（`4.22×`，较此前 `20.93 s` 再快 `25.6%`）** 和 **`347.52 / 37.60 秒`（`9.24×`，较此前 `45.14 s` 再快 `16.7%`）**。 | [大盒报告](../reports/v432-current-large-lightcones.json) · [常驻大盒与回归](../reports/v432-resident-large-lightcones.json) |
| 多线程 PRNG 与 8 线程大盒 | 5 种 CUDA GSL PRNG（`mt`/`gfsr`/`cmrg`/`mrg`/`taus2`）各 4 种子 × 100,000 字零错配；1–16 线程种子一致性零错配；`64³ / 32³`（1/2/4/5/8 线程）初始场逐位一致。8 线程 `192³ / 64³`（200 Mpc，43 节点）`E-INTEGRAL` 光锥通过 8 个光锥场（`Tb` 最大差 `1.53 × 10⁻⁵ mK`）、903 个全局值、1,376 个亮度功率谱分箱及 20,800 个横向功率谱点；同平台 `a8` 8 线程 CPU / GPU 总耗时 **`34.93 / 12.88 秒`（`2.71×`**；较此前 `a9` `55.69 s` 提速 **`4.32×`**）。 | [多线程校验](../reports/v432-multithread-rng.json) |
| `CHMF-SAMPLER` 多线程与 `hybrid`/`prefix` 提速 | 为 `cmrg`/`mrg`（Worker 余数 `2/3`）补齐 31-bit GSL 浮点归一化并修复多线程流与子网格寄存器竞态，正式支持 `N_THREADS >= 1`。`32³ / 16³`（3 节点）1/2/4/8 线程 × `hybrid`/`prefix` **全部 8 组零失败且 5 项目录字段逐位一致**（暖态 GPU `1.64–1.99 s`，加速 `2.73×–6.78×`）。`96³ / 32³`（5 节点，`6.51M–6.68M` 晕）全部目录逐位一致：`1T hybrid` 全零失败且暖态仅 **`4.81 s`（`4.94×` vs CPU `23.78 s`，较旧 `prefix` `74.15 s` 提速 `15.4×`**）；`2T/8T hybrid` 暖态分别 **`3.80 s`（`3.78×`） / `5.46 s`（`1.58×`）**，仅电离边界面元余 1 / 2 个 CPU OpenMP float32 累加序体素差异；`1T prefix` 从 `74.15 s` 改善至 `48.52 s`（仍受 `(1,),(1,)` 串行核制约）。 | [CHMF 多线程与提速](../reports/v432-chmf-multithread-and-speed.json) |
| LZF 压缩 HDF5 与单卡并发批跑 | 实现 `device-lzf` / `parallel-lzf`（含 `chunks_written`、字节对齐写出与 fsync）和统一并发入口 [run_batch.py](../v4/run_batch.py)。`192³` 与 `384³` LZF 往返校验均逐比特一致（`none` `0.382 / 0.679 s`，`device-lzf` **`0.352 / 0.685 s`**、压缩比 **`1.35× / 1.31×`**，分别写入 `328 / 1,056` chunks 且 `skipped_lzf_chunks=0`）。单张 A6000（2 线程/worker，`device-lzf`）并发吞吐：`64³ verify` `2/2` 全过（`25.40 s`）；`192³` `conc=1/2/4` 分别 **`234.1 / 286.9 / 134.3` 光锥/小时**（`conc=2` 最优、峰值显存 `4,480 MiB`）；`384³` `conc=1/2/4` 分别 **`98.0 / 111.4 / 118.9` 光锥/小时**（`conc=4` 较串行提升 `21.3%`、峰值显存 `35,885 MiB`）。 | [LZF 与批跑](../reports/v432-lzf-and-batch.json) |
| 物质谱和初始条件 | EH、BBKS、EFSTATHIOU、PEEBLES、WHITE、CLASS。四种新增解析谱通过欧拉热演化小盒。CLASS 密度谱通过五种源模型小盒，CHMF 和 DexM 的有序目录一致。 | [解析谱](../reports/v432-analytic-spectra.json) · [CLASS 五模型](../reports/v432-class-density-only.json) |
| 高斯物质窗口 | EH、BBKS、CLASS 质量方差与导数表及 11 个种子初始场逐位一致。小盒 CLASS `E-INTEGRAL` 与 EH `CHMF-SAMPLER` 各通过 13 个光锥场、57 个全局值及 776 个横向功率谱点。大盒 `192³ / 64³` BBKS + 高斯窗口（2 线程，43 节点）全通过：同 `a8` 节点 CPU **`63.76 s`**，冷/暖态 GPU 降至 **`14.96 s` / `11.55 s`（`5.52×`**，较此前 `a9` 暖态 `49.93 s` 提速 **`4.32×`**）。 | [高斯窗口](../reports/v432-gaussian-matter-window.json) · [大盒新物理](../reports/v432-large-new-physics.json) |
| 大盒分子冷却新物理 | `192³ / 64³`（2 线程，10 节点 `z=20..10`）CLASS `FLUCTS` + 分子冷却 + Lyα 多重散射：`lowres_vcb` 262,144 格点零差异，全 10 节点共动 `Emissivity`/`Radiation`/`Ts`/`Ionized` 箱零超差，14/15 个光锥场、220 个全局值、4 个归一化分子场和 11,520 个横向功率谱全通过；光锥 `Tb` 在 1,474,560 点中有 37 点（0.0025%）及共动 `BrightnessTemp` 在 `z=12.5/11.6/10.8` 有 `1/12/15` 点于 $T_s \approx T_{\mathrm{CMB}}$ 近零区超容差，全局最大 `Tb` 差 `3.05 × 10⁻⁵ mK`；同 `a8` 节点 CPU **`248.89 s`**（不含 CLASS `196.48 s`），GPU **`47.96 s`**（不含 CLASS **`4.10×`**，总耗时 **`5.19×`**）。 | [大盒新物理](../reports/v432-large-new-physics.json) |
| 分子冷却与热演化（小盒） | 欧拉及拉格朗日 Lyα 多重散射、空间相对流速随机晕均通过小盒整链验收，共 552,960 个源项、原始辐射、热演化和电离快照值通过逐点相对误差 `10⁻⁵`。随机晕的 CLASS 平均流速也单独通过；最新常驻分支下常效热/均匀复合小盒、`128` 射线角向光锥与 `32³` 1T 热 `CHMF` 小盒均零失败通过。 | [三组小盒整链](../reports/v432-joined-small-lightcones.json) · [平均流速](../reports/v432-chmf-class-avg-auto.json) · [常驻大盒与回归](../reports/v432-resident-large-lightcones.json) |
| 辐射流水线 | 相邻壳层复用发射率快照并跳过冗余主机回传，FFT 直接读取 GPU 插值结果，Lyα 与 LW 共用正向 FFT，滤波网格、电离半径迭代与累计值可全程常驻 GPU，并经 `DeviceFieldRegistry` 跨阶段直连光锥切片与亮温核。 | [快照复用](../reports/v432-emissivity-history-reuse.json) · [FFT 交接](../reports/v432-emissivity-filter-handoff.json) · [累加驻留](../reports/v432-radiation-accumulation-residency.json) |
| 光锥几何与光子校准 | 矩形（`line_of_sight_axis` 0/1/2）和角向光锥均有 CPU 对照，支持零上传切片与亮温输出，并已支持 `run_coeval`/`generate_coeval` 与 `photon_calibration=None` 自动校准。欧拉质量相关源支持 `z`/`alpha`/`f-photoncons` 校正；`CONST-ION-EFF` + `z-photoncons` 与原生 4.3.2 一致触发 `KeyError: 'CONST-ION-EFF'`；校准拟合使用原生 CPU。 | [角向光锥](../reports/v432-angular-lightcone.json) · [光子校准](../reports/v432-photon-conservation.json) |
| 全选项开关、API 与一般网格 | 由 [verify_full_coverage.py](../v4/verify_full_coverage.py) 覆盖 Suite A（`40/40`：`run_coeval`/`generate_coeval`、`line_of_sight_axis`、`MINIMIZE_MEMORY`、`KEEP_3D_VELOCITIES`、全部 7 种 `HMF`、`HALO_FILTER`、`SAMPLE_METHOD`、`GAMMA-APPROX`、`USE_INTERPOLATION_TABLES`）、Suite B（`31/31`：无光致加热 MCG、整球电离、`USE_X_RAY_HEATING`/`USE_LYA_HEATING`/`USE_CMB_HEATING`/`USE_METALLICITY`/复合开关、`V_CB_MODEL`）、Suite C（`9/9`：自动 `z`/`alpha`/`f-photoncons`）与 Suite D（`22/22`：`NON_CUBIC_FACTOR=1.5/2.0/1.4`、奇数 `15/5`、`USE_FFTW_WISDOM=True`），合计 **`102/102` 通过（`0` 失败）**；非立方 `ncf15_e_mcg` 仅在中间 4D 数组 `nion_conditional_filtered_mcg` 出现 `2` 个 1-ULP 阈值阶跃翻转（`<= 0.1%` 容差内）。改动后 `192³ / 64³` 回归耗时 `16.15 s`（`4.07×` vs `65.78 s` CPU，8 场 / 全局量 / 1,376 个 PS32 分箱零失败）且 `32³` `CHMF` 多线程 `8/8` 目录一致（`1.57–1.97 s` 暖态，`3.75×–7.21×`）。 | [全选项覆盖与回归](../reports/v432-full-option-coverage.json) |

## 未完成或仍有限制

本地安装、测试入口和可选 glibc/GSL 源码目录的环境变量见 [v4 使用说明](../v4/README.md)。公开运行入口默认对 GPU 未覆盖的配置回退到原生 CPU，验证脚本使用 `fallback_to_cpu=False` 确认 GPU 路径。

- **大盒与随机晕覆盖**。`E-INTEGRAL` 已补齐常驻化大盒、BBKS 高斯窗口和 8 线程 `192³` 验收；`192³` CLASS 空间流速 + 分子冷却 + 多重散射仅余 37 个 $T_s \approx T_{\mathrm{CMB}}$ 近零交叉微差体素（≤ `3.05 × 10⁻⁵ mK`）。`96³` 热演化 `CHMF-SAMPLER` 默认 `hybrid` 已可在 `1/2/8` 线程取得 `4.94× / 3.78× / 1.58×` 加速（1 线程零失败），`2/8` 线程仍因 CPU OpenMP atomic 浮点累加序在电离边界存在 1–2 体素偏移，`prefix` 模式在高晕数下（`96³ 1T = 48.52 s`）仍受单块串行核限制。CLASS 随机晕与 `CONST-ION-EFF`、`L-INTEGRAL`、`DEXM-ESF`、`EFSTATHIOU`/`PEEBLES`/`WHITE` 及角向光锥目前仍主要是小盒证据。
- **原生 CPU 工作与回退**。CLASS 求解、Astropy 距离与壳层几何、部分标量系数与光子校准拟合仍由主机准备。`CHMF` 仍保留原生 GSL 随机数与数值边界回退；`run_batch.py` 在 `192³` 单卡 `conc=4` 下存在多进程 GPU 调度竞争（`134.3/h`，建议 `conc=2` 获 `286.9/h`），可用 `backend-routing.json` 核对实际执行路径。
- **参数覆盖、分级误差与已确认上游行为**。全五种源模型（含 `CHMF-SAMPLER`）支持 `N_THREADS >= 1` 与 5 种 CUDA PRNG；公开入口已覆盖 `run_coeval`/`generate_coeval`/`run_lightcone`/`generate_lightcone`、全部 7 种 HMF、全套热演化/复合/采样开关、自动 `photon_calibration=None`，以及经 [fft_general.py](../v4/fft_general.py)（cuFFT）实现的 `NON_CUBIC_FACTOR != 1`、奇数 `DIM`/`HII_DIM` 与 `USE_FFTW_WISDOM=True`（102/102 通过）。误差采用两级口径：偶数立方且 `!USE_FFTW_WISDOM` 走 `fft_plan_cuda.transform` 严格回放 FFTW 累加顺序（`1e-8 + 1e-4 * |CPU|` 全通过）；一般 cuFFT 路径允许 `1e-5 + 1e-4 * |CPU|` 物理场容差，且 `USE_MCGS=True` 时中间数组 `nion_conditional_filtered_mcg` 允许 `<= 0.1%` 阶跃阈值翻转（1 ULP 跨越 `atomic_cooling_threshold * 1.001`）。两项已确认的原生 4.3.2 上游行为：`USE_MCGS=True, USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False, SOURCE_MODEL="E-INTEGRAL"` 在原生 C 中会覆盖单一 1D ACG 查找表（GPU 分键 `(previous, mini)` 可安全运行，`mcg_only_e_gpu_safe`）；`SOURCE_MODEL="CONST-ION-EFF", PHOTON_CONS_TYPE="z-photoncons"` 在原生 Python `calibrate_photon_cons` 中抛出 `KeyError: 'CONST-ION-EFF'`（CPU 与 GPU 一致）。
- **跨版本耗时归因**。v3 与 4.3.2 整条光锥的历史计时采用不同输入、节点数和计时边界。按确认跳过重测 `v3 CPU` 与 `v3 GPU` 整条光锥对照并已取消集群基准作业；严格四方对齐证据为单节点 2LPT，见 [四方摄动对照](../reports/v3-v432-matched-perturbation.json) 和 [计时口径](../v4/README.md#comparing-recorded-run-times-across-versions)。
- **版本归档**。当前移植与优化成果包含工作树中尚未纳入 Git 的代码和报告。单独检出当前 HEAD 无法得到这份索引对应的完整实现。

## 代码与验证文件索引

| 用途 | 文件 |
| --- | --- |
| 入口、并发批跑与输入限制 | [gpu_lightcone.py](../v4/gpu_lightcone.py) · [run_batch.py](../v4/run_batch.py) · [v4 使用说明](../v4/README.md) |
| 4.3.2 原生适配与 GPU 常驻注册表 | [evolution_backend.py](../v4/evolution_backend.py) · [device_registry.py](../v4/device_registry.py) · [native_shim.py](../v4/native_shim.py) · [evolution_grid.cu](../v4/evolution_grid.cu) |
| 功率谱、方差与初始条件 | [power_setup.c](../v4/power_setup.c) · [sigma_moments.cu](../v4/sigma_moments.cu) · [fft_general.py](../v4/fft_general.py) · [analytic_power.py](../v4/analytic_power.py) · [classy_power.py](../v4/classy_power.py) · [ic_random.py](../v4/ic_random.py) · [ic_random.cu](../v4/ic_random.cu) · [perturb.py](../v4/perturb.py) · [cic_ordered.py](../v4/cic_ordered.py) |
| 随机晕、多线程 PRNG、解析晕与源项 | [halo_catalog.py](../v4/halo_catalog.py) · [halo_catalog_random.h](../v4/halo_catalog_random.h) · [halo_catalog_batch.h](../v4/halo_catalog_batch.h) · [halo_progenitor_batch.h](../v4/halo_progenitor_batch.h) · [catalog_progenitor.py](../v4/catalog_progenitor.py) · [halo_rng.py](../v4/halo_rng.py) · [halo_dexm.py](../v4/halo_dexm.py) · [halo_grid.py](../v4/halo_grid.py) · [source_integrals.cu](../v4/source_integrals.cu) |
| 辐射、热演化及电离 | [halo_radiation.py](../v4/halo_radiation.py) · [lyman_scattering.py](../v4/lyman_scattering.py) · [halo_ionization.py](../v4/halo_ionization.py) · [eulerian_mcg_ion.py](../v4/eulerian_mcg_ion.py) · [eulerian_mcg_history.py](../v4/eulerian_mcg_history.py) · [halo_mcg_turnover.py](../v4/halo_mcg_turnover.py) · [photon_conservation.py](../v4/photon_conservation.py) |
| 光锥插值、亮温与 LZF I/O | [lightcone_device.py](../v4/lightcone_device.py) · [angular_slices.py](../v4/angular_slices.py) · [brightness.py](../v4/brightness.py) · [lightcone_io.py](../v4/lightcone_io.py) |
| 复用的 v3 CUDA 核 | [FFT 计划](../lightcone/plan_cuda/fft_plan_cuda.py) · [自适应求积](../lightcone/table_qag.cuh) · [X 射线积分](../lightcone/xray_quadrature.cu) · [热演化](../lightcone/update_stabilized.cu) · [电离](../lightcone/ion.cu) · [切片插值](../lightcone/device_slices.py) |
| 大盒与多模型验收入口 | [verify_full_coverage.py](../v4/verify_full_coverage.py) · [lightcone_ps.py](../v4/lightcone_ps.py) · [lightcone_pair.py](../v4/lightcone_pair.py) · [compare_lightcone_ps.py](../v4/compare_lightcone_ps.py) · [verify_multithread_rng.py](../v4/verify_multithread_rng.py) · [verify_chmf_multithread.py](../v4/verify_chmf_multithread.py) · [verify_large_new_physics.py](../v4/verify_large_new_physics.py) · [verify_chmf_lightcone.py](../v4/verify_chmf_lightcone.py) · [verify_constant_lightcone.py](../v4/verify_constant_lightcone.py) · [verify_angular_lightcone.py](../v4/verify_angular_lightcone.py) |
| 单步核验 | [质量方差](../v4/verify_power_setup.py) · [分子辐射](../v4/verify_molecular_radiation.py) · [Lyα 多重散射](../v4/verify_lya_scattering.py) · [电离与温度](../v4/verify_ion_reuse.py) · [标量与切片复用](../v4/verify_scalar_slice_reuse.py) · [CHMF 属性随机流](../v4/verify_chmf_properties.py) |
