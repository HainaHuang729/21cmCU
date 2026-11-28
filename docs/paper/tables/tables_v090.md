# 21cmCU v0.9.0 Reproducible Benchmark Tables

## Colab Tesla T4 Science & Multi-Probe Suite (`benchmarks/results_t4_v090.json`)

| Workload | Baseline [s] | T4 GPU [s] | Speedup | Parity / Notes |
| --- | ---: | ---: | ---: | --- |
| Bispectrum 3D ($128^3$, equil.) | `0.8812` | `0.1420` | **`6.21×`** | $\max|\Delta B|/|B|=1.31\times 10^{-6}$ |
| Bispectrum 3D ($256^3$, equil.) | `11.3274` | `0.4252` | **`26.64×`** | $\max|\Delta B|/|B|=8.21\times 10^{-7}$ |
| MFP BSD ($128^3$, 200k rays) | `0.6947` | `0.0095` | **`72.87×`** | $\langle R\rangle$ rel diff $3.66\times 10^{-3}$ |
| MFP BSD ($256^3$, 200k rays) | `2.3150` | `0.0370` | **`62.53×`** | $\langle R\rangle$ rel diff $5.0\times 10^{-4}$ |
| Minkowski ($128^3$, 15 thresh.) | `0.2960` | `0.0357` | **`8.28×`** | Exact Euler $\chi$ |
| Minkowski ($256^3$, 15 thresh.) | `3.0784` | `0.3632` | **`8.48×`** | Exact Euler $\chi$ |
| \texttt{cic\_deposit\_3d} ($256^3$, 1M) | `1.4336` | `0.0365` | **`39.29×`** | $\max|\Delta\delta|=3.81\times 10^{-6}$ |
| \texttt{cross\_power\_21cm\_galaxy} ($256^3$, 1M) | `4.8076` | `0.2324` | **`20.68×`** | $\max|\Delta r(k)|=2.91\times 10^{-8}$ |
| \texttt{BatchPool} Warm ($4\times 2$, T4) | `26.7845 (cold)` | `11.3101 (warm)` | **`2.368×`** | Round 4: 1.2729\,s; 1 PID |

## RTX 4090 Single-GPU Core Suite (`benchmarks/results_rtx4090_v060.json`)

| Workload | CPU 16T [s] | 1× RTX 4090 [s] | Speedup |
| --- | ---: | ---: | ---: |
| \texttt{apply\_rsds} (512 slices) | `37.6455` | `0.0764` | **`492.74×`** |
| Rectilinear LC ($z=6.5\text{--}11$) | `248.773` | `3.032` | **`82.05×`** |
| Angular LC ($z=7.8\text{--}9.8$) | `97.018` | `4.698` | **`20.65×`** |
| \texttt{include\_dvdr\_in\_tau21} | `0.305` | `0.0238` | **`12.81×`** |
| Perturb Field (2LPT, $256^3$) | `0.915` | `0.073` | **`12.58×`** |
| Coeval 3z (\texttt{L-INT+TS}, 512/256) | `55.079` | `4.456` | **`12.36×`** |
| Ionize Box ($256^3$, \texttt{E-INT}) | `2.851` | `0.264` | **`10.81×`** |
| \texttt{CLASS+FLUCTS} IC (256/128) | `2.827` | `0.295` | **`9.57×`** |
| EoR Coeval ($768^3$, $z=7.0$) | `110.209` | `16.435` | **`6.71×`** |
| 2LPT IC (512/256) | `6.244` | `1.160` | **`5.38×`** |
| IC + PF (1024/512, 2LPT lo) | `54.245` | `10.495` | **`5.17×`** |
| Concurrent Batch ($4\times 192/96$) | `14.184` | `3.097` | **`4.58×`** |
