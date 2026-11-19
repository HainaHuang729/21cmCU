# 21cmFAST-GPU

21cmFAST 的 GPU 移植與效能驗證歸檔。包含 CUDA 核心、原生 C 接口補丁、完整 coeval 入口、LZF 並行輸出、固定點輸入及驗收資料。基礎來源版本為 [21cmFAST aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273)，目錄中的基礎程式另有本研究的先前修改。上游授權見 [LICENSE.upstream](LICENSE.upstream)。

## 已驗證配置

- 固定點 80000，seed 1850877074，盒邊長 250 Mpc，IC 網格 512³，演化網格 128³。
- RTX A6000 單卡及 8 個主機執行緒。完整輸出包含 14 個 IC 欄位、32 個紅移的 416 個演化場、全局歷史和兩種功率譜。
- 暖機後從啟動到共享 NFS 同步落盤，基準 98.51 秒，雙檔並行壓縮 85.69 秒。兩組均成功讀回完整 HDF5，14 個 IC 欄位雜湊一致。
- 生產 32-bin PS 兩組均 0/1024 超差。60-bin PS 兩組均 26/1920 超差，驗收容差為 `rtol=1e-4`、`atol=1e-8`。
- 85.69 秒是此盒尺寸、解析度與輸出模式的實測。單卡多任務完整輸出和更大網格未用此版本量測。

[完整重測](reports/coeval-retest.json) · [雙檔輸出對照](reports/paired-output.json) · [並行 LZF 對照](reports/parallel-lzf.json)

## 重跑

在 Linux CUDA 節點上準備 Python 3.10、NumPy、SciPy、Astropy、h5py（含 LZF）、cffi、CuPy、PyYAML，以及具 OpenMP 的 C 編譯器、GSL 和單精度 FFTW/OpenMP。擴充套件會在啟動時依賴這些套件重建。並行直寫 HDF5 需要 h5py 或系統 liblzf 提供 `lzf_compress`。

在工作樹根目錄準備下列路徑，可用符號連結接到節點上的已安裝依賴。CUDA toolkit 需與 CuPy 環境匹配。

- `include/` 包含 `gsl/` 和 `fftw3.h`
- `lib/` 包含 GSL、gslcblas、fftw3f 和 fftw3f_omp
- `env310/bin/python` 使用具備上述 Python 套件的環境
- `CUDA_PATH` 指到 CUDA toolkit，並將其 `include/` 加到 `CPATH`

```sh
export CUDA_PATH=/path/to/cuda
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
env310/bin/python gpu/launch_ps.py runs/point-80000 --cache runs/cache \
  --seed --full-coeval --include-ic --gpu-full-ic \
  --compression lzf --coeval-lzf-workers 4 --paired-archive
```

Slurm A/B 的入口是 `gpu/integrated-lzf.sbatch`。節點和分區依集群設定調整。腳本先暖機，兩組依序從同一個種子執行，每組讀回驗證後清除暫存 HDF5。小型 `reference.npz` 是固定點的驗收資料。倉庫內的重跑入口使用固定點參數，其他盒大小、解析度和物理開關需要另做數值與效能驗證。

## 其他輸出模式

較早的矩形 lightcone 實驗使用相同盒尺寸與固定點，輸出亮溫和中性氫兩個場。暖機單條 38.35 秒，單卡四並發批次 99.83 秒，約 144 條／小時。這兩份歷史 lightcone 報告使用當時的獨立實驗入口。

[Lightcone 並行計時](reports/lightcone-batch4.json) · [Lightcone 資料核對](reports/lightcone-comparison.json)
