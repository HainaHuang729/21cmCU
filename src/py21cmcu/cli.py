"""Command-line interface for 21cmCU (extends upstream 21cmFAST commands with batch/doctor/bench)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from py21cmfast.cli import app

_cns = Console()


def _load_param_rows(
    *,
    params_path: Path | None,
    sobol_samples: int,
    bounds_json: str | None,
) -> list[dict[str, float | int]]:
    import py21cmcu

    if params_path is not None:
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {params_path}")
        suffix = params_path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(params_path.read_text())
            if not isinstance(raw, list):
                raise ValueError("JSON parameter file must contain a list of parameter dicts")
            return [
                {
                    k: (int(v) if k == "random_seed" else float(v))
                    for k, v in item.items()
                }
                for item in raw
            ]
        if suffix == ".csv":
            rows: list[dict[str, float | int]] = []
            with params_path.open("r", newline="") as fl:
                reader = csv.DictReader(fl)
                for item in reader:
                    rows.append(
                        {
                            k: (int(v) if k == "random_seed" else float(v))
                            for k, v in item.items()
                            if v is not None and v != ""
                        }
                    )
            return rows
        raise ValueError(f"Unsupported params file extension {suffix!r}; use .json or .csv")

    if sobol_samples > 0:
        bounds = (
            json.loads(bounds_json)
            if bounds_json
            else {"F_STAR10_ACG": [-1.8, -0.7], "F_ESC10_ACG": [-1.8, -0.5]}
        )
        parsed_bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        return py21cmcu.generate_sobol_parameter_grid(parsed_bounds, n_samples=sobol_samples)

    raise ValueError("Provide either --params <file.csv|json> or --sobol-samples N (> 0)")


@app.command(name="batch")
def batch_cmd(
    params: Path | None = None,
    sobol_samples: int = 0,
    bounds: str | None = None,
    out: Path = Path("batch_summary.h5"),
    mode: Literal["coeval", "lightcone"] = "coeval",
    dim: int = 128,
    hii_dim: int = 64,
    box_len: float = 128.0,
    redshifts: str = "8.0",
    devices: str = "0",
    workers_per_gpu: int = 1,
    threads_per_worker: int = 2,
    cuda_mode: Literal["strict", "fast"] = "fast",
    use_cuda: bool = True,
    resume: bool = False,
    wedge_filter: bool = False,
    wedge_buffer: float = 0.0,
    noise_sigma_mk: float | None = None,
    ps_n_bins: int = 12,
    compute_ps_2d: bool = False,
    in_process: bool | None = None,
) -> None:
    """Run a GPU-accelerated parameter sweep from CSV/JSON or Sobol bounds and save HDF5 summaries."""
    import py21cmcu

    rows = _load_param_rows(
        params_path=params,
        sobol_samples=sobol_samples,
        bounds_json=bounds,
    )
    z_list = [float(z.strip()) for z in redshifts.split(",") if z.strip()]
    if not z_list:
        z_list = [8.0]
    dev_tuple = tuple(int(d.strip()) for d in devices.split(",") if d.strip()) or (0,)

    _alias_map = {
        "F_STAR10": "F_STAR10_ACG",
        "F_ESC10": "F_ESC10_ACG",
        "F_STAR7_MINI": "F_STAR7_MCG",
        "F_ESC7_MINI": "F_ESC7_MCG",
        "ALPHA_STAR": "ALPHA_STAR_ACG",
        "ALPHA_STAR_MINI": "ALPHA_STAR_MCG",
        "L_X": "LX_OVER_SFR_ACG",
        "L_X_MINI": "LX_OVER_SFR_MCG",
    }
    tasks: list[py21cmcu.BatchTask] = []
    for idx, row in enumerate(rows):
        seed = int(row.get("random_seed", 1000 + idx))
        extra_params: dict[str, Any] = {
            _alias_map.get(k, k): v for k, v in row.items() if k != "random_seed"
        }
        inp = py21cmcu.InputParameters(
            random_seed=seed,
            node_redshifts=z_list,
        ).evolve_input_structs(
            DIM=dim,
            HII_DIM=hii_dim,
            BOX_LEN=box_len,
            N_THREADS=threads_per_worker,
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
            **extra_params,
        )
        tasks.append(
            py21cmcu.BatchTask(
                task_id=f"task_{idx:04d}",
                inputs=inp,
                mode=mode,
                out_redshifts=tuple(z_list) if mode == "coeval" else None,
                min_redshift=min(z_list) + 0.05 if mode == "lightcone" and len(z_list) >= 2 else None,
                max_redshift=max(z_list) - 0.05 if mode == "lightcone" and len(z_list) >= 2 else None,
                cuda_mode=cuda_mode,
                ps_n_bins=ps_n_bins,
                compute_ps_2d=compute_ps_2d,
                wedge_filter=wedge_filter,
                wedge_buffer=wedge_buffer,
                noise_sigma_mk=noise_sigma_mk,
            )
        )

    results = py21cmcu.run_batch(
        tasks,
        devices=dev_tuple,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=threads_per_worker,
        output_h5=out,
        checkpoint_h5=out,
        resume=resume,
        use_cuda=use_cuda,
        in_process=in_process,
    )
    n_ok = sum(1 for r in results if r.ok)
    _cns.print(f"[green]21cmcu batch finished[/green]: {n_ok}/{len(results)} tasks succeeded -> {out}")


@app.command(name="doctor")
def doctor_cmd(*, as_json: bool = False) -> None:
    """Inspect CUDA runtime, GPU topology, C toolchain, and grid capacity."""
    from .doctor import doctor

    report = doctor(as_json=as_json)
    if as_json:
        return

    ver = report["versions"]
    cu = report["cuda"]
    tc = report["toolchain"]

    tbl = Table(title=f"21cmCU Diagnostics (v{ver['py21cmcu']})")
    tbl.add_column("Check", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("Python / NumPy", f"{ver['python']} / {ver['numpy']} ({ver['cpu_count']} CPUs)")
    tbl.add_row(
        "CUDA Runtime",
        f"compiled={cu['compiled']}, available={cu['available']}, devices={cu['device_count']}, "
        f"active={cu['active_devices']}",
    )
    tbl.add_row(
        "Mode / Lazy D2H",
        f"mode={cu['cuda_mode']}, lazy_d2h={cu['cuda_lazy_d2h']}",
    )
    for g in cu["gpus"]:
        tbl.add_row(f"GPU {g['index']}", f"{g['name']} ({g['memory_mib']} MiB, sm_{g['compute_cap']})")
    tbl.add_row("Toolchain", f"nvcc={tc['nvcc']}, gcc={tc['gcc']}, gsl={tc['libgsl']}, fftw3f={tc['libfftw3f']}")
    _cns.print(tbl)

    if cu["nvlink_or_pcie_topo"]:
        _cns.print("\n[bold]GPU Topology (`nvidia-smi topo -m`):[/bold]")
        _cns.print(cu["nvlink_or_pcie_topo"])

    sizing = Table(title="Recommended Grid Sizing (24 GB GPUs)")
    sizing.add_column("Workload")
    sizing.add_column("Min GPUs")
    sizing.add_column("VRAM / GPU [GB]")
    sizing.add_column("Host RAM [GB]")
    for row in report["grid_sizing"]:
        sizing.add_row(
            row["grid"],
            str(row["min_gpus_24gb"]),
            row["vram_per_gpu_gb"],
            row["host_ram_gb"],
        )
    _cns.print(sizing)


@app.command(name="bench")
def bench_cmd(
    suite: str = "lightcone,batch",
    json_out: Path | None = None,
    resume: bool = False,
) -> None:
    """Run 21cmCU benchmark suites from the repository harness."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmarks import bench_21cmcu

    argv = ["--suite", suite]
    if json_out is not None:
        argv.extend(["--json", str(json_out)])
    if resume:
        argv.append("--resume")
    old_argv = sys.argv
    try:
        sys.argv = ["bench_21cmcu.py", *argv]
        bench_21cmcu.main()
    finally:
        sys.argv = old_argv


__all__ = ["app", "batch_cmd", "bench_cmd", "doctor_cmd"]


if __name__ == "__main__":
    app()
