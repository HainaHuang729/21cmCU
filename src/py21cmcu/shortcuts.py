"""High-level ergonomic simulation entrypoints and multi-GPU redshift-parallel lightcones."""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import z_at_value

from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers.coeval import run_coeval
from py21cmfast.drivers.lightcone import LightCone, run_lightcone, setup_lightcone_instance
from py21cmfast.drivers.single_field import compute_initial_conditions
from py21cmfast.lightconers import ObliqueRectilinearLightconer, RectilinearLightconer
from py21cmfast.wrapper.inputs import InputParameters

from .inference import _ALIAS_MAP

if TYPE_CHECKING:
    from py21cmfast.drivers.coeval import Coeval


@contextlib.contextmanager
def _mode_ctx(mode: Literal["strict", "fast"], *, use_cuda: bool, lazy_d2h: bool | None = None, devices: Any = None):
    extra: dict[str, Any] = {}
    if lazy_d2h is not None:
        extra["CUDA_LAZY_D2H"] = bool(lazy_d2h)
    if devices is not None:
        extra["CUDA_DEVICES"] = devices
    with config.use(USE_CUDA=use_cuda, CUDA_MODE=mode, **extra) as cfg:
        yield cfg


def _split_input_overrides(
    astro_params: dict[str, Any] | None,
    cosmo_params: dict[str, Any] | None,
    extra_sim_kwargs: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for src in (cosmo_params or {}, astro_params or {}, extra_sim_kwargs):
        for k, v in src.items():
            overrides[_ALIAS_MAP.get(k, k)] = v
    src_model = overrides.get("SOURCE_MODEL")
    if src_model in ("L-INTEGRAL", "E-INTEGRAL", "CONST-ION-EFF"):
        overrides.setdefault("USE_UPPER_STELLAR_TURNOVER", False)
    if src_model in ("E-INTEGRAL", "CONST-ION-EFF"):
        overrides.setdefault("USE_EXP_FILTER", False)
    if src_model == "L-INTEGRAL":
        overrides.setdefault("PHOTON_CONS_TYPE", "no-photoncons")
    return overrides


def _build_node_redshifts(z_min: float, z_max: float, *, n_nodes: int | None) -> list[float]:
    if z_max <= z_min:
        raise ValueError(f"z_max ({z_max}) must be strictly greater than z_min ({z_min})")
    if n_nodes is None:
        ratio = (1.0 + z_max) / (1.0 + z_min)
        n_nodes = max(4, min(32, int(np.ceil(np.log(ratio) / np.log(1.025))) + 1))
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
    zs = np.geomspace(1.0 + float(z_max), 1.0 + float(z_min), int(n_nodes)) - 1.0
    zs[0] = float(z_max)
    zs[-1] = float(z_min)
    return [float(x) for x in zs]


def quick_coeval(
    redshifts: float | Sequence[float] = 8.0,
    *,
    box_len: float = 200.0,
    dim: int = 128,
    hii_dim: int | None = None,
    seed: int = 42,
    mode: Literal["strict", "fast"] = "fast",
    use_cuda: bool | None = None,
    lazy_d2h: bool = True,
    use_ts_fluct: bool = False,
    recomb_model: str = "none",
    devices: Any = None,
    astro_params: dict[str, Any] | None = None,
    cosmo_params: dict[str, Any] | None = None,
    **extra_sim_kwargs: Any,
) -> Coeval | list[Coeval]:
    """Run a coeval 21cm simulation with clean 1-line defaults.

    Returns a single :class:`Coeval` when ``redshifts`` is a scalar float, or a list of
    :class:`Coeval` boxes when a sequence of redshifts is provided.
    """
    is_scalar = np.ndim(redshifts) == 0
    z_list = [float(redshifts)] if is_scalar else [float(z) for z in redshifts]
    if not z_list:
        raise ValueError("redshifts must not be empty")
    z_sorted = sorted(set(z_list), reverse=True)

    eff_hii = int(dim) // 2 if hii_dim is None else int(hii_dim)
    overrides = _split_input_overrides(astro_params, cosmo_params, extra_sim_kwargs)
    overrides.setdefault("DIM", int(dim))
    overrides.setdefault("HII_DIM", eff_hii)
    overrides.setdefault("BOX_LEN", float(box_len))
    overrides.setdefault("USE_TS_FLUCT", bool(use_ts_fluct))
    overrides.setdefault("RECOMB_MODEL", str(recomb_model))

    inputs = InputParameters(
        random_seed=int(seed),
        node_redshifts=z_sorted,
    ).evolve_input_structs(**overrides)

    eff_cuda = (
        bool(lib.p21_cuda_is_compiled() and lib.p21_cuda_is_available())
        if use_cuda is None
        else bool(use_cuda)
    )
    with _mode_ctx(mode, use_cuda=eff_cuda, lazy_d2h=lazy_d2h, devices=devices):
        boxes = run_coeval(inputs=inputs, out_redshifts=z_list, write=False)
    return boxes[0] if is_scalar else boxes


def _lightcone_worker(task: dict[str, Any]) -> dict[str, Any]:
    if task["cuda_visible"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task["cuda_visible"])

    from collections import deque

    from py21cmfast.drivers.coeval import _setup_ics_and_pfs_for_scrolling
    from py21cmfast.drivers.lightcone import _run_lightcone_from_perturbed_fields
    from py21cmfast.io.caching import CacheConfig
    from py21cmfast.wrapper.outputs import InitialConditions

    inputs = InputParameters.from_template(task["base_inputs_dict"]) if isinstance(
        task["base_inputs_dict"], dict
    ) else task["base_inputs_dict"]
    inputs = inputs.clone(node_redshifts=task["node_redshifts"])

    ic_data = np.load(task["ic_npz"])
    ic = InitialConditions.new(inputs=inputs)
    for k in ic_data.files:
        ic.set(k, ic_data[k])

    if task.get("oblique", False):
        lcr = ObliqueRectilinearLightconer(
            lc_distances=task["lc_distances_mpc"] * u.Mpc,
            quantities=tuple(task["quantities"]),
            cosmo=inputs.cosmo_params.cosmo,
            index_offset=task["index_offset"],
            direction=tuple(task["oblique_direction"]),
        )
    else:
        lcr = RectilinearLightconer(
            lc_distances=task["lc_distances_mpc"] * u.Mpc,
            quantities=tuple(task["quantities"]),
            cosmo=inputs.cosmo_params.cosmo,
            index_offset=task["index_offset"],
        )

    write_off = CacheConfig.off()
    with _mode_ctx(task["mode"], use_cuda=task["use_cuda"], lazy_d2h=task["lazy_d2h"]):
        ic_out, perturbed_fields, halofield_list, photon_data = _setup_ics_and_pfs_for_scrolling(
            all_redshifts=inputs.node_redshifts,
            initial_conditions=ic,
            inputs=inputs,
            write=write_off,
            progressbar=False,
            stream_perturbed_fields=task["stream_perturbed_fields"],
            cache=None,
            regenerate=True,
        )
        lc = deque(
            _run_lightcone_from_perturbed_fields(
                initial_conditions=ic_out,
                perturbed_fields=perturbed_fields,
                lightconer=lcr,
                inputs=inputs,
                lc_distances=lcr.lc_distances.copy(),
                halofield_list=halofield_list,
                photon_nonconservation_data=photon_data,
                include_dvdr_in_tau21=False,
                apply_rsds=False,
                n_rsd_subcells=4,
                write=write_off,
                progressbar=False,
                lightcone_filename=None,
                global_quantities=tuple(task["global_quantities"]),
                cache=None,
                regenerate=True,
            ),
            maxlen=1,
        )[0][-1]

    return {
        "worker_idx": task["worker_idx"],
        "node_start": task["node_start"],
        "node_end": task["node_end"],
        "slice_idx": task["slice_idx"],
        "lightcones": {k: v[..., task["slice_idx"]].copy() for k, v in lc.lightcones.items()},
        "global_quantities": {
            k: np.asarray(v).copy() for k, v in (lc.global_quantities or {}).items()
        },
        "photon_nonconservation_data": lc.photon_nonconservation_data,
    }


def _run_parallel_lightcone(
    *,
    inputs: InputParameters,
    lightconer: RectilinearLightconer,
    lc_distances_orig: u.Quantity,
    global_quantities: tuple[str, ...],
    include_dvdr_in_tau21: bool,
    apply_rsds: bool,
    n_rsd_subcells: int,
    mode: Literal["strict", "fast"],
    eff_cuda: bool,
    lazy_d2h: bool,
    stream_perturbed_fields: bool,
    n_workers: int,
    gpu_ids: list[int] | None,
) -> LightCone:
    full_nodes = list(inputs.node_redshifts)
    n_nodes = len(full_nodes)
    n_workers = max(1, min(n_workers, n_nodes - 1))

    # Balanced partitioning of consecutive overlapping node intervals [s_w .. e_w]
    edges = np.linspace(0, n_nodes - 1, n_workers + 1, dtype=int)
    intervals: list[tuple[int, int]] = []
    for w in range(n_workers):
        s_w = int(edges[w])
        e_w = int(edges[w + 1])
        if e_w > s_w:
            intervals.append((s_w, e_w))

    with _mode_ctx(mode, use_cuda=eff_cuda, lazy_d2h=False):
        ic = compute_initial_conditions(inputs=inputs, write=False)

    stitched = setup_lightcone_instance(
        lightconer=lightconer,
        scrollz=full_nodes,
        inputs=inputs,
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
        photon_nonconservation_data={},
        global_quantities=global_quantities,
    )

    cosmo = inputs.cosmo_params.cosmo
    pixeleq = u.pixel_scale(inputs.simulation_options.cell_size / u.pixel)
    pixlcdist = lightconer.get_lc_distances_in_pixels(inputs.simulation_options.cell_size)

    with tempfile.TemporaryDirectory(prefix="21cmcu_plc_") as tmp_dir:
        ic_npz = Path(tmp_dir) / "ic.npz"
        ic_arrays = {name: ic.get(name) for name in ic.arrays if ic.has(name)}
        np.savez(ic_npz, **ic_arrays)

        tasks: list[dict[str, Any]] = []
        for w_idx, (s_w, e_w) in enumerate(intervals):
            z_sub = full_nodes[s_w : e_w + 1]
            dc_a = cosmo.comoving_distance(z_sub[0]).to(u.pixel, equivalencies=pixeleq)
            dc_b = cosmo.comoving_distance(z_sub[-1]).to(u.pixel, equivalencies=pixeleq)
            dc_lo = min(dc_a, dc_b)
            dc_hi = max(dc_a, dc_b)
            slice_idx = np.nonzero((pixlcdist >= dc_lo * (1.0 - 1e-6)) & (pixlcdist < dc_hi))[0]

            cuda_vis = None
            if eff_cuda and gpu_ids:
                cuda_vis = gpu_ids[w_idx % len(gpu_ids)]

            tasks.append(
                {
                    "worker_idx": w_idx,
                    "node_start": s_w,
                    "node_end": e_w,
                    "slice_idx": slice_idx,
                    "base_inputs_dict": inputs,
                    "node_redshifts": z_sub,
                    "ic_npz": str(ic_npz),
                    "lc_distances_mpc": lightconer.lc_distances.to_value(u.Mpc),
                    "quantities": list(lightconer.quantities),
                    "index_offset": int(lightconer.index_offset),
                    "oblique": isinstance(lightconer, ObliqueRectilinearLightconer),
                    "oblique_direction": (
                        tuple(lightconer.direction)
                        if isinstance(lightconer, ObliqueRectilinearLightconer)
                        else (3.0, 5.0, 11.0)
                    ),
                    "global_quantities": list(global_quantities),
                    "mode": mode,
                    "use_cuda": eff_cuda,
                    "lazy_d2h": lazy_d2h,
                    "stream_perturbed_fields": stream_perturbed_fields,
                    "cuda_visible": cuda_vis,
                }
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            results = pool.map(_lightcone_worker, tasks)

    for res in results:
        idx = res["slice_idx"]
        for q, arr in res["lightcones"].items():
            if idx.size > 0 and q in stitched.lightcones:
                stitched.lightcones[q][..., idx] = arr
        s_w = res["node_start"]
        e_w = res["node_end"]
        for q, gvals in res["global_quantities"].items():
            if q in stitched.global_quantities:
                stitched.global_quantities[q][s_w : e_w + 1] = gvals
        stitched.photon_nonconservation_data |= res["photon_nonconservation_data"]

    lowz_buf = int(np.sum(lc_distances_orig.min() > stitched.lightcone_distances))
    highz_buf = int(np.sum(lc_distances_orig.max() < stitched.lightcone_distances))
    return stitched._finalize_lightcone_at_last_redshift(
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
        n_rsd_subcells=n_rsd_subcells,
        inputs=inputs,
        lightconer=lightconer,
        lc_distances=lc_distances_orig,
        lowz_buffer_pixels=lowz_buf,
        highz_buffer_pixels=highz_buf,
        lightcone_filename=None,
    )


def quick_lightcone(
    z_min: float = 6.5,
    z_max: float = 11.0,
    *,
    box_len: float = 200.0,
    dim: int = 128,
    hii_dim: int | None = None,
    n_nodes: int | None = None,
    quantities: Sequence[str] = ("brightness_temp", "neutral_fraction"),
    global_quantities: Sequence[str] | None = None,
    seed: int = 42,
    mode: Literal["strict", "fast"] = "fast",
    use_cuda: bool | None = None,
    lazy_d2h: bool = True,
    stream_perturbed_fields: bool = True,
    use_ts_fluct: bool = False,
    recomb_model: str = "none",
    apply_rsds: bool = False,
    include_dvdr_in_tau21: bool = True,
    n_rsd_subcells: int = 4,
    devices: Any = None,
    parallel_redshifts: bool = False,
    n_workers: int | None = None,
    oblique: bool = False,
    oblique_direction: tuple[float, float, float] = (3.0, 5.0, 11.0),
    instrument: str | Any = None,
    mock_seed: int = 0,
    html_3d_path: str | Path | None = None,
    astro_params: dict[str, Any] | None = None,
    cosmo_params: dict[str, Any] | None = None,
    **extra_sim_kwargs: Any,
) -> LightCone:
    """Build and run a 21cm rectilinear lightcone in a single call with GPU optimizations.

    Combines ``lazy_d2h=True``, zero-D2H GPU global means, on-demand ``PerturbedField``
    streaming, optional oblique periodic box traversal (``oblique=True`` via
    :class:`ObliqueRectilinearLightconer`), optional multi-GPU/multi-worker redshift-parallel
    evolution (for history-independent EoR configurations), optional interferometer mock
    observation (``brightness_temp_mock``), and optional interactive 3D WebGL HTML export.
    """
    eff_hii = int(dim) // 2 if hii_dim is None else int(hii_dim)
    overrides = _split_input_overrides(astro_params, cosmo_params, extra_sim_kwargs)
    overrides.setdefault("DIM", int(dim))
    overrides.setdefault("HII_DIM", eff_hii)
    overrides.setdefault("BOX_LEN", float(box_len))
    overrides.setdefault("USE_TS_FLUCT", bool(use_ts_fluct))
    overrides.setdefault("RECOMB_MODEL", str(recomb_model))
    if oblique and (apply_rsds or include_dvdr_in_tau21):
        overrides.setdefault("KEEP_3D_VELOCITIES", True)

    custom_nodes = overrides.pop("node_redshifts", None)
    base_inputs = InputParameters(random_seed=int(seed)).evolve_input_structs(**overrides)
    cosmo = base_inputs.cosmo_params.cosmo

    q_tuple = tuple(quantities)
    g_tuple = tuple(q_tuple) if global_quantities is None else tuple(global_quantities)
    res_mpc = float(box_len) / float(eff_hii)
    if oblique:
        lcr: RectilinearLightconer = ObliqueRectilinearLightconer.between_redshifts(
            min_redshift=float(z_min),
            max_redshift=float(z_max),
            resolution=res_mpc * u.Mpc,
            quantities=q_tuple,
            cosmo=cosmo,
            direction=oblique_direction,
        )
    else:
        lcr = RectilinearLightconer.between_redshifts(
            min_redshift=float(z_min),
            max_redshift=float(z_max),
            resolution=res_mpc * u.Mpc,
            quantities=q_tuple,
            cosmo=cosmo,
        )

    if custom_nodes is not None:
        node_z = [float(z) for z in custom_nodes]
    else:
        pad_mpc = (35.0 if apply_rsds else 0.0) + max(2.5 * res_mpc, 5.0)
        d_lo_mpc = max(10.0, float(lcr.lc_distances.min().to_value(u.Mpc)) - pad_mpc)
        d_hi_mpc = float(lcr.lc_distances.max().to_value(u.Mpc)) + pad_mpc
        z_lo_pad = min(float(z_min), float(z_at_value(cosmo.comoving_distance, d_lo_mpc * u.Mpc).value))
        z_hi_pad = max(float(z_max), float(z_at_value(cosmo.comoving_distance, d_hi_mpc * u.Mpc).value))
        node_z = _build_node_redshifts(z_lo_pad, z_hi_pad, n_nodes=n_nodes)

    inputs = base_inputs.clone(node_redshifts=node_z)

    eff_cuda = (
        bool(lib.p21_cuda_is_compiled() and lib.p21_cuda_is_available())
        if use_cuda is None
        else bool(use_cuda)
    )

    causal_history = bool(
        inputs.evolution_required
        or inputs.matter_options.has_discrete_halos
        or inputs.astro_options.PHOTON_CONS_TYPE != "no-photoncons"
    )
    lc_dist_orig = lcr.lc_distances.copy()
    validated_lcr = lcr.validate_options(
        inputs=inputs,
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
    )

    gpu_ids: list[int] | None = None
    if devices is not None:
        with _mode_ctx(mode, use_cuda=eff_cuda, devices=devices):
            gpu_ids = list(range(int(lib.p21_cuda_device_count()))) if eff_cuda else None
            if isinstance(config["CUDA_DEVICES"], (tuple, list)):
                gpu_ids = list(config["CUDA_DEVICES"])
    elif eff_cuda:
        n_dev = int(lib.p21_cuda_device_count())
        gpu_ids = list(range(n_dev)) if n_dev > 0 else [0]

    eff_workers = n_workers
    if eff_workers is None:
        eff_workers = (len(gpu_ids) if gpu_ids else 1) if eff_cuda else 1

    if parallel_redshifts and not causal_history and eff_workers >= 2 and len(node_z) >= 3:
        lc = _run_parallel_lightcone(
            inputs=inputs,
            lightconer=validated_lcr,
            lc_distances_orig=lc_dist_orig,
            global_quantities=g_tuple,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            n_rsd_subcells=n_rsd_subcells,
            mode=mode,
            eff_cuda=eff_cuda,
            lazy_d2h=lazy_d2h,
            stream_perturbed_fields=stream_perturbed_fields,
            n_workers=eff_workers,
            gpu_ids=gpu_ids,
        )
    else:
        with _mode_ctx(mode, use_cuda=eff_cuda, lazy_d2h=lazy_d2h, devices=devices):
            lc = run_lightcone(
                lightconer=lcr,
                inputs=inputs,
                include_dvdr_in_tau21=include_dvdr_in_tau21,
                apply_rsds=apply_rsds,
                n_rsd_subcells=n_rsd_subcells,
                global_quantities=g_tuple,
                stream_perturbed_fields=stream_perturbed_fields,
                write=False,
            )

    if instrument is not None:
        from .instrument import mock_observation_3d

        tb_key = "brightness_temp_with_rsds" if "brightness_temp_with_rsds" in lc.lightcones else "brightness_temp"
        if tb_key in lc.lightcones:
            mid_z = float(np.median(lc.lightcone_redshifts))
            lc.lightcones["brightness_temp_mock"] = mock_observation_3d(
                lc.lightcones[tb_key],
                box_length=lc.lightcone_dimensions,
                redshift=mid_z,
                instrument=instrument,
                seed=mock_seed,
                use_cuda=eff_cuda,
            )

    if html_3d_path is not None:
        from .viz import export_lightcone_3d_html

        out_qty = "brightness_temp_mock" if "brightness_temp_mock" in lc.lightcones else "brightness_temp"
        export_lightcone_3d_html(lc, html_3d_path, quantity=out_qty)

    return lc


__all__ = ["quick_coeval", "quick_lightcone"]
