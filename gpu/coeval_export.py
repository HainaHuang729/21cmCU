"""Persist all evolved coeval fields for the requested output redshifts."""

import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np


def export_evolved(boxes, config, path, compression, *, workers=1):
    path = Path(path)
    start = perf_counter()
    count = 0
    uncompressed_bytes = 0
    field_schema = {}
    export_stats = {}

    def fields(archive):
        nonlocal count, uncompressed_bytes
        for index, box in enumerate(boxes):
            group = archive.create_group(f"z{index:02d}")
            group.attrs["redshift"] = box.redshift
            parts = (box.perturb_struct, box.spin_temp_struct, box.ionization_struct,
                     box.brightness_temp_struct)
            field_names = set()
            for part in parts:
                if part is None:
                    continue
                for field in part._get_box_structures():
                    if field == "J_21_LW_box" and not box.flag_options.USE_MINI_HALOS:
                        continue
                    if field in field_names:
                        raise ValueError(f"Duplicate {field} at z={box.redshift}")
                    field_names.add(field)
                    values = np.asarray(getattr(part, field))
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {field} at z={box.redshift}")
                    schema = {"shape": list(values.shape), "dtype": str(values.dtype)}
                    if field_schema.setdefault(field, schema) != schema:
                        raise ValueError(f"Inconsistent field shape or dtype for {field}")
                    chunks = tuple(min(n, b) for n, b in zip(values.shape, (1, 16, 64, 64)[-values.ndim:]))
                    count += 1
                    uncompressed_bytes += values.nbytes
                    yield group, field, values, chunks
                for field in ("mean_f_coll", "mean_f_coll_MINI", "log10_Mturnover_ave",
                              "log10_Mturnover_MINI_ave"):
                    if hasattr(part, field):
                        group.attrs[field] = getattr(part, field)
            if field_names != set(field_schema):
                raise ValueError(f"Missing field at z={box.redshift}")

    with h5py.File(path, "w") as archive:
        archive.attrs["schema_version"] = 1
        archive.attrs["point"] = config["index"]
        archive.attrs["seed"] = config["seed"]
        archive.attrs["configuration_json"] = json.dumps(config)
        archive.create_dataset("redshifts", data=config["z"])
        if workers > 1:
            if compression != "lzf":
                raise ValueError("Parallel encoding requires the LZF filter")
            from parallel_lzf import ParallelLZFWriter
            with ParallelLZFWriter(workers) as writer:
                writer.write(fields(archive))
                export_stats = {"workers": workers, "encoder_cpu_seconds": writer.encoder_cpu_seconds,
                                "encoded_chunks": writer.chunks, "skipped_lzf_chunks": writer.skipped_lzf_chunks}
        else:
            for group, field, values, chunks in fields(archive):
                options = {"compression": "lzf", "shuffle": True, "chunks": chunks} if compression == "lzf" else {}
                group.create_dataset(field, data=values, **options)
    if len(boxes) != len(config["z"]) or count != len(boxes) * len(field_schema):
        raise ValueError("The full coeval archive is incomplete")
    return {"archive_seconds": perf_counter()-start, "archive_bytes": path.stat().st_size,
            "raw_field_bytes": uncompressed_bytes, "field_count": len(field_schema),
            "dataset_count": count, "redshift_count": len(boxes), "fields": field_schema,
            "compression": compression, **export_stats}
