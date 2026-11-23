"""Prepare bounded batches of HDF5 shuffle/LZF chunks on CUDA."""

from pathlib import Path

import cupy as cp
import numpy as np


class DeviceLZFWriter:
    def __init__(self, workers=None):
        module = cp.RawModule(code=Path(__file__).with_suffix(".cu").read_text())
        self.shuffle, self.encode, self.pack = [
            module.get_function(name)
            for name in ("shuffle_chunks", "encode_segments", "pack_segments")
        ]
        self.chunks = 0
        self.skipped_lzf_chunks = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write(self, fields):
        for group, name, values, chunks in fields:
            if values.dtype != np.float32 or values.ndim != 3:
                raise ValueError("Device lightcone writer expects 3D float32 fields")
            source = cp.ascontiguousarray(cp.asarray(values))
            if not bool(cp.isfinite(source).all()):
                raise ValueError(f"Nonfinite lightcone field {name}")
            ds = group.create_dataset(
                name,
                shape=values.shape,
                dtype=values.dtype,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            grid = tuple((n + c - 1) // c for n, c in zip(values.shape, chunks))
            total = int(np.prod(grid))
            size = int(np.prod(chunks)) * 4
            segment = 2048
            # A stream of alternating short matches and single literals can use
            # one length byte per literal. Two bytes per input byte bounds it.
            stride = 2 * segment
            ns = (size + segment - 1) // segment
            for first in range(0, total, 32):
                batch = min(32, total - first)
                raw = cp.empty((batch, size), cp.uint8)
                self.shuffle(
                    ((raw.size + 255) // 256,),
                    (256,),
                    (
                        source,
                        raw,
                        *map(
                            np.int32,
                            (
                                values.shape[1],
                                values.shape[2],
                                values.shape[0],
                                *chunks,
                                first,
                                raw.size,
                                *grid[1:],
                            ),
                        ),
                    ),
                )
                count = batch * ns
                tokens = cp.empty((count, stride), cp.uint8)
                sizes = cp.empty(count, cp.int32)
                self.encode(
                    ((count + 31) // 32,),
                    (32,),
                    (
                        raw,
                        tokens,
                        sizes,
                        *map(np.int32, (segment, stride, count, size)),
                    ),
                )
                ends = cp.cumsum(sizes, dtype=cp.int32)
                stops = ends[ns - 1 :: ns].get()
                packed = cp.empty(int(stops[-1]), cp.uint8)
                self.pack(
                    (count,),
                    (64,),
                    (
                        tokens,
                        ends - sizes,
                        sizes,
                        packed,
                        np.int32(stride),
                        np.int32(count),
                    ),
                )
                data = packed.get()
                start = 0
                for j, stop in enumerate(stops):
                    skip = int(stop) - start >= size
                    payload = raw[j].get() if skip else data[start:stop]
                    offset = tuple(
                        int(i * c)
                        for i, c in zip(np.unravel_index(first + j, grid), chunks)
                    )
                    ds.id.write_direct_chunk(
                        offset, payload, filter_mask=2 if skip else 0
                    )
                    self.chunks += 1
                    self.skipped_lzf_chunks += int(skip)
                    start = int(stop)
