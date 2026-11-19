"""Prepare numeric shuffle/LZF chunks in parallel and write them with HDF5."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes.util import find_library
from itertools import product
from time import thread_time

import h5py.h5z
import numpy as np

LZF_SKIPPED = 2


class LZFEncoder:
    def __init__(self):
        library_path = find_library("lzf")
        locations = [h5py.h5z.__file__]
        if library_path:
            locations.append(library_path)
        for location in locations:
            library = ctypes.CDLL(location)
            if hasattr(library, "lzf_compress"):
                self.compress = library.lzf_compress
                break
        else:
            raise RuntimeError("No liblzf compression symbol available")
        self.compress.argtypes = (
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
        )
        self.compress.restype = ctypes.c_uint

    def encode(self, name, values, chunks):
        started = thread_time()
        output = []
        for offset in product(*(range(0, size, step) for size, step in zip(values.shape, chunks))):
            sub = values[tuple(slice(start, start + step) for start, step in zip(offset, chunks))]
            if sub.shape != chunks:
                padded = np.zeros(chunks, dtype=values.dtype)
                padded[tuple(slice(n) for n in sub.shape)] = sub
                sub = padded
            raw = np.ascontiguousarray(sub).view(np.uint8).reshape(-1, values.dtype.itemsize)
            shuffled = np.ascontiguousarray(raw.T)
            # A skipped LZF filter still leaves the HDF5 byte-shuffle applied.
            compressed = bytearray(shuffled.nbytes)
            destination = ctypes.addressof(ctypes.c_char.from_buffer(compressed))
            size = self.compress(shuffled.ctypes.data, shuffled.nbytes, destination, shuffled.nbytes)
            if size:
                output.append((offset, memoryview(compressed)[:size], 0))
            else:
                output.append((offset, shuffled, LZF_SKIPPED))
        return name, values.shape, values.dtype, chunks, output, thread_time() - started


class ParallelLZFWriter:
    def __init__(self, workers):
        self.workers = workers
        self.encoder = LZFEncoder()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.encoder_cpu_seconds = 0.0
        self.chunks = 0
        self.skipped_lzf_chunks = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def write(self, fields):
        pending = deque()

        def drain():
            group, future = pending.popleft()
            name, shape, dtype, chunks, data, work_seconds = future.result()
            node = group.create_dataset(
                name, shape=shape, dtype=dtype, chunks=chunks, compression="lzf", shuffle=True,
            )
            pipeline = node.id.get_create_plist()
            ids = [pipeline.get_filter(i)[0] for i in range(pipeline.get_nfilters())]
            if ids != [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_LZF]:
                raise RuntimeError(f"Unexpected HDF5 filter pipeline {ids}")
            for offset, payload, mask in data:
                node.id.write_direct_chunk(offset, payload, filter_mask=mask)
                self.chunks += 1
                self.skipped_lzf_chunks += bool(mask)
            self.encoder_cpu_seconds += work_seconds

        for group, name, values, chunks in fields:
            pending.append((group, self.pool.submit(self.encoder.encode, name, values, chunks)))
            if len(pending) == 2 * self.workers:
                drain()
        while pending:
            drain()
