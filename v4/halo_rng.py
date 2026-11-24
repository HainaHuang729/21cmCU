"""Ordered GSL property draws for the 4.3.2 resolved DexM catalogue."""

import cupy as cp
import numpy as np

from ic_random import RNG_ORDER, Stream, gaussian, stream_module


def _thread_seeds(module, seed, threads):
    master = Stream(module, "mt", seed)
    words = np.empty(0, dtype=np.uint32)
    pos = 0
    while True:
        if pos + threads > words.size:
            tail = words[pos:]
            nxt = master.integers(1).get()
            words = np.concatenate((tail, nxt))
            pos = 0
        # sample_n_unique_integers draws `threads` words and retries the full tuple on any duplicate.
        cand = [int(w) for w in words[pos : pos + threads]]
        pos += threads
        if len(set(cand)) == threads:
            return cand


def assign_properties(seed, threads, workers, targets):
    """Fill the native star, SFR and X-ray float arrays in static OpenMP order."""
    if not 1 <= workers <= threads:
        raise ValueError("Expected at least one worker and at most N_THREADS streams")
    count = len(targets[0])
    if not count:
        return
    module = stream_module()
    seeds = _thread_seeds(module, seed, threads)
    quotient, extra = divmod(count, workers)
    for thread, subseed in enumerate(seeds[:workers]):
        size = quotient + (thread < extra)
        if not size:
            continue
        start = thread * quotient + min(thread, extra)
        stream = Stream(module, RNG_ORDER[thread % 5], subseed)
        values = gaussian(stream, 3 * size).astype("f").reshape(size, 3)
        for index, target in enumerate(targets):
            cp.ascontiguousarray(values[:, index]).get(out=target[start : start + size])
