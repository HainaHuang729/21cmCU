"""Bounded LRU registry pairing host NumPy arrays with resident CUDA copies."""

from collections import OrderedDict
import ctypes as C

import cupy as cp
import numpy as np


class DeviceFieldRegistry:
    """Reuse resident CuPy arrays across 4.3.2 stages by host buffer address."""

    def __init__(self, *, max_entries: int = 128, budget_fraction: float = 0.25):
        self.max_entries = max_entries
        free_bytes = cp.cuda.Device().mem_info[0]
        self.budget_bytes = max(int(free_bytes * budget_fraction), 64 << 20)
        self._entries: OrderedDict[int, tuple[object, cp.ndarray]] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.hit_bytes = 0
        self.misses = 0
        self.miss_bytes = 0
        self.records = 0
        self.evictions = 0

    def _evict_until_fit(self, incoming: int) -> None:
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.bytes + incoming > self.budget_bytes
        ):
            _, (_, evicted) = self._entries.popitem(last=False)
            self.bytes -= evicted.nbytes
            self.evictions += 1

    def record(self, host_array: np.ndarray, device_array: cp.ndarray) -> None:
        """Register an existing NumPy array and its matching GPU buffer."""
        if not isinstance(host_array, np.ndarray) or not host_array.flags.c_contiguous:
            return
        if host_array.size == 0 or device_array.size != host_array.size:
            return
        if device_array.nbytes > self.budget_bytes:
            return
        addr = int(host_array.ctypes.data)
        existing = self._entries.pop(addr, None)
        if existing is not None:
            self.bytes -= existing[1].nbytes
        self._evict_until_fit(device_array.nbytes)
        dev = cp.ascontiguousarray(device_array)
        self._entries[addr] = (host_array, dev)
        self.bytes += dev.nbytes
        self.records += 1

    def record_ptr(self, addr: int, owner: object, device_array: cp.ndarray) -> None:
        """Register a raw C pointer paired with a Python owner keeping it alive."""
        if not addr or owner is None or device_array.size == 0:
            return
        if device_array.nbytes > self.budget_bytes:
            return
        addr = int(addr)
        existing = self._entries.pop(addr, None)
        if existing is not None:
            self.bytes -= existing[1].nbytes
        self._evict_until_fit(device_array.nbytes)
        dev = cp.ascontiguousarray(device_array)
        self._entries[addr] = (owner, dev)
        self.bytes += dev.nbytes
        self.records += 1

    def lookup_ptr(self, addr: int, size: int, dtype) -> cp.ndarray | None:
        """Look up a resident GPU vector by raw host pointer if shape/dtype match."""
        entry = self._entries.get(int(addr))
        if entry is None:
            return None
        _, dev = entry
        target_dtype = cp.dtype(dtype)
        if dev.size != size or dev.dtype != target_dtype:
            return None
        self._entries.move_to_end(int(addr))
        self.hits += 1
        self.hit_bytes += dev.nbytes
        return dev.reshape(-1)

    def lookup_host(self, host_array: np.ndarray, *, dtype=None) -> cp.ndarray:
        """Return the cached GPU copy of `host_array`, uploading only on cache miss."""
        target_dtype = cp.dtype(host_array.dtype if dtype is None else dtype)
        if isinstance(host_array, np.ndarray) and host_array.flags.c_contiguous:
            addr = int(host_array.ctypes.data)
            entry = self._entries.get(addr)
            if entry is not None:
                owner, dev = entry
                if (
                    dev.size == host_array.size
                    and dev.dtype == cp.dtype(host_array.dtype)
                    and (
                        not isinstance(owner, np.ndarray)
                        or (
                            owner.shape == host_array.shape
                            or dev.size == host_array.size
                        )
                    )
                ):
                    self._entries.move_to_end(addr)
                    self.hits += 1
                    self.hit_bytes += dev.nbytes
                    out = dev.reshape(host_array.shape)
                    return (
                        out if dev.dtype == target_dtype else out.astype(target_dtype)
                    )
        uploaded = cp.asarray(host_array, dtype=target_dtype)
        self.misses += 1
        self.miss_bytes += uploaded.nbytes
        return uploaded

    def invalidate_ptr(self, addr: int) -> None:
        """Remove a single pointer entry if its backing host buffer is invalidated."""
        entry = self._entries.pop(int(addr), None)
        if entry is not None:
            self.bytes -= entry[1].nbytes

    def clear(self) -> None:
        """Release all cached device buffers."""
        self._entries.clear()
        self.bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "bytes": self.bytes,
            "budget_bytes": self.budget_bytes,
            "hits": self.hits,
            "hit_bytes": self.hit_bytes,
            "misses": self.misses,
            "miss_bytes": self.miss_bytes,
            "records": self.records,
            "evictions": self.evictions,
        }


def c_ptr_addr(pointer) -> int:
    return int(C.cast(pointer, C.c_void_p).value or 0)
