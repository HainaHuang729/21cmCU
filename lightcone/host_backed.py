"""Bound device-side history using the native host snapshots."""
import ctypes as C
import cupy as cp
from gpu_heat_runtime import arr

class LatestSnapshots(dict):
    def __init__(self):
        super().__init__()
        self.evictions = 0

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > 12:
            del self[next(iter(self))]
            self.evictions += 1

def resident_or_host(cache, pointer, count):
    addr = C.cast(pointer, C.c_void_p).value
    value = cache.get(addr)
    if value is None:
        value = cp.asarray(arr(pointer, count))
        cache[addr] = value
    return value

def resident_box(cache, host):
    value = cache.get(host.ctypes.data)
    return value if value is not None else cp.asarray(host)
