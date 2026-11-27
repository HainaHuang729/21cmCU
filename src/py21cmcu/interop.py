"""Zero-copy CUDA device export via DLPack and __cuda_array_interface__."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np

from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

if TYPE_CHECKING:
    from py21cmfast.drivers.lightcone import LightCone

_K_DL_CUDA = 2
_K_DL_FLOAT = 2

_PYCAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PYCAPSULE_NEW.restype = ctypes.py_object
_PYCAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_PYCAPSULE_GET_NAME = ctypes.pythonapi.PyCapsule_GetName
_PYCAPSULE_GET_NAME.restype = ctypes.c_char_p
_PYCAPSULE_GET_NAME.argtypes = [ctypes.c_void_p]

_PYCAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
_PYCAPSULE_GET_POINTER.restype = ctypes.c_void_p
_PYCAPSULE_GET_POINTER.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DELETER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DELETER_FUNC),
    ]


_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_ACTIVE_MANAGED: dict[int, tuple[_DLManagedTensor, Any, Any, Any, int]] = {}


@_DELETER_FUNC
def _dlpack_managed_deleter(managed_ptr: int) -> None:
    entry = _ACTIVE_MANAGED.pop(int(managed_ptr), None)
    if entry is None:
        return
    ptr_val = entry[4]
    if ptr_val:
        _lib.p21_cuda_free_exported_device_ptr(int(ptr_val))


@_CAPSULE_DESTRUCTOR
def _dlpack_capsule_destructor(capsule_void: int) -> None:
    if not capsule_void:
        return
    name = _PYCAPSULE_GET_NAME(capsule_void)
    if name == b"dltensor":
        managed_ptr = _PYCAPSULE_GET_POINTER(capsule_void, b"dltensor")
        if managed_ptr:
            _dlpack_managed_deleter(int(managed_ptr))


class CudaFieldView:
    """Device-memory handle for a float32 21cmCU array supporting DLPack & CUDA array interface."""

    def __init__(self, shape: tuple[int, ...], ptr: int, device_id: int) -> None:
        self._shape = tuple(int(s) for s in shape)
        self._ptr = int(ptr)
        self._device_id = int(device_id)
        self._owns_ptr = True
        self._closed = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def ptr(self) -> int:
        if self._closed or self._ptr == 0:
            raise RuntimeError("CudaFieldView buffer has been closed or transferred")
        return self._ptr

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "shape": self._shape,
            "typestr": "<f4",
            "data": (self.ptr, False),
            "version": 3,
        }

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_K_DL_CUDA, self._device_id)

    def __dlpack__(self, *, stream: int | None = None) -> Any:
        del stream  # export is already synchronized on the originating stream
        if not self._owns_ptr or self._closed or self._ptr == 0:
            raise RuntimeError(
                "DLPack ownership was already transferred or this CudaFieldView was closed"
            )
        ndim = len(self._shape)
        shape_arr = (ctypes.c_int64 * ndim)(*self._shape)
        strides_list = []
        stride = 1
        for dim in reversed(self._shape):
            strides_list.append(stride)
            stride *= dim
        strides_list.reverse()
        strides_arr = (ctypes.c_int64 * ndim)(*strides_list)

        managed = _DLManagedTensor()
        managed.dl_tensor.data = ctypes.c_void_p(self._ptr)
        managed.dl_tensor.device = _DLDevice(_K_DL_CUDA, self._device_id)
        managed.dl_tensor.ndim = ndim
        managed.dl_tensor.dtype = _DLDataType(_K_DL_FLOAT, 32, 1)
        managed.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
        managed.dl_tensor.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int64))
        managed.dl_tensor.byte_offset = 0
        managed.manager_ctx = None
        managed.deleter = _dlpack_managed_deleter

        addr = ctypes.addressof(managed)
        _ACTIVE_MANAGED[addr] = (managed, shape_arr, strides_arr, _dlpack_managed_deleter, self._ptr)
        self._owns_ptr = False
        self._ptr = 0
        return _PYCAPSULE_NEW(
            addr,
            b"dltensor",
            ctypes.cast(_dlpack_capsule_destructor, ctypes.c_void_p),
        )

    def to_numpy(self) -> np.ndarray:
        """Copy the exported CUDA buffer back to a contiguous float32 NumPy array."""
        p = self.ptr
        out = np.empty(self._shape, dtype=np.float32)
        rc = _lib.p21_cuda_device_copy_to_host_f32(
            int(p),
            _ffi.cast("float *", _ffi.from_buffer(out)),
            int(out.size),
        )
        if rc != 0:
            raise RuntimeError("Failed to copy exported CUDA buffer back to host")
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_ptr and self._ptr != 0:
            _lib.p21_cuda_free_exported_device_ptr(int(self._ptr))
        self._owns_ptr = False
        self._ptr = 0

    def __enter__(self) -> CudaFieldView:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def to_cuda_view(field: np.ndarray) -> CudaFieldView:
    """Export a 21cmCU float32 array as a CudaFieldView without D2H if already GPU-resident."""
    if _lib.p21_cuda_is_compiled() == 0 or _lib.p21_cuda_is_available() == 0:
        raise RuntimeError("Native CUDA runtime is required for to_cuda_view()")
    arr = np.ascontiguousarray(field, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot export an empty array to CUDA")
    d_ptr = _ffi.new("unsigned long long *")
    dev_id = _ffi.new("int *")
    rc = _lib.p21_cuda_export_device_f32(
        _ffi.cast("const float *", _ffi.from_buffer(arr)),
        int(arr.size),
        d_ptr,
        dev_id,
    )
    if rc != 0 or int(d_ptr[0]) == 0:
        raise RuntimeError("Failed to export field to standalone CUDA device buffer")
    return CudaFieldView(arr.shape, int(d_ptr[0]), int(dev_id[0]))


def lightcone_to_cuda_views(
    lightcone: LightCone,
    quantities: tuple[str, ...] = ("brightness_temp",),
) -> dict[str, CudaFieldView]:
    """Export selected lightcone fields directly to GPU-resident CudaFieldView handles."""
    return {qty: to_cuda_view(lightcone.lightcones[qty]) for qty in quantities}


def to_torch(field: np.ndarray | CudaFieldView) -> Any:
    """Convert a 21cmCU field or CudaFieldView to a GPU PyTorch tensor via zero-copy DLPack."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for py21cmcu.to_torch(); install torch with CUDA support."
        ) from exc

    view = field if isinstance(field, CudaFieldView) else to_cuda_view(field)
    return torch.from_dlpack(view)


__all__ = [
    "CudaFieldView",
    "lightcone_to_cuda_views",
    "to_cuda_view",
    "to_torch",
]
