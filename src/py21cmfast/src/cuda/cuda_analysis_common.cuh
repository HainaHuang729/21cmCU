#ifndef P21_CUDA_ANALYSIS_COMMON_CUH
#define P21_CUDA_ANALYSIS_COMMON_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <stddef.h>

#include "cuda_common.cuh"

/* Unified grid-stride loop avoiding > 65535 * 256 tail-element truncation. */
#define P21_GRID_STRIDE_LOOP(idx, total)                                           \
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x,               \
                _p21_stride_##idx = (size_t)gridDim.x * blockDim.x;                \
         idx < (size_t)(total); idx += _p21_stride_##idx)

inline int p21_grid_blocks(size_t total, int threads = 256) {
    if (total == 0 || threads <= 0) {
        return 0;
    }
    const size_t max_blocks = 65535ULL;
    const size_t blocks = (total + (size_t)threads - 1) / (size_t)threads;
    return (int)std::min(blocks, max_blocks);
}

struct DeviceFloatView {
    const float *d_ptr = nullptr;
    bool owned = false;
};

inline DeviceFloatView acquire_float_view(const float *h_ptr, size_t nbytes,
                                          cudaStream_t stream) {
    DeviceFloatView view{};
    if (!h_ptr || nbytes == 0) {
        return view;
    }
    if (nbytes <= p21_cuda_registry_max_entry_bytes()) {
        view.d_ptr = (const float *)p21_cuda_registry_get_or_upload(h_ptr, nbytes, stream);
        if (view.d_ptr) {
            view.owned = false;
            return view;
        }
    }
    float *d_tmp = (float *)p21_cuda_pool_alloc(nbytes, stream);
    if (!d_tmp) {
        return view;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_tmp, h_ptr, nbytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_tmp, stream);
        return view;
    }
    p21_cuda_stats_mut()->registry_h2d_bytes += (unsigned long long int)nbytes;
    view.d_ptr = d_tmp;
    view.owned = true;
    return view;
}

inline void release_float_view(DeviceFloatView &view, cudaStream_t stream) {
    if (view.owned && view.d_ptr) {
        p21_cuda_pool_free(const_cast<float *>(view.d_ptr), stream);
    }
    view.d_ptr = nullptr;
    view.owned = false;
}

class ScopedFloatView {
   public:
    ScopedFloatView() = default;
    ScopedFloatView(const float *h_ptr, size_t nbytes, cudaStream_t stream)
        : view_(acquire_float_view(h_ptr, nbytes, stream)), stream_(stream) {}
    ScopedFloatView(const ScopedFloatView &) = delete;
    ScopedFloatView &operator=(const ScopedFloatView &) = delete;
    ~ScopedFloatView() { release(); }

    void reset(const float *h_ptr, size_t nbytes, cudaStream_t stream) {
        release();
        stream_ = stream;
        view_ = acquire_float_view(h_ptr, nbytes, stream);
    }

    void release() { release_float_view(view_, stream_); }
    const float *ptr() const { return view_.d_ptr; }
    explicit operator bool() const { return view_.d_ptr != nullptr; }

   private:
    DeviceFloatView view_{};
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class ScopedPoolBuf {
   public:
    ScopedPoolBuf() = default;
    ScopedPoolBuf(size_t n_elems, cudaStream_t stream) { allocate(n_elems, stream); }
    ScopedPoolBuf(const ScopedPoolBuf &) = delete;
    ScopedPoolBuf &operator=(const ScopedPoolBuf &) = delete;
    ~ScopedPoolBuf() { release(); }

    bool allocate(size_t n_elems, cudaStream_t stream) {
        release();
        stream_ = stream;
        if (n_elems == 0) {
            return true;
        }
        ptr_ = (T *)p21_cuda_pool_alloc(n_elems * sizeof(T), stream);
        return ptr_ != nullptr;
    }

    void release() {
        if (ptr_) {
            p21_cuda_pool_free(ptr_, stream_);
            ptr_ = nullptr;
        }
    }

    T *ptr() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

   private:
    T *ptr_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

#endif /* P21_CUDA_ANALYSIS_COMMON_CUH */
