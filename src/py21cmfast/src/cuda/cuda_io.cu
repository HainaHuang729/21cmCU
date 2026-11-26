#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cuda_common.cuh"

namespace {

constexpr size_t kLzfSegmentBytes = 2048ULL;
constexpr size_t kLzfSegStride = kLzfSegmentBytes + (kLzfSegmentBytes / 32ULL) + 64ULL;
constexpr int kLzfHashBuckets = 1024;
constexpr int kLzfThreads = 256;

__global__ void lzf_prep_f32_kernel(const float *__restrict__ in_f32, size_t n_chunks,
                                    size_t chunk_elems, int shuffle,
                                    unsigned char *__restrict__ out_bytes,
                                    int *__restrict__ has_nonfinite) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = n_chunks * chunk_elems;
    if (idx >= total) {
        return;
    }

    float v = in_f32[idx];
    if (!isfinite(v)) {
        atomicOr(has_nonfinite, 1);
    }

    uint32_t bits = __float_as_uint(v);
    size_t chunk = idx / chunk_elems;
    size_t elem = idx - chunk * chunk_elems;
    size_t chunk_bytes = chunk_elems * sizeof(float);
    unsigned char *dst = out_bytes + chunk * chunk_bytes;

    if (shuffle) {
        dst[0 * chunk_elems + elem] = (unsigned char)(bits & 0xffU);
        dst[1 * chunk_elems + elem] = (unsigned char)((bits >> 8) & 0xffU);
        dst[2 * chunk_elems + elem] = (unsigned char)((bits >> 16) & 0xffU);
        dst[3 * chunk_elems + elem] = (unsigned char)((bits >> 24) & 0xffU);
    } else {
        size_t base = elem * 4ULL;
        dst[base + 0] = (unsigned char)(bits & 0xffU);
        dst[base + 1] = (unsigned char)((bits >> 8) & 0xffU);
        dst[base + 2] = (unsigned char)((bits >> 16) & 0xffU);
        dst[base + 3] = (unsigned char)((bits >> 24) & 0xffU);
    }
}

__device__ size_t lzf_emit_literals(const unsigned char *in, int start, int end,
                                    unsigned char *out, size_t op, size_t out_cap) {
    int pos = start;
    while (pos < end) {
        int run = end - pos;
        if (run > 32) {
            run = 32;
        }
        if (op + 1ULL + (size_t)run <= out_cap) {
            out[op++] = (unsigned char)(run - 1);
            for (int i = 0; i < run; ++i) {
                out[op++] = in[pos + i];
            }
        } else {
            op += 1ULL + (size_t)run;
        }
        pos += run;
    }
    return op;
}

__global__ void lzf_compress_segments_kernel(const unsigned char *__restrict__ in_bytes,
                                             size_t n_chunks, size_t chunk_bytes, size_t n_segs,
                                             unsigned char *__restrict__ seg_out,
                                             uint32_t *__restrict__ seg_len_out,
                                             uint16_t *__restrict__ htab_scratch) {
    size_t seg_global = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total_segs = n_chunks * n_segs;
    if (seg_global >= total_segs) {
        return;
    }

    size_t chunk = seg_global / n_segs;
    size_t seg = seg_global - chunk * n_segs;
    size_t seg_offset = seg * kLzfSegmentBytes;
    int seg_len = (int)min(kLzfSegmentBytes, chunk_bytes - seg_offset);

    const unsigned char *in = in_bytes + chunk * chunk_bytes + seg_offset;
    unsigned char *out = seg_out + seg_global * kLzfSegStride;
    uint16_t *htab = htab_scratch + seg_global * (size_t)kLzfHashBuckets;

    for (int i = 0; i < kLzfHashBuckets; ++i) {
        htab[i] = 0;
    }

    size_t op = 0;
    int ip = 0;
    int lit_start = 0;

    while (ip + 2 < seg_len) {
        uint32_t hval = ((uint32_t)in[ip] << 16) | ((uint32_t)in[ip + 1] << 8) | (uint32_t)in[ip + 2];
        uint32_t hidx = (((hval ^ (hval >> 5)) * 2654435761U) >> 22) & (uint32_t)(kLzfHashBuckets - 1);
        uint16_t prev = htab[hidx];
        htab[hidx] = (uint16_t)(ip + 1);

        if (prev != 0) {
            int ref = (int)prev - 1;
            int off = ip - ref;
            if (off > 0 && off <= 8192 && in[ref] == in[ip] && in[ref + 1] == in[ip + 1] &&
                in[ref + 2] == in[ip + 2]) {
                op = lzf_emit_literals(in, lit_start, ip, out, op, kLzfSegStride);

                int max_match = min(264, seg_len - ip);
                int mlen = 3;
                while (mlen < max_match && in[ref + mlen] == in[ip + mlen]) {
                    mlen++;
                }

                uint32_t off_m1 = (uint32_t)(off - 1);
                if (mlen <= 8) {
                    if (op + 2ULL <= kLzfSegStride) {
                        out[op++] = (unsigned char)(((mlen - 2) << 5) | (off_m1 >> 8));
                        out[op++] = (unsigned char)(off_m1 & 0xffU);
                    } else {
                        op += 2ULL;
                    }
                } else {
                    if (op + 3ULL <= kLzfSegStride) {
                        out[op++] = (unsigned char)((7U << 5) | (off_m1 >> 8));
                        out[op++] = (unsigned char)(mlen - 9);
                        out[op++] = (unsigned char)(off_m1 & 0xffU);
                    } else {
                        op += 3ULL;
                    }
                }

                ip += mlen;
                lit_start = ip;
                continue;
            }
        }
        ip++;
        if (ip - lit_start == 32) {
            op = lzf_emit_literals(in, lit_start, ip, out, op, kLzfSegStride);
            lit_start = ip;
        }
    }

    op = lzf_emit_literals(in, lit_start, seg_len, out, op, kLzfSegStride);
    if (op > kLzfSegStride) {
        op = lzf_emit_literals(in, 0, seg_len, out, 0ULL, kLzfSegStride);
    }
    seg_len_out[seg_global] = (uint32_t)op;
}

__global__ void lzf_pack_chunks_kernel(size_t n_chunks, size_t n_segs, size_t chunk_bytes,
                                       size_t max_out_bytes,
                                       const unsigned char *__restrict__ seg_out,
                                       const uint32_t *__restrict__ seg_len,
                                       unsigned char *__restrict__ packed_out,
                                       size_t *__restrict__ out_sizes) {
    size_t chunk = (size_t)blockIdx.x;
    if (chunk >= n_chunks || threadIdx.x != 0) {
        return;
    }

    size_t seg_base = chunk * n_segs;
    size_t total_comp = 0ULL;
    for (size_t s = 0; s < n_segs; ++s) {
        total_comp += (size_t)seg_len[seg_base + s];
    }

    if (total_comp == 0ULL || total_comp >= chunk_bytes || total_comp > max_out_bytes) {
        out_sizes[chunk] = 0ULL;
        return;
    }

    unsigned char *dst = packed_out + chunk * max_out_bytes;
    size_t wp = 0ULL;
    for (size_t s = 0; s < n_segs; ++s) {
        size_t slen = (size_t)seg_len[seg_base + s];
        const unsigned char *src = seg_out + (seg_base + s) * kLzfSegStride;
        for (size_t b = 0; b < slen; ++b) {
            dst[wp++] = src[b];
        }
    }
    out_sizes[chunk] = total_comp;
}

}  // namespace

extern "C" int p21_cuda_lzf_compress_batch_f32(const float *host_chunks, int n_chunks,
                                               size_t chunk_elems, int shuffle,
                                               unsigned char *host_out, size_t max_out_bytes,
                                               size_t *out_sizes, int *has_nonfinite_out) {
    if (!host_chunks || !host_out || !out_sizes || n_chunks <= 0 || chunk_elems == 0 ||
        !p21_cuda_get_enabled() || !p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    const size_t n_c = (size_t)n_chunks;
    const size_t total_elems = n_c * chunk_elems;
    const size_t chunk_bytes = chunk_elems * sizeof(float);
    const size_t raw_bytes = n_c * chunk_bytes;
    const size_t out_stride = max_out_bytes > 0 ? max_out_bytes : chunk_bytes;
    const size_t n_segs = (chunk_bytes + kLzfSegmentBytes - 1ULL) / kLzfSegmentBytes;
    const size_t total_segs = n_c * n_segs;

    float *d_in = (float *)p21_cuda_pool_alloc(raw_bytes, stream);
    unsigned char *d_work = (unsigned char *)p21_cuda_pool_alloc(raw_bytes, stream);
    unsigned char *d_seg_out =
        (unsigned char *)p21_cuda_pool_alloc(total_segs * kLzfSegStride, stream);
    uint32_t *d_seg_len = (uint32_t *)p21_cuda_pool_alloc(total_segs * sizeof(uint32_t), stream);
    uint16_t *d_htab = (uint16_t *)p21_cuda_pool_alloc(
        total_segs * (size_t)kLzfHashBuckets * sizeof(uint16_t), stream);
    unsigned char *d_packed = (unsigned char *)p21_cuda_pool_alloc(n_c * out_stride, stream);
    size_t *d_out_sizes = (size_t *)p21_cuda_pool_alloc(n_c * sizeof(size_t), stream);
    int *d_nonfinite = (int *)p21_cuda_pool_alloc(sizeof(int), stream);

    if (!d_in || !d_work || !d_seg_out || !d_seg_len || !d_htab || !d_packed || !d_out_sizes ||
        !d_nonfinite) {
        p21_cuda_pool_free(d_in, stream);
        p21_cuda_pool_free(d_work, stream);
        p21_cuda_pool_free(d_seg_out, stream);
        p21_cuda_pool_free(d_seg_len, stream);
        p21_cuda_pool_free(d_htab, stream);
        p21_cuda_pool_free(d_packed, stream);
        p21_cuda_pool_free(d_out_sizes, stream);
        p21_cuda_pool_free(d_nonfinite, stream);
        return -1;
    }

    int h_nonfinite = 0;
    bool ok =
        P21_CUDA_OK(cudaMemsetAsync(d_nonfinite, 0, sizeof(int), stream)) &&
        P21_CUDA_OK(cudaMemcpyAsync(d_in, host_chunks, raw_bytes, cudaMemcpyHostToDevice, stream));

    if (ok) {
        size_t prep_grid = (total_elems + kLzfThreads - 1ULL) / kLzfThreads;
        lzf_prep_f32_kernel<<<(unsigned int)prep_grid, kLzfThreads, 0, stream>>>(
            d_in, n_c, chunk_elems, shuffle ? 1 : 0, d_work, d_nonfinite);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        size_t comp_grid = (total_segs + kLzfThreads - 1ULL) / kLzfThreads;
        lzf_compress_segments_kernel<<<(unsigned int)comp_grid, kLzfThreads, 0, stream>>>(
            d_work, n_c, chunk_bytes, n_segs, d_seg_out, d_seg_len, d_htab);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        lzf_pack_chunks_kernel<<<(unsigned int)n_c, 1, 0, stream>>>(
            n_c, n_segs, chunk_bytes, out_stride, d_seg_out, d_seg_len, d_packed, d_out_sizes);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(out_sizes, d_out_sizes, n_c * sizeof(size_t),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, d_nonfinite, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(host_out, d_packed, n_c * out_stride,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
        if (ok) {
            p21_cuda_stats_add_d2h(n_c * out_stride);
        }
    }

    p21_cuda_pool_free(d_in, stream);
    p21_cuda_pool_free(d_work, stream);
    p21_cuda_pool_free(d_seg_out, stream);
    p21_cuda_pool_free(d_seg_len, stream);
    p21_cuda_pool_free(d_htab, stream);
    p21_cuda_pool_free(d_packed, stream);
    p21_cuda_pool_free(d_out_sizes, stream);
    p21_cuda_pool_free(d_nonfinite, stream);

    if (has_nonfinite_out) {
        *has_nonfinite_out = h_nonfinite;
    }
    return ok ? 0 : -1;
}
