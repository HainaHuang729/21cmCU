#include <algorithm>
#include <mutex>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_common.cuh"

namespace {

bool g_pool_supported = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
cudaMemPool_t g_mem_pool = nullptr;
#endif

struct RegistryEntry {
    void *d_ptr;
    size_t nbytes;
    uint64_t hash;
    uint64_t lru_tick;
    bool device_only;
};

std::mutex g_reg_mu;
std::unordered_map<const void *, RegistryEntry> g_registry;
size_t g_registry_bytes = 0;
uint64_t g_lru_counter = 0;

constexpr size_t kHashChunkBytes = 16384ULL;
constexpr size_t kGpuHashThresholdBytes = 1ULL << 20;

__host__ __device__ static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__host__ __device__ static inline uint64_t read_u64_unaligned(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

__host__ __device__ uint64_t p21_hash_chunk(const uint8_t *p, size_t nbytes, uint64_t chunk_idx) {
    uint64_t h0 = 0x243f6a8885a308d3ULL ^ (uint64_t)nbytes ^ chunk_idx;
    uint64_t h1 = 0x13198a2e03707344ULL ^ (uint64_t)nbytes;
    uint64_t h2 = 0xa4093822299f31d0ULL;
    uint64_t h3 = 0x082efa98ec4e6c89ULL;

    size_t i = 0;
    for (; i + 32 <= nbytes; i += 32) {
        h0 = (h0 ^ read_u64_unaligned(p + i)) * 0x9e3779b97f4a7c15ULL;
        h1 = (h1 ^ read_u64_unaligned(p + i + 8)) * 0xbf58476d1ce4e5b9ULL;
        h2 = (h2 ^ read_u64_unaligned(p + i + 16)) * 0x94d049bb133111ebULL;
        h3 = (h3 ^ read_u64_unaligned(p + i + 24)) * 0xd6e8feb86659fd93ULL;
    }
    for (; i + 8 <= nbytes; i += 8) {
        h0 = splitmix64(h0 ^ read_u64_unaligned(p + i));
    }
    uint64_t tail = 0;
    for (size_t b = 0; i + b < nbytes; b++) {
        tail |= ((uint64_t)p[i + b]) << (8 * b);
    }
    if (i < nbytes) {
        h1 = splitmix64(h1 ^ tail);
    }
    return splitmix64(h0 ^ ((h1 << 17) | (h1 >> 47)) ^ h2 ^ ((h3 << 31) | (h3 >> 33)));
}

uint64_t fold_chunk_hashes(const uint64_t *chunks, size_t n_chunks, size_t total_nbytes) {
    uint64_t acc = splitmix64((uint64_t)total_nbytes);
    for (size_t c = 0; c < n_chunks; ++c) {
        acc = splitmix64(acc ^ chunks[c]);
    }
    return acc;
}

__global__ void hash_chunks_device_kernel(const uint8_t *data, size_t nbytes, size_t n_chunks,
                                          uint64_t *chunk_hashes) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chunks) {
        return;
    }
    size_t off = c * kHashChunkBytes;
    size_t len = nbytes - off;
    if (len > kHashChunkBytes) {
        len = kHashChunkBytes;
    }
    chunk_hashes[c] = p21_hash_chunk(data + off, len, (uint64_t)c);
}

/* Deterministic parallel chunk-tree fingerprint matching GPU chunk hashing bit-for-bit. */
uint64_t p21_hash_host_bytes(const void *ptr, size_t nbytes) {
    const uint8_t *p = (const uint8_t *)ptr;
    size_t n_chunks = (nbytes + kHashChunkBytes - 1) / kHashChunkBytes;
    if (n_chunks == 0) {
        return splitmix64(0ULL);
    }
    if (nbytes < kGpuHashThresholdBytes || n_chunks <= 8) {
        std::vector<uint64_t> chunks(n_chunks);
        for (size_t c = 0; c < n_chunks; ++c) {
            size_t off = c * kHashChunkBytes;
            size_t len = std::min(kHashChunkBytes, nbytes - off);
            chunks[c] = p21_hash_chunk(p + off, len, (uint64_t)c);
        }
        return fold_chunk_hashes(chunks.data(), n_chunks, nbytes);
    }

    unsigned int nw = std::thread::hardware_concurrency();
    if (nw == 0u) nw = 4u;
    if (nw > 8u) nw = 8u;
    if ((size_t)nw > n_chunks) nw = (unsigned int)n_chunks;

    std::vector<uint64_t> chunks(n_chunks);
    std::vector<std::thread> workers;
    workers.reserve(nw);
    for (unsigned int w = 0; w < nw; ++w) {
        size_t c_start = (n_chunks * (size_t)w) / (size_t)nw;
        size_t c_end = (n_chunks * (size_t)(w + 1)) / (size_t)nw;
        workers.emplace_back([p, nbytes, c_start, c_end, &chunks]() {
            for (size_t c = c_start; c < c_end; ++c) {
                size_t off = c * kHashChunkBytes;
                size_t len = std::min(kHashChunkBytes, nbytes - off);
                chunks[c] = p21_hash_chunk(p + off, len, (uint64_t)c);
            }
        });
    }
    for (std::thread &th : workers) {
        th.join();
    }
    return fold_chunk_hashes(chunks.data(), n_chunks, nbytes);
}

bool p21_hash_device_bytes(const void *d_ptr, size_t nbytes, cudaStream_t stream,
                           uint64_t *hash_out) {
    if (!d_ptr || !hash_out || nbytes == 0) {
        return false;
    }
    size_t n_chunks = (nbytes + kHashChunkBytes - 1) / kHashChunkBytes;
    uint64_t *d_chunks = (uint64_t *)p21_cuda_pool_alloc(n_chunks * sizeof(uint64_t), stream);
    if (!d_chunks) {
        return false;
    }
    constexpr int kTpb = 256;
    int blocks = (int)((n_chunks + kTpb - 1) / kTpb);
    hash_chunks_device_kernel<<<blocks, kTpb, 0, stream>>>((const uint8_t *)d_ptr, nbytes,
                                                            n_chunks, d_chunks);
    std::vector<uint64_t> h_chunks(n_chunks);
    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_chunks.data(), d_chunks, n_chunks * sizeof(uint64_t),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    p21_cuda_pool_free(d_chunks, stream);
    if (!ok) {
        return false;
    }
    *hash_out = fold_chunk_hashes(h_chunks.data(), n_chunks, nbytes);
    return true;
}

size_t registry_budget_bytes(void) {
    size_t total = p21_cuda_total_mem_bytes();
    if (total == 0) {
        return 1024ULL << 20;
    }
    const char *env_mb = getenv("P21_CUDA_REGISTRY_MB");
    if (env_mb && *env_mb) {
        char *end = nullptr;
        unsigned long long mb = strtoull(env_mb, &end, 10);
        if (end != env_mb && *end == '\0' && mb > 0ULL) {
            return std::min<size_t>((size_t)mb << 20, (size_t)(((unsigned long long)total * 3ULL) / 4ULL));
        }
    }
    if (total > (size_t)(12ULL << 30)) {
        return std::min<size_t>(total / 2, total - (size_t)(6ULL << 30));
    }
    return total / 2;
}

void release_evicted_entry(const void *host_ptr, void *d_old, size_t nbytes, bool device_only,
                           cudaStream_t stream) {
    if (!d_old) {
        return;
    }
    if (device_only && host_ptr && nbytes > 0) {
        if (P21_CUDA_OK(cudaMemcpyAsync(const_cast<void *>(host_ptr), d_old, nbytes,
                                        cudaMemcpyDeviceToHost, stream)) &&
            P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            p21_cuda_stats_add_d2h(nbytes);
        }
    }
    p21_cuda_pool_free(d_old, stream);
}

void evict_lru_locked(size_t incoming_bytes, const void *keep_host_ptr, cudaStream_t stream) {
    size_t budget = registry_budget_bytes();
    while (!g_registry.empty() && g_registry_bytes + incoming_bytes > budget) {
        auto victim = g_registry.end();
        uint64_t best_tick = UINT64_MAX;
        for (auto it = g_registry.begin(); it != g_registry.end(); ++it) {
            if (it->first == keep_host_ptr) {
                continue;
            }
            if (it->second.lru_tick < best_tick) {
                best_tick = it->second.lru_tick;
                victim = it;
            }
        }
        if (victim == g_registry.end()) {
            break;
        }
        const void *h_old = victim->first;
        void *d_old = victim->second.d_ptr;
        size_t old_bytes = victim->second.nbytes;
        bool old_lazy = victim->second.device_only;
        g_registry_bytes -= old_bytes;
        g_registry.erase(victim);
        release_evicted_entry(h_old, d_old, old_bytes, old_lazy, stream);
        p21_cuda_stats_mut()->registry_invalidations++;
    }
}

}  // namespace

void p21_cuda_pool_init(int device_id) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    int pool_attr = 0;
    if (cudaDeviceGetAttribute(&pool_attr, cudaDevAttrMemoryPoolsSupported, device_id) ==
            cudaSuccess &&
        pool_attr > 0) {
        cudaMemPool_t pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&pool, device_id) == cudaSuccess && pool != nullptr) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            cudaGetLastError();
            g_mem_pool = pool;
            g_pool_supported = true;
            return;
        }
    }
    cudaGetLastError();
    g_mem_pool = nullptr;
    g_pool_supported = false;
#else
    (void)device_id;
    g_pool_supported = false;
#endif
}

bool reclaim_stale_registry_for_oom(size_t target_bytes, cudaStream_t stream) {
    struct EvictedItem {
        const void *host_ptr;
        void *d_ptr;
        size_t nbytes;
        bool device_only;
    };
    std::vector<EvictedItem> freed_items;
    size_t freed_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        if (g_registry.empty()) {
            return false;
        }
        /* Keep recently touched entries (within last 16 accesses) so active kernel inputs remain valid. */
        uint64_t safe_cutoff = (g_lru_counter > 16ULL) ? (g_lru_counter - 16ULL) : 0ULL;
        std::vector<std::pair<uint64_t, const void *>> candidates;
        candidates.reserve(g_registry.size());
        for (const auto &kv : g_registry) {
            if (kv.second.lru_tick <= safe_cutoff) {
                candidates.push_back({kv.second.lru_tick, kv.first});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &cand : candidates) {
            auto it = g_registry.find(cand.second);
            if (it == g_registry.end()) {
                continue;
            }
            freed_items.push_back(
                {it->first, it->second.d_ptr, it->second.nbytes, it->second.device_only});
            freed_bytes += it->second.nbytes;
            g_registry_bytes -= it->second.nbytes;
            g_registry.erase(it);
            p21_cuda_stats_mut()->registry_invalidations++;
            if (freed_bytes >= target_bytes) {
                break;
            }
        }
    }
    for (const auto &item : freed_items) {
        release_evicted_entry(item.host_ptr, item.d_ptr, item.nbytes, item.device_only, stream);
    }
    p21_cuda_pool_trim();
    return !freed_items.empty();
}

void *p21_cuda_pool_alloc(size_t nbytes, cudaStream_t stream) {
    size_t bytes = nbytes > 0 ? nbytes : 256;
    auto try_alloc = [&]() -> std::pair<void *, cudaError_t> {
        void *p = nullptr;
        cudaError_t e;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        if (g_pool_supported && g_mem_pool != nullptr && stream != nullptr) {
            e = cudaMallocFromPoolAsync(&p, bytes, g_mem_pool, stream);
        } else if (stream != nullptr) {
            e = cudaMallocAsync(&p, bytes, stream);
            if (e != cudaSuccess) {
                cudaGetLastError();
                e = cudaMalloc(&p, bytes);
            }
        } else {
            e = cudaMalloc(&p, bytes);
        }
#else
        e = cudaMalloc(&p, bytes);
#endif
        return {p, e};
    };

    auto [ptr, err] = try_alloc();
    if (err != cudaSuccess) {
        cudaGetLastError();
        p21_cuda_pool_trim();
        std::tie(ptr, err) = try_alloc();
        if (err != cudaSuccess) {
            cudaGetLastError();
            if (reclaim_stale_registry_for_oom(bytes, stream)) {
                std::tie(ptr, err) = try_alloc();
            }
        }
    }
    if (!P21_CUDA_OK(err)) {
        return nullptr;
    }
    P21CudaStats *st = p21_cuda_stats_mut();
    st->mempool_alloc_calls++;
    st->mempool_bytes_allocated += (unsigned long long int)bytes;
    return ptr;
}

void p21_cuda_pool_free(void *ptr, cudaStream_t stream) {
    if (!ptr) {
        return;
    }
    p21_cuda_stats_mut()->mempool_free_calls++;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (g_pool_supported && stream != nullptr) {
        if (cudaFreeAsync(ptr, stream) == cudaSuccess) {
            return;
        }
        cudaGetLastError();
    }
#else
    (void)stream;
#endif
    cudaFree(ptr);
}

void p21_cuda_pool_trim(void) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (g_pool_supported && g_mem_pool != nullptr) {
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(g_mem_pool, 0);
        cudaGetLastError();
    }
#endif
}

size_t p21_cuda_free_mem_bytes(void) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return free_b;
}

size_t p21_cuda_registry_max_entry_bytes(void) {
    const char *env_mb = getenv("P21_CUDA_REGISTRY_MAX_ENTRY_MB");
    if (env_mb && *env_mb) {
        char *end = nullptr;
        unsigned long long mb = strtoull(env_mb, &end, 10);
        if (end != env_mb && *end == '\0' && mb > 0ULL) {
            return (size_t)mb << 20;
        }
    }
    return registry_budget_bytes() / 4;
}

void p21_cuda_registry_evict_to_free(size_t target_free_bytes, cudaStream_t stream) {
    if (target_free_bytes == 0 || p21_cuda_free_mem_bytes() >= target_free_bytes) {
        return;
    }
    p21_cuda_pool_trim();
    if (p21_cuda_free_mem_bytes() >= target_free_bytes) {
        return;
    }

    while (p21_cuda_free_mem_bytes() < target_free_bytes) {
        const void *h_old = nullptr;
        void *d_old = nullptr;
        size_t old_bytes = 0;
        bool old_lazy = false;
        {
            std::lock_guard<std::mutex> lock(g_reg_mu);
            if (g_registry.empty()) {
                break;
            }
            auto victim = g_registry.end();
            uint64_t best_tick = UINT64_MAX;
            for (auto it = g_registry.begin(); it != g_registry.end(); ++it) {
                if (it->second.lru_tick < best_tick) {
                    best_tick = it->second.lru_tick;
                    victim = it;
                }
            }
            if (victim == g_registry.end()) {
                break;
            }
            h_old = victim->first;
            d_old = victim->second.d_ptr;
            old_bytes = victim->second.nbytes;
            old_lazy = victim->second.device_only;
            g_registry_bytes -= old_bytes;
            g_registry.erase(victim);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
        if (d_old) {
            release_evicted_entry(h_old, d_old, old_bytes, old_lazy, stream);
            p21_cuda_pool_trim();
        }
    }
}

void *p21_cuda_registry_get_or_upload(const void *host_ptr, size_t nbytes, cudaStream_t stream) {
    if (!host_ptr || nbytes == 0) {
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it != g_registry.end() && it->second.device_only) {
            if (it->second.nbytes == nbytes) {
                it->second.lru_tick = ++g_lru_counter;
                p21_cuda_stats_mut()->registry_hits++;
                return it->second.d_ptr;
            }
            void *d_stale = it->second.d_ptr;
            g_registry_bytes -= it->second.nbytes;
            g_registry.erase(it);
            p21_cuda_pool_free(d_stale, stream);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
    }

    uint64_t hash = p21_hash_host_bytes(host_ptr, nbytes);

    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it != g_registry.end()) {
            if (it->second.nbytes == nbytes && (it->second.device_only || it->second.hash == hash)) {
                it->second.lru_tick = ++g_lru_counter;
                p21_cuda_stats_mut()->registry_hits++;
                return it->second.d_ptr;
            }
            void *d_stale = it->second.d_ptr;
            g_registry_bytes -= it->second.nbytes;
            g_registry.erase(it);
            p21_cuda_pool_free(d_stale, stream);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
        evict_lru_locked(nbytes, host_ptr, stream);
    }

    void *d_buf = p21_cuda_pool_alloc(nbytes, stream);
    if (!d_buf) {
        return nullptr;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_buf, host_ptr, nbytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_buf, stream);
        return nullptr;
    }

    P21CudaStats *st = p21_cuda_stats_mut();
    st->registry_misses++;
    st->registry_h2d_bytes += (unsigned long long int)nbytes;

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto existing = g_registry.find(host_ptr);
    if (existing != g_registry.end()) {
        p21_cuda_pool_free(existing->second.d_ptr, stream);
        g_registry_bytes -= existing->second.nbytes;
        g_registry.erase(existing);
    }
    evict_lru_locked(nbytes, host_ptr, stream);
    g_registry.emplace(host_ptr, RegistryEntry{d_buf, nbytes, hash, ++g_lru_counter, false});
    g_registry_bytes += nbytes;
    return d_buf;
}

bool p21_cuda_registry_put_copy(const void *host_ptr, const void *d_src, size_t nbytes,
                                cudaStream_t stream) {
    if (!host_ptr || !d_src || nbytes == 0) {
        return false;
    }
    auto erase_existing = [&]() {
        void *d_stale = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_reg_mu);
            auto it = g_registry.find(host_ptr);
            if (it != g_registry.end()) {
                d_stale = it->second.d_ptr;
                g_registry_bytes -= it->second.nbytes;
                g_registry.erase(it);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
        if (d_stale) {
            p21_cuda_pool_free(d_stale, stream);
        }
    };

    if (nbytes > p21_cuda_registry_max_entry_bytes()) {
        erase_existing();
        return false;
    }
    if (nbytes >= 64ULL * 1024 * 1024) {
        size_t headroom = std::min<size_t>(nbytes, 2ULL * 1024 * 1024 * 1024);
        if (p21_cuda_free_mem_bytes() < nbytes + headroom) {
            erase_existing();
            return false;
        }
    }

    /* Compute hash after the stream finishes D2H copy so host_ptr matches device data. */
    uint64_t hash = 0ULL;
    if (nbytes < kGpuHashThresholdBytes || !p21_hash_device_bytes(d_src, nbytes, stream, &hash)) {
        hash = p21_hash_host_bytes(host_ptr, nbytes);
    }

    void *d_cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it != g_registry.end()) {
            if (it->second.nbytes == nbytes) {
                d_cached = it->second.d_ptr;
            } else {
                p21_cuda_pool_free(it->second.d_ptr, stream);
                g_registry_bytes -= it->second.nbytes;
                g_registry.erase(it);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
        if (!d_cached) {
            evict_lru_locked(nbytes, host_ptr, stream);
        }
    }

    if (!d_cached) {
        d_cached = p21_cuda_pool_alloc(nbytes, stream);
        if (!d_cached) {
            return false;
        }
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_cached, d_src, nbytes, cudaMemcpyDeviceToDevice, stream))) {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it == g_registry.end() || it->second.d_ptr != d_cached) {
            p21_cuda_pool_free(d_cached, stream);
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto it = g_registry.find(host_ptr);
    if (it != g_registry.end()) {
        if (it->second.d_ptr != d_cached) {
            p21_cuda_pool_free(it->second.d_ptr, stream);
            g_registry_bytes -= it->second.nbytes;
            it->second.d_ptr = d_cached;
            it->second.nbytes = nbytes;
            g_registry_bytes += nbytes;
        }
        it->second.hash = hash;
        it->second.lru_tick = ++g_lru_counter;
        it->second.device_only = false;
    } else {
        evict_lru_locked(nbytes, host_ptr, stream);
        g_registry.emplace(host_ptr,
                           RegistryEntry{d_cached, nbytes, hash, ++g_lru_counter, false});
        g_registry_bytes += nbytes;
    }
    return true;
}

bool p21_cuda_registry_put_owned(const void *host_ptr, void *d_buf, size_t nbytes,
                                 cudaStream_t stream) {
    if (!host_ptr || !d_buf || nbytes == 0) {
        return false;
    }
    if (nbytes > p21_cuda_registry_max_entry_bytes()) {
        void *d_stale = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_reg_mu);
            auto existing = g_registry.find(host_ptr);
            if (existing != g_registry.end()) {
                if (existing->second.d_ptr != d_buf) {
                    d_stale = existing->second.d_ptr;
                }
                g_registry_bytes -= existing->second.nbytes;
                g_registry.erase(existing);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
        if (d_stale) {
            p21_cuda_pool_free(d_stale, stream);
        }
        return false;
    }
    /* Caller guarantees host_ptr already reflects completed D2H contents. */
    uint64_t hash = 0ULL;
    if (nbytes < kGpuHashThresholdBytes || !p21_hash_device_bytes(d_buf, nbytes, stream, &hash)) {
        hash = p21_hash_host_bytes(host_ptr, nbytes);
    }

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto existing = g_registry.find(host_ptr);
    if (existing != g_registry.end()) {
        if (existing->second.d_ptr != d_buf) {
            p21_cuda_pool_free(existing->second.d_ptr, stream);
        }
        g_registry_bytes -= existing->second.nbytes;
        g_registry.erase(existing);
        p21_cuda_stats_mut()->registry_invalidations++;
    }
    evict_lru_locked(nbytes, host_ptr, stream);
    g_registry.emplace(host_ptr, RegistryEntry{d_buf, nbytes, hash, ++g_lru_counter, false});
    g_registry_bytes += nbytes;
    return true;
}

bool p21_cuda_registry_put_owned_lazy(const void *host_ptr, void *d_buf, size_t nbytes,
                                      cudaStream_t stream) {
    if (!host_ptr || !d_buf || nbytes == 0) {
        return false;
    }
    if (nbytes > p21_cuda_registry_max_entry_bytes()) {
        return false;
    }
    if (nbytes >= 64ULL * 1024 * 1024) {
        size_t headroom = std::min<size_t>(nbytes, 2ULL * 1024 * 1024 * 1024);
        if (p21_cuda_free_mem_bytes() < nbytes + headroom) {
            return false;
        }
    }

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto existing = g_registry.find(host_ptr);
    if (existing != g_registry.end()) {
        if (existing->second.d_ptr != d_buf) {
            p21_cuda_pool_free(existing->second.d_ptr, stream);
        }
        g_registry_bytes -= existing->second.nbytes;
        g_registry.erase(existing);
        p21_cuda_stats_mut()->registry_invalidations++;
    }
    evict_lru_locked(nbytes, host_ptr, stream);
    g_registry.emplace(host_ptr, RegistryEntry{d_buf, nbytes, 0ULL, ++g_lru_counter, true});
    g_registry_bytes += nbytes;
    return true;
}

extern "C" {

int p21_cuda_registry_sync_to_host(const void *host_ptr) {
    if (!host_ptr) {
        return 0;
    }
    void *d_ptr = nullptr;
    size_t nbytes = 0;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it == g_registry.end() || !it->second.device_only) {
            return 0;
        }
        d_ptr = it->second.d_ptr;
        nbytes = it->second.nbytes;
    }
    cudaStream_t stream = p21_cuda_stream();
    if (!d_ptr || nbytes == 0 ||
        !P21_CUDA_OK(cudaMemcpyAsync(const_cast<void *>(host_ptr), d_ptr, nbytes,
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return 0;
    }
    p21_cuda_stats_add_d2h(nbytes);

    uint64_t hash = 0ULL;
    if (nbytes < kGpuHashThresholdBytes || !p21_hash_device_bytes(d_ptr, nbytes, stream, &hash)) {
        hash = p21_hash_host_bytes(host_ptr, nbytes);
    }
    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto it = g_registry.find(host_ptr);
    if (it != g_registry.end() && it->second.d_ptr == d_ptr) {
        it->second.hash = hash;
        it->second.device_only = false;
        it->second.lru_tick = ++g_lru_counter;
    }
    return 1;
}

void p21_cuda_registry_sync_all_lazy(void) {
    std::vector<const void *> keys;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        for (const auto &kv : g_registry) {
            if (kv.second.device_only) {
                keys.push_back(kv.first);
            }
        }
    }
    for (const void *k : keys) {
        p21_cuda_registry_sync_to_host(k);
    }
}

void p21_cuda_invalidate_host_ptr(const void *ptr) {
    if (!ptr) {
        return;
    }
    void *d_old = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        if (!g_registry.empty()) {
            auto it = g_registry.find(ptr);
            if (it != g_registry.end()) {
                d_old = it->second.d_ptr;
                g_registry_bytes -= it->second.nbytes;
                g_registry.erase(it);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
    }
    if (d_old) {
        p21_cuda_pool_free(d_old, p21_cuda_stream());
    }
    p21_cuda_radiation_invalidate_host_ptr(ptr);
}

void p21_cuda_registry_clear(void) {
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        cudaStream_t stream = p21_cuda_stream();
        for (auto &kv : g_registry) {
            release_evicted_entry(kv.first, kv.second.d_ptr, kv.second.nbytes,
                                  kv.second.device_only, stream);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
        g_registry.clear();
        g_registry_bytes = 0;
    }
    p21_cuda_radiation_clear();
}

int p21_cuda_export_device_f32(const float *h_ptr, size_t n_elems, unsigned long long *d_ptr_out,
                               int *device_id_out) {
    if (!h_ptr || n_elems == 0 || !d_ptr_out || !device_id_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }
    cudaStream_t stream = p21_cuda_stream();
    size_t nbytes = n_elems * sizeof(float);

    void *d_export = nullptr;
    if (!P21_CUDA_OK(cudaMalloc(&d_export, nbytes))) {
        p21_cuda_registry_evict_to_free(nbytes, stream);
        if (!P21_CUDA_OK(cudaMalloc(&d_export, nbytes))) {
            return -1;
        }
    }

    const void *d_cached = nullptr;
    if (nbytes <= p21_cuda_registry_max_entry_bytes()) {
        d_cached = p21_cuda_registry_get_or_upload(h_ptr, nbytes, stream);
    }

    bool ok = false;
    if (d_cached) {
        ok = P21_CUDA_OK(
            cudaMemcpyAsync(d_export, d_cached, nbytes, cudaMemcpyDeviceToDevice, stream));
    } else {
        ok = P21_CUDA_OK(
            cudaMemcpyAsync(d_export, h_ptr, nbytes, cudaMemcpyHostToDevice, stream));
        if (ok) {
            p21_cuda_stats_mut()->registry_h2d_bytes += (unsigned long long int)nbytes;
        }
    }
    if (!ok || !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        cudaFree(d_export);
        return -1;
    }

    *d_ptr_out = (unsigned long long)(uintptr_t)d_export;
    *device_id_out = p21_cuda_current_device();
    return 0;
}

void p21_cuda_free_exported_device_ptr(unsigned long long d_ptr) {
    if (d_ptr == 0ULL) {
        return;
    }
    cudaFree((void *)(uintptr_t)d_ptr);
}

int p21_cuda_device_copy_to_host_f32(unsigned long long d_ptr, float *h_out, size_t n_elems) {
    if (d_ptr == 0ULL || !h_out || n_elems == 0) {
        return -1;
    }
    size_t nbytes = n_elems * sizeof(float);
    if (!P21_CUDA_OK(cudaMemcpy(h_out, (const void *)(uintptr_t)d_ptr, nbytes,
                                cudaMemcpyDeviceToHost))) {
        return -1;
    }
    p21_cuda_stats_add_d2h(nbytes);
    return 0;
}

}  // extern "C"
