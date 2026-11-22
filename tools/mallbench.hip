// mallbench.hip — does a streaming read of a weight-sized buffer run faster when it is already in the 96 MB Infinity Cache (MALL)?
// build/run on the box (server stopped): hipcc -O3 --offload-arch=gfx1100 tools/mallbench.hip -o /tmp/mallbench && /tmp/mallbench
// prints us and GB/s for: cold read (after flushing with a 512 MB read), warm re-read, and a read after a concurrent "prefetch" on a side stream.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

int main() {
    const size_t flushB = 512ull << 20;
    uint4 *flush, *w; unsigned* o;
    CK(hipMalloc(&flush, flushB)); CK(hipMalloc(&o, 4)); CK(hipMemset(flush, 1, flushB));
    hipStream_t s, s2; CK(hipStreamCreate(&s)); CK(hipStreamCreate(&s2));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    int grid = 96 * 8;
    auto timed = [&](const uint4* p, size_t bytes, int g) {
        hipEventRecord(a, s); rd<<<g, 256, 0, s>>>(p, bytes / 16, o); hipEventRecord(b, s); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); return ms * 1000.0f;
    };
    for (size_t mb : {4, 8, 16, 32, 48, 64, 80, 128}) {
        const size_t B = mb << 20;
        CK(hipMalloc(&w, B)); CK(hipMemset(w, 2, B));
        float cold = 0, warm = 0;
        const int R = 10;
        for (int r = 0; r < R; ++r) {
            rd<<<grid, 256, 0, s>>>(flush, flushB / 16, o);
            cold += timed(w, B, grid);
            warm += timed(w, B, grid);
        }
        cold /= R; warm /= R;
        // a prefetch by a small grid (24 blocks) on a side stream, then the full read
        float pf = 0;
        for (int r = 0; r < R; ++r) {
            rd<<<grid, 256, 0, s>>>(flush, flushB / 16, o); hipStreamSynchronize(s);
            rd<<<24, 256, 0, s2>>>(w, B / 16, o); hipStreamSynchronize(s2);
            pf += timed(w, B, grid);
        }
        pf /= R;
        printf("%4zu MB  cold %8.1f us %6.0f GB/s | warm %8.1f us %6.0f GB/s | after side prefetch %8.1f us %6.0f GB/s\n", mb,
               cold, B / cold / 1e3, warm, B / warm / 1e3, pf, B / pf / 1e3);
        hipFree(w);
    }
    // how fast can a small grid pull from DRAM (the prefetch's own rate)?
    for (int g : {8, 16, 24, 48, 96}) {
        const size_t B = 64ull << 20; CK(hipMalloc(&w, B)); CK(hipMemset(w, 2, B));
        float t = 0; for (int r = 0; r < 5; ++r) { rd<<<grid, 256, 0, s>>>(flush, flushB / 16, o); t += timed(w, B, g); }
        t /= 5; printf("prefetch grid %3d blocks: 64 MB cold %8.1f us %6.0f GB/s\n", g, t, B / t / 1e3); hipFree(w);
    }
    return 0;
}
