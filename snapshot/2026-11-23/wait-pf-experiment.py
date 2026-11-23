# wpf_patch.py — (arg2 "revert" undoes it) STRATA_WAIT_PF: combine's CPU-expert wait warms the next layer's weights in the MALL (edits the engine tree in place)
import sys, re
E = sys.argv[1]
REVERT = len(sys.argv) > 2 and sys.argv[2] == 'revert'

def edit(path, old, new, count=1):
    p = f"{E}/{path}"; s = open(p).read()
    if REVERT: old, new = new, old
    if new in s and old not in s: print("already:", path); return
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new)); print("edited:", path)

# ---- kernel: body as a device function, the stock kernel as a wrapper, plus the prefetching variant
K = "src/kernels/cuda/native_moe.cu"
edit(K, """template <bool GATED>
__global__ void combine_cpu_k10_vec4(const volatile float4* cpu4, const float4* __restrict__ hit4,
                                     const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                     const float* __restrict__ weights, const float4* __restrict__ shared4,
                                     const float* __restrict__ sg, float4* __restrict__ output4, int64_t n4,
                                     const volatile uint32_t* flag, uint32_t value) {""",
"""template <bool GATED>
__device__ __forceinline__ void combine_cpu_body(const volatile float4* cpu4, const float4* __restrict__ hit4,
                                     const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                     const float* __restrict__ weights, const float4* __restrict__ shared4,
                                     const float* __restrict__ sg, float4* __restrict__ output4, int64_t n4,
                                     const volatile uint32_t* flag, uint32_t value) {""")
edit(K, """    output4[tk * n4 + c4] = sum;
}
__global__ void combine_k10_vec4_hits(""",
"""    output4[tk * n4 + c4] = sum;
}
template <bool GATED>
__global__ void combine_cpu_k10_vec4(const volatile float4* cpu4, const float4* __restrict__ hit4,
                                     const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                     const float* __restrict__ weights, const float4* __restrict__ shared4,
                                     const float* __restrict__ sg, float4* __restrict__ output4, int64_t n4,
                                     const volatile uint32_t* flag, uint32_t value) {
    combine_cpu_body<GATED>(cpu4, hit4, hit_rows, count, weights, shared4, sg, output4, n4, flag, value);
}
// rdna3-moe-tuning STRATA_WAIT_PF: blocks past the combine's own (row 0 of the grid only) read the next layer's weight
// ranges while the CPU experts are still running, so those reads land in the 96 MB Infinity Cache and the next
// layer's HC read / GEMVs hit it.  They stop as soon as the flag is up (polled before the first chunk, then once per
// chunk with the chunk's loads already in flight).  Pure loads: nothing is written, outputs are bitwise unchanged.
struct CombinePf { const uint4* p[kCombinePfMax]; unsigned n16[kCombinePfMax]; int n; int nbx; };
template <int CH>
__device__ __forceinline__ void combine_pf_body(const CombinePf& pf, int pb, int nb, const volatile uint32_t* flag, uint32_t value) {
    __shared__ int stop[2];
    if (threadIdx.x == 0) stop[0] = *flag >= value;
    __syncthreads();
    if (stop[0]) return;
    uint32_t acc = 0;
    int it = 1;
    constexpr unsigned kChunk16 = 64u * CH;   // one block pass: 64 threads x CH uint4
    for (int r = 0; r < pf.n; ++r) {
        const uint4* __restrict__ p = pf.p[r];
        const unsigned m = pf.n16[r], nch = (m + kChunk16 - 1) / kChunk16;
        for (unsigned c = (unsigned) pb; c < nch; c += (unsigned) nb, ++it) {
            const unsigned base = c * kChunk16 + threadIdx.x;
            uint4 v[CH];
#pragma unroll
            for (int i = 0; i < CH; ++i) { const unsigned j = base + 64u * i; v[i] = j < m ? p[j] : make_uint4(0u, 0u, 0u, 0u); }
            if (threadIdx.x == 0) stop[it & 1] = *flag >= value;   // the poll's round trip overlaps the loads
#pragma unroll
            for (int i = 0; i < CH; ++i) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
            __syncthreads();
            if (stop[it & 1]) { asm volatile("" :: "v"(acc)); return; }
        }
    }
    asm volatile("" :: "v"(acc));
}
template <bool GATED, int CH>
__global__ void combine_cpu_k10_vec4_pf(const volatile float4* cpu4, const float4* __restrict__ hit4,
                                        const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                        const float* __restrict__ weights, const float4* __restrict__ shared4,
                                        const float* __restrict__ sg, float4* __restrict__ output4, int64_t n4,
                                        const volatile uint32_t* flag, uint32_t value, CombinePf pf) {
    if ((int) blockIdx.x >= pf.nbx) {
        if (blockIdx.y == 0) combine_pf_body<CH>(pf, (int) blockIdx.x - pf.nbx, (int) gridDim.x - pf.nbx, flag, value);
        return;
    }
    combine_cpu_body<GATED>(cpu4, hit4, hit_rows, count, weights, shared4, sg, output4, n4, flag, value);
}
__global__ void combine_k10_vec4_hits(""")

edit(K, """    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (shared_gate)
        combine_cpu_k10_vec4<true><<<grid, 64, 0, st>>>(""",
"""    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (wait_flag != nullptr && pf_n > 0 && pf_blocks > 0) {   // STRATA_WAIT_PF
        CombinePf pf{}; int k = 0;
        for (int i = 0; i < pf_n && k < kCombinePfMax; ++i)
            if (pf_ptrs[i] != nullptr && pf_bytes[i] >= 16) { pf.p[k] = (const uint4*) pf_ptrs[i]; pf.n16[k] = (unsigned) (pf_bytes[i] / 16); ++k; }
        pf.n = k; pf.nbx = (int) grid.x;
        if (k > 0) {
            const dim3 g2(grid.x + (unsigned) pf_blocks, grid.y);
            const float* sgp = shared_gate;
#define STRATA_WPF(G, CH) combine_cpu_k10_vec4_pf<G, CH><<<g2, 64, 0, st>>>(reinterpret_cast<const volatile float4*>(cpu), \\
                reinterpret_cast<const float4*>(hit), hit_rows, count, weights, reinterpret_cast<const float4*>(shared), sgp, \\
                reinterpret_cast<float4*>(output), n4, wait_flag, wait_value, pf)
            if (shared_gate) { if (pf_chunk >= 16) STRATA_WPF(true, 16); else if (pf_chunk >= 8) STRATA_WPF(true, 8); else STRATA_WPF(true, 4); }
            else             { if (pf_chunk >= 16) STRATA_WPF(false, 16); else if (pf_chunk >= 8) STRATA_WPF(false, 8); else STRATA_WPF(false, 4); }
#undef STRATA_WPF
            const auto error = cudaGetLastError();
            if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
            return;
        }
    }
    if (shared_gate)
        combine_cpu_k10_vec4<true><<<grid, 64, 0, st>>>(""")
edit(K, """                                  int64_t n_embd, int n_tok, void* stream, const uint32_t* wait_flag, uint32_t wait_value) {""",
"""                                  int64_t n_embd, int n_tok, void* stream, const uint32_t* wait_flag, uint32_t wait_value,
                                  const void* const* pf_ptrs, const unsigned long long* pf_bytes, int pf_n, int pf_blocks, int pf_chunk) {""")

H = "include/strata/kernels/native_moe.hpp"
edit(H, """                                  int64_t n_embd, int n_tok, void* stream, const uint32_t* wait_flag = nullptr,
                                  uint32_t wait_value = 0);""",
"""                                  int64_t n_embd, int n_tok, void* stream, const uint32_t* wait_flag = nullptr,
                                  uint32_t wait_value = 0, const void* const* pf_ptrs = nullptr,
                                  const unsigned long long* pf_bytes = nullptr, int pf_n = 0, int pf_blocks = 0, int pf_chunk = 16);
/// rdna3-moe-tuning STRATA_WAIT_PF: the most ranges the combine's wait can warm (see native_moe.cu)
constexpr int kCombinePfMax = 16;""")

# ---- verify.cpp: the next layer's ranges at the combine
V = "src/core/verify.cpp"
edit(V, """        if (!ar_on() && cfuse_on(n)) {
            try {
                native_moe_combine_multi_cpu(m_ymiss_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, w_ + tb * K,
                                             shared_ + tb * N, sg_gated_[grp] ? sh_g_ + tb : nullptr, bo_ + tb * N, N, n, cs,
                                             g_wait_fuse ? m_flag_ : nullptr, ring);""",
"""        if (!ar_on() && cfuse_on(n)) {
            // rdna3-moe-tuning STRATA_WAIT_PF=<blocks> (0 off): while the combine waits for the CPU experts, extra blocks
            // read layer l+1's weights (STRATA_WAIT_PF_SET bits: 1 hc_attn 2 qkv/q,k,v 4 z 8 out 16 hc_ffn 32 router
            // 64 shexp; default 127) so they are in the Infinity Cache; STRATA_WAIT_PF_CH 4/8/16 uint4 a thread a pass
            static const int wpf = [] { const char* e = std::getenv("STRATA_WAIT_PF"); return e ? std::atoi(e) : 0; }();
            static const int wpf_set = [] { const char* e = std::getenv("STRATA_WAIT_PF_SET"); return e ? std::atoi(e) : 127; }();
            static const int wpf_ch = [] { const char* e = std::getenv("STRATA_WAIT_PF_CH"); return e ? std::atoi(e) : 16; }();
            const void* pf_ptr[strata::kernels::kCombinePfMax]; unsigned long long pf_by[strata::kernels::kCombinePfMax]; int pf_k = 0;
            if (wpf > 0 && g_wait_fuse && l + 1 < le_ && !tp_) {
                const LayerView pv(wt, l + 1);
                auto addn = [&](const char* name) {
                    const WeightRef* w = pv.get(name);
                    if (w && w->native_data && w->native_type >= 0 && pf_k < strata::kernels::kCombinePfMax) {
                        pf_ptr[pf_k] = w->native_data;
                        pf_by[pf_k++] = strata::kernels::native_mmvq_weight_bytes(w->native_type, (int) w->ne0, (int) w->ne1);
                    }
                };
                auto addb = [&](const char* name) {
                    const WeightRef* w = pv.get(name);
                    if (w && w->data && pf_k < strata::kernels::kCombinePfMax) { pf_ptr[pf_k] = w->data; pf_by[pf_k++] = w->bytes; }
                };
                auto addr8 = [&](const char* name) {
                    const WeightRef* w = pv.get(name);
                    if (w && w->hc_r8 && pf_k < strata::kernels::kCombinePfMax) {
                        const unsigned long long q = (unsigned long long) w->ne0 * (unsigned long long) w->ne1;
                        pf_ptr[pf_k] = w->hc_r8; pf_by[pf_k++] = ((q + 255ull) & ~255ull) + q / 16;
                    }
                };
                if (wpf_set & 1) { addr8("hc_attn_down.weight"); addr8("hc_attn_up.weight"); addr8("hc_attn_inject.weight"); }
                if (!is_qsa_layer(g, l + 1)) {
                    if (wpf_set & 2) addn("attn_qkv.weight");
                    if (wpf_set & 4) addn("attn_gate.weight");
                    if (wpf_set & 8) addn("ssm_out.weight");
                } else {
                    if (wpf_set & 2) { addn("attn_q.weight"); addn("attn_k.weight"); addn("attn_v.weight"); }
                    if (wpf_set & 8) addn("attn_output.weight");
                }
                if (wpf_set & 16) { addr8("hc_ffn_down.weight"); addr8("hc_ffn_up.weight"); addr8("hc_ffn_inject.weight"); }
                if (wpf_set & 32) addb("ffn_gate_inp.weight");
                if (wpf_set & 64) { addn("ffn_gate_shexp.weight"); addn("ffn_up_shexp.weight"); addn("ffn_down_shexp.weight"); }
            }
            try {
                native_moe_combine_multi_cpu(m_ymiss_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, w_ + tb * K,
                                             shared_ + tb * N, sg_gated_[grp] ? sh_g_ + tb : nullptr, bo_ + tb * N, N, n, cs,
                                             g_wait_fuse ? m_flag_ : nullptr, ring, pf_ptr, pf_by, pf_k, wpf, wpf_ch);""")
