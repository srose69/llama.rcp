#pragma once

#include "common.cuh"
#include "vecdotq_q4.cuh"
#include "vecdotq.cuh"

// SPDX-License-Identifier: PolyForm-Shield-1.0.0
// Copyright (c) 2026 Ivan K
// @ai-training prohibited
// Source: https://github.com/srose69/llama.rcp

// Q4_K MMQ kernels for Pascal (DP4A path)
// Extracted for optimization experiments

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int lane_id = threadIdx.x % 32;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        
        // Prefetch N+1 and N+2 tiles to L1/L2 cache
        if (i0 + nrows*nwarps < mmq_y) {
            const block_q4_K * bxi_next = bxi + nrows*nwarps*stride;
            asm volatile("prefetch.global.L1 [%0];" :: "l"(&bxi_next->qs[txi * 4]));
        }
        if (i0 + 2*nrows*nwarps < mmq_y) {
            const block_q4_K * bxi_next2 = bxi + 2*nrows*nwarps*stride;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(&bxi_next2->qs[txi * 4]));
        }
        
        // Load with cache-at-all-levels hint (data reused across warps)
        int qs0;
        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(qs0) : "l"(&bxi->qs[txi * 4]));

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        // ZERO-COST SWAR UNPACKING: Move unpacking HERE (load_tiles) instead of vec_dot
        // This eliminates SHR+AND in the hot compute loop
        // Cost: 2x shared memory, Gain: Zero ALU overhead in vec_dot
        
        // Extract nibbles using XMAD trick (faster than SHR on Pascal)
        // Lo nibbles: multiply by 1, mask low bits
        const unsigned int v_lo4 = qs0 & 0x0F0F0F0F;        // [n6|n4|n2|n0]
        // Hi nibbles: shift right 4 using XMAD.PSL equivalent
        unsigned int v_hi4;
        asm("shr.u32 %0, %1, 4;" : "=r"(v_hi4) : "r"(qs0));
        v_hi4 &= 0x0F0F0F0F;  // [n7|n5|n3|n1]
        
        // Pack with holes: [0|n3|0|n2|0|n1|0|n0] in one int32
        const unsigned int packed_lo = __byte_perm(v_lo4, 0, 0x6420);  // Even nibbles with holes
        const unsigned int packed_hi = __byte_perm(v_hi4, 0, 0x6420);  // Odd nibbles with holes
        
        // CRITICAL: Store lo/hi in SEPARATE banks to avoid unpacking in vec_dot
        // Layout: [all_lo_data | all_hi_data] instead of interleaved [lo|hi|lo|hi...]
        constexpr int bank_stride = (MMQ_TILE_NE_K + 1);  // Stride per row
        const int base_offset = i * bank_stride * 2;  // 2x for lo+hi banks
        
        // Bank 0: lo nibbles
        x_qs[base_offset + txi*2 + 0] = packed_lo;
        // Bank 1: hi nibbles (offset by bank_stride)
        x_qs[base_offset + bank_stride + txi*2 + 0] = packed_hi;
        // Duplicate for butterfly shuffle (thread pair access)
        x_qs[base_offset + txi*2 + 1] = packed_lo;
        x_qs[base_offset + bank_stride + txi*2 + 1] = packed_hi;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < mmq_y) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll 4  // Maximum ILP for generation (batch=1), balance code size vs parallelism
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);

                // ZERO-COST UNPACKING: x_qs now has dual-bank layout [lo_data | hi_data]
                // Each row uses bank_stride*2 space instead of (MMQ_TILE_NE_K + 1)
                constexpr int bank_stride = (MMQ_TILE_NE_K + 1);
                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq_ptx(
                    &x_qs[i*bank_stride*2 + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}
