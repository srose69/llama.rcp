#pragma once

#include "common.cuh"

// SPDX-License-Identifier: PolyForm-Shield-1.0.0

// Copyright (c) 2026 Ivan K

// @ai-training prohibited

// Source: https://github.com/srose69/llama.rcp

/*
// ============================================================================
// OLD BASELINE IMPLEMENTATION (DEPRECATED - REPLACED BY 6-PACK)
// ============================================================================
// This is the old single-warp implementation with poor memory utilization.
// Issues: 90% time spent waiting on global memory, XMAD utilization <2%
// Kept for reference only.
// ============================================================================

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    // Worker1: i=0
    {
        const int v0i_w1 = (v[0] >> 0) & 0x0F0F0F0F;
        const int v1i_w1 = (v[1] >> 0) & 0x0F0F0F0F;

        const int dot1_w1 = ggml_cuda_dp4a(v1i_w1, u[1], ggml_cuda_dp4a(v0i_w1, u[0], 0));
        const int dot2_w1 = ggml_cuda_dp4a(0x01010101, u[1], ggml_cuda_dp4a(0x01010101, u[0], 0));

        sumf_d += d8[0] * (dot1_w1 * sc[0]);
        sumf_m += d8[0] * (dot2_w1 * m[0]);
    }

    __syncwarp();

    // Worker2: i=1
    {
        const int v0i_w2 = (v[0] >> 4) & 0x0F0F0F0F;
        const int v1i_w2 = (v[1] >> 4) & 0x0F0F0F0F;

        const int dot1_w2 = ggml_cuda_dp4a(v1i_w2, u[3], ggml_cuda_dp4a(v0i_w2, u[2], 0));
        const int dot2_w2 = ggml_cuda_dp4a(0x01010101, u[3], ggml_cuda_dp4a(0x01010101, u[2], 0));

        sumf_d += d8[1] * (dot1_w2 * sc[1]);
        sumf_m += d8[1] * (dot2_w2 * m[1]);
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    
    const uint16_t s0 = scales[j+0];
    const uint16_t s2 = scales[j+2];
    const uint16_t sm2 = scales[j-2];
    
    const uint16_t path1_0 = s0 & 0x3f3f;
    const uint16_t path1_1 = s2 & 0x3f3f;
    const uint16_t path2_0 = ((s2 >> 0) & 0x0f0f) | ((sm2 & 0xc0c0) >> 2);
    const uint16_t path2_1 = ((s2 >> 4) & 0x0f0f) | ((s0 & 0xc0c0) >> 2);
    
    const int mask = -((j < 2) ? 1 : 0);
    
    int aux0_int, aux1_int;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(aux0_int) : "r"(mask), "r"((int)path1_0), "r"((int)path2_0));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(aux1_int) : "r"(mask), "r"((int)path1_1), "r"((int)path2_1));
    
    aux[0] = (uint16_t)aux0_int;
    aux[1] = (uint16_t)aux1_int;
    
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll 8
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
*/

// ============================================================================
// 6-PACK SOFTWARE PIPELINE IMPLEMENTATION
// ============================================================================
// Architecture: 6 specialized warps with async execution and latency hiding
// - Warp 1: Load Q4 data from global → shared staging
// - Warp 2: Load Q8 data from global → shared staging  
// - Warp 3: Format Q4 data + padding
// - Warp 4: Format Q8 data + padding
// - Warp 5: XMAD compute using formatted data
// - Warp 6: Output reduction and store
//
// Synchronization: BAR.SYNC n, count for partial barriers
// Key benefit: Overlaps global memory loads with compute (latency hiding)
// ============================================================================

#define WARP_SIZE 32
#define NUM_PIPELINE_WARPS 6

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_6pack(
    const block_q4_K * __restrict__ bq4_K, 
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs) {

    __shared__ int smem_q4_staging[WARP_SIZE * 2];
    __shared__ int smem_q8_staging[WARP_SIZE * 4];
    __shared__ int smem_formatted_q4[WARP_SIZE * 8];
    __shared__ int smem_formatted_q8[WARP_SIZE * 8];
    __shared__ int smem_scaled_dot_result;
    __shared__ int smem_offset_result;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = (threadIdx.x / WARP_SIZE) % NUM_PIPELINE_WARPS + 1;
    
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j = bq8_offset/2;
    
    const uint16_t s0 = scales[j+0];
    const uint16_t s2 = scales[j+2];
    const uint16_t sm2 = scales[j-2];
    
    const uint16_t path1_0 = s0 & 0x3f3f;
    const uint16_t path1_1 = s2 & 0x3f3f;
    const uint16_t path2_0 = ((s2 >> 0) & 0x0f0f) | ((sm2 & 0xc0c0) >> 2);
    const uint16_t path2_1 = ((s2 >> 4) & 0x0f0f) | ((s0 & 0xc0c0) >> 2);
    
    const int mask = -((j < 2) ? 1 : 0);
    
    uint16_t aux[2];
    int aux0_int, aux1_int;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(aux0_int) : "r"(mask), "r"((int)path1_0), "r"((int)path2_0));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(aux1_int) : "r"(mask), "r"((int)path1_1), "r"((int)path2_1));
    
    aux[0] = (uint16_t)aux0_int;
    aux[1] = (uint16_t)aux1_int;
    
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;
    
    float d8[QR4_K];
    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
    }
    
    const half2 dm4 = bq4_K->dm;
    
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    // ========================================================================
    // WARP 1: LOADER Q4 (Load Q4 weights from global memory)
    // ========================================================================
    if (warp_id == 1) {
        const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
        
        if (lane_id < 2) {
            smem_q4_staging[lane_id] = q4[lane_id * 4];
        }
        
        asm volatile("bar.sync 1, %0;" :: "r"(WARP_SIZE));
    }

    // ========================================================================
    // WARP 2: LOADER Q8 (Load Q8 activations from global memory)
    // ========================================================================
    if (warp_id == 2) {
        #pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            if (lane_id < 2) {
                const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
                const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
                smem_q8_staging[i * 2 + lane_id] = q8[lane_id * 4];
            }
        }
        
        asm volatile("bar.sync 2, %0;" :: "r"(WARP_SIZE));
    }

    // ========================================================================
    // WARP 3: SORTER Q4 (Pack 2×Q4 into 16-bit)
    // ========================================================================
    if (warp_id == 3) {
        asm volatile("bar.sync 1, %0;" :: "r"(WARP_SIZE));
        
        if (lane_id == 0) {
            int q4_val0 = smem_q4_staging[0];
            int q4_val1 = smem_q4_staging[1];
            
            int packed = ((q4_val1 & 0xFF) << 8) | (q4_val0 & 0xFF);
            smem_formatted_q4[0] = packed;
        }
        
        asm volatile("bar.sync 3, %0;" :: "r"(WARP_SIZE));
    }

    // ========================================================================
    // WARP 4: SORTER Q8 (Pack 2×Q8 into 16-bit)
    // ========================================================================
    if (warp_id == 4) {
        asm volatile("bar.sync 2, %0;" :: "r"(WARP_SIZE));
        
        if (lane_id < 8) {
            int q8_val0 = smem_q8_staging[lane_id * 2 + 0];
            int q8_val1 = smem_q8_staging[lane_id * 2 + 1];
            
            int packed = ((q8_val1 & 0xFF) << 8) | (q8_val0 & 0xFF);
            smem_formatted_q8[lane_id] = packed;
        }
        
        asm volatile("bar.sync 4, %0;" :: "r"(WARP_SIZE));
    }

    // ========================================================================
    // WARP 5: WORKER (VMAD Q4×Q8 + XMAD dot×sc)
    // ========================================================================
    if (warp_id == 5) {
        asm volatile("bar.sync 3, %0;" :: "r"(WARP_SIZE));
        asm volatile("bar.sync 4, %0;" :: "r"(WARP_SIZE));
        
        if (lane_id == 0) {
            int q4_packed = smem_formatted_q4[0];
            int q8_packed = smem_formatted_q8[0];
            
            // VMAD: Q4 × Q8
            int dot_result = 0;
            asm volatile(
                "vmad.u8.u8 %0, %1, %2, 0;\n\t"
                : "=r"(dot_result)
                : "r"(q4_packed), "r"(q8_packed)
            );
            
            // XMAD: dot × sc
            int sc_packed = ((int)sc[1] << 8) | (int)sc[0];
            int scaled_dot;
            asm volatile(
                "xmad %0, %1, %2, RZ;\n\t"
                : "=r"(scaled_dot)
                : "r"(dot_result), "r"(sc_packed)
            );
            
            // Write to shared memory for FP block
            smem_scaled_dot_result = scaled_dot;
        }
        
        asm volatile("bar.sync 5, %0;" :: "r"(WARP_SIZE));
    }

    // ========================================================================
    // WARP 6: COURIER (VMAD m×Q8 for offset)
    // ========================================================================
    if (warp_id == 6) {
        asm volatile("bar.sync 5, %0;" :: "r"(WARP_SIZE));
        
        if (lane_id == 0) {
            int q8_packed = smem_formatted_q8[0];
            
            // VMAD: m × Q8
            int m_packed = ((int)m[1] << 8) | (int)m[0];
            int offset;
            asm volatile(
                "vmad.u8.u8 %0, %1, %2, 0;\n\t"
                : "=r"(offset)
                : "r"(m_packed), "r"(q8_packed)
            );
            
            // Write to shared memory for FP block
            smem_offset_result = offset;
        }
    }
    
    // ========================================================================
    // FP BLOCK: INT→FLOAT conversion + multiply by d8
    // ========================================================================
    // Wait for all threads to finish writing to shared memory
    __syncthreads();
    
    if (lane_id == 0) {
        // Read from shared memory
        int scaled_dot_result = smem_scaled_dot_result;
        int offset_result = smem_offset_result;
        
        // Extract INT values
        int scaled_dot1 = scaled_dot_result & 0xFFFF;
        int scaled_dot2 = (scaled_dot_result >> 16) & 0xFFFF;
        int offset1 = offset_result & 0xFF;
        int offset2 = (offset_result >> 8) & 0xFF;
        
        // INT→FLOAT conversion for scaled_dot1
        float scaled_f1;
        asm volatile(
            "cvt.rn.f32.s32 %0, %1;\n\t"
            : "=f"(scaled_f1)
            : "r"(scaled_dot1)
        );
        sumf_d += scaled_f1 * d8[0];
        
        // INT→FLOAT conversion for scaled_dot2
        float scaled_f2;
        asm volatile(
            "cvt.rn.f32.s32 %0, %1;\n\t"
            : "=f"(scaled_f2)
            : "r"(scaled_dot2)
        );
        sumf_d += scaled_f2 * d8[1];
        
        // INT→FLOAT conversion for offset1
        float offset_f1;
        asm volatile(
            "cvt.rn.f32.s32 %0, %1;\n\t"
            : "=f"(offset_f1)
            : "r"(offset1)
        );
        sumf_m += offset_f1 * d8[0];
        
        // INT→FLOAT conversion for offset2
        float offset_f2;
        asm volatile(
            "cvt.rn.f32.s32 %0, %1;\n\t"
            : "=f"(offset_f2)
            : "r"(offset2)
        );
        sumf_m += offset_f2 * d8[1];
    }

    // Final computation
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// Wrapper function with same signature as original
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_6pack(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
    
    return vec_dot_q4_K_q8_1_impl_6pack(bq4_K, bq8_1, iqs);
}

// Alias for compatibility with mmvq.cu
#define vec_dot_q4_K_q8_1 vec_dot_q4_K_q8_1_6pack