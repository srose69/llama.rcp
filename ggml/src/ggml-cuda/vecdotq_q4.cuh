#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

// PTX-optimized Q4 vector dot product for Pascal
// Bypasses DP4A, uses LDG.E.128, __byte_perm, HFMA2

// Load 128-bit with cache hint (bypass L1, go to L2)
static __device__ __forceinline__ int4 ldg_cg_128(const void * ptr) {
    int4 result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile (
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "l"(ptr)
    );
#else
    result = *((const int4 *)ptr);
#endif
    return result;
}

// Load with non-coherent hint (read-only data)
static __device__ __forceinline__ int4 ldg_nc_128(const void * ptr) {
    int4 result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    asm volatile (
        "ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "l"(ptr)
    );
#else
    result = *((const int4 *)ptr);
#endif
    return result;
}

// Unpack 4-bit values using __byte_perm (PRMT instruction)
static __device__ __forceinline__ void unpack_4bit_to_int8(
    const int packed, int8_t * out) {
    
    // Extract low nibbles (0x0, 0x4, 0x8, 0xC bytes)
    const int lo = __byte_perm(packed, 0, 0x4040);
    const int hi = __byte_perm(packed, 0, 0x4141);
    
    // Mask to get 4-bit values
    *((int *)&out[0]) = lo & 0x0F0F0F0F;
    *((int *)&out[4]) = (hi >> 4) & 0x0F0F0F0F;
}

// Convert int8 to half2 for HFMA2 operations
static __device__ __forceinline__ half2 int8x2_to_half2(const int8_t a, const int8_t b) {
#ifdef FAST_FP16_AVAILABLE
    return __floats2half2_rn((float)a, (float)b);
#else
    return make_half2(__int2half_rn(a), __int2half_rn(b));
#endif
}

// Q4_K vec dot using PTX optimizations
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_ptx(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
    
    // Load Q4 data with cache hints
    const int4 q4_data = ldg_nc_128(&bq4_K->qs[iqs * 4]);
    const half2 q4_dm = bq4_K->dm;
    
    // Load Q8 data
    const int * q8 = (const int *) bq8_1->qs + iqs;
    const half2 q8_ds = bq8_1->ds;
    
    // Unpack 4-bit values
    int8_t q4_unpacked[16];
    unpack_4bit_to_int8(q4_data.x, &q4_unpacked[0]);
    unpack_4bit_to_int8(q4_data.y, &q4_unpacked[8]);
    
    // Compute dot product using HFMA2
    float sum = 0.0f;
    
#ifdef FAST_FP16_AVAILABLE
    half2 acc = make_half2(0.0f, 0.0f);
    
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        // Convert to half2
        const half2 vq4 = int8x2_to_half2(q4_unpacked[i], q4_unpacked[i+1]);
        
        // Load Q8 as int8
        const int q8_packed = q8[i/2];
        const int8_t * q8_bytes = (const int8_t *)&q8_packed;
        const half2 vq8 = int8x2_to_half2(q8_bytes[0], q8_bytes[1]);
        
        // HFMA2: acc += vq4 * vq8
        acc = __hfma2(vq4, vq8, acc);
    }
    
    sum = __low2float(acc) + __high2float(acc);
#else
    // Fallback to int32 accumulation
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int8_t * q8_bytes = (const int8_t *)&q8[i/4];
        sumi += (int)q4_unpacked[i] * (int)q8_bytes[i%4];
    }
    sum = (float)sumi;
#endif
    
    // Scale by dequantization factors
    const float d = __half2float(q4_dm.x) * __low2float(q8_ds);
    const float m = __half2float(q4_dm.y) * __high2float(q8_ds);
    
    return d * sum - m * 8.0f; // 8 values
}

// Butterfly-Shuffle DP4A: pre-packed data with holes in shared memory, shuffle between thread pairs
// Each thread pair processes 8 values via 1 DP4A per thread (2 total)
// Key: 15*15=225 fits in 8 bits, no overflow with zero-padding
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq_ptx(
    const int * __restrict__ v, const int * __restrict__ u, 
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m, 
    const half2 & dm4, const half2 * __restrict__ ds8) {
    
    const int lane_id = threadIdx.x % 32;
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;
        
        #pragma unroll
        for (int j = 0; j < QI8_1; j += 2) {
            // ZERO-COST UNPACKING: lo/hi nibbles stored in SEPARATE banks by load_tiles
            // No SHR+AND here - just LDS from different offsets
            // Layout: v[0..bank_stride-1] = lo nibbles, v[bank_stride..2*bank_stride-1] = hi nibbles
            constexpr int bank_stride = (MMQ_TILE_NE_K + 1);
            
            // Load lo nibbles (already packed with holes)
            const unsigned int v_my_lo = v[j*2 + 0];  // Lo bank
            // Load hi nibbles from offset bank (already packed with holes)
            const unsigned int v_my_hi = v[bank_stride + j*2 + 0];  // Hi bank
            
            // Butterfly shuffle: swap with neighbor (lane_id XOR 1)
            const unsigned int v_neighbor_lo = __shfl_xor_sync(0xFFFFFFFF, v_my_lo, 1);
            const unsigned int v_neighbor_hi = __shfl_xor_sync(0xFFFFFFFF, v_my_hi, 1);
            
            // Select which half to use based on lane parity
            const unsigned int v_packed = (lane_id & 1) ? v_neighbor_hi : v_my_lo;
            
            // Load 8 Q8 values (two int32 = 8 bytes)
            const int u_val0 = u[i*QI8_1 + j];
            const int u_val1 = u[i*QI8_1 + j + 1];
            
            // Pack Q8 with holes: select based on lane parity
            // Even lanes: [0|u0[3]|0|u0[2]|0|u0[1]|0|u0[0]]
            // Odd lanes:  [0|u1[3]|0|u1[2]|0|u1[1]|0|u1[0]]
            const int u_src = (lane_id & 1) ? u_val1 : u_val0;
            const unsigned int u_packed = __byte_perm(u_src, 0, 0x6420);  // Pack 4 bytes with holes
            
            // ONE DP4A processes 4 Q4×Q8 products
            // ZERO ALU overhead before DP4A (no SHR+AND)
            sumi_d = ggml_cuda_dp4a(v_packed, u_packed, sumi_d);
        }
        
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }
    
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

