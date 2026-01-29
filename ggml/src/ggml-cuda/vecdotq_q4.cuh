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

// Double-Packed DP4A: pack 8 Q4 nibbles with holes, process 8 values per iteration
// Key: 15*15=225 fits in 8 bits, nibbles with zero-padding don't overflow
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq_ptx(
    const int * __restrict__ v, const int * __restrict__ u, 
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m, 
    const half2 & dm4, const half2 * __restrict__ ds8) {
    
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;
        
        #pragma unroll
        for (int j = 0; j < QI8_1; j += 2) {
            // Load FULL 32 bits = 8 nibbles (no mask, no shift on nibbles)
            // v[j] contains [n7|n6|n5|n4|n3|n2|n1|n0] in 4-bit nibbles
            const unsigned int v_full = v[j] >> (4*i);
            
            // Unpack 8 nibbles: low 4 bytes [n3|n2|n1|n0], high 4 bytes [n7|n6|n5|n4]
            const unsigned int v_lo4 = v_full & 0x0F0F0F0F;        // [0|n6|0|n4|0|n2|0|n0] after perm
            const unsigned int v_hi4 = (v_full >> 4) & 0x0F0F0F0F; // [0|n7|0|n5|0|n3|0|n1] after perm
            
            // Pack with holes using __byte_perm: even/odd nibbles
            const unsigned int v_even = __byte_perm(v_lo4, 0, 0x6420);  // [0|n6|0|n4|0|n2|0|n0]
            const unsigned int v_odd  = __byte_perm(v_hi4, 0, 0x6420);  // [0|n7|0|n5|0|n3|0|n1]
            
            // Load 8 Q8 values (two int32 = 8 bytes)
            const int u_val0 = u[i*QI8_1 + j];
            const int u_val1 = u[i*QI8_1 + j + 1];
            
            // Pack Q8 with holes: 4 from u_val0, 4 from u_val1
            // u_even gets [0|u0[2]|0|u0[0]] in low 16 bits, [0|u1[2]|0|u1[0]] in high 16 bits
            const unsigned int u_even = __byte_perm(u_val0, u_val1, 0x6240);  // Select bytes for even positions
            const unsigned int u_odd  = __byte_perm(u_val0, u_val1, 0x7351);  // Select bytes for odd positions
            
            // Two DP4A calls process 8 Q4×Q8 products
            sumi_d = ggml_cuda_dp4a(v_even, u_even, sumi_d);
            sumi_d = ggml_cuda_dp4a(v_odd,  u_odd,  sumi_d);
        }
        
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }
    
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

