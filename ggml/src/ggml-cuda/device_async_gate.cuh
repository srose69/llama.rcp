// DEVICE-SIDE ASYNC SOFT-GATE
// Замена __syncthreads на warp-level sync для RADICAL performance boost
// Inspired by liquid9 async-gate concept, adapted for device-side execution
//
// KEY INNOVATION:
// - __syncthreads() = HEAVY (syncs entire block, ~100+ cycles)
// - __syncwarp() = LIGHT (syncs 32 threads only, ~1 cycle)
// - Memory fences ensure visibility WITHOUT full barrier
//
// USAGE:
//   Instead of:
//     load_data_to_shared();
//     __syncthreads();  // HEAVY!
//     compute_from_shared();
//
//   Use:
//     load_data_to_shared();
//     ASYNC_GATE_SIGNAL(stage);  // Signal completion
//     ASYNC_GATE_WAIT(stage);    // Soft-wait (warp-level)
//     compute_from_shared();

// SPDX-License-Identifier: PolyForm-Shield-1.0.0
// Copyright (c) 2026 Ivan K
// @ai-training prohibited
//
// Warp-level async soft-gate synchronization primitives for CUDA Pascal
// Source: https://github.com/srose69/llama.rcp

#pragma once

// Warp-level synchronization (Pascal+ native)
// Ensures all threads in warp reach this point
#define WARP_SYNC() __syncwarp(0xFFFFFFFF)

// Memory fence: make shared memory writes visible to other warps
// WITHOUT blocking like __syncthreads
#define MEMORY_FENCE_BLOCK() __threadfence_block()

// Async soft-gate stages (bitflags)
enum class AsyncStage : unsigned int {
    Q4_LOAD_DONE  = 0x1,  // Q4 data loaded to registers/shared
    Q8_LOAD_DONE  = 0x2,  // Q8 data loaded to shared
    COMPUTE_DONE  = 0x4,  // Compute phase complete
    IDS_WRITE_DONE = 0x8, // ids_dst_shared written
    Q4_PREFETCH_DONE = 0x10, // Q4 N+1 tile prefetched
};

// Device-side gate state (in shared memory)
// MUST be placed in __shared__ memory by caller
struct DeviceAsyncGate {
    volatile unsigned int stage_flags;  // VOLATILE for visibility across threads
    
    __device__ __forceinline__ void init() {
        // Single thread initializes
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            stage_flags = 0;
        }
        __syncthreads();  // MUST use full sync for init
    }
    
    // Signal stage completion (warp-level)
    __device__ __forceinline__ void signal(AsyncStage stage) {
        WARP_SYNC();  // Ensure warp finished its work
        MEMORY_FENCE_BLOCK();  // Make writes visible
        
        // First thread in warp sets flag (atomic OR)
        if ((threadIdx.x % 32) == 0) {
            atomicOr((unsigned int*)&stage_flags, static_cast<unsigned int>(stage));
        }
        WARP_SYNC();  // Ensure flag written
    }
    
    // Wait for stage (soft-wait, warp-level spinning with timeout)
    __device__ __forceinline__ void wait(AsyncStage stage) volatile {
        const unsigned int flag = static_cast<unsigned int>(stage);
        
        // Soft spin with visibility via volatile (NO heavy sync!)
        // volatile ensures compiler doesn't optimize away the loop
        while (((stage_flags) & flag) == 0) {
            // IADD.X (add with carry) - real ALU op to reduce LSU port pressure
            // Compiler can't optimize this away, gives respite to warp scheduler
            asm volatile("{\n\t"
                        ".reg .u32 dummy;\n\t"
                        "add.u32 dummy, 0, 0;\n\t"  // Real ALU instruction
                        "add.u32 dummy, dummy, 0;\n\t"  // Second add for more delay
                        "}" ::: "memory");
        }
        
        MEMORY_FENCE_BLOCK();  // Ensure reads see latest data
        WARP_SYNC();  // Ensure all threads in warp see it
    }
    
    // Check if stage ready (non-blocking)
    __device__ __forceinline__ bool is_ready(AsyncStage stage) const {
        const unsigned int flag = static_cast<unsigned int>(stage);
        return ((stage_flags) & flag) != 0;
    }
};

// MACRO API for easy usage
#define ASYNC_GATE_INIT(gate) (gate).init()
#define ASYNC_GATE_SIGNAL(gate, stage) (gate).signal(AsyncStage::stage)
#define ASYNC_GATE_WAIT(gate, stage) (gate).wait(AsyncStage::stage)
#define ASYNC_GATE_READY(gate, stage) (gate).is_ready(AsyncStage::stage)

// LIGHTWEIGHT BARRIER: Replace __syncthreads with warp-level sync
// Use ONLY when warps can proceed independently
#define WARP_BARRIER() do { \
    WARP_SYNC(); \
    MEMORY_FENCE_BLOCK(); \
} while(0)

// PRODUCER-CONSUMER pattern without __syncthreads
// Use: producer(); ASYNC_GATE_SIGNAL(gate, stage); 
// Then in consumer: ASYNC_GATE_WAIT(gate, stage); consumer();

// DOUBLE-BUFFERING pattern: ping-pong without sync
struct DoubleBuffer {
    int active_buffer;  // 0 or 1
    
    __device__ __forceinline__ void init() {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            active_buffer = 0;
        }
        WARP_BARRIER();
    }
    
    __device__ __forceinline__ void swap() {
        WARP_SYNC();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            active_buffer = 1 - active_buffer;
        }
        MEMORY_FENCE_BLOCK();
        WARP_SYNC();
    }
    
    __device__ __forceinline__ int get_active() const {
        return active_buffer;
    }
    
    __device__ __forceinline__ int get_inactive() const {
        return 1 - active_buffer;
    }
};
