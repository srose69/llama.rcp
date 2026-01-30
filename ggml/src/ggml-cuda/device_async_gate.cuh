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
// Sense-reversing barrier: avoids deadlock on counter reset
template<int NWARPS>
struct DeviceAsyncGate {
    volatile unsigned int warp_counter;  // Count of warps that reached barrier
    volatile unsigned int sense;         // Sense flag for barrier phase
    
    __device__ __forceinline__ void init() {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            warp_counter = 0;
            sense = 0;
        }
        __syncwarp();
        __threadfence_block();
    }
    
    // Barrier: wait for ALL warps to reach this point
    // Sense-reversing pattern: safe reset without deadlock
    __device__ __forceinline__ void barrier() {
        __syncwarp();  // Sync within warp first
        __threadfence_block();  // Make writes visible
        
        // Read current sense
        const unsigned int my_sense = sense;
        
        // Lane 0 of each warp increments counter
        unsigned int arrived = 0;
        if ((threadIdx.x & 31) == 0) {
            arrived = atomicAdd((unsigned int*)&warp_counter, 1) + 1;
        }
        // Broadcast arrived to all lanes in warp
        arrived = __shfl_sync(0xFFFFFFFF, arrived, 0);
        
        if (arrived == NWARPS) {
            // Last warp: reset counter and flip sense
            warp_counter = 0;
            __threadfence_block();
            sense = 1 - my_sense;  // Flip sense to release waiters
        } else {
            // Wait for sense to flip
            while (sense == my_sense) {
                asm volatile("" ::: "memory");
            }
        }
        
        __threadfence_block();
        __syncwarp();
    }
};

// MACRO API for easy usage
#define ASYNC_GATE_INIT(gate) (gate).init()
#define ASYNC_GATE_BARRIER(gate) (gate).barrier()

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
