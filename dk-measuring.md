# DK Measuring - Pascal MMQ Baseline

## Hardware
- **GPU:** NVIDIA GeForce GTX 1080 (Pascal sm_61, 8GB VRAM)
- **GPU 2:** Tesla P4 (Pascal sm_61, 8GB VRAM)
- **Backend:** CUDA + MMQ (default)

## Model
- **Model:** Qwen3-4B-Instruct Q4_K_M
- **Size:** 2.32 GiB
- **Params:** 4.02B
- **Offload:** 99 layers (full GPU)

## Baseline Results (Build 2b80748)

| Metric | Speed (t/s) | Notes |
|--------|-------------|-------|
| Prompt Processing (512 tok) | **809.23** ± 24.98 | Batch processing |
| Token Generation (128 tok) | **42.48** ± 0.15 | Autoregressive |

## Configuration
- **Build:** pascal branch @ 2b80748
- **MMQ:** Enabled (default DP4A path for Pascal)
- **Quantization:** Q4_K_M
- **Context:** 512 prompt + 128 generation

## Q4_K Code Extraction Test (Build cce2320)

| Metric | Speed (t/s) | Δ from baseline |
|--------|-------------|-----------------|
| Prompt Processing (512 tok) | **798.47** ± 8.07 | -1.3% |
| Token Generation (128 tok) | **42.14** ± 0.72 | -0.8% |

**Status:** Q4_K MMQ code successfully extracted to `ggml/src/ggml-cuda/mmq_q4.cuh`  
**Performance:** Identical (within measurement error)

### Optimization Attempts

| Optimization | Prompt Δ | Generation Δ | Note |
|--------------|----------|--------------|------|
| `__ldg` read-only cache | -3.6% | -0.8% | Memory already efficient |
| Full K-unroll | -3.0% | -0.8% | Increases register pressure |
| Butterfly shuffle | -48.9% | +10.5% | Warp divergence, double shuffle overhead |
| Log-space (LFP8 approach) | -87.6% | +13.1% | Expensive exp2f, accuracy loss |
| LOP3.LUT nibble extraction | -82.0% | +4.0% | 10+ instructions vs 1 DP4A |
| Double-Packed DP4A (4-value) | -34.1% | +10.4% | Packing overhead dominates |
| **Double-Packed DP4A (8-value)** | **+2.9%** | **+10.8%** | **First win on both metrics** |

## Double-Packed DP4A Optimization (Build cce2320)

**Approach:** Pack 8 Q4 nibbles with zero-padding holes, process 8 values per 2 DP4A calls instead of 4 values per 1 DP4A.

| Metric | Baseline | Optimized | Δ |
|--------|----------|-----------|---|
| Prompt Processing (512 tok) | 809.23 t/s | **832.38** ± 6.63 t/s | **+2.9%** ✅ |
| Token Generation (128 tok) | 42.48 t/s | **47.07** ± 0.46 t/s | **+10.8%** ✅ |

**Technical Details:**
- Load full 32 bits (8 nibbles) per iteration instead of 4
- Use `__byte_perm` (PRMT) to pack nibbles with holes: `[0|n7|0|n5|0|n3|0|n1]` and `[0|n6|0|n4|0|n2|0|n0]`
- Pack Q8 values similarly from two int32 using `__byte_perm`
- Two DP4A calls process 8 Q4×Q8 products with no overflow (15×15=225 fits in 8 bits)
- Reduced loop iterations by 2x (j += 2)
- 4 PRMT + 2 DP4A = 6 cycles for 8 values vs 1 DP4A = 1 cycle for 4 values
- Lower loop overhead + better instruction pipelining = net win

**Files Modified:**
- `ggml/src/ggml-cuda/vecdotq_q4.cuh` - `vec_dot_q4_K_q8_1_impl_mmq_ptx()`

## Target
- **Goal:** 2-3x current performance via PRMT/tiling/PTX optimizations
- **Current:** 1.03x prompt, 1.11x generation
- **Expected:** 20-25+ TFLOPS on Pascal hardware
