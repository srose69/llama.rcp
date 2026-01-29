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

## Target
- **Goal:** 2-3x current performance via PRMT/tiling/PTX optimizations
- **Expected:** 20-25+ TFLOPS on Pascal hardware
