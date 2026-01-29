<!--
SPDX-License-Identifier: PolyForm-Shield-1.0.0
Copyright (c) 2026 Ivan K
Co-licensed under LLaMa.RCP project terms
@ai-training prohibited

Source: https://github.com/srose69/llama.rcp
-->

# PrisMoE: Parallel Pipe Tensor Architecture

Theoretical architecture for model-to-model connections using 3 parallel transformation pipes with 512 independent workers per pipe.

## Project Structure

```
testtensors/
├── prismoe.py              # Main entry point with CLI
├── src/                    # Core implementations
│   ├── prismoe_core.py     # PrisMoE core architecture
│   ├── prism_3d_universal.py     # Universal 3D decomposition
│   └── llama_centipede_benchmark.py  # Benchmarking tools
├── utils/                  # Utility scripts
│   ├── extract_lm_head.py      # Extract lm_head from models
│   ├── extract_embedding.py    # Extract embedding layers
│   ├── compare_heads.py        # Compare weight matrices
│   ├── inspect_tile.py         # Inspect tile files
│   ├── tile_direct.py          # Tile directly from model
│   ├── tile_matrices.py        # Tile from saved files
│   └── visualize_3d_tensor.py  # 3D tensor visualization
├── docs/                   # Documentation
│   ├── LOGIC.md           # Architecture & theory
│   └── HOWTO.md           # Detailed how-to guide
└── output/                # Generated files
    ├── tiled_head/        # Tiled lm_head weights
    └── tiled_embed/       # Tiled embedding weights
```

## Quick Start

### Main Interface

```bash
# Train PrisMoE adapter
python prismoe.py train \
    --upper-model "meta-llama/Llama-3.1-8B" \
    --lower-model "Qwen/Qwen-2-7B" \
    --output adapter.pt

# Run inference
python prismoe.py infer \
    --adapter adapter.pt \
    --input "Hello world"

# Benchmark
python prismoe.py benchmark \
    --adapter adapter.pt \
    --batch-size 32
```

### Utilities

**Extract model layers:**
```bash
# Extract lm_head
python utils/extract_lm_head.py \
    --model "LiquidAI/LFM2.5-1.2B-Base" \
    --output output/lm_head.npy

# Extract embedding
python utils/extract_embedding.py \
    --model "meta-llama/Llama-3.1-8B" \
    --output output/embedding.npy \
    --layer-name "model.embed_tokens"
```

**Tile weights:**
```bash
# Direct tiling from model (memory efficient)
python utils/tile_direct.py \
    --model "LiquidAI/LFM2.5-1.2B-Base" \
    --output-dir output \
    --tile-size 512

# Tile from saved files
python utils/tile_matrices.py \
    --input-head output/lm_head.npy \
    --input-embed output/embedding.npy \
    --tile-size 512
```

**Compare matrices:**
```bash
python utils/compare_heads.py \
    --matrix1 "meta-llama/Llama-3.1-8B" \
    --matrix2 output/lm_head.npy \
    --name1 "Llama-3.1 lm_head" \
    --name2 "Saved lm_head"
```

**Inspect tiles:**
```bash
python utils/inspect_tile.py \
    output/tiled_head/tile_000.npy \
    --tile-id 0 \
    --grid-cols 4
```

**Visualize 3D tensors:**
```bash
python utils/visualize_3d_tensor.py \
    tensor_file.npy \
    --output-dir plots \
    --prefix my_tensor
```

## Architecture Overview

PrisMoE uses **3 parallel pipes** instead of one monolithic transformation matrix:

- **Pipe 1**: [512×512×512] - Cubic tensor
- **Pipe 2**: [512×512×512] - Cubic tensor  
- **Pipe 3**: [512×N×K] - Flexible tensor (N,K powers of 2)

Each pipe contains **512 independent workers** (micro-experts) that:
- Process input in parallel
- Solve independent least-squares problems
- Are managed by a MicroMoE router (~30% sparse activation)

**Key Benefits:**
- Fast training via one-shot least squares
- Lower memory through sparse activation
- Model-agnostic design
- High modularity

## Documentation

- **`docs/LOGIC.md`** - Mathematical foundations, architecture details, training procedures
- **`docs/HOWTO.md`** - Complete guide with diagrams and step-by-step explanations

## Design Principles

All utilities follow these principles:

✅ **Universal** - Work with any model, any dimensions  
✅ **Argument-based** - No hardcoded paths or model names  
✅ **Flexible** - Support both .npy and .pt formats  
✅ **Memory-efficient** - Stream processing where possible  

## Status

**Current**: Theoretical architecture with utility implementations  
**Next**: Complete PrisMoE core implementation and empirical validation

## Requirements

```bash
pip install torch transformers numpy matplotlib
```

## Examples

See individual script help:
```bash
python utils/extract_lm_head.py --help
python utils/tile_direct.py --help
python prismoe.py --help
```
