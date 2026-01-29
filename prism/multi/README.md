<!--
SPDX-License-Identifier: PolyForm-Shield-1.0.0
Copyright (c) 2026 Ivan K
Co-licensed under LLaMa.RCP project terms
@ai-training prohibited

Source: https://github.com/srose69/llama.rcp
-->

# PRISM Multi - LLM-to-LLM Adapters

**Sequential composition of language models through learned translation layers.**

## Overview

PRISM Multi enables **chaining different language models** with incompatible vocabularies and architectures. Unlike weight merging (SLERP, TIES, DARE), which averages parameters, PRISM preserves each model's specialization through learned adapters.

## The Problem

Modern LLMs use different tokenization schemes:

- **Llama-3.1-8B**: 128,256 vocabulary (general English + multilingual)
- **Qwen2.5**: 151,936 vocabulary (optimized for Western European languages)
- **DeepSeek-Coder**: Custom vocabulary optimized for code
- **Multilingual models**: Language-specific tokenizers

**Example:** The text "hello world" tokenizes differently:
- Llama: `[15339, 1917]` (2 tokens)
- Qwen: `[24748, 1917, 2]` (3 tokens, different splits)

**Direct integration is impossible:**
- Token IDs don't correspond
- Embedding spaces are incompatible
- Architectures expect specific inputs

## PRISM Solution

PRISM adapters act as **learned translation layers** between model spaces, converting representations at the semantic level rather than token level.

### Architecture Overview

```
Text Input
    ↓
Model_A Tokenizer
    ↓
Model_A Embeddings
    ↓
Model_A Encoder
    ↓
Model_A Hidden States + LM Head Logits
    ↓
╔═══════════════════════════════════╗
║       PRISM ADAPTER               ║
║                                   ║
║  Input: Hidden states + Logits    ║
║  Process: Vocabulary translation  ║
║  Output: Model_B embeddings       ║
╚═══════════════════════════════════╝
    ↓
Model_B Decoder (no text embedding)
    ↓
Output
```

### Key Components

**1. Input Layer**
- Accepts Model_A hidden states `[B, L_A, D_A]`
- Accepts Model_A lm_head logits `[B, L_A, V_A]` (optional but recommended)
- Fuses both representations for richer semantic information

**2. Vocabulary Translation**
- Learns mapping between different token spaces
- Example: Llama token 15 → Qwen token 87 for same semantic content
- Handles different tokenization granularity

**3. Length Regulation**
- Expands or compresses sequence length as needed
- Llama: 50 tokens → Qwen: 65 tokens (different BPE splits)
- Learned duration/expansion patterns

**4. Feature Refinement**
- Cross-attention for semantic alignment
- Layer normalization for stability
- Residual connections

**5. Output Projection**
- Projects to Model_B's embedding space `[B, L_B, D_B]`
- Output replaces Model_B's text embedding layer

## Training

### Data Requirements

For each training example:

**From Model_A:**
- Text input
- Hidden states from encoder
- LM head logits (vocabulary distribution)

**From Model_B (ground truth):**
- Same text tokenized with Model_B's tokenizer
- Embeddings from Model_B's embedding layer

### Training Process

Both Model_A and Model_B remain **frozen**. Only PRISM adapter is trained.

**Loss function:** MSE + Cosine similarity between PRISM output and Model_B embeddings.

**Ground truth extraction:** 
- Use Model_A and Model_B tokenizers + embedding layers
- No paired audio/vision data needed
- Pure text-to-text mapping

## Use Cases

### 1. Cross-Domain Expertise

Combine specialized models:

```
Math-LLM (4B) → PRISM → Code-LLM (4B) → PRISM → Science-LLM (4B)
```

Each model trained on focused, high-quality domain data. PRISM bridges the gaps.

**Advantage:** Preserves specialization without catastrophic forgetting.

### 2. Language Chaining

```
English-LLM → PRISM → Russian-LLM → PRISM → Chinese-LLM
```

Each model trained on monolingual data for best language quality. PRISM handles cross-lingual representation transfer.

### 3. Quantization Recovery

```
Big-Model-Q2 (70B quantized to 2-bit) → PRISM → Small-Model-FP16 (7B full precision)
```

PRISM learns to compensate for quantization artifacts in the large model's output.

### 4. Architecture Bridging

Connect models with different architectures:

```
Decoder-only (GPT) → PRISM → Encoder-decoder (T5)
Transformer → PRISM → Mamba/SSM
```

## Design Principles

### Vocabulary Agnostic

PRISM doesn't require aligned vocabularies. Learns semantic mapping at feature level, not token level.

### Frozen Models

Source and target models remain unchanged:
- Preserves pre-trained capabilities
- Small training cost (PRISM: 100M-500M params)
- Modular: swap models without retraining entire pipeline

### Compositional

Stack multiple PRISM adapters:

```
Model_1 → PRISM_1→2 → Model_2 → PRISM_2→3 → Model_3 → ... → Model_N
```

Each PRISM is independent. Train once, compose freely.

### Deterministic

Well-trained PRISM produces consistent mappings. No sampling in the adapter itself.

## Scaling

### Parameter Efficiency

Example: Llama-3.1-8B → Qwen2.5-7B

- Llama: 8,000M params (frozen)
- PRISM: ~150M params (trainable)
- Qwen: 7,000M params (frozen)
- **Total:** 15,150M params, only **1%** trainable

### Latency

Sequential execution adds latency:

```
Latency_total = Σ(Latency_model_i + Latency_PRISM_i)
```

PRISM overhead: Typically 5-10% of base model latency.

### Quality Hypothesis

**Sequential specialized models > Merged generalists**

- SLERP(4B + 4B) ≈ 4-5B effective quality
- Sequential(4B → PRISM → 4B) ≈ 8B+ effective quality

**Caveat:** Requires domain complementarity and quality training data.

## Comparison to Alternatives

### vs Weight Merging (SLERP, TIES, DARE)

| Aspect | Weight Merging | PRISM |
|--------|----------------|-------|
| Vocabularies | Must be identical | Can be different |
| Architecture | Must be identical | Can be different |
| Specialization | Lost via averaging | Preserved |
| Training | None | Required for adapter |
| Inference | Single model | Sequential pipeline |
| Quality | ~Average of inputs | Compositional |

### vs Fine-tuning

| Aspect | Fine-tuning | PRISM |
|--------|-------------|-------|
| Base Model | Modified | Frozen |
| Forgetting | Catastrophic forgetting risk | No forgetting |
| Data | Full dataset | Paired features only |
| Modularity | Monolithic | Compositional |
| Cost | Retrain full model | Train small adapter |

### vs LoRA/Adapters

| Aspect | LoRA | PRISM |
|--------|------|-------|
| Purpose | Efficient fine-tuning | Model bridging |
| Vocabularies | Same as base | Different allowed |
| Composition | Single model + LoRA | Multiple models + PRISM |
| Use case | Adapt one model | Connect multiple models |

## Differences from Audio Projection

**PRISM Multi (this folder):**
- Purpose: Text-to-text LLM bridging
- Input: LLM hidden states + logits
- Output: Another LLM's embeddings
- Domain: Language models only

**PRISM Audio (../audio/):**
- Purpose: Text-to-speech generation
- Input: LLM hidden states
- Output: Acoustic features for TTS decoder
- Domain: Cross-modal (text → audio)

Both share the core PRISM concept but target different modalities.

## Training Ground Truth

Unlike audio/vision tasks requiring paired multimodal data, LLM-to-LLM PRISM uses:

**Tokenizers as ground truth generators:**
1. Take text: "Hello world"
2. Model_A tokenizer: `[15, 42]`
3. Model_B tokenizer: `[87, 12, 56]`
4. Extract Model_B embeddings for tokens `[87, 12, 56]`
5. Train PRISM to produce these embeddings from Model_A's output

**No manual annotation needed.** Entirely self-supervised from existing tokenizers.

## Future Directions

### Automatic Architecture Generation

Given Model_A and Model_B specifications:
1. Analyze vocabulary and architecture differences
2. Generate optimal PRISM configuration
3. Train with minimal supervision

### Multi-Path PRISM

Support graph structures, not just chains:

```
        ┌─> Model_B ─┐
Model_A ─┤           ├─> Fusion → Output
        └─> Model_C ─┘
```

### Online Adaptation

PRISM adapters that learn continuously from user feedback during inference.

### Compression-Aware Design

Co-design PRISM with model quantization for minimal quality loss in quantized pipelines.

## Related Concepts

- **Mixture of Experts (MoE):** Dynamic routing vs PRISM's sequential deterministic flow
- **Model Merging:** Parameter averaging vs PRISM's learned translation
- **Adapter Methods:** Fine-tuning within model vs bridging between models
- **Cross-lingual Transfer:** PRISM generalizes this to arbitrary model pairs

## See Also

- PRISM Audio: `../audio/README.md` - Text-to-speech projection
- Root documentation: `../../README.md` - Project overview
- Contributing: `../../CONTRIBUTING.md` - Contribution guidelines
- License: `../../LICENSE` - Licensing terms
