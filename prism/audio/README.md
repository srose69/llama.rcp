<!--
SPDX-License-Identifier: PolyForm-Shield-1.0.0
Copyright (c) 2026 Ivan K
Co-licensed under LLaMa.RCP project terms
@ai-training prohibited

Source: https://github.com/srose69/llama.rcp
-->

# PRISM

**P**rojection **R**esearch **I**n **S**peech **M**odeling

Text-to-Speech via LLM Hidden State Projection

## Overview

PRISM bridges pretrained language models and TTS systems through a lightweight trainable adapter. Unlike end-to-end LLM-based TTS models, PRISM keeps both the LLM encoder and TTS decoder frozen, training only a small projection layer between them.

### How PRISM Differs from mmproj

**mmproj (LLaVA-style vision adapters):**
- Projects **INTO** LLM token space: `Image → Vision Encoder → mmproj → LLM`
- External modality → LLM direction
- Adds visual tokens to LLM's input

**PRISM:**
- Projects **OUT OF** LLM space: `Text → LLM → PRISM → TTS Decoder`
- LLM → External modality direction  
- Replaces LLM's output head + TTS's input embedding

Think of PRISM as "gluing" two pretrained models by removing their boundary layers (LLM's `lm_head` and TTS's input embedding) and inserting a trainable projector between them.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────┐
│   LLM       │────▶│   Audio      │────▶│     TTS     │────▶│  Audio │
│  (frozen)   │     │  Projector   │     │   Decoder   │     │ Output │
│             │     │  (trainable) │     │  (frozen)   │     │        │
└─────────────┘     └──────────────┘     └─────────────┘     └────────┘
```

### Components

1. **LLM Encoder**: Any pretrained language model (frozen)
   - Hidden dimension: typically 2048-8192
   - Extracts semantic features from text
   - Examples: Llama, Qwen, GPT, etc.

2. **PRISM Projector**: Trainable adapter (~2-8M params)
   - Bridges LLM feature space → TTS feature space
   - Architecture: Multi-layer projection with normalization
   - Supports duration prediction and alignment

3. **TTS Decoder**: Any pretrained TTS model (frozen or fine-tuned)
   - Generates audio from projected features
   - Can use codec-based, mel-spectrogram, or other approaches

## Training Strategy

**Core Principle:** Modular composition over monolithic training

- **Frozen models**: Both LLM encoder and TTS decoder remain frozen
- **Trainable**: Only PRISM projector (~2-8M parameters, <1% of total)
- **Dataset**: Paired text-audio data (LibriSpeech, Common Voice, etc.)
- **Loss**: Acoustic reconstruction + duration prediction

## Project Structure

```
prism/
├── README.md                 # This file
├── docs/
│   ├── architecture.md       # Detailed architecture
│   └── training.md           # Training guide
├── src/
│   ├── projector.py          # Audio projector implementation
│   ├── data_loader.py        # Dataset handling
│   ├── train.py              # Training script
│   └── inference.py          # Inference pipeline
├── configs/
│   └── default.yaml          # Default configuration
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- GGUF Python bindings
- Audio processing libraries (librosa, soundfile)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Train projector
python src/train.py --config configs/default.yaml

# Run inference
python src/inference.py --text "Hello world" --output output.wav
```

## Why LLM Encoder?

### The Semantic Understanding Advantage

Traditional TTS models use simple token embeddings - static lookup tables where each token maps to a fixed vector. This lacks contextual awareness:

**TTS Embedding Layer (Simple):**
- Static vectors per token
- No context understanding
- "bank" (river) = "bank" (finance) - same vector
- No prosody or emotional information

**LLM Hidden States (Rich):**
- Contextually aware representations
- Processed through multi-layer transformer
- Disambiguates meaning through attention
- Encodes prosody hints from punctuation
- Captures long-range dependencies
- Semantic richness from pre-training

### Concrete Benefits

**1. Semantic Disambiguation**
- "I saw her duck" - LLM understands context (she ducked vs her pet duck)
- Attention mechanism resolves ambiguity
- Better acoustic features for TTS decoder

**2. Prosody Encoding**
- "Really?" (question) vs "Really." (statement) - different hidden states
- Punctuation patterns learned during LLM training
- Sarcasm, irony, emotional tone encoded implicitly

**3. Context Handling**
- Pronouns, references, sentence structure
- Important vs filler words (affects duration)
- Natural phrasing and rhythm

### PRISM's Role

PRISM doesn't create semantic information - it **translates** what LLM already encoded:

1. LLM processes text → rich semantic hidden states
2. PRISM learned mapping → TTS-compatible acoustic features  
3. TTS decoder generates audio from those features

**Key Insight:** Better input semantics (from LLM) → better acoustic features → better audio quality from TTS decoder.

## Vocabulary Translation

PRISM enables using **different tokenizers** for LLM and TTS:

### The Problem

- Llama-3.1: 128K vocabulary (optimized for general text)
- Qwen TTS: 151K vocabulary (optimized for multilingual)
- Token IDs don't correspond
- Different tokenization granularity

##<!--
SPDX-License-Identifier: PolyForm-Shield-1.0.0
Copyright (c) 2026 Ivan K
Co-licensed under LLaMa.RCP project terms
@ai-training prohibited

Source: https://github.com/srose69/llama.rcp
-->

# PRISM Audiolution

Works at **feature level**, not token level:

```
Text: "Hello world"
↓
Llama Tokenizer: [15339, 1917] (2 tokens)
↓
Llama Encoder: hidden states [2, 4096]
↓
PRISM Adapter: acoustic features [frames, 2048]
↓
Qwen TTS Decoder: audio (bypasses Qwen tokenizer entirely!)
```

**Advantages:**
- Use best LLM encoder (Llama's better semantic understanding)
- Use best TTS decoder (Qwen's audio synthesis)
- No need for aligned vocabularies

## Multilingual Support: Micro-MoE Architecture

### Current Implementation (Vanilla)

Single path for all languages - no language-specific optimization.

### Future: Language-Specific Experts (Micro-MoE)

**Architecture:**

```
LLM Hidden States
    ↓
Shared Input Projection
    ↓
Language Router (learned)
    ↓
┌─────────────┬─────────────┐
│  EN Expert  │  RU Expert  │
│  (English)  │  (Russian)  │
└─────────────┴─────────────┘
    ↓              ↓
EN Output     RU Output
Projection    Projection
    ↓              ↓
Acoustic Features for TTS
```

**Components:**

1. **Language Router** (Micro-MoE)
   - Analyzes LLM features
   - Detects language from semantic patterns
   - Learned (not hardcoded rules)
   - Outputs probabilities: P(EN), P(RU), etc.

2. **Language-Specific Experts**
   - EN Expert: Optimized for English phonetics/prosody
   - RU Expert: Optimized for Russian phonetics/prosody
   - Each has own layers + output projection
   - Train independently, no catastrophic forgetting

3. **Conditional Compute**
   - Only active expert processes each input
   - Efficient inference (not all experts active)

### Training Strategies

**Mixed Training (Recommended):**
- 50% English examples + 50% Russian examples
- Shuffle in each batch
- Router learns language detection
- Experts specialize automatically
- No forgetting issues

**Sequential Training (Alternative):**
1. Pre-train on English (step 1-100K)
2. Add Russian expert + router
3. Fine-tune with replay:
   - 70% new Russian data
   - 30% English replay data
4. Final mixed fine-tune (10K steps)

**Advantages:**
- Prevents catastrophic forgetting
- Each language gets optimized path
- Scalable to more languages
- Better quality than unified model

### Russian Language Improvement

**Current Challenge:**
- Qwen tokenizer optimized for Western European languages
- Russian underrepresented in vocabulary
- Inefficient tokenization for Cyrillic

**With Llama + PRISM:**
- Llama has better Russian support (128K vocab covers Cyrillic efficiently)
- Better understanding of Russian idioms, context
- Train PRISM on Russian dataset
- Qwen decoder can generate Russian audio (already capable)

**Result:** Llama semantic understanding + Qwen audio synthesis = better Russian TTS quality.

## Scaling Considerations

### Parameter Efficiency

Example: Llama-8B → Qwen-1.7B

- Llama: 8,000M params (frozen)
- PRISM: ~147M params (trainable)
- Qwen: 1,700M params (frozen)
- **Total:** 9,847M params, only **1.5%** trainable

With Micro-MoE (EN + RU):
- Shared layers: 100M
- EN expert: 50M
- RU expert: 50M  
- Router: 5M
- **Total PRISM:** ~205M (still <3% of total pipeline)

### Bottleneck Analysis

**Q: Is Qwen TTS decoder the bottleneck?**

**A: Depends on task complexity.**

For most TTS tasks: **No**
- Qwen-1.7B sufficient for natural speech
- Bottleneck was text understanding (Qwen tokenizer)
- PRISM + Llama solves this

For complex tasks: **Potentially**
- Celebrity voice cloning
- Highly emotional/expressive speech
- Singing voice synthesis
- Multiple speakers

**Solution:** Use larger TTS decoder (7B+) if needed.

**Balanced Pipeline Examples:**
- Llama-8B + PRISM-150M + Qwen-1.7B: Good for standard TTS
- Llama-70B + PRISM-500M + Qwen-1.7B: Better semantics, Qwen starts limiting
- Llama-70B + PRISM-1B + BigTTS-7B: Maximum quality, balanced

### Why Not Distill Qwen Into PRISM?

**Different Tasks:**
- PRISM: Feature translation (LLM space → TTS space)
- Qwen Decoder: Audio synthesis (acoustic features → waveform)

**Qwen Decoder Specialization:**
- Vocoder components (HiFi-GAN, WaveNet)
- Flow matching / diffusion models
- Codec integration for audio tokens
- Phase reconstruction

These are **architectural specializations**, not just learned weights. Even 1B PRISM can't replicate without proper TTS architecture.

## Training Ground Truth

**For audio PRISM (unlike text-to-text):**

Requires **paired (text, audio) datasets**:

1. Text processed through Llama → hidden states
2. Audio processed through Qwen TTS encoder → acoustic features
3. Train PRISM: hidden states → acoustic features
4. Loss: MSE + Cosine similarity

**Why audio needed:**
- Acoustic features contain frequency, amplitude, timing information
- Cannot be derived from text alone
- Need real speech examples for prosody, pronunciation

**Datasets:**
- LibriSpeech (English): Professional audiobook recordings
- Common Voice (Multilingual): Crowdsourced recordings  
- Custom datasets for specific languages/voices

## Design Goals

1. **Modularity**: Use pretrained models as black boxes
2. **Efficiency**: Train only small adapter vs billions of parameters
3. **Flexibility**: Support any LLM + TTS combination
4. **Quality**: Achieve natural speech through semantic enrichment
5. **Multilingual**: Language-specific experts without catastrophic forgetting
6. **Scalability**: Easily add new languages or upgrade components

## Status

**Active Research**
- ✅ Core vanilla architecture implemented (147M params)
- ✅ English training on LJSpeech dataset
- ✅ Checkpoints available (500 steps trained)
- 🚧 Micro-MoE multilingual architecture (planned)
- 🚧 Russian language support (planned)
- 🚧 Evaluation metrics and quality assessment

## Related

- **PRISM Multi** (`../multi/`): Text-to-text LLM cascading
- **Architecture Details**: [`docs/architecture.md`](docs/architecture.md)
- **MFA Setup**: [`docs/MFA_SETUP.md`](docs/MFA_SETUP.md)
