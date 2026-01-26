# Audio Projection Adapter Architecture

## System Overview

The audio projection adapter enables text-to-audio generation by bridging frozen LLM hidden states to a TTS decoder through a lightweight, trainable projection layer.

## Component Details

### 1. Base LLM: Meta-Llama-3.1-8B-Instruct

**Specifications:**
- Parameters: 8.03B
- Hidden dimension: 4096
- Layers: 32
- Attention heads: 32
- Context length: 128K tokens

**Usage:**
- Frozen during training
- Extract hidden states from final layer
- No gradient computation required

**Input:** Text prompt (tokenized)
**Output:** Hidden states tensor `[batch_size, seq_len, 4096]`

### 2. Audio Projector

**Architecture:**

```python
AudioProjector(
  linear1: Linear(4096 -> 2048)
  norm1: LayerNorm(2048)
  activation: GELU()
  linear2: Linear(2048 -> 2048)
  norm2: LayerNorm(2048)
)
```

**Parameters:** ~8.4M trainable

**Forward Pass:**
```
hidden_states [B, L, 4096]
    ↓ linear1
features [B, L, 2048]
    ↓ norm1 + gelu
intermediate [B, L, 2048]
    ↓ linear2
output [B, L, 2048]
    ↓ norm2
projected_features [B, L, 2048]
```

**Design Rationale:**
- Two-layer MLP for better feature transformation
- LayerNorm for training stability
- GELU activation for smooth gradients
- Dimension reduction: 4096 → 2048 matches TTS decoder input

### 3. TTS Decoder: Qwen3-TTS-1.7B

**Specifications:**
- Parameters: 1.7B
- Input dimension: 2048
- Output: Audio codec tokens (12Hz)
- Vocabulary: 2048 tokens per codebook
- Codebooks: 16 groups

**Usage:**
- Can be frozen or fine-tuned
- Generates discrete audio tokens
- Uses codec-based approach

**Pipeline:**
```
projected_features [B, L, 2048]
    ↓ TTS encoder
audio_features [B, L', 2048]
    ↓ Code predictor
audio_tokens [B, L', 16] (16 codebook groups)
    ↓ Vocoder
waveform [B, samples]
```

## Training Strategy

### Phase 1: Projector Only (Recommended)

**Frozen:**
- Llama-3.1-8B (100% frozen)
- Qwen3-TTS decoder (100% frozen)

**Trainable:**
- Audio projector only (~8.4M params)

**Advantages:**
- Low compute requirements (1x A100 sufficient)
- Fast training (days, not weeks)
- No risk of degrading LLM quality

### Phase 2: Joint Fine-tuning (Optional)

**Frozen:**
- Llama-3.1-8B (100% frozen)

**Trainable:**
- Audio projector
- TTS decoder layers (LoRA adapters)

**Advantages:**
- Better audio quality
- More expressive speech

**Disadvantages:**
- Higher compute requirements
- Longer training time

## Data Flow

### Training

```
Text → LLM (frozen) → Hidden States → Projector (train) → TTS (frozen) → Audio
                                          ↑                                 ↓
                                          └──────── Loss ←────────────────┘
```

### Inference

```
Text Input
    ↓
Llama-3.1-8B (frozen)
    ↓
Hidden States [B, L, 4096]
    ↓
Audio Projector
    ↓
Projected Features [B, L, 2048]
    ↓
Qwen3-TTS Decoder
    ↓
Audio Tokens [B, L', 16]
    ↓
Vocoder
    ↓
Audio Output (.wav)
```

## Loss Functions

### Primary Loss: Token Reconstruction

```python
loss = CrossEntropyLoss(
    predicted_tokens,  # [B, L', 16, 2048]
    target_tokens      # [B, L', 16]
)
```

### Optional: Perceptual Loss

```python
# Convert to waveform and compare
pred_audio = vocoder(predicted_tokens)
target_audio = vocoder(target_tokens)
perceptual_loss = MelSpectrogramLoss(pred_audio, target_audio)
```

### Combined Loss

```python
total_loss = token_loss + λ * perceptual_loss
```

Where λ = 0.1 (default weight)

## Model Export

### GGUF Format Support

The audio projector can be exported to GGUF format for efficient inference:

```python
# Export projector weights
projector_state = {
    'linear1.weight': ...,
    'linear1.bias': ...,
    'norm1.weight': ...,
    'norm1.bias': ...,
    # etc.
}

# Convert to GGUF
gguf_writer.write_tensor_data(projector_state)
```

### Usage with llama.cpp

```bash
# Load LLM with audio projection adapter
llama-server \
    -m llama-3.1-8b-instruct.gguf \
    --audio-proj audio-projector.gguf \
    --tts-model qwen3-tts-1.7b.gguf
```

## Performance Targets

### Compute Requirements

**Training:**
- GPU: 1x A100 (40GB) or 2x RTX 4090
- RAM: 64GB system memory
- Training time: 2-5 days (100K steps)

**Inference:**
- GPU: 1x RTX 4090 or equivalent
- Latency: ~200ms for 5 seconds of audio
- Throughput: ~25x realtime on A100

### Quality Metrics

**Target:**
- WER (Word Error Rate): < 5%
- MOS (Mean Opinion Score): > 4.0
- Naturalness: Human-like prosody

## Future Enhancements

1. **Streaming Support**
   - Real-time audio generation
   - Chunk-based processing

2. **Multi-speaker**
   - Speaker embedding conditioning
   - Voice cloning capabilities

3. **Emotion Control**
   - Prosody modeling
   - Emotional speech synthesis

4. **Multi-lingual**
   - Cross-lingual projection
   - Language-specific adapters
