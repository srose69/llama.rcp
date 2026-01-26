# Audio Projection Adapter

Research project for text-to-audio generation via LLM hidden state projection.

## Overview

This project implements an audio projection adapter similar to multimodal projectors (like mmproj for vision), but in reverse - projecting LLM hidden states to audio space for speech synthesis.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────┐
│   LLM       │────▶│   Audio      │────▶│     TTS     │────▶│  Audio │
│  (frozen)   │     │  Projector   │     │   Decoder   │     │ Output │
│             │     │  (trainable) │     │  (frozen)   │     │        │
└─────────────┘     └──────────────┘     └─────────────┘     └────────┘
```

### Components

1. **Base LLM**: Meta-Llama-3.1-8B-Instruct (frozen)
   - Hidden dimension: 4096
   - Used only for feature extraction from last hidden layer

2. **Audio Projector**: Trainable adapter layer
   - Input: LLM hidden states (4096 dim)
   - Output: TTS conditioning features (2048 dim)
   - Architecture: Linear projection + LayerNorm + activation

3. **TTS Decoder**: Qwen3-TTS-1.7B (frozen or fine-tuned)
   - Generates audio tokens from projected features
   - Uses codec-based approach for discrete audio generation

## Training Strategy

- **Frozen LLM**: Keep Llama-3.1-8B weights fixed
- **Trainable**: Only audio projector (~8M parameters)
- **Dataset**: LibriSpeech, Common Voice, or similar text-audio pairs
- **Loss**: Reconstruction loss on audio tokens + optional perceptual loss

## Project Structure

```
audio-projection/
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

## Goals

1. Enable audio generation from LLM hidden states
2. Minimize training compute (only small adapter)
3. Support GGUF format for efficient inference
4. Achieve reasonable speech quality with minimal parameters

## Status

🚧 **Work in Progress** - Initial research phase
