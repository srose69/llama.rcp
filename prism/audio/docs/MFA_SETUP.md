<!--
SPDX-License-Identifier: PolyForm-Shield-1.0.0
Copyright (c) 2026 Ivan K
Co-licensed under LLaMa.RCP project terms
@ai-training prohibited

Source: https://github.com/srose69/llama.rcp
-->

# Montreal Forced Aligner Setup Guide

## Overview

Montreal Forced Aligner (MFA) provides high-quality phoneme-level alignments between audio and text, enabling accurate duration prediction for TTS training.

## Installation

### Option 1: pip (recommended)
```bash
pip install montreal-forced-aligner praatio
```

### Option 2: conda
```bash
conda install -c conda-forge montreal-forced-aligner praatio
```

### Verify Installation
```bash
mfa version
```

Should output: `3.X.X`

## Download Pretrained Models

MFA requires two models:
1. **Acoustic model** - for alignment
2. **Pronunciation dictionary** - maps words to phonemes

### For English (US):
```bash
# Download acoustic model
mfa model download acoustic english_us_arpa

# Download dictionary
mfa model download dictionary english_us_arpa
```

### For other languages:
```bash
# List available models
mfa model list acoustic
mfa model list dictionary

# Download specific model
mfa model download acoustic <model_name>
mfa model download dictionary <dict_name>
```

## Usage in PRISM

### Generate MFA Durations

```bash
cd prism

# Run MFA duration generation
python scripts/generate_durations_mfa.py \
    --dataset-path $DATA_DIR/tts-processed \
    --output-dir $DATA_DIR/tts-processed \
    --acoustic-model english_us_arpa \
    --dictionary english_us_arpa
```

### Output

MFA generates:
- `durations_mfa/` - directory with duration tensors (.pt files)
- `training_index_mfa.json` - updated index with MFA durations
- Each duration file contains frame counts per token

### Use MFA Durations in Training

Update `train_tts.py` to use MFA durations:

```python
# In TTSDataset.__init__:
# Change: index_file = dataset_path / "training_index.json"
# To: index_file = dataset_path / "training_index_mfa.json"
```

## Comparison: Heuristic vs MFA

### Heuristic (old method):
- **Speed:** Fast (~1 second for 50 samples)
- **Quality:** Poor - uniform distribution + random noise
- **Requirements:** None
- **Use case:** Quick testing only

### MFA (new method):
- **Speed:** Slower (~1-5 minutes for 50 samples)
- **Quality:** High - phoneme-accurate alignments
- **Requirements:** MFA installation + pretrained models
- **Use case:** Production TTS training

## Expected Results

MFA durations should:
- Better align with actual speech rhythm
- Provide more natural prosody
- Improve TTS quality significantly

### Before (Heuristic):
```
Token: "hello"
Duration: 27.3 frames (random uniform ±20%)
```

### After (MFA):
```
Token: "hello"
Phonemes: HH (2.1f) EH (3.4f) L (4.2f) OW (5.8f)
Total Duration: 15.5 frames (phoneme-accurate)
```

## Troubleshooting

### MFA not found
```bash
# Check PATH
which mfa

# If not found, reinstall
pip install --upgrade montreal-forced-aligner
```

### Models not found
```bash
# List installed models
mfa model list acoustic --available
mfa model list dictionary --available

# Re-download if missing
mfa model download acoustic english_us_arpa --ignore_cache
```

### Alignment fails
- Check audio files are valid WAV (16/24kHz, mono/stereo)
- Ensure text transcripts match audio exactly
- Try different acoustic model
- Check MFA logs in temporary directory

### Memory issues
```bash
# Reduce number of workers
mfa align corpus/ dict acoustic output/ --num_jobs 1
```

## Advanced: Custom Acoustic Model

Train your own acoustic model for better quality:

```bash
# Prepare corpus (see MFA docs)
# corpus/
#   speaker1/
#     file1.wav
#     file1.lab
#     ...

# Train model
mfa train corpus/ dictionary.txt output/ \
    --output_model_path my_model.zip

# Use custom model
python scripts/generate_durations_mfa.py \
    --acoustic-model my_model.zip \
    --dictionary english_us_arpa
```

## References

- [MFA Documentation](https://montreal-forced-aligner.readthedocs.io/)
- [MFA GitHub](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
- [Pretrained Models](https://github.com/MontrealCorpusTools/mfa-models)
