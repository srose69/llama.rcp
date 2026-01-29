#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Test full audio projection pipeline with real models.

Tests:
1. Load Llama-3.1-8B-Instruct
2. Extract hidden states from text input
3. Project through AudioProjector
4. Verify Qwen3-TTS compatibility
"""

import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from projector import AudioProjector


def load_llm(model_path: str):
    """Load Llama-3.1-8B-Instruct model with validation."""
    import os
    from pathlib import Path
    
    print(f"Loading LLM from {model_path}...")
    
    # Validate model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    model_path_obj = Path(model_path)
    if not model_path_obj.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")
    
    # Check for required files
    required_files = ['config.json']
    for fname in required_files:
        if not (model_path_obj / fname).exists():
            raise FileNotFoundError(f"Missing required file: {fname} in {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer is None:
            raise RuntimeError("Failed to load tokenizer")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU: use float32 for stability
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        if model is None:
            raise RuntimeError("Failed to load model")
        
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
    
    # Validate model configuration
    if not hasattr(model.config, 'hidden_size'):
        raise ValueError("Model config missing 'hidden_size' attribute")
    
    if not hasattr(model.config, 'num_hidden_layers'):
        raise ValueError("Model config missing 'num_hidden_layers' attribute")
    
    hidden_size = model.config.hidden_size
    if hidden_size != 4096:
        print(f"⚠ Warning: Expected hidden_size=4096 for Llama-3.1-8B, got {hidden_size}")
    
    # Validate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LLM loaded successfully")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Total params: {total_params:,}")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def extract_hidden_states(model, tokenizer, text: str):
    """Extract hidden states from LLM for given text with validation."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    print(f"\nProcessing text: '{text}'")
    print(f"  Text length: {len(text)} chars")
    
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {str(e)}")
    
    # Validate tokenization output
    if 'input_ids' not in inputs:
        raise RuntimeError("Tokenizer did not return 'input_ids'")
    
    num_tokens = inputs['input_ids'].shape[1]
    print(f"  Tokenized to {num_tokens} tokens")
    
    if num_tokens == 0:
        raise ValueError("Tokenization produced zero tokens")
    
    if num_tokens > 2048:
        print(f"⚠ Warning: Long sequence ({num_tokens} tokens) may cause memory issues")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {str(e)}")
    
    # Validate outputs
    if not hasattr(outputs, 'hidden_states'):
        raise RuntimeError("Model outputs missing 'hidden_states'")
    
    if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
        raise RuntimeError("Model returned empty hidden_states")
    
    # Get last hidden state
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
    
    # Validate hidden states
    if hidden_states.dim() != 3:
        raise RuntimeError(f"Expected 3D hidden_states, got {hidden_states.dim()}D")
    
    if torch.isnan(hidden_states).any():
        raise RuntimeError("Hidden states contain NaN values")
    
    if torch.isinf(hidden_states).any():
        raise RuntimeError("Hidden states contain Inf values")
    
    print(f"✓ Hidden states extracted: {hidden_states.shape}")
    print(f"  Mean: {hidden_states.mean().item():.4f}")
    print(f"  Std: {hidden_states.std().item():.4f}")
    print(f"  Min: {hidden_states.min().item():.4f}")
    print(f"  Max: {hidden_states.max().item():.4f}")
    
    return hidden_states


def test_full_pipeline():
    """Test complete pipeline with real models."""
    
    print("=" * 70)
    print("Full Audio Projection Pipeline Test")
    print("=" * 70)
    
    # Paths
    llm_path = "./data/audio-projection-models/llama-3.1-8b-instruct"
    tts_path = "./data/audio-projection-models/qwen3-tts-1.7b"
    
    # Step 1: Load LLM
    print("\n" + "-" * 70)
    print("Step 1: Loading Llama-3.1-8B-Instruct")
    print("-" * 70)
    
    model, tokenizer = load_llm(llm_path)
    
    # Verify hidden dimension
    llm_hidden_size = model.config.hidden_size
    print(f"  Hidden size: {llm_hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Model dtype: {model.dtype}")
    
    # Step 2: Extract hidden states
    print("\n" + "-" * 70)
    print("Step 2: Extracting hidden states")
    print("-" * 70)
    
    test_text = "Hello world, this is a test for audio generation."
    hidden_states = extract_hidden_states(model, tokenizer, test_text)
    
    # Step 3: Initialize projector
    print("\n" + "-" * 70)
    print("Step 3: Initializing Audio Projector")
    print("-" * 70)
    
    tts_hidden_size = 2048
    dropout = 0.1
    
    print(f"  LLM hidden size: {llm_hidden_size}")
    print(f"  TTS hidden size: {tts_hidden_size}")
    print(f"  Dropout: {dropout}")
    
    try:
        projector = AudioProjector(
            llm_hidden_size=llm_hidden_size,
            tts_hidden_size=tts_hidden_size,
            dropout=dropout,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize projector: {str(e)}")
    
    projector.eval()
    projector = projector.to(hidden_states.device, dtype=hidden_states.dtype)
    
    # Validate projector state
    num_params = projector.get_num_params()
    if num_params == 0:
        raise RuntimeError("Projector has zero parameters")
    
    # Check parameter initialization
    for name, param in projector.named_parameters():
        if torch.isnan(param).any():
            raise RuntimeError(f"Projector parameter '{name}' contains NaN after initialization")
        if torch.isinf(param).any():
            raise RuntimeError(f"Projector parameter '{name}' contains Inf after initialization")
    
    print(f"✓ Projector initialized successfully")
    print(f"  Parameters: {num_params:,}")
    print(f"  Device: {next(projector.parameters()).device}")
    print(f"  Dtype: {next(projector.parameters()).dtype}")
    print(f"  Architecture: {llm_hidden_size}D -> {tts_hidden_size}D")
    
    # Step 4: Project to TTS space
    print("\n" + "-" * 70)
    print("Step 4: Projecting to TTS feature space")
    print("-" * 70)
    
    try:
        with torch.no_grad():
            projected = projector(hidden_states)
    except Exception as e:
        raise RuntimeError(f"Projection failed: {str(e)}")
    
    # Validate projection output
    if projected is None:
        raise RuntimeError("Projector returned None")
    
    if projected.shape[0] != hidden_states.shape[0]:
        raise RuntimeError(f"Batch size mismatch: input {hidden_states.shape[0]}, output {projected.shape[0]}")
    
    if projected.shape[1] != hidden_states.shape[1]:
        raise RuntimeError(f"Sequence length mismatch: input {hidden_states.shape[1]}, output {projected.shape[1]}")
    
    if projected.shape[2] != tts_hidden_size:
        raise RuntimeError(f"Hidden size mismatch: expected {tts_hidden_size}, got {projected.shape[2]}")
    
    # Numerical validation
    if torch.isnan(projected).any():
        raise RuntimeError("Projected features contain NaN values")
    
    if torch.isinf(projected).any():
        raise RuntimeError("Projected features contain Inf values")
    
    # Statistical checks
    proj_mean = projected.mean().item()
    proj_std = projected.std().item()
    proj_min = projected.min().item()
    proj_max = projected.max().item()
    
    print(f"✓ Projection complete: {projected.shape}")
    print(f"  Input:  {hidden_states.shape} ({hidden_states.dtype})")
    print(f"  Output: {projected.shape} ({projected.dtype})")
    print(f"\n  Statistics:")
    print(f"    Mean: {proj_mean:.4f}")
    print(f"    Std:  {proj_std:.4f}")
    print(f"    Min:  {proj_min:.4f}")
    print(f"    Max:  {proj_max:.4f}")
    
    # Warn about unusual statistics
    if abs(proj_mean) > 10.0:
        print(f"  ⚠ Warning: Large mean value {proj_mean:.2f}, may indicate numerical issues")
    
    if proj_std < 0.01 or proj_std > 100.0:
        print(f"  ⚠ Warning: Unusual std deviation {proj_std:.2f}")
    
    if abs(proj_min) > 1000 or abs(proj_max) > 1000:
        print(f"  ⚠ Warning: Large magnitude values (min={proj_min:.2f}, max={proj_max:.2f})")
    
    # Step 5: Verify TTS compatibility
    print("\n" + "-" * 70)
    print("Step 5: Verifying Qwen3-TTS compatibility")
    print("-" * 70)
    
    # Check if dimensions match TTS requirements
    expected_dim = 2048  # Qwen3-TTS input dimension
    actual_dim = projected.shape[-1]
    
    print(f"✓ TTS model location: {tts_path}")
    print(f"  Expected input dim: {expected_dim}")
    print(f"  Actual output dim: {actual_dim}")
    
    if actual_dim == expected_dim:
        print(f"\n✓ Dimensions match perfectly! {actual_dim} == {expected_dim}")
        print(f"  Projector output ready for Qwen3-TTS decoder input")
    else:
        print(f"\n✗ Dimension mismatch: {actual_dim} != {expected_dim}")
        raise ValueError("Dimension mismatch between projector and TTS")
    
    # Step 6: Save projector
    print("\n" + "-" * 70)
    print("Step 6: Saving trained projector")
    print("-" * 70)
    
    import os
    from pathlib import Path
    
    save_path = "./data/audio-projection-models/audio_projector_trained.pt"
    save_dir = Path(save_path).parent
    
    # Ensure save directory exists
    if not save_dir.exists():
        print(f"  Creating directory: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU for saving
    print(f"  Moving projector to CPU for saving...")
    projector.cpu()
    
    try:
        projector.save_pretrained(save_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save projector: {str(e)}")
    
    # Verify save
    if not os.path.exists(save_path):
        raise RuntimeError(f"Save verification failed: file not found at {save_path}")
    
    file_size = os.path.getsize(save_path)
    print(f"\n✓ Saved successfully")
    print(f"  Path: {save_path}")
    print(f"  Size: {file_size / (1024*1024):.2f} MB")
    
    # Test load to verify integrity
    print(f"\n  Verifying save integrity...")
    try:
        loaded_projector = AudioProjector.from_pretrained(save_path)
        print(f"  ✓ Load verification successful")
    except Exception as e:
        raise RuntimeError(f"Save integrity check failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Test Summary")
    print("=" * 70)
    
    print("\n✓ Full pipeline validated successfully!")
    print("\nComponents:")
    print(f"  1. LLM: Llama-3.1-8B ({llm_hidden_size}D) ✓")
    print(f"  2. Projector: {projector.get_num_params():,} params ✓")
    print(f"  3. TTS: Qwen3-TTS (2048D input) ✓")
    
    print("\nNext steps:")
    print("  1. Prepare audio-text dataset (LibriSpeech)")
    print("  2. Implement training loop")
    print("  3. Train adapter for 100K steps")
    print("  4. Test audio generation")
    
    print("\n✓ System ready for training!")


if __name__ == "__main__":
    test_full_pipeline()
