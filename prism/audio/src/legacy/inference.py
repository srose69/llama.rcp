#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Inference pipeline for Audio Projection Adapter.

Provides end-to-end inference from text to audio features:
1. Load LLM and tokenize input
2. Extract LLM hidden states
3. Project through trained adapter
4. Output TTS-ready features
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from projector import AudioProjector


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    llm_path: str
    projector_path: str
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # Generation parameters
    max_length: int = 512
    use_cache: bool = True
    
    def validate(self):
        """Validate configuration."""
        if not Path(self.llm_path).exists():
            raise FileNotFoundError(f"LLM path not found: {self.llm_path}")
        
        if not Path(self.projector_path).exists():
            raise FileNotFoundError(f"Projector path not found: {self.projector_path}")
        
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}")
        
        print("✓ Inference config validated")


class AudioProjectionInference:
    """
    End-to-end inference pipeline.
    
    Usage:
        config = InferenceConfig(
            llm_path="path/to/llama",
            projector_path="path/to/projector.pt"
        )
        
        pipeline = AudioProjectionInference(config)
        features = pipeline.generate("Hello world!")
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.config.validate()
        
        print("=" * 70)
        print("Initializing Audio Projection Pipeline")
        print("=" * 70)
        
        # Load LLM
        print("\n1. Loading LLM...")
        self.load_llm()
        
        # Load projector
        print("\n2. Loading Audio Projector...")
        self.load_projector()
        
        print("\n✓ Pipeline ready")
        print("=" * 70)
    
    def load_llm(self):
        """Load LLM with validation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_path)
            if self.tokenizer is None:
                raise RuntimeError("Failed to load tokenizer")
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_path,
                torch_dtype=self.config.dtype,
                device_map=self.config.device,
                low_cpu_mem_usage=True,
            )
            
            if self.llm is None:
                raise RuntimeError("Failed to load LLM")
            
            self.llm.eval()
            
        except Exception as e:
            raise RuntimeError(f"Error loading LLM: {str(e)}")
        
        # Validate LLM
        if not hasattr(self.llm.config, 'hidden_size'):
            raise ValueError("LLM config missing 'hidden_size'")
        
        self.llm_hidden_size = self.llm.config.hidden_size
        
        total_params = sum(p.numel() for p in self.llm.parameters())
        print(f"  ✓ LLM loaded")
        print(f"    Hidden size: {self.llm_hidden_size}")
        print(f"    Parameters: {total_params:,}")
        print(f"    Device: {next(self.llm.parameters()).device}")
        print(f"    Dtype: {next(self.llm.parameters()).dtype}")
    
    def load_projector(self):
        """Load trained projector with validation."""
        try:
            self.projector = AudioProjector.from_pretrained(self.config.projector_path)
            
            if self.projector is None:
                raise RuntimeError("Failed to load projector")
            
            self.projector.eval()
            self.projector = self.projector.to(self.config.device, dtype=self.config.dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error loading projector: {str(e)}")
        
        # Validate projector dimensions
        if self.projector.llm_hidden_size != self.llm_hidden_size:
            raise ValueError(
                f"Dimension mismatch: LLM hidden_size={self.llm_hidden_size}, "
                f"Projector expects {self.projector.llm_hidden_size}"
            )
        
        # Validate projector parameters
        for name, param in self.projector.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"Projector parameter '{name}' contains NaN")
            if torch.isinf(param).any():
                raise RuntimeError(f"Projector parameter '{name}' contains Inf")
        
        print(f"  ✓ Projector loaded")
        print(f"    Parameters: {self.projector.get_num_params():,}")
        print(f"    Architecture: {self.projector.llm_hidden_size}D -> {self.projector.tts_hidden_size}D")
        print(f"    Device: {next(self.projector.parameters()).device}")
        print(f"    Dtype: {next(self.projector.parameters()).dtype}")
    
    def extract_hidden_states(
        self,
        text: str,
        return_attention_mask: bool = False
    ) -> tuple:
        """
        Extract LLM hidden states from text.
        
        Args:
            text: Input text
            return_attention_mask: Whether to return attention mask
        
        Returns:
            Tuple of (hidden_states, attention_mask) if return_attention_mask=True,
            otherwise just hidden_states
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Tokenize
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=False,
            ).to(self.config.device)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {str(e)}")
        
        # Validate inputs
        if 'input_ids' not in inputs:
            raise RuntimeError("Tokenizer did not return 'input_ids'")
        
        num_tokens = inputs['input_ids'].shape[1]
        if num_tokens == 0:
            raise ValueError("Tokenization produced zero tokens")
        
        # Extract hidden states
        try:
            with torch.no_grad():
                outputs = self.llm(**inputs, output_hidden_states=True)
        except Exception as e:
            raise RuntimeError(f"LLM forward pass failed: {str(e)}")
        
        # Validate outputs
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise RuntimeError("LLM did not return hidden_states")
        
        hidden_states = outputs.hidden_states[-1]
        
        # Validate hidden states
        if hidden_states.dim() != 3:
            raise RuntimeError(f"Expected 3D hidden_states, got {hidden_states.dim()}D")
        
        if torch.isnan(hidden_states).any():
            raise RuntimeError("Hidden states contain NaN")
        
        if torch.isinf(hidden_states).any():
            raise RuntimeError("Hidden states contain Inf")
        
        if return_attention_mask:
            attention_mask = inputs.get('attention_mask', None)
            return hidden_states, attention_mask
        
        return hidden_states
    
    def project_to_tts_space(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project LLM hidden states to TTS feature space.
        
        Args:
            hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, seq_len]
        
        Returns:
            Projected TTS features [batch, seq_len, tts_hidden_dim]
        """
        try:
            with torch.no_grad():
                projected = self.projector(hidden_states, attention_mask)
        except Exception as e:
            raise RuntimeError(f"Projection failed: {str(e)}")
        
        # Validate output
        if projected is None:
            raise RuntimeError("Projector returned None")
        
        if torch.isnan(projected).any():
            raise RuntimeError("Projected features contain NaN")
        
        if torch.isinf(projected).any():
            raise RuntimeError("Projected features contain Inf")
        
        return projected
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Generate TTS features from input text.
        
        Args:
            text: Input text
            return_intermediates: If True, return intermediate representations
        
        Returns:
            Dictionary containing:
                - tts_features: Projected TTS features
                - text: Input text
                - num_tokens: Number of tokens
                - (optional) llm_hidden_states: LLM hidden states
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Extract hidden states
        hidden_states, attention_mask = self.extract_hidden_states(
            text,
            return_attention_mask=True
        )
        
        # Project to TTS space
        tts_features = self.project_to_tts_space(hidden_states, attention_mask)
        
        result = {
            'tts_features': tts_features,
            'text': text,
            'num_tokens': hidden_states.shape[1],
            'feature_shape': tuple(tts_features.shape),
        }
        
        if return_intermediates:
            result['llm_hidden_states'] = hidden_states
            result['attention_mask'] = attention_mask
        
        return result
    
    def generate_batch(
        self,
        texts: List[str],
        return_intermediates: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate TTS features for multiple texts.
        
        Args:
            texts: List of input texts
            return_intermediates: If True, return intermediate representations
        
        Returns:
            List of result dictionaries
        """
        if not texts:
            raise ValueError("Input text list is empty")
        
        results = []
        for text in texts:
            try:
                result = self.generate(text, return_intermediates)
                results.append(result)
            except Exception as e:
                print(f"✗ Error processing text '{text[:50]}...': {str(e)}")
                results.append({'error': str(e), 'text': text})
        
        return results
    
    def get_statistics(self, features: torch.Tensor) -> Dict[str, Any]:
        """Get feature statistics."""
        return {
            'mean': features.mean().item(),
            'std': features.std().item(),
            'min': features.min().item(),
            'max': features.max().item(),
            'shape': tuple(features.shape),
        }


def main():
    """Demo inference pipeline."""
    
    print("=" * 70)
    print("Audio Projection Inference Demo")
    print("=" * 70)
    
    # Configuration
    config = InferenceConfig(
        llm_path="./data/audio-projection-models/llama-3.1-8b-instruct",
        projector_path="./data/audio-projection-models/audio_projector_trained.pt",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Initialize pipeline
    pipeline = AudioProjectionInference(config)
    
    # Test texts
    test_texts = [
        "Hello world, this is a test for audio generation.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
    ]
    
    print("\n" + "=" * 70)
    print("Running Inference")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        
        try:
            result = pipeline.generate(text, return_intermediates=False)
            
            features = result['tts_features']
            stats = pipeline.get_statistics(features)
            
            print(f"   ✓ Generated features: {stats['shape']}")
            print(f"     Tokens: {result['num_tokens']}")
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"     Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
