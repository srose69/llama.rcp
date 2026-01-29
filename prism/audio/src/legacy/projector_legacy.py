#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Audio Projection Adapter

Implements a trainable projection layer that maps LLM hidden states
to TTS decoder input space.
"""

import torch
import torch.nn as nn
from typing import Optional


class AudioProjector(nn.Module):
    """
    Audio projection adapter for LLM-to-TTS bridge.
    
    Architecture:
        LLM hidden states (4096) -> Linear -> LayerNorm -> GELU
        -> Linear -> LayerNorm -> TTS features (2048)
    
    Args:
        llm_hidden_size: Hidden dimension of base LLM (default: 4096 for Llama-3.1-8B)
        tts_hidden_size: Input dimension for TTS decoder (default: 2048 for Qwen3-TTS)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 4096,
        tts_hidden_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.tts_hidden_size = tts_hidden_size
        
        # First projection layer
        self.linear1 = nn.Linear(llm_hidden_size, tts_hidden_size)
        self.norm1 = nn.LayerNorm(tts_hidden_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second projection layer
        self.linear2 = nn.Linear(tts_hidden_size, tts_hidden_size)
        self.norm2 = nn.LayerNorm(tts_hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layer weights."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project LLM hidden states to TTS feature space.
        
        Args:
            hidden_states: LLM hidden states [batch_size, seq_len, llm_hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            Projected features [batch_size, seq_len, tts_hidden_size]
        """
        # Input validation
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states to be 3D [batch, seq, hidden], "
                f"got {hidden_states.dim()}D with shape {hidden_states.shape}"
            )
        
        if hidden_states.shape[-1] != self.llm_hidden_size:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.llm_hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        
        if torch.isnan(hidden_states).any():
            raise ValueError("Input hidden_states contains NaN values")
        
        if torch.isinf(hidden_states).any():
            raise ValueError("Input hidden_states contains Inf values")
        
        # Validate attention mask if provided
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"Attention mask shape {attention_mask.shape} does not match "
                    f"hidden_states batch and sequence dimensions {hidden_states.shape[:2]}"
                )
        
        # First projection
        x = self.linear1(hidden_states)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Check intermediate numerical stability
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                "Numerical instability detected after first projection layer. "
                f"Stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )
        
        # Second projection
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        # Final numerical stability check
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                "Numerical instability detected in final output. "
                f"Stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match hidden dimension
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask
        
        # Validate output shape
        expected_shape = (hidden_states.shape[0], hidden_states.shape[1], self.tts_hidden_size)
        if x.shape != expected_shape:
            raise RuntimeError(
                f"Output shape mismatch: expected {expected_shape}, got {x.shape}"
            )
        
        return x
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_path: str):
        """
        Save projector weights.
        
        Args:
            save_path: Path to save weights (e.g., 'audio_projector.pt')
        """
        # Validate model state before saving
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"Cannot save model: parameter '{name}' contains NaN values")
            if torch.isinf(param).any():
                raise RuntimeError(f"Cannot save model: parameter '{name}' contains Inf values")
        
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'llm_hidden_size': self.llm_hidden_size,
                'tts_hidden_size': self.tts_hidden_size,
            },
            'num_params': self.get_num_params(),
        }, save_path)
        print(f"✓ Saved audio projector to {save_path}")
        print(f"  Parameters: {self.get_num_params():,}")
        print(f"  Config: {self.llm_hidden_size}D -> {self.tts_hidden_size}D")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'AudioProjector':
        """
        Load projector from saved weights.
        
        Args:
            load_path: Path to saved weights
        
        Returns:
            Loaded AudioProjector instance
        """
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # Validate checkpoint structure
        if 'state_dict' not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing 'state_dict' key")
        if 'config' not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing 'config' key")
        
        config = checkpoint['config']
        
        # Validate config
        required_keys = ['llm_hidden_size', 'tts_hidden_size']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Invalid checkpoint config: missing '{key}' key")
        
        model = cls(
            llm_hidden_size=config['llm_hidden_size'],
            tts_hidden_size=config['tts_hidden_size'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        # Validate loaded parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"Loaded parameter '{name}' contains NaN values")
            if torch.isinf(param).any():
                raise RuntimeError(f"Loaded parameter '{name}' contains Inf values")
        
        print(f"✓ Loaded audio projector from {load_path}")
        print(f"  Parameters: {model.get_num_params():,}")
        print(f"  Config: {config['llm_hidden_size']}D -> {config['tts_hidden_size']}D")
        
        return model


class AudioProjectorWithPooling(AudioProjector):
    """
    Enhanced audio projector with sequence pooling.
    
    Adds temporal pooling to reduce sequence length before TTS,
    which can improve training stability and inference speed.
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 4096,
        tts_hidden_size: int = 2048,
        dropout: float = 0.1,
        pooling_type: str = 'avg',  # 'avg', 'max', or 'attention'
    ):
        super().__init__(llm_hidden_size, tts_hidden_size, dropout)
        
        self.pooling_type = pooling_type
        
        if pooling_type == 'attention':
            # Learnable attention pooling
            self.attention_pool = nn.Linear(tts_hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional pooling.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional mask
            return_pooled: If True, return pooled features instead of projected
        
        Returns:
            Projected features [B, L, H] or pooled features [B, H] if return_pooled=True
        """
        # Standard projection
        projected = super().forward(hidden_states, attention_mask)
        
        if not return_pooled:
            return projected
        
        # Apply pooling
        if self.pooling_type == 'avg':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = projected.mean(dim=1)
        
        elif self.pooling_type == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                projected_masked = projected.masked_fill(~mask.bool(), float('-inf'))
                pooled = projected_masked.max(dim=1)[0]
            else:
                pooled = projected.max(dim=1)[0]
        
        elif self.pooling_type == 'attention':
            # Compute attention weights
            attn_weights = self.attention_pool(projected).squeeze(-1)  # [B, L]
            
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            
            attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)  # [B, L, 1]
            pooled = (projected * attn_weights).sum(dim=1)  # [B, H]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return pooled


if __name__ == '__main__':
    # Test audio projector
    print("Testing AudioProjector...")
    
    batch_size = 2
    seq_len = 128
    llm_hidden_size = 4096
    
    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_len, llm_hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Initialize projector
    projector = AudioProjector()
    print(f"Number of parameters: {projector.get_num_params():,}")
    
    # Forward pass
    output = projector(hidden_states, attention_mask)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test save/load
    projector.save_pretrained('test_projector.pt')
    loaded_projector = AudioProjector.from_pretrained('test_projector.pt')
    
    print("\nTesting AudioProjectorWithPooling...")
    projector_pool = AudioProjectorWithPooling(pooling_type='attention')
    projected = projector_pool(hidden_states, attention_mask, return_pooled=False)
    pooled = projector_pool(hidden_states, attention_mask, return_pooled=True)
    print(f"Projected shape: {projected.shape}")
    print(f"Pooled shape: {pooled.shape}")
    
    print("\n✓ All tests passed!")
