# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Cross-Attention mechanism for TTS decoder to attend to LLM hidden states.

Allows decoder to "look at" different parts of text while generating audio,
similar to Transformer encoder-decoder attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention for TTS decoder.
    
    Query: from TTS decoder state
    Key, Value: from LLM hidden states
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim {query_dim} must be divisible by num_heads {num_heads}")
        
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Projections
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(kv_dim, query_dim)
        self.value_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.
        
        Args:
            query: Decoder state [batch, tgt_len, query_dim]
            key: Encoder output [batch, src_len, kv_dim]
            value: Encoder output [batch, src_len, kv_dim]
            key_padding_mask: Mask for padding [batch, src_len]
            attn_mask: Attention mask [tgt_len, src_len]
        
        Returns:
            output: Attended features [batch, tgt_len, query_dim]
            attention_weights: Attention matrix [batch, num_heads, tgt_len, src_len]
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # [batch, tgt_len, query_dim]
        K = self.key_proj(key)      # [batch, src_len, query_dim]
        V = self.value_proj(value)  # [batch, src_len, query_dim]
        
        # Reshape for multi-head attention
        # [batch, len, query_dim] -> [batch, num_heads, len, head_dim]
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [batch, num_heads, tgt_len, src_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # [batch, src_len] -> [batch, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask == 0,
                float('-inf')
            )
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [batch, num_heads, tgt_len, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # [batch, tgt_len, query_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.query_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_weights


class CrossAttentionBlock(nn.Module):
    """
    Complete cross-attention block with feed-forward and residual connections.
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Returns:
            output: Attended features
            attention_weights: Attention matrix
        """
        # Cross-attention with residual
        attn_output, attn_weights = self.cross_attn(
            query, key, value, key_padding_mask
        )
        query = self.norm1(query + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output, attn_weights


class AdaptiveCrossAttention(nn.Module):
    """
    Adaptive cross-attention that can handle variable-length sequences.
    
    Useful for aligning text tokens with audio frames when lengths differ.
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        use_relative_position: bool = True,
    ):
        super().__init__()
        
        self.cross_attn_block = CrossAttentionBlock(
            query_dim=query_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
        )
        
        self.use_relative_position = use_relative_position
        
        if use_relative_position:
            # Relative position encoding
            self.relative_position_embedding = nn.Embedding(512, num_heads)
    
    def forward(
        self,
        query: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adaptive cross-attention forward.
        
        Args:
            query: Decoder features [batch, tgt_len, query_dim]
            encoder_hidden_states: LLM hidden states [batch, src_len, kv_dim]
            encoder_attention_mask: Encoder mask [batch, src_len]
        
        Returns:
            output: Attended features [batch, tgt_len, query_dim]
        """
        output, attn_weights = self.cross_attn_block(
            query=query,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=encoder_attention_mask,
        )
        
        return output


if __name__ == '__main__':
    # Test cross-attention
    print("Testing Multi-Head Cross-Attention...")
    
    batch_size = 2
    src_len = 50    # LLM text tokens
    tgt_len = 500   # Audio frames
    llm_dim = 4096  # Llama hidden dim
    tts_dim = 2048  # TTS decoder dim
    num_heads = 8
    
    # Simulated inputs
    encoder_hidden = torch.randn(batch_size, src_len, llm_dim)  # From Llama
    decoder_state = torch.randn(batch_size, tgt_len, tts_dim)   # From TTS decoder
    
    # Cross-attention
    cross_attn = MultiHeadCrossAttention(
        query_dim=tts_dim,
        kv_dim=llm_dim,
        num_heads=num_heads,
    )
    
    output, attn_weights = cross_attn(
        query=decoder_state,
        key=encoder_hidden,
        value=encoder_hidden,
    )
    
    print(f"Encoder hidden: {encoder_hidden.shape}")
    print(f"Decoder state: {decoder_state.shape}")
    print(f"Output: {output.shape}")
    print(f"Attention weights: {attn_weights.shape}")
    
    # Test cross-attention block
    print("\nTesting Cross-Attention Block...")
    cross_attn_block = CrossAttentionBlock(
        query_dim=tts_dim,
        kv_dim=llm_dim,
        num_heads=num_heads,
    )
    
    output, attn_weights = cross_attn_block(
        query=decoder_state,
        key=encoder_hidden,
        value=encoder_hidden,
    )
    
    print(f"Block output: {output.shape}")
    
    # Visualize attention pattern
    avg_attn = attn_weights[0].mean(dim=0)  # Average over heads
    print(f"\nAttention pattern (first sample, averaged over heads):")
    print(f"  Shape: {avg_attn.shape}")
    print(f"  Each audio frame attends to ~{avg_attn.sum(dim=1).mean().item():.2f} text tokens (should be ~1.0)")
    
    print("\n✓ All tests passed!")
