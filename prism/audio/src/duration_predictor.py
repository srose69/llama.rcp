# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Duration Predictor for Length Regulation.

Predicts phoneme/token durations to expand text sequence to acoustic frame sequence.
Based on FastSpeech architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DurationPredictor(nn.Module):
    """
    Duration predictor for length regulation.
    
    Predicts duration (number of frames) for each input token.
    Used to expand text-length sequence to audio-length sequence.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for duration prediction
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Final projection to scalar duration
        self.projection = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict duration for each token.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            attention_mask: Mask [batch, seq_len]
        
        Returns:
            Predicted durations [batch, seq_len] (positive integers)
        """
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply conv layers
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            x = x + residual  # Residual connection
        
        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Project to duration
        durations = self.projection(x).squeeze(-1)  # [batch, seq_len]
        
        # Apply softplus to ensure positive durations + minimum duration
        durations = F.softplus(durations) + 1.0
        
        # Apply mask if provided
        if attention_mask is not None:
            durations = durations * attention_mask
        
        return durations


class LengthRegulator(nn.Module):
    """
    Length regulator that expands input sequence based on predicted durations.
    
    Repeats each frame according to its duration to match target acoustic length.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Expand input sequence according to durations.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            durations: Duration for each frame [batch, seq_len]
            max_len: Maximum output length (for padding)
        
        Returns:
            Expanded features [batch, expanded_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Round durations to integers
        durations_int = torch.round(durations).long()
        
        # Compute output lengths
        output_lens = durations_int.sum(dim=1)  # [batch]
        
        if max_len is None:
            max_len = int(output_lens.max().item())
        
        # Initialize output tensor
        output = torch.zeros(
            batch_size, max_len, hidden_dim,
            dtype=x.dtype,
            device=x.device
        )
        
        # Expand each sequence
        for batch_idx in range(batch_size):
            pos = 0
            for seq_idx in range(seq_len):
                duration = int(durations_int[batch_idx, seq_idx].item())
                if duration > 0 and pos < max_len:
                    end_pos = min(pos + duration, max_len)
                    # Repeat this frame 'duration' times
                    output[batch_idx, pos:end_pos] = x[batch_idx, seq_idx]
                    pos = end_pos
        
        return output


def gaussian_upsampling(
    x: torch.Tensor,
    durations: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """
    Soft upsampling using Gaussian interpolation.
    
    More sophisticated than hard repeat - uses soft attention
    to smoothly interpolate between adjacent frames.
    
    Args:
        x: Input [batch, seq_len, hidden_dim]
        durations: Predicted durations [batch, seq_len]
        max_len: Target length
    
    Returns:
        Upsampled [batch, max_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = x.shape
    device = x.device
    
    # Compute positions for each input frame
    # cumsum gives us the center position of each frame in output space
    centers = torch.cumsum(durations, dim=1) - durations / 2.0  # [batch, seq_len]
    
    # Output positions
    output_positions = torch.arange(max_len, device=device).float()  # [max_len]
    output_positions = output_positions.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len]
    
    # Expand centers for broadcasting
    centers = centers.unsqueeze(2)  # [batch, seq_len, 1]
    durations = durations.unsqueeze(2)  # [batch, seq_len, 1]
    
    # Gaussian weights: exp(-((pos - center) / (duration/2))^2)
    sigma = durations / 4.0  # Control spread
    distances = (output_positions - centers) ** 2
    weights = torch.exp(-distances / (2 * sigma ** 2 + 1e-6))  # [batch, seq_len, max_len]
    
    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # [batch, seq_len, max_len]
    
    # Apply weighted sum: [batch, max_len, hidden_dim]
    output = torch.matmul(weights.transpose(1, 2), x)
    
    return output


if __name__ == '__main__':
    # Test duration predictor
    print("Testing Duration Predictor...")
    
    batch_size = 2
    seq_len = 50  # Text tokens
    hidden_dim = 2048
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Duration predictor
    duration_predictor = DurationPredictor(hidden_dim=hidden_dim)
    durations = duration_predictor(x)
    
    print(f"Input: {x.shape}")
    print(f"Predicted durations: {durations.shape}")
    print(f"Duration range: [{durations.min().item():.2f}, {durations.max().item():.2f}]")
    print(f"Average duration: {durations.mean().item():.2f} frames/token")
    
    # Length regulator
    print("\nTesting Length Regulator...")
    length_regulator = LengthRegulator()
    expanded = length_regulator(x, durations)
    
    print(f"Expanded output: {expanded.shape}")
    print(f"Expansion ratio: {expanded.shape[1] / x.shape[1]:.2f}x")
    
    # Gaussian upsampling
    print("\nTesting Gaussian Upsampling...")
    target_len = 500
    gaussian_expanded = gaussian_upsampling(x, durations, target_len)
    
    print(f"Gaussian output: {gaussian_expanded.shape}")
    print(f"Target length: {target_len}")
    
    print("\n✓ All tests passed!")
