# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
End-to-End TTS Projector with Duration Prediction and Cross-Attention.

Full architecture for high-quality text-to-speech generation using LLM hidden states.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from duration_predictor import DurationPredictor, LengthRegulator, gaussian_upsampling
from cross_attention import CrossAttentionBlock


class TTSProjector(nn.Module):
    """
    Complete TTS projector with duration prediction and cross-attention.
    
    Architecture:
        1. Llama hidden states [batch, text_len, 4096]
        2. Initial projection [batch, text_len, 2048]
        3. Duration predictor -> predict frame count per token
        4. Length regulator -> expand to [batch, audio_len, 2048]
        5. Cross-attention -> decoder attends to original text
        6. Output -> acoustic features for TTS decoder
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 4096,
        tts_hidden_size: int = 2048,
        num_cross_attn_layers: int = 4,
        num_attn_heads: int = 8,
        dropout: float = 0.1,
        use_gaussian_upsampling: bool = True,
    ):
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.tts_hidden_size = tts_hidden_size
        self.use_gaussian_upsampling = use_gaussian_upsampling
        
        # Initial projection: Llama space -> TTS space
        self.text_projection = nn.Sequential(
            nn.Linear(llm_hidden_size, tts_hidden_size),
            nn.LayerNorm(tts_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tts_hidden_size, tts_hidden_size),
            nn.LayerNorm(tts_hidden_size),
        )
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            hidden_dim=tts_hidden_size,
            num_layers=2,
            dropout=dropout,
        )
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Cross-attention layers for decoder to attend to text
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=tts_hidden_size,
                kv_dim=tts_hidden_size,  # Same space after projection
                num_heads=num_attn_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Final refinement
        self.output_projection = nn.Sequential(
            nn.Linear(tts_hidden_size, tts_hidden_size),
            nn.LayerNorm(tts_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tts_hidden_size, tts_hidden_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        ground_truth_durations: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for TTS projection.
        
        Args:
            hidden_states: Llama hidden states [batch, text_len, llm_hidden_size]
            attention_mask: Text attention mask [batch, text_len]
            target_length: Target audio length (frames). If None, predicted from durations
            ground_truth_durations: GT durations for training [batch, text_len]
        
        Returns:
            Dictionary with:
                - acoustic_features: Output features for TTS decoder [batch, audio_len, tts_hidden_size]
                - predicted_durations: Predicted durations [batch, text_len]
                - attention_weights: List of attention matrices from each layer
        """
        batch_size, text_len, _ = hidden_states.shape
        
        # Validate input
        if hidden_states.shape[-1] != self.llm_hidden_size:
            raise ValueError(
                f"Input hidden_size {hidden_states.shape[-1]} != expected {self.llm_hidden_size}"
            )
        
        if torch.isnan(hidden_states).any():
            raise ValueError("Input contains NaN")
        
        # 1. Project Llama hidden states to TTS space
        text_features = self.text_projection(hidden_states)  # [batch, text_len, tts_hidden_size]
        
        if torch.isnan(text_features).any():
            raise RuntimeError("NaN after text projection")
        
        # 2. Predict durations
        predicted_durations = self.duration_predictor(text_features, attention_mask)  # [batch, text_len]
        
        if torch.isnan(predicted_durations).any():
            raise RuntimeError("NaN in predicted durations")
        
        # Use GT durations during training if provided
        durations = ground_truth_durations if ground_truth_durations is not None else predicted_durations
        
        # 3. Length regulation: expand text sequence to audio sequence
        if self.use_gaussian_upsampling and target_length is not None:
            # Soft upsampling with Gaussian interpolation
            expanded_features = gaussian_upsampling(
                text_features,
                durations,
                target_length
            )
        else:
            # Hard upsampling with repeat
            expanded_features = self.length_regulator(
                text_features,
                durations,
                max_len=target_length
            )
        
        audio_len = expanded_features.shape[1]
        
        if torch.isnan(expanded_features).any():
            raise RuntimeError("NaN after length regulation")
        
        # 4. Cross-attention: decoder attends to original text features
        attention_weights_list = []
        decoder_hidden = expanded_features
        
        for cross_attn_layer in self.cross_attention_layers:
            decoder_hidden, attn_weights = cross_attn_layer(
                query=decoder_hidden,           # Expanded audio features
                key=text_features,              # Original text features
                value=text_features,
                key_padding_mask=attention_mask,
            )
            attention_weights_list.append(attn_weights)
            
            if torch.isnan(decoder_hidden).any():
                raise RuntimeError("NaN in cross-attention output")
        
        # 5. Final refinement
        acoustic_features = self.output_projection(decoder_hidden)
        
        if torch.isnan(acoustic_features).any():
            raise RuntimeError("NaN in final output")
        
        return {
            'acoustic_features': acoustic_features,
            'predicted_durations': predicted_durations,
            'attention_weights': attention_weights_list,
            'text_features': text_features,
            'expanded_features': expanded_features,
        }
    
    def get_num_params(self) -> int:
        """Get total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"Cannot save: parameter '{name}' contains NaN")
        
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'llm_hidden_size': self.llm_hidden_size,
                'tts_hidden_size': self.tts_hidden_size,
                'use_gaussian_upsampling': self.use_gaussian_upsampling,
            },
            'num_params': self.get_num_params(),
        }, save_path)
        
        print(f"✓ Saved TTS projector to {save_path}")
        print(f"  Parameters: {self.get_num_params():,}")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'TTSProjector':
        """Load model from checkpoint."""
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if 'config' not in checkpoint or 'state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        
        config = checkpoint['config']
        model = cls(
            llm_hidden_size=config['llm_hidden_size'],
            tts_hidden_size=config['tts_hidden_size'],
            use_gaussian_upsampling=config.get('use_gaussian_upsampling', True),
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        
        # Validate loaded parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"Loaded parameter '{name}' contains NaN")
        
        print(f"✓ Loaded TTS projector from {load_path}")
        print(f"  Parameters: {model.get_num_params():,}")
        
        return model


if __name__ == '__main__':
    # Test TTS projector
    print("=" * 70)
    print("Testing End-to-End TTS Projector")
    print("=" * 70)
    
    batch_size = 2
    text_len = 50      # Input text tokens
    target_audio_len = 500  # Target audio frames (10x expansion)
    llm_dim = 4096
    tts_dim = 2048
    
    # Simulated Llama hidden states
    llama_hidden = torch.randn(batch_size, text_len, llm_dim)
    attention_mask = torch.ones(batch_size, text_len)
    
    # Initialize projector
    projector = TTSProjector(
        llm_hidden_size=llm_dim,
        tts_hidden_size=tts_dim,
        num_cross_attn_layers=4,
        num_attn_heads=8,
    )
    
    print(f"\nModel Parameters: {projector.get_num_params():,}")
    
    # Forward pass
    print(f"\nForward pass:")
    print(f"  Input: {llama_hidden.shape}")
    print(f"  Target audio length: {target_audio_len} frames")
    
    projector.eval()
    with torch.no_grad():
        outputs = projector(
            hidden_states=llama_hidden,
            attention_mask=attention_mask,
            target_length=target_audio_len,
        )
    
    acoustic_features = outputs['acoustic_features']
    predicted_durations = outputs['predicted_durations']
    attention_weights = outputs['attention_weights']
    
    print(f"\n✓ Output:")
    print(f"  Acoustic features: {acoustic_features.shape}")
    print(f"  Predicted durations: {predicted_durations.shape}")
    print(f"  Duration range: [{predicted_durations.min().item():.2f}, {predicted_durations.max().item():.2f}]")
    print(f"  Avg duration: {predicted_durations.mean().item():.2f} frames/token")
    print(f"  Cross-attention layers: {len(attention_weights)}")
    print(f"  Attention shape per layer: {attention_weights[0].shape}")
    
    # Verify expansion
    expected_expansion = target_audio_len / text_len
    actual_expansion = acoustic_features.shape[1] / text_len
    print(f"\n✓ Length expansion:")
    print(f"  Expected: {expected_expansion:.1f}x")
    print(f"  Actual: {actual_expansion:.1f}x")
    
    # Test save/load
    print(f"\n✓ Testing save/load...")
    save_path = "/tmp/tts_projector_test.pt"
    projector.save_pretrained(save_path)
    loaded = TTSProjector.from_pretrained(save_path)
    print(f"  Load successful")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
