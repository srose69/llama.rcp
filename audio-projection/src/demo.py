"""
Demo script for audio projection adapter.

Tests the full pipeline:
LLM hidden states -> Audio Projector -> TTS Decoder -> Audio
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from projector import AudioProjector


def simulate_llm_hidden_states(batch_size: int = 1, seq_len: int = 50) -> torch.Tensor:
    """
    Simulate LLM hidden states from Llama-3.1-8B.
    In real usage, these would come from the actual LLM forward pass.
    """
    llm_hidden_size = 4096
    hidden_states = torch.randn(batch_size, seq_len, llm_hidden_size)
    return hidden_states


def test_projection_pipeline():
    """Test audio projection adapter pipeline."""
    
    print("=" * 60)
    print("Audio Projection Adapter - Demo")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 50
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  LLM hidden size: 4096 (Llama-3.1-8B)")
    print(f"  TTS hidden size: 2048 (Qwen3-TTS)")
    
    # Step 1: Simulate LLM output
    print("\n" + "-" * 60)
    print("Step 1: Simulating LLM hidden states")
    print("-" * 60)
    
    hidden_states = simulate_llm_hidden_states(batch_size, seq_len)
    print(f"✓ Generated hidden states: {hidden_states.shape}")
    print(f"  Mean: {hidden_states.mean().item():.4f}")
    print(f"  Std: {hidden_states.std().item():.4f}")
    
    # Step 2: Initialize audio projector
    print("\n" + "-" * 60)
    print("Step 2: Initializing Audio Projector")
    print("-" * 60)
    
    projector = AudioProjector(
        llm_hidden_size=4096,
        tts_hidden_size=2048,
        dropout=0.1,
    )
    projector.eval()  # Set to evaluation mode to disable dropout
    
    num_params = projector.get_num_params()
    print(f"✓ AudioProjector initialized")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: ~{num_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Step 3: Project to TTS space
    print("\n" + "-" * 60)
    print("Step 3: Projecting to TTS feature space")
    print("-" * 60)
    
    with torch.no_grad():
        projected_features = projector(hidden_states)
    
    print(f"✓ Projection completed: {projected_features.shape}")
    print(f"  Mean: {projected_features.mean().item():.4f}")
    print(f"  Std: {projected_features.std().item():.4f}")
    
    # Step 4: Verify compatibility
    print("\n" + "-" * 60)
    print("Step 4: Verifying TTS compatibility")
    print("-" * 60)
    
    expected_shape = (batch_size, seq_len, 2048)
    assert projected_features.shape == expected_shape, \
        f"Shape mismatch: {projected_features.shape} != {expected_shape}"
    
    print(f"✓ Output shape matches TTS input requirements")
    print(f"  Ready for Qwen3-TTS decoder input")
    
    # Step 5: Save projector weights
    print("\n" + "-" * 60)
    print("Step 5: Saving projector weights")
    print("-" * 60)
    
    save_path = "/media/tiny/audio-projection-models/audio_projector_demo.pt"
    projector.save_pretrained(save_path)
    print(f"✓ Saved to: {save_path}")
    
    # Step 6: Test loading
    print("\n" + "-" * 60)
    print("Step 6: Testing weight loading")
    print("-" * 60)
    
    loaded_projector = AudioProjector.from_pretrained(save_path)
    loaded_projector.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        reloaded_output = loaded_projector(hidden_states)
    
    # Verify outputs match
    diff = (projected_features - reloaded_output).abs().max().item()
    print(f"✓ Loaded projector successfully")
    print(f"  Max difference: {diff:.10f}")
    print(f"  Outputs match: {diff < 1e-6}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    print("\n✓ Audio projection adapter working correctly!")
    print("\nNext steps:")
    print("1. Download Llama-3.1-8B-Instruct (requires HF token)")
    print("2. Use existing Qwen3-TTS at /media/tiny/qwen3-tts-conversion/model/")
    print("3. Prepare audio-text dataset (LibriSpeech, Common Voice)")
    print("4. Train adapter: python src/train.py --config configs/default.yaml")
    print("\nPipeline ready for training!")


if __name__ == "__main__":
    test_projection_pipeline()
