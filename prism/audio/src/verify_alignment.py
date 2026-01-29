# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Feature Space Alignment Verification Utilities.

Tools to validate that Llama and Qwen feature spaces are compatible
and that the projector is learning the correct mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import json
from dataclasses import dataclass


@dataclass
class AlignmentMetrics:
    """Metrics for feature alignment validation."""
    
    cosine_similarity: float
    euclidean_distance: float
    mse: float
    pearson_correlation: float
    
    def __str__(self) -> str:
        return (
            f"Alignment Metrics:\n"
            f"  Cosine Similarity: {self.cosine_similarity:.4f} (target: >0.85)\n"
            f"  Euclidean Distance: {self.euclidean_distance:.4f}\n"
            f"  MSE: {self.mse:.4f}\n"
            f"  Pearson Correlation: {self.pearson_correlation:.4f}"
        )
    
    def is_good(self) -> bool:
        """Check if alignment is acceptable."""
        return (
            self.cosine_similarity > 0.85 and
            self.pearson_correlation > 0.7
        )


def compute_alignment_metrics(
    projected: torch.Tensor,
    target: torch.Tensor,
) -> AlignmentMetrics:
    """
    Compute comprehensive alignment metrics.
    
    Args:
        projected: Projected features from Llama [batch, seq_len, hidden_dim]
        target: Target features from Qwen [batch, seq_len, hidden_dim]
    
    Returns:
        AlignmentMetrics object
    """
    if projected.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: projected={projected.shape}, target={target.shape}"
        )
    
    # Flatten for global metrics
    proj_flat = projected.reshape(-1, projected.shape[-1])
    tgt_flat = target.reshape(-1, target.shape[-1])
    
    # Cosine similarity
    cosine_sim = F.cosine_similarity(proj_flat, tgt_flat, dim=-1).mean().item()
    
    # Euclidean distance
    euclidean = torch.norm(proj_flat - tgt_flat, dim=-1).mean().item()
    
    # MSE
    mse = F.mse_loss(proj_flat, tgt_flat).item()
    
    # Pearson correlation (per dimension, then average)
    proj_centered = proj_flat - proj_flat.mean(dim=0, keepdim=True)
    tgt_centered = tgt_flat - tgt_flat.mean(dim=0, keepdim=True)
    
    proj_std = proj_flat.std(dim=0, keepdim=True) + 1e-8
    tgt_std = tgt_flat.std(dim=0, keepdim=True) + 1e-8
    
    correlation = (proj_centered * tgt_centered).mean(dim=0) / (proj_std * tgt_std)
    pearson = correlation.mean().item()
    
    return AlignmentMetrics(
        cosine_similarity=cosine_sim,
        euclidean_distance=euclidean,
        mse=mse,
        pearson_correlation=pearson,
    )


def verify_sequence_alignment(
    llm_hidden_states: torch.Tensor,
    tts_features: torch.Tensor,
) -> Dict[str, Any]:
    """
    Verify that sequence lengths are compatible.
    
    Args:
        llm_hidden_states: Llama hidden states
        tts_features: Qwen text encoder features
    
    Returns:
        Dictionary with alignment info and warnings
    """
    results = {
        'compatible': False,
        'llm_shape': tuple(llm_hidden_states.shape),
        'tts_shape': tuple(tts_features.shape),
        'issues': [],
        'warnings': [],
    }
    
    # Check batch size
    if llm_hidden_states.shape[0] != tts_features.shape[0]:
        results['issues'].append(
            f"Batch size mismatch: {llm_hidden_states.shape[0]} vs {tts_features.shape[0]}"
        )
        return results
    
    # Check sequence length
    if llm_hidden_states.shape[1] != tts_features.shape[1]:
        results['issues'].append(
            f"Sequence length mismatch: {llm_hidden_states.shape[1]} vs {tts_features.shape[1]}"
        )
        results['warnings'].append(
            "This indicates you may need a Length Regulator or are targeting acoustic features"
        )
        return results
    
    # Check dimensions
    results['llm_hidden_dim'] = llm_hidden_states.shape[2]
    results['tts_hidden_dim'] = tts_features.shape[2]
    
    if llm_hidden_states.shape[2] != 4096:
        results['warnings'].append(
            f"Expected Llama hidden dim 4096, got {llm_hidden_states.shape[2]}"
        )
    
    if tts_features.shape[2] != 2048:
        results['warnings'].append(
            f"Expected Qwen text encoder dim 2048, got {tts_features.shape[2]}"
        )
    
    results['compatible'] = True
    return results


def compare_tokenization(
    text: str,
    llama_tokenizer,
    qwen_tokenizer,
) -> Dict[str, Any]:
    """
    Compare how Llama and Qwen tokenize the same text.
    
    This helps understand the tokenization mismatch risk.
    """
    llama_tokens = llama_tokenizer(text, return_tensors="pt")
    qwen_tokens = qwen_tokenizer(text, return_tensors="pt")
    
    llama_ids = llama_tokens['input_ids'][0].tolist()
    qwen_ids = qwen_tokens['input_ids'][0].tolist()
    
    return {
        'text': text,
        'llama_tokens': len(llama_ids),
        'qwen_tokens': len(qwen_ids),
        'length_ratio': len(qwen_ids) / len(llama_ids),
        'llama_decoded': [llama_tokenizer.decode([t]) for t in llama_ids],
        'qwen_decoded': [qwen_tokenizer.decode([t]) for t in qwen_ids],
    }


def test_projector_quality(
    projector: nn.Module,
    test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Test projector quality on a set of samples.
    
    Args:
        projector: Trained projector (any nn.Module with forward pass)
        test_samples: List of (llm_hidden, tts_target) pairs
    
    Returns:
        Quality metrics and diagnostics
    """
    projector.eval()
    
    all_metrics = []
    cosine_sims = []
    mse_losses = []
    
    with torch.no_grad():
        for llm_hidden, tts_target in test_samples:
            # Validate shapes
            alignment = verify_sequence_alignment(llm_hidden, tts_target)
            if not alignment['compatible']:
                print(f"⚠ Sample incompatible: {alignment['issues']}")
                continue
            
            # Project
            projected = projector(llm_hidden)
            
            # Compute metrics
            metrics = compute_alignment_metrics(projected, tts_target)
            all_metrics.append(metrics)
            
            cosine_sims.append(metrics.cosine_similarity)
            mse_losses.append(metrics.mse)
    
    if not all_metrics:
        return {'error': 'No compatible samples found'}
    
    # Aggregate
    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    avg_mse = sum(mse_losses) / len(mse_losses)
    min_cosine = min(cosine_sims)
    max_cosine = max(cosine_sims)
    
    quality = {
        'num_samples': len(all_metrics),
        'avg_cosine_similarity': avg_cosine,
        'min_cosine_similarity': min_cosine,
        'max_cosine_similarity': max_cosine,
        'avg_mse': avg_mse,
        'quality_rating': 'GOOD' if avg_cosine > 0.85 else 'POOR',
        'all_metrics': all_metrics,
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("Projector Quality Test Results")
    print("=" * 70)
    print(f"  Samples tested: {quality['num_samples']}")
    print(f"  Avg Cosine Similarity: {avg_cosine:.4f}")
    print(f"  Range: [{min_cosine:.4f}, {max_cosine:.4f}]")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Quality: {quality['quality_rating']}")
    
    if avg_cosine < 0.85:
        print("\n⚠ WARNING: Low cosine similarity indicates poor feature alignment")
        print("  Possible causes:")
        print("    - Insufficient training")
        print("    - Incompatible feature spaces")
        print("    - Wrong target features (acoustic vs text encoder)")
    
    return quality


def visualize_feature_distribution(
    projected: torch.Tensor,
    target: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Visualize distribution of projected vs target features.
    
    Useful for debugging topology mismatches.
    """
    proj_flat = projected.reshape(-1, projected.shape[-1]).cpu()
    tgt_flat = target.reshape(-1, target.shape[-1]).cpu()
    
    stats = {
        'projected': {
            'mean': proj_flat.mean(dim=0).mean().item(),
            'std': proj_flat.std(dim=0).mean().item(),
            'min': proj_flat.min().item(),
            'max': proj_flat.max().item(),
        },
        'target': {
            'mean': tgt_flat.mean(dim=0).mean().item(),
            'std': tgt_flat.std(dim=0).mean().item(),
            'min': tgt_flat.min().item(),
            'max': tgt_flat.max().item(),
        }
    }
    
    print("\n" + "=" * 70)
    print("Feature Distribution Analysis")
    print("=" * 70)
    print("\nProjected Features:")
    print(f"  Mean: {stats['projected']['mean']:.4f}")
    print(f"  Std:  {stats['projected']['std']:.4f}")
    print(f"  Range: [{stats['projected']['min']:.4f}, {stats['projected']['max']:.4f}]")
    
    print("\nTarget Features:")
    print(f"  Mean: {stats['target']['mean']:.4f}")
    print(f"  Std:  {stats['target']['std']:.4f}")
    print(f"  Range: [{stats['target']['min']:.4f}, {stats['target']['max']:.4f}]")
    
    # Check for scale mismatch
    scale_ratio = stats['projected']['std'] / (stats['target']['std'] + 1e-8)
    if scale_ratio < 0.5 or scale_ratio > 2.0:
        print(f"\n⚠ WARNING: Scale mismatch detected (ratio: {scale_ratio:.2f})")
        print("  Consider normalizing features or adjusting loss weights")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Stats saved to {save_path}")
    
    return stats


def main():
    """Demo verification utilities."""
    print("=" * 70)
    print("Feature Alignment Verification Utilities")
    print("=" * 70)
    print("\nThese tools help validate:")
    print("  1. Sequence length compatibility")
    print("  2. Feature space alignment (cosine similarity)")
    print("  3. Tokenization differences")
    print("  4. Projector quality")
    print("\nUsage:")
    print("  from verify_alignment import compute_alignment_metrics")
    print("  metrics = compute_alignment_metrics(projected, target)")
    print("  print(metrics)")


if __name__ == "__main__":
    main()
