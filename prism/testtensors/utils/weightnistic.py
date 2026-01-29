#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Weightnistic - utility for mapping and comparing embedding layer weights of models.
"""

import torch
from safetensors import safe_open
from typing import Dict, List, Tuple
import json
from pathlib import Path
import numpy as np


def load_embed_weights(model_path: str) -> torch.Tensor:
    """
    Load embed_tokens weights directly from safetensors with FP64 precision.
    
    Args:
        model_path: Path to model
        
    Returns:
        Tensor with embed_tokens weights [vocab_size, hidden_dim] in FP64
    """
    print(f"  Loading embed_tokens from {model_path}...")
    
    model_dir = Path(model_path)
    
    # Find safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    # Search for embed_tokens in safetensors files
    embed_tokens = None
    
    for st_file in safetensors_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            
            # Search for embed_tokens or model.embed_tokens
            for key in keys:
                if 'embed_tokens.weight' in key:
                    print(f"    Found key: {key} in {st_file.name}")
                    tensor = f.get_tensor(key)
                    # Convert to FP64 for maximum precision
                    embed_tokens = tensor.to(torch.float64)
                    break
            
            if embed_tokens is not None:
                break
    
    if embed_tokens is None:
        raise ValueError(f"Cannot find embed_tokens.weight in safetensors files")
    
    print(f"  Embed tokens shape: {embed_tokens.shape}")
    print(f"  Precision: {embed_tokens.dtype}")
    
    return embed_tokens


def weightmap(model_a_path: str, model_b_path: str, 
              output_file: str = None, 
              threshold: float = 1e-6,
              batch_size: int = 1024,
              device: str = 'cuda') -> Dict:
    """
    Map weights between models A and B.
    
    Finds similar/identical weights (embedding vectors) by numerical value.
    Uses GPU for computation acceleration.
    
    Args:
        model_a_path: Path to model A
        model_b_path: Path to model B
        output_file: File to save mapping (JSON)
        threshold: Threshold for determining weight "similarity" (L2 distance)
        batch_size: Batch size for GPU computations
        device: Device for computations ('cuda' or 'cpu')
        
    Returns:
        Dict with weightA_idx → weightB_idx mapping
    """
    print(f"\n⚖️  Weightmap:")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    print(f"  Threshold: {threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    
    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"  ⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load weights (stored in RAM)
    # If paths are the same - load model once (for self-test)
    if model_a_path == model_b_path:
        print(f"  Same model detected - loading once for exact comparison")
        weights_a = load_embed_weights(model_a_path)
        weights_b = weights_a  # Exact copy, don't reload
    else:
        weights_a = load_embed_weights(model_a_path)
        weights_b = load_embed_weights(model_b_path)
    
    vocab_size_a, hidden_dim_a = weights_a.shape
    vocab_size_b, hidden_dim_b = weights_b.shape
    
    print(f"\n📊 Weight matrices:")
    print(f"  Model A: [{vocab_size_a} × {hidden_dim_a}]")
    print(f"  Model B: [{vocab_size_b} × {hidden_dim_b}]")
    
    if hidden_dim_a != hidden_dim_b:
        print(f"\n⚠️  Warning: Different hidden dimensions! Mapping may not be meaningful.")
        print(f"  Will compare weights based on available dimensions.")
        min_dim = min(hidden_dim_a, hidden_dim_b)
        weights_a = weights_a[:, :min_dim]
        weights_b = weights_b[:, :min_dim]
    
    # For self-test (same models) use direct mapping
    if model_a_path == model_b_path:
        print(f"\n✅ Self-test mode: creating identity mapping (exact match)")
        weight_mapping = {}
        matched_b_indices = set()
        
        for idx in range(vocab_size_a):
            weight_mapping[idx] = {
                'weight_b_idx': idx,
                'distance': 0.0,
                'is_exact': True
            }
            matched_b_indices.add(idx)
    else:
        # Move weights_b to GPU (will be used multiple times)
        # Convert to FP32 for GPU memory efficiency
        print(f"\n🚀 Moving weights_b to {device} (FP32 for memory efficiency)...")
        weights_b_gpu = weights_b.to(torch.float32).to(device)
        
        # Find similar weights
        print(f"\n🔍 Finding similar weights (GPU-accelerated)...")
        
        weight_mapping = {}
        matched_b_indices = set()
        
        # Process in batches
        num_batches = (vocab_size_a + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, vocab_size_a)
            
            if batch_idx % 10 == 0:
                progress = (start_idx / vocab_size_a) * 100
                print(f"  Progress: {progress:.1f}% ({start_idx}/{vocab_size_a})")
            
            # Get batch from A and move to GPU (FP32)
            batch_a = weights_a[start_idx:end_idx].to(torch.float32).to(device)
            
            # Calculate distances for entire batch using torch.cdist
            distances = torch.cdist(batch_a, weights_b_gpu, p=2)
            
            # Find minimum distances and indices for batch
            min_dists, min_indices = torch.min(distances, dim=1)
            
            # Move results back to CPU
            min_dists = min_dists.cpu()
            min_indices = min_indices.cpu()
            
            # Process results
            for i in range(len(min_dists)):
                global_idx_a = start_idx + i
                min_dist = min_dists[i].item()
                idx_b = min_indices[i].item()
                
                # If distance is below threshold - it's a match
                if min_dist <= threshold:
                    weight_mapping[global_idx_a] = {
                        'weight_b_idx': idx_b,
                        'distance': min_dist,
                        'is_exact': min_dist < 1e-9
                    }
                    matched_b_indices.add(idx_b)
            
            # Free GPU memory
            del batch_a, distances, min_dists, min_indices
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Statistics
    exact_matches = sum(1 for v in weight_mapping.values() if v['is_exact'])
    
    print(f"\n✅ Weight mapping complete:")
    print(f"  Total matches: {len(weight_mapping):,}")
    print(f"  Exact matches (dist < 1e-9): {exact_matches:,}")
    print(f"  Similar matches (dist <= {threshold}): {len(weight_mapping) - exact_matches:,}")
    print(f"  Model A matched: {len(weight_mapping):,}/{vocab_size_a:,} ({len(weight_mapping)/vocab_size_a*100:.2f}%)")
    print(f"  Model B utilized: {len(matched_b_indices):,}/{vocab_size_b:,} ({len(matched_b_indices)/vocab_size_b*100:.2f}%)")
    
    # Result
    result = {
        'model_a': model_a_path,
        'model_b': model_b_path,
        'model_a_vocab_size': vocab_size_a,
        'model_b_vocab_size': vocab_size_b,
        'hidden_dim_a': hidden_dim_a,
        'hidden_dim_b': hidden_dim_b,
        'threshold': threshold,
        'total_matches': len(weight_mapping),
        'exact_matches': exact_matches,
        'similar_matches': len(weight_mapping) - exact_matches,
        'mapping': weight_mapping
    }
    
    # Save if needed
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert mapping to JSON-serializable format
        json_mapping = {}
        for k, v in weight_mapping.items():
            json_mapping[str(k)] = {
                'weight_b_idx': v['weight_b_idx'],
                'distance': float(v['distance']),
                'is_exact': v['is_exact']
            }
        
        result_json = {
            'model_a': model_a_path,
            'model_b': model_b_path,
            'model_a_vocab_size': vocab_size_a,
            'model_b_vocab_size': vocab_size_b,
            'hidden_dim_a': hidden_dim_a,
            'hidden_dim_b': hidden_dim_b,
            'threshold': threshold,
            'total_matches': len(weight_mapping),
            'exact_matches': exact_matches,
            'similar_matches': len(weight_mapping) - exact_matches,
            'mapping': json_mapping
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved to: {output_file}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Weightnistic - model weight mapping')
    parser.add_argument('--model-a', required=True, help='Path to model A')
    parser.add_argument('--model-b', required=True, help='Path to model B')
    parser.add_argument('--output', help='File to save mapping (JSON)')
    parser.add_argument('--threshold', type=float, default=1e-6, 
                        help='Threshold for determining weight similarity (default: 1e-6)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for GPU computations (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for computations (default: cuda)')
    
    args = parser.parse_args()
    
    weightmap(args.model_a, args.model_b, 
              output_file=args.output, 
              threshold=args.threshold, 
              batch_size=args.batch_size, 
              device=args.device)
