#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Validate trained workers by checking their reconstruction quality.

CORRECT VALIDATION:
We CUT OUT lm_head from Llama and embedding from Qwen, replace them with PRISM.
Validation should check transformation lm_head.weight → embedding.weight
with token mapping (same 109K pairs used during training).

For each worker:
1. Load trained weights W [output_dim, input_dim]
2. Extract input chunk X_chunk from lm_head.weight (with token mapping)
3. Extract target chunk Z_chunk from embedding.weight (with token mapping)
4. Compute output: Z_pred = X_chunk @ W.T
5. Compare Z_pred with Z_chunk using MSE, MAE, correlation

Usage:
    python validate_workers.py --trained-pipes trained_pipes --upper-model /path/to/llama --lower-model /path/to/qwen
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_training_data(lm_head_file, embedding_file, upper_model_path, lower_model_path):
    """
    Extract training data using token mapping.
    
    Load untiled weights and apply token mapping (as during training).
    """
    import numpy as np
    from utils.tokenistic import create_token_mapping
    
    print("🗺️  Creating token mapping...")
    upper_indices, lower_indices = create_token_mapping(
        upper_model_path,
        lower_model_path
    )
    
    print(f"  Token pairs: {len(upper_indices):,}")
    
    # Load weights
    print(f"📂 Loading lm_head from {lm_head_file}...")
    lm_head_weights = np.load(lm_head_file)
    print(f"  Loaded: {lm_head_weights.shape}")
    
    print(f"📂 Loading embedding from {embedding_file}...")
    embedding_weights = np.load(embedding_file)
    print(f"  Loaded: {embedding_weights.shape}")
    
    # DO NOT transpose - in Qwen weight tying
    # Extract mapped tokens
    X_upper = torch.from_numpy(lm_head_weights[upper_indices])  # [N, hidden_dim]
    Z_lower = torch.from_numpy(embedding_weights[lower_indices])  # [N, hidden_dim]
    
    print(f"  X: {X_upper.shape}")
    print(f"  Z: {Z_lower.shape}")
    
    return X_upper, Z_lower


def validate_worker(worker_weights, X_chunk, Z_chunk, worker_id, pipe_id):
    """
    Validate single worker.
    
    Args:
        worker_weights: [output_dim, input_dim] - trained worker tile
        X_chunk: [num_samples, input_dim] - input from upper matrix
        Z_chunk: [num_samples, output_dim] - target from lower matrix
        worker_id: Worker ID
        pipe_id: Pipe ID
    
    Returns:
        dict with validation metrics
    """
    # Forward pass: Z_pred = X_chunk @ W.T
    Z_pred = X_chunk @ worker_weights.T
    
    # Compute metrics
    mse = torch.mean((Z_pred - Z_chunk) ** 2).item()
    mae = torch.mean(torch.abs(Z_pred - Z_chunk)).item()
    
    # Correlation (per output dimension)
    correlations = []
    for i in range(Z_chunk.shape[1]):
        if Z_chunk[:, i].std() > 0 and Z_pred[:, i].std() > 0:
            corr = torch.corrcoef(torch.stack([Z_chunk[:, i], Z_pred[:, i]]))[0, 1].item()
            correlations.append(corr)
    
    avg_corr = np.mean(correlations) if correlations else 0.0
    
    # Relative error
    rel_error = mse / (torch.mean(Z_chunk ** 2).item() + 1e-8)
    
    return {
        'worker_id': worker_id,
        'pipe_id': pipe_id,
        'mse': mse,
        'mae': mae,
        'avg_correlation': avg_corr,
        'relative_error': rel_error,
        'input_shape': list(X_chunk.shape),
        'output_shape': list(Z_chunk.shape),
        'weight_shape': list(worker_weights.shape)
    }


def validate_pipe(pipe_path, pipe_metadata, X_upper, Z_lower, pipe_id, specific_worker=None):
    """
    Validate all workers in a pipe.
    
    Args:
        pipe_path: Path to trained pipe .pt file
        pipe_metadata: Worker metadata from calculated_pipes
        X_upper: Full upper matrix [vocab, dim_upper]
        Z_lower: Full lower matrix [vocab, dim_lower]
        pipe_id: Pipe ID
        specific_worker: If not None, only validate this worker ID
    
    Returns:
        List of validation results
    """
    # Load trained pipe
    pipe_state = torch.load(pipe_path, map_location='cpu', weights_only=False)
    
    results = []
    workers_to_validate = pipe_metadata if specific_worker is None else [pipe_metadata[specific_worker]]
    
    for worker_info in tqdm(workers_to_validate, desc=f"Validating Pipe {pipe_id}"):
        worker_id = worker_info['worker_id']
        input_slice = worker_info['input_slice']
        output_slice = worker_info['output_slice']
        
        # Load worker weights
        worker_key = f'workers.{worker_id}.tile'
        if worker_key not in pipe_state:
            print(f"Warning: Worker {worker_id} not found in pipe state")
            continue
        
        worker_weights = pipe_state[worker_key]
        
        # Extract chunks
        X_chunk = X_upper[:, input_slice[0]:input_slice[1]]
        Z_chunk = Z_lower[:, output_slice[0]:output_slice[1]]
        
        # Validate
        result = validate_worker(worker_weights, X_chunk, Z_chunk, worker_id, pipe_id)
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained workers (lm_head → embedding transformation with token mapping)'
    )
    parser.add_argument(
        '--trained-pipes',
        type=str,
        required=True,
        help='Directory with trained pipes'
    )
    parser.add_argument(
        '--upper-model',
        type=str,
        required=True,
        help='Path to upper model (for tokenizer)'
    )
    parser.add_argument(
        '--lower-model',
        type=str,
        required=True,
        help='Path to lower model (for tokenizer)'
    )
    parser.add_argument(
        '--lm-head-file',
        type=str,
        default='output/untiled_head/qwen_lm_head.npy',
        help='Path to lm_head .npy file'
    )
    parser.add_argument(
        '--embedding-file',
        type=str,
        default='output/untiled_embed/qwen_embedding.npy',
        help='Path to embedding .npy file'
    )
    parser.add_argument(
        '--pipe',
        type=int,
        choices=[1, 2, 3],
        help='Validate specific pipe only'
    )
    parser.add_argument(
        '--worker',
        type=int,
        help='Validate specific worker only (requires --pipe)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results.json',
        help='Output file for validation results'
    )
    
    args = parser.parse_args()
    
    if args.worker is not None and args.pipe is None:
        parser.error("--worker requires --pipe")
    
    print("=" * 70)
    print("WORKER VALIDATION (lm_head → embedding)")
    print("=" * 70)
    print()
    
    # Load architecture metadata
    calculated_pipes_path = Path('calculated_pipes')
    if not calculated_pipes_path.exists():
        raise FileNotFoundError("calculated_pipes/ not found")
    
    with open(calculated_pipes_path / 'architecture.json', 'r') as f:
        architecture = json.load(f)
    
    print("📋 Architecture:")
    print(f"  Transformation: {architecture['transformation']['input_dim']} → {architecture['transformation']['output_dim']}")
    print(f"  Total workers: {architecture['total_workers']}")
    print()
    
    # Load worker metadata
    pipe_workers = {}
    for pipe_name in ['pipe1', 'pipe2', 'pipe3']:
        metadata_file = calculated_pipes_path / f'{pipe_name}_workers.json'
        with open(metadata_file, 'r') as f:
            pipe_workers[pipe_name] = json.load(f)
    
    # Extract training data (with token mapping)
    print("🔄 Loading validation data...")
    X_upper, Z_lower = extract_training_data(
        args.lm_head_file,
        args.embedding_file,
        args.upper_model,
        args.lower_model
    )
    print(f"📊 Validation data: {X_upper.shape[0]:,} samples")
    print()
    
    # Validate pipes
    trained_pipes_path = Path(args.trained_pipes)
    all_results = []
    
    pipes_to_validate = [args.pipe] if args.pipe else [1, 2, 3]
    
    for pipe_id in pipes_to_validate:
        pipe_name = f'pipe{pipe_id}'
        pipe_path = trained_pipes_path / f'{pipe_name}_trained.pt'
        
        if not pipe_path.exists():
            print(f"Warning: {pipe_path} not found, skipping")
            continue
        
        results = validate_pipe(
            pipe_path,
            pipe_workers[pipe_name],
            X_upper,
            Z_lower,
            pipe_id,
            specific_worker=args.worker
        )
        
        all_results.extend(results)
    
    # Summary statistics
    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print()
    
    if not all_results:
        print("No results to display")
        return
    
    # Overall statistics
    mses = [r['mse'] for r in all_results]
    maes = [r['mae'] for r in all_results]
    corrs = [r['avg_correlation'] for r in all_results]
    rel_errors = [r['relative_error'] for r in all_results]
    
    print(f"📊 Overall Statistics ({len(all_results)} workers):")
    print(f"  MSE:        mean={np.mean(mses):.6f}, std={np.std(mses):.6f}, max={np.max(mses):.6f}")
    print(f"  MAE:        mean={np.mean(maes):.6f}, std={np.std(maes):.6f}, max={np.max(maes):.6f}")
    print(f"  Correlation: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}, min={np.min(corrs):.4f}")
    print(f"  Rel. Error: mean={np.mean(rel_errors):.6f}, std={np.std(rel_errors):.6f}")
    print()
    
    # Per-pipe statistics
    for pipe_id in [1, 2, 3]:
        pipe_results = [r for r in all_results if r['pipe_id'] == pipe_id]
        if not pipe_results:
            continue
        
        pipe_mses = [r['mse'] for r in pipe_results]
        pipe_corrs = [r['avg_correlation'] for r in pipe_results]
        
        print(f"Pipe {pipe_id} ({len(pipe_results)} workers):")
        print(f"  MSE:        {np.mean(pipe_mses):.6f} ± {np.std(pipe_mses):.6f}")
        print(f"  Correlation: {np.mean(pipe_corrs):.4f} ± {np.std(pipe_corrs):.4f}")
    
    print()
    
    # Worst performers
    worst_5 = sorted(all_results, key=lambda r: r['mse'], reverse=True)[:5]
    print("⚠️  Top 5 worst workers (by MSE):")
    for r in worst_5:
        print(f"  Pipe {r['pipe_id']} Worker {r['worker_id']:3d}: MSE={r['mse']:.6f}, Corr={r['avg_correlation']:.4f}")
    
    print()
    
    # Best performers
    best_5 = sorted(all_results, key=lambda r: r['mse'])[:5]
    print("✅ Top 5 best workers (by MSE):")
    for r in best_5:
        print(f"  Pipe {r['pipe_id']} Worker {r['worker_id']:3d}: MSE={r['mse']:.6f}, Corr={r['avg_correlation']:.4f}")
    
    print()
    
    # Save results
    output_path = Path(args.output)
    summary = {
        'num_workers_validated': len(all_results),
        'overall_statistics': {
            'mse': {'mean': float(np.mean(mses)), 'std': float(np.std(mses)), 'max': float(np.max(mses))},
            'mae': {'mean': float(np.mean(maes)), 'std': float(np.std(maes)), 'max': float(np.max(maes))},
            'correlation': {'mean': float(np.mean(corrs)), 'std': float(np.std(corrs)), 'min': float(np.min(corrs))},
            'relative_error': {'mean': float(np.mean(rel_errors)), 'std': float(np.std(rel_errors))}
        },
        'per_worker_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    print()
    
    # Quality assessment
    avg_mse = np.mean(mses)
    avg_corr = np.mean(corrs)
    
    print("=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    if avg_corr > 0.9:
        print("✅ EXCELLENT: High correlation (>0.9), workers are learning correct transformations")
    elif avg_corr > 0.7:
        print("✓  GOOD: Decent correlation (>0.7), workers are reasonably accurate")
    elif avg_corr > 0.5:
        print("⚠️  MODERATE: Low correlation (>0.5), workers may need more training data")
    else:
        print("❌ POOR: Very low correlation (<0.5), check training procedure")
    
    print()


if __name__ == '__main__':
    main()
