#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Phase 1: Train Workers via One-Shot Least Squares

Formula: Input + X = Target
Where:
- Input = lm_head.weight (upper model) [vocab, dim_upper]
- Target = embedding.weight (lower model) [vocab, dim_lower]
- X = learned transformation (our workers)

Workers are trained ONCE via least squares, then FROZEN forever.
MicroMoE routers are FROZEN during this phase.

NO TEXT GENERATION - we use weight matrices directly!

Usage:
    python train_workers.py \
        --pipes-dir builded_pipes \
        --upper-model /path/to/llama \
        --lower-model /path/to/liquidai \
        --output-dir trained_pipes
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(upper_model_path, lower_model_path, device='cpu'):
    """
    Load upper and lower models for activation extraction.
    
    Strategy:
    - Upper (Llama 8B): CPU/RAM, inference on CPU (slow but stable)
    - Lower (Qwen 1.7B): cuda:1 (Tesla P4 8GB) persistent, fast inference
    - Training lstsq: cuda:0 (GTX 1080)
    """
    from pathlib import Path as PathLib
    
    upper_path = PathLib(upper_model_path)
    lower_path = PathLib(lower_model_path)
    
    upper_is_local = upper_path.exists()
    lower_is_local = lower_path.exists()
    
    print(f"Loading upper model: {upper_model_path} (CPU/RAM for weight access, local={upper_is_local})")
    
    upper_model = AutoModelForCausalLM.from_pretrained(
        str(upper_path) if upper_is_local else upper_model_path,
        torch_dtype=torch.float32,
        device_map='cpu',  # CPU for weight access
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=upper_is_local
    )
    upper_tokenizer = AutoTokenizer.from_pretrained(
        str(upper_path) if upper_is_local else upper_model_path,
        trust_remote_code=True,
        local_files_only=upper_is_local
    )
    if upper_tokenizer.pad_token is None:
        upper_tokenizer.pad_token = upper_tokenizer.eos_token
    
    print(f"Loading lower model: {lower_model_path} (CPU/RAM, local={lower_is_local})")
    lower_model = AutoModelForCausalLM.from_pretrained(
        str(lower_path) if lower_is_local else lower_model_path,
        torch_dtype=torch.float32,
        device_map='cpu',  # In RAM
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=lower_is_local
    )
    lower_tokenizer = AutoTokenizer.from_pretrained(
        str(lower_path) if lower_is_local else lower_model_path,
        trust_remote_code=True,
        local_files_only=lower_is_local
    )
    if lower_tokenizer.pad_token is None:
        lower_tokenizer.pad_token = lower_tokenizer.eos_token
    
    print(f"  Both models: CPU/RAM (for weight access)")
    print(f"  Strategy: Extract lm_head.weight and embedding.weight directly")
    print(f"  Lstsq training: cuda:0 GTX 1080")
    
    return upper_model, upper_tokenizer, lower_model, lower_tokenizer


def extract_hidden_states_batch(model, texts, tokenizer, model_device='cpu'):
    """
    Extract hidden states from model.
    
    Model is already on the required device (cpu or cuda:1).
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].mean(dim=1)
    
    return hidden_states.cpu()


def generate_training_data(upper_model_path, lower_model_path, num_samples=10000, 
                          lm_head_file='output/untiled_head/llama_lm_head.npy',
                          embedding_file='output/untiled_embed/qwen_embedding.npy'):
    """
    Generate training data from pre-extracted weight files.
    
    CORRECT LOGIC:
    
    Llama says: "My vector for 'apple' is like this (from lm_head)" -> X
    Qwen says: "And my vector for 'apple' is like this (from embedding)" -> Z
    PRISM learns to transform X into Z
    
    DO NOT LOAD FULL MODELS - only tokenizers and pre-extracted weights from .npy
    
    Returns:
        X_upper: [N, dim_upper] - lm_head weights for common tokens
        Z_lower: [N, dim_lower] - embedding weights for common tokens
    """
    import numpy as np
    from utils.tokenistic import create_token_mapping
    from pathlib import Path
    
    print(f"\n🔄 Loading training data from pre-extracted weights...")
    print(f"  lm_head: {lm_head_file}")
    print(f"  embedding: {embedding_file}")
    print(f"  Strategy: NOT loading full models")
    
    # Create token mapping (load only tokenizers)
    print(f"\n🗺️  Creating token mapping...")
    upper_indices, lower_indices = create_token_mapping(
        upper_model_path,
        lower_model_path
    )
    
    N = min(len(upper_indices), num_samples)
    upper_indices = upper_indices[:N]
    lower_indices = lower_indices[:N]
    
    print(f"\n📊 Training data info:")
    print(f"  Common tokens: {len(upper_indices):,}")
    print(f"  Using: {N:,} token pairs")
    
    # STEP 1: Load lm_head weights from .npy
    print(f"\n⬆️  Loading upper model lm_head from {lm_head_file}...")
    lm_head_weights = np.load(lm_head_file)  # [vocab_size, hidden_dim]
    print(f"  Loaded shape: {lm_head_weights.shape}")
    
    # DO NOT transpose - in Qwen weight tying (lm_head = embedding)
    # Take only rows for our tokens
    X_upper = torch.from_numpy(lm_head_weights[upper_indices])  # [N, dim_upper]
    print(f"  X shape: {X_upper.shape}")
    print(f"  X dtype: {X_upper.dtype}")
    
    # STEP 2: Load embedding weights from .npy
    print(f"\n⬇️  Loading lower model embedding from {embedding_file}...")
    embedding_weights = np.load(embedding_file)  # [vocab_size, hidden_dim]
    print(f"  Loaded shape: {embedding_weights.shape}")
    
    # Take only rows for our tokens
    Z_lower = torch.from_numpy(embedding_weights[lower_indices])  # [N, dim_lower]
    print(f"  Z shape: {Z_lower.shape}")
    print(f"  Z dtype: {Z_lower.dtype}")
    
    print(f"\n✅ Training data ready:")
    print(f"  X_upper: {X_upper.shape} - lm_head.weight for common tokens")
    print(f"  Z_lower: {Z_lower.shape} - embedding.weight for common tokens")
    print(f"  Total samples: {N:,} token pairs")
    print(f"\n💡 Mapping: Llama lm_head vectors → Qwen embedding vectors")
    print(f"   Garbage and special tokens excluded")
    print(f"   Full models NOT loaded")
    
    return X_upper, Z_lower


def load_weight_matrices_from_tiles(tiled_dir, target_shape=None):
    """
    Load weight matrix from pre-tiled .npy files.
    
    Args:
        tiled_dir: Directory containing tile_000.npy, tile_001.npy, etc.
        target_shape: Expected matrix shape (vocab, hidden_dim) from architecture.json
    
    Returns:
        matrix: Assembled weight matrix [vocab, hidden_dim]
    """
    import numpy as np
    from pathlib import Path
    
    tiled_path = Path(tiled_dir)
    if not tiled_path.exists():
        raise FileNotFoundError(f"Tiled directory not found: {tiled_dir}")
    
    # Find all tile files
    tile_files = sorted(tiled_path.glob('tile_*.npy'))
    if not tile_files:
        raise FileNotFoundError(f"No tile files found in {tiled_dir}")
    
    print(f"  Loading {len(tile_files)} tiles from {tiled_dir}")
    
    # Load first tile to get tile size
    first_tile = np.load(tile_files[0])
    tile_size = first_tile.shape[0]
    num_tiles = len(tile_files)
    
    if target_shape is not None:
        # Use target_shape from architecture.json
        vocab_size, hidden_dim = target_shape
        num_row_tiles = (vocab_size + tile_size - 1) // tile_size
        num_col_tiles = (hidden_dim + tile_size - 1) // tile_size
        print(f"  Target shape: [{vocab_size}, {hidden_dim}]")
        print(f"  Grid: {num_row_tiles}×{num_col_tiles} tiles")
    else:
        # Fallback: try to determine grid dimensions
        possible_grids = []
        for rows in range(1, num_tiles + 1):
            if num_tiles % rows == 0:
                cols = num_tiles // rows
                possible_grids.append((rows, cols))
        
        target_aspect_ratio = 32
        best_grid = min(possible_grids, key=lambda g: abs((g[0] / g[1]) - target_aspect_ratio))
        num_row_tiles, num_col_tiles = best_grid
        vocab_size = num_row_tiles * tile_size
        hidden_dim = num_col_tiles * tile_size
        print(f"  Grid: {num_row_tiles}×{num_col_tiles}, Matrix: [{vocab_size}, {hidden_dim}]")
    
    # Assemble matrix from tiles
    matrix = np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    
    for tile_id, tile_file in enumerate(tile_files):
        tile = np.load(tile_file)
        
        row_idx = tile_id // num_col_tiles
        col_idx = tile_id % num_col_tiles
        
        row_start = row_idx * tile_size
        row_end = min(row_start + tile_size, vocab_size)
        col_start = col_idx * tile_size
        col_end = min(col_start + tile_size, hidden_dim)
        
        actual_rows = row_end - row_start
        actual_cols = col_end - col_start
        
        # Handle tiles that might be smaller than tile_size
        tile_rows, tile_cols = tile.shape
        copy_rows = min(actual_rows, tile_rows)
        copy_cols = min(actual_cols, tile_cols)
        
        matrix[row_start:row_start+copy_rows, col_start:col_start+copy_cols] = tile[:copy_rows, :copy_cols]
    
    print(f"  ✓ Assembled matrix: {matrix.shape}")
    return torch.from_numpy(matrix)


def extract_training_data(upper_lm_head, lower_embedding):
    """
    Extract training data from weight matrices.
    
    Formula: Input + X = Target
    - Input (X_upper): rows from upper model lm_head [vocab, dim_upper]
    - Target (Z_lower): rows from lower model embedding [vocab, dim_lower]
    
    Each row corresponds to ONE TOKEN representation.
    We train workers to map: lm_head_row → embedding_row
    
    Args:
        upper_lm_head: [vocab_upper, dim_upper]
        lower_embedding: [vocab_lower, dim_lower]
    
    Returns:
        X_upper: [num_samples, dim_upper] - input data
        Z_lower: [num_samples, dim_lower] - target data
    """
    print("\nExtracting training data from weight matrices...")
    print("  Formula: Input + X = Target")
    print(f"  Input (lm_head): {upper_lm_head.shape}")
    print(f"  Target (embedding): {lower_embedding.shape}")
    
    # Use overlapping vocabulary tokens
    # Take minimum vocab size
    vocab_upper, dim_upper = upper_lm_head.shape
    vocab_lower, dim_lower = lower_embedding.shape
    
    # Use smaller vocab size for training
    num_samples = min(vocab_upper, vocab_lower)
    
    # Extract rows (token representations)
    X_upper = upper_lm_head[:num_samples, :]  # [num_samples, dim_upper]
    Z_lower = lower_embedding[:num_samples, :]  # [num_samples, dim_lower]
    
    print(f"\n✓ Extracted training data:")
    print(f"  X_upper (Input): {X_upper.shape} - token reps from lm_head")
    print(f"  Z_lower (Target): {Z_lower.shape} - token reps from embedding")
    print(f"  Using {num_samples:,} tokens from shared vocabulary")
    
    return X_upper, Z_lower


def train_worker_least_squares(X_chunk, Z_chunk, device='cpu'):
    """
    Train single worker via one-shot least squares.
    
    Solve: X_chunk @ W.T = Z_chunk
    
    Args:
        X_chunk: [num_samples, input_dim]
        Z_chunk: [num_samples, output_dim]
        device: 'cpu' or 'cuda'
    
    Returns:
        W: [output_dim, input_dim] - trained worker tile
    """
    # DEBUG
    print(f"\n    🔬 train_worker_least_squares:")
    print(f"       X_chunk: {X_chunk.shape}, device={X_chunk.device}")
    print(f"       Z_chunk: {Z_chunk.shape}, device={Z_chunk.device}")
    
    # Move to device for computation
    X_chunk = X_chunk.to(device)
    Z_chunk = Z_chunk.to(device)
    
    print(f"       After .to(device): X={X_chunk.shape}, Z={Z_chunk.shape}")
    
    # Least squares: W = (X^T X)^{-1} X^T Z
    # Equivalent: W = lstsq(X, Z).T
    solution = torch.linalg.lstsq(X_chunk, Z_chunk).solution
    print(f"       solution shape: {solution.shape}")
    
    W = solution.T  # Transpose to get [output_dim, input_dim]
    print(f"       W shape after transpose: {W.shape}")
    
    # Return to CPU for storage
    result = W.cpu()
    print(f"       Returning: {result.shape}")
    return result


def train_single_worker_wrapper(args):
    """Wrapper for parallel worker training."""
    worker_info, X_upper, Z_lower, device, pipe_input_start, pipe_output_start = args
    
    worker_id = worker_info['worker_id']
    input_slice = worker_info['input_slice']
    output_slice = worker_info['output_slice']
    
    # Convert absolute slices to pipe-relative slices
    input_slice_rel = [input_slice[0] - pipe_input_start, input_slice[1] - pipe_input_start]
    output_slice_rel = [output_slice[0] - pipe_output_start, output_slice[1] - pipe_output_start]
    
    # Extract chunks for this worker using RELATIVE slices
    X_chunk = X_upper[:, input_slice_rel[0]:input_slice_rel[1]]
    Z_chunk = Z_lower[:, output_slice_rel[0]:output_slice_rel[1]]
    
    # DEBUG first 3 workers
    if worker_id < 3:
        print(f"\n  🔍 Worker {worker_id} DEBUG:")
        print(f"     Absolute slices: input={input_slice}, output={output_slice}")
        print(f"     Relative slices: input={input_slice_rel}, output={output_slice_rel}")
        print(f"     X_upper shape: {X_upper.shape}, Z_lower shape: {Z_lower.shape}")
        print(f"     X_chunk shape: {X_chunk.shape}, Z_chunk shape: {Z_chunk.shape}")
    
    # Train via least squares on GPU
    W_worker = train_worker_least_squares(X_chunk, Z_chunk, device)
    
    if worker_id < 3:
        print(f"     W_worker shape: {W_worker.shape}")
    
    return {
        'worker_id': worker_id,
        'tile': W_worker,
        'input_slice': input_slice,
        'output_slice': output_slice
    }


def train_all_workers(X_upper, Z_lower, pipe_metadata, pipe_id, parallel=True, num_workers=None, device='cpu', pipe_input_start=0, pipe_output_start=0):
    """
    Train all 512 workers for a pipe via one-shot least squares.
    
    ONESHOT: X_upper and Z_lower are loaded ONCE and reused for ALL workers.
    
    Args:
        pipe_input_start, pipe_output_start: Offsets to convert absolute worker slices to pipe-relative slices
    Each worker trains on ITS slice (input_slice, output_slice).
    
    Args:
        X_upper: [num_samples, dim_upper] - SHARED by all workers
        Z_lower: [num_samples, dim_lower] - SHARED by all workers
        pipe_metadata: Worker metadata from calculated_pipes
        pipe_id: 1, 2, or 3
        parallel: If True, use parallel training (default)
        num_workers: Number of parallel workers (default: CPU count)
        device: 'cpu' or 'cuda' for training
    
    Returns:
        trained_workers: List of 512 trained worker tiles
    """
    print(f"\nTraining Pipe {pipe_id} workers...")
    print(f"  Data: X_upper={X_upper.shape}, Z_lower={Z_lower.shape}")
    print(f"  Training method: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    print(f"  Device: {device}")
    
    if parallel:
        # Parallel training via multiprocessing
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        print(f"  Using {num_workers} CPU cores for parallel training")
        
        # Prepare arguments for each worker
        worker_args = [(worker, X_upper, Z_lower, device, pipe_input_start, pipe_output_start) for worker in pipe_metadata]
        
        trained_workers = []
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(train_single_worker_wrapper, args) for args in worker_args]
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Pipe {pipe_id}"):
                trained_workers.append(future.result())
        
        # Sort by worker_id to maintain order
        trained_workers.sort(key=lambda x: x['worker_id'])
    else:
        # Sequential training (for debugging)
        trained_workers = []
        
        for worker in tqdm(pipe_metadata, desc=f"Pipe {pipe_id} workers"):
            result = train_single_worker_wrapper((worker, X_upper, Z_lower, device, pipe_input_start, pipe_output_start))
            trained_workers.append(result)
    
    print(f"✓ Trained {len(trained_workers)} workers for Pipe {pipe_id}")
    
    return trained_workers


def save_trained_pipe(pipe_state, trained_workers, output_file):
    """
    Update pipe state dict with trained worker tiles.
    """
    # Load existing pipe state
    new_state = pipe_state.copy()
    
    print(f"\n🔍 DEBUG save_trained_pipe:")
    print(f"  Total trained workers: {len(trained_workers)}")
    print(f"  Pipe state keys: {len(new_state)}")
    
    # Replace worker tiles with trained ones
    updated_count = 0
    for worker_data in trained_workers:
        worker_id = worker_data['worker_id']
        tile = worker_data['tile']
        
        print(f"  Worker {worker_id}: tile shape={tile.shape}, dtype={tile.dtype}")
        
        # Find worker tile key in state dict
        worker_key = f'workers.{worker_id}.tile'
        if worker_key in new_state:
            new_state[worker_key] = tile
            updated_count += 1
        else:
            print(f"    ⚠️  Key {worker_key} NOT FOUND in pipe state!")
    
    print(f"  ✓ Updated {updated_count} workers")
    
    # Save updated state
    torch.save(new_state, output_file)
    print(f"✓ Saved trained pipe: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Train Workers via One-Shot Least Squares'
    )
    parser.add_argument(
        '--pipes-dir',
        type=str,
        required=True,
        help='Directory with builded_pipes'
    )
    parser.add_argument(
        '--upper-model',
        type=str,
        required=True,
        help='Path to upper model (e.g., Llama)'
    )
    parser.add_argument(
        '--lower-model',
        type=str,
        required=True,
        help='Path to lower model (e.g., LiquidAI)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of training samples to generate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for activation extraction'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_pipes',
        help='Output directory for trained pipes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for model inference'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel training for workers (recommended for CPU)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 1: TRAIN WORKERS (ONE-SHOT LEAST SQUARES)")
    print("=" * 70)
    print()
    
    # Load architecture metadata from calculated_pipes
    calculated_pipes_path = Path('calculated_pipes')
    if not calculated_pipes_path.exists():
        raise FileNotFoundError("calculated_pipes/ not found. Run calcpipes first!")
    
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
        if not metadata_file.exists():
            raise FileNotFoundError(f"Worker metadata not found: {metadata_file}")
        with open(metadata_file, 'r') as f:
            pipe_workers[pipe_name] = json.load(f)
    
    # Generate training data (NOT loading models, using pre-extracted .npy)
    X_upper, Z_lower = generate_training_data(
        upper_model_path=args.upper_model,
        lower_model_path=args.lower_model,
        num_samples=args.num_samples,
        lm_head_file='output/untiled_head/qwen_lm_head.npy',
        embedding_file='output/untiled_embed/qwen_embedding.npy'
    )
    
    # Train workers for each pipe
    pipes_path = Path(args.pipes_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TEMPORARY: Train only pipe3
    for pipe_name in ['pipe3']:  # Changed from ['pipe1', 'pipe2', 'pipe3']
        pipe_id = int(pipe_name[-1])
        
        print(f"\n📊 Training Pipe {pipe_id}:")
        print(f"  Using FULL data: X_upper={X_upper.shape}, Z_lower={Z_lower.shape}")
        print(f"  Workers use absolute slices within [0:{X_upper.shape[1]}]")
        
        # Load pipe state
        pipe_state = torch.load(pipes_path / f'{pipe_name}.pt', map_location='cpu')
        
        # Train workers using FULL data
        # Worker slices are absolute indices into X_upper/Z_lower
        trained_workers = train_all_workers(
            X_upper, Z_lower,  # Use FULL data for all pipes
            pipe_workers[pipe_name], 
            pipe_id,
            parallel=args.parallel,
            num_workers=args.num_workers,
            device=args.device,
            pipe_input_start=0,  # No offset - workers use absolute slices
            pipe_output_start=0
        )
        
        # Save trained pipe
        save_trained_pipe(
            pipe_state, 
            trained_workers, 
            output_path / f'{pipe_name}_trained.pt'
        )
    
    # Save training metadata
    training_info = {
        'training_method': 'one-shot-least-squares',
        'data_source': 'hidden_states_from_text',
        'num_samples': X_upper.shape[0],
        'upper_model': args.upper_model,
        'lower_model': args.lower_model,
        'transformation': f"{architecture['transformation']['input_dim']} → {architecture['transformation']['output_dim']}",
        'workers_frozen': True,
        'routers_frozen': True,
        'formula': 'Input + X = Target (lm_head.weight + X = embedding.weight)',
        'next_step': 'Run train_mmoe.py to train MicroMoE routers'
    }
    
    with open(output_path / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print()
    print("=" * 70)
    print("✅ WORKER TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("📦 Trained pipes saved to:", output_path)
    print()
    print("Workers are now FROZEN.")
    print("Next step: Train MicroMoE routers")
    print(f"  python prismoe.py train-mmoe --trained-pipes {output_path}")
    print()


if __name__ == '__main__':
    main()
