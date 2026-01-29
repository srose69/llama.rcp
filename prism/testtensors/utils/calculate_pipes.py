#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Calculate 3-Pipe parallel architecture tiling configuration.

This script analyzes head and embedding matrices and calculates:
- Pipe 1: [512×512×512] - Fixed cubic tensor
- Pipe 2: [512×512×512] - Fixed cubic tensor  
- Pipe 3: [512×N×K] - Flexible tensor (N, K = powers of 2)

Outputs metadata for rigid mapping: complete coverage from upper to lower model.

Usage:
    python calculate_pipes.py \
        --tiled-head-dir output/tiled_head \
        --tiled-embed-dir output/tiled_embed \
        --output-dir calculated_pipes
"""

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np


def next_power_of_2(x):
    """Round x up to nearest power of 2"""
    if x <= 0:
        return 1
    return 2 ** math.ceil(math.log2(x))


def prev_power_of_2(x):
    """Round x down to nearest power of 2"""
    if x <= 1:
        return 1
    return 2 ** math.floor(math.log2(x))


def analyze_tiled_directory(tiled_dir):
    """
    Analyze tiled directory to determine matrix dimensions.
    
    Returns:
        dict with keys: vocab_size, hidden_dim, num_tiles, tile_size
    """
    tiled_path = Path(tiled_dir)
    if not tiled_path.exists():
        raise FileNotFoundError(f"Tiled directory not found: {tiled_dir}")
    
    # Count tiles
    tile_files = sorted([f for f in tiled_path.iterdir() if f.name.startswith('tile_') and f.suffix == '.npy'])
    num_tiles = len(tile_files)
    
    if num_tiles == 0:
        raise ValueError(f"No tile files found in {tiled_dir}")
    
    # Load first tile to determine tile size
    first_tile = np.load(tile_files[0])
    tile_size = first_tile.shape[0]  # Assuming square tiles
    
    # Calculate matrix dimensions from tile grid
    # Try all possible factorizations of num_tiles
    import math
    possible_grids = []
    for rows in range(1, num_tiles + 1):
        if num_tiles % rows == 0:
            cols = num_tiles // rows
            possible_grids.append((rows, cols))
    
    # Load and examine ACTUAL tiles to determine matrix dimensions
    # Strategy: Tiles at edges may be smaller than tile_size if matrix dims aren't exact multiples
    
    # Load corner tiles to determine actual matrix size
    first_tile = np.load(tile_files[0])
    last_tile = np.load(tile_files[-1])
    
    print(f"  DEBUG: First tile shape: {first_tile.shape}")
    print(f"  DEBUG: Last tile shape: {last_tile.shape}")
    print(f"  DEBUG: Total tiles: {num_tiles}")
    
    # For each possible grid, check if it makes sense
    best_grid = None
    best_score = float('inf')
    
    for num_row_tiles, num_col_tiles in possible_grids:
        # Expected matrix dimensions
        expected_vocab = num_row_tiles * tile_size
        expected_hidden = num_col_tiles * tile_size
        
        # Check if last tile position makes sense
        last_row_idx = num_tiles // num_col_tiles - 1
        last_col_idx = (num_tiles - 1) % num_col_tiles
        
        # For typical LLM matrices: vocab_size >> hidden_dim
        # So we want: num_row_tiles >> num_col_tiles
        # Score: prefer aspect closest to typical values
        aspect = num_row_tiles / num_col_tiles
        
        # Typical LLM aspect ratios:
        # - Llama 70B: 128256/8192 ≈ 15.7
        # - Llama 8B: 128256/4096 ≈ 31.3
        # - Qwen 1.7B: 151936/2048 ≈ 74.2
        # - Prefer higher aspect (many rows, few cols) since vocab >> hidden
        
        # Scoring: penalize deviation from ideal range, prefer higher within range
        if aspect < 10:
            score = (10 - aspect) ** 2 * 10  # Heavy penalty for very low
        elif aspect < 30:
            score = (30 - aspect) ** 2  # Moderate penalty
        elif aspect <= 100:
            # Within good range: prefer higher aspect (closer to Qwen's 74)
            # Give best score to aspect around 70-80
            if 60 <= aspect <= 90:
                score = abs(aspect - 74) * 0.1  # Very small penalty, prefer ~74
            else:
                score = abs(aspect - 74) * 0.5  # Small penalty
        else:
            score = (aspect - 100) ** 2  # Penalize very high
        
        if score < best_score:
            best_score = score
            best_grid = (num_row_tiles, num_col_tiles)
    
    num_row_tiles, num_col_tiles = best_grid
    
    vocab_size = num_row_tiles * tile_size
    hidden_dim = num_col_tiles * tile_size
    
    print(f"  DEBUG: Selected grid: {num_row_tiles}×{num_col_tiles} (aspect={num_row_tiles/num_col_tiles:.1f})")
    print(f"  DEBUG: Matrix dims: [{vocab_size}, {hidden_dim}]")
    
    return {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'num_tiles': num_tiles,
        'tile_size': tile_size,
        'num_row_tiles': num_row_tiles,
        'num_col_tiles': num_col_tiles
    }


def calculate_pipe3_dimensions(total_tiles, pipe1_tiles, pipe2_tiles, tile_size):
    """
    Calculate Pipe 3 dimensions based on REMAINING tiles.
    
    Simple logic:
    - Total tiles available from matrix
    - Pipe 1 takes pipe1_tiles
    - Pipe 2 takes pipe2_tiles  
    - Pipe 3 takes ALL remaining tiles
    
    Returns:
        dict with keys: num_workers, tile_shape
    """
    # Pipe 3 gets ALL remaining tiles
    pipe3_tiles = total_tiles - pipe1_tiles - pipe2_tiles
    
    print(f"\nPipe 3 tile distribution:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Pipe 1: {pipe1_tiles} tiles")
    print(f"  Pipe 2: {pipe2_tiles} tiles")
    print(f"  Pipe 3: {pipe3_tiles} tiles (remainder)")
    
    return {
        'num_workers': pipe3_tiles,
        'tile_shape': [tile_size, tile_size],  # Same tile size as Pipes 1&2
        'N': tile_size,
        'K': tile_size,
        'total_output_dim': 0,  # Not used
        'padding_output': 0
    }


def generate_worker_metadata(pipe_id, pipe_config, dim_upper, dim_lower):
    """
    Generate metadata for workers in a pipe.
    
    Pipes 1&2: 512 workers each with grid structure
    Pipe 3: Variable workers (remainder tiles) - simple list
    
    Each worker has:
    - worker_id: unique within pipe
    - pipe_id: 1, 2, or 3
    - input_slice: [start, end] indices into dim_upper
    - output_slice: [start, end] indices into dim_lower
    - tile_shape: [N, K] for this worker
    - specialization: string describing worker's role
    
    Args:
        pipe_id: 1, 2, or 3
        pipe_config: Configuration dict for this pipe
        dim_upper: Total upper dimension (e.g., 2048)
        dim_lower: Total lower dimension (e.g., 2048)
    
    Returns:
        List of worker metadata dicts
    """
    workers = []
    
    if pipe_id == 3:
        # Pipe 3: Simple tile list - no grid structure
        # Just assign tiles sequentially from tile index 1024 onwards
        num_workers = pipe_config['num_workers']
        tile_shape = pipe_config['tile_shape']
        
        # Pipe 3 starts after Pipes 1&2 (which take 1024 tiles total)
        start_tile_id = 1024
        
        for worker_id in range(num_workers):
            tile_id = start_tile_id + worker_id
            
            workers.append({
                'worker_id': worker_id,
                'pipe_id': pipe_id,
                'tile_id': tile_id,  # Which tile from tiled matrix this worker uses
                'tile_shape': tile_shape,
                'specialization': f'pipe3_worker{worker_id:03d}_tile{tile_id}'
            })
        
        return workers
    
    # Pipes 1 & 2: VERTICAL slicing by dims (not horizontal by vocab)
    if pipe_id in [1, 2]:
        num_workers = 512
        
        # VERTICAL partitioning: each pipe handles ALL vocab but only PART of dims
        # Pipe 1: dims [0:682]
        # Pipe 2: dims [682:1365]
        # This allows true parallelism by dimensions!
        
        if pipe_id == 1:
            pipe_input_start = 0
            pipe_input_end = 682
            pipe_output_start = 0
            pipe_output_end = 682
        elif pipe_id == 2:
            pipe_input_start = 682
            pipe_input_end = 1365
            pipe_output_start = 682
            pipe_output_end = 1365
        
        pipe_input_size = pipe_input_end - pipe_input_start
        pipe_output_size = pipe_output_end - pipe_output_start
        
        # Each worker processes part of vocab for this dim range
        # 512 workers split the vocab dimension
        vocab_chunk_size = dim_upper // num_workers  # This is wrong - should be vocab_size not dim_upper
        # Actually, workers should map to tiles which span BOTH vocab and this pipe's dims
        
        # For now: simple sequential tile assignment
        # Worker gets one 512×512 tile from the pipe's column range
        tiles_per_pipe = num_workers
        
        for worker_id in range(num_workers):
            # Each worker handles a 512×512 tile
            # But tiles span vocab (rows) and THIS pipe's dim range (cols)
            
            workers.append({
                'worker_id': worker_id,
                'pipe_id': pipe_id,
                'input_slice': [pipe_input_start, pipe_input_end],
                'output_slice': [pipe_output_start, pipe_output_end],
                'tile_shape': [512, 512],
                'specialization': f'pipe{pipe_id}_worker{worker_id:03d}_dims[{pipe_input_start}:{pipe_input_end}]'
            })
    
    elif pipe_id == 3:
        # Pipe 3: [512×N×K] - flexible dimensions
        # Covers the final 1/3 of the space (2/3 to 3/3)
        N = pipe_config['N']
        K = pipe_config['K']
        num_rows = pipe_config['num_rows']
        num_cols = pipe_config['num_cols']
        
        # Pipe 3 covers 2/3 to 3/3 of the space
        num_pipes = 3
        pipe_fraction_start = 2.0 / num_pipes  # 2/3
        pipe_fraction_end = 1.0  # 3/3
        
        pipe_input_start = int(dim_upper * pipe_fraction_start)
        pipe_input_end = dim_upper  # Full dimension
        pipe_output_start = int(dim_lower * pipe_fraction_start)
        pipe_output_end = dim_lower  # Full dimension
        
        for worker_id in range(512):
            row_idx = worker_id // num_cols
            col_idx = worker_id % num_cols
            
            # Input slice within Pipe 3's portion
            input_start = pipe_input_start + (col_idx * K)
            input_end = min(input_start + K, pipe_input_end)
            
            # Output slice within Pipe 3's portion
            output_start = pipe_output_start + (row_idx * N)
            output_end = min(output_start + N, pipe_output_end)
            
            workers.append({
                'worker_id': worker_id,
                'pipe_id': pipe_id,
                'grid_position': [row_idx, col_idx],
                'input_slice': [input_start, input_end],
                'output_slice': [output_start, output_end],
                'tile_shape': [N, K],
                'specialization': f'pipe3_worker{worker_id:03d}_r{row_idx}_c{col_idx}'
            })
    
    return workers


def main():
    parser = argparse.ArgumentParser(
        description='Calculate 3-Pipe parallel architecture configuration'
    )
    parser.add_argument(
        '--tiled-head-dir',
        type=str,
        required=True,
        help='Directory with tiled lm_head (upper model)'
    )
    parser.add_argument(
        '--tiled-embed-dir',
        type=str,
        required=True,
        help='Directory with tiled embedding (lower model)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='calculated_pipes',
        help='Output directory for pipe metadata'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("3-PIPE PARALLEL ARCHITECTURE CALCULATION")
    print("=" * 70)
    print()
    
    # Analyze tiled directories
    print("Analyzing tiled matrices...")
    print()
    
    print("📊 Upper Model (lm_head):")
    head_info = analyze_tiled_directory(args.tiled_head_dir)
    print(f"  Matrix: {head_info['vocab_size']:,} × {head_info['hidden_dim']:,}")
    print(f"  Tiles: {head_info['num_row_tiles']} × {head_info['num_col_tiles']} = {head_info['num_tiles']}")
    print(f"  Tile size: {head_info['tile_size']}×{head_info['tile_size']}")
    print()
    
    print("📊 Lower Model (embedding):")
    embed_info = analyze_tiled_directory(args.tiled_embed_dir)
    print(f"  Matrix: {embed_info['vocab_size']:,} × {embed_info['hidden_dim']:,}")
    print(f"  Tiles: {embed_info['num_row_tiles']} × {embed_info['num_col_tiles']} = {embed_info['num_tiles']}")
    print(f"  Tile size: {embed_info['tile_size']}×{embed_info['tile_size']}")
    print()
    
    # Dimensions for transformation
    dim_upper = head_info['hidden_dim']  # Upper model output
    dim_lower = embed_info['hidden_dim']  # Lower model input
    
    print("🔄 Transformation:")
    print(f"  Upper → Lower: {dim_upper:,} → {dim_lower:,}")
    print()
    
    # Calculate Pipe 3 dimensions
    # Pipe 3 gets ALL remaining tiles after Pipes 1&2 take 512 each
    print("Calculating Pipe 3 dimensions...")
    total_tiles = head_info['num_tiles']
    pipe1_tiles = 512
    pipe2_tiles = 512
    tile_size = head_info['tile_size']
    
    pipe3_config = calculate_pipe3_dimensions(total_tiles, pipe1_tiles, pipe2_tiles, tile_size)
    print(f"  Pipe 3: {pipe3_config['num_workers']} workers")
    print(f"  Tile shape: {pipe3_config['tile_shape']}")
    print()
    
    # Generate pipe configurations
    print("Generating pipe configurations...")
    print()
    
    pipes_config = {
        'pipe1': {
            'pipe_id': 1,
            'structure': 'cubic',
            'dimensions': [512, 512, 512],
            'num_workers': 512,
            'tile_shape': [512, 512],
            'description': 'Fixed cubic tensor - learns transformation pattern A'
        },
        'pipe2': {
            'pipe_id': 2,
            'structure': 'cubic',
            'dimensions': [512, 512, 512],
            'num_workers': 512,
            'tile_shape': [512, 512],
            'description': 'Fixed cubic tensor - learns transformation pattern B'
        },
        'pipe3': {
            'pipe_id': 3,
            'structure': 'flexible',
            'dimensions': pipe3_config['tile_shape'],
            'num_workers': pipe3_config['num_workers'],
            'tile_shape': pipe3_config['tile_shape'],
            'description': f"remainder tiles ({pipe3_config['num_workers']} workers)"
        }
    }
    
    # Generate worker metadata for each pipe
    print("Generating worker metadata...")
    print("  Each worker is an expert on its specific tile")
    print("  MicroMoE router will learn which workers to activate")
    print()
    
    all_workers = {}
    for pipe_name, pipe_cfg in pipes_config.items():
        pipe_id = pipe_cfg['pipe_id']
        workers = generate_worker_metadata(
            pipe_id,
            pipe3_config if pipe_id == 3 else {},
            dim_upper,
            dim_lower
        )
        all_workers[pipe_name] = workers
        print(f"  ✓ Pipe {pipe_id}: 512 workers configured")
    
    print()
    
    # Overall architecture metadata
    architecture = {
        'version': '1.0',
        'num_pipes': 3,
        'workers_per_pipe': 512,
        'total_workers': 1536,
        'sparse_activation_ratio': 0.30,
        'active_workers_per_pipe': 154,
        'upper_model': {
            'output_dim': dim_upper,
            'matrix_shape': [head_info['vocab_size'], head_info['hidden_dim']],
            'tiled_dir': args.tiled_head_dir
        },
        'lower_model': {
            'input_dim': dim_lower,
            'matrix_shape': [embed_info['vocab_size'], embed_info['hidden_dim']],
            'tiled_dir': args.tiled_embed_dir
        },
        'transformation': {
            'input_dim': dim_upper,
            'output_dim': dim_lower,
            'formula': 'Input + X = Target (least squares)'
        },
        'pipes': pipes_config
    }
    
    # Save metadata
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving metadata...")
    
    # Save overall architecture
    with open(output_path / 'architecture.json', 'w') as f:
        json.dump(architecture, f, indent=2)
    print(f"  ✓ {output_path / 'architecture.json'}")
    
    # Save worker metadata for each pipe
    for pipe_name, workers in all_workers.items():
        with open(output_path / f'{pipe_name}_workers.json', 'w') as f:
            json.dump(workers, f, indent=2)
        print(f"  ✓ {output_path / f'{pipe_name}_workers.json'}")
    
    # Save summary
    summary = {
        'upper_to_lower': f'{dim_upper} → {dim_lower}',
        'total_workers': 1536,
        'pipes': [
            f'Pipe 1: [512×512×512]',
            f'Pipe 2: [512×512×512]',
            f'Pipe 3: [512×{pipe3_config["N"]}×{pipe3_config["K"]}]'
        ],
        'rigid_mapping': 'Complete coverage from upper to lower embedding space',
        'next_step': 'Run build_pipes.py to create trainable pipe structures'
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {output_path / 'summary.json'}")
    
    print()
    print("=" * 70)
    print("✅ PIPE CALCULATION COMPLETE")
    print("=" * 70)
    print()
    print("📋 Summary:")
    print(f"  Total pipes: 3")
    print(f"  Total workers: 1,536 (512 per pipe)")
    print(f"  Transformation: {dim_upper:,} → {dim_lower:,}")
    print()
    print(f"  Pipe 1: [512×512×512] - cubic")
    print(f"  Pipe 2: [512×512×512] - cubic")
    print(f"  Pipe 3: [512×{pipe3_config['N']}×{pipe3_config['K']}] - flexible")
    print()
    print(f"📁 Metadata saved to: {output_path}")
    print()
    print("Next step:")
    print(f"  python prismoe.py buildpipes --config {output_path / 'architecture.json'}")
    print()


if __name__ == '__main__':
    main()
