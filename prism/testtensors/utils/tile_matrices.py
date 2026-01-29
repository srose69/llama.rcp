#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Tile weight matrices from saved files (.npy or .pt)
"""
import torch
import numpy as np
from pathlib import Path
import argparse

def load_matrix(filepath):
    """Load matrix from .npy or .pt file"""
    path = Path(filepath)
    if path.suffix == '.npy':
        return torch.from_numpy(np.load(path))
    elif path.suffix == '.pt':
        return torch.load(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Tile weight matrices from files')
    parser.add_argument('--input-head', help='Input lm_head file (.npy or .pt)')
    parser.add_argument('--input-embed', help='Input embedding file (.npy or .pt)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=512, help='Tile size')
    parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    args = parser.parse_args()
    
    if not args.input_head and not args.input_embed:
        parser.error("At least one of --input-head or --input-embed is required")
    
    print("="*80)
    print("TILING MATRICES FROM FILES")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    (output_dir / 'tiled_head').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tiled_embed').mkdir(parents=True, exist_ok=True)
    
    tile_size = args.tile_size

    
    # Process lm_head
    if args.input_head:
        print(f"\nLoading lm_head from: {args.input_head}")
        lm_head = load_matrix(args.input_head)
        print(f"  Shape: {lm_head.shape}")
        
        num_row_tiles = (lm_head.shape[0] + tile_size - 1) // tile_size
        num_col_tiles = (lm_head.shape[1] + tile_size - 1) // tile_size
        total_tiles = num_row_tiles * num_col_tiles
        
        print(f"\nTiling lm_head...")
        print(f"  Grid: {num_row_tiles} × {num_col_tiles} = {total_tiles} tiles")
        
        for tile_id in range(total_tiles):
            row_idx = tile_id // num_col_tiles
            col_idx = tile_id % num_col_tiles
            
            row_start = row_idx * tile_size
            row_end = min((row_idx + 1) * tile_size, lm_head.shape[0])
            col_start = col_idx * tile_size
            col_end = min((col_idx + 1) * tile_size, lm_head.shape[1])
            
            tile = lm_head[row_start:row_end, col_start:col_end]
            
            ext = '.npy' if args.format == 'npy' else '.pt'
            filename = output_dir / 'tiled_head' / f'tile_{tile_id:03d}{ext}'
            
            if args.format == 'npy':
                np.save(filename, tile.cpu().numpy())
            else:
                torch.save(tile, filename)
            
            if (tile_id + 1) % 100 == 0:
                print(f"  Saved {tile_id + 1}/{total_tiles} tiles...")
        
        print(f"  ✓ Saved {total_tiles} lm_head tiles")
    
    # Process embedding
    if args.input_embed:
        print(f"\nLoading embedding from: {args.input_embed}")
        embedding = load_matrix(args.input_embed)
        print(f"  Shape: {embedding.shape}")
        
        num_row_tiles = (embedding.shape[0] + tile_size - 1) // tile_size
        num_col_tiles = (embedding.shape[1] + tile_size - 1) // tile_size
        total_tiles = num_row_tiles * num_col_tiles
        
        print(f"\nTiling embedding...")
        print(f"  Grid: {num_row_tiles} × {num_col_tiles} = {total_tiles} tiles")
        
        for tile_id in range(total_tiles):
            row_idx = tile_id // num_col_tiles
            col_idx = tile_id % num_col_tiles
            
            row_start = row_idx * tile_size
            row_end = min((row_idx + 1) * tile_size, embedding.shape[0])
            col_start = col_idx * tile_size
            col_end = min((col_idx + 1) * tile_size, embedding.shape[1])
            
            tile = embedding[row_start:row_end, col_start:col_end]
            
            ext = '.npy' if args.format == 'npy' else '.pt'
            filename = output_dir / 'tiled_embed' / f'tile_{tile_id:03d}{ext}'
            
            if args.format == 'npy':
                np.save(filename, tile.cpu().numpy())
            else:
                torch.save(tile, filename)
            
            if (tile_id + 1) % 100 == 0:
                print(f"  Saved {tile_id + 1}/{total_tiles} tiles...")
        
        print(f"  ✓ Saved {total_tiles} embedding tiles")
    
    print("\n" + "="*80)
    print("✓ TILING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
