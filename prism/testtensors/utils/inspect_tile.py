#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Inspect tile file (npy or pt format)
"""
import numpy as np
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Inspect tile file')
    parser.add_argument('tile_file', help='Path to tile file (.npy or .pt)')
    parser.add_argument('--tile-id', type=int, help='Tile ID for mapping calculation')
    parser.add_argument('--grid-cols', type=int, help='Number of column tiles in grid')
    parser.add_argument('--tile-size', type=int, default=512, help='Tile size')
    args = parser.parse_args()
    
    tile_path = Path(args.tile_file)
    
    # Load tile
    if tile_path.suffix == '.npy':
        tile = np.load(tile_path)
    elif tile_path.suffix == '.pt':
        tile = torch.load(tile_path).cpu().numpy()
    else:
        raise ValueError(f"Unsupported format: {tile_path.suffix}")
    
    print("="*80)
    print(f"TILE INSPECTION: {tile_path.name}")
    print("="*80)
    
    print(f"\nShape: {tile.shape}")
    print(f"Dtype: {tile.dtype}")
    print(f"Size: {tile.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\nStatistics:")
    print(f"  Min: {tile.min():.6f}")
    print(f"  Max: {tile.max():.6f}")
    print(f"  Mean: {tile.mean():.6f}")
    print(f"  Std: {tile.std():.6f}")
    print(f"  Median: {np.median(tile):.6f}")
    
    print(f"\nFirst 5x5 values:")
    print(tile[:5, :5])
    
    print(f"\nLast 5x5 values:")
    print(tile[-5:, -5:])
    
    print(f"\nValue distribution (histogram):")
    hist, bins = np.histogram(tile.flatten(), bins=10)
    for i, (count, bin_start, bin_end) in enumerate(zip(hist, bins[:-1], bins[1:])):
        bar = '█' * int(count / hist.max() * 50)
        print(f"  [{bin_start:7.3f}, {bin_end:7.3f}): {count:6d} {bar}")
    
    print(f"\nNon-zero elements: {np.count_nonzero(tile)} / {tile.size} ({np.count_nonzero(tile)/tile.size*100:.2f}%)")
    
    if args.tile_id is not None and args.grid_cols is not None:
        row_idx = args.tile_id // args.grid_cols
        col_idx = args.tile_id % args.grid_cols
        row_start = row_idx * args.tile_size
        row_end = (row_idx + 1) * args.tile_size
        col_start = col_idx * args.tile_size
        col_end = (col_idx + 1) * args.tile_size
        
        print("\n" + "="*80)
        print("TILE MAPPING:")
        print(f"  Tile ID: {args.tile_id}")
        print(f"  Grid position: row {row_idx}, col {col_idx}")
        print(f"  Matrix region: rows [{row_start}:{row_end}] × cols [{col_start}:{col_end}]")
        print("="*80)

if __name__ == '__main__':
    main()
