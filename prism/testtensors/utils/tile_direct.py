#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Tile lm_head and embedding directly from model WITHOUT intermediate files
Memory efficient: processes one tile at a time
"""
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
import argparse

def main():
    parser = argparse.ArgumentParser(description='Tile model layers directly (memory efficient)')
    parser.add_argument('--model', required=True, help='Model path or HuggingFace model ID')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=512, help='Tile size')
    parser.add_argument('--layer-lm-head', default='lm_head', help='LM head layer name')
    parser.add_argument('--layer-embed', default='model.embed_tokens', help='Embedding layer path')
    parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    parser.add_argument('--skip-head', action='store_true', help='Skip lm_head tiling')
    parser.add_argument('--skip-embed', action='store_true', help='Skip embedding tiling')
    args = parser.parse_args()
    
    print("="*80)
    print("DIRECT TILING: MODEL → TILES (NO INTERMEDIATE FILES)")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    (output_dir / 'tiled_head').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tiled_embed').mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    # Get lm_head
    lm_head = model
    for attr in args.layer_lm_head.split('.'):
        lm_head = getattr(lm_head, attr)
    lm_head = lm_head.weight.data
    
    # Get embedding
    embedding = model
    for attr in args.layer_embed.split('.'):
        embedding = getattr(embedding, attr)
    embedding = embedding.weight.data
    
    print(f"  lm_head: {lm_head.shape}")
    print(f"  embedding: {embedding.shape}")
    
    tile_size = args.tile_size
    num_row_tiles = (lm_head.shape[0] + tile_size - 1) // tile_size
    num_col_tiles = (lm_head.shape[1] + tile_size - 1) // tile_size
    total_tiles = num_row_tiles * num_col_tiles
    
    print(f"\nTiling scheme:")
    print(f"  Matrix: {list(lm_head.shape)}")
    print(f"  Tile size: {tile_size}×{tile_size}")
    print(f"  Row tiles: {num_row_tiles}")
    print(f"  Col tiles: {num_col_tiles}")
    print(f"  Total tiles: {total_tiles}")

    
    # Tile lm_head
    if not args.skip_head:
        print("\nTiling lm_head (direct slice → save → free)...")
        for tile_id in range(total_tiles):
            row_idx = tile_id // num_col_tiles
            col_idx = tile_id % num_col_tiles
            
            row_start = row_idx * tile_size
            row_end = min((row_idx + 1) * tile_size, lm_head.shape[0])
            col_start = col_idx * tile_size
            col_end = min((col_idx + 1) * tile_size, lm_head.shape[1])
            
            tile = lm_head[row_start:row_end, col_start:col_end].clone()
            
            ext = '.npy' if args.format == 'npy' else '.pt'
            filename = output_dir / 'tiled_head' / f'tile_{tile_id:03d}{ext}'
            
            if args.format == 'npy':
                np.save(filename, tile.cpu().numpy())
            else:
                torch.save(tile, filename)
            
            del tile
            
            if (tile_id + 1) % 100 == 0:
                print(f"  {tile_id + 1}/{total_tiles} tiles...")
        
        print(f"  ✓ Saved {total_tiles} lm_head tiles")
    
    # Tile embedding
    if not args.skip_embed:
        print("\nTiling embedding (direct slice → save → free)...")
        for tile_id in range(total_tiles):
            row_idx = tile_id // num_col_tiles
            col_idx = tile_id % num_col_tiles
            
            row_start = row_idx * tile_size
            row_end = min((row_idx + 1) * tile_size, embedding.shape[0])
            col_start = col_idx * tile_size
            col_end = min((col_idx + 1) * tile_size, embedding.shape[1])
            
            tile = embedding[row_start:row_end, col_start:col_end].clone()
            
            ext = '.npy' if args.format == 'npy' else '.pt'
            filename = output_dir / 'tiled_embed' / f'tile_{tile_id:03d}{ext}'
            
            if args.format == 'npy':
                np.save(filename, tile.cpu().numpy())
            else:
                torch.save(tile, filename)
            
            del tile
            
            if (tile_id + 1) % 100 == 0:
                print(f"  {tile_id + 1}/{total_tiles} tiles...")
        
        print(f"  ✓ Saved {total_tiles} embedding tiles")
    
    del model
    
    print("\n" + "="*80)
    print("✓ TILING COMPLETE!")
    print("="*80)
    if not args.skip_head:
        print(f"  lm_head: {total_tiles} tiles in {output_dir}/tiled_head/")
    if not args.skip_embed:
        print(f"  embedding: {total_tiles} tiles in {output_dir}/tiled_embed/")

if __name__ == '__main__':
    main()
