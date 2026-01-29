#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
PrisMoE: Parallel Pipe Architecture for Model-to-Model Connections
Main entry point with proper relative imports
"""
import sys
from pathlib import Path

# Add src and utils to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'utils'))

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='PrisMoE: 3-Pipe Parallel Architecture for Model Connections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and tile model layers
  python prismoe.py extract-head --model "meta-llama/Llama-3.1-8B" --output output/lm_head.npy
  python prismoe.py tile-model --model "LiquidAI/LFM2.5-1.2B-Base" --output-dir output
  
  # Compare matrices
  python prismoe.py compare --matrix1 model1 --matrix2 output/lm_head.npy
  
  # Visualize 3D tensor
  python prismoe.py visualize tensor_file.npy --output-dir plots
  
  # Benchmark (future)
  python prismoe.py benchmark --model llama-3.1-8b --workers 512
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract lm_head command
    extract_head_parser = subparsers.add_parser('extract-head', help='Extract lm_head from model')
    extract_head_parser.add_argument('--model', required=True, help='Model path or HuggingFace ID')
    extract_head_parser.add_argument('--output', required=True, help='Output file path')
    extract_head_parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    
    # Extract embedding command
    extract_embed_parser = subparsers.add_parser('extract-embed', help='Extract embedding from model')
    extract_embed_parser.add_argument('--model', required=True, help='Model path or HuggingFace ID')
    extract_embed_parser.add_argument('--output', required=True, help='Output file path')
    extract_embed_parser.add_argument('--layer-name', default='model.embed_tokens', help='Embedding layer path')
    extract_embed_parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    
    # Tile model command
    tile_parser = subparsers.add_parser('tile-model', help='Tile model layers directly (memory efficient)')
    tile_parser.add_argument('--model', required=True, help='Model path or HuggingFace ID')
    tile_parser.add_argument('--output-dir', default='output', help='Output directory')
    tile_parser.add_argument('--tile-size', type=int, default=512, help='Tile size')
    tile_parser.add_argument('--skip-head', action='store_true', help='Skip lm_head tiling')
    tile_parser.add_argument('--skip-embed', action='store_true', help='Skip embedding tiling')
    
    # Compare matrices command
    compare_parser = subparsers.add_parser('compare', help='Compare two weight matrices')
    compare_parser.add_argument('--matrix1', required=True, help='First matrix (file or model ID)')
    compare_parser.add_argument('--matrix2', required=True, help='Second matrix (file or model ID)')
    compare_parser.add_argument('--name1', help='Name for first matrix')
    compare_parser.add_argument('--name2', help='Name for second matrix')
    
    # Visualize tensor command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize 3D tensor')
    visualize_parser.add_argument('tensor_file', help='Path to tensor file (.npy or .pt)')
    visualize_parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    visualize_parser.add_argument('--prefix', default='tensor', help='Prefix for output files')
    
    # Inspect tile command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect tile file')
    inspect_parser.add_argument('tile_file', help='Path to tile file')
    inspect_parser.add_argument('--tile-id', type=int, default=0, help='Tile ID')
    inspect_parser.add_argument('--grid-cols', type=int, default=4, help='Grid columns')
    
    # Calculate pipes command
    calcpipes_parser = subparsers.add_parser('calcpipes', help='Calculate 3-Pipe architecture configuration')
    calcpipes_parser.add_argument('--tiled-head-dir', required=True, help='Directory with tiled lm_head')
    calcpipes_parser.add_argument('--tiled-embed-dir', required=True, help='Directory with tiled embedding')
    calcpipes_parser.add_argument('--output-dir', default='calculated_pipes', help='Output directory for metadata')
    
    # Build pipes command
    buildpipes_parser = subparsers.add_parser('buildpipes', help='Build 3-Pipe architecture with MicroMoE routers')
    buildpipes_parser.add_argument('--config', required=True, help='Path to architecture.json from calcpipes')
    buildpipes_parser.add_argument('--output-dir', default='builded_pipes', help='Output directory for pipe structures')
    buildpipes_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to build on')
    
    # Train workers command (Phase 1)
    train_workers_parser = subparsers.add_parser('train-workers', help='Phase 1: Train workers via one-shot least squares')
    train_workers_parser.add_argument('--pipes-dir', required=True, help='Directory with builded_pipes')
    train_workers_parser.add_argument('--upper-model', required=True, help='Path to upper model')
    train_workers_parser.add_argument('--lower-model', required=True, help='Path to lower model')
    train_workers_parser.add_argument('--num-samples', type=int, default=10000, help='Number of training samples')
    train_workers_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_workers_parser.add_argument('--output-dir', default='trained_pipes', help='Output directory')
    train_workers_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    train_workers_parser.add_argument('--parallel', action='store_true', help='Use parallel training')
    train_workers_parser.add_argument('--num-workers', type=int, help='Number of parallel workers')
    
    # Train MicroMoE command (Phase 2)
    train_mmoe_parser = subparsers.add_parser('train-mmoe', help='Phase 2: Train MicroMoE routers')
    train_mmoe_parser.add_argument('--trained-pipes', required=True, help='Directory with trained pipes from Phase 1')
    train_mmoe_parser.add_argument('--upper-model', required=True, help='Path to upper model')
    train_mmoe_parser.add_argument('--lower-model', required=True, help='Path to lower model')
    train_mmoe_parser.add_argument('--num-samples', type=int, default=10000, help='Number of training samples')
    train_mmoe_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_mmoe_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_mmoe_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_mmoe_parser.add_argument('--output-dir', default='final_pipes', help='Output directory')
    train_mmoe_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    
    args = parser.parse_args()
    
    if args.command == 'extract-head':
        from extract_lm_head import main as extract_head_main
        sys.argv = ['extract_lm_head.py', '--model', args.model, '--output', args.output, '--format', args.format]
        extract_head_main()
        
    elif args.command == 'extract-embed':
        from extract_embedding import main as extract_embed_main
        sys.argv = ['extract_embedding.py', '--model', args.model, '--output', args.output, 
                    '--layer-name', args.layer_name, '--format', args.format]
        extract_embed_main()
        
    elif args.command == 'tile-model':
        from tile_direct import main as tile_main
        sys.argv = ['tile_direct.py', '--model', args.model, '--output-dir', args.output_dir,
                    '--tile-size', str(args.tile_size)]
        if args.skip_head:
            sys.argv.append('--skip-head')
        if args.skip_embed:
            sys.argv.append('--skip-embed')
        tile_main()
        
    elif args.command == 'compare':
        from compare_heads import main as compare_main
        sys.argv = ['compare_heads.py', '--matrix1', args.matrix1, '--matrix2', args.matrix2]
        if args.name1:
            sys.argv.extend(['--name1', args.name1])
        if args.name2:
            sys.argv.extend(['--name2', args.name2])
        compare_main()
        
    elif args.command == 'visualize':
        from visualize_3d_tensor import main as visualize_main
        sys.argv = ['visualize_3d_tensor.py', args.tensor_file, '--output-dir', args.output_dir,
                    '--prefix', args.prefix]
        visualize_main()
        
    elif args.command == 'inspect':
        from inspect_tile import main as inspect_main
        sys.argv = ['inspect_tile.py', args.tile_file, '--tile-id', str(args.tile_id),
                    '--grid-cols', str(args.grid_cols)]
        inspect_main()
        
    elif args.command == 'calcpipes':
        from calculate_pipes import main as calcpipes_main
        sys.argv = ['calculate_pipes.py', '--tiled-head-dir', args.tiled_head_dir,
                    '--tiled-embed-dir', args.tiled_embed_dir, '--output-dir', args.output_dir]
        calcpipes_main()
        
    elif args.command == 'buildpipes':
        from build_pipes import main as buildpipes_main
        sys.argv = ['build_pipes.py', '--config', args.config, '--output-dir', args.output_dir,
                    '--device', args.device]
        buildpipes_main()
        
    elif args.command == 'train-workers':
        from train_workers import main as train_workers_main
        sys.argv = ['train_workers.py', '--pipes-dir', args.pipes_dir,
                    '--upper-model', args.upper_model, '--lower-model', args.lower_model,
                    '--num-samples', str(args.num_samples), '--batch-size', str(args.batch_size),
                    '--output-dir', args.output_dir, '--device', args.device]
        if args.parallel:
            sys.argv.append('--parallel')
        if args.num_workers:
            sys.argv.extend(['--num-workers', str(args.num_workers)])
        train_workers_main()
        
    elif args.command == 'train-mmoe':
        from train_mmoe import main as train_mmoe_main
        sys.argv = ['train_mmoe.py', '--trained-pipes', args.trained_pipes,
                    '--upper-model', args.upper_model, '--lower-model', args.lower_model,
                    '--num-samples', str(args.num_samples), '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size), '--lr', str(args.lr),
                    '--output-dir', args.output_dir, '--device', args.device]
        train_mmoe_main()
        
    else:
        parser.print_help()
        print("\n💡 Available commands:")
        print("  extract-head  - Extract lm_head weight matrix from model")
        print("  extract-embed - Extract embedding layer from model")
        print("  tile-model    - Tile model layers into 512×512 tiles")
        print("  compare       - Compare two weight matrices")
        print("  visualize     - Visualize 3D tensor with plots")
        print("  inspect       - Inspect individual tile file")
        print("  calcpipes     - Calculate 3-Pipe architecture configuration")
        print("  buildpipes    - Build 3-Pipe architecture with MicroMoE routers")
        print("  train-workers - Phase 1: Train workers via one-shot least squares")
        print("  train-mmoe    - Phase 2: Train MicroMoE routers")
        print("\nFor utility help: python prismoe.py <command> --help")

if __name__ == '__main__':
    main()
